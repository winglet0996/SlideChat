import re
import os
import csv
import torch
import numpy as np
from typing import Any, Sequence, Dict, List, Optional

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


class PathologyMetric(BaseMetric):
    """
    A streamlined metric for pathology report evaluation.
    It generates predictions from logits, then computes BLEU and ROUGE scores
    for both the full report and the specific diagnosis section. It saves
    samples and a quantitative metrics table to a specified
    directory. It calculates and displays metrics with 95% confidence intervals
    using bootstrapping.
    """
    DEFAULT_METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L']
    DIAGNOSIS_PATTERN = re.compile(
        r'Final diagnosis:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL)

    def __init__(self,
                 tokenizer: Dict,
                 print_first_n_samples: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 *args, **kwargs):
        """
        Initializes the metric.

        Args:
            tokenizer (Dict): Configuration for building the tokenizer.
            print_first_n_samples (Optional[int]): The number of initial samples to
                save for qualitative review. If None, save all samples. Defaults to None.
            output_dir (str, optional): The directory to save output files.
                If provided, samples will be saved as .txt files
                and the final metrics will be saved as a .csv file.
                Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = BUILDER.build(tokenizer)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        self.print_first_n_samples = print_first_n_samples
        self.output_dir = output_dir
        self._sample_count = 0

        # Create output directories if specified
        if self.output_dir:
            self.samples_dir = os.path.join(self.output_dir, 'samples')
            os.makedirs(self.samples_dir, exist_ok=True)

    def _decode_full_input(self, token_ids: torch.Tensor) -> str:
        """
        Decodes token IDs into a string, correctly handling special image tokens.
        """
        decoded_parts = []
        special_token_indices = (token_ids == IMAGE_TOKEN_INDEX).nonzero(
            as_tuple=True)[0]

        last_idx = 0
        for special_idx in special_token_indices:
            if special_idx > last_idx:
                decoded_parts.append(
                    self.tokenizer.decode(token_ids[last_idx:special_idx]))
            decoded_parts.append(DEFAULT_IMAGE_TOKEN)
            last_idx = special_idx + 1

        if last_idx < len(token_ids):
            decoded_parts.append(self.tokenizer.decode(token_ids[last_idx:]))

        return "".join(decoded_parts)

    def process(self, data_batch: Any, data_samples: Sequence[Dict]) -> None:
        """
        Process a batch of data, decoding inputs, targets, and predictions.
        Saves the first N samples to individual text files if output_dir is set.
        If print_first_n_samples is None, saves all samples.
        """
        input_ids_batch = data_batch['data']['input_ids']
        labels_batch = data_batch['data']['labels']

        for i, sample in enumerate(data_samples):
            pred_str = sample.get('prediction_text')
            if pred_str is None:
                continue

            labels = labels_batch[i]
            valid_labels = labels[labels != -100]
            target_str = self.tokenizer.decode(
                valid_labels, skip_special_tokens=True)

            if pred_str and target_str:
                # Extract diagnoses
                pred_diag = self._extract_diagnosis(pred_str)
                target_diag = self._extract_diagnosis(target_str)
                
                # Calculate scores for full text
                full_scores = self._compute_scores_for_pair(pred_str.strip(), target_str.strip())
                
                # Calculate scores for diagnosis and check correctness
                diag_scores = {}
                diag_correct = False
                if pred_diag and target_diag:
                    diag_scores = self._compute_scores_for_pair(pred_diag, target_diag)
                    diag_correct = self._is_diagnosis_correct(pred_diag, target_diag)
                else:
                    diag_scores = {m: 0.0 for m in self.DEFAULT_METRICS}
                
                self.results.append({
                    'prediction': pred_str.strip(),
                    'target': target_str.strip(),
                    'pred_diagnosis': pred_diag,
                    'target_diagnosis': target_diag,
                    'diagnosis_correct': diag_correct,
                    'full_scores': full_scores,
                    'diag_scores': diag_scores
                })

            # Save the first N samples to individual .txt files, or all if print_first_n_samples is None
            should_save = False
            if self.output_dir:
                if self.print_first_n_samples is None:
                    should_save = True
                elif self._sample_count < self.print_first_n_samples:
                    should_save = True
            if should_save:
                input_str = self._decode_full_input(input_ids_batch[i])
                
                # Get the most recent result for this sample
                if self.results:
                    result = self.results[-1]
                    full_scores = result.get('full_scores', {})
                    diag_scores = result.get('diag_scores', {})
                    pred_diag = result.get('pred_diagnosis', '')
                    target_diag = result.get('target_diagnosis', '')
                    diag_correct = result.get('diagnosis_correct', False)
                    
                    # Format scores
                    full_scores_str = ', '.join([f"{k}: {v:.4f}" for k, v in full_scores.items()])
                    diag_scores_str = ', '.join([f"{k}: {v:.4f}" for k, v in diag_scores.items()])
                    
                    sample_content = (
                        f"==================== [INPUT & GROUND TRUTH] ====================\n"
                        f"{input_str.strip()}\n\n"
                        f"==================== [PREDICTION] ====================\n"
                        f"{pred_str.strip()}\n\n"
                        f"==================== [EVALUATION METRICS] ====================\n"
                        f"Full Text Scores: {full_scores_str}\n"
                        f"Diagnosis Scores: {diag_scores_str}\n\n"
                        f"==================== [DIAGNOSIS EXTRACTION] ====================\n"
                        f"Predicted Diagnosis: {pred_diag}\n"
                        f"Ground Truth Diagnosis: {target_diag}\n"
                        f"Diagnosis Correct: {'YES' if diag_correct else 'NO'}\n"
                    )
                else:
                    sample_content = (
                        f"==================== [INPUT & GROUND TRUTH] ====================\n"
                        f"{input_str.strip()}\n\n"
                        f"==================== [PREDICTION] ====================\n"
                        f"{pred_str.strip()}\n"
                    )
                
                # Use zero-padding for proper file sorting
                file_path = os.path.join(
                    self.samples_dir, f'sample_{self._sample_count + 1:03d}.txt')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(sample_content)
                self._sample_count += 1

    def process_predictions_and_targets(self, predictions: List[str], targets: List[str]) -> None:
        """
        Simplified process method that directly takes predictions and targets as strings.
        This bypasses the need to create fake input_ids and labels.
        
        Args:
            predictions (List[str]): List of prediction strings
            targets (List[str]): List of target/ground truth strings
        """
        for pred_str, target_str in zip(predictions, targets):
            if pred_str and target_str:
                # Extract diagnoses
                pred_diag = self._extract_diagnosis(pred_str)
                target_diag = self._extract_diagnosis(target_str)
                
                # Calculate scores for full text
                full_scores = self._compute_scores_for_pair(pred_str.strip(), target_str.strip())
                
                # Calculate scores for diagnosis and check correctness
                diag_scores = {}
                diag_correct = False
                if pred_diag and target_diag:
                    diag_scores = self._compute_scores_for_pair(pred_diag, target_diag)
                    diag_correct = self._is_diagnosis_correct(pred_diag, target_diag)
                else:
                    diag_scores = {m: 0.0 for m in self.DEFAULT_METRICS}
                
                self.results.append({
                    'prediction': pred_str.strip(),
                    'target': target_str.strip(),
                    'pred_diagnosis': pred_diag,
                    'target_diagnosis': target_diag,
                    'diagnosis_correct': diag_correct,
                    'full_scores': full_scores,
                    'diag_scores': diag_scores
                })

    @staticmethod
    def _bootstrap_ci(scores: List[float], n_bootstraps: int = 1000, ci_level: float = 0.95) -> tuple[float, float, float]:
        """Calculates the mean and confidence interval using bootstrapping."""
        scores_arr = np.array(scores)
        n_size = len(scores_arr)
        if n_size == 0:
            return 0.0, 0.0, 0.0

        bootstrapped_means = []
        for _ in range(n_bootstraps):
            resample_indices = np.random.randint(0, n_size, size=n_size)
            resample = scores_arr[resample_indices]
            bootstrapped_means.append(np.mean(resample))

        mean_score = np.mean(bootstrapped_means)
        alpha = (1.0 - ci_level) / 2.0
        lower_bound = np.percentile(bootstrapped_means, alpha * 100)
        upper_bound = np.percentile(bootstrapped_means, 100 - alpha * 100)

        return mean_score, lower_bound, upper_bound

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Calculate, print, and save the metrics with bootstrapped CIs."""
        if not results:
            print_log("Warning: No valid results to compute metrics on.", 'current', level='warning')
            return {}

        metrics_scores = {f'full_{m}': [] for m in self.DEFAULT_METRICS}
        metrics_scores.update({f'diag_{m}': [] for m in self.DEFAULT_METRICS})
        diagnosis_accuracy = []

        for res in results:
            # Use pre-computed scores if available
            if 'full_scores' in res and 'diag_scores' in res:
                full_scores = res['full_scores']
                diag_scores = res['diag_scores']
                diag_correct = res.get('diagnosis_correct', False)
            else:
                # Fallback to old computation method
                pred, target = res['prediction'], res['target']
                full_scores = self._compute_scores_for_pair(pred, target)
                
                pred_diag = self._extract_diagnosis(pred)
                target_diag = self._extract_diagnosis(target)
                if pred_diag and target_diag:
                    diag_scores = self._compute_scores_for_pair(pred_diag, target_diag)
                    diag_correct = self._is_diagnosis_correct(pred_diag, target_diag)
                else:
                    diag_scores = {m: 0.0 for m in self.DEFAULT_METRICS}
                    diag_correct = False
            
            # Add scores to lists
            for m in self.DEFAULT_METRICS:
                metrics_scores[f'full_{m}'].append(full_scores.get(m, 0.0))
                metrics_scores[f'diag_{m}'].append(diag_scores.get(m, 0.0))
            
            # Add diagnosis accuracy
            diagnosis_accuracy.append(1.0 if diag_correct else 0.0)

        # Calculate bootstrapped CI for all metrics including diagnosis accuracy
        final_metrics_with_ci = {}
        # This dict is what mmengine expects as the return value
        avg_metrics_prefixed = {}

        for name, scores in metrics_scores.items():
            mean, lower, upper = self._bootstrap_ci(scores)
            final_metrics_with_ci[name] = {'mean': mean, 'lower': lower, 'upper': upper}
            # mmengine evaluator requires a single float value per metric
            avg_metrics_prefixed[f'eval/{name}'] = mean
        
        # Add diagnosis accuracy
        diag_acc_mean, diag_acc_lower, diag_acc_upper = self._bootstrap_ci(diagnosis_accuracy)
        final_metrics_with_ci['diag_accuracy'] = {
            'mean': diag_acc_mean, 
            'lower': diag_acc_lower, 
            'upper': diag_acc_upper
        }
        avg_metrics_prefixed['eval/diag_accuracy'] = diag_acc_mean

        # Print the plain text table to the console
        self._print_results_table(final_metrics_with_ci)
        
        # Save the metrics with CI to a CSV file if a directory is provided
        if self.output_dir:
            self._save_metrics_to_csv(final_metrics_with_ci)
            
        return avg_metrics_prefixed

    def _extract_diagnosis(self, text: str) -> str:
        """Extract the final diagnosis from a report using regex."""
        match = self.DIAGNOSIS_PATTERN.search(text)
        return match.group(1).strip() if match else ""
    
    def _is_diagnosis_correct(self, pred_diagnosis: str, target_diagnosis: str) -> bool:
        """Check if the ground truth diagnosis is contained in the predicted diagnosis."""
        if not pred_diagnosis or not target_diagnosis:
            return False
        return target_diagnosis.lower() in pred_diagnosis.lower()

    def _compute_scores_for_pair(self, pred: str, target: str) -> Dict[str, float]:
        """Compute BLEU and ROUGE scores for a single prediction-target pair."""
        if not pred or not target:
            return {m: 0.0 for m in self.DEFAULT_METRICS}

        pred_tokens = pred.lower().split()
        target_tokens_list = [target.lower().split()]

        try:
            return {
                'ROUGE-L': self.rouge_scorer.score(target, pred)['rougeL'].fmeasure,
                'BLEU-1': sentence_bleu(target_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothie),
                'BLEU-2': sentence_bleu(target_tokens_list, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothie),
                'BLEU-3': sentence_bleu(target_tokens_list, pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=self.smoothie),
                'BLEU-4': sentence_bleu(target_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothie)
            }
        except (ValueError, ZeroDivisionError) as e:
            print_log(f"Warning: Could not compute scores for a pair: {e}", 'current', level='warning')
            return {m: 0.0 for m in self.DEFAULT_METRICS}

    def _print_results_table(self, metrics_with_ci: Dict[str, Dict[str, float]]) -> None:
        """Format and print the final results as a plain text table."""
        header = f"{'Metric Category':<20} | {'Metric Name':<15} | {'Score (Mean & 95% CI)':<35}"
        separator = '-' * len(header)
        
        print_log("\n" + separator, 'current')
        print_log("Pathology Text Generation Evaluation".center(len(header)), 'current')
        print_log(separator, 'current')
        print_log(header, 'current')
        print_log(separator, 'current')

        # Print Full Report Metrics
        print_log(f"{'Full Text Report':<20} |", 'current')
        for metric_key in self.DEFAULT_METRICS:
            full_metric_name = f'full_{metric_key}'
            if full_metric_name in metrics_with_ci:
                stats = metrics_with_ci[full_metric_name]
                mean, lower, upper = stats['mean'], stats['lower'], stats['upper']
                score_str = f"{mean:.4f} ({lower:.4f} - {upper:.4f})"
                print_log(f"{'':<20} | {metric_key:<15} | {score_str:<35}", 'current')

        print_log(separator, 'current')
        
        # Print Diagnosis Metrics
        print_log(f"{'Final Diagnosis':<20} |", 'current')
        for metric_key in self.DEFAULT_METRICS:
            diag_metric_name = f'diag_{metric_key}'
            if diag_metric_name in metrics_with_ci:
                stats = metrics_with_ci[diag_metric_name]
                mean, lower, upper = stats['mean'], stats['lower'], stats['upper']
                score_str = f"{mean:.4f} ({lower:.4f} - {upper:.4f})"
                print_log(f"{'':<20} | {metric_key:<15} | {score_str:<35}", 'current')
        
        # Print Diagnosis Accuracy
        if 'diag_accuracy' in metrics_with_ci:
            stats = metrics_with_ci['diag_accuracy']
            mean, lower, upper = stats['mean'], stats['lower'], stats['upper']
            score_str = f"{mean:.4f} ({lower:.4f} - {upper:.4f})"
            print_log(f"{'':<20} | {'ACCURACY':<15} | {score_str:<35}", 'current')
                
        print_log(separator + "\n", 'current')

    def _save_metrics_to_csv(self, metrics_with_ci: Dict[str, Dict[str, float]]) -> None:
        """Save the final metrics with CI to a CSV file."""
        csv_path = os.path.join(self.output_dir, 'evaluation_metrics.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Metric', 'Mean', '95% CI Lower', '95% CI Upper'])
            
            # Write full report metrics
            for metric_key in self.DEFAULT_METRICS:
                full_metric_name = f'full_{metric_key}'
                if full_metric_name in metrics_with_ci:
                    stats = metrics_with_ci[full_metric_name]
                    writer.writerow([
                        'Full Text Report',
                        metric_key,
                        f"{stats['mean']:.4f}",
                        f"{stats['lower']:.4f}",
                        f"{stats['upper']:.4f}",
                    ])
            
            # Write diagnosis metrics
            for metric_key in self.DEFAULT_METRICS:
                diag_metric_name = f'diag_{metric_key}'
                if diag_metric_name in metrics_with_ci:
                    stats = metrics_with_ci[diag_metric_name]
                    writer.writerow([
                        'Final Diagnosis',
                        metric_key,
                        f"{stats['mean']:.4f}",
                        f"{stats['lower']:.4f}",
                        f"{stats['upper']:.4f}",
                    ])
            
            # Write diagnosis accuracy
            if 'diag_accuracy' in metrics_with_ci:
                stats = metrics_with_ci['diag_accuracy']
                writer.writerow([
                    'Final Diagnosis',
                    'ACCURACY',
                    f"{stats['mean']:.4f}",
                    f"{stats['lower']:.4f}",
                    f"{stats['upper']:.4f}",
                ])
        
        print_log(f"Metrics table saved to: {csv_path}", 'current')