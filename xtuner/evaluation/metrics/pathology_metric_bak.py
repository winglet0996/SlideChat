import re
import os
import csv
import json
import torch
import numpy as np
from typing import Any, Sequence, Dict, List, Optional
from scipy import stats

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


class PathologyMetric(BaseMetric):
    """
    A comprehensive metric for pathology evaluation supporting three task types:
      1) Free-text generation: computes BLEU/ROUGE scores for full report and diagnosis
      2) Multi-Choice Question Answering (MCQA): computes per-category accuracy
      3) Regression: computes MSE, Pearson correlation, and Spearman correlation
    
    Task type is automatically determined by the category field:
    - Regression: category contains 'regression' (case-insensitive)
    - MCQA: category provided and target is single letter choice
    - Free-text: default case
    
    All metrics include 95% bootstrap confidence intervals and separate output files.
    """

    DEFAULT_METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L']
    REGRESSION_METRICS = ['MSE', 'Pearson', 'Spearman']
    DIAGNOSIS_PATTERN = re.compile(
        r'Final diagnosis:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL)
    CHOICE_PATTERN = re.compile(r'^[A-Z]$')

    def __init__(self,
                 tokenizer: Dict,
                 save_first_n_samples: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 *args, **kwargs):
        """
        Initialize the multi-task pathology metric evaluator.

        Args:
            tokenizer (Dict): Configuration for building the tokenizer
            save_first_n_samples (Optional[int]): Number of samples to save for review
            output_dir (Optional[str]): Directory to save evaluation outputs
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = BUILDER.build(tokenizer)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        self.save_first_n_samples = save_first_n_samples
        self.output_dir = output_dir

        # Prepare output directories on main process only
        if self.output_dir and is_main_process():
            self.samples_dir = os.path.join(self.output_dir, 'samples')
            self.mcqa_samples_dir = os.path.join(self.output_dir, 'mcqa_samples')
            self.regression_samples_dir = os.path.join(self.output_dir, 'regression_samples')
            os.makedirs(self.samples_dir, exist_ok=True)
            os.makedirs(self.mcqa_samples_dir, exist_ok=True)
            os.makedirs(self.regression_samples_dir, exist_ok=True)

    def process(self, data_batch: Any, data_samples: Sequence[Dict]) -> None:
        """
        Process a batch of data samples across different task types.
        Automatically determines task type based on category and target format:
        - Regression: category contains 'regression'
        - MCQA: category provided and target is single letter
        - Free-text: default case
        """
        input_ids_batch = data_batch['data']['input_ids']
        labels_batch = data_batch['data']['labels']
        category_batch = data_batch['data'].get('category', None)
        image_file_batch = data_batch['data'].get('image_file', None)
        regression_targets_batch = data_batch['data'].get('regression_targets', None)

        for i, sample in enumerate(data_samples):
            pred_str = sample.get('prediction_text')
            if pred_str is None:
                continue

            # Extract target text and regression target
            target_str = self._get_target_text(data_batch, i, labels_batch)
            regression_target = self._get_regression_target(regression_targets_batch, i)

            if target_str is None and regression_target is None:
                continue

            pred_str = str(pred_str).strip()
            input_str = self._decode_full_input(input_ids_batch[i])
            filename = self._extract_filename_from_image_file(image_file_batch, i)

            # Determine task type based on category
            category_i = self._extract_category(category_batch, i)
            
            if self._is_regression_task(category_i):
                self._process_regression_sample(
                    sample, input_str, pred_str, target_str, regression_target, category_i, filename
                )
            elif self._is_mcqa_task(category_i, target_str):
                self._process_mcqa_sample(
                    sample, input_str, pred_str, target_str, category_i, filename
                )
            else:
                self._process_text_sample(
                    sample, input_str, pred_str, target_str, filename
                )

    def _extract_category(self, category_batch: Any, i: int) -> str:
        """Extract category string from batch data."""
        if category_batch is None:
            return ""
        
        if isinstance(category_batch, (list, tuple)):
            category_i = category_batch[i] if i < len(category_batch) else ""
        elif isinstance(category_batch, str):
            category_i = category_batch
        else:
            category_i = ""
            
        # Handle nested lists like [['Regression Task']]
        if isinstance(category_i, (list, tuple)) and category_i:
            category_i = category_i[0]
            
        return str(category_i) if category_i else ""

    def _get_regression_target(self, regression_targets_batch: Any, i: int) -> Optional[float]:
        """Extract regression target value for sample i."""
        if regression_targets_batch is None:
            return None
            
        try:
            if isinstance(regression_targets_batch, (list, tuple)):
                if i < len(regression_targets_batch):
                    target = regression_targets_batch[i]
                    return float(target) if target is not None else None
            elif hasattr(regression_targets_batch, '__getitem__'):
                target = regression_targets_batch[i]
                return float(target) if target is not None else None
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def _is_regression_task(self, category: str) -> bool:
        """Check if task is regression based on category."""
        return 'regression' in category.lower()

    def _is_mcqa_task(self, category: str, target_str: Optional[str]) -> bool:
        """Check if task is MCQA based on category and target format."""
        return (category.strip() != "" and 
                target_str is not None and 
                self._looks_like_mcqa(target_str))

    def _process_regression_sample(self, sample: Dict, input_str: str, pred_str: str, 
                                 target_str: Optional[str], regression_target: Optional[float],
                                 category: str, filename: str) -> None:
        """Process regression task sample."""
        # Extract numerical prediction from prediction text
        pred_value = self._extract_numerical_value(pred_str)
        
        # Use regression target if available, otherwise try to parse target_str
        if regression_target is not None:
            target_value = regression_target
        elif target_str is not None:
            target_value = self._extract_numerical_value(target_str)
        else:
            target_value = None

        if pred_value is not None and target_value is not None:
            self.results.append({
                'task_type': 'regression',
                'filename': filename,
                'input': input_str.strip(),
                'prediction': pred_str.strip(),
                'pred_value': pred_value,
                'target_value': target_value,
                'category': category
            })

            # Log regression details
            if is_main_process():
                print_log(f"Regression Input: {input_str.strip()}", 'current')
                print_log(f"Regression Prediction: {pred_str.strip()} -> {pred_value}", 'current')
                print_log(f"Regression Target: {target_value}", 'current')
                print_log(f"Category: {category}", 'current')
                print_log(f"Squared Error: {(pred_value - target_value) ** 2:.4f}", 'current')

    def _process_mcqa_sample(self, sample: Dict, input_str: str, pred_str: str,
                           target_str: str, category: str, filename: str) -> None:
        """Process MCQA task sample."""
        pred_choice = self._extract_mcqa_choice(pred_str)
        target_choice = self._extract_mcqa_choice(target_str)

        if not pred_choice or not target_choice:
            return

        correct = (pred_choice == target_choice)

        self.results.append({
            'task_type': 'mcqa',
            'filename': filename,
            'input': input_str.strip(),
            'prediction': pred_str.strip(),
            'pred_choice': pred_choice,
            'target': target_choice,
            'category': category,
            'mcqa_correct': correct
        })

        # Log MCQA details
        if is_main_process():
            print_log(f"MCQA Input: {input_str.strip()}", 'current')
            print_log(f"MCQA Prediction: {pred_str.strip()} -> {pred_choice}", 'current')
            print_log(f"MCQA Target: {target_choice}", 'current')
            print_log(f"Category: {category}", 'current')
            print_log(f"MCQA Correct: {'YES' if correct else 'NO'}", 'current')

    def _process_text_sample(self, sample: Dict, input_str: str, pred_str: str,
                           target_str: str, filename: str) -> None:
        """Process free-text generation sample."""
        if not target_str.strip():
            return
            
        target_str = target_str.strip()
        
        # Extract diagnosis sections
        pred_diag = self._extract_diagnosis(pred_str)
        target_diag = self._extract_diagnosis(target_str)

        # Compute text similarity scores
        full_scores = self._compute_scores_for_pair(pred_str, target_str)

        diag_correct = False
        if pred_diag and target_diag:
            diag_scores = self._compute_scores_for_pair(pred_diag, target_diag)
            diag_correct = self._is_diagnosis_correct(pred_diag, target_diag)
        else:
            diag_scores = {m: 0.0 for m in self.DEFAULT_METRICS}

        self.results.append({
            'task_type': 'text',
            'filename': filename,
            'input': input_str.strip(),
            'prediction': pred_str.strip(),
            'target': target_str,
            'pred_diagnosis': pred_diag,
            'target_diagnosis': target_diag,
            'diagnosis_correct': diag_correct,
            'full_scores': full_scores,
            'diag_scores': diag_scores
        })

        # Log free-text details
        if is_main_process():
            print_log(f"Text Input: {input_str.strip()}", 'current')
            print_log(f"Text Prediction: {pred_str.strip()}", 'current')
            print_log(f"Text Target: {target_str}", 'current')
            print_log(f"Predicted Diagnosis: {pred_diag}", 'current')
            print_log(f"Target Diagnosis: {target_diag}", 'current')
            print_log(f"Diagnosis Correct: {'YES' if diag_correct else 'NO'}", 'current')

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """
        Compute metrics for all task types on gathered results from all ranks.
        Handles Free-text, MCQA, and Regression tasks separately with appropriate metrics.
        """
        if not results:
            print_log("Warning: No valid results to compute metrics on.", 'current')
            return {}

        # Split results by task type
        text_results = [r for r in results if r.get('task_type', 'text') == 'text' or 'full_scores' in r]
        mcqa_results = [r for r in results if r.get('task_type') == 'mcqa']
        regression_results = [r for r in results if r.get('task_type') == 'regression']

        # Aggregate metrics to return to the engine
        avg_metrics_for_engine: Dict[str, float] = {}

        # 1) Free-text evaluation
        if text_results:
            if self.output_dir:
                self._save_samples_from_results(text_results)
            text_metrics = self._compute_text_metrics(text_results)
            avg_metrics_for_engine.update(text_metrics)
            self._print_results_table(text_metrics)
            if self.output_dir:
                self._save_metrics_to_csv(text_metrics)

        # 2) MCQA evaluation
        if mcqa_results:
            if self.output_dir:
                self._save_mcqa_samples_from_results(mcqa_results)
            mcqa_metrics = self._compute_mcqa_metrics(mcqa_results)
            avg_metrics_for_engine.update(mcqa_metrics)
            self._print_mcqa_results_table(mcqa_metrics)
            if self.output_dir:
                self._save_mcqa_metrics_to_csv(mcqa_metrics)

        # 3) Regression evaluation
        if regression_results:
            if self.output_dir:
                self._save_regression_samples_from_results(regression_results)
            regression_metrics = self._compute_regression_metrics(regression_results)
            avg_metrics_for_engine.update(regression_metrics)
            self._print_regression_results_table(regression_metrics)
            if self.output_dir:
                self._save_regression_metrics_to_csv(regression_metrics)

        return avg_metrics_for_engine

    def _compute_text_metrics(self, text_results: List[Dict]) -> Dict[str, float]:
        """Compute BLEU/ROUGE metrics for free-text generation."""
        metrics_scores = {f'full_{m}': [] for m in self.DEFAULT_METRICS}
        metrics_scores.update({f'diag_{m}': [] for m in self.DEFAULT_METRICS})
        diagnosis_accuracy = []

        for res in text_results:
            full_scores = res['full_scores']
            diag_scores = res['diag_scores']
            diag_correct = res['diagnosis_correct']

            for m in self.DEFAULT_METRICS:
                metrics_scores[f'full_{m}'].append(full_scores.get(m, 0.0))
                metrics_scores[f'diag_{m}'].append(diag_scores.get(m, 0.0))

            diagnosis_accuracy.append(1.0 if diag_correct else 0.0)

        # Compute metrics with confidence intervals
        final_metrics_with_ci = {}
        engine_metrics = {}
        
        for name, scores in metrics_scores.items():
            mean, lower, upper = self._bootstrap_ci(scores)
            final_metrics_with_ci[name] = {'mean': mean, 'lower': lower, 'upper': upper}
            engine_metrics[f'eval/{name}'] = mean

        diag_acc_mean, diag_acc_lower, diag_acc_upper = self._bootstrap_ci(diagnosis_accuracy)
        final_metrics_with_ci['diag_accuracy'] = {
            'mean': diag_acc_mean, 'lower': diag_acc_lower, 'upper': diag_acc_upper
        }
        engine_metrics['eval/diag_accuracy'] = diag_acc_mean

        # Store for printing and saving
        self._text_metrics_ci = final_metrics_with_ci
        return engine_metrics

    def _compute_mcqa_metrics(self, mcqa_results: List[Dict]) -> Dict[str, float]:
        """Compute accuracy metrics for MCQA tasks."""
        category_to_corrects: Dict[str, List[float]] = {}
        overall_corrects: List[float] = []

        for res in mcqa_results:
            cat = res.get('category', 'UNKNOWN')
            correct = 1.0 if res.get('mcqa_correct', False) else 0.0
            overall_corrects.append(correct)
            category_to_corrects.setdefault(cat, []).append(correct)

        # Compute per-category and overall accuracy with bootstrap CIs
        mcqa_metrics_with_ci: Dict[str, Dict[str, float]] = {}
        engine_metrics = {}
        
        for cat, values in category_to_corrects.items():
            mean, lower, upper = self._bootstrap_ci(values)
            mcqa_metrics_with_ci[cat] = {'mean': mean, 'lower': lower, 'upper': upper, 'count': len(values)}
            engine_metrics[f"eval/mcqa/{self._slugify_category(cat)}/accuracy"] = mean

        overall_mean, overall_lower, overall_upper = self._bootstrap_ci(overall_corrects)
        mcqa_metrics_with_ci['__OVERALL__'] = {
            'mean': overall_mean, 'lower': overall_lower, 'upper': overall_upper, 'count': len(overall_corrects)
        }
        engine_metrics['eval/mcqa_overall_accuracy'] = overall_mean

        # Store for printing and saving
        self._mcqa_metrics_ci = mcqa_metrics_with_ci
        return engine_metrics

    def _compute_regression_metrics(self, regression_results: List[Dict]) -> Dict[str, float]:
        """Compute MSE, Pearson, and Spearman metrics for regression tasks."""
        category_to_predictions: Dict[str, List[tuple]] = {}
        overall_predictions = []
        overall_targets = []

        for res in regression_results:
            cat = res.get('category', 'UNKNOWN')
            pred_val = res['pred_value']
            target_val = res['target_value']
            
            overall_predictions.append(pred_val)
            overall_targets.append(target_val)
            
            category_to_predictions.setdefault(cat, []).append((pred_val, target_val))

        # Compute per-category and overall regression metrics
        regression_metrics_with_ci: Dict[str, Dict[str, float]] = {}
        engine_metrics = {}
        
        # Per-category metrics
        for cat, pred_target_pairs in category_to_predictions.items():
            predictions = [p[0] for p in pred_target_pairs]
            targets = [p[1] for p in pred_target_pairs]
            
            cat_metrics = self._compute_regression_metrics_for_pairs(predictions, targets)
            regression_metrics_with_ci[cat] = cat_metrics
            regression_metrics_with_ci[cat]['count'] = len(pred_target_pairs)
            
            # Add to engine metrics
            for metric_name in self.REGRESSION_METRICS:
                engine_metrics[f"eval/regression/{self._slugify_category(cat)}/{metric_name.lower()}"] = cat_metrics[metric_name]['mean']

        # Overall metrics
        if overall_predictions and overall_targets:
            overall_metrics = self._compute_regression_metrics_for_pairs(overall_predictions, overall_targets)
            regression_metrics_with_ci['__OVERALL__'] = overall_metrics
            regression_metrics_with_ci['__OVERALL__']['count'] = len(overall_predictions)
            
            # Add to engine metrics
            for metric_name in self.REGRESSION_METRICS:
                engine_metrics[f"eval/regression_overall_{metric_name.lower()}"] = overall_metrics[metric_name]['mean']

        # Store for printing and saving
        self._regression_metrics_ci = regression_metrics_with_ci
        return engine_metrics

    def _compute_regression_metrics_for_pairs(self, predictions: List[float], targets: List[float]) -> Dict[str, Dict[str, float]]:
        """Compute regression metrics with bootstrap confidence intervals."""
        if not predictions or not targets or len(predictions) != len(targets):
            return {metric: {'mean': 0.0, 'lower': 0.0, 'upper': 0.0} for metric in self.REGRESSION_METRICS}

        predictions = np.array(predictions)
        targets = np.array(targets)

        # MSE with bootstrap CI
        mse_scores = []
        pearson_scores = []
        spearman_scores = []
        
        n_size = len(predictions)
        n_bootstraps = min(1000, n_size * 10)  # Adjust bootstrap size for small samples
        
        for _ in range(n_bootstraps):
            indices = np.random.randint(0, n_size, size=n_size)
            boot_pred = predictions[indices]
            boot_target = targets[indices]
            
            # MSE
            mse = np.mean((boot_pred - boot_target) ** 2)
            mse_scores.append(float(mse))
            
            # Pearson correlation
            if len(np.unique(boot_pred)) > 1 and len(np.unique(boot_target)) > 1:
                try:
                    pearson_r, _ = stats.pearsonr(boot_pred, boot_target)
                    pearson_scores.append(float(pearson_r) if not np.isnan(pearson_r) else 0.0)
                except:
                    pearson_scores.append(0.0)
            else:
                pearson_scores.append(0.0)
            
            # Spearman correlation
            if len(np.unique(boot_pred)) > 1 and len(np.unique(boot_target)) > 1:
                try:
                    spearman_r, _ = stats.spearmanr(boot_pred, boot_target)
                    spearman_scores.append(float(spearman_r) if not np.isnan(spearman_r) else 0.0)
                except:
                    spearman_scores.append(0.0)
            else:
                spearman_scores.append(0.0)

        # Compute confidence intervals
        metrics = {}
        for metric_name, scores in zip(self.REGRESSION_METRICS, [mse_scores, pearson_scores, spearman_scores]):
            mean, lower, upper = self._bootstrap_ci(scores, n_bootstraps=len(scores))
            metrics[metric_name] = {'mean': mean, 'lower': lower, 'upper': upper}
        
        return metrics

    def _extract_numerical_value(self, text: str) -> Optional[float]:
        """Extract numerical value from text string."""
        if not text:
            return None
            
        # Try to convert entire string to float first
        try:
            return float(text.strip())
        except ValueError:
            pass
        
        # Extract first number found in the text
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        return None

    # -----------------------------
    # Saving (Free-text) Samples
    # -----------------------------
    def _save_samples_from_results(self, results: List[Dict]):
        """Saves free-text samples to JSON files from the gathered results list."""
        print_log(f"Saving free-text samples to {self.samples_dir}...", 'current')
        num_to_save = len(results) if self.save_first_n_samples is None else self.save_first_n_samples

        # Group samples by filename
        filename_to_samples = {}
        
        for i, res in enumerate(results[:num_to_save]):
            filename = res.get('filename', f'sample_{i + 1:05d}')
            
            sample_data = {
                'input': res['input'],
                'prediction': res['prediction'],
                'target': res['target'],
                'pred_diagnosis': res['pred_diagnosis'],
                'target_diagnosis': res['target_diagnosis'],
                'diagnosis_correct': res['diagnosis_correct'],
                'full_scores': res['full_scores'],
                'diag_scores': res['diag_scores']
            }
            
            if filename not in filename_to_samples:
                filename_to_samples[filename] = []
            filename_to_samples[filename].append(sample_data)

        # Save each file as JSON
        saved_count = 0
        for filename, samples in filename_to_samples.items():
            file_path = os.path.join(self.samples_dir, f'{filename}.json')
            
            # If file exists, load existing data and append
            existing_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except Exception:
                    existing_data = []
            
            # Append new samples
            existing_data.extend(samples)
            
            # Save updated data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            saved_count += len(samples)
            
    # -----------------------------
    # Saving (MCQA) Samples
    # -----------------------------
    def _save_mcqa_samples_from_results(self, results: List[Dict]):
        """Save MCQA samples to JSON files from the gathered results list."""
        print_log(f"Saving MCQA samples to {self.mcqa_samples_dir}...", 'current')
        num_to_save = len(results) if self.save_first_n_samples is None else self.save_first_n_samples

        # Group samples by filename
        filename_to_samples = {}
        
        for i, res in enumerate(results[:num_to_save]):
            filename = res.get('filename', f'mcqa_sample_{i + 1:05d}')
            
            sample_data = {
                'input': res['input'],
                'category': res.get('category', 'UNKNOWN'),
                'prediction': res['prediction'],
                'pred_choice': res['pred_choice'],
                'target': res['target'],
                'mcqa_correct': res.get('mcqa_correct', False)
            }
            
            if filename not in filename_to_samples:
                filename_to_samples[filename] = []
            filename_to_samples[filename].append(sample_data)

        # Save each file as JSON
        saved_count = 0
        for filename, samples in filename_to_samples.items():
            file_path = os.path.join(self.mcqa_samples_dir, f'{filename}.json')
            
            # If file exists, load existing data and append
            existing_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except Exception:
                    existing_data = []
            
            # Append new samples
            existing_data.extend(samples)
            
            # Save updated data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            saved_count += len(samples)
            
        print_log(f"Saved {saved_count} MCQA samples to {len(filename_to_samples)} JSON files.", 'current')
    def _save_regression_samples_from_results(self, results: List[Dict]):
        """Save regression samples to JSON files from the gathered results list."""
        print_log(f"Saving regression samples to {self.regression_samples_dir}...", 'current')
        num_to_save = len(results) if self.save_first_n_samples is None else self.save_first_n_samples

        # Group samples by filename
        filename_to_samples = {}
        
        for i, res in enumerate(results[:num_to_save]):
            filename = res.get('filename', f'regression_sample_{i + 1:05d}')
            
            sample_data = {
                'input': res['input'],
                'category': res.get('category', 'UNKNOWN'),
                'prediction': res['prediction'],
                'pred_value': res['pred_value'],
                'target_value': res['target_value'],
                'squared_error': (res['pred_value'] - res['target_value']) ** 2
            }
            
            if filename not in filename_to_samples:
                filename_to_samples[filename] = []
            filename_to_samples[filename].append(sample_data)

        # Save each file as JSON
        saved_count = 0
        for filename, samples in filename_to_samples.items():
            file_path = os.path.join(self.regression_samples_dir, f'{filename}.json')
            
            # If file exists, load existing data and append
            existing_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except Exception:
                    existing_data = []
            
            # Append new samples
            existing_data.extend(samples)
            
            # Save updated data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            saved_count += len(samples)
            
        print_log(f"Saved {saved_count} regression samples to {len(filename_to_samples)} JSON files.", 'current')

    # -----------------------------
    # Printing (Regression) Results
    # -----------------------------
    def _print_regression_results_table(self, regression_metrics_ci: Dict[str, Dict[str, float]]) -> None:
        """Print regression evaluation results in a formatted table."""
        if not hasattr(self, '_regression_metrics_ci'):
            return
            
        metrics_with_ci = self._regression_metrics_ci
        header = f"{'Regression Category':<40} | {'Metric':<10} | {'Score (Mean & 95% CI)':<35} | {'Count':<7}"
        separator = '-' * len(header)
        
        print_log("\n" + separator, 'current')
        print_log("Regression Evaluation".center(len(header)), 'current')
        print_log(separator, 'current')
        print_log(header, 'current')
        print_log(separator, 'current')

        # Per-category metrics
        for cat, cat_metrics in metrics_with_ci.items():
            if cat == '__OVERALL__':
                continue
                
            count_str = f"{int(cat_metrics.get('count', 0))}"
            
            # Print each regression metric for this category
            for i, metric_name in enumerate(self.REGRESSION_METRICS):
                stats = cat_metrics.get(metric_name, {'mean': 0.0, 'lower': 0.0, 'upper': 0.0})
                score_str = f"{stats['mean']:.4f} ({stats['lower']:.4f} - {stats['upper']:.4f})"
                
                # Only show category name on first metric line
                cat_display = cat if i == 0 else ""
                count_display = count_str if i == 0 else ""
                print_log(f"{cat_display:<40} | {metric_name:<10} | {score_str:<35} | {count_display:<7}", 'current')

        # Overall metrics
        if '__OVERALL__' in metrics_with_ci:
            print_log(separator, 'current')
            overall_metrics = metrics_with_ci['__OVERALL__']
            count_str = f"{int(overall_metrics.get('count', 0))}"
            
            for i, metric_name in enumerate(self.REGRESSION_METRICS):
                stats = overall_metrics.get(metric_name, {'mean': 0.0, 'lower': 0.0, 'upper': 0.0})
                score_str = f"{stats['mean']:.4f} ({stats['lower']:.4f} - {stats['upper']:.4f})"
                
                cat_display = "OVERALL" if i == 0 else ""
                count_display = count_str if i == 0 else ""
                print_log(f"{cat_display:<40} | {metric_name:<10} | {score_str:<35} | {count_display:<7}", 'current')
                
        print_log(separator + "\n", 'current')

    # -----------------------------
    # CSV Saving (Regression)
    # -----------------------------
    def _save_regression_metrics_to_csv(self, regression_metrics_ci: Dict[str, Dict[str, float]]) -> None:
        """Save regression metrics to CSV file."""
        if not hasattr(self, '_regression_metrics_ci'):
            return
            
        metrics_with_ci = self._regression_metrics_ci
        csv_path = os.path.join(self.output_dir, 'regression_evaluation_metrics.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Regression Category', 'Metric', 'Mean', '95% CI Lower', '95% CI Upper', 'Count'])
            
            # Per-category metrics
            for cat, cat_metrics in metrics_with_ci.items():
                if cat == '__OVERALL__':
                    continue
                    
                count = int(cat_metrics.get('count', 0))
                for metric_name in self.REGRESSION_METRICS:
                    stats = cat_metrics.get(metric_name, {'mean': 0.0, 'lower': 0.0, 'upper': 0.0})
                    writer.writerow([
                        cat, metric_name,
                        f"{stats['mean']:.4f}", f"{stats['lower']:.4f}", f"{stats['upper']:.4f}",
                        count
                    ])
            
            # Overall metrics
            if '__OVERALL__' in metrics_with_ci:
                overall_metrics = metrics_with_ci['__OVERALL__']
                count = int(overall_metrics.get('count', 0))
                
                for metric_name in self.REGRESSION_METRICS:
                    stats = overall_metrics.get(metric_name, {'mean': 0.0, 'lower': 0.0, 'upper': 0.0})
                    writer.writerow([
                        'OVERALL', metric_name,
                        f"{stats['mean']:.4f}", f"{stats['lower']:.4f}", f"{stats['upper']:.4f}",
                        count
                    ])
        
        print_log(f"Regression metrics table saved to: {csv_path}", 'current')
    def _save_mcqa_samples_from_results(self, results: List[Dict]):
        """Saves MCQA samples to JSON files from the gathered results list."""
        print_log(f"Saving MCQA samples to {self.mcqa_samples_dir}...", 'current')
        num_to_save = len(results) if self.save_first_n_samples is None else self.save_first_n_samples

        # Group samples by filename
        filename_to_samples = {}
        
        for i, res in enumerate(results[:num_to_save]):
            filename = res.get('filename', f'mcqa_sample_{i + 1:05d}')
            
            sample_data = {
                'input': res['input'],
                'category': res.get('category', 'UNKNOWN'),
                'prediction': res['prediction'],
                'pred_choice': res['pred_choice'],
                'target': res['target'],
                'mcqa_correct': res.get('mcqa_correct', False)
            }
            
            if filename not in filename_to_samples:
                filename_to_samples[filename] = []
            filename_to_samples[filename].append(sample_data)

        # Save each file as JSON
        saved_count = 0
        for filename, samples in filename_to_samples.items():
            file_path = os.path.join(self.mcqa_samples_dir, f'{filename}.json')
            
            # If file exists, load existing data and append
            existing_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except Exception:
                    existing_data = []
            
            # Append new samples
            existing_data.extend(samples)
            
            # Save updated data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            saved_count += len(samples)
            
        print_log(f"Saved {saved_count} MCQA samples to {len(filename_to_samples)} JSON files.", 'current')

    # -----------------------------
    # Decoding helpers
    # -----------------------------
    def _decode_full_input(self, token_ids: torch.Tensor) -> str:
        decoded_parts = []
        special_token_indices = (token_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        last_idx = 0
        for special_idx in special_token_indices:
            if special_idx > last_idx:
                decoded_parts.append(self.tokenizer.decode(token_ids[last_idx:special_idx]))
            decoded_parts.append(DEFAULT_IMAGE_TOKEN)
            last_idx = special_idx + 1
        if last_idx < len(token_ids):
            decoded_parts.append(self.tokenizer.decode(token_ids[last_idx:]))
        return "".join(decoded_parts)

    def _get_target_text(self, data_batch: Any, i: int, labels_batch: torch.Tensor) -> Optional[str]:
        """Robustly obtain target text: use labels_text if present; fallback to decoding labels."""
        labels_text_list = data_batch['data'].get('labels_text', None)
        if labels_text_list is not None:
            try:
                return labels_text_list[i]
            except Exception:
                pass

        # Fallback: decode labels by skipping masked positions (-100)
        if labels_batch is not None:
            labels = labels_batch[i]
            if isinstance(labels, torch.Tensor):
                valid_labels = labels[labels != -100]
                try:
                    return self.tokenizer.decode(valid_labels, skip_special_tokens=True)
                except Exception:
                    # As a last resort, decode without skipping special tokens
                    return self.tokenizer.decode(valid_labels)
        return None

    def _extract_filename_from_image_file(self, image_file_batch: Any, i: int) -> str:
        """Extract filename from image_file batch, removing path and extension."""
        if image_file_batch is None:
            return f"sample_{i + 1:05d}"
        
        try:
            # Handle different possible formats of image_file_batch
            if isinstance(image_file_batch, (list, tuple)):
                if i < len(image_file_batch):
                    image_file = image_file_batch[i]
                else:
                    return f"sample_{i + 1:05d}"
            else:
                image_file = image_file_batch
            
            # Handle nested lists like [['path/to/file.h5']]
            if isinstance(image_file, (list, tuple)) and len(image_file) > 0:
                image_file = image_file[0]
                if isinstance(image_file, (list, tuple)) and len(image_file) > 0:
                    image_file = image_file[0]
            
            if isinstance(image_file, str):
                # Extract filename from path and remove extension
                filename = os.path.basename(image_file)
                # Remove extension (e.g., .h5)
                filename_without_ext = os.path.splitext(filename)[0]
                return filename_without_ext
            else:
                return f"sample_{i + 1:05d}"
        except Exception:
            return f"sample_{i + 1:05d}"

    # -----------------------------
    # Statistics and scoring
    # -----------------------------
    @staticmethod
    def _bootstrap_ci(scores: List[float], n_bootstraps: int = 1000, ci_level: float = 0.95) -> tuple[float, float, float]:
        scores_arr = np.array(scores, dtype=float)
        n_size = len(scores_arr)
        if n_size == 0:
            return 0.0, 0.0, 0.0

        bootstrapped_means = [
            float(np.mean(scores_arr[np.random.randint(0, n_size, size=n_size)]))
            for _ in range(n_bootstraps)
        ]
        mean_score = float(np.mean(bootstrapped_means))
        alpha = (1.0 - ci_level) / 2.0
        lower_bound = float(np.percentile(bootstrapped_means, alpha * 100))
        upper_bound = float(np.percentile(bootstrapped_means, 100 - alpha * 100))
        return mean_score, lower_bound, upper_bound

    def _extract_diagnosis(self, text: str) -> str:
        match = self.DIAGNOSIS_PATTERN.search(text)
        return match.group(1).strip() if match else ""

    def _is_diagnosis_correct(self, pred_diagnosis: str, target_diagnosis: str) -> bool:
        if not pred_diagnosis or not target_diagnosis:
            return False
        return target_diagnosis.lower() in pred_diagnosis.lower()

    def _compute_scores_for_pair(self, pred: str, target: str) -> Dict[str, float]:
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
            print_log(f"Warning: Could not compute scores for a pair: {e}", 'current')
            return {m: 0.0 for m in self.DEFAULT_METRICS}

    # -----------------------------
    # Printing (Free-text) Results
    # -----------------------------
    def _print_results_table(self, engine_metrics: Dict[str, float]) -> None:
        """Print free-text evaluation results in a formatted table."""
        if not hasattr(self, '_text_metrics_ci'):
            return
            
        metrics_with_ci = self._text_metrics_ci
        header = f"{'Metric Category':<20} | {'Metric Name':<15} | {'Score (Mean & 95% CI)':<35}"
        separator = '-' * len(header)
        print_log("\n" + separator, 'current')
        print_log("Pathology Text Generation Evaluation".center(len(header)), 'current')
        print_log(separator, 'current')
        print_log(header, 'current')
        print_log(separator, 'current')

        # Full Report
        print_log(f"{'Full Text Report':<20} |", 'current')
        for metric_key in self.DEFAULT_METRICS:
            stats = metrics_with_ci.get(f'full_{metric_key}')
            if stats:
                score_str = f"{stats['mean']:.4f} ({stats['lower']:.4f} - {stats['upper']:.4f})"
                print_log(f"{'':<20} | {metric_key:<15} | {score_str:<35}", 'current')
        print_log(separator, 'current')

        # Diagnosis
        print_log(f"{'Final Diagnosis':<20} |", 'current')
        for metric_key in self.DEFAULT_METRICS:
            stats = metrics_with_ci.get(f'diag_{metric_key}')
            if stats:
                score_str = f"{stats['mean']:.4f} ({stats['lower']:.4f} - {stats['upper']:.4f})"
                print_log(f"{'':<20} | {metric_key:<15} | {score_str:<35}", 'current')
        if 'diag_accuracy' in metrics_with_ci:
            stats = metrics_with_ci['diag_accuracy']
            score_str = f"{stats['mean']:.4f} ({stats['lower']:.4f} - {stats['upper']:.4f})"
            print_log(f"{'':<20} | {'ACCURACY':<15} | {score_str:<35}", 'current')
        print_log(separator + "\n", 'current')

    # -----------------------------
    # Printing (MCQA) Results
    # -----------------------------
    def _print_mcqa_results_table(self, engine_metrics: Dict[str, float]) -> None:
        """Print MCQA evaluation results in a formatted table."""
        if not hasattr(self, '_mcqa_metrics_ci'):
            return
            
        mcqa_metrics_with_ci = self._mcqa_metrics_ci
        header = f"{'MCQA Category':<40} | {'Accuracy (Mean & 95% CI)':<35} | {'Count':<7}"
        separator = '-' * len(header)
        print_log("\n" + separator, 'current')
        print_log("MCQA Evaluation".center(len(header)), 'current')
        print_log(separator, 'current')
        print_log(header, 'current')
        print_log(separator, 'current')

        # Per-category
        for cat, stats in mcqa_metrics_with_ci.items():
            if cat == '__OVERALL__':
                continue
            score_str = f"{stats['mean']:.4f} ({stats['lower']:.4f} - {stats['upper']:.4f})"
            count_str = f"{int(stats.get('count', 0))}"
            print_log(f"{cat:<40} | {score_str:<35} | {count_str:<7}", 'current')

        # Overall
        if '__OVERALL__' in mcqa_metrics_with_ci:
            print_log(separator, 'current')
            stats = mcqa_metrics_with_ci['__OVERALL__']
            score_str = f"{stats['mean']:.4f} ({stats['lower']:.4f} - {stats['upper']:.4f})"
            count_str = f"{int(stats.get('count', 0))}"
            print_log(f"{'OVERALL':<40} | {score_str:<35} | {count_str:<7}", 'current')
        print_log(separator + "\n", 'current')

    # -----------------------------
    # CSV Saving (Free-text)
    # -----------------------------
    def _save_metrics_to_csv(self, engine_metrics: Dict[str, float]) -> None:
        """Save free-text metrics to CSV file."""
        if not hasattr(self, '_text_metrics_ci'):
            return
            
        metrics_with_ci = self._text_metrics_ci
        csv_path = os.path.join(self.output_dir, 'evaluation_metrics.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Metric', 'Mean', '95% CI Lower', '95% CI Upper'])
            categories = {'Full Text Report': 'full_', 'Final Diagnosis': 'diag_'}
            for cat_name, prefix in categories.items():
                for metric_key in self.DEFAULT_METRICS:
                    stats = metrics_with_ci.get(f'{prefix}{metric_key}')
                    if stats:
                        writer.writerow([
                            cat_name, metric_key,
                            f"{stats['mean']:.4f}", f"{stats['lower']:.4f}", f"{stats['upper']:.4f}"
                        ])
            if 'diag_accuracy' in metrics_with_ci:
                stats = metrics_with_ci['diag_accuracy']
                writer.writerow([
                    'Final Diagnosis', 'ACCURACY',
                    f"{stats['mean']:.4f}", f"{stats['lower']:.4f}", f"{stats['upper']:.4f}"
                ])
        print_log(f"Metrics table saved to: {csv_path}", 'current')

    # -----------------------------
    # CSV Saving (MCQA)
    # -----------------------------
    def _save_mcqa_metrics_to_csv(self, engine_metrics: Dict[str, float]) -> None:
        """Save MCQA metrics to CSV file."""
        if not hasattr(self, '_mcqa_metrics_ci'):
            return
            
        mcqa_metrics_with_ci = self._mcqa_metrics_ci
        csv_path = os.path.join(self.output_dir, 'mcqa_evaluation_metrics.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['MCQA Category', 'Metric', 'Mean', '95% CI Lower', '95% CI Upper', 'Count'])
            for cat, stats in mcqa_metrics_with_ci.items():
                if cat == '__OVERALL__':
                    continue
                writer.writerow([
                    cat, 'ACCURACY',
                    f"{stats['mean']:.4f}", f"{stats['lower']:.4f}", f"{stats['upper']:.4f}",
                    int(stats.get('count', 0))
                ])
            # Overall row
            if '__OVERALL__' in mcqa_metrics_with_ci:
                o = mcqa_metrics_with_ci['__OVERALL__']
                writer.writerow([
                    'OVERALL', 'ACCURACY',
                    f"{o['mean']:.4f}", f"{o['lower']:.4f}", f"{o['upper']:.4f}",
                    int(o.get('count', 0))
                ])
        print_log(f"MCQA metrics table saved to: {csv_path}", 'current')

    # -----------------------------
    # MCQA helpers
    # -----------------------------
    def _looks_like_mcqa(self, text: str) -> bool:
        """Return True if text is a single capital letter or contains such choice-like token."""
        t = (text or "").strip().upper()
        return bool(self.CHOICE_PATTERN.match(t))

    def _extract_mcqa_choice(self, text: str) -> str:
        """
        Extract a single capital-letter choice from text.
        - If text is exactly a single capital letter, return it.
        - Else, try to find a standalone capital letter token as a fallback.
        """
        t = (text or "").strip().upper()
        if self.CHOICE_PATTERN.fullmatch(t):
            return t
        return ""

        # Fallback: try to find a single capital-letter word boundary
        m = re.search(r'\b([A-Z])\b', t)
        if m:
            return m.group(1)

        # As a last fallback, take the first capital letter if any
        m = re.search(r'([A-Z])', t)
        return m.group(1) if m else ""

    @staticmethod
    def _slugify_category(text: str) -> str:
        """Sanitize category names to be used in metric keys."""
        return re.sub(r'[^a-zA-Z0-9]+', '_', str(text)).strip('_').lower()