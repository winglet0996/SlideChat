import os
import re
import json
import torch
import numpy as np
from typing import Any, Sequence, Dict, List, Optional, Union
from scipy import stats
from collections import defaultdict

from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX



class PathologyMetric(BaseMetric):
    """
    Comprehensive pathology evaluation metric supporting multiple task types:
    - Free-text generation: BLEU/ROUGE scores for reports and diagnosis
    - Multi-Choice QA (MCQA): Per-category accuracy with confidence intervals  
    - Regression: RMSE, MAE, R², Pearson/Spearman correlation with bootstrap CI
    - Survival prediction: C-index, Integrated Brier Score, time-dependent AUC
    
    Task type is automatically determined from data structure.
    All metrics include 95% bootstrap confidence intervals.
    Results are saved to comprehensive JSON files with complete evaluation data.
    """

    # Metric configurations
    TEXT_METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L']
    REGRESSION_METRICS = ['RMSE', 'MAE', 'R2', 'Pearson', 'Spearman'] 
    SURVIVAL_METRICS = ['C-Index', 'IBS', 'Time-AUC']
    
    # Pattern matching
    DIAGNOSIS_PATTERN = re.compile(r'Final diagnosis:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL)
    CHOICE_PATTERN = re.compile(r'^[A-Z]$')

    def __init__(self,
                 tokenizer: Dict,
                 output_dir: Optional[str] = None,
                 survival_time_intervals: Optional[List[float]] = None,
                 survival_training_data_path: Optional[str] = '/home/ps/pathology/codes/project/TCGA/dataset_pp/PathoVerse_stage2_regression_train_no-knowledge.json',
                 *args, **kwargs):
        """
        Initialize the multi-task pathology metric evaluator.

        Args:
            tokenizer: Configuration for building the tokenizer
            output_dir: Directory to save evaluation JSON results
            survival_time_intervals: Time intervals for survival analysis
            survival_training_data_path: Path to JSON file containing training survival data
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = BUILDER.build(tokenizer)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        self.output_dir = output_dir
        self.survival_training_data_path = survival_training_data_path
        
        # Survival metrics: lazily initialize from model outputs if intervals not specified
        self._survival_intervals = np.array(survival_time_intervals) if survival_time_intervals is not None else None
        self.survival_metrics = None

        # Prepare output directory for JSON results only
        if self.output_dir and is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)

    def process(self, data_batch: Any, data_samples: Sequence[Dict]) -> None:
        """
        Process a batch of data samples, automatically determining task type.
        """
        # Extract batch data
        batch_data = self._extract_batch_data(data_batch)
        
        # Lazy init survival metrics using intervals coming from the first survival prediction
        if batch_data['survival_targets'] is not None and self.survival_metrics is None and self._survival_intervals is None:
            for s in data_samples:
                sp = s.get('survival_prediction') if isinstance(s, dict) else None
                if sp and isinstance(sp, dict) and sp.get('time_intervals') is not None:
                    try:
                        self._survival_intervals = np.array(sp['time_intervals'], dtype=float)
                        break
                    except Exception:
                        pass
            # Fallback to simple default if nothing found yet
            if self._survival_intervals is None:
                self._survival_intervals = np.array([0, 1, 2, 3, 5, 7, 10], dtype=float)
            self.survival_metrics = SurvivalMetrics(self._survival_intervals)
            
            # Load training data if path is provided
            if self.survival_training_data_path:
                self.survival_metrics.load_training_data_from_json(self.survival_training_data_path)

        for i, sample in enumerate(data_samples):
            # Decode input and predictions
            input_str = self._decode_full_input(batch_data['input_ids'][i])
            
            # Handle different sample structures
            pred_str = self._extract_prediction_text(sample)
            
            # Extract metadata
            metadata = self._extract_sample_metadata(batch_data, i)
            
            # Determine task type and process accordingly
            task_type = self._determine_task_type(metadata, pred_str, sample)
            
            if task_type == 'survival':
                self._process_survival_sample(sample, input_str, pred_str, metadata)
            elif task_type == 'regression':
                self._process_regression_sample(sample, input_str, pred_str, metadata)
            elif task_type == 'mcqa':
                self._process_mcqa_sample(sample, input_str, pred_str, metadata)
            else:  # Free-text generation
                self._process_text_sample(sample, input_str, pred_str, metadata)

    def _extract_prediction_text(self, sample: Dict) -> str:
        """Extract prediction text written by model."""
        # The model always writes a plain string under 'prediction_text'
        return sample.get('prediction_text', "")

    def _extract_batch_data(self, data_batch: Any) -> Dict[str, Any]:
        """Extract relevant data from batch."""
        return {
            'input_ids': data_batch['data']['input_ids'],
            'labels': data_batch['data']['labels'],
            'category': data_batch['data'].get('category', None),
            'image_file': data_batch['data'].get('image_file', None),
            'regression_targets': data_batch['data'].get('regression_targets', None),
            'survival_targets': data_batch['data'].get('survival_targets', None),
            'survival_times': data_batch['data'].get('survival_times', None),
            'survival_events': data_batch['data'].get('survival_events', None),
        }

    def _extract_sample_metadata(self, batch_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Extract metadata for a specific sample."""
        metadata = {}
        
        # Extract category
        if batch_data['category'] is not None:
            if isinstance(batch_data['category'], (list, tuple)):
                cat = batch_data['category'][idx] if idx < len(batch_data['category']) else ""
            else:
                cat = str(batch_data['category'])
            
            # Handle nested categories
            if isinstance(cat, (list, tuple)) and cat:
                cat = str(cat[0])
            metadata['category'] = str(cat) if cat else ""
        else:
            metadata['category'] = ""
        
        # Extract targets
        metadata['target_str'] = self._get_target_text(batch_data, idx)
        metadata['regression_target'] = self._get_regression_target(batch_data['regression_targets'], idx)
        metadata['survival_data'] = self._get_survival_data(batch_data, idx)
        
        # Extract filename
        metadata['filename'] = self._extract_filename_from_image_file(batch_data['image_file'], idx)
        
        return metadata

    def _determine_task_type(self, metadata: Dict[str, Any], pred_str: str, sample: Dict) -> str:
        """Determine task type using explicit model fields."""
        if 'survival_prediction' in sample:
            return 'survival'
        if 'regression_prediction' in sample:
            return 'regression'
        # MCQA is inferred from prediction text pattern only
        return 'mcqa' if self._looks_like_mcqa(pred_str) else 'text'

    def _process_survival_sample(self, sample: Dict, input_str: str, pred_str: str, 
                               metadata: Dict[str, Any]) -> None:
        """Process survival prediction sample using model's survival_prediction dict."""
        survival_data = metadata['survival_data']
        if not survival_data:
            return
        sp = sample['survival_prediction']  # dict with keys: logits, survival_probs, risk_score, median_survival_time
        self.results.append({
            'task_type': 'survival',
            'filename': metadata['filename'],
            'input': input_str.strip(),
            'prediction': pred_str.strip(),
            'category': metadata['category'],
            'survival_probs': sp.get('survival_probs'),
            'risk_score': sp.get('risk_score'),
            'event_time': survival_data['time'],
            'event_indicator': survival_data['event']
        })

    def _process_regression_sample(self, sample: Dict, input_str: str, pred_str: str,
                                 metadata: Dict[str, Any]) -> None:
        """Process regression sample using explicit fields."""
        pred_value = sample['regression_prediction']
        target_value = metadata['regression_target']
        if pred_value is None or target_value is None:
            return
        self.results.append({
            'task_type': 'regression',
            'filename': metadata['filename'],
            'input': input_str.strip(),
            'prediction': pred_str.strip(),
            'pred_value': float(pred_value),
            'target_value': float(target_value),
            'category': metadata['category']
        })

    def _process_mcqa_sample(self, sample: Dict, input_str: str, pred_str: str,
                           metadata: Dict[str, Any]) -> None:
        """Process MCQA sample."""
        pred_choice = self._extract_mcqa_choice(pred_str)
        target_choice = self._extract_mcqa_choice(metadata['target_str'] or "")
        
        if not pred_choice or not target_choice:
            return
        
        correct = (pred_choice == target_choice)
        
        self.results.append({
            'task_type': 'mcqa',
            'filename': metadata['filename'],
            'input': input_str.strip(),
            'prediction': pred_str.strip(),
            'pred_choice': pred_choice,
            'target_choice': target_choice,
            'category': metadata['category'],
            'correct': correct
        })

    def _process_text_sample(self, sample: Dict, input_str: str, pred_str: str,
                           metadata: Dict[str, Any]) -> None:
        """Process free-text generation sample."""
        target_str = metadata['target_str']
        if not target_str or not target_str.strip():
            return
        
        target_str = target_str.strip()
        
        # Extract diagnosis sections
        pred_diag = self._extract_diagnosis(pred_str)
        target_diag = self._extract_diagnosis(target_str)
        
        # Compute similarity scores
        full_scores = self._compute_text_scores(pred_str, target_str)
        diag_scores = self._compute_text_scores(pred_diag, target_diag) if pred_diag and target_diag else {}
        diag_correct = self._is_diagnosis_correct(pred_diag, target_diag)
        
        self.results.append({
            'task_type': 'text',
            'filename': metadata['filename'],
            'input': input_str.strip(),
            'prediction': pred_str.strip(),
            'target': target_str,
            'pred_diagnosis': pred_diag,
            'target_diagnosis': target_diag,
            'diagnosis_correct': diag_correct,
            'full_scores': full_scores,
            'diag_scores': diag_scores
        })

    def compute_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """
        Compute metrics for all task types with proper error handling.
        """
        if not results:
            return {}

        # Group results by task type
        task_results = defaultdict(list)
        for result in results:
            task_type = result.get('task_type', 'text')
            task_results[task_type].append(result)

        # Store task results for summary printing
        self._computed_task_results = task_results

        # Compute metrics for each task type
        all_metrics = {}
        
        if task_results['text']:
            all_metrics.update(self._compute_text_metrics(task_results['text']))
        
        if task_results['mcqa']:
            all_metrics.update(self._compute_mcqa_metrics(task_results['mcqa']))
        
        if task_results['regression']:
            all_metrics.update(self._compute_regression_metrics(task_results['regression']))
        
        if task_results['survival'] and self.survival_metrics:
            all_metrics.update(self._compute_survival_metrics(task_results['survival']))

        # Save results and print summaries
        if is_main_process() and self.output_dir:
            self._save_all_results(task_results)
            self._print_metric_summaries(all_metrics)

        return all_metrics

    # (Old _compute_survival_metrics removed; per-category version defined later.)

    # Helper methods for text scoring, MCQA, regression, etc.
    def _compute_text_scores(self, pred: str, target: str) -> Dict[str, float]:
        """Compute BLEU and ROUGE scores for text pair."""
        if not pred.strip() or not target.strip():
            return {metric: 0.0 for metric in self.TEXT_METRICS}
        
        scores = {}
        
        # BLEU scores
        pred_tokens = pred.lower().split()
        target_tokens = target.lower().split()
        
        for n in range(1, 5):
            bleu = sentence_bleu([target_tokens], pred_tokens, 
                               weights=tuple([1/n] * n + [0] * (4-n)),
                               smoothing_function=self.smoothie)
            scores[f'BLEU-{n}'] = bleu
        
        # ROUGE-L score
        rouge_score = self.rouge_scorer.score(target, pred)
        scores['ROUGE-L'] = rouge_score['rougeL'].fmeasure
        
        return scores

    def _looks_like_mcqa(self, text: str) -> bool:
        """Check if text looks like MCQA response."""
        text = text.strip()
        return bool(self.CHOICE_PATTERN.match(text)) or any(
            choice in text.upper()[:10] for choice in ['A)', 'B)', 'C)', 'D)', 'E)']
        )

    def _extract_mcqa_choice(self, text: str) -> str:
        """Extract choice letter from MCQA response."""
        if not text:
            return ""
        
        text = text.strip().upper()
        
        # Direct match
        if self.CHOICE_PATTERN.match(text):
            return text
        
        # Extract from patterns like "A)", "A:", "A."
        for pattern in [r'^([A-E])[):\.]', r'([A-E])[):\.]', r'^([A-E])']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""

    def _extract_diagnosis(self, text: str) -> str:
        """Extract diagnosis from text."""
        if not text:
            return ""
        
        match = self.DIAGNOSIS_PATTERN.search(text)
        return match.group(1).strip() if match else ""

    def _is_diagnosis_correct(self, pred_diag: str, target_diag: str) -> bool:
        """Check if diagnosis prediction is correct."""
        if not pred_diag or not target_diag:
            return False
        
        # Simple exact match after normalization
        pred_norm = re.sub(r'[^\w\s]', '', pred_diag.lower()).strip()
        target_norm = re.sub(r'[^\w\s]', '', target_diag.lower()).strip()
        
        return pred_norm == target_norm

    def _extract_numerical_value(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        if not text:
            return None
        
        # Try direct conversion first
        try:
            return float(text.strip())
        except ValueError:
            pass
        
        # Extract first number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        return None

    @staticmethod
    def _bootstrap_ci(values: List[float], n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple:
        """Compute bootstrap confidence interval."""
        if not values:
            return 0.0, 0.0, 0.0
        
        values = np.array(values)
        n = len(values)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        mean_val = np.mean(values)
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return mean_val, lower, upper

    # Data extraction helper methods
    def _decode_full_input(self, token_ids: torch.Tensor) -> str:
        """Decode full input including special tokens."""
        decoded_parts = []
        special_token_indices = (token_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        last_idx = 0
        
        for special_idx in special_token_indices:
            if last_idx < special_idx:
                decoded_parts.append(self.tokenizer.decode(token_ids[last_idx:special_idx], skip_special_tokens=True))
            decoded_parts.append(DEFAULT_IMAGE_TOKEN)
            last_idx = special_idx + 1
        
        if last_idx < len(token_ids):
            decoded_parts.append(self.tokenizer.decode(token_ids[last_idx:], skip_special_tokens=True))
        
        return ''.join(decoded_parts)

    def _get_target_text(self, batch_data: Dict[str, Any], idx: int) -> Optional[str]:
        """Extract target text for sample."""
        labels = batch_data['labels'][idx]
        target_ids = labels[labels != -100]
        
        if len(target_ids) == 0:
            return None
        
        return self.tokenizer.decode(target_ids, skip_special_tokens=True)

    def _get_regression_target(self, regression_targets: Any, idx: int) -> Optional[float]:
        """Extract regression target for sample."""
        if regression_targets is None:
            return None
        
        try:
            if isinstance(regression_targets, (list, tuple)):
                return float(regression_targets[idx]) if idx < len(regression_targets) else None
            else:
                return float(regression_targets[idx])
        except (IndexError, ValueError, TypeError):
            return None

    def _get_survival_data(self, batch_data: Dict[str, Any], idx: int) -> Optional[Dict[str, float]]:
        """Extract survival time and event from discretized survival_targets.

        Uses the interval index derived from target_y/at_risk_mask and maps
        it to a continuous time via interval midpoints.
        """
        st = batch_data.get('survival_targets')
        if st is None:
            return None

        target_y = st['target_y'][idx]
        at_risk_mask = st['at_risk_mask'][idx]

        # Convert to numpy for indexing
        if isinstance(target_y, torch.Tensor):
            target_y = target_y.detach().cpu().numpy()
        if isinstance(at_risk_mask, torch.Tensor):
            at_risk_mask = at_risk_mask.detach().cpu().numpy()

        # Event occurs in the interval where target_y == 1, else censored at last at-risk interval
        if (target_y > 0.5).any():
            k = int(np.argmax(target_y))
            event = 1.0
        else:
            at_risk_indices = np.where(at_risk_mask > 0.5)[0]
            k = int(at_risk_indices.max()) if len(at_risk_indices) > 0 else 0
            event = 0.0

        # Ensure k is within valid range for interval_endpoints
        k = min(k, len(self.survival_metrics.interval_endpoints) - 1)
        
        # Map interval index to time using interval right endpoints (ensures within bounds)
        time = float(self.survival_metrics.interval_endpoints[k])
        return {'time': time, 'event': event}

    def _extract_filename_from_image_file(self, image_file_batch: Any, idx: int) -> str:
        """Extract filename from image file batch."""
        if image_file_batch is None:
            return f"sample_{idx}"
        
        try:
            if isinstance(image_file_batch, (list, tuple)):
                filename = image_file_batch[idx] if idx < len(image_file_batch) else f"sample_{idx}"
            else:
                filename = str(image_file_batch)
            
            return os.path.basename(str(filename)) if filename else f"sample_{idx}"
        except (IndexError, TypeError):
            return f"sample_{idx}"

    def _compute_text_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute text generation metrics."""
        if not results:
            return {}
        
        # Collect scores for all metrics
        metrics_scores = {f'full_{m}': [] for m in self.TEXT_METRICS}
        metrics_scores.update({f'diag_{m}': [] for m in self.TEXT_METRICS})
        diagnosis_accuracy = []
        
        for res in results:
            # Full text scores
            full_scores = res.get('full_scores', {})
            for metric in self.TEXT_METRICS:
                if metric in full_scores:
                    metrics_scores[f'full_{metric}'].append(full_scores[metric])
            
            # Diagnosis scores
            diag_scores = res.get('diag_scores', {})
            for metric in self.TEXT_METRICS:
                if metric in diag_scores:
                    metrics_scores[f'diag_{metric}'].append(diag_scores[metric])
            
            # Diagnosis accuracy
            diagnosis_accuracy.append(float(res.get('diagnosis_correct', False)))
        
        # Compute metrics with confidence intervals
        engine_metrics = {}
        for name, scores in metrics_scores.items():
            if scores:
                mean, lower, upper = self._bootstrap_ci(scores)
                engine_metrics[f'eval/{name}'] = mean
        
        # Diagnosis accuracy
        if diagnosis_accuracy:
            diag_acc_mean, _, _ = self._bootstrap_ci(diagnosis_accuracy)
            engine_metrics['eval/diag_accuracy'] = diag_acc_mean
        
        return engine_metrics

    def _compute_mcqa_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute MCQA metrics."""
        if not results:
            return {}
        
        category_to_corrects = defaultdict(list)
        overall_corrects = []
        
        for res in results:
            correct = float(res.get('correct', False))
            category = res.get('category', 'Unknown')
            
            category_to_corrects[category].append(correct)
            overall_corrects.append(correct)
        
        # Compute per-category and overall accuracy
        engine_metrics = {}
        for cat, values in category_to_corrects.items():
            if values:
                mean, _, _ = self._bootstrap_ci(values)
                safe_cat = cat.replace('/', '_').replace(' ', '_')
                engine_metrics[f'eval/mcqa_{safe_cat}_accuracy'] = mean
        
        # Overall accuracy
        if overall_corrects:
            overall_mean, _, _ = self._bootstrap_ci(overall_corrects)
            engine_metrics['eval/mcqa_overall_accuracy'] = overall_mean
        
        return engine_metrics

    def _compute_regression_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute regression metrics per category and overall."""
        if not results:
            return {}
            
        # Group results by category
        cat_groups = defaultdict(list)
        for r in results:
            cat_groups[r.get('category', 'regression')].append(r)

        def compute_category_metrics(group: List[Dict]) -> Optional[Dict[str, float]]:
            """Compute metrics for a single category group."""
            preds, targets = [], []
            for g in group:
                pv, tv = g.get('pred_value'), g.get('target_value')
                if pv is not None and tv is not None:
                    preds.append(float(pv))
                    targets.append(float(tv))
            
            if not preds:
                return None
                
            preds, targets = np.array(preds), np.array(targets)
            residuals = preds - targets
            mse = float(np.mean(residuals**2))
            mae = float(np.mean(np.abs(residuals)))
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((targets - targets.mean())**2))
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
            
            metrics = {'rmse': float(np.sqrt(mse)), 'mae': mae, 'r2': r2}
            
            # Add correlation metrics if sufficient samples
            if len(preds) > 1:
                try:
                    pearson_r, _ = stats.pearsonr(preds, targets)
                    spearman_r, _ = stats.spearmanr(preds, targets)
                    metrics.update({'pearson': float(pearson_r), 'spearman': float(spearman_r)})
                except Exception:
                    metrics.update({'pearson': float('nan'), 'spearman': float('nan')})
            
            return metrics

        # Compute metrics for all categories
        engine_metrics = {}
        for cat, group in cat_groups.items():
            metrics = compute_category_metrics(group)
            if metrics:
                safe_cat = cat.lower().replace(' ', '_').replace('/', '_')
                for metric_name, value in metrics.items():
                    engine_metrics[f'eval/reg_{safe_cat}_{metric_name}'] = float(value)
        
        # Compute overall metrics
        overall_metrics = compute_category_metrics(results)
        if overall_metrics:
            for metric_name, value in overall_metrics.items():
                engine_metrics[f'eval/reg_overall_{metric_name}'] = float(value)
                
        return engine_metrics

    def _compute_survival_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute survival metrics per category and overall."""
        if not self.survival_metrics or not results:
            return {}
            
        # Group by category
        cat_groups = defaultdict(list)
        for r in results:
            cat_groups[r.get('category', 'Survival')].append(r)
            
        def compute_category_survival_metrics(group: List[Dict]) -> Dict[str, float]:
            """Compute survival metrics for a single category group."""
            survival_probs = np.array([g['survival_probs'] for g in group], dtype=float)
            risk_scores = np.array([g['risk_score'] for g in group], dtype=float)
            event_times = np.array([g['event_time'] for g in group], dtype=float)
            event_indicators = np.array([g['event_indicator'] for g in group], dtype=float)
            return self.survival_metrics.compute_all_metrics(survival_probs, risk_scores, event_times, event_indicators)
        
        engine_metrics = {}
        
        # Compute per-category metrics
        for cat, group in cat_groups.items():
            try:
                metrics = compute_category_survival_metrics(group)
                safe_cat = cat.lower().replace(' ', '_').replace('/', '_')
                for metric_name, value in metrics.items():
                    engine_metrics[f'eval/surv_{safe_cat}_{metric_name}'] = float(value)
            except Exception as e:
                print_log(f'Survival metrics failed for category {cat}: {e}', 'current')
        
        # Compute overall metrics
        try:
            overall_metrics = compute_category_survival_metrics(results)
            for metric_name, value in overall_metrics.items():
                engine_metrics[f'eval/surv_overall_{metric_name}'] = float(value)
            self._survival_metrics = overall_metrics
        except Exception as e:
            print_log(f'Overall survival metrics failed: {e}', 'current')
            
        return engine_metrics

    def _save_all_results(self, task_results: Dict[str, List[Dict]]) -> None:
        """Save results with category split for structured tasks."""
        for task_type, results in task_results.items():
            if not results:
                continue
                
            # Save merged file
            merged_path = os.path.join(self.output_dir, f'{task_type}_results.json')
            self._save_json_file(results, merged_path, f"{len(results)} {task_type} results")
            
            # Save per-category files for structured tasks
            if task_type in ['mcqa', 'regression', 'survival']:
                self._save_category_results(results, task_type)

    def _save_json_file(self, data: List[Dict], file_path: str, description: str) -> None:
        """Save data to JSON file with error handling."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print_log(f"Saved {description} to {file_path}", 'current')
        except Exception as e:
            print_log(f"Error saving {description}: {e}", 'current')

    def _save_category_results(self, results: List[Dict], task_type: str) -> None:
        """Save results split by category."""
        category_results = defaultdict(list)
        for result in results:
            category = result.get('category', 'Unknown')
            category_results[category].append(result)
            
        for category, category_data in category_results.items():
            safe_category = category.lower().replace(' ', '_').replace('/', '_')
            file_path = os.path.join(self.output_dir, f'{task_type}_{safe_category}_results.json')
            description = f"{task_type} category {category} ({len(category_data)})"
            self._save_json_file(category_data, file_path, description)

    def _print_metric_summaries(self, all_metrics: Dict[str, float]) -> None:
        """Print formatted metric summaries by task type in table format."""
        print_log("=" * 100, 'current')
        print_log(" " * 35 + "EVALUATION RESULTS SUMMARY", 'current')
        print_log("=" * 100, 'current')
        
        if not hasattr(self, '_computed_task_results'):
            print_log("=" * 100, 'current')
            return
            
        task_results = self._computed_task_results
        task_printers = {
            'text': self._print_text_summary,
            'mcqa': lambda m: self._print_mcqa_summary(m, task_results['mcqa']),
            'regression': lambda m: self._print_regression_summary(m, task_results['regression']),
            'survival': lambda m: self._print_survival_summary_detailed(m, task_results['survival'])
        }
        
        for task_type, printer in task_printers.items():
            if task_type in task_results and task_results[task_type]:
                printer(all_metrics)
        
        print_log("=" * 100, 'current')

    def _print_text_summary(self, all_metrics: Dict[str, float]) -> None:
        """Print text generation metrics summary."""
        print_log("\nFREE-TEXT GENERATION RESULTS:", 'current')
        print_log("-" * 80, 'current')
        print_log(f"{'Metric':<30} {'Value':<15}", 'current')
        print_log("-" * 80, 'current')
        
        text_metrics = {
            'eval/full_BLEU-4': 'Full Text BLEU-4',
            'eval/full_ROUGE-L': 'Full Text ROUGE-L',
            'eval/diag_BLEU-4': 'Diagnosis BLEU-4',
            'eval/diag_ROUGE-L': 'Diagnosis ROUGE-L',
            'eval/diag_accuracy': 'Diagnosis Accuracy'
        }
        
        for key, name in text_metrics.items():
            if key in all_metrics:
                value = all_metrics[key]
                print_log(f"{name:<30} {value:<15.4f}", 'current')
        print_log("-" * 80, 'current')

    def _print_mcqa_summary(self, all_metrics: Dict[str, float], results: List[Dict]) -> None:
        """Print MCQA metrics summary by category."""
        print_log("\nMULTI-CHOICE QA RESULTS:", 'current')
        print_log("-" * 60, 'current')
        print_log(f"{'Category':<25} {'Accuracy':<15} {'Count':<10}", 'current')
        print_log("-" * 60, 'current')
        
        category_counts = defaultdict(int)
        for result in results:
            category_counts[result.get('category', 'Unknown')] += 1
        
        # Print per-category results
        for key, value in all_metrics.items():
            if not (key.startswith('eval/mcqa_') and key.endswith('_accuracy') and 'overall' not in key):
                continue
                
            cat_part = key.replace('eval/mcqa_', '').replace('_accuracy', '')
            original_cat = self._find_original_category(cat_part, category_counts.keys())
            
            if original_cat:
                count = category_counts[original_cat]
                print_log(f"{original_cat:<25} {value:<15.4f} {count:<10}", 'current')
        
        # Print overall summary
        if 'eval/mcqa_overall_accuracy' in all_metrics:
            total_count = sum(category_counts.values())
            print_log("-" * 60, 'current')
            print_log(f"{'OVERALL':<25} {all_metrics['eval/mcqa_overall_accuracy']:<15.4f} {total_count:<10}", 'current')
        
        print_log("-" * 60, 'current')

    def _print_regression_summary(self, all_metrics: Dict[str, float], results: List[Dict]) -> None:
        """Print regression metrics summary by category."""
        print_log("\nREGRESSION RESULTS:", 'current')
        print_log("-" * 100, 'current')
        print_log(f"{'Category':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Pearson':<10} {'Spearman':<10} {'Count':<10}", 'current')
        print_log("-" * 100, 'current')
        
        category_counts = defaultdict(int)
        for result in results:
            category_counts[result.get('category', 'regression')] += 1
        
        # Extract categories directly from metric keys, excluding overall
        categories_with_metrics = {}
        for key, value in all_metrics.items():
            if key.startswith('eval/reg_') and not key.startswith('eval/reg_overall_') and key.endswith('_rmse'):
                # Extract category from key like 'eval/reg_hrd_regression_rmse' -> 'hrd_regression'
                cat_part = key.replace('eval/reg_', '').replace('_rmse', '')
                categories_with_metrics[cat_part] = True
        
        # Print per-category results
        for cat_part in sorted(categories_with_metrics.keys()):
            original_cat = self._find_original_category(cat_part, category_counts.keys()) or cat_part.replace('_', ' ').title()
            metrics = self._get_regression_metrics(all_metrics, cat_part)
            count = category_counts.get(original_cat, 0)
            
            print_log(f"{original_cat:<20} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f} "
                     f"{metrics['pearson']:<10.4f} {metrics['spearman']:<10.4f} {count:<10}", 'current')
        
        # Print overall summary
        if any(k.startswith('eval/reg_overall_') for k in all_metrics):
            overall = self._get_regression_metrics(all_metrics, 'overall')
            total_count = sum(category_counts.values())
            print_log("-" * 100, 'current')
            print_log(f"{'OVERALL':<20} {overall['rmse']:<10.4f} {overall['mae']:<10.4f} {overall['r2']:<10.4f} "
                     f"{overall['pearson']:<10.4f} {overall['spearman']:<10.4f} {total_count:<10}", 'current')
        
        print_log("-" * 100, 'current')

    def _print_survival_summary_detailed(self, all_metrics: Dict[str, float], results: List[Dict]) -> None:
        """Print survival metrics summary by category."""
        print_log("\nSURVIVAL ANALYSIS RESULTS:", 'current')
        print_log("-" * 90, 'current')
        print_log(f"{'Category':<20} {'C-Index':<12} {'IBS':<12} {'Time-AUC':<12} {'Count':<10}", 'current')
        print_log("-" * 90, 'current')
        
        category_counts = defaultdict(int)
        for result in results:
            category_counts[result.get('category', 'Survival')] += 1
        
        # Extract categories directly from metric keys, excluding overall
        categories_with_metrics = {}
        for key, value in all_metrics.items():
            if key.startswith('eval/surv_') and not key.startswith('eval/surv_overall_') and key.endswith('_c_index'):
                # Extract category from key like 'eval/surv_survival_os_c_index' -> 'survival_os'
                cat_part = key.replace('eval/surv_', '').replace('_c_index', '')
                categories_with_metrics[cat_part] = True
        
        # Print per-category results
        for cat_part in sorted(categories_with_metrics.keys()):
            original_cat = self._find_original_category(cat_part, category_counts.keys()) or cat_part.replace('_', ' ').title()
            metrics = self._get_survival_metrics(all_metrics, cat_part)
            count = category_counts.get(original_cat, 0)
            
            print_log(f"{original_cat:<20} {metrics['c_index']:<12.4f} {metrics['ibs']:<12.4f} "
                     f"{metrics['auc']:<12.4f} {count:<10}", 'current')
        
        # Print overall summary
        if any(k.startswith('eval/surv_overall_') for k in all_metrics):
            overall = self._get_survival_metrics(all_metrics, 'overall')
            total_count = sum(category_counts.values())
            print_log("-" * 90, 'current')
            print_log(f"{'OVERALL':<20} {overall['c_index']:<12.4f} {overall['ibs']:<12.4f} "
                     f"{overall['auc']:<12.4f} {total_count:<10}", 'current')
        
        print_log("-" * 90, 'current')

    def _find_original_category(self, cat_part: str, category_names) -> Optional[str]:
        """Find original category name from safe category part."""
        cat_part_lower = cat_part.lower()
        
        for cat in category_names:
            # Exact match after normalization
            safe_cat = cat.lower().replace(' ', '_').replace('/', '_')
            if safe_cat == cat_part_lower:
                return cat
                
            # Try removing common prefixes/suffixes for partial matching
            # For example: "Survival OS" -> "survival_os", but metric might be "survival_os" or just "os"
            if cat_part_lower in safe_cat or safe_cat in cat_part_lower:
                return cat
                
        return None

    def _extract_categories_from_metrics(self, all_metrics: Dict[str, float], prefix: str, 
                                       exclude_prefix: str, metric_offset: int = -1) -> set:
        """Extract category parts from metric keys."""
        categories = set()
        for key in all_metrics.keys():
            if key.startswith(prefix) and not key.startswith(exclude_prefix):
                parts = key.split('_')
                if len(parts) >= 3:
                    # Extract everything between prefix and metric name
                    # For 'eval/reg_hrd_regression_rmse', we want 'hrd_regression'
                    # For 'eval/surv_survival_os_c_index', we want 'survival_os'  
                    if metric_offset == -1:
                        # Default: take everything except the last part (metric name)
                        cat_part = '_'.join(parts[2:-1])
                    elif metric_offset == -2:
                        # For survival: take everything except last 2 parts (compound metric names)
                        cat_part = '_'.join(parts[2:-2])
                    else:
                        cat_part = '_'.join(parts[2:metric_offset])
                    
                    if cat_part:  # Only add non-empty categories
                        categories.add(cat_part)
        return categories

    def _get_regression_metrics(self, all_metrics: Dict[str, float], cat_part: str) -> Dict[str, float]:
        """Get regression metrics for a specific category."""
        prefix = f'eval/reg_{cat_part}_' if cat_part != 'overall' else 'eval/reg_overall_'
        return {
            'rmse': all_metrics.get(f'{prefix}rmse', 0.0),
            'mae': all_metrics.get(f'{prefix}mae', 0.0),
            'r2': all_metrics.get(f'{prefix}r2', 0.0),
            'pearson': all_metrics.get(f'{prefix}pearson', 0.0),
            'spearman': all_metrics.get(f'{prefix}spearman', 0.0)
        }

    def _get_survival_metrics(self, all_metrics: Dict[str, float], cat_part: str) -> Dict[str, float]:
        """Get survival metrics for a specific category."""
        prefix = f'eval/surv_{cat_part}_' if cat_part != 'overall' else 'eval/surv_overall_'
        return {
            'c_index': all_metrics.get(f'{prefix}c_index', 0.0),
            'ibs': all_metrics.get(f'{prefix}integrated_brier_score', 0.0),
            'auc': all_metrics.get(f'{prefix}time_dependent_auc', 0.0)
        }

    def _print_survival_summary(self) -> None:
        """Print survival metrics in table format (legacy method)."""
        if not hasattr(self, '_survival_metrics') or not self._survival_metrics:
            return
            
        print_log("\nSURVIVAL ANALYSIS RESULTS:", 'current')
        print_log("-" * 60, 'current')
        print_log(f"{'Metric':<25} {'Value':<15}", 'current')
        print_log("-" * 60, 'current')
        
        metric_names = {
            'c_index': 'C-Index',
            'integrated_brier_score': 'Integrated Brier Score', 
            'time_dependent_auc': 'Time-dependent AUC'
        }
        
        for key, name in metric_names.items():
            if key in self._survival_metrics:
                value = self._survival_metrics[key]
                if isinstance(value, float):
                    print_log(f"{name:<25} {value:<15.4f}", 'current')
                else:
                    print_log(f"{name:<25} {value:<15}", 'current')
        print_log("-" * 60, 'current')


class SurvivalMetrics:
    """Survival analysis metrics: C-index, Integrated Brier Score, time-dependent AUC."""
    
    def __init__(self, time_intervals: np.ndarray):
        """Initialize with time interval boundaries (len = n+1)."""
        self.time_intervals = time_intervals
        self.interval_midpoints = (time_intervals[:-1] + time_intervals[1:]) / 2
        # Use right endpoints to ensure all times are within bounds
        self.interval_endpoints = time_intervals[1:]
        # Optional training data for IPCW estimation in IBS calculation
        self._train_event_times: Optional[np.ndarray] = None
        self._train_event_indicators: Optional[np.ndarray] = None

    def set_training_data(self, event_times: np.ndarray, event_indicators: np.ndarray) -> None:
        """Set training data for IPCW estimation in IBS calculation."""
        self._train_event_times = np.asarray(event_times, dtype=float)
        self._train_event_indicators = np.asarray(event_indicators, dtype=float)

    def load_training_data_from_json(self, path: str) -> int:
        """Load survival samples from JSON file and set as training data.
        
        Returns:
            Number of loaded samples.
        """
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        event_times, event_indicators = [], []
        n_intervals = len(self.interval_endpoints)

        for sample in samples:
            category = sample.get('category', '')
            if not category.startswith('Survival'):
                continue
            
            survival_targets = sample.get('survival_targets')
            if not survival_targets:
                continue
                
            target_y = list(survival_targets.get('target_y', []))
            at_risk_mask = list(survival_targets.get('at_risk_mask', []))
            
            if len(target_y) != n_intervals or len(at_risk_mask) != n_intervals:
                continue
            
            # Determine event time and indicator
            if any(v == 1 for v in target_y):
                k = target_y.index(1)
                event = 1.0
            else:
                at_risk_indices = [i for i, v in enumerate(at_risk_mask) if v == 1]
                k = max(at_risk_indices) if at_risk_indices else 0
                event = 0.0
            
            k = min(k, n_intervals - 1)
            time = float(self.interval_endpoints[k])
            event_times.append(time)
            event_indicators.append(event)

        if event_times:
            self.set_training_data(np.array(event_times), np.array(event_indicators))
        
        return len(event_times)
    
    def compute_concordance_index(self,
                                  risk_scores: np.ndarray,
                                  event_times: np.ndarray,
                                  event_indicators: np.ndarray) -> float:
        """Harrell's C-index via lifelines (assumed available)."""
        from lifelines.utils import concordance_index
        # lifelines interprets smaller values as higher risk -> negate scores
        return float(concordance_index(event_times, -risk_scores, event_indicators))
    
    def compute_integrated_brier_score(self,
                                        survival_probs: np.ndarray,
                                        event_times: np.ndarray,
                                        event_indicators: np.ndarray,
                                        max_time: Optional[float] = None) -> float:
        """Integrated Brier Score (IBS) using sksurv.

        - Aligns predicted survival probability grid to interval endpoints or
          midpoints; interpolates if necessary.
        - Uses only evaluation times strictly within follow-up (< max observed time)
          and optionally <= max_time.
        - Returns NaN if no events or no valid evaluation times.
        """
        try:
            from sksurv.metrics import integrated_brier_score
            from sksurv.util import Surv
            import warnings
        except ImportError:
            return float('nan')

        # Test (evaluation) data
        y_test = Surv.from_arrays(event_indicators.astype(bool), event_times)
        if not y_test['event'].any():  # no events -> IBS undefined
            return float('nan')

        # Training (reference) data for IPCW; fallback to test set if not provided
        if self._train_event_times is not None and self._train_event_indicators is not None:
            y_train = Surv.from_arrays(self._train_event_indicators.astype(bool), self._train_event_times)
            # If training set has no events, fallback gracefully
            if not y_train['event'].any():
                y_train = y_test
        else:
            y_train = y_test

        n_times_pred = survival_probs.shape[1]
        # Prefer endpoints (right boundaries), fallback to midpoints; else interpolate to endpoints
        candidate_grids = [self.interval_endpoints, self.interval_midpoints]
        eval_times = None
        for grid in candidate_grids:
            if len(grid) == n_times_pred:
                eval_times = grid
                break
        if eval_times is None:  # interpolate to endpoints
            target = self.interval_endpoints
            orig_x = np.linspace(0, 1, n_times_pred)
            new_x = np.linspace(0, 1, len(target))
            survival_probs = np.vstack([np.interp(new_x, orig_x, row) for row in survival_probs])
            eval_times = target

        max_follow_up = float(np.max(event_times))  # right-open upper bound
        mask = eval_times < max_follow_up
        if max_time is not None:
            mask &= (eval_times <= max_time)
        if not mask.any():
            return float('nan')
        eval_times = eval_times[mask]
        survival_probs = survival_probs[:, mask]
        if eval_times.size == 0:
            return float('nan')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Correct usage: first argument = reference (training) data, second = evaluation (test) data
            try:
                ibs = integrated_brier_score(y_train, y_test, survival_probs, eval_times)
            except Exception:
                # Fallback: clip eval_times to strictly inside (min_evt, max_evt) if possible
                evt_times = y_test['time'][y_test['event']]
                if len(evt_times) == 0:
                    return float('nan')
                lo, hi = float(np.min(evt_times)), float(np.max(evt_times))
                tight_mask = (eval_times > lo) & (eval_times < hi)
                if tight_mask.any():
                    et2 = eval_times[tight_mask]
                    sp2 = survival_probs[:, tight_mask]
                    try:
                        ibs = integrated_brier_score(y_train, y_test, sp2, et2)
                    except Exception:
                        return float('nan')
                else:
                    return float('nan')
        return float(ibs)

    def compute_time_dependent_auc(self,
                                   risk_scores: np.ndarray,
                                   event_times: np.ndarray,
                                   event_indicators: np.ndarray,
                                   prediction_time: float) -> float:
        """Time-dependent AUC at a single prediction_time (sksurv)."""
        try:
            from sksurv.metrics import cumulative_dynamic_auc
            from sksurv.util import Surv
            import warnings
        except ImportError:
            return float('nan')

        y = Surv.from_arrays(event_indicators.astype(bool), event_times)
        if not y['event'].any():
            return float('nan')
        event_times_only = y['time'][y['event']]
        first_evt = event_times_only.min()
        last_evt = event_times_only.max()
        if not (first_evt <= prediction_time < last_evt):
            return float('nan')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            auc, _ = cumulative_dynamic_auc(y, y, risk_scores, [prediction_time])
        return float(auc[0])
    
    def compute_all_metrics(self,
                          survival_probs: np.ndarray,
                          risk_scores: np.ndarray, 
                          event_times: np.ndarray,
                          event_indicators: np.ndarray) -> Dict[str, float]:
        """
        Compute all survival metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # C-index
        metrics['c_index'] = self.compute_concordance_index(
            risk_scores, event_times, event_indicators
        )
        
        # Integrated Brier Score
        metrics['integrated_brier_score'] = self.compute_integrated_brier_score(
            survival_probs, event_times, event_indicators
        )
        
        # Time-dependent AUC at median time
        median_time = np.median(self.time_intervals)
        metrics['time_dependent_auc'] = self.compute_time_dependent_auc(
            risk_scores, event_times, event_indicators, median_time
        )
        
        return metrics