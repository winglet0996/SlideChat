import os
import re
import json
import torch
import numpy as np
from typing import Any, Sequence, Dict, List, Optional, Union
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
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
    SURVIVAL_METRICS = ['C-Index']
    
    # Pattern matching
    DIAGNOSIS_PATTERN = re.compile(r'Final diagnosis:\s*(.+?)(?:\n|$)', re.IGNORECASE | re.DOTALL)

    def __init__(self,
                 tokenizer: Union[Dict, Any],
                 output_dir: Optional[str] = None,
                 survival_time_intervals: Optional[List[float]] = None,
                 *args, **kwargs):
        """
        Initialize the multi-task pathology metric evaluator.

        Args:
            tokenizer: Configuration dict for building the tokenizer, or tokenizer object directly
            output_dir: Directory to save evaluation JSON results
            survival_time_intervals: Time intervals for survival analysis
        """
        super().__init__(*args, **kwargs)
        # Allow passing tokenizer object directly or build from config
        if isinstance(tokenizer, dict):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        self.output_dir = output_dir
        
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
        for i, sample in enumerate(data_samples):
            # Decode input and predictions
            # We only want the prompt part for the 'input' field in results.
            # The prompt is where labels are -100.
            labels = batch_data['labels'][i]
            input_ids = batch_data['input_ids'][i]
            attn_mask = batch_data['attention_mask'][i] if batch_data['attention_mask'] is not None else None

            # Find where the answer starts (first non-ignore label)
            non_ignore_mask = (labels != -100)
            if non_ignore_mask.any():
                answer_start = non_ignore_mask.nonzero()[0].item()
            else:
                answer_start = len(input_ids)
            
            # Extract prompt part and filter out padding using attention_mask
            prompt_ids = input_ids[:answer_start]
            if attn_mask is not None:
                prompt_attn_mask = attn_mask[:answer_start].to(torch.bool)
                prompt_ids = prompt_ids[prompt_attn_mask]

            input_str = self._decode_full_input(prompt_ids)
            
            # Handle different sample structures
            pred_str = self._extract_prediction_text(sample)
            
            # Extract metadata
            metadata = self._extract_sample_metadata(batch_data, i)
            
            # Determine task type and process accordingly
            task_type = self._determine_task_type(metadata, pred_str, sample, input_str)
            
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
            'attention_mask': data_batch['data'].get('attention_mask', None),
            'category': data_batch['data'].get('category', None),
            'image_file': data_batch['data'].get('image_file', None),
            'regression_targets': data_batch['data'].get('regression_targets', None),
            'survival_targets': data_batch['data'].get('survival_targets', None),
            'survival_times': data_batch['data'].get('survival_times', None),
            'survival_events': data_batch['data'].get('survival_events', None),
            'project': data_batch['data'].get('project', None),
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
        
        # Extract project
        if batch_data['project'] is not None:
            if isinstance(batch_data['project'], (list, tuple)):
                proj = batch_data['project'][idx] if idx < len(batch_data['project']) else ""
            else:
                proj = str(batch_data['project'])
            
            # Handle nested projects
            if isinstance(proj, (list, tuple)) and proj:
                proj = str(proj[0])
            metadata['project'] = str(proj) if proj else "Unknown"
        else:
            metadata['project'] = "Unknown"
        
        # Extract targets
        metadata['target_str'] = self._get_target_text(batch_data, idx)
        metadata['regression_target'] = self._get_regression_target(batch_data['regression_targets'], idx)
        metadata['survival_data'] = self._get_survival_data(batch_data, idx)
        
        # Extract filename
        metadata['filename'] = self._extract_filename_from_image_file(batch_data['image_file'], idx)
        
        return metadata

    def _determine_task_type(self, metadata: Dict[str, Any], pred_str: str, sample: Dict, input_str: str = "") -> str:
        """Determine task type using explicit model fields and category."""
        if 'survival_prediction' in sample:
            return 'survival'
        if 'regression_prediction' in sample:
            return 'regression'
            
        # 1. Check for MCQA markers in input or target (most reliable)
        target_str = metadata.get('target_str', '') or ""
        if '<CHOICES>' in input_str or self._extract_mcqa_choice(target_str):
            return 'mcqa'

        category = metadata.get('category', '').lower()
        
        # 2. Check for explicit task markers in category
        if 'mcqa' in category:
            return 'mcqa'
        if any(kw in category for kw in ['text', 'caption', 'report', 'diagnosis', 'generation']):
            return 'text'
            
        # Default to text for safety if it doesn't look like MCQA
        return 'text'

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
            'project': metadata['project'],
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
            'project': metadata['project'],
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
        
        # If target_choice is empty, it's likely not an MCQA task or ground truth is broken
        if not target_choice:
            return
        
        # If pred_choice is empty, it's a model failure to follow format, count as incorrect
        # but DO NOT drop the sample from evaluation
        correct = (pred_choice == target_choice) if pred_choice else False

        # Optional: capture model-side choice logits for AUROC (binary K=2)
        choice_logits = None
        if isinstance(sample, dict):
            choice_logits = (sample.get('mcqa_choice_logits')
                             or sample.get('choice_logits')
                             or sample.get('mcqa_logits'))
        if choice_logits is not None and not isinstance(choice_logits, dict):
            choice_logits = None
        
        self.results.append({
            'task_type': 'mcqa',
            'filename': metadata['filename'],
            'project': metadata['project'],
            'input': input_str.strip(),
            'prediction': pred_str.strip(),
            'pred_choice': pred_choice,
            'target_choice': target_choice,
            'category': metadata['category'],
            'correct': correct,
            'choice_logits': choice_logits
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
            'project': metadata['project'],
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

        Text, MCQA, and regression metrics are aggregated by category with overall summaries,
        while survival metrics continue to report per-project and per-category breakdowns.
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
            all_metrics.update(self._compute_text_metrics_by_category(task_results['text']))

        if task_results['mcqa']:
            all_metrics.update(self._compute_mcqa_metrics(task_results['mcqa']))

        if task_results['regression']:
            all_metrics.update(self._compute_regression_metrics(task_results['regression']))

        if task_results['survival'] and self.survival_metrics:
            all_metrics.update(self._compute_survival_metrics_with_projects(task_results['survival']))

        # Save results and print summaries
        if is_main_process() and self.output_dir:
            self._save_all_results(task_results)
            self._print_metric_summaries(all_metrics)

        return all_metrics
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

    @staticmethod
    def _make_safe_key(value: str, fallback: str = 'uncategorized') -> str:
        """Convert arbitrary string into a safe lowercase identifier."""
        if value is None:
            value = ''
        safe = re.sub(r'[^0-9a-zA-Z]+', '_', value.strip()).strip('_').lower()
        return safe or fallback

    def _extract_mcqa_choice(self, text: str) -> str:
        """Extract choice letter from MCQA response.
        
        Supports formats like:
        - "A"
        - "A) Positive"
        - "A: Positive"
        - "A. Positive"
        - "<SRV>1" (returns as is)
        """
        if not text:
            return ""
        
        text = text.strip()
        
        # Handle special tokens <SRV> and <REG> as single tokens
        if text.startswith(('<SRV>', '<REG>')):
            return text
            
        # Direct single letter match
        if len(text) == 1 and text.isalpha():
            return text.upper()
        
        # Extract from patterns like "A)", "A:", "A."
        # We look for a letter followed by punctuation or space at the start
        # or a letter followed by punctuation anywhere.
        patterns = [
            r'^([A-Z])[\)\.\:\s]',  # "A)", "A.", "A:", "A " at start
            r'([A-Z])[\)\.\:]',     # "A)", "A.", "A:" anywhere
            r'^([A-Z])$',           # "A" at start and end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
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
        """Prefer continuous survival_times/survival_events; fallback to survival_targets if needed."""
        # 1) Best: use continuous labels if provided
        times = batch_data.get("survival_times", None)
        events = batch_data.get("survival_events", None)
        if times is not None and events is not None:
            try:
                t = times[idx] if isinstance(times, (list, tuple)) else times[idx]
                e = events[idx] if isinstance(events, (list, tuple)) else events[idx]
                # torch / numpy / python scalar all ok
                if isinstance(t, torch.Tensor):
                    t = float(t.detach().cpu().item())
                else:
                    t = float(t)
                if isinstance(e, torch.Tensor):
                    e = float(e.detach().cpu().item())
                else:
                    e = float(e)
                return {"time": t, "event": e}
            except Exception:
                pass  # fallback below
        # 2) Fallback: reconstruct from discretized survival_targets (less ideal)
        st = batch_data.get("survival_targets", None)
        if st is None:
            return None
        if self.survival_metrics is None:
            # if still not initialized, set default/fallback intervals
            if self._survival_intervals is None:
                self._survival_intervals = np.array([0, 1, 2, 3, 5, 7, 10], dtype=float)
            self.survival_metrics = SurvivalMetrics(self._survival_intervals)
        target_y = st["target_y"][idx]
        at_risk_mask = st["at_risk_mask"][idx]
        if isinstance(target_y, torch.Tensor):
            target_y = target_y.detach().cpu().numpy()
        if isinstance(at_risk_mask, torch.Tensor):
            at_risk_mask = at_risk_mask.detach().cpu().numpy()
        if (target_y > 0.5).any():
            k = int(np.argmax(target_y))
            event = 1.0
        else:
            at_risk_indices = np.where(at_risk_mask > 0.5)[0]
            k = int(at_risk_indices.max()) if len(at_risk_indices) > 0 else 0
            event = 0.0
        k = min(k, len(self.survival_metrics.interval_endpoints) - 1)
        time = float(self.survival_metrics.interval_endpoints[k])  # right endpoint
        return {"time": time, "event": event}

    def _extract_filename_from_image_file(self, image_file_batch: Any, idx: int) -> str:
        """Extract filename from image file batch."""
        if image_file_batch is None:
            return f"sample_{idx}"
        
        try:
            if isinstance(image_file_batch, (list, tuple)):
                filename = image_file_batch[idx] if idx < len(image_file_batch) else f"sample_{idx}"
            else:
                filename = image_file_batch
            
            # Handle list of filenames (e.g. from multi-image datasets)
            if isinstance(filename, (list, tuple)) and len(filename) > 0:
                filename = filename[0]
            
            return os.path.basename(str(filename)) if filename else f"sample_{idx}"
        except (IndexError, TypeError):
            return f"sample_{idx}"

    def _compute_text_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute text generation metrics (legacy method, kept for compatibility)."""
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
        metrics = {}
        for name, scores in metrics_scores.items():
            if scores:
                mean, _, _ = self._bootstrap_ci(scores)
                metrics[name] = mean

        # Diagnosis accuracy
        if diagnosis_accuracy:
            diag_acc_mean, _, _ = self._bootstrap_ci(diagnosis_accuracy)
            metrics['diag_accuracy'] = diag_acc_mean

        return metrics

    def _compute_text_metrics_by_category(self, results: List[Dict]) -> Dict[str, float]:
        """Compute text metrics grouped by category with overall summary."""
        if not results:
            self._text_metrics_table = []
            return {}

        category_groups = defaultdict(list)
        for res in results:
            category = res.get('category') or 'Uncategorized'
            category_groups[category].append(res)

        metrics = {}
        table = []

        for category in sorted(category_groups.keys()):
            group = category_groups[category]
            cat_metrics = self._compute_text_metrics(group)
            safe_cat = self._make_safe_key(category)
            for metric_name, value in cat_metrics.items():
                metrics[f'eval/text_{safe_cat}_{metric_name}'] = value
            table.append({'category': category, 'metrics': cat_metrics, 'count': len(group)})

        overall_metrics = self._compute_text_metrics(results)
        for metric_name, value in overall_metrics.items():
            metrics[f'eval/text_overall_{metric_name}'] = value
        table.append({'category': 'Overall', 'metrics': overall_metrics, 'count': len(results)})

        self._text_metrics_table = table
        return metrics

    def _extract_choices_block(self, input_text: str) -> str:
        """Extract the raw <CHOICES>...</CHOICES> block from an input string."""
        if not input_text:
            return ""
        for pattern in [
            r'<CHOICES>(.*?)</CHOICES>',
            r'<CHOICES>(.*?)(?:assistant|$)',
        ]:
            match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""


    def _extract_choices_from_input(self, input_text: str) -> frozenset:
        """Extract actual choice content from MCQA input text."""
        choices_text = self._extract_choices_block(input_text)
        if choices_text:
            # Extract choice content after A), B), C), etc.
            choice_contents = re.findall(r'^[A-Z]\)\s*(.+?)$', choices_text, re.MULTILINE)
            return frozenset(c.strip() for c in choice_contents)
        return frozenset()

    def _map_choice_to_content(self, choice_letter: str, input_text: str) -> str:
        """Map choice letter (A/B/C/D/E) to actual choice content."""
        if not choice_letter:
            return choice_letter
        
        # Normalize choice letter to uppercase
        choice_letter = choice_letter.strip().upper()
        
        # Extract choices block
        choices_text = self._extract_choices_block(input_text)
        if choices_text:
            # Try multiple patterns to match the choice
            patterns = [
                rf'^{re.escape(choice_letter)}\)\s*(.+?)$',  # A) content
                rf'^{re.escape(choice_letter)}\.\s*(.+?)$',  # A. content
                rf'^{re.escape(choice_letter)}:\s*(.+?)$',   # A: content
                rf'^{re.escape(choice_letter)}\s+(.+?)$',    # A content
            ]
            
            for pattern in patterns:
                content_match = re.search(pattern, choices_text, re.MULTILINE)
                if content_match:
                    mapped_content = content_match.group(1).strip()
                    # Remove trailing punctuation/newlines
                    mapped_content = re.sub(r'[\n\r]+.*$', '', mapped_content).strip()
                    return mapped_content
        
        # Fallback: return original letter if mapping fails
        return choice_letter

    def _infer_num_choices(self, input_text: str) -> int:
        """Infer number of options (K) for a MCQA question from its input."""
        block = self._extract_choices_block(input_text)
        if not block:
            return 0
        raw_letters = re.findall(r'^\s*([A-E])[\)\.:]', block, flags=re.MULTILINE | re.IGNORECASE)
        letters = {self._extract_mcqa_choice(x) for x in raw_letters}
        letters.discard('')
        if letters:
            return len(letters)
        # Fallback: count extracted contents
        contents = self._extract_choices_from_input(input_text)
        return len(contents)

    def _compute_mcqa_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute MCQA metrics per category.

        Rules:
                - If K=2: compute weighted-F1, BACC, and AUROC (if choice logits exist) + ACC.
                - If K>2:
                    - If all questions share the same global choice set (same choice contents): compute weighted-F1 + BACC + ACC.
                    - Otherwise: ACC only.
        """
        if not results:
            self._mcqa_metrics_table = []
            return {}
        
        # Group by category: collect predictions, targets, and inputs
        category_groups = defaultdict(lambda: {'preds': [], 'targets': [], 'inputs': [], 'choice_logits': []})
        
        for res in results:
            pred_choice = res.get('pred_choice', '')
            target_choice = res.get('target_choice', '')
            category = res.get('category', 'Unknown')
            input_text = res.get('input', '')
            
            if pred_choice and target_choice:
                category_groups[category]['preds'].append(pred_choice)
                category_groups[category]['targets'].append(target_choice)
                category_groups[category]['inputs'].append(input_text)
                category_groups[category]['choice_logits'].append(res.get('choice_logits'))
        
        # Compute per-category metrics
        engine_metrics = {}
        table = []

        # Overall summaries (kept simple and explicit)
        total_correct = 0
        total_samples = 0
        auroc_weighted = []  # (auroc, n)
        wf1_weighted = []    # (weighted_f1, n)
        bacc_weighted = []   # (bacc, n)

        for cat in sorted(category_groups.keys()):
            group = category_groups[cat]
            if len(group['preds']) > 0:
                sample_count = len(group['preds'])
                total_samples += sample_count
                cat_correct = int(np.sum([p == t for p, t in zip(group['preds'], group['targets'])]))
                total_correct += cat_correct
                acc = float(cat_correct / sample_count) if sample_count > 0 else 0.0

                # Infer option count K (best-effort; fall back to observed label space)
                ks = [self._infer_num_choices(inp) for inp in group['inputs']]
                known_ks = [k for k in ks if k > 0]
                observed_labels = sorted({x for x in (group['targets'] + group['preds']) if x})
                if known_ks:
                    is_binary = all(k == 2 for k in known_ks)
                    has_multichoice = any(k > 2 for k in known_ks)
                else:
                    is_binary = (len(observed_labels) == 2)
                    has_multichoice = False

                # If any sample indicates K>2, do not treat as binary
                all_binary = bool(is_binary and not has_multichoice)

                safe_cat = self._make_safe_key(cat, fallback='unknown')

                if all_binary:
                    # Weighted-F1 + BACC on observed labels
                    weighted_f1 = None
                    bacc = None
                    if observed_labels:
                        try:
                            weighted_f1 = float(f1_score(group['targets'], group['preds'], average='weighted', labels=observed_labels))
                        except Exception:
                            weighted_f1 = 0.0
                        try:
                            bacc = float(balanced_accuracy_score(group['targets'], group['preds']))
                        except Exception:
                            bacc = None

                    # AUROC: use logits for the two labels when available (pos label defaults to 'A' if present)
                    auroc = None
                    if len(observed_labels) == 2:
                        pos_label = 'A' if 'A' in observed_labels else observed_labels[0]
                        neg_label = observed_labels[1] if observed_labels[0] == pos_label else observed_labels[0]
                        y_true, y_score = [], []
                        for t, logits in zip(group['targets'], group['choice_logits']):
                            if t not in (pos_label, neg_label):
                                continue
                            if not isinstance(logits, dict) or pos_label not in logits or neg_label not in logits:
                                continue
                            try:
                                y_true.append(1 if t == pos_label else 0)
                                # Use a robust difference that handles -inf to avoid NaN and allow ranking
                                l_pos, l_neg = float(logits[pos_label]), float(logits[neg_label])
                                if l_pos == -float('inf') and l_neg == -float('inf'):
                                    y_score.append(0.0)
                                elif l_pos == -float('inf'):
                                    y_score.append(-10000.0)
                                elif l_neg == -float('inf'):
                                    y_score.append(10000.0)
                                else:
                                    y_score.append(l_pos - l_neg)
                            except Exception:
                                continue
                        if len(set(y_true)) >= 2 and len(y_true) >= 2:
                            try:
                                auroc = float(roc_auc_score(y_true, y_score))
                            except Exception:
                                auroc = None

                    # Emit metrics
                    engine_metrics[f'eval/mcqa_{safe_cat}_acc'] = acc
                    if isinstance(weighted_f1, (float, int)):
                        engine_metrics[f'eval/mcqa_{safe_cat}_weighted_f1'] = float(weighted_f1)
                        wf1_weighted.append((float(weighted_f1), sample_count))
                    if isinstance(bacc, (float, int)):
                        engine_metrics[f'eval/mcqa_{safe_cat}_bacc'] = float(bacc)
                        bacc_weighted.append((float(bacc), sample_count))
                    if auroc is not None:
                        engine_metrics[f'eval/mcqa_{safe_cat}_auroc'] = float(auroc)
                        auroc_weighted.append((float(auroc), sample_count))

                    table.append({
                        'category': cat,
                        'metrics': {
                            'primary': 'AUROC',
                            'auroc': float(auroc) if auroc is not None else None,
                            'weighted_f1': float(weighted_f1) if isinstance(weighted_f1, (float, int)) else None,
                            'bacc': float(bacc) if isinstance(bacc, (float, int)) else None,
                            'acc': acc,
                            'K': 2,
                            'has_choice_logits': auroc is not None,
                        },
                        'count': sample_count
                    })
                    continue

                # K>2 path: check whether all samples share the same choice set
                choice_sets = [self._extract_choices_from_input(inp) for inp in group['inputs']]
                unique_choice_sets = set(choice_sets)
                has_global_label_space = (len(unique_choice_sets) == 1 and unique_choice_sets != {frozenset()})

                if has_global_label_space:
                    global_choices = sorted(list(next(iter(unique_choice_sets))))
                    idx_map = {c: i for i, c in enumerate(global_choices)}
                    mapped_preds, mapped_targets = [], []
                    mapping_failed = False

                    for i in range(sample_count):
                        mp = self._map_choice_to_content(group['preds'][i], group['inputs'][i])
                        mt = self._map_choice_to_content(group['targets'][i], group['inputs'][i])
                        if mp not in idx_map or mt not in idx_map:
                            mapping_failed = True
                            break
                        mapped_preds.append(idx_map[mp])
                        mapped_targets.append(idx_map[mt])

                    if not mapping_failed and mapped_targets:
                        try:
                            weighted_f1 = float(f1_score(mapped_targets, mapped_preds, average='weighted'))
                        except Exception:
                            weighted_f1 = 0.0
                        try:
                            bacc = float(balanced_accuracy_score(mapped_targets, mapped_preds))
                        except Exception:
                            bacc = None

                        engine_metrics[f'eval/mcqa_{safe_cat}_weighted_f1'] = float(weighted_f1)
                        wf1_weighted.append((float(weighted_f1), sample_count))
                        if isinstance(bacc, (float, int)):
                            engine_metrics[f'eval/mcqa_{safe_cat}_bacc'] = float(bacc)
                            bacc_weighted.append((float(bacc), sample_count))
                        engine_metrics[f'eval/mcqa_{safe_cat}_acc'] = acc
                        table.append({
                            'category': cat,
                            'metrics': {
                                'primary': 'Weighted-F1',
                                'weighted_f1': float(weighted_f1),
                                'bacc': float(bacc) if isinstance(bacc, (float, int)) else None,
                                'acc': acc,
                                'K': (max(known_ks) if known_ks else (len(global_choices) if global_choices else None)),
                                'global_label_space': True,
                            },
                            'count': sample_count
                        })
                        continue

                # Fallback: ACC only
                engine_metrics[f'eval/mcqa_{safe_cat}_acc'] = acc
                table.append({
                    'category': cat,
                    'metrics': {
                        'primary': 'ACC',
                        'acc': acc,
                        'K': max(known_ks) if known_ks else None,
                        'global_label_space': False,
                    },
                    'count': sample_count
                })

        # Overall metrics
        overall_acc = float(total_correct / total_samples) if total_samples > 0 else 0.0
        engine_metrics['eval/mcqa_overall_acc'] = overall_acc

        overall_auroc = None
        if auroc_weighted:
            denom = sum(n for _, n in auroc_weighted)
            if denom > 0:
                overall_auroc = float(sum(v * n for v, n in auroc_weighted) / denom)
                engine_metrics['eval/mcqa_overall_auroc'] = overall_auroc

        overall_weighted_f1 = None
        if wf1_weighted:
            denom = sum(n for _, n in wf1_weighted)
            if denom > 0:
                overall_weighted_f1 = float(sum(v * n for v, n in wf1_weighted) / denom)
                engine_metrics['eval/mcqa_overall_weighted_f1'] = overall_weighted_f1

        overall_bacc = None
        if bacc_weighted:
            denom = sum(n for _, n in bacc_weighted)
            if denom > 0:
                overall_bacc = float(sum(v * n for v, n in bacc_weighted) / denom)
                engine_metrics['eval/mcqa_overall_bacc'] = overall_bacc

        table.append({
            'category': 'Overall',
            'metrics': {
                'primary': 'ACC',
                'acc': overall_acc,
                'auroc_weighted': overall_auroc,
                'weighted_f1_weighted': overall_weighted_f1,
                'bacc_weighted': overall_bacc,
            },
            'count': total_samples
        })

        self._mcqa_metrics_table = table
        
        return engine_metrics


    def _compute_regression_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute regression metrics per category and overall (legacy method)."""
        if not results:
            return {}
            
        # Group results by category
        cat_groups = defaultdict(list)
        for r in results:
            category = r.get('category') or 'Regression'
            cat_groups[category].append(r)

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
        table = []
        for cat in sorted(cat_groups.keys()):
            group = cat_groups[cat]
            display_cat = cat if cat else 'Regression'
            metrics = compute_category_metrics(group)
            if metrics:
                safe_cat = self._make_safe_key(display_cat, fallback='regression')
                for metric_name, value in metrics.items():
                    engine_metrics[f'eval/reg_{safe_cat}_{metric_name}'] = float(value)
                table.append({'category': display_cat, 'metrics': metrics, 'count': len(group)})
        
        # Compute overall metrics
        overall_metrics = compute_category_metrics(results)
        if overall_metrics:
            for metric_name, value in overall_metrics.items():
                engine_metrics[f'eval/reg_overall_{metric_name}'] = float(value)
            table.append({'category': 'Overall', 'metrics': overall_metrics, 'count': len(results)})

        self._regression_metrics_table = table
                
        return engine_metrics


    def _compute_survival_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute survival metrics per category and overall (legacy method)."""
        if not self.survival_metrics or not results:
            return {}
            
        # Group by category
        cat_groups = defaultdict(list)
        for r in results:
            cat_groups[r.get('category', 'Survival')].append(r)
            
        def compute_category_survival_metrics(group: List[Dict]) -> Optional[Dict[str, float]]:
            """Compute survival metrics for a single category group."""
            risk_scores, event_times, event_indicators = [], [], []
            for g in group:
                rs = g.get('risk_score')
                et = g.get('event_time')
                ei = g.get('event_indicator')
                if rs is None or et is None or ei is None:
                    continue
                risk_scores.append(float(rs))
                event_times.append(float(et))
                event_indicators.append(float(ei))

            if not risk_scores:
                return None

            risk_scores = np.array(risk_scores, dtype=float)
            event_times = np.array(event_times, dtype=float)
            event_indicators = np.array(event_indicators, dtype=float)

            try:
                c_index = self.survival_metrics.compute_concordance_index(
                    risk_scores, event_times, event_indicators)
            except Exception as e:
                print_log(f'Survival C-index failed for category computation: {e}', 'current')
                return None

            return {'c_index': float(c_index)}
        
        engine_metrics = {}
        
        # Compute per-category metrics
        for cat, group in cat_groups.items():
            metrics = compute_category_survival_metrics(group)
            if not metrics:
                continue
            safe_cat = cat.lower().replace(' ', '_').replace('/', '_')
            engine_metrics[f'eval/surv_{safe_cat}_c_index'] = metrics['c_index']
        
        # Compute overall metrics
        overall_metrics = compute_category_survival_metrics(results)
        if overall_metrics:
            engine_metrics['eval/surv_overall_c_index'] = overall_metrics['c_index']
            self._survival_metrics = overall_metrics
        
        return engine_metrics

    def _compute_survival_metrics_with_projects(self, results: List[Dict]) -> Dict[str, float]:
        """Compute survival metrics per category for each project and aggregate across projects."""
        if not self.survival_metrics or not results:
            return {}
        
        # Group by project
        project_results = defaultdict(list)
        for res in results:
            project = res.get('project', 'Unknown')
            project_results[project].append(res)
        
        all_metrics = {}
        # Structure: category -> list of (c_index, count) from each project
        category_metrics_across_projects = defaultdict(list)
        
        # Compute per-project, per-category metrics
        for project, proj_results in project_results.items():
            metrics = self._compute_single_project_survival_metrics(proj_results)
            safe_proj = project.replace('-', '_').replace(' ', '_')
            
            # Add per-project per-category metrics
            if metrics:
                for key, value in metrics.items():
                    all_metrics[f'eval/surv_{safe_proj}_{key}'] = value
                    # Track category metrics for cross-project aggregation
                    # metrics keys are like: 'survival_os_c_index', 'mutation_c_index', etc.
                    if key.endswith('_c_index'):
                        category_name = key[:-8]  # Remove '_c_index' suffix
                        # Count samples in this category for this project
                        cat_count = sum(1 for r in proj_results if self._make_safe_key(r.get('category', '')) == category_name)
                        category_metrics_across_projects[category_name].append((value, cat_count))
        
        # Compute weighted average across projects for each category
        for category, metrics_list in category_metrics_across_projects.items():
            if metrics_list:
                total_samples = sum(count for _, count in metrics_list)
                if total_samples > 0:
                    weighted_c_index = sum(c_index * count for c_index, count in metrics_list) / total_samples
                    all_metrics[f'eval/surv_overall_{category}_c_index'] = float(weighted_c_index)
        
        return all_metrics
    
    def _compute_single_project_survival_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Compute survival metrics per category for a single project (no overall metric).
        
        Returns per-category C-index metrics only.
        """
        # Group by category
        category_data = defaultdict(lambda: {'risk_scores': [], 'event_times': [], 'event_indicators': []})
        
        for res in results:
            rs = res.get('risk_score')
            et = res.get('event_time')
            ei = res.get('event_indicator')
            if rs is None or et is None or ei is None:
                continue
            
            # Group by category
            category = res.get('category', 'Unknown')
            category_data[category]['risk_scores'].append(float(rs))
            category_data[category]['event_times'].append(float(et))
            category_data[category]['event_indicators'].append(float(ei))

        if not category_data:
            return {}

        metrics = {}
        
        # Compute per-category C-index only (no overall)
        for category, cat_data in category_data.items():
            if len(cat_data['risk_scores']) < 2:
                continue  # Skip categories with insufficient data
                
            cat_risk = np.array(cat_data['risk_scores'], dtype=float)
            cat_time = np.array(cat_data['event_times'], dtype=float)
            cat_event = np.array(cat_data['event_indicators'], dtype=float)
            
            try:
                cat_c_index = self.survival_metrics.compute_concordance_index(
                    cat_risk, cat_time, cat_event)
                safe_cat = self._make_safe_key(category)
                metrics[f'{safe_cat}_c_index'] = float(cat_c_index)
            except Exception as e:
                # Silently skip categories that cannot compute C-index (e.g., no admissible pairs)
                continue
        
        return metrics

    def _save_all_results(self, task_results: Dict[str, List[Dict]]) -> None:
        """Persist raw evaluation results for each task type."""
        for task_type, results in task_results.items():
            if not results:
                continue

            merged_path = os.path.join(self.output_dir, f'{task_type}_results.json')
            self._save_json_file(results, merged_path, f"{len(results)} {task_type} results")
            self._save_project_results(results, task_type)

            # Save MCQA metric summary table for easy inspection
            if task_type == 'mcqa' and hasattr(self, '_mcqa_metrics_table'):
                summary_path = os.path.join(self.output_dir, 'mcqa_metrics_summary.json')
                self._save_json_file(self._mcqa_metrics_table, summary_path, 'MCQA metric summary')

    def _save_project_results(self, results: List[Dict], task_type: str) -> None:
        """Persist results split by project, adding per-category files for survival."""
        project_results = defaultdict(list)
        for result in results:
            project = result.get('project', 'Unknown')
            project_results[project].append(result)

        for project, project_data in project_results.items():
            safe_project = project.replace('-', '_').replace(' ', '_')
            file_path = os.path.join(self.output_dir, f'{task_type}_{safe_project}_results.json')
            description = f"{task_type} project {project} ({len(project_data)} samples)"
            self._save_json_file(project_data, file_path, description)

            if task_type == 'survival':
                category_results = defaultdict(list)
                for item in project_data:
                    category = item.get('category', 'Unknown')
                    category_results[category].append(item)

                for category, category_data in category_results.items():
                    safe_category = category.replace('/', '_').replace(' ', '_').replace('-', '_')
                    cat_file_path = os.path.join(
                        self.output_dir,
                        f'{task_type}_{safe_project}_{safe_category}_results.json'
                    )
                    cat_description = (
                        f"{task_type} project {project} category {category} ({len(category_data)} samples)"
                    )
                    self._save_json_file(category_data, cat_file_path, cat_description)

    def _save_json_file(self, data: List[Dict], file_path: str, description: str) -> None:
        """Write JSON data to disk with logging."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print_log(f"Saved {description} to {file_path}", 'current')
        except Exception as e:
            print_log(f"Error saving {description}: {e}", 'current')

    def _print_metric_summaries(self, all_metrics: Dict[str, float]) -> None:
        """Render formatted metric summaries for each task."""
        print_log("=" * 120, 'current')
        print_log(" " * 45 + "EVALUATION RESULTS SUMMARY", 'current')
        print_log("=" * 120, 'current')

        if not hasattr(self, '_computed_task_results'):
            print_log("=" * 120, 'current')
            return

        task_results = self._computed_task_results

        if task_results.get('text'):
            self._print_text_summary()

        if task_results.get('mcqa'):
            self._print_mcqa_summary()

        if task_results.get('regression'):
            self._print_regression_summary()

        if task_results.get('survival'):
            self._print_survival_summary_with_projects(all_metrics, task_results['survival'])

        print_log("=" * 120, 'current')

    def _print_text_summary(self) -> None:
        """Print text generation metrics per category."""
        table = getattr(self, '_text_metrics_table', None)
        if not table:
            return

        print_log("\nFREE-TEXT GENERATION RESULTS (Per Category):", 'current')
        print_log("-" * 120, 'current')
        print_log(
            f"{'Category':<25} {'BLEU-4':<12} {'ROUGE-L':<12} "
            f"{'Diag BLEU-4':<12} {'Diag ROUGE-L':<15} {'Diag Acc':<12} {'Count':<10}",
            'current'
        )
        print_log("-" * 120, 'current')

        for row in table:
            metrics = row.get('metrics', {})
            count = row.get('count', 0)
            category = row.get('category', 'Uncategorized')

            print_log(
                f"{category:<25} "
                f"{metrics.get('full_BLEU-4', 0.0):<12.4f} "
                f"{metrics.get('full_ROUGE-L', 0.0):<12.4f} "
                f"{metrics.get('diag_BLEU-4', 0.0):<12.4f} "
                f"{metrics.get('diag_ROUGE-L', 0.0):<15.4f} "
                f"{metrics.get('diag_accuracy', 0.0):<12.4f} "
                f"{count:<10}",
                'current'
            )

        print_log("-" * 120, 'current')

    def _print_mcqa_summary(self) -> None:
        """Print MCQA metrics per category with the selected primary metric."""
        table = getattr(self, '_mcqa_metrics_table', None)
        if not table:
            return

        print_log("\nMULTI-CHOICE QA RESULTS (Per Category):", 'current')
        print_log("-" * 120, 'current')
        print_log(f"{'Category':<30} {'Primary':<12} {'AUROC':<12} {'W-F1':<12} {'BACC':<12} {'ACC':<12} {'K':<6} {'Count':<10}", 'current')
        print_log("-" * 120, 'current')

        for row in table:
            metrics = row.get('metrics', {})
            primary = metrics.get('primary', 'ACC')
            auroc = metrics.get('auroc', metrics.get('auroc_weighted', None))
            # New preferred keys; fall back to legacy names if present
            weighted_f1 = metrics.get('weighted_f1', metrics.get('weighted_f1_weighted', None))
            if weighted_f1 is None:
                weighted_f1 = metrics.get('macro_f1', metrics.get('macro_f1_weighted', None))
            bacc = metrics.get('bacc', metrics.get('bacc_weighted', None))
            acc = metrics.get('acc', 0.0)
            k = metrics.get('K', '')
            count = row.get('count', 0)
            category = row.get('category', 'Unknown')
            auroc_s = f"{auroc:<12.4f}" if isinstance(auroc, (float, int)) else f"{'-':<12}"
            wf1_s = f"{weighted_f1:<12.4f}" if isinstance(weighted_f1, (float, int)) else f"{'-':<12}"
            bacc_s = f"{bacc:<12.4f}" if isinstance(bacc, (float, int)) else f"{'-':<12}"
            k_s = str(k) if k is not None else ''
            print_log(
                f"{category:<30} {str(primary):<12} {auroc_s} {wf1_s} {bacc_s} {acc:<12.4f} {k_s:<6} {count:<10}",
                'current'
            )

        print_log("-" * 120, 'current')

    def _print_regression_summary(self) -> None:
        """Print regression metrics per category."""
        table = getattr(self, '_regression_metrics_table', None)
        if not table:
            return

        print_log("\nREGRESSION RESULTS (Per Category):", 'current')
        print_log("-" * 120, 'current')
        print_log(f"{'Category':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12} "
                  f"{'Pearson':<12} {'Spearman':<12} {'Count':<10}", 'current')
        print_log("-" * 120, 'current')

        for row in table:
            metrics = row.get('metrics', {})
            count = row.get('count', 0)
            category = row.get('category', 'Regression')

            print_log(
                f"{category:<25} "
                f"{metrics.get('rmse', 0.0):<12.4f} "
                f"{metrics.get('mae', 0.0):<12.4f} "
                f"{metrics.get('r2', 0.0):<12.4f} "
                f"{metrics.get('pearson', 0.0):<12.4f} "
                f"{metrics.get('spearman', 0.0):<12.4f} "
                f"{count:<10}",
                'current'
            )

        print_log("-" * 120, 'current')

    def _print_survival_summary_with_projects(self, all_metrics: Dict[str, float], results: List[Dict]) -> None:
        """Print survival metrics summary per category for each project and overall."""
        print_log("\nSURVIVAL ANALYSIS RESULTS (Per Project & Category):", 'current')
        print_log("-" * 100, 'current')
        print_log(f"{'Project':<25} {'Category':<25} {'C-Index':<15} {'Count':<10}", 'current')
        print_log("-" * 100, 'current')
        
        # Count samples per project and category
        project_category_counts = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            project = result.get('project', 'Unknown')
            category = result.get('category', 'Unknown')
            project_category_counts[project][category] += 1
        
        # Group metrics by project and category
        project_metrics = defaultdict(dict)
        for key, value in all_metrics.items():
            if key.startswith('eval/surv_') and not key.startswith('eval/surv_overall_'):
                # Parse key: 'eval/surv_TCGA_CESC_survival_os_c_index'
                remainder = key.replace('eval/surv_', '')
                
                if not remainder.endswith('_c_index'):
                    continue
                    
                # Remove _c_index suffix
                prefix = remainder.replace('_c_index', '')
                
                # Find matching project by reconstructing project name from parts
                for proj in project_category_counts.keys():
                    safe_proj = proj.replace('-', '_').replace(' ', '_')
                    
                    # Check if prefix starts with this project
                    if prefix.startswith(safe_proj + '_'):
                        # This is a category metric
                        category_part = prefix[len(safe_proj)+1:]
                        project_metrics[proj][category_part] = value
                        break
        
        # Print per-project per-category results
        for project in sorted(project_category_counts.keys()):
            metrics = project_metrics.get(project, {})
            
            # Print per-category C-index for this project
            for cat_safe in sorted(metrics.keys()):
                c_index = metrics[cat_safe]
                
                # Find original category name
                original_cat = None
                for cat in project_category_counts[project].keys():
                    if self._make_safe_key(cat) == cat_safe:
                        original_cat = cat
                        break
                
                if original_cat is None:
                    original_cat = cat_safe.replace('_', ' ').title()
                
                count = project_category_counts[project].get(original_cat, 0)
                print_log(f"{project:<25} {original_cat:<25} {c_index:<15.4f} {count:<10}", 'current')
        
        # Print cross-project overall metrics per category
        print_log("-" * 100, 'current')
        print_log("CROSS-PROJECT OVERALL (Per Category):", 'current')
        print_log("-" * 100, 'current')
        
        # Collect overall metrics per category
        overall_category_metrics = {}
        overall_category_counts = defaultdict(int)
        
        for key, value in all_metrics.items():
            if key.startswith('eval/surv_overall_') and key.endswith('_c_index'):
                category_part = key.replace('eval/surv_overall_', '').replace('_c_index', '')
                overall_category_metrics[category_part] = value
                
                # Count total samples for this category across all projects
                for project, cats in project_category_counts.items():
                    for cat, count in cats.items():
                        if self._make_safe_key(cat) == category_part:
                            overall_category_counts[category_part] += count
        
        # Print overall category metrics
        for cat_safe in sorted(overall_category_metrics.keys()):
            c_index = overall_category_metrics[cat_safe]
            count = overall_category_counts[cat_safe]
            
            # Try to find original category name
            original_cat = None
            for project, cats in project_category_counts.items():
                for cat in cats.keys():
                    if self._make_safe_key(cat) == cat_safe:
                        original_cat = cat
                        break
                if original_cat:
                    break
            
            if original_cat is None:
                original_cat = cat_safe.replace('_', ' ').title()
            
            print_log(f"{'All Projects':<25} {original_cat:<25} {c_index:<15.4f} {count:<10}", 'current')
        
        print_log("-" * 100, 'current')


class SurvivalMetrics:
    """Survival analysis metric utilities (C-index only)."""
    
    def __init__(self, time_intervals: np.ndarray):
        """Initialize with time interval boundaries (len = n+1)."""
        self.time_intervals = time_intervals
        # Use right endpoints to ensure all times are within bounds
        self.interval_endpoints = time_intervals[1:]

    def compute_concordance_index(self,
                                  risk_scores: np.ndarray,
                                  event_times: np.ndarray,
                                  event_indicators: np.ndarray) -> float:
        """Harrell's C-index via lifelines (assumed available)."""
        from lifelines.utils import concordance_index
        # lifelines interprets smaller values as higher risk -> negate scores
        return float(concordance_index(event_times, -risk_scores, event_indicators))
    
    def compute_all_metrics(self,
                          _survival_probs: np.ndarray,
                          risk_scores: np.ndarray, 
                          event_times: np.ndarray,
                          event_indicators: np.ndarray) -> Dict[str, float]:
        """Compute survival metrics usable by legacy callers (C-index only)."""
        return {
            'c_index': self.compute_concordance_index(
                risk_scores, event_times, event_indicators
            )
        }