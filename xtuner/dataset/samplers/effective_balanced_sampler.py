# Copyright (c) OpenMMLab. All rights reserved.
"""Effective-sample-aware sampler for PathOverse multitask training."""
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import time
from contextlib import contextmanager
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, Sized, Tuple

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.logging import print_log
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


_TCGA_PATIENT_RE = re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')


def _weighted_choice(rng: random.Random,
                     items: Sequence,
                     weights: Sequence[float]):
    total = float(sum(weights))
    if total <= 0:
        return rng.choice(list(items))
    threshold = rng.random() * total
    upto = 0.0
    for item, weight in zip(items, weights):
        upto += float(weight)
        if upto >= threshold:
            return item
    return items[-1]


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        return [1.0 / len(weights)] * len(weights) if weights else []
    return [float(w) / total for w in weights]


SAMPLER_CACHE_VERSION = 1


class _FenwickTree:
    """Mutable prefix-sum tree for weighted unit sampling."""

    def __init__(self, weights: Sequence[float]) -> None:
        self.n = len(weights)
        self.tree = [0.0] * (self.n + 1)
        self.values = [float(weight) for weight in weights]
        for idx, weight in enumerate(self.values):
            self._add(idx, weight)

    def _add(self, idx: int, delta: float) -> None:
        idx += 1
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def total(self) -> float:
        return self.prefix_sum(self.n - 1) if self.n else 0.0

    def prefix_sum(self, idx: int) -> float:
        idx += 1
        total = 0.0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx
        return total

    def set_zero(self, idx: int) -> None:
        value = self.values[idx]
        if value != 0.0:
            self.values[idx] = 0.0
            self._add(idx, -value)

    def find_prefix(self, threshold: float) -> int:
        idx = 0
        bit = 1 << (self.n.bit_length() - 1) if self.n else 0
        while bit:
            next_idx = idx + bit
            if next_idx <= self.n and self.tree[next_idx] < threshold:
                threshold -= self.tree[next_idx]
                idx = next_idx
            bit >>= 1
        return min(idx, self.n - 1)


class _UnitSelectionState:
    """Per-iterator mutable state for one sampler group."""

    def __init__(self, group: dict) -> None:
        self.weights = [float(weight) for weight in group['unit_weights']]
        self.weight_tree = _FenwickTree(self.weights)
        self.active_mask = [True] * len(self.weights)
        self.active_count = len(self.weights)

    def deactivate(self, unit_idx: int) -> None:
        if not self.active_mask[unit_idx]:
            return
        self.active_mask[unit_idx] = False
        self.active_count -= 1
        self.weight_tree.set_zero(unit_idx)

    def choose(self, rng: random.Random) -> int:
        total = self.weight_tree.total()
        if total <= 0:
            if self.active_count > 0:
                active = [
                    idx for idx, is_active in enumerate(self.active_mask)
                    if is_active
                ]
                return rng.choice(active)
            return _weighted_choice(
                rng, list(range(len(self.weights))), self.weights)
        threshold = rng.random() * total
        return self.weight_tree.find_prefix(threshold)


@contextmanager
def _sampler_cache_lock(lock_path, timeout_seconds=1800, poll_interval=1.0):
    start_time = time.time()
    while True:
        try:
            os.mkdir(lock_path)
            break
        except FileExistsError:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"Timeout while waiting for sampler cache lock: {lock_path}")
            time.sleep(poll_interval)

    try:
        yield
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def _dataset_cache_identity(dataset: Sized) -> dict:
    text_data = getattr(dataset, "text_data", None)
    identity = {
        "length": len(dataset),
        "text_cache_key": getattr(dataset, "text_cache_key", None),
        "text_cache_path": getattr(dataset, "text_cache_path", None),
        "text_cache_signature": getattr(dataset, "text_cache_signature", None),
        "text_fingerprint": getattr(text_data, "_fingerprint", None),
    }
    if identity["text_cache_key"] is None and identity["text_cache_signature"] is None:
        cache_files = getattr(text_data, "cache_files", None)
        if cache_files:
            identity["text_cache_files"] = cache_files
    return identity


@DATA_SAMPLERS.register_module()
class EffectiveBalancedSampler(Sampler):
    """Sampler that balances multitask data by effective units.

    The sampler chooses a task family, then a task group, then an effective
    unit such as a slide or patient, and finally a row inside that unit. By
    default, it repeats that choice per sample so low-frequency tasks are mixed
    into many batches instead of being concentrated into rare task-only batches.
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True,
                 family_weights: Optional[Dict[str, float]] = None,
                 size_alpha: float = 0.5,
                 original_mix_by_family: Optional[Dict[str, float]] = None,
                 subfamily_weights_by_family: Optional[Dict[str, Dict[str, float]]] = None,
                 balance_mcqa_labels: bool = True,
                 label_balance_power: float = 0.5,
                 min_group_size_for_label_balance: int = 32,
                 tiny_group_size: int = 32,
                 tiny_group_max_fraction: float = 0.1,
                 max_unit_repeats_per_epoch: Optional[int] = 4,
                 survival_unit: str = 'patient',
                 default_unit: str = 'slide',
                 mix_within_batch: bool = True,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        self.shuffle = shuffle
        self.round_up = round_up
        self.family_weights = family_weights or {
            'mcqa': 0.55,
            'regression': 0.35,
            'survival': 0.10,
        }
        self.size_alpha = float(size_alpha)
        self.original_mix_by_family = {
            str(k): float(v) for k, v in (original_mix_by_family or {}).items()
        }
        self.subfamily_weights_by_family = {
            str(family): {
                str(subfamily): float(weight)
                for subfamily, weight in weights.items()
            }
            for family, weights in (subfamily_weights_by_family or {}).items()
        }
        self.balance_mcqa_labels = balance_mcqa_labels
        self.label_balance_power = float(label_balance_power)
        self.min_group_size_for_label_balance = int(min_group_size_for_label_balance)
        self.tiny_group_size = int(tiny_group_size)
        self.tiny_group_max_fraction = float(tiny_group_max_fraction)
        self.max_unit_repeats_per_epoch = max_unit_repeats_per_epoch
        self.survival_unit = survival_unit
        self.default_unit = default_unit
        self.mix_within_batch = bool(mix_within_batch)
        self.cache_dir = cache_dir
        self.verbose = verbose

        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        total_len = len(self.dataset)
        if self.round_up:
            num_batches = math.ceil(total_len / (world_size * self.batch_size))
            self.num_samples = num_batches * self.batch_size
        else:
            self.num_samples = math.ceil((total_len - rank) / world_size)
        self.total_size = self.num_samples * world_size

        self._built = False
        self._families: List[str] = []
        self._family_probs: List[float] = []
        self._groups_by_family: Dict[str, List[dict]] = {}
        self._group_probs_by_family: Dict[str, List[float]] = {}

    def _sample_at(self, index: int) -> dict:
        text_data = getattr(self.dataset, 'text_data', None)
        if text_data is None:
            return {}
        try:
            return text_data[index]
        except Exception:
            return {}

    def _family(self, category: str) -> str:
        lower = (category or '').lower()
        if lower.startswith('mcqa::') or 'mcqa' in lower:
            return 'mcqa'
        if lower.startswith('regression::') or 'regression' in lower:
            return 'regression'
        if lower.startswith('survival::') or 'survival' in lower:
            return 'survival'
        return 'text'

    def _group_key(self, sample: dict) -> Tuple[str, str, str]:
        category = sample.get('category', 'default') or 'default'
        family = self._family(category)
        if family == 'survival':
            project = sample.get('project', 'default') or 'default'
        else:
            project = 'all'
        return family, category, project

    def _subfamily(self, family: str, category: str) -> str:
        if family != 'regression':
            return 'all'
        task = (category.split('::', 1)[1]
                if '::' in category else category).lower()
        if (task.startswith('rna_pathway_')
                or task.startswith('rna_regression_')
                or task.startswith('rna_')):
            return 'RNA'
        if task.startswith('protein_pathway_') or task.startswith('protein_'):
            return 'protein'
        if (task.startswith('immune_infil_')
                or task.startswith('immune_infiltration')
                or task.startswith('immune_')):
            return 'immune_infil'
        return 'openTME'

    def _first_path(self, sample: dict) -> Optional[str]:
        for key in ('image', 'image_file', 'wsi_features'):
            value = sample.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)) and value:
                return str(value[0])
        return None

    def _unit_key(self, sample: dict, family: str) -> str:
        path = self._first_path(sample)
        sample_id = str(sample.get('id', 'unknown'))

        if family == 'survival' and self.survival_unit == 'patient':
            for value in (path, sample_id):
                if not value:
                    continue
                match = _TCGA_PATIENT_RE.search(str(value))
                if match:
                    return match.group(1)

        if path and self.default_unit == 'slide':
            return os.path.splitext(os.path.basename(str(path)))[0]
        return sample_id

    def _mcqa_label(self, sample: dict) -> str:
        conversations = sample.get('conversations') or []
        answer = ''
        if conversations and isinstance(conversations, (list, tuple)):
            last = conversations[-1]
            if isinstance(last, dict):
                answer = str(last.get('value', ''))
        match = re.match(r'\s*([A-Z])\)', answer)
        if match:
            return match.group(1)
        return answer.strip().lower()[:64] or 'unknown'

    def _unit_weight(self, rows: List[dict], label_counts: Counter,
                     enable_label_balance: bool) -> float:
        if not enable_label_balance:
            return 1.0
        weights = []
        for row in rows:
            label = row.get('label', 'unknown')
            count = max(1, label_counts.get(label, 1))
            weights.append(count ** (-self.label_balance_power))
        return float(sum(weights) / len(weights)) if weights else 1.0

    def _cache_signature(self) -> dict:
        return {
            'cache_version': SAMPLER_CACHE_VERSION,
            'dataset': _dataset_cache_identity(self.dataset),
            'world_size': self.world_size,
            'round_up': self.round_up,
            'batch_size': self.batch_size,
            'family_weights': self.family_weights,
            'size_alpha': self.size_alpha,
            'original_mix_by_family': self.original_mix_by_family,
            'subfamily_weights_by_family': self.subfamily_weights_by_family,
            'balance_mcqa_labels': self.balance_mcqa_labels,
            'label_balance_power': self.label_balance_power,
            'min_group_size_for_label_balance': self.min_group_size_for_label_balance,
            'tiny_group_size': self.tiny_group_size,
            'tiny_group_max_fraction': self.tiny_group_max_fraction,
            'max_unit_repeats_per_epoch': self.max_unit_repeats_per_epoch,
            'survival_unit': self.survival_unit,
            'default_unit': self.default_unit,
            'mix_within_batch': self.mix_within_batch,
        }

    def _cache_paths(self):
        if not self.cache_dir:
            return None, None, None
        signature = self._cache_signature()
        cache_key = hashlib.sha256(
            json.dumps(signature, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        cache_root = os.path.join(os.path.abspath(self.cache_dir),
                                  'effective_balanced_sampler')
        cache_path = os.path.join(cache_root, f'{cache_key}.pkl')
        lock_path = os.path.join(cache_root, f'{cache_key}.lock')
        return cache_path, lock_path, signature

    def _try_load_cache(self, cache_path, signature) -> bool:
        if not cache_path or not os.path.isfile(cache_path):
            return False
        try:
            with open(cache_path, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('signature') != signature:
                return False
            self._families = payload['families']
            self._family_probs = payload['family_probs']
            self._groups_by_family = payload['groups_by_family']
            self._group_probs_by_family = payload['group_probs_by_family']
            self._built = True
            if self.verbose and self.rank == 0:
                print_log(
                    f'Loading EffectiveBalancedSampler cache from {cache_path}',
                    logger='current')
            return True
        except Exception as e:
            if self.verbose and self.rank == 0:
                print_log(
                    f'Failed to load EffectiveBalancedSampler cache from {cache_path}: {e}. '
                    'The cache will be rebuilt.',
                    logger='current')
            return False

    def _save_cache(self, cache_path, signature) -> None:
        if not cache_path:
            return
        cache_root = os.path.dirname(cache_path)
        os.makedirs(cache_root, exist_ok=True)
        tmp_path = os.path.join(
            cache_root,
            f'.tmp_{os.path.basename(cache_path)}_{os.getpid()}_{time.time_ns()}')
        payload = {
            'signature': signature,
            'families': self._families,
            'family_probs': self._family_probs,
            'groups_by_family': self._groups_by_family,
            'group_probs_by_family': self._group_probs_by_family,
        }
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            if self.verbose and self.rank == 0:
                print_log(
                    f'Saved EffectiveBalancedSampler cache to {cache_path}',
                    logger='current')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _build(self) -> None:
        if self._built:
            return

        cache_path, lock_path, signature = self._cache_paths()
        if self._try_load_cache(cache_path, signature):
            return

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with _sampler_cache_lock(lock_path):
                if self._try_load_cache(cache_path, signature):
                    return
                self._build_index()
                self._save_cache(cache_path, signature)
            return

        self._build_index()

    def _build_index(self) -> None:
        if self._built:
            return

        groups = defaultdict(lambda: {
            'family': None,
            'category': None,
            'project': None,
            'units': defaultdict(list),
            'row_count': 0,
        })

        for index in range(len(self.dataset)):
            sample = self._sample_at(index)
            family, category, project = self._group_key(sample)
            unit = self._unit_key(sample, family)
            label = self._mcqa_label(sample) if family == 'mcqa' else None
            key = (family, category, project)
            group = groups[key]
            group['family'] = family
            group['category'] = category
            group['project'] = project
            group['subfamily'] = self._subfamily(family, category)
            group['units'][unit].append({'index': index, 'label': label})
            group['row_count'] += 1

        groups_by_family = defaultdict(list)
        for key in sorted(groups.keys()):
            group = groups[key]
            units = dict(group['units'])
            n_eff = len(units)
            labels = Counter(
                row['label']
                for rows in units.values()
                for row in rows
                if row.get('label') is not None)
            enable_label_balance = (
                self.balance_mcqa_labels
                and group['family'] == 'mcqa'
                and n_eff >= self.min_group_size_for_label_balance
                and len(labels) > 1)
            unit_items = sorted(units.items(), key=lambda item: item[0])
            group['unit_keys'] = [item[0] for item in unit_items]
            group['unit_rows'] = [item[1] for item in unit_items]
            group['unit_weights'] = [
                self._unit_weight(rows, labels, enable_label_balance)
                for _, rows in unit_items
            ]
            group.pop('units', None)
            group['n_eff'] = n_eff
            group['tiny'] = n_eff < self.tiny_group_size
            groups_by_family[group['family']].append(group)

        self._groups_by_family = dict(groups_by_family)
        self._group_probs_by_family = {
            family: self._compute_group_probs(family, groups)
            for family, groups in self._groups_by_family.items()
        }
        self._families = [
            family for family in self.family_weights
            if family in self._groups_by_family and self.family_weights[family] > 0
        ]
        if not self._families:
            self._families = sorted(self._groups_by_family.keys())
        self._family_probs = _normalize_weights(
            [self.family_weights.get(family, 1.0) for family in self._families])
        self._built = True

        if self.verbose and self.rank == 0:
            summary = []
            for family in self._families:
                family_groups = self._groups_by_family[family]
                rows = sum(group['row_count'] for group in family_groups)
                units = sum(group['n_eff'] for group in family_groups)
                summary.append(
                    f'{family}: groups={len(family_groups)} rows={rows} units={units}')
                if family in self.subfamily_weights_by_family:
                    sub_summary = []
                    probs = self._group_probs_by_family[family]
                    for subfamily in sorted({g['subfamily'] for g in family_groups}):
                        sub_groups = [
                            (group, prob)
                            for group, prob in zip(family_groups, probs)
                            if group['subfamily'] == subfamily
                        ]
                        sub_rows = sum(group['row_count'] for group, _ in sub_groups)
                        sub_units = sum(group['n_eff'] for group, _ in sub_groups)
                        sub_prob = sum(prob for _, prob in sub_groups)
                        sub_summary.append(
                            f'{subfamily}: groups={len(sub_groups)} rows={sub_rows} '
                            f'units={sub_units} prob={sub_prob:.4f}')
                    summary.append(f'{family} subfamilies: ' + '; '.join(sub_summary))
            print_log(
                'EffectiveBalancedSampler is used. ' + '; '.join(summary),
                logger='current')

    def _compute_size_probs(self, family: str, groups: List[dict]) -> List[float]:
        if not groups:
            return []
        eff_counts = [max(1, group['n_eff']) for group in groups]
        original = _normalize_weights(eff_counts)
        tempered = _normalize_weights([
            count ** self.size_alpha for count in eff_counts
        ])
        if family not in self.original_mix_by_family:
            raise KeyError(f'original_mix_by_family is missing family: {family}')
        original_mix = self.original_mix_by_family[family]
        original_mix = max(0.0, min(1.0, float(original_mix)))
        probs = [
            original_mix * p_orig + (1.0 - original_mix) * p_temp
            for p_orig, p_temp in zip(original, tempered)
        ]

        tiny_indices = [i for i, group in enumerate(groups) if group['tiny']]
        if tiny_indices and len(tiny_indices) < len(groups):
            tiny_total = sum(probs[i] for i in tiny_indices)
            cap = max(0.0, min(1.0, self.tiny_group_max_fraction))
            if tiny_total > cap:
                scale_tiny = cap / tiny_total if tiny_total > 0 else 0.0
                non_tiny_indices = [
                    i for i in range(len(groups)) if i not in tiny_indices
                ]
                non_tiny_total = sum(probs[i] for i in non_tiny_indices)
                scale_non_tiny = (
                    (1.0 - cap) / non_tiny_total
                    if non_tiny_total > 0 else 1.0)
                probs = [
                    prob * (scale_tiny if i in tiny_indices else scale_non_tiny)
                    for i, prob in enumerate(probs)
                ]
        return _normalize_weights(probs)

    def _compute_group_probs(self, family: str, groups: List[dict]) -> List[float]:
        subfamily_weights = self.subfamily_weights_by_family.get(family)
        if not subfamily_weights:
            return self._compute_size_probs(family, groups)

        subfamilies = sorted({group['subfamily'] for group in groups})
        sub_weights = [
            max(0.0, float(subfamily_weights.get(subfamily, 1.0)))
            for subfamily in subfamilies
        ]
        if sum(sub_weights) <= 0:
            return self._compute_size_probs(family, groups)
        sub_probs = _normalize_weights(sub_weights)

        probs = [0.0] * len(groups)
        for subfamily, sub_prob in zip(subfamilies, sub_probs):
            indices = [
                idx for idx, group in enumerate(groups)
                if group['subfamily'] == subfamily
            ]
            if not indices or sub_prob <= 0:
                continue
            inner_groups = [groups[idx] for idx in indices]
            inner_probs = self._compute_size_probs(family, inner_groups)
            for idx, inner_prob in zip(indices, inner_probs):
                probs[idx] = sub_prob * inner_prob
        return _normalize_weights(probs)

    def _select_unit_index(self, rng: random.Random, group: dict,
                           unit_counts: Dict[Tuple[str, str, str, str], int],
                           unit_states: Dict[int, _UnitSelectionState]) -> int:
        state = unit_states.get(id(group))
        if state is None:
            state = _UnitSelectionState(group)
            unit_states[id(group)] = state
        return state.choose(rng)

    def _choose_group(self, rng: random.Random) -> dict:
        family = _weighted_choice(rng, self._families, self._family_probs)
        groups = self._groups_by_family[family]
        group_probs = self._group_probs_by_family[family]
        return _weighted_choice(rng, groups, group_probs)

    def __iter__(self) -> Iterator[int]:
        self._build()
        group_rng = random.Random(self.seed + self.epoch * 1009)
        sample_rng = random.Random(self.seed + self.epoch * 1009 + self.rank)
        unit_counts = defaultdict(int)
        unit_states: Dict[int, _UnitSelectionState] = {}
        num_batches = math.ceil(self.num_samples / self.batch_size)
        yielded = 0

        for _ in range(num_batches):
            batch_group = None
            if not self.mix_within_batch:
                batch_group = self._choose_group(group_rng)

            for _ in range(self.batch_size):
                if yielded >= self.num_samples:
                    return
                group = batch_group
                if self.mix_within_batch:
                    group = self._choose_group(group_rng)

                unit_idx = self._select_unit_index(
                    sample_rng, group, unit_counts, unit_states)
                unit_key = group['unit_keys'][unit_idx]
                rows = group['unit_rows'][unit_idx]
                row = sample_rng.choice(rows) if self.shuffle else rows[0]
                count_key = (
                    group['family'], group['category'], group['project'],
                    unit_key)
                unit_counts[count_key] += 1
                if (self.max_unit_repeats_per_epoch is not None
                        and unit_counts[count_key] >= self.max_unit_repeats_per_epoch):
                    unit_states[id(group)].deactivate(unit_idx)
                yielded += 1
                yield row['index']

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
