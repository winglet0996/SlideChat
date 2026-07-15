# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def masked_collated_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):
    # Defensive: a few datasets/map_fns may return None for broken samples.
    # MMEngine's loops cannot handle model outputs being None, so fail fast
    # with a clear error rather than crashing later in _update_losses.
    if instances is None:
        raise RuntimeError('masked_collated_fn received instances=None (dataloader yielded None batch).')
    if any(inst is None for inst in instances):
        bad = [i for i, inst in enumerate(instances) if inst is None]
        raise RuntimeError(
            f'masked_collated_fn received None sample(s) at batch positions {bad}. '
            'This usually means the dataset/map_fn returned None for a corrupt/missing file. '
            'Fix the data (or change the dataset to skip invalid samples) and rerun.'
        )
    if len(instances) == 0:
        raise RuntimeError('masked_collated_fn received an empty batch (len(instances)==0).')

    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    has_image = any(inst.get('features') is not None for inst in instances)
    has_wsi_features = any(inst.get('wsi_features') is not None for inst in instances)
    
    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        features = []
        image_batch_indices = []  # map each image to its sample index
        feature_shapes = []  # original (H, W) per image before batch padding
        feature_paths = []  # source path aligned with each flattened feature
    
    # WSI features collection
    if has_wsi_features:
        wsi_features_list = []  # List of List[Tensor] for each sample

    # Optional regression and survival targets collection
    regression_targets = []
    survival_targets = []

    for b_idx, example in enumerate(instances):
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))

        # collect regression targets if present
        if 'regression_targets' in example and example['regression_targets'] is not None:
            # support scalar or list
            tgt = example['regression_targets']
            if isinstance(tgt, (list, tuple, np.ndarray)):
                # use the first value for this sample
                tgt = float(tgt[0])
            regression_targets.append([tgt])
        else:
            regression_targets.append([np.nan])  # placeholder, will be masked later if unused

        # collect survival targets if present
        if 'survival_targets' in example and example['survival_targets'] is not None:
            survival_targets.append(example['survival_targets'])
        else:
            survival_targets.append(None)  # placeholder

        if has_image:
            # Handle features
            if isinstance(example['features'], list):
                image_files = example.get('image_file', None)
                if isinstance(image_files, (list, tuple)):
                    image_files = list(image_files)
                elif image_files is None:
                    image_files = []
                else:
                    image_files = [image_files]
                if len(image_files) < len(example['features']):
                    image_files.extend([None] * (len(example['features']) - len(image_files)))
                feature_shapes.extend([(f.shape[1], f.shape[2]) for f in example['features']])
                features.extend(example['features'])
                image_batch_indices.extend([b_idx] * len(example['features']))
                feature_paths.extend(image_files[:len(example['features'])])
            else:
                feature_shapes.append((example['features'].shape[1], example['features'].shape[2]))
                features.append(example['features'])
                image_batch_indices.append(b_idx)
                feature_paths.append(example.get('image_file', None))
        
        # Handle WSI features
        if has_wsi_features:
            if 'wsi_features' in example and example['wsi_features'] is not None:
                # wsi_features is a list of tensors [Tensor(D1), Tensor(D2), ...]
                wsi_features_list.append(example['wsi_features'])
            else:
                # Placeholder - will need to handle this case in the model
                wsi_features_list.append(None)

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros_like(input_ids).bool()
        for row, i in enumerate(ori_length):
            attention_mask[row, :i] = True

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)

    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }

    data_dict['category'] = [
        inst.get('category', None) for inst in instances
    ]
    data_dict['route_family'] = [
        inst.get('route_family', None) for inst in instances
    ]
    data_dict['project'] = [
        inst.get('project', None) for inst in instances
    ]
    data_dict['id'] = [
        inst.get('id', None) for inst in instances
    ]
    data_dict['image_file'] = [
        inst.get('image_file', None) for inst in instances
    ]
    data_dict['division'] = [
        inst.get('division', None) for inst in instances
    ]
    data_dict['wsi_feature_paths'] = [
        inst.get('wsi_feature_paths', None) for inst in instances
    ]
    raw_drop_keys = {
        'features', 'wsi_features', 'input_ids', 'labels', 'attention_mask',
        'position_ids', 'cumulative_len'
    }
    data_dict['raw_sample_json'] = [
        json.dumps(
            {k: v for k, v in inst.items() if k not in raw_drop_keys},
            ensure_ascii=False,
            default=str,
        )
        for inst in instances
    ]

    if has_image:
        # Pad features to the max size in the batch
        max_h = max(f.shape[1] for f in features)
        max_w = max(f.shape[2] for f in features)

        padded_features = []
        for f in features:
            pad_h = max_h - f.shape[1]
            pad_w = max_w - f.shape[2]
            if pad_h > 0 or pad_w > 0:
                f = torch.nn.functional.pad(f, (0, pad_w, 0, pad_h), value=0.0)
            padded_features.append(f)
        features = torch.stack(padded_features)

        data_dict['features'] = features
        data_dict['labels_text'] = [
            inst.get('conversations', [])[-1].get('value', '') for inst in instances
        ]
        # Add mapping from image to sample index
        data_dict['image_batch_indices'] = torch.as_tensor(
            image_batch_indices, dtype=torch.long)
        data_dict['feature_shapes'] = torch.as_tensor(
            feature_shapes, dtype=torch.long)
        data_dict['feature_paths'] = feature_paths

    # Add WSI features to data_dict
    # WSI features are kept as a list of lists (not stacked) since different sources may have different dims
    if has_wsi_features:
        # Filter out None values and only include if all samples have WSI features
        valid_wsi_features = [w for w in wsi_features_list if w is not None]
        if len(valid_wsi_features) == len(wsi_features_list) and len(valid_wsi_features) > 0:
            data_dict['wsi_features'] = valid_wsi_features
        else:
            # If some samples are missing WSI features, don't include any to avoid shape mismatches
            data_dict['wsi_features'] = None

    # stack regression targets
    if any(not np.isnan(t[0]) for t in regression_targets):
        data_dict['regression_targets'] = torch.tensor(
            regression_targets, dtype=torch.float32).squeeze(-1)

    # stack survival targets (supports both Cox and Discrete formats)
    # Cox format: {'time': float, 'event': float} or [time, event]
    # Discrete format: {'time': float, 'event': float, 'bins': list} or {'bins': list, ...}
    if any(t is not None for t in survival_targets):
        survival_times = []
        survival_events = []
        survival_bins = []  # For discrete method: list of K-length tensors
        has_bins = False
        
        for t in survival_targets:
            if t is not None:
                # Check for discrete bins format
                if isinstance(t, dict) and 'bins' in t:
                    has_bins = True
                    survival_bins.append(t['bins'])
                    # Also extract time/event for C-index evaluation
                    survival_times.append(float(t.get('time', 0.0)))
                    survival_events.append(float(t.get('event', 0.0)))
                # Handle [time, event] format from JSON (Cox)
                elif isinstance(t, (list, tuple)) and len(t) == 2:
                    survival_times.append(float(t[0]))
                    survival_events.append(float(t[1]))
                    survival_bins.append(None)
                # Handle dict format (Cox)
                elif isinstance(t, dict):
                    survival_times.append(float(t.get('time', 0.0)))
                    survival_events.append(float(t.get('event', 0.0)))
                    survival_bins.append(None)
                else:
                    # Fallback: treat as invalid
                    survival_times.append(float('nan'))
                    survival_events.append(float('nan'))
                    survival_bins.append(None)
            else:
                survival_times.append(float('nan'))
                survival_events.append(float('nan'))
                survival_bins.append(None)
        
        survival_dict = {
            'time': torch.tensor(survival_times, dtype=torch.float32),
            'event': torch.tensor(survival_events, dtype=torch.float32),
            'method': 'discrete' if has_bins else 'cox'
        }
        
        # Add bins data for discrete method
        if has_bins:
            # Stack bins into tensors if all samples have bins
            valid_bins = [b for b in survival_bins if b is not None]
            if valid_bins:
                # Assume all bins have the same structure: {'target_y': [...], 'at_risk_mask': [...]}
                if isinstance(valid_bins[0], dict):
                    target_y_list = []
                    at_risk_mask_list = []
                    for i, b in enumerate(survival_bins):
                        if b is not None:
                            target_y_list.append(torch.tensor(b['target_y'], dtype=torch.float32))
                            at_risk_mask_list.append(torch.tensor(b['at_risk_mask'], dtype=torch.float32))
                        else:
                            # Placeholder for samples without bins (use zeros)
                            K = len(valid_bins[0]['target_y'])
                            target_y_list.append(torch.zeros(K, dtype=torch.float32))
                            at_risk_mask_list.append(torch.zeros(K, dtype=torch.float32))
                    survival_dict['target_y'] = torch.stack(target_y_list)
                    survival_dict['at_risk_mask'] = torch.stack(at_risk_mask_list)
                else:
                    # bins is a list of values [y1, y2, ..., yK]
                    target_y_list = []
                    for i, b in enumerate(survival_bins):
                        if b is not None:
                            target_y_list.append(torch.tensor(b, dtype=torch.float32))
                        else:
                            K = len(valid_bins[0])
                            target_y_list.append(torch.zeros(K, dtype=torch.float32))
                    survival_dict['target_y'] = torch.stack(target_y_list)
        
        data_dict['survival_targets'] = survival_dict

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
