# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence
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
    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    has_image = any(inst.get('features') is not None for inst in instances)
    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        features = []
        masks = []
        image_batch_indices = []  # map each image to its sample index

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
                features.extend(example['features'])
                image_batch_indices.extend([b_idx] * len(example['features']))
            else:
                features.append(example['features'])
                image_batch_indices.append(b_idx)
            
            # Handle masks
            if isinstance(example['masks'], list):
                masks.extend(example['masks'])
            else:
                masks.append(example['masks'])

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

    if has_image:
        features = torch.stack(features)
        masks = torch.stack(masks) if masks[0] is not None else None
        data_dict['features'] = features
        if masks is not None:
            data_dict['masks'] = masks
        data_dict['labels_text'] = [
            inst.get('conversations', [])[-1].get('value', '') for inst in instances
        ]
        data_dict['category'] = [
            inst.get('category', None) for inst in instances
        ]
        data_dict['image_file'] = [
            inst.get('image_file', None) for inst in instances
        ]
        data_dict['project'] = [
            inst.get('project', None) for inst in instances
        ]
        # Add mapping from image to sample index
        data_dict['image_batch_indices'] = torch.as_tensor(
            image_batch_indices, dtype=torch.long)

    # stack regression targets
    if any(not np.isnan(t[0]) for t in regression_targets):
        data_dict['regression_targets'] = torch.tensor(
            regression_targets, dtype=torch.float32).squeeze(-1)

    # stack survival targets 
    if any(t is not None for t in survival_targets):
        # Filter out None values and stack the valid ones
        valid_survival_targets = [t for t in survival_targets if t is not None]
        if valid_survival_targets:
            # Assume survival_targets have 'target_y' and 'at_risk_mask' keys
            target_y_list = [t['target_y'] for t in valid_survival_targets]
            at_risk_mask_list = [t['at_risk_mask'] for t in valid_survival_targets]
            
            data_dict['survival_targets'] = {
                'target_y': torch.tensor(target_y_list, dtype=torch.float32),
                'at_risk_mask': torch.tensor(at_risk_mask_list, dtype=torch.float32)
            }

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
