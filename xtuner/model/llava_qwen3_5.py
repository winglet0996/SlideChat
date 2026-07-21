# Copyright (c) OpenMMLab. All rights reserved.
"""Prompt-conditioned pathology adapter for Qwen3.5 text models."""

import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModel
from mmengine.utils import is_list_of
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AddedToken, AutoConfig, GenerationConfig, StoppingCriteriaList
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.utils import IMAGE_TOKEN_INDEX, IGNORE_INDEX, StopWordStoppingCriteria

from .custom_model import (
    InputComposer,
    MRoPEPositionIDGenerator,
    PromptConditionedPatchResampler,
    ROUTE_FAMILIES,
    RoutedLoRALinear,
    RegressionHead,
    SurvivalHead,
    WSIProjector,
    cox_ph_loss,
    logistic_hazard_loss,
    replace_linear_with_routed_lora,
)
from .modules import dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (
    LoadWoInit,
    find_all_linear_names,
    get_peft_model_state_dict,
    guess_load_checkpoint,
    make_inputs_require_grad,
    traverse_dict,
)


def _safe_token_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    safe_ids = input_ids.clone()
    safe_ids[safe_ids < 0] = pad_token_id
    return safe_ids


def _strip_image_tokens(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    padding_side: str,
    pad_token_id: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attn = attention_mask.bool()
    keep = attn & (input_ids != IMAGE_TOKEN_INDEX)
    batch_size, seq_len = input_ids.shape
    lengths = keep.sum(dim=1, keepdim=True)
    rank = keep.long().cumsum(dim=1) - 1
    if padding_side == 'right':
        dst = rank
    else:
        dst = seq_len - lengths + rank
    dst = dst.clamp(min=0, max=max(seq_len - 1, 0))

    dummy_col = torch.full_like(dst, seq_len)
    scatter_dst = torch.where(keep, dst, dummy_col)
    new_ids_ext = torch.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    new_mask_ext = torch.zeros((batch_size, seq_len + 1), dtype=torch.bool, device=input_ids.device)
    new_ids_ext.scatter_(1, scatter_dst, torch.where(keep, input_ids, torch.full_like(input_ids, pad_token_id)))
    new_mask_ext.scatter_(1, scatter_dst, keep)
    new_ids = new_ids_ext[:, :seq_len]
    new_mask = new_mask_ext[:, :seq_len]

    new_labels = None
    if labels is not None:
        new_labels_ext = torch.full((batch_size, seq_len + 1), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        new_labels_ext.scatter_(1, scatter_dst, torch.where(keep, labels, torch.full_like(labels, IGNORE_INDEX)))
        new_labels = new_labels_ext[:, :seq_len]
    return new_ids, new_labels, new_mask


def _linear_position_ids(attention_mask: torch.Tensor, rows: int = 4) -> torch.Tensor:
    valid = attention_mask.bool()
    pos = valid.long().cumsum(dim=1).sub_(1).clamp_min_(0)
    pos = pos.masked_fill(~valid, 0)
    return pos.unsqueeze(0).expand(rows, -1, -1).contiguous()


def _raise_if_nonfinite(name: str, value: torch.Tensor, extra: str = "") -> None:
    if torch.isfinite(value).all():
        return
    value32 = value.detach().float()
    finite = value32[torch.isfinite(value32)]
    if finite.numel() > 0:
        min_value = float(finite.min().item())
        max_value = float(finite.max().item())
    else:
        min_value = float('nan')
        max_value = float('nan')
    suffix = f" {extra}" if extra else ""
    raise RuntimeError(
        f"Non-finite tensor detected in {name}:{suffix} "
        f"shape={tuple(value.shape)} dtype={value.dtype} "
        f"min={min_value:.6g} max={max_value:.6g}"
    )


def _sample_keep_mask(
    value: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    value = value.to(device=device).bool().view(-1)
    if value.numel() != batch_size:
        raise ValueError(
            f"sample keep mask must have {batch_size} entries, got {value.numel()}")
    return value


def _prepare_text_or_wsi_inputs(
    llm,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    wsi_embeddings: Optional[torch.Tensor],
    padding_side: str,
    wsi_sample_keep: Optional[torch.Tensor] = None,
):
    pad_token_id = getattr(llm.config, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(llm.config, 'eos_token_id', 0)
    input_ids, labels, attention_mask = _strip_image_tokens(
        input_ids, labels, attention_mask, padding_side, int(pad_token_id)
    )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask = attention_mask.bool()
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    token_embed = llm.get_input_embeddings()
    safe_ids = _safe_token_ids(input_ids, int(pad_token_id))
    text_embeds = token_embed(safe_ids)
    if wsi_embeddings is None or wsi_embeddings.numel() == 0:
        return {
            'input_ids': None,
            'inputs_embeds': text_embeds,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': _linear_position_ids(attention_mask),
        }

    batch_size = input_ids.size(0)
    wsi_sample_keep = _sample_keep_mask(wsi_sample_keep, batch_size, input_ids.device)

    embeds_list, labels_list, attention_list, pos_list = [], [], [], []
    for b_idx in range(batch_size):
        valid = attention_mask[b_idx]
        text = text_embeds[b_idx, valid]
        lbl = labels[b_idx, valid]
        if bool(wsi_sample_keep[b_idx].item()):
            wsi = wsi_embeddings[b_idx].to(device=text.device, dtype=text.dtype)
            cur_emb = torch.cat([wsi, text], dim=0)
            cur_lbl = torch.cat([
                torch.full((wsi.size(0),), IGNORE_INDEX, dtype=labels.dtype, device=labels.device),
                lbl,
            ])
        else:
            cur_emb = text
            cur_lbl = lbl
        cur_attn = torch.ones(cur_emb.size(0), dtype=torch.bool, device=cur_emb.device)
        seq = torch.arange(cur_emb.size(0), device=cur_emb.device, dtype=torch.long)
        embeds_list.append(cur_emb)
        labels_list.append(cur_lbl)
        attention_list.append(cur_attn)
        pos_list.append(seq.unsqueeze(0).expand(4, -1))

    composed = InputComposer()(embeds_list, labels_list, attention_list, pos_list, padding_side, IGNORE_INDEX)
    return {'input_ids': None, **composed}


def prepare_inputs_labels_for_qwen3_5(
    llm,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_batch_indices: Optional[torch.Tensor] = None,
    vision_token_positions: Optional[torch.Tensor] = None,
    vision_token_valid: Optional[torch.Tensor] = None,
    wsi_embeddings: Optional[torch.Tensor] = None,
    patch_sample_keep: Optional[torch.Tensor] = None,
    wsi_sample_keep: Optional[torch.Tensor] = None,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    position_generator: Optional[MRoPEPositionIDGenerator] = None,
    patch_position_encoding: str = 'linear',
    composer: Optional[InputComposer] = None,
    padding_side: str = 'right',
    **kwargs,
):
    batch_size = input_ids.size(0)
    patch_sample_keep = _sample_keep_mask(patch_sample_keep, batch_size, input_ids.device)
    wsi_sample_keep = _sample_keep_mask(wsi_sample_keep, batch_size, input_ids.device)
    has_patch = pixel_values is not None and pixel_values.numel() > 0
    has_wsi = wsi_embeddings is not None and wsi_embeddings.numel() > 0
    if not has_patch:
        return _prepare_text_or_wsi_inputs(
            llm, input_ids, labels, attention_mask, wsi_embeddings, padding_side,
            wsi_sample_keep=wsi_sample_keep)

    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask = attention_mask.bool()
    patch_position_encoding = str(patch_position_encoding).lower()
    if patch_position_encoding not in ('linear', 'mrope'):
        raise ValueError("patch_position_encoding must be 'linear' or 'mrope'.")
    if position_generator is None:
        position_generator = MRoPEPositionIDGenerator()
    if composer is None:
        composer = InputComposer()
    if vision_start_token_id is None or vision_end_token_id is None:
        raise ValueError("Qwen3.5 patch injection requires vision_start_token_id and vision_end_token_id.")

    token_embed = llm.get_input_embeddings()
    pad_token_id = getattr(llm.config, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(llm.config, 'eos_token_id', 0)
    device = input_ids.device
    dtype = token_embed.weight.dtype
    hidden_dim = token_embed.weight.shape[1]
    special_ids = torch.tensor([vision_start_token_id, vision_end_token_id], device=device, dtype=torch.long)
    vision_start_embed, vision_end_embed = token_embed(special_ids).view(2, 1, hidden_dim)

    payloads_by_sample = [[] for _ in range(batch_size)]
    if image_batch_indices is None:
        image_batch_indices = torch.arange(pixel_values.size(0), device=device).clamp(max=batch_size - 1)
    for img_idx, b_idx in enumerate(image_batch_indices.to(device=device).tolist()):
        payloads_by_sample[int(b_idx)].append((
            pixel_values[img_idx],
            vision_token_positions[img_idx],
            vision_token_valid[img_idx],
        ))

    embeds_list, labels_list, attention_list, positions_list = [], [], [], []
    patch_spans_by_sample = []
    for b_idx in range(batch_size):
        cur_ids = input_ids[b_idx, attention_mask[b_idx]]
        cur_labels = labels[b_idx, attention_mask[b_idx]]
        image_positions = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        boundaries = [-1] + image_positions + [cur_ids.numel()]
        payload_idx = 0
        keep_patch = bool(patch_sample_keep[b_idx].item())
        keep_wsi = has_wsi and bool(wsi_sample_keep[b_idx].item())
        wsi_inserted = False
        pieces_embeds, pieces_labels, pieces_attn, patch_blocks = [], [], [], []

        for seg_idx in range(len(boundaries) - 1):
            text_start = boundaries[seg_idx] + 1
            text_end = boundaries[seg_idx + 1]
            text_ids = cur_ids[text_start:text_end]
            text_labels = cur_labels[text_start:text_end]
            if text_ids.numel() > 0:
                safe_ids = _safe_token_ids(text_ids, int(pad_token_id))
                text_embeds = token_embed(safe_ids)
                pieces_embeds.append(text_embeds)
                pieces_labels.append(text_labels)
                pieces_attn.append(torch.ones(text_embeds.size(0), dtype=torch.bool, device=device))

            is_image_slot = seg_idx < len(boundaries) - 2
            if is_image_slot:
                if keep_wsi and not wsi_inserted:
                    cur_wsi = wsi_embeddings[b_idx].to(device=device, dtype=dtype)
                    pieces_embeds.append(cur_wsi)
                    pieces_labels.append(torch.full((cur_wsi.size(0),), IGNORE_INDEX,
                                                    dtype=labels.dtype, device=device))
                    pieces_attn.append(torch.ones(cur_wsi.size(0), dtype=torch.bool, device=device))
                    wsi_inserted = True

                if not keep_patch:
                    if payload_idx < len(payloads_by_sample[b_idx]):
                        payload_idx += 1
                    continue

                if payload_idx >= len(payloads_by_sample[b_idx]):
                    raise ValueError(f"Sample {b_idx} has more <image> tokens than visual feature groups.")
                patch_embeds, patch_positions, patch_valid = payloads_by_sample[b_idx][payload_idx]
                payload_idx += 1
                patch_embeds = patch_embeds.to(device=device, dtype=dtype)
                patch_positions = patch_positions.to(device=device)
                patch_valid = patch_valid.to(device=device).bool()

                current_offset = sum(x.size(0) for x in pieces_embeds)
                pieces_embeds.extend([vision_start_embed, patch_embeds, vision_end_embed])
                pieces_labels.extend([
                    torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=device),
                    torch.full((patch_embeds.size(0),), IGNORE_INDEX, dtype=labels.dtype, device=device),
                    torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=device),
                ])
                pieces_attn.extend([
                    torch.ones(1, dtype=torch.bool, device=device),
                    patch_valid,
                    torch.ones(1, dtype=torch.bool, device=device),
                ])
                patch_blocks.append((current_offset + 1, patch_positions, patch_valid,
                                     current_offset + 1 + patch_embeds.size(0)))

        if payload_idx != len(payloads_by_sample[b_idx]):
            raise ValueError(f"Sample {b_idx} has unused visual feature groups.")
        if keep_wsi and not wsi_inserted:
            cur_wsi = wsi_embeddings[b_idx].to(device=device, dtype=dtype)
            pieces_embeds = [cur_wsi] + pieces_embeds
            pieces_labels = [torch.full((cur_wsi.size(0),), IGNORE_INDEX, dtype=labels.dtype, device=device)] + pieces_labels
            pieces_attn = [torch.ones(cur_wsi.size(0), dtype=torch.bool, device=device)] + pieces_attn
            patch_blocks = [
                (start + cur_wsi.size(0), pos, valid, end + cur_wsi.size(0))
                for start, pos, valid, end in patch_blocks
            ]

        if not pieces_embeds:
            pieces_embeds.append(token_embed(torch.tensor([int(pad_token_id)], device=device)))
            pieces_labels.append(torch.full((1,), IGNORE_INDEX, dtype=labels.dtype, device=device))
            pieces_attn.append(torch.zeros(1, dtype=torch.bool, device=device))

        cur_embeds = torch.cat(pieces_embeds, dim=0)
        cur_labels = torch.cat(pieces_labels, dim=0)
        cur_attn = torch.cat(pieces_attn, dim=0)
        seq_pos = torch.arange(cur_embeds.size(0), device=device, dtype=torch.long)
        cur_pos = seq_pos.unsqueeze(0).expand(4, -1).clone()
        if patch_position_encoding == 'mrope':
            for patch_start, patch_positions, patch_valid, vision_end_idx in patch_blocks:
                cur_pos = position_generator(
                    seq_pos,
                    patch_start=patch_start,
                    token_positions=patch_positions,
                    token_valid=patch_valid,
                    vision_end_index=vision_end_idx,
                    base_position_ids=cur_pos,
                )
        embeds_list.append(cur_embeds)
        labels_list.append(cur_labels)
        attention_list.append(cur_attn)
        positions_list.append(cur_pos)
        patch_spans_by_sample.append([(int(start), int(end)) for start, _, _, end in patch_blocks])

    composed = composer(embeds_list, labels_list, attention_list, positions_list, padding_side, IGNORE_INDEX)
    max_spans = max((len(spans) for spans in patch_spans_by_sample), default=0)
    if max_spans > 0:
        span_tensor = torch.full((batch_size, max_spans, 2), -1, dtype=torch.long, device=device)
        max_len = composed['attention_mask'].size(1)
        for b_idx, spans in enumerate(patch_spans_by_sample):
            pad_offset = max_len - embeds_list[b_idx].size(0) if padding_side == 'left' else 0
            for span_idx, (start, end) in enumerate(spans):
                span_tensor[b_idx, span_idx, 0] = start + pad_offset
                span_tensor[b_idx, span_idx, 1] = end + pad_offset
        composed['vision_token_spans'] = span_tensor
    return {'input_ids': None, **composed}


class LLaVAModel_qwen3_5(BaseModel):
    """Qwen3.5 multimodal model with prompt-conditioned patch resampling."""

    SUPPORT_CONFIGS = {
        'SDPA': ('LlamaConfig', 'GemmaConfig', 'MistralConfig', 'MixtralConfig',
                 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config',
                 'Qwen3Config', 'Qwen3_5Config'),
        'FLASH2': ('InternLM2Config', 'LlamaConfig', 'GemmaConfig', 'MistralConfig',
                   'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config',
                   'Phi3Config', 'Qwen3Config', 'Qwen3_5Config'),
    }

    def __init__(
        self,
        llm,
        tokenizer,
        freeze_llm: bool = True,
        pretrained_pth: Optional[str] = None,
        pretrained_patch_resampler: Optional[str] = None,
        llm_lora: Optional[Dict] = None,
        use_activation_checkpointing: bool = True,
        max_position_embeddings: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
        stop_words: Optional[List[str]] = None,
        enable_regression: bool = True,
        reg_token: str = '<REG>',
        enable_survival: bool = True,
        srv_token: str = '<SRV>',
        num_survival_intervals: int = 6,
        survival_method: str = 'cox',
        gen_forcing: bool = True,
        lambda_llm: float = 0.1,
        lambda_reg: float = 1.0,
        lambda_srv: float = 1.0,
        prompt_resampler_cfg: Optional[Dict] = None,
        prompt_context_mode: str = 'llm_hidden',
        prompt_context_layer: Union[int, str] = 'auto',
        prompt_context_detach: bool = True,
        patch_position_encoding: str = 'linear',
        enable_nonfinite_checks: bool = False,
        wsi_feature_dims: Optional[List[int]] = None,
        wsi_dropout: float = 0.1,
        survival_head_dropout: float = 0.3,
        head_scaling: Union[float, List[float]] = (0.0, 0.0, 0.5),
        patch_modality_dropout: float = 0.0,
        wsi_modality_dropout: float = 0.0,
        modality_dropout_allow_text_only: bool = False,
        force_drop_patch: bool = False,
        force_drop_wsi: bool = False,
        trainable_module_prefixes: Optional[List[str]] = None,
        save_attention_heatmap: bool = False,
        attention_heatmap_dir: Optional[str] = None,
        save_patch_attention_h5: bool = False,
        patch_attention_h5_dir: Optional[str] = None,
        patch_attention_h5_dtype: str = 'float16',
        route_families: Optional[List[str]] = None,
        routed_lora_family_rank: Optional[int] = None,
        routed_lora_family_alpha: Optional[float] = None,
        routed_lora_trainable: str = 'all',
        freeze_patch_route_residual: bool = False,
    ):
        super().__init__()
        self.route_families = tuple(route_families or ROUTE_FAMILIES)
        if not self.route_families or len(set(self.route_families)) != len(self.route_families):
            raise ValueError(
                'route_families must contain unique non-empty family names, '
                f'got {self.route_families!r}.')
        self._route_family_context = None
        self.routed_lora_family_rank = routed_lora_family_rank
        self.routed_lora_family_alpha = routed_lora_family_alpha
        self.routed_lora_trainable = str(routed_lora_trainable).lower()
        if self.routed_lora_trainable not in ('all', 'family_only', 'shared_only'):
            raise ValueError(
                "routed_lora_trainable must be 'all', 'family_only', or "
                f"'shared_only', got {routed_lora_trainable!r}.")
        self.freeze_patch_route_residual = bool(freeze_patch_route_residual)
        self.freeze_llm = freeze_llm
        self.enable_regression = enable_regression
        self.enable_survival = enable_survival
        self.reg_token = reg_token
        self.srv_token = srv_token
        self.num_survival_intervals = num_survival_intervals
        self.survival_method = survival_method
        self.gen_forcing = gen_forcing
        self.lambda_llm = lambda_llm
        self.lambda_reg = lambda_reg
        self.lambda_srv = lambda_srv
        self.prompt_resampler_cfg = prompt_resampler_cfg
        self.prompt_context_mode = prompt_context_mode
        self.prompt_context_layer = prompt_context_layer if str(prompt_context_layer).lower() == 'auto' else int(prompt_context_layer)
        self.prompt_context_detach = bool(prompt_context_detach)
        self.patch_position_encoding = str(patch_position_encoding).lower()
        self.enable_nonfinite_checks = bool(enable_nonfinite_checks)
        self.enable_vision = prompt_resampler_cfg is not None
        self.wsi_feature_dims = wsi_feature_dims
        self.enable_wsi_injection = wsi_feature_dims is not None and len(wsi_feature_dims) > 0
        self.wsi_dropout = float(wsi_dropout)
        self.survival_head_dropout = float(survival_head_dropout)
        self.patch_modality_dropout = float(patch_modality_dropout)
        self.wsi_modality_dropout = float(wsi_modality_dropout)
        self.modality_dropout_allow_text_only = bool(modality_dropout_allow_text_only)
        self.force_drop_patch = bool(force_drop_patch)
        self.force_drop_wsi = bool(force_drop_wsi)
        self.trainable_module_prefixes = trainable_module_prefixes
        self.save_attention_heatmap = bool(save_attention_heatmap)
        self.attention_heatmap_dir = attention_heatmap_dir
        self.save_patch_attention_h5 = bool(save_patch_attention_h5)
        self.patch_attention_h5_dir = patch_attention_h5_dir
        self.patch_attention_h5_dtype = str(patch_attention_h5_dtype)
        if self.patch_attention_h5_dtype not in ('float16', 'float32'):
            raise ValueError("patch_attention_h5_dtype must be 'float16' or 'float32'.")
        self.use_llm_lora = llm_lora is not None
        self._use_llm_lora = self.use_llm_lora
        self.routed_lora_enabled = False
        self.reg_token_id = None
        self.srv_token_id = None
        self._last_hidden_state = None
        self._last_patch_attention = None
        self._last_patch_valid_mask = None
        self.is_first_iter = True
        if isinstance(head_scaling, (int, float)):
            self.head_scaling = [float(head_scaling)] * 3
        else:
            self.head_scaling = [float(x) for x in head_scaling]
            if len(self.head_scaling) != 3:
                raise ValueError("head_scaling must be a float or a length-3 list.")
        if self.prompt_context_mode not in ('embedding', 'llm_hidden'):
            raise ValueError("prompt_context_mode must be 'embedding' or 'llm_hidden'.")
        if self.patch_position_encoding not in ('linear', 'mrope'):
            raise ValueError("patch_position_encoding must be 'linear' or 'mrope'.")
        for name, value in (
            ('patch_modality_dropout', self.patch_modality_dropout),
            ('wsi_modality_dropout', self.wsi_modality_dropout),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

        self._init_llm(llm, max_position_embeddings)
        self._setup_tokenizer_and_tokens(tokenizer)
        if self.enable_vision:
            self._init_prompt_resampler()
            if self.freeze_patch_route_residual:
                self.patch_resampler.family_query_residual.requires_grad_(False)
        if self.enable_wsi_injection:
            self._init_wsi_projector()
        if self.enable_regression or self.enable_survival:
            self._init_prediction_modules()
            self._init_special_lm_head()
        self._configure_training(llm_lora, use_activation_checkpointing)
        self._setup_generation(generation_kwargs, stop_words)
        self.position_generator = MRoPEPositionIDGenerator()
        self.input_composer = InputComposer()
        if pretrained_pth:
            self.load_state_dict(guess_load_checkpoint(pretrained_pth), strict=False)
        if pretrained_patch_resampler:
            self._load_pretrained_patch_resampler(pretrained_patch_resampler)
        self._apply_trainable_module_filter()
        self._apply_routed_lora_trainable_filter()

    def _init_llm(self, llm, max_position_embeddings: Optional[int]) -> None:
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(ConfigDict(llm), max_position_embeddings)
            self.llm = self._build_from_cfg_or_module(llm)
        if hasattr(self.llm.config, 'use_cache'):
            self.llm.config.use_cache = False
        if hasattr(self.llm.config, 'text_config'):
            self.llm.config.text_config.use_cache = False
        dispatch_modules(self.llm)

    def _get_llm_hidden_size(self) -> int:
        if hasattr(self.llm.config, 'hidden_size'):
            return int(self.llm.config.hidden_size)
        if hasattr(self.llm.config, 'text_config'):
            return int(self.llm.config.text_config.hidden_size)
        raise AttributeError("Could not determine hidden_size from LLM config.")

    def _get_language_model_norm(self):
        paths_to_try = [
            lambda: self.llm.model.norm,
            lambda: self.llm.model.language_model.norm,
            lambda: self.llm.base_model.model.model.norm,
            lambda: self.llm.base_model.model.model.language_model.norm,
        ]
        for get_norm in paths_to_try:
            try:
                norm = get_norm()
                if norm is not None:
                    return norm
            except (AttributeError, TypeError):
                continue
        return None

    def _get_prompt_context_model(self):
        paths_to_try = [
            lambda: self.llm.model.language_model,
            lambda: self.llm.base_model.model.model.language_model,
            lambda: self.llm.base_model.model.language_model,
            lambda: self.llm.model,
            lambda: self.llm.base_model.model.model,
            lambda: self.llm.base_model.model,
        ]
        for get_model in paths_to_try:
            try:
                model = get_model()
                if model is not None:
                    return model
            except (AttributeError, TypeError):
                continue
        return self.llm

    def _run_prompt_context_model(self, model, **kwargs):
        kwargs.setdefault('use_cache', False)
        with self._temporary_attn_implementation('sdpa'):
            if self.prompt_context_detach:
                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        return model(**kwargs)
                finally:
                    model.train(was_training)
            return model(**kwargs)

    def _setup_tokenizer_and_tokens(self, tokenizer) -> None:
        if isinstance(tokenizer, (dict, Config, ConfigDict)):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer
        self._original_vocab_size = len(self.tokenizer)
        added_tokens = []
        if self.enable_regression:
            added_tokens.append(self.reg_token)
        if self.enable_survival:
            added_tokens.append(self.srv_token)
        if added_tokens:
            self._add_special_tokens(added_tokens)
        if self.enable_regression:
            self.reg_token_id = self.tokenizer.convert_tokens_to_ids(self.reg_token)
            self._init_token_embedding(self.reg_token_id, 'regression')
        if self.enable_survival:
            self.srv_token_id = self.tokenizer.convert_tokens_to_ids(self.srv_token)
            self._init_token_embedding(self.srv_token_id, 'survival')

        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        if self.vision_start_token_id is None or self.vision_start_token_id < 0:
            self.vision_start_token_id = getattr(self.llm.config, 'vision_start_token_id', None)
        if self.vision_end_token_id is None or self.vision_end_token_id < 0:
            self.vision_end_token_id = getattr(self.llm.config, 'vision_end_token_id', None)

        if getattr(self.llm.config, 'pad_token_id', None) is None:
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._register_embedding_grad_hook()

    def _add_special_tokens(self, tokens: List[str]) -> None:
        added = [AddedToken(t, normalized=False, special=True) for t in tokens]
        try:
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': added})
        except Exception:
            num_added = self.tokenizer.add_tokens(added, special_tokens=True)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            if hasattr(self.llm, 'tie_weights'):
                self.llm.tie_weights()

    def _init_token_embedding(self, token_id: int, task_type: str) -> None:
        emb = self.llm.get_input_embeddings()
        out = self.llm.get_output_embeddings()
        if emb is None or token_id is None or token_id >= emb.weight.size(0):
            return
        init_words = ["value", "number", "score"] if task_type == 'regression' else ["survival", "risk", "hazard"]
        valid_ids = []
        for word in init_words:
            tid = self.tokenizer.convert_tokens_to_ids(word)
            if tid is not None and 0 <= tid < emb.weight.size(0):
                valid_ids.append(tid)
        with torch.no_grad():
            base = emb.weight[valid_ids].mean(dim=0) if valid_ids else emb.weight.mean(dim=0)
            emb.weight[token_id] = base + 1e-3 * torch.randn_like(base)
            if out is not None and out is not emb and hasattr(out, 'weight') and token_id < out.weight.size(0):
                out.weight[token_id] = out.weight[valid_ids].mean(dim=0) if valid_ids else out.weight.mean(dim=0)

    def _register_embedding_grad_hook(self) -> None:
        old_vocab_size = self._original_vocab_size

        def _get_zero_hook(_name: str):
            def _zero_old_token_grad(grad: torch.Tensor) -> torch.Tensor:
                if grad is not None:
                    grad[:old_vocab_size] = 0
                return grad
            return _zero_old_token_grad

        emb = self.llm.get_input_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            if hasattr(self, '_embedding_grad_hook_handle'):
                self._embedding_grad_hook_handle.remove()
            self._embedding_grad_hook_handle = emb.weight.register_hook(_get_zero_hook('input'))

        is_tied = getattr(self.llm.config, 'tie_word_embeddings', True)
        if not is_tied:
            out = self.llm.get_output_embeddings()
            if out is not None and hasattr(out, 'weight'):
                if hasattr(self, '_output_grad_hook_handle'):
                    self._output_grad_hook_handle.remove()
                self._output_grad_hook_handle = out.weight.register_hook(_get_zero_hook('output'))

    def _estimate_embedding_rms(self) -> Optional[float]:
        emb = self.llm.get_input_embeddings()
        if emb is None or not hasattr(emb, 'weight'):
            return None
        with torch.no_grad():
            return float(emb.weight.detach().float().pow(2).mean(dim=-1).sqrt().mean().item())

    def _init_prompt_resampler(self) -> None:
        cfg = dict(
            patch_dim=768,
            llm_hidden_size=self._get_llm_hidden_size(),
            resampler_dim=min(2048, self._get_llm_hidden_size()),
            num_query=8,
            num_layers=2,
            num_heads=8,
            dropout=0.0,
            use_local_conv=True,
            query_init_std=0.5,
        )
        cfg.update(self.prompt_resampler_cfg or {})
        cfg['llm_hidden_size'] = self._get_llm_hidden_size()
        cfg['route_families'] = self.route_families
        self.patch_resampler = PromptConditionedPatchResampler(**cfg)
        self.patch_resampler.set_output_rms(self._estimate_embedding_rms())
        print_log(f"[PromptResampler] cfg={cfg}", 'current')

    def _load_pretrained_patch_resampler(self, ckpt_path: str) -> None:
        """Initialize patch_resampler from a baseline perceiver checkpoint.

        Accepts the baseline `3_linear_prob_v5_patch_baseline_perceiver.py`
        outputs (e.g. `unified_perceiver_*.pt` / `unified_masked_mlp_*.pt`).
        Those carry the resampler weights under `llava_patch_resampler`
        (already prefixed `patch_resampler.`) with `perceiver`/`patch_resampler`
        as fallbacks. Only `self.patch_resampler` is touched; everything else in
        this model keeps its warm-start / fresh init. Initialization-only.
        """
        if not self.enable_vision or not hasattr(self, 'patch_resampler'):
            print_log(
                f"[PatchResamplerInit] vision disabled; skip {ckpt_path}",
                'current')
            return
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Locate the resampler sub-state-dict and normalize keys to bare
        # PromptConditionedPatchResampler names (strip a `patch_resampler.`
        # prefix when present).
        resampler_state = None
        for block_key in ('llava_patch_resampler', 'patch_resampler', 'perceiver'):
            block = ckpt.get(block_key) if isinstance(ckpt, dict) else None
            if not isinstance(block, dict) or not block:
                continue
            if block_key == 'perceiver':
                block = block.get('resampler') or block.get('patch_resampler')
                if not isinstance(block, dict) or not block:
                    continue
            resampler_state = OrderedDict(
                (k[len('patch_resampler.'):] if k.startswith('patch_resampler.') else k, v)
                for k, v in block.items()
            )
            break
        if resampler_state is None:
            raise KeyError(
                f"No patch-resampler weights found in {ckpt_path}; expected one "
                "of keys 'llava_patch_resampler' / 'patch_resampler' / 'perceiver'.")

        # Guard against silent dim mismatch (wrong Qwen size / patch_dim).
        target = self.patch_resampler.state_dict()
        mismatched = [
            k for k, v in resampler_state.items()
            if k in target and tuple(v.shape) != tuple(target[k].shape)
        ]
        if mismatched:
            raise ValueError(
                f"Patch-resampler shape mismatch loading {ckpt_path} "
                f"(check Qwen hidden size / patch_dim): {mismatched[:5]}"
                f"{' ...' if len(mismatched) > 5 else ''}")

        incompatible = self.patch_resampler.load_state_dict(resampler_state, strict=False)
        if is_main_process():
            ck_query = ckpt.get('perceiver_num_query') if isinstance(ckpt, dict) else None
            print_log(
                f"[PatchResamplerInit] loaded {len(resampler_state)} tensors from "
                f"{ckpt_path} | baseline_num_query={ck_query} "
                f"model_num_query={self.patch_resampler.num_query} | "
                f"missing={len(getattr(incompatible, 'missing_keys', []))} "
                f"unexpected={len(getattr(incompatible, 'unexpected_keys', []))}",
                'current')

    def _init_wsi_projector(self) -> None:
        self.wsi_projector = WSIProjector(
            wsi_input_dims=self.wsi_feature_dims,
            llm_hidden_size=self._get_llm_hidden_size(),
            hidden_mult=self.head_scaling[2],
            dropout=self.wsi_dropout,
        )

    def _init_prediction_modules(self) -> None:
        hidden = self._get_llm_hidden_size()
        if self.enable_regression:
            self.regression_head = RegressionHead(hidden, hidden_mult=self.head_scaling[0])
            self.regression_loss_fn = nn.SmoothL1Loss(beta=1.0)
        if self.enable_survival:
            self.survival_head = SurvivalHead(
                hidden,
                method=self.survival_method,
                num_intervals=self.num_survival_intervals,
                hidden_mult=self.head_scaling[1],
                dropout=self.survival_head_dropout,
            )
            self.survival_loss_fn = cox_ph_loss if self.survival_method == 'cox' else logistic_hazard_loss

    def _task_token_ids(self) -> List[int]:
        token_ids = []
        if self.enable_regression and self.reg_token_id is not None:
            token_ids.append(int(self.reg_token_id))
        if self.enable_survival and self.srv_token_id is not None:
            token_ids.append(int(self.srv_token_id))
        return token_ids

    def _init_special_lm_head(self) -> None:
        token_ids = self._task_token_ids()
        if not token_ids:
            self.special_lm_token_ids = []
            self.special_lm_head = None
            return

        hidden = self._get_llm_hidden_size()
        self.special_lm_token_ids = token_ids
        self.special_lm_head = nn.Linear(hidden, len(token_ids), bias=True)

        out = self.llm.get_output_embeddings()
        if out is not None and hasattr(out, 'weight'):
            with torch.no_grad():
                weight = out.weight[token_ids].detach().float()
                self.special_lm_head.weight.copy_(weight.to(dtype=self.special_lm_head.weight.dtype))
                self.special_lm_head.bias.zero_()

    def _apply_special_lm_logits(
        self,
        active_logits: torch.Tensor,
        active_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if not getattr(self, 'special_lm_token_ids', None) or self.special_lm_head is None:
            return active_logits
        token_ids = torch.as_tensor(self.special_lm_token_ids, device=active_logits.device, dtype=torch.long)
        special_logits = self.special_lm_head(
            active_hidden.to(dtype=next(self.special_lm_head.parameters()).dtype)
        )
        active_logits = active_logits.to(dtype=special_logits.dtype).clone()
        active_logits.index_copy_(1, token_ids, special_logits)
        return active_logits

    def _configure_training(self, llm_lora: Optional[Dict], use_activation_checkpointing: bool) -> None:
        if llm_lora is not None:
            lora_cfg = self._build_from_cfg_or_module(llm_lora)
            rank = int(getattr(lora_cfg, 'r', 0))
            alpha = float(getattr(lora_cfg, 'lora_alpha', rank))
            dropout = float(getattr(lora_cfg, 'lora_dropout', 0.0) or 0.0)
            family_rank = (
                rank if self.routed_lora_family_rank is None
                else int(self.routed_lora_family_rank))
            family_alpha = (
                alpha if self.routed_lora_family_alpha is None
                else float(self.routed_lora_family_alpha))
            if rank <= 0:
                raise ValueError(f'LoRA config must define a positive r, got {rank}.')
            if family_rank <= 0:
                raise ValueError(
                    'routed_lora_family_rank must be positive, '
                    f'got {family_rank}.')

            # Freeze the complete Qwen base before inserting the additive
            # routed branches. This also keeps the base weights out of the
            # optimizer and makes Stage 1/Stage 2 boundaries explicit.
            self.llm.requires_grad_(False)
            if getattr(lora_cfg, 'target_modules', None) is None:
                # For Qwen3-VL, we must avoid targeting 'proj' which matches Conv3d in visual encoder.
                # Searching only in language_model avoids finding 'proj' from vision blocks.
                target_model = getattr(self.llm, 'model', self.llm)
                target_model = getattr(target_model, 'language_model', target_model)
                lora_cfg.target_modules = find_all_linear_names(target_model)
            target_model = getattr(self.llm, 'model', self.llm)
            target_model = getattr(target_model, 'language_model', target_model)
            replaced = replace_linear_with_routed_lora(
                target_model,
                lora_cfg.target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                route_families=self.route_families,
                family_rank=family_rank,
                family_alpha=family_alpha,
            )
            if replaced == 0:
                raise RuntimeError(
                    f'No language-model linear layers matched LoRA targets '
                    f'{lora_cfg.target_modules!r}.')
            self.routed_lora_enabled = True
            if is_main_process():
                print_log(
                    f'[RoutedLoRA] replaced {replaced} linear layers; '
                    f'shared_rank={rank}, shared_alpha={alpha}, '
                    f'family_rank={family_rank}, family_alpha={family_alpha}, '
                    f'dropout={dropout}, '
                    f'families={self.route_families}',
                    'current')
        elif self.freeze_llm:
            self.llm.requires_grad_(False)

        # Special-token rows are initialized before LoRA setup. Keep the full
        # BF16 vocab matrices out of AdamW; LoRA and task heads learn the task.
        emb = self.llm.get_input_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            emb.weight.requires_grad = False
        out = self.llm.get_output_embeddings()
        if out is not None and hasattr(out, 'weight'):
            out.weight.requires_grad = False
        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()
        if self.routed_lora_trainable == 'all':
            self._log_trainable_parameters()

    def _apply_trainable_module_filter(self) -> None:
        if self.trainable_module_prefixes is None:
            return

        prefixes = tuple(str(prefix) for prefix in self.trainable_module_prefixes)
        for name, param in self.named_parameters():
            param.requires_grad = any(
                name == prefix or name.startswith(f'{prefix}.')
                for prefix in prefixes
            )

        if is_main_process():
            print_log(
                f"[TrainableFilter] trainable_module_prefixes={list(prefixes)}",
                'current',
            )
        self._log_trainable_parameters()

    def _apply_routed_lora_trainable_filter(self) -> None:
        mode = self.routed_lora_trainable
        if mode == 'all':
            return
        if not self.routed_lora_enabled:
            raise ValueError(
                f'routed_lora_trainable={mode!r} requires routed LoRA.')

        marker = '.family_lora.' if mode == 'family_only' else '.shared_lora.'
        trainable = 0
        for name, param in self.named_parameters():
            param.requires_grad_(marker in name)
            if param.requires_grad:
                trainable += param.numel()
        if trainable == 0:
            raise RuntimeError(
                f'routed_lora_trainable={mode!r} selected no parameters.')
        if is_main_process():
            print_log(
                f'[RoutedLoRA] trainable_mode={mode}; '
                f'trainable_parameters={trainable:,}',
                'current')
        self._log_trainable_parameters()

    def _setup_generation(self, generation_kwargs: Optional[Dict], stop_words: Optional[List[str]]) -> None:
        gen_kwargs = dict(generation_kwargs or {})
        # Training disables the model-level cache for gradient checkpointing,
        # but autoregressive inference should explicitly re-enable it.
        gen_kwargs.setdefault('use_cache', True)
        if 'eos_token_id' not in gen_kwargs and self.tokenizer.eos_token_id is not None:
            gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        if 'pad_token_id' not in gen_kwargs and self.tokenizer.pad_token_id is not None:
            gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        self.generation_config = GenerationConfig(**gen_kwargs)
        if hasattr(self.llm, 'generation_config'):
            self.llm.generation_config = self.generation_config
        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words or []:
            self.stop_criteria.append(StopWordStoppingCriteria(self.tokenizer, word))

    def _maybe_raise_if_nonfinite(self, name: str, value: torch.Tensor, extra: str = "") -> None:
        if not self.enable_nonfinite_checks:
            return
        _raise_if_nonfinite(name, value, extra=extra)

    @contextmanager
    def _temporary_attn_implementation(self, implementation: Optional[str]):
        """Temporarily override runtime attention backend for fragile eval paths."""
        if implementation is None:
            yield
            return

        targets = []
        llm_config = getattr(self.llm, 'config', None)
        configs = [llm_config, getattr(llm_config, 'text_config', None)]
        try:
            prompt_config = getattr(self._get_prompt_context_model(), 'config', None)
            configs.extend([prompt_config, getattr(prompt_config, 'text_config', None)])
        except Exception:
            pass

        seen_configs = set()
        for cfg in configs:
            if cfg is None or id(cfg) in seen_configs:
                continue
            seen_configs.add(id(cfg))
            old_values = {}
            for attr in ('_attn_implementation', 'attn_implementation'):
                if hasattr(cfg, attr):
                    old_values[attr] = getattr(cfg, attr)
                    setattr(cfg, attr, implementation)
            if old_values:
                targets.append((cfg, old_values))

        try:
            yield
        finally:
            for cfg, old_values in targets:
                for attr, old_value in old_values.items():
                    setattr(cfg, attr, old_value)

    def _resolve_prompt_context_layer(self, hidden_states) -> int:
        if str(self.prompt_context_layer).lower() == 'auto':
            num_llm_layers = max(len(hidden_states) - 1, 1)
            return min(len(hidden_states) - 1, max(1, round(num_llm_layers * 2 / 3)))
        layer = int(self.prompt_context_layer)
        if not (-len(hidden_states) <= layer < len(hidden_states)):
            raise IndexError(
                f"prompt_context_layer={layer} is out of range for "
                f"{len(hidden_states)} hidden-state tensors."
            )
        return layer if layer >= 0 else len(hidden_states) + layer

    def _prompt_mask(self, data: Dict[str, Any]) -> torch.Tensor:
        input_ids = data['input_ids']
        labels = data.get('labels')
        attention_mask = data.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        prompt_mask = attention_mask.bool() & (input_ids != IMAGE_TOKEN_INDEX)
        if labels is not None:
            prompt_mask = prompt_mask & (labels == IGNORE_INDEX)
        empty = ~prompt_mask.any(dim=1)
        fallback_mask = attention_mask.bool() & (input_ids != IMAGE_TOKEN_INDEX)
        fallback_idx = fallback_mask.long().argmax(dim=1, keepdim=True)
        fallback_one = torch.zeros_like(prompt_mask)
        fallback_one.scatter_(1, fallback_idx, True)
        prompt_mask = torch.where(empty.unsqueeze(1), fallback_one, prompt_mask)
        return prompt_mask

    def _text_context_embeds(
        self,
        input_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_token_id = self.llm.config.pad_token_id or self.tokenizer.eos_token_id or 0
        safe_ids = _safe_token_ids(input_ids, int(pad_token_id))
        token_embed = self.llm.get_input_embeddings()
        inputs_embeds = token_embed(safe_ids)
        embeds_list, labels_list, attention_list, pos_list = [], [], [], []
        for b_idx in range(input_ids.size(0)):
            cur_mask = prompt_mask[b_idx]
            cur_embeds = inputs_embeds[b_idx, cur_mask]
            cur_len = cur_embeds.size(0)
            cur_attn = torch.ones(cur_len, dtype=torch.bool, device=cur_embeds.device)
            cur_pos = torch.arange(cur_len, device=cur_embeds.device, dtype=torch.long)
            embeds_list.append(cur_embeds)
            labels_list.append(torch.full((cur_len,), IGNORE_INDEX, dtype=torch.long, device=cur_embeds.device))
            attention_list.append(cur_attn)
            pos_list.append(cur_pos.unsqueeze(0).expand(4, -1))
        prompt_inputs = self.input_composer(
            embeds_list,
            labels_list,
            attention_list,
            pos_list,
            'left',
            IGNORE_INDEX,
        )

        if self.prompt_context_mode == 'embedding':
            return prompt_inputs['inputs_embeds'], prompt_inputs['attention_mask']

        prompt_model = self._get_prompt_context_model()
        outputs = self._run_prompt_context_model(
            prompt_model,
            inputs_embeds=prompt_inputs['inputs_embeds'],
            attention_mask=prompt_inputs['attention_mask'],
            position_ids=prompt_inputs['position_ids'],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise TypeError(f"Prompt context model did not return hidden_states: {type(outputs)!r}")
        selected_layer = self._resolve_prompt_context_layer(hidden_states)
        context_embeds = hidden_states[selected_layer]
        if selected_layer == len(hidden_states) - 1:
            final_norm = self._get_language_model_norm()
            if final_norm is not None:
                context_embeds = final_norm(context_embeds)
        self._maybe_raise_if_nonfinite(
            'prompt_context_embeds',
            context_embeds,
            extra=f"mode={self.prompt_context_mode} layer={self.prompt_context_layer}",
        )
        if self.prompt_context_detach:
            context_embeds = context_embeds.detach()
        return context_embeds, prompt_inputs['attention_mask']

    def _prompt_embeds_for_images(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = data['input_ids']
        prompt_mask = self._prompt_mask(data)
        prompt_embeds, prompt_mask = self._text_context_embeds(input_ids, prompt_mask)
        image_batch_indices = data.get('image_batch_indices')
        if image_batch_indices is None:
            image_batch_indices = torch.arange(data['features'].size(0), device=input_ids.device).clamp(max=input_ids.size(0) - 1)
        image_batch_indices = image_batch_indices.to(device=input_ids.device, dtype=torch.long)
        return prompt_embeds[image_batch_indices], prompt_mask[image_batch_indices]

    def _normalize_route_family(self, route_family: Any, batch_size: int,
                                device: torch.device) -> torch.Tensor:
        """Validate and encode the required per-sample route field."""
        if route_family is None:
            raise ValueError(
                'Missing route_family. Every sample must provide one of '
                f'{self.route_families!r}; route is never inferred from the prompt.')
        if torch.is_tensor(route_family):
            if route_family.dtype.is_floating_point:
                raise ValueError('route_family tensor must contain integer ids, not floats.')
            route_ids = route_family.to(device=device, dtype=torch.long).view(-1)
        else:
            if isinstance(route_family, str):
                route_family = [route_family]
            try:
                route_values = list(route_family)
            except TypeError as exc:
                raise ValueError('route_family must be a per-sample sequence.') from exc
            if len(route_values) != batch_size:
                raise ValueError(
                    f'route_family must have {batch_size} entries, got {len(route_values)}.')
            route_to_id = {family: idx for idx, family in enumerate(self.route_families)}
            invalid = [value for value in route_values if value not in route_to_id]
            if invalid:
                raise ValueError(
                    f'Invalid route_family values {invalid!r}; expected one of '
                    f'{self.route_families!r}.')
            route_ids = torch.tensor(
                [route_to_id[value] for value in route_values],
                device=device,
                dtype=torch.long,
            )
        if route_ids.numel() != batch_size:
            raise ValueError(
                f'route_family must be per-sample with shape [{batch_size}], '
                f'got {tuple(route_ids.shape)}.')
        if bool(((route_ids < 0) | (route_ids >= len(self.route_families))).any().item()):
            raise ValueError(
                f'Invalid route_family ids {route_ids.tolist()}; expected '
                f'0..{len(self.route_families) - 1}.')
        return route_ids

    @contextmanager
    def _routed_lora_context(self, route_family: Optional[torch.Tensor]):
        """Install one route for all LoRA calls in a forward/generation pass."""
        if not self.routed_lora_enabled:
            yield
            return
        previous = []
        for module in self.llm.modules():
            if isinstance(module, RoutedLoRALinear):
                previous.append((module, module._route_family))
                module.set_route_family(route_family)
        if not previous:
            raise RuntimeError('routed_lora_enabled=True but no routed linear modules exist.')
        try:
            yield
        finally:
            # Gradient-checkpointed transformer blocks are recomputed during
            # backward, after this forward context has exited. Keep the route
            # installed until the next serial training forward overwrites it;
            # otherwise recomputation sees route=None. Eval/generation has no
            # deferred recomputation and can restore the previous context.
            keep_for_backward = (
                self.training
                and torch.is_grad_enabled()
                and bool(getattr(self.llm, 'is_gradient_checkpointing', False))
            )
            if not keep_for_backward:
                for module, old_route in previous:
                    module.set_route_family(old_route)

    def _make_modality_keep_masks(
        self,
        batch_size: int,
        has_patch: bool,
        has_wsi: bool,
        device: torch.device,
        mode: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        patch_keep = torch.ones(batch_size, dtype=torch.bool, device=device) if has_patch else None
        wsi_keep = torch.ones(batch_size, dtype=torch.bool, device=device) if has_wsi else None

        if patch_keep is not None and self.force_drop_patch:
            patch_keep.zero_()
        if wsi_keep is not None and self.force_drop_wsi:
            wsi_keep.zero_()

        if mode == 'loss' and self.training:
            if patch_keep is not None and not self.force_drop_patch and self.patch_modality_dropout > 0:
                patch_keep &= torch.rand(batch_size, device=device) >= self.patch_modality_dropout
            if wsi_keep is not None and not self.force_drop_wsi and self.wsi_modality_dropout > 0:
                wsi_keep &= torch.rand(batch_size, device=device) >= self.wsi_modality_dropout

        if (
            not self.modality_dropout_allow_text_only
            and patch_keep is not None
            and wsi_keep is not None
        ):
            both_dropped = ~patch_keep & ~wsi_keep
            if both_dropped.any():
                can_restore_patch = not self.force_drop_patch
                can_restore_wsi = not self.force_drop_wsi
                if can_restore_patch and can_restore_wsi:
                    restore_patch = torch.rand(batch_size, device=device) < 0.5
                    restore_patch &= both_dropped
                    patch_keep |= restore_patch
                    wsi_keep |= both_dropped & ~restore_patch
                elif can_restore_patch:
                    patch_keep |= both_dropped
                elif can_restore_wsi:
                    wsi_keep |= both_dropped

        return patch_keep, wsi_keep

    def _project_vision_features(
        self,
        data: Dict[str, Any],
        route_family: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.enable_vision:
            raise RuntimeError("Received patch features but prompt_resampler_cfg is None.")
        param = next(self.patch_resampler.parameters())
        features = data['features'].to(device=param.device, dtype=param.dtype)
        feature_shapes = data.get('feature_shapes')
        if feature_shapes is not None:
            feature_shapes = feature_shapes.to(device=param.device)
        prompt_embeds, prompt_mask = self._prompt_embeds_for_images(data)
        prompt_embeds = prompt_embeds.to(device=param.device, dtype=param.dtype)
        prompt_mask = prompt_mask.to(device=param.device)
        image_batch_indices = data.get('image_batch_indices')
        if image_batch_indices is None:
            image_batch_indices = torch.arange(
                features.size(0), device=route_family.device, dtype=torch.long)
        image_batch_indices = image_batch_indices.to(
            device=route_family.device, dtype=torch.long).view(-1)
        if image_batch_indices.numel() != features.size(0):
            raise ValueError(
                'image_batch_indices must contain one sample index per image: '
                f'{image_batch_indices.numel()} vs {features.size(0)}.')
        image_route_family = route_family.index_select(0, image_batch_indices)
        out = self.patch_resampler(
            features=features,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            feature_shapes=feature_shapes,
            route_family=image_route_family,
        )
        self._last_patch_attention = out['patch_attention'].detach()
        self._last_patch_valid_mask = out['patch_valid_mask'].detach()
        return {
            'pixel_values': out['visual_tokens'],
            'visual_tokens': out['visual_tokens'],
            'vision_token_positions': out['token_positions'],
            'vision_token_valid': out['token_valid'],
            'patch_attention': out['patch_attention'],
            'patch_valid_mask': out['patch_valid_mask'],
            'region_attention': out.get('region_attention'),
            'region_attention_heads': out.get('region_attention_heads'),
            'visual_to_region_attention': out.get('visual_to_region_attention'),
            'visual_to_region_attention_heads': out.get('visual_to_region_attention_heads'),
            'token_positions': out['token_positions'],
        }

    @staticmethod
    def _safe_path_name(value: Any, default: str) -> str:
        value = default if value is None else str(value).rstrip('/')
        value = value or default
        value = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._')
        return value or default

    @staticmethod
    def _normalize_heatmap_image(heatmap: np.ndarray, valid_mask: np.ndarray) -> Image.Image:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        values = heatmap[valid_mask]
        if values.size == 0:
            values = heatmap.reshape(-1)
        lo = float(np.nanmin(values)) if values.size else 0.0
        hi = float(np.nanmax(values)) if values.size else 0.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            arr = np.zeros_like(heatmap, dtype=np.uint8)
        else:
            arr = np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        arr = np.where(valid_mask, arr, 0).astype(np.uint8)
        return Image.fromarray(arr, mode='L')

    @staticmethod
    def _cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        flat = x.reshape(x.shape[0], -1).astype(np.float64, copy=False)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        sim = (flat @ flat.T) / (norms * norms.T)
        return sim.astype(np.float32, copy=False)

    @staticmethod
    def _write_h5_string(group, name: str, value: Any) -> None:
        dtype = h5py.string_dtype(encoding='utf-8')
        group.create_dataset(name, data='' if value is None else str(value), dtype=dtype)

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)

    def _patch_attention_np_dtype(self):
        return np.float16 if self.patch_attention_h5_dtype == 'float16' else np.float32

    def _source_patch_index_grid(self, feature_path: Optional[str], feature_shape: Tuple[int, int]) -> np.ndarray:
        h_final, w_final = int(feature_shape[0]), int(feature_shape[1])
        dense = np.full((max(h_final, 0), max(w_final, 0)), -1, dtype=np.int64)
        if not feature_path or h_final <= 0 or w_final <= 0:
            return dense.reshape(-1)

        cache = getattr(self, '_patch_source_index_cache', None)
        if cache is None:
            cache = {}
            self._patch_source_index_cache = cache
        cache_key = (str(feature_path), h_final, w_final)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            with h5py.File(feature_path, 'r') as f:
                coords = np.asarray(f['coords'][:], dtype=np.int64)
                patch_size = int(f['coords'].attrs.get('patch_size_level0', 512))
        except Exception:
            cache[cache_key] = dense.reshape(-1)
            return cache[cache_key]

        if coords.size == 0:
            cache[cache_key] = dense.reshape(-1)
            return cache[cache_key]
        grid_coords = coords // max(patch_size, 1)
        min_coords = grid_coords.min(axis=0)
        max_coords = grid_coords.max(axis=0)
        shifted = grid_coords - min_coords
        orig_h = int(max_coords[1] - min_coords[1] + 1)
        orig_w = int(max_coords[0] - min_coords[0] + 1)
        padded_h = max(orig_h, h_final)
        padded_w = max(orig_w, w_final)
        top = (padded_h - h_final) // 2
        left = (padded_w - w_final) // 2
        rows = shifted[:, 1] - top
        cols = shifted[:, 0] - left
        keep = (rows >= 0) & (rows < h_final) & (cols >= 0) & (cols < w_final)
        for src_idx, row, col in zip(np.nonzero(keep)[0], rows[keep], cols[keep]):
            dense[int(row), int(col)] = int(src_idx)
        cache[cache_key] = dense.reshape(-1)
        return cache[cache_key]

    def _build_patch_attention_payload(self, data: Dict[str, Any], projected: Dict[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
        if not self.save_patch_attention_h5 or not self.patch_attention_h5_dir:
            return None
        heads = projected.get('region_attention_heads')
        valid_mask = projected.get('patch_valid_mask')
        if heads is None or valid_mask is None:
            return None
        image_batch_indices = data.get('image_batch_indices')
        if torch.is_tensor(image_batch_indices):
            image_batch_indices = image_batch_indices.detach().cpu().tolist()
        else:
            image_batch_indices = list(range(int(heads.size(0))))
        feature_shapes = data.get('feature_shapes')
        if torch.is_tensor(feature_shapes):
            feature_shapes = feature_shapes.detach().cpu().tolist()
        return {
            'region_attention_heads': heads.detach(),
            'patch_valid_mask': valid_mask.detach(),
            'token_positions': projected.get('token_positions').detach() if projected.get('token_positions') is not None else None,
            'vision_token_valid': projected.get('vision_token_valid').detach() if projected.get('vision_token_valid') is not None else None,
            'image_batch_indices': image_batch_indices,
            'feature_shapes': feature_shapes,
            'feature_paths': list(data.get('feature_paths') or []),
            'sample_ids': list(data.get('id') or []),
            'categories': list(data.get('category') or []),
            'projects': list(data.get('project') or []),
            'divisions': list(data.get('division') or []),
            'labels_text': list(data.get('labels_text') or []),
            'raw_sample_json': list(data.get('raw_sample_json') or []),
            'wsi_feature_paths': list(data.get('wsi_feature_paths') or []),
        }

    def _collect_next_token_visual_attention(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        if not self.save_patch_attention_h5 or not self.patch_attention_h5_dir:
            return None
        spans = data.get('vision_token_spans')
        if spans is None or not torch.is_tensor(spans) or spans.numel() == 0:
            return None
        spans_cpu = spans.detach().cpu()
        attention_mask = data.get('attention_mask')
        if attention_mask is None:
            return None
        last_valid = attention_mask.bool().long()
        seq_idx = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
        last_valid = (seq_idx * last_valid).max(dim=1).values
        llm_kwargs = {k: data[k] for k in ['inputs_embeds', 'attention_mask', 'position_ids'] if k in data}
        with torch.no_grad(), self._temporary_attn_implementation('eager'):
            outputs = self.llm(
                **llm_kwargs,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        attentions = getattr(outputs, 'attentions', None)
        if not attentions:
            return None
        batch_size, max_spans = spans_cpu.shape[:2]
        max_query = int((spans_cpu[..., 1] - spans_cpu[..., 0]).clamp_min(0).max().item())
        if max_query <= 0:
            return None
        num_layers = len(attentions)
        num_heads = int(attentions[0].size(1))
        out = np.zeros((batch_size, num_layers, num_heads, max_spans, max_query), dtype=self._patch_attention_np_dtype())
        for layer_idx, layer_attn in enumerate(attentions):
            for b_idx in range(batch_size):
                target = int(last_valid[b_idx].item())
                for span_idx in range(max_spans):
                    start = int(spans_cpu[b_idx, span_idx, 0].item())
                    end = int(spans_cpu[b_idx, span_idx, 1].item())
                    if start < 0 or end <= start:
                        continue
                    row = layer_attn[b_idx, :, target, start:end].detach().float().cpu().numpy()
                    out[b_idx, layer_idx, :, span_idx, :row.shape[-1]] = row.astype(out.dtype, copy=False)
        return out

    def _patch_attention_h5_path(self, sample_id: Any, category: Any, feature_path: Any, image_idx: int) -> str:
        slide_stem = os.path.basename(str(feature_path or f'image_{image_idx}'))
        if slide_stem.endswith('.h5'):
            slide_stem = slide_stem[:-3]
        safe_slide = self._safe_path_name(slide_stem, f'image_{image_idx}')
        safe_id = self._safe_path_name(sample_id, f'sample_{image_idx}')
        digest = hashlib.sha1(f'{sample_id}|{feature_path}|{image_idx}'.encode('utf-8')).hexdigest()[:8]
        filename = f'{safe_slide}__{safe_id}__{digest}.attn.h5'
        safe_category = self._safe_path_name(category, 'unknown_category')
        shard = digest[:2]
        return os.path.join(self.patch_attention_h5_dir, safe_category, shard, filename)

    def _save_patch_attention_h5_files(
        self,
        payload: Optional[Dict[str, Any]],
        llm_visual_attention: Optional[np.ndarray],
        data_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not payload or not self.save_patch_attention_h5 or not self.patch_attention_h5_dir:
            return
        attn_heads = payload['region_attention_heads'].float().cpu().numpy()
        valid_masks = payload['patch_valid_mask'].cpu().numpy().astype(bool)
        token_positions = payload.get('token_positions')
        if token_positions is not None:
            token_positions = token_positions.cpu().numpy()
        token_valid = payload.get('vision_token_valid')
        if token_valid is not None:
            token_valid = token_valid.cpu().numpy().astype(bool)

        dtype = self._patch_attention_np_dtype()
        counts_by_sample = {}
        for image_idx in range(attn_heads.shape[0]):
            sample_idx = int(payload['image_batch_indices'][image_idx]) if image_idx < len(payload['image_batch_indices']) else image_idx
            span_idx = counts_by_sample.get(sample_idx, 0)
            counts_by_sample[sample_idx] = span_idx + 1
            sample_id = payload['sample_ids'][sample_idx] if sample_idx < len(payload['sample_ids']) else None
            category = payload['categories'][sample_idx] if sample_idx < len(payload['categories']) else None
            project = payload['projects'][sample_idx] if sample_idx < len(payload['projects']) else None
            division = payload['divisions'][sample_idx] if sample_idx < len(payload['divisions']) else None
            feature_path = payload['feature_paths'][image_idx] if image_idx < len(payload['feature_paths']) else None
            feature_shape = list(valid_masks[image_idx].shape)
            if payload.get('feature_shapes') is not None and image_idx < len(payload['feature_shapes']):
                feature_shape = [int(x) for x in payload['feature_shapes'][image_idx]]

            mask_flat = valid_masks[image_idx].reshape(-1)
            valid_flat = np.nonzero(mask_flat)[0].astype(np.int32)
            sparse_attn = attn_heads[image_idx].reshape(attn_heads.shape[1], attn_heads.shape[2], -1)[:, :, valid_flat]
            sparse_attn = sparse_attn.astype(dtype, copy=False)
            source_grid = self._source_patch_index_grid(feature_path, tuple(feature_shape))
            source_indices = source_grid[valid_flat].astype(np.int64, copy=False) if valid_flat.size else np.empty((0,), dtype=np.int64)

            b_attn = np.empty((0, 0, 0), dtype=dtype)
            if llm_visual_attention is not None and sample_idx < llm_visual_attention.shape[0] and span_idx < llm_visual_attention.shape[3]:
                b_attn = llm_visual_attention[sample_idx, :, :, span_idx, :sparse_attn.shape[1]].astype(dtype, copy=False)

            out_path = self._patch_attention_h5_path(sample_id, category, feature_path, image_idx)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            tmp_path = f'{out_path}.tmp'
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            with h5py.File(tmp_path, 'w') as h5:
                h5.attrs['schema_version'] = 'patch_resampler_attention.v1'
                h5.attrs['qa_id'] = '' if sample_id is None else str(sample_id)
                h5.attrs['category'] = '' if category is None else str(category)
                h5.attrs['project'] = '' if project is None else str(project)
                h5.attrs['division'] = '' if division is None else str(division)
                h5.attrs['patch_feature_path'] = '' if feature_path is None else str(feature_path)
                h5.attrs['attention_dtype'] = self.patch_attention_h5_dtype
                h5.attrs['llm_visual_attention_saved'] = bool(b_attn.size)

                qa = h5.create_group('qa')
                raw_json = payload['raw_sample_json'][sample_idx] if sample_idx < len(payload['raw_sample_json']) else ''
                self._write_h5_string(qa, 'raw_sample_json', raw_json)
                self._write_h5_string(qa, 'answer_text', payload['labels_text'][sample_idx] if sample_idx < len(payload['labels_text']) else '')
                if data_samples is not None and sample_idx < len(data_samples):
                    self._write_h5_string(qa, 'prediction_json', self._json_dumps(data_samples[sample_idx]))

                ref = h5.create_group('patch_ref')
                self._write_h5_string(ref, 'patch_feature_path', feature_path)
                self._write_h5_string(ref, 'feature_key', 'features')
                self._write_h5_string(ref, 'coords_key', 'coords')
                ref.create_dataset('grid_shape', data=np.asarray(feature_shape, dtype=np.int32))
                ref.create_dataset('valid_flat_indices', data=valid_flat, compression='lzf', shuffle=True)
                ref.create_dataset('source_patch_indices', data=source_indices, compression='lzf', shuffle=True)
                wsi_paths = payload['wsi_feature_paths'][sample_idx] if sample_idx < len(payload['wsi_feature_paths']) else None
                self._write_h5_string(ref, 'wsi_feature_paths_json', self._json_dumps(wsi_paths))

                attn = h5.create_group('attention')
                attn.create_dataset('resampler_cross_attn', data=sparse_attn, compression='lzf', shuffle=True)
                attn.create_dataset('next_token_source_attn', data=b_attn, compression='lzf', shuffle=True)
                if token_positions is not None:
                    attn.create_dataset('token_positions', data=token_positions[image_idx].astype(np.int16, copy=False))
                if token_valid is not None:
                    attn.create_dataset('token_valid', data=token_valid[image_idx])
            os.replace(tmp_path, out_path)

    def _save_attention_heatmaps(self, data: Dict[str, Any], projected: Dict[str, torch.Tensor]) -> None:
        if not self.save_attention_heatmap or not self.attention_heatmap_dir:
            return
        patch_attention = projected.get('patch_attention')
        valid_mask = projected.get('patch_valid_mask')
        region_attention = projected.get('region_attention')
        region_attention_heads = projected.get('region_attention_heads')
        visual_to_region_attention = projected.get('visual_to_region_attention')
        visual_to_region_attention_heads = projected.get('visual_to_region_attention_heads')
        visual_tokens = projected.get('visual_tokens')
        token_positions = projected.get('token_positions')
        if patch_attention is None or valid_mask is None:
            return

        patch_attention = patch_attention.detach().float().cpu().numpy()
        valid_mask = valid_mask.detach().cpu().numpy().astype(bool)
        if region_attention is not None:
            region_attention = region_attention.detach().float().cpu().numpy()
        if region_attention_heads is not None:
            region_attention_heads = region_attention_heads.detach().float().cpu().numpy()
        if visual_to_region_attention is not None:
            visual_to_region_attention = visual_to_region_attention.detach().float().cpu().numpy()
        if visual_to_region_attention_heads is not None:
            visual_to_region_attention_heads = visual_to_region_attention_heads.detach().float().cpu().numpy()
        if visual_tokens is not None:
            visual_tokens = visual_tokens.detach().float().cpu().numpy()
        token_positions = token_positions.detach().cpu().numpy() if token_positions is not None else None
        feature_shapes = data.get('feature_shapes')
        if torch.is_tensor(feature_shapes):
            feature_shapes = feature_shapes.detach().cpu().tolist()
        image_batch_indices = data.get('image_batch_indices')
        if torch.is_tensor(image_batch_indices):
            image_batch_indices = image_batch_indices.detach().cpu().tolist()
        else:
            image_batch_indices = list(range(patch_attention.shape[0]))
        feature_paths = data.get('feature_paths') or [None] * patch_attention.shape[0]
        sample_ids = data.get('id') or []
        categories = data.get('category') or []
        projects = data.get('project') or []

        os.makedirs(self.attention_heatmap_dir, exist_ok=True)
        counts_by_sample = {}
        for image_idx in range(patch_attention.shape[0]):
            if image_idx < len(image_batch_indices):
                sample_idx = int(image_batch_indices[image_idx])
            else:
                sample_idx = image_idx
            raw_id = sample_ids[sample_idx] if sample_idx < len(sample_ids) else None
            fallback = feature_paths[image_idx] if image_idx < len(feature_paths) else None
            fallback = os.path.basename(str(fallback)) if fallback else None
            fallback_id = self._safe_path_name(fallback, f'sample_{sample_idx}')
            case_id = self._safe_path_name(raw_id, fallback_id)
            case_dir = os.path.join(self.attention_heatmap_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)

            per_sample_count = counts_by_sample.get(sample_idx, 0)
            counts_by_sample[sample_idx] = per_sample_count + 1
            suffix = '' if counts_by_sample[sample_idx] == 1 else f'_image{per_sample_count}'

            attn = patch_attention[image_idx]
            mask = valid_mask[image_idx]
            mean_heatmap = attn.mean(axis=0)
            max_heatmap = attn.max(axis=0)
            region_attn = None
            region_mean_heatmap = None
            region_max_heatmap = None
            if region_attention is not None:
                region_attn = region_attention[image_idx].reshape(region_attention.shape[1], *mask.shape)
                region_attn = np.where(mask[None, :, :], region_attn, 0.0)
                region_mean_heatmap = region_attn.mean(axis=0)
                region_max_heatmap = region_attn.max(axis=0)
            visual_to_region = (
                visual_to_region_attention[image_idx]
                if visual_to_region_attention is not None
                else np.empty((0, 0), dtype=np.float32)
            )
            region_attn_heads = (
                region_attention_heads[image_idx]
                if region_attention_heads is not None
                else np.empty((0, 0, 0), dtype=np.float32)
            )
            visual_to_region_heads = (
                visual_to_region_attention_heads[image_idx]
                if visual_to_region_attention_heads is not None
                else np.empty((0, 0, 0), dtype=np.float32)
            )
            visual_token_embeds = (
                visual_tokens[image_idx]
                if visual_tokens is not None
                else np.empty((0, 0), dtype=np.float32)
            )
            visual_token_cosine = self._cosine_similarity_matrix(visual_token_embeds)
            if visual_to_region_heads.size and region_attn_heads.size:
                composed_patch_attention_same_head = np.einsum(
                    'hvr,hrp->hvp',
                    visual_to_region_heads.astype(np.float32, copy=False),
                    region_attn_heads.astype(np.float32, copy=False),
                )
                composed_patch_attention_same_head = np.where(
                    mask.reshape(1, 1, -1),
                    composed_patch_attention_same_head,
                    0.0,
                )
                composed_patch_attention_same_head = composed_patch_attention_same_head / np.clip(
                    composed_patch_attention_same_head.sum(axis=-1, keepdims=True),
                    1e-6,
                    None,
                )
                composed_patch_attention_same_head = composed_patch_attention_same_head.reshape(
                    composed_patch_attention_same_head.shape[0],
                    composed_patch_attention_same_head.shape[1],
                    *mask.shape,
                )
            else:
                composed_patch_attention_same_head = np.empty((0, 0, *mask.shape), dtype=np.float32)
            token_pos = (
                token_positions[image_idx]
                if token_positions is not None
                else np.empty((0, 2), dtype=np.int64)
            )
            np.savez_compressed(
                os.path.join(case_dir, f'attention{suffix}.npz'),
                patch_attention=attn,
                mean_heatmap=mean_heatmap,
                max_heatmap=max_heatmap,
                valid_mask=mask,
                token_positions=token_pos,
                region_attention=(
                    region_attn if region_attn is not None
                    else np.empty((0, *mask.shape), dtype=np.float32)
                ),
                region_mean_heatmap=(
                    region_mean_heatmap if region_mean_heatmap is not None
                    else np.empty(mask.shape, dtype=np.float32)
                ),
                region_max_heatmap=(
                    region_max_heatmap if region_max_heatmap is not None
                    else np.empty(mask.shape, dtype=np.float32)
                ),
                visual_tokens=visual_token_embeds,
                visual_token_cosine=visual_token_cosine,
                visual_to_region_attention=visual_to_region,
                region_attention_heads=region_attn_heads,
                visual_to_region_attention_heads=visual_to_region_heads,
                composed_patch_attention_same_head=composed_patch_attention_same_head,
            )
            self._normalize_heatmap_image(mean_heatmap, mask).save(
                os.path.join(case_dir, f'heatmap_mean{suffix}.png'))
            self._normalize_heatmap_image(max_heatmap, mask).save(
                os.path.join(case_dir, f'heatmap_max{suffix}.png'))
            if region_mean_heatmap is not None and region_max_heatmap is not None:
                self._normalize_heatmap_image(region_mean_heatmap, mask).save(
                    os.path.join(case_dir, f'region_heatmap_mean{suffix}.png'))
                self._normalize_heatmap_image(region_max_heatmap, mask).save(
                    os.path.join(case_dir, f'region_heatmap_max{suffix}.png'))

            feature_shape = list(mask.shape)
            if feature_shapes is not None and image_idx < len(feature_shapes):
                feature_shape = feature_shapes[image_idx]
            metadata = {
                'id': raw_id,
                'sample_index': sample_idx,
                'image_index': image_idx,
                'image_file': feature_paths[image_idx] if image_idx < len(feature_paths) else None,
                'feature_shape': feature_shape,
                'category': categories[sample_idx] if sample_idx < len(categories) else None,
                'project': projects[sample_idx] if sample_idx < len(projects) else None,
            }
            metadata_path = os.path.join(case_dir, f'metadata{suffix}.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _project_wsi_features(self, wsi_features: List[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        if not self.enable_wsi_injection or not hasattr(self, 'wsi_projector'):
            return None
        param = next(self.wsi_projector.parameters())
        batch_size = len(wsi_features)
        source_features = []
        for src_idx in range(len(wsi_features[0])):
            source_features.append(torch.stack([
                wsi_features[b][src_idx].to(device=param.device, dtype=param.dtype)
                for b in range(batch_size)
            ]))
        return self.wsi_projector(source_features)

    def forward(self, data: Dict[str, Any], data_samples: Optional[List] = None,
                mode: str = 'loss') -> Any:
        routed_data = dict(data)
        batch_size = int(routed_data['input_ids'].size(0))
        route_family = routed_data.pop('route_family', None)
        if route_family is None:
            route_family = routed_data.get('route_family_ids')
        route_family_ids = self._normalize_route_family(
            route_family, batch_size, routed_data['input_ids'].device)
        routed_data['route_family_ids'] = route_family_ids
        with self._routed_lora_context(route_family_ids):
            return self._forward_impl(routed_data, data_samples, mode)

    def _forward_impl(self, data: Dict[str, Any],
                      data_samples: Optional[List] = None,
                      mode: str = 'loss') -> Any:
        data = dict(data)
        if self.is_first_iter:
            first_tensor = next((v for v in data.values() if torch.is_tensor(v)), None)
            if first_tensor is not None:
                self.to(first_tensor.device)
            self.is_first_iter = False

        regression_targets = data.pop('regression_targets', None)
        survival_targets = data.pop('survival_targets', None)
        task_categories = data.get('category', None)
        has_visual = data.get('features') is not None
        has_wsi_features = self.enable_wsi_injection and data.get('wsi_features') is not None
        batch_size = int(data['input_ids'].size(0))
        mask_device = data['input_ids'].device
        patch_sample_keep, wsi_sample_keep = self._make_modality_keep_masks(
            batch_size=batch_size,
            has_patch=has_visual,
            has_wsi=has_wsi_features,
            device=mask_device,
            mode=mode,
        )

        attention_save_payload = None
        if has_visual:
            projected = self._project_vision_features(data, data['route_family_ids'])
            if mode == 'predict':
                self._save_attention_heatmaps(data, projected)
                attention_save_payload = self._build_patch_attention_payload(data, projected)
            data['pixel_values'] = projected['pixel_values']
            data['vision_token_positions'] = projected['vision_token_positions']
            data['vision_token_valid'] = projected['vision_token_valid']
            data['patch_sample_keep'] = patch_sample_keep
        else:
            data['pixel_values'] = None
            data['vision_token_positions'] = None
            data['vision_token_valid'] = None
            data['image_batch_indices'] = None
            data['patch_sample_keep'] = None
        data.pop('features', None)
        data.pop('feature_shapes', None)
        data.pop('feature_paths', None)

        wsi_embeddings = None
        if self.enable_wsi_injection and data.get('wsi_features') is not None:
            wsi_embeddings = self._project_wsi_features(data.pop('wsi_features'))
        else:
            data.pop('wsi_features', None)
        data['wsi_embeddings'] = wsi_embeddings
        data['wsi_sample_keep'] = wsi_sample_keep

        if mode == 'predict':
            for key in ('input_ids', 'labels', 'attention_mask'):
                if torch.is_tensor(data.get(key)):
                    data[key] = data[key].clone()
            self._strip_assistant_targets(data)
        padding_side = 'left'
        data = prepare_inputs_labels_for_qwen3_5(
            llm=self.llm,
            padding_side=padding_side,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            position_generator=self.position_generator,
            patch_position_encoding=self.patch_position_encoding,
            composer=self.input_composer,
            **data,
        )
        if attention_save_payload is not None:
            data['_patch_attention_save_payload'] = attention_save_payload
        if mode == 'loss':
            return self.compute_loss(data, data_samples, regression_targets, survival_targets)
        if mode == 'tensor':
            return self._forward(data, data_samples)
        if mode == 'predict':
            return self.predict(data, data_samples, regression_targets, survival_targets, task_categories)
        raise NotImplementedError(f"Unsupported mode: {mode}")

    def _strip_assistant_targets(self, data: Dict[str, torch.Tensor]) -> None:
        labels = data.get('labels')
        input_ids = data.get('input_ids')
        attention_mask = data.get('attention_mask')
        if labels is None or input_ids is None or attention_mask is None:
            return
        pad_token_id = self.llm.config.pad_token_id or self.tokenizer.eos_token_id or 0
        for b_idx in range(input_ids.size(0)):
            supervised = (labels[b_idx] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if supervised.numel() == 0:
                continue
            start = int(supervised[0].item())
            input_ids[b_idx, start:] = int(pad_token_id)
            labels[b_idx, start:] = IGNORE_INDEX
            attention_mask[b_idx, start:] = False

    def _forward(self, data: Dict[str, torch.Tensor], data_samples: Optional[List] = None):
        kwargs = {k: data[k] for k in ['input_ids', 'inputs_embeds', 'attention_mask', 'position_ids'] if k in data}
        return self.llm(**kwargs, use_cache=False, output_hidden_states=True, return_dict=True)

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Use the pre-weighted total loss for backward and keep components for logging only."""
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        if 'loss' in losses:
            loss = losses['loss']
        else:
            loss = sum(value for key, value in log_vars if 'loss' in key)

        log_vars_dict = OrderedDict()
        log_vars_dict['loss'] = loss
        for name, val in log_vars:
            if name != 'loss':
                log_vars_dict[name] = val

        return loss, log_vars_dict

    def compute_loss(
        self,
        data: Dict[str, torch.Tensor],
        data_samples: Optional[List] = None,
        regression_targets: Optional[torch.Tensor] = None,
        survival_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self._forward(data, data_samples)
        logits = outputs.logits
        labels = data.get('labels')
        last_hidden = outputs.hidden_states[-1]
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_hidden = last_hidden[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            active = shift_labels != IGNORE_INDEX
            if active.any():
                active_logits = self._apply_special_lm_logits(
                    shift_logits[active],
                    shift_hidden[active],
                )
                lm_loss = nn.CrossEntropyLoss()(
                    active_logits.float(),
                    shift_labels[active],
                )
            else:
                lm_loss = shift_logits.new_zeros(())
        else:
            lm_loss = logits.new_zeros(())
        self._last_hidden_state = last_hidden
        reg_loss = self._compute_task_loss(labels, regression_targets, 'regression') if (
            self.enable_regression and regression_targets is not None and labels is not None
        ) else self._regularization_loss('regression', lm_loss)
        srv_loss = self._compute_task_loss(labels, survival_targets, 'survival') if (
            self.enable_survival and survival_targets is not None and labels is not None
        ) else self._regularization_loss('survival', lm_loss)
        self._maybe_raise_if_nonfinite('lm_loss', lm_loss)
        self._maybe_raise_if_nonfinite('reg_loss', reg_loss)
        self._maybe_raise_if_nonfinite('srv_loss', srv_loss)
        loss = self.lambda_llm * lm_loss + self.lambda_reg * reg_loss + self.lambda_srv * srv_loss
        return {
            'lm_loss': lm_loss.detach(),
            'reg_loss': reg_loss.detach(),
            'srv_loss': srv_loss.detach(),
            'loss': loss,
        }

    def _compute_task_loss(self, labels: torch.Tensor, targets: Union[torch.Tensor, Dict], task: str) -> torch.Tensor:
        token_id = self.reg_token_id if task == 'regression' else self.srv_token_id
        head = self.regression_head if task == 'regression' else self.survival_head
        token_mask = labels == token_id
        if token_id is None or not token_mask.any():
            return self._regularization_loss(task, labels.sum() * 0)
        seq = torch.arange(labels.size(1), device=labels.device).unsqueeze(0).expand_as(labels)
        last_pos = torch.where(token_mask, seq, torch.full_like(seq, -1)).max(dim=1).values
        valid = last_pos >= 0
        if not valid.any():
            return self._regularization_loss(task, labels.sum() * 0)
        b_idx = valid.nonzero(as_tuple=True)[0]
        task_embeds = self._last_hidden_state[b_idx, last_pos[valid]].to(dtype=next(head.parameters()).dtype)
        if task == 'regression':
            pred = self.regression_head(task_embeds).squeeze(-1)
            target = targets.to(device=pred.device, dtype=pred.dtype)[b_idx]
            keep = torch.isfinite(target)
            if not keep.any():
                return self._regularization_loss(task, pred.sum() * 0)
            return self.regression_loss_fn(pred[keep], target[keep])
        out = self.survival_head(task_embeds)
        if self.survival_method == 'discrete':
            target_y = targets['target_y'].to(device=out.device, dtype=out.dtype)[b_idx]
            at_risk = targets['at_risk_mask'].to(device=out.device, dtype=out.dtype)[b_idx]
            return logistic_hazard_loss(out, target_y, at_risk)
        time = targets['time'].to(device=out.device, dtype=out.dtype)[b_idx]
        event = targets['event'].to(device=out.device, dtype=out.dtype)[b_idx]
        keep = ~(torch.isnan(time) | torch.isnan(event))
        if not keep.any():
            return self._regularization_loss(task, out.sum() * 0)
        return cox_ph_loss(out[keep], time[keep], event[keep])

    def _regularization_loss(self, task: str, base: torch.Tensor, scale: float = 1e-6) -> torch.Tensor:
        module = None
        if task == 'regression' and hasattr(self, 'regression_head'):
            module = self.regression_head
        if task == 'survival' and hasattr(self, 'survival_head'):
            module = self.survival_head
        if module is None:
            return base * 0
        reg = sum((p.float() ** 2).mean() for p in module.parameters() if p.requires_grad)
        return reg.to(device=base.device, dtype=base.dtype) * scale if isinstance(reg, torch.Tensor) else base * 0

    @torch.no_grad()
    def predict(
        self,
        data: Dict[str, torch.Tensor],
        data_samples: Optional[List] = None,
        regression_targets: Optional[torch.Tensor] = None,
        survival_targets: Optional[Dict[str, torch.Tensor]] = None,
        task_categories: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Generate text and compute regression/survival predictions from special tokens."""
        try:
            # Save prefix to reconstruct the full sequence
            prefix_inputs_embeds = data['inputs_embeds']  # (B, Lp, H)
            prefix_attention_mask = data['attention_mask']  # (B, Lp)
            prefix_position_ids = data.get('position_ids', None)
            B, Lp, _ = prefix_inputs_embeds.shape
            attention_save_payload = data.get('_patch_attention_save_payload')
            llm_visual_attention = self._collect_next_token_visual_attention(data) if attention_save_payload is not None else None

            # 1) Text generation with logits capture for MCQA
            with torch.no_grad(), self._temporary_attn_implementation('sdpa'):
                captured_logits = []
                def capture_logits_processor(input_ids, scores):
                    if not captured_logits:
                        captured_logits.append(scores.detach().cpu())
                    return scores

                gen_config = GenerationConfig.from_dict(self.generation_config.to_dict())
                gen_config.return_dict_in_generate = True
                gen_config.output_scores = True
                if self.tokenizer.bos_token_id is not None:
                    gen_config.bos_token_id = self.tokenizer.bos_token_id

                gen_kwargs = {k: data[k] for k in ['inputs_embeds', 'attention_mask', 'position_ids'] if k in data}
                gen_out = self.llm.generate(
                    **gen_kwargs,
                    generation_config=gen_config,
                    stopping_criteria=self.stop_criteria,
                    logits_processor=[capture_logits_processor],
                )

            generate_ids = getattr(gen_out, 'sequences', gen_out)
            gen_scores = getattr(gen_out, 'scores', None)

            batch_size = generate_ids.size(0)
            if data_samples is None:
                data_samples = [{} for _ in range(batch_size)]

            # Decode and detect special tokens in generated continuation
            has_regression, has_survival = [], []

            # MCQA choice token id cache
            def _choice_token_id(letter: str) -> Optional[int]:
                cache = getattr(self, '_mcqa_choice_token_id_cache', None)
                if cache is None:
                    cache = {}
                    setattr(self, '_mcqa_choice_token_id_cache', cache)
                if letter in cache:
                    return cache[letter]
                candidates = [letter, f" {letter}"]
                token_id = None
                for cand in candidates:
                    try:
                        ids = self.tokenizer.encode(cand, add_special_tokens=False)
                        if isinstance(ids, list) and len(ids) == 1:
                            token_id = int(ids[0])
                            break
                    except Exception:
                        continue
                cache[letter] = token_id
                return token_id

            first_step_logits = captured_logits[0] if captured_logits else None
            if first_step_logits is None and gen_scores is not None and len(gen_scores) > 0:
                first_step_logits = gen_scores[0]

            for i, gen_id in enumerate(generate_ids):
                clean_text = self.tokenizer.decode(gen_id, skip_special_tokens=True).strip()
                data_samples[i]['prediction_text'] = clean_text

                # Attach choice logits for MCQA metrics
                if first_step_logits is not None and first_step_logits.size(0) >= batch_size:
                    beam_factor = first_step_logits.size(0) // batch_size
                    logits_row = first_step_logits[i * beam_factor]
                    choice_logits = {}
                    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        tid = _choice_token_id(letter)
                        if tid is None:
                            continue
                        try:
                            choice_logits[letter] = float(logits_row[tid].detach().cpu().item())
                        except Exception:
                            continue
                    if choice_logits:
                        data_samples[i]['mcqa_choice_logits'] = choice_logits

                has_reg = (self.enable_regression and self.reg_token_id is not None
                           and (gen_id == self.reg_token_id).any().item())
                has_srv = (self.enable_survival and self.srv_token_id is not None
                           and (gen_id == self.srv_token_id).any().item())

                # When gen_forcing=False, predict only tasks supervised for this sample
                if not self.gen_forcing:
                    has_reg = (
                        self.enable_regression
                        and self.reg_token_id is not None
                        and (
                            self._has_regression_target(regression_targets, i)
                            or self._category_contains(task_categories, i, 'regression')
                        )
                    )
                    has_srv = (
                        self.enable_survival
                        and self.srv_token_id is not None
                        and self._has_survival_target(survival_targets, i)
                    )

                has_regression.append(bool(has_reg))
                has_survival.append(bool(has_srv))

            # If no sample generated any special tokens, return text-only predictions
            if not (any(has_regression) or any(has_survival)):
                self._save_patch_attention_h5_files(attention_save_payload, llm_visual_attention, data_samples)
                return data_samples

            # 2) Task predictions from generated special tokens
            with self._temporary_attn_implementation('sdpa'):
                data_samples = self._predict_tasks_from_generation(
                    generate_ids=generate_ids,
                    data_samples=data_samples,
                    has_regression=has_regression,
                    has_survival=has_survival,
                    prefix_inputs_embeds=prefix_inputs_embeds,
                    prefix_attention_mask=prefix_attention_mask,
                    prefix_position_ids=prefix_position_ids,
                )
            self._save_patch_attention_h5_files(attention_save_payload, llm_visual_attention, data_samples)
            return data_samples

        finally:
            self._cleanup_prediction_state()

    def _predict_tasks_from_generation(
        self,
        generate_ids: torch.Tensor,
        data_samples: List[Dict[str, Any]],
        has_regression: List[bool],
        has_survival: List[bool],
        prefix_inputs_embeds: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Compute task predictions at generated special-token positions."""
        device = prefix_inputs_embeds.device
        dtype = prefix_inputs_embeds.dtype
        B, Lp, H = prefix_inputs_embeds.shape
        Lg = generate_ids.size(1)

        # When gen_forcing=False, always use explicit task token construction
        if not self.gen_forcing:
            return self._predict_tasks_from_prefix_tokens(
                data_samples=data_samples,
                has_regression=has_regression,
                has_survival=has_survival,
                prefix_inputs_embeds=prefix_inputs_embeds,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids,
            )

        # Build embeddings for generated tokens
        with torch.no_grad():
            tok_emb = self.llm.get_input_embeddings()
            gen_embeds = tok_emb(generate_ids.to(device))  # (B, Lg, H)

            # Assemble full sequence: [prefix embeds] + [generated embeds]
            full_inputs_embeds = torch.cat([prefix_inputs_embeds, gen_embeds.to(dtype)], dim=1)

            # Attention mask for generated tokens
            pad_id = self.llm.config.pad_token_id
            if pad_id is None and hasattr(self.tokenizer, 'pad_token_id'):
                pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                gen_attn = (generate_ids != pad_id).to(dtype=prefix_attention_mask.dtype, device=device)
            else:
                gen_attn = torch.ones((B, Lg), dtype=prefix_attention_mask.dtype, device=device)
            full_attention_mask = torch.cat([prefix_attention_mask.to(device), gen_attn], dim=1)

            # Position IDs: Qwen3.5 uses (4, B, L) MRoPE format
            full_position_ids = None
            if prefix_position_ids is not None:
                prefix_position_ids = prefix_position_ids.to(device)
                if prefix_position_ids.dim() == 3 and prefix_position_ids.size(0) >= 2:
                    # MRoPE: (rows, B, Lp) - continue monotonically for generated text
                    last_pos = prefix_position_ids[:, :, -1:].max(dim=0, keepdim=False)[0]  # (B, 1)
                    incr = torch.arange(1, Lg + 1, device=device).view(1, -1)  # (1, Lg)
                    gen_pos = last_pos + incr  # (B, Lg)
                    rows = prefix_position_ids.size(0)
                    gen_pos_expanded = gen_pos.unsqueeze(0).expand(rows, -1, -1)  # (rows, B, Lg)
                    full_position_ids = torch.cat([prefix_position_ids, gen_pos_expanded], dim=2)
                else:
                    # Standard 2D position_ids (B, L)
                    last_pos = prefix_position_ids[:, -1].unsqueeze(1)  # (B, 1)
                    incr = torch.arange(1, Lg + 1, device=device).view(1, -1)
                    gen_pos = last_pos + incr
                    full_position_ids = torch.cat([prefix_position_ids, gen_pos], dim=1)

            # Forward through LLM to get hidden states
            llm_kwargs = {
                'inputs_embeds': full_inputs_embeds,
                'attention_mask': full_attention_mask,
                'position_ids': full_position_ids,
                'output_hidden_states': True,
                'return_dict': True,
            }
            outputs = self.llm(**llm_kwargs)
            hidden = outputs.hidden_states[-1]  # (B, Lp+Lg, H)

            # Find last valid position per sample (left-padding aware)
            _seq_indices = torch.arange(full_attention_mask.size(1), device=device).unsqueeze(0)
            last_valid_pos = (_seq_indices * full_attention_mask.bool().long()).max(dim=1).values  # (B,)

            # For each batch, locate generated special-token positions and predict
            for b in range(B):
                # Regression
                if has_regression[b] and self.enable_regression and self.reg_token_id is not None:
                    pos_in_gen = torch.nonzero(generate_ids[b] == self.reg_token_id, as_tuple=False).flatten()
                    embed = None
                    if pos_in_gen.numel() > 0:
                        pos_full = int(Lp + pos_in_gen[-1].item())
                        if pos_full < hidden.size(1):
                            embed = hidden[b, pos_full]  # (H,)
                    if embed is not None:
                        task_embeds = embed.unsqueeze(0).to(dtype=next(self.regression_head.parameters()).dtype)
                        pred_out = self.regression_head(task_embeds)
                        pred = float(pred_out.squeeze(-1).item()) if pred_out is not None else None
                        data_samples[b]['regression_prediction'] = pred
                        prev = data_samples[b].get('prediction_text', '')
                        text_suffix = f"[Regression: {pred:.4f}]" if pred is not None else ""
                        data_samples[b]['prediction_text'] = f"{prev} {text_suffix}".strip()

                # Survival (supports both Cox and Discrete methods)
                if has_survival[b] and self.enable_survival and self.srv_token_id is not None:
                    pos_in_gen = torch.nonzero(generate_ids[b] == self.srv_token_id, as_tuple=False).flatten()
                    embed = None
                    if pos_in_gen.numel() > 0:
                        pos_full = int(Lp + pos_in_gen[-1].item())
                        if pos_full < hidden.size(1):
                            embed = hidden[b, pos_full]  # (H,)
                    if embed is None:
                        continue

                    task_embeds = embed.unsqueeze(0).to(dtype=next(self.survival_head.parameters()).dtype)
                    pred_dict = {}

                    if self.survival_method == 'discrete':
                        survival_probs_out = self.survival_head.predict_survival_probs(task_embeds)
                        survival_probs = survival_probs_out.squeeze(0).cpu().tolist() if survival_probs_out is not None else None

                        risk_score_out = self.survival_head.predict_risk_scores(task_embeds)
                        risk_score = float(risk_score_out.squeeze(0).item()) if risk_score_out is not None else None

                        median_time_out = self.survival_head.predict_median_survival_time(task_embeds)
                        median_time = float(median_time_out.squeeze(0).item()) if median_time_out is not None else None

                        pred_dict = {
                            "method": "discrete",
                            "risk_score": risk_score,
                            "survival_probs": survival_probs,
                            "median_survival_time": median_time,
                        }

                        suffixes = []
                        if risk_score is not None:
                            suffixes.append(f"Risk: {risk_score:.4f}")
                        if median_time is not None:
                            suffixes.append(f"Median: {median_time:.2f}")
                        text_suffix = f"[{', '.join(suffixes)}]" if suffixes else ""
                    else:
                        # Cox method: predict risk score only
                        risk_score_out = self.survival_head.predict_risk_scores(task_embeds)
                        risk_score = float(risk_score_out.squeeze(0).item()) if risk_score_out is not None else None
                        pred_dict = {
                            "method": "cox",
                            "risk_score": risk_score,
                        }
                        text_suffix = f"[Risk Score: {risk_score:.4f}]" if risk_score is not None else ""

                    data_samples[b]["survival_prediction"] = pred_dict
                    data_samples[b]["risk_score"] = pred_dict["risk_score"]
                    prev = data_samples[b].get("prediction_text", "")
                    data_samples[b]["prediction_text"] = f"{prev} {text_suffix}".strip()

        return data_samples

    def _predict_tasks_from_prefix_tokens(
        self,
        data_samples: List[Dict[str, Any]],
        has_regression: List[bool],
        has_survival: List[bool],
        prefix_inputs_embeds: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Append task tokens to the prefix and predict from those hidden states.

        When gen_forcing=False, instead of using the last generated token's hidden
        state (distribution mismatch), we explicitly append <REG>/<SRV> tokens to
        the prefix and run one forward pass, reading hidden states at those positions.
        This matches the training setup exactly.
        """
        device = prefix_inputs_embeds.device
        dtype = prefix_inputs_embeds.dtype
        tok_emb = self.llm.get_input_embeddings()
        embed_list = []
        attn_list = []
        pos_list = []
        reg_positions = []
        srv_positions = []

        if prefix_position_ids is not None:
            prefix_position_ids = prefix_position_ids.to(device)

        for b_idx in range(prefix_inputs_embeds.size(0)):
            cur_mask = prefix_attention_mask[b_idx].bool()
            valid_len = int(cur_mask.sum().item())
            if valid_len == 0:
                embed_list.append(torch.zeros(0, prefix_inputs_embeds.size(-1), device=device, dtype=dtype))
                attn_list.append(torch.zeros(0, device=device, dtype=torch.bool))
                pos_list.append(None)
                reg_positions.append(None)
                srv_positions.append(None)
                continue

            # Gather attended prefix tokens by mask
            cur_embeds = prefix_inputs_embeds[b_idx][cur_mask]
            cur_attn = cur_mask[cur_mask]

            cur_pos = None
            if prefix_position_ids is not None:
                if prefix_position_ids.dim() == 3:
                    # (rows, B, L) -> (rows, valid_len)
                    cur_pos = prefix_position_ids[:, b_idx, :][:, cur_mask]
                else:
                    cur_pos = prefix_position_ids[b_idx][cur_mask]

            suffix_ids = []
            reg_pos = None
            srv_pos = None
            if has_regression[b_idx] and self.enable_regression and self.reg_token_id is not None:
                reg_pos = valid_len + len(suffix_ids)
                suffix_ids.append(self.reg_token_id)
            if has_survival[b_idx] and self.enable_survival and self.srv_token_id is not None:
                srv_pos = valid_len + len(suffix_ids)
                suffix_ids.append(self.srv_token_id)

            if suffix_ids:
                suffix_tensor = torch.tensor(suffix_ids, device=device, dtype=torch.long)
                suffix_embeds = tok_emb(suffix_tensor).to(dtype=dtype)
                cur_embeds = torch.cat([cur_embeds, suffix_embeds], dim=0)
                cur_attn = torch.cat([
                    cur_attn,
                    torch.ones(len(suffix_ids), device=device, dtype=torch.bool),
                ], dim=0)

                # Extend position IDs
                if cur_pos is not None:
                    if cur_pos.dim() == 2:
                        # MRoPE (rows, valid_len) - text tokens use same linear position
                        last_scalar = int(cur_pos.max().item()) if cur_pos.numel() > 0 else -1
                        suffix_pos = torch.arange(
                            last_scalar + 1,
                            last_scalar + 1 + len(suffix_ids),
                            device=device,
                            dtype=cur_pos.dtype,
                        ).view(1, -1).expand(cur_pos.size(0), -1)
                        cur_pos = torch.cat([cur_pos, suffix_pos], dim=1)
                    else:
                        last_p = int(cur_pos[-1].item()) if cur_pos.numel() > 0 else -1
                        cur_pos = torch.cat([
                            cur_pos,
                            torch.arange(last_p + 1, last_p + 1 + len(suffix_ids),
                                         device=device, dtype=cur_pos.dtype),
                        ], dim=0)

            embed_list.append(cur_embeds)
            attn_list.append(cur_attn)
            pos_list.append(cur_pos)
            reg_positions.append(reg_pos)
            srv_positions.append(srv_pos)

        # Pad sequences
        max_len = max(x.size(0) for x in embed_list)
        batch_size = len(embed_list)
        hidden_dim = embed_list[0].size(-1)
        inputs_embeds = torch.zeros((batch_size, max_len, hidden_dim), device=device, dtype=dtype)
        attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.bool)

        has_mrope_pos = pos_list[0] is not None and pos_list[0].dim() == 2
        position_ids = None
        if pos_list[0] is not None:
            if has_mrope_pos:
                rows = pos_list[0].size(0)
                position_ids = torch.zeros((rows, batch_size, max_len), device=device, dtype=pos_list[0].dtype)
            else:
                position_ids = torch.zeros((batch_size, max_len), device=device, dtype=pos_list[0].dtype)

        for b_idx, (emb, attn, pids) in enumerate(zip(embed_list, attn_list, pos_list)):
            cur_len = emb.size(0)
            inputs_embeds[b_idx, :cur_len] = emb
            attention_mask[b_idx, :cur_len] = attn
            if position_ids is not None and pids is not None:
                if has_mrope_pos:
                    position_ids[:, b_idx, :cur_len] = pids
                else:
                    position_ids[b_idx, :cur_len] = pids

        with torch.no_grad():
            llm_kwargs = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'output_hidden_states': True,
                'return_dict': True,
            }
            outputs = self.llm(**llm_kwargs)
            hidden = outputs.hidden_states[-1]

        for b_idx in range(batch_size):
            if reg_positions[b_idx] is not None and self.enable_regression:
                reg_embed = hidden[b_idx, reg_positions[b_idx]].unsqueeze(0)
                reg_embed = reg_embed.to(dtype=next(self.regression_head.parameters()).dtype)
                pred_out = self.regression_head(reg_embed)
                pred = float(pred_out.squeeze(-1).item())
                data_samples[b_idx]['regression_prediction'] = pred
                prev = data_samples[b_idx].get('prediction_text', '')
                data_samples[b_idx]['prediction_text'] = f"{prev} [Regression: {pred:.4f}]".strip()

            if srv_positions[b_idx] is not None and self.enable_survival:
                srv_embed = hidden[b_idx, srv_positions[b_idx]].unsqueeze(0)
                srv_embed = srv_embed.to(dtype=next(self.survival_head.parameters()).dtype)
                pred_dict = {}
                if self.survival_method == 'discrete':
                    survival_probs_out = self.survival_head.predict_survival_probs(srv_embed)
                    risk_score_out = self.survival_head.predict_risk_scores(srv_embed)
                    median_time_out = self.survival_head.predict_median_survival_time(srv_embed)
                    pred_dict = {
                        'method': 'discrete',
                        'risk_score': float(risk_score_out.squeeze(0).item()) if risk_score_out is not None else None,
                        'survival_probs': survival_probs_out.squeeze(0).cpu().tolist() if survival_probs_out is not None else None,
                        'median_survival_time': float(median_time_out.squeeze(0).item()) if median_time_out is not None else None,
                    }
                else:
                    risk_score_out = self.survival_head.predict_risk_scores(srv_embed)
                    pred_dict = {
                        'method': 'cox',
                        'risk_score': float(risk_score_out.squeeze(0).item()) if risk_score_out is not None else None,
                    }
                data_samples[b_idx]['survival_prediction'] = pred_dict
                data_samples[b_idx]['risk_score'] = pred_dict['risk_score']

        return data_samples

    @staticmethod
    def _has_regression_target(targets: Optional[torch.Tensor], idx: int) -> bool:
        if targets is None:
            return False
        try:
            value = targets[idx]
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all().item())
            return math.isfinite(float(value))
        except Exception:
            return False

    @staticmethod
    def _has_survival_target(targets: Optional[Dict[str, torch.Tensor]], idx: int) -> bool:
        if not isinstance(targets, dict):
            return False
        try:
            time = targets.get('time', None)
            event = targets.get('event', None)
            if time is None or event is None:
                return False
            time_value = time[idx]
            event_value = event[idx]
            if isinstance(time_value, torch.Tensor):
                time_ok = bool(torch.isfinite(time_value).all().item())
            else:
                time_ok = math.isfinite(float(time_value))
            if isinstance(event_value, torch.Tensor):
                event_ok = bool(torch.isfinite(event_value).all().item())
            else:
                event_ok = math.isfinite(float(event_value))
            return time_ok and event_ok
        except Exception:
            return False

    @staticmethod
    def _category_contains(categories: Optional[Any], idx: int, keyword: str) -> bool:
        if categories is None:
            return False
        if isinstance(categories, (list, tuple)):
            category = categories[idx] if idx < len(categories) else ''
        else:
            category = categories
        if isinstance(category, (list, tuple)):
            category = category[0] if category else ''
        return keyword.lower() in str(category).lower()

    def _cleanup_prediction_state(self) -> None:
        """Clean up temporary prediction state."""
        pass

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.llm, 'enable_input_require_grads'):
            self.llm.enable_input_require_grads()
        else:
            self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.llm.gradient_checkpointing_disable()

    activation_checkpointing_enable = gradient_checkpointing_enable
    activation_checkpointing_disable = gradient_checkpointing_disable

    def init_weights(self) -> None:
        pass

    def _special_token_row_items(self) -> List[Tuple[str, int]]:
        items = []
        if self.enable_regression and self.reg_token_id is not None:
            items.append(('reg', int(self.reg_token_id)))
        if self.enable_survival and self.srv_token_id is not None:
            items.append(('srv', int(self.srv_token_id)))
        return items

    def _add_special_token_rows_to_state_dict(self, keep: OrderedDict) -> None:
        token_items = self._special_token_row_items()
        if not token_items:
            return

        emb = self.llm.get_input_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            for name, token_id in token_items:
                if 0 <= token_id < emb.weight.size(0):
                    keep[f'special_token_embeddings.input.{name}'] = emb.weight[token_id].detach().clone()

        out = self.llm.get_output_embeddings()
        if out is not None and out is not emb and hasattr(out, 'weight'):
            for name, token_id in token_items:
                if 0 <= token_id < out.weight.size(0):
                    keep[f'special_token_embeddings.output.{name}'] = out.weight[token_id].detach().clone()

    def _load_special_token_rows(self, rows: Dict[str, torch.Tensor]) -> None:
        if not rows:
            return
        token_items = dict(self._special_token_row_items())

        emb = self.llm.get_input_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            with torch.no_grad():
                for name, token_id in token_items.items():
                    key = f'special_token_embeddings.input.{name}'
                    if key in rows and 0 <= token_id < emb.weight.size(0):
                        emb.weight[token_id].copy_(rows[key].to(device=emb.weight.device, dtype=emb.weight.dtype))

        out = self.llm.get_output_embeddings()
        if out is not None and out is not emb and hasattr(out, 'weight'):
            with torch.no_grad():
                for name, token_id in token_items.items():
                    key = f'special_token_embeddings.output.{name}'
                    if key in rows and 0 <= token_id < out.weight.size(0):
                        out.weight[token_id].copy_(rows[key].to(device=out.weight.device, dtype=out.weight.dtype))

    @staticmethod
    def _routed_external_key(key: str) -> Optional[str]:
        """Map an internal wrapped-LLM key to a stable checkpoint key."""
        if not key.startswith('llm.'):
            return None
        shared_marker = '.shared_lora.'
        if shared_marker in key:
            layer, parameter = key[4:].split(shared_marker, 1)
            return f'routed_lora.shared.{layer}.{parameter}'
        family_marker = '.family_lora.'
        if family_marker in key:
            layer, family_and_parameter = key[4:].split(family_marker, 1)
            family, parameter = family_and_parameter.split('.', 1)
            return f'routed_lora.family.{family}.{layer}.{parameter}'
        return None

    @staticmethod
    def _routed_internal_key(key: str) -> Optional[str]:
        """Map stable routed checkpoint keys back to wrapped-LLM keys."""
        if key.startswith('routed_lora.shared.'):
            rest = key[len('routed_lora.shared.'):]
            layer, parameter = rest.rsplit('.', 2)[0], '.'.join(rest.rsplit('.', 2)[1:])
            return f'llm.{layer}.shared_lora.{parameter}'
        if key.startswith('routed_lora.family.'):
            rest = key[len('routed_lora.family.'):]
            family, rest = rest.split('.', 1)
            layer, parameter = rest.rsplit('.', 2)[0], '.'.join(rest.rsplit('.', 2)[1:])
            return f'llm.{layer}.family_lora.{family}.{parameter}'
        return None

    def state_dict(self, *args, **kwargs) -> OrderedDict:
        state_dict = super().state_dict(*args, **kwargs)
        keep = OrderedDict()
        if self.routed_lora_enabled:
            for key, value in state_dict.items():
                external_key = self._routed_external_key(key)
                if external_key is not None:
                    keep[external_key] = value
        elif self.use_llm_lora:
            keep.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            keep.update({k: v for k, v in state_dict.items() if k.startswith('llm.')})
        trainable_keys = [
            'patch_resampler.',
            'wsi_projector.',
            'regression_head.',
            'survival_head.',
            'special_lm_head.',
        ]
        keep.update({k: v for k, v in state_dict.items() if any(key in k for key in trainable_keys)})
        gate_keys = ('vision_token_gate', 'wsi_token_gate')
        keep.update({k: v for k, v in state_dict.items() if k in gate_keys})
        self._add_special_token_rows_to_state_dict(keep)
        return keep

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        state_dict = OrderedDict(state_dict)
        special_rows = OrderedDict(
            (k, state_dict.pop(k))
            for k in list(state_dict.keys())
            if k.startswith('special_token_embeddings.')
        )

        new_state_dict = OrderedDict()
        is_lora_model = bool(self.routed_lora_enabled or self.use_llm_lora)
        is_lora_ckpt = any('lora_A' in k or 'routed_lora.' in k for k in state_dict)
        mapped_count = 0
        llm_keys_count = 0

        for key, value in state_dict.items():
            new_key = key
            if self.routed_lora_enabled:
                routed_key = self._routed_internal_key(key)
                if routed_key is not None:
                    new_state_dict[routed_key] = value
                    continue
            if key.startswith('llm.'):
                llm_keys_count += 1
                if self.routed_lora_enabled and ('lora_A' in key or 'lora_B' in key):
                    # Old PEFT checkpoints are treated as the shared branch;
                    # newly introduced family branches retain their zero init.
                    new_key = key.replace('llm.base_model.model.', 'llm.', 1)
                    if '.lora_A.default.' in new_key:
                        new_key = new_key.replace('.lora_A.default.', '.shared_lora.lora_A.')
                    elif '.lora_A.' in new_key:
                        new_key = new_key.replace('.lora_A.', '.shared_lora.lora_A.')
                    if '.lora_B.default.' in new_key:
                        new_key = new_key.replace('.lora_B.default.', '.shared_lora.lora_B.')
                    elif '.lora_B.' in new_key:
                        new_key = new_key.replace('.lora_B.', '.shared_lora.lora_B.')
                    mapped_count += 1
                elif not is_lora_model and is_lora_ckpt:
                    new_key = key.replace('llm.base_model.model.', 'llm.', 1)
                    mapped_count += 1
            new_state_dict[new_key] = value

        if not strict:
            full_model_keys = set(super().state_dict().keys())
            unexpected_keys = sorted(set(new_state_dict) - full_model_keys)

            checkpoint_keys = {
                key for key in full_model_keys
                if (
                    '.shared_lora.' in key
                    or '.family_lora.' in key
                    or key.startswith('patch_resampler.')
                    or key.startswith('wsi_projector.')
                    or key.startswith('regression_head.')
                    or key.startswith('survival_head.')
                    or key.startswith('special_lm_head.')
                    or key in ('vision_token_gate', 'wsi_token_gate')
                )
            }
            missing_checkpoint_keys = sorted(checkpoint_keys - set(new_state_dict))
            is_legacy_checkpoint = not any(
                key.startswith('routed_lora.') for key in state_dict)
            if is_legacy_checkpoint:
                missing_checkpoint_keys = [
                    key for key in missing_checkpoint_keys
                    if not (
                        '.shared_lora.' in key
                        or '.family_lora.' in key
                        or key.startswith('patch_resampler.family_query_residual.')
                    )
                ]
            if missing_checkpoint_keys or unexpected_keys:
                details = []
                if missing_checkpoint_keys:
                    details.append(
                        f'missing non-routed keys: {missing_checkpoint_keys[:20]!r}')
                if unexpected_keys:
                    details.append(
                        f'unexpected keys: {unexpected_keys[:20]!r}')
                raise RuntimeError(
                    'Checkpoint is incompatible with the current model; '
                    + '; '.join(details))

        incompatible = super().load_state_dict(new_state_dict, strict=strict)
        self._load_special_token_rows(special_rows)
        if is_main_process():
            mode = 'LoRA' if is_lora_model else 'Full/Alignment'
            ckpt_type = 'LoRA' if is_lora_ckpt else 'Full/Alignment'
            missing = len(getattr(incompatible, 'missing_keys', []))
            unexpected = len(getattr(incompatible, 'unexpected_keys', []))
            print_log(
                f'[WeightLoading] Loaded {len(state_dict)} keys | '
                f'LLM: {llm_keys_count} | Remapped: {mapped_count} | '
                f'Checkpoint: {ckpt_type} -> Model: {mode} | '
                f'SpecialRows: {len(special_rows)} | Missing: {missing} | Unexpected: {unexpected}',
                'current',
            )
        return incompatible

    @staticmethod
    def _is_kbit_model(model: nn.Module) -> bool:
        return bool(
            getattr(model, 'is_loaded_in_4bit', False)
            or getattr(model, 'is_loaded_in_8bit', False)
            or getattr(getattr(model, 'config', None), 'quantization_config', None) is not None
        )

    def _get_peft_model_without_bnb_dispatch(self,
                                             model: nn.Module,
                                             lora_config: Any,
                                             disable_bnb_dispatch: bool = False) -> nn.Module:
        """Build a PEFT LoRA model without touching bitsandbytes when k-bit is unused."""
        if not disable_bnb_dispatch:
            return get_peft_model(model, lora_config)

        try:
            import peft.import_utils as peft_import_utils
            import peft.tuners.lora.model as peft_lora_model
        except Exception:
            return get_peft_model(model, lora_config)

        patched_symbols = []
        for module in (peft_import_utils, peft_lora_model):
            for symbol in ('is_bnb_available', 'is_bnb_4bit_available'):
                if hasattr(module, symbol):
                    patched_symbols.append((module, symbol, getattr(module, symbol)))
                    setattr(module, symbol, lambda: False)

        try:
            print_log('[LoRA] Detected non-quantized model; skip bitsandbytes PEFT dispatch.', 'current')
            return get_peft_model(model, lora_config)
        finally:
            for module, symbol, original in patched_symbols:
                setattr(module, symbol, original)

    @staticmethod
    def _get_torch_dtype() -> torch.dtype:
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def _dispatch_lm_model_cfg(self, cfg: ConfigDict, max_position_embeddings: Optional[int] = None) -> Any:
        cfg = self._prepare_for_qlora_zero3(cfg)
        llm_cfg = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _prepare_for_qlora_zero3(self, cfg: ConfigDict) -> ConfigDict:
        if is_deepspeed_zero3_enabled() and hasattr(cfg, 'quantization_config'):
            torch_dtype = self._get_torch_dtype()
            cfg.torch_dtype = torch_dtype
            cfg.quantization_config.bnb_4bit_compute_dtype = torch_dtype
            cfg.quantization_config.bnb_4bit_quant_storage = torch_dtype
        return cfg

    def _prepare_for_flash_attn(self, cfg: ConfigDict, llm_cfg) -> Tuple[ConfigDict, Any]:
        cls_name = type(llm_cfg).__name__
        if getattr(cfg, 'attn_implementation', None) == 'flash_attention_2':
            cfg.torch_dtype = self._get_torch_dtype()
        elif SUPPORT_FLASH2 and cls_name in self.SUPPORT_CONFIGS['FLASH2']:
            cfg.torch_dtype = self._get_torch_dtype()
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in self.SUPPORT_CONFIGS['SDPA']:
            cfg.attn_implementation = 'sdpa'
        return cfg, llm_cfg

    def _prepare_for_long_context_training(self, cfg: ConfigDict, llm_cfg, max_position_embeddings: int):
        text_cfg = getattr(llm_cfg, 'text_config', llm_cfg)
        orig_ctx_len = getattr(text_cfg, 'max_position_embeddings', None)
        if orig_ctx_len and max_position_embeddings > orig_ctx_len:
            scaling_factor = float(math.ceil(max_position_embeddings / orig_ctx_len))
            text_cfg.rope_scaling = {'type': 'linear', 'factor': scaling_factor}
        cfg.config = llm_cfg
        return cfg, llm_cfg

    def _build_from_cfg_or_module(self, cfg_or_mod) -> nn.Module:
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        if isinstance(cfg_or_mod, (dict, Config, ConfigDict)):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        return cfg_or_mod

    def _log_trainable_parameters(self):
        if not is_main_process():
            return
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_log(f"[Qwen3.5Adapter] trainable parameters: {total:,}", 'current')

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'use_llm_lora':
                return getattr(self, '_use_llm_lora', False)
            return getattr(self.llm, name)
