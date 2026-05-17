# Copyright (c) OpenMMLab. All rights reserved.
"""Prompt-conditioned pathology adapter for Qwen3.5 text models."""

import math
from contextlib import contextmanager
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
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
    RegressionHead,
    SurvivalHead,
    WSIProjector,
    cox_ph_loss,
    logistic_hazard_loss,
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


def _prepare_text_or_wsi_inputs(
    llm,
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    wsi_embeddings: Optional[torch.Tensor],
    padding_side: str,
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

    embeds_list, labels_list, attention_list, pos_list = [], [], [], []
    for b_idx in range(input_ids.size(0)):
        valid = attention_mask[b_idx]
        text = text_embeds[b_idx, valid]
        lbl = labels[b_idx, valid]
        wsi = wsi_embeddings[b_idx].to(device=text.device, dtype=text.dtype)
        cur_emb = torch.cat([wsi, text], dim=0)
        cur_lbl = torch.cat([
            torch.full((wsi.size(0),), IGNORE_INDEX, dtype=labels.dtype, device=labels.device),
            lbl,
        ])
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
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    position_generator: Optional[MRoPEPositionIDGenerator] = None,
    composer: Optional[InputComposer] = None,
    padding_side: str = 'right',
    **kwargs,
):
    has_patch = pixel_values is not None and pixel_values.numel() > 0
    has_wsi = wsi_embeddings is not None and wsi_embeddings.numel() > 0
    if not has_patch:
        return _prepare_text_or_wsi_inputs(llm, input_ids, labels, attention_mask, wsi_embeddings, padding_side)

    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask = attention_mask.bool()
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

    batch_size = input_ids.size(0)
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
    for b_idx in range(batch_size):
        cur_ids = input_ids[b_idx, attention_mask[b_idx]]
        cur_labels = labels[b_idx, attention_mask[b_idx]]
        image_positions = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        boundaries = [-1] + image_positions + [cur_ids.numel()]
        payload_idx = 0
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
                if has_wsi and not wsi_inserted:
                    cur_wsi = wsi_embeddings[b_idx].to(device=device, dtype=dtype)
                    pieces_embeds.append(cur_wsi)
                    pieces_labels.append(torch.full((cur_wsi.size(0),), IGNORE_INDEX,
                                                    dtype=labels.dtype, device=device))
                    pieces_attn.append(torch.ones(cur_wsi.size(0), dtype=torch.bool, device=device))
                    wsi_inserted = True

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
        if has_wsi and not wsi_inserted:
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

    composed = composer(embeds_list, labels_list, attention_list, positions_list, padding_side, IGNORE_INDEX)
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
        prompt_context_layer: int = -1,
        prompt_context_detach: bool = True,
        enable_nonfinite_checks: bool = False,
        wsi_feature_dims: Optional[List[int]] = None,
        wsi_dropout: float = 0.1,
        survival_head_dropout: float = 0.3,
        head_scaling: Union[float, List[float]] = (0.0, 0.0, 0.5),
    ):
        super().__init__()
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
        self.prompt_context_layer = int(prompt_context_layer)
        self.prompt_context_detach = bool(prompt_context_detach)
        self.enable_nonfinite_checks = bool(enable_nonfinite_checks)
        self.enable_vision = prompt_resampler_cfg is not None
        self.wsi_feature_dims = wsi_feature_dims
        self.enable_wsi_injection = wsi_feature_dims is not None and len(wsi_feature_dims) > 0
        self.wsi_dropout = float(wsi_dropout)
        self.survival_head_dropout = float(survival_head_dropout)
        self.use_llm_lora = llm_lora is not None
        self._use_llm_lora = self.use_llm_lora
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

        self._init_llm(llm, max_position_embeddings)
        self._setup_tokenizer_and_tokens(tokenizer)
        if self.enable_vision:
            self._init_prompt_resampler()
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
            resampler_dim=min(1024, self._get_llm_hidden_size()),
            num_region_tokens=128,
            num_visual_tokens=64,
            num_heads=8,
            dropout=0.0,
            use_local_conv=True,
        )
        cfg.update(self.prompt_resampler_cfg or {})
        cfg['llm_hidden_size'] = self._get_llm_hidden_size()
        self.patch_resampler = PromptConditionedPatchResampler(**cfg)
        self.patch_resampler.set_output_rms(self._estimate_embedding_rms())
        print_log(f"[PromptResampler] cfg={cfg}", 'current')

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
            uses_kbit = self._is_kbit_model(self.llm)
            if uses_kbit:
                self.llm = prepare_model_for_kbit_training(self.llm, use_gradient_checkpointing=use_activation_checkpointing)
            if getattr(lora_cfg, 'target_modules', None) is None:
                # For Qwen3-VL, we must avoid targeting 'proj' which matches Conv3d in visual encoder.
                # Searching only in language_model avoids finding 'proj' from vision blocks.
                target_model = getattr(self.llm, 'model', self.llm)
                target_model = getattr(target_model, 'language_model', target_model)
                lora_cfg.target_modules = find_all_linear_names(target_model)
            self.llm = self._get_peft_model_without_bnb_dispatch(
                self.llm,
                lora_cfg,
                disable_bnb_dispatch=not uses_kbit,
            )
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
        self._log_trainable_parameters()

    def _setup_generation(self, generation_kwargs: Optional[Dict], stop_words: Optional[List[str]]) -> None:
        gen_kwargs = dict(generation_kwargs or {})
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
        for cfg in (llm_config, getattr(llm_config, 'text_config', None)):
            if cfg is None:
                continue
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

        layer = self.prompt_context_layer
        prompt_model = self._get_prompt_context_model()
        if layer == -1:
            outputs = self._run_prompt_context_model(
                prompt_model,
                inputs_embeds=prompt_inputs['inputs_embeds'],
                attention_mask=prompt_inputs['attention_mask'],
                position_ids=prompt_inputs['position_ids'],
                return_dict=True,
            )
            context_embeds = getattr(outputs, 'last_hidden_state', None)
            if context_embeds is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                context_embeds = outputs[0]
            if context_embeds is None:
                raise TypeError(f"Prompt context model returned unsupported output type: {type(outputs)!r}")
        else:
            outputs = self._run_prompt_context_model(
                prompt_model,
                inputs_embeds=prompt_inputs['inputs_embeds'],
                attention_mask=prompt_inputs['attention_mask'],
                position_ids=prompt_inputs['position_ids'],
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if not (-len(hidden_states) <= layer < len(hidden_states)):
                raise IndexError(
                    f"prompt_context_layer={layer} is out of range for "
                    f"{len(hidden_states)} hidden-state tensors."
                )
            context_embeds = hidden_states[layer]
            selected_layer = layer if layer >= 0 else len(hidden_states) + layer
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

    def _project_vision_features(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
        out = self.patch_resampler(
            features=features,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            feature_shapes=feature_shapes,
        )
        self._last_patch_attention = out['patch_attention'].detach()
        self._last_patch_valid_mask = out['patch_valid_mask'].detach()
        return {
            'pixel_values': out['visual_tokens'],
            'vision_token_positions': out['token_positions'],
            'vision_token_valid': out['token_valid'],
            'patch_attention': out['patch_attention'],
        }

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

    def forward(self, data: Dict[str, Any], data_samples: Optional[List] = None, mode: str = 'loss') -> Any:
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
        if has_visual:
            projected = self._project_vision_features(data)
            data['pixel_values'] = projected['pixel_values']
            data['vision_token_positions'] = projected['vision_token_positions']
            data['vision_token_valid'] = projected['vision_token_valid']
        else:
            data['pixel_values'] = None
            data['vision_token_positions'] = None
            data['vision_token_valid'] = None
            data['image_batch_indices'] = None
        data.pop('features', None)
        data.pop('feature_shapes', None)
        data.pop('feature_paths', None)

        wsi_embeddings = None
        if self.enable_wsi_injection and data.get('wsi_features') is not None:
            wsi_embeddings = self._project_wsi_features(data.pop('wsi_features'))
        else:
            data.pop('wsi_features', None)
        data['wsi_embeddings'] = wsi_embeddings

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
            composer=self.input_composer,
            **data,
        )
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
                if first_step_logits is not None and i < first_step_logits.size(0):
                    logits_row = first_step_logits[i]
                    choice_logits = {}
                    for letter in "ABCDE":
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
                return data_samples

            # 2) Task predictions from generated special tokens
            with self._temporary_attn_implementation('sdpa'):
                return self._predict_tasks_from_generation(
                    generate_ids=generate_ids,
                    data_samples=data_samples,
                    has_regression=has_regression,
                    has_survival=has_survival,
                    prefix_inputs_embeds=prefix_inputs_embeds,
                    prefix_attention_mask=prefix_attention_mask,
                    prefix_position_ids=prefix_position_ids,
                )

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

    def state_dict(self, *args, **kwargs) -> OrderedDict:
        state_dict = super().state_dict(*args, **kwargs)
        keep = OrderedDict()
        if self.use_llm_lora:
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
        return keep

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
