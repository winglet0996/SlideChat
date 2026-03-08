# Copyright (c) OpenMMLab. All rights reserved.
"""
Unified LLaVA model for Qwen3 text-only models (Qwen3-4B/8B).

This model supports:
1. Text-only mode: Pure text input with no visual features
2. Multimodal mode: Text + patch-level visual features  
3. WSI global features: Additional slide-level features from multiple encoders

Key differences from llava_conv_qwen3_vl.py:
- No DeepStack visual feature injection
- No 3D-RoPE position encoding (uses standard 1D position IDs)
- Simpler forward path without Qwen3-VL specific handling
- Supports pure text training for regression/survival tasks
"""
import math
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.utils import is_list_of
from mmengine.dist import is_main_process
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, GenerationConfig, StoppingCriteriaList)
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names, get_peft_model_state_dict, 
                    guess_load_checkpoint, make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)
from .custom_model import (HighResConvNeXtV2Pyramid, PositionalEmbedding2DSinusoidal, 
                           AttentionPooling, RegressionHead, SurvivalHead, 
                           cox_ph_loss, logistic_hazard_loss, WSIProjector)


def _strip_image_tokens(input_ids: torch.Tensor, labels: torch.Tensor, 
                        attention_mask: torch.Tensor,
                        padding_side: str = 'right',
                        pad_token_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove <image> tokens from input_ids and corresponding positions in labels/attention_mask.
    Properly handles padding based on padding_side.
    
    Args:
        input_ids: (B, L) input token IDs
        labels: (B, L) labels
        attention_mask: (B, L) attention mask
        padding_side: 'left' or 'right' - where to place padding
        pad_token_id: Token ID to use for padding
    
    Returns:
        Tuple of (new_input_ids, new_labels, new_attention_mask) with <image> tokens removed
    """
    B, L = input_ids.shape
    device = input_ids.device
    
    # Ensure attention_mask is bool for logical operations
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attn_bool = attention_mask.bool()
    
    # Find valid (non-image, non-padding) positions for each sample
    # A position is valid if: it's attended AND not an image token
    is_image = (input_ids == IMAGE_TOKEN_INDEX)
    valid_mask = attn_bool & ~is_image  # (B, L) - positions to keep
    
    # Early exit: no attended image tokens to strip → return None (caller skips unpacking)
    if not (is_image & attn_bool).any():
        return None
    
    # Get new lengths after removing image tokens
    new_lengths = valid_mask.sum(dim=1)  # (B,)
    max_new_len = new_lengths.max().item()
    
    # Create new tensors with proper padding values
    new_input_ids = torch.full((B, max_new_len), pad_token_id, dtype=input_ids.dtype, device=device)
    new_labels = torch.full((B, max_new_len), IGNORE_INDEX, dtype=labels.dtype, device=device) if labels is not None else None
    new_attention_mask = torch.zeros((B, max_new_len), dtype=attention_mask.dtype, device=device)
    
    for b in range(B):
        # Get indices of valid positions (non-image, attended)
        valid_indices = valid_mask[b].nonzero(as_tuple=True)[0]
        length = valid_indices.shape[0]
        
        if length == 0:
            continue
        
        if padding_side == 'right':
            # Content at the beginning, padding at the end
            new_input_ids[b, :length] = input_ids[b, valid_indices]
            if new_labels is not None:
                new_labels[b, :length] = labels[b, valid_indices]
            new_attention_mask[b, :length] = 1
        else:
            # Left padding: padding at the beginning, content at the end
            start_idx = max_new_len - length
            new_input_ids[b, start_idx:] = input_ids[b, valid_indices]
            if new_labels is not None:
                new_labels[b, start_idx:] = labels[b, valid_indices]
            new_attention_mask[b, start_idx:] = 1
    
    return new_input_ids, new_labels, new_attention_mask


def _prepare_text_with_wsi(llm, input_ids: torch.Tensor, labels: torch.Tensor,
                           attention_mask: torch.Tensor, position_ids: torch.Tensor,
                           past_key_values: Any, wsi_embeddings: torch.Tensor,
                           padding_side: str) -> Dict[str, torch.Tensor]:
    """
    Prepare text + WSI embeddings (no patch features).
    WSI embeddings are prepended to the text sequence.
    Properly handles padding based on padding_side.
    """
    B, L = input_ids.shape
    num_wsi_tokens = wsi_embeddings.size(1)
    device = input_ids.device
    dtype = wsi_embeddings.dtype
    hidden_dim = wsi_embeddings.size(-1)
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    
    # Calculate actual text lengths for each sample
    text_lengths = attention_mask.sum(dim=1)  # (B,)
    new_total_len = L + num_wsi_tokens  # Total length after adding WSI tokens
    
    # Get text embeddings
    text_embeds = llm.get_input_embeddings()(input_ids)  # (B, L, H)
    
    # Create output tensors
    inputs_embeds = torch.zeros((B, new_total_len, hidden_dim), dtype=dtype, device=device)
    new_labels = torch.full((B, new_total_len), IGNORE_INDEX, dtype=labels.dtype, device=device)
    new_attention_mask = torch.zeros((B, new_total_len), dtype=torch.bool, device=device)
    new_position_ids = torch.zeros((B, new_total_len), dtype=torch.long, device=device)
    
    for b in range(B):
        text_len = text_lengths[b].item()
        total_content_len = num_wsi_tokens + text_len
        
        if padding_side == 'right':
            # [WSI, Text, Padding]
            # WSI tokens
            inputs_embeds[b, :num_wsi_tokens] = wsi_embeddings[b]
            new_attention_mask[b, :num_wsi_tokens] = True
            # Text tokens (first text_len valid tokens from input)
            inputs_embeds[b, num_wsi_tokens:num_wsi_tokens + text_len] = text_embeds[b, :text_len]
            new_labels[b, num_wsi_tokens:num_wsi_tokens + text_len] = labels[b, :text_len]
            new_attention_mask[b, num_wsi_tokens:num_wsi_tokens + text_len] = True
            # Position IDs
            new_position_ids[b, :total_content_len] = torch.arange(total_content_len, device=device)
        else:
            # Left padding: [Padding, WSI, Text]
            start_idx = new_total_len - total_content_len
            # WSI tokens
            inputs_embeds[b, start_idx:start_idx + num_wsi_tokens] = wsi_embeddings[b]
            new_attention_mask[b, start_idx:start_idx + num_wsi_tokens] = True
            # Text tokens (last text_len valid tokens from input)
            inputs_embeds[b, start_idx + num_wsi_tokens:] = text_embeds[b, -text_len:]
            new_labels[b, start_idx + num_wsi_tokens:] = labels[b, -text_len:]
            new_attention_mask[b, start_idx + num_wsi_tokens:] = True
            # Position IDs (start from 0 for actual content)
            new_position_ids[b, start_idx:] = torch.arange(total_content_len, device=device)
    
    return {
        'input_ids': None,
        'inputs_embeds': inputs_embeds,
        'labels': new_labels,
        'attention_mask': new_attention_mask,
        'position_ids': new_position_ids,
        'past_key_values': past_key_values,
    }


def prepare_inputs_labels_for_text_and_wsi(
    llm,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    past_key_values: Any = None,
    pixel_values: torch.Tensor = None,
    wsi_embeddings: torch.Tensor = None,
    padding_side: str = 'right',
    **kwargs
):
    """
    Prepare inputs and labels for text-only or text+visual training.
    
    Supports all combinations of modalities:
    1. Pure text mode: No visual features (automatically strips <image> tokens if present)
    2. Text + WSI: WSI global features prepended to text (strips <image> tokens)
    3. Text + Patch: <image> tokens replaced with patch features
    4. Full multimodal: Text + Patch + WSI features
    
    Key feature: If pixel_values is None but input_ids contains <image> tokens,
    those tokens are automatically stripped, allowing use of full-modal data
    with any subset of modalities via config only.
    
    Args:
        llm: Language model
        input_ids: (B, L) input token IDs
        labels: (B, L) labels for loss computation
        attention_mask: (B, L) attention mask
        position_ids: (B, L) position IDs
        pixel_values: (N, S, H) projected visual features, or None to skip images
        wsi_embeddings: (B, Num_WSI_Sources, H) WSI global features, or None to skip
        padding_side: 'left' or 'right' padding
    
    Returns:
        Dict with processed inputs for LLM forward
    """
    has_pixel_values = pixel_values is not None
    has_wsi = wsi_embeddings is not None
    
    # Check if any sample has <image> tokens
    has_image_tokens = (input_ids == IMAGE_TOKEN_INDEX).any()
    
    # Get pad_token_id for proper padding
    pad_token_id = getattr(llm.config, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = getattr(llm.config, 'eos_token_id', 0)
    
    # Case 1: Pure text mode - no visual features and no <image> tokens (or we strip them)
    if not has_pixel_values and not has_wsi:
        if has_image_tokens:
            # Strip <image> tokens from input_ids and labels (returns None if nothing to strip)
            strip_result = _strip_image_tokens(
                input_ids, labels, attention_mask, padding_side, pad_token_id)
            if strip_result is not None:
                input_ids, labels, attention_mask = strip_result
        inputs_embeds = llm.get_input_embeddings()(input_ids)
        # Generate position_ids that respect padding (attention_mask based)
        B, L = input_ids.shape
        device = input_ids.device
        if padding_side == 'right':
            new_position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        else:
            # Left padding: position_ids should start from 0 for actual content
            new_position_ids = torch.zeros((B, L), dtype=torch.long, device=device)
            for b in range(B):
                seq_len = attention_mask[b].sum().item()
                new_position_ids[b, -seq_len:] = torch.arange(seq_len, device=device)
        return {
            'input_ids': None,
            'inputs_embeds': inputs_embeds,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': new_position_ids,
            'past_key_values': past_key_values,
        }
    
    # Case 2: Text + WSI only (no patch features) - strip <image> tokens if present
    if not has_pixel_values and has_wsi:
        if has_image_tokens:
            strip_result = _strip_image_tokens(
                input_ids, labels, attention_mask, padding_side, pad_token_id)
            if strip_result is not None:
                input_ids, labels, attention_mask = strip_result
        return _prepare_text_with_wsi(llm, input_ids, labels, attention_mask, 
                                      position_ids, past_key_values, wsi_embeddings, padding_side)
    
    # Case 3 & 4: Has patch features (with or without WSI)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
    
    num_wsi_tokens = wsi_embeddings.size(1) if has_wsi else 0
    
    # Remove padding
    input_ids_list = [cur_input_ids[cur_attn] 
                      for cur_input_ids, cur_attn in zip(input_ids, attention_mask)]
    labels_list = [cur_labels[cur_attn] 
                   for cur_labels, cur_attn in zip(labels, attention_mask)]
    
    new_inputs_embeds = []
    new_labels = []
    new_position_ids = []
    
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids_list):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        
        if num_images == 0:
            # No image tokens - simple text with optional WSI prepending
            cur_inputs_embeds = llm.get_input_embeddings()(cur_input_ids)
            cur_labels = labels_list[batch_idx]
            
            if has_wsi and num_wsi_tokens > 0:
                cur_wsi_embeds = wsi_embeddings[batch_idx]
                wsi_labels = torch.full((num_wsi_tokens,), IGNORE_INDEX, 
                                        device=cur_labels.device, dtype=cur_labels.dtype)
                cur_inputs_embeds = torch.cat([cur_wsi_embeds, cur_inputs_embeds], dim=0)
                cur_labels = torch.cat([wsi_labels, cur_labels], dim=0)
            
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(cur_labels)
            new_position_ids.append(torch.arange(cur_inputs_embeds.shape[0], 
                                                  device=cur_inputs_embeds.device))
            continue
        
        # Has image tokens - replace with patch features
        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        
        cur_new_inputs_embeds = []
        cur_new_labels = []
        
        for i in range(len(image_token_indices) - 1):
            # Text segment
            text_ids = cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i+1]]
            if text_ids.shape[0] > 0:
                text_embeds = llm.get_input_embeddings()(text_ids)
                text_labels = labels_list[batch_idx][image_token_indices[i] + 1 : image_token_indices[i+1]]
                cur_new_inputs_embeds.append(text_embeds)
                cur_new_labels.append(text_labels)
            
            # Image segment
            if i < len(image_token_indices) - 2:
                # Insert WSI before first image only
                if i == 0 and has_wsi and num_wsi_tokens > 0:
                    cur_wsi_embeds = wsi_embeddings[batch_idx]
                    wsi_labels = torch.full((num_wsi_tokens,), IGNORE_INDEX, 
                                            device=labels.device, dtype=labels.dtype)
                    cur_new_inputs_embeds.append(cur_wsi_embeds)
                    cur_new_labels.append(wsi_labels)
                
                # Insert patch features
                cur_pixel_values = pixel_values[cur_image_idx]
                cur_image_idx += 1
                cur_new_inputs_embeds.append(cur_pixel_values)
                cur_new_labels.append(torch.full((cur_pixel_values.shape[0],), IGNORE_INDEX,
                                                 device=labels.device, dtype=labels.dtype))
        
        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds, dim=0)
        cur_new_labels = torch.cat(cur_new_labels, dim=0)
        
        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)
        new_position_ids.append(torch.arange(cur_new_inputs_embeds.shape[0],
                                              device=cur_new_inputs_embeds.device))
    
    # Pad sequences
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)
    hidden_dim = new_inputs_embeds[0].shape[-1]
    device = new_inputs_embeds[0].device
    dtype = new_inputs_embeds[0].dtype
    
    final_inputs_embeds = torch.zeros((batch_size, max_len, hidden_dim), dtype=dtype, device=device)
    final_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=device)
    final_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    final_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    
    for i, (emb, lbl, pids) in enumerate(zip(new_inputs_embeds, new_labels, new_position_ids)):
        cur_len = emb.shape[0]
        if padding_side == 'right':
            final_inputs_embeds[i, :cur_len] = emb
            final_labels[i, :cur_len] = lbl
            final_attention_mask[i, :cur_len] = True
            final_position_ids[i, :cur_len] = pids
        else:  # left padding
            final_inputs_embeds[i, -cur_len:] = emb
            final_labels[i, -cur_len:] = lbl
            final_attention_mask[i, -cur_len:] = True
            final_position_ids[i, -cur_len:] = pids
    
    return {
        'input_ids': None,
        'inputs_embeds': final_inputs_embeds,
        'labels': final_labels,
        'attention_mask': final_attention_mask,
        'position_ids': final_position_ids,
        'past_key_values': past_key_values,
    }


def convert_state_dict_to_hf(state_dict: Dict[str, torch.Tensor], 
                           mapping: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Convert state dict using key mapping, excluding frequency parameters."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.inv_freq'):
            continue
        new_key = key
        for old_key, replacement in mapping.items():
            if old_key in new_key:
                new_key = new_key.replace(old_key, replacement)
        new_state_dict[new_key] = value
    return new_state_dict


class LLaVAModel_conv_unified(BaseModel):
    """
    Unified LLaVA model for Qwen3 text-only models (Qwen3-4B/8B).
    
    Supports:
    1. Text-only mode: Pure text with regression/survival prediction
    2. Multimodal mode: Text + patch-level visual features
    3. WSI injection: Slide-level features from multiple encoders
    
    Key features:
    - Clean token-only regression approach  
    - Visual-aware survival prediction (optional)
    - End-to-end differentiable training
    - Multi-GPU training compatibility with gradient synchronization deadlock prevention
    """

    # Supported model configurations for flash attention
    SUPPORT_CONFIGS = {
        'SDPA': ('LlamaConfig', 'GemmaConfig', 'MistralConfig', 'MixtralConfig', 
                 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config', 'Qwen3Config'),
        'FLASH2': ('InternLM2Config', 'LlamaConfig', 'GemmaConfig', 'MistralConfig', 
                   'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config', 'Qwen3Config')
    }

    def __init__(self, llm, tokenizer, freeze_llm: bool = True, visual_select_layer: int = -2,
                 pretrained_pth: Optional[str] = None, projector_depth: int = 2,
                 llm_lora: Optional[Dict] = None, use_activation_checkpointing: bool = True,
                 max_position_embeddings: Optional[int] = None, hidden_size: Optional[int] = None,
                 generation_kwargs: Optional[Dict] = None, stop_words: Optional[List[str]] = None,
                 enable_regression: bool = True, reg_token: str = '<REG>',
                 enable_survival: bool = True, srv_token: str = '<SRV>',
                 num_survival_intervals: int = 6,
                 survival_method: str = 'cox',  # 'cox' or 'discrete'
                 gen_forcing: bool = True,  # If False, skip token generation check for task prediction
                 lambda_llm: float = 0.1, lambda_reg: float = 1.0, lambda_srv: float = 1.0,
                 vision_conv_cfg: Optional[Dict] = None,
                 wsi_feature_dims: Optional[List[int]] = None,
                 head_scaling: Union[float, List[float]] = [0.0, 0.0, 0.5]):
        """
        Unified LLaVA model with regression and survival prediction capabilities.
        
        Args:
            survival_method: 'cox' for Cox proportional hazards or 'discrete' for discrete-time survival
            num_survival_intervals: Number of intervals (K) for discrete method
            gen_forcing: If True (default), the model must correctly generate the special token
                         (<REG>/<SRV>) during inference to trigger task prediction. If False,
                         task predictions are always computed using the learned special token
                         embedding directly, regardless of whether the model generated it.
            wsi_feature_dims: List of input dimensions for each WSI encoder source.
                              E.g., [768, 1024, 768] for three different encoders (TITAN, CONCH, UNI).
                              If None, WSI feature injection is disabled.
            head_scaling: Multiplier(s) for hidden layer dimension in task heads and WSI projector.
                          - If float: applied to all (reg, srv, wsi).
                          - If list of 3: [reg_mult, srv_mult, wsi_mult].
                          Set to 0 for a single linear layer (most lightweight).
            vision_conv_cfg: Configuration for the ConvNeXtV2 pyramid vision backbone.
                             If None, vision components are disabled (text-only mode).
        """
        super().__init__()

        # Initialize core attributes
        self._init_attributes(
            freeze_llm=freeze_llm,
            enable_regression=enable_regression,
            reg_token=reg_token,
            enable_survival=enable_survival,
            srv_token=srv_token,
            num_survival_intervals=num_survival_intervals,
            survival_method=survival_method,
            gen_forcing=gen_forcing,
            lambda_llm=lambda_llm,
            lambda_reg=lambda_reg,
            lambda_srv=lambda_srv,
            vision_conv_cfg=vision_conv_cfg,
            wsi_feature_dims=wsi_feature_dims,
            head_scaling=head_scaling
        )

        # Initialize model components
        self._init_llm(llm, max_position_embeddings)
        
        # Initialize vision components only if config provided
        if self.enable_vision:
            self._init_vision_components()
            self._init_projector(projector_depth)
        
        # Initialize WSI projector if wsi_feature_dims is provided
        if self.enable_wsi_injection:
            self._init_wsi_projector()
        
        self._setup_tokenizer_and_tokens(tokenizer, enable_regression, enable_survival)

        # Initialize prediction modules if needed
        if enable_regression or enable_survival:
            self._init_prediction_modules()

        # Configure training
        self._configure_training(llm_lora, use_activation_checkpointing, freeze_llm)

        # Load pretrained weights
        if pretrained_pth:
            self._load_pretrained_weights(pretrained_pth)

        # Setup generation
        self._setup_generation(generation_kwargs, stop_words)

        # Initialize state
        self._init_state(visual_select_layer)

    # ========== Initialization helpers ==========

    def _init_attributes(self, freeze_llm: bool, enable_regression: bool, reg_token: str,
                         enable_survival: bool, srv_token: str, num_survival_intervals: int,
                         survival_method: str, gen_forcing: bool,
                         lambda_llm: float, lambda_reg: float, lambda_srv: float, 
                         vision_conv_cfg: Optional[Dict],
                         wsi_feature_dims: Optional[List[int]],
                         head_scaling: Union[float, List[float]]) -> None:
        """Initialize core model attributes."""
        self.freeze_llm = freeze_llm
        self.enable_regression = enable_regression
        self.reg_token = reg_token
        self.enable_survival = enable_survival
        self.srv_token = srv_token
        self.num_survival_intervals = num_survival_intervals
        self.survival_method = survival_method
        self.gen_forcing = gen_forcing  # If False, skip token generation check
        self.lambda_llm = lambda_llm
        self.lambda_reg = lambda_reg
        self.lambda_srv = lambda_srv
        
        # LoRA flags
        self.use_llm_lora = False
        self._use_llm_lora = False
        
        # Token ids will be filled later
        self.reg_token_id = None
        self.srv_token_id = None
        
        # Vision config - if None, model runs in text-only mode
        self.vision_conv_cfg = vision_conv_cfg
        self.enable_vision = vision_conv_cfg is not None
        
        # WSI feature injection configuration
        self.wsi_feature_dims = wsi_feature_dims
        self.enable_wsi_injection = wsi_feature_dims is not None and len(wsi_feature_dims) > 0
        
        # Handle scaling factors: [reg_mult, srv_mult, wsi_mult]
        if isinstance(head_scaling, (int, float)):
            self.head_scaling = [float(head_scaling)] * 3
        else:
            assert len(head_scaling) == 3, f"head_scaling must be float or list of length 3, got {head_scaling}"
            self.head_scaling = [float(x) for x in head_scaling]

    def _init_llm(self, llm, max_position_embeddings: Optional[int]) -> None:
        """Initialize the language model."""
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
            self.llm = self._build_from_cfg_or_module(llm)
        
        # Handle both standard and composite configs for use_cache
        if hasattr(self.llm.config, 'use_cache'):
            self.llm.config.use_cache = False
        if hasattr(self.llm.config, 'text_config'):
            self.llm.config.text_config.use_cache = False

        dispatch_modules(self.llm)

    def _get_llm_hidden_size(self) -> int:
        """Get the hidden size of the LLM, handling both standard and composite configs."""
        if hasattr(self.llm.config, 'hidden_size'):
            return self.llm.config.hidden_size
        if hasattr(self.llm.config, 'text_config'):
            return getattr(self.llm.config.text_config, 'hidden_size')
        raise AttributeError("Could not determine hidden_size from LLM config")

    def _get_language_model_norm(self):
        """Get the final RMSNorm layer from the language model.
        
        This is needed because hidden_states[-1] from output_hidden_states=True
        are PRE-normalization. The model internally applies RMSNorm before lm_head.
        We need to apply the same normalization for regression/survival heads.
        """
        paths_to_try = [
            # Standard Qwen2/3 or Llama-like
            lambda: self.llm.model.norm,
            # Qwen3-VL style
            lambda: self.llm.model.language_model.norm,
            # PEFT-wrapped standard
            lambda: self.llm.base_model.model.model.norm,
            # PEFT-wrapped Qwen3-VL
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

    def _init_vision_components(self) -> None:
        """Initialize vision processing components."""
        # depths=[2,4]: balanced; deeper pyramid-stage is more powerful than [1,3]
        # without the gradient issues of [3,9,3].  drop_path_rate=0.15 keeps
        # stochastic depth regularisation mild so gradients flow cleanly.
        default_conv_cfg = dict(
            in_chans=768,
            depths=[2, 4],
            dims=[1024, 2048],
            drop_path_rate=0.15,
            num_downsamples=1,
        )
        if self.vision_conv_cfg is not None:
            default_conv_cfg.update(self.vision_conv_cfg)
        self.conv = HighResConvNeXtV2Pyramid(**default_conv_cfg).to(self.llm.dtype)
        print_log(f"[VisionBackbone] Using {self.conv.__class__.__name__} with cfg={default_conv_cfg}", 'current')
        self.pos_emb_2d = PositionalEmbedding2DSinusoidal(
            d_model=self.conv.dims[-1],
            scale_mode='learned',
            init_pe_scale=0.1
        ).to(self.llm.dtype)

    def _init_projector(self, depth: int = 2) -> None:
        """Initialize the vision-language projector."""
        import math
        llm_hidden_size = self._get_llm_hidden_size()
        projector_config = ProjectorConfig(
            visual_hidden_size=self.conv.dims[-1],
            llm_hidden_size=llm_hidden_size,
            depth=depth
        )
        self.projector = ProjectorModel(projector_config).to(self.llm.dtype)

        # Post-projection LayerNorm to stabilize visual feature distribution.
        #
        # Initialization note: we use standard LayerNorm init (weight=1.0, bias=0),
        # NOT the 1/sqrt(D) "scale-to-unit-norm" trick used in the VL model.
        #
        # Reason: in the unified model visual tokens enter the LLM ONLY via the input
        # sequence (no DeepStack).  If weight << 1 at init, the backward gradient to
        # projector/conv is scaled by weight, creating a large attenuation factor
        # (e.g. 1/sqrt(4096) ≈ 0.016 → 64× gradient throttle).  In bf16 this causes
        # gradient underflow for the visual pathway, especially for tasks with small
        # gradient magnitude such as SmoothL1 regression.  Survival (hazard-log-loss)
        # and MCQA survive because their gradients are larger / text-driven.
        #
        # With weight=1.0 the initial output norm is ~sqrt(D) (~64 for D=4096), which
        # is larger than text embeddings, but that is intentional: visual tokens
        # dominate <REG> hidden states early in training, forcing the regression head
        # to actually use visual information.  The trainable weight converges to the
        # right scale within a few hundred steps.
        norm = nn.LayerNorm(llm_hidden_size, elementwise_affine=True)
        # Standard init: weight=1, bias=0 – do NOT override with small constant.
        nn.init.ones_(norm.weight)
        nn.init.zeros_(norm.bias)
        self.proj_norm = norm.to(self.llm.dtype)
        print_log(
            f"[ProjectorNorm] Added LayerNorm after projector (weight=1.0, standard init). "
            f"Initial output norm ≈ sqrt({llm_hidden_size}) ≈ {math.sqrt(llm_hidden_size):.1f} "
            f"(intentionally large for full gradient flow).",
            'current'
        )

    def _init_wsi_projector(self) -> None:
        """Initialize the WSI feature projector for multi-source WSI injection."""
        llm_hidden_size = self._get_llm_hidden_size()
        wsi_mult = self.head_scaling[2]
        self.wsi_projector = WSIProjector(
            wsi_input_dims=self.wsi_feature_dims,
            llm_hidden_size=llm_hidden_size,
            hidden_mult=wsi_mult,
            dropout=0.1
        ).to(self.llm.dtype)
        
        print_log(f"[WSIProjector] Initialized with dims={self.wsi_feature_dims} -> {llm_hidden_size}, "
                  f"mult={wsi_mult}", 'current')

    def _setup_tokenizer_and_tokens(self, tokenizer, enable_regression: bool, enable_survival: bool) -> None:
        """Setup tokenizer and add special tokens efficiently."""
        self.tokenizer = BUILDER.build(tokenizer)
        
        # Store original vocab size before adding special tokens
        self._original_vocab_size = len(self.tokenizer)

        # Add special tokens in batch
        special_tokens = []
        if enable_regression:
            special_tokens.append(self.reg_token)
        if enable_survival:
            special_tokens.append(self.srv_token)
        if special_tokens:
            self._add_special_tokens(special_tokens)

        # Get token IDs after adding
        if enable_regression:
            self.reg_token_id = self.tokenizer.convert_tokens_to_ids(self.reg_token)
            self._init_token_embedding(self.reg_token_id, 'regression')
        if enable_survival:
            self.srv_token_id = self.tokenizer.convert_tokens_to_ids(self.srv_token)
            self._init_token_embedding(self.srv_token_id, 'survival')

        # Ensure pad_token_id is set for the model
        if getattr(self.llm.config, 'pad_token_id', None) is None:
            if self.tokenizer.pad_token_id is not None:
                self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            elif self.tokenizer.eos_token_id is not None:
                self.llm.config.pad_token_id = self.tokenizer.eos_token_id

        # Enable training for new special tokens
        if enable_regression or enable_survival:
            self._enable_selective_training()

    def _enable_selective_training(self):
        """Enable training for special tokens and task-specific components."""
        new_token_ids = [tid for tid in [self.reg_token_id, self.srv_token_id] if tid is not None]
        if not new_token_ids:
            return

        # Register gradient hook to only train new token embeddings
        self._register_embedding_grad_hook()
        print_log("[SelectiveTraining] Configured gradient flow for special token learning", 'current')

    def _register_embedding_grad_hook(self) -> None:
        """Register gradient hook to zero out gradients for original vocabulary."""
        old_vocab_size = self._original_vocab_size
        
        def _get_zero_hook(name):
            def _zero_old_token_grad(grad: torch.Tensor) -> torch.Tensor:
                if grad is not None:
                    grad[:old_vocab_size] = 0
                return grad
            return _zero_old_token_grad

        # 1. Input embeddings
        embedding = self.llm.get_input_embeddings()
        if embedding is not None and hasattr(embedding, 'weight'):
            if hasattr(self, '_embedding_grad_hook_handle'):
                self._embedding_grad_hook_handle.remove()
            self._embedding_grad_hook_handle = embedding.weight.register_hook(_get_zero_hook('input'))
        
        # 2. Output embeddings (only if not tied)
        is_tied = getattr(self.llm.config, 'tie_word_embeddings', True)
        if not is_tied:
            output_layer = self.llm.get_output_embeddings()
            if output_layer is not None and hasattr(output_layer, 'weight'):
                if hasattr(self, '_output_grad_hook_handle'):
                    self._output_grad_hook_handle.remove()
                self._output_grad_hook_handle = output_layer.weight.register_hook(_get_zero_hook('output'))
        
        new_vocab_size = len(self.tokenizer)
        num_new_tokens = new_vocab_size - old_vocab_size
        status = "tied" if is_tied else "untied"
        print_log(f"[EmbeddingGradHook] Registered ({status}): old_vocab={old_vocab_size}, "
                  f"new_vocab={new_vocab_size}, trainable_tokens={num_new_tokens}", 'current')

    def _add_special_tokens(self, tokens: List[str]) -> None:
        """Add special tokens and resize embeddings properly."""
        added = [AddedToken(t, normalized=False, special=True) for t in tokens]
        num_added = 0
        try:
            num_added = self.tokenizer.add_special_tokens({
                'additional_special_tokens': added
            })
        except Exception:
            try:
                num_added = self.tokenizer.add_tokens(added, special_tokens=True)
            except Exception:
                num_added = self.tokenizer.add_tokens(tokens)

        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            if hasattr(self.llm, 'tie_weights'):
                try:
                    self.llm.tie_weights()
                except Exception:
                    pass

    def _init_token_embedding(self, token_id: int, task_type: str) -> None:
        """Initialize special token embedding with semantic meaning."""
        is_tied = getattr(self.llm.config, 'tie_word_embeddings', True)
        emb = self.llm.get_input_embeddings()
        out = self.llm.get_output_embeddings() if not is_tied else None
        
        if emb is None or not hasattr(emb, 'weight') or token_id >= emb.weight.size(0):
            return

        with torch.no_grad():
            if task_type == 'regression':
                init_tokens = ["value", "number", "result", "score", "level"]
            else:
                init_tokens = ["survival", "time", "risk", "hazard", "outcome"]

            valid_ids = []
            for tk in init_tokens:
                tid = self.tokenizer.convert_tokens_to_ids(tk)
                if (tid is not None and tid != self.tokenizer.unk_token_id and 0 <= tid < emb.weight.size(0)):
                    valid_ids.append(tid)

            # Initialize input embedding
            if valid_ids:
                base_vec = emb.weight[valid_ids].mean(dim=0)
            else:
                base_vec = emb.weight.mean(dim=0)
            
            noise = 1e-3 * torch.randn_like(base_vec)
            emb.weight[token_id] = 1.05 * base_vec + noise

            # Initialize output head weight (only for untied models like 8B)
            if out is not None and hasattr(out, 'weight') and token_id < out.weight.size(0):
                if valid_ids:
                    base_out = out.weight[valid_ids].mean(dim=0)
                else:
                    base_out = out.weight.mean(dim=0)
                out.weight[token_id] = 1.05 * base_out + 1e-3 * torch.randn_like(base_out)
                
            print_log(f"[TokenInit] Initialized {task_type} token ({token_id}) for "
                      f"{'tied' if is_tied else 'untied'} embeddings", 'current')

    def _init_prediction_modules(self) -> None:
        """Initialize prediction-specific modules with scaling control."""
        llm_hidden = self._get_llm_hidden_size()
        reg_mult, srv_mult, _ = self.head_scaling

        # Regression head using only special token embeddings
        if self.enable_regression:
            self.regression_head = RegressionHead(
                in_dim=llm_hidden, 
                hidden_mult=reg_mult
            ).to(self.llm.dtype)
            self.regression_loss_fn = nn.SmoothL1Loss(beta=1.0)

        # Survival head - supports both Cox and Discrete methods
        if self.enable_survival:
            self.survival_head = SurvivalHead(
                in_dim=llm_hidden,
                method=self.survival_method,
                num_intervals=self.num_survival_intervals,
                time_intervals=None,
                hidden_mult=srv_mult
            ).to(dtype=self.llm.dtype)
            
            if self.survival_method == 'cox':
                self.survival_loss_fn = cox_ph_loss
            else:
                self.survival_loss_fn = logistic_hazard_loss
            
            print_log(f"[SurvivalHead] Initialized with method='{self.survival_method}', "
                      f"num_intervals={self.num_survival_intervals}, mult={srv_mult}", 'current')

    def _configure_training(self, llm_lora: Optional[Dict], use_activation_checkpointing: bool, freeze_llm: bool) -> None:
        """Configure training settings including LoRA and checkpointing."""
        self.use_llm_lora = llm_lora is not None
        self._use_llm_lora = self.use_llm_lora

        if self.use_llm_lora:
            self._setup_lora(llm_lora, use_activation_checkpointing)

        if freeze_llm:
            self._freeze_llm_with_exceptions()
        
        self._configure_parameter_gradients()

        if use_activation_checkpointing:
            self._setup_checkpointing()

    def _setup_lora(self, lora_config: Dict, use_activation_checkpointing: bool) -> None:
        """Setup LoRA configuration."""
        lora_config = self._build_from_cfg_or_module(lora_config)
        self.llm = prepare_model_for_kbit_training(self.llm, use_activation_checkpointing)

        if lora_config.target_modules is None:
            lora_config.target_modules = find_all_linear_names(self.llm)

        self.llm = get_peft_model(self.llm, lora_config)

    def _freeze_llm_with_exceptions(self) -> None:
        """Freeze LLM parameters while keeping task-critical components trainable."""
        if self.use_llm_lora:
            return
            
        self.llm.requires_grad_(False)
        print_log("Froze base LLM parameters", 'current')

    def _configure_parameter_gradients(self) -> None:
        """Centralized parameter gradient configuration for consistent multi-GPU behavior."""
        trainable_params = []
        
        # Handle Embeddings and LM Head
        is_tied = getattr(self.llm.config, 'tie_word_embeddings', True)
        embed_layer = self.llm.get_input_embeddings()
        output_layer = self.llm.get_output_embeddings()
        
        if embed_layer is not None and hasattr(embed_layer, 'weight'):
            embed_layer.weight.requires_grad = True
        
        if not is_tied and output_layer is not None and hasattr(output_layer, 'weight'):
            output_layer.weight.requires_grad = True

        # Log trainable parameters
        if hasattr(self, '_original_vocab_size'):
            num_new = len(self.tokenizer) - self._original_vocab_size
            hidden_dim = self._get_llm_hidden_size()
            if is_tied:
                trainable_params.append(f"embed+lm_head (tied, new tokens): {num_new * hidden_dim:,}")
            else:
                trainable_params.append(f"embed+lm_head (untied, new tokens): {num_new * hidden_dim * 2:,}")

        # LoRA parameters
        if self.use_llm_lora:
            lora_param_count = 0
            for name, param in self.llm.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    lora_param_count += param.numel()
            if lora_param_count > 0:
                trainable_params.append(f"LoRA: {lora_param_count:,}")

        # Task-specific modules
        if self.enable_regression and hasattr(self, 'regression_head'):
            reg_params = sum(p.numel() for p in self.regression_head.parameters())
            trainable_params.append(f"regression_head: {reg_params:,}")
            
        if self.enable_survival and hasattr(self, 'survival_head'):
            srv_params = sum(p.numel() for p in self.survival_head.parameters()) 
            trainable_params.append(f"survival_head: {srv_params:,}")

        # Vision components
        if self.enable_vision:
            conv_params = sum(p.numel() for p in self.conv.parameters())
            proj_params = sum(p.numel() for p in self.projector.parameters())
            pos_params = sum(p.numel() for p in self.pos_emb_2d.parameters())
            norm_params = sum(p.numel() for p in self.proj_norm.parameters())
            trainable_params.extend([
                f"conv: {conv_params:,}",
                f"projector: {proj_params:,}",
                f"proj_norm: {norm_params:,}",
                f"pos_emb_2d: {pos_params:,}"
            ])
        
        # WSI projector
        if self.enable_wsi_injection and hasattr(self, 'wsi_projector'):
            wsi_proj_params = sum(p.numel() for p in self.wsi_projector.parameters())
            trainable_params.append(f"wsi_projector: {wsi_proj_params:,}")

        if is_main_process():
            print_log(f"[ParameterConfig] Trainable components: {', '.join(trainable_params)}", 'current')
            
        self._validate_parameter_consistency()

    def _validate_parameter_consistency(self) -> None:
        """Validate that trainable parameters are consistent across all GPUs."""
        trainable_param_names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
        
        param_hash = hash(tuple(sorted(trainable_param_names)))
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if is_main_process():
            print_log(f"[MultiGPUValidation] Parameter hash: {param_hash}, "
                      f"Total trainable params: {total_trainable:,}", 'current')

    def _setup_checkpointing(self) -> None:
        """Setup gradient checkpointing."""
        if hasattr(self.llm, 'enable_input_require_grads'):
            self.llm.enable_input_require_grads()
        else:
            self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if self.enable_vision:
            self.projector.enable_input_require_grads()
        self.gradient_checkpointing_enable()

    def _load_pretrained_weights(self, pretrained_pth: str) -> None:
        """Load pretrained model weights."""
        pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
        self.load_state_dict(pretrained_state_dict, strict=False)

    def _setup_generation(self, generation_kwargs: Optional[Dict], stop_words: Optional[List[str]]) -> None:
        """Setup generation configuration and stopping criteria."""
        gen_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        
        if 'eos_token_id' not in gen_kwargs and self.tokenizer.eos_token_id is not None:
            gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        if 'pad_token_id' not in gen_kwargs and self.tokenizer.pad_token_id is not None:
            gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
            
        self.generation_config = GenerationConfig(**gen_kwargs)

        for model in [self.llm, getattr(self.llm, 'base_model', None), 
                      getattr(getattr(self.llm, 'base_model', {}), 'model', None)]:
            if model and hasattr(model, 'generation_config'):
                try:
                    model.generation_config = self.generation_config
                except Exception:
                    pass

        self.stop_criteria = StoppingCriteriaList()
        if stop_words:
            for word in stop_words:
                self.stop_criteria.append(StopWordStoppingCriteria(self.tokenizer, word))

    def _init_state(self, visual_select_layer: int) -> None:
        """Initialize training state variables."""
        self.visual_select_layer = visual_select_layer
        self._is_init = True
        self.is_first_iter = True
        self._last_hidden_state = None
        self._original_input_ids = None

    # ========== Core utilities ==========

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.llm.gradient_checkpointing_enable()
        if self.enable_vision:
            self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.llm.gradient_checkpointing_disable()
        if self.enable_vision:
            self.projector.gradient_checkpointing_disable()

    activation_checkpointing_enable = gradient_checkpointing_enable
    activation_checkpointing_disable = gradient_checkpointing_disable

    def init_weights(self) -> None:
        """Initialize model weights - placeholder for compatibility."""
        pass

    def state_dict(self, *args, **kwargs) -> OrderedDict:
        """Return state dict with relevant components based on training configuration."""
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()

        # Save LLM weights (LoRA or full)
        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if 'llm.' in k})

        # Save vision and projection components
        if self.enable_vision:
            vision_keys = ['projector.', 'proj_norm.', 'conv.', 'pos_emb_2d.']
            to_return.update({k: v for k, v in state_dict.items() 
                              if any(key in k for key in vision_keys)})

        # Save WSI projector
        if self.enable_wsi_injection:
            to_return.update({k: v for k, v in state_dict.items() if 'wsi_projector.' in k})

        # Save prediction components + embeddings for new tokens
        if self.enable_regression or self.enable_survival:
            pred_keys = ['regression_head.', 'survival_head.']
            to_return.update({k: v for k, v in state_dict.items() 
                              if any(key in k for key in pred_keys)})
            embedding_keys = ['embed_tokens.weight', 'tok_embeddings.weight', 'lm_head.weight']
            for emb_key in embedding_keys:
                matching_keys = [k for k in state_dict.keys() if k.endswith(emb_key)]
                for k in matching_keys:
                    if k.startswith('llm.'):
                        to_return[k] = state_dict[k]

        return to_return

    def _project_vision_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project vision features through conv, positional embedding, projector, and proj_norm."""
        if not self.enable_vision:
            raise RuntimeError("Vision components not initialized. Set vision_conv_cfg to enable vision.")
        
        conv_input = features.to(self.llm.dtype)
        B, C, H, W = conv_input.shape

        stage_outputs = self.conv(conv_input)
        conv_output = stage_outputs[-1]

        conv_output = self.pos_emb_2d(conv_output)
        _, C_new, H_new, W_new = conv_output.shape
        feat_to_proj = conv_output.permute(0, 2, 3, 1).view(B, H_new * W_new, C_new)
        projected = self.projector(feat_to_proj.to(self.llm.dtype))
        # Normalize to match LLM embedding scale (prevents visual-feature suppression)
        return self.proj_norm(projected)

    def _project_wsi_features(self, wsi_features: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Project WSI features from multiple sources through the WSI projector.
        
        Args:
            wsi_features: List of lists, where each inner list contains tensors from
                         different WSI encoders for one sample.
        
        Returns:
            Tensor of shape (B, Num_WSI_Sources, LLM_Dim) containing all projected features.
        """
        if not hasattr(self, 'wsi_projector') or self.wsi_projector is None:
            return None
        
        B = len(wsi_features)
        num_sources = len(wsi_features[0])
        device = next(self.wsi_projector.parameters()).device
        
        source_features = []
        for source_idx in range(num_sources):
            source_batch = torch.stack([
                wsi_features[b][source_idx].to(device=device, dtype=self.llm.dtype) for b in range(B)
            ])
            source_features.append(source_batch)
        
        projected = self.wsi_projector(source_features)
        return projected.to(self.llm.dtype)

    @staticmethod
    def _get_torch_dtype() -> torch.dtype:
        """Get optimal torch dtype based on hardware capabilities."""
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings: Optional[int] = None) -> Any:
        """Process and configure language model."""
        cfg = self._prepare_for_qlora_zero3(cfg)
        llm_cfg = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)

        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings)

        return cfg

    def _prepare_for_qlora_zero3(self, cfg) -> Any:
        """Configure QLoRA with DeepSpeed ZeRO3."""
        if is_deepspeed_zero3_enabled() and hasattr(cfg, 'quantization_config'):
            torch_dtype = self._get_torch_dtype()
            cfg.torch_dtype = torch_dtype
            cfg.quantization_config.bnb_4bit_compute_dtype = torch_dtype
            cfg.quantization_config.bnb_4bit_quant_storage = torch_dtype
        return cfg

    def _prepare_for_flash_attn(self, cfg, llm_cfg) -> Tuple[Any, Any]:
        """Configure flash attention based on model type."""
        cls_name = type(llm_cfg).__name__
        torch_dtype = self._get_torch_dtype()

        if getattr(cfg, 'attn_implementation', None) == 'flash_attention_2':
            cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in self.SUPPORT_CONFIGS['FLASH2']:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in self.SUPPORT_CONFIGS['SDPA']:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    def _prepare_for_long_context_training(self, cfg, llm_cfg, max_position_embeddings: int) -> Tuple[Any, Any]:
        """Configure model for long context training with RoPE scaling."""
        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None) or {'factor': 1}
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)

        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling.get('factor', 1)
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {'type': 'linear', 'factor': scaling_factor}

        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg
        return cfg, llm_cfg

    def _build_from_cfg_or_module(self, cfg_or_mod) -> nn.Module:
        """Build module from config or return existing module."""
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError(f"Unsupported type: {type(cfg_or_mod)}")

    # ========== Forward paths ==========

    def forward(self, data: Dict[str, Any], data_samples: Optional[List] = None, mode: str = 'loss') -> Any:
        """Main forward pass with support for loss computation, prediction, and tensor modes."""
        # Initialize device on first iteration
        if self.is_first_iter:
            self.to(data['input_ids'].device)
            self.is_first_iter = False

        # Extract targets
        regression_targets = data.pop('regression_targets', None)
        survival_targets = data.pop('survival_targets', None)
        self._original_input_ids = (data['input_ids'].clone()
                                    if (self.enable_regression or self.enable_survival) and 'input_ids' in data
                                    else None)

        # Process vision features if present
        has_visual_features = 'features' in data and data['features'] is not None
        if has_visual_features and self.enable_vision:
            projected_features = self._project_vision_features(data['features'])
            data['pixel_values'] = projected_features
            data.pop('features', None)
        else:
            data['pixel_values'] = None
            data.pop('features', None)

        # Process WSI features if present
        wsi_embeddings = None
        if self.enable_wsi_injection and 'wsi_features' in data:
            wsi_features = data.pop('wsi_features')
            if wsi_features is not None and len(wsi_features) > 0:
                wsi_embeddings = self._project_wsi_features(wsi_features)
        data['wsi_embeddings'] = wsi_embeddings

        if mode == 'predict':
            self._strip_assistant_targets(data)

        # Prepare inputs using unified function
        padding_side = 'left' if mode == 'predict' else 'right'
        data = prepare_inputs_labels_for_text_and_wsi(llm=self.llm, padding_side=padding_side, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples, regression_targets, survival_targets)

        elif mode == 'predict':
            gen_data = {k: data[k] for k in ['inputs_embeds', 'attention_mask', 'position_ids'] if k in data}
            return self.predict(gen_data, data_samples, regression_targets, survival_targets)

        elif mode == 'tensor':
            return self._forward(data, data_samples)

        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

    def _strip_assistant_targets(self, data: Dict[str, torch.Tensor]) -> None:
        """Remove ground-truth continuations from prompts before generation."""
        input_ids = data.get('input_ids')
        labels = data.get('labels')
        if input_ids is None or labels is None or input_ids.ndim != 2:
            return

        pad_token_id = self.llm.config.pad_token_id
        if pad_token_id is None and getattr(self.tokenizer, 'pad_token_id', None) is not None:
            pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None and getattr(self.tokenizer, 'eos_token_id', None) is not None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        attention_mask = data.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            data['attention_mask'] = attention_mask

        for b_idx in range(input_ids.size(0)):
            gt_positions = torch.nonzero(labels[b_idx] != IGNORE_INDEX, as_tuple=False)
            if gt_positions.numel() == 0:
                continue
            cut = gt_positions[0].item()
            if cut == 0:
                cut = 1
            if cut >= input_ids.size(1):
                continue
            input_ids[b_idx, cut:] = pad_token_id
            attention_mask[b_idx, cut:] = False

    # ========== Predict and helpers ==========

    def predict(self, data: Dict[str, torch.Tensor], data_samples: Optional[List] = None, 
                regression_targets: Optional[torch.Tensor] = None,
                survival_targets: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, Any]]:
        """Generate text and compute regression/survival predictions."""
        try:
            prefix_inputs_embeds = data['inputs_embeds']
            prefix_attention_mask = data['attention_mask']
            prefix_position_ids = data.get('position_ids', None)
            B, Lp, _ = prefix_inputs_embeds.shape

            with torch.no_grad():
                captured_logits = []
                def capture_logits_processor(input_ids, scores):
                    if not captured_logits:
                        captured_logits.append(scores.detach().cpu())
                    return scores

                gen_out = self.llm.generate(
                    **data,
                    generation_config=self.generation_config,
                    stopping_criteria=self.stop_criteria,
                    bos_token_id=self.tokenizer.bos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    logits_processor=[capture_logits_processor]
                )

            generate_ids = getattr(gen_out, 'sequences', gen_out)
            gen_scores = getattr(gen_out, 'scores', None)

            batch_size = generate_ids.size(0)
            if data_samples is None:
                data_samples = [{} for _ in range(batch_size)]

            has_regression, has_survival = [], []

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

                has_reg = (self.enable_regression and self.reg_token_id is not None and 
                          (gen_id == self.reg_token_id).any().item())
                has_srv = (self.enable_survival and self.srv_token_id is not None and 
                          (gen_id == self.srv_token_id).any().item())
                
                # When gen_forcing=False, always attempt task predictions based on enabled tasks
                if not self.gen_forcing:
                    has_reg = self.enable_regression and self.reg_token_id is not None
                    has_srv = self.enable_survival and self.srv_token_id is not None
                    
                has_regression.append(bool(has_reg))
                has_survival.append(bool(has_srv))

            # If no sample generated any special tokens (or no tasks enabled), return text-only predictions
            if not (any(has_regression) or any(has_survival)):
                return data_samples

            return self._predict_tasks_from_generation(
                generate_ids=generate_ids,
                data_samples=data_samples,
                has_regression=has_regression,
                has_survival=has_survival,
                prefix_inputs_embeds=prefix_inputs_embeds,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids
            )

        finally:
            self._cleanup_prediction_state()

    def _predict_tasks_from_generation(self,
                                       generate_ids: torch.Tensor,
                                       data_samples: List[Dict[str, Any]],
                                       has_regression: List[bool],
                                       has_survival: List[bool],
                                       prefix_inputs_embeds: torch.Tensor,
                                       prefix_attention_mask: torch.Tensor,
                                       prefix_position_ids: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """Compute task predictions at generated special-token positions.
        
        When gen_forcing=False and special tokens are not generated, we use the
        hidden state at the last valid prefix position (causal LM: prefix hidden
        states are identical regardless of what is generated after, so a prefix-only
        forward pass is sufficient and avoids re-processing all generated tokens).
        When gen_forcing=True, run over the full prefix + generated sequence.
        """
        device = prefix_inputs_embeds.device
        dtype = prefix_inputs_embeds.dtype
        B, Lp, H = prefix_inputs_embeds.shape
        Lg = generate_ids.size(1)

        with torch.no_grad():
            norm = self._get_language_model_norm()

            if not self.gen_forcing:
                # Fast path: prefix-only forward – much cheaper than
                # re-processing prefix + all generated tokens.
                fwd_inputs_embeds = prefix_inputs_embeds
                fwd_attention_mask = prefix_attention_mask.to(device)
                fwd_position_ids = (prefix_position_ids.to(device)
                                    if prefix_position_ids is not None else None)
                last_valid_pos = fwd_attention_mask.bool().sum(dim=1) - 1  # (B,)
            else:
                # Full-sequence path: need hidden states at exact generated positions.
                tok_emb = self.llm.get_input_embeddings()
                gen_embeds = tok_emb(generate_ids.to(device))
                fwd_inputs_embeds = torch.cat([prefix_inputs_embeds, gen_embeds.to(dtype)], dim=1)

                pad_id = self.llm.config.pad_token_id
                if pad_id is None and hasattr(self.tokenizer, 'pad_token_id'):
                    pad_id = self.tokenizer.pad_token_id
                if pad_id is not None:
                    gen_attn = (generate_ids != pad_id).to(dtype=prefix_attention_mask.dtype, device=device)
                else:
                    gen_attn = torch.ones((B, Lg), dtype=prefix_attention_mask.dtype, device=device)
                fwd_attention_mask = torch.cat([prefix_attention_mask.to(device), gen_attn], dim=1)

                if prefix_position_ids is not None:
                    prefix_position_ids = prefix_position_ids.to(device)
                    last_pos = prefix_position_ids[:, -1].unsqueeze(1)
                    incr = torch.arange(1, Lg + 1, device=device).view(1, -1)
                    gen_pos = last_pos + incr
                    fwd_position_ids = torch.cat([prefix_position_ids, gen_pos], dim=1)
                else:
                    fwd_position_ids = None
                last_valid_pos = fwd_attention_mask.bool().sum(dim=1) - 1  # (B,)

            outputs = self.llm(inputs_embeds=fwd_inputs_embeds,
                               attention_mask=fwd_attention_mask,
                               position_ids=fwd_position_ids,
                               output_hidden_states=True,
                               return_dict=True)
            hidden = outputs.hidden_states[-1]

            # Apply RMSNorm – same as training path (compute_loss)
            if norm is not None:
                hidden = norm(hidden)

        for b in range(B):
            # Regression
            if has_regression[b] and self.enable_regression and self.reg_token_id is not None:
                embed = None

                if not self.gen_forcing:
                    # prefix-only hidden: last_valid_pos is always within bounds
                    embed = hidden[b, last_valid_pos[b]]  # (H,)
                else:
                    pos_in_gen = torch.nonzero(generate_ids[b] == self.reg_token_id, as_tuple=False).flatten()
                    if pos_in_gen.numel() > 0:
                        pos_full = int(Lp + pos_in_gen[-1].item())
                        if pos_full < hidden.size(1):
                            embed = hidden[b, pos_full]  # (H,)
                
                if embed is not None:
                    fused = embed.unsqueeze(0)
                    pred_out = self.regression_head(fused)
                    pred = float(pred_out.squeeze(-1).item()) if pred_out is not None else None
                    data_samples[b]['regression_prediction'] = pred
                    prev = data_samples[b].get('prediction_text', '')
                    text_suffix = f"[Regression: {pred:.4f}]" if pred is not None else ""
                    data_samples[b]['prediction_text'] = f"{prev} {text_suffix}".strip()

            # Survival
            if has_survival[b] and self.enable_survival and self.srv_token_id is not None:
                embed = None

                if not self.gen_forcing:
                    # prefix-only hidden: last_valid_pos is always within bounds
                    embed = hidden[b, last_valid_pos[b]]  # (H,)
                else:
                    pos_in_gen = torch.nonzero(generate_ids[b] == self.srv_token_id, as_tuple=False).flatten()
                    if pos_in_gen.numel() > 0:
                        pos_full = int(Lp + pos_in_gen[-1].item())
                        if pos_full < hidden.size(1):
                            embed = hidden[b, pos_full]  # (H,)

                if embed is None:
                    continue
                
                fused = embed.unsqueeze(0)
                
                pred_dict = {}
                
                if self.survival_method == 'discrete':
                    survival_probs_out = self.survival_head.predict_survival_probs(fused)
                    survival_probs = survival_probs_out.squeeze(0).cpu().tolist() if survival_probs_out is not None else None
                    
                    risk_score_out = self.survival_head.predict_risk_scores(fused)
                    risk_score = float(risk_score_out.squeeze(0).item()) if risk_score_out is not None else None
                    
                    median_time_out = self.survival_head.predict_median_survival_time(fused)
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
                    risk_score_out = self.survival_head.predict_risk_scores(fused)
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

    def _cleanup_prediction_state(self) -> None:
        """Clean up temporary prediction state."""
        self._original_input_ids = None
        for attr in ['_reg_token_positions', '_srv_token_positions']:
            if hasattr(self, attr):
                delattr(self, attr)

    # ========== Loss path ==========

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Custom parse_losses to ensure only 'loss' is used for backward,
        preventing double-counting of components already included in total_loss.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        # If we manually provided 'loss', use it as the primary loss for backward
        if 'loss' in losses:
            loss = losses['loss']
        else:
            # Fallback to default behavior: sum all components containing 'loss'
            loss = sum(value for key, value in log_vars if 'loss' in key)
        
        # Ensure 'loss' is the first element for MMEngine logging
        log_vars_dict = OrderedDict()
        log_vars_dict['loss'] = loss
        for name, val in log_vars:
            if name != 'loss':
                log_vars_dict[name] = val

        return loss, log_vars_dict

    def compute_loss(self, data: Dict[str, torch.Tensor], data_samples: Optional[List] = None, 
                     regression_targets: Optional[torch.Tensor] = None,
                     survival_targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute combined language modeling, regression, and survival loss."""
        self._last_hidden_state = None

        input_kwargs = {k: data[k] for k in ['input_ids', 'inputs_embeds', 'attention_mask', 'position_ids'] if k in data}
        outputs = self.llm(**input_kwargs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]

        # Compute LM loss
        logits = outputs.logits
        labels = data.get('labels', None)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            vocab_size = shift_logits.size(-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        else:
            lm_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        self._last_hidden_state = last_hidden

        # For regression/survival heads, apply final normalization to be consistent 
        # with how the lm_head receives hidden states.
        norm = self._get_language_model_norm()
        if norm is not None:
            self._last_hidden_state = norm(last_hidden)
        else:
            self._last_hidden_state = last_hidden

        # Debug logging
        if not hasattr(self, '_logged_token_stats'):
            try:
                if is_main_process() and 'labels' in data and data['labels'] is not None:
                    lbl = data['labels']
                    reg_cnt = int(((lbl == self.reg_token_id).sum().item()) if (self.enable_regression and self.reg_token_id is not None) else 0)
                    srv_cnt = int(((lbl == self.srv_token_id).sum().item()) if (self.enable_survival and self.srv_token_id is not None) else 0)
                    print_log(f"[SpecialTokenDebug] reg_token_id={self.reg_token_id}, srv_token_id={self.srv_token_id}, label_counts: REG={reg_cnt}, SRV={srv_cnt}", 'current')
            except Exception:
                pass
            self._logged_token_stats = True

        # Task losses
        reg_loss = self._compute_task_loss_safe(data['labels'], regression_targets, 'regression') if (
            self.enable_regression and regression_targets is not None and
            self.reg_token_id is not None and 'labels' in data
        ) else self._get_zero_loss_with_grad_connectivity('regression', lm_loss)

        srv_loss = self._compute_task_loss_safe(data['labels'], survival_targets, 'survival') if (
            self.enable_survival and survival_targets is not None and
            self.srv_token_id is not None and 'labels' in data
        ) else self._get_zero_loss_with_grad_connectivity('survival', lm_loss)

        total_loss = (self.lambda_llm * lm_loss +
                      self.lambda_reg * reg_loss +
                      self.lambda_srv * srv_loss)

        # NaN protection: if any individual loss is NaN/Inf, replace with small regularization
        # This keeps gradients flowing while preventing training collapse
        if not torch.isfinite(lm_loss):
            lm_loss = torch.zeros_like(lm_loss)
        if not torch.isfinite(reg_loss):
            reg_loss = self._get_regularization_loss('regression')
        if not torch.isfinite(srv_loss):
            srv_loss = self._get_regularization_loss('survival')
        
        # Recompute total_loss with sanitized components
        total_loss = (self.lambda_llm * lm_loss +
                      self.lambda_reg * reg_loss +
                      self.lambda_srv * srv_loss)

        return {
            'lm_loss': lm_loss.detach(), 
            'reg_loss': reg_loss.detach(), 
            'srv_loss': srv_loss.detach(), 
            'loss': total_loss
        }

    def _compute_task_loss_safe(self, labels: torch.Tensor, targets: Union[torch.Tensor, Dict], 
                                task: str) -> torch.Tensor:
        """Compute task loss with guaranteed gradient connectivity for multi-GPU training."""
        hidden = self._last_hidden_state
        if hidden is None:
            return self._get_zero_loss_with_grad_connectivity(task, 
                torch.zeros((), device=labels.device, dtype=self.llm.dtype))

        token_id = self.reg_token_id if task == 'regression' else self.srv_token_id
        if token_id is None:
            return self._get_zero_loss_with_grad_connectivity(task, 
                torch.zeros((), device=labels.device, dtype=self.llm.dtype))

        token_mask = (labels == token_id)

        if not torch.any(token_mask):
            return self._get_zero_loss_with_grad_connectivity(task, 
                torch.zeros((), device=labels.device, dtype=self.llm.dtype))

        seq_positions = torch.arange(hidden.size(1), device=hidden.device, dtype=torch.long)
        seq_positions = seq_positions.unsqueeze(0).expand_as(token_mask)
        weighted_positions = torch.where(token_mask, seq_positions, torch.full_like(seq_positions, -1))
        last_positions = weighted_positions.max(dim=1).values
        valid_mask = last_positions >= 0

        if not torch.any(valid_mask):
            return self._get_zero_loss_with_grad_connectivity(task, 
                torch.zeros((), device=labels.device, dtype=self.llm.dtype))

        b_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        s_idx = last_positions[valid_mask]
        task_embeds = hidden[b_idx, s_idx]

        if task == 'regression':
            predictions = self.regression_head(task_embeds).squeeze(-1)
            task_targets = targets.to(device=predictions.device)[b_idx].to(predictions.dtype)
            return self.regression_loss_fn(predictions, task_targets)
        else:  # survival
            output = self.survival_head(task_embeds)
            
            if self.survival_method == 'discrete':
                if isinstance(targets, dict) and 'target_y' in targets:
                    target_y = targets['target_y'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                    at_risk_mask = targets['at_risk_mask'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                    return logistic_hazard_loss(output, target_y, at_risk_mask)
                else:
                    raise ValueError(
                        f"Survival method is '{self.survival_method}' but data missing 'bins'. "
                        "Please ensure dataset includes pre-computed bins for discrete survival."
                    )
            else:
                time = targets['time'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                event = targets['event'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                
                valid = ~(torch.isnan(time) | torch.isnan(event))
                if not torch.any(valid):
                    return self._get_zero_loss_with_grad_connectivity(task, 
                        torch.zeros((), device=labels.device, dtype=self.llm.dtype))
                
                return cox_ph_loss(output[valid], time[valid], event[valid])

    def _get_zero_loss_with_grad_connectivity(self, task: str, base_loss: torch.Tensor) -> torch.Tensor:
        """Return small L2 regularization loss to keep gradients flowing.
        
        This prevents gradient vanishing when no valid samples exist in a batch.
        """
        return self._get_regularization_loss(task, scale=1e-6)
    
    def _get_regularization_loss(self, task: str, scale: float = 1e-6) -> torch.Tensor:
        """Return small L2 regularization loss for the specified task head.
        
        This keeps gradients flowing through task-specific parameters.
        """
        if task == 'regression' and self.enable_regression and hasattr(self, 'regression_head'):
            # L2 regularization on head weights
            reg = sum((p ** 2).mean() for p in self.regression_head.parameters() if p.requires_grad)
            if isinstance(reg, torch.Tensor):
                return scale * reg
                
        elif task == 'survival' and self.enable_survival and hasattr(self, 'survival_head'):
            reg = sum((p ** 2).mean() for p in self.survival_head.parameters() if p.requires_grad)
            if isinstance(reg, torch.Tensor):
                return scale * reg
        
        # Fallback: return zero tensor on the correct device
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return torch.zeros((), device=device, dtype=dtype)

    # ========== Misc ==========

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying LLM model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'use_llm_lora':
                return getattr(self, '_use_llm_lora', False)
            return getattr(self.llm, name)


# Alias for backward compatibility
LLaVAModel_conv = LLaVAModel_conv_unified
