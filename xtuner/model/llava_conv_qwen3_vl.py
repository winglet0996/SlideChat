# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple, Union
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.dist import is_main_process
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, GenerationConfig, StoppingCriteriaList, Qwen3VLConfig, AutoModelForCausalLM)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLForConditionalGeneration, Qwen3VLTextModel
from transformers.integrations import is_deepspeed_zero3_enabled

def patch_qwen3_vl_deepstack():
    """Patch Qwen3-VL to support sparse DeepStack injection and None-safe processing."""
    # Make _deepstack_process no-op when visual_embeds is None
    old_process = Qwen3VLTextModel._deepstack_process
    @functools.wraps(old_process)
    def new_process(self, hidden_states, visual_pos_masks, visual_embeds):
        if visual_embeds is None or visual_pos_masks is None:
            return hidden_states
        
        if visual_pos_masks.shape[1] != hidden_states.shape[1]:
            # Handle generation case where hidden_states is sliced but visual_pos_masks is not
            visual_pos_masks = visual_pos_masks[:, -hidden_states.shape[1]:]
            
        if not visual_pos_masks.any():
            return hidden_states
            
        return old_process(self, hidden_states, visual_pos_masks, visual_embeds)
    Qwen3VLTextModel._deepstack_process = new_process

    def patch_forward(cls):
        old_fwd = cls.forward
        # Do NOT use functools.wraps here to ensure transformers.generate sees the new signature
        def new_fwd(self, *args, visual_pos_masks=None, deepstack_visual_embeds=None, **kwargs):
            v_mask = visual_pos_masks if visual_pos_masks is not None else kwargs.get('visual_pos_masks', None)
            v_embeds = deepstack_visual_embeds if deepstack_visual_embeds is not None else kwargs.get('deepstack_visual_embeds', None)

            if isinstance(self, Qwen3VLModel):
                # Intercept language_model call to inject our features
                old_lm_fwd = self.language_model.forward
                def patched_lm_fwd(*la, **lk):
                    if v_mask is not None: lk['visual_pos_masks'] = v_mask
                    if v_embeds is not None: lk['deepstack_visual_embeds'] = v_embeds
                    return old_lm_fwd(*la, **lk)
                self.language_model.forward = patched_lm_fwd
                
                # Pop from kwargs to avoid multiple values for keyword argument in old_fwd
                # but only for the Model call where we manually inject into language_model
                kwargs.pop('visual_pos_masks', None)
                kwargs.pop('deepstack_visual_embeds', None)
                
                try:
                    return old_fwd(self, *args, **kwargs)
                finally:
                    self.language_model.forward = old_lm_fwd
            else:
                # For Qwen3VLForConditionalGeneration, ensure they are in kwargs for the Model call
                if v_mask is not None: kwargs['visual_pos_masks'] = v_mask
                if v_embeds is not None: kwargs['deepstack_visual_embeds'] = v_embeds
                return old_fwd(self, *args, **kwargs)
        
        new_fwd.__doc__ = old_fwd.__doc__
        new_fwd.__module__ = old_fwd.__module__
        new_fwd.__name__ = old_fwd.__name__
        cls.forward = new_fwd

    patch_forward(Qwen3VLModel)
    patch_forward(Qwen3VLForConditionalGeneration)

patch_qwen3_vl_deepstack()

AutoModelForCausalLM.register(Qwen3VLConfig, Qwen3VLForConditionalGeneration)

from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names, get_peft_model_state_dict, 
                    guess_load_checkpoint, make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)
from .custom_model import (HighResPartialConvNeXt, PositionalEmbedding2DSinusoidal, 
                           AttentionPooling, RegressionHead, SurvivalHead, 
                           cox_ph_loss, logistic_hazard_loss, WSIProjector)


def prepare_inputs_labels_for_qwen3_vl(llm, input_ids, pixel_values, 
                                    labels=None, attention_mask=None, 
                                    position_ids=None, past_key_values=None,
                                    image_grid_thw=None,
                                    deepstack_pixel_values=None,
                                    wsi_embeddings=None,
                                    padding_side='right',
                                    **kwargs):
    """
    Custom multimodal preparation for Qwen3-VL that also returns visual_pos_masks.
    Supports hierarchical DeepStack visual features and WSI global feature injection.
    
    WSI Injection Logic:
    - wsi_embeddings: Tensor of shape (B, Num_WSI_Sources, LLM_Dim)
    - Injected immediately BEFORE patch grid tokens (pixel_values)
    - Uses linear/text-like Position IDs (no 3D-RoPE)
    - Marked as visual tokens in visual_pos_masks for DeepStack
    - Labels set to IGNORE_INDEX
    """
    if pixel_values is None or len(pixel_values) == 0:
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': None,
            'labels': labels,
            'visual_pos_masks': None,
            'image_grid_thw': None,
            'deepstack_visual_embeds': None,
            'wsi_token_counts': None
        }

    # Get WSI embeddings info for this batch
    # wsi_embeddings shape: (B, Num_WSI_Sources, LLM_Dim) or None
    num_wsi_tokens = wsi_embeddings.size(1) if wsi_embeddings is not None else 0

    # remove the padding using attention_mask
    input_ids_list = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels_list = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_inputs_embeds = []
    new_labels = []
    new_visual_masks = []
    new_position_ids = []
    all_deepstack_embeds = []
    all_wsi_embeds_for_deepstack = []  # Track WSI embeddings for DeepStack injection
    new_image_grid_thw = []
    
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids_list):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_inputs_embeds = llm.get_input_embeddings()(cur_input_ids)
            new_inputs_embeds.append(cur_inputs_embeds)
            new_labels.append(labels_list[batch_idx])
            new_visual_masks.append(torch.zeros(cur_inputs_embeds.shape[0], dtype=torch.bool, device=cur_inputs_embeds.device))
            # Text position IDs: (3, seq_len)
            seq_len = cur_inputs_embeds.shape[0]
            new_position_ids.append(torch.arange(seq_len, device=cur_inputs_embeds.device).view(1, -1).expand(3, -1))
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        
        cur_new_inputs_embeds = []
        cur_new_labels = []
        cur_new_visual_mask = []
        cur_new_position_ids = []
        
        st_idx = 0
        for i in range(len(image_token_indices) - 1):
            # Text part
            text_ids = cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i+1]]
            if text_ids.shape[0] > 0:
                text_len = text_ids.shape[0]
                cur_new_inputs_embeds.append(llm.get_input_embeddings()(text_ids))
                cur_new_labels.append(labels_list[batch_idx][image_token_indices[i] + 1 : image_token_indices[i+1]])
                cur_new_visual_mask.append(torch.zeros(text_len, dtype=torch.bool, device=text_ids.device))
                # Text position IDs: (3, text_len)
                cur_new_position_ids.append(torch.arange(text_len, device=text_ids.device).view(1, -1).expand(3, -1) + st_idx)
                st_idx += text_len
            
            # Image part (with WSI injection before patch tokens)
            if i < len(image_token_indices) - 2:
                # === WSI EMBEDDING INJECTION (before patch tokens) ===
                if wsi_embeddings is not None and num_wsi_tokens > 0:
                    # Get WSI embeddings for this batch sample: (Num_WSI_Sources, LLM_Dim)
                    cur_wsi_embeds = wsi_embeddings[batch_idx]  # (Num_WSI_Sources, LLM_Dim)
                    wsi_seq_len = cur_wsi_embeds.size(0)
                    
                    # Add WSI embeddings to sequence
                    cur_new_inputs_embeds.append(cur_wsi_embeds)
                    
                    # Labels: IGNORE_INDEX for WSI tokens
                    cur_new_labels.append(torch.full((wsi_seq_len,), IGNORE_INDEX, 
                                                     device=cur_wsi_embeds.device, dtype=labels.dtype))
                    
                    # Visual mask: True for WSI tokens (they are visual features)
                    cur_new_visual_mask.append(torch.ones(wsi_seq_len, dtype=torch.bool, 
                                                         device=cur_wsi_embeds.device))
                    
                    # Position IDs: Linear/text-like (no 3D-RoPE for WSI global tokens)
                    # All 3 dimensions use the same linear increment
                    wsi_pos = torch.arange(wsi_seq_len, device=cur_wsi_embeds.device).view(1, -1) + st_idx
                    cur_new_position_ids.append(wsi_pos.expand(3, -1))  # (3, wsi_seq_len)
                    st_idx += wsi_seq_len
                    
                    # Store WSI embeds for DeepStack injection
                    all_wsi_embeds_for_deepstack.append(cur_wsi_embeds)
                
                # === PATCH GRID TOKENS ===
                cur_pixel_values = pixel_values[cur_image_idx]
                if image_grid_thw is not None:
                    grid_thw = image_grid_thw[cur_image_idx]
                    new_image_grid_thw.append(grid_thw)
                    
                    # Calculate 3D position IDs for Qwen3-VL
                    # Since pixel_values are already projected/merged features, 
                    # we use the grid dimensions directly without further spatial merging.
                    t, h, w = grid_thw
                    llm_grid_t, llm_grid_h, llm_grid_w = t.item(), h.item(), w.item()
                    
                    t_index = torch.arange(llm_grid_t, device=cur_input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h, device=cur_input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w, device=cur_input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    cur_new_position_ids.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                    st_idx += llm_grid_t * llm_grid_h * llm_grid_w
                
                if deepstack_pixel_values is not None:
                    # Collect all stages for this image, preserving None for sparse layers
                    all_deepstack_embeds.append([v[cur_image_idx] if v is not None else None for v in deepstack_pixel_values])
                else:
                    all_deepstack_embeds.append([cur_pixel_values])

                cur_image_idx += 1
                cur_new_inputs_embeds.append(cur_pixel_values)
                cur_new_labels.append(torch.full((cur_pixel_values.shape[0],), IGNORE_INDEX, device=cur_pixel_values.device, dtype=labels.dtype))
                cur_new_visual_mask.append(torch.ones(cur_pixel_values.shape[0], dtype=torch.bool, device=cur_pixel_values.device))

        new_inputs_embeds.append(torch.cat(cur_new_inputs_embeds))
        new_labels.append(torch.cat(cur_new_labels))
        new_visual_masks.append(torch.cat(cur_new_visual_mask))
        new_position_ids.append(torch.cat(cur_new_position_ids, dim=1))

    # Pad
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    final_inputs_embeds = torch.zeros((batch_size, max_len, new_inputs_embeds[0].shape[-1]), dtype=new_inputs_embeds[0].dtype, device=new_inputs_embeds[0].device)
    final_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    final_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_inputs_embeds[0].device)
    final_visual_pos_masks = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_inputs_embeds[0].device)
    final_position_ids = torch.zeros((3, batch_size, max_len), dtype=torch.long, device=new_inputs_embeds[0].device)

    for i, (emb, lbl, vmask, pids) in enumerate(zip(new_inputs_embeds, new_labels, new_visual_masks, new_position_ids)):
        cur_len = emb.shape[0]
        if padding_side == 'right':
            final_inputs_embeds[i, :cur_len] = emb
            final_labels[i, :cur_len] = lbl
            final_attention_mask[i, :cur_len] = True
            final_visual_pos_masks[i, :cur_len] = vmask
            final_position_ids[:, i, :cur_len] = pids
        else:
            final_inputs_embeds[i, -cur_len:] = emb
            final_labels[i, -cur_len:] = lbl
            final_attention_mask[i, -cur_len:] = True
            final_visual_pos_masks[i, -cur_len:] = vmask
            final_position_ids[:, i, -cur_len:] = pids

    # Prepare DeepStack embeds: List[Tensor] where each tensor is (total_num_images * seq_len, hidden_dim)
    # Now also includes WSI embeddings concatenated at the beginning for enhanced task predictions
    if all_deepstack_embeds:
        num_stages = len(all_deepstack_embeds[0])
        deepstack_visual_embeds = []
        
        # Prepare WSI embeddings for DeepStack: concatenate all WSI embeds across batch
        wsi_embeds_concat = None
        if all_wsi_embeds_for_deepstack:
            wsi_embeds_concat = torch.cat(all_wsi_embeds_for_deepstack, dim=0)  # (Total_WSI_Tokens, LLM_Dim)
        
        for s in range(num_stages):
            # Collect only non-None tensors for this stage
            stage_tensors = [img_stages[s] for img_stages in all_deepstack_embeds if img_stages[s] is not None]
            if stage_tensors:
                patch_embeds = torch.cat(stage_tensors, dim=0)  # (Total_Patch_Tokens, LLM_Dim)
                
                # Concatenate WSI embeds at the beginning of patch features for this stage
                if wsi_embeds_concat is not None:
                    combined = torch.cat([wsi_embeds_concat, patch_embeds], dim=0)
                    deepstack_visual_embeds.append(combined)
                else:
                    deepstack_visual_embeds.append(patch_embeds)
            else:
                deepstack_visual_embeds.append(None)
    else:
        deepstack_visual_embeds = None

    return {
        'inputs_embeds': final_inputs_embeds,
        'labels': final_labels,
        'attention_mask': final_attention_mask,
        'visual_pos_masks': final_visual_pos_masks,
        'deepstack_visual_embeds': deepstack_visual_embeds,
        'image_grid_thw': torch.stack(new_image_grid_thw) if new_image_grid_thw else None,
        'position_ids': final_position_ids,
        'wsi_token_counts': num_wsi_tokens  # Number of WSI tokens injected per sample
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


class LLaVAModel_conv(BaseModel):
    """
    Multi-modal LLaVA model with convolution-based vision processing and regression/survival prediction.

    The model leverages special tokens (<REG> for regression, <SRV> for survival) to trigger 
    downstream predictions. Regression predictions rely solely on the special token embeddings
    processed by the LLM, while survival predictions fuse token embeddings with visual features
    through attention pooling.

    Key features:
    - Clean token-only regression approach  
    - Visual-aware survival prediction
    - End-to-end differentiable training
    - Multi-GPU training compatibility with gradient synchronization deadlock prevention
    
    Multi-GPU Safety:
    The loss computation ensures all task-specific parameters (regression_head, survival_head)
    participate in the computation graph on every GPU, even when special tokens are absent
    from some data shards. This prevents gradient synchronization deadlocks in distributed training.
    """

    # Supported model configurations for flash attention
    SUPPORT_CONFIGS = {
        'SDPA': ('LlamaConfig', 'GemmaConfig', 'MistralConfig', 'MixtralConfig', 
                 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config', 'Qwen3VLConfig'),
        'FLASH2': ('InternLM2Config', 'LlamaConfig', 'GemmaConfig', 'MistralConfig', 
                   'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config', 'Qwen3VLConfig')
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
                 lambda_llm: float = 0.1, lambda_reg: float = 1.0, lambda_srv: float = 1.0,
                 vision_conv_cfg: Optional[Dict] = None,
                 deepstack_visual_indexes: List[int] = [8, 16, 24],
                 deepstack_reverse_injection: bool = False,
                 wsi_feature_dims: Optional[List[int]] = None,
                 head_scaling: Union[float, List[float]] = [0.0, 0.0, 1.0]):
        """
        Multi-modal LLaVA model with regression and survival prediction capabilities.
        
        Args:
            survival_method: 'cox' for Cox proportional hazards or 'discrete' for discrete-time survival
            num_survival_intervals: Number of intervals (K) for discrete method
            wsi_feature_dims: List of input dimensions for each WSI encoder source.
                              E.g., [768, 1024, 768] for three different encoders (TITAN, CONCH, UNI).
                              If None, WSI feature injection is disabled.
            head_scaling: Multiplier(s) for hidden layer dimension in task heads and WSI projector.
                          - If float: applied to all (reg, srv, wsi).
                          - If list of 3: [reg_mult, srv_mult, wsi_mult].
                          Set to 0 for a single linear layer (most lightweight).
            deepstack_reverse_injection: Whether to reverse the order of visual features 
                                        injected into DeepStack (deep features to shallow layers).
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
            lambda_llm=lambda_llm,
            lambda_reg=lambda_reg,
            lambda_srv=lambda_srv,
            vision_conv_cfg=vision_conv_cfg,
            deepstack_visual_indexes=deepstack_visual_indexes,
            deepstack_reverse_injection=deepstack_reverse_injection,
            wsi_feature_dims=wsi_feature_dims,
            head_scaling=head_scaling
        )

        # Initialize model components
        self._init_llm(llm, max_position_embeddings)
        self._init_vision_components()
        self._init_projector(projector_depth)
        
        # Initialize WSI projector if wsi_feature_dims is provided
        if wsi_feature_dims:
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
                         survival_method: str,
                         lambda_llm: float, lambda_reg: float, lambda_srv: float, vision_conv_cfg: Optional[Dict],
                         deepstack_visual_indexes: List[int],
                         deepstack_reverse_injection: bool = False,
                         wsi_feature_dims: Optional[List[int]] = None,
                         head_scaling: Union[float, List[float]] = 1.0) -> None:
        """Initialize core model attributes."""
        self.freeze_llm = freeze_llm
        self.enable_regression = enable_regression
        self.reg_token = reg_token
        self.enable_survival = enable_survival
        self.srv_token = srv_token
        self.num_survival_intervals = num_survival_intervals
        self.survival_method = survival_method  # 'cox' or 'discrete'
        self.lambda_llm = lambda_llm
        self.lambda_reg = lambda_reg
        self.lambda_srv = lambda_srv
        # LoRA flags
        self.use_llm_lora = False
        self._use_llm_lora = False
        # Token ids will be filled later
        self.reg_token_id = None
        self.srv_token_id = None
        self.vision_conv_cfg = vision_conv_cfg
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.deepstack_reverse_injection = deepstack_reverse_injection
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
        
        # Handle composite configs for use_cache
        if hasattr(self.llm.config, 'use_cache'):
            self.llm.config.use_cache = False
        if hasattr(self.llm.config, 'text_config'):
            self.llm.config.text_config.use_cache = False

        dispatch_modules(self.llm)

    def _init_vision_components(self) -> None:
        """Initialize vision processing components."""
        default_conv_cfg = dict(
            in_chans=768,
            depths=[1, 3],
            dims=[1024, 2048],
            drop_path_rate=0.3,
            num_downsamples=1,
        )
        if self.vision_conv_cfg is not None:
            default_conv_cfg.update(self.vision_conv_cfg)
        self.conv = HighResPartialConvNeXt(
            **default_conv_cfg
        ).to(self.llm.dtype)

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
        
        Handles both:
        - Direct model: self.llm.model.language_model.norm
        - PEFT-wrapped: self.llm.base_model.model.model.language_model.norm
        """
        # Try different paths for PEFT and non-PEFT models
        paths_to_try = [
            # PEFT-wrapped Qwen3-VL
            lambda: self.llm.base_model.model.model.language_model.norm,
            # Non-PEFT Qwen3-VL
            lambda: self.llm.model.language_model.norm,
            # Alternative PEFT path
            lambda: self.llm.model.model.language_model.norm,
        ]
        
        for get_norm in paths_to_try:
            try:
                norm = get_norm()
                if norm is not None:
                    return norm
            except AttributeError:
                continue
        
        return None

    def _init_projector(self, depth: int = 2) -> None:
        """Initialize the vision-language projector."""
        self.projectors = nn.ModuleList()
        llm_hidden_size = self._get_llm_hidden_size()
        for dim in self.conv.dims:
            projector_config = ProjectorConfig(
                visual_hidden_size=dim,
                llm_hidden_size=llm_hidden_size,
                depth=depth
            )
            self.projectors.append(ProjectorModel(projector_config).to(self.llm.dtype))
        
        # For backward compatibility and easy access to the main projector
        self.projector = self.projectors[-1]

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

        # Ensure pad_token_id is set for the model to avoid defaulting to 0 ('!' in Qwen)
        if getattr(self.llm.config, 'pad_token_id', None) is None:
            if self.tokenizer.pad_token_id is not None:
                self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            elif self.tokenizer.eos_token_id is not None:
                self.llm.config.pad_token_id = self.tokenizer.eos_token_id

        # Enable training for new special tokens while allowing broader gradient flow
        if enable_regression or enable_survival:
            self._enable_selective_training()

    def _enable_selective_training(self):
        """Enable training for special tokens and task-specific components while preserving gradient flow."""
        new_token_ids = [tid for tid in [self.reg_token_id, self.srv_token_id] if tid is not None]
        if not new_token_ids:
            return

        # Register gradient hook to only train new token embeddings
        self._register_embedding_grad_hook()
        print_log("[SelectiveTraining] Configured gradient flow for special token learning", 'current')

    def _register_embedding_grad_hook(self) -> None:
        """Register gradient hook to zero out gradients for original vocabulary.
        
        Handles both tied and untied word embeddings (e.g., 4B vs 8B).
        """
        old_vocab_size = self._original_vocab_size
        
        def _get_zero_hook(name):
            def _zero_old_token_grad(grad: torch.Tensor) -> torch.Tensor:
                if grad is not None:
                    # Only allow gradients for new tokens added at the end
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
        """Initialize special token embedding with semantic meaning for both input and output."""
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

            # 1. Initialize input embedding
            if valid_ids:
                base_vec = emb.weight[valid_ids].mean(dim=0)
            else:
                base_vec = emb.weight.mean(dim=0)
            
            # Use small perturbation to avoid identical embeddings if multiple tokens added
            noise = 1e-3 * torch.randn_like(base_vec)
            emb.weight[token_id] = 1.05 * base_vec + noise

            # 2. Initialize output head weight (only for untied models like 8B)
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
                time_intervals=None,  # Time intervals not needed for training/C-index
                hidden_mult=srv_mult
            ).to(dtype=self.llm.dtype)
            
            # Set appropriate loss function based on method
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

        # Configure parameter training after all components are initialized
        if freeze_llm:
            self._freeze_llm_with_exceptions()
        
        # Final parameter check and configuration
        self._configure_parameter_gradients()

        if use_activation_checkpointing:
            self._setup_checkpointing()

    def _setup_lora(self, lora_config: Dict, use_activation_checkpointing: bool) -> None:
        """Setup LoRA configuration."""
        lora_config = self._build_from_cfg_or_module(lora_config)
        self.llm = prepare_model_for_kbit_training(self.llm, use_activation_checkpointing)

        if lora_config.target_modules is None:
            # For Qwen3-VL, we must avoid targeting 'proj' which matches Conv3d in visual encoder.
            # Searching only in language_model avoids finding 'proj' from vision blocks.
            target_model = getattr(self.llm, 'model', self.llm)
            target_model = getattr(target_model, 'language_model', target_model)
            lora_config.target_modules = find_all_linear_names(target_model)

        self.llm = get_peft_model(self.llm, lora_config)

    def _freeze_llm_with_exceptions(self) -> None:
        """Freeze LLM parameters while keeping task-critical components trainable."""
        if self.use_llm_lora:
            # PEFT handles freezing the base model automatically.
            # Manual freezing here could accidentally disable LoRA adapters.
            return
            
        self.llm.requires_grad_(False)
        print_log("Froze base LLM parameters", 'current')

    def _configure_parameter_gradients(self) -> None:
        """Centralized parameter gradient configuration for consistent multi-GPU behavior."""
        trainable_params = []
        
        # 1. Handle Embeddings and LM Head (special token learning)
        is_tied = getattr(self.llm.config, 'tie_word_embeddings', True)
        embed_layer = self.llm.get_input_embeddings()
        output_layer = self.llm.get_output_embeddings()
        
        # Enable input embeddings
        if embed_layer is not None and hasattr(embed_layer, 'weight'):
            embed_layer.weight.requires_grad = True
        
        # Enable output head if not tied
        if not is_tied and output_layer is not None and hasattr(output_layer, 'weight'):
            output_layer.weight.requires_grad = True

        # Log trainable parameters for embeddings
        if hasattr(self, '_original_vocab_size'):
            num_new = len(self.tokenizer) - self._original_vocab_size
            hidden_dim = self._get_llm_hidden_size()
            if is_tied:
                trainable_params.append(f"embed+lm_head (tied, new tokens): {num_new * hidden_dim:,}")
            else:
                trainable_params.append(f"embed+lm_head (untied, new tokens): {num_new * hidden_dim * 2:,}")
        else:
            total_emb = (embed_layer.weight.numel() if embed_layer is not None else 0) + \
                        (output_layer.weight.numel() if not is_tied and output_layer is not None else 0)
            trainable_params.append(f"embeddings: {total_emb:,}")

        # 2. Enable LoRA parameters if applicable
        if self.use_llm_lora:
            lora_param_count = 0
            for name, param in self.llm.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    lora_param_count += param.numel()
            if lora_param_count > 0:
                trainable_params.append(f"LoRA: {lora_param_count:,}")

        # Task-specific modules are trainable by default
        if self.enable_regression and hasattr(self, 'regression_head'):
            reg_params = sum(p.numel() for p in self.regression_head.parameters())
            trainable_params.append(f"regression_head: {reg_params:,}")
            
        if self.enable_survival and hasattr(self, 'survival_head'):
            srv_params = sum(p.numel() for p in self.survival_head.parameters()) 
            trainable_params.append(f"survival_head: {srv_params:,}")

        # Vision components are trainable by default
        conv_params = sum(p.numel() for p in self.conv.parameters())
        proj_params = sum(p.numel() for p in self.projectors.parameters())
        trainable_params.extend([
            f"conv: {conv_params:,}",
            f"projectors: {proj_params:,}"
        ])
        
        # WSI projector if enabled
        if self.enable_wsi_injection and hasattr(self, 'wsi_projector'):
            wsi_proj_params = sum(p.numel() for p in self.wsi_projector.parameters())
            trainable_params.append(f"wsi_projector: {wsi_proj_params:,}")

        if is_main_process():
            print_log(f"[ParameterConfig] Trainable components: {', '.join(trainable_params)}", 'current')
            
        # Validate parameter consistency for multi-GPU training
        self._validate_parameter_consistency()

    def _validate_parameter_consistency(self) -> None:
        """Validate that trainable parameters are consistent across all GPUs."""
        trainable_param_names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
        
        # Create deterministic hash of trainable parameter names for consistency check
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

        for projector in self.projectors:
            projector.enable_input_require_grads()
        self.gradient_checkpointing_enable()

    def _load_pretrained_weights(self, pretrained_pth: str) -> None:
        """Load pretrained model weights."""
        pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
        self.load_state_dict(pretrained_state_dict, strict=False)

    def _setup_generation(self, generation_kwargs: Optional[Dict], stop_words: Optional[List[str]]) -> None:
        """Setup generation configuration and stopping criteria."""
        gen_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        
        # Ensure eos_token_id is set from tokenizer if not specified
        if 'eos_token_id' not in gen_kwargs and self.tokenizer.eos_token_id is not None:
            gen_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        if 'pad_token_id' not in gen_kwargs and self.tokenizer.pad_token_id is not None:
            gen_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
            
        self.generation_config = GenerationConfig(**gen_kwargs)

        # Set generation config on all relevant models
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
        for projector in self.projectors:
            projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.llm.gradient_checkpointing_disable()
        for projector in self.projectors:
            projector.gradient_checkpointing_disable()

    activation_checkpointing_enable = gradient_checkpointing_enable
    activation_checkpointing_disable = gradient_checkpointing_disable

    def init_weights(self) -> None:
        """Initialize model weights - placeholder for compatibility."""
        pass

    def state_dict(self, *args, **kwargs) -> OrderedDict:
        """Return state dict with relevant components based on training configuration."""
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()

        # 1. Save LLM weights (LoRA or full)
        if self.use_llm_lora:
            # get_peft_model_state_dict filters keys with 'lora_'
            # Note: xtuner's get_peft_model_state_dict preserves the 'llm.' prefix if present in state_dict
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if 'llm.' in k})

        # 2. Save vision and projection components (including WSI projector)
        vision_keys = ['projectors.', 'projector.', 'conv.', 'wsi_projector.']
        to_return.update({k: v for k, v in state_dict.items() 
                          if any(key in k for key in vision_keys)})

        # 3. Save prediction components
        if self.enable_regression or self.enable_survival:
            pred_keys = ['regression_head.', 'survival_head.']
            to_return.update({k: v for k, v in state_dict.items() 
                              if any(key in k for key in pred_keys)})
            
        # 4. CRITICAL: Always save embeddings and lm_head if they were trainable
        # This ensures special tokens and their predictions are preserved.
        embedding_keys = ['embed_tokens.weight', 'tok_embeddings.weight', 'lm_head.weight']
        for emb_key in embedding_keys:
            matching_keys = [k for k in state_dict.keys() if k.endswith(emb_key)]
            for k in matching_keys:
                if k.startswith('llm.'):
                    to_return[k] = state_dict[k]

        return to_return

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """
        Custom load_state_dict with explicit and safe key remapping between
        LoRA and non-LoRA (full/alignment) checkpoints.

        Supported paths:
        1) non-LoRA ckpt  -> LoRA model
        2) LoRA ckpt      -> non-LoRA model
        3) LoRA ckpt      -> LoRA model (direct, no remap)
        4) non-LoRA ckpt  -> non-LoRA model (direct)
        """
        new_state_dict = {}

        is_lora_model = bool(self.use_llm_lora)
        is_lora_ckpt = any(
            k.startswith("llm.") and "base_model.model" in k
            for k in state_dict
        )

        mapped_count = 0
        llm_keys_count = 0

        for k, v in state_dict.items():
            new_key = k

            if k.startswith("llm."):
                llm_keys_count += 1

                # Case 1: non-LoRA ckpt -> LoRA model
                if is_lora_model and not is_lora_ckpt:
                    new_key = k.replace("llm.", "llm.base_model.model.", 1)
                    mapped_count += 1

                # Case 2: LoRA ckpt -> non-LoRA model
                elif not is_lora_model and is_lora_ckpt:
                    new_key = k.replace("llm.base_model.model.", "llm.", 1)
                    mapped_count += 1

                # Case 3 & 4:
                #   LoRA ckpt -> LoRA model
                #   non-LoRA ckpt -> non-LoRA model
                # Keys are already correct; no remapping needed.

            new_state_dict[new_key] = v

        if is_main_process():
            mode = "LoRA" if is_lora_model else "Full/Alignment"
            ckpt_type = "LoRA" if is_lora_ckpt else "Full/Alignment"

            print_log(
                f"[WeightLoading] Loaded {len(state_dict)} keys | "
                f"LLM: {llm_keys_count} | Remapped: {mapped_count} | "
                f"Checkpoint: {ckpt_type} -> Model: {mode}",
                "current",
            )

        return super().load_state_dict(new_state_dict, strict=strict)

    def _project_vision_features(self, features: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Project vision features through conv and hierarchical projectors.
        Returns (list_of_projected_features, grid_thw)."""
        conv_input = features.to(self.llm.dtype)
        B, C, H, W = conv_input.shape

        mask = (torch.ones(B, 1, H, W, device=conv_input.device, dtype=conv_input.dtype) 
                if masks is None else masks.to(conv_input.device, dtype=conv_input.dtype))

        stage_outputs, updated_mask = self.conv(conv_input, mask)
        
        # Final resolution
        final_feat = stage_outputs[-1]
        _, _, H_final, W_final = final_feat.shape
        
        projected_stages = []
        for i, (feat, projector) in enumerate(zip(stage_outputs, self.projectors)):
            # Downsample to final resolution if needed
            if feat.shape[2:] != (H_final, W_final):
                feat = F.adaptive_avg_pool2d(feat, (H_final, W_final))
            
            # Project
            feat_to_proj = feat.permute(0, 2, 3, 1).reshape(B, H_final * W_final, -1)
            projected_stages.append(projector(feat_to_proj.to(self.llm.dtype)))
            
        grid_thw = torch.tensor([[1, H_final, W_final]] * B, device=conv_input.device, dtype=torch.long)
        
        return projected_stages, grid_thw

    def _project_wsi_features(self, wsi_features: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Project WSI features from multiple sources through the WSI projector.
        
        Args:
            wsi_features: List of lists, where each inner list contains tensors from
                         different WSI encoders for one sample.
                         Shape: [[Tensor(D1), Tensor(D2), Tensor(D3)], ...] for B samples.
        
        Returns:
            Tensor of shape (B, Num_WSI_Sources, LLM_Dim) containing all projected features.
        """
        if not hasattr(self, 'wsi_projector') or self.wsi_projector is None:
            return None
        
        B = len(wsi_features)
        num_sources = len(wsi_features[0])
        device = next(self.wsi_projector.parameters()).device
        
        # Stack features from each source across batch
        # wsi_features: [[D1, D2, D3], [D1, D2, D3], ...] for B samples
        source_features = []
        for source_idx in range(num_sources):
            # Collect this source's features across all batch samples
            source_batch = torch.stack([
                wsi_features[b][source_idx].to(device=device, dtype=self.llm.dtype) for b in range(B)
            ])  # (B, D_source)
            source_features.append(source_batch)
        
        # Project through WSI projector
        projected = self.wsi_projector(source_features)  # (B, Num_WSI_Sources, LLM_Dim)
        
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
            cfg.dtype = torch_dtype
            cfg.quantization_config.bnb_4bit_compute_dtype = torch_dtype
            cfg.quantization_config.bnb_4bit_quant_storage = torch_dtype
        return cfg

    def _prepare_for_flash_attn(self, cfg, llm_cfg) -> Tuple[Any, Any]:
        """Configure flash attention based on model type."""
        cls_name = type(llm_cfg).__name__
        torch_dtype = self._get_torch_dtype()

        if getattr(cfg, 'attn_implementation', None) == 'flash_attention_2':
            cfg.dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in self.SUPPORT_CONFIGS['FLASH2']:
            cfg.dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in self.SUPPORT_CONFIGS['SDPA']:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    def _prepare_for_long_context_training(self, cfg, llm_cfg, max_position_embeddings: int) -> Tuple[Any, Any]:
        """Configure model for long context training with RoPE scaling."""
        # Handle composite configs like Qwen3VLConfig
        target_cfg = llm_cfg
        if hasattr(llm_cfg, 'text_config'):
            target_cfg = llm_cfg.text_config

        orig_rope_scaling = getattr(target_cfg, 'rope_scaling', None) or {'factor': 1}
        orig_ctx_len = getattr(target_cfg, 'max_position_embeddings', None)

        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling.get('factor', 1)
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(math.ceil(max_position_embeddings / orig_ctx_len))
                target_cfg.rope_scaling = {'type': 'linear', 'factor': scaling_factor}

        llm_cfg.attn_implementation = 'flash_attention_2'
        if hasattr(llm_cfg, 'text_config'):
            llm_cfg.text_config.attn_implementation = 'flash_attention_2'

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

        # Extract targets and preserve original input_ids for predict-time reference
        regression_targets = data.pop('regression_targets', None)
        survival_targets = data.pop('survival_targets', None)
        self._original_input_ids = (data['input_ids'].clone()
                                    if (self.enable_regression or self.enable_survival) and 'input_ids' in data
                                    else None)

        # Process vision features
        projected_stages, grid_thw = self._project_vision_features(data['features'], data.get('masks'))
        
        # Map multi-stage features to specific LLM layers (e.g., 8, 16, 24)
        ds_indexes = self.deepstack_visual_indexes
        num_layers = self.llm.config.text_config.num_hidden_layers

        # Handle injection order (normal or reversed)
        stages_to_inject = projected_stages
        if self.deepstack_reverse_injection:
            stages_to_inject = list(reversed(projected_stages))

        # Build sparse list aligned to total LLM layers; only target indices receive features
        deepstack_embeds = [None] * num_layers
        for idx, feat in zip(ds_indexes, stages_to_inject):
            if idx < num_layers:
                deepstack_embeds[idx] = feat

        data['pixel_values'] = projected_stages[-1]  # Main feature
        data['deepstack_pixel_values'] = deepstack_embeds  # Sparse deepstack features
        data['image_grid_thw'] = grid_thw
        data.pop('features', None)
        data.pop('masks', None)
        
        # Process WSI features if available
        wsi_embeddings = None
        if self.enable_wsi_injection and 'wsi_features' in data:
            wsi_features = data.pop('wsi_features')  # List of List[Tensor]: [[D1, D2, D3], ...]
            if wsi_features is not None and len(wsi_features) > 0:
                wsi_embeddings = self._project_wsi_features(wsi_features)
        data['wsi_embeddings'] = wsi_embeddings

        if mode == 'predict':
            self._strip_assistant_targets(data)

        # Prepare multimodal inputs
        is_qwen3_vl = getattr(self.llm.config, 'model_type', None) == 'qwen3_vl'
        padding_side = 'left' if mode == 'predict' else 'right'
        if is_qwen3_vl:
            data = prepare_inputs_labels_for_qwen3_vl(llm=self.llm, padding_side=padding_side, **data)
        else:
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples, regression_targets, survival_targets)

        elif mode == 'predict':
            # Filter fields required by generation
            gen_fields = ['inputs_embeds', 'attention_mask', 'position_ids', 'visual_pos_masks', 'deepstack_visual_embeds', 'image_grid_thw']
            gen_data = {k: data[k] for k in gen_fields if k in data}
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
        """Generate text and compute regression/survival predictions only if special tokens are generated."""
        try:
            # Save prefix to reconstruct the full sequence (vision + generated)
            prefix_inputs_embeds = data['inputs_embeds']  # (B, Lp, H)
            prefix_attention_mask = data['attention_mask']  # (B, Lp)
            prefix_position_ids = data.get('position_ids', None)
            B, Lp, _ = prefix_inputs_embeds.shape

            # 1) Text generation (depends on generated special tokens)
            with torch.no_grad():
                # Capture raw logits before masking (top_p/top_k) for accurate AUROC
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

            # HF generate may return a tensor or a GenerateOutput depending on model/version
            generate_ids = getattr(gen_out, 'sequences', gen_out)
            gen_scores = getattr(gen_out, 'scores', None)

            batch_size = generate_ids.size(0)
            if data_samples is None:
                data_samples = [{} for _ in range(batch_size)]

            # Decode and detect special tokens in generated continuation
            has_regression, has_survival = [], []

            # Precompute MCQA choice token ids for (A..E) if possible
            def _choice_token_id(letter: str) -> Optional[int]:
                cache = getattr(self, '_mcqa_choice_token_id_cache', None)
                if cache is None:
                    cache = {}
                    setattr(self, '_mcqa_choice_token_id_cache', cache)
                if letter in cache:
                    return cache[letter]

                # Prefer single-token encodings for stable logit extraction
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

            # Use captured raw logits for first step if available (avoids top_p masking)
            first_step_logits = captured_logits[0] if captured_logits else None
            if first_step_logits is None and gen_scores is not None and len(gen_scores) > 0:
                first_step_logits = gen_scores[0]  # (B, vocab)

            for i, gen_id in enumerate(generate_ids):
                clean_text = self.tokenizer.decode(gen_id, skip_special_tokens=True).strip()
                data_samples[i]['prediction_text'] = clean_text

                # Attach choice logits for MCQA metrics (used by AUROC when K=2)
                if first_step_logits is not None and i < first_step_logits.size(0):
                    logits_row = first_step_logits[i]
                    choice_logits = {}
                    # Support A-Z for broader MCQA compatibility
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

                has_reg = (self.enable_regression and self.reg_token_id is not None and (gen_id == self.reg_token_id).any().item())
                has_srv = (self.enable_survival and self.srv_token_id is not None and (gen_id == self.srv_token_id).any().item())
                has_regression.append(bool(has_reg))
                has_survival.append(bool(has_srv))

            # If no sample generated any special tokens, return text-only predictions
            if not (any(has_regression) or any(has_survival)):
                return data_samples

            # 2) Task predictions from generated special tokens (use full sequence with vision prefix)
            return self._predict_tasks_from_generation(
                generate_ids=generate_ids,
                data_samples=data_samples,
                has_regression=has_regression,
                has_survival=has_survival,
                prefix_inputs_embeds=prefix_inputs_embeds,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids,
                visual_pos_masks=data.get('visual_pos_masks'),
                deepstack_visual_embeds=data.get('deepstack_visual_embeds'),
                image_grid_thw=data.get('image_grid_thw')
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
                                       prefix_position_ids: Optional[torch.Tensor] = None,
                                       visual_pos_masks: Optional[torch.Tensor] = None,
                                       deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
                                       image_grid_thw: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """Compute task predictions at generated special-token positions using only token embeddings.
        
        The visual context is already encoded through the language model's cross-attention mechanism
        during the generation process, so the special token embeddings contain sufficient multimodal
        information for downstream predictions.
        """
        device = prefix_inputs_embeds.device
        dtype = prefix_inputs_embeds.dtype
        B, Lp, H = prefix_inputs_embeds.shape
        Lg = generate_ids.size(1)

        # Build embeddings for generated tokens
        with torch.no_grad():
            tok_emb = self.llm.get_input_embeddings()
            gen_embeds = tok_emb(generate_ids.to(device))  # (B, Lg, H), dtype matches tok_emb

            # Assemble full sequence embeddings: [vision/text prefix embeds] + [generated embeds]
            full_inputs_embeds = torch.cat([prefix_inputs_embeds, gen_embeds.to(dtype)], dim=1)  # (B, Lp+Lg, H)

            # Build attention mask for generated tokens (respect pad if exists)
            pad_id = self.llm.config.pad_token_id
            if pad_id is None and hasattr(self.tokenizer, 'pad_token_id'):
                pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                gen_attn = (generate_ids != pad_id).to(dtype=prefix_attention_mask.dtype, device=device)
            else:
                gen_attn = torch.ones((B, Lg), dtype=prefix_attention_mask.dtype, device=device)

            full_attention_mask = torch.cat([prefix_attention_mask.to(device), gen_attn], dim=1)  # (B, Lp+Lg)

            # Position IDs (optional). If prefix provided, continue monotonically; else omit.
            # For Qwen3-VL, position_ids has shape (3, B, Lp) for 3D M-RoPE (temporal, height, width)
            if prefix_position_ids is not None:
                prefix_position_ids = prefix_position_ids.to(device)
                
                # Check if this is Qwen3-VL style 3D position IDs (3, B, L) or standard 2D (B, L)
                if prefix_position_ids.dim() == 3 and prefix_position_ids.size(0) == 3:
                    # Qwen3-VL: position_ids shape is (3, B, Lp)
                    # For generated text tokens, all 3 dimensions should increment monotonically from last position
                    last_pos = prefix_position_ids[:, :, -1:].max(dim=0, keepdim=False)[0]  # (B, 1) - use max across 3 dims
                    incr = torch.arange(1, Lg + 1, device=device).view(1, -1)  # (1, Lg)
                    gen_pos = last_pos + incr  # (B, Lg)
                    # Expand to (3, B, Lg) - text tokens use same position for all 3 dimensions
                    gen_pos_3d = gen_pos.unsqueeze(0).expand(3, -1, -1)  # (3, B, Lg)
                    full_position_ids = torch.cat([prefix_position_ids, gen_pos_3d], dim=2)  # (3, B, Lp+Lg)
                else:
                    # Standard 2D position_ids (B, L)
                    last_pos = prefix_position_ids[:, -1].unsqueeze(1)  # (B, 1)
                    incr = torch.arange(1, Lg + 1, device=device).view(1, -1)  # (1, Lg)
                    gen_pos = last_pos + incr  # (B, Lg)
                    full_position_ids = torch.cat([prefix_position_ids, gen_pos], dim=1)  # (B, Lp+Lg)
            else:
                full_position_ids = None

            # Prepare visual masks for full sequence
            full_visual_pos_masks = None
            if visual_pos_masks is not None:
                gen_vmask = torch.zeros((B, Lg), dtype=torch.bool, device=device)
                full_visual_pos_masks = torch.cat([visual_pos_masks.to(device), gen_vmask], dim=1)

            # Forward through base model to get last hidden state for the full sequence
            # Get hidden states from full sequence (vision + generated tokens)
            llm_kwargs = {
                'inputs_embeds': full_inputs_embeds,
                'attention_mask': full_attention_mask,
                'position_ids': full_position_ids,
                'output_hidden_states': True,
                'return_dict': True
            }
            if full_visual_pos_masks is not None:
                llm_kwargs['visual_pos_masks'] = full_visual_pos_masks
            if deepstack_visual_embeds is not None:
                llm_kwargs['deepstack_visual_embeds'] = deepstack_visual_embeds
            if image_grid_thw is not None:
                llm_kwargs['image_grid_thw'] = image_grid_thw

            outputs = self.llm(**llm_kwargs)
            hidden = outputs.hidden_states[-1]  # (B, Lp+Lg, H)

            # Apply RMSNorm to hidden states - same as training path (compute_loss)
            # hidden_states[-1] are PRE-normalization; the model applies RMSNorm before lm_head.
            # Regression/survival heads were trained on normalized hidden states,
            # so we must apply the same normalization at inference time.
            norm = self._get_language_model_norm()
            if norm is not None:
                hidden = norm(hidden)

        # For each batch, locate generated special-token positions and predict
        for b in range(B):
            # Regression
            if has_regression[b] and self.enable_regression and self.reg_token_id is not None:
                pos_in_gen = torch.nonzero(generate_ids[b] == self.reg_token_id, as_tuple=False).flatten()
                if pos_in_gen.numel() > 0:
                    pos_full = int(Lp + pos_in_gen[-1].item())
                    if pos_full < hidden.size(1):
                        embed = hidden[b, pos_full]  # (H,)
                        fused = self._fuse_token_with_vision(embed, b, 'regression')  # Shape depends on mode
                        pred_out = self.regression_head(fused)
                        pred = float(pred_out.squeeze(-1).item()) if pred_out is not None else None
                        data_samples[b]['regression_prediction'] = pred
                        prev = data_samples[b].get('prediction_text', '')
                        text_suffix = f"[Regression: {pred:.4f}]" if pred is not None else ""
                        data_samples[b]['prediction_text'] = f"{prev} {text_suffix}".strip()

            # Survival (supports both Cox and Discrete methods)
            if has_survival[b] and self.enable_survival and self.srv_token_id is not None:
                pos_in_gen = torch.nonzero(generate_ids[b] == self.srv_token_id, as_tuple=False).flatten()
                if pos_in_gen.numel() > 0:
                    pos_full = int(Lp + pos_in_gen[-1].item())
                    if pos_full < hidden.size(1):
                        embed = hidden[b, pos_full]  # (H,)
                        fused = self._fuse_token_with_vision(embed, b, 'survival')  # (1, H)
                        
                        # Build prediction dict based on survival method
                        pred_dict = {}
                        
                        if self.survival_method == 'discrete':
                            # Discrete method: predict survival probabilities and derive risk
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
                            # Cox method: predict risk score only
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

    def _fuse_token_with_vision(self, token_embed: torch.Tensor, b: int, task_type: str = 'survival') -> torch.Tensor:
        """Process token embedding for regression or survival tasks."""
        # Both regression and survival use only the special token embedding
        return token_embed.unsqueeze(0)  # (1, H)

    def _cleanup_prediction_state(self) -> None:
        """Clean up temporary prediction state."""
        self._original_input_ids = None
        for attr in ['_reg_token_positions', '_srv_token_positions']:
            if hasattr(self, attr):
                delattr(self, attr)

    # ========== Loss path ==========

    def compute_loss(self, data: Dict[str, torch.Tensor], data_samples: Optional[List] = None, 
                     regression_targets: Optional[torch.Tensor] = None,
                     survival_targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute combined language modeling, regression, and survival loss."""
        self._last_hidden_state = None

        # Get last hidden state - force output_hidden_states=True for LoRA compatibility
        input_kwargs = {k: data[k] for k in ['input_ids', 'inputs_embeds', 'attention_mask', 'position_ids', 'visual_pos_masks', 'deepstack_visual_embeds', 'image_grid_thw'] if k in data}
        outputs = self.llm(**input_kwargs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]

        # Compute LM loss (causal shift)
        # CRITICAL: Use outputs.logits directly, NOT self.llm.lm_head(last_hidden)!
        # Qwen3-VL applies RMSNorm to hidden states before lm_head internally.
        # Using raw hidden states produces incorrect logits (off by ~150 in magnitude).
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

        # For regression/survival heads, we also need to apply RMSNorm to be consistent
        # with how the lm_head receives hidden states. This ensures the task heads 
        # receive properly normalized features.
        norm = self._get_language_model_norm()
        if norm is not None:
            self._last_hidden_state = norm(last_hidden)
        else:
            # Fallback for models without this structure
            self._last_hidden_state = last_hidden

        # One-time token debug
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

        # Task losses with gradient connectivity guarantee
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

        return {'lm_loss': lm_loss, 'reg_loss': reg_loss, 'srv_loss': srv_loss, 'loss': total_loss}

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

        # Vectorized discovery of the latest special token per batch using masked max
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
            # Ensure targets is on the same device before indexing
            task_targets = targets.to(device=predictions.device)[b_idx].to(predictions.dtype)
            return self.regression_loss_fn(predictions, task_targets)
        else:  # survival
            output = self.survival_head(task_embeds)
            
            # Robust dispatch: always match loss path to model's configured head method
            if self.survival_method == 'discrete':
                # Discrete head pathway
                if isinstance(targets, dict) and 'target_y' in targets:
                    target_y = targets['target_y'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                    at_risk_mask = targets['at_risk_mask'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                    return logistic_hazard_loss(output, target_y, at_risk_mask)
                else:
                    # If model is discrete but data has no bins, we cannot train
                    raise ValueError(
                        f"Survival method is '{self.survival_method}' but data missing 'bins'. "
                        "Please ensure dataset includes pre-computed bins for discrete survival."
                    )
            else:
                # Cox head pathway (Default)
                # Even if data contains 'bins', we use 'time'/'event' fields for CoxPH head
                time = targets['time'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                event = targets['event'].to(device=output.device)[b_idx].to(dtype=output.dtype)
                
                # Exclude samples with missing data (NaNs)
                valid = ~(torch.isnan(time) | torch.isnan(event))
                if not torch.any(valid):
                    return self._get_zero_loss_with_grad_connectivity(task, 
                        torch.zeros((), device=labels.device, dtype=self.llm.dtype))
                
                return cox_ph_loss(output[valid], time[valid], event[valid])

    def _get_zero_loss_with_grad_connectivity(self, task: str, base_loss: torch.Tensor) -> torch.Tensor:
        """Return zero loss while ensuring task parameters remain in computation graph.
        
        This prevents gradient synchronization deadlocks in multi-GPU training by ensuring
        all trainable parameters participate in the computation graph on all devices.
        """
        zero_loss = torch.zeros_like(base_loss)
        
        # Add minimal parameter connectivity to ensure gradients flow through task heads
        if task == 'regression' and self.enable_regression and hasattr(self, 'regression_head'):
            # Sum all parameters and multiply by epsilon to maintain gradient connectivity
            param_sum = sum(p.sum() for p in self.regression_head.parameters() if p.requires_grad)
            if isinstance(param_sum, torch.Tensor):
                zero_loss = zero_loss + 1e-12 * param_sum
                
        elif task == 'survival' and self.enable_survival and hasattr(self, 'survival_head'):
            # Sum all parameters and multiply by epsilon to maintain gradient connectivity  
            param_sum = sum(p.sum() for p in self.survival_head.parameters() if p.requires_grad)
            if isinstance(param_sum, torch.Tensor):
                zero_loss = zero_loss + 1e-12 * param_sum
                
        return zero_loss

    # ========== Misc ==========

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying LLM model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'use_llm_lora':
                return getattr(self, '_use_llm_lora', False)
            return getattr(self.llm, name)