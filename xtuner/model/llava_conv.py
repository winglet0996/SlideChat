# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import torch.nn as nn
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine.dist import is_main_process
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, GenerationConfig, StoppingCriteriaList)
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names, get_peft_model_state_dict, 
                    guess_load_checkpoint, make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)
from .custom_model import HighResPartialConvNeXt, PositionalEmbedding2DSinusoidal, AttentionPooling, RegressionHead, SurvivalHead


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
    """Multi-modal LLaVA model with convolution-based vision processing and regression capabilities."""
    
    # Supported model configurations for flash attention
    SUPPORT_CONFIGS = {
        'SDPA': ('LlamaConfig', 'GemmaConfig', 'MistralConfig', 'MixtralConfig', 
                'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config'),
        'FLASH2': ('InternLM2Config', 'LlamaConfig', 'GemmaConfig', 'MistralConfig', 
                  'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config')
    }

    def __init__(self, llm, tokenizer, freeze_llm: bool = True, visual_select_layer: int = -2,
                 pretrained_pth: Optional[str] = None, projector_depth: int = 2,
                 llm_lora: Optional[Dict] = None, use_activation_checkpointing: bool = True,
                 max_position_embeddings: Optional[int] = None, hidden_size: Optional[int] = None,
                 generation_kwargs: Optional[Dict] = None, stop_words: Optional[List[str]] = None,
                 enable_regression: bool = True, reg_token: str = '<REG>',
                 enable_survival: bool = True, srv_token: str = '<SRV>',
                 num_survival_intervals: int = 6,
                 lambda_llm: float = 1.0, lambda_reg: float = 10.0, lambda_srv: float = 10.0):
        super().__init__()
        
        # Initialize core attributes
        self._init_attributes(freeze_llm, enable_regression, reg_token, enable_survival, 
                            srv_token, num_survival_intervals, lambda_llm, lambda_reg, lambda_srv)
        
        # Initialize model components
        self._init_llm(llm, max_position_embeddings)
        self._init_vision_components()
        self._init_projector(projector_depth)
        
        # Setup tokenizer and special tokens
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

    def _init_attributes(self, freeze_llm: bool, enable_regression: bool, reg_token: str,
                        enable_survival: bool, srv_token: str, num_survival_intervals: int,
                        lambda_llm: float, lambda_reg: float, lambda_srv: float) -> None:
        """Initialize core model attributes."""
        self.freeze_llm = freeze_llm
        self.enable_regression = enable_regression
        self.reg_token = reg_token
        self.enable_survival = enable_survival
        self.srv_token = srv_token
        self.num_survival_intervals = num_survival_intervals
        self.lambda_llm = lambda_llm
        self.lambda_reg = lambda_reg
        self.lambda_srv = lambda_srv
        self.use_llm_lora = False
        self._use_llm_lora = False
        self.reg_token_id = None
        self.srv_token_id = None

    def _init_llm(self, llm, max_position_embeddings: Optional[int]) -> None:
        """Initialize the language model."""
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
            self.llm = self._build_from_cfg_or_module(llm)
        
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

    def _init_vision_components(self) -> None:
        """Initialize vision processing components."""
        self.conv = HighResPartialConvNeXt(
            in_chans=768, depths=[1, 1, 1], dims=[768, 1024, 2048],
            drop_path_rate=0.1, num_downsamples=2
        ).to(self.llm.dtype)
        
        self.pos_emb_2d = PositionalEmbedding2DSinusoidal(
            d_model=self.conv.dims[-1], scale_mode='learned', init_pe_scale=0.1
        ).to(self.llm.dtype)

    def _init_projector(self, depth: int = 2) -> None:
        """Initialize the vision-language projector."""
        projector_config = ProjectorConfig(
            visual_hidden_size=self.conv.dims[-1],
            llm_hidden_size=self.llm.config.hidden_size,
            depth=depth
        )
        self.projector = ProjectorModel(projector_config).to(self.llm.dtype)

    def _setup_tokenizer_and_tokens(self, tokenizer, enable_regression: bool, enable_survival: bool) -> None:
        """Setup tokenizer and add special tokens efficiently."""
        self.tokenizer = BUILDER.build(tokenizer)
        
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

    def _add_special_tokens(self, tokens: List[str]) -> None:
        """Add special tokens and resize embeddings."""
        try:
            special_tokens = [AddedToken(token, normalized=False, special=True) for token in tokens]
            num_added = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        except Exception:
            num_added = self.tokenizer.add_tokens(tokens)
        
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))

    def _init_token_embedding(self, token_id: int, task_type: str) -> None:
        """Initialize special token embedding with semantic meaning."""
        emb = self.llm.get_input_embeddings()
        if emb is None or not hasattr(emb, 'weight') or token_id >= emb.weight.size(0):
            return

        with torch.no_grad():
            # Choose initialization tokens based on task
            if task_type == 'regression':
                init_tokens = ["value", "number", "result", "score", "level"]
            else:  # survival
                init_tokens = ["survival", "time", "risk", "hazard", "outcome"]
            
            # Get valid token embeddings
            valid_ids = []
            for token in init_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if (token_id is not None and 
                    token_id != self.tokenizer.unk_token_id and 
                    token_id < len(self.tokenizer) - 10):
                    valid_ids.append(token_id)
            
            if valid_ids:
                base_vec = emb.weight[valid_ids].mean(dim=0) * 1.5
            else:
                base_vec = emb.weight[:-10].mean(dim=0)
            
            emb.weight[token_id] = base_vec
        
        emb.weight.requires_grad = True

    def _init_prediction_modules(self) -> None:
        """Initialize prediction-specific modules."""
        llm_hidden = self.llm.config.hidden_size
        vis_channels = self.conv.dims[-1]
        
        self.attention_pool = AttentionPooling(
            q_dim=llm_hidden, kv_dim=vis_channels, hidden_dim=llm_hidden
        ).to(self.llm.dtype)
        
        if self.enable_regression:
            self.regression_head = RegressionHead(
                in_dim=llm_hidden * 2, hidden_dim=llm_hidden
            ).to(self.llm.dtype)
            self.regression_loss_fn = nn.SmoothL1Loss(beta=1.0)
        
        if self.enable_survival:
            from .custom_model import logistic_hazard_loss
            # Single source of truth for survival intervals; shared with metrics via outputs
            self.survival_time_intervals = [0, 1, 2, 3, 5, 7, 10]
            self.survival_head = SurvivalHead(
                in_dim=llm_hidden * 2, hidden_dim=llm_hidden, 
                num_intervals=self.num_survival_intervals,
                time_intervals=self.survival_time_intervals
            ).to(self.llm.dtype)
            self.survival_loss_fn = logistic_hazard_loss

    def _configure_training(self, llm_lora: Optional[Dict], use_activation_checkpointing: bool, freeze_llm: bool) -> None:
        """Configure training settings including LoRA and checkpointing."""
        self.use_llm_lora = llm_lora is not None
        self._use_llm_lora = self.use_llm_lora
        
        if self.use_llm_lora:
            self._setup_lora(llm_lora, use_activation_checkpointing)
        
        if freeze_llm and (self.enable_regression or self.enable_survival):
            self.llm.requires_grad_(False)
            
        if use_activation_checkpointing:
            self._setup_checkpointing()

    def _setup_lora(self, lora_config: Dict, use_activation_checkpointing: bool) -> None:
        """Setup LoRA configuration."""
        lora_config = self._build_from_cfg_or_module(lora_config)
        self.llm = prepare_model_for_kbit_training(self.llm, use_activation_checkpointing)
        
        if lora_config.target_modules is None:
            lora_config.target_modules = find_all_linear_names(self.llm)
        
        self.llm = get_peft_model(self.llm, lora_config)

    def _setup_checkpointing(self) -> None:
        """Setup gradient checkpointing."""
        if hasattr(self.llm, 'enable_input_require_grads'):
            self.llm.enable_input_require_grads()
        else:
            self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        self.projector.enable_input_require_grads()
        self.gradient_checkpointing_enable()

    def _load_pretrained_weights(self, pretrained_pth: str) -> None:
        """Load pretrained model weights."""
        pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
        self.load_state_dict(pretrained_state_dict, strict=False)

    def _setup_generation(self, generation_kwargs: Optional[Dict], stop_words: Optional[List[str]]) -> None:
        """Setup generation configuration and stopping criteria."""
        self.generation_config = GenerationConfig(**(generation_kwargs or {}))
        
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
        self.last_conv_output = None
        self.last_conv_mask = None
        self._last_hidden_state = None
        self._original_input_ids = None
        
    # Hidden states are captured directly from model outputs where needed
    # to avoid fragile hooks under LoRA/PEFT wrappers.

    # Removed hook-based hidden state capture. Hidden states are read
    # directly from model outputs in compute_loss and _predict_task.

    # Core functionality methods
    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.llm.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.llm.gradient_checkpointing_disable()
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
        vision_keys = ['projector.', 'conv.', 'pos_emb_2d.']
        to_return.update({k: v for k, v in state_dict.items() 
                         if any(key in k for key in vision_keys)})
        
        # Save prediction components
        if self.enable_regression or self.enable_survival:
            pred_keys = ['attention_pool.', 'regression_head.', 'survival_head.']
            to_return.update({k: v for k, v in state_dict.items() 
                             if any(key in k for key in pred_keys)})
            
            # Save embedding weights to preserve special tokens
            embedding_keys = ['embed_tokens.weight', 'tok_embeddings.weight', 'lm_head.weight']
            for emb_key in embedding_keys:
                matching_keys = [k for k in state_dict.keys() if k.endswith(emb_key)]
                for k in matching_keys:
                    to_return[k] = state_dict[k]
        
        return to_return

    def _project_vision_features(self, features: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project vision features through conv, positional embedding, and final projector."""
        conv_input = features.to(self.llm.dtype)
        B, C, H, W = conv_input.shape
        
        # Create default mask if not provided
        mask = (torch.ones(B, 1, H, W, device=conv_input.device, dtype=conv_input.dtype) 
                if masks is None else masks.to(conv_input.device, dtype=conv_input.dtype))
        
        # Process through conv and cache for regression
        conv_output, updated_mask = self.conv(conv_input, mask)
        self.last_conv_output = conv_output
        self.last_conv_mask = updated_mask
        
        # Add positional embedding and reshape for projection
        conv_output = self.pos_emb_2d(conv_output)
        _, C_new, H_new, W_new = conv_output.shape
        feat_to_proj = conv_output.permute(0, 2, 3, 1).view(B, H_new * W_new, C_new)
        
        return self.projector(feat_to_proj.to(self.llm.dtype))

    def forward(self, data: Dict[str, Any], data_samples: Optional[List] = None, mode: str = 'loss') -> Any:
        """Main forward pass with support for loss computation, prediction, and tensor modes."""
        # Initialize device on first iteration
        if self.is_first_iter:
            self.to(data['input_ids'].device)
            self.is_first_iter = False
        
        # Extract targets and preserve original input_ids for special token processing
        regression_targets = data.pop('regression_targets', None)
        survival_targets = data.pop('survival_targets', None)
        self._original_input_ids = (data['input_ids'].clone() 
                                   if (self.enable_regression or self.enable_survival) and 'input_ids' in data 
                                   else None)
        
        # Process vision features
        projected_features = self._project_vision_features(data['features'], data.get('masks'))
        data['pixel_values'] = projected_features
        data.pop('features', None)
        data.pop('masks', None)
        
        # Handle prediction mode data slicing
        if mode == 'predict':
            start_idx = ((data['labels'] != -100).cumsum(dim=1) == 0).sum(dim=1).min().item()
            self._preserve_special_token_positions(start_idx)
            
            slice_keys = ['input_ids', 'attention_mask', 'position_ids', 'labels']
            for key in slice_keys:
                if key in data:
                    data[key] = data[key][:, :start_idx]
        
        # Prepare multimodal inputs
        data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)
        
        # Route to appropriate method
        if mode == 'loss':
            return self.compute_loss(data, data_samples, regression_targets, survival_targets)
        elif mode == 'predict':
            filtered_data = {k: data[k] for k in ['inputs_embeds', 'attention_mask'] if k in data}
            return self.predict(filtered_data, data_samples, regression_targets, survival_targets)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

    def _preserve_special_token_positions(self, start_idx: int) -> None:
        """Preserve special token positions for prediction mode."""
        if self._original_input_ids is None:
            return
            
        for token_id, attr_name in [(self.reg_token_id, '_reg_token_positions'), 
                                   (self.srv_token_id, '_srv_token_positions')]:
            if token_id is not None:
                positions = []
                for b in range(self._original_input_ids.size(0)):
                    mask = (self._original_input_ids[b] == token_id)
                    if mask.any():
                        pos = torch.nonzero(mask, as_tuple=False).flatten()
                        if len(pos) > 0:
                            positions.append((b, pos[-1].item()))
                setattr(self, attr_name, positions)
        
        # self._original_input_ids = self._original_input_ids[:, :start_idx]

    # Configuration helper methods
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

    # Token extraction methods
    def _extract_token_embeddings(self, hidden: torch.Tensor, token_id: int, 
                                 positions: Optional[List] = None) -> Tuple[List[torch.Tensor], List[int]]:
        """Extract embeddings for a specific token from hidden states."""
        if hidden is None or token_id is None:
            return [], []
        
        embeds, valid_indices = [], []
        batch_size = hidden.size(0)
        
        # Use preserved positions if available (predict mode)
        if positions:
            for batch_idx, pos in positions:
                if (batch_idx < batch_size and batch_idx < self._original_input_ids.size(0) 
                    and pos < hidden.size(1)):
                    embeds.append(hidden[batch_idx, pos, :])
                    valid_indices.append(batch_idx)
        else:
            # Standard approach for training mode
            for b in range(min(batch_size, self._original_input_ids.size(0))):
                mask = (self._original_input_ids[b] == token_id)
                if mask.any():
                    positions = torch.nonzero(mask, as_tuple=False).flatten()
                    if len(positions) > 0:
                        pos = positions[-1].item()  # Use last occurrence
                        if pos < hidden.size(1):
                            embeds.append(hidden[b, pos, :])
                            valid_indices.append(b)
        
        return embeds, valid_indices

    def _compute_task_predictions(self, embeds: List[torch.Tensor], valid_indices: List[int], 
                                task: str) -> List:
        """Compute predictions for regression or survival task."""
        if not embeds:
            return []
        
        predictions = []
        for i, embed in enumerate(embeds):
            b = valid_indices[i]
            
            # Get visual features for attention pooling
            if self.last_conv_output is not None and b < self.last_conv_output.size(0):
                vis_feats = self.last_conv_output[b:b+1]
                vis_mask = None if self.last_conv_mask is None else self.last_conv_mask[b:b+1]
                fused_vis = self.attention_pool(embed.unsqueeze(0), vis_feats, vis_mask)
            else:
                fused_vis = torch.zeros_like(embed.unsqueeze(0))
            
            # Fuse and predict
            fused = torch.cat([embed.unsqueeze(0), fused_vis], dim=-1)
            
            if task == 'regression':
                pred = self.regression_head(fused).squeeze(-1).item()
                predictions.append(pred)
            else:  # survival
                logits = self.survival_head(fused).squeeze(0)
                
                # Compute comprehensive survival predictions
                survival_probs = self.survival_head.predict_survival_probs(fused).squeeze(0)
                risk_score = self.survival_head.predict_risk_scores(fused).squeeze(0).item()
                
                # If time intervals are available, compute median survival time
                median_time = self.survival_head.predict_median_survival_time(
                        fused, self.survival_time_intervals
                    ).squeeze(0).item()
                
                predictions.append({
                    'logits': logits.cpu().float().numpy(),
                    'survival_probs': survival_probs.cpu().float().numpy(),
                    'risk_score': risk_score,
                    'median_survival_time': median_time,
                    # Provide time intervals for evaluators to consume
                    'time_intervals': list(getattr(self, 'survival_time_intervals', getattr(self.survival_head, 'time_intervals', [])))
                })
        
        return predictions

    # Main prediction and loss methods
    def predict(self, data: Dict[str, torch.Tensor], data_samples: Optional[List] = None, 
                regression_targets: Optional[torch.Tensor] = None,
                survival_targets: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, Any]]:
        """Generate predictions for text generation, regression, and survival tasks."""
        try:
            # Check if input contains special tokens to determine task type
            if self._original_input_ids is not None:
                batch_size = self._original_input_ids.size(0)
                
                # Check for regression tokens
                if (self.enable_regression and self.reg_token_id is not None and
                    any((self._original_input_ids[b] == self.reg_token_id).any() for b in range(batch_size))):
                    return self._predict_task(data, data_samples, 'regression')
                
                # Check for survival tokens  
                if (self.enable_survival and self.srv_token_id is not None and
                    any((self._original_input_ids[b] == self.srv_token_id).any() for b in range(batch_size))):
                    return self._predict_task(data, data_samples, 'survival')
            
            # Standard text generation
            return self._predict_text_generation(data, data_samples)
        
        finally:
            self._cleanup_prediction_state()

    def _predict_task(self, data: Dict[str, torch.Tensor], data_samples: Optional[List], 
                     task: str) -> List[Dict[str, Any]]:
        """Handle regression or survival prediction."""
        self._last_hidden_state = None
        
        with torch.no_grad():
            outputs = self.llm(**data, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1] if getattr(outputs, 'hidden_states', None) is not None else None
            self._last_hidden_state = hidden
            
            batch_size = data.get('inputs_embeds', data.get('input_ids')).size(0)
            if hidden is None:
                return self._create_empty_predictions(batch_size, task)
            
            # Extract embeddings and compute predictions
            token_id = self.reg_token_id if task == 'regression' else self.srv_token_id
            positions_attr = '_reg_token_positions' if task == 'regression' else '_srv_token_positions'
            positions = getattr(self, positions_attr, None)
            embeds, valid_indices = self._extract_token_embeddings(hidden, token_id, positions)
            predictions = self._compute_task_predictions(embeds, valid_indices, task)
            
            return self._format_task_predictions(predictions, valid_indices, data_samples, batch_size, task)

    def _predict_text_generation(self, data: Dict[str, torch.Tensor], 
                                data_samples: Optional[List]) -> List[Dict[str, Any]]:
        """Handle standard text generation."""
        generate_ids = self.llm.generate(
            **data, generation_config=self.generation_config, stopping_criteria=self.stop_criteria,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        
        if data_samples is None:
            data_samples = [{} for _ in range(len(generate_ids))]
        
        for i, gen_id in enumerate(generate_ids[:len(data_samples)]):
            generated_text = self.tokenizer.decode(gen_id, skip_special_tokens=True).strip()
            data_samples[i]['prediction_text'] = generated_text
        
        return data_samples

    def _create_empty_predictions(self, batch_size: int, task: str) -> List[Dict[str, Any]]:
        """Create empty prediction results."""
        if task == 'regression':
            return [{'regression_prediction': None, 'prediction_text': "N/A"} for _ in range(batch_size)]
        else:  # survival
            return [{
                'survival_prediction': {
                    'logits': None,
                    'survival_probs': None,
                    'risk_score': None,
                    'median_survival_time': None,
                    'time_intervals': getattr(self, 'survival_time_intervals', None)
                },
                'prediction_text': "N/A"
            } for _ in range(batch_size)]

    def _format_task_predictions(self, predictions: List, valid_indices: List[int], 
                                data_samples: Optional[List], batch_size: int, task: str) -> List[Dict[str, Any]]:
        """Format task predictions into result format."""
        if data_samples is None:
            data_samples = [{} for _ in range(batch_size)]
        
        pred_idx = 0
        for i in range(min(batch_size, len(data_samples))):
            if i in valid_indices and pred_idx < len(predictions):
                if task == 'regression':
                    data_samples[i]['regression_prediction'] = predictions[pred_idx]
                    data_samples[i]['prediction_text'] = str(predictions[pred_idx])
                else:  # survival
                    pred_dict = predictions[pred_idx]
                    # Ensure time intervals are present for downstream metrics
                    if 'time_intervals' not in pred_dict:
                        intervals = getattr(self, 'survival_time_intervals', getattr(self, 'survival_head', None))
                        if isinstance(intervals, list):
                            pass
                        elif hasattr(self, 'survival_head') and hasattr(self.survival_head, 'time_intervals'):
                            intervals = self.survival_head.time_intervals
                        else:
                            intervals = None
                        if intervals is not None:
                            try:
                                pred_dict['time_intervals'] = list(intervals)
                            except Exception:
                                pred_dict['time_intervals'] = intervals
                    data_samples[i]['survival_prediction'] = pred_dict
                    
                    # Create readable prediction text
                    risk_score = pred_dict.get('risk_score', 'N/A')
                    median_time = pred_dict.get('median_survival_time', 'N/A')
                    data_samples[i]['prediction_text'] = (
                        f"Risk Score: {risk_score:.4f}, "
                        f"Median Survival: {median_time}" if median_time != 'N/A' 
                        else f"Risk Score: {risk_score}"
                    )
                    
                    # Add individual components for compatibility
                    data_samples[i]['risk_score'] = pred_dict.get('risk_score')
                    data_samples[i]['survival_probs'] = pred_dict.get('survival_probs')
                    data_samples[i]['median_survival_time'] = pred_dict.get('median_survival_time')
                pred_idx += 1
            else:
                # Create appropriate empty prediction
                if task == 'regression':
                    data_samples[i]['regression_prediction'] = None
                    data_samples[i]['prediction_text'] = "N/A"
                else:  # survival
                    empty_pred = {
                        'logits': None,
                        'survival_probs': None,
                        'risk_score': None,
                        'median_survival_time': None
                    }
                    data_samples[i]['survival_prediction'] = empty_pred
                    data_samples[i]['risk_score'] = None
                    data_samples[i]['survival_probs'] = None
                    data_samples[i]['median_survival_time'] = None
                    data_samples[i]['prediction_text'] = "N/A"
        
        return data_samples

    def compute_loss(self, data: Dict[str, torch.Tensor], data_samples: Optional[List] = None, 
                    regression_targets: Optional[torch.Tensor] = None,
                    survival_targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute combined language modeling, regression, and survival loss."""
        # Get LLM output and capture hidden states
        self._last_hidden_state = None
        outputs = self.llm(**data, output_hidden_states=True, return_dict=True)
        lm_loss = outputs.loss
        # Capture hidden states precisely from outputs
        self._last_hidden_state = outputs.hidden_states[-1] if getattr(outputs, 'hidden_states', None) is not None else None
        
        # Initialize task losses
        reg_loss = torch.zeros_like(lm_loss)
        srv_loss = torch.zeros_like(lm_loss)
        
        # Compute task-specific losses
        if (self.enable_regression and regression_targets is not None and 
            self.reg_token_id is not None and 'labels' in data):
            reg_loss = self._compute_task_loss(data['labels'], regression_targets, 'regression')
        
        if (self.enable_survival and survival_targets is not None and
            self.srv_token_id is not None and 'labels' in data):
            srv_loss = self._compute_task_loss(data['labels'], survival_targets, 'survival')
        
        # Combine losses
        total_loss = (self.lambda_llm * lm_loss + 
                     self.lambda_reg * reg_loss + 
                     self.lambda_srv * srv_loss)
        
        return {'lm_loss': lm_loss, 'reg_loss': reg_loss, 'srv_loss': srv_loss, 'loss': total_loss}

    def _compute_task_loss(self, labels: torch.Tensor, targets: Union[torch.Tensor, Dict], 
                          task: str) -> torch.Tensor:
        """Compute loss for regression or survival task."""
        hidden = self._last_hidden_state
        if hidden is None:
            return torch.zeros((), device=labels.device, dtype=self.llm.dtype)
        
        # Get appropriate token ID and mask
        token_id = self.reg_token_id if task == 'regression' else self.srv_token_id
        if token_id is None:
            return torch.zeros((), device=labels.device, dtype=self.llm.dtype)
        
        # Find token positions
        token_mask = (labels == token_id)
        if not token_mask.any():
            return torch.zeros((), device=labels.device, dtype=self.llm.dtype)
        
        # Extract valid token positions
        batch_indices, seq_indices = [], []
        for b in range(token_mask.size(0)):
            positions = torch.nonzero(token_mask[b], as_tuple=False)
            if positions.numel() > 0:
                seq_pos = positions[-1, 0].item()  # Use last occurrence
                if seq_pos < hidden.size(1):
                    batch_indices.append(b)
                    seq_indices.append(seq_pos)
        
        if not batch_indices:
            return torch.zeros((), device=labels.device, dtype=self.llm.dtype)
        
        # Get embeddings and compute predictions
        b_idx = torch.tensor(batch_indices, device=hidden.device, dtype=torch.long)
        s_idx = torch.tensor(seq_indices, device=hidden.device, dtype=torch.long)
        task_embeds = hidden[b_idx, s_idx]
        
        # Pool visual features
        if self.last_conv_output is not None:
            vis_feats = self.last_conv_output[b_idx]
            vis_mask = None if self.last_conv_mask is None else self.last_conv_mask[b_idx]
            pooled_vis = self.attention_pool(task_embeds, vis_feats, vis_mask)
        else:
            pooled_vis = torch.zeros_like(task_embeds)
        
        # Fuse and compute loss
        fused = torch.cat([task_embeds, pooled_vis], dim=-1)
        
        if task == 'regression':
            predictions = self.regression_head(fused).squeeze(-1)
            task_targets = targets[b_idx].to(predictions.dtype)
            return self.regression_loss_fn(predictions, task_targets)
        else:  # survival
            logits = self.survival_head(fused)
            target_y = targets['target_y'][b_idx].to(logits.dtype)
            at_risk_mask = targets['at_risk_mask'][b_idx].to(logits.dtype)
            return self.survival_loss_fn(logits, target_y, at_risk_mask)

    def _cleanup_prediction_state(self) -> None:
        """Clean up temporary prediction state."""
        self._original_input_ids = None
        for attr in ['_reg_token_positions', '_srv_token_positions']:
            if hasattr(self, attr):
                delattr(self, attr)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying LLM model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'use_llm_lora':
                return getattr(self, '_use_llm_lora', False)
            return getattr(self.llm, name)

    # No destructor required; no forward hooks are registered.
