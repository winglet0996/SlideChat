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
from xtuner.utils import StopWordStoppingCriteria, IGNORE_INDEX
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names, get_peft_model_state_dict, 
                    guess_load_checkpoint, make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)
from .custom_model import HighResConvNeXtV2Pyramid, PositionalEmbedding2DSinusoidal, AttentionPooling, RegressionHead, SurvivalHead


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
                 lambda_llm: float = 0.1, lambda_reg: float = 1.0, lambda_srv: float = 1.0,
                 vision_conv_cfg: Optional[Dict] = None):
        """
        Multi-modal LLaVA model with regression and survival prediction capabilities.
        Regression relies solely on special token embeddings processed by LLM.
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
            lambda_llm=lambda_llm,
            lambda_reg=lambda_reg,
            lambda_srv=lambda_srv,
            vision_conv_cfg=vision_conv_cfg
        )

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

    # ========== Initialization helpers ==========

    def _init_attributes(self, freeze_llm: bool, enable_regression: bool, reg_token: str,
                         enable_survival: bool, srv_token: str, num_survival_intervals: int,
                         lambda_llm: float, lambda_reg: float, lambda_srv: float, vision_conv_cfg: Optional[Dict]) -> None:
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
        # LoRA flags
        self.use_llm_lora = False
        self._use_llm_lora = False
        # Token ids will be filled later
        self.reg_token_id = None
        self.srv_token_id = None
        self.vision_conv_cfg = vision_conv_cfg

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
        default_conv_cfg = dict(
            in_chans=768,
            depths=[1, 3],
            dims=[1024, 2048],
            drop_path_rate=0.3,
            num_downsamples=1,
        )
        if self.vision_conv_cfg is not None:
            default_conv_cfg.update(self.vision_conv_cfg)
        self.conv = HighResConvNeXtV2Pyramid(
            **default_conv_cfg
        ).to(self.llm.dtype)
        self.pos_emb_2d = PositionalEmbedding2DSinusoidal(
            d_model=self.conv.dims[-1],
            scale_mode='learned',
            init_pe_scale=0.1
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

        # Enable training for new special tokens while allowing broader gradient flow
        if enable_regression or enable_survival:
            self._enable_selective_training()

    def _enable_selective_training(self):
        """Enable training for special tokens and task-specific components while preserving gradient flow."""
        new_token_ids = [tid for tid in [self.reg_token_id, self.srv_token_id] if tid is not None]
        if not new_token_ids:
            return

        # Consolidated gradient management for new special tokens
        self._configure_parameter_gradients()
        print_log("[SelectiveTraining] Configured gradient flow for special token learning", 'current')

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
        emb = self.llm.get_input_embeddings()
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

            if valid_ids:
                base_vec = emb.weight[valid_ids].mean(dim=0)
            else:
                base_vec = emb.weight.mean(dim=0)

            base_vec = 1.05 * base_vec + 1e-3 * torch.randn_like(base_vec)
            emb.weight[token_id] = base_vec

    def _init_prediction_modules(self) -> None:
        """Initialize prediction-specific modules."""
        llm_hidden = self.llm.config.hidden_size

        # Regression head using only special token embeddings
        if self.enable_regression:
            self.regression_head = RegressionHead(
                in_dim=llm_hidden, hidden_dim=llm_hidden
            ).to(self.llm.dtype)
            self.regression_loss_fn = nn.SmoothL1Loss(beta=1.0)

        # Survival head using only special token embeddings
        if self.enable_survival:
            from .custom_model import logistic_hazard_loss

            self.survival_head = SurvivalHead(
                in_dim=llm_hidden,
                hidden_dim=llm_hidden,
                time_intervals=(0, 1, 2, 3, 5, 7, 10),
                dropout=0.1,
            ).to(dtype=self.llm.dtype)
            self.survival_loss_fn = logistic_hazard_loss

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
            lora_config.target_modules = find_all_linear_names(self.llm)

        self.llm = get_peft_model(self.llm, lora_config)

    def _freeze_llm_with_exceptions(self) -> None:
        """Freeze LLM parameters while keeping task-critical components trainable."""
        if not (self.enable_regression or self.enable_survival):
            return
            
        self.llm.requires_grad_(False)
        print_log("Froze base LLM parameters", 'current')

    def _configure_parameter_gradients(self) -> None:
        """Centralized parameter gradient configuration for consistent multi-GPU behavior."""
        trainable_params = []
        
        # Enable embeddings for special token learning
        if hasattr(self.llm, 'get_input_embeddings'):
            embed_layer = self.llm.get_input_embeddings()
            if embed_layer is not None and hasattr(embed_layer, 'weight'):
                embed_layer.weight.requires_grad = True
                trainable_params.append(f"embeddings: {embed_layer.weight.numel():,}")

        # Enable lm_head for special token prediction
        if hasattr(self.llm, 'lm_head') and hasattr(self.llm.lm_head, 'weight'):
            self.llm.lm_head.weight.requires_grad = True
            trainable_params.append(f"lm_head: {self.llm.lm_head.weight.numel():,}")

        # Enable LoRA parameters if applicable
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
        proj_params = sum(p.numel() for p in self.projector.parameters())
        pos_params = sum(p.numel() for p in self.pos_emb_2d.parameters())
        trainable_params.extend([
            f"conv: {conv_params:,}",
            f"projector: {proj_params:,}", 
            f"pos_emb_2d: {pos_params:,}"
        ])

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
        self._last_hidden_state = None
        self._original_input_ids = None

    # ========== Core utilities ==========

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

        # Save prediction components + embeddings for new tokens
        if self.enable_regression or self.enable_survival:
            pred_keys = ['regression_head.', 'survival_head.']
            to_return.update({k: v for k, v in state_dict.items() 
                              if any(key in k for key in pred_keys)})
            embedding_keys = ['embed_tokens.weight', 'tok_embeddings.weight', 'lm_head.weight']
            for emb_key in embedding_keys:
                matching_keys = [k for k in state_dict.keys() if k.endswith(emb_key)]
                for k in matching_keys:
                    to_return[k] = state_dict[k]

        return to_return

    def _project_vision_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project vision features through conv, positional embedding, and final projector."""
        conv_input = features.to(self.llm.dtype)
        B, C, H, W = conv_input.shape

        stage_outputs = self.conv(conv_input)
        conv_output = stage_outputs[-1]

        conv_output = self.pos_emb_2d(conv_output)
        _, C_new, H_new, W_new = conv_output.shape
        feat_to_proj = conv_output.permute(0, 2, 3, 1).view(B, H_new * W_new, C_new)
        return self.projector(feat_to_proj.to(self.llm.dtype))

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

        # Extract targets and preserve original input_ids for predict-time reference
        regression_targets = data.pop('regression_targets', None)
        survival_targets = data.pop('survival_targets', None)
        self._original_input_ids = (data['input_ids'].clone()
                                    if (self.enable_regression or self.enable_survival) and 'input_ids' in data
                                    else None)

        # Process vision features
        projected_features = self._project_vision_features(data['features'])
        data['pixel_values'] = projected_features
        data.pop('features', None)

        if mode == 'predict':
            self._strip_assistant_targets(data)

        # Prepare multimodal inputs
        data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples, regression_targets, survival_targets)

        elif mode == 'predict':
            # Filter fields required by generation
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
        """Generate text and compute regression/survival predictions only if special tokens are generated."""
        try:
            # Save prefix to reconstruct the full sequence (vision + generated)
            prefix_inputs_embeds = data['inputs_embeds']  # (B, Lp, H)
            prefix_attention_mask = data['attention_mask']  # (B, Lp)
            prefix_position_ids = data.get('position_ids', None)
            B, Lp, _ = prefix_inputs_embeds.shape

            # 1) Text generation (depends on generated special tokens)
            with torch.no_grad():
                gen_out = self.llm.generate(
                    **data,
                    generation_config=self.generation_config,
                    stopping_criteria=self.stop_criteria,
                    bos_token_id=self.tokenizer.bos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
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

            first_step_logits = None
            if gen_scores is not None and isinstance(gen_scores, (list, tuple)) and len(gen_scores) > 0:
                first_step_logits = gen_scores[0]  # (B, vocab)

            for i, gen_id in enumerate(generate_ids):
                clean_text = self.tokenizer.decode(gen_id, skip_special_tokens=True).strip()
                data_samples[i]['prediction_text'] = clean_text

                # Attach choice logits for MCQA metrics (used by AUROC when K=2)
                if first_step_logits is not None and i < first_step_logits.size(0):
                    logits_row = first_step_logits[i]
                    choice_logits = {}
                    for letter in ['A', 'B', 'C', 'D', 'E']:
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
            if prefix_position_ids is not None:
                prefix_position_ids = prefix_position_ids.to(device)
                last_pos = prefix_position_ids[:, -1].unsqueeze(1)  # (B, 1)
                incr = torch.arange(1, Lg + 1, device=device).view(1, -1)  # (1, Lg)
                gen_pos = last_pos + incr  # (B, Lg)
                full_position_ids = torch.cat([prefix_position_ids, gen_pos], dim=1)  # (B, Lp+Lg)
            else:
                full_position_ids = None

            # Forward through base model to get last hidden state for the full sequence
            # Get hidden states from full sequence (vision + generated tokens)
            outputs = self.llm(inputs_embeds=full_inputs_embeds,
                               attention_mask=full_attention_mask,
                               position_ids=full_position_ids,
                               output_hidden_states=True,
                               return_dict=True)
            hidden = outputs.hidden_states[-1]  # (B, Lp+Lg, H)

        # For each batch, locate generated special-token positions and predict
        for b in range(B):
            # Regression
            if has_regression[b] and self.enable_regression and self.reg_token_id is not None:
                pos_in_gen = torch.nonzero(generate_ids[b] == self.reg_token_id, as_tuple=False).flatten()
                if pos_in_gen.numel() > 0:
                    pos_full = int(Lp + pos_in_gen[-1].item())
                    if pos_full < hidden.size(1):
                        embed = hidden[b, pos_full]  # (H,)
                        fused = self._fuse_with_vision(embed, b, 'regression')  # Shape depends on mode
                        pred = self.regression_head(fused).squeeze(-1).item()
                        data_samples[b]['regression_prediction'] = float(pred)
                        prev = data_samples[b].get('prediction_text', '')
                        data_samples[b]['prediction_text'] = f"{prev} [Regression: {pred:.4f}]".strip()

            # Survival
            if has_survival[b] and self.enable_survival and self.srv_token_id is not None:
                pos_in_gen = torch.nonzero(generate_ids[b] == self.srv_token_id, as_tuple=False).flatten()
                if pos_in_gen.numel() > 0:
                    pos_full = int(Lp + pos_in_gen[-1].item())
                    if pos_full < hidden.size(1):
                        embed = hidden[b, pos_full]  # (H,)
                        fused = self._fuse_with_vision(embed, b, 'survival')  # (1, 2H)
                        logits = self.survival_head(fused)
                        survival_probs = self.survival_head.predict_survival_probs(fused)
                        risk_score = float(self.survival_head.predict_risk_scores(fused).squeeze(0).item())
                        median_time = float(
                            self.survival_head.predict_median_survival_time(fused).squeeze(0).item()
                        )
                        time_intervals = self.survival_head.time_intervals.detach().cpu().float().tolist()
                        pred_dict = {
                            "logits": logits.squeeze(0).detach().cpu().float().tolist(),
                            "survival_probs": survival_probs.squeeze(0).detach().cpu().float().tolist(),
                            "risk_score": risk_score,
                            "median_survival_time": median_time,
                            "time_intervals": time_intervals,
                        }
                        data_samples[b]["survival_prediction"] = pred_dict
                        data_samples[b]["risk_score"] = risk_score
                        data_samples[b]["survival_probs"] = pred_dict["survival_probs"]
                        data_samples[b]["median_survival_time"] = median_time
                        prev = data_samples[b].get("prediction_text", "")
                        data_samples[b]["prediction_text"] = (
                            f"{prev} [Risk Score: {risk_score:.4f}, Median Survival: {median_time}]"
                        ).strip()
        return data_samples

    def _fuse_with_vision(self, token_embed: torch.Tensor, b: int, task_type: str = 'survival') -> torch.Tensor:
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
        input_kwargs = {k: data[k] for k in ['input_ids', 'inputs_embeds', 'attention_mask', 'position_ids'] if k in data}
        outputs = self.llm(**input_kwargs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]

        # Compute LM loss (causal shift)
        logits = self.llm.lm_head(last_hidden)
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
            task_targets = targets[b_idx].to(predictions.dtype)
            return self.regression_loss_fn(predictions, task_targets)
        else:  # survival
            logits = self.survival_head(task_embeds)
            target_y = targets['target_y'][b_idx].to(device=logits.device, dtype=logits.dtype)
            at_risk_mask = targets['at_risk_mask'][b_idx].to(device=logits.device, dtype=logits.dtype)
            return self.survival_loss_fn(logits, target_y, at_risk_mask)

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