# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, CLIPImageProcessor,
                          CLIPVisionModel, LlamaForCausalLM,
                          LlamaTokenizerFast, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor,
                          GenerationConfig, StoppingCriteriaList)
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, StopWordStoppingCriteria)
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)
from .custom_model import HighResPartialConvNeXt, RotaryEmbedding2D

from .torchscale.model.LongNet import make_longnet_from_name
import torch.nn.functional as F


def convert_state_dict_to_hf(state_dict, mapping):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.inv_freq'):
            continue
        for key_to_modify, new_key in mapping.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        new_state_dict[key] = value
    return new_state_dict



class LLaVAModel_conv_longnet(BaseModel):
    def __init__(self,
                 llm,
                 tokenizer,
                 freeze_llm=True, # also for lora
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 hidden_size=None,
                 generation_kwargs=None,
                 stop_words=None):
        super().__init__()

        self.freeze_llm = freeze_llm

        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)
        
        # High-resolution partial convolution for feature preprocessing
        self.conv = HighResPartialConvNeXt().to(self.llm.dtype)
        
        # 2D rotary embedding for positional encoding (apply to half dimensions)
        self.rotary_emb = RotaryEmbedding2D(dim=hidden_size//2).to(self.llm.dtype)

        self.encoder_name = "LongNet_{}_layers_{}_dim".format(2, hidden_size)
        self.LongNet_encoder = make_longnet_from_name(self.encoder_name,
                                                      checkpoint_activations=use_activation_checkpointing) # , drop_path_rate=0.3, dropout=0.3, segment_length=1024
        self.LongNet_encoder = self.LongNet_encoder.to(self.llm.dtype)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        self.projector_depth = projector_depth

        projector_config = ProjectorConfig(
            visual_hidden_size=hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=self.projector_depth)        

        self.projector = ProjectorModel(projector_config).to(
            self.llm.dtype)
        
        if self.freeze_llm:
            print('freeze_llm')
            self.llm.requires_grad_(False)
        
        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
                
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}',
                      'current')

        self.visual_select_layer = visual_select_layer

        self.tokenizer = BUILDER.build(tokenizer)
        if generation_kwargs:
            self.generation_config = GenerationConfig(**generation_kwargs)
        else:
            self.generation_config = GenerationConfig()
        self.stop_criteria = StoppingCriteriaList()
        if stop_words:
            for word in stop_words:
                self.stop_criteria.append(
                    StopWordStoppingCriteria(self.tokenizer, word))
        
        self._is_init = True

        self.is_first_iter = True

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. Vision Encoder: none
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        
        # Step 4. Conv and RotaryEmb
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'conv.' in k or 'rotary_emb.' in k})
        
        # Step 5. LongNet_encoder
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'LongNet_encoder.' in k})
        return to_return

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        # hardcode for internlm2
        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig',
                             'Starcoder2Config', 'Starcoder2Config',
                             'Phi3Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Qwen2MoeConfig', 'Starcoder2Config',
                               'Starcoder2Config', 'Phi3Config')

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        if getattr(cfg, 'attn_implementation', None) is not None:
            # Flash Attention 2.0 only supports torch.float16 and
            # torch.bfloat16 dtypes
            if cfg.attn_implementation == 'flash_attention_2':
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(
                cfg, 'quantization_config')):
            return cfg

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def _project_vision_features(self, features, masks=None):
        """Projects vision features through HighResPartialConvNeXt, 
        RotaryEmbedding2D, and LongNet encoder before final projection."""
        
        # features: (B, C, H, W), masks: (B, 1, H, W) or None
        conv_input = features.to(self.llm.dtype)  # Ensure correct dtype
        B, C, H, W = conv_input.shape
        
        # Process masks
        if masks is None:
            # Create default mask (all valid)
            mask = torch.ones(B, 1, H, W, device=conv_input.device, dtype=conv_input.dtype)
        else:
            mask = masks.to(conv_input.device, dtype=conv_input.dtype)
        
        # Pass through HighResPartialConvNeXt
        conv_output, updated_mask = self.conv(conv_input, mask)
        
        # Apply 2D rotary positional embedding
        conv_output = self.rotary_emb(conv_output)
        
        # Reshape back to sequence format for LongNet
        _, C_new, H_new, W_new = conv_output.shape
        feat_to_proj = conv_output.permute(0, 2, 3, 1).view(B, H_new * W_new, C_new)
        feat_to_proj = feat_to_proj.to(self.llm.dtype)
        
        # Pass through LongNet encoder
        long_net_output = self.LongNet_encoder(
            src_tokens=None, token_embeddings=feat_to_proj.permute(1, 0, 2)
        )["encoder_out"]
        feat_to_proj = long_net_output.permute(1, 0, 2)
        
        # Final projection
        projected_features = self.projector(feat_to_proj.to(self.llm.dtype))
        return projected_features

    def forward(self, data, data_samples=None, mode='loss'):
        if self.is_first_iter:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            self.to(data['input_ids'].device)
            self.is_first_iter = False
        
        # features (B, C, H, W) and masks (B, 1, H, W)
        projected_features = self._project_vision_features(data['features'], data['masks'])
        # Replace with projected features, keep original key for compatibility
        data['pixel_values'] = projected_features
        # Clean up original keys
        data.pop('features', None)
        data.pop('masks', None)
        data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):
        outputs = self.llm(**data)
        return outputs

    def predict(self, data, data_samples=None):
        
        generate_ids = self.llm.generate(
            **data,
            generation_config=self.generation_config,
            stopping_criteria=self.stop_criteria,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        
        # Decode the output.
        if data_samples is None:
            data_samples = [{} for _ in range(len(generate_ids))]

        for i in range(len(generate_ids)):
            generation_output = self.tokenizer.decode(
                generate_ids[i], 
                skip_special_tokens=True
            )
            generated_text = generation_output.strip()
            data_samples[i]['prediction_text'] = generated_text
            
        return data_samples

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)