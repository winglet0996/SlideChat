# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TwoGroupOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Build adapter/head and vision groups, with optional family-LoRA LR."""

    def add_params(self, params: List[dict], module, **kwargs) -> None:
        cfg = self.paramwise_cfg or {}
        vision_prefixes = tuple(cfg.get(
            'vision_prefixes', ('patch_resampler.', 'wsi_projector.')))
        vision_lr_mult = float(cfg.get('vision_lr_mult', 0.1))
        family_lora_lr_mult = cfg.get('family_lora_lr_mult')
        if family_lora_lr_mult is not None:
            family_lora_lr_mult = float(family_lora_lr_mult)

        adapter_and_head = []
        vision_alignment = []
        family_lora = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            is_family_lora = (
                name.startswith('family_lora.')
                or '.family_lora.' in name
            )
            if family_lora_lr_mult is not None and is_family_lora:
                target = family_lora
            elif name.startswith(vision_prefixes):
                target = vision_alignment
            else:
                target = adapter_and_head
            target.append(parameter)

        if not adapter_and_head or not vision_alignment:
            raise RuntimeError(
                'TwoGroupOptimWrapperConstructor requires non-empty adapter/head '
                f'and vision groups, got {len(adapter_and_head)} and '
                f'{len(vision_alignment)} parameters.')
        params.extend([
            dict(params=adapter_and_head, lr=self.base_lr),
            dict(params=vision_alignment, lr=self.base_lr * vision_lr_mult),
        ])
        if family_lora_lr_mult is not None:
            if not family_lora:
                raise RuntimeError(
                    'family_lora_lr_mult was configured but no family LoRA '
                    'parameters were found.')
            params.append(dict(
                params=family_lora,
                lr=self.base_lr * family_lora_lr_mult,
            ))
