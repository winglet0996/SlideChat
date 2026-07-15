# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TwoGroupOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Build exactly one adapter/head group and one vision-alignment group."""

    def add_params(self, params: List[dict], module, **kwargs) -> None:
        cfg = self.paramwise_cfg or {}
        vision_prefixes = tuple(cfg.get(
            'vision_prefixes', ('patch_resampler.', 'wsi_projector.')))
        vision_lr_mult = float(cfg.get('vision_lr_mult', 0.1))

        adapter_and_head = []
        vision_alignment = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            target = (
                vision_alignment
                if name.startswith(vision_prefixes)
                else adapter_and_head
            )
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
