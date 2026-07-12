# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper


class ModalityDropoutSchedulerHook(Hook):
    """Piecewise linear schedule for model modality dropout."""

    def __init__(self,
                 patch_start_dropout=0.0,
                 wsi_start_dropout=0.0,
                 begin_ratio=0.1,
                 end_ratio=0.5):
        if not 0 <= begin_ratio < end_ratio <= 1:
            raise ValueError('Require 0 <= begin_ratio < end_ratio <= 1.')
        self.patch_start_dropout = self._check_dropout(
            'patch_start_dropout', patch_start_dropout)
        self.wsi_start_dropout = self._check_dropout(
            'wsi_start_dropout', wsi_start_dropout)
        self.begin_ratio = float(begin_ratio)
        self.end_ratio = float(end_ratio)
        self.begin_iter = None
        self.end_iter = None
        self.patch_final_dropout = None
        self.wsi_final_dropout = None

    @staticmethod
    def _check_dropout(name, value):
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0, 1], got {value}.')
        return value

    @staticmethod
    def _unwrap_model(model):
        return model.module if is_model_wrapper(model) else model

    def before_train(self, runner) -> None:
        model = self._unwrap_model(runner.model)
        self.patch_final_dropout = self._check_dropout(
            'patch_modality_dropout', model.patch_modality_dropout)
        self.wsi_final_dropout = self._check_dropout(
            'wsi_modality_dropout', model.wsi_modality_dropout)
        max_iters = runner.train_loop.max_iters
        self.begin_iter = int(round(max_iters * self.begin_ratio))
        self.end_iter = max(
            self.begin_iter + 1, int(round(max_iters * self.end_ratio)))
        self._apply(runner, runner.iter)
        runner.logger.info(
            'Modality dropout schedule: '
            f'patch {self.patch_start_dropout}->{self.patch_final_dropout}, '
            f'wsi {self.wsi_start_dropout}->{self.wsi_final_dropout}, '
            f'begin_iter={self.begin_iter}, end_iter={self.end_iter}.')

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        self._apply(runner, runner.iter)

    def _apply(self, runner, cur_iter):
        scale = self._scale(cur_iter)
        model = self._unwrap_model(runner.model)
        model.patch_modality_dropout = self._lerp(
            self.patch_start_dropout, self.patch_final_dropout, scale)
        model.wsi_modality_dropout = self._lerp(
            self.wsi_start_dropout, self.wsi_final_dropout, scale)

    def _scale(self, cur_iter):
        if cur_iter <= self.begin_iter:
            return 0.0
        if cur_iter >= self.end_iter:
            return 1.0
        return (cur_iter - self.begin_iter) / (self.end_iter - self.begin_iter)

    @staticmethod
    def _lerp(start, end, scale):
        return float(start + (end - start) * scale)
