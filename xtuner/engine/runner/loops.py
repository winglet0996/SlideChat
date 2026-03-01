# Copyright (c) OpenMMLab. All rights reserved.
import logging
import math
import time
from typing import Any, Dict, Optional, Sequence, Union

from mmengine.logging import print_log
from mmengine.runner import IterBasedTrainLoop
from torch.utils.data import DataLoader


class _InfiniteDataloaderIteratorSkip:

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iterator: Any = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Sequence[dict]:
        return self._next_data()

    def skip_iter(self, iter: int) -> None:
        for _ in range(iter):
            self._next_data(skip_loading=True)

    def _next_data(self, skip_loading: bool = False) -> Any:
        data = None
        try:
            if skip_loading and hasattr(self._iterator, '_next_index'):
                self._iterator._next_index()
            elif skip_loading:
                next(self._iterator)
            else:
                data = next(self._iterator)
        except StopIteration:
            print_log(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.',
                logger='current',
                level=logging.WARNING)
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)

            bypass_mypy_checking: Any = iter(self._dataloader)
            self._iterator = bypass_mypy_checking

            if skip_loading and hasattr(bypass_mypy_checking, '_next_index'):
                bypass_mypy_checking._next_index()
            elif skip_loading:
                next(self._iterator)
            else:
                data = next(self._iterator)
        return data


class TrainLoop(IterBasedTrainLoop):

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: Optional[int] = None,
                 max_epochs: Union[int, float] = None,
                 **kwargs) -> None:

        if max_iters is None and max_epochs is None:
            raise RuntimeError('Please specify the `max_iters` or '
                               '`max_epochs` in `train_cfg`.')
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError('Only one of `max_iters` or `max_epochs` can '
                               'exist in `train_cfg`.')
        else:
            if max_iters is not None:
                iters = int(max_iters)
                assert iters == max_iters, ('`max_iters` should be a integer '
                                            f'number, but get {max_iters}')
            elif max_epochs is not None:
                if isinstance(dataloader, dict):
                    diff_rank_seed = runner._randomness_cfg.get(
                        'diff_rank_seed', False)
                    dataloader = runner.build_dataloader(
                        dataloader,
                        seed=runner.seed,
                        diff_rank_seed=diff_rank_seed)
                iters = max_epochs * len(dataloader)
            else:
                raise NotImplementedError

        self._configured_max_epochs = int(max_epochs) if max_epochs is not None else None
        super().__init__(
            runner=runner, dataloader=dataloader, max_iters=iters, **kwargs)

        self.dataloader_iterator = _InfiniteDataloaderIteratorSkip(self.dataloader)

        self._iters_per_epoch = len(self.dataloader)
        if self._configured_max_epochs is not None:
            self._max_epochs = self._configured_max_epochs
        else:
            self._max_epochs = max(1, math.ceil(self._max_iters / self._iters_per_epoch))

    def run(self):
        self.runner.call_hook('before_train')

        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            self.dataloader_iterator.skip_iter(self._iter)

        self._epoch = self._iter // self._iters_per_epoch
        self.runner.call_hook('before_train_epoch')

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()
            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

            reached_epoch_end = (self._iter % self._iters_per_epoch == 0)
            reached_train_end = (self._iter == self._max_iters)
            if reached_epoch_end or reached_train_end:
                self.runner.call_hook('after_train_epoch')
                self._epoch += 1
                if self._iter < self._max_iters:
                    self.runner.call_hook('before_train_epoch')

        self.runner.call_hook('after_train')
        return self.runner.model
