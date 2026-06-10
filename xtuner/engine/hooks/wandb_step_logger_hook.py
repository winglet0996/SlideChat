from typing import Dict, Optional

from mmengine.hooks import LoggerHook


class WandbStepLoggerHook(LoggerHook):
    """Align iter-based validation logging with train-step semantics.

    MMEngine logs train scalars at runner.iter + 1 in after_train_iter,
    but logs validation scalars at runner.iter in after_val_epoch.
    When a wandb backend uses explicit step=..., that causes validation
    records to be dropped as out-of-order right after a train log.
    """

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), "val")
        runner.logger.info(log_str)
        if self.log_metric_by_epoch:
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(
                tag, step=epoch, file_path=self.json_log_path)
        else:
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                step = 0
            else:
                step = runner.iter + 1
            runner.visualizer.add_scalars(
                tag, step=step, file_path=self.json_log_path)
