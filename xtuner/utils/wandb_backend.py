import math
from typing import Optional, Union

import numpy as np
import torch
from mmengine.registry import VISBACKENDS
from mmengine.visualization import WandbVisBackend


@VISBACKENDS.register_module()
class ResumeWandbVisBackend(WandbVisBackend):
    """Wandb backend that preserves the runner step on resume."""

    def __init__(self, *args, step_divisor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if step_divisor <= 0:
            raise ValueError("step_divisor must be positive")
        self.step_divisor = float(step_divisor)

    def _scale_step(self, step: int) -> int:
        if step <= 0:
            return 0
        # Preserve monotonicity after down-scaling. For example with a divisor
        # of 10, train step 1500 should map to 150 while the following val step
        # 1501 must advance to 151 instead of collapsing back to 150.
        return math.ceil(step / self.step_divisor)

    def _init_env(self):
        if not hasattr(self, '_init_kwargs') or self._init_kwargs is None:
            self._init_kwargs = {}
        self._init_kwargs.setdefault('resume', 'allow')
        super()._init_env()

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        image = self._wandb.Image(image)
        self._wandb.log(
            {name: image}, step=self._scale_step(step), commit=self._commit)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        self._wandb.log(
            {name: value}, step=self._scale_step(step), commit=self._commit)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        self._wandb.log(
            scalar_dict, step=self._scale_step(step), commit=self._commit)
