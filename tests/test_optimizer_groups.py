import importlib.util
from pathlib import Path
import unittest

from torch import nn

_MODULE_PATH = Path(__file__).parents[1] / 'xtuner/engine/optimizers.py'
_SPEC = importlib.util.spec_from_file_location(
    'optimizer_groups_under_test', _MODULE_PATH)
_OPTIMIZERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OPTIMIZERS)
TwoGroupOptimWrapperConstructor = (
    _OPTIMIZERS.TwoGroupOptimWrapperConstructor)


class _ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(2, 2)
        self.patch_resampler = nn.Linear(2, 2)
        self.llm = nn.Module()
        self.llm.family_lora = nn.Linear(2, 2)


def _constructor(*, allow_empty_groups=False):
    return TwoGroupOptimWrapperConstructor(
        optim_wrapper_cfg=dict(
            type='OptimWrapper',
            optimizer=dict(type='AdamW', lr=1e-3),
        ),
        paramwise_cfg=dict(
            vision_prefixes=('patch_resampler.', 'wsi_projector.'),
            vision_lr_mult=0.1,
            family_lora_lr_mult=2.0,
            allow_empty_groups=allow_empty_groups,
        ),
    )


class TestTwoGroupOptimWrapperConstructor(unittest.TestCase):

    def test_family_only_builds_one_non_empty_group(self):
        model = _ToyModel()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('.family_lora.' in name)

        groups = []
        _constructor(allow_empty_groups=True).add_params(groups, model)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['lr'], 2e-3)
        self.assertTrue(groups[0]['params'])
        self.assertTrue(all(
            parameter.requires_grad for parameter in groups[0]['params']))

    def test_strict_mode_preserves_non_empty_group_check(self):
        model = _ToyModel()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('.family_lora.' in name)

        with self.assertRaisesRegex(RuntimeError, 'adapter/head and vision'):
            _constructor().add_params([], model)

    def test_allow_empty_groups_still_rejects_no_trainable_parameters(self):
        model = _ToyModel().requires_grad_(False)

        with self.assertRaisesRegex(RuntimeError, 'no trainable parameters'):
            _constructor(allow_empty_groups=True).add_params([], model)


if __name__ == '__main__':
    unittest.main()
