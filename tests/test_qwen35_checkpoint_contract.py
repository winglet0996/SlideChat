"""Regression tests for sparse alignment and routed-LoRA checkpoints."""

import unittest

import torch
from torch import nn

from xtuner.model.llava_qwen3_5 import LLaVAModel_qwen3_5


class _Adapter(nn.Module):

    def __init__(self):
        super().__init__()
        self.lora_A = nn.Linear(2, 2, bias=False)
        self.lora_B = nn.Linear(2, 2, bias=False)


class _TinyLLM(nn.Module):

    def __init__(self, routed: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(2, 2, bias=False)
        self.model.language_model = nn.Module()
        self.model.language_model.block = nn.Module()
        self.model.language_model.block.base = nn.Linear(2, 2, bias=False)
        if routed:
            self.model.language_model.block.shared_lora = _Adapter()
            self.model.language_model.block.family_lora = nn.ModuleDict({
                'family_a': _Adapter(),
                'family_b': _Adapter(),
            })

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None


class _CheckpointHarness(LLaVAModel_qwen3_5):
    """A construction-free model exposing only the checkpoint contract."""

    def __init__(self, *, routed: bool, freeze_llm: bool,
                 trainable_mode: str = 'all'):
        nn.Module.__init__(self)
        self.freeze_llm = freeze_llm
        self.routed_lora_enabled = routed
        self.use_llm_lora = routed
        self.routed_lora_trainable = trainable_mode
        self.enable_regression = False
        self.enable_survival = False
        self.reg_token_id = None
        self.srv_token_id = None
        self.llm = _TinyLLM(routed)
        self.patch_resampler = nn.Linear(2, 2, bias=False)
        self.wsi_projector = nn.Linear(2, 2, bias=False)
        self.regression_head = nn.Linear(2, 1, bias=False)
        self.survival_head = nn.Linear(2, 1, bias=False)
        self.special_lm_head = nn.Linear(2, 2, bias=False)


class TestQwen35CheckpointContract(unittest.TestCase):

    def test_alignment_sparse_checkpoint_loads_with_strict_true(self):
        source = _CheckpointHarness(routed=False, freeze_llm=True)
        checkpoint = source.state_dict()
        self.assertNotIn('llm.model.visual.weight', checkpoint)
        self.assertIn(source._CHECKPOINT_FORMAT_KEY, checkpoint)

        target = _CheckpointHarness(routed=False, freeze_llm=True)
        target.load_state_dict(checkpoint, strict=True)
        torch.testing.assert_close(
            target.patch_resampler.weight, source.patch_resampler.weight)

    def test_routed_lora_all_round_trip_requires_every_route(self):
        source = _CheckpointHarness(routed=True, freeze_llm=True,
                                    trainable_mode='all')
        checkpoint = source.state_dict()
        self.assertTrue(any(key.startswith('routed_lora.shared.')
                            for key in checkpoint))
        self.assertTrue(any(key.startswith('routed_lora.family.family_a.')
                            for key in checkpoint))

        target = _CheckpointHarness(routed=True, freeze_llm=True,
                                    trainable_mode='all')
        target.load_state_dict(checkpoint, strict=True)
        torch.testing.assert_close(
            target.llm.model.language_model.block.shared_lora.lora_A.weight,
            source.llm.model.language_model.block.shared_lora.lora_A.weight)

        truncated = checkpoint.copy()
        family_key = next(key for key in truncated
                          if key.startswith('routed_lora.family.family_a.'))
        truncated.pop(family_key)
        with self.assertRaisesRegex(RuntimeError, 'missing required keys'):
            target.load_state_dict(truncated, strict=True)

    def test_family_only_saves_and_loads_shared_and_all_family_routes(self):
        source = _CheckpointHarness(routed=True, freeze_llm=True,
                                    trainable_mode='family_only')
        checkpoint = source.state_dict()
        self.assertTrue(any(key.startswith('routed_lora.shared.')
                            for key in checkpoint))
        self.assertTrue(any(key.startswith('routed_lora.family.family_b.')
                            for key in checkpoint))

        # Evaluation can build the same routed architecture with ``all``;
        # trainability is intentionally not part of checkpoint compatibility.
        target = _CheckpointHarness(routed=True, freeze_llm=True,
                                    trainable_mode='all')
        target.load_state_dict(checkpoint, strict=True)
        torch.testing.assert_close(
            target.llm.model.language_model.block.family_lora['family_b']
            .lora_B.weight,
            source.llm.model.language_model.block.family_lora['family_b']
            .lora_B.weight)

    def test_alignment_is_an_explicit_warm_start_for_routed_lora(self):
        alignment = _CheckpointHarness(routed=False, freeze_llm=True)
        routed = _CheckpointHarness(routed=True, freeze_llm=True)
        routed.load_state_dict(alignment.state_dict(), strict=True)
        torch.testing.assert_close(
            routed.patch_resampler.weight, alignment.patch_resampler.weight)

    def test_full_parameter_checkpoint_keeps_real_strictness(self):
        source = _CheckpointHarness(routed=False, freeze_llm=False)
        checkpoint = source.state_dict()
        target = _CheckpointHarness(routed=False, freeze_llm=False)
        target.load_state_dict(checkpoint, strict=True)

        checkpoint.pop('llm.model.visual.weight')
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            target.load_state_dict(checkpoint, strict=True)


if __name__ == '__main__':
    unittest.main()
