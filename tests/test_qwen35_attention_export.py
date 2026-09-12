"""Regression tests for decision-state attention export."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch import nn
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.model.llava_qwen3_5 import (
    LLaVAModel_qwen3_5,
    prepare_inputs_labels_for_qwen3_5,
)


class _FakeLLM(nn.Module):

    def __init__(self, hidden_size=4, vocab_size=32, num_layers=2, num_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(
            _attn_implementation='sdpa',
            pad_token_id=0,
            eos_token_id=0,
        )
        self.num_layers = num_layers
        self.num_heads = num_heads

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, inputs_embeds=None, logits_processor=None, **kwargs):
        del kwargs
        batch_size = inputs_embeds.size(0)
        scores = torch.zeros(
            batch_size,
            self.embedding.num_embeddings,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=inputs_embeds.device)
        for processor in logits_processor or []:
            scores = processor(input_ids, scores)
        return SimpleNamespace(
            sequences=torch.full(
                (batch_size, 1), 9, dtype=torch.long, device=inputs_embeds.device),
            scores=(scores,),
        )

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
        output_attentions=False,
        **kwargs,
    ):
        del attention_mask, position_ids, kwargs
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        hidden = torch.zeros_like(inputs_embeds)
        hidden[:, :, 0] = torch.arange(
            seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        hidden_states = (hidden,) if output_hidden_states else None

        attentions = None
        if output_attentions:
            attentions = []
            for layer_idx in range(self.num_layers):
                layer = torch.zeros(
                    batch_size,
                    self.num_heads,
                    seq_len,
                    seq_len,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                for head_idx in range(self.num_heads):
                    values = float(layer_idx * 100 + head_idx * 10)
                    row_values = values + torch.arange(
                        seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    layer[:, head_idx] = row_values.view(1, seq_len, 1)
                attentions.append(layer)

        return SimpleNamespace(hidden_states=hidden_states, attentions=attentions)


class _FakeSurvivalHead(nn.Module):

    def __init__(self, hidden_size=4):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(hidden_size))

    def predict_survival_probs(self, embed):
        return embed.new_zeros((embed.size(0), 2))

    def predict_risk_scores(self, embed):
        return embed[:, :1]

    def predict_median_survival_time(self, embed):
        return embed[:, :1]


class _FakeTokenizer:

    bos_token_id = None
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, value, add_special_tokens=False):
        del add_special_tokens
        if value in ('A', ' A'):
            return [1]
        return []

    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return 'A'

def _make_harness(*, gen_forcing=False, regression=True, survival=False):
    model = object.__new__(LLaVAModel_qwen3_5)
    nn.Module.__init__(model)
    model.llm = _FakeLLM()
    model.tokenizer = _FakeTokenizer()
    model.generation_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=0,
    )
    model.stop_criteria = StoppingCriteriaList()
    model.save_patch_attention_h5 = True
    model.patch_attention_h5_dir = '/tmp'
    model.patch_attention_h5_dtype = 'float32'
    model.enable_regression = regression
    model.enable_survival = survival
    model.reg_token_id = 7 if regression else None
    model.srv_token_id = 8 if survival else None
    model.survival_method = 'discrete'
    model.gen_forcing = gen_forcing
    model.regression_head = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.regression_head.weight.zero_()
        model.regression_head.weight[0, 0] = 1
    model.survival_head = _FakeSurvivalHead() if survival else None
    return model


class Qwen35AttentionExportTest(unittest.TestCase):

    def test_mcqa_collects_the_last_prefix_decision_row(self):
        model = _make_harness(regression=False)
        visual_attention, wsi_attention = model._collect_decision_attentions({
            'inputs_embeds': torch.zeros(1, 5, 4),
            'attention_mask': torch.tensor([[False, True, True, True, True]]),
            'vision_token_spans': torch.tensor([[[1, 3]]]),
            'wsi_token_spans': torch.tensor([[[3, 5]]]),
        })

        self.assertEqual(visual_attention.shape, (1, 2, 2, 1, 2))
        self.assertEqual(visual_attention[0, 0, 0, 0].tolist(), [4.0, 4.0])
        self.assertEqual(visual_attention[0, 1, 1, 0].tolist(), [114.0, 114.0])
        self.assertEqual(wsi_attention.shape, (1, 2, 2, 1, 2))
        self.assertEqual(wsi_attention[0, 0, 0, 0].tolist(), [4.0, 4.0])
        self.assertEqual(model.llm.config._attn_implementation, 'sdpa')

    def test_regression_collects_the_appended_reg_token_row(self):
        model = _make_harness(regression=True)
        samples, attention, wsi_attention = model._predict_tasks_from_prefix_tokens(
            data_samples=[{}],
            has_regression=[True],
            has_survival=[False],
            prefix_inputs_embeds=torch.zeros(1, 5, 4),
            prefix_attention_mask=torch.tensor([[False, True, True, True, True]]),
            vision_token_spans=torch.tensor([[[1, 3]]]),
            wsi_token_spans=torch.tensor([[[3, 5]]]),
            collect_visual_attention=True,
        )

        self.assertEqual(samples[0]['regression_prediction'], 4.0)
        self.assertEqual(attention.shape, (1, 2, 2, 1, 2))
        self.assertEqual(attention[0, 0, 0, 0].tolist(), [4.0, 4.0])
        self.assertEqual(wsi_attention.shape, (1, 2, 2, 1, 2))
        self.assertEqual(wsi_attention[0, 0, 0, 0].tolist(), [4.0, 4.0])

    def test_survival_collects_the_appended_srv_token_row(self):
        model = _make_harness(regression=False, survival=True)
        samples, attention, wsi_attention = model._predict_tasks_from_prefix_tokens(
            data_samples=[{}],
            has_regression=[False],
            has_survival=[True],
            prefix_inputs_embeds=torch.zeros(1, 5, 4),
            prefix_attention_mask=torch.tensor([[False, True, True, True, True]]),
            vision_token_spans=torch.tensor([[[1, 3]]]),
            wsi_token_spans=torch.tensor([[[3, 5]]]),
            collect_visual_attention=True,
        )

        self.assertIn('survival_prediction', samples[0])
        self.assertEqual(float(samples[0]['risk_score']), 4.0)
        self.assertEqual(attention[0, 0, 0, 0].tolist(), [4.0, 4.0])
        self.assertEqual(wsi_attention.shape, (1, 2, 2, 1, 2))

    def test_generated_task_token_row_is_used_when_forcing_generation(self):
        model = _make_harness(gen_forcing=True)
        samples, attention, wsi_attention = model._predict_tasks_from_generation(
            generate_ids=torch.tensor([[7, 9]]),
            data_samples=[{}],
            has_regression=[True],
            has_survival=[False],
            prefix_inputs_embeds=torch.zeros(1, 4, 4),
            prefix_attention_mask=torch.tensor([[False, True, True, True]]),
            vision_token_spans=torch.tensor([[[1, 3]]]),
            wsi_token_spans=torch.tensor([[[3, 4]]]),
            collect_visual_attention=True,
        )

        self.assertEqual(samples[0]['regression_prediction'], 4.0)
        self.assertEqual(attention[0, 0, 0, 0].tolist(), [4.0, 4.0])
        self.assertEqual(wsi_attention.shape, (1, 2, 2, 1, 1))

    def test_predict_passes_task_decision_attention_to_h5_writer(self):
        model = _make_harness(regression=True)
        captured = {}
        model._save_patch_attention_h5_files = (
            lambda payload, attention, samples, wsi_attention: captured.update(
                payload=payload, attention=attention, samples=samples,
                wsi_attention=wsi_attention))

        samples = model.predict(
            data={
                'inputs_embeds': torch.zeros(1, 5, 4),
                'attention_mask': torch.tensor([[False, True, True, True, True]]),
                'vision_token_spans': torch.tensor([[[1, 3]]]),
                'wsi_token_spans': torch.tensor([[[3, 5]]]),
                '_patch_attention_save_payload': {'present': True},
            },
            data_samples=[{}],
            regression_targets=torch.tensor([1.0]),
            task_categories=['regression::test'],
        )

        self.assertEqual(samples[0]['regression_prediction'], 4.0)
        self.assertEqual(captured['payload'], {'present': True})
        self.assertEqual(captured['attention'][0, 0, 0, 0].tolist(), [4.0, 4.0])
        self.assertEqual(captured['wsi_attention'][0, 0, 0, 0].tolist(), [4.0, 4.0])

    def test_input_composer_records_left_padded_wsi_span(self):
        model = _make_harness(regression=False)
        result = prepare_inputs_labels_for_qwen3_5(
            llm=model.llm,
            input_ids=torch.tensor([[1, -200, 2]]),
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            pixel_values=torch.zeros(1, 2, 4),
            image_batch_indices=torch.tensor([0]),
            vision_token_positions=torch.zeros(1, 2, 2, dtype=torch.long),
            vision_token_valid=torch.ones(1, 2, dtype=torch.bool),
            wsi_embeddings=torch.zeros(1, 4, 4),
            vision_start_token_id=10,
            vision_end_token_id=11,
            padding_side='left',
        )

        self.assertEqual(result['wsi_token_spans'].tolist(), [[[1, 5]]])
        self.assertEqual(result['vision_token_spans'].tolist(), [[[6, 8]]])

    def test_h5_schema_and_source_mapping(self):
        model = _make_harness(regression=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / 'source.h5'
            with h5py.File(source_path, 'w') as source:
                source.create_dataset('features', data=np.zeros((2, 1), dtype=np.float32))
                coords = source.create_dataset('coords', data=np.asarray([[0, 0], [1, 1]]))
                coords.attrs['patch_size_level0'] = 1

            model.patch_attention_h5_dir = tmp_dir
            payload = {
                'region_attention_heads': torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4),
                'patch_valid_mask': torch.tensor([[[True, False], [False, True]]]),
                'token_positions': torch.tensor([[[0, 0], [1, 1]]]),
                'vision_token_valid': torch.tensor([[True, True]]),
                'image_batch_indices': [0],
                'feature_shapes': [(2, 2)],
                'feature_paths': [str(source_path)],
                'sample_ids': ['sample'],
                'categories': ['mcqa::test'],
                'projects': ['project'],
                'divisions': ['test'],
                'labels_text': ['A'],
                'raw_sample_json': ['{}'],
                'wsi_feature_paths': [['titan', 'prism', 'gigapath', 'chief']],
            }
            model._save_patch_attention_h5_files(
                payload,
                np.ones((1, 2, 2, 1, 2), dtype=np.float32),
                data_samples=[{'prediction_text': 'A'}],
                llm_wsi_attention=np.arange(16, dtype=np.float32).reshape(1, 2, 2, 1, 4),
            )

            files = list(Path(tmp_dir).rglob('*.attn.h5'))
            self.assertEqual(len(files), 1)
            self.assertFalse(list(Path(tmp_dir).rglob('*.npz')))
            self.assertFalse(list(Path(tmp_dir).rglob('*.png')))
            with h5py.File(files[0], 'r') as output:
                self.assertEqual(output.attrs['schema_version'], 'patch_resampler_attention.v2')
                self.assertEqual(
                    set(output['attention'].keys()),
                    {'resampler_cross_attn', 'next_token_source_attn',
                     'wsi_source_attn', 'token_positions', 'token_valid'},
                )
                np.testing.assert_array_equal(
                    output['patch_ref/valid_flat_indices'][:], [0, 3])
                np.testing.assert_array_equal(
                    output['patch_ref/source_patch_indices'][:], [0, 1])
                self.assertEqual(
                    output['attention/next_token_source_attn'].shape,
                    (2, 2, 2),
                )
                self.assertEqual(
                    output['attention/wsi_source_attn'].shape,
                    (2, 2, 4),
                )
                np.testing.assert_array_equal(
                    output['attention/wsi_source_attn'][0, 0], [0, 1, 2, 3])

    def test_source_mapping_reproduces_sparse_center_fallback(self):
        model = _make_harness(regression=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / 'sparse.h5'
            with h5py.File(source_path, 'w') as source:
                source.create_dataset(
                    'features',
                    data=np.asarray([[1.0], [2.0]], dtype=np.float32),
                )
                coords = source.create_dataset(
                    'coords',
                    data=np.asarray([[0, 0], [9, 9]], dtype=np.int64),
                )
                coords.attrs['patch_size_level0'] = 1

            mapped = model._source_patch_index_grid(str(source_path), (4, 4))
            self.assertEqual(mapped.reshape(4, 4)[0, 0], 0)
            self.assertEqual(mapped.reshape(4, 4)[3, 3], -1)


if __name__ == '__main__':
    unittest.main()
