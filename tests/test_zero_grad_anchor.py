import importlib.util
from pathlib import Path
import unittest

import torch

_MODULE_PATH = Path(__file__).parents[1] / 'xtuner/model/custom_model.py'
_SPEC = importlib.util.spec_from_file_location(
    'zero_grad_anchor_custom_model_under_test', _MODULE_PATH)
_CUSTOM_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CUSTOM_MODEL)
zero_grad_anchor = _CUSTOM_MODEL.zero_grad_anchor


class ZeroGradAnchorTest(unittest.TestCase):

    def test_materializes_zero_gradients_for_upstream_parameters(self):
        layer = torch.nn.Linear(5, 3)
        output = layer(torch.randn(4, 5))

        anchor = zero_grad_anchor(output)
        self.assertEqual(anchor.shape, torch.Size([]))
        self.assertEqual(anchor.item(), 0.0)
        anchor.backward()

        for name, parameter in layer.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad,
                torch.zeros_like(parameter.grad),
                rtol=0,
                atol=0,
            )

    def test_does_not_read_nonfinite_values(self):
        value = torch.tensor(
            [float('nan'), float('inf'), -float('inf')],
            requires_grad=True,
        )

        anchor = zero_grad_anchor(value)
        self.assertTrue(torch.isfinite(anchor).item())
        self.assertEqual(anchor.item(), 0.0)
        anchor.backward()

        torch.testing.assert_close(
            value.grad,
            torch.zeros_like(value),
            rtol=0,
            atol=0,
        )

    def test_does_not_change_existing_gradients(self):
        torch.manual_seed(11)
        reference = torch.nn.Linear(5, 3)
        anchored = torch.nn.Linear(5, 3)
        anchored.load_state_dict(reference.state_dict())
        inputs = torch.randn(4, 5)

        reference_output = reference(inputs)
        reference_output.square().mean().backward()
        anchored_output = anchored(inputs)
        (anchored_output.square().mean() +
         zero_grad_anchor(anchored_output)).backward()

        anchored_parameters = dict(anchored.named_parameters())
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                anchored_parameters[name].grad,
                rtol=0,
                atol=0,
            )


if __name__ == '__main__':
    unittest.main()
