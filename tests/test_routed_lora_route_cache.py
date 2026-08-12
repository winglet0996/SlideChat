import copy
import importlib.util
from pathlib import Path
import unittest

import torch
from torch.utils.checkpoint import checkpoint

_MODULE_PATH = Path(__file__).parents[1] / 'xtuner/model/custom_model.py'
_SPEC = importlib.util.spec_from_file_location(
    'routed_lora_custom_model_under_test', _MODULE_PATH)
_CUSTOM_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CUSTOM_MODEL)
RoutedLoRALinear = _CUSTOM_MODEL.RoutedLoRALinear
_RoutedLoRARouteCache = _CUSTOM_MODEL._RoutedLoRARouteCache


FAMILIES = ('family_a', 'family_b', 'family_c')


def _make_pair(dropout: float):
    torch.manual_seed(7)
    reference = RoutedLoRALinear(
        torch.nn.Linear(7, 5, bias=False),
        rank=3,
        alpha=3,
        dropout=dropout,
        route_families=FAMILIES,
        family_rank={'family_a': 2, 'family_b': 3, 'family_c': 4},
        family_alpha={'family_a': 2, 'family_b': 3, 'family_c': 4},
    )
    with torch.no_grad():
        reference.shared_lora.lora_B.weight.normal_()
        for adapter in reference.family_lora.values():
            adapter.lora_B.weight.normal_()
    return reference, copy.deepcopy(reference)


def _assert_gradients_equal(test_case, reference, cached, x_ref, x_cached):
    torch.testing.assert_close(x_ref.grad, x_cached.grad, rtol=0, atol=0)
    cached_params = dict(cached.named_parameters())
    for name, parameter in reference.named_parameters():
        other = cached_params[name]
        if parameter.grad is None:
            test_case.assertIsNone(other.grad, name)
        else:
            torch.testing.assert_close(
                parameter.grad, other.grad, rtol=0, atol=0)


class RoutedLoRARouteCacheTest(unittest.TestCase):

    def test_mixed_family_forward_and_backward_are_exact(self):
        reference, cached = _make_pair(dropout=0.25)
        route = torch.tensor([0, 2, 0, 2])
        reference.set_route_family(route)
        cached.set_route_family(_RoutedLoRARouteCache(route, len(FAMILIES)))
        x_ref = torch.randn(4, 5, 7, requires_grad=True)
        x_cached = x_ref.detach().clone().requires_grad_(True)

        torch.manual_seed(19)
        out_ref = reference(x_ref)
        torch.manual_seed(19)
        out_cached = cached(x_cached)
        torch.testing.assert_close(out_ref, out_cached, rtol=0, atol=0)

        out_ref.square().sum().backward()
        out_cached.square().sum().backward()
        _assert_gradients_equal(
            self, reference, cached, x_ref, x_cached)

    def test_flattened_route_expansion_is_exact(self):
        reference, cached = _make_pair(dropout=0.0)
        route = torch.tensor([2, 0, 1])
        route_cache = _RoutedLoRARouteCache(route, len(FAMILIES))
        reference.set_route_family(route)
        cached.set_route_family(route_cache)
        x = torch.randn(12, 7)

        torch.testing.assert_close(reference(x), cached(x), rtol=0, atol=0)
        torch.testing.assert_close(
            route_cache.route_for_input(x),
            route.repeat_interleave(4),
            rtol=0,
            atol=0,
        )

    def test_indices_are_shared_between_layers(self):
        first, second = _make_pair(dropout=0.0)
        route_cache = _RoutedLoRARouteCache(
            torch.tensor([0, 1, 2, 0]), len(FAMILIES))
        first.set_route_family(route_cache)
        second.set_route_family(route_cache)
        x = torch.randn(4, 3, 7)

        first_indices = first._route_indices_for_input(x)
        second_indices = second._route_indices_for_input(x)
        self.assertEqual(len(route_cache._indices), 1)
        for first_idx, second_idx in zip(first_indices, second_indices):
            self.assertIs(first_idx, second_idx)

    def test_checkpoint_recomputation_is_exact(self):
        reference, cached = _make_pair(dropout=0.25)
        route = torch.tensor([0, 2, 1, 0])
        reference.set_route_family(route)
        cached.set_route_family(_RoutedLoRARouteCache(route, len(FAMILIES)))
        x_ref = torch.randn(4, 3, 7, requires_grad=True)
        x_cached = x_ref.detach().clone().requires_grad_(True)

        torch.manual_seed(23)
        out_ref = checkpoint(reference, x_ref, use_reentrant=True)
        out_ref.square().sum().backward()
        torch.manual_seed(23)
        out_cached = checkpoint(cached, x_cached, use_reentrant=True)
        out_cached.square().sum().backward()

        torch.testing.assert_close(out_ref, out_cached, rtol=0, atol=0)
        _assert_gradients_equal(
            self, reference, cached, x_ref, x_cached)

    def test_cache_does_not_change_checkpoint_keys(self):
        module, _ = _make_pair(dropout=0.0)
        keys_before = tuple(module.state_dict())
        module.set_route_family(_RoutedLoRARouteCache(
            torch.tensor([0, 1]), len(FAMILIES)))
        self.assertEqual(keys_before, tuple(module.state_dict()))

    def test_invalid_leading_dimension_still_raises(self):
        module, _ = _make_pair(dropout=0.0)
        module.set_route_family(_RoutedLoRARouteCache(
            torch.tensor([0, 1, 2]), len(FAMILIES)))
        with self.assertRaisesRegex(ValueError, 'Cannot align per-sample route'):
            module(torch.randn(5, 7))


if __name__ == '__main__':
    unittest.main()
