"""Benchmark behavior-equivalent routed-LoRA index caching on one GPU."""

import argparse
import importlib.util
import statistics
import time
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / 'model/custom_model.py'
SPEC = importlib.util.spec_from_file_location(
    'routed_lora_custom_model_benchmark', MODULE_PATH)
CUSTOM_MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CUSTOM_MODEL)
RoutedLoRALinear = CUSTOM_MODEL.RoutedLoRALinear
RoutedLoRARouteCache = CUSTOM_MODEL._RoutedLoRARouteCache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--sequence-length', type=int, default=64)
    parser.add_argument('--hidden-size', type=int, default=4096)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--route-layer-calls', type=int, default=248)
    parser.add_argument('--forward-layer-calls', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=7)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    families = tuple(f'family_{index}' for index in range(6))
    module = RoutedLoRALinear(
        torch.nn.Linear(args.hidden_size, args.hidden_size, bias=False),
        rank=args.rank,
        alpha=args.rank,
        dropout=0.0,
        route_families=families,
        family_rank=args.rank,
        family_alpha=args.rank,
    ).to(device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    route = torch.arange(args.batch_size, device=device) % len(families)

    module.set_route_family(route)
    legacy_output = module(x)
    module.set_route_family(RoutedLoRARouteCache(route, len(families)))
    cached_output = module(x)
    if not torch.equal(legacy_output, cached_output):
        raise AssertionError('Cached and legacy outputs differ.')

    def run_once(use_cache: bool, layer_calls: int, full_forward: bool):
        route_context = (
            RoutedLoRARouteCache(route, len(families))
            if use_cache else route)
        module.set_route_family(route_context)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(layer_calls):
            if full_forward:
                module(x)
            else:
                module._route_indices_for_input(x)
        torch.cuda.synchronize(device)
        return (time.perf_counter() - started) * 1000.0

    for full_forward, layer_calls, name in (
        (False, args.route_layer_calls, 'route_only'),
        (True, args.forward_layer_calls, 'full_forward'),
    ):
        run_once(False, layer_calls, full_forward)
        run_once(True, layer_calls, full_forward)
        samples = {False: [], True: []}
        for repeat in range(args.repeats):
            order = (False, True) if repeat % 2 == 0 else (True, False)
            for use_cache in order:
                samples[use_cache].append(
                    run_once(use_cache, layer_calls, full_forward))
        legacy_ms = statistics.median(samples[False])
        cached_ms = statistics.median(samples[True])
        speedup = legacy_ms / cached_ms
        reduction = (legacy_ms - cached_ms) / legacy_ms * 100.0
        print(
            f'{name}: calls={layer_calls} legacy_ms={legacy_ms:.3f} '
            f'cached_ms={cached_ms:.3f} speedup={speedup:.3f}x '
            f'reduction={reduction:.2f}%')


if __name__ == '__main__':
    main()
