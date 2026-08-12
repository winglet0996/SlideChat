#!/usr/bin/env python3
"""Small NCCL/DeepSpeed smoke test for a torchrun multi-node allocation."""

import argparse
import json
import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tensor-mib', type=int, default=16,
        help='Size of the NCCL all-reduce tensor on each rank (default: 16 MiB).')
    parser.add_argument(
        '--warmup-iters', type=int, default=5,
        help='Untimed NCCL all-reduce warmup iterations (default: 5).')
    parser.add_argument(
        '--benchmark-iters', type=int, default=20,
        help='Timed NCCL all-reduce iterations (default: 20).')
    parser.add_argument(
        '--timeout-seconds', type=int, default=300,
        help='Distributed initialization/collective timeout.')
    parser.add_argument(
        '--check-deepspeed', action='store_true',
        help='Also import DeepSpeed on every GPU rank.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tensor_mib <= 0:
        raise ValueError('--tensor-mib must be positive.')
    if args.warmup_iters < 0:
        raise ValueError('--warmup-iters cannot be negative.')
    if args.benchmark_iters <= 0:
        raise ValueError('--benchmark-iters must be positive.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing to run an NCCL smoke test.')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=torch.device('cuda', local_rank))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    element_size = torch.tensor([], dtype=torch.float32).element_size()
    count = args.tensor_mib * 1024 * 1024 // element_size
    value = torch.full(
        (count,), float(rank + 1), dtype=torch.float32, device=local_rank)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)

    expected = world_size * (world_size + 1) / 2
    max_error = float((value - expected).abs().max().item())
    if max_error != 0.0:
        raise RuntimeError(
            f'NCCL all-reduce mismatch on rank {rank}: max_error={max_error}')

    # Measure a large, steady-state collective as well as checking correctness.
    # The maximum rank duration is used because the slowest rank determines the
    # wall-clock time of synchronous data-parallel training.
    value.zero_()
    for _ in range(args.warmup_iters):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    dist.barrier()
    started = time.perf_counter()
    for _ in range(args.benchmark_iters):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    elapsed = time.perf_counter() - started
    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=local_rank)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    max_elapsed = float(elapsed_tensor.item())
    seconds_per_collective = max_elapsed / args.benchmark_iters
    tensor_bytes = value.numel() * value.element_size()
    algorithmic_gbps = tensor_bytes / seconds_per_collective / 1e9
    bus_gbps = algorithmic_gbps * (2.0 * (world_size - 1) / world_size)

    deepspeed_version = None
    if args.check_deepspeed:
        import deepspeed
        deepspeed_version = deepspeed.__version__
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        nccl_version = list(nccl_version)

    metadata = {
        'rank': rank,
        'local_rank': local_rank,
        'node_rank': int(os.environ.get('GROUP_RANK', os.environ.get(
            'SENSECORE_PYTORCH_NODE_RANK', '-1'))),
        'hostname': socket.gethostname(),
        'gpu': torch.cuda.get_device_name(local_rank),
        'nccl_version': nccl_version,
        'deepspeed': deepspeed_version,
    }
    all_metadata = [None] * world_size
    dist.all_gather_object(all_metadata, metadata)
    dist.barrier()

    if rank == 0:
        hostnames = sorted({item['hostname'] for item in all_metadata})
        expected_nodes = int(os.environ.get('SENSECORE_PYTORCH_NNODES', '0'))
        if expected_nodes and len(hostnames) != expected_nodes:
            raise RuntimeError(
                f'Expected {expected_nodes} nodes but ranks reported {hostnames!r}')
        print(json.dumps({
            'status': 'ok',
            'backend': dist.get_backend(),
            'world_size': world_size,
            'hostnames': hostnames,
            'tensor_mib_per_rank': args.tensor_mib,
            'warmup_iters': args.warmup_iters,
            'benchmark_iters': args.benchmark_iters,
            'seconds_per_all_reduce': seconds_per_collective,
            'algorithmic_bandwidth_gbps': algorithmic_gbps,
            'bus_bandwidth_gbps': bus_gbps,
            'expected_all_reduce_value': expected,
            'nccl_environment': {
                name: os.environ.get(name)
                for name in (
                    'NCCL_IB_DISABLE',
                    'NCCL_IB_HCA',
                    'NCCL_NET',
                    'NCCL_SOCKET_IFNAME',
                )
            },
            'ranks': all_metadata,
        }, indent=2, ensure_ascii=False), flush=True)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
