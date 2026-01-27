# Copyright (c) OpenMMLab. All rights reserved.
"""
Category and Project-Aware Sampler for Multi-task Learning.
Optimized for DeepSpeed ZeRO-2: Independent Task Scheduling per Rank.
"""
import math
import random
from typing import Iterator, List, Optional, Dict, Tuple, Sized
from collections import defaultdict

import torch
from torch.utils.data import Sampler
from mmengine.dist import sync_random_seed, get_dist_info
from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class CategoryProjectSampler(Sampler):
    """
    A sampler where:
    1. Data is sharded across ranks (standard DDP behavior).
    2. Each rank maintains independent shuffling cycles for each (Category, Project) group.
    3. Different ranks can process different categories simultaneously.
    
    Args:
        dataset: The dataset.
        batch_size: Local batch size per GPU.
        shuffle: Whether to shuffle indices.
        seed: Random seed.
        round_up: Whether to round up total length.
    """
    
    def __init__(self,
                 dataset: Sized,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.round_up = round_up
        
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        
        # 1. Calculate number of samples this rank is responsible for
        total_len = len(self.dataset)
        if self.round_up:
            self.num_samples = math.ceil(total_len / world_size)
        else:
            self.num_samples = math.ceil((total_len - rank) / world_size)
            
        # 2. Lazy init state
        self._index_groups_built = False
        self.groups: Dict[Tuple[str, str], List[int]] = {}
        self.group_keys: List[Tuple[str, str]] = []

    def _build_index_groups(self) -> None:
        """
        Build index groups, but ONLY for the subset of data belonging to this rank.
        This implements the 'Distributed' part of DistributedSampler.
        """
        if self._index_groups_built:
            return
            
        # Structure: {(category, project): [indices]}
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        
        # Get all indices available in the dataset
        all_indices = torch.arange(len(self.dataset))
        
        # Deterministically shuffle all indices first based on seed 0 
        # (so all ranks agree on the global permutation before sharding)
        g = torch.Generator()
        g.manual_seed(self.seed) # Fixed seed for sharding consistency
        if self.shuffle:
            perm = torch.randperm(len(self.dataset), generator=g)
            all_indices = all_indices[perm]
            
        # Subsample indices for THIS rank
        # e.g., Rank 0 gets indices [0, 2, 4...], Rank 1 gets [1, 3, 5...]
        # This ensures strict non-overlapping data across GPUs.
        rank_indices = all_indices[self.rank::self.world_size].tolist()
        
        # Now classify these rank-specific indices into groups
        text_data = getattr(self.dataset, 'text_data', None)
        
        if text_data is None:
            groups[('default', 'default')] = rank_indices
        else:
            for idx in rank_indices:
                try:
                    sample = text_data[idx]
                    category = sample.get('category', 'default') or 'default'
                    
                    if self._is_survival_category(category):
                        project = sample.get('project', 'default') or 'default'
                    else:
                        project = 'all' 
                    
                    groups[(category, project)].append(idx)
                except Exception:
                    groups[('default', 'default')].append(idx)
        
        # Filter out empty groups (possible if dataset is small or very imbalanced)
        self.groups = {k: v for k, v in groups.items() if len(v) > 0}
        self.group_keys = list(self.groups.keys())
        self._index_groups_built = True

    def _is_survival_category(self, category: str) -> bool:
        return 'survival' in category.lower()

    def _infinite_group_iterator(self, indices: List[int], seed: int) -> Iterator[int]:
        """
        Yields indices from a specific group indefinitely with reshuffling.
        """
        g = torch.Generator()
        g.manual_seed(seed)
        
        while True:
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                current_indices = [indices[i] for i in perm]
            else:
                current_indices = indices[:]
            
            yield from current_indices

    def __iter__(self) -> Iterator[int]:
        self._build_index_groups()
        
        # Seed for task selection logic
        # Use (seed + epoch + rank) so each rank makes DIFFERENT choices about 
        # which category to train on at step t.
        rng = random.Random(self.seed + self.epoch + self.rank)
        
        # Create independent iterators for each group
        # Seed includes rank to ensure randomness divergence
        group_iterators = {
            key: self._infinite_group_iterator(indices, self.seed + self.epoch + hash(key))
            for key, indices in self.groups.items()
        }
        
        final_indices = []
        
        # Calculate how many batches we need to yield
        num_batches = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            num_batches += 1
            
        for _ in range(num_batches):
            if not self.group_keys:
                break
                
            # 1. Randomly select a group (Rank-specific choice)
            selected_key = rng.choice(self.group_keys)
            iterator = group_iterators[selected_key]
            
            # 2. Draw batch_size samples
            # Since we already sharded data in _build_index_groups, 
            # we just take 'batch_size' items directly.
            for _ in range(self.batch_size):
                final_indices.append(next(iterator))
            
        # Truncate to exact length required by Sampler contract
        return iter(final_indices[:self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch