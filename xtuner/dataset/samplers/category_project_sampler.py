# Copyright (c) OpenMMLab. All rights reserved.
"""
Category and Project-Aware Sampler for Multi-task Learning with Cox Loss.

This sampler ensures that:
1. Each batch contains samples from the SAME category AND project
2. ALL tasks (including mcqa) are grouped by project (cancer type) for balanced training
3. Tasks and projects are balanced during training through weighted sampling
4. Batch boundaries are preserved across distributed training
5. **Samples are not repeated within an epoch** (sampling without replacement)

Key Design:
- Pre-generate complete batches where each batch is homogeneous (same category+project)
- Shuffle batches (not individual indices) to maintain batch integrity
- Distribute batches to GPUs, not individual indices
- Use shuffled pools per group, consumed sequentially to avoid repetition
- Groups with more remaining samples get higher sampling probability

Distributed Training (DDP) Compatibility:
- All ranks generate identical batch assignments using the same seed
- Each rank takes every world_size-th batch starting from its rank
- This ensures no overlap between ranks and complete coverage
- Gradient synchronization happens after local batch processing
"""
import math
import random
from typing import Iterator, List, Optional, Dict, Any, Sized, Tuple
from collections import defaultdict, Counter

import torch
from torch.utils.data import Sampler
from mmengine.dist import sync_random_seed, get_dist_info, is_main_process
from mmengine.registry import DATA_SAMPLERS

# Debug flag - set to True to enable detailed sampling logs
DEBUG_SAMPLER = True


@DATA_SAMPLERS.register_module()
class CategoryProjectSampler(Sampler):
    """
    A sampler that groups samples by category and project for multi-task learning.
    
    CRITICAL: Each batch contains samples from the SAME (category, project) group.
    This is essential for:
    - Cox proportional hazards loss (survival) - requires same cancer type cohort
    - Balanced multi-task learning - ensures all tasks and cancer types are trained
    
    Sampling Strategy (Optimized for Data Utilization):
    1. Group all samples by (category, project)
    2. Shuffle each group's indices at epoch start (deterministic per epoch+seed)
    3. Consume samples sequentially from shuffled pools (no repetition within epoch)
    4. Balance groups using weighted sampling based on remaining samples
    5. When a group is exhausted mid-epoch, reshuffle and continue
    6. Shuffle batches (not indices) to maintain batch homogeneity
    7. Distribute complete batches to each GPU
    
    This ensures:
    - Maximum data utilization: each sample seen ~once per epoch
    - Balanced training: groups sampled proportionally to their remaining size
    - No repetition: sequential consumption from shuffled pools
    - DDP compatibility: identical batch generation across all ranks
    
    Args:
        dataset: The dataset to sample from
        batch_size: Number of samples per batch (must match DataLoader batch_size)
        shuffle: Whether to shuffle batches
        seed: Random seed for reproducibility
        round_up: Whether to round up the number of samples
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
        
        # Set seed - synchronized across all ranks
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        
        # Lazily build index groups - defer until first iteration
        self._index_groups_built = False
        # Structure: {(category, project): [indices]}
        self.group_indices: Dict[Tuple[str, str], List[int]] = {}
        self.groups: List[Tuple[str, str]] = []  # List of (category, project) tuples
        
        # Calculate number of samples per GPU
        # We need to ensure batch alignment across GPUs
        total_batches = math.ceil(len(self.dataset) / self.batch_size)
        batches_per_gpu = math.ceil(total_batches / world_size)
        
        if self.round_up:
            self.num_samples = batches_per_gpu * self.batch_size
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)
        
        # Debug: Print initialization info
        if DEBUG_SAMPLER and is_main_process():
            print(f"\n[CategoryProjectSampler.__init__] Rank {self.rank}/{self.world_size}")
            print(f"  - Dataset size: {len(self.dataset)}")
            print(f"  - Batch size: {self.batch_size}")
            print(f"  - Total batches: {total_batches}")
            print(f"  - Batches per GPU: {batches_per_gpu}")
            print(f"  - Shuffle: {self.shuffle}")
            print(f"  - Round up: {self.round_up}")
            print(f"  - Num samples per GPU: {self.num_samples}")
            print(f"  - Total size: {self.total_size}")
            print(f"  - Seed: {self.seed}")
            print(f"  - Sampling: WITHOUT replacement (epoch-based shuffled pools)")
    
    def _build_index_groups(self) -> None:
        """Build index groups organized by (category, project) tuples.
        
        For survival tasks, also tracks event indices separately for 
        event-aware sampling to ensure Cox loss has enough events per batch.
        """
        if self._index_groups_built:
            return
            
        # Structure: {(category, project): [indices]}
        group_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        # For survival groups: {(category, project): {'event': [idx], 'censored': [idx]}}
        self.survival_event_indices: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
        
        # Access the underlying text_data from the dataset
        text_data = getattr(self.dataset, 'text_data', None)
        if text_data is None:
            # Fallback: treat all samples as one group
            group_indices[('default', 'default')] = list(range(len(self.dataset)))
            self.group_indices = dict(group_indices)
            self.groups = [('default', 'default')]
            self._index_groups_built = True
            return
        
        for idx in range(len(text_data)):
            try:
                sample = text_data[idx]
                category = sample.get('category', 'default') or 'default'
                project = sample.get('project', 'default') or 'default'
                
                # Group by (category, project) tuple - ALL tasks get project grouping
                group_key = (category, project)
                group_indices[group_key].append(idx)
                
                # For survival tasks, track event vs censored samples
                if 'survival' in category.lower():
                    survival_targets = sample.get('survival_targets')
                    if group_key not in self.survival_event_indices:
                        self.survival_event_indices[group_key] = {'event': [], 'censored': []}
                    
                    if survival_targets is not None and len(survival_targets) >= 2:
                        # survival_targets is [time, event] where event=1 means event occurred
                        event = survival_targets[1] if isinstance(survival_targets, (list, tuple)) else 0
                        if event == 1:
                            self.survival_event_indices[group_key]['event'].append(idx)
                        else:
                            self.survival_event_indices[group_key]['censored'].append(idx)
                    else:
                        # No valid survival data, treat as censored
                        self.survival_event_indices[group_key]['censored'].append(idx)
                
            except Exception:
                # Fallback for problematic samples
                group_indices[('default', 'default')].append(idx)
        
        # Convert to regular dict and get group list
        self.group_indices = dict(group_indices)
        self.groups = list(self.group_indices.keys())
        self._index_groups_built = True
        
        # Debug: Print group statistics
        if DEBUG_SAMPLER and is_main_process():
            print(f"\n{'='*70}")
            print(f"[CategoryProjectSampler] Index groups built")
            print(f"{'='*70}")
            print(f"Total dataset size: {len(self.dataset)}")
            print(f"Total groups (category, project): {len(self.groups)}")
            
            # Group by category for summary
            category_stats: Dict[str, Dict[str, int]] = defaultdict(dict)
            for (cat, proj), indices in self.group_indices.items():
                category_stats[cat][proj] = len(indices)
            
            print(f"\nDetailed distribution by category and project:")
            for cat in sorted(category_stats.keys()):
                projects = category_stats[cat]
                total = sum(projects.values())
                print(f"\n  [{cat}]: {total} total samples, {len(projects)} projects")
                for proj, count in sorted(projects.items()):
                    print(f"    - {proj}: {count} samples")
            
            # Print survival event statistics
            if self.survival_event_indices:
                print(f"\n{'='*70}")
                print(f"[Survival Event-Aware Sampling] Event distribution:")
                low_event_groups = []
                for group_key, event_info in sorted(self.survival_event_indices.items()):
                    n_events = len(event_info['event'])
                    n_censored = len(event_info['censored'])
                    total = n_events + n_censored
                    event_rate = n_events / total * 100 if total > 0 else 0
                    cat, proj = group_key
                    print(f"  {proj}: {n_events} events ({event_rate:.1f}%), {n_censored} censored")
                    if n_events < 2:
                        low_event_groups.append((proj, n_events))
                
                if low_event_groups:
                    print(f"\n⚠ Warning: Groups with <2 events (may have many zero-loss batches):")
                    for proj, n in low_event_groups:
                        print(f"    - {proj}: only {n} event(s)")
            
            print(f"{'='*70}\n")
    
    def _sample_batch_from_pool(self, 
                                group: Tuple[str, str], 
                                group_pools: Dict[Tuple[str, str], List[int]],
                                group_positions: Dict[Tuple[str, str], int],
                                rng: random.Random) -> List[int]:
        """
        Sample a batch from a group's shuffled pool WITHOUT replacement.
        
        For survival groups, uses event-aware sampling to ensure sufficient events per batch.
        For other groups, takes the next batch_size samples from the shuffled pool.
        
        When a pool is exhausted, it is reshuffled and consumption continues.
        This ensures maximum data utilization while maintaining randomness.
        
        Args:
            group: (category, project) tuple
            group_pools: Dict mapping groups to their shuffled index pools
            group_positions: Dict tracking consumption position in each pool
            rng: Random number generator for reproducibility
            
        Returns:
            List of sample indices for the batch
        """
        pool = group_pools[group]
        pos = group_positions[group]
        pool_size = len(pool)
        
        # Check if this is a survival group requiring event-aware sampling
        if group in self.survival_event_indices:
            return self._sample_batch_event_aware_from_pool(
                group, group_pools, group_positions, rng
            )
        
        # Standard sampling: take next batch_size samples from shuffled pool
        batch = []
        while len(batch) < self.batch_size:
            remaining_in_pool = pool_size - pos
            needed = self.batch_size - len(batch)
            
            if remaining_in_pool >= needed:
                # Enough samples remaining
                batch.extend(pool[pos:pos + needed])
                group_positions[group] = pos + needed
                break
            else:
                # Take what's left, then reshuffle and continue
                batch.extend(pool[pos:])
                # Reshuffle pool for next round
                rng.shuffle(pool)
                group_positions[group] = 0
                pos = 0
        
        return batch
    
    def _sample_batch_event_aware_from_pool(self,
                                            group: Tuple[str, str],
                                            group_pools: Dict[Tuple[str, str], List[int]],
                                            group_positions: Dict[Tuple[str, str], int],
                                            rng: random.Random) -> List[int]:
        """
        Event-aware batch sampling for survival groups WITHOUT replacement.
        
        Ensures at least min_events_per_batch events are included in each batch
        to prevent Cox loss from returning 0, while still consuming from pools
        sequentially to maximize data utilization.
        
        Strategy:
        1. Track separate pools for events and censored samples
        2. Take min_events from event pool, rest from censored pool
        3. When either pool is exhausted, reshuffle it
        
        Args:
            group: (category, project) tuple
            group_pools: Dict mapping groups to their shuffled index pools
            group_positions: Dict tracking consumption position in each pool  
            rng: Random number generator
            
        Returns:
            List of sample indices for the batch
        """
        min_events_per_batch = 2  # Cox loss needs >= 1 event, target 2 for robustness
        
        event_info = self.survival_event_indices[group]
        event_pool_key = (group, 'event')
        censored_pool_key = (group, 'censored')
        
        # Initialize event/censored pools if not already done
        if event_pool_key not in group_pools:
            event_indices = event_info['event'].copy()
            censored_indices = event_info['censored'].copy()
            rng.shuffle(event_indices)
            rng.shuffle(censored_indices)
            group_pools[event_pool_key] = event_indices
            group_pools[censored_pool_key] = censored_indices
            group_positions[event_pool_key] = 0
            group_positions[censored_pool_key] = 0
        
        event_pool = group_pools[event_pool_key]
        censored_pool = group_pools[censored_pool_key]
        event_pos = group_positions[event_pool_key]
        censored_pos = group_positions[censored_pool_key]
        
        batch = []
        
        # Calculate how many events we should include
        n_events_target = min(min_events_per_batch, len(event_info['event']), self.batch_size)
        n_censored_target = self.batch_size - n_events_target
        
        # Sample events
        events_sampled = 0
        while events_sampled < n_events_target and event_pool:
            remaining = len(event_pool) - event_pos
            needed = n_events_target - events_sampled
            
            if remaining >= needed:
                batch.extend(event_pool[event_pos:event_pos + needed])
                group_positions[event_pool_key] = event_pos + needed
                events_sampled += needed
                break
            else:
                batch.extend(event_pool[event_pos:])
                events_sampled += remaining
                # Reshuffle and reset
                rng.shuffle(event_pool)
                group_positions[event_pool_key] = 0
                event_pos = 0
        
        # Sample censored to fill the rest
        censored_sampled = 0
        while censored_sampled < n_censored_target:
            if not censored_pool:
                # No censored samples, fill from the general pool
                general_pool = group_pools.get(group, self.group_indices[group])
                needed = n_censored_target - censored_sampled
                batch.extend(rng.choices(general_pool, k=needed))
                break
                
            remaining = len(censored_pool) - censored_pos
            needed = n_censored_target - censored_sampled
            
            if remaining >= needed:
                batch.extend(censored_pool[censored_pos:censored_pos + needed])
                group_positions[censored_pool_key] = censored_pos + needed
                censored_sampled += needed
                break
            else:
                batch.extend(censored_pool[censored_pos:])
                censored_sampled += remaining
                # Reshuffle and reset
                rng.shuffle(censored_pool)
                group_positions[censored_pool_key] = 0
                censored_pos = 0
        
        # Fill any remaining slots if batch is still short
        while len(batch) < self.batch_size:
            # Use the general group pool
            general_pool = self.group_indices[group]
            batch.append(rng.choice(general_pool))
        
        return batch[:self.batch_size]
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for sampling with batch-level homogeneity and NO repetition.
        
        Key Algorithm:
        1. Initialize shuffled pools for each group (deterministic per epoch+seed)
        2. Use weighted group selection based on remaining samples (groups with more 
           remaining samples are more likely to be selected)
        3. Consume samples sequentially from pools (no repetition within epoch)
        4. When a pool is exhausted, reshuffle it (allows continuation if needed)
        5. Shuffle batches (not indices) to maintain batch homogeneity
        6. Distribute batches to ranks for DDP
        
        DDP Correctness:
        - All ranks use the same seed (self.seed + self.epoch)
        - All ranks generate identical batch lists
        - Each rank takes every world_size-th batch starting from its rank
        - This ensures no overlap and complete coverage
        """
        # Build index groups on first iteration (lazy initialization)
        self._build_index_groups()
        
        # Set random seed based on epoch - MUST be same across all ranks for DDP
        rng = random.Random(self.seed + self.epoch)
        
        # Step 1: Initialize shuffled pools for each group
        # Each pool is a shuffled copy of the group's indices
        group_pools: Dict[Tuple[str, str], List[int]] = {}
        group_positions: Dict[Tuple[str, str], int] = {}
        
        for group, indices in self.group_indices.items():
            pool = indices.copy()
            rng.shuffle(pool)
            group_pools[group] = pool
            group_positions[group] = 0
        
        # Step 2: Generate batches where each batch is from the same (category, project) group
        all_batches: List[List[int]] = []
        batch_group_log: List[Tuple[str, str]] = []
        
        # Calculate how many batches we need total (before distribution)
        num_batches_needed = math.ceil(self.total_size / self.batch_size)
        
        # Track samples consumed per group for statistics
        samples_consumed: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Calculate target batches per group for balanced sampling
        # Each group should get batches proportional to its size
        total_samples = sum(len(indices) for indices in self.group_indices.values())
        target_batches_per_group: Dict[Tuple[str, str], float] = {}
        for group, indices in self.group_indices.items():
            group_fraction = len(indices) / total_samples if total_samples > 0 else 1.0 / len(self.groups)
            target_batches_per_group[group] = group_fraction * num_batches_needed
        
        # Track batches assigned to each group
        batches_assigned: Dict[Tuple[str, str], int] = defaultdict(int)
        
        while len(all_batches) < num_batches_needed:
            # Weighted group selection based on how far each group is from its target
            # Groups that are behind their target get higher weight
            weights = []
            available_groups = []
            
            for group in self.groups:
                pool = group_pools[group]
                pos = group_positions[group]
                remaining = len(pool) - pos
                
                # Calculate how far behind this group is from its target
                target = target_batches_per_group[group]
                assigned = batches_assigned[group]
                deficit = max(0, target - assigned)
                
                # Weight combines: 1) deficit from target, 2) remaining samples in pool
                # Groups behind their target AND with remaining samples get priority
                if remaining > 0:
                    # Prefer groups with remaining samples and behind their target
                    weight = deficit + 1.0  # +1 to ensure all groups have some weight
                else:
                    # Pool exhausted, lower weight but keep in rotation
                    weight = max(0.1, deficit * 0.5)
                
                weights.append(weight)
                available_groups.append(group)
            
            if not available_groups:
                break
            
            # Weighted random selection
            total_weight = sum(weights)
            r = rng.random() * total_weight
            cumsum = 0
            selected_group = available_groups[0]
            for group, weight in zip(available_groups, weights):
                cumsum += weight
                if r < cumsum:
                    selected_group = group
                    break
            
            # Sample a batch from the selected group's pool (without replacement)
            batch = self._sample_batch_from_pool(
                selected_group, group_pools, group_positions, rng
            )
            
            all_batches.append(batch)
            batch_group_log.append(selected_group)
            samples_consumed[selected_group] += len(batch)
            batches_assigned[selected_group] += 1
        
        # Step 3: Shuffle batches (not indices within batches) to randomize order
        if self.shuffle:
            paired = list(zip(all_batches, batch_group_log))
            rng.shuffle(paired)
            if paired:
                all_batches, batch_group_log = zip(*paired)
                all_batches = list(all_batches)
                batch_group_log = list(batch_group_log)
            else:
                all_batches = []
                batch_group_log = []
        
        # Step 4: Distribute batches to this rank
        # Each GPU gets every world_size-th batch starting from its rank
        # This ensures: rank 0 gets [0, n, 2n, ...], rank 1 gets [1, n+1, 2n+1, ...], etc.
        my_batches = all_batches[self.rank::self.world_size]
        my_batch_groups = batch_group_log[self.rank::self.world_size]
        
        # Step 5: Flatten batches to indices
        indices = []
        for batch in my_batches:
            indices.extend(batch)
        
        # Ensure we have exactly num_samples (pad if needed)
        if len(indices) < self.num_samples:
            if my_batches and my_batch_groups:
                last_group = my_batch_groups[-1]
                pool = self.group_indices[last_group]
                padding = rng.choices(pool, k=self.num_samples - len(indices))
                indices.extend(padding)
            elif self.groups:
                # Fallback: use any group
                pool = self.group_indices[self.groups[0]]
                padding = rng.choices(pool, k=self.num_samples - len(indices))
                indices.extend(padding)
        indices = indices[:self.num_samples]
        
        # Debug: Print sampling statistics
        if DEBUG_SAMPLER and is_main_process():
            self._print_sampling_stats(
                all_batches, batch_group_log, my_batches, my_batch_groups,
                indices, samples_consumed, group_pools, group_positions
            )
        
        return iter(indices)
    
    def _print_sampling_stats(self, 
                              all_batches: List[List[int]],
                              batch_group_log: List[Tuple[str, str]],
                              my_batches: List[List[int]],
                              my_batch_groups: List[Tuple[str, str]],
                              indices: List[int],
                              samples_consumed: Dict[Tuple[str, str], int],
                              group_pools: Dict[Tuple[str, str], List[int]],
                              group_positions: Dict[Tuple[str, str], int]) -> None:
        """Print detailed sampling statistics for debugging."""
        print(f"\n{'='*70}")
        print(f"[CategoryProjectSampler] Epoch {self.epoch} - Rank {self.rank}/{self.world_size}")
        print(f"{'='*70}")
        print(f"Seed: {self.seed + self.epoch}")
        print(f"Total batches generated: {len(all_batches)}")
        print(f"Batches for this rank: {len(my_batches)}")
        print(f"Indices for this rank: {len(indices)}")
        print(f"Batch size: {self.batch_size}")
        
        # Count by category
        cat_counts = Counter(cat for cat, _ in batch_group_log)
        print(f"\nBatch distribution by category (total: {len(batch_group_log)}):")
        for cat, count in sorted(cat_counts.items()):
            pct = 100.0 * count / len(batch_group_log) if batch_group_log else 0
            print(f"  - {cat}: {count} batches ({pct:.1f}%)")
        
        # Count by (category, project) with data utilization stats
        group_counts = Counter(batch_group_log)
        print(f"\nBatch distribution by (category, project) with data utilization:")
        for (cat, proj), count in sorted(group_counts.items()):
            pct = 100.0 * count / len(batch_group_log) if batch_group_log else 0
            group = (cat, proj)
            total_samples = len(self.group_indices.get(group, []))
            consumed = samples_consumed.get(group, 0)
            utilization = 100.0 * consumed / total_samples if total_samples > 0 else 0
            print(f"  - {cat} | {proj}: {count} batches ({pct:.1f}%), "
                  f"samples: {consumed}/{total_samples} ({utilization:.1f}% utilized)")
        
        # Show first 10 batch assignments
        print(f"\nFirst 10 batch assignments:")
        for i, (cat, proj) in enumerate(batch_group_log[:10]):
            print(f"  [Batch {i}] Category: {cat}, Project: {proj}")
        
        # Verify batch homogeneity for this rank
        print(f"\nBatch homogeneity check (this rank's first 5 batches):")
        for i, batch in enumerate(my_batches[:5]):
            if my_batch_groups and i < len(my_batch_groups):
                cat, proj = my_batch_groups[i]
                print(f"  [Batch {i}] Group: ({cat}, {proj}), Size: {len(batch)}")
        
        # Check for repeated samples within this rank's indices
        unique_indices = set(indices)
        repetition_rate = 1.0 - len(unique_indices) / len(indices) if indices else 0
        print(f"\nData utilization for this rank:")
        print(f"  - Total indices: {len(indices)}")
        print(f"  - Unique indices: {len(unique_indices)}")
        print(f"  - Repetition rate: {repetition_rate:.2%}")
        
        # Indices sanity check
        if indices:
            print(f"\nIndices sanity check:")
            print(f"  - Min index: {min(indices)}, Max index: {max(indices)}")
            print(f"  - Dataset size: {len(self.dataset)}")
            print(f"  - All indices valid: {all(0 <= i < len(self.dataset) for i in indices)}")
        print(f"{'='*70}\n")
    
    def __len__(self) -> int:
        """Return the number of samples for this rank."""
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.epoch = epoch
