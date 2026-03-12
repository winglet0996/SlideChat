# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils import mkdir_or_exist
from mmengine.utils.misc import get_object_from_string
from mmengine.dataset import DefaultSampler
from mmengine.dist import get_world_size, get_rank, collect_results, is_main_process
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GenerationConfig

from xtuner.dataset.llava_dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn, masked_collated_fn
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX)
from xtuner.evaluation.metrics.pathology_metric import PathologyMetric

class EvaluateChatHook_conv_longnet(Hook):
    """
    Elegant evaluation hook using LLaVADataset for data loading
    
    IMPORTANT: This hook is designed to work correctly in distributed (multi-GPU) environments.
    Key features for distributed evaluation:
    - Uses DistributedSampler to ensure each GPU processes different data samples
    - Avoids data duplication across GPUs
    - Collects results from all GPUs for complete evaluation using collect_results
    - Only logs metrics on the main process to avoid duplicate logging
    
    FIXED ISSUE: Previously, in a distributed setup with N GPUs, only ~1/N of the samples
    were being evaluated because each GPU was independently processing its subset without
    collecting results from other GPUs. Now all results are properly collected and aggregated.
    """

    priority = 'LOW'

    def __init__(self,
                 tokenizer,
                 evaluation_data_path,
                 image_folder=None,
                 image_path_list=None,
                 per_image_length=None,
                 max_patch_num=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=None,
                 batch_size=None,
                 num_workers=None,
                 every_n_iters=None,
                 max_new_tokens=None,
                 generation_kwargs={}):
        """
        Initialize evaluation hook with dataset-based approach
        
        Args:
            evaluation_data_path: Path to JSON/JSONL file containing evaluation data
            image_folder: Base folder for images
            image_path_list: List of image paths
            per_image_length: Number of tokens per image
            max_patch_num: Maximum number of patches for WSI
            dataset_map_fn: Function to map dataset
            template_map_fn: Function to map templates
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            num_workers: Number of workers for DataLoader
            every_n_iters: Evaluation frequency
            max_new_tokens: Maximum new tokens for generation
            generation_kwargs: Additional generation arguments
        """
        
        self.evaluation_data_path = evaluation_data_path
        self.image_folder = image_folder
        self.image_path_list = image_path_list
        self.per_image_length = per_image_length
        self.max_patch_num = max_patch_num
        self.dataset_map_fn = dataset_map_fn
        self.template_map_fn = template_map_fn
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        
        # Build tokenizer
        self.tokenizer = BUILDER.build(tokenizer)
        
        # Setup generation config
        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id)
        default_generation_kwargs.update(generation_kwargs)
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        # Initialize pathology metric
        self.pathology_metric = PathologyMetric(tokenizer=tokenizer)
        self.is_first_run = True
        
        # Will be initialized in before_train
        self.eval_dataset = None
        self.eval_dataloader = None

    def _create_distributed_sampler(self, dataset):
        """Create appropriate sampler based on distributed setting"""
        if get_world_size() > 1:
            # Use DistributedSampler for multi-GPU evaluation
            # drop_last=False ensures all samples are evaluated
            return DistributedSampler(
                dataset, 
                shuffle=False, 
                drop_last=False
            )
        else:
            return DefaultSampler(dataset, shuffle=False)

    def _create_eval_dataset(self, runner):
        """Create evaluation dataset using LLaVADataset"""
        
        # Create evaluation dataset
        self.eval_dataset = LLaVADataset(
            image_folder=self.image_folder,
            image_path_list=self.image_path_list,
            per_image_length=self.per_image_length,
            data_path=self.evaluation_data_path,
            tokenizer=self.tokenizer,
            max_dataset_length=None,
            dataset_map_fn=self.dataset_map_fn,
            template_map_fn=self.template_map_fn,
            max_length=self.max_length,
            mode='eval',  # Set to eval mode
            max_patch_num=self.max_patch_num,
            input_ids_with_output=True
        )
        
        # Create distributed-aware sampler
        sampler = self._create_distributed_sampler(self.eval_dataset)
            
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=masked_collated_fn
        )
        
        # Log dataset information
        total_samples = len(self.eval_dataset)
        if get_world_size() > 1:
            samples_per_gpu = total_samples // get_world_size()
            runner.logger.info(
                f"Created distributed evaluation dataset: "
                f"Total samples: {total_samples}, "
                f"Samples per GPU: ~{samples_per_gpu}, "
                f"World size: {get_world_size()}, "
                f"Current rank: {get_rank()}"
            )
        else:
            runner.logger.info(f"Created evaluation dataset with {total_samples} samples")

    def _calculate_metrics(self, predictions, targets, runner):
        """Calculate metrics using PathologyMetric with proper distributed collection"""
        # Process predictions and targets on current rank
        self.pathology_metric.process_predictions_and_targets(predictions, targets)
        
        # Get local results from current rank
        local_results = self.pathology_metric.results
        
        # Collect results from all ranks using MMEngine's collect_results
        # collect_results automatically handles distributed communication
        all_results = collect_results(local_results, len(local_results))
        
        # Only compute metrics on the main process with all collected results
        if is_main_process():
            runner.logger.info(f"Computing metrics on {len(all_results)} total samples across all ranks")
            metrics = self.pathology_metric.compute_metrics(all_results)
        else:
            metrics = {}
        
        # Clear local results for next evaluation
        self.pathology_metric.results = []
        return metrics

    @master_only
    def _log_metrics_to_wandb(self, runner, metrics):
        """Log metrics to wandb (only on main process)"""
        if not metrics:  # Skip if no metrics (happens on non-main processes)
            return
            
        if hasattr(runner, 'visualizer') and runner.visualizer is not None:
            # Add iteration info
            metrics['eval/iteration'] = runner.iter
            runner.visualizer.add_scalars(metrics)
        
        # Also log to runner logger
        for metric_name, value in metrics.items():
            runner.logger.info(f'{metric_name}: {value:.4f}')

    @master_only
    def _save_eval_output(self, runner, eval_outputs):
        """Save evaluation outputs to file"""
        save_path = os.path.join(runner.log_dir, 'vis_data',
                                 f'eval_outputs_iter_{runner.iter}.txt')
        mkdir_or_exist(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            for i, output in enumerate(eval_outputs):
                f.write(f'Eval output {i + 1}:\n{output}\n\n')

    def _evaluate_with_dataset(self, runner, model, device, save_eval_output=False):
        """Evaluate model using dataset and dataloader"""
        if save_eval_output:
            eval_outputs = []

        predictions = []
        targets = []

        # Set epoch for distributed sampler
        if hasattr(self.eval_dataloader.sampler, 'set_epoch'):
            self.eval_dataloader.sampler.set_epoch(runner.epoch if hasattr(runner, 'epoch') else 0)

        for batch_idx, data_batch in enumerate(self.eval_dataloader):
            data_batch = data_batch['data']
            # Move data to device
            for key, value in data_batch.items():
                if isinstance(value, torch.Tensor):
                    data_batch[key] = value.to(device)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    data_batch[key] = [v.to(device) for v in value]

            # Get ground truth if available
            ground_truth = self._decode_input_ids(data_batch['labels'][0], skip_special_tokens=True)
            targets.append(ground_truth)

            # Generate prediction using model's predict method
            try:
                # Use model's predict method for inference
                data_samples = model.forward(data_batch, mode='predict')
                generated_text = data_samples[0]['prediction_text']
                predictions.append(generated_text)
                
                # Log sample result (only for rank 0 to avoid spam)
                if get_rank() == 0:
                    input_text = self._decode_input_ids(data_batch['input_ids'][0], skip_special_tokens=False)
                    runner.logger.info(
                        "\n" +
                        "╔" + "═" * 60 + "╗\n"
                        "║" + " " * 21 + "EVAL EXAMPLE START" + " " * 21 + "║\n"
                        "╚" + "═" * 60 + "╝\n"
                        f"<<<Input>>>\n{input_text}\n<<</Input>>>\n"
                        f"<<<Prediction>>>\n{generated_text}\n<<</Prediction>>>\n"
                        f"<<<Ground Truth>>>\n{ground_truth}\n<<</Ground Truth>>>\n"
                        "╔" + "═" * 60 + "╗\n"
                        "║" + " " * 22 + "EVAL EXAMPLE END" + " " * 22 + "║\n"
                        "╚" + "═" * 60 + "╝\n"
                    )
                
                if save_eval_output:
                    input_text = self._decode_input_ids(data_batch['input_ids'][0], skip_special_tokens=False)
                    eval_outputs.append(f'{input_text}\n{generated_text}\n')
                    
            except Exception as e:
                runner.logger.error(f"Error during evaluation batch {batch_idx} on rank {get_rank()}: {e}")
                continue

        # Log local sample count for debugging
        runner.logger.info(f"Rank {get_rank()}: Processed {len(predictions)} samples")

        # Calculate metrics (PathologyMetric handles distributed collection internally)
        if targets and len(targets) == len(predictions):
            metrics = self._calculate_metrics(predictions, targets, runner)
            # Only log metrics on main process (metrics will be empty dict on other processes)
            if metrics:
                self._log_metrics_to_wandb(runner, metrics)

        if save_eval_output and eval_outputs:
            self._save_eval_output(runner, eval_outputs)

    def _generate_samples(self, runner, save_eval_output=False):
        """Main evaluation function using dataset approach"""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        if self.is_first_run:
            # Initialize dataset and dataloader
            self._create_eval_dataset(runner)
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to device
            model.to(device)
            self.is_first_run = False

        # Save current training state
        is_checkpointing = model.llm.is_gradient_checkpointing
        use_cache = model.llm.config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        model.llm.config.use_cache = True
        model.eval()
        
        # Run evaluation
        self._evaluate_with_dataset(runner, model, device, save_eval_output)

        # Restore training mode
        if is_checkpointing:
            model.activation_checkpointing_enable()
        model.llm.config.use_cache = use_cache
        model.train()

    def before_train(self, runner):
        """Run evaluation before training starts"""
        runner.logger.info('before_train in EvaluateChatHook.')
        self._generate_samples(runner)

    def _is_save_checkpoint(self, runner):
        """Check if this iteration should save checkpoint"""
        hooks = runner.hooks
        checkpoint_hook = None
        for hook in hooks:
            if type(hook).__name__ == 'CheckpointHook':
                checkpoint_hook = hook
                break
        if checkpoint_hook is None or checkpoint_hook.by_epoch:
            return False

        if checkpoint_hook.every_n_train_iters(
            runner, checkpoint_hook.interval, checkpoint_hook.save_begin) or \
                (checkpoint_hook.save_last and
                 checkpoint_hook.is_last_train_iter(runner)):
            return True

        return False

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        """Run evaluation after training iteration"""
        if self.every_n_iters is None:
            return

        save_eval_output = self._is_save_checkpoint(runner)

        do_chat = (
            save_eval_output
            or self.every_n_train_iters(runner, self.every_n_iters))
        if not do_chat:
            return

        runner.logger.info('after_train_iter in EvaluateChatHook.')
        self._generate_samples(runner, save_eval_output=save_eval_output)

    def after_train(self, runner):
        """Run evaluation after training completes"""
        runner.logger.info('after_train in EvaluateChatHook.')
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        """Run evaluation after validation"""
        if self.every_n_iters is not None:
            return
        runner.logger.info('after_val in EvaluateChatHook.')
        self._generate_samples(runner)

    def _decode_input_ids(self,
                          input_ids,
                          image_token_index=IMAGE_TOKEN_INDEX,
                          default_image_token=DEFAULT_IMAGE_TOKEN,
                          ignore_index=IGNORE_INDEX,
                          skip_special_tokens=True):
        ids_list = input_ids.tolist()
        ids_list = [tid for tid in ids_list if tid != ignore_index]
        chunks = []
        current_chunk = []
        has_image_token = False
        for token_id in ids_list:
            if token_id == image_token_index:
                has_image_token = True
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            else:
                current_chunk.append(token_id)
        if current_chunk:
            chunks.append(current_chunk)
        decoded_parts = []
        if has_image_token:
            for i, chunk in enumerate(chunks):
                decoded = self.tokenizer.decode(chunk, skip_special_tokens=skip_special_tokens)
                decoded_parts.append(decoded)
                if i != len(chunks) - 1:
                    decoded_parts.append(default_image_token)
            return ''.join(decoded_parts)
        else:
            return self.tokenizer.decode(ids_list, skip_special_tokens=skip_special_tokens)