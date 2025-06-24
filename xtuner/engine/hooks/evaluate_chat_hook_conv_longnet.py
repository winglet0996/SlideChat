# Copyright (c) OpenMMLab. All rights reserved.
import os
import re
import warnings

import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils import mkdir_or_exist
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList
from torchvision import transforms
from xtuner.dataset.utils import load_wsi_feature, PadToGrid
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)
from xtuner.evaluation.metrics.pathology_metric import PathologyMetric

class EvaluateChatHook_conv_longnet(Hook):

    priority = 'LOW'

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_images=None,
                 evaluation_targets=None,  # Ground truth targets for metric calculation
                 image_processor=None,
                 system='',
                 prompt_template=None,
                 every_n_iters=None,
                 max_new_tokens=None,
                 stop_word=None,
                 stop_words=[],
                 generation_kwargs={},
                 max_patch_num=None):
        self.max_patch_num = max_patch_num

        # Define transform for WSI feature processing
        if self.max_patch_num is None:
            self.transform = transforms.Compose([
                PadToGrid(),
            ])
        else:
            self.transform = None

        self.evaluation_inputs = evaluation_inputs

        if isinstance(self.evaluation_inputs, str):
            self.evaluation_inputs = [self.evaluation_inputs]

        # Handle ground truth targets
        self.evaluation_targets = evaluation_targets
        if self.evaluation_targets is not None:
            if isinstance(self.evaluation_targets, str):
                self.evaluation_targets = [self.evaluation_targets]
            assert len(self.evaluation_targets) == len(self.evaluation_inputs)

        self.evaluation_images = evaluation_images
        if isinstance(self.evaluation_images, str):
            self.evaluation_images = [self.evaluation_images]
        if self.evaluation_images is not None:
            assert len(
                self.evaluation_images) in [1, len(self.evaluation_inputs)]
            if len(self.evaluation_images) == 1:
                self.evaluation_images = [self.evaluation_images[0]] * len(
                    self.evaluation_inputs)
            
            # Load WSI features and handle potential tuple returns
            loaded_features = []
            for img in self.evaluation_images:
                try:
                    feature_data = load_wsi_feature(img, self.max_patch_num, self.transform)
                    loaded_features.append(feature_data)
                        
                except Exception as e:
                    print(f"Error loading WSI feature from {img}: {e}")
                    raise e
                    
            self.evaluation_images = loaded_features
        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
            stop_words += prompt_template.get('STOP_WORDS', [])
        if stop_word is not None:
            # TODO: deprecation, v0.3.0
            warnings.warn(
                ('The `stop_word` argument is deprecated and will be removed '
                 'in v0.3.0, use `stop_words` instead.'), DeprecationWarning)
            stop_words.append(stop_word)
        self.instruction = instruction
        self.system = system
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        if image_processor is not None:
            self.image_processor = BUILDER.build(image_processor)
        self.stop_criteria = StoppingCriteriaList()

        # default generation config
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

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))

        self.pathology_metric = PathologyMetric(tokenizer=tokenizer)
        self.is_first_run = True
    
    @master_only
    def _calculate_metrics(self, predictions, targets):
        """Calculate metrics using PathologyMetric"""
        self.pathology_metric.process_predictions_and_targets(predictions, targets)
        results = self.pathology_metric.results
        metrics = self.pathology_metric.compute_metrics(results)
        self.pathology_metric.results = []
        
        return metrics

    @master_only
    def _log_metrics_to_wandb(self, runner, metrics):
        """Log metrics to wandb"""
        if hasattr(runner, 'visualizer') and runner.visualizer is not None:
            # Add iteration info
            metrics['eval/iteration'] = runner.iter
            runner.visualizer.add_scalars(metrics)
        
        # Also log to runner logger
        for metric_name, value in metrics.items():
            runner.logger.info(f'{metric_name}: {value:.4f}')
 

    @master_only
    def _save_eval_output(self, runner, eval_outputs):
        save_path = os.path.join(runner.log_dir, 'vis_data',
                                 f'eval_outputs_iter_{runner.iter}.txt')
        mkdir_or_exist(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            for i, output in enumerate(eval_outputs):
                f.write(f'Eval output {i + 1}:\n{output}\n\n')

    def _eval_images(self,
                     runner,
                     model,
                     device,
                     max_new_tokens=None,
                     save_eval_output=False):
        """Evaluate model on image inputs by reusing model's forward methods"""
        if save_eval_output:
            eval_outputs = []

        predictions = []

        for sample_idx, (sample_image, sample_input) in enumerate(zip(self.evaluation_images,
                                                                       self.evaluation_inputs)):
            if runner.cfg.sample_type=='wsi':
                # assert tuple here
                assert isinstance(sample_image, tuple) and len(sample_image) == 2, \
                    "For WSI evaluation, sample_image should be a tuple (feature, mask)"
                feature, mask = sample_image
                # Prepare features and masks for model input
                features = feature.unsqueeze(0).to(device)  # (1, C, H, W)
                masks = mask.unsqueeze(0).to(device)  # (1, 1, H, W)
                
                sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
                inputs = (self.system + self.instruction).format(
                    input=sample_input, round=1, **runner.cfg)
                input_ids = self._tokenize_with_image_token(inputs, device)
                data_batch = {
                    'input_ids': input_ids.unsqueeze(0),
                    'features': features,  # Use 'features' instead of 'pixel_values'
                    'masks': masks,
                    'attention_mask': torch.ones_like(input_ids).unsqueeze(0),
                    'labels': torch.full_like(input_ids, -100).unsqueeze(0)  # Add dummy labels for predict method
                }
                
                # Use model's predict method for consistent inference
                data_samples = model.forward(data_batch, mode='predict')
                generated_text = data_samples[0]['prediction_text']

                predictions.append(generated_text)
                
                gt = self.evaluation_targets[sample_idx] if self.evaluation_targets else "N/A"

                runner.logger.info(
                    "\n" +
                    "╔" + "═" * 60 + "╗\n"
                    "║" + " " * 21 + "EVAL EXAMPLE START" + " " * 21 + "║\n"
                    "╚" + "═" * 60 + "╝\n"
                    f"<<<Input>>>\n{inputs}\n<<</Input>>>\n"
                    f"<<<Prediction>>>\n{generated_text}\n<<</Prediction>>>\n"
                    f"<<<Ground Truth>>>\n{gt}\n<<</Ground Truth>>>\n"
                    "╔" + "═" * 60 + "╗\n"
                    "║" + " " * 22 + "EVAL EXAMPLE END" + " " * 22 + "║\n"
                    "╚" + "═" * 60 + "╝\n"
                )
            
                if save_eval_output:
                    eval_outputs.append(f'{inputs + generated_text}\n')

        # Calculate metrics if targets are provided
        if self.evaluation_targets:
            metrics = self._calculate_metrics(predictions, self.evaluation_targets)
            self._log_metrics_to_wandb(runner, metrics)

        if save_eval_output:
            self._save_eval_output(runner, eval_outputs)
            
    def _generate_samples(self,
                          runner,
                          max_new_tokens=None,
                          save_eval_output=False):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(iter(model.parameters())).device

        if self.is_first_run:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            model.to(device)
            self.is_first_run = False

        is_checkpointing = model.llm.is_gradient_checkpointing
        use_cache = model.llm.config.use_cache

        # Cast to inference mode
        model.activation_checkpointing_disable()
        model.llm.config.use_cache = True
        model.eval()
        if self.evaluation_images is not None:
            self._eval_images(runner, model, device, max_new_tokens,
                              save_eval_output)
        else:
            self._eval_language(runner, model, device, max_new_tokens,
                                save_eval_output)

        # Cast to training mode
        if is_checkpointing:
            model.activation_checkpointing_enable()
        model.llm.config.use_cache = use_cache
        model.train()

    def before_train(self, runner):
        runner.logger.info('before_train in EvaluateChatHook.')
        self._generate_samples(runner, max_new_tokens=self.max_new_tokens) 

    def _is_save_checkpoint(self, runner):
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

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
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
        runner.logger.info('after_train in EvaluateChatHook.')
        self._generate_samples(runner)

    def after_val(self, runner) -> None:
        if self.every_n_iters is not None:
            return
        runner.logger.info('after_val in EvaluateChatHook.')
        self._generate_samples(runner)

    def _tokenize_with_image_token(self, inputs, device):
        """Tokenize input text with proper image token handling
        
        Reusable helper to avoid duplicating tokenization logic from model.
        """
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        
        input_ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            input_ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                input_ids.append(IMAGE_TOKEN_INDEX)
        
        return torch.tensor(input_ids).to(device)