# Copyright (c) OpenMMLab. All rights reserved.
# xtuner/configs/llava/internlm2_chat_7b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain.py
import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.visualization import Visualizer, WandbVisBackend

from torch.optim import AdamW
from sophia import SophiaG 
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)
from peft import LoraConfig
from xtuner.dataset import LLaVADataset_conv_longnet
from xtuner.dataset.collate_fns import default_collate_fn, masked_collated_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook_conv_longnet, HFCheckpointHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel_conv_longnet
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.configs.slidechat.eval_samples import evaluation_images, evaluation_inputs, evaluation_targets
from xtuner.evaluation.metrics.pathology_metric import PathologyMetric
#######################################################################
#                          PART 1  Settings                           #
#######################################################################

llm_name_or_path = '/home/winglet/models/Qwen3-8B'
train_data_path = '/home/winglet/pathology/vqa/dataset_pp/PathoVerse_train_stage1_caption.json'
test_data_path = '/home/winglet/pathology/vqa/dataset_pp/PathoVerse_train_stage1_caption_test.json'
# ckpt_path = '/home/winglet/pathology/vqa/train_s1/iter_1.pth'
ckpt_path = None
work_dir = '/home/winglet/pathology/vqa/train_s1/'
test_output_path = work_dir + 'test_results'
print_n_samples_in_test = None
image_path_list = None

prompt_template = PROMPT_TEMPLATE.qwen_chat_no_think


max_length = 32768
max_patch_num = None
max_new_tokens = 256
repetition_penalty = 1.1
per_image_length = None
sample_type='wsi' # 'wsi'or'image'


# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 1
dataloader_num_workers = 1
max_epochs = 50
optim_type = SophiaG
lr = 2e-4
betas = (0.965, 0.999)
rho = 0.01
weight_decay = 1e-1
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right'
    )

# removed image_processor

model = dict(
    type=LLaVAModel_conv_longnet,
    tokenizer=tokenizer,
    freeze_llm=True,
    hidden_size=768,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # quantization_config=dict(
        #     type=BitsAndBytesConfig,
        #     load_in_4bit=True,
        #     load_in_8bit=False,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4')
    ),
    generation_kwargs=dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=repetition_penalty
    ),
    # llm_lora=dict(
    #     type=LoraConfig,
    #     r=64,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     bias='none',
    #     task_type='CAUSAL_LM')
    )

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_llava_dataset = dict(
    type=LLaVADataset_conv_longnet,
    data_path=train_data_path,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=per_image_length,
    mode='train')

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=masked_collated_fn))

#######################################################################
#                     Test Dataset & Dataloader                       #
#######################################################################
test_llava_dataset = dict(
    type=LLaVADataset_conv_longnet,
    data_path=test_data_path,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=per_image_length,
    mode='test',
    input_ids_with_output=False)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=test_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),  # Don't shuffle for test
    collate_fn=dict(type=masked_collated_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay, rho=rho),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

# Test configuration
test_cfg = dict(type="TestLoop")

# Test evaluator
test_evaluator = dict(
    type=PathologyMetric,
    tokenizer=tokenizer,
    print_first_n_samples=print_n_samples_in_test,
    output_dir=test_output_path,
    prefix="test"
)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook_conv_longnet,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        evaluation_targets=evaluation_targets,
        system=SYSTEM,
        max_new_tokens=max_new_tokens,
        prompt_template=prompt_template,
        max_patch_num=max_patch_num,
        generation_kwargs={'repetition_penalty': repetition_penalty,
                           'max_new_tokens': max_new_tokens,
                           'do_sample': True,
                           'temperature': 0.7,
                           'top_p': 0.8,
                           'top_k': 20,})
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None
# visualizer = dict(
#     type=Visualizer,
#     vis_backends=[
#         dict(
#             type=WandbVisBackend,
#             init_kwargs=dict(
#                 project='pathoverse_capgen',
#                 name='qwen3_8b_freeze_llm'
#             )
#         )
#     ]
# )

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = ckpt_path

# whether to resume training from the loaded checkpoint
resume = False
 
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
