# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 multitask config with prompt-conditioned WSI patch resampling."""
import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.visualization import Visualizer, WandbVisBackend
from peft import LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForImageTextToText, AutoTokenizer

from xtuner.configs.slidechat.eval_samples import evaluation_images, evaluation_inputs, evaluation_targets
from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import masked_collated_fn
from xtuner.dataset.map_fns import llava_map_fn, llava_text_only_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.evaluation.metrics.pathology_metric import PathologyMetric
from xtuner.model import LLaVAModel_qwen3_5
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
# Settings
#######################################################################

setting = 'lora'
model_type = 'multimodal'  # text_only, text_patch, text_wsi, multimodal
model_size = '9B'

llm_lora = dict(
    type=LoraConfig,
    r=64,
    lora_alpha=64,
    lora_dropout=0.2,
    bias='none',
    task_type='CAUSAL_LM',
)
freeze_llm = True
lr = 2e-5
max_epochs = 10
resume = False
ckpt_path = None
save_best_metrics = None

model_zoo = '/mnt/petrelfs/zhaoweike/hwfile_share/model/model_zoo'
llm_name_or_path = f'{model_zoo}/Qwen3.5-{model_size}'

dataset_cache_dir = '/mnt/petrelfs/zhaoweike/project/TCGA/.cache/'
train_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_train/tcga_aligned_train_survival_os.json'
val_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/tcga_aligned_test_survival_os.json'
test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/tcga_aligned_test_survival_os.json'

work_dir = f'/mnt/petrelfs/zhaoweike/project/TCGA/train_s2_qwen3_5_{model_size}_{model_type}_{setting}_prompt_resampler/'
vis_name = f'qwen3_5_{model_size}_{model_type}_{setting}_prompt_resampler'
ckpt_out_path = None

visualizer = None if vis_name is None else dict(
    type=Visualizer,
    vis_backends=[
        dict(
            type=WandbVisBackend,
            init_kwargs=dict(project='pathoverse_srv', name=vis_name),
        )
    ],
)

val_output_path = work_dir + 'val_results'
test_output_path = work_dir + 'test_results'

by_epoch = True
interval = 250
save_total_limit = 1
evaluation_freq = 50
image_path_list = None
prompt_template = PROMPT_TEMPLATE.qwen_chat
dataset_map_fn = llava_text_only_map_fn if model_type == 'text_only' else llava_map_fn

max_length = 256000
max_patch_num = None
max_new_tokens = 32
per_image_length = None
sample_type = 'wsi'

batch_size = 8
accumulative_counts = 1
dataloader_num_workers = 8
optim_type = AdamW
betas = (0.9, 0.999)
weight_decay = 1e-1
max_norm = 1
warmup_ratio = 0.05

SYSTEM = ''

#######################################################################
# Model
#######################################################################

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
)

prompt_resampler_cfg = None
if model_type in ('text_patch', 'multimodal'):
    prompt_resampler_cfg = dict(
        patch_dim=768,
        resampler_dim=1024,
        num_region_tokens=128,
        num_visual_tokens=64,
        num_heads=8,
        dropout=0.1,
        use_local_conv=True,
    )

wsi_feature_dims = [768, 1280] if model_type in ('text_wsi', 'multimodal') else None

model = dict(
    type=LLaVAModel_qwen3_5,
    tokenizer=tokenizer,
    freeze_llm=freeze_llm,
    llm=dict(
        type=AutoModelForImageTextToText.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ),
    generation_kwargs=dict(max_new_tokens=max_new_tokens, do_sample=False),
    stop_words=['<|im_end|>', '<|endoftext|>'],
    llm_lora=llm_lora,
    enable_regression=True,
    enable_survival=True,
    reg_token='<REG>',
    srv_token='<SRV>',
    survival_method='discrete',
    gen_forcing=False,
    num_survival_intervals=6,
    lambda_llm=1.0,
    lambda_reg=1.0,
    lambda_srv=1.0,
    prompt_resampler_cfg=prompt_resampler_cfg,
    wsi_feature_dims=wsi_feature_dims,
    wsi_dropout=0.1,
    head_scaling=[1, 1, 1],
)

#######################################################################
# Dataset
#######################################################################

common_dataset_cfg = dict(
    type=LLaVADataset,
    cache_dir=dataset_cache_dir,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=dataset_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=per_image_length,
    text_only=model_type == 'text_only',
    load_patch_features=model_type in ('text_patch', 'multimodal'),
    load_wsi_features=model_type in ('text_wsi', 'multimodal'),
)

train_llava_dataset = dict(
    **common_dataset_cfg,
    data_path=train_data_path,
    mode='train',
)

val_llava_dataset = dict(
    **common_dataset_cfg,
    data_path=val_data_path,
    mode='test',
    input_ids_with_output=True,
)

test_llava_dataset = dict(
    **common_dataset_cfg,
    data_path=test_data_path,
    mode='test',
    input_ids_with_output=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=masked_collated_fn),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=val_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=masked_collated_fn),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=test_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=masked_collated_fn),
)

val_evaluator = dict(type=PathologyMetric, tokenizer=tokenizer, output_dir=val_output_path)
test_evaluator = dict(type=PathologyMetric, tokenizer=tokenizer, output_dir=test_output_path)

#######################################################################
# Runtime
#######################################################################

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16',
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=evaluation_freq)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [dict(type=DatasetInfoHook, tokenizer=tokenizer)]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=by_epoch,
        interval=interval,
        max_keep_ckpts=save_total_limit,
        save_best=save_best_metrics,
        rule='greater',
        out_dir=ckpt_out_path,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = ckpt_path
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)
