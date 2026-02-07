# Copyright (c) OpenMMLab. All rights reserved.
# Config for Qwen3 text-only model with optional vision support (unified architecture)
# Supports: Qwen3-4B, Qwen3-8B (text-only, text+WSI, text+vision+WSI)
import torch
from mmengine.dataset import DefaultSampler, InfiniteSampler
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
from xtuner.dataset.collate_fns import masked_collated_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel_conv_unified
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.configs.slidechat.eval_samples import evaluation_images, evaluation_inputs, evaluation_targets
from xtuner.evaluation.metrics.pathology_metric import PathologyMetric

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

setting = 'lora'

if setting == 'alignment':
    llm_lora = None
    freeze_llm = True
    lr = 2e-5
    ckpt_path = None
    max_epochs = 10
    save_best_metrics = None
if setting == 'lora':
    llm_lora = dict(
        type=LoraConfig,
        r=64,
        lora_alpha=64,
        lora_dropout=0.2,
        bias='none',
        task_type='CAUSAL_LM')
    save_best_metrics = None
    ckpt_path = None
    # ckpt_path = '/mnt/petrelfs/zhouxiao/project/TCGA/train_s2_qwen3_unified_text_patch_alignment/iter_100_cp.pth'
    lr = 2e-5
    freeze_llm = True
    max_epochs = 10
if setting == 'full_param':
    llm_lora = None
    freeze_llm = False
    lr = 1e-5
    save_best_metrics = ['eval/reg_overall_rmse']
    ckpt_path = None
    max_epochs = 25
    
resume = False

# =====================================================================
# MODALITY SELECTION (use same full-modal data, control via config only)
# =====================================================================
# You can use the SAME full-modal dataset (containing <image> tokens, 
# patch features, and WSI features) with any of these modes.
# The model automatically handles unused modalities:
#
# MODEL MODES (Qwen3-4B/8B text-only backbone):
# ┌──────────────┬──────────────────┬────────────────┬─────────────────────┐
# │ Mode         │ vision_conv_cfg  │ wsi_feature_dims│ Behavior           │
# ├──────────────┼──────────────────┼────────────────┼─────────────────────┤
# │ text_only    │ None             │ None           │ Pure text, strips   │
# │              │                  │                │ <image> tokens      │
# ├──────────────┼──────────────────┼────────────────┼─────────────────────┤
# │ text_wsi     │ None             │ [768,1024,...] │ Text + WSI features,│
# │              │                  │                │ strips <image> tok  │
# ├──────────────┼──────────────────┼────────────────┼─────────────────────┤
# │ text_patch   │ {...}            │ None           │ Text + patch visual │
# │              │                  │                │ uses <image> tokens │
# ├──────────────┼──────────────────┼────────────────┼─────────────────────┤
# │ multimodal   │ {...}            │ [768,1024,...] │ Full: Text+Patch+WSI│
# └──────────────┴──────────────────┴────────────────┴─────────────────────┘
#
# Data format: Your JSON data can contain all modalities:
#   - "image": path to patch features (.h5 file)
#   - "wsi_features": paths to WSI global features [titan.pt, prism.pt, ...]
#   - "conversations": with or without <image> tokens in text
#
# The model will automatically skip unused modalities based on config.
# =====================================================================
model_type = 'text_patch'  # Options: 'text_only', 'text_wsi', 'text_patch', 'multimodal'

# Model paths - change to Qwen3-4B or Qwen3-8B as needed
llm_name_or_path = '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B'

# Data paths (same full-modal data for all modes)
train_data_path = '/mnt/petrelfs/zhouxiao/project/TCGA/dataset_pp/data_pipeline/tcga_train/tcga_aligned_survival_survival_os_context.json'
val_data_path = '/mnt/petrelfs/zhouxiao/project/TCGA/dataset_pp/data_pipeline/tcga_test/tcga_aligned_survival_survival_os_context.json'
test_data_path = '/mnt/petrelfs/zhouxiao/project/TCGA/dataset_pp/data_pipeline/tcga_test/tcga_aligned_survival_survival_os_context.json'

# Output paths
ckpt_out_path = None
work_dir = f'/mnt/petrelfs/zhouxiao/project/TCGA/train_s2_qwen3_unified_{model_type}_{setting}'
vis_name = f'qwen3_8b_lm_multitask_mcqa_srv_random_discrete_dynamic_{model_type}_{setting}'
# vis_name = None

visualizer = None if vis_name is None else dict(
    type=Visualizer,
    vis_backends=[
        dict(
            type=WandbVisBackend,
            init_kwargs=dict(
                project='pathoverse_multitask_conv_srv_debug',
                name=vis_name
            )
        )
    ]
)

val_output_path = work_dir + '/val_results'
test_output_path = work_dir + '/test_results'

# Save settings
save_steps = 100
save_total_limit = 1

# Evaluation frequency
evaluation_freq = 50

image_path_list = None
prompt_template = PROMPT_TEMPLATE.qwen_chat


def _get_latest_valid_deepspeed_checkpoint(work_dir, num_gpus=8):
    import os
    import glob
    import re
    if not os.path.isdir(work_dir):
        return None
    all_ckpt_dirs = [p for p in glob.glob(os.path.join(work_dir, 'iter_*.pth')) if os.path.isdir(p)]
    
    if not all_ckpt_dirs:
        return None

    try:
        sorted_ckpts = sorted(
            all_ckpt_dirs,
            key=lambda p: int(re.search(r'iter_(\d+)\.pth', os.path.basename(p)).group(1)),
            reverse=True
        )
    except (AttributeError, ValueError):
        print("Warning: Found directories with malformed names, skipping them.")
        return None

    expected_file_count = num_gpus + 1
    
    for ckpt_dir in sorted_ckpts:
        try:
            if len(os.listdir(ckpt_dir)) == expected_file_count:
                return ckpt_dir
            else:
                print(f"Warning: Checkpoint '{os.path.basename(ckpt_dir)}' is incomplete. Skipping.")
        except OSError as e:
            print(f"Warning: Could not access checkpoint '{os.path.basename(ckpt_dir)}'. Error: {e}. Skipping.")
            continue
    return None

if resume:
    latest_valid_ckpt = _get_latest_valid_deepspeed_checkpoint(work_dir, num_gpus=8)
    
    if latest_valid_ckpt:
        ckpt_path = latest_valid_ckpt
        print(f"Resuming from latest valid checkpoint: {ckpt_path}")
    else:
        print(f"Resume is True, but no complete checkpoints were found in {work_dir}. Starting from scratch.")
        
del _get_latest_valid_deepspeed_checkpoint

max_length = 32768
max_patch_num = None
max_new_tokens = 32
repetition_penalty = 1.0
per_image_length = None
sample_type = 'wsi'  # 'wsi' or 'image'


# Scheduler & Optimizer
batch_size = 8
accumulative_counts = 1
dataloader_num_workers = 8
optim_type = SophiaG
betas = (0.9, 0.999)
rho = 0.01
weight_decay = 1e-1
max_norm = 1
warmup_ratio = 0.03


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

# Configure model based on model_type
# Vision config - set to None for text-only modes
if model_type == 'text_only':
    vision_conv_cfg = None
    wsi_feature_dims = None
elif model_type == 'text_wsi':
    vision_conv_cfg = None  # No patch-level vision
    # WSI feature dimensions: [TITAN, PRISM, GigaPath, CHIEF]
    wsi_feature_dims = [768, 1280]
elif model_type == 'text_patch':
    vision_conv_cfg = {
        "in_chans": 768,
        "depths": [3, 9, 3],
        "dims": [768, 1024, 2048],
        "drop_path_rate": 0.3,
        "num_downsamples": 2,
    }
    wsi_feature_dims = None  # No WSI features
elif model_type == 'multimodal':
    vision_conv_cfg = {
        "in_chans": 768,
        "depths": [3, 9, 3],
        "dims": [768, 1024, 2048],
        "drop_path_rate": 0.3,
        "num_downsamples": 2,
    }
    wsi_feature_dims = [768, 1280]
else:
    raise ValueError(f"Unknown model_type: {model_type}. Options: 'text_only', 'text_wsi', 'text_patch', 'multimodal'")

model = dict(
    type=LLaVAModel_conv_unified,
    tokenizer=tokenizer,
    freeze_llm=freeze_llm,
    hidden_size=4096,  # Adjust based on model size (2560 for 4B, 4096 for 8B)
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ),
    generation_kwargs=dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
    ),
    llm_lora=llm_lora,
    enable_regression=True,
    enable_survival=True,
    reg_token='<REG>',
    srv_token='<SRV>',
    num_survival_intervals=6,
    survival_method='discrete',  # 'cox' or 'discrete'
    gen_forcing = False,
    lambda_llm=1.0,
    lambda_reg=1.0,
    lambda_srv=1.0,
    head_scaling=[1, 1, 1],  # [reg_mult, srv_mult, wsi_mult]
    vision_conv_cfg=vision_conv_cfg,
    wsi_feature_dims=wsi_feature_dims,
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

val_llava_dataset = dict(
    type=LLaVADataset_conv_longnet,
    data_path=val_data_path,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=per_image_length,
    mode='test',
    input_ids_with_output=True)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=val_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=masked_collated_fn))

val_evaluator = dict(type=PathologyMetric,
            tokenizer=tokenizer,
            output_dir=val_output_path)

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
    input_ids_with_output=True)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=test_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=masked_collated_fn))

test_evaluator = dict(type=PathologyMetric,
            tokenizer=tokenizer,
            output_dir=test_output_path)

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
train_cfg = dict(type=TrainLoop,
                 max_epochs=max_epochs,
                 val_interval=evaluation_freq)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type="TestLoop")

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer)
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
        save_best=save_best_metrics,
        rule='greater',
        out_dir=ckpt_out_path,),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'
load_from = ckpt_path
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
