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
#                          PART 1  Settings                           #
#######################################################################

setting = 'lora'

if setting == 'alignment':
    llm_lora = None
    freeze_llm = True
    lr = 2e-5  # Reduced from 1e-4 for better stability
    ckpt_path = None
    # ckpt_path = '/mnt/petrelfs/zhaoweike/project/TCGA/train_s2_multitask_qwen3_4b_conv_alignment_multitask_mcqa_srv/iter_16500.pth'
    max_epochs = 1
    save_best_metrics = None
if setting == 'lora':
    llm_lora = dict(
        type=LoraConfig,
        r=64,
        lora_alpha=64,
        lora_dropout=0.2,
        bias='none',
        task_type='CAUSAL_LM')
    # save_best_metrics = ['eval/mcqa_overall_accuracy', 'eval/reg_overall_r2', 'eval/surv_overall_survival_os_c_index']
    save_best_metrics = None
    ckpt_path = None
    # ckpt_path = '/mnt/petrelfs/zhaoweike/project/TCGA/8B_vl_multimodal_alignment_resnet/epoch_3.pth'
    lr = 2e-5
    freeze_llm = True
    max_epochs = 5
if setting == 'full_param':
    llm_lora = None
    freeze_llm = False
    lr = 1e-5
    save_best_metrics = ['eval/reg_overall_rmse']
    ckpt_path = '/mnt/petrelfs/zhaoweike/project/TCGA/train_s2_multitask_qwen3_4b_conv_alignment_rna_regression_multitask/iter_1000.pth'
    max_epochs = 25
    
resume = False

model_type = 'text_patch'  # Options: 'text_only', 'text_patch', 'text_patch_no_deepstack', 'text_wsi', 'text_patch_pooling', 'multimodal'
model_size = '9B'

# exp = 'noctx_notrt_xena_noaug-r1'
# exp = 'ctx_notrt_xena_aug-d0.5-r2'
# exp = 'ctx_notrt_xena_aug-d0.25-r1'
# exp = 'ctx_notrt_xena_aug-d0.75-r4'
# exp = 'ctx_trt_landmark_aug-d0.5-r2'
# exp = 'ctx_notrt_xena_aug-d0.2-k0.8-s0.1-r2-c1'

llm_name_or_path = f'/mnt/petrelfs/zhaoweike/hwfile_share/model/model_zoo/Qwen3.5-{model_size}'
# train_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/survival_generated_qa_{exp}/train.json'
# val_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/survival_generated_qa_{exp}/test.json'
# test_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/survival_generated_qa_{exp}/test.json'
dataset_cache_dir = '/mnt/petrelfs/zhaoweike/project/TCGA/.cache/'
train_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse/train.json'
val_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse/test.json'
test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse/test.json'

# train_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_train/supercategories/mcqa_mutation_train.json'
# val_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/mcqa_mutation_test.json'
# test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/mcqa_mutation_test.json'

# train_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_train/supercategories/regression__train.json'
# val_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/regression__test.json'
# test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/regression__test.json'


# ckpt_out_path = 's3://zhaoweike/ckpt'
ckpt_out_path = None

# work_dir = f'/mnt/petrelfs/zhaoweike/project/TCGA/{model_size}_vl_{model_type}_{setting}_{exp}/'
# vis_name = f'{model_size}_vl_{model_type}_{setting}_{exp}'
work_dir = f'/mnt/petrelfs/zhaoweike/project/TCGA/{model_size}_{model_type}_{setting}/'
vis_name = f'{model_size}_vl_{model_type}_{setting}'
# vis_name = None


# set visualizer
visualizer = None if vis_name is None else dict(
    type=Visualizer,
    vis_backends=[
        dict(
            type=WandbVisBackend,
            init_kwargs=dict(
                project='pathoverse_qwen3_5',
                name=vis_name
            )
        )
    ]
)

val_output_path = work_dir + 'val_results'
test_output_path = work_dir + 'test_results'

# Save
by_epoch = True
interval = 250
# interval = 1
save_total_limit = 3

# Evaluate the generation performance during the training
evaluation_freq = 1000  # More frequent evaluation for alignment debugging
image_path_list = None

prompt_template = PROMPT_TEMPLATE.qwen_chat


dataset_map_fn = llava_text_only_map_fn if model_type == 'text_only' else llava_map_fn


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
    latest_valid_ckpt = _get_latest_valid_deepspeed_checkpoint(work_dir, num_gpus=4)
    
    if latest_valid_ckpt:
        ckpt_path = latest_valid_ckpt
        print(f"Resuming from latest valid checkpoint: {ckpt_path}")
    else:
        print(f"Resume is True, but no complete checkpoints were found in {work_dir}. Starting from scratch.")
        
del _get_latest_valid_deepspeed_checkpoint

max_length = 256000
max_patch_num = None
max_new_tokens = 32
repetition_penalty = 1.0
per_image_length = None
sample_type='wsi' # 'wsi'or'image'


# Scheduler & Optimizer
batch_size = 8
accumulative_counts = 1
dataloader_num_workers = 8
optim_type = AdamW
betas = (0.9, 0.999)
rho = 0.01
weight_decay = 1e-1
max_norm = 1  # grad clip
warmup_ratio = 0.05


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
else:
    prompt_resampler_cfg = None

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
    prompt_context_mode='llm_hidden',
    prompt_context_layer=-1,
    wsi_feature_dims=wsi_feature_dims,
    wsi_dropout=0.1,
    head_scaling=[1, 1, 1],
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_llava_dataset = dict(
    type=LLaVADataset,
    data_path=train_data_path,
    cache_dir=dataset_cache_dir,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=dataset_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=per_image_length,
    mode='train',
    text_only=model_type == 'text_only',
    load_patch_features=model_type in ('text_patch', 'multimodal'),
    load_wsi_features=model_type in ('text_wsi', 'multimodal'))

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_llava_dataset,
    # sampler=dict(type=CategoryProjectSampler, batch_size=batch_size, shuffle=True),
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=masked_collated_fn))

val_llava_dataset = dict(
    type=LLaVADataset,
    data_path=val_data_path,
    cache_dir=dataset_cache_dir,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=dataset_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=per_image_length,
    mode='test',
    input_ids_with_output=True,
    text_only=model_type == 'text_only',
    load_patch_features=model_type in ('text_patch', 'multimodal'),
    load_wsi_features=model_type in ('text_wsi', 'multimodal'))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=val_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=masked_collated_fn))

val_evaluator = dict(type=PathologyMetric,
            tokenizer=tokenizer,
            output_dir= val_output_path
            )

test_llava_dataset = dict(
    type=LLaVADataset,
    data_path=test_data_path,
    cache_dir=dataset_cache_dir,
    image_folder='',
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    dataset_map_fn=dataset_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=per_image_length,
    mode='test',
    input_ids_with_output=True,
    text_only=model_type == 'text_only',
    load_patch_features=model_type in ('text_patch', 'multimodal'),
    load_wsi_features=model_type in ('text_wsi', 'multimodal'))

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=test_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=masked_collated_fn)
)

test_evaluator = dict(type=PathologyMetric,
            tokenizer=tokenizer,
            output_dir=test_output_path
            )

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        # type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay, rho=rho),
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16')

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
train_cfg = dict(type=TrainLoop,
                 max_epochs=max_epochs,
                 val_interval=evaluation_freq)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type="TestLoop")

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer)
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
        by_epoch=by_epoch,
        interval=interval,
        max_keep_ckpts=save_total_limit,
        save_best=save_best_metrics,
        rule='greater',
        out_dir=ckpt_out_path,),
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


# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = ckpt_path
 
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=42, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
