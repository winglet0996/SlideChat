# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 multitask config with prompt-conditioned WSI patch resampling."""
from os import listdir
from os.path import basename, isdir, isfile, join

from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.visualization import Visualizer, WandbVisBackend
from peft import LoraConfig
from torch import bfloat16
from torch.optim import AdamW
from transformers import AutoModelForImageTextToText, AutoTokenizer

from xtuner.configs.slidechat.eval_samples import evaluation_images, evaluation_inputs, evaluation_targets
from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import masked_collated_fn
from xtuner.dataset.map_fns import llava_map_fn, llava_text_only_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import EffectiveBalancedSampler
from xtuner.engine.hooks import DatasetInfoHook, ModalityDropoutSchedulerHook
from xtuner.engine.runner import TrainLoop
from xtuner.evaluation.metrics.pathology_metric import PathologyMetric
from xtuner.model import LLaVAModel_qwen3_5
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

setting = 'lora'

# Ablation knobs. Update these together for patch/WSI/position-encoding runs.
ablation_version = 'v12'
patch_keep_tokens = 8
wsi_feature_source = ('titan', 'prism', 'gigapath', 'chief')
patch_position_encoding = 'linear'  # 'linear' or 'mrope'
head_scaling = (0.5, 0, 0.5) # regression, survival, wsi_projector
lora_r = 128
lora_alpha = 128
run_suffix = 'lr2e-5_sampler'

if setting == 'alignment':
    llm_lora = None
    freeze_llm = True
    lr = 1e-4  # Reduced from 1e-4 for better stability
    ckpt_path = None
    # ckpt_path = '/mnt/petrelfs/zhaoweike/project/TCGA/9B_multimodal_alignment_wo_knowledge_v12_4token_keep_4wsi_dropout20_sampler/iter_3000.pth'
    max_epochs = 3
    save_best_metrics = None
if setting == 'lora':
    llm_lora = dict(
        type=LoraConfig,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.2,
        bias='none',
        task_type='CAUSAL_LM')
    # save_best_metrics = ['eval/mcqa_overall_accuracy', 'eval/reg_overall_r2', 'eval/surv_overall_survival_os_c_index']
    save_best_metrics = None
    # ckpt_path = None
    ckpt_path = '/mnt/petrelfs/zhaoweike/project/TCGA/9B_multimodal_alignment_wo_knowledge_v12_8token_4wsi_linear_sampler/iter_19440.pth'
    lr = 2e-5
    freeze_llm = True
    max_epochs = 3
if setting == 'full_param':
    llm_lora = None
    freeze_llm = False
    lr = 1e-5
    save_best_metrics = ['eval/reg_overall_rmse']
    ckpt_path = '/mnt/petrelfs/zhaoweike/project/TCGA/train_s2_multitask_qwen3_4b_conv_alignment_rna_regression_multitask/iter_1000.pth'
    max_epochs = 25
    
resume = False

model_type = 'multimodal'  # Options: 'text_only', 'text_patch', 'text_patch_no_deepstack', 'text_wsi', 'text_patch_pooling', 'multimodal'
model_size = '9B'
kg_status='wo_knowledge'

llm_name_or_path = f'/mnt/petrelfs/zhaoweike/hwfile_share/model/model_zoo/Qwen3.5-{model_size}'
# train_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/survival_generated_qa_{exp}/train.json'
# val_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/survival_generated_qa_{exp}/test.json'
# test_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/survival_generated_qa_{exp}/test.json'
dataset_cache_dir = '/mnt/petrelfs/zhaoweike/project/TCGA/.cache/'
# train_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse_{kg_status}_r2/train.json'
# val_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse_{kg_status}_r2/test.json'
# test_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse_{kg_status}_r2/test.json'

train_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse_wo_knowledge_r2/chief_gigapath_prism_titan/train.json'
val_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse_wo_knowledge_r2/chief_gigapath_prism_titan/test.json'
test_data_path = f'/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline_v2/pathoverse_wo_knowledge_r2/chief_gigapath_prism_titan/test.json'

# train_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_train/supercategories/mcqa_mutation_train.json'
# val_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/mcqa_mutation_test.json'
# test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/mcqa_mutation_test.json'

# train_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_train/supercategories/regression__train.json'
# val_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/regression__test.json'
# test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/data_pipeline/tcga_test/supercategories/regression__test.json'


# ckpt_out_path = 's3://zhaoweike/ckpt'
ckpt_out_path = None

wsi_feature_specs = dict(
    titan=dict(field='slide_features_titan', dim=768),
    prism=dict(field='slide_features_prism', dim=1280),
    chief=dict(field='slide_features_chief', dim=768),
    gigapath=dict(field='slide_features_gigapath', dim=768),
)
if isinstance(wsi_feature_source, str):
    wsi_feature_source = (wsi_feature_source,)
else:
    wsi_feature_source = tuple(wsi_feature_source)
valid_wsi_feature_sources = tuple(wsi_feature_specs)
unknown_wsi_feature_sources = [src for src in wsi_feature_source if src not in wsi_feature_specs]
if unknown_wsi_feature_sources:
    raise ValueError(
        f"Unknown wsi_feature_source: {unknown_wsi_feature_sources!r}. "
        f"Expected subset of {valid_wsi_feature_sources!r}")
if not wsi_feature_source:
    raise ValueError('wsi_feature_source must include at least one source.')
if patch_position_encoding not in ('linear', 'mrope'):
    raise ValueError("patch_position_encoding must be 'linear' or 'mrope'.")

wsi_source_tag = '4wsi' if len(wsi_feature_source) == 4 else '-'.join(wsi_feature_source)
head_scaling_tag = 'hs_' + '_'.join(str(scale).replace('.', 'p') for scale in head_scaling)
ablation_tag = f'{patch_keep_tokens}token_{wsi_source_tag}_{patch_position_encoding}_{head_scaling_tag}'
if setting == 'lora':
    ablation_tag = f'{ablation_tag}_lora_r{lora_r}_a{lora_alpha}'
exp_tag = f'{ablation_version}_{ablation_tag}'
if run_suffix:
    exp_tag = f'{exp_tag}_{run_suffix}'

# work_dir = f'/mnt/petrelfs/zhaoweike/project/TCGA/{model_size}_vl_{model_type}_{setting}_{exp}/'
# vis_name = f'{model_size}_vl_{model_type}_{setting}_{exp}'
work_dir = f'/mnt/petrelfs/zhaoweike/project/TCGA/{model_size}_{model_type}_{setting}_{kg_status}_{exp_tag}/'
vis_name = f'{model_size}_{model_type}_{setting}_{kg_status}_{exp_tag}'
# vis_name = None


# set visualizer
visualizer = None if vis_name is None else dict(
    type=Visualizer,
    vis_backends=[
        dict(
            type=WandbVisBackend,
            init_kwargs=dict(
                project='pathoverse_qwen3_5_r2',
                name=vis_name
            )
        )
    ]
)

val_output_path = work_dir + 'val_results'
test_output_path = work_dir + 'test_results'

# Save
by_epoch = False
interval = 1000
# interval = 1
save_total_limit = 10

# Evaluate the generation performance during the training
evaluation_freq = 1000  # More frequent evaluation for alignment debugging
image_path_list = None

prompt_template = PROMPT_TEMPLATE.qwen_chat


dataset_map_fn = llava_text_only_map_fn if model_type == 'text_only' else llava_map_fn


def _get_latest_valid_deepspeed_checkpoint(work_dir, num_gpus=8):
    import glob
    import re
    if not isdir(work_dir):
        return None
    all_ckpt_dirs = [p for p in glob.glob(join(work_dir, 'iter_*.pth')) if isdir(p)]
    
    if not all_ckpt_dirs:
        return None

    try:
        sorted_ckpts = sorted(
            all_ckpt_dirs,
            key=lambda p: int(re.search(r'iter_(\d+)\.pth', basename(p)).group(1)),
            reverse=True
        )
    except (AttributeError, ValueError):
        print("Warning: Found directories with malformed names, skipping them.")
        return None

    expected_file_count = num_gpus + 1
    
    for ckpt_dir in sorted_ckpts:
        try:
            if len(listdir(ckpt_dir)) == expected_file_count:
                return ckpt_dir
            else:
                print(f"Warning: Checkpoint '{basename(ckpt_dir)}' is incomplete. Skipping.")
        except OSError as e:
            print(f"Warning: Could not access checkpoint '{basename(ckpt_dir)}'. Error: {e}. Skipping.")
            continue
    return None

if resume:
    latest_valid_ckpt = _get_latest_valid_deepspeed_checkpoint(work_dir, num_gpus=24)
    
    if latest_valid_ckpt:
        ckpt_path = latest_valid_ckpt
        print(f"Resuming from latest valid checkpoint: {ckpt_path}")
    else:
        print(f"Resume is True, but no complete checkpoints were found in {work_dir}. Starting from scratch.")
        
del _get_latest_valid_deepspeed_checkpoint

pretrained_pth = None
if not resume and ckpt_path is not None:
    if isdir(ckpt_path):
        pretrained_pth = join(ckpt_path, 'mp_rank_00_model_states.pt')
    else:
        pretrained_pth = ckpt_path

    if not isfile(pretrained_pth):
        raise FileNotFoundError(
            f'Warm-start checkpoint file not found: {pretrained_pth}')

max_length = 256000
max_patch_num = None
max_new_tokens = 32
repetition_penalty = 1.0
per_image_length = None
sample_type='wsi' # 'wsi'or'image'

# Data worker settings
preprocess_num_workers = 8
dataloader_num_workers = 8

# EffectiveBalancedSampler settings
sampler_family_weights = dict(
    mcqa=0.35,
    regression=0.62,
    survival=0.03,
)
sampler_size_alpha = 0.5
sampler_original_mix_by_family = dict(
    mcqa=0.6,
    regression=1.0,
    survival=1.0,
)
sampler_balance_mcqa_labels = True
sampler_label_balance_power = 0.5
sampler_min_group_size_for_label_balance = 32
sampler_tiny_group_size = 32
sampler_tiny_group_max_fraction = 0.05
sampler_max_unit_repeats_per_epoch = 3
sampler_survival_unit = 'patient'
sampler_default_unit = 'slide'
sampler_mix_within_batch = True

# Scheduler & Optimizer
batch_size = 16
accumulative_counts = 1
optim_type = AdamW
betas = (0.9, 0.999)
rho = 0.01
weight_decay = 1e-1
max_norm = 1  # grad clip
warmup_ratio = 0.1
patch_modality_dropout_start = 0
wsi_modality_dropout_start = 0.8
modality_dropout_begin_ratio = 0.2
modality_dropout_end_ratio = 0.5


SYSTEM = ''

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
)


if model_type in ('text_patch', 'multimodal'):
    prompt_resampler_cfg = dict(
        patch_dim=768,
        resampler_dim=2048,
        num_query=patch_keep_tokens,
        num_layers=2,
        num_heads=16,
        dropout=0.2,
        use_local_conv=True,
        query_init_std=0.5,
    )
else:
    prompt_resampler_cfg = None

if model_type in ('text_wsi', 'multimodal'):
    if len(wsi_feature_source) == 1:
        wsi_feature_field = wsi_feature_specs[wsi_feature_source[0]]['field']
    else:
        wsi_feature_field = tuple(wsi_feature_specs[src]['field'] for src in wsi_feature_source)
    wsi_feature_dims = [wsi_feature_specs[src]['dim'] for src in wsi_feature_source]
else:
    wsi_feature_field = 'wsi_features'
    wsi_feature_dims = None

model = dict(
    type=LLaVAModel_qwen3_5,
    tokenizer=tokenizer,
    freeze_llm=freeze_llm,
    llm=dict(
        type=AutoModelForImageTextToText.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        dtype=bfloat16,
        attn_implementation='flash_attention_2',
        # attn_implementation='sdpa',
    ),
    generation_kwargs=dict(max_new_tokens=max_new_tokens, do_sample=False),
    stop_words=['<|im_end|>', '<|endoftext|>'],
    pretrained_pth=pretrained_pth,
    # Optional: warm-start patch_resampler from a baseline perceiver checkpoint
    # (3_linear_prob_v5_patch_baseline_perceiver.py -> unified_perceiver_*.pt).
    # Resampler hparams above must match the baseline run. None disables it.
    pretrained_patch_resampler=None,
    llm_lora=llm_lora,
    enable_regression=True,
    enable_survival=True,
    reg_token='<REG>',
    srv_token='<SRV>',
    survival_method='discrete',
    gen_forcing=False,
    num_survival_intervals=6,
    lambda_llm=8.0,
    lambda_reg=0.5,
    lambda_srv=0.03,
    prompt_resampler_cfg=prompt_resampler_cfg,
    prompt_context_mode='llm_hidden',
    # prompt_context_mode='embedding',
    prompt_context_layer='auto',
    patch_position_encoding=patch_position_encoding,
    enable_nonfinite_checks=False,
    wsi_feature_dims=wsi_feature_dims,
    wsi_dropout=0.3,  # Projector hidden dropout, not modality dropout.
    survival_head_dropout=0.6,
    head_scaling=list(head_scaling),
    patch_modality_dropout=0.2,
    wsi_modality_dropout=0.2,
    modality_dropout_allow_text_only=False,
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
    preprocess_num_workers=preprocess_num_workers,
    load_patch_features=model_type in ('text_patch', 'multimodal'),
    load_wsi_features=model_type in ('text_wsi', 'multimodal'),
    wsi_feature_field=wsi_feature_field)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_llava_dataset,
    # sampler=dict(type=CategoryProjectSampler, batch_size=batch_size, shuffle=True),
    sampler=dict(
        type=EffectiveBalancedSampler,
        batch_size=batch_size,
        shuffle=True,
        family_weights=sampler_family_weights,
        size_alpha=sampler_size_alpha,
        original_mix_by_family=sampler_original_mix_by_family,
        balance_mcqa_labels=sampler_balance_mcqa_labels,
        label_balance_power=sampler_label_balance_power,
        min_group_size_for_label_balance=sampler_min_group_size_for_label_balance,
        tiny_group_size=sampler_tiny_group_size,
        tiny_group_max_fraction=sampler_tiny_group_max_fraction,
        max_unit_repeats_per_epoch=sampler_max_unit_repeats_per_epoch,
        survival_unit=sampler_survival_unit,
        default_unit=sampler_default_unit,
        mix_within_batch=sampler_mix_within_batch,
        cache_dir=dataset_cache_dir,
        verbose=True),
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
    preprocess_num_workers=preprocess_num_workers,
    load_patch_features=model_type in ('text_patch', 'multimodal'),
    load_wsi_features=model_type in ('text_wsi', 'multimodal'),
    wsi_feature_field=wsi_feature_field)

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
    preprocess_num_workers=preprocess_num_workers,
    load_patch_features=model_type in ('text_patch', 'multimodal'),
    load_wsi_features=model_type in ('text_wsi', 'multimodal'),
    wsi_feature_field=wsi_feature_field)

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
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    # dict(
    #     type=ModalityDropoutSchedulerHook,
    #     patch_start_dropout=patch_modality_dropout_start,
    #     wsi_start_dropout=wsi_modality_dropout_start,
    #     begin_ratio=modality_dropout_begin_ratio,
    #     end_ratio=modality_dropout_end_ratio,
    # ),
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
load_from = ckpt_path if resume else None
 
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=42, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
