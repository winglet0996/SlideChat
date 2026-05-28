# Copyright (c) OpenMMLab. All rights reserved.
"""Official SlideChat LongNet baseline on current H5/json evaluation pipeline."""

from mmengine.dataset import DefaultSampler
from mmengine.hooks import DistSamplerSeedHook, IterTimerHook, LoggerHook
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import LLaVADataset_longnet
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.evaluation.metrics.pathology_metric import PathologyMetric
from xtuner.model import LLaVAModel_longnet
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

llm_name_or_path = '/mnt/petrelfs/zhaoweike/hwfile_share/model/model_zoo/Qwen2.5-7B-Instruct'
# test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/slidechat_dataset/slidechat_mcqa_conch_test.json'
test_data_path = '/mnt/petrelfs/zhaoweike/project/TCGA/dataset_pp/slidechat_dataset/SlideBench_test_aligned_conch.json'
dataset_cache_dir = '/mnt/petrelfs/zhaoweike/project/TCGA/.cache/'
work_dir = '/mnt/petrelfs/zhaoweike/project/TCGA/baseline/slidechat_original_longnet_slidebench_test'
test_output_path = work_dir + '/test_results'

prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 19600
max_patch_num = 10240
max_new_tokens = 500
batch_size = 1
dataloader_num_workers = 2

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=LLaVAModel_longnet,
    tokenizer=tokenizer,
    freeze_llm=False,
    hidden_size=512,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    generation_kwargs=dict(
        max_new_tokens=max_new_tokens,
        do_sample=False),
    stop_words=['<|im_end|>', '<|endoftext|>'])

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

test_llava_dataset = dict(
    type=LLaVADataset_longnet,
    data_path=test_data_path,
    cache_dir=dataset_cache_dir,
    image_folder='',
    image_path_list=None,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    max_patch_num=max_patch_num,
    per_image_length=None,
    mode='test',
    input_ids_with_output=True)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=test_llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=default_collate_fn))

test_evaluator = dict(
    type=PathologyMetric,
    tokenizer=tokenizer,
    output_dir=test_output_path)

#######################################################################
#                           PART 4  Runtime                           #
#######################################################################

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)
