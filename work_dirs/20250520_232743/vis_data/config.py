SYSTEM = ''
accumulative_counts = 1
batch_size = 1
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path=
            '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.DatasetInfoHook'),
    dict(
        evaluation_images=
        '/mnt/f/TCGA_KEEP_features1024/TCGA-GBM/TCGA-08-0347-01Z-00-DX1.CA3BDEAA-1F85-43A9-99D8-9FB7ED6BCDBD.h5',
        evaluation_inputs=[
            'Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.',
        ],
        every_n_iters=1000,
        prompt_template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
        system='',
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path=
            '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.EvaluateChatHook'),
]
data_path = '/mnt/petrelfs/zhouxiao/project/TCGA/dataset_pp/PathoVerse_train_stage1_caption.json'
dataloader_num_workers = 2
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=2,
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=10,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation_freq = 1000
evaluation_images = '/mnt/f/TCGA_KEEP_features1024/TCGA-GBM/TCGA-08-0347-01Z-00-DX1.CA3BDEAA-1F85-43A9-99D8-9FB7ED6BCDBD.h5'
evaluation_inputs = [
    'Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.',
]
image_path_list = None
launcher = 'none'
llava_dataset = dict(
    data_path=
    '/mnt/petrelfs/zhouxiao/project/TCGA/dataset_pp/PathoVerse_train_stage1_caption.json',
    dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
    image_folder='',
    image_path_list=None,
    max_length=51200,
    pad_image_to_square=False,
    per_image_length=None,
    template_map_fn=dict(
        template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
        type='xtuner.dataset.map_fns.template_map_fn_factory'),
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path=
        '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B',
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'),
    type='xtuner.dataset.LLaVADataset')
llm_name_or_path = '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 0.001
max_epochs = 1
max_length = 51200
max_norm = 1
model = dict(
    freeze_llm=True,
    llm=dict(
        pretrained_model_name_or_path=
        '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B',
        quantization_config=dict(
            bnb_4bit_compute_dtype='torch.float16',
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            llm_int8_has_fp16_weight=False,
            llm_int8_threshold=6.0,
            load_in_4bit=False,
            load_in_8bit=False,
            type='transformers.BitsAndBytesConfig'),
        torch_dtype='torch.float16',
        trust_remote_code=True,
        type='transformers.AutoModelForCausalLM.from_pretrained'),
    train_stage='1',
    type='xtuner.model.LLaVAModel')
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    accumulative_counts=1,
    clip_grad=dict(error_if_nonfinite=False, max_norm=1),
    dtype='float16',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=0.001,
        type='torch.optim.AdamW',
        weight_decay=0),
    type='mmengine.optim.AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=0.03,
        start_factor=1e-05,
        type='mmengine.optim.LinearLR'),
    dict(
        begin=0.03,
        by_epoch=True,
        convert_to_iter_based=True,
        end=1,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
per_image_length = None
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.qwen_chat'
randomness = dict(deterministic=False, seed=None)
resume = False
sample_type = 'wsi'
save_steps = 500
save_total_limit = 2
tokenizer = dict(
    padding_side='right',
    pretrained_model_name_or_path=
    '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(max_epochs=1, type='xtuner.engine.runner.TrainLoop')
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        data_path=
        '/mnt/petrelfs/zhouxiao/project/TCGA/dataset_pp/PathoVerse_train_stage1_caption.json',
        dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
        image_folder='',
        image_path_list=None,
        max_length=51200,
        pad_image_to_square=False,
        per_image_length=None,
        template_map_fn=dict(
            template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
            type='xtuner.dataset.map_fns.template_map_fn_factory'),
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path=
            '/mnt/petrelfs/zhouxiao/hwfile_share/model/model_zoo/Qwen3-8B',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.dataset.LLaVADataset'),
    num_workers=2,
    pin_memory=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
visualizer = None
warmup_ratio = 0.03
weight_decay = 0
work_dir = '/mnt/petrelfs/zhouxiao/project/TCGA/SlideChat/work_dirs'
