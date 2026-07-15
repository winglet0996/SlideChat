# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# # Set a different port for distributed training to avoid EADDRINUSE error
# if 'MASTER_PORT' not in os.environ:
#     os.environ['MASTER_PORT'] = '29565'
import os.path as osp
from types import FunctionType
from types import MethodType
from mmengine.config import Config, DictAction
from mmengine.config.lazy import LazyObject
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version

from xtuner.configs import cfgs_name_path
from xtuner.registry import MAP_FUNC
from xtuner.tools.utils import auto_dtype_of_deepspeed_config


def parse_args():
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument("--config", help="config file name or path.")
    parser.add_argument("--checkpoint", help="checkpoint file, e.g., xxx/iter_1.pth/mp_rank_00_model_states.pt")
    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='the path to the .json file for deepspeed')
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    args = parser.parse_args()
    # args.config = '/home/xiaozhou/data/project/TCGA/SlideChat/xtuner/configs/slidechat/stage_2_qwen3_8b_conv_multitask.py'
    # args.checkpoint = '/home/xiaozhou/data/project/TCGA/train_s2_multitask_qwen3_8b_conv_lora_multitask/best_eval_reg_overall_r2_iter_31000.pth'

    return args


def _resolve_config_path(path):
    if osp.isfile(path):
        return path
    try:
        return cfgs_name_path[path]
    except KeyError:
        raise FileNotFoundError(f"Cannot find {path}")


def _as_model_checkpoint_file(checkpoint):
    """Match train-config warm-start semantics for DeepSpeed checkpoint dirs."""
    if osp.isdir(checkpoint):
        return osp.join(checkpoint, 'mp_rank_00_model_states.pt')
    return checkpoint


def _enable_deepspeed(cfg, deepspeed_path):
    try:
        import deepspeed
    except ImportError as e:
        raise ImportError(
            'deepspeed is not installed properly, please check.') from e
    if digit_version(deepspeed.__version__) < digit_version('0.12.3'):
        raise RuntimeError('Please upgrade your DeepSpeed version '
                           'by using `deepspeed>=0.12.3`')

    deepspeed_path = _resolve_config_path(deepspeed_path)
    with open(deepspeed_path) as f:
        ds_cfg = json.load(f)

    ds_cfg = auto_dtype_of_deepspeed_config(ds_cfg)
    test_dataloader = cfg.get('test_dataloader', None)
    train_bs = test_dataloader.batch_size if test_dataloader else cfg.train_dataloader.batch_size

    optim_wrapper = cfg.get('optim_wrapper', None)
    if optim_wrapper is None:
        # ZeRO-2 requires an optimizer even for eval-only engine creation.
        # It is never stepped in this path; model weights come from pretrained_pth.
        optimizer = dict(type='AdamW', lr=0.0, weight_decay=0.0)
        grad_accum = 1
        grad_clip = 1.0
    else:
        optimizer = optim_wrapper.optimizer
        grad_accum = optim_wrapper.get('accumulative_counts', 1)
        clip_grad = optim_wrapper.get('clip_grad', None)
        grad_clip = clip_grad.get('max_norm', 1.0) if clip_grad else 1.0

    exclude_frozen_parameters = True if digit_version(
        deepspeed.__version__) >= digit_version('0.10.1') else None

    cfg.__setitem__(
        'strategy',
        dict(
            type=LazyObject('xtuner.engine', 'DeepSpeedStrategy'),
            config=ds_cfg,
            gradient_accumulation_steps=grad_accum,
            train_micro_batch_size_per_gpu=train_bs,
            gradient_clipping=grad_clip,
            exclude_frozen_parameters=exclude_frozen_parameters,
            sequence_parallel_size=getattr(cfg, 'sequence_parallel_size', 1)))
    deepspeed_optim_wrapper = dict(
        type='DeepSpeedOptimWrapper',
        optimizer=optimizer)
    if optim_wrapper is not None and optim_wrapper.get('constructor', None) is not None:
        deepspeed_optim_wrapper['constructor'] = optim_wrapper.constructor
    if optim_wrapper is not None and optim_wrapper.get('paramwise_cfg', None) is not None:
        deepspeed_optim_wrapper['paramwise_cfg'] = optim_wrapper.paramwise_cfg
    cfg.__setitem__('optim_wrapper', deepspeed_optim_wrapper)
    cfg.runner_type = 'FlexibleRunner'


def _run_deepspeed_val_with_optimizer(runner):
    """Prepare DeepSpeed like train(), then run only the validation loop."""
    if runner._val_loop is None:
        raise RuntimeError('val loop is required for DeepSpeed eval.')

    runner._val_loop = runner.build_val_loop(runner._val_loop)
    dispatch_kwargs = dict(
        init_weights_for_test_or_val=runner.cfg.get('init_weights_for_test_or_val', True))
    runner.strategy.prepare(
        runner.model,
        optim_wrapper=runner.optim_wrapper,
        dispatch_kwargs=dispatch_kwargs)
    runner.model = runner.strategy.model
    runner.optim_wrapper = runner.strategy.optim_wrapper

    runner.load_or_resume()
    runner.call_hook('before_run')
    metrics = runner.val_loop.run()
    runner.call_hook('after_run')
    return metrics


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)


def main():
    args = parse_args()

    args.config = _resolve_config_path(args.config)
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # avoid auto loading
    if cfg.load_from is not None:
        cfg.load_from = None
    cfg.resume = False

    register_function(cfg._cfg_dict)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    # Default evaluator outputs follow the final runtime work_dir, but do not
    # overwrite explicit output_dir values defined in the config.
    if cfg.get("val_evaluator", None) is not None and not cfg.val_evaluator.get("output_dir"):
        cfg.val_evaluator["output_dir"] = osp.join(cfg.work_dir, "val_results")
    if cfg.get("test_evaluator", None) is not None and not cfg.test_evaluator.get("output_dir"):
        cfg.test_evaluator["output_dir"] = osp.join(cfg.work_dir, "test_results")

    if args.deepspeed:
        pretrained_pth = _as_model_checkpoint_file(args.checkpoint) if args.checkpoint else None
        cfg.model.pretrained_pth = pretrained_pth
        cfg.load_from = None
        # Reuse the train-time validation path, but feed it the requested test set.
        cfg.val_dataloader = cfg.test_dataloader
        cfg.val_evaluator = cfg.test_evaluator
        cfg.val_cfg = dict(type='ValLoop')
        # FlexibleRunner requires train_dataloader/train_cfg/optim_wrapper to
        # be all present once DeepSpeed ZeRO-2 needs an optimizer. This loop is
        # never executed; eval still runs only through val_loop below.
        cfg.train_dataloader = cfg.get('train_dataloader', cfg.test_dataloader)
        cfg.train_cfg = cfg.get('train_cfg', dict(type='IterBasedTrainLoop', max_iters=1, val_interval=1))
        _enable_deepspeed(cfg, args.deepspeed)
    else:
        cfg.load_from = args.checkpoint
    cfg.resume = False

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Debug wrapper is only safe for the eager MMEngine Runner path.
    # FlexibleRunner prepares DeepSpeed lazily inside runner.test().
    if not args.deepspeed:
        _orig_test_step = runner.model.test_step

        def _debug_test_step(self, data_batch, *args, **kwargs):
            try:
                outputs = _orig_test_step(data_batch, *args, **kwargs)
            except Exception as e:
                try:
                    meta = {}
                    if isinstance(data_batch, dict):
                        d = data_batch.get('data', {}) if isinstance(data_batch.get('data', None), dict) else {}
                        for k in ['image_file', 'category', 'project']:
                            if k in d:
                                meta[k] = d.get(k)
                    runner.logger.error(f'test_step exception. batch_meta={meta}')
                except Exception:
                    pass
                raise

            if outputs is None:
                meta = {}
                if isinstance(data_batch, dict):
                    d = data_batch.get('data', {}) if isinstance(data_batch.get('data', None), dict) else {}
                    for k in ['image_file', 'category', 'project']:
                        if k in d:
                            meta[k] = d.get(k)
                raise RuntimeError(
                    f'model.test_step returned None. batch_meta={meta}. '
                    'This usually means the dataloader produced an invalid/empty batch or the model predict path bailed out.'
                )
            return outputs

        runner.model.test_step = MethodType(_debug_test_step, runner.model)

    if args.deepspeed:
        if cfg.model.pretrained_pth:
            runner.logger.info(
                f"Checkpoint will be warm-started via model.pretrained_pth from {cfg.model.pretrained_pth}.")
        else:
            runner.logger.info('No checkpoint supplied; evaluating the configured base model initialization.')
        _run_deepspeed_val_with_optimizer(runner)
    else:
        runner.model.eval()
        runner.test()

if __name__ == "__main__":
    main()
