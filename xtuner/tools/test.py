# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# # Set a different port for distributed training to avoid EADDRINUSE error
# if 'MASTER_PORT' not in os.environ:
#     os.environ['MASTER_PORT'] = '29565'
import os.path as osp
from types import FunctionType
from types import MethodType
from collections import OrderedDict
import torch

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from xtuner.configs import cfgs_name_path
from xtuner.registry import MAP_FUNC


def parse_args():
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument("--config", help="config file name or path.")
    parser.add_argument("--checkpoint", help="checkpoint file, e.g., xxx/iter_1.pth/mp_rank_00_model_states.pt")
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

    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")
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

    # only pretrained weights
    runner = RUNNERS.build(cfg)

    # Debug: MMEngine will crash with a cryptic error if model.test_step returns None.
    # Wrap it so we can print the offending batch metadata (often a corrupt/missing sample).
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

    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        runner.logger.info(f"Checkpoint file loading from {args.checkpoint}.")

        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        else: # standard case
            state_dict = checkpoint.get('state_dict', checkpoint)

        model_has_module_prefix = next(iter(runner.model.state_dict())).startswith('module.')
        ckpt_has_module_prefix = next(iter(state_dict)).startswith('module.')

        if model_has_module_prefix and not ckpt_has_module_prefix:
            runner.logger.info("Adding 'module.' prefix to checkpoint keys to match model.")
            corrected_state_dict = OrderedDict((f'module.{k}', v) for k, v in state_dict.items())
        elif not model_has_module_prefix and ckpt_has_module_prefix:
            runner.logger.info("Removing 'module.' prefix from checkpoint keys to match model.")
            corrected_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
        else:
            runner.logger.info("Prefixes match. No correction needed.")
            corrected_state_dict = state_dict
        
        missing_keys, unexpected_keys = runner.model.load_state_dict(corrected_state_dict, strict=False)
        runner.logger.info("Checkpoint loaded into model successfully!")
        
    except Exception as e:
        runner.logger.error(f"Failed to load checkpoint: {e}")
        raise e
    
    runner.model.eval()
    runner.model.to(torch.bfloat16)
    runner.test()

if __name__ == "__main__":
    main()