# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from types import FunctionType
from collections import OrderedDict
import torch

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from xtuner.configs import cfgs_name_path
from xtuner.registry import MAP_FUNC

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
    args.config = '/home/winglet/pathology/vqa/SlideChat/xtuner/configs/slidechat/stage_1_qwen3_8b_conv_longnet.py'
    args.checkpoint = '/home/winglet/pathology/vqa/train_s1/iter_1.pth/mp_rank_00_model_states.pt'
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = '0'
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

    register_function(cfg._cfg_dict)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    # only pretrained weights
    runner = RUNNERS.build(cfg)

    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        runner.logger.info("Checkpoint file loaded.")

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