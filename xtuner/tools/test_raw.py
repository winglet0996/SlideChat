# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from types import FunctionType

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import MAP_FUNC

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def parse_args():
    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument("--config", default=None, help="config file name or path.")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument(
        "--load-method",
        choices=["manual", "runner"],
        default="auto",
        help="Method to load checkpoint: 'manual' uses guess_load_checkpoint, 'runner' uses Runner load_from config",
    )
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    
    args.config = '/home/winglet/pathology/vqa/SlideChat/xtuner/configs/slidechat/stage_1_qwen3_8b.py'
    args.checkpoint = '/home/winglet/pathology/vqa/train_s1/iter_1.pth'
    args.work_dir = '/home/winglet/pathology/vqa/train_s1'

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
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

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # build the runner from config
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    if args.load_method == "manual":
        # manual loading similar to default test
        state_dict = guess_load_checkpoint(args.checkpoint)
        runner.model.load_state_dict(state_dict, strict=False)
        runner.logger.info(f"Manual load checkpoint from {args.checkpoint}")
    elif args.load_method == "runner":
        # use Runner load_from and build from config like train
        cfg.load_from = args.checkpoint
        runner = Runner.from_cfg(cfg)
        runner.logger.info(f"Runner load checkpoint via config: {args.checkpoint}")
    else:
        raise ValueError(f"Unknown load_method {args.load_method}")

    # start testing
    runner.test()


if __name__ == "__main__":
    main()