from utils import *
from runner import *
import argparse
from os.path import join
import warnings
warnings.filterwarnings('ignore')
import os


def exp():
    runner = DARunner(args)
    runner.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="/data/xuanlin/DA/Domain_Adaptation/output/output_target")
    parser.add_argument("--savedir", type=str, default="/data/xuanlin/DA/Domain_Adaptation/saved/saved_target")
    parser.add_argument("--savedir_source", type=str, default="/data/xuanlin/DA/Domain_Adaptation/saved/saved_source")
    parser.add_argument("--config_fn_t", type=str, default="target")
    parser.add_argument("--config_fn_s", type=str, default="source")
    parser.add_argument("--source", type=str, default="YelpHotel", choices=["YelpRes", "YelpHotel", "YelpNYC", "Amazon"])
    parser.add_argument("--target", type=str, default="YelpRes", choices=["YelpRes", "YelpHotel", "YelpNYC", "Amazon"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(os.getcwd())

    if args.config_fn_s is not None:
        config_s = load_yaml(join("/data/xuanlin/DA/Domain_Adaptation/config", args.config_fn_s + ".yaml"))
        if args.source in config_s:
            source_config = config_s[args.source]
            args = argparse.Namespace(**{**vars(args), **source_config})
    else:
        raise RuntimeError("Source config file not found!")
    
    if args.config_fn_t is not None:
        config_t = load_yaml(join("/data/xuanlin/DA/Domain_Adaptation/config", args.config_fn_t + ".yaml"))
        if args.target in config_t:
            target_config = config_t[args.target]
            args = argparse.Namespace(**{**vars(args), **target_config})
    else:
        raise RuntimeError("Target config file not found!")
    
    print(f"Source Domain: {args.source}")
    print(f"Target Domain: {args.target}\n")
    print(args)
    print()
    exp()