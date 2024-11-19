import argparse
import os
from os.path import join
from utils import load_yaml
from datetime import datetime
from random import seed, randint
import pandas as pd
import os
import torch
import utils
from models import *
from runner import *


def exp():
    device = torch.device(f'cuda:{args.device}')
    utils.set_random_seeds(seed_value=args.seed, device=device)

    data_src, label_src = utils.load_data(args.source, if_split=True)
    data_tgt, label_tgt = utils.load_data(args.target, if_split=False)
    source_encoder = GCN_Encoder(nhids = args.nhids_source_encoder,
                                 dropout = args.dropout_source_encoder,
                                 with_bn = args.with_bn_source_encoder)
    classifier = Classifier(nhids = args.nhids_classifier,
                            dropout = args.dropout_classifier,
                            with_bn = args.with_bn_classifier)


    print(f"Source Domain: {args.source}")
    print(f"Target Domain: {args.target}")
    runner = SupRunner(data_src, label_src, data_tgt, label_tgt, source_encoder, classifier, device)
    runner.train(printing = True, 
                 lr = args.lr, 
                 n_epoch = args.n_epoch, 
                 n_stopping = args.n_stopping, 
                 early_stopping = args.early_stopping)
    runner.save('GCN', args.outdir, args.savedir, args.source)
    runner.clear()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="/data/xuanlin/DA/Domain_Adaptation/output/output_source")
    parser.add_argument("--savedir", type=str, default="/data/xuanlin/DA/Domain_Adaptation/saved/saved_source")
    parser.add_argument("--config_fn", type=str, default="source")
    parser.add_argument("--target", type=str, default="YelpRes", choices=["YelpRes", "YelpHotel", "YelpNYC", "Amazon"])
    parser.add_argument("--source", type=str, default="YelpHotel", choices=["YelpRes", "YelpHotel", "YelpNYC", "Amazon"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(os.getcwd())


    if args.config_fn is not None:
        config = load_yaml(join("config", args.config_fn + ".yaml"))
        if args.source in config:
            source_config = config[args.source]
            args = argparse.Namespace(**{**vars(args), **source_config})
    else:
        raise RuntimeError("Config file not found!")

    exp()
