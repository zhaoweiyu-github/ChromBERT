import os 
import sys 
import h5py 
import argparse

import numpy as np 
import pandas as pd 

import torch
from torch import nn 
from tqdm import tqdm 

import chrombert 
from chrombert import ChromBERTConfig, DatasetConfig
from .utils import HDF5Manager

DEFAULT_BASEDIR = os.path.expanduser("~/.cache/chrombert/data")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract region embeddings from ChromBERT")
    parser.add_argument("supervised_file", type=str, help="Path to the supervised file")
    parser.add_argument("-o", "--oname", type=str, required=True, help="Path to the output hdf5 file")

    parser.add_argument("--basedir", type=str, default = DEFAULT_BASEDIR, help="Base directory for the required files")
    
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. only hg38 is supported now.")
    parser.add_argument("-k", "--pretrain_ckpt", type=str, required=False, default=None, help="Path to the pretrain checkpoint. Optial if it could infered from other arguments")
    parser.add_argument("-d","--hdf5-file", type=str, required=False, default=None, help="Path to the hdf5 file that contains the dataset. Optional if it could infered from other arguments")
    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")

    parser.add_argument("--gpu", type=int, default = 0, help="GPU index") 
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, required=False, default=8, help="number of workers for dataloader")

    return parser.parse_args()

def validate_args(args):
    assert os.path.exists(args.supervised_file), f"Supervised file does not exist: {args.supervised_file}"

def get_pretrain_config(args):

    if args.pretrain_ckpt is not None:
        pretrain_ckpt = args.pretrain_ckpt
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.hr:
            res = "200bp"
        else:
            res = "1kb"
        pretrain_ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_6k_{res}_pretrain.ckpt")
    
    assert os.path.exists(pretrain_ckpt), f"Pretrain checkpoint does not exist: {pretrain_ckpt}"
    assert ChromBERTConfig.get_ckpt_type(pretrain_ckpt) == "pretrain", f"Invalid pretrain checkpoint: {args.pretrain_ckpt}. Only pretrain checkpoint is allowed." 

    config = ChromBERTConfig(genome=args.genome, dropout = 0, ckpt = pretrain_ckpt)
    return config

def get_dataset_config(args):
    if args.hr:
        res = "200bp"
    else:
        res = "1kb"
    if args.hdf5_file is not None:
        hdf5_file = args.hdf5_file
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        hdf5_file = os.path.join(args.basedir, f"{args.genome}_6k_{res}.hdf5")

    dataset_config = DatasetConfig(
        kind = "GeneralDataset", 
        supervised_file = args.supervised_file,
        hdf5_file = hdf5_file,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        )
    return dataset_config

def main():
    args = parse_args()
    validate_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = get_pretrain_config(args)
    model = config.init_model().cuda().bfloat16().eval()
    dc = get_dataset_config(args)
    dl = dc.init_dataloader()
    ds = dc.init_dataset()
    with HDF5Manager(args.oname, region=[(len(ds),4), np.int64], emb = [(len(ds), 768), np.float16]) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total = len(dl)):
                emb = model(batch["input_ids"].cuda(), batch["position_ids"].cuda()).mean(dim=1).float().cpu().detach().numpy()
                region = np.concatenate([
                    batch["region"].long().cpu().numpy(), 
                    batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
                    ], axis = 1
                )
                h5.insert(region = region, emb = emb)
    return None
    

if __name__ == "__main__":
    main()
