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
from chrombert import ChromBERTFTConfig, DatasetConfig
from .utils import HDF5Manager

DEFAULT_BASEDIR = os.path.expanduser("~/.cache/chrombert/data")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract region embeddings from ChromBERT")
    parser.add_argument("supervised_file", type=str, help="Path to the supervised file")
    parser.add_argument("-o", "--oname", type=str, required=True, help="Path to the output hdf5 file")

    parser.add_argument("--basedir", type=str, default = DEFAULT_BASEDIR, help="Base directory for the required files")
    
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. only hg38 is supported now.")
    parser.add_argument("-k", "--ckpt", type=str, required=False, default=None, help="Path to the pretrain or fine-tuned checkpoint. Optial if it could infered from other arguments")
    parser.add_argument("--mask", type=str, required=False, default=None, help="Path to the mtx mask file. Optional if it could infered from other arguments")

    parser.add_argument("-d","--hdf5-file", type=str, required=False, default=None, help="Path to the hdf5 file that contains the dataset. Optional if it could infered from other arguments")
    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")

    parser.add_argument("--batch-size", dest="batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--num-workers",dest="num_workers", type=int, required=False, default=8, help="number of workers for dataloader")

    return parser.parse_args()

def validate_args(args):
    assert os.path.exists(args.supervised_file), f"Supervised file does not exist: {args.supervised_file}"

def get_model_config(args):
    assert args.genome == "hg38", "Only hg38 is supported now"  
    if args.ckpt is not None:
        ckpt = args.ckpt
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.hr:
            res = "200bp"
        else:
            res = "1kb"
        ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_6k_{res}_pretrain.ckpt")
    parameters = {
        "genome": args.genome,
        "dropout": 0,
        "preset": "general",
    }
    if ChromBERTFTConfig.get_ckpt_type(ckpt) == "pretrain":
        parameters["pretrain_ckpt"] = ckpt
    else:
        parameters["finetune_ckpt"] = ckpt

    if args.mask is not None:
        parameters["mtx_mask"] = args.mask

    config = chrombert.get_preset_model_config(
        basedir = args.basedir,
        **parameters
    )
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
    config = get_model_config(args)
    model = config.init_model().get_embedding_manager().cuda().bfloat16()
    dc = get_dataset_config(args)
    dl = dc.init_dataloader()
    ds = dc.init_dataset()

    with HDF5Manager(args.oname, region=[(len(ds),4), np.int64], emb = [(len(ds), 768), np.float16]) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total = len(dl)):
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                model(batch) # initialize the cache 
                region = np.concatenate([
                    batch["region"].long().cpu().numpy(), 
                    batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
                    ], axis = 1
                )
                h5.insert(region = region, emb = model.get_region_embedding().float().cpu().detach().numpy())
    return None
    

if __name__ == "__main__":
    main()
