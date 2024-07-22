import os 
import sys 
import h5py 
import json 
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
    parser.add_argument("ids", nargs="+", type=str, help="IDs to extract. can be GSMID or regulator:cellline format id. To generate cache file for prompt, use 'regulator:cellline' format. ")

    parser.add_argument("-o", "--oname", type=str, required=True, help="Path to the output hdf5 file")

    parser.add_argument("--basedir", type=str, default = DEFAULT_BASEDIR, help="Base directory for the required files")
    
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. only hg38 is supported now.")
    parser.add_argument("-k", "--ckpt", type=str, required=False, default=None, help="Path to the pretrain checkpoint. Optial if it could infered from other arguments")
    parser.add_argument("--meta", type=str, required=False, default=None, help="Path to the meta file. Optional if it could infered from other arguments")
    parser.add_argument("--mask", type=str, required=False, default=None, help="Path to the mtx mask file. Optional if it could infered from other arguments")

    parser.add_argument("-d","--hdf5-file", type=str, required=False, default=None, help="Path to the hdf5 file that contains the dataset. Optional if it could infered from other arguments")
    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")

    parser.add_argument("--gpu", type=int, default = 0, help="GPU index") 
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, required=False, default=8, help="number of workers for dataloader")

    return parser.parse_args()

def validate_args(args):
    assert os.path.exists(args.supervised_file), f"Supervised file does not exist: {args.supervised_file}"
    print(f"Extracting embeddings for {len(args.ids)} ids")
    print(f"{args.ids}")


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

def get_meta_file(meta_file,basedir, genome):
    
    if meta_file is None:
        if genome == "hg38":
            meta_file = os.path.join(basedir, "config", f"{genome}_6k_meta.json")
        else:
            raise ValueError(f"Genome {genome} is not supported now")
    return meta_file


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

def get_cistrome_ids(ids, meta_file):

    ids = [i.strip() for i in ids]
    gsm_ids = [i for i in ids if ":" not in i ]
    reg_ids = [i for i in ids if ":" in i]

    with open(meta_file) as f:
        meta = json.load(f)

    dict_ids = {i:i for i in gsm_ids}
    dict_ids.update({k:meta[k] for k in reg_ids})

    return dict_ids



def main():
    args = parse_args()
    validate_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = get_model_config(args)
    model = config.init_model().get_embedding_manager().cuda().bfloat16()
    dc = get_dataset_config(args)
    dl = dc.init_dataloader()
    ds = dc.init_dataset()

    meta_file = get_meta_file(args.meta, args.basedir, args.genome)
    dict_ids = get_cistrome_ids(args.ids, meta_file)

    shapes = {f"emb/{k}": [(len(ds),768), np.float16] for k in dict_ids}
    with HDF5Manager(args.oname, region=[(len(ds),4), np.int64],**shapes) as h5:
        with torch.no_grad():
            for batch in tqdm(dl, total = len(dl)):
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                emb = model(batch).mean(dim=1).float().cpu().detach().numpy()
                region = np.concatenate([
                    batch["region"].long().cpu().numpy(), 
                    batch["build_region_index"].long().cpu().unsqueeze(-1).numpy()
                    ], axis = 1
                )
                embs = {
                    f"emb/{k}": model.get_cistrome_embedding(v).float().cpu().detach().numpy()
                    for k,v in dict_ids.items()
                }
                h5.insert(region = region, **embs)
    return None
    
