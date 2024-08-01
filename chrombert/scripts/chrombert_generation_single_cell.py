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
from chrombert import ChromBERTFTConfig, DatasetConfig, LitChromBERTFTDataModule

import pyBigWig
import bioframe as bf

DEFAULT_BASEDIR = os.path.expanduser("~/.cache/chrombert/data")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract region embeddings from ChromBERT")
    parser.add_argument("--supervised_file", type=str, help="Path to the supervised file")
    parser.add_argument("--o_h5", type=str, required=True, help="Path of the hdf5 file")

    parser.add_argument("--basedir", type=str, default = DEFAULT_BASEDIR, help="Base directory for the required files")
    
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. only hg38 is supported now.")
    parser.add_argument("-pc", "--pretrain_ckpt", type=str, required=False, default=None, help="Path to the pretrain checkpoint. Optial if it could infered from other arguments")
    parser.add_argument("-m", "--mtx_mask", type=str, required=False, default=None, help="Path to the mtx mask file for the mean pooling of the embedding of regulators")
    parser.add_argument("-d","--hdf5-file", type=str, required=False, default=None, help="Path to the hdf5 file that contains the dataset. Optional if it could infered from other arguments")
    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")

    parser.add_argument("-fc", "--finetune_ckpt", type=str, required=True, default=None, help="Path to the finetune checkpoint")
    parser.add_argument("-pk", "--prompt_kind", type=str, required=True, default=None, help="prompt data class, choose from 'cistrome' or 'expression'")
    parser.add_argument("--prompt_dim_external", type=int, required=False, default=512, help="dimension of external data. use 512 for scgpt")
    
    parser.add_argument("-file_ct", "--prompt_celltype_cache_file", type=str, required=False, default=None, help="the path to the cell type specific prompt cache file")
    parser.add_argument("-file_rl", "--prompt_regulator_cache_file", type=str, required=False, default=None, help="the path to the regulator prompt cache file")
    parser.add_argument("-prompt_ct", "--prompt_celltype", type=str, required=False, default=None, help="the cell-type-specific prompt. For example, 'dnase:k562' for cistrome prompt and 'k562' for expression prompt. It can also be provided in the supervised file if the format supports.")
    parser.add_argument("-prompt_rl", "--prompt_regulator", type=str, required=False, default=None, help="the regulator prompt. Determine the kind of output. For example, 'ctcf' or 'h3k27ac'. It can also be provided in the supervised file if the format supports.")

    parser.add_argument("--gpu", type=int, default = 0, help="GPU index") 
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, required=False, default=8, help="number of workers for dataloader")

    return parser.parse_args()

def validate_args(args):
    assert os.path.exists(args.supervised_file), f"Supervised file does not exist: {args.supervised_file}"

def get_finetune_config(args):

    if args.mtx_mask is not None:
        mtx_mask = args.mtx_mask
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        mtx_mask = os.path.join(args.basedir, 'config', 'hg38_6k_mask_matrix.pt')
    if args.pretrain_ckpt is not None:
        pretrain_ckpt = args.pretrain_ckpt
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.hr:
            res = "200bp"
        else:
            res = "1kb"
        pretrain_ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_6k_{res}_pretrain.ckpt")

    config = ChromBERTFTConfig(mtx_mask=mtx_mask,pretrain_ckpt=pretrain_ckpt, task='prompt', prompt_kind=args.prompt_kind, finetune_ckpt=args.finetune_ckpt, prompt_dim_external = args.prompt_dim_external, dropout=0)
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
        kind = "PromptDataset", 
        prompt_kind = args.prompt_kind,
        supervised_file = args.supervised_file,
        prompt_celltype_cache_file = args.prompt_celltype_cache_file,
        prompt_regulator_cache_file = args.prompt_regulator_cache_file,
        prompt_regulator = args.prompt_regulator,
        prompt_celltype = args.prompt_celltype,
        hdf5_file = hdf5_file,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        meta_file = os.path.join(args.basedir, 'config/hg38_6k_meta.json')
        )
    return dataset_config

def predict(dl, ds, model, o_h5):
    regions, logits, probs, labels= [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dl,total=len(dl)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
            y = model(batch).float().cpu()
            logits.append(y)

    logits = torch.cat(logits)
    probs = torch.sigmoid(logits).cpu().numpy()
    logits = logits.cpu().numpy()
    regions = ds.h5_regions
    cells = ds.prompt_celltype
    regulators = ds.prompt_regulator

    get_hdf5(regions, probs, logits, cells, regulators, o_h5)


def get_hdf5(regions, probs, logits, cells, regulators, o_h5):
    num_regions = regions.shape[0]
    probs = probs.reshape(num_regions, -1)
    logits = logits.reshape(num_regions, -1)
    with h5py.File(o_h5, 'w') as f:
        f1 = f.create_dataset("regions", data=regions)
        f2= f.create_dataset("probs", data=probs)
        f3= f.create_dataset("logits", data=logits)
        f2.attrs['cells'] = cells
        f2.attrs['regulators'] = regulators


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = get_finetune_config(args)
    model = config.init_model().cuda().bfloat16().eval()

    dc = get_dataset_config(args)
    dl = dc.init_dataloader()
    ds = dc.init_dataset().dataset.sv_dataset
    

    predict(dl, ds, model, args.o_h5)


if __name__ == "__main__":
    main()