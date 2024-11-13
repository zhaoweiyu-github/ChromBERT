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
    parser = argparse.ArgumentParser(description="Imputation the data in new cell types or regulators")
    parser.add_argument("supervised_file", type=str, help="Path to the supervised file")
    parser.add_argument("--o-bw", dest="o_bw", type=str, required=False, default=None, help="Path of the bw file")
    parser.add_argument("--o-table", dest="o_table", type=str, required=False, default=None, help="Path to the output table if you want to output the table")

    parser.add_argument("--basedir", type=str, default = DEFAULT_BASEDIR, help="Base directory for the required files")
    
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. only hg38 is supported now.")
    parser.add_argument("--pretrain-ckpt", dest="pretrain_ckpt", type=str, required=False, default=None, help="Path to the pretrain checkpoint. Optial if it could infered from other arguments")
    parser.add_argument("-d","--hdf5-file", type=str, required=False, default=None, help="Path to the hdf5 file that contains the dataset. Optional if it could infered from other arguments")
    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")


    parser.add_argument("--prompt-kind", dest='prompt_kind', type=str, required=True, default=None, help="prompt data class, choose from 'cistrome' or 'expression'")
    parser.add_argument("--finetune-ckpt", dest='finetune_ckpt', type=str, required=False, default=None, help="Path to the finetune checkpoint. Optial if it could infered from other arguments")
    parser.add_argument("--prompt-dim-external", dest ='prompt_dim_external', type=int, required=False, default=512, help="Dimension of external data. use 512 for scGPT")
    
    parser.add_argument("--prompt-celltype-cache-file", dest="prompt_celltype_cache_file", type=str, required=False, default=None, help="Path to the cell type specific prompt cache file")
    parser.add_argument("--prompt-regulator-cache-file", dest="prompt_regulator_cache_file", type=str, required=False, default=None, help="Path to the regulator prompt cache file")
    parser.add_argument("--prompt-celltype", dest="prompt_celltype", type=str, required=False, default=None, help="The cell-type-specific prompt. For example, 'dnase:k562' for cistrome prompt and 'k562' for expression prompt. It can also be provided in the supervised file if the format supports.")
    parser.add_argument("--prompt-regulator", dest="prompt_regulator", type=str, required=False, default=None, help="The regulator prompt. Determine the kind of output. For example, 'ctcf' or 'h3k27ac'. It can also be provided in the supervised file if the format supports.")

    parser.add_argument("--batch-size", dest="batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--num-workers", dest="num_workers", type=int, required=False, default=8, help="number of workers for dataloader")

    return parser.parse_args()

def validate_args(args):
    assert os.path.exists(args.supervised_file), f"Supervised file does not exist: {args.supervised_file}"
    if args.prompt_kind == "regression":
        assert os.path.exists(args.prompt_regulator_cache_file), "prompt-regulator-cache-file must be provided if the prompt kind is regression"


def get_finetune_config(args):

    if args.pretrain_ckpt is not None:
        pretrain_ckpt = args.pretrain_ckpt
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.hr:
            res = "200bp"
        else:
            res = "1kb"
        pretrain_ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_6k_{res}_pretrain.ckpt")

    if args.finetune_ckpt:
        finetune_ckpt = args.finetune_ckpt
    else:
        finetune_ckpt = os.path.join(DEFAULT_BASEDIR, "checkpoint", f"hg38_6k_1kb_prompt_{args.prompt_kind}.ckpt")

    config = ChromBERTFTConfig(pretrain_ckpt=pretrain_ckpt, task='prompt', prompt_kind=args.prompt_kind, finetune_ckpt=finetune_ckpt, prompt_dim_external = args.prompt_dim_external, dropout=0)
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

def predict(args, dl, ds, model, o_table, o_bw):
    region_indices, logits, probs, labels= [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dl,total=len(dl)):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].cuda()
            y = model(batch).float().cpu()
            if 'label' in batch.keys():
                label = batch['label'].cpu()
                labels.append(label)
            logits.append(y)
            region_indices.append(batch['build_region_index'])

    logits = torch.cat(logits)
    probs = torch.sigmoid(logits).cpu().numpy()
    logits = logits.cpu().numpy()
    indices = torch.cat(region_indices).cpu().numpy()
    cells = ds.prompt_celltype
    regulators = ds.prompt_regulator

    df = get_table(args, indices, logits, probs, labels, cells, regulators)
    if o_table:
        output_table(df, o_table)
    if o_bw:
        assert len(set(cells)) == 1, f"only predicition of single cell is support when output bigwig files, but get {len(set(cells))} cells"
        get_bw(df, o_bw)

    return df

def get_table(args, indices, logits, probs, labels, cells, regulators):
    df_supervised = pd.read_csv(args.supervised_file)[['chrom', 'start', 'end', 'build_region_index']]
    df_result = pd.DataFrame({'build_region_index':indices, 'logit': logits, 'prob': probs, 'cell':cells, 'regulator':regulators})
    df = df_supervised.merge(df_result, how='left', on=['build_region_index'])
    if len(labels) > 0:
        labels = torch.cat(labels).cpu().numpy()
        df['label'] = labels
    return df

def output_table(df, opath):
    if opath.endswith('.parquet'):
        df.to_parquet(opath, engine='pyarrow')
    elif opath.endswith('.csv'):
        df.to_csv(opath, index=False)
    elif opath.endswith('.tsv'):
        df.to_csv(opath, sep='\t', index=False)
    else:
        file_extension = opath.split('.')[-1]
        assert False, f"outfile should be .csv, .tsv, or .parquet, but got {file_extension}"

def get_bw(df, opath):
    chrom_sizes = bf.fetch_chromsizes('hg38')
    chrom_order = list(chrom_sizes.keys())

    df['chrom'] = pd.Categorical(df['chrom'], categories=chrom_order, ordered=True)
    df = df.sort_values(by=['chrom', 'start'])

    bw = pyBigWig.open(opath, 'w')
    bw.addHeader(list(chrom_sizes.items()))
    bw.addEntries(df['chrom'].tolist(), df['start'].tolist(), ends=df['end'].tolist(), values=df['prob'].tolist())
    bw.close()

def main():
    args = parse_args()
    config = get_finetune_config(args)
    model = config.init_model().cuda().bfloat16().eval()

    dc = get_dataset_config(args)
    dl = dc.init_dataloader()
    ds = dc.init_dataset()
    ds = ds.dataset.sv_dataset

    predict(args, dl, ds, model, args.o_table, args.o_bw)


if __name__ == "__main__":
    main()
