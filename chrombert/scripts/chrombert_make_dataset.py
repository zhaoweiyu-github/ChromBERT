import os 
import io 
import sys 
import subprocess as sp 

import numpy as np 
import pandas as pd

import argparse

DEFAULT_BASEDIR = os.path.expanduser("~/.cache/chrombert/data")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate general datasets for ChromBERT from bed3 files")
    parser.add_argument("bed", type=str, help="Path to bed file")
    parser.add_argument("-o","--oname", type=str, required = False, default=None, help="Path to output file. Stdout if not specified. Must end with .tsv or .txt. ")
    
    parser.add_argument("--mode", type=str, choices=["region","all"], default="region", help="Mode to generate the dataset. \nregion: only consider overlap between input regions to determine the label generated. Useful for narrowPeak like input. \nall: report all overlapping status like bedtools intersect -wao. You should determine the label column by your self. ")
    parser.add_argument("--center", action="store_true", help="If used, only consider the center of the input regions." )
    parser.add_argument("--label", type=int, default = 4, help="if mode is not region, this column will be used as label. Default is 4th. 1-based. ")

    parser.add_argument("--no-filter",dest="no_filter", default=False, action = "store_true", help="Do not filter the regions that are not overlapped. ")

    parser.add_argument("--basedir", type=str, default = DEFAULT_BASEDIR, help="Base directory for the required files")
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. ")
    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")

    return parser.parse_args()

def validate_args(args):
    assert os.path.exists(args.bed), f"Bed file does not exist: {args.bed}"
    assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
    assert args.genome in ["hg38", "mm10"], f"Genome version {args.genome} is not supported. "
    assert args.hr == False, "200-bp resolution is not supported now. "
    if args.oname is not None:
        assert isinstance(args.oname, str), f"Output file name should be string. Given: {args.oname}"
        assert args.oname.endswith(".tsv") or args.oname.endswith(".txt"), f"Output file should be tsv or txt file. Given: {args.oname}"

def run_cmd(cmd):
    try:
        run = sp.run(cmd, shell = True, stdout = sp.PIPE, stderr = sp.PIPE, check = True, text = True)
    except sp.CalledProcessError as e:
        print(e)
        print(e.stderr, file = sys.stderr)
        sys.exit(1)
    except Exception as e:
        raise(e)
        sys.exit(1)
    return run

def get_regions(basedir = DEFAULT_BASEDIR, genome="hg38", high_resolution = False):
    if genome == "hg38":
        if high_resolution:
            oname = os.path.join(basedir, "config", f"{genome}_6k_200bp_region.bed")
        else:
            oname = os.path.join(basedir, "config", f"{genome}_6k_1kb_region.bed")
    elif genome == "mm10":
        if high_resolution:
            oname = os.path.join(basedir, "config", f"{genome}_5k_200bp_region.bed")
        else:
            oname = os.path.join(basedir, "config", f"{genome}_5k_1kb_region.bed")
    else:
        raise ValueError(f"Genome {genome} is not supported. ")
    return oname

def get_overlap(supervised, regions, no_filter = False, center = False):
    assert os.path.exists(supervised), f"Supervised file does not exist: {supervised}"
    if center:
        cmd = f'''
        cut -f 1-3 {supervised} | awk 'BEGIN{{OFS="\\t"}}{{c=int(($2+$3)/2);$2=c;$3=$2+1;print $0;}}' | sort -k1,1 -k2,2n | bedtools merge | bedtools intersect -c -e -f 0.5 -F 0.5 -a {regions} -b - \
            '''
    else:
        cmd = f'''
        cut -f 1-3 {supervised} | sort -k1,1 -k2,2n | bedtools merge | bedtools intersect -c -e -f 0.5 -F 0.5 -a {regions} -b - \
            '''
    if not no_filter:
        cmd += ''' | awk '$5 > 0' '''

    run = run_cmd(cmd)

    if len(run.stdout) == 0:
        print("No overlapping regions found. ", file = sys.stderr)
        sys.exit(1)

    df_supervised = pd.read_csv(io.StringIO(run.stdout), sep = "\t", header = None)
    df_supervised.columns = ["chrom", "start", "end", "build_region_index", "label"]

    return df_supervised


def get_overlap_all(supervised, regions, no_filter = False, col_label = 4, center = False):
    assert os.path.exists(supervised), f"Supervised file does not exist: {supervised}"

    if center:
            cmd = f'''
        cat {supervised} | awk -F '\\t' 'BEGIN{{OFS="\\t"}}{{c=int(($2+$3)/2);$2=c;$3=$2+1;print $0;}}' | sort -k1,1 -k2,2n | bedtools intersect -wao -e -f 0.5 -F 0.5 -a {regions} -b - \
        '''
    else:
        cmd = f'''
        cat {supervised} | sort -k1,1 -k2,2n | bedtools intersect -wao -e -f 0.5 -F 0.5 -a {regions} -b - \
            '''
    if not no_filter:
        cmd += ''' | awk '$5 != "." ' '''
    
    run = run_cmd(cmd)

    if len(run.stdout) == 0:
        print("No overlapping regions found. ", file = sys.stderr)
        sys.exit(1)
    
    df_supervised = pd.read_csv(io.StringIO(run.stdout), sep = "\t", header = None)
    n_cols = df_supervised.shape[1]
    col_label += 4 - 1 # 1-based to 0-based, and shift to the right
    assert n_cols >= 8, f"Input file should have at least 3 columns. Given: {n_cols - 5 }"
    assert col_label < n_cols - 1, f"Label column {col_label -3 } is out of range. Total columns of your input file: {n_cols-4}"
    colnames = ["chrom","start","end","build_region_index","chrom_s","start_s","end_s"]

    n = 1
    for i in range(7, n_cols-1):
        if i != col_label:
            colnames.append(f"extra_{n}")
            n += 1
        else:
            colnames.append("label")
    colnames.append("coverage")
    df_supervised.columns = colnames
    df_supervised = df_supervised[["chrom","start","end","build_region_index","label", "chrom_s","start_s","end_s", "label", "coverage"] + [f"extra_{i}" for i in range(1,n)]]

    return df_supervised

def process(supervised, regions, mode = "region", no_filter = False, col_label = 4, center = False):
    if mode == "region":
        df =  get_overlap(supervised, regions, no_filter = no_filter, center = center)
    elif mode == "all":
        df = get_overlap_all(supervised, regions, no_filter = no_filter, col_label = col_label, center = center)
    else:
        raise ValueError(f"Mode {mode} is not supported. ")

    return df 

def main():
    args = parse_args()
    validate_args(args)
    regions = get_regions(args.basedir, args.genome, args.hr)
    # df_supervised = get_overlap(args.bed, regions, no_merge = args.no_merge)
    df_supervised = process(args.bed, regions, mode = args.mode, no_filter = args.no_filter, col_label = args.label, center = args.center)
    if args.oname is None:
        text = df_supervised.to_csv(sep = "\t", index = False)
        try:
            sys.stdout.write(text)
        except BrokenPipeError:
            pass
    else:
        df_supervised.to_csv(args.oname, sep = "\t", index = False)
    return 0

if __name__ == "__main__":
    main()


