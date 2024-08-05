import os
import sys 
import shutil 
import subprocess
import argparse

class FileManager:
    @staticmethod
    def create_directories(directories):
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def decompress_file(file_path):
        command = ["gzip", "-d", file_path]
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Error decompressing {file_path}")

    @staticmethod
    def unpack_tar(file_path, output_dir):
        command = ["tar", "-xzf", file_path, "-C", output_dir]
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Error unpacking {file_path}")

class HuggingFaceDownloader:
    @staticmethod
    def download(ifile, odir, hf_endpoint="https://huggingface.co"):
        # huggingface_cli_path = os.path.join(os.path.dirname(sys.executable), "huggingface-cli")
        huggingface_cli_path = shutil.which("huggingface-cli")
        if huggingface_cli_path is None:
            raise FileNotFoundError("The 'huggingface-cli' command was not found in the system PATH.")
        
        cmd = [
            huggingface_cli_path,
            "download",
            "--repo-type",
            "dataset",
            "--local-dir",
            odir,
            "TongjiZhanglab/chrombert",
            ifile
        ]
        # cmd = f"huggingface-cli download --repo-type dataset --local-dir {odir} TongjiZhanglab/chrombert {ifile}"
        result = subprocess.run(cmd, env={"HF_ENDPOINT": hf_endpoint})
        if result.returncode != 0:
            print(f"Error downloading {ifile}")


def download(basedir = "~/.cache/chrombert/data", hf_endpoint="https://huggingface.co"):
    basedir = os.path.expanduser(basedir)
    os.makedirs(basedir, exist_ok=True)
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint[:-1]

    print(f"Downloading files to {basedir}")

    directories = [
        "config",
        "checkpoint",
        "cache",
        "other",
        "demo"
    ]
    directories = [os.path.join(basedir, directory) for directory in directories]
    
    FileManager.create_directories(directories)
    
    files_to_download = [
        ("hg38_6k_1kb.hdf5.gz", "."),
        ("hg38_6k_1kb_cistrome_cell_prompt_chr1_cache.h5", "cache"),
        ("hg38_6k_1kb_expression_cell_prompt_cache.pkl", "cache"),
        ("hg38_6k_1kb_regulator_prompt_chr1_cache.h5", "cache"),
        ("pbmc10k_scgpt_cell_prompt_cache.pkl","cache"), 
        ("hg38_6k_1kb_pretrain.ckpt", "checkpoint"),
        ("hg38_6k_1kb_prompt_cistrome.ckpt", "checkpoint"),
        ("hg38_6k_1kb_prompt_expression.ckpt", "checkpoint"),
        ("hg38_6k_factors_list.txt", "config"),
        ("hg38_6k_meta.tsv", "config"),
        ("hg38_6k_regulators_list.txt", "config"),
        ("hg38_6k_1kb_region.bed", "config"),
        ("hg38_6k_meta.json", "config"),
        ("hg38_6k_mask_matrix.tsv", "config"),
        ("hg38.fa", "other"),
        ("demo.tar.gz",".")
        
    ]

    files_to_decompress = [
        "hg38_6k_1kb.hdf5.gz",
    ]

    files_to_unpack = [
        ("demo.tar.gz", ".")
    ]

    for ifile, odir in files_to_download:
        if ifile in files_to_decompress and  os.path.exists(os.path.join(basedir, ifile.replace(".gz", ""))):
            continue
        HuggingFaceDownloader.download(ifile, os.path.join(basedir, odir), hf_endpoint)
    
    for file in files_to_decompress:
        if not os.path.exists(os.path.join(basedir, file.replace(".gz", ""))):
            FileManager.decompress_file(os.path.join(basedir, file))
        
    for file, output_dir in files_to_unpack:
        FileManager.unpack_tar(os.path.join(basedir, file), os.path.join(basedir, output_dir))
    return basedir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default="~/.cache/chrombert/data", help="Base directory to download files")
    parser.add_argument("--hf-endpoint", type=str, default="https://huggingface.co", help="Huggingface endpoint")
    args = parser.parse_args()
    download(args.basedir, args.hf_endpoint)
    return None

if __name__ == "__main__":
    main()
