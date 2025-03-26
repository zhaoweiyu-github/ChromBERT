import os 
import sys 
import torch
import numpy 
import argparse

import chrombert 
import lightning.pytorch as pl 

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune the ChromBERT with a multiflank window setting to predict genome-wide changes in transcriptome")
    # training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. ")
    parser.add_argument("--warmup-ratio",dest="warmup_ratio", type=float, default=0.1, help="Warmup ratio. ")
    parser.add_argument("--grad-samples", dest="grad_samples", type=int, default=128, help="Number of gradient samples. Automatically scaled according to the batch size and gpu number. ")
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=10, help="Number of epochs to train. ")
    parser.add_argument("--pretrain-trainable",dest="pretrain_trainable", type=int, default = 2, help="Number of pretrained layers to be trainable. ")    
    parser.add_argument("--tag", type=str, default="default", help="Tag of the trainer, used for grouping logged results. ")

    # validation arguments
    parser.add_argument("--limit-val-batches", dest="limit_val_batches", type=float, default=64, help="Number of batches to use for each validation. ")
    parser.add_argument("--val-check-interval", dest="val_check_interval", type=float, default=64, help="Validation check interval. ")

    # checkpoint arguments
    parser.add_argument("--name", type=str, default="chrombert-ft-gep", help="Name of the trainer. ")
    parser.add_argument("--save-top-k", dest="save_top_k", type=int, default=3, help="Save top k checkpoints. ")
    parser.add_argument("--checkpoint-metric", dest="checkpoint_metric", type=str, default="pcc", help="Checkpoint metric (Pearson correlation coefficient). ")
    parser.add_argument("--checkpoint-mode", dest="checkpoint_mode", type=str, default="max", help="Checkpoint mode. ")
    parser.add_argument("--log-every-n-steps",dest="log_every_n_steps", type=int, default=50, help="Log every n steps. ")
    # loss arguments
    parser.add_argument("--kind", choices=["regression", "zero_inflation"], default="regression", help="Kind of the task. ")
    parser.add_argument("--loss", type=str, default="rmse", help="Loss function. ")

    # data arguments
    parser.add_argument("--train", type=str, required=True, help="Path to the training data. ")
    parser.add_argument("--valid", type=str, required=True, help="Path to the validation data. ")
    parser.add_argument("--test", type=str, required=True, help="Path to the test data. ")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2, help="Batch size. ")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4, help="Number of workers. ")

    # model arguments
    parser.add_argument("--basedir", type=str, default=os.path.expanduser("~/.cache/chrombert/data"), help="Path to the base directory. ")
    parser.add_argument("-g", "--genome", type=str, default = "hg38", help="genome version. For example, hg38 or mm10. ")

    parser.add_argument("-k", "--ckpt", type=str, required=False, default=None, help="Path to the pretrain checkpoint. Optial if it could infered from other arguments")
    parser.add_argument("--mask", type=str, required=False, default=None, help="Path to the mtx mask file. Optional if it could infered from other arguments")

    parser.add_argument("-d","--hdf5-file", type=str, required=False, default=None, help="Path to the hdf5 file that contains the dataset. Optional if it could infered from other arguments")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate. ")

    parser.add_argument("-hr","--high-resolution", dest = "hr", action = "store_true", help="Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.")
        
    # gep arguments
    parser.add_argument("--flank-window", dest="flank_window", type=int, default=4, help="Flank window size.")
    parser.add_argument("--gep-zero-inflation", action="store_true", default=False, help="Use GEP header that includes zero inflation output. ")
    parser.add_argument("--gep-parallel-embedding", action="store_true", help="Parallel pick embedding, which is quick but requires more GPU memory. ")
    parser.add_argument("--gep-gradient-checkpoint", action="store_true", help="Use gradient checkpoint, which requires less GPU memory but is slower ")    
    
    return parser.parse_args()

def get_datamodule(args):
    assert os.path.exists(args.train), f"Training file does not exist: {args.train}"
    assert os.path.exists(args.valid), f"Validation file does not exist: {args.valid}"
    assert os.path.exists(args.test), f"Test file does not exist: {args.test}"
    if args.hr:
        res = "200bp"
    else:
        res = "1kb"
    if args.hdf5_file is not None:
        hdf5_file = args.hdf5_file
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.genome == "hg38":
            hdf5_file = os.path.join(args.basedir, f"{args.genome}_6k_{res}.hdf5")
        elif args.genome == "mm10":
            hdf5_file = os.path.join(args.basedir, f"{args.genome}_5k_{res}.hdf5")
        else:
            raise ValueError(f"Genome {args.genome} is not supported. ")

    if args.genome == "hg38":
        meta_file = os.path.join(args.basedir, "config", f"{args.genome}_6k_meta.json")
    elif args.genome == "mm10":
        meta_file = os.path.join(args.basedir, "config", f"{args.genome}_5k_meta.json")
    else:
        raise ValueError(f"Genome {args.genome} is not supported. ")


    params = {
        "hdf5_file": hdf5_file,
        "meta_file": meta_file,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "flank_window": args.flank_window,
        "basedir": args.basedir,
    }
    dc = chrombert.get_preset_dataset_config("multi_flank_window",**params, supervised_file = None)
    data_module = chrombert.LitChromBERTFTDataModule(
        config = dc,
        train_params={"supervised_file": args.train},
        val_params={"supervised_file": args.valid},
        test_params={"supervised_file": args.test},
    )
    return data_module

def get_model_config(args):
    assert args.genome in ["hg38", "mm10"], "Only hg38 and mm10 are supported now"  
    if args.ckpt is not None:
        ckpt = args.ckpt
    else:
        assert os.path.exists(args.basedir), f"Basedir does not exist: {args.basedir}. If you use default basedir, please make sure environment initialized correctly (see readme of the repo). "
        if args.hr:
            res = "200bp"
        else:
            res = "1kb"
        if args.genome == "hg38":
            ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_6k_{res}_pretrain.ckpt")
        elif args.genome == "mm10":
            ckpt = os.path.join(args.basedir, "checkpoint", f"{args.genome}_5k_{res}_pretrain.ckpt")
        else:
            raise ValueError(f"Genome {args.genome} is not supported. ")
    parameters = {
        "genome": args.genome,
        "dropout": args.dropout,
        "preset": "gep",
        "gep_flank_window": args.flank_window,
        "gep_zero_inflation": args.gep_zero_inflation,
        "gep_parallel_embedding": args.gep_parallel_embedding,
        "gep_gradient_checkpoint": args.gep_gradient_checkpoint,

    }
    if chrombert.ChromBERTFTConfig.get_ckpt_type(ckpt) == "pretrain":
        parameters["pretrain_ckpt"] = ckpt
    else:
        print("Warning: You are using a finetune checkpoint. Make sure it is the correct one!")
        parameters["finetune_ckpt"] = ckpt

    if args.mask is not None:
        parameters["mtx_mask"] = args.mask
    else:
        if args.genome == "hg38":
            parameters["mtx_mask"] = os.path.join(args.basedir, "config", f"{args.genome}_6k_mask_matrix.tsv")
        elif args.genome == "mm10":
            parameters["mtx_mask"] = os.path.join(args.basedir, "config", f"{args.genome}_5k_mask_matrix.tsv")
        else:
            raise ValueError(f"Genome {args.genome} is not supported. ")

    config = chrombert.get_preset_model_config(
        basedir = args.basedir,
        **parameters
    )

    return config

def get_train_config(args):
    if args.limit_val_batches >1:
        args.limit_val_batches = int(args.limit_val_batches)
    if args.val_check_interval > 1:
        args.val_check_interval = int(args.val_check_interval)
    if args.checkpoint_metric is None:
        checkpoint_metric = args.loss
    else:
        checkpoint_metric = args.checkpoint_metric
    config = chrombert.finetune.TrainConfig(
        kind = args.kind,
        loss = args.loss,
        tag = args.tag,
        max_epochs = args.max_epochs,
        lr = args.lr,
        warmup_ratio = args.warmup_ratio,
        accumulate_grad_batches = args.grad_samples // args.batch_size // torch.cuda.device_count(),
        limit_val_batches = args.limit_val_batches,
        val_check_interval = args.val_check_interval,
        checkpoint_metric = checkpoint_metric, 
        checkpoint_mode = args.checkpoint_mode

    )

    return config

def main():
    print("Welcome to ChromBERT fine-tuning. ")
    print(f"Total {torch.cuda.device_count()} GPUs available. If you want to use specific GPUs, you can set `CUDA_VISIBLE_DEVICES`= 'gpu_num' ")
    print("Parsing the arguments ... ")
    args = get_args()
    print("initiating the datasets ... ")
    data_module = get_datamodule(args)
    print("initiating the model ... ")
    model_config = get_model_config(args)
    model = model_config.init_model()
    model.freeze_pretrain(trainable = args.pretrain_trainable)
    model.display_trainable_parameters()
    train_config = get_train_config(args)
    pl_module = train_config.init_pl_module(model)
    trainer = train_config.init_trainer(name = args.name, 
                                        precision="bf16-mixed", 
                                        save_top_k = args.save_top_k, 
                                        log_every_n_steps = args.log_every_n_steps
                                        )
    trainer.fit(pl_module, datamodule = data_module)
    print("Training finished. See you~")

if __name__ == "__main__":
    main()