# A pre-trained foundation model for context-specific transcription regulatory network 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version: 1.0.0](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0)

**ChromBERT** is a pre-trained deep learning model designed to learn latent representations for genome-wide co-association among approximately one thousand transcription regulators. This enables the accurate prediction and interpretation of context-specific human regulatory networks in various downstream tasks. ChromBERT aims to establish a foundational model for transcription regulatory omics through transfer learning.

![ChromBERT Framework](docs/_static/ChromBERT_framework.png "Framework")


## Installation
You must install [PyTorch](https://pytorch.org/get-started/locally/) and [flash-attention](https://github.com/Dao-AILab/flash-attention) before installing ChromBERT. 
```shell
# Below is instruction for installing flash-attention. For PyTorch, please refer to the official website.
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```
<!-- ChromBERT is being developed into a Python package and will be released soon. -->
```shell
git clone git@github.com:qianachen/ChromBERT_reorder.git
cd ChromBERT_reorder
pip install .
chrombert_prepare_env # download required files to ~/.cache/chrombert/data
```

After installation, you can import ChromBERT in Python:
```python
import chrombert
```
For detail usage, please refer to the [examples](examples) directory.


## Directory structure
```
|-- chrombert                    # Main Python package
    |-- base                     # Pre-trained part of the model
    |-- finetune                 # For downstream tasks
    └-- scripts                  # Scripts for convinient usage 
|-- data                         # Data for testing.
|-- examples                     # Tutorials and train scripts
|-- docs                         # Documentation files
|-- pyproject.toml               # Depencency management file
|-- LICENSE
└-- README.md
```

## To-do List
- [ ] Publish to pypi
- [ ] Fine-tune scripts 
- [ ] Tutorials for Prompt-based model 
- [ ] Tutorial for cellular dynamics TRN inference
- [ ] Documentation website with readthedocs 
- [ ] Bump up to flash-attention v3


## Pre-trained Model Zoo

ChromBERT has been pre-trained on the human Cistrome-Human-6K dataset at 1-kb resolution and the mouse Cistrome-Mouse-5K dataset at 1-kb resolution. Further training is planned on the Cistrome-Human-6K dataset at 200-bp resolution. The available pre-trained models include:

| Model Name                | Description                                              | Download Link                                                                                     |
| :------------------------ | :------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| Human-6K-1kb | Pre-trained on Cistrome-Human-6K dataset at 1-kb resolution | [Download](https://huggingface.co/datasets/TongjiZhanglab/chrombert) |

Notion: you can also download by running `chrombert_prepare_env` in the terminal.

## Fine-Tuning ChromBERT

For detailed guidance on fine-tuning ChromBERT to uncover context-specific regulatory networks, please refer to our tutorials in [examples](examples), which we will continue to update.

## Citing ChromBERT

Our work is currently in progress.
