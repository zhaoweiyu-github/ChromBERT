# ChromBERT: Uncovering Chromatin Regulatory Architecture with Transfer Learning

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version: 1.2.3](https://img.shields.io/badge/Version-1.2.3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0)

**ChromBERT** is a pre-trained deep learning model designed to learn latent representations for genome-wide co-association among approximately one thousand chromatin regulators. This enables the accurate prediction and interpretation of context-specific human regulatory networks in various downstream tasks. ChromBERT aims to establish a foundational model for chromatin regulatory omics through transfer learning.

![ChromBERT Framework](docs/_static/ChromBERT_framework.png "Framework")

## Directory structure
```
├── chrombert                    # Main Python package
├── config                       # Model data config file 
├── data                         # Data files, including pretrain checkpoint
├── finetune                     # Finetune task package
├── experiments                  # Experiments and case studies
├── docs                         # Documentation files
├── tests                        # Unit tests for the Python package
├── env.yaml                     # Reproducible Python environment via conda
├── LICENSE
└── README.md
```

## Installation

ChromBERT is being developed into a Python package and will be released soon.

## Pre-trained Model Zoo

ChromBERT has been pre-trained on the human Cistrome-Human-6K dataset at 1-kb resolution and the mouse Cistrome-Mouse-5K dataset at 1-kb resolution. Further training is planned on the Cistrome-Human-6K dataset at 200-bp resolution. The available pre-trained models include:

| Model Name                | Description                                              | Download Link                                                                                     |
| :------------------------ | :------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| Human-6K-1kb (recommended) | Pre-trained on Cistrome-Human-6K dataset at 1-kb resolution | [Download not available now](https://github.com/zhaoweiyu-github/ChromBERT/tree/main) |
| Mouse-5K-1kb              | Pre-trained on Cistrome-Mouse-5K dataset at 1-kb resolution  | [Download not available now](https://github.com/zhaoweiyu-github/ChromBERT/tree/main) |

## Fine-Tuning ChromBERT

For detailed guidance on fine-tuning ChromBERT to uncover context-specific regulatory networks, please refer to our tutorials in [finetune/README.md](finetune/README.md), which we will continue to update.

## Citing ChromBERT

Our work is currently in progress.