# ChromBERT: Uncovering chromatin regulatory architecture with transfer learning

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version: 1.2.3](https://img.shields.io/badge/Version-1.2.3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0)

**ChromBERT** is a deep language model learning latent representations for genome-wide co-assocation of about one thousand chromatin regulators, to enable accurate prediction or interpretation of context-specific human regulatory network in downstream tasks. ChromBERT is towards building a foundational model for chromatin regulatory omics with transfer learning. 

![docs/_static/ChromBERT_framework.png]( "Framework")

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

Our pre-trained checkpoint and fint-tuning code are in progress, which will be released soon.