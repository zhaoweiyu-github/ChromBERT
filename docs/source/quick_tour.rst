Quick Tour
==========

Before starting this quick tour, ensure you are familiar with the basics
of the PyTorch Lightning framework, including the
`LightningDataModule <https://lightning.ai/docs/pytorch/2.5.0/data/datamodule.html#lightningdatamodule>`__
and
`LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`__.
What’s more, make sure you have downloaded the necessary file by
executing the ``chrombert_prepare_env`` command, see :doc:`chrombert_prepare_env <scripts/chrombert_prepare_env>` for
details.

OK! Let’s get started! 

1 Customize the input
---------------------

You can customize the model’s input by assigning parameters to the ``chrombert.DatasetConfig`` class. Key
parameters include:

``hdf5_file``: A preprocessed HDF5 file containing features for 1kb bins
across the genome. This file is cached in the default directory
(``~/.cache/chrombert/data/hg38_6k_1kb.hdf5``) upon installation, unless
customized.

``kind``: Specifies the input format, which varies across tasks. It is
crucial to assign this based on your specific task. Different tasks may
require additional parameters, which you can find
`here <https://github.com/zhaoweiyu-github/ChromBERT/blob/main/chrombert/finetune/dataset/dataset_config.py>`__.

``supervised_file``: A input dataset containing at least four columns:
``chrom``,\ ``start``,\ ``end``, ``build_region_index``. These four
columns are used to locate and retrieve features for the regions. Depending on the task, you can add additional
columns, such as: ``label``

You can also configure other parameters like ``batch_size`` and
``num_workers``.

.. code:: python

   import chrombert

   # Create a DatasetConfig object with your settings
   dc = chrombert.DatasetConfig(hdf5_file="~/.cache/chrombert/data/hg38_6k_1kb.hdf5", 
   kind="GeneralDataset", supervised_file="<your_path_input_data>")

   # Initialize inputs in whatever formats you want
   ds = dc.init_dataset()  # Dataset

   dl = dc.init_dataloader()  # Dataloader

   dm = chrombert.LitChromBERTFTDataModule(config=dc, 
   train_params={"supervised_file": args.train}, 
   val_params={"supervised_file": args.valid},
   test_params={"supervised_file": args.test})  # LightningDataModule

2 Customize the model
---------------------

The model structure depends on the task at hand. Use the
``ChromBERTFTConfig``\ class to specify the task and configure its
parameters. Remember to assign the ``pretrain_ckpt`` parameter if you
want to use the pre-trained ChromBERT model. The checkpoint file is
cached in the default directory
(``~/.cache/chrombert/data/checkpoint/hg38_6k_1kb_pretrain.ckpt``) upon
installation, unless customized.

Different tasks may require additional parameters, which you can find
`here <https://github.com/zhaoweiyu-github/ChromBERT/blob/main/chrombert/finetune/model/model_config.py>`__.

.. code:: python

   # Configure the model for your task
   mc = chrombert.ChromBERTFTConfig(task='general', 
                        pretrain_ckpt="~/.cache/chrombert/data/checkpoint/hg38_6k_1kb_pretrain.ckpt")
   model = mc.init_model()

   # Optional: Manage trainable parameters
   # model.freeze_pretrain(trainable:int)  # Freeze transformer layers
   # model.display_trainable_parameters()  # Display the number of trainable layers

3 Customize the training process
--------------------------------

Once the input and model are configured, you can customize the training
process, including:

• Loss Function (``loss``): Specify the type of loss (e.g., ``"bce"`` for binary cross-entropy). 
• Learning Rate (``lr``): Set the desired learning rate. 
• Task Type (``kind``): Choose from ["classification", "regression", "zero_inflation"].

Explore other customizable training parameters `here <https://github.com/zhaoweiyu-github/ChromBERT/blob/main/chrombert/finetune/train/train_config.py>`__.

.. code:: python

   config_train = chrombert.finetune.TrainConfig(kind="classification", 
                                                loss="bce", lr=1e-4)
   pl_module = config_train.init_pl_module(model)
   trainer = config_train.init_trainer()

4 Start training !
------------------

With everything in place, you’re ready to train the model:

.. code:: python

   trainer.fit(pl_module, datamodule = dm)

5 Task templates
------------------

To make your workflow easier, we’ve prepared a collection of ready-to-use scripts for different tasks. You can find detailed instructions and examples :doc:`here <finetune>`.
