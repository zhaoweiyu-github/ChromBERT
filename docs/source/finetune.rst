Fine-tuning of ChromBERT
-------------------------

For a hands-on tutorial, see the documentation on :doc:`tutorial_finetuning_ChromBERT_from_scratch`.  

We provide three scripts for fine-tuning, designed for your convenience. All scripts can be downloaded and executed anywhere, provided that your installation is correct.

For detailed usage instructions, run the following command:

.. code-block:: bash

    python <script.py> --help

.. csv-table:: Fine-Tune Scripts
    :header: "Type", "Description"

    "`General`_", "Suitable for most purposes."
    "Prompt", "Designed for prompt-enhanced tasks, such as cistrome generation."
    "Multi-window", "Intended for tasks that use multiple 1-kb bins as input, such as gene expression prediction."

.. _General: https://raw.githubusercontent.com/TongjiZhanglab/ChromBERT/main/examples/train/ft_general.py 



.. toctree:: 
    :maxdepth: 1 

    scripts/finetune
    tutorial_finetuning_ChromBERT_from_scratch