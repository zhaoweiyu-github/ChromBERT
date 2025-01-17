Prompt-enhanced
*******************

This task is designed for situations where the pre-trained model's default single-region embeddings are insufficient. By incorporating additional prompts, such as cell-type specific embeddings or DNA sequence embeddings, the model can make more informed predictions by leveraging extra contextual knowledge.

.. code-block:: shell

    python ft_prompt_enhanced.py [OPTIONS] --prompt-kind KIND \  
        --train TRAIN_PATH \  
        --valid VALID_PATH \  
        --test TEST_PATH  

    # use cache file for acceleration 
    python ft_prompt_enhanced.py [OPTIONS] \  
        --prompt-kind KIND  \  
        --prompt-regulator-cache-file CACHE_PATH1 \  
        --prompt-celltype-cache-file CACHE_PATH2 \  
        --train TRAIN_PATH \  
        --valid VALID_PATH \  
        --test TEST_PATH 


.. rubric:: Options

.. option:: --lr

    Learning rate. Default is *1e-4*.

.. option:: --warmup-ratio

    Warmup ratio. Default is *0.1*.

.. option:: --grad-samples

    Number of gradient samples. Automatically scaled according to the batch size and GPU number. Default is *512*.

.. option:: --pretrain-trainable

    Number of pretrained layers to be trainable. Default is *0*.

.. option:: --max-epochs

    Number of epochs to train. Default is *10*.

.. option:: --tag

    Tag of the trainer, used for grouping logged results. Default is *default*.

.. option:: --limit-val-batches

    Number of batches to use for each validation. Default is *64*.

.. option:: --val-check-interval

    Validation check interval. Default is *64*.

.. option:: --name

    Name of the trainer. Default is *chrombert-ft-prompt-enhanced*.

.. option:: --save-top-k

    Save top k checkpoints. Default is *3*.

.. option:: --checkpoint-metric

    Checkpoint metric. Default is *bce*.

.. option:: --checkpoint-mode

    Checkpoint mode. Default is *min*.

.. option:: --log-every-n-steps

    Log every n steps. Default is *50*.

.. option:: --kind

    Kind of the task. Choose from *classification*, *regression*, or *zero_inflation*. Default is *classification*.

.. option:: --loss

    Loss function. Default is *focal*.

.. option:: --train

    Path to the training data. This option is required.

.. option:: --valid

    Path to the validation data. This option is required.

.. option:: --test

    Path to the test data. This option is required.

.. option:: --batch-size

    Batch size. Default is *8*. It's suggested to set a larger number to accelerate training here. 

.. option:: --num-workers

    Number of workers. Default is *4*.

.. option:: --basedir

    Path to the base directory. Default is set to the value of ``os.path.expanduser("~/.cache/chrombert/data")``.

.. option:: -g, --genome

    Genome version. For example, *hg38* or *mm10*. Only *hg38* is supported now. Default is *hg38*.

.. option:: -k, --ckpt

    Path to the checkpoints used to initialize the model. Optional. Defualt is the pretrain checkpoint provided in the base directory.

.. option:: --mask
    Path to the mtx mask file. Optional if it could infered from other arguments. 

.. option:: -d, --hdf5-file

    Path to the HDF5 file that contains the dataset. Optional if it could be inferred from other arguments.

.. option:: --dropout

    Dropout rate. Default is *0.1*.

.. option:: -hr, --high-resolution

    Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.

.. option:: --prompt-kind

    Prompt data class. Choose from *cistrome* or *expression*. Default is *None*. This option is required.

.. option:: --prompt-dim-external

    Dimension of external data. Use *512* for *scGPT*, and *768* for *ChromBERT*'s embedding. Default is *512*.

.. option:: --prompt-celltype-cache-file

    Path to the cell-type-specific prompt cache file. Provided if you want to use cache file to accelerate the training process. Optional. Default is not use it. 

.. option:: --prompt-regulator-cache-file

    Path to the regulator prompt cache file. Provided if you want to use cache file to accelerate the training process. Optional.  Default is not use it. 
