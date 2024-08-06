Gep
*********************

Fine-tune the ChromBERT model with a multiflank window setting to predict genome-wide changes in the transcriptome.

.. code-block:: shell

    python ft_gep.py [OPTIONS] --flank-window FLANK_WINDOW_SIZE \
    --train TRAIN_PATH \  
    --valid VALID_PATH \  
    --test TEST_PATH 

.. rubric:: Options

.. option:: --lr

    Learning rate. Default is *1e-4*.

.. option:: --warmup-ratio

    Warmup ratio for the learning rate. Default is *0.1*.

.. option:: --grad-samples

    Number of gradient samples, scaled by batch size and GPU count. Default is *128*.

.. option:: --pretrain-trainable

    Number of pretrained layers to be trainable. Default is *2*.

.. option:: --max-epochs

    Maximum number of training epochs. Default is *10*.

.. option:: --tag

    Tag of the trainer, used for grouping logged results. Default is *default*.

.. option:: --limit-val-batches

    Number of batches to use for each validation. Default is *64*.

.. option:: --val-check-interval

    Interval for validation checks. Default is *64*.

.. option:: --name

    Name of the training session. Default is *chrombert-ft-gep*.

.. option:: --save-top-k

    Number of top-performing checkpoints to save. Default is *3*.

.. option:: --checkpoint-metric

    Metric for checkpointing. Default is *zero_inflation*.

.. option:: --checkpoint-mode

    Mode for checkpointing. Default is *min*.

.. option:: --log-every-n-steps

    Logging frequency in terms of steps. Default is *50*.

.. option:: --kind

    Type of task, such as *regression*, *zero_inflation*. Default is *zero_inflation*.

.. option:: --loss

    Loss function to be used. Default is *zero_inflation*.

.. option:: --train

    Path to the training data. This option is required.

.. option:: --valid

    Path to the validation data. This option is required.

.. option:: --test

    Path to the test data. This option is required.

.. option:: --batch-size

    Batch size for training. Default is *2*.

.. option:: --num-workers

    Number of workers for data loading. Default is *4*.

.. option:: --basedir

    Path to the base directory for model and data files. Default is ``os.path.expanduser("~/.cache/chrombert/data")``.

.. option:: -g, --genome

    Genome version. Only *hg38* is supported now. Default is *hg38*.

.. option:: -k, --ckpt

    Path to the pretrained checkpoint. Optional if it could be inferred from other arguments.

.. option:: --mask

    Path to the mtx mask file. Optional if it could be inferred from other arguments.

.. option:: -d, --hdf5-file

    Path to the HDF5 file that contains the dataset. Optional if it could be inferred from other arguments.

.. option:: --dropout

    Dropout rate for the model. Default is *0.1*.

.. option:: -hr, --high-resolution

    Use 200-bp resolution instead of 1-kb. Note: 200-bp resolution is not available yet, preparing for future release.

.. option:: --flank-window

    Flank window size for genomic data embedding. Default is *4*.

.. option:: --gep-zero-inflation

    Specifies whether to include zero inflation in the GEP header. Default is *True*.

.. option:: --gep-parallel-embedding

    Enable parallel embedding, which is faster but requires more GPU memory.

.. option:: --gep-gradient-checkpoint

    Use gradient checkpointing to reduce GPU memory usage during training.
