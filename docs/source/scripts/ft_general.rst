Cell-type-specific regulatory effects
****************************************

This script enables fine-tuning ChromBERT for analyzing cell-type-specific regulatory effects. Users can selectively perturb or omit specific genomic features, making it valuable for simulating regulatory changes and testing hypotheses about the role of individual regulatory elements in cell-type-specific gene regulation.

.. code-block:: shell

    python ft_general.py [OPTIONS] --train TRAIN_PATH --valid VALID_PATH --test TEST_PATH


.. rubric:: Options

.. option:: --lr

    Learning rate. Default is *1e-4*.

.. option:: --warmup-ratio

    Warmup ratio. Default is *0.1*.

.. option:: --grad-samples

    Number of gradient samples. Automatically scaled according to the batch size and GPU number. Default is *512*.

.. option:: --max-epochs

    Number of epochs to train. Default is *10*.

.. option:: --pretrain-trainable

    Number of pretrained layers to be trainable. Default is *2*.

.. option:: --tag

    Tag of the trainer, used for grouping logged results. Default is *default*.

.. option:: --limit-val-batches

    Number of batches to use for each validation. Default is *64*.

.. option:: --val-check-interval

    Validation check interval. Default is *64*.

.. option:: --name

    Name of the trainer. Default is *chrombert-ft-general*.

.. option:: --save-top-k

    Save top k checkpoints. Default is *3*.

.. option:: --checkpoint-metric

    Checkpoint metric. Default is the same as the loss function if not specified.

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

    Batch size. Default is *8*.

.. option:: --num-workers

    Number of workers. Default is *4*.

.. option:: --basedir

    Path to the base directory. Default is set to the value of ``os.path.expanduser("~/.cache/chrombert/data")``.

.. option:: -g, --genome

    Genome version. For example, *hg38* or *mm10*. Only *hg38* is supported now. Default is *hg38*.

.. option:: -k, --ckpt

    Path to the pretrain checkpoint. Optional if it could be inferred from other arguments.

.. option:: --mask

    Path to the mtx mask file. Optional if it could be inferred from other arguments.

.. option:: -d, --hdf5-file

    Path to the HDF5 file that contains the dataset. Optional if it could be inferred from other arguments.

.. option:: --dropout

    Dropout rate. Default is *0.1*.

.. option:: -hr, --high-resolution

    Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.

.. option:: --ignore

    Ignore given targets.

.. option:: --ignore-object

    Ignore object. Regulator, or dataset IDs separated by *;*.

.. option:: --perturbation

    Use perturbation model.

.. option:: --perturbation-object

    Perturbation object. Regulator, or dataset IDs separated by *;*.

.. option:: --perturbation-value

    Perturbation target level. *0* means knock-out perturbation, and *4* means over-expression perturbation. Default is *0*.

