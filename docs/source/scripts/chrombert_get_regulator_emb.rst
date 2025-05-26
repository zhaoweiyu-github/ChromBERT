chrombert_get_regulator_emb
*****************************

Extract regulator embeddings from ChromBERT.

.. code-block:: shell

    chrombert_get_regulator_emb [OPTIONS] SUPERVISED_FILE IDS... -o ONAME 

.. rubric:: Options 

.. option:: SUPERVISED_FILE

    Path to the supervised file.

.. option:: IDS

    Regulator names to extract. Must be in lower case.

.. option:: -o, --oname

    Path to the output HDF5 file. This option is required.

.. option:: --basedir

    Base directory for the required files. Default is set to the value of `DEFAULT_BASEDIR`.

.. option:: -g, --genome

    Genome version. For example, hg38 or mm10. Only hg38 is supported now. Default is *hg38*.

.. option:: -k, --ckpt

    Path to the pretrain or **fine-tuned** checkpoint. Optional if it can be inferred from other arguments.

.. option:: --meta

    Path to the meta file. Optional if it can be inferred from other arguments.

.. option:: --mask

    Path to the matrix mask file. Optional if it can be inferred from other arguments.

.. option:: -d, --hdf5-file

    Path to the HDF5 file that contains the dataset. Optional if it can be inferred from other arguments.

.. option:: -hr, --high-resolution

    Use 200-bp resolution instead of 1-kb resolution. Caution: 200-bp resolution is preparing for the future release of ChromBERT, which is not available yet.

.. option:: --batch-size

    Batch size. Default is *8*.

.. option:: --num-workers

    Number of workers for the dataloader. Default is *8*.
