CLI Reference
=============

Overview
-----------

We provide a set of command line scripts for your convenience. All scripts can be called in your terminal directly. See the following sections for more details.

.. csv-table:: Scripts Instruction 
    :header: "Script", "Description"

    "`chrombert_prepare_env`_", "Download required files to ~/.cache/chrombert/data, or other path your like."
    "chrombert_make_dataset", "Make dataset for ChromBERT forward. "
    "chrombert_get_region_emb", "Get mean pooled TRN embedding (region embedding) and store in a file."
    "chrombert_get_cistrome_emb", "Get cistrome embedding and store in a file. "
    "chrombert_get_regulator_emb", "Get regulator embedding and store in a file."


Details 
---------

.. include:: scripts/chrombert_prepare_env.rst 
