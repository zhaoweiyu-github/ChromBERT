CLI Reference
=============

Overview
-----------

We provide a set of command line scripts for your convenience. All scripts can be called in your terminal directly. See the following sections for more details.

.. csv-table:: Scripts Instruction 
    :header: "Script", "Description"

    "`chrombert_prepare_env`_", "Download required files to ~/.cache/chrombert/data, or other path your like."
    "`chrombert_make_dataset`_", "Make dataset for ChromBERT forward. "
    "`chrombert_get_region_emb`_", "Get mean pooled TRN embedding (region embedding) and store in a file."
    "`chrombert_get_cistrome_emb`_", "Get cistrome embedding and store in a file. "
    "`chrombert_get_regulator_emb`_", "Get regulator embedding and store in a file."
    "`chrombert_imputation_cistrome`_", "Generate cistromes using prompt-enhanced ChromBERT. "
    "`chrombert_imputation_cistrome_sc`_", "Generate cistromes using prompt-enhanced ChromBERT, specified for single-cell data. "

-----

Details 
---------

.. include:: scripts/chrombert_prepare_env.rst 

----

.. include:: scripts/chrombert_make_dataset.rst

----

.. include:: scripts/chrombert_get_region_emb.rst

----

.. include:: scripts/chrombert_get_cistrome_emb.rst

----

.. include:: scripts/chrombert_get_regulator_emb.rst

----

.. include:: scripts/chrombert_imputation_cistrome.rst

---- 

.. include:: scripts/chrombert_imputation_cistrome_sc.rst

