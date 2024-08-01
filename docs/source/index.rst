Welcome to ChromBERT's documentation!
======================================

**ChromBERT** is a pre-trained deep learning model designed to capture the genome-wide co-association patterns of approximately one thousand transcription regulators, thereby enabling accurate representations of context-specific transcriptional regulatory networks (TRNs). As a foundational model, ChromBERT can be fine-tuned to adapt to various biological contexts through transfer learning and provide insights into the roles of transcription regulators in the specific biological contexts without the need of additional genomic.

Check out the :doc:`installation` section to get started.

.. note::

   This project is under active development.


.. toctree:: 
    :maxdepth: 2    
    :caption: Basics:

    installation
    scripts
    developing

.. toctree::
    :maxdepth: 2
    :caption: Tutorials: 

    finetune
    tutorial_embedding_extraction

.. toctree:: 
    :maxdepth: 1
    :caption: Examples:

    tutorial_prompt_cistrome_generation
    tutorial_locus_specific_TRN_eqtl
    tutorial_locus_specific_TRN_ezh2
    tutorial_locus_specific_TRN_starr
    tutorial_transdifferentiation
