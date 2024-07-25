Installation
============

You must install `PyTorch <https://pytorch.org/get-started/locally/>`_ and `flash-attention <https://github.com/Dao-AILab/flash-attention>`_ before installing ChromBERT. 

.. code-block:: shell

    # Below is instruction for installing flash-attention. For PyTorch, please refer to the official website.    
    # Flash-attention 3 is not supported yet. Please install flash-attention 2 
    pip install packaging  
    pip install ninja  
    pip install flash-attn==2.* --no-build-isolation  

Then, you can install ChromBERT by running the following commands:

.. code-block:: shell 

    git clone git@github.com:qianachen/ChromBERT_reorder.git
    cd ChromBERT_reorder
    pip install .
    chrombert_prepare_env # download required files to ~/.cache/chrombert/data


After installation, you can import ChromBERT in Python:

.. code-block:: python

    import chrombert
