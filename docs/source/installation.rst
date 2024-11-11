Installation
============

``ChromBERT`` is compatible with Python versions 3.8 or higher and requires ``PyTorch`` version 2.0 or above, along with ``FlashAttention-2``. These dependencies must be installed prior to ``ChromBERT``.

Installing PyTorch
------------------
Follow the detailed instructions on `PyTorch’s official site <https://pytorch.org/get-started/locally/>`__ to install ``PyTorch`` according to your device and CUDA version specifications.

.. note::
    ChromBERT has been tested with Python 3.9+ and Torch 2.0 to 2.4 (inclusive). Compatibility with other environments is not guaranteed.

Installing FlashAttention-2
---------------------------
Execute the following commands to install the required packages and `FlashAttention-2 <https://github.com/Dao-AILab/flash-attention>`__.

.. code-block:: shell

    # Install the required packages for FlashAttention-2
    pip install packaging
    pip install ninja
    # FlashAttention-3 is not supported yet, please install FlashAttention-2
    pip install flash-attn==2.4.* --no-build-isolation

Installing ChromBERT
--------------------
Clone the repository and install ``ChromBERT`` using the commands below:

.. code-block:: shell

    git clone https://github.com/TongjiZhanglab/ChromBERT.git
    cd ChromBERT
    pip install .
    
Then download required pre-trained model and annotation data files from Hugging Face to ~/.cache/chrombert/data.

.. code-block:: shell
    
    chrombert_prepare_env

Alternatively, if you're experiencing significant connectivity issues with Hugging Face, you can try to use the ``--hf-endpoint`` option to connect to an available mirror of Hugging Face for you.

.. code-block:: shell
    
    chrombert_prepare_env --hf-endpoint <Hugging Face endpoint>

For built-in dataset preparation, it is recommended to install `bedtools <https://bedtools.readthedocs.io/en/latest/content/installation.html>`_.

Verifying Installation
----------------------
To verify installation, execute the following python code:

.. code-block:: python

    import chrombert