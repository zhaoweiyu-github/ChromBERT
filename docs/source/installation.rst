Installation
============

ChromBERT is compatible with Python versions 3.8 or higher and requires PyTorch 2.0 or above, along with FlashAttention-2. These dependencies must be installed prior to ChromBERT.

Installing PyTorch
------------------
Follow the detailed instructions on `PyTorch’s official site <https://pytorch.org/get-started/locally/>`__ to install ``PyTorch`` according to your device and CUDA version specifications.

Installing FlashAttention-2
---------------------------
Execute the following commands to install the required packages and `FlashAttention-2 <https://github.com/Dao-AILab/flash-attention>`__.

.. code-block:: shell

    # Install the required packages for FlashAttention-2
    pip install packaging
    pip install ninja
    # FlashAttention-3 is not supported yet, please install FlashAttention-2
    pip install flash-attn==2.* --no-build-isolation

Installing ChromBERT
--------------------
Clone the repository and install ChromBERT using the commands below:

.. code-block:: shell

    git clone git@github.com:TongjiZhanglab/ChromBERT.git
    cd ChromBERT
    pip install .
    # Download required pre-trained model and annotation data files from Hugging Face to ~/.cache/chrombert/data
    chrombert_prepare_env

Verifying Installation
----------------------
To verify installation, execute the following command:

.. code-block:: python

    import chrombert