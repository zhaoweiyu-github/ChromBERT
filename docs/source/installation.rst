Installation
============

ChromBERT is compatible with Python versions 3.8 or higher and requires PyTorch 2.0 or above, along with FlashAttention-2. These dependencies must be installed prior to ChromBERT.

.. You must install `PyTorch <https://pytorch.org/get-started/locally/>`_ and `flash-attention <https://github.com/Dao-AILab/flash-attention>`_ before installing ChromBERT. 

Installing PyTorch 
------------------
Follow the detailed instructions on `PyTorch’s official site <https://pytorch.org/get-started/locally/>` to install PyTorch according to your device and CUDA version specifications.

Installing FlashAttention-2
---------------------------
Execute the following commands to install the requried packages and `FlashAttention-2<https://github.com/Dao-AILab/flash-attention>`.
.. code-block:: shell
    # install the required packages for FlashAttention-2
    pip install packaging
    pip install ninja

    pip install flash-attn==2.* --no-build-isolation # FlashAttention-3 is not supported yet, please install FlashAttention-2

Installing ChromBERT
--------------------
Clone the repository and install ChromBERT using the commands below:
.. code-block:: shell 
    git clone git@github.com:TongjiZhanglab/ChromBERT.git
    cd ChromBERT
    pip install .
    chrombert_prepare_env # download required pre-trained model and annotation data files from Hugging Face to ~/.cache/chrombert/data

Verifying Installation
----------------------
To verify installation, execute the following command:

.. code-block:: python
    import chrombert