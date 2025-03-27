FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Install essential packages
RUN apt-get update && apt-get install -y wget bzip2 tar gzip && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH="/opt/miniconda/bin:${PATH}"

# Install python 3.9 using conda
RUN conda install -y python=3.9 && \
    conda clean -a -y

# Install PyTorch for CUDA 12.1 using pip
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

# Install packaging and ninja (required by flash_attn)
RUN pip install packaging ninja && pip cache purge

# Copy and install flash_attn
COPY ./flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl /tmp/
RUN pip install /tmp/flash_attn-2.4.3.post1+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl && \
    rm /tmp/flash_attn-*.whl && \
    pip cache purge

# Copy, extract, and install ChromBERT
COPY ./ChromBERT.tar.gz /tmp/
RUN tar -xzf /tmp/ChromBERT.tar.gz -C /tmp && \
    pip install /tmp/ChromBERT && \
    rm -rf /tmp/ChromBERT* && \
    pip cache purge

# Install other dependencies
RUN pip install scipy jupyterlab && pip cache purge

# Set the working directory
WORKDIR /workspace

# Set Python as entrypoint
ENTRYPOINT ["python"]
