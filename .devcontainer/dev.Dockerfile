# Use a base image with Python 3.7+ and CUDA
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS base

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y git \
    && apt-get install -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

FROM base AS dev_environment

# Set up environment variables for CUDA and PyTorch
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0"
ENV IABN_FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ r-base \
    && apt-get clean

# Copy deepbiosphere src
COPY . /workspace/deepbiosphere
WORKDIR /workspace

# Install conda environment
RUN conda env create -f deepbiosphere/.devcontainer/environment.yml

# Set up R environment for reticulate
RUN Rscript -e "install.packages('reticulate')"

# Default command to open bash for devcontainer interaction
CMD ["/bin/bash"]