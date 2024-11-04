#########################
### BASE REQUIREMENTS ###
#########################

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
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

###############################
### DEVELOPMENT ENVIRONMENT ###
###############################

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
WORKDIR /workspace/deepbiosphere

# Install deepbiosphere package dependencies using pip, within the conda environment
RUN conda env create -f .devcontainer/environment.yml

# Set up R environment for reticulate
RUN Rscript -e "install.packages('reticulate')"

###########################
### RUNTIME ENVIRONMENT ###
###########################

FROM base AS runtime_environment

# Make .netrc file for auth with GBIF
RUN touch ~/.netrc \
    && echo "machine api.gbif.org login YOUR_GBIF_LOGIN password YOUR_GBIF_PASSWORD" > ~/.netrc

# Make directories for local data storage
RUN mkdir -p /workspaces/deepbiosphere/data/occs \
    && mkdir /workspaces/deepbiosphere/data/shpfiles/ \
    && mkdir /workspaces/deepbiosphere/data/models/ \
    && mkdir /workspaces/deepbiosphere/data/images/ \
    && mkdir /workspaces/deepbiosphere/data/rasters/ \
    && mkdir /workspaces/deepbiosphere/data/baselines/ \
    && mkdir /workspaces/deepbiosphere/data/results/ \
    && mkdir /workspaces/deepbiosphere/data/misc/ \
    && mkdir /workspaces/deepbiosphere/data/docs/ \
    && mkdir /workspaces/deepbiosphere/data/scratch/ \
    && mkdir /workspaces/deepbiosphere/data/runs/

# Copy the conda environment from the environment stage
COPY --from=dev_environment /opt/conda /opt/conda

# Default command to open bash for devcontainer interaction
CMD ["/bin/bash"]