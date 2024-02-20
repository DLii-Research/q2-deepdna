FROM quay.io/jupyter/minimal-notebook:2024-02-13 AS build-base

LABEL maintainer="Joshua L. Phillips <https://www.cs.mtsu.edu/~jphillips/>"
LABEL release-date="2024-02-20"

USER root

# Maybe not needed for most stacks...
# RUN echo -e "y\ny" | unminimize

# Additional tools
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get update && \
    apt-get install -y \
    autoconf \
    curl \
    emacs-nox \
    g++ \
    g++-multilib \
    gcc \
    gcc-multilib \
    gdb \
    graphviz \
    less \
    libtool \
    make \
    man-db \
    poppler-utils \
    python3-opengl \
    rsync \
    s3cmd \
    ssh \
    time \
    tmux \
    vim \
    zip && \
    apt-get dist-upgrade -y && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN mamba install --yes \
    bash_kernel \
    expect \
    jupyter-server-proxy \
    && \
    mamba clean --all -f -y

FROM build-base AS build-qiime2
ARG Q2_VERSION=2024.2
ARG Q2_DISTRIBUTION=amplicon
ENV QIIME2_ENV=qiime2-$Q2_DISTRIBUTION-$Q2_VERSION
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda/envs/qiime2-$Q2_DISTRIBUTION-$Q2_VERSION/pkgs/cuda-toolkit"

USER root
RUN echo -e "/opt/conda/envs/${QIIME2_ENV}/lib" | tee /etc/ld.so.conf.d/zzz-conda.conf
USER ${NB_UID}

# Qiime2
RUN wget https://data.qiime2.org/distro/$Q2_DISTRIBUTION/$QIIME2_ENV-py38-linux-conda.yml && \
    mamba env create -n $QIIME2_ENV --file $QIIME2_ENV-py38-linux-conda.yml  && \
    rm $QIIME2_ENV-py38-linux-conda.yml
RUN mamba run -n ${QIIME2_ENV} python -m ipykernel install --name ${QIIME2_ENV} --prefix /opt/conda --display-name "Python 3 (Qiime2)"

# Not working... and probably never will.
# SHELL ["mamba", "run", "-n", "${QIIME2_ENV}", "/bin/bash", "-o", "pipefail", "-c"]
# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# RESCRIPt - we use this for much of our work
# even if it (was) not in the Qiime2 amplicon
# distribution by default, so added here (no longer needed).
# RUN mamba install -n ${QIIME2_ENV} -c conda-forge -c bioconda -c qiime2 \
#     -c https://packages.qiime2.org/qiime2/$Q2_VERSION/shotgun/released/ -c defaults \
#     xmltodict "q2-types-genomics>$Q2_VERSION" ncbi-datasets-pylib && \
#     mamba run -n ${QIIME2_ENV} pip install --no-cache-dir git+https://github.com/bokulich-lab/RESCRIPt.git

FROM build-qiime2 AS build-deeplearning
# GPU-enabled Tensorflow
RUN CONDA_OVERRIDE_CUDA="11.8" mamba install -n $QIIME2_ENV --yes \
    bokeh \
    cudatoolkit==11.8.0 \
    cudnn==8.8.0.121 \
    numpy \
    plotly \
    pydot \
    python-graphviz \
    tensorflow==2.9.1 \
    wandb \
    && \
    mamba clean --all -f -y

# PyTorch
RUN mamba run -n ${QIIME2_ENV} pip install --no-cache-dir \
    deepspeed \
    jsonargparse[signatures] \
    lightning \
    tokenizers \
    torch \
    torchaudio \
    torchinfo \
    torchmetrics \
    torchtext \
    torchview \
    torchvision \
    transformers \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Note that we cannot clean mamba -at all- after
# this point - or it will remove the dev tools from the stack...
RUN mamba install -n $QIIME2_ENV --yes \
    cudatoolkit-dev==11.7.0 && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Configure user environment (needed for singularity/apptainer)
RUN cp /etc/skel/.bash_logout /etc/skel/.bashrc /etc/skel/.profile /home/${NB_USER}/. && conda init && mamba init

FROM build-deeplearning AS build-deep-dna

# SetBERT Packages
RUN git clone https://github.com/DLii-Research/deep-dna && \
    cd deep-dna && \
    git checkout dev && \
    mamba run -n ${QIIME2_ENV} pip install --no-cache-dir -e . && \
    rm -rf deep-dna && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
