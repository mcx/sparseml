# This torch version was used when we did not have
# support for torch 2.2.x
ARG TORCH_VERSION=1.9.1 

# torchvsion version was determined from pypi
# by taking into account the torch version needed for it
ARG TORCHVISION_VERSION=0.10.1 

# The CUDA version was needed because installing torch
# would not install certain CUDA dependencies that were
# needed for GPU support

# The two versions of CUDA that we supported at the time
# were 10.2 and 11.1.1 these versions are important because
# we start from a base CUDA image (Done to avoid installing
# CUDA from scratch) and then install other dependencies
# on top of it.
ARG CUDA_VERSION=11.1.1

# The VERSION arg is used to specify the version of sparseml
# that we want to install. If it is not specified, the latest
# version of sparseml will be installed from pypi.
# valid values would be 1.1, 1.2, ..., 1.6 ... etc.
ARG VERSION

# DEPS is used to specify the extra that we want to install
# with sparseml. Valid values are all, base, dev (dev is not tested)
# base: will install sparseml with only torch and torchvision
# all: will install sparseml with all dependencies, namely 
# onnxruntime, torchvision, transformers, yolov5, ultralytics 
ARG DEPS="all"

# VENV arg specifies the path to the virtual environment
# that we want to create. This makes it easier to change
# the virtual environment path if needed.
# + allows us to create a slimmer image by not having to
# install the virtual environment in the same layer as the
# dependencies.

ARG VENV="/venv"

# BRANCH is used to specify the branch that we want to install
# sparseml from. If it is not specified, defaults to main
ARG BRANCH

# Setup the base image & install dependencies
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04 as base
# As of 05/05/22 nvidia images are broken. Two lines below are a temporary fix.
# Source: https://github.com/NVIDIA/nvidia-docker/issues/1632

ARG VENV
RUN if [ -f /etc/apt/sources.list.d/cuda.list ] ; then rm /etc/apt/sources.list.d/cuda.list ; fi
RUN if [ -f /etc/apt/sources.list.d/nvidia-ml.list ] ; then rm /etc/apt/sources.list.d/nvidia-ml.list ; fi

# Install system dependencies, including python and pip
RUN set -Eeuxo \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        git-all \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3-pip\
        build-essential libssl-dev libffi-dev \
        ffmpeg libsm6 libxext6 \
        curl 

# Install virtualenv and create a virtual environment
RUN set -Eeuxo \
    && apt-get update \
    && python3.8 -m pip install --upgrade virtualenv \
    && python3.8 -m virtualenv --python=/usr/bin/python3.8 $VENV \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add the virtual environment to the $PATH
# This is a workaround for activating the virtual environment
ENV PATH="${VENV}/bin:$PATH"

# Next Two stages are used to install torch and torchvision
# on top of the base CUDA image
FROM base as cuda-10.2
ARG TORCH_VERSION
ARG VENV
ARG TORCHVISION_VERSION
RUN $VENV/bin/pip install --upgrade pip torch==${TORCH_VERSION}+cu102 torchvision==${TORCHVISION_VERSION}+cu102  -f https://download.pytorch.org/whl/torch_stable.html

FROM base as cuda-11.1.1
ARG TORCH_VERSION
ARG VENV
ARG TORCHVISION_VERSION
RUN $VENV/bin/pip install --upgrade pip torch==${TORCH_VERSION}+cu111 torchvision==${TORCHVISION_VERSION}+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# This stage is used to clone `sparseml` from the specified branch
# if needed, if cloning is not needed, this just echo(s) a message
# and does nothing.

FROM cuda-$CUDA_VERSION as cuda_builder
ARG VENV
RUN $VENV/bin/pip install --upgrade setuptools wheel
ARG BRANCH
RUN \
   if [ -z "${BRANCH}"] ; then  \
    echo Will install from pypi based on mode and version ; \
  else  \
    echo cloning from "${BRANCH}" &&  \
    git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b "${BRANCH}" ; \
  fi;

# The stage responsible for installing sparseml
# Either installs from a BRANCH (if specified) and assumes
# that the branch is already cloned

# If BRANCH is not specified, it installs from pypi
# Based on the MODE and VERSION specified
# MODE can be set to nightly to install nightly VERSIONS
# VERSION can be set to a specific version of sparseml
FROM cuda_builder AS container_branch_base
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE=""
ARG BRANCH

RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH && \
      $VENV/bin/pip install --no-cache-dir "./sparseml"; \
    elif [ "$MODE" = "nightly" ] ; then \
      echo Installing nightlies ... && \
      if [ -z "$VERSION" ] ; then \
        echo Installing latest nightlies ...  && \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly"; \
      else \
        echo Installing nightlies ... with $VERSION && \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly==$VERSION"; \
      fi; \
    elif [ -z "$VERSION" ] ; then \
      echo Installing latest sparseml ... from pypi && \
      $VENV/bin/pip install --no-cache-dir "sparseml"; \
    else \
      echo Installing latest sparseml version="$VERSION" from pypi && \
      $VENV/bin/pip install --no-cache-dir "sparseml==$VERSION"; \
    fi;

# Similar to the previous stage, but installs all dependencies
# on top of sparseml
FROM cuda_builder AS container_branch_all
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE
ARG BRANCH

RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH && \
      $VENV/bin/pip install --no-cache-dir "./sparseml[onnxruntime,torchvision,transformers,yolov5,ultralytics]"; \
    elif [ "$MODE" = "nightly" ] ; then \
      if [ -z $VERSION] ; then \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly[onnxruntime,torchvision,transformers,yolov5,ultralytics]"; \
      else \
        $VENV/bin/pip install --no-cache-dir "sparseml-nightly[onnxruntime,torchvision,transformers,yolov5,ultralytics]==$VERSION"; \
      fi; \
    elif [ -z $VERSION] ; then \
      $VENV/bin/pip install --no-cache-dir "sparseml[onnxruntime,torchvision,transformers,yolov5,ultralytics]"; \
    else \
      $VENV/bin/pip install --no-cache-dir "sparseml[onnxruntime,torchvision,transformers,yolov5,ultralytics]==$VERSION"; \
    fi;

# Installs dev dependencies on top of sparseml
FROM cuda_builder AS container_branch_dev
ARG VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=200
ARG VERSION
ARG MODE
ARG BRANCH

RUN \
    if [ -n "$BRANCH" ] ; then \
      echo Installing from BRANCH with editable mode && \
      $VENV/bin/pip install -e  "./sparseml[dev]"; \
    else \
      echo Installing from main  with editable mode && \
      git clone https://github.com/neuralmagic/sparseml.git --depth 1 -b main && \
      $VENV/bin/pip install -e "./sparseml[dev]"; \
    fi;

# At this point, we have installed sparseml and its dependencies
# in the container. This is needed to provide an alias for the
# container that we can use to copy the virtual environment

FROM container_branch_${DEPS} AS build
RUN echo Build complete, going onto prod

# The final step is used to copy the virtual environment
# from the build stage to the final stage. This is done to
# create a slimmer image that only contains the virtual
# environment with the dependencies that we need.

# A separate stage is needed for dev and prod because
# dev installs sparseml in editable mode, which means
# that the virtual environment does not contain sparseml
# in the venv site-packages directory, but rather points
# to the sparseml directory in the container. So, we need
# to copy the sparseml directory from the container to the
# final stage.

FROM base as dev
ARG VENV
COPY --from=build $VENV $VENV
COPY --from=build sparseml sparseml
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import sparseml'
CMD bash

# The prod image's venv contains sparseml in the site-packages
# directory, so we do not need to copy the sparseml directory
# from the container to the final stage. We can just copy the
# virtual environment and set the PATH to the virtual environment
# bin directory.

FROM base as prod
ARG VENV
COPY --from=build $VENV $VENV
ENV PATH="${VENV}/bin:$PATH"
HEALTHCHECK CMD python -c 'import sparseml'
RUN pip list | grep sparseml
CMD bash

