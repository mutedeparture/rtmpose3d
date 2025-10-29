# https://hub.docker.com/r/pytorch/pytorch/tags
ARG PYTORCH_VERSION=2.4.0
ARG CUDA_VERSION=12.1

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-runtime


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
