# docker build -t mmdetection3:pytorch1.10.0-cuda11.3-cudnn8-mmcv2.0.0rc3 .

ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    TZ=Asia/Tokyo \
    CUDA_DEVICE_ORDER="PCI_BUS_ID" \
    LANG="C.UTF-8"

# add japanese font
COPY src/mmdetection/docker/takao-gothic /usr/share/fonts/truetype/

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install -U pip && pip install openmim && \
    mim install "mmengine==0.4.0" "mmcv==2.0.0rc3"

RUN conda clean --all
WORKDIR /workspace

# Install MMDetection
COPY src/mmdetection /workspace/
RUN cd /workspace/mmdetection \
    && pip install --no-cache-dir -e . \
    && pip install -r requirements/albu.txt

# Install MMYOLO
RUN mim install "mmyolo"

# Install MMClassification
COPY src/mmclassification /workspace/
RUN cd /workspace/mmclassification \
    && pip install --no-cache-dir -e .
