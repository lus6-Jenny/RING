FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility,display

# RUN \
#     # Update nvidia GPG key
#     rm /etc/apt/sources.list.d/cuda.list && \
#     apt-key del 7fa2af80 && \
#     apt-get update && apt-get install -y --no-install-recommends wget && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
#     dpkg -i cuda-keyring_1.0-1_all.deb

# install packages
RUN apt-get update && apt-get install -y \
    build-essential ca-certificates lsb-release gnupg net-tools iputils-ping \
    cmake git curl wget vim zip unzip tmux openssh-server \
    python3-pip python3-setuptools python3-dev libpcl-dev

# install pytorch
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

RUN mkdir /home/RING
WORKDIR /home/RING
COPY . /home/RING
RUN pip3 install --no-cache-dir -r requirements.txt

# install torch-radon
RUN cd utils/torch-radon && \
    python3 setup.py install

# install fast_gicp
RUN cd utils/fast_gicp && \
    python3 setup.py install --user

# install BEV generation utils
RUN cd utils/generate_bev_occ_cython && python3 setup.py install && \
    cd ../generate_bev_pointfeat_cython && python3 setup.py install

# clean apt cache
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf /var/cache/apt/archives/*

SHELL ["/bin/bash", "-c"]
