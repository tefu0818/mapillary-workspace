FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # && apt-get update \
    # && apt-get install -y sudo \
    # && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    # && chmod 0440 /etc/sudoers.d/$USERNAME


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y software-properties-common tzdata

RUN apt update && \
      apt install -y \
      wget \
      bzip2 \
      build-essential \
      git \
      git-lfs \
      curl \
      ca-certificates \
      libsndfile1-dev \
      libgl1 \
      libgeos-dev

RUN apt update && apt install -y python3 python3-pip

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME