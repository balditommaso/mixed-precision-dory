FROM    ubuntu:focal
SHELL   ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        sudo \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        liblzma-dev \
        tk-dev \
        uuid-dev \
        libftdi-dev \
        libftdi1 \
        doxygen \
        libsdl2-dev \
        cmake \
        libusb-1.0-0-dev \
        gtkwave \
        libsndfile1-dev \
        rsync \
        autoconf \
        automake \
        texinfo \
        libtool \
        pkg-config \
        libsdl2-ttf-dev \
        unzip \
        ssh \
        sed && \
    cd /usr/src && \
    curl -fsSL -O https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz && \
    tar -xf Python-3.10.15.tgz && \
    cd Python-3.10.15 && \
    ./configure --enable-optimizations --enable-shared --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    ldconfig && \
    cd / && rm -rf /usr/src/Python-3.10.15* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/local/bin/python3.10

RUN     python3 -m venv /dory_env && \
        source /dory_env/bin/activate && \
        git clone https://github.com/GreenWaves-Technologies/gap8_openocd.git && \
        cd gap8_openocd && \
        ./bootstrap && \
        ./configure --program-prefix=gap8- --prefix=/usr --datarootdir=/usr/share/gap8-openocd && \
        make -j && \
        make -j install

RUN     source /dory_env/bin/activate && \
        cd / && \
        git clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18.git && \
        cd /gap_riscv_toolchain_ubuntu_18 && \
        ./install.sh /usr/lib/gap_riscv_toolchain && \
        git clone https://github.com/GreenWaves-Technologies/gap_sdk/ && \
        cd gap_sdk && \
        git checkout a3dedd5cd8a680a88d2dca2ab7a4ae65cebf4c8d

COPY requirements-build.txt /tmp/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    source /dory_env/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements-build.txt

RUN     source /dory_env/bin/activate && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk && \
        source sourceme.sh && \
        make minimal && \
        make gvsoc && \
        # PULP-SDK INSTALLATION
        cd / && \
        git clone https://github.com/pulp-platform/pulp-sdk.git && \
        cd pulp-sdk && \
        git checkout 1f6a59e9b3def1585cd726872ea93b134a46faa2 && \
        source configs/pulp-open-nn.sh && \
        make all && \
        # PULP-NN TOOLCHAIN DOWNLOAD
        cd / && \
        wget https://iis-nextcloud.ee.ethz.ch/s/aYESyR5W9FrHgYa/download/riscv-nn-toolchain.zip && \
        unzip riscv-nn-toolchain


WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    source /dory_env/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip uninstall -y torch torchvision || true && \
    python3 -m pip install -r /tmp/requirements.txt


ARG USERNAME=dory
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN chown -R $USER_UID:$USER_GID /dory_env

USER $USERNAME

RUN echo "source /dory_env/bin/activate" >> /home/$USERNAME/.bashrc && \
    echo "alias gap-init='source /gap_riscv_toolchain_ubuntu_18/gap_sdk/sourceme.sh'" >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]