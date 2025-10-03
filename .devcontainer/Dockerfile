FROM    ubuntu:focal
SHELL   ["/bin/bash", "-c"]
# Set non-interactive mode globally
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.9 \
        python3.9-venv \
        python3-pip \
        build-essential \
        git \
        libftdi-dev \
        libftdi1 \
        doxygen \
        libsdl2-dev \
        curl \
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
        wget \
        unzip \
        sed && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1


#RUN     stat /etc/udev/
#RUN     cp /usr/share/gap8-openocd/openocd/contrib/60-openocd.rules /etc/udev/rules.d
#RUN     udevadm control --reload-rules && sudo udevadm trigger
#RUN     usermod -a -G dialout ubuntu
# GAP-SDK & TOOLCHAIN INSTALLATION
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
        git checkout a3dedd5cd8a680a88d2dca2ab7a4ae65cebf4c8d && \
        python3 -m pip install wheel && \
        python3 -m pip install scons && \
        python3 -m pip install Cython && \
        python3 -m pip install --upgrade pip setuptools wheel && \
        python3 -m pip install setuptools_rust && \
        python3 -m pip install -r requirements.txt
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
# FINALIZE PYTHON VENV
WORKDIR /gap_riscv_toolchain_ubuntu_18/gap_sdk/
RUN     source /dory_env/bin/activate && \
        python3 -m pip install python-dev-tools --upgrade && \
        python3 -m pip install numpy && \
        python3 -m pip install onnx onnxscript onnxoptimizer && \
        python3 -m pip install future-fstrings && \
        python3 -m pip install pandas && \
        python3 -m pip install ortools && \
        python3 -m pip install mako && \
        python3 -m pip install pytest && \
        python3 -m pip install Pygments && \
        python3 -m pip install MarkupSafe && \
        python3 -m pip install prettytable && \
        cd /gap_riscv_toolchain_ubuntu_18/gap_sdk/ && \
        python3 -m pip uninstall -y torch torchvision && \
        python3 -m pip install torch torchvision && \
        python3 -m pip install brevitas pytorch-lightning tensorboard

# -------------------------
# Add vscode user for VS Code
# -------------------------
ARG USERNAME=dory
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
