ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

# Use newer PyTorch base image
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Configure CUDA architectures for modern GPUs
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Add NVIDIA GPG keys (updated for Ubuntu 22.04)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
	
# Install system dependencies (updated packages)
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
	
# Update pip and install basic Python packages
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install Cython with newer version
RUN pip install --no-cache-dir cython==3.0.8

RUN pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# Install OpenMMLab packages using MIM
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmdet>=3.0.0"
	
# Install pycocotools
RUN pip install --no-cache-dir pycocotools

# Install MMPose (latest stable version)
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN pip install -r requirements.txt && pip install --no-cache-dir -v -e .

# Install modern Jupyter ecosystem
RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    jupyterlab \
    ipykernel \
    ipywidgets \
    nbformat>=5.7.0 \
    jupyter_server>=2.0.0
	
# Register Jupyter kernel
RUN python -m ipykernel install --user --name mmpose_new --display-name "Python (MMPose_New)"

# Configure Jupyter
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py
	
# Install additional useful packages
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm
	
# Expose port for Jupyter
EXPOSE 8888

# Set working directory
WORKDIR /workspace

# Start Jupyter Lab instead of classic notebook
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]