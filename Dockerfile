FROM ubuntu:24.04

RUN apt-get update

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_BREAK_SYSTEM_PACKAGES=true

RUN apt-get -y install python3-pip wget

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
RUN mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-60

RUN wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb

RUN dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb

RUN cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

RUN apt-get update && apt-get -y install cuda-toolkit-13-0 sox libsox-fmt-all

RUN apt-get update && apt-get -y install \
    python3-pip python3-venv git ffmpeg curl ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y --no-modify-path

RUN python3 -m venv /opt/venv
ENV PATH="/root/.cargo/bin:/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir "qwentts-cpp-python==0.3.1+cu130" \
    -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130

# Install core dependencies + CUDA 12 runtime libs required by qwentts-cpp
RUN pip install --no-cache-dir torch torchvision torchaudio deepfilternet \
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cudnn-cu12 \
    && ln -s /opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 /usr/lib/libcudart.so.12 \
    && ln -s /opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12 /usr/lib/libcublas.so.12 \
    && ln -s /opt/venv/lib/python3.12/site-packages/nvidia/cufft/lib/libcufft.so.11 /usr/lib/libcufft.so.11 \
    && ln -s /opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/lib/libcudnn.so.9

# Install the local speech-to-speech with faster-whisper extra
COPY . /tmp/speech-to-speech/
RUN pip install --no-cache-dir "/tmp/speech-to-speech[faster-whisper]" \
    && rm -rf /tmp/speech-to-speech \
    && speech-to-speech --help > /dev/null \
    && echo "speech-to-speech installed successfully"

# Install flash-attention (from git submodule)
COPY flash-attention/ /tmp/flash-attention/
RUN MAX_JOBS=4 pip install /tmp/flash-attention --no-build-isolation -v \
    && rm -rf /tmp/flash-attention \
    && python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} loaded successfully')"

WORKDIR /app

CMD ["speech-to-speech"]
