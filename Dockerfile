FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN ln -s /runpod-volume /workspace 2>/dev/null || true

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

COPY ./requirements.txt /content/requirements.txt
RUN pip install -r /content/requirements.txt && \
    GIT_LFS_SKIP_SMUDGE=1 git clone -b dev https://github.com/camenduru/diffusers-image-outpaint-hf /content/outpaint

RUN [ ! -f /runpod-volume/model/lightning/model_index.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/model_index.json -d /runpod-volume/model/lightning -o model_index.json || true && \
    [ ! -f /runpod-volume/model/lightning/scheduler/scheduler_config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/scheduler/scheduler_config.json -d /runpod-volume/model/lightning/scheduler -o scheduler_config.json || true && \
    [ ! -f /runpod-volume/model/lightning/text_encoder/config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/text_encoder/config.json -d /runpod-volume/model/lightning/text_encoder -o config.json || true && \
    [ ! -f /runpod-volume/model/lightning/text_encoder/model.fp16.safetensors ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/resolve/main/lightning/text_encoder/model.fp16.safetensors -d /runpod-volume/model/lightning/text_encoder -o model.fp16.safetensors || true && \
    [ ! -f /runpod-volume/model/lightning/text_encoder_2/config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/text_encoder_2/config.json -d /runpod-volume/model/lightning/text_encoder_2 -o config.json || true && \
    [ ! -f /runpod-volume/model/lightning/text_encoder_2/model.fp16.safetensors ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/resolve/main/lightning/text_encoder_2/model.fp16.safetensors -d /runpod-volume/model/lightning/text_encoder_2 -o model.fp16.safetensors || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer/merges.txt ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer/merges.txt -d /runpod-volume/model/lightning/tokenizer -o merges.txt || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer/special_tokens_map.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer/special_tokens_map.json -d /runpod-volume/model/lightning/tokenizer -o special_tokens_map.json || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer/tokenizer_config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer/tokenizer_config.json -d /runpod-volume/model/lightning/tokenizer -o tokenizer_config.json || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer/vocab.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer/vocab.json -d /runpod-volume/model/lightning/tokenizer -o vocab.json || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer_2/merges.txt ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer_2/merges.txt -d /runpod-volume/model/lightning/tokenizer_2 -o merges.txt || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer_2/special_tokens_map.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer_2/special_tokens_map.json -d /runpod-volume/model/lightning/tokenizer_2 -o special_tokens_map.json || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer_2/tokenizer_config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer_2/tokenizer_config.json -d /runpod-volume/model/lightning/tokenizer_2 -o tokenizer_config.json || true && \
    [ ! -f /runpod-volume/model/lightning/tokenizer_2/vocab.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/tokenizer_2/vocab.json -d /runpod-volume/model/lightning/tokenizer_2 -o vocab.json || true && \
    [ ! -f /runpod-volume/model/lightning/unet/config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/unet/config.json -d /runpod-volume/model/lightning/unet -o config.json || true && \
    [ ! -f /runpod-volume/model/lightning/unet/diffusion_pytorch_model.fp16.safetensors ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/resolve/main/lightning/unet/diffusion_pytorch_model.fp16.safetensors -d /runpod-volume/model/lightning/unet -o diffusion_pytorch_model.fp16.safetensors || true && \
    [ ! -f /runpod-volume/model/lightning/unet/diffusion_pytorch_model.safetensors.index.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/unet/diffusion_pytorch_model.safetensors.index.json -d /runpod-volume/model/lightning/unet -o diffusion_pytorch_model.safetensors.index.json || true && \
    [ ! -f /runpod-volume/model/lightning/vae/config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/lightning/vae/config.json -d /runpod-volume/model/lightning/vae -o config.json || true && \
    [ ! -f /runpod-volume/model/lightning/vae/diffusion_pytorch_model.fp16.safetensors ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/resolve/main/lightning/vae/diffusion_pytorch_model.fp16.safetensors -d /runpod-volume/model/lightning/vae -o diffusion_pytorch_model.fp16.safetensors || true && \
    [ ! -f /runpod-volume/model/vae-fix/config.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/vae-fix/config.json -d /runpod-volume/model/vae-fix -o config.json || true && \
    [ ! -f /runpod-volume/model/vae-fix/diffusion_pytorch_model.safetensors ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/resolve/main/vae-fix/diffusion_pytorch_model.safetensors -d /runpod-volume/model/vae-fix -o diffusion_pytorch_model.safetensors || true && \
    [ ! -f /runpod-volume/model/union/config_promax.json ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/raw/main/union/config_promax.json -d /runpod-volume/model/union -o config_promax.json || true && \
    [ ! -f /runpod-volume/model/union/diffusion_pytorch_model_promax.safetensors ] && aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/outpaint/resolve/main/union/diffusion_pytorch_model_promax.safetensors -d /runpod-volume/model/union -o diffusion_pytorch_model_promax.safetensors || true

COPY ./worker_runpod.py /content/outpaint/worker_runpod.py
WORKDIR /content/outpaint
CMD python worker_runpod.py