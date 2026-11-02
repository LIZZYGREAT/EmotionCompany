# CUDA 12.8 matches the tested torch 2.9.0+cu128 / vLLM 0.11.1 runtime.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG MINIFORGE_VERSION=26.1.0-0
ARG MINIFORGE_SHA256=127b5e14cfe6c83b787f624487cdc2168645ed82fdca1b0c1937caa086aed6d5

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_EXE=/opt/conda/bin/conda
ENV CONDA_PREFIX=/opt/conda/envs/conda_env
ENV CONDA_DEFAULT_ENV=conda_env
ENV PATH=/opt/conda/envs/conda_env/bin:/opt/conda/bin:${PATH}
ENV PYTHON_BIN=/opt/conda/envs/conda_env/bin/python
ENV VLLM_BIN=/opt/conda/envs/conda_env/bin/vllm
ENV EVALUATION_INFERENCE_CONFIG=/root/configs/deployment/e02_vllm_docker.json
ENV HF_HOME=/tmp/huggingface
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV VLLM_CACHE_ROOT=/tmp/vllm-cache
ENV TRITON_CACHE_DIR=/tmp/triton-cache
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --silent --show-error \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
      --output /tmp/miniforge.sh \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum --check --status \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh

COPY docker/e02/conda-linux-64.explicit.txt /tmp/conda-linux-64.explicit.txt
COPY docker/e02/requirements-linux-64.lock /tmp/requirements-linux-64.lock
RUN /opt/conda/bin/conda create --yes --name conda_env --file /tmp/conda-linux-64.explicit.txt \
    && /opt/conda/envs/conda_env/bin/python -m pip install \
      --no-cache-dir --disable-pip-version-check \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --requirement /tmp/requirements-linux-64.lock \
    && /opt/conda/envs/conda_env/bin/python -m pip check \
    && rm -f /tmp/conda-linux-64.explicit.txt /tmp/requirements-linux-64.lock \
    && /opt/conda/bin/conda clean --all --yes

WORKDIR /root
COPY participant/start.sh /root/participant/start.sh
COPY participant/run_inference.py /root/participant/run_inference.py
COPY participant/templates/submission_schema.json /root/participant/templates/submission_schema.json
COPY configs/deployment/e02_vllm_docker.json /root/configs/deployment/e02_vllm_docker.json
COPY configs/prompts/p0.txt /root/configs/prompts/p0.txt
COPY src/shared /root/src/shared
COPY scripts/e02_vllm_probe.py /root/scripts/e02_vllm_probe.py
COPY participant/model /root/participant/model
COPY participant/adapter /root/participant/adapter
COPY package_manifest.json /root/package_manifest.json
COPY docker/e02/entrypoint.sh /usr/local/bin/emotionalcompany-e02-entrypoint

RUN chmod 0755 /usr/local/bin/emotionalcompany-e02-entrypoint \
    && test -x /opt/conda/envs/conda_env/bin/vllm \
    && test -f /root/participant/model/config.json \
    && test -f /root/participant/adapter/adapter_config.json \
    && test -f /root/package_manifest.json

ENTRYPOINT ["/usr/local/bin/emotionalcompany-e02-entrypoint"]
