#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/entrypoint_args.sh"
final_dual_parse_entrypoint_args "$@" || exit $?

if [[ "${ENTRYPOINT_IDLE:-false}" == "true" ]]; then
  exec sleep infinity
fi

unset VLLM_API_KEY EVALUATION_API_KEY
export CONDA_EXE=/opt/conda/bin/conda
export CONDA_PREFIX=/opt/conda/envs/conda_env
export CONDA_DEFAULT_ENV=conda_env
export PATH="$CONDA_PREFIX/bin:/opt/conda/bin:$PATH"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export VLLM_BIN="$CONDA_PREFIX/bin/vllm"
export VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-900}"
export EVALUATION_INFERENCE_CONFIG=/root/configs/deployment/final_dual_docker.json
export EVALUATION_BACKEND=vllm
export HF_HOME=/tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache
export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"
