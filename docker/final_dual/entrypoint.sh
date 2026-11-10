#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/entrypoint_args.sh"
final_dual_parse_entrypoint_args "$@" || exit $?

if [[ "${ENTRYPOINT_IDLE:-false}" == "true" ]]; then
  exec sleep infinity
fi

# Defaults only.  The caller may override any of these; do not reject the
# container just because the host or platform provides a different value.
export CONDA_EXE="${CONDA_EXE:-/opt/conda/bin/conda}"
export CONDA_PREFIX="${CONDA_PREFIX:-/opt/conda/envs/conda_env}"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-conda_env}"
# Always put the image's fixed conda_env first.  The platform may expose a
# different CONDA_PREFIX, but the vLLM runtime is installed in this one.
export PATH="/opt/conda/envs/conda_env/bin:/opt/conda/bin:${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/conda_env/bin/python}"
export VLLM_BIN="${VLLM_BIN:-/usr/local/bin/emotionalcompany-vllm}"
export VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-900}"
export EVALUATION_BACKEND="${EVALUATION_BACKEND:-openai_compatible}"
export EVALUATION_INFERENCE_CONFIG="${EVALUATION_INFERENCE_CONFIG:-/root/participant/configs/inference_config.json}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${ENTRYPOINT_START_SH:-false}" == "true" ]]; then
  exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"
fi

if [[ -n "${ENTRYPOINT_SHELL_COMMAND:-}" ]]; then
  exec bash -lc "$ENTRYPOINT_SHELL_COMMAND"
fi

# Pass the caller's command through unchanged.  This covers
# `bash /root/participant/start.sh ...`, `/root/participant/start.sh ...`,
# `bash -lc "..."`, `/bin/bash -c "..."`, `sh -c "..."`, docker exec
# commands, and any other wrapper the platform may use.
exec "$@"
