#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: docker run IMAGE TEST_FILE RESULT_DIR" >&2
  exit 2
fi

TEST_FILE="$1"
RESULT_DIR="$2"
if [[ ! -f "$TEST_FILE" || "$TEST_FILE" != *.jsonl ]]; then
  echo "ERROR: TEST_FILE must be an existing .jsonl file: $TEST_FILE" >&2
  exit 1
fi
if [[ -e "$RESULT_DIR" && ! -d "$RESULT_DIR" ]]; then
  echo "ERROR: RESULT_DIR exists but is not a directory: $RESULT_DIR" >&2
  exit 1
fi
mkdir -p -- "$RESULT_DIR" || {
  echo "ERROR: cannot create RESULT_DIR: $RESULT_DIR" >&2
  exit 1
}
for artifact in submission.jsonl performance_report.json run_progress.json; do
  if [[ -e "$RESULT_DIR/$artifact" ]]; then
    echo "ERROR: RESULT_DIR already contains $artifact; use a new run directory" >&2
    exit 1
  fi
done
write_probe="$(mktemp "$RESULT_DIR/.docker-write-test.XXXXXX")" || {
  echo "ERROR: RESULT_DIR is not writable: $RESULT_DIR" >&2
  exit 1
}
rm -f -- "$write_probe"

unset VLLM_API_KEY EVALUATION_API_KEY

export CONDA_EXE=/opt/conda/bin/conda
export CONDA_PREFIX=/opt/conda/envs/conda_env
export CONDA_DEFAULT_ENV=conda_env
export PATH="$CONDA_PREFIX/bin:/opt/conda/bin:$PATH"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export VLLM_BIN="$CONDA_PREFIX/bin/vllm"
export EVALUATION_INFERENCE_CONFIG=/root/configs/deployment/e02_vllm_docker.json
export EVALUATION_HARDWARE_LABEL="${EVALUATION_HARDWARE_LABEL:-Docker GPU Smoke}"
export HF_HOME=/tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache
export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache
export PYTHONUNBUFFERED=1

exec bash /root/participant/start.sh "$@"
