#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/entrypoint_args.sh"
final_dual_parse_entrypoint_args "$@" || exit $?
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
export VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-900}"
export EVALUATION_INFERENCE_CONFIG=/root/configs/deployment/final_dual_docker.json
export EVALUATION_BACKEND=vllm
export EVALUATION_RESUME=0
export HF_HOME=/tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache
export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" - <<'PY'
import shutil
import subprocess
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("DOCKER_GPU_PREFLIGHT=failed: torch cannot access a CUDA device")
nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi is None:
    raise SystemExit("DOCKER_GPU_PREFLIGHT=failed: nvidia-smi is unavailable")
probe = subprocess.run(
    [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True,
    text=True,
    timeout=10,
    check=False,
)
if probe.returncode != 0 or not probe.stdout.strip():
    raise SystemExit("DOCKER_GPU_PREFLIGHT=failed: nvidia-smi could not observe a GPU")
print(f"DOCKER_GPU_PREFLIGHT=passed device={torch.cuda.get_device_name(0)}")
print(probe.stdout.strip())
PY

exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"
