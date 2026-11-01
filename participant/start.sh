#!/usr/bin/env bash
set -euo pipefail

# 启动虚拟环境conda_env, 名称固定，不要修改
activate_project_conda_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "conda_env" && -n "${CONDA_PREFIX:-}" ]]; then
    return 0
  fi

  local conda_command="${CONDA_EXE:-}"
  if [[ -z "$conda_command" || ! -x "$conda_command" ]]; then
    conda_command="$(type -P conda || true)"
  fi
  if [[ -z "$conda_command" || ! -x "$conda_command" ]]; then
    echo "ERROR: 找不到 Conda 可执行文件；请确保 conda 在 PATH 中，或设置 CONDA_EXE。入口需要环境 conda_env。" >&2
    exit 1
  fi

  local conda_hook
  if ! conda_hook="$("$conda_command" shell.bash hook 2>/dev/null)"; then
    echo "ERROR: 无法初始化 Conda shell hook: $conda_command" >&2
    exit 1
  fi
  eval "$conda_hook"
  if ! conda activate conda_env; then
    echo "ERROR: 无法激活 Conda 环境 conda_env。请检查该环境是否已创建。" >&2
    exit 1
  fi
}
activate_project_conda_env

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROBE_SCRIPT="$WORKSPACE_ROOT/scripts/e02_vllm_probe.py"
if [[ -n "${EVALUATION_INFERENCE_CONFIG:-}" && "$EVALUATION_INFERENCE_CONFIG" != /* ]]; then
  EVALUATION_INFERENCE_CONFIG="$WORKSPACE_ROOT/$EVALUATION_INFERENCE_CONFIG"
  export EVALUATION_INFERENCE_CONFIG
fi
CONFIG_FILE="${EVALUATION_INFERENCE_CONFIG:-$SCRIPT_DIR/configs/inference_config.json}"
[[ -f "$CONFIG_FILE" ]] || {
  echo "ERROR: 推理配置文件不存在: $CONFIG_FILE" >&2
  exit 1
}

# 服务默认配置；可直接修改，也可用同名环境变量覆盖。
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-}"
START_TIMEOUT="${VLLM_START_TIMEOUT:-600}"
PYTHON_CMD="${PYTHON_BIN:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_ARGS=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"

if [[ $# -ne 2 ]]; then
  echo "用法: $0 TEST_FILE RESULT_DIR" >&2
  exit 2
fi

TEST_FILE="$1"
RESULT_DIR="$2"

[[ -f "$TEST_FILE" && "$TEST_FILE" == *.jsonl ]] || {
  echo "ERROR: TEST_FILE 必须是存在的 .jsonl 文件: $TEST_FILE" >&2
  exit 1
}
if [[ -e "$RESULT_DIR" && ! -d "$RESULT_DIR" ]]; then
  echo "ERROR: RESULT_DIR 已存在但不是目录: $RESULT_DIR" >&2
  exit 1
fi
command -v "$PYTHON_CMD" >/dev/null 2>&1 || {
  echo "ERROR: 找不到 Python 命令: $PYTHON_CMD" >&2
  exit 1
}

mapfile -d '' -t CONFIG_FACTS < <("$PYTHON_CMD" - "$CONFIG_FILE" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
adapter = config.get("adapter") or {}
vllm = config.get("deployment", {}).get("vllm", {})
values = [
    config.get("backend", "transformers"),
    config.get("model_path", ""),
    config.get("model_id", ""),
    config.get("api_model", ""),
    "true" if adapter.get("enabled") is True else "false",
    adapter.get("path", ""),
    adapter.get("expected_base_weight_sha256", ""),
    adapter.get("expected_adapter_weight_sha256", ""),
    adapter.get("expected_adapter_config_sha256", ""),
    adapter.get("expected_rank", adapter.get("r", "")),
    adapter.get("expected_server_model_id", ""),
    vllm.get("expected_version", ""),
    vllm.get("dtype", ""),
    vllm.get("max_num_seqs", ""),
    vllm.get("max_num_batched_tokens", ""),
    vllm.get("gpu_memory_utilization", ""),
]
for value in values:
    sys.stdout.buffer.write(str(value).encode("utf-8") + b"\0")
PY
)
[[ "${#CONFIG_FACTS[@]}" -eq 16 ]] || {
  echo "ERROR: 无法从推理配置读取部署参数: $CONFIG_FILE" >&2
  exit 1
}
CONFIG_BACKEND="${CONFIG_FACTS[0]:-transformers}"
CONFIG_MODEL_PATH="${CONFIG_FACTS[1]:-}"
CONFIG_MODEL_ID="${CONFIG_FACTS[2]:-}"
CONFIG_API_MODEL="${CONFIG_FACTS[3]:-}"
CONFIG_ADAPTER_ENABLED="${CONFIG_FACTS[4]:-false}"
CONFIG_ADAPTER_PATH="${CONFIG_FACTS[5]:-}"
CONFIG_ADAPTER_BASE_SHA256="${CONFIG_FACTS[6]:-}"
CONFIG_ADAPTER_WEIGHT_SHA256="${CONFIG_FACTS[7]:-}"
CONFIG_ADAPTER_CONFIG_SHA256="${CONFIG_FACTS[8]:-}"
CONFIG_ADAPTER_RANK="${CONFIG_FACTS[9]:-}"
CONFIG_ADAPTER_SERVER_MODEL_ID="${CONFIG_FACTS[10]:-}"
CONFIG_VLLM_EXPECTED_VERSION="${CONFIG_FACTS[11]:-}"
CONFIG_VLLM_DTYPE="${CONFIG_FACTS[12]:-}"
CONFIG_VLLM_MAX_NUM_SEQS="${CONFIG_FACTS[13]:-}"
CONFIG_VLLM_MAX_NUM_BATCHED_TOKENS="${CONFIG_FACTS[14]:-}"
CONFIG_VLLM_GPU_MEMORY_UTILIZATION="${CONFIG_FACTS[15]:-}"

BACKEND="${EVALUATION_BACKEND:-}"
if [[ -z "$BACKEND" ]]; then
  BACKEND="$CONFIG_BACKEND"
fi

if [[ "$BACKEND" == "transformers" ]]; then
  echo "后端: transformers（直接加载本地模型）"
  export EVALUATION_BACKEND="transformers"
  exec "$PYTHON_CMD" "$SCRIPT_DIR/run_inference.py" "$TEST_FILE" "$RESULT_DIR"
fi

if [[ "$BACKEND" != "openai_compatible" ]]; then
  echo "ERROR: backend 必须为 transformers 或 openai_compatible: $BACKEND" >&2
  exit 1
fi

mapfile -d '' -t GPU_MEMORY_FACTS < <("$PYTHON_CMD" "$PROBE_SCRIPT" resolve-gpu-memory \
  --config-value "$CONFIG_VLLM_GPU_MEMORY_UTILIZATION" \
  --override-value "$VLLM_GPU_MEMORY_UTILIZATION" \
  --format nul)
[[ "${#GPU_MEMORY_FACTS[@]}" -eq 4 ]] || {
  echo "ERROR: 无法解析 deployment.vllm.gpu_memory_utilization" >&2
  exit 1
}
GPU_MEMORY_CONFIG_VALUE="${GPU_MEMORY_FACTS[0]}"
GPU_MEMORY_OVERRIDE_VALUE="${GPU_MEMORY_FACTS[1]}"
GPU_MEMORY_EFFECTIVE_VALUE="${GPU_MEMORY_FACTS[2]}"
GPU_MEMORY_SOURCE="${GPU_MEMORY_FACTS[3]}"
if [[ -n "$GPU_MEMORY_EFFECTIVE_VALUE" ]]; then
  VLLM_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_EFFECTIVE_VALUE")
fi

[[ "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: VLLM_TENSOR_PARALLEL_SIZE 必须为正整数" >&2
  exit 1
}
if [[ -n "$CONFIG_VLLM_DTYPE" ]]; then
  case "$CONFIG_VLLM_DTYPE" in
    auto|bfloat16|float16|float32) VLLM_ARGS+=(--dtype "$CONFIG_VLLM_DTYPE") ;;
    *) echo "ERROR: deployment.vllm.dtype 必须为 auto/bfloat16/float16/float32" >&2; exit 1 ;;
  esac
fi
if [[ -n "$CONFIG_VLLM_MAX_NUM_SEQS" ]]; then
  [[ "$CONFIG_VLLM_MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: deployment.vllm.max_num_seqs 必须为正整数" >&2
    exit 1
  }
  VLLM_ARGS+=(--max-num-seqs "$CONFIG_VLLM_MAX_NUM_SEQS")
fi
if [[ -n "$CONFIG_VLLM_MAX_NUM_BATCHED_TOKENS" ]]; then
  [[ "$CONFIG_VLLM_MAX_NUM_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: deployment.vllm.max_num_batched_tokens 必须为正整数" >&2
    exit 1
  }
  VLLM_ARGS+=(--max-num-batched-tokens "$CONFIG_VLLM_MAX_NUM_BATCHED_TOKENS")
fi

read -r PROMPT_LIMIT GENERATION_LIMIT CONTEXT_LIMIT < <("$PYTHON_CMD" -c \
  'import json, os, sys; c=json.load(open(sys.argv[1], encoding="utf-8")); x=c.get("context", {}); g=c.get("generation", {}); p=os.environ.get("EVALUATION_MAX_PROMPT_TOKENS", x.get("max_prompt_tokens")); k=os.environ.get("EVALUATION_SERVING_CONTEXT_LIMIT", x.get("serving_context_limit")); print(p or "", g.get("max_new_tokens", ""), k or "")' \
  "$CONFIG_FILE")
[[ "$PROMPT_LIMIT" =~ ^[1-9][0-9]*$ && "$GENERATION_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: 配置必须显式提供正整数 P 和 G" >&2
  exit 1
}
if [[ -n "$CONTEXT_LIMIT" ]]; then
  [[ "$CONTEXT_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: EVALUATION_SERVING_CONTEXT_LIMIT 必须为正整数或留空" >&2
    exit 1
  }
  ((PROMPT_LIMIT + GENERATION_LIMIT <= CONTEXT_LIMIT)) || {
    echo "ERROR: 部署预算不兼容：P($PROMPT_LIMIT) + G($GENERATION_LIMIT) > K($CONTEXT_LIMIT)" >&2
    exit 1
  }
  if [[ -n "$CONFIG_VLLM_MAX_NUM_BATCHED_TOKENS" ]] && ((CONFIG_VLLM_MAX_NUM_BATCHED_TOKENS < PROMPT_LIMIT + GENERATION_LIMIT)); then
    echo "ERROR: deployment.vllm.max_num_batched_tokens 小于 P+G 预算" >&2
    exit 1
  fi
  VLLM_ARGS+=(--max-model-len "$CONTEXT_LIMIT")
  echo "部署预算预检通过：P=$PROMPT_LIMIT, G=$GENERATION_LIMIT, K=$CONTEXT_LIMIT"
else
  echo "部署预算：P=$PROMPT_LIMIT, G=$GENERATION_LIMIT, K 未配置；服务上下文能力 not_verified"
fi

MODEL_DIR="${VLLM_MODEL_DIR:-${EVALUATION_MODEL_PATH:-$CONFIG_MODEL_PATH}}"
if [[ -n "$MODEL_DIR" && "$MODEL_DIR" != /* ]]; then
  if [[ -d "$WORKSPACE_ROOT/$MODEL_DIR" ]]; then
    MODEL_DIR="$WORKSPACE_ROOT/$MODEL_DIR"
  else
    MODEL_DIR="$SCRIPT_DIR/$MODEL_DIR"
  fi
fi
MODEL_BASENAME="$(basename -- "$MODEL_DIR")"
MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$MODEL_BASENAME}"

[[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] || {
  echo "ERROR: vLLM 模型目录不存在或缺少 config.json: $MODEL_DIR" >&2
  exit 1
}
command -v "$VLLM_BIN" >/dev/null 2>&1 || {
  echo "ERROR: 找不到 vLLM 命令: $VLLM_BIN" >&2
  exit 1
}

LORA_ENABLED="${VLLM_LORA_ENABLED:-$CONFIG_ADAPTER_ENABLED}"
case "$LORA_ENABLED" in
  1|true|TRUE|yes|YES) LORA_ENABLED=true ;;
  0|false|FALSE|no|NO) LORA_ENABLED=false ;;
  *)
    echo "ERROR: VLLM_LORA_ENABLED 必须为 true/false" >&2
    exit 1
    ;;
esac
if [[ "$CONFIG_ADAPTER_ENABLED" == "true" && "$LORA_ENABLED" != "true" ]]; then
  echo "ERROR: 当前推理配置启用了 Adapter，但 VLLM_LORA_ENABLED 关闭；拒绝退化为 Base-only" >&2
  exit 1
fi

REQUEST_MODEL_NAME="$MODEL_NAME"
if [[ "$LORA_ENABLED" == "true" ]]; then
  LORA_NAME="${VLLM_LORA_NAME:-${EVALUATION_API_MODEL:-$CONFIG_API_MODEL}}"
  LORA_PATH="${VLLM_LORA_PATH:-${EVALUATION_ADAPTER_PATH:-$CONFIG_ADAPTER_PATH}}"
  MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-$CONFIG_ADAPTER_RANK}"
  [[ -n "$CONFIG_ADAPTER_BASE_SHA256" && -n "$CONFIG_ADAPTER_WEIGHT_SHA256" && -n "$CONFIG_ADAPTER_CONFIG_SHA256" ]] || {
    echo "ERROR: Adapter 已启用，但配置缺少冻结的 Base/Adapter SHA-256" >&2
    exit 1
  }
  [[ -n "$LORA_NAME" && "$LORA_NAME" != "$MODEL_NAME" ]] || {
    echo "ERROR: VLLM_LORA_NAME 必须非空且不同于 Base served-model-name" >&2
    exit 1
  }
  if [[ -n "$CONFIG_ADAPTER_SERVER_MODEL_ID" && "$LORA_NAME" != "$CONFIG_ADAPTER_SERVER_MODEL_ID" ]]; then
    echo "ERROR: LoRA 别名与配置的 expected_server_model_id 不匹配" >&2
    exit 1
  fi
  [[ "$MAX_LORA_RANK" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: VLLM_MAX_LORA_RANK 必须为正整数" >&2
    exit 1
  }
  if [[ -n "$LORA_PATH" && "$LORA_PATH" != /* ]]; then
    if [[ -d "$WORKSPACE_ROOT/$LORA_PATH" ]]; then
      LORA_PATH="$WORKSPACE_ROOT/$LORA_PATH"
    else
      LORA_PATH="$SCRIPT_DIR/$LORA_PATH"
    fi
  fi
  [[ -d "$LORA_PATH" && -f "$LORA_PATH/adapter_config.json" ]] || {
    echo "ERROR: Adapter 路径不存在或缺少 adapter_config.json: $LORA_PATH" >&2
    exit 1
  }
  [[ -f "$PROBE_SCRIPT" ]] || {
    echo "ERROR: 缺少 E02 vLLM 身份检查程序: $PROBE_SCRIPT" >&2
    exit 1
  }
  "$PYTHON_CMD" "$PROBE_SCRIPT" preflight \
    --base-path "$MODEL_DIR" \
    --adapter-path "$LORA_PATH" \
    --model-id "$CONFIG_MODEL_ID" \
    --alias "$LORA_NAME" \
    --base-alias "$MODEL_NAME" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --expected-base-sha256 "$CONFIG_ADAPTER_BASE_SHA256" \
    --expected-adapter-sha256 "$CONFIG_ADAPTER_WEIGHT_SHA256" \
    --expected-adapter-config-sha256 "$CONFIG_ADAPTER_CONFIG_SHA256" \
    --expected-vllm-version "$CONFIG_VLLM_EXPECTED_VERSION" \
    --vllm-bin "$VLLM_BIN"
  VLLM_ARGS+=(--enable-lora --lora-modules "$LORA_NAME=$LORA_PATH" --max-lora-rank "$MAX_LORA_RANK")
  REQUEST_MODEL_NAME="$LORA_NAME"
fi
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) || {
  echo "ERROR: VLLM_PORT 必须是 1 到 65535 之间的整数" >&2
  exit 1
}
[[ "$START_TIMEOUT" =~ ^[0-9]+$ ]] && ((START_TIMEOUT >= 1)) || {
  echo "ERROR: VLLM_START_TIMEOUT 必须为正整数" >&2
  exit 1
}

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

CLIENT_HOST="$HOST"
[[ "$CLIENT_HOST" == "0.0.0.0" || "$CLIENT_HOST" == "::" ]] && CLIENT_HOST="127.0.0.1"
BASE_URL="http://$CLIENT_HOST:$PORT/v1"
HEALTH_URL="http://$CLIENT_HOST:$PORT/health"
if [[ -n "${VLLM_LOG_FILE:-}" ]]; then
  LOG_FILE="$VLLM_LOG_FILE"
  mkdir -p -- "$(dirname -- "$LOG_FILE")"
  [[ ! -e "$LOG_FILE" ]] || {
    echo "ERROR: VLLM_LOG_FILE 已存在，拒绝覆盖: $LOG_FILE" >&2
    exit 1
  }
else
  LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/competition-vllm.XXXXXX.log")"
fi

COMMAND=(
  "$VLLM_BIN" serve "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$MODEL_NAME"
  "${VLLM_ARGS[@]}"
)
[[ -n "$API_KEY" ]] && COMMAND+=(--api-key "$API_KEY")
SAFE_COMMAND=()
REDACT_NEXT_ARGUMENT=false
for argument in "${COMMAND[@]}"; do
  if [[ "$REDACT_NEXT_ARGUMENT" == "true" ]]; then
    SAFE_COMMAND+=("<redacted>")
    REDACT_NEXT_ARGUMENT=false
  elif [[ "$argument" == "--api-key" ]]; then
    SAFE_COMMAND+=("$argument")
    REDACT_NEXT_ARGUMENT=true
  else
    SAFE_COMMAND+=("$argument")
  fi
done
export EVALUATION_VLLM_LAUNCH_FACTS_JSON="$("$PYTHON_CMD" - "${SAFE_COMMAND[@]}" \
  "$GPU_MEMORY_CONFIG_VALUE" "$GPU_MEMORY_OVERRIDE_VALUE" \
  "$GPU_MEMORY_EFFECTIVE_VALUE" "$GPU_MEMORY_SOURCE" <<'PY'
import json
import sys

arguments = sys.argv[1:-4]
memory_config, memory_override, memory_effective, memory_source = sys.argv[-4:]
print(json.dumps({
    "argv": arguments,
    "api_key_configured": "--api-key" in arguments,
    "api_key_value": "redacted" if "--api-key" in arguments else None,
    "gpu_memory_utilization": memory_effective or "vLLM default; not overridden",
    "gpu_memory_utilization_config_value": memory_config or None,
    "gpu_memory_utilization_override_value": memory_override or None,
    "gpu_memory_utilization_effective_value": memory_effective or None,
    "gpu_memory_utilization_source": memory_source,
}, ensure_ascii=False, separators=(",", ":")))
PY
)"

VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "启动 vLLM: $MODEL_DIR -> $BASE_URL"
echo "vLLM tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "请求模型 ID: $REQUEST_MODEL_NAME"
echo "vLLM 日志: $LOG_FILE"
"${COMMAND[@]}" >"$LOG_FILE" 2>&1 &
VLLM_PID=$!

READY=0
for ((second = 0; second < START_TIMEOUT; second++)); do
  kill -0 "$VLLM_PID" 2>/dev/null || break
  if "$PYTHON_CMD" -c \
    'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' \
    "$HEALTH_URL" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if ((READY == 0)); then
  echo "ERROR: vLLM 未能在 ${START_TIMEOUT}s 内就绪" >&2
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "vLLM 已就绪，开始推理测试"
export EVALUATION_GPU_FACTS_JSON="$("$PYTHON_CMD" "$WORKSPACE_ROOT/scripts/e02_vllm_probe.py" gpu-facts)"
SERVER_MODELS_JSON="$("$PYTHON_CMD" "$WORKSPACE_ROOT/scripts/e02_vllm_probe.py" models \
  --base-url "$BASE_URL" \
  --requested-model "$REQUEST_MODEL_NAME" \
  --api-key "$API_KEY")"
export EVALUATION_SERVER_MODELS_JSON="$("$PYTHON_CMD" -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["server_model_ids"]))' <<<"$SERVER_MODELS_JSON")"
if [[ "$LORA_ENABLED" == "true" && -n "$CONFIG_VLLM_EXPECTED_VERSION" ]]; then
  export EVALUATION_VLLM_VERSION="$CONFIG_VLLM_EXPECTED_VERSION"
else
  export EVALUATION_VLLM_VERSION="$("$VLLM_BIN" --version 2>&1 | head -n 1)"
fi
export EVALUATION_BACKEND="openai_compatible"
export EVALUATION_BASE_URL="$BASE_URL"
export EVALUATION_API_MODEL="$REQUEST_MODEL_NAME"
export EVALUATION_API_KEY="$API_KEY"
export EVALUATION_MODEL_PATH="$MODEL_DIR"
[[ "$LORA_ENABLED" != "true" ]] || export EVALUATION_ADAPTER_PATH="$LORA_PATH"
export EVALUATION_HARDWARE_LABEL="${EVALUATION_HARDWARE_LABEL:-$(hostname)}"

"$PYTHON_CMD" "$SCRIPT_DIR/run_inference.py" "$TEST_FILE" "$RESULT_DIR"

if [[ -n "${E02_VLLM_RUN_ROOT:-}" ]]; then
  RUN_ROOT="$E02_VLLM_RUN_ROOT"
  if [[ "$RUN_ROOT" != /* ]]; then
    RUN_ROOT="$WORKSPACE_ROOT/$RUN_ROOT"
  fi
  "$PYTHON_CMD" "$WORKSPACE_ROOT/scripts/e02_vllm_probe.py" write-manifest \
    --run-root "$RUN_ROOT" \
    --runner-dir "$RESULT_DIR" \
    --test-file "$TEST_FILE" \
    --config "$CONFIG_FILE" \
    --vllm-version "$EVALUATION_VLLM_VERSION" \
    --server-model-ids-json "$EVALUATION_SERVER_MODELS_JSON" \
    --requested-model "$REQUEST_MODEL_NAME" \
    --gpu-facts-json "$EVALUATION_GPU_FACTS_JSON"
fi

echo "测试完成: $RESULT_DIR"
echo "vLLM 日志: $LOG_FILE"
