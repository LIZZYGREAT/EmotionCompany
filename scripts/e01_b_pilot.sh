#!/usr/bin/env bash
# Explicit user entry point: Pilot preflight -> checkpoint pause -> same-Run resume.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_python="${PYTHON_BIN:-$task_root/.venv/bin/python}"
task_run_id="${1:?Usage: bash scripts/e01_b_pilot.sh E01-B-PILOT-UNIQUE_ID}"
if [[ ! "$task_run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,75}$ ]]; then
  echo 'Pilot Run ID must use 3-76 ASCII letters, digits, dots, underscores or hyphens.' >&2
  exit 2
fi
task_config="configs/training/e01_b_pilot.yaml"
task_preflight_id="${task_run_id}-preflight"
task_run_root="outputs/training/$task_run_id"
"$task_python" -m src.training.training_sft --config "$task_config" \
  --mode preflight --run-id "$task_preflight_id"
"$task_python" -m src.training.training_sft --config "$task_config" \
  --mode train --run-id "$task_run_id" \
  --preflight-run-root "outputs/training/$task_preflight_id" --stop-after-step 32
"$task_python" - "$task_run_root" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
if manifest.get('status') != 'interrupted' or manifest.get('stop_reason') != 'requested_checkpoint_pause':
    raise SystemExit('Pilot did not reach its requested checkpoint pause; resume was not started.')
PY
"$task_python" -m src.training.training_sft --config "$task_config" \
  --mode resume --run-root "$task_run_root" --checkpoint "$task_run_root/trainer/checkpoint-32"
"$task_python" scripts/e01_b_config.py review --pilot-run-root "$task_run_root"
