"""Run E01-A train-fit artifact checks and score the fixed E00 Dev Smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference
from src.evaluation.score import score_files
from src.training.training_sft import sha256_file


SMOKE_ROOT = ROOT / "outputs/evaluation/e00_1/dev_adapter_v2"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(path: Path, value: Any, *, yaml_format: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation snapshot: {path}")
    if yaml_format:
        path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_run(run_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = run_root.resolve()
    output_root = (ROOT / "outputs/training").resolve()
    if not root.is_relative_to(output_root):
        raise ValueError("--run-root must be an E01 run under outputs/training")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"E01 manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E01-A" or manifest.get("status") != "passed":
        raise ValueError("Dev Smoke accepts only an engineering-passed E01-A run")
    adapter_identity = manifest.get("adapter_identity")
    if not isinstance(adapter_identity, dict) or adapter_identity.get("identity_status") != "observed":
        raise ValueError("E01-A run has no observed local Adapter identity")
    adapter_path = root / "adapter"
    identity, adapter_config = run_inference.adapter_weight_identity(adapter_path)
    if identity["weight_sha256"] != adapter_identity.get("weight_sha256"):
        raise ValueError("saved Adapter weights changed after the E01-A run was recorded")
    if identity["config_sha256"] != adapter_identity.get("config_sha256"):
        raise ValueError("saved Adapter config changed after the E01-A run was recorded")
    if adapter_config.get("base_model_name_or_path") != "Qwen/Qwen3-1.7B":
        raise ValueError("E01-A Adapter config does not identify Qwen/Qwen3-1.7B")
    return root, manifest, identity


def _evaluation_attempt_root(root: Path, attempt_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", attempt_id):
        raise ValueError("--attempt-id must use 1-96 ASCII letters, digits, dots, underscores, or hyphens")
    evaluation_root = root / "evaluation" / "dev_smoke" / attempt_id
    if evaluation_root.exists():
        raise FileExistsError(f"refusing to overwrite Dev Smoke attempt: {evaluation_root}")
    return evaluation_root


def _prepare_snapshots(
    root: Path,
    manifest: dict[str, Any],
    identity: dict[str, Any],
    attempt_id: str,
) -> tuple[Path, Path, Path]:
    if not (SMOKE_ROOT / "smoke_input.jsonl").is_file() or not (SMOKE_ROOT / "smoke_references.jsonl").is_file():
        raise FileNotFoundError("fixed E00.1 Dev Smoke inputs/references are missing; rebuild via docs/workflow.md")
    base_sha = manifest.get("base_weight_sha256")
    if not isinstance(base_sha, str) or len(base_sha) != 64:
        raise ValueError("E01-A manifest has no verified Base weight SHA-256")

    inference_config_path = ROOT / "participant/configs/inference_config.json"
    inference_config = json.loads(inference_config_path.read_text(encoding="utf-8"))
    inference_config["experiment_id"] = "E01-A"
    inference_config["quantization"]["enabled"] = True
    inference_config["adapter"] = {
        "enabled": True,
        "path": str((root / "adapter").resolve()),
        "expected_base_weight_sha256": base_sha,
        "expected_adapter_weight_sha256": identity["weight_sha256"],
        "expected_adapter_config_sha256": identity["config_sha256"],
    }

    evaluation_config_path = ROOT / "configs/evaluation/e00.yaml"
    evaluation_config = yaml.safe_load(evaluation_config_path.read_text(encoding="utf-8"))
    evaluation_config["experiment_id"] = "E01-A"
    evaluation_config["model"]["adapter"] = {
        "enabled": True,
        "expected_base_weight_sha256": base_sha,
        "expected_adapter_weight_sha256": identity["weight_sha256"],
        "expected_adapter_config_sha256": identity["config_sha256"],
    }
    evaluation_root = _evaluation_attempt_root(root, attempt_id)
    evaluation_root.mkdir(parents=True)
    infer_snapshot = evaluation_root / "inference_config.json"
    _snapshot(infer_snapshot, inference_config)
    evaluation_snapshot = evaluation_root / "evaluation_config.yaml"
    _snapshot(evaluation_snapshot, evaluation_config, yaml_format=True)
    return infer_snapshot, evaluation_snapshot, evaluation_root


def run_dev_smoke(run_root: Path, *, attempt_id: str) -> dict[str, Any]:
    root, manifest, _recorded_identity = _load_run(run_root)
    identity, _adapter_cfg = run_inference.adapter_weight_identity(root / "adapter")
    infer_snapshot, evaluation_snapshot, evaluation_root = _prepare_snapshots(root, manifest, identity, attempt_id)
    inference_output = evaluation_root / "runner"
    raw_path = evaluation_root / "raw_predictions.jsonl"
    run_id = f"{root.name}-dev-smoke-{attempt_id}"
    environment = os.environ.copy()
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EVALUATION_INFERENCE_CONFIG": str(infer_snapshot),
        "EVALUATION_EVALUATION_CONFIG": str(evaluation_snapshot),
        "EVALUATION_RUN_ID": run_id,
        "EVALUATION_EXPERIMENT_ID": "E01-A",
        "EVALUATION_ENABLE_NF4": "1",
        "EVALUATION_RAW_OUTPUT_PATH": str(raw_path),
    })
    run_record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": "running",
        "inference_config_sha256": sha256_file(infer_snapshot),
        "evaluation_config_sha256": sha256_file(evaluation_snapshot),
        "smoke_manifest_sha256": sha256_file(SMOKE_ROOT / "manifest.json"),
        "training_fit_scope": "train_fitting_only",
        "dev_scope": "fixed_20_row_E00_1_smoke",
        "references_read_by_inference": False,
    }
    run_manifest_path = evaluation_root / "run_manifest.json"
    _write_json(run_manifest_path, run_record)
    try:
        runner_result = subprocess.run(
            [sys.executable, str(ROOT / "participant/run_inference.py"), str(SMOKE_ROOT / "smoke_input.jsonl"), str(inference_output)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        run_record.update(status="runner_launch_failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(run_manifest_path, run_record)
        raise
    run_record.update(
        status="inference_failed" if runner_result.returncode else "inference_completed",
        return_code=runner_result.returncode,
        runner_stdout_tail=runner_result.stdout[-4000:],
        runner_stderr_tail=runner_result.stderr[-4000:],
    )
    _write_json(run_manifest_path, run_record)
    if runner_result.returncode:
        return run_record
    eval_cfg = yaml.safe_load(evaluation_snapshot.read_text(encoding="utf-8"))
    try:
        metrics = score_files(
            config=eval_cfg,
            config_sha256=sha256_file(evaluation_snapshot),
            references_path=SMOKE_ROOT / "smoke_references.jsonl",
            submission_path=inference_output / "submission.jsonl",
            raw_path=raw_path,
            output_dir=evaluation_root / "scored",
            manifest_path=SMOKE_ROOT / "manifest.json",
            id_map_path=SMOKE_ROOT / "smoke_id_map.jsonl",
            run_metadata_path=inference_output / "performance_report.json",
        )
        metrics_sha256 = sha256_file(evaluation_root / "scored/metrics.json")
    except Exception as exc:
        run_record.update(status="scoring_failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(run_manifest_path, run_record)
        raise
    run_record.update({"status": "scored", "score_metrics_sha256": metrics_sha256, "score_status": metrics["model_identity_verification"]["status"]})
    _write_json(evaluation_root / "run_manifest.json", run_record)
    return run_record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--scope", choices=("train-and-dev-smoke", "dev-smoke"), default="train-and-dev-smoke")
    parser.add_argument("--attempt-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.run_root).expanduser().resolve()
    if args.scope == "train-and-dev-smoke":
        diagnostics = sorted((root / "evaluation").glob("train_diagnostics_step_*.json"))
        if not diagnostics:
            raise ValueError("train-and-dev-smoke requires completed train_fitting_only diagnostics")
    attempt_id = args.attempt_id or datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S%fZ")
    result = run_dev_smoke(root, attempt_id=attempt_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "scored" else 2


if __name__ == "__main__":
    raise SystemExit(main())
