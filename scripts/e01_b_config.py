#!/usr/bin/env python3
"""Review real Pilot evidence and explicitly freeze a separately approved Full B budget."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference
from src.training.training_sft import approved_config_sha256, sha256_file, _validate_staged_budget


def validate_pilot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_relative_to((ROOT / "outputs/training").resolve()):
        raise ValueError("Pilot Run must remain under outputs/training")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != root.name or manifest.get("experiment_id") != "E01-B-PILOT"
        or manifest.get("training_scope") != "pilot" or manifest.get("status") != "passed"
        or manifest.get("train_rows") != 512 or not manifest.get("resume_history")
        or manifest.get("init_adapter") is not None or manifest.get("base_weights_unchanged") is not True
        or manifest.get("reload_check_status") != "passed"
    ):
        raise ValueError("review/approval requires the completed 512-row Pilot with real checkpoint/resume and reload evidence")
    if sha256_file(root / "config_snapshot.yaml") != manifest.get("config_sha256"):
        raise ValueError("Pilot config snapshot changed")
    identity, _ = run_inference.adapter_weight_identity(root / "adapter")
    if any(identity.get(key) != (manifest.get("adapter_identity") or {}).get(key) for key in ("weight_sha256", "config_sha256")):
        raise ValueError("Pilot Adapter identity changed")
    return manifest


def review(root: Path) -> dict[str, Any]:
    manifest = validate_pilot(root)
    snapshot = yaml.safe_load((root / "config_snapshot.yaml").read_text(encoding="utf-8"))
    train_path = ROOT / snapshot["data"]["train_path"]
    if sha256_file(train_path) != manifest.get("train_sha256"):
        raise ValueError("effective Train changed since the Pilot")
    lengths = []
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lengths.append(len(row["input_ids"]))
    selected = json.loads((root / "selected_samples.json").read_text(encoding="utf-8"))["rows"]
    seconds = float(manifest["training_seconds"])
    tokens = int(manifest["consumed_input_tokens"])
    if seconds <= 0 or tokens <= 0:
        raise ValueError("Pilot has no positive actual time/token throughput evidence")
    log = [json.loads(line) for line in (root / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    memory_keys = sorted({key for row in log for key, value in row.get("cuda_memory", {}).items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    memory_observations = [row.get("cuda_memory", {}) for row in log] + [attempt.get("cuda_memory", {}) for attempt in manifest.get("training_attempts", [])]
    checkpoint_bytes = sum(path.stat().st_size for path in (root / "trainer").rglob("*") if path.is_file())
    return {
        "pilot_run_root": str(root.resolve()), "pilot_manifest_sha256": sha256_file(root / "manifest.json"),
        "optimizer_steps": manifest["optimizer_steps"], "train_rows": manifest["train_rows"],
        "training_seconds_including_dev_eval_and_checkpoint": seconds,
        "consumed_input_tokens": tokens, "input_tokens_per_second": tokens / seconds,
        "pilot_length_min_max": [min(row["sequence_length"] for row in selected), max(row["sequence_length"] for row in selected)],
        "full_train_length_min_max": [min(lengths), max(lengths)],
        "full_train_input_tokens": sum(lengths),
        "one_epoch_linear_token_time_estimate_hours": seconds * sum(lengths) / tokens / 3600,
        "estimate_limitations": "Token-linear estimate only; sequence-length cost is nonlinear and Full Dev Loss/checkpoint overhead differs. This estimate does not approve Full B.",
        "peak_cuda_memory_observations": {key: max(row.get(key, 0) for row in memory_observations) for key in memory_keys},
        "cuda_peak_memory_scope": manifest.get("model_facts", {}).get("cuda_peak_memory_scope"),
        "retained_checkpoint_bytes": checkpoint_bytes,
        "available_disk_bytes": shutil.disk_usage(root).free,
        "training_loss_first_10": manifest["training_loss_first_10"], "training_loss_last_10": manifest["training_loss_last_10"],
        "sft_dev_loss_last": manifest["sft_dev_loss_last"],
        "resume_history": manifest["resume_history"],
        "reload_check_status": manifest["reload_check_status"], "cuda": manifest.get("cuda"),
    }


def approve(root: Path, output: Path, *, approved_by: str, learning_rate: float, epochs: int, save_steps: int, eval_steps: int, max_training_hours: float) -> dict[str, Any]:
    evidence = review(root)
    if not approved_by.strip() or epochs < 1 or not math.isfinite(max_training_hours) or max_training_hours <= 0:
        raise ValueError("approval requires an operator name, positive integer epochs and a positive resource budget")
    config = copy.deepcopy(yaml.safe_load((ROOT / "configs/training/e01_b.yaml").read_text(encoding="utf-8")))
    config["status"] = "approved_for_full_training"
    config["optimizer"].update(learning_rate=learning_rate, weight_decay=0.0)
    training = config["training"]
    batches = math.ceil(int(config["data"]["expected_train_rows"]) / int(training["per_device_micro_batch_size"]))
    training.update(
        epochs=epochs,
        max_optimizer_steps=math.ceil(batches / int(training["gradient_accumulation_steps"])) * epochs,
        save_interval_steps=save_steps, evaluation_interval_steps=eval_steps,
        max_training_seconds=max_training_hours * 3600,
        execution_gate="approved_config_plus_explicit_allow_full_training_flag",
    )
    config["approval"] = {
        "scope": "E01-B-FULL", "approved_by": approved_by.strip(),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_run_root": str(root.resolve()), "pilot_manifest_sha256": evidence["pilot_manifest_sha256"],
        "config_sha256": approved_config_sha256(config),
    }
    _validate_staged_budget(config)
    output = output.resolve()
    if not output.is_relative_to((ROOT / "outputs/training/configs").resolve()) or output.suffix != ".yaml":
        raise ValueError("approved snapshots must use a new .yaml file under outputs/training/configs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return {"status": "approved_config_written", "config_path": str(output), "config_file_sha256": sha256_file(output), "approval": config["approval"], "optimizer_steps": training["max_optimizer_steps"], "training_started": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("review", "approve"))
    parser.add_argument("--pilot-run-root", required=True, type=Path)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--approved-by")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--max-training-hours", type=float)
    parser.add_argument("--approve-full-training", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "review":
        result = review(args.pilot_run_root)
    else:
        required = (args.output_config, args.approved_by, args.learning_rate, args.epochs, args.save_steps, args.eval_steps, args.max_training_hours)
        if not args.approve_full_training or any(value is None for value in required):
            parser.error("approve requires --approve-full-training, --approved-by and every explicit budget/output option")
        result = approve(args.pilot_run_root, args.output_config, approved_by=args.approved_by, learning_rate=args.learning_rate, epochs=args.epochs, save_steps=args.save_steps, eval_steps=args.eval_steps, max_training_hours=args.max_training_hours)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
