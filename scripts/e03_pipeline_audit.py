#!/usr/bin/env python3
"""Summarize local training/evaluation artifacts without loading models or data rows."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]] | None, str]:
    if not path.is_file():
        return None, "missing"
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    return None, f"blank_line:{line_number}"
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None, f"non_object_line:{line_number}"
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid:{type(exc).__name__}"
    return rows, "present"


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _series_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50": statistics.median(values) if values else None,
        "p90": _quantile(values, 0.90),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "sum": sum(values) if values else None,
        "status": "measured" if values else "unknown",
    }


def _directory_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    failed = False
    for root, _directories, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                failed = True
    return None if failed else total


def _checkpoint_facts(run_root: Path) -> dict[str, Any]:
    trainer_root = run_root / "trainer"
    if not trainer_root.is_dir():
        return {"status": "unknown", "checkpoint_count": None, "bytes": None, "checkpoints": []}
    checkpoints: list[dict[str, Any]] = []
    for path in sorted(trainer_root.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        checkpoint_manifest = _read_json(path / "checkpoint_manifest.json") or {}
        checkpoints.append({
            "name": path.name,
            "bytes": _directory_bytes(path),
            "manifest_status": checkpoint_manifest.get("status", "unknown"),
        })
    byte_sizes = [item["bytes"] for item in checkpoints]
    return {
        "status": "measured",
        "checkpoint_count": len(checkpoints),
        "bytes": sum(byte_sizes) if all(value is not None for value in byte_sizes) else None,
        "checkpoints": checkpoints,
    }


def analyze_training_run(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    if not run_root.is_dir():
        return {
            "run_root": str(run_root),
            "status": "missing_run_root",
            "run_id": "unknown",
            "training_seconds": None,
            "optimizer_steps": None,
            "training_step_timing": {"status": "unknown"},
            "dev_evaluation": {"status": "unknown"},
            "checkpoints": {"status": "unknown"},
            "run_disk_bytes": None,
            "attempts": "unknown",
            "resume_history": "unknown",
        }

    manifest = _read_json(run_root / "manifest.json")
    if manifest is None:
        return {
            "run_root": str(run_root),
            "status": "manifest_missing_or_invalid",
            "run_id": "unknown",
            "training_seconds": None,
            "optimizer_steps": None,
            "training_step_timing": {"status": "unknown"},
            "dev_evaluation": {"status": "unknown"},
            "checkpoints": _checkpoint_facts(run_root),
            "run_disk_bytes": _directory_bytes(run_root),
            "attempts": "unknown",
            "resume_history": "unknown",
        }

    run_id = manifest.get("run_id", "unknown")
    identity_status = "verified_run_id_path" if run_id == run_root.name else "run_id_path_mismatch"
    if identity_status != "verified_run_id_path":
        return {
            "run_root": str(run_root),
            "run_id": run_id,
            "identity_status": identity_status,
            "manifest_status": manifest.get("status", "unknown"),
            "training_seconds": manifest.get("training_seconds"),
            "optimizer_steps_manifest": manifest.get("optimizer_steps"),
            "training_log_status": "not_read_identity_mismatch",
            "optimizer_steps_logged": None,
            "training_step_timing": {"status": "unknown", "reason": "run_id_path_mismatch"},
            "dev_evaluation": {"status": "unknown", "reason": "run_id_path_mismatch"},
            "checkpoints": {"status": "unknown", "checkpoint_count": None, "bytes": None, "checkpoints": []},
            "run_disk_bytes": _directory_bytes(run_root),
            "attempts": "unknown",
            "resume_history": "unknown",
        }
    train_rows, train_status = _read_jsonl(run_root / "train_log.jsonl")
    eval_rows, eval_status = _read_jsonl(run_root / "eval_log.jsonl")
    logged_steps: list[int] = []
    if train_rows is None:
        train_timing: dict[str, Any] = {
            "status": "unknown" if train_status == "missing" else "invalid",
            "source_status": train_status,
            "optimizer_step_seconds": None,
            "step_count": None,
            "unique_steps": None,
            "duplicate_step_count": None,
            "gap_count": None,
        }
    else:
        timing_values: list[float] = []
        for row in train_rows:
            step = row.get("optimizer_step")
            if isinstance(step, int) and not isinstance(step, bool):
                logged_steps.append(step)
            seconds = row.get("optimizer_step_seconds")
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0:
                timing_values.append(float(seconds))
        unique_steps = sorted(set(logged_steps))
        train_timing = {
            **_series_summary(timing_values),
            "source_status": train_status,
            "step_count": len(train_rows),
            "unique_steps": len(unique_steps),
            "duplicate_step_count": len(logged_steps) - len(unique_steps),
            "gap_count": sum(
                max(0, right - left - 1) for left, right in zip(unique_steps, unique_steps[1:])
            ),
            "timing_scope": "optimizer_step_seconds from existing train_log; includes the measured step callback interval, not isolated GPU kernel time",
        }

    eval_steps: list[int] = []
    eval_counts: list[int] = []
    if eval_rows is not None:
        for row in eval_rows:
            step = row.get("optimizer_step")
            samples = row.get("eval_samples")
            if isinstance(step, int) and not isinstance(step, bool):
                eval_steps.append(step)
            if isinstance(samples, int) and not isinstance(samples, bool):
                eval_counts.append(samples)
    progress = _read_json(run_root / "run_progress.json") or _read_json(run_root / "progress.json")
    checkpoint_facts = _checkpoint_facts(run_root)
    if isinstance(manifest.get("training_attempts"), list):
        raw_attempts = manifest["training_attempts"]
        attempts: Any = {
            "status": "measured",
            "count": len(raw_attempts),
            "training_seconds": [
                item.get("training_seconds")
                for item in raw_attempts
                if isinstance(item, dict)
            ],
            "step_ranges": [
                {
                    "starting_optimizer_step": item.get("starting_optimizer_step"),
                    "ending_optimizer_step": item.get("ending_optimizer_step"),
                    "stop_reason": item.get("stop_reason", "unknown"),
                }
                for item in raw_attempts
                if isinstance(item, dict)
            ],
            "loss_formula_audit_calls": [
                item.get("loss_formula_audit_calls")
                for item in raw_attempts
                if isinstance(item, dict) and "loss_formula_audit_calls" in item
            ],
        }
        resume_summary: Any = {
            "status": "multiple_attempts_recorded" if len(raw_attempts) > 1 else "single_attempt_recorded",
            "attempt_count": len(raw_attempts),
            "same_run_checkpoint_resume_explicit": manifest.get("resume_from_checkpoint", "unknown"),
        }
    else:
        attempt_rows, attempt_status = _read_jsonl(run_root / "training_attempts.jsonl")
        attempts = len(attempt_rows) if attempt_rows is not None else "unknown"
        if attempts == "unknown" and attempt_status != "missing":
            attempts = f"unknown:{attempt_status}"
        resume_summary = "unknown"
    resume_fields = {
        key: manifest[key]
        for key in ("resume_count", "resume_attempts", "resume_from_checkpoint", "resumed_from")
        if key in manifest
    }
    step_sum = train_timing.get("sum")
    total_training_seconds = manifest.get("training_seconds")
    residual = (
        float(total_training_seconds) - float(step_sum)
        if isinstance(total_training_seconds, (int, float))
        and not isinstance(total_training_seconds, bool)
        and isinstance(step_sum, (int, float))
        else None
    )
    return {
        "run_root": str(run_root),
        "run_id": run_id,
        "identity_status": identity_status,
        "manifest_status": manifest.get("status", "unknown"),
        "manifest_stage": manifest.get("stage", "unknown"),
        "experiment_id": manifest.get("experiment_id", "unknown"),
        "training_seconds": manifest.get("training_seconds"),
        "training_time_minus_logged_step_sum_seconds": residual,
        "training_time_residual_scope": "unattributed; do not interpret as a single phase",
        "optimizer_steps_manifest": manifest.get("optimizer_steps"),
        "optimizer_steps_logged": len(set(logged_steps)) if train_rows is not None else None,
        "training_log_status": train_status,
        "training_step_timing": train_timing,
        "dev_evaluation": {
            "status": "measured" if eval_rows is not None else "unknown",
            "log_status": eval_status,
            "evaluation_count": len(eval_rows) if eval_rows is not None else None,
            "optimizer_steps": sorted(eval_steps) if eval_rows is not None else None,
            "samples_per_evaluation": sorted(set(eval_counts)) if eval_rows is not None else None,
            "frequency_status": "measured_from_eval_log" if eval_rows is not None else "unknown",
        },
        "checkpoints": checkpoint_facts,
        "run_progress": progress if progress is not None else "unknown",
        "run_disk_bytes": _directory_bytes(run_root),
        "attempts": attempts,
        "resume_history": resume_summary if resume_summary != "unknown" else resume_fields or "unknown",
        "isolated_phase_timings": {
            key: "unknown"
            for key in (
                "preflight.hash", "preflight.data", "mapping", "model_load",
                "gradient_audit", "checkpoint_save_and_hash", "dev_loss",
                "adapter_reload", "quick_profile_generation", "full_generation",
            )
        },
    }


def analyze_runner_report(report_path: Path) -> dict[str, Any]:
    report_path = report_path.resolve()
    report = _read_json(report_path)
    if report is None:
        return {
            "report_path": str(report_path),
            "status": "unknown" if not report_path.exists() else "invalid",
            "run_id": "unknown",
            "latency_ms": {"status": "unknown"},
            "model_load_seconds": None,
            "cuda_peak_memory": "unknown",
        }
    values = [
        float(value) for value in report.get("latencies_ms", [])
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    ]
    latency = _series_summary(values)
    latency.update({"unit": "ms", "source": "performance_report.latencies_ms"})
    peak_memory = report.get("cuda_memory_after_model_load", {}).get("allocator_peak_allocated_bytes")
    return {
        "report_path": str(report_path),
        "status": "measured",
        "run_id": report.get("run_id", "unknown"),
        "samples": report.get("samples"),
        "success_count": report.get("success_count"),
        "failure_counts": report.get("failure_counts"),
        "model_load_seconds": report.get("model_load_seconds"),
        "latency_ms": latency,
        "cuda_peak_allocated_bytes_reported": peak_memory,
        "timing_scope": report.get("timing_scope", "unknown"),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    def display(value: Any) -> str:
        return "unknown" if value is None else str(value)

    lines = [
        "# E03 Pipeline Audit",
        "",
        f"Created: {summary['created_at_utc']}",
        "",
        "This is a read-only summary of existing local Run artifacts. Missing component timings remain unknown; this report does not attribute time to a GPU kernel or callback without an explicit measurement.",
        "",
        "## Training runs",
        "",
        "| Run | Status | Logged steps | Step P50 (s) | Step P90 (s) | Manifest time (s) | Unattributed delta (s) | Dev eval count | Checkpoint bytes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["training_runs"]:
        timing = run.get("training_step_timing", {})
        evaluation = run.get("dev_evaluation", {})
        checkpoints = run.get("checkpoints", {})
        lines.append(
            f"| {run.get('run_id', 'unknown')} | {run.get('manifest_status', run.get('status', 'unknown'))} "
            f"| {display(run.get('optimizer_steps_logged'))} "
            f"| {display(timing.get('p50'))} | {display(timing.get('p90'))} "
            f"| {display(run.get('training_seconds'))} "
            f"| {display(run.get('training_time_minus_logged_step_sum_seconds'))} "
            f"| {display(evaluation.get('evaluation_count'))} "
            f"| {display(checkpoints.get('bytes'))} |"
        )
    lines.extend(["", "Run status counts for the supplied roots:"])
    for status, count in summary["training_run_status_counts"].items():
        lines.append(f"- {status}: {count}")
    lines.append("")
    lines.append("Training attempt records:")
    for run in summary["training_runs"]:
        attempts = run.get("attempts", "unknown")
        if isinstance(attempts, dict):
            lines.append(f"- {run.get('run_id', 'unknown')}: {attempts.get('count', 'unknown')} recorded attempt(s)")
        else:
            lines.append(f"- {run.get('run_id', 'unknown')}: {attempts}")
    lines.extend(["", "## Runner reports", ""])
    if summary["runner_reports"]:
        lines.extend([
            "| Run | Samples | Latency P50 (ms) | Latency P90 (ms) | Model load (s) |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in summary["runner_reports"]:
            latency = item.get("latency_ms", {})
            lines.append(
                f"| {item.get('run_id', 'unknown')} | {display(item.get('samples'))} "
                f"| {display(latency.get('p50'))} | {display(latency.get('p90'))} "
                f"| {display(item.get('model_load_seconds'))} |"
            )
    else:
        lines.append("No Runner report was supplied.")
    lines.extend([
        "",
        "## Unmeasured phases",
        "",
        "Preflight hashing/data checks, mapping, isolated gradient audit, checkpoint hashing, Dev loss, Adapter reload, and quick/full generation are unknown unless present in a future explicitly instrumented Run. The existing optimizer_step_seconds value combines the step interval and cannot isolate those components.",
        "",
        f"Filesystem context: {summary['filesystem_context']}.",
        "",
    ])
    return "\n".join(lines)


def build_audit(run_roots: list[Path], runner_reports: list[Path]) -> dict[str, Any]:
    platform_name = platform.platform()
    path_values = [str(path.resolve()) for path in run_roots + runner_reports]
    if "microsoft-standard-WSL" in platform_name:
        filesystem = "WSL2"
    elif any(value.startswith("/mnt/c/") for value in path_values):
        filesystem = "WSL2 with one or more /mnt/c paths"
    else:
        filesystem = "Linux/native; storage mount type not independently verified"
    training_runs = [analyze_training_run(path) for path in run_roots]
    statuses: dict[str, int] = {}
    for run in training_runs:
        status = str(run.get("manifest_status", run.get("status", "unknown")))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "status": "read_only_audit_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform_name,
        "filesystem_context": filesystem,
        "training_runs": training_runs,
        "training_run_status_counts": dict(sorted(statuses.items())),
        "runner_reports": [analyze_runner_report(path) for path in runner_reports],
        "conclusions": {
            "isolated_phase_timing_claims": "none",
            "model_loading_performed": False,
            "training_performed": False,
        "raw_predictions_read": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", default=[], help="training Run root; repeat per Run")
    parser.add_argument("--runs-dir", action="append", default=[], help="scan immediate child Run folders; repeatable")
    parser.add_argument("--runner-report", action="append", default=[], help="Runner performance_report.json; repeatable")
    parser.add_argument("--output-dir", default="outputs/diagnostics/e03_engineering/pipeline_audit")
    args = parser.parse_args(argv)
    if not args.run_root and not args.runs_dir and not args.runner_report:
        parser.error("supply at least one --run-root or --runner-report")
    run_roots = [_resolve(value) for value in args.run_root]
    for parent in args.runs_dir:
        run_roots.extend(
            child for child in sorted(_resolve(parent).iterdir())
            if child.is_dir() and (child / "manifest.json").is_file()
        )
    run_roots = list(dict.fromkeys(path.resolve() for path in run_roots))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = build_audit(
        run_roots,
        [_resolve(value) for value in args.runner_report],
    )
    summary_path = output_dir / "timing_summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "training_runs": len(summary["training_runs"]),
        "runner_reports": len(summary["runner_reports"]),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
