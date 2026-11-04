#!/usr/bin/env python3
"""Small read-only orchestration CLI for E03 status, audit, and CPU gates."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e03_pipeline_audit import _render_markdown, analyze_training_run, build_audit
from src.training.experiment_spec import validate_experiment_spec


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def status(run_roots: list[Path]) -> dict[str, Any]:
    return {
        "status": "read_only_status",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [analyze_training_run(path) for path in run_roots],
    }


def preflight(spec_path: Path, observed_adapter_path: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment spec must be a YAML object")
    observed_adapter = _read_json(observed_adapter_path) if observed_adapter_path else None
    schema_order = None
    if raw.get("objective") == "sft_plus_bce":
        from src.evaluation import load_submission_schema, schema_enums

        enums = schema_enums(load_submission_schema(ROOT))
        schema_order = [
            f"{dimension}:{label}"
            for dimension in ("personality_traits", "interests", "style")
            for label in enums[dimension]
        ]
    result = validate_experiment_spec(
        raw,
        observed_adapter=observed_adapter,
        expected_label_order=schema_order,
    )
    return {
        **result,
        "spec_path": str(spec_path),
        "preflight_only": True,
        "training_started": False,
    }


def validate_cpu_gates() -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "participant", "tests"],
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_e03_profile_prefix.py",
         "tests/test_e03_profile_fast_eval.py",
         "tests/test_e03_pipeline_audit.py",
         "tests/test_e03_experiment_spec.py",
         "tests/test_e03_experiment_ctl.py"],
        ["git", "diff", "--check"],
        ["git", "diff", "--cached", "--check"],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        results.append({
            "command": command,
            "return_code": completed.returncode,
            "status": "passed" if completed.returncode == 0 else "failed",
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        })
        if completed.returncode:
            break
    return {
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
        "scope": "compileall + focused E03 CPU tests + git diff --check",
        "gpu_validation": "not_run",
        "full_repository_suite": "not_run",
        "commands": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="summarize immutable Run Manifests")
    status_parser.add_argument("--run-root", action="append", default=[])
    status_parser.add_argument("--runs-dir", action="append", default=[])

    preflight_parser = subparsers.add_parser("preflight", help="validate a future spec; never starts training")
    preflight_parser.add_argument("--spec", required=True)
    preflight_parser.add_argument("--observed-adapter")

    audit_parser = subparsers.add_parser("audit", help="read-only resource audit of existing Runs")
    audit_parser.add_argument("--run-root", action="append", default=[])
    audit_parser.add_argument("--runs-dir", action="append", default=[])
    audit_parser.add_argument("--runner-report", action="append", default=[])
    audit_parser.add_argument("--output-dir", required=True)

    subparsers.add_parser("validate", help="run focused E03 CPU gates only")
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            roots = [_resolve(value) for value in args.run_root]
            for parent in args.runs_dir:
                roots.extend(
                    child for child in sorted(_resolve(parent).iterdir())
                    if child.is_dir() and (child / "manifest.json").is_file()
                )
            roots = list(dict.fromkeys(path.resolve() for path in roots))
            if not roots:
                parser.error("status requires at least one Run root or runs directory")
            result = status(roots)
        elif args.command == "preflight":
            result = preflight(
                _resolve(args.spec),
                _resolve(args.observed_adapter) if args.observed_adapter else None,
            )
        elif args.command == "audit":
            if not args.run_root and not args.runs_dir and not args.runner_report:
                parser.error("audit requires at least one Run root or Runner report")
            roots = [_resolve(value) for value in args.run_root]
            for parent in args.runs_dir:
                roots.extend(
                    child for child in sorted(_resolve(parent).iterdir())
                    if child.is_dir() and (child / "manifest.json").is_file()
                )
            roots = list(dict.fromkeys(path.resolve() for path in roots))
            result = build_audit(
                roots,
                [_resolve(value) for value in args.runner_report],
            )
            output_dir = _resolve(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
            (output_dir / "timing_summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (output_dir / "report.md").write_text(_render_markdown(result), encoding="utf-8")
        else:
            result = validate_cpu_gates()
    except (OSError, ValueError, KeyError) as exc:
        result = {
            "status": "preflight_failed" if args.command == "preflight" else "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "training_started": False,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"read_only_status", "validation_passed", "read_only_audit_completed", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
