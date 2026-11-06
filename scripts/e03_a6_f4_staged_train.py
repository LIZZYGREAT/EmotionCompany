#!/usr/bin/env python3
"""Run the gated E03-A6 F4 staged-full preflight, dry-step, training, and resume flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.e03_a6_f4_staged_contract import (
    STAGE_STEPS,
    USER_AUTHORIZATION_SOURCE,
    USER_AUTHORIZATION_TEXT,
    write_stage_authorization,
)
from src.training.e03_a6_f4_staged_full import (
    DEFAULT_CONFIG,
    _load_config,
    _read_json,
    _resolve,
    _write_json,
    config_sha256,
    execute_dry_step,
    execute_resume,
    execute_start,
    prepare_preflight,
    status_run,
)
from src.training.e03_a6_f4_staged_full import sha256_file


def _record_continuation_authorization(
    config_path: Path,
    run_id: str,
    from_step: int,
    stop_at_step: int,
    *,
    allow_staged_training: bool,
) -> Path:
    if not allow_staged_training:
        raise ValueError("writing a staged authorization receipt requires --allow-staged-training")
    if from_step not in STAGE_STEPS[:-1]:
        raise ValueError("continuation authorization must follow node 512, 1024, or 2048")
    config, _ = _load_config(config_path)
    expected_target = {512: 1024, 1024: 2048, 2048: 4048}[from_step]
    if stop_at_step != expected_target:
        raise ValueError("authorization target differs from the next frozen milestone")
    run_root = _resolve(config["data"]["output_root"]) / run_id
    manifest = _read_json(run_root / "manifest.json")
    if manifest.get("config_sha256") != config_sha256(config):
        raise ValueError("authorization Run config differs from the frozen staged-full config")
    node_dir = run_root / f"step-{from_step:04d}"
    evaluation_path = node_dir / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError("continuation requires a completed node Fast-only evaluation")
    evaluation = _read_json(evaluation_path)
    if evaluation.get("engineering_gate") != "passed":
        raise ValueError("continuation is blocked because the hard engineering evaluation gate did not pass")
    checkpoint_path = run_root / "trainer" / f"checkpoint-{from_step}"
    checkpoint_manifest = checkpoint_path / "checkpoint_manifest.json"
    if not checkpoint_manifest.is_file():
        raise FileNotFoundError("continuation checkpoint manifest is missing")
    receipt_path = run_root / "approvals" / f"step-{from_step:04d}.json"
    record = write_stage_authorization(
        receipt_path,
        run_id=run_id,
        from_step=from_step,
        to_step=stop_at_step,
        checkpoint_manifest_sha256=sha256_file(checkpoint_manifest),
    )
    manifest.setdefault("nodes", {}).setdefault(str(from_step), {})[
        "progression_authorization_status"
    ] = "preauthorized_by_explicit_user_instruction"
    manifest["nodes"][str(from_step)]["approval_file"] = str(receipt_path)
    manifest["nodes"][str(from_step)]["approval_sha256"] = sha256_file(receipt_path)
    manifest["nodes"][str(from_step)]["node_metrics_individually_reviewed"] = False
    manifest["nodes"][str(from_step)]["authorization_source"] = USER_AUTHORIZATION_SOURCE
    manifest["nodes"][str(from_step)]["authorization_text"] = USER_AUTHORIZATION_TEXT
    _write_json(run_root / "manifest.json", manifest)
    evaluation["progression_authorization_status"] = "preauthorized_by_explicit_user_instruction"
    evaluation["authorization_source"] = USER_AUTHORIZATION_SOURCE
    evaluation["authorization_text"] = USER_AUTHORIZATION_TEXT
    evaluation["node_metrics_individually_reviewed"] = False
    _write_json(evaluation_path, evaluation)
    from scripts.e03_a6_f4_staged_eval import EVAL_CONFIG, _load_eval_config, _render_node_report

    eval_config, _eval_path = _load_eval_config(EVAL_CONFIG)
    (node_dir / "report.md").write_text(_render_node_report(evaluation), encoding="utf-8")
    eval_manifest_path = _resolve(eval_config["output_root"]) / run_id / f"step-{from_step:04d}" / "run_manifest.json"
    eval_manifest = _read_json(eval_manifest_path)
    eval_manifest.update(
        progression_authorization_status="preauthorized_by_explicit_user_instruction",
        authorization_file=str(receipt_path),
        authorization_sha256=sha256_file(receipt_path),
        node_metrics_individually_reviewed=False,
        summary_sha256=sha256_file(evaluation_path),
    )
    _write_json(eval_manifest_path, eval_manifest)
    manifest["nodes"][str(from_step)]["evaluation_summary_sha256"] = sha256_file(evaluation_path)
    _write_json(run_root / "manifest.json", manifest)
    return receipt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gated F4 Full training: CPU preflight re-encodes and audits all 16,192 fixed Train rows; "
            "GPU dry-run performs exactly one isolated optimizer update; formal training uses one fixed "
            "4,048-step schedule with retained resumable checkpoints at 512/1024/2048/4048. "
            "All model/data outputs remain under outputs/ or ignored data/processed; official Val is not read."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="frozen F4 staged-full training YAML")
    parser.add_argument("--mode", required=True, choices=("preflight", "dry-run", "start", "resume", "authorize", "status"))
    parser.add_argument("--run-id", help="unique formal F4 staged-full Run ID")
    parser.add_argument("--preflight-run-root", type=Path, help="passed CPU-preflight artifact directory")
    parser.add_argument("--dry-run-root", type=Path, help="passed one-step GPU dry-run artifact directory")
    parser.add_argument("--stop-at-step", type=int, choices=STAGE_STEPS, help="fixed stage boundary; never changes max_steps=4048")
    parser.add_argument("--from-step", type=int, help="exact latest checkpoint step to restore from")
    parser.add_argument("--approved-from-step", type=int, choices=STAGE_STEPS[:-1], help="node-specific authorization receipt source")
    parser.add_argument("--approval-file", type=Path, help="must bind Run, previous node, next node, and checkpoint SHA")
    parser.add_argument("--allow-staged-training", action="store_true", help="explicit GPU gate required for dry-run, start, resume, and node authorization")
    parser.add_argument("--help-resource-budget", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        if not args.run_id:
            parser.error("preflight requires --run-id")
        path = prepare_preflight(args.config, args.run_id)
        print(json.dumps({"status": "passed", "preflight_run_root": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "dry-run":
        if not args.run_id or not args.preflight_run_root:
            parser.error("dry-run requires --run-id and --preflight-run-root")
        path = execute_dry_step(args.config, args.run_id, args.preflight_run_root, allow_staged_training=args.allow_staged_training)
        print(json.dumps({"status": "passed", "dry_run_root": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "start":
        if not args.run_id or not args.preflight_run_root or not args.dry_run_root or args.stop_at_step is None:
            parser.error("start requires --run-id, --preflight-run-root, --dry-run-root, and --stop-at-step")
        path = execute_start(
            args.config, args.run_id, args.preflight_run_root, args.dry_run_root,
            stop_at_step=args.stop_at_step, allow_staged_training=args.allow_staged_training,
        )
        print(json.dumps({"status": _read_json(path / "manifest.json")["status"], "run_root": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "resume":
        if not args.run_id or args.from_step is None or args.approved_from_step is None or args.stop_at_step is None or not args.approval_file:
            parser.error("resume requires --run-id, --from-step, --approved-from-step, --stop-at-step, and --approval-file")
        path = execute_resume(
            args.config, args.run_id,
            from_step=args.from_step,
            approved_from_step=args.approved_from_step,
            stop_at_step=args.stop_at_step,
            approval_file=args.approval_file,
            allow_staged_training=args.allow_staged_training,
        )
        print(json.dumps({"status": _read_json(path / "manifest.json")["status"], "run_root": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "authorize":
        if not args.run_id or args.approved_from_step is None or args.stop_at_step is None:
            parser.error("authorize requires --run-id, --approved-from-step, and --stop-at-step")
        path = _record_continuation_authorization(
            args.config, args.run_id, args.approved_from_step, args.stop_at_step,
            allow_staged_training=args.allow_staged_training,
        )
        print(json.dumps({"status": "preauthorized_by_explicit_user_instruction", "approval_file": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "status":
        if not args.run_id:
            parser.error("status requires --run-id")
        print(json.dumps(status_run(args.config, args.run_id), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
