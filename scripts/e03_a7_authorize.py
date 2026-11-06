#!/usr/bin/env python3
"""Record a user-reviewed continuation permit for exactly one E03-A7 Arm/node."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e03_a7_train
from src.data.e03_a5_labels import sha256_file
from src.training.e03_a7_staged import (
    STAGE_STEPS,
    StageAuthorizationError,
    canonical_sha256,
    sha256_file as staged_sha256_file,
    validate_stage_transition,
    verify_checkpoint_manifest,
    write_continuation_receipt,
    write_manifest,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def record_continuation(
    config_path: Path,
    run_id: str,
    arm: str,
    from_step: int,
    to_step: int,
    *,
    user_approval_text: str,
) -> Path:
    validate_stage_transition(from_step, to_step)
    if arm not in {"P", "EP"}:
        raise ValueError("A7 approval Arm must be P or EP")
    config, config_sha = e03_a7_train._load_config(config_path.resolve())
    preflight, pair_root = e03_a7_train._load_preflight_for_training(config, config_sha)
    expected_run_id = f"{config['training']['pair_id']}-{arm}"
    if run_id != expected_run_id:
        raise ValueError("A7 approval Run ID does not bind the requested pair and Arm")
    run_root = e03_a7_train._resolve(config["outputs"]["training_root"]) / run_id
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = _json(run_manifest_path)
    node = run_manifest.get("nodes", {}).get(str(from_step), {})
    if (
        run_manifest.get("status") != "awaiting_manual_review"
        or run_manifest.get("manual_review_state") != f"awaiting_manual_review_step_{from_step:04d}"
        or run_manifest.get("optimizer_step") != from_step
        or run_manifest.get("config_sha256") != config_sha
        or run_manifest.get("code_sha256") != preflight["code_sha256"]
        or node.get("training_status") != "passed"
        or node.get("evaluation_status") != "passed"
    ):
        raise StageAuthorizationError("A7 continuation requires this Arm's reviewed training and evaluation node")

    pair_eval_root = e03_a7_train._resolve(config["outputs"]["evaluation_root"]) / "pairs" / config["training"]["pair_id"] / f"step-{from_step:04d}"
    pair_report_path = pair_eval_root / "paired_report.json"
    if not pair_report_path.is_file():
        raise FileNotFoundError("A7 continuation requires the pair/partial node review report")
    pair_report = _json(pair_report_path)
    if pair_report.get("checkpoint_step") != from_step or pair_report.get("pair_id") != config["training"]["pair_id"]:
        raise ValueError("A7 pair node report does not bind the requested review node")
    if pair_report.get("status") not in {"awaiting_manual_review", "awaiting_manual_review_missing_matched_arm"}:
        raise ValueError("A7 pair node report is not ready for user review")

    checkpoint_manifest_path = run_root / "trainer" / f"checkpoint-{from_step}" / "checkpoint_manifest.json"
    checkpoint = verify_checkpoint_manifest(
        checkpoint_manifest_path,
        expected_run_id=run_id,
        expected_arm=arm,
        expected_step=from_step,
        expected_identity=run_manifest["checkpoint_identity"],
        expected_schedule_sha256=run_manifest["schedule_sha256"],
    )
    if (
        node.get("checkpoint_manifest_sha256") != sha256_file(checkpoint_manifest_path)
        or checkpoint.get("optimizer_step") != from_step
    ):
        raise ValueError("A7 training Run Manifest does not bind the reviewed checkpoint")

    eval_dir = e03_a7_train._resolve(config["outputs"]["evaluation_root"]) / run_id / f"step-{from_step:04d}"
    eval_report_path = eval_dir / "evaluation_report.json"
    eval_manifest_path = eval_dir / "evaluation_manifest.json"
    if not eval_report_path.is_file() or not eval_manifest_path.is_file():
        raise FileNotFoundError("A7 continuation requires this Arm's frozen 224-point scored evaluation")
    eval_report = _json(eval_report_path)
    eval_manifest = _json(eval_manifest_path)
    if (
        eval_report.get("status") != "scored"
        or eval_report.get("run_id") != run_id
        or eval_report.get("checkpoint_step") != from_step
        or eval_report.get("engineering_gate") != "passed"
        or eval_report.get("progression_authorization_status") != "awaiting_manual_review"
        or eval_manifest.get("status") != "scored"
        or eval_manifest.get("run_id") != run_id
        or eval_manifest.get("checkpoint_step") != from_step
        or eval_manifest.get("gold_read_after_predictions_locked") is not True
        or eval_manifest.get("predictions_sha256") != eval_report.get("locked_predictions_sha256")
        or node.get("evaluation_manifest_sha256") != sha256_file(eval_manifest_path)
        or node.get("evaluation_report_sha256") != sha256_file(eval_report_path)
    ):
        raise ValueError("A7 evaluation artifacts are incomplete or changed after scoring")
    if not user_approval_text.strip():
        raise ValueError("A7 continuation authorization must include the user's explicit instruction text")

    receipt_path = run_root / "reviews" / f"step-{from_step:04d}.json"
    if receipt_path.exists():
        raise FileExistsError("A7 continuation receipt already exists and cannot be replaced")
    receipt = write_continuation_receipt(
        receipt_path,
        run_id=run_id,
        arm=arm,
        from_step=from_step,
        to_step=to_step,
        checkpoint_manifest_sha256=sha256_file(checkpoint_manifest_path),
        evaluation_manifest_sha256=sha256_file(eval_manifest_path),
        config_sha256=config_sha,
        code_sha256=preflight["code_sha256"],
        user_approval_text=user_approval_text,
    )
    run_manifest["nodes"][str(from_step)].update(
        progression_authorization_status="approved",
        approval_receipt_path=str(receipt_path),
        approval_receipt_sha256=sha256_file(receipt_path),
        approval_text=receipt["authorization_text"],
        approved_to_step=to_step,
        pair_review_report_sha256=sha256_file(pair_report_path),
    )
    run_manifest["last_updated_at_utc"] = e03_a7_train._now()
    write_manifest(run_manifest_path, run_manifest, overwrite=True)
    print(json.dumps({"status": "continuation_authorized", "run_id": run_id, "arm": arm, "from_step": from_step, "to_step": to_step, "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)}, ensure_ascii=False))
    return receipt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=e03_a7_train.DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True, choices=("P", "EP"))
    parser.add_argument("--from-step", required=True, type=int, choices=STAGE_STEPS[:-1])
    parser.add_argument("--to-step", required=True, type=int, choices=STAGE_STEPS[1:])
    parser.add_argument("--user-approval", required=True, help="Exact natural-language continuation instruction from the user")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record_continuation(
        args.config,
        args.run_id,
        args.arm,
        args.from_step,
        args.to_step,
        user_approval_text=args.user_approval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
