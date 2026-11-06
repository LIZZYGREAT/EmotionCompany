#!/usr/bin/env python3
"""Record a user-authorized transition after the fixed Fast-only evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e03_a6_f4_staged_eval import EVAL_CONFIG, _load_eval_config, _render_node_report
from src.training.e03_a6_f4_staged_contract import (
    STAGE_STEPS,
    USER_AUTHORIZATION_SOURCE,
    write_stage_authorization,
)
from src.training.e03_a6_f4_staged_full import (
    DEFAULT_CONFIG,
    _load_config,
    _read_json,
    _resolve,
    _write_json,
    config_sha256,
    sha256_file,
)


def record_continuation_authorization(
    config_path: Path,
    eval_config_path: Path,
    run_id: str,
    from_step: int,
    to_step: int,
    *,
    allow_staged_training: bool,
) -> Path:
    if not allow_staged_training:
        raise ValueError("writing a staged authorization receipt requires --allow-staged-training")
    if from_step not in STAGE_STEPS[:-1]:
        raise ValueError("continuation authorization must follow node 512, 1024, or 2048")
    expected_target = {512: 1024, 1024: 2048, 2048: 4048}[from_step]
    if to_step != expected_target:
        raise ValueError("authorization target differs from the next frozen milestone")

    train_config, _ = _load_config(config_path)
    eval_config, _ = _load_eval_config(eval_config_path)
    run_root = _resolve(train_config["data"]["output_root"]) / run_id
    training_manifest_path = run_root / "manifest.json"
    training_manifest = _read_json(training_manifest_path)
    if (
        training_manifest.get("run_id") != run_id
        or training_manifest.get("config_sha256") != config_sha256(train_config)
    ):
        raise ValueError("authorization Run identity or frozen training config differs")
    if training_manifest.get("status") not in {"awaiting_manual_review", "interrupted"}:
        raise ValueError("authorization requires a paused same-Run checkpoint")
    if training_manifest.get("optimizer_steps") != from_step:
        raise ValueError("authorization node differs from the Run's current optimizer step")

    checkpoint = run_root / "trainer" / f"checkpoint-{from_step}"
    checkpoint_manifest = checkpoint / "checkpoint_manifest.json"
    if not checkpoint_manifest.is_file():
        raise FileNotFoundError("continuation checkpoint manifest is missing")
    checkpoint_sha = sha256_file(checkpoint_manifest)
    checkpoint_record = _read_json(checkpoint_manifest)
    if checkpoint_record.get("run_id") != run_id or checkpoint_record.get("optimizer_step") != from_step:
        raise ValueError("authorization checkpoint identity or optimizer step differs")
    node = training_manifest.get("nodes", {}).get(str(from_step), {})
    if (
        node.get("training_status") != "passed"
        or node.get("checkpoint_manifest_sha256") != checkpoint_sha
    ):
        raise ValueError("training Run Manifest does not bind a completed node checkpoint")

    eval_node = _resolve(eval_config["output_root"]) / run_id / f"step-{from_step:04d}"
    evaluation_path = eval_node / "evaluation.json"
    eval_manifest_path = eval_node / "run_manifest.json"
    predictions_path = eval_node / "predictions.jsonl"
    if not evaluation_path.is_file() or not eval_manifest_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("continuation requires completed Fast-only evaluation artifacts")
    evaluation = _read_json(evaluation_path)
    eval_manifest = _read_json(eval_manifest_path)
    evaluation_identity = evaluation.get("evaluation_identity")
    if (
        eval_manifest.get("run_id") != run_id
        or evaluation.get("run_id") != run_id
        or evaluation.get("checkpoint_step") != from_step
        or evaluation.get("engineering_gate") != "passed"
        or evaluation.get("progression_authorization_status") != "awaiting_manual_review"
        or not isinstance(evaluation_identity, dict)
        or evaluation_identity.get("checkpoint_manifest_sha256") != checkpoint_sha
    ):
        raise ValueError("Fast-only summary is not eligible for this Run/node/checkpoint authorization")
    prediction_sha = sha256_file(predictions_path)
    if (
        eval_manifest.get("status") != "complete"
        or eval_manifest.get("summary_sha256") != sha256_file(evaluation_path)
        or eval_manifest.get("evaluation_identity") != evaluation_identity
        or eval_manifest.get("predictions_sha256") != prediction_sha
        or evaluation.get("generation_protocol", {}).get("prediction_sha256") != prediction_sha
        or eval_manifest.get("engineering_gate") != "passed"
    ):
        raise ValueError("Fast-only artifacts are incomplete or have changed since scoring")

    if (
        node.get("evaluation_engineering_gate") != "passed"
        or node.get("evaluation_summary_sha256") != sha256_file(evaluation_path)
        or _resolve(node.get("evaluation_summary_path", "")) != evaluation_path.resolve()
    ):
        raise ValueError("training Run Manifest does not bind the completed Fast-only summary")

    receipt_path = run_root / "approvals" / f"step-{from_step:04d}.json"
    write_stage_authorization(
        receipt_path,
        run_id=run_id,
        from_step=from_step,
        to_step=to_step,
        checkpoint_manifest_sha256=checkpoint_sha,
    )
    receipt_sha = sha256_file(receipt_path)
    receipt_relative = str(receipt_path.relative_to(ROOT))

    authorization_facts = {
        "progression_authorization_status": "preauthorized_by_explicit_user_instruction",
        "authorization_source": USER_AUTHORIZATION_SOURCE,
        "authorization_file": receipt_relative,
        "authorization_sha256": receipt_sha,
        "node_metrics_individually_reviewed": False,
    }
    evaluation.update(authorization_facts)
    _write_json(evaluation_path, evaluation)
    (eval_node / "report.md").write_text(_render_node_report(evaluation), encoding="utf-8")
    eval_manifest.update(authorization_facts)
    eval_manifest["summary_sha256"] = sha256_file(evaluation_path)
    _write_json(eval_manifest_path, eval_manifest)

    training_node = training_manifest["nodes"][str(from_step)]
    training_node.update(
        progression_authorization_status="preauthorized_by_explicit_user_instruction",
        approval_file=receipt_relative,
        approval_sha256=receipt_sha,
        authorization_source=USER_AUTHORIZATION_SOURCE,
        authorization_text=_read_json(receipt_path)["authorization_text"],
        node_metrics_individually_reviewed=False,
        evaluation_summary_sha256=sha256_file(evaluation_path),
    )
    _write_json(training_manifest_path, training_manifest)
    return receipt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a node-specific F4 Full continuation receipt after Fast-only engineering gates pass."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--from-step", required=True, type=int, choices=STAGE_STEPS[:-1])
    parser.add_argument("--to-step", required=True, type=int, choices=STAGE_STEPS[1:])
    parser.add_argument("--allow-staged-training", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = record_continuation_authorization(
        args.config,
        args.eval_config,
        args.run_id,
        args.from_step,
        args.to_step,
        allow_staged_training=args.allow_staged_training,
    )
    print(json.dumps({"status": "preauthorized_by_explicit_user_instruction", "approval_file": str(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
