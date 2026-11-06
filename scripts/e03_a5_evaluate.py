#!/usr/bin/env python3
"""Prepare separately authorized A5 Fast-only Runs and compare them on CPU."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e03_a5_probe as probe_runner
from scripts import e03_a5_train as train_runner
from src.data.e03_a5_labels import canonical_sha256, iter_jsonl, schema_label_order, sha256_file
from src.evaluation import schema_enums
from src.evaluation.metrics import emotion_metrics, profile_metrics
from src.evaluation.e03_a4_stability_metrics import paired_stability_summary
from src.evaluation.profile_fast_eval import selected_id_sha256
from src.evaluation.profile_reconcile import empty_profile, index_profile_prediction_rows
from src.evaluation.source_integrity import unique_index
from src.training.e03_a5_contract import RunAuthorizationError, verify_run_ticket


QUALITY_GATES = {
    "candidate_control_micro_f1_min_delta": 0.015,
    "candidate_e01_b_micro_f1_ci_lower_min_delta": -0.01,
    "candidate_e01_b_precision_min_delta": -0.02,
    "candidate_e01_b_fp_max_relative_increase": 0.10,
    "candidate_e01_b_emotion_accuracy_min_delta": -0.02,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ValueError(f"{name} must be a simple filesystem-safe Run ID")
    return value


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_a5(config_path: Path) -> tuple[dict[str, Any], str]:
    return probe_runner._load_config(probe_runner._resolve(config_path))


def _eval_root(config: Mapping[str, Any]) -> Path:
    root = probe_runner._resolve(config["outputs"]["evaluation_root"]).resolve()
    if not root.is_relative_to(probe_runner._resolve("outputs").resolve()):
        raise ValueError("A5 evaluation artifacts must remain under ignored outputs/")
    return root


def _pair_eval_root(config: Mapping[str, Any], pair_id: str) -> Path:
    train_runner._validate_id(pair_id, "pair_id")
    return _eval_root(config) / "pairs" / pair_id


def _plan_dir(config: Mapping[str, Any], pair_id: str, training_run_id: str) -> Path:
    _validate_id(training_run_id, "training_run_id")
    return _pair_eval_root(config, pair_id) / "plans" / training_run_id


def _audit(config: Mapping[str, Any], config_sha: str, audit_run_id: str) -> dict[str, Any]:
    return probe_runner._verify_audit_run(config, config_sha, audit_run_id)


def _fast_evaluator_source_hashes() -> dict[str, str]:
    return {
        "fast_eval_script": sha256_file(ROOT / "scripts/e03_profile_fast_eval.py"),
        "fast_eval_helpers": sha256_file(ROOT / "src/evaluation/profile_fast_eval.py"),
        "profile_prefix_parser": sha256_file(ROOT / "src/evaluation/profile_prefix.py"),
        "source_integrity": sha256_file(ROOT / "src/evaluation/source_integrity.py"),
        "inference_runner": sha256_file(ROOT / "participant/run_inference.py"),
    }


def _load_a4_config() -> tuple[dict[str, Any], str]:
    path = ROOT / "configs/training/e03_a4_lr_stability.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("experiment_id") != "E03-A4-LR2E-05":
        raise ValueError("pinned A4 cohort and E01-B baseline config is invalid")
    return value, sha256_file(path)


def _load_eval_assets(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the already frozen 224-point development proxy and E01-B baseline."""
    a4, a4_config_sha = _load_a4_config()
    identity, evaluation = a4["identity"], a4["evaluation"]
    if (
        identity["expected_frozen_eval_ids_sha256"] != config["identity"]["frozen_dev_224_ids_sha256"]
        or int(evaluation["frozen_eval_points"]) != int(config["candidate_evaluation"]["frozen_proxy_points"])
        or int(evaluation["frozen_eval_conversations"]) != int(config["candidate_evaluation"]["frozen_proxy_conversations"])
    ):
        raise ValueError("A4 frozen Dev proxy cohort differs from the A5 224-point contract")

    diagnosis_root = probe_runner._resolve(evaluation["frozen_diagnosis_run_root"])
    source_config_path = ROOT / "configs/evaluation/e03_profile_candidate.yaml"
    source_fast_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    if not isinstance(source_fast_config, dict) or source_fast_config.get("version") != "E03-profile-fast-v1":
        raise ValueError("frozen Fast-only source config is invalid")
    source_paths = {
        "frozen_input": diagnosis_root / "frozen_eval_input.jsonl",
        "references": diagnosis_root / "frozen_eval_references.jsonl",
        "id_map": diagnosis_root / "frozen_eval_id_map.jsonl",
        "generation_input": probe_runner._resolve(source_fast_config["sources"]["input_path"]),
        "generation_id_map": probe_runner._resolve(source_fast_config["sources"]["id_map_path"]),
        "normalized_dev": probe_runner._resolve(source_fast_config["sources"]["normalized_dev_path"]),
        "raw_source": probe_runner._resolve(source_fast_config["sources"]["raw_source_path"]),
        "dataset_manifest": probe_runner._resolve(source_fast_config["sources"]["dataset_manifest_path"]),
    }
    expected_hashes = {
        "frozen_input": evaluation["frozen_eval_input_sha256"],
        "references": evaluation["frozen_eval_references_sha256"],
        "id_map": evaluation["frozen_eval_id_map_sha256"],
    }
    source_hashes = {}
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen A5 evaluation asset {name}: {path}")
        observed = sha256_file(path)
        if name in expected_hashes and observed != expected_hashes[name]:
            raise ValueError(f"frozen A5 evaluation asset changed: {name}")
        source_hashes[name] = observed

    frozen_rows = [row for _n, row, _sha in iter_jsonl(source_paths["frozen_input"])]
    frozen_ids = [row.get("id") for row in frozen_rows]
    if (
        len(frozen_ids) != 224
        or any(not isinstance(value, str) or not value for value in frozen_ids)
        or len(frozen_ids) != len(set(frozen_ids))
        or selected_id_sha256(frozen_ids) != config["identity"]["frozen_dev_224_ids_sha256"]
    ):
        raise ValueError("frozen Dev input IDs differ from the locked 224-point A5 cohort")

    baseline = a4["fast_baseline"]
    baseline_paths = {
        "report": probe_runner._resolve(baseline["report_path"]),
        "predictions": probe_runner._resolve(baseline["predictions_path"]),
        "alignment": probe_runner._resolve(baseline["alignment_path"]),
    }
    pinned = {
        "report": baseline["report_sha256"],
        "predictions": baseline["predictions_sha256"],
        "alignment": baseline["alignment_sha256"],
    }
    for name, path in baseline_paths.items():
        if sha256_file(path) != pinned[name]:
            raise ValueError(f"E01-B Fast-only baseline {name} SHA differs from its A4 pin")
    baseline_report = _json(baseline_paths["report"])
    alignment = _json(baseline_paths["alignment"])
    if (
        baseline_report.get("status") != "passed"
        or baseline_report.get("sample_count") != 224
        or baseline_report.get("valid_count") != 224
        or baseline_report.get("profile_predictions", {}).get("ids_sha256") != config["identity"]["frozen_dev_224_ids_sha256"]
        or alignment.get("status") != "reconciled"
        or alignment.get("pairwise_comparisons", {}).get("R_raw_prefix_vs_F_fast_only", {}).get("different_sample_count") != 0
    ):
        raise ValueError("pinned E01-B Fast-only baseline failed its frozen 224-point contract")
    baseline_rows = [row for _n, row, _sha in iter_jsonl(baseline_paths["predictions"])]
    if len(baseline_rows) != 224 or selected_id_sha256([row.get("id") for row in baseline_rows]) != config["identity"]["frozen_dev_224_ids_sha256"]:
        raise ValueError("E01-B Fast-only predictions differ from the frozen 224-point IDs")

    output_schema = json.loads(probe_runner._resolve(config["data"]["official_output_schema"]).read_text(encoding="utf-8"))
    enums = schema_enums(output_schema)
    _baseline_index, baseline_statuses = index_profile_prediction_rows(
        baseline_rows, frozen_ids, enums, require_all_valid=True,
    )
    if baseline_statuses.get("valid") != 224:
        raise ValueError("E01-B baseline must be 224/224 valid before A5 pairing")

    full_eval_root = probe_runner._resolve(evaluation["e01_b_full_eval_run_root"])
    full_eval_manifest_path = full_eval_root / "run_manifest.json"
    inference_path = full_eval_root / "inference_config.json"
    if (
        sha256_file(full_eval_manifest_path) != evaluation["e01_b_full_eval_manifest_sha256"]
        or sha256_file(inference_path) != _json(full_eval_manifest_path).get("inference_config_sha256")
    ):
        raise ValueError("E01-B historical inference config/manifest identity changed")
    inference_config = _json(inference_path)
    tokenizer_config = inference_config.get("tokenizer", {})
    prompt_config = inference_config.get("prompt", {})
    context_config = inference_config.get("context", {})
    if (
        inference_config.get("model_id") != config["model_assets"]["base_model_id"]
        or inference_config.get("model_revision") != config["model_assets"]["base_model_revision"]
        or inference_config.get("expected_weight_sha256") != config["identity"]["base_weight_sha256"]
        or prompt_config.get("version") != "P0"
        or prompt_config.get("sha256") != config["identity"]["p0_sha256"]
        or tokenizer_config.get("expected_signature_sha256") != config["identity"]["tokenizer_signature_sha256"]
        or tokenizer_config.get("expected_template_sha256") != config["identity"]["chat_template_sha256"]
        or context_config.get("max_prompt_tokens") != int(config["identity"]["max_prompt_tokens"])
    ):
        raise ValueError("E01-B inference protocol/tokenizer config differs from the frozen A5 identity")
    return {
        "a4_config": a4,
        "a4_config_sha256": a4_config_sha,
        "fast_source_config": source_fast_config,
        "fast_source_config_sha256": sha256_file(source_config_path),
        "source_paths": source_paths,
        "source_hashes": source_hashes,
        "frozen_ids": frozen_ids,
        "frozen_input_path": source_paths["frozen_input"],
        "baseline": baseline,
        "baseline_paths": baseline_paths,
        "baseline_hashes": pinned,
        "baseline_status_counts": baseline_statuses,
        "e01b_full_eval_manifest_sha256": sha256_file(full_eval_manifest_path),
        "e01b_inference_path": inference_path,
        "e01b_inference_sha256": sha256_file(inference_path),
        "e01b_inference_config": inference_config,
        "output_enums": enums,
    }


def _verify_training_run(
    config: Mapping[str, Any], config_sha: str, audit: Mapping[str, Any],
    pair: Mapping[str, Any], training_run_id: str, expected_group: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _validate_id(training_run_id, "training_run_id")
    run_root = train_runner._run_root(config) / training_run_id
    manifest_path, identity_path = run_root / "run_manifest.json", run_root / "run_identity.json"
    manifest, identity = _json(manifest_path), _json(identity_path)
    if (
        manifest.get("status") != "trained_pending_eval"
        or manifest.get("run_id") != training_run_id
        or manifest.get("group") != expected_group
        or manifest.get("config_sha256") != config_sha
        or manifest.get("a5_audit_manifest_sha256") != audit["manifest_sha256"]
        or manifest.get("pair_id") != pair["pair_id"]
        or manifest.get("pair_manifest_sha256") != sha256_file(train_runner._pair_paths(config, pair["pair_id"])["manifest"])
        or manifest.get("train_schedule_sha256") != pair["train_schedule_sha256"]
        or manifest.get("base_weights_unchanged") is not True
        or manifest.get("initial_adapter_unchanged") is not True
        or manifest.get("adapter_reload_exact") is not True
        or manifest.get("optimizer_step_count") != int(config["training_proposal"]["max_optimizer_steps"])
        or manifest.get("optimizer", {}).get("draw_count") != int(config["training_proposal"]["sample_draw_count"])
        or identity.get("run_id") != training_run_id
        or identity.get("group") != expected_group
        or identity.get("config_sha256") != config_sha
        or identity.get("a5_audit_manifest_sha256") != audit["manifest_sha256"]
        or manifest.get("run_identity_sha256") != sha256_file(identity_path)
    ):
        raise ValueError(f"A5 {expected_group} training Run is incomplete or failed its reload/source gates")
    expected_lambda = 0.0 if expected_group == "control_sft" else float(config["training_proposal"]["candidate_lambda"])
    if (
        manifest.get("bce_lambda") != expected_lambda
        or manifest.get("candidate_lambda_status") != "candidate_lambda_frozen"
        or not isinstance(manifest.get("lambda_freeze_run_id"), str)
        or not isinstance(manifest.get("lambda_freeze_manifest_sha256"), str)
        or identity.get("lambda_freeze_manifest_sha256") != manifest.get("lambda_freeze_manifest_sha256")
    ):
        raise ValueError("A5 training Run is not bound to the reviewed frozen lambda")
    freeze_root = train_runner._run_root(config) / "lambda_freezes" / manifest["lambda_freeze_run_id"]
    freeze_path = freeze_root / "lambda_freeze.json"
    freeze_manifest_path = freeze_root / "lambda_freeze_manifest.json"
    freeze, freeze_manifest = _json(freeze_path), _json(freeze_manifest_path)
    if (
        sha256_file(freeze_manifest_path) != manifest["lambda_freeze_manifest_sha256"]
        or freeze_manifest.get("status") != "candidate_lambda_frozen"
        or freeze_manifest.get("outputs", {}).get("lambda_freeze.json") != sha256_file(freeze_path)
        or freeze.get("status") != "candidate_lambda_frozen"
        or freeze.get("approved") is not True
        or freeze.get("candidate_lambda") != float(config["training_proposal"]["candidate_lambda"])
        or freeze.get("config_sha256") != config_sha
        or freeze.get("audit_manifest_sha256") != audit["manifest_sha256"]
        or freeze.get("pair_id") != pair["pair_id"]
        or freeze.get("pair_manifest_sha256") != manifest["pair_manifest_sha256"]
        or freeze.get("probe_gate_manifest_sha256") != manifest.get("probe_gate_manifest_sha256")
        or freeze.get("both_dry_steps_passed_without_optimizer_updates") is not True
    ):
        raise ValueError("A5 training Run does not match its approved lambda freeze artifact")
    adapter_dir = run_root / "adapter"
    from participant import run_inference as inference
    adapter_identity, adapter_config = inference.adapter_weight_identity(adapter_dir)
    expected_identity = manifest.get("adapter_identity", {})
    if (
        adapter_identity.get("weight_sha256") != expected_identity.get("weight_sha256")
        or adapter_identity.get("config_sha256") != expected_identity.get("config_sha256")
        or adapter_config.get("r") != 8
        or adapter_config.get("lora_alpha") != 16
        or set(adapter_config.get("target_modules", [])) != {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        }
    ):
        raise ValueError("A5 trained Adapter differs from its passed Run Manifest")
    return run_root, manifest, identity, adapter_identity


def _prepare_fast(args: argparse.Namespace) -> int:
    config, config_sha = _load_a5(args.config)
    audit = _audit(config, config_sha, args.audit_run_id)
    pair, _schedule, _pair_root = train_runner._load_pair(config, config_sha, audit, args.pair_id)
    group_contract = "control_sft" if args.group == "control" else "candidate_sft_bce"
    train_root, train_manifest, train_identity, adapter_identity = _verify_training_run(
        config, config_sha, audit, pair, args.training_run_id, group_contract,
    )
    assets = _load_eval_assets(config)
    plan_dir = _plan_dir(config, args.pair_id, args.training_run_id)
    plan_dir.mkdir(parents=True, exist_ok=False)

    inference_config = copy.deepcopy(assets["e01b_inference_config"])
    inference_config["experiment_id"] = config["experiment_id"]
    inference_config["model_path"] = str(probe_runner._resolve(config["model_assets"]["base_model_path"]))
    inference_config["model_id"] = config["model_assets"]["base_model_id"]
    inference_config["model_revision"] = config["model_assets"]["base_model_revision"]
    inference_config["expected_weight_sha256"] = config["identity"]["base_weight_sha256"]
    inference_config.setdefault("context", {})["max_prompt_tokens"] = int(config["identity"]["max_prompt_tokens"])
    inference_config["adapter"] = {
        **inference_config.get("adapter", {}),
        "enabled": True,
        "path": str((train_root / "adapter").resolve()),
        "expected_base_weight_sha256": config["identity"]["base_weight_sha256"],
        "expected_adapter_weight_sha256": adapter_identity["weight_sha256"],
        "expected_adapter_config_sha256": adapter_identity["config_sha256"],
    }
    inference_config["generation"] = {
        **inference_config.get("generation", {}),
        "max_new_tokens": int(config["identity"]["max_new_tokens"]),
        "do_sample": False,
    }

    fast_config = copy.deepcopy(assets["fast_source_config"])
    fast_config["experiment_version"] = "E03-A5-fastonly-paired-v1"
    inference_path, fast_path = plan_dir / "inference_config.json", plan_dir / "fast_eval_config.yaml"
    fast_config["sources"]["inference_config_path"] = str(inference_path.resolve())
    frozen_ids_path = str(assets["frozen_input_path"])
    fast_config["selection"] = {
        "mode": "frozen_ids",
        "frozen_ids_path": frozen_ids_path,
        "expected_count": int(config["candidate_evaluation"]["frozen_proxy_points"]),
        "expected_conversation_count": int(config["candidate_evaluation"]["frozen_proxy_conversations"]),
        "expected_ids_sha256": config["identity"]["frozen_dev_224_ids_sha256"],
    }
    fast_config["protocol"] = {
        "prompt_version": "P0",
        "prompt_sha256": config["identity"]["p0_sha256"],
        "format_version": config["identity"]["protocol"]["format_version"],
        "truncation_version": config["identity"]["protocol"]["history_policy"].split("_")[0],
    }
    # Preserve the repository's exact T1 wire value; the human-readable history
    # policy string above is independently pinned by the A5 config SHA.
    fast_config["protocol"]["truncation_version"] = "T1"
    fast_config["generation"] = {
        "max_new_tokens": int(config["identity"]["max_new_tokens"]),
        "do_sample": False,
    }
    fast_config["gpu_budget"] = {"max_seconds": 900}
    fast_config["adoption_gates"] = {
        "minimum_valid_coverage": 1.0,
        "require_fixed_denominator_metrics": True,
    }
    fast_root = _eval_root(config) / "fast_only"
    fast_config["output_root"] = str(fast_root)
    _write_once(inference_path, inference_config)
    with fast_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(fast_config, allow_unicode=True, sort_keys=False))
        handle.flush()
        os.fsync(handle.fileno())

    plan = {
        "status": "prepared",
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "pair_id": args.pair_id,
        "group": args.group,
        "training_run_id": args.training_run_id,
        "training_manifest_sha256": sha256_file(train_root / "run_manifest.json"),
        "training_identity_sha256": sha256_file(train_root / "run_identity.json"),
        "adapter_identity": adapter_identity,
        "config_sha256": config_sha,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "pair_manifest_sha256": sha256_file(train_runner._pair_paths(config, args.pair_id)["manifest"]),
        "frozen_ids_sha256": config["identity"]["frozen_dev_224_ids_sha256"],
        "frozen_point_count": len(assets["frozen_ids"]),
        "frozen_conversation_count": int(config["candidate_evaluation"]["frozen_proxy_conversations"]),
        "frozen_input_path": str(assets["frozen_input_path"].resolve()),
        "frozen_references_path": str(assets["source_paths"]["references"].resolve()),
        "frozen_id_map_path": str(assets["source_paths"]["id_map"].resolve()),
        "frozen_input_sha256": assets["source_hashes"]["frozen_input"],
        "frozen_references_sha256": assets["source_hashes"]["references"],
        "frozen_id_map_sha256": assets["source_hashes"]["id_map"],
        "fast_run_id": args.fast_run_id,
        "fast_budget_seconds": 900,
        "generation_protocol": "P0_F2_T1_greedy_G1200",
        "inference_config_path": str(inference_path.resolve()),
        "inference_config_sha256": sha256_file(inference_path),
        "fast_config_path": str(fast_path.resolve()),
        "fast_config_sha256": sha256_file(fast_path),
        "fast_source_config_sha256": assets["fast_source_config_sha256"],
        "fast_evaluator_source_hashes": _fast_evaluator_source_hashes(),
        "a4_config_sha256": assets["a4_config_sha256"],
        "e01_b_full_eval_manifest_sha256": assets["e01b_full_eval_manifest_sha256"],
        "e01_b_inference_config_sha256": assets["e01b_inference_sha256"],
        "baseline_fast_report_sha256": assets["baseline_hashes"]["report"],
        "baseline_fast_predictions_sha256": assets["baseline_hashes"]["predictions"],
        "baseline_alignment_sha256": assets["baseline_hashes"]["alignment"],
        "evaluation_source_hashes": assets["source_hashes"],
        "response_text_generated_or_saved": False,
        "full_generation_performed": False,
        "generation_authorization": "requires_separate_candidate_generation_Run_ticket",
    }
    _write_once(plan_dir / "evaluation_plan.json", plan)
    print(json.dumps({
        "status": "prepared",
        "pair_id": args.pair_id,
        "group": args.group,
        "training_run_id": args.training_run_id,
        "fast_run_id": args.fast_run_id,
        "fast_config_sha256": plan["fast_config_sha256"],
        "inference_config_sha256": plan["inference_config_sha256"],
        "candidate_generation_authorized": False,
        "output_dir": str(plan_dir),
    }, ensure_ascii=False))
    return 0


def _load_plan(
    config: Mapping[str, Any], config_sha: str, audit: Mapping[str, Any],
    pair_id: str, training_run_id: str,
) -> tuple[dict[str, Any], Path]:
    plan_dir = _plan_dir(config, pair_id, training_run_id)
    plan_path = plan_dir / "evaluation_plan.json"
    plan = _json(plan_path)
    if (
        plan.get("status") != "prepared"
        or plan.get("pair_id") != pair_id
        or plan.get("training_run_id") != training_run_id
        or plan.get("config_sha256") != config_sha
        or plan.get("audit_manifest_sha256") != audit["manifest_sha256"]
        or plan.get("fast_evaluator_source_hashes") != _fast_evaluator_source_hashes()
        or sha256_file(plan["fast_config_path"]) != plan.get("fast_config_sha256")
        or sha256_file(plan["inference_config_path"]) != plan.get("inference_config_sha256")
    ):
        raise ValueError("A5 Fast-only evaluation plan or snapshots are missing or changed")
    return plan, plan_path


def _run_fast(args: argparse.Namespace) -> int:
    config, config_sha = _load_a5(args.config)
    audit = _audit(config, config_sha, args.audit_run_id)
    plan, plan_path = _load_plan(config, config_sha, audit, args.pair_id, args.training_run_id)
    if plan.get("group") != args.group or plan.get("fast_run_id") != args.fast_run_id:
        raise ValueError("A5 Fast-only request differs from its frozen per-Adapter plan")
    run_root, train_manifest, _train_identity, _adapter_id = _verify_training_run(
        config, config_sha, audit,
        train_runner._load_pair(config, config_sha, audit, args.pair_id)[0],
        args.training_run_id,
        "control_sft" if args.group == "control" else "candidate_sft_bce",
    )
    ticket_path = probe_runner._resolve(args.ticket)
    ticket = _json(ticket_path)
    verify_run_ticket(
        ticket,
        stage="candidate_generation",
        run_id=args.fast_run_id,
        config_sha256=config_sha,
        audit_manifest_sha256=audit["manifest_sha256"],
        max_sample_count=224,
    )
    if float(ticket["budget"]["max_seconds"]) > 900:
        raise RunAuthorizationError("A5 Fast-only Run ticket may not exceed 900 seconds")
    expected_bindings = {
        "pair_id": args.pair_id,
        "group": args.group,
        "training_run_id": args.training_run_id,
        "training_manifest_sha256": plan["training_manifest_sha256"],
        "evaluation_plan_sha256": sha256_file(plan_path),
        "fast_config_sha256": plan["fast_config_sha256"],
        "inference_config_sha256": plan["inference_config_sha256"],
        "frozen_ids_sha256": plan["frozen_ids_sha256"],
        "baseline_fast_predictions_sha256": plan["baseline_fast_predictions_sha256"],
        "generation_protocol": plan["generation_protocol"],
        "fast_evaluator_source_hashes": plan["fast_evaluator_source_hashes"],
    }
    mismatches = [key for key, value in expected_bindings.items() if ticket.get(key) != value]
    if mismatches:
        raise RunAuthorizationError(f"A5 candidate-generation ticket does not bind its frozen plan: {mismatches}")

    fast_root = _eval_root(config) / "fast_only"
    output_root = (fast_root / args.fast_run_id).resolve()
    if not output_root.is_relative_to(fast_root.resolve()):
        raise ValueError("A5 Fast-only output escaped the configured outputs root")
    if output_root.exists():
        raise FileExistsError(f"A5 Fast-only Run ID is immutable and already exists: {output_root}")
    command = [
        ".venv/bin/python", "scripts/e03_profile_fast_eval.py",
        "--mode", "gpu-fast-only",
        "--config", plan["fast_config_path"],
        "--run-config", plan["inference_config_path"],
        "--run-id", args.fast_run_id,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=float(ticket["budget"]["max_seconds"]) + 30,
        )
        report_path = output_root / "profile_fast_metrics.json"
        report = _json(report_path) if report_path.is_file() else None
        wrapper = {
            "status": "generation_wrapper_completed" if completed.returncode == 0 else "generation_wrapper_failed",
            "created_at_utc": _now(),
            "experiment_id": config["experiment_id"],
            "run_id": args.fast_run_id,
            "pair_id": args.pair_id,
            "group": args.group,
            "training_run_id": args.training_run_id,
            "ticket_sha256": sha256_file(ticket_path),
            "ticket_path": str(ticket_path.resolve()),
            "evaluation_plan_sha256": sha256_file(plan_path),
            "subprocess_exit_code": completed.returncode,
            "subprocess_stdout": completed.stdout.strip()[-4000:],
            "subprocess_stderr_tail": completed.stderr.strip()[-4000:],
            "fast_report_sha256": sha256_file(report_path) if report_path.is_file() else None,
            "fast_report_status": report.get("status") if isinstance(report, dict) else None,
            "elapsed_seconds": time.monotonic() - started,
            "response_text_generated_or_saved": False,
            "full_generation_performed": False,
        }
        _write_once(output_root / "a5_generation_wrapper.json", wrapper)
    except subprocess.TimeoutExpired as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_once(output_root / "a5_generation_wrapper.json", {
            "status": "generation_budget_exhausted",
            "created_at_utc": _now(),
            "run_id": args.fast_run_id,
            "pair_id": args.pair_id,
            "group": args.group,
            "ticket_sha256": sha256_file(ticket_path),
            "evaluation_plan_sha256": sha256_file(plan_path),
            "elapsed_seconds": time.monotonic() - started,
            "error": str(exc),
            "response_text_generated_or_saved": False,
        })
        print(json.dumps({"status": "generation_budget_exhausted", "run_id": args.fast_run_id}, ensure_ascii=False))
        return 1
    report_status = wrapper.get("fast_report_status")
    print(json.dumps({
        "status": wrapper["status"],
        "run_id": args.fast_run_id,
        "fast_report_status": report_status,
        "subprocess_exit_code": wrapper["subprocess_exit_code"],
        "output_dir": str(output_root),
    }, ensure_ascii=False))
    return 0 if completed.returncode == 0 and report is not None else 1


def _freeze_quality_gates(args: argparse.Namespace) -> int:
    if not args.approved_by.strip() or not args.rationale.strip():
        raise ValueError("freezing A5 quality thresholds requires a reviewer and rationale")
    config, config_sha = _load_a5(args.config)
    audit = _audit(config, config_sha, args.audit_run_id)
    pair, _schedule, _pair_root = train_runner._load_pair(config, config_sha, audit, args.pair_id)
    control_plan, control_path = _load_plan(config, config_sha, audit, args.pair_id, args.control_training_run_id)
    candidate_plan, candidate_path = _load_plan(config, config_sha, audit, args.pair_id, args.candidate_training_run_id)
    if (
        control_plan.get("group") != "control"
        or candidate_plan.get("group") != "candidate_sft_bce"
        or control_plan.get("pair_id") != args.pair_id
        or candidate_plan.get("pair_id") != args.pair_id
    ):
        raise ValueError("quality-gate freeze requires matching Control and Candidate Fast-only plans")
    root = _pair_eval_root(config, args.pair_id) / "quality_gates" / _validate_id(args.run_id, "quality_gate_run_id")
    root.mkdir(parents=True, exist_ok=False)
    task_plan_path = ROOT / "docs/task/E0.3-A5_画像BCE辅助监督.md"
    gate = {
        "status": "a5_quality_gates_frozen",
        "experiment_id": config["experiment_id"],
        "pair_id": args.pair_id,
        "run_id": args.run_id,
        "created_at_utc": _now(),
        "approved": True,
        "approved_by": args.approved_by.strip(),
        "rationale": args.rationale.strip(),
        "config_sha256": config_sha,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "pair_manifest_sha256": sha256_file(train_runner._pair_paths(config, args.pair_id)["manifest"]),
        "task_plan_sha256": sha256_file(task_plan_path),
        "control_evaluation_plan_sha256": sha256_file(control_path),
        "candidate_evaluation_plan_sha256": sha256_file(candidate_path),
        "frozen_ids_sha256": config["identity"]["frozen_dev_224_ids_sha256"],
        "decision_rule": "engineering_screen_not_statistical_noninferiority",
        "thresholds": dict(QUALITY_GATES),
        "development_proxy_only": True,
        "final_retention_requires_separate_full_json_and_reply_evaluation": True,
    }
    gate_path = root / "quality_gate.json"
    _write_once(gate_path, gate)
    manifest = {
        "status": "a5_quality_gates_frozen",
        "run_id": args.run_id,
        "pair_id": args.pair_id,
        "config_sha256": config_sha,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "outputs": {"quality_gate.json": sha256_file(gate_path)},
        "created_at_utc": gate["created_at_utc"],
    }
    _write_once(root / "quality_gate_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "pair_id": args.pair_id,
        "approved_by": gate["approved_by"], "thresholds": gate["thresholds"],
        "output_dir": str(root),
    }, ensure_ascii=False))
    return 0


def _load_fast_run(
    config: Mapping[str, Any], plan: Mapping[str, Any], training_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int], Path]:
    root = _eval_root(config) / "fast_only" / plan["fast_run_id"]
    report_path, predictions_path = root / "profile_fast_metrics.json", root / "profile_predictions.jsonl"
    wrapper_path = root / "a5_generation_wrapper.json"
    if not report_path.is_file() or not predictions_path.is_file() or not wrapper_path.is_file():
        raise FileNotFoundError(f"A5 Fast-only Run is incomplete: {root}")
    report = _json(report_path)
    wrapper = _json(wrapper_path)
    predictions = [row for _n, row, _sha in iter_jsonl(predictions_path)]
    artifact = report.get("profile_predictions", {})
    plan_path = _plan_dir(config, plan["pair_id"], plan["training_run_id"]) / "evaluation_plan.json"
    ticket_path = Path(wrapper.get("ticket_path", ""))
    if not ticket_path.is_file():
        raise ValueError("A5 Fast-only Run ticket file is missing")
    ticket = _json(ticket_path)
    verify_run_ticket(
        ticket,
        stage="candidate_generation",
        run_id=plan["fast_run_id"],
        config_sha256=plan["config_sha256"],
        audit_manifest_sha256=plan["audit_manifest_sha256"],
        max_sample_count=224,
    )
    expected_ticket_bindings = {
        "pair_id": plan["pair_id"],
        "group": plan["group"],
        "training_run_id": plan["training_run_id"],
        "training_manifest_sha256": plan["training_manifest_sha256"],
        "evaluation_plan_sha256": sha256_file(plan_path),
        "fast_config_sha256": plan["fast_config_sha256"],
        "inference_config_sha256": plan["inference_config_sha256"],
        "frozen_ids_sha256": plan["frozen_ids_sha256"],
        "baseline_fast_predictions_sha256": plan["baseline_fast_predictions_sha256"],
        "generation_protocol": plan["generation_protocol"],
        "fast_evaluator_source_hashes": plan["fast_evaluator_source_hashes"],
    }
    if (
        float(ticket["budget"]["max_seconds"]) > 900
        or any(ticket.get(key) != value for key, value in expected_ticket_bindings.items())
        or wrapper.get("status") != "generation_wrapper_completed"
        or wrapper.get("run_id") != plan["fast_run_id"]
        or wrapper.get("pair_id") != plan["pair_id"]
        or wrapper.get("group") != plan["group"]
        or wrapper.get("training_run_id") != plan["training_run_id"]
        or wrapper.get("subprocess_exit_code") != 0
        or wrapper.get("fast_report_status") != report.get("status")
        or wrapper.get("ticket_sha256") != sha256_file(ticket_path)
        or wrapper.get("evaluation_plan_sha256") != sha256_file(plan_path)
        or wrapper.get("fast_report_sha256") != sha256_file(report_path)
        or wrapper.get("response_text_generated_or_saved") is not False
        or wrapper.get("full_generation_performed") is not False
    ):
        raise ValueError("A5 Fast-only generation wrapper/ticket failed its frozen authorization contract")
    if (
        report.get("status") not in {"passed", "not_adopted", "budget_exhausted", "incomplete", "blocked_environment"}
        or report.get("run_id") != plan["fast_run_id"]
        or report.get("sample_count") != 224
        or report.get("accounted_count") != 224
        or report.get("references_read_by_generator") is not False
        or report.get("response_text_generated_or_saved") is not False
        or artifact.get("rows") != 224
        or artifact.get("expected_rows") != 224
        or artifact.get("response_text_saved") is not False
        or artifact.get("sha256") != sha256_file(predictions_path)
        or artifact.get("ids_sha256") != plan["frozen_ids_sha256"]
        or report.get("run_identity", {}).get("config_sha256") != plan["fast_config_sha256"]
        or report.get("run_identity", {}).get("inference_config_sha256") != plan["inference_config_sha256"]
        or report.get("run_identity", {}).get("source_code_hashes") != plan["fast_evaluator_source_hashes"]
        or report.get("response_text_generated_or_saved") is not False
        or report.get("full_generation_performed") is not False
    ):
        raise ValueError(f"A5 Fast-only report failed its frozen identity/protocol contract: {plan['fast_run_id']}")
    run_identity = report.get("run_identity", {})
    for key, expected in (
        ("adapter_weight_sha256", training_manifest["adapter_identity"]["weight_sha256"]),
        ("adapter_config_sha256", training_manifest["adapter_identity"]["config_sha256"]),
        ("base_weight_sha256", config["identity"]["base_weight_sha256"]),
    ):
        observed = run_identity.get(key)
        if report["status"] == "passed" and observed != expected:
            raise ValueError(f"A5 passed Fast-only Run has a mismatched {key}")
        if observed is not None and observed != expected:
            raise ValueError(f"A5 Fast-only Run has a mismatched {key}")
    generation = report.get("generation")
    if isinstance(generation, Mapping) and (
        generation.get("full_generation_performed", False) is not False
        or generation.get("response_text_generated_or_saved", False) is not False
    ):
        raise ValueError("A5 Fast-only report unexpectedly performed full response generation")
    source_selection = report.get("source_manifest", {}).get("selection", {})
    if (
        source_selection.get("selected_ids_sha256") != plan["frozen_ids_sha256"]
        or source_selection.get("selected_count") != 224
        or source_selection.get("selected_conversation_count") != 101
    ):
        raise ValueError("A5 Fast-only Run differs from its frozen 224-point/101-conversation cohort")
    # The expected ID order is supplied by the immutable frozen input file in the plan.
    frozen_input_path = Path(plan["frozen_input_path"])
    if sha256_file(frozen_input_path) != plan.get("frozen_input_sha256"):
        raise ValueError("A5 Fast-only plan's frozen ID source SHA changed")
    frozen_rows = [row for _n, row, _sha in iter_jsonl(frozen_input_path)]
    frozen_ids = [row["id"] for row in frozen_rows]
    if selected_id_sha256([row.get("id") for row in predictions]) != plan["frozen_ids_sha256"]:
        raise ValueError("A5 Fast-only prediction ID set differs from the pinned 224 IDs")
    enums = schema_enums(json.loads(probe_runner._resolve(config["data"]["official_output_schema"]).read_text(encoding="utf-8")))
    indexed, counts = index_profile_prediction_rows(predictions, frozen_ids, enums, require_all_valid=False)
    all_valid = counts.get("valid") == 224
    if (
        artifact.get("complete_id_coverage") is not True
        or artifact.get("all_selected_rows_valid") is not all_valid
        or artifact.get("strict_quality_comparison_ready") is not all_valid
        or (report.get("status") == "passed") is not all_valid
    ):
        raise ValueError("A5 Fast-only prediction status summary differs from its validated JSONL rows")
    return report, predictions, indexed, counts, predictions_path


def _profile_row_summary(
    ids: Sequence[str], gold: Sequence[Mapping[str, Sequence[str]]],
    predicted: Sequence[Mapping[str, Sequence[str]]], enums: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    metric = profile_metrics([dict(value) for value in gold], [dict(value) for value in predicted], {key: list(value) for key, value in enums.items() if key in {"personality_traits", "interests", "style"}})
    return {
        "sample_count": len(ids),
        "overall_33_label_tp": metric["overall_33_label_tp"],
        "overall_33_label_fp": metric["overall_33_label_fp"],
        "overall_33_label_fn": metric["overall_33_label_fn"],
        "overall_33_label_micro_precision": metric["overall_33_label_micro_precision"],
        "overall_33_label_micro_recall": metric["overall_33_label_micro_recall"],
        "overall_33_label_micro_f1": metric["overall_33_label_micro_f1"],
        "overall_33_label_macro_f1": metric["overall_33_label_macro_f1"],
        "all_three_dimensions_exact_match": metric["all_three_dimensions_exact_match"],
        "per_dimension": metric["per_dimension"],
        "per_label": {
            dimension: metric["per_dimension"][dimension]["per_label"]
            for dimension in ("personality_traits", "interests", "style")
        },
    }


def _pair_metrics(
    *, ids: Sequence[str], conversations: Sequence[str], history_lengths: Sequence[int],
    gold_profiles: Sequence[Mapping[str, Sequence[str]]], left_profiles: Sequence[Mapping[str, Sequence[str]]],
    right_profiles: Sequence[Mapping[str, Sequence[str]]], gold_emotions: Sequence[str],
    left_emotions: Sequence[str], right_emotions: Sequence[str], profile_enums: dict[str, list[str]],
    emotion_labels: list[str], seed: int, repetitions: int, short_history_max: int,
) -> dict[str, Any]:
    value = paired_stability_summary(
        ids=ids,
        conversation_ids=conversations,
        history_lengths=history_lengths,
        gold_profiles=gold_profiles,
        baseline_profiles=left_profiles,
        candidate_profiles=right_profiles,
        gold_emotions=gold_emotions,
        baseline_emotions=left_emotions,
        candidate_emotions=right_emotions,
        profile_enums=profile_enums,
        emotion_labels=emotion_labels,
        seed=seed,
        bootstrap_repetitions=repetitions,
        short_history_max_messages=short_history_max,
    )
    return {
        key: item for key, item in value.items()
        if key not in {"status", "quality_thresholds", "threshold_failures"}
    }


def _pairwise_comparisons(
    *, ids: Sequence[str], conversations: Sequence[str], history_lengths: Sequence[int],
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    baseline_profiles: Sequence[Mapping[str, Sequence[str]]],
    control_profiles: Sequence[Mapping[str, Sequence[str]]],
    candidate_profiles: Sequence[Mapping[str, Sequence[str]]],
    gold_emotions: Sequence[str], baseline_emotions: Sequence[str],
    control_emotions: Sequence[str], candidate_emotions: Sequence[str],
    profile_enums: dict[str, list[str]], emotion_labels: list[str],
    seed: int, repetitions: int, short_history_max: int,
) -> dict[str, Any]:
    common = {
        "ids": ids,
        "conversations": conversations,
        "history_lengths": history_lengths,
        "gold_profiles": gold_profiles,
        "gold_emotions": gold_emotions,
        "profile_enums": profile_enums,
        "emotion_labels": emotion_labels,
        "seed": seed,
        "repetitions": repetitions,
        "short_history_max": short_history_max,
    }
    return {
        "Candidate-Control": _pair_metrics(
            **common, left_profiles=control_profiles, right_profiles=candidate_profiles,
            left_emotions=control_emotions, right_emotions=candidate_emotions,
        ),
        "Candidate-E01-B": _pair_metrics(
            **common, left_profiles=baseline_profiles, right_profiles=candidate_profiles,
            left_emotions=baseline_emotions, right_emotions=candidate_emotions,
        ),
        "Control-E01-B": _pair_metrics(
            **common, left_profiles=baseline_profiles, right_profiles=control_profiles,
            left_emotions=baseline_emotions, right_emotions=control_emotions,
        ),
    }


def _transition_summary(
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    control_profiles: Sequence[Mapping[str, Sequence[str]]],
    candidate_profiles: Sequence[Mapping[str, Sequence[str]]],
    profile_enums: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    added_tp = added_fp = removed_tp = fixed_fp = 0
    for dimension in ("personality_traits", "interests", "style"):
        counts = {"candidate_added_tp": 0, "candidate_added_fp": 0, "candidate_removed_tp": 0, "candidate_fixed_fp": 0}
        for gold, control, candidate in zip(gold_profiles, control_profiles, candidate_profiles):
            gold_set, control_set, candidate_set = set(gold[dimension]), set(control[dimension]), set(candidate[dimension])
            for label in profile_enums[dimension]:
                gold_has, control_has, candidate_has = label in gold_set, label in control_set, label in candidate_set
                if not control_has and candidate_has:
                    counts["candidate_added_tp" if gold_has else "candidate_added_fp"] += 1
                if control_has and not candidate_has:
                    counts["candidate_removed_tp" if gold_has else "candidate_fixed_fp"] += 1
        counts["net_tp_change"] = counts["candidate_added_tp"] - counts["candidate_removed_tp"]
        counts["net_fp_change"] = counts["candidate_added_fp"] - counts["candidate_fixed_fp"]
        dimensions[dimension] = counts
        added_tp += counts["candidate_added_tp"]
        added_fp += counts["candidate_added_fp"]
        removed_tp += counts["candidate_removed_tp"]
        fixed_fp += counts["candidate_fixed_fp"]
    return {
        "candidate_added_tp_vs_control": added_tp,
        "candidate_added_fp_vs_control": added_fp,
        "candidate_added_label_precision": added_tp / (added_tp + added_fp) if added_tp + added_fp else 0.0,
        "candidate_removed_tp_vs_control": removed_tp,
        "candidate_fixed_fp_vs_control": fixed_fp,
        "per_dimension": dimensions,
    }


def _empty_gold_fp_summary(
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    predictions: Mapping[str, Sequence[Mapping[str, Sequence[str]]]],
) -> dict[str, Any]:
    empty_indices = [index for index, gold in enumerate(gold_profiles) if all(not gold[dimension] for dimension in ("personality_traits", "interests", "style"))]
    result: dict[str, Any] = {"gold_all_dimensions_empty_count": len(empty_indices), "per_model": {}}
    for name, profiles in predictions.items():
        per_dimension = {}
        total = 0
        samples_with_fp = 0
        for dimension in ("personality_traits", "interests", "style"):
            count = sum(len(profiles[index][dimension]) for index in empty_indices)
            per_dimension[dimension] = count
            total += count
        samples_with_fp = sum(
            any(profiles[index][dimension] for dimension in ("personality_traits", "interests", "style"))
            for index in empty_indices
        )
        result["per_model"][name] = {
            "false_positive_label_count": total,
            "samples_with_any_false_positive": samples_with_fp,
            "false_positive_by_dimension": per_dimension,
        }
    return result


def _label_count_summary(
    profiles_by_model: Mapping[str, Sequence[Mapping[str, Sequence[str]]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model, profiles in profiles_by_model.items():
        counts = [sum(len(profile[dimension]) for dimension in ("personality_traits", "interests", "style")) for profile in profiles]
        histogram: dict[str, int] = {}
        for count in counts:
            key = str(count)
            histogram[key] = histogram.get(key, 0) + 1
        result[model] = {
            "sample_count": len(counts),
            "mean_labels_per_sample": sum(counts) / len(counts) if counts else 0.0,
            "histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        }
    return result


def _rare_label_summary(
    config: Mapping[str, Any], audit: Mapping[str, Any],
    metrics_by_model: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    train_schema = json.loads(probe_runner._resolve(config["data"]["official_train_schema"]).read_text(encoding="utf-8"))
    output_schema = json.loads(probe_runner._resolve(config["data"]["official_output_schema"]).read_text(encoding="utf-8"))
    label_order, _enums = schema_label_order(train_schema, output_schema)
    supervision = probe_runner._read_supervision(audit["supervision"])
    support = [0] * len(label_order)
    for row in supervision.values():
        for index, value in enumerate(row["y"]):
            support[index] += int(value)
    results = []
    for index, count in enumerate(support):
        if count >= 50:
            continue
        dimension, label = label_order[index].split(":", 1)
        results.append({
            "label": label_order[index],
            "train_positive_support": count,
            "models": {
                name: metrics["per_label"][dimension][label]
                for name, metrics in metrics_by_model.items()
            },
        })
    return results


def _write_comparison_failure(
    config: Mapping[str, Any], args: argparse.Namespace, status: str,
    message: str, facts: Mapping[str, Any],
) -> int:
    output_dir = _pair_eval_root(config, args.pair_id) / "comparisons" / _validate_id(args.run_id, "comparison_run_id")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path, manifest_path = output_dir / "comparison_summary.json", output_dir / "evaluation_manifest.json"
    summary = {
        "status": status,
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "pair_id": args.pair_id,
        "failure_reason": message,
        "evidence": dict(facts),
        "candidate_adoption": "not_allowed",
        "full_json_and_reply_evaluation": "not_run",
    }
    _write_once(summary_path, summary)
    _write_once(manifest_path, {
        "status": status,
        "pair_id": args.pair_id,
        "comparison_run_id": args.run_id,
        "summary_sha256": sha256_file(summary_path),
        "candidate_adoption": "not_allowed",
        "created_at_utc": summary["created_at_utc"],
    })
    print(json.dumps({"status": status, "reason": message, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


def _compare(args: argparse.Namespace) -> int:
    config, config_sha = _load_a5(args.config)
    audit = _audit(config, config_sha, args.audit_run_id)
    pair, _schedule, _pair_root = train_runner._load_pair(config, config_sha, audit, args.pair_id)
    control_plan, control_plan_path = _load_plan(config, config_sha, audit, args.pair_id, args.control_training_run_id)
    candidate_plan, candidate_plan_path = _load_plan(config, config_sha, audit, args.pair_id, args.candidate_training_run_id)
    if control_plan.get("group") != "control" or candidate_plan.get("group") != "candidate_sft_bce":
        raise ValueError("A5 comparison requires one Control plan and one Candidate plan")
    gate_root = _pair_eval_root(config, args.pair_id) / "quality_gates" / _validate_id(args.quality_gate_run_id, "quality_gate_run_id")
    gate_path, gate_manifest_path = gate_root / "quality_gate.json", gate_root / "quality_gate_manifest.json"
    gate, gate_manifest = _json(gate_path), _json(gate_manifest_path)
    if (
        gate.get("status") != "a5_quality_gates_frozen"
        or gate.get("approved") is not True
        or not isinstance(gate.get("approved_by"), str)
        or not gate.get("approved_by", "").strip()
        or not isinstance(gate.get("rationale"), str)
        or not gate.get("rationale", "").strip()
        or gate.get("run_id") != args.quality_gate_run_id
        or gate.get("pair_id") != args.pair_id
        or gate.get("config_sha256") != config_sha
        or gate.get("audit_manifest_sha256") != audit["manifest_sha256"]
        or gate.get("pair_manifest_sha256") != sha256_file(train_runner._pair_paths(config, args.pair_id)["manifest"])
        or gate.get("frozen_ids_sha256") != config["identity"]["frozen_dev_224_ids_sha256"]
        or gate.get("task_plan_sha256") != sha256_file(ROOT / "docs/task/E0.3-A5_画像BCE辅助监督.md")
        or gate.get("thresholds") != QUALITY_GATES
        or gate.get("control_evaluation_plan_sha256") != sha256_file(control_plan_path)
        or gate.get("candidate_evaluation_plan_sha256") != sha256_file(candidate_plan_path)
        or gate_manifest.get("status") != "a5_quality_gates_frozen"
        or gate_manifest.get("run_id") != args.quality_gate_run_id
        or gate_manifest.get("pair_id") != args.pair_id
        or gate_manifest.get("outputs", {}).get("quality_gate.json") != sha256_file(gate_path)
    ):
        raise ValueError("A5 quality thresholds are not frozen against these comparison plans")
    for training_run_id, fast_run_id in (
        (args.control_training_run_id, control_plan["fast_run_id"]),
        (args.candidate_training_run_id, candidate_plan["fast_run_id"]),
    ):
        if fast_run_id != (args.control_fast_run_id if training_run_id == args.control_training_run_id else args.candidate_fast_run_id):
            raise ValueError("A5 comparison Fast Run IDs differ from their frozen plans")

    a4_config, _a4_sha = _load_a4_config()
    assets = _load_eval_assets(config)
    _control_root, control_training, _control_identity, _control_adapter = _verify_training_run(
        config, config_sha, audit, pair, args.control_training_run_id, "control_sft",
    )
    _candidate_root, candidate_training, _candidate_identity, _candidate_adapter = _verify_training_run(
        config, config_sha, audit, pair, args.candidate_training_run_id, "candidate_sft_bce",
    )
    paired_fields = (
        "pair_id", "pair_manifest_sha256", "train_schedule_sha256", "sample_ids_sha256",
        "lambda_freeze_manifest_sha256", "optimizer", "base_weight_sha256_before",
        "base_weight_sha256_after", "base_weights_unchanged", "initial_adapter_unchanged",
    )
    differences = [
        field for field in paired_fields
        if control_training.get(field) != candidate_training.get(field)
    ]
    if differences:
        raise ValueError(f"A5 Control and Candidate training manifests are not a matched pair: {differences}")
    baseline_predictions = [row for _n, row, _sha in iter_jsonl(assets["baseline_paths"]["predictions"])]
    baseline_report = _json(assets["baseline_paths"]["report"])
    control_report, control_rows, control_index, control_counts, control_prediction_path = _load_fast_run(config, control_plan, control_training)
    candidate_report, candidate_rows, candidate_index, candidate_counts, candidate_prediction_path = _load_fast_run(config, candidate_plan, candidate_training)
    output_dir = _pair_eval_root(config, args.pair_id) / "comparisons" / _validate_id(args.run_id, "comparison_run_id")
    status_counts = {
        "E01-B": assets["baseline_status_counts"],
        "Control": control_counts,
        "Candidate": candidate_counts,
    }
    failures = {
        name: counts for name, counts in status_counts.items()
        if counts.get("valid") != 224
    }
    if failures:
        status = "candidate_rejected" if failures.get("Candidate") else "comparison_incomplete"
        return _write_comparison_failure(
            config, args, status,
            "all three frozen-cohort Fast-only prediction artifacts must be 224/224 valid for the strict quality comparison",
            {
                "status_counts": status_counts,
                "baseline_report_sha256": sha256_file(assets["baseline_paths"]["report"]),
                "control_report_sha256": sha256_file(_eval_root(config) / "fast_only" / control_plan["fast_run_id"] / "profile_fast_metrics.json"),
                "candidate_report_sha256": sha256_file(_eval_root(config) / "fast_only" / candidate_plan["fast_run_id"] / "profile_fast_metrics.json"),
                "control_valid_rows": sum(row.get("status") == "valid" for row in control_rows),
                "candidate_valid_rows": sum(row.get("status") == "valid" for row in candidate_rows),
            },
        )

    ids = list(assets["frozen_ids"])
    baseline_index, _ = index_profile_prediction_rows(
        baseline_predictions, ids, assets["output_enums"], require_all_valid=True,
    )
    input_rows = [row for _n, row, _sha in iter_jsonl(assets["source_paths"]["frozen_input"])]
    reference_rows = [row for _n, row, _sha in iter_jsonl(assets["source_paths"]["references"])]
    id_map_rows = [row for _n, row, _sha in iter_jsonl(assets["source_paths"]["id_map"])]
    input_index = unique_index(input_rows, "id", "A5 frozen target-free inputs")
    reference_index = unique_index(reference_rows, "eval_id", "A5 frozen references")
    id_map_index = unique_index(id_map_rows, "eval_id", "A5 frozen ID map")
    if set(ids) != set(input_index) or set(ids) != set(reference_index) or set(ids) != set(id_map_index):
        raise ValueError("A5 frozen input, Gold, ID map, and Fast-only predictions do not share the same 224 IDs")
    conversations, history_lengths = [], []
    gold_profiles, gold_emotions = [], []
    baseline_profiles, control_profiles, candidate_profiles = [], [], []
    baseline_emotions, control_emotions, candidate_emotions = [], [], []
    for sample_id in ids:
        history = input_index[sample_id].get("history")
        conversation_id = id_map_index[sample_id].get("conversation_id")
        target = reference_index[sample_id].get("target")
        profile = target.get("user_profile") if isinstance(target, dict) else None
        emotion = target.get("emotion_label") if isinstance(target, dict) else None
        if (
            not isinstance(history, list) or not history
            or not isinstance(conversation_id, str) or not conversation_id
            or not isinstance(profile, dict) or not isinstance(emotion, str)
        ):
            raise ValueError(f"frozen Dev sample {sample_id} has invalid history/Gold/cluster metadata")
        conversations.append(conversation_id)
        history_lengths.append(len(history))
        gold_profiles.append({dimension: list(profile[dimension]) for dimension in ("personality_traits", "interests", "style")})
        gold_emotions.append(emotion)
        baseline_profiles.append(baseline_index[sample_id]["user_profile"])
        control_profiles.append(control_index[sample_id]["user_profile"])
        candidate_profiles.append(candidate_index[sample_id]["user_profile"])
        baseline_emotions.append(baseline_index[sample_id]["emotion_label"])
        control_emotions.append(control_index[sample_id]["emotion_label"])
        candidate_emotions.append(candidate_index[sample_id]["emotion_label"])

    profile_enums = {dimension: assets["output_enums"][dimension] for dimension in ("personality_traits", "interests", "style")}
    emotion_labels = assets["output_enums"]["emotion_label"]
    repetitions = int(a4_config["evaluation"]["bootstrap_repetitions"])
    seed = int(a4_config["evaluation"]["bootstrap_seed"])
    short_max = int(a4_config["evaluation"]["short_history_max_messages"])
    pairwise = _pairwise_comparisons(
        ids=ids, conversations=conversations, history_lengths=history_lengths,
        gold_profiles=gold_profiles, baseline_profiles=baseline_profiles,
        control_profiles=control_profiles, candidate_profiles=candidate_profiles,
        gold_emotions=gold_emotions, baseline_emotions=baseline_emotions,
        control_emotions=control_emotions, candidate_emotions=candidate_emotions,
        profile_enums=profile_enums, emotion_labels=emotion_labels,
        seed=seed, repetitions=repetitions, short_history_max=short_max,
    )
    metrics_by_model = {
        "E01-B": _profile_row_summary(ids, gold_profiles, baseline_profiles, profile_enums),
        "Control": _profile_row_summary(ids, gold_profiles, control_profiles, profile_enums),
        "Candidate": _profile_row_summary(ids, gold_profiles, candidate_profiles, profile_enums),
    }
    emotion_by_model = {
        "E01-B": emotion_metrics(gold_emotions, baseline_emotions, emotion_labels),
        "Control": emotion_metrics(gold_emotions, control_emotions, emotion_labels),
        "Candidate": emotion_metrics(gold_emotions, candidate_emotions, emotion_labels),
    }
    transitions = _transition_summary(gold_profiles, control_profiles, candidate_profiles, profile_enums)
    empty_fp = _empty_gold_fp_summary(
        gold_profiles,
        {"E01-B": baseline_profiles, "Control": control_profiles, "Candidate": candidate_profiles},
    )
    label_counts = _label_count_summary(
        {"E01-B": baseline_profiles, "Control": control_profiles, "Candidate": candidate_profiles},
    )
    rare = _rare_label_summary(config, audit, metrics_by_model)

    thresholds = gate["thresholds"]
    candidate_control_f1 = pairwise["Candidate-Control"]["candidate_minus_baseline"]["profile"]["overall_33_label_micro_f1"]
    candidate_e01b = pairwise["Candidate-E01-B"]["candidate_minus_baseline"]
    f1_ci = pairwise["Candidate-E01-B"]["conversation_cluster_bootstrap"]["confidence_intervals"]["micro_f1"]
    base_fp = metrics_by_model["E01-B"]["overall_33_label_fp"]
    candidate_fp = metrics_by_model["Candidate"]["overall_33_label_fp"]
    fp_relative = (candidate_fp - base_fp) / base_fp if base_fp else (0.0 if candidate_fp == 0 else None)
    checks = {
        "candidate_control_micro_f1_gain": candidate_control_f1 >= thresholds["candidate_control_micro_f1_min_delta"],
        "candidate_e01b_micro_f1_ci_lower": f1_ci["lower_95"] >= thresholds["candidate_e01_b_micro_f1_ci_lower_min_delta"],
        "candidate_e01b_precision": candidate_e01b["profile"]["overall_33_label_micro_precision"] >= thresholds["candidate_e01_b_precision_min_delta"],
        "candidate_e01b_false_positive_increase": fp_relative is not None and fp_relative <= thresholds["candidate_e01_b_fp_max_relative_increase"],
        "candidate_e01b_emotion_accuracy": candidate_e01b["emotion_accuracy_fixed_denominator"] >= thresholds["candidate_e01_b_emotion_accuracy_min_delta"],
    }
    threshold_failures = [name for name, passed in checks.items() if not passed]
    result_status = "candidate_preliminarily_passed" if not threshold_failures else "candidate_rejected"

    difference_rows = []
    for sample_id, gold, control_profile, candidate_profile in zip(ids, gold_profiles, control_profiles, candidate_profiles):
        changed = any(set(control_profile[dim]) != set(candidate_profile[dim]) for dim in ("personality_traits", "interests", "style"))
        if changed:
            difference_rows.append({
                "id": sample_id,
                "gold_labels": dict(gold),
                "control_labels": dict(control_profile),
                "candidate_labels": dict(candidate_profile),
                "status": "profile_changed",
            })
    output_dir = _pair_eval_root(config, args.pair_id) / "comparisons" / _validate_id(args.run_id, "comparison_run_id")
    output_dir.mkdir(parents=True, exist_ok=False)
    diff_path = output_dir / "profile_differences.jsonl"
    _write_jsonl_once(diff_path, difference_rows)
    summary_path, manifest_path = output_dir / "comparison_summary.json", output_dir / "evaluation_manifest.json"
    summary = {
        "status": result_status,
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "pair_id": args.pair_id,
        "development_proxy_only": True,
        "decision_scope": "preliminary engineering screen on the already-used frozen 224-point development proxy; not a statistical non-inferiority proof",
        "sample_count": len(ids),
        "conversation_count": len(set(conversations)),
        "history_short_max_messages": short_max,
        "status_counts": status_counts,
        "pairwise_comparisons": pairwise,
        "profile_metrics_by_model": metrics_by_model,
        "emotion_metrics_by_model": emotion_by_model,
        "candidate_added_label_transitions_vs_control": transitions,
        "gold_all_empty_false_positives": empty_fp,
        "predicted_profile_label_counts": label_counts,
        "rare_labels_train_support_below_50": rare,
        "probe_head_vs_final_json_disagreement": "not_run",
        "fast_run_performance": {
            "E01-B": _fast_performance(baseline_report),
            "Control": _fast_performance(control_report),
            "Candidate": _fast_performance(candidate_report),
        },
        "frozen_quality_gate": {
            "run_id": args.quality_gate_run_id,
            "manifest_sha256": sha256_file(gate_manifest_path),
            "approved_by": gate["approved_by"],
            "rationale": gate["rationale"],
            "thresholds": thresholds,
            "checks": checks,
            "threshold_failures": threshold_failures,
        },
        "profile_difference_table": {
            "path": str(diff_path),
            "sha256": sha256_file(diff_path),
            "changed_rows": len(difference_rows),
            "contains_raw_text": False,
        },
        "fast_report_sha256": {
            "E01-B": sha256_file(assets["baseline_paths"]["report"]),
            "Control": sha256_file(control_prediction_path.with_name("profile_fast_metrics.json")),
            "Candidate": sha256_file(candidate_prediction_path.with_name("profile_fast_metrics.json")),
        },
        "full_json_and_reply_smoke": "not_run",
        "official_val_used": False,
        "candidate_adoption": "not_allowed_by_preliminary_proxy_alone",
    }
    _write_once(summary_path, summary)
    manifest = {
        "status": result_status,
        "created_at_utc": summary["created_at_utc"],
        "experiment_id": config["experiment_id"],
        "pair_id": args.pair_id,
        "comparison_run_id": args.run_id,
        "config_sha256": config_sha,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "pair_manifest_sha256": sha256_file(train_runner._pair_paths(config, args.pair_id)["manifest"]),
        "control_training_manifest_sha256": sha256_file(train_runner._run_root(config) / args.control_training_run_id / "run_manifest.json"),
        "candidate_training_manifest_sha256": sha256_file(train_runner._run_root(config) / args.candidate_training_run_id / "run_manifest.json"),
        "control_fast_report_sha256": sha256_file(control_prediction_path.with_name("profile_fast_metrics.json")),
        "candidate_fast_report_sha256": sha256_file(candidate_prediction_path.with_name("profile_fast_metrics.json")),
        "baseline_fast_report_sha256": sha256_file(assets["baseline_paths"]["report"]),
        "baseline_fast_predictions_sha256": sha256_file(assets["baseline_paths"]["predictions"]),
        "quality_gate_manifest_sha256": sha256_file(gate_manifest_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "profile_difference_table_sha256": sha256_file(diff_path),
        "candidate_scoring_status": result_status,
        "full_json_and_reply_smoke": "not_run",
        "official_val_used": False,
    }
    _write_once(manifest_path, manifest)
    print(json.dumps({
        "status": result_status,
        "pair_id": args.pair_id,
        "candidate_control_micro_f1_delta": candidate_control_f1,
        "threshold_failures": threshold_failures,
        "changed_profile_rows": len(difference_rows),
        "summary": str(summary_path),
    }, ensure_ascii=False))
    return 0


def _fast_performance(report: Mapping[str, Any]) -> dict[str, Any]:
    timing = report.get("timing", {})
    generation = report.get("generation", {})
    return {
        "status": report.get("status"),
        "elapsed_seconds_including_model_load": timing.get("elapsed_seconds_including_model_load", timing.get("elapsed_seconds")),
        "generation_seconds": generation.get("generation_seconds", timing.get("fast_generation_seconds")),
        "peak_memory_bytes": generation.get("peak_memory_bytes", timing.get("fast_peak_memory_bytes")),
        "valid_count": report.get("valid_count"),
        "attempted_count": report.get("attempted_count"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/e03_a5.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-fast")
    prepare.add_argument("--audit-run-id", required=True)
    prepare.add_argument("--pair-id", required=True)
    prepare.add_argument("--training-run-id", required=True)
    prepare.add_argument("--group", required=True, choices=("control", "candidate"))
    prepare.add_argument("--fast-run-id", required=True)
    generate = commands.add_parser("run-fast")
    generate.add_argument("--audit-run-id", required=True)
    generate.add_argument("--pair-id", required=True)
    generate.add_argument("--training-run-id", required=True)
    generate.add_argument("--group", required=True, choices=("control", "candidate"))
    generate.add_argument("--fast-run-id", required=True)
    generate.add_argument("--ticket", type=Path, required=True)
    freeze = commands.add_parser("freeze-quality-gates")
    freeze.add_argument("--audit-run-id", required=True)
    freeze.add_argument("--pair-id", required=True)
    freeze.add_argument("--control-training-run-id", required=True)
    freeze.add_argument("--candidate-training-run-id", required=True)
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--approved-by", required=True)
    freeze.add_argument("--rationale", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--audit-run-id", required=True)
    compare.add_argument("--pair-id", required=True)
    compare.add_argument("--control-training-run-id", required=True)
    compare.add_argument("--candidate-training-run-id", required=True)
    compare.add_argument("--control-fast-run-id", required=True)
    compare.add_argument("--candidate-fast-run-id", required=True)
    compare.add_argument("--quality-gate-run-id", required=True)
    compare.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-fast":
            _validate_id(args.fast_run_id, "fast_run_id")
            return _prepare_fast(args)
        if args.command == "run-fast":
            _validate_id(args.fast_run_id, "fast_run_id")
            return _run_fast(args)
        if args.command == "freeze-quality-gates":
            return _freeze_quality_gates(args)
        return _compare(args)
    except RunAuthorizationError as exc:
        print(json.dumps({"status": "blocked_run_authorization", "error": str(exc)}, ensure_ascii=False))
        return 3
    except Exception as exc:
        print(json.dumps({
            "status": f"{args.command}_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "candidate_generation_started": False if args.command != "run-fast" else None,
        }, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
