#!/usr/bin/env python3
"""Freeze and score one strict, paired E03-A4 Fast-only 224-point evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import read_jsonl, schema_enums, sha256_file
from src.evaluation.profile_fast_eval import selected_id_sha256
from src.evaluation.profile_reconcile import index_profile_prediction_rows
from src.evaluation.e03_a4_stability_metrics import paired_stability_summary


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_config(config_path: Path) -> dict[str, Any]:
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "E03-A4-lr-stability-v1":
        raise ValueError("unsupported E03-A4 stability configuration")
    return value


def _run_paths(config: dict[str, Any]) -> tuple[Path, Path, str]:
    training_root = _resolve(config["data"]["output_root"]) / config["run_id"]
    fast_run_id = f"{config['run_id']}-FAST-224"
    fast_root = _resolve(config["fast_baseline"]["report_path"]).parents[1] / fast_run_id
    return training_root, fast_root, fast_run_id


def build_candidate_inference_config(
    frozen_config: dict[str, Any], config: dict[str, Any],
    training_root: Path, adapter_identity: dict[str, Any],
) -> dict[str, Any]:
    value = deepcopy(frozen_config)
    if value.get("backend") != "transformers":
        raise ValueError("A4 paired Fast-only evaluation requires Transformers")
    value["experiment_id"] = config["experiment_id"]
    value["model_path"] = str(_resolve(config["base_model"]["local_path"]))
    value["model_id"] = config["base_model"]["model_id"]
    value["model_revision"] = config["base_model"]["revision"]
    value["expected_weight_sha256"] = config["base_model"]["expected_weight_sha256"]
    value["adapter"] = {
        **value.get("adapter", {}),
        "enabled": True,
        "path": str((training_root / "adapter").resolve()),
        "expected_base_weight_sha256": config["base_model"]["expected_weight_sha256"],
        "expected_adapter_weight_sha256": adapter_identity["weight_sha256"],
        "expected_adapter_config_sha256": adapter_identity["config_sha256"],
    }
    value["generation"] = {
        **value.get("generation", {}),
        "max_new_tokens": 1200,
        "do_sample": False,
    }
    return value


def build_candidate_fast_config(
    frozen_config: dict[str, Any], config: dict[str, Any], inference_path: Path,
) -> dict[str, Any]:
    value = deepcopy(frozen_config)
    value["experiment_version"] = "E03-A4-lr-stability-fastonly-v1"
    value["sources"]["inference_config_path"] = str(inference_path)
    value["selection"].update({
        "mode": "frozen_ids",
        "frozen_ids_path": config["evaluation"]["frozen_diagnosis_run_root"] + "/frozen_eval_input.jsonl",
        "expected_count": int(config["evaluation"]["frozen_eval_points"]),
        "expected_conversation_count": int(config["evaluation"]["frozen_eval_conversations"]),
        "expected_ids_sha256": config["identity"]["expected_frozen_eval_ids_sha256"],
    })
    value["generation"] = {"max_new_tokens": 1200, "do_sample": False}
    value["gpu_budget"] = {"max_seconds": int(config["evaluation"]["fast_budget_seconds"])}
    value["adoption_gates"] = {
        **value.get("adoption_gates", {}),
        "minimum_valid_coverage": 1.0,
        "require_fixed_denominator_metrics": True,
    }
    value["output_root"] = config["evaluation"]["run_root"]
    return value


def _verify_completed_training(config: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from participant import run_inference as inference

    training_root, _fast_root, _fast_run_id = _run_paths(config)
    manifest_path = training_root / "run_manifest.json"
    identity_path = training_root / "run_identity.json"
    manifest = _json(manifest_path)
    identity = _json(identity_path)
    if (
        manifest.get("status") != "trained_pending_eval"
        or manifest.get("run_id") != config["run_id"]
        or manifest.get("optimizer_steps") != 64
        or manifest.get("draw_count") != 256
        or manifest.get("reload_check", {}).get("status") != "passed"
        or manifest.get("reload_check", {}).get("saved_reloaded_tensor_values_exact_match") is not True
        or manifest.get("base_weights_unchanged") is not True
        or manifest.get("initial_source_adapter_unchanged") is not True
        or identity.get("config_sha256") != sha256_file(_resolve("configs/training/e03_a4_lr_stability.yaml"))
    ):
        raise ValueError("A4 training is incomplete or failed its Adapter/Base reload gates")
    adapter_path = training_root / "adapter"
    adapter_identity, adapter_config = inference.adapter_weight_identity(adapter_path)
    expected_identity = manifest.get("adapter_identity", {})
    if (
        adapter_identity.get("weight_sha256") != expected_identity.get("weight_sha256")
        or adapter_identity.get("config_sha256") != expected_identity.get("config_sha256")
        or adapter_config.get("r") != config["lora"]["r"]
        or adapter_config.get("lora_alpha") != config["lora"]["alpha"]
        or set(adapter_config.get("target_modules", [])) != set(config["lora"]["target_modules"])
    ):
        raise ValueError("A4 saved Adapter identity differs from the completed training Manifest")
    return training_root, manifest, adapter_identity


def prepare(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    training_root, _fast_root, fast_run_id = _run_paths(config)
    training_root, training_manifest, adapter_identity = _verify_completed_training(config)
    evaluation_root = training_root / "evaluation"
    if evaluation_root.exists():
        raise FileExistsError(f"A4 evaluation config snapshots are immutable: {evaluation_root}")
    evaluation_root.mkdir()

    e01_eval_root = _resolve(config["evaluation"]["e01_b_full_eval_run_root"])
    e01_eval_manifest = _json(e01_eval_root / "run_manifest.json")
    base_inference_path = e01_eval_root / "inference_config.json"
    if sha256_file(base_inference_path) != e01_eval_manifest.get("inference_config_sha256"):
        raise ValueError("frozen E01-B inference config differs from its historical Run Manifest")
    fast_config_path = ROOT / "configs/evaluation/e03_profile_candidate.yaml"
    frozen_fast_config = yaml.safe_load(fast_config_path.read_text(encoding="utf-8"))
    if not isinstance(frozen_fast_config, dict):
        raise ValueError("frozen A4 candidate Fast-only config is invalid")
    inference_config = build_candidate_inference_config(
        _json(base_inference_path), config, training_root, adapter_identity,
    )
    fast_config = build_candidate_fast_config(
        frozen_fast_config, config, evaluation_root / "inference_config.json",
    )

    inference_path = evaluation_root / "inference_config.json"
    fast_config_snapshot = evaluation_root / "fast_eval_config.yaml"
    _write_once(inference_path, inference_config)
    with fast_config_snapshot.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(yaml.safe_dump(fast_config, allow_unicode=True, sort_keys=False))
        handle.flush()
        os.fsync(handle.fileno())
    plan_path = evaluation_root / "evaluation_plan.json"
    plan = {
        "status": "prepared",
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "training_run_id": config["run_id"],
        "training_manifest_sha256": sha256_file(training_root / "run_manifest.json"),
        "training_adapter_identity": adapter_identity,
        "baseline_fast_report_sha256": config["fast_baseline"]["report_sha256"],
        "baseline_prediction_sha256": config["fast_baseline"]["predictions_sha256"],
        "frozen_ids_sha256": config["identity"]["expected_frozen_eval_ids_sha256"],
        "frozen_points": int(config["evaluation"]["frozen_eval_points"]),
        "frozen_conversations": int(config["evaluation"]["frozen_eval_conversations"]),
        "fast_run_id": fast_run_id,
        "fast_budget_seconds": int(config["evaluation"]["fast_budget_seconds"]),
        "inference_config_path": str(inference_path),
        "inference_config_sha256": sha256_file(inference_path),
        "fast_config_path": str(fast_config_snapshot),
        "fast_config_sha256": sha256_file(fast_config_snapshot),
        "frozen_fast_config_source_sha256": sha256_file(fast_config_path),
        "full_generation_or_response_text": False,
    }
    _write_once(plan_path, plan)
    return {
        "status": "prepared",
        "evaluation_root": str(evaluation_root),
        "fast_run_id": fast_run_id,
        "fast_config_path": str(fast_config_snapshot),
        "inference_config_path": str(inference_path),
        "fast_config_sha256": plan["fast_config_sha256"],
        "inference_config_sha256": plan["inference_config_sha256"],
        "command": [
            ".venv/bin/python", "scripts/e03_profile_fast_eval.py", "--mode", "gpu-fast-only",
            "--config", str(fast_config_snapshot), "--run-config", str(inference_path),
            "--run-id", fast_run_id, "--require-passed",
        ],
    }


def _validate_fast_report(
    config: dict[str, Any], training_root: Path, fast_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    training_manifest = _json(training_root / "run_manifest.json")
    evaluation_root = training_root / "evaluation"
    plan = _json(evaluation_root / "evaluation_plan.json")
    report_path = fast_root / "profile_fast_metrics.json"
    prediction_path = fast_root / "profile_predictions.jsonl"
    report = _json(report_path)
    artifact = report.get("profile_predictions", {})
    identity = report.get("run_identity", {})
    generation = report.get("generation", {})
    model_facts = identity.get("model_facts", {})
    tokenizer_facts = identity.get("tokenizer_facts", {})
    expected_adapter = training_manifest["adapter_identity"]
    expected_ids_sha = config["identity"]["expected_frozen_eval_ids_sha256"]
    if (
        report.get("status") != "passed"
        or report.get("gpu_used") is not True
        or report.get("sample_count") != 224
        or report.get("valid_count") != 224
        or report.get("attempted_count") != 224
        or report.get("references_read_by_generator") is not False
        or report.get("response_text_generated_or_saved") is not False
        or artifact.get("path") != str(prediction_path.resolve())
        or artifact.get("sha256") != sha256_file(prediction_path)
        or artifact.get("rows") != 224
        or artifact.get("expected_rows") != 224
        or artifact.get("ids_sha256") != expected_ids_sha
        or artifact.get("all_selected_rows_valid") is not True
        or artifact.get("strict_quality_comparison_ready") is not True
        or generation.get("model_generate_call_count") != 224
        or generation.get("full_generation_performed") is not False
        or generation.get("response_text_generated_or_saved") is not False
        or identity.get("config_sha256") != plan["fast_config_sha256"]
        or identity.get("inference_config_sha256") != plan["inference_config_sha256"]
        or identity.get("base_weight_sha256") != config["base_model"]["expected_weight_sha256"]
        or identity.get("adapter_weight_sha256") != expected_adapter["weight_sha256"]
        or identity.get("adapter_config_sha256") != expected_adapter["config_sha256"]
        or identity.get("prompt_facts", {}).get("sha256") != config["protocol"]["prompt_sha256"]
        or identity.get("generation", {}).get("do_sample") is not False
        or identity.get("generation", {}).get("max_new_tokens") != 1200
        or tokenizer_facts.get("signature_sha256") != config["protocol"]["tokenizer_signature_sha256"]
        or tokenizer_facts.get("template_sha256") != config["protocol"]["chat_template_sha256"]
        or model_facts.get("quantization", {}).get("real_nf4_verified") is not True
        or model_facts.get("offload_devices")
    ):
        raise ValueError("A4 candidate Fast-only Run failed identity, coverage, protocol, or generation gates")
    source_selection = report.get("source_manifest", {}).get("selection", {})
    if (
        source_selection.get("selected_ids_sha256") != expected_ids_sha
        or source_selection.get("expected_ids_sha256") != expected_ids_sha
        or source_selection.get("selected_count") != 224
        or source_selection.get("selected_conversation_count") != 101
    ):
        raise ValueError("A4 candidate Fast-only Run differs from the frozen 224-point / 101-conversation cohort")
    rows = read_jsonl(prediction_path)
    ids = [row.get("id") for row in rows]
    if any(not isinstance(sample_id, str) for sample_id in ids) or selected_id_sha256(ids) != expected_ids_sha:
        raise ValueError("A4 candidate Fast-only prediction IDs differ from the frozen cohort")
    return report, rows, ids


def compare(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    training_root, fast_root, fast_run_id = _run_paths(config)
    training_root, training_manifest, _adapter_identity = _verify_completed_training(config)
    plan_path = training_root / "evaluation" / "evaluation_plan.json"
    plan = _json(plan_path)
    if plan.get("status") != "prepared" or plan.get("fast_run_id") != fast_run_id:
        raise ValueError("A4 Fast-only evaluation plan is missing or mismatched")
    if not fast_root.is_dir():
        raise FileNotFoundError(f"candidate Fast-only Run has not been generated: {fast_root}")

    report, candidate_rows, ids = _validate_fast_report(config, training_root, fast_root)
    baseline_report_path = _resolve(config["fast_baseline"]["report_path"])
    baseline_prediction_path = _resolve(config["fast_baseline"]["predictions_path"])
    if (
        sha256_file(baseline_report_path) != config["fast_baseline"]["report_sha256"]
        or sha256_file(baseline_prediction_path) != config["fast_baseline"]["predictions_sha256"]
    ):
        raise ValueError("A4-1 reconciled E01-B Fast-only baseline has changed")
    baseline_report = _json(baseline_report_path)
    baseline_rows = read_jsonl(baseline_prediction_path)
    if baseline_report.get("status") != "passed" or baseline_report.get("sample_count") != 224:
        raise ValueError("A4-1 E01-B Fast-only baseline did not pass the fixed cohort")
    diagnosis_root = _resolve(config["evaluation"]["frozen_diagnosis_run_root"])
    input_rows = read_jsonl(diagnosis_root / "frozen_eval_input.jsonl")
    reference_rows = read_jsonl(diagnosis_root / "frozen_eval_references.jsonl")
    id_map_rows = read_jsonl(diagnosis_root / "frozen_eval_id_map.jsonl")
    if sha256_file(diagnosis_root / "frozen_eval_input.jsonl") != config["evaluation"]["frozen_eval_input_sha256"]:
        raise ValueError("A4 frozen target-free inputs changed")
    if sha256_file(diagnosis_root / "frozen_eval_references.jsonl") != config["evaluation"]["frozen_eval_references_sha256"]:
        raise ValueError("A4 frozen evaluation References changed")
    if sha256_file(diagnosis_root / "frozen_eval_id_map.jsonl") != config["evaluation"]["frozen_eval_id_map_sha256"]:
        raise ValueError("A4 frozen evaluation ID map changed")

    from src.evaluation.source_integrity import unique_index
    input_index = unique_index(input_rows, "id", "frozen evaluation inputs")
    reference_index = unique_index(reference_rows, "eval_id", "frozen evaluation References")
    id_map_index = unique_index(id_map_rows, "eval_id", "frozen evaluation ID map")
    expected_ids = set(ids)
    if set(input_index) != expected_ids or set(reference_index) != expected_ids or set(id_map_index) != expected_ids:
        raise ValueError("E01-B, A4 predictions, frozen input, Gold, and ID map do not share the same 224 IDs")

    official_output_schema = _json(_resolve(config["protocol"]["output_schema"]))
    enums = schema_enums(official_output_schema)
    baseline_index, baseline_statuses = index_profile_prediction_rows(
        baseline_rows, ids, enums, require_all_valid=True,
    )
    candidate_index, candidate_statuses = index_profile_prediction_rows(
        candidate_rows, ids, enums, require_all_valid=True,
    )
    if baseline_statuses.get("valid") != 224 or candidate_statuses.get("valid") != 224:
        raise ValueError("strict A4 quality comparison requires 224 valid baseline and candidate profiles")

    conversation_ids: list[str] = []
    history_lengths: list[int] = []
    gold_profiles: list[dict[str, list[str]]] = []
    baseline_profiles: list[dict[str, list[str]]] = []
    candidate_profiles: list[dict[str, list[str]]] = []
    gold_emotions: list[str] = []
    baseline_emotions: list[str] = []
    candidate_emotions: list[str] = []
    emotion_labels = enums["emotion_label"]
    for sample_id in ids:
        input_row = input_index[sample_id]
        reference = reference_index[sample_id]
        id_map = id_map_index[sample_id]
        if set(input_row) != {"id", "history"}:
            raise ValueError(f"frozen input {sample_id} is not target-free")
        history = input_row.get("history")
        if not isinstance(history, list) or not history or history[-1].get("role") != "user":
            raise ValueError(f"frozen input {sample_id} has no complete current user history")
        if not isinstance(id_map.get("conversation_id"), str) or not id_map["conversation_id"]:
            raise ValueError(f"frozen ID {sample_id} has no conversation-cluster identity")
        target = reference.get("target")
        if not isinstance(target, dict):
            raise ValueError(f"frozen Gold {sample_id} is missing its target object")
        profile = target.get("user_profile")
        if not isinstance(profile, dict):
            raise ValueError(f"frozen Gold {sample_id} is missing its user profile")
        emotion = target.get("emotion_label")
        if not isinstance(emotion, str) or emotion not in emotion_labels:
            raise ValueError(f"frozen Gold {sample_id} has an invalid emotion label")
        conversation_ids.append(id_map["conversation_id"])
        history_lengths.append(len(history))
        gold_profiles.append(profile)
        baseline_profiles.append(baseline_index[sample_id]["user_profile"])
        candidate_profiles.append(candidate_index[sample_id]["user_profile"])
        gold_emotions.append(emotion)
        baseline_emotions.append(baseline_index[sample_id]["emotion_label"])
        candidate_emotions.append(candidate_index[sample_id]["emotion_label"])

    metrics = paired_stability_summary(
        ids=ids,
        conversation_ids=conversation_ids,
        history_lengths=history_lengths,
        gold_profiles=gold_profiles,
        baseline_profiles=baseline_profiles,
        candidate_profiles=candidate_profiles,
        gold_emotions=gold_emotions,
        baseline_emotions=baseline_emotions,
        candidate_emotions=candidate_emotions,
        profile_enums={key: enums[key] for key in ("personality_traits", "interests", "style")},
        emotion_labels=emotion_labels,
        seed=int(config["evaluation"]["bootstrap_seed"]),
        bootstrap_repetitions=int(config["evaluation"]["bootstrap_repetitions"]),
        short_history_max_messages=int(config["evaluation"]["short_history_max_messages"]),
    )
    if int(report.get("timing", {}).get("elapsed_seconds_including_model_load", 0)) > int(config["evaluation"]["fast_budget_seconds"]):
        raise ValueError("A4 candidate Fast-only evaluation exceeded its 900-second budget")

    evaluation_root = training_root / "evaluation"
    summary_path = evaluation_root / "paired_stability_summary.json"
    manifest_path = evaluation_root / "evaluation_manifest.json"
    if summary_path.exists() or manifest_path.exists():
        raise FileExistsError("A4 paired evaluation outputs are immutable and already exist")
    summary = {
        **metrics,
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "training_run_id": config["run_id"],
        "fast_run_id": fast_run_id,
        "sample_ids_sha256": config["identity"]["expected_frozen_eval_ids_sha256"],
        "candidate_fast_run": {
            "report_sha256": sha256_file(fast_root / "profile_fast_metrics.json"),
            "predictions_sha256": sha256_file(fast_root / "profile_predictions.jsonl"),
            "elapsed_seconds": report["timing"]["elapsed_seconds_including_model_load"],
            "generation_seconds": report["generation"]["generation_seconds"],
            "peak_memory_bytes": report["generation"]["peak_memory_bytes"],
            "adapter_weight_sha256": report["run_identity"]["adapter_weight_sha256"],
            "adapter_config_sha256": report["run_identity"]["adapter_config_sha256"],
            "base_weight_sha256": report["run_identity"]["base_weight_sha256"],
        },
        "baseline_fast_run": {
            "run_id": config["fast_baseline"]["run_id"],
            "report_sha256": config["fast_baseline"]["report_sha256"],
            "predictions_sha256": config["fast_baseline"]["predictions_sha256"],
            "adapter_weight_sha256": config["fast_baseline"]["expected_adapter_weight_sha256"],
        },
    }
    _write_once(summary_path, summary)
    manifest = {
        "status": "completed",
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "training_run_id": config["run_id"],
        "training_manifest_sha256": sha256_file(training_root / "run_manifest.json"),
        "evaluation_plan_sha256": sha256_file(evaluation_root / "evaluation_plan.json"),
        "fast_report_sha256": sha256_file(fast_root / "profile_fast_metrics.json"),
        "fast_predictions_sha256": sha256_file(fast_root / "profile_predictions.jsonl"),
        "baseline_fast_report_sha256": config["fast_baseline"]["report_sha256"],
        "baseline_fast_predictions_sha256": config["fast_baseline"]["predictions_sha256"],
        "reference_sha256": sha256_file(diagnosis_root / "frozen_eval_references.jsonl"),
        "input_sha256": sha256_file(diagnosis_root / "frozen_eval_input.jsonl"),
        "id_map_sha256": sha256_file(diagnosis_root / "frozen_eval_id_map.jsonl"),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "candidate_scoring_status": metrics["status"],
        "full_json_and_reply_smoke": "not_run",
    }
    _write_once(manifest_path, manifest)
    return {
        "status": metrics["status"],
        "evaluation_manifest": str(manifest_path),
        "summary": str(summary_path),
        "micro_f1_delta": metrics["candidate_minus_baseline"]["profile"]["overall_33_label_micro_f1"],
        "precision_delta": metrics["candidate_minus_baseline"]["profile"]["overall_33_label_micro_precision"],
        "fp_relative_increase": metrics["candidate_minus_baseline"]["false_positive_relative_increase"],
        "emotion_accuracy_delta": metrics["candidate_minus_baseline"]["emotion_accuracy_fixed_denominator"],
        "threshold_failures": metrics["threshold_failures"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "compare"))
    parser.add_argument("--config", default="configs/training/e03_a4_lr_stability.yaml")
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    result = prepare(config_path) if args.mode == "prepare" else compare(config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("status") in {"prepared", "stability_preliminarily_passed", "stability_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
