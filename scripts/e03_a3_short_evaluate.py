#!/usr/bin/env python3
"""Run and compare A3 adapters on the frozen 224-point development-proxy set."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as runner
from src.diagnostics.e03_a3_profile import validate_history
from src.evaluation import (
    jsonl_bytes,
    load_submission_schema,
    read_jsonl,
    schema_enums,
    sha256_file,
)
from src.evaluation.metrics import emotion_metrics, profile_metrics, raw_diagnostics, response_metrics
from src.evaluation.score import _model_identity_verification, _validate_runtime_compatibility

DIMENSIONS = ("personality_traits", "interests", "style")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,95}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_compatible(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("version") != "E03-A3-short-training-v1"
        or value.get("authorization", {}).get("authorized") is not True
        or value.get("evaluation", {}).get("full_dev_generation_authorization") != "separate_required"
    ):
        raise ValueError("evaluation requires the authorized A3 short-run config and separate Full Dev gate")
    return value


def _unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{label} has a missing or duplicate {key} at row {index}")
        result[value] = row
    return result


def _verify_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{description} SHA-256 differs from the frozen configuration")


def _frozen_dataset(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity, evaluation = config["identity"], config["evaluation"]
    root = _resolve(evaluation["frozen_diagnosis_run_root"])
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("run_id") != root.name:
        raise ValueError("frozen A3 diagnosis Run is incomplete")
    selection_path = root / "frozen_eval_selection.json"
    selection = _json(selection_path)
    _verify_hash(selection_path, identity["expected_frozen_eval_selection_sha256"], "frozen eval selection")
    if (
        selection.get("status") != "frozen"
        or selection.get("selected_points") != int(evaluation["frozen_eval_points"])
        or selection.get("selected_conversations") != int(evaluation["frozen_eval_conversations"])
        or selection.get("ids_sha256") != identity["expected_frozen_eval_ids_sha256"]
        or selection.get("selection_uses_predictions") is not False
        or selection.get("conversation_groups_split") is not False
    ):
        raise ValueError("frozen evaluation selection differs from the locked membership")
    names = {
        "input": ("frozen_eval_input.jsonl", evaluation["frozen_eval_input_sha256"]),
        "references": ("frozen_eval_references.jsonl", evaluation["frozen_eval_references_sha256"]),
        "id_map": ("frozen_eval_id_map.jsonl", evaluation["frozen_eval_id_map_sha256"]),
    }
    for filename, digest in names.values():
        _verify_hash(root / filename, digest, filename)
        if manifest.get("outputs", {}).get(filename, {}).get("sha256") != digest:
            raise ValueError(f"diagnosis Manifest does not bind {filename}")
    inputs = _unique(read_jsonl(root / names["input"][0]), "id", "frozen input")
    references = _unique(read_jsonl(root / names["references"][0]), "eval_id", "frozen References")
    id_map = _unique(read_jsonl(root / names["id_map"][0]), "eval_id", "frozen ID map")
    expected_ids = set(references)
    if set(inputs) != expected_ids or set(id_map) != expected_ids or len(expected_ids) != int(evaluation["frozen_eval_points"]):
        raise ValueError("frozen input, References, and ID map do not share the same complete ID set")
    conversation_ids: set[str] = set()
    for eval_id in expected_ids:
        row, reference, mapping = inputs[eval_id], references[eval_id], id_map[eval_id]
        if set(row) != {"id", "history"}:
            raise ValueError("frozen generation input must contain only id and history")
        validate_history(row["history"])
        if not row["history"] or row["history"][-1].get("role") != "user":
            raise ValueError("frozen history must be non-empty and end with a user turn")
        if reference.get("sample_id") != mapping.get("sample_id"):
            raise ValueError("frozen reference and ID map sample IDs differ")
        conversation = mapping.get("conversation_id")
        if not isinstance(conversation, str) or not conversation:
            raise ValueError("frozen ID map lacks a conversation ID")
        conversation_ids.add(conversation)
    if len(conversation_ids) != int(evaluation["frozen_eval_conversations"]):
        raise ValueError("frozen ID map has the wrong number of conversation clusters")
    dataset = {
        "root": str(root),
        "run_id": root.name,
        "diagnosis_manifest_sha256": sha256_file(manifest_path),
        "selection_sha256": sha256_file(selection_path),
        "input": names["input"][0],
        "input_sha256": names["input"][1],
        "references": names["references"][0],
        "references_sha256": names["references"][1],
        "id_map": names["id_map"][0],
        "id_map_sha256": names["id_map"][1],
        "sample_count": len(expected_ids),
        "conversation_count": len(conversation_ids),
        "ids_sha256": identity["expected_frozen_eval_ids_sha256"],
    }
    return dataset, inputs, references, id_map


def _training_run(config: dict[str, Any], path: Path, expected_group: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = path.resolve()
    if not root.is_relative_to(_resolve(config["data"]["output_root"])):
        raise ValueError("evaluation accepts only a Run under outputs/training/E03-A3-short")
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("run_id") != root.name
        or manifest.get("experiment_id") != "E03-A3"
        or manifest.get("training_scope") != "short_contrast"
        or manifest.get("status") != "trained_pending_eval"
        or manifest.get("optimizer_steps") != int(config["training"]["max_optimizer_steps"])
        or manifest.get("reload_check_status") != "passed"
        or manifest.get("base_weights_unchanged") is not True
        or manifest.get("optimizer_state_resume_enabled") is not False
        or manifest.get("authorization_receipt") != _json_compatible(config["authorization"])
        or (expected_group is not None and manifest.get("group") != expected_group)
    ):
        raise ValueError("evaluation requires an authorized, reloaded 128-step A3 training Run")
    snapshot_path = root / "config_snapshot.yaml"
    _verify_hash(snapshot_path, manifest.get("config_sha256", ""), "training config snapshot")
    if yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) != config:
        raise ValueError("training config snapshot differs from the frozen evaluation config")
    adapter_path = root / "adapter"
    adapter_identity, adapter_config = runner.adapter_weight_identity(adapter_path)
    expected = manifest.get("adapter_identity", {})
    if any(adapter_identity.get(key) != expected.get(key) for key in ("weight_sha256", "config_sha256")):
        raise ValueError("Adapter identity differs from the training Manifest")
    runner.validate_adapter_base_reference(
        adapter_config, config["base_model"]["model_id"], _resolve(config["base_model"]["local_path"]),
    )
    return {
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "group": manifest["group"],
        "optimizer_steps": manifest["optimizer_steps"],
        "adapter_identity": adapter_identity,
        "base_weight_sha256": manifest.get("base_weight_sha256"),
    }, adapter_identity


def _snapshots(
    config: dict[str, Any],
    eval_root: Path,
    training_root: Path,
    adapter_identity: dict[str, Any],
    hardware_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inference_config = _json(_resolve("participant/configs/inference_config.json"))
    baseline = yaml.safe_load(_resolve("configs/evaluation/e01_b_full.yaml").read_text(encoding="utf-8"))
    if (
        config["protocol"]["prompt_sha256"] != baseline["protocol"]["prompt_sha256"]
        or config["protocol"]["max_prompt_tokens"] != baseline["protocol"]["max_prompt_tokens"]
        or config["protocol"]["tokenizer_signature_sha256"] != baseline["protocol"]["tokenizer_signature_sha256"]
        or config["protocol"]["chat_template_sha256"] != baseline["protocol"]["chat_template_sha256"]
        or config["protocol"]["tokenizer_revision"] != baseline["protocol"]["tokenizer_revision"]
        or config["evaluation"]["max_new_tokens"] != baseline["generation"]["max_new_tokens"]
        or config["evaluation"]["do_sample"] is not False
    ):
        raise ValueError("A3 generation protocol differs from the frozen E01-B P0/F2/T1/G protocol")
    base_sha = config["base_model"]["expected_weight_sha256"]
    inference_config.update({
        "backend": "transformers",
        "model_id": config["base_model"]["model_id"],
        "model_path": str(_resolve(config["base_model"]["local_path"])),
        "model_revision": config["base_model"]["revision"],
        "experiment_id": "E03-A3",
        "hardware_label": hardware_label,
        "expected_weight_sha256": base_sha,
        "generation": {"max_new_tokens": int(config["evaluation"]["max_new_tokens"]), "do_sample": False},
        "adapter": {
            "enabled": True,
            "path": str(training_root / "adapter"),
            "expected_base_weight_sha256": base_sha,
            "expected_adapter_weight_sha256": adapter_identity["weight_sha256"],
            "expected_adapter_config_sha256": adapter_identity["config_sha256"],
        },
    })
    inference_config["quantization"].update({
        "enabled": True,
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "use_double_quant": True,
    })
    inference_config["context"]["max_prompt_tokens"] = int(config["protocol"]["max_prompt_tokens"])
    evaluation_config = copy.deepcopy(baseline)
    evaluation_config["experiment_id"] = "E03-A3"
    evaluation_config["execution"]["scope"] = "frozen-224-development-proxy"
    evaluation_config["execution"]["output_root"] = str(eval_root)
    evaluation_config["execution"]["expected_base_weight_sha256"] = base_sha
    evaluation_config["model"]["adapter"] = {
        key: value for key, value in inference_config["adapter"].items() if key != "path"
    }
    evaluation_config["data"]["expected_dev_count"] = int(config["evaluation"]["frozen_eval_points"])
    evaluation_config["generation"] = {
        "max_new_tokens": int(config["evaluation"]["max_new_tokens"]), "do_sample": False,
    }
    return inference_config, evaluation_config


def _invoke_runner(eval_root: Path, record: dict[str, Any], *, resume: bool) -> int:
    env = {key: value for key, value in os.environ.items() if not key.startswith("EVALUATION_")}
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "EVALUATION_INFERENCE_CONFIG": str(eval_root / "inference_config.json"),
        "EVALUATION_EVALUATION_CONFIG": str(eval_root / "evaluation_config.yaml"),
        "EVALUATION_RUN_ID": record["run_id"],
        "EVALUATION_EXPERIMENT_ID": "E03-A3",
        "EVALUATION_RAW_OUTPUT_PATH": str(eval_root / "raw_predictions.jsonl"),
    })
    if resume:
        env["EVALUATION_RESUME"] = "1"
    command = [
        sys.executable,
        str(ROOT / "participant/run_inference.py"),
        str(Path(record["dataset"]["root"]) / record["dataset"]["input"]),
        str(eval_root / "runner"),
    ]
    with (eval_root / "inference.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def _score_run(
    eval_root: Path,
    record: dict[str, Any],
    evaluation_config: dict[str, Any],
    inputs: dict[str, Any],
    references: dict[str, Any],
    id_map: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = set(references)
    predictions = _unique(read_jsonl(eval_root / "runner/submission.jsonl"), "id", "A3 submission")
    raw_rows = _unique(read_jsonl(eval_root / "raw_predictions.jsonl"), "id", "A3 raw output")
    if set(predictions) != expected_ids or set(raw_rows) != expected_ids or set(inputs) != expected_ids:
        raise ValueError("A3 generation is incomplete or IDs differ from the frozen 224-point set")
    report_path = eval_root / "runner/performance_report.json"
    report = _json(report_path)
    if (
        report.get("run_id") != record["run_id"]
        or report.get("experiment_id") != "E03-A3"
        or report.get("samples", report.get("submission_rows")) != len(expected_ids)
        or report.get("submission_sha256") != sha256_file(eval_root / "runner/submission.jsonl")
        or report.get("raw_predictions_sha256") != sha256_file(eval_root / "raw_predictions.jsonl")
        or report.get("execution_mode", "model_inference") != "model_inference"
    ):
        raise ValueError("A3 inference report does not bind complete model-inference outputs")
    runtime = _validate_runtime_compatibility(report, evaluation_config, "D0.3_dev")
    model_identity = _model_identity_verification(report, "D0.3_dev", evaluation_config)
    if (
        runtime.get("status") != "verified"
        or model_identity.get("status") != "verified_base_and_adapter"
        or model_identity.get("observed_weight_hash") != evaluation_config["execution"]["expected_base_weight_sha256"]
    ):
        raise ValueError("A3 runtime protocol or model identity check failed")
    schema = load_submission_schema(ROOT)
    validator = Draft202012Validator(schema)
    enums = schema_enums(schema)
    ordered_ids = list(references)
    ordered_refs = [references[key]["target"] for key in ordered_ids]
    ordered_predictions = [predictions[key] for key in ordered_ids]
    ordered_raw = [raw_rows[key] for key in ordered_ids]
    schema_validity = []
    for eval_id, prediction in zip(ordered_ids, ordered_predictions):
        errors = list(validator.iter_errors(prediction))
        if errors:
            raise ValueError(f"Submission Schema failure for {eval_id}: {errors[0].message}")
        schema_validity.append(True)
    gold_emotions = [row["emotion_label"] for row in ordered_refs]
    predicted_emotions = [row["emotion_label"] for row in ordered_predictions]
    gold_profiles = [row["user_profile"] for row in ordered_refs]
    predicted_profiles = [row["user_profile"] for row in ordered_predictions]
    gold_responses = [row["response_text"] for row in ordered_refs]
    predicted_responses = [row["response_text"] for row in ordered_predictions]
    metrics = {
        "metrics_scope": "local_proxy_development_proxy",
        "run_id": record["run_id"],
        "experiment_id": "E03-A3",
        "sample_count": len(expected_ids),
        "dataset": record["dataset"],
        "hashes": {
            "inference_config_sha256": sha256_file(eval_root / "inference_config.json"),
            "evaluation_config_sha256": sha256_file(eval_root / "evaluation_config.yaml"),
            "input_sha256": record["dataset"]["input_sha256"],
            "references_sha256": record["dataset"]["references_sha256"],
            "id_map_sha256": record["dataset"]["id_map_sha256"],
            "submission_sha256": sha256_file(eval_root / "runner/submission.jsonl"),
            "raw_predictions_sha256": sha256_file(eval_root / "raw_predictions.jsonl"),
            "performance_report_sha256": sha256_file(report_path),
        },
        "runtime_compatibility": runtime,
        "model_identity_verification": model_identity,
        "emotion": emotion_metrics(gold_emotions, predicted_emotions, enums["emotion_label"]),
        "profile": profile_metrics(gold_profiles, predicted_profiles, enums),
        "response": response_metrics(
            gold_responses, predicted_responses, evaluation_config["scoring"], workspace_root=ROOT,
        ),
        "raw_diagnostics": raw_diagnostics(
            ordered_raw, enums, ordered_predictions, schema_validity,
            generation_cap=int(evaluation_config["generation"]["max_new_tokens"]),
        ),
        "history_strata": {},
        "limitations": [
            "This is a previously used development-proxy subset, not a blind test.",
            "Local proxy metrics are not official competition points.",
            "The full 1,833-point Dev set was not regenerated by this A3 runner.",
        ],
    }
    for label, selected in (
        ("raw_history_1_4", [key for key in ordered_ids if len(inputs[key]["history"]) <= 4]),
        ("raw_history_5_plus", [key for key in ordered_ids if len(inputs[key]["history"]) > 4]),
    ):
        gold_nonempty = sum(
            any(references[key]["target"]["user_profile"][dimension] for dimension in DIMENSIONS)
            for key in selected
        )
        false_empty = sum(
            any(references[key]["target"]["user_profile"][dimension] for dimension in DIMENSIONS)
            and all(not predictions[key]["user_profile"][dimension] for dimension in DIMENSIONS)
            for key in selected
        )
        metrics["history_strata"][label] = {
            "sample_count": len(selected),
            "profile": profile_metrics(
                [references[key]["target"]["user_profile"] for key in selected],
                [predictions[key]["user_profile"] for key in selected],
                enums,
            ),
            "gold_nonempty_profile_points": gold_nonempty,
            "predicted_empty_profile_points": sum(
                all(not predictions[key]["user_profile"][dimension] for dimension in DIMENSIONS)
                for key in selected
            ),
            "gold_nonempty_predicted_empty_points": false_empty,
            "gold_nonempty_to_predicted_empty_rate": false_empty / gold_nonempty if gold_nonempty else None,
        }
    cases = []
    for eval_id in ordered_ids:
        target, prediction, raw = references[eval_id]["target"], predictions[eval_id], raw_rows[eval_id]
        one_raw = raw_diagnostics(
            [raw], enums, [prediction], [True],
            generation_cap=int(evaluation_config["generation"]["max_new_tokens"]),
        )
        profile_counts = {}
        for dimension in DIMENSIONS:
            gold, predicted = set(target["user_profile"][dimension]), set(prediction["user_profile"][dimension])
            profile_counts[dimension] = {
                "tp": len(gold & predicted), "fp": len(predicted - gold), "fn": len(gold - predicted),
            }
        cases.append({
            "id": eval_id,
            "conversation_id": id_map[eval_id]["conversation_id"],
            "raw_history_length": len(inputs[eval_id]["history"]),
            "gold_profile_empty": not any(target["user_profile"][dimension] for dimension in DIMENSIONS),
            "emotion_correct": prediction["emotion_label"] == target["emotion_label"],
            "profile_counts": profile_counts,
            "raw_json_valid": one_raw["raw_json_parse_rate"] == 1.0,
            "raw_a0_schema_valid": one_raw["raw_schema_valid_rate"] == 1.0,
            "response_text": prediction["response_text"],
            "gold_response_text": target["response_text"],
        })
    _write_json(eval_root / "metrics.json", metrics)
    (eval_root / "cases.jsonl").write_bytes(jsonl_bytes(cases))
    return metrics


def execute(
    config: dict[str, Any],
    config_path: Path,
    training_run_root: Path,
    run_id: str,
    *,
    hardware_label: str,
    resume: bool = False,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("--run-id must be 3-96 safe ASCII characters")
    dataset, inputs, references, id_map = _frozen_dataset(config)
    training, adapter_identity = _training_run(config, training_run_root)
    eval_root = _resolve(config["evaluation"]["run_root"]) / run_id
    inference_config, evaluation_config = _snapshots(
        config, eval_root, Path(training["root"]), adapter_identity, hardware_label,
    )
    inference_bytes = (json.dumps(inference_config, ensure_ascii=False, indent=2) + "\n").encode()
    evaluation_bytes = yaml.safe_dump(evaluation_config, allow_unicode=True, sort_keys=False).encode()
    identity = {
        "run_id": run_id,
        "model": "adapter",
        "group": training["group"],
        "training_run": training,
        "dataset": dataset,
        "config_sha256": sha256_file(config_path),
        "source_code_sha256": sha256_file(Path(__file__).resolve()),
        "inference_config_sha256": hashlib.sha256(inference_bytes).hexdigest(),
        "evaluation_config_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
    }
    manifest_path = eval_root / "run_manifest.json"
    if resume:
        record = _json(manifest_path)
        if record.get("status") in {"scored", "completed"} or any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("A3 resume manifest is complete or its identity/config/data/Adapter/code changed")
        for filename, digest in (
            ("inference_config.json", identity["inference_config_sha256"]),
            ("evaluation_config.yaml", identity["evaluation_config_sha256"]),
        ):
            _verify_hash(eval_root / filename, digest, filename)
    else:
        eval_root.mkdir(parents=True, exist_ok=False)
        (eval_root / "inference_config.json").write_bytes(inference_bytes)
        (eval_root / "evaluation_config.yaml").write_bytes(evaluation_bytes)
        record = {
            **identity, "status": "prepared", "created_at_utc": _now(),
            "references_read_by_generator": False,
        }
        _write_json(manifest_path, record)
    if not (eval_root / "runner/performance_report.json").is_file():
        record.update(status="running", started_at_utc=record.get("started_at_utc", _now()))
        _write_json(manifest_path, record)
        return_code = _invoke_runner(eval_root, record, resume=resume)
        if return_code:
            record.update(status="inference_failed", return_code=return_code, ended_at_utc=_now())
            _write_json(manifest_path, record)
            return {"status": "inference_failed", "run_root": str(eval_root)}
    record.update(status="inference_completed")
    _write_json(manifest_path, record)
    try:
        metrics = _score_run(eval_root, record, evaluation_config, inputs, references, id_map)
    except BaseException as exc:
        record.update(status="scoring_failed", error=f"{type(exc).__name__}: {exc}", ended_at_utc=_now())
        _write_json(manifest_path, record)
        raise
    paths = (
        "runner/submission.jsonl", "runner/performance_report.json", "raw_predictions.jsonl",
        "metrics.json", "cases.jsonl",
    )
    record.update(
        status="scored",
        ended_at_utc=_now(),
        metrics_sha256=sha256_file(eval_root / "metrics.json"),
        artifacts={name: sha256_file(eval_root / name) for name in paths},
        metric_summary={
            "profile_micro_f1": metrics["profile"]["overall_33_label_micro_f1"],
            "profile_tp": metrics["profile"]["overall_33_label_tp"],
            "profile_fp": metrics["profile"]["overall_33_label_fp"],
            "profile_fn": metrics["profile"]["overall_33_label_fn"],
            "emotion_accuracy": metrics["emotion"]["accuracy"],
            "raw_json_valid_rate": metrics["raw_diagnostics"]["raw_json_parse_rate"],
            "raw_a0_schema_valid_rate": metrics["raw_diagnostics"]["raw_schema_valid_rate"],
        },
    )
    _write_json(manifest_path, record)
    return {"status": "scored", "run_root": str(eval_root), "metrics": record["metric_summary"]}


def _profile_counts(cases: list[dict[str, Any]], indices: list[int], model: str) -> dict[str, int]:
    tp = fp = fn = 0
    for index in indices:
        for dimension in DIMENSIONS:
            row = cases[index][model]["profile_counts"][dimension]
            tp += row["tp"]
            fp += row["fp"]
            fn += row["fn"]
    return {"tp": tp, "fp": fp, "fn": fn}


def _rates(cases: list[dict[str, Any]], indices: list[int], model: str) -> dict[str, float]:
    if not indices:
        return {key: 0.0 for key in (
            "profile_micro_f1", "profile_precision", "profile_recall",
            "emotion_accuracy", "raw_json_valid_rate", "raw_a0_schema_valid_rate",
        )}
    counts = _profile_counts(cases, indices, model)
    f1_denom = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    precision_denom = counts["tp"] + counts["fp"]
    recall_denom = counts["tp"] + counts["fn"]
    return {
        "profile_micro_f1": 2 * counts["tp"] / f1_denom if f1_denom else 0.0,
        "profile_precision": counts["tp"] / precision_denom if precision_denom else 0.0,
        "profile_recall": counts["tp"] / recall_denom if recall_denom else 0.0,
        "emotion_accuracy": sum(bool(cases[i][model]["emotion_correct"]) for i in indices) / len(indices),
        "raw_json_valid_rate": sum(bool(cases[i][model]["raw_json_valid"]) for i in indices) / len(indices),
        "raw_a0_schema_valid_rate": sum(bool(cases[i][model]["raw_a0_schema_valid"]) for i in indices) / len(indices),
    }


def paired_cluster_bootstrap(
    cases: list[dict[str, Any]],
    groups: dict[str, str],
    candidate_key: str,
    control_key: str,
    *,
    seed: int = 42,
    repetitions: int = 1000,
) -> dict[str, Any]:
    if not cases or repetitions < 1 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("paired bootstrap needs unique cases and positive repetitions")
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        conversation = groups[case["id"]]
        if not isinstance(conversation, str) or not conversation:
            raise ValueError("every case needs a conversation cluster")
        clusters[conversation].append(index)
    cluster_values = list(clusters.values())
    indices = list(range(len(cases)))
    point_candidate, point_control = _rates(cases, indices, candidate_key), _rates(cases, indices, control_key)
    rng = random.Random(seed)
    deltas = {
        key: [] for key in (
            "profile_micro_f1", "profile_precision", "profile_recall",
            "emotion_accuracy", "raw_json_valid_rate", "raw_a0_schema_valid_rate",
        )
    }
    for _ in range(repetitions):
        selected: list[int] = []
        for _cluster in cluster_values:
            selected.extend(rng.choice(cluster_values))
        candidate, control = _rates(cases, selected, candidate_key), _rates(cases, selected, control_key)
        for key in deltas:
            deltas[key].append(candidate[key] - control[key])
    intervals = {}
    for key, values in deltas.items():
        values.sort()
        intervals[key] = [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]
    return {
        "candidate_key": candidate_key,
        "control_key": control_key,
        "candidate_minus_control": {key: point_candidate[key] - point_control[key] for key in deltas},
        "cluster_bootstrap_95_percent_intervals": intervals,
        "interval_method": "paired_conversation_cluster_percentile_bootstrap_v1",
        "cluster_count": len(cluster_values),
        "seed": seed,
        "repetitions": repetitions,
    }


def _full_b_baseline(config: dict[str, Any], expected_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = config["evaluation"]
    root = _resolve(evaluation["e01_b_full_eval_run_root"])
    manifest_path = root / "run_manifest.json"
    _verify_hash(manifest_path, evaluation["e01_b_full_eval_manifest_sha256"], "E01-B evaluation Manifest")
    manifest = _json(manifest_path)
    if (
        manifest.get("status") != "scored"
        or manifest.get("scope") != "full-dev"
        or manifest.get("model") != "adapter"
        or manifest.get("run_id") != root.name
        or manifest.get("training_run", {}).get("manifest_sha256") != config["identity"]["expected_e01_b_manifest_sha256"]
    ):
        raise ValueError("E01-B Full Dev reference is not the frozen Full Adapter evaluation")
    for relative in ("runner/submission.jsonl", "raw_predictions.jsonl"):
        _verify_hash(root / relative, manifest.get("artifacts", {}).get(relative, ""), f"E01-B {relative}")
    submission = _unique(read_jsonl(root / "runner/submission.jsonl"), "id", "E01-B Full submission")
    raw = _unique(read_jsonl(root / "raw_predictions.jsonl"), "id", "E01-B Full raw")
    if not expected_ids <= set(submission) or not expected_ids <= set(raw):
        raise ValueError("E01-B Full Run lacks frozen A3 evaluation IDs")
    return [submission[key] for key in expected_ids], [raw[key] for key in expected_ids]


def _comparison_summary(
    references: dict[str, Any],
    id_map: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    raw_predictions: dict[str, list[dict[str, Any]]],
    eval_config: dict[str, Any],
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = load_submission_schema(ROOT)
    validator = Draft202012Validator(schema)
    enums = schema_enums(schema)
    ordered_ids = list(references)
    metrics: dict[str, Any] = {}
    maps = {
        model: (_unique(predictions[model], "id", f"{model} submission"),
                _unique(raw_predictions[model], "id", f"{model} raw"))
        for model in ("control", "candidate", "e01_b_full")
    }
    for model, (pred_map, raw_map) in maps.items():
        if set(pred_map) != set(references) or set(raw_map) != set(references):
            raise ValueError(f"{model} output IDs differ from frozen References")
        ordered_predictions = [pred_map[key] for key in ordered_ids]
        ordered_raw = [raw_map[key] for key in ordered_ids]
        schema_validity = []
        for eval_id, row in zip(ordered_ids, ordered_predictions):
            errors = list(validator.iter_errors(row))
            if errors:
                raise ValueError(f"{model} submission schema failure for {eval_id}: {errors[0].message}")
            schema_validity.append(True)
        gold = [references[key]["target"] for key in ordered_ids]
        metrics[model] = {
            "emotion": emotion_metrics(
                [row["emotion_label"] for row in gold],
                [row["emotion_label"] for row in ordered_predictions],
                enums["emotion_label"],
            ),
            "profile": profile_metrics(
                [row["user_profile"] for row in gold],
                [row["user_profile"] for row in ordered_predictions],
                enums,
            ),
            "response": response_metrics(
                [row["response_text"] for row in gold],
                [row["response_text"] for row in ordered_predictions],
                eval_config["scoring"],
                workspace_root=ROOT,
            ),
            "raw_diagnostics": raw_diagnostics(
                ordered_raw, enums, ordered_predictions, schema_validity,
                generation_cap=int(eval_config["generation"]["max_new_tokens"]),
            ),
        }
        metrics[model]["history_strata"] = {}
        for label, selected in (
            ("raw_history_1_4", [key for key in ordered_ids if id_map[key]["raw_history_length"] <= 4]),
            ("raw_history_5_plus", [key for key in ordered_ids if id_map[key]["raw_history_length"] > 4]),
        ):
            gold_nonempty = sum(
                any(references[key]["target"]["user_profile"][dimension] for dimension in DIMENSIONS)
                for key in selected
            )
            false_empty = sum(
                any(references[key]["target"]["user_profile"][dimension] for dimension in DIMENSIONS)
                and all(not pred_map[key]["user_profile"][dimension] for dimension in DIMENSIONS)
                for key in selected
            )
            metrics[model]["history_strata"][label] = {
                "sample_count": len(selected),
                "profile": profile_metrics(
                    [references[key]["target"]["user_profile"] for key in selected],
                    [pred_map[key]["user_profile"] for key in selected],
                    enums,
                ),
                "gold_nonempty_profile_points": gold_nonempty,
                "gold_nonempty_predicted_empty_points": false_empty,
                "gold_nonempty_to_predicted_empty_rate": false_empty / gold_nonempty if gold_nonempty else None,
            }
    cases = []
    for eval_id in ordered_ids:
        target = references[eval_id]["target"]
        case = {
            "id": eval_id,
            "conversation_id": id_map[eval_id]["conversation_id"],
            "raw_history_length": id_map[eval_id]["raw_history_length"],
            "gold_profile_empty": not any(target["user_profile"][dimension] for dimension in DIMENSIONS),
        }
        for model, (pred_map, raw_map) in maps.items():
            prediction, raw = pred_map[eval_id], raw_map[eval_id]
            one_raw = raw_diagnostics(
                [raw], enums, [prediction], [True],
                generation_cap=int(eval_config["generation"]["max_new_tokens"]),
            )
            profile_counts = {}
            for dimension in DIMENSIONS:
                gold_values = set(target["user_profile"][dimension])
                predicted_values = set(prediction["user_profile"][dimension])
                profile_counts[dimension] = {
                    "tp": len(gold_values & predicted_values),
                    "fp": len(predicted_values - gold_values),
                    "fn": len(gold_values - predicted_values),
                }
            case[model] = {
                "emotion_correct": prediction["emotion_label"] == target["emotion_label"],
                "profile_counts": profile_counts,
                "raw_json_valid": one_raw["raw_json_parse_rate"] == 1.0,
                "raw_a0_schema_valid": one_raw["raw_schema_valid_rate"] == 1.0,
            }
        cases.append(case)
    groups = {key: id_map[key]["conversation_id"] for key in ordered_ids}
    comparisons = {}
    for comparator in ("control", "e01_b_full"):
        comparisons[f"candidate_minus_{comparator}"] = paired_cluster_bootstrap(
            cases, groups, "candidate", comparator,
            seed=bootstrap_seed, repetitions=bootstrap_repetitions,
        )
    transition_sources = {}
    for label, selected in (
        ("raw_history_1_4", [case for case in cases if case["raw_history_length"] <= 4]),
        ("raw_history_5_plus", [case for case in cases if case["raw_history_length"] > 4]),
        ("gold_profile_empty", [case for case in cases if case["gold_profile_empty"]]),
        ("gold_profile_nonempty", [case for case in cases if not case["gold_profile_empty"]]),
    ):
        counts_by_model = {}
        false_empty_by_model = {}
        for model in ("control", "candidate"):
            totals = {"tp": 0, "fp": 0, "fn": 0}
            false_empty = 0
            for case in selected:
                predicted_label_count = 0
                for dimension in DIMENSIONS:
                    item = case[model]["profile_counts"][dimension]
                    for key in totals:
                        totals[key] += item[key]
                    predicted_label_count += item["tp"] + item["fp"]
                if not case["gold_profile_empty"] and predicted_label_count == 0:
                    false_empty += 1
            counts_by_model[model] = totals
            false_empty_by_model[model] = false_empty
        transition_sources[label] = {
            "sample_count": len(selected),
            "control": counts_by_model["control"],
            "candidate": counts_by_model["candidate"],
            "candidate_minus_control": {
                key: counts_by_model["candidate"][key] - counts_by_model["control"][key]
                for key in ("tp", "fp", "fn")
            },
            "gold_nonempty_predicted_empty_points": false_empty_by_model,
        }
    summary = {
        "status": "completed",
        "metrics_scope": "local_proxy_development_proxy",
        "sample_count": len(ordered_ids),
        "conversation_count": len(set(groups.values())),
        "dataset_ids_sha256": _canonical_sha256(ordered_ids),
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "improvement_sources": transition_sources,
        "stop_gates": _stop_gates(metrics, comparisons),
        "limitations": [
            "All comparisons use the frozen 224-point development-proxy subset.",
            "Candidate versus Control is primary; E01-B Full is a secondary reference.",
            "Conversation-cluster bootstrap describes sampling uncertainty, not causal certainty.",
            "No full 1,833-point Dev or official Val generation was run.",
        ],
    }
    return summary, cases


def _stop_gates(metrics: dict[str, Any], comparisons: dict[str, Any]) -> dict[str, Any]:
    candidate, control, base = metrics["candidate"], metrics["control"], metrics["e01_b_full"]
    primary = comparisons["candidate_minus_control"]
    deltas = primary["candidate_minus_control"]
    f1_interval = primary["cluster_bootstrap_95_percent_intervals"]["profile_micro_f1"]
    tp_delta = candidate["profile"]["overall_33_label_tp"] - control["profile"]["overall_33_label_tp"]
    fp_delta = candidate["profile"]["overall_33_label_fp"] - control["profile"]["overall_33_label_fp"]
    control_fp = control["profile"]["overall_33_label_fp"]
    precision_delta = candidate["profile"]["overall_33_label_micro_precision"] - control["profile"]["overall_33_label_micro_precision"]
    short_f1_delta = (
        candidate["history_strata"]["raw_history_1_4"]["profile"]["overall_33_label_micro_f1"]
        - control["history_strata"]["raw_history_1_4"]["profile"]["overall_33_label_micro_f1"]
    )
    non_short_f1_delta = (
        candidate["history_strata"]["raw_history_5_plus"]["profile"]["overall_33_label_micro_f1"]
        - control["history_strata"]["raw_history_5_plus"]["profile"]["overall_33_label_micro_f1"]
    )
    dimension_f1_deltas = {
        dimension: (
            candidate["profile"]["per_dimension"][dimension]["local_micro_f1"]
            - control["profile"]["per_dimension"][dimension]["local_micro_f1"]
        )
        for dimension in DIMENSIONS
    }
    bleu_candidate = candidate["response"]["local_corpus_bleu4"]["score_percent"]
    bleu_control = control["response"]["local_corpus_bleu4"]["score_percent"]
    rouge_candidate = candidate["response"]["local_rouge_l_macro_f1"]["value"]
    rouge_control = control["response"]["local_rouge_l_macro_f1"]["value"]
    bleu_delta = (bleu_candidate - bleu_control) / bleu_control if bleu_control else None
    rouge_delta = (rouge_candidate - rouge_control) / rouge_control if rouge_control else None
    both_below_base = (
        candidate["profile"]["overall_33_label_micro_f1"] < base["profile"]["overall_33_label_micro_f1"]
        and control["profile"]["overall_33_label_micro_f1"] < base["profile"]["overall_33_label_micro_f1"]
    )
    gates = {
        "profile_micro_f1_gain_at_least_0_03": deltas["profile_micro_f1"] >= 0.03,
        "profile_tp_increased": tp_delta > 0,
        "profile_f1_interval_excludes_zero": f1_interval[0] > 0 or f1_interval[1] < 0,
        "profile_fp_increase_at_most_10_percent": fp_delta <= 0.10 * control_fp if control_fp else fp_delta <= 0,
        "profile_precision_not_lower_by_more_than_0_01": precision_delta >= -0.01,
        "target_short_history_profile_f1_improved": short_f1_delta > 0,
        "non_target_profile_f1_not_lower_by_more_than_0_01": non_short_f1_delta >= -0.01,
        "each_dimension_profile_f1_not_lower_by_more_than_0_01": all(value >= -0.01 for value in dimension_f1_deltas.values()),
        "emotion_accuracy_drop_at_most_0_01": deltas["emotion_accuracy"] >= -0.01,
        "raw_json_drop_at_most_0_01": deltas["raw_json_valid_rate"] >= -0.01,
        "raw_a0_drop_at_most_0_01": deltas["raw_a0_schema_valid_rate"] >= -0.01,
        "response_bleu_drop_at_most_5_percent": bleu_delta is not None and bleu_delta >= -0.05,
        "response_rouge_l_drop_at_most_5_percent": rouge_delta is not None and rouge_delta >= -0.05,
        "control_and_candidate_not_both_below_e01_b_full": not both_below_base,
    }
    if both_below_base or (gates["profile_micro_f1_gain_at_least_0_03"] and not gates["profile_tp_increased"]):
        decision = "negative_result"
    elif not gates["profile_f1_interval_excludes_zero"]:
        decision = "inconclusive"
    elif all(gates.values()):
        decision = "promising_candidate"
    else:
        decision = "negative_result"
    return {
        "decision": decision,
        "thresholds_are_pre_registered": True,
        "gates": gates,
        "observed": {
            "profile_micro_f1_delta_candidate_minus_control": deltas["profile_micro_f1"],
            "profile_tp_delta": tp_delta,
            "profile_fp_delta": fp_delta,
            "profile_precision_delta": precision_delta,
            "short_history_profile_micro_f1_delta": short_f1_delta,
            "non_short_history_profile_micro_f1_delta": non_short_f1_delta,
            "per_dimension_micro_f1_deltas": dimension_f1_deltas,
            "emotion_accuracy_delta": deltas["emotion_accuracy"],
            "raw_json_valid_rate_delta": deltas["raw_json_valid_rate"],
            "raw_a0_schema_valid_rate_delta": deltas["raw_a0_schema_valid_rate"],
            "response_bleu_relative_delta": bleu_delta,
            "response_rouge_l_relative_delta": rouge_delta,
        },
        "next_step": "separate_expansion_authorization_required",
    }


def compare(
    config: dict[str, Any],
    config_path: Path,
    control_root: Path,
    candidate_root: Path,
    run_id: str,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("--run-id must be 3-96 safe ASCII characters")
    dataset, inputs, references, id_map = _frozen_dataset(config)
    control = _load_eval_run(config, control_root, "control", config_path)
    candidate = _load_eval_run(config, candidate_root, "candidate", config_path)
    if control["dataset"] != dataset or candidate["dataset"] != dataset or control["dataset"] != candidate["dataset"]:
        raise ValueError("Control and Candidate must use the identical frozen evaluation dataset")
    control_root, candidate_root = control_root.resolve(), candidate_root.resolve()
    for record, root in ((control, control_root), (candidate, candidate_root)):
        if record.get("training_run", {}).get("group") != record["group"]:
            raise ValueError(f"{record['group']} evaluation points to the wrong training group")
    control_config = _json(control_root / "inference_config.json")
    candidate_config = _json(candidate_root / "inference_config.json")
    if control_config["hardware_label"] != candidate_config["hardware_label"]:
        raise ValueError("Control and Candidate hardware labels differ")
    left, right = copy.deepcopy(control_config), copy.deepcopy(candidate_config)
    for value in (left, right):
        value.pop("experiment_id", None)
        value.pop("adapter", None)
        value.pop("hardware_label", None)
    if left != right:
        raise ValueError("Control and Candidate generation/quantization/protocol settings differ")
    base_root = _resolve(config["evaluation"]["e01_b_full_eval_run_root"])
    base_manifest = _json(base_root / "run_manifest.json")
    base_predictions, base_raw = _full_b_baseline(config, set(references))
    prediction_maps = {
        "control": _unique(read_jsonl(control_root / "runner/submission.jsonl"), "id", "control"),
        "candidate": _unique(read_jsonl(candidate_root / "runner/submission.jsonl"), "id", "candidate"),
        "e01_b_full": _unique(base_predictions, "id", "E01-B Full subset"),
    }
    raw_maps = {
        "control": _unique(read_jsonl(control_root / "raw_predictions.jsonl"), "id", "control raw"),
        "candidate": _unique(read_jsonl(candidate_root / "raw_predictions.jsonl"), "id", "candidate raw"),
        "e01_b_full": _unique(base_raw, "id", "E01-B Full subset raw"),
    }
    input_lengths = {key: len(row["history"]) for key, row in inputs.items()}
    compared_id_map = {
        key: {**id_map[key], "raw_history_length": input_lengths[key]} for key in id_map
    }
    eval_config = yaml.safe_load((control_root / "evaluation_config.yaml").read_text(encoding="utf-8"))
    summary, cases = _comparison_summary(
        references,
        compared_id_map,
        {key: list(value.values()) for key, value in prediction_maps.items()},
        {key: list(value.values()) for key, value in raw_maps.items()},
        eval_config,
        int(config["evaluation"]["bootstrap_seed"]),
        int(config["evaluation"]["bootstrap_repetitions"]),
    )
    output_root = _resolve(config["evaluation"]["run_root"]).parent / "comparison" / run_id
    if not output_root.resolve().is_relative_to(_resolve(config["evaluation"]["run_root"]).parent):
        raise ValueError("comparison output escaped outputs/evaluation/e03_a3")
    output_root.mkdir(parents=True, exist_ok=False)
    summary.update({
        "run_id": run_id,
        "created_at_utc": _now(),
        "config_sha256": sha256_file(config_path),
        "control_run": {
            "root": str(control_root),
            "manifest_sha256": sha256_file(control_root / "run_manifest.json"),
            "metrics_sha256": control.get("metrics_sha256"),
        },
        "candidate_run": {
            "root": str(candidate_root),
            "manifest_sha256": sha256_file(candidate_root / "run_manifest.json"),
            "metrics_sha256": candidate.get("metrics_sha256"),
        },
        "e01_b_full_reference": {
            "root": str(base_root),
            "manifest_sha256": sha256_file(base_root / "run_manifest.json"),
            "run_id": base_manifest["run_id"],
        },
    })
    _write_json(output_root / "comparison.json", summary)
    (output_root / "comparison_cases.jsonl").write_bytes(jsonl_bytes(cases))
    report = [
        "# E03-A3 冻结 224 点短程对照",
        "",
        f"Decision: {summary['stop_gates']['decision']}. All metrics are development-proxy local proxies.",
        "",
        "| Metric | E01-B Full | Control-128 | Candidate-128 |",
        "|---|---:|---:|---:|",
    ]
    for title, section, key in (
        ("Profile Micro-F1", "profile", "overall_33_label_micro_f1"),
        ("Profile Macro-F1", "profile", "overall_33_label_macro_f1"),
        ("Profile TP", "profile", "overall_33_label_tp"),
        ("Profile FP", "profile", "overall_33_label_fp"),
        ("Profile FN", "profile", "overall_33_label_fn"),
        ("Three-dimension Exact Match", "profile", "all_three_dimensions_exact_match"),
        ("Emotion Accuracy", "emotion", "accuracy"),
        ("Emotion Macro-F1", "emotion", "macro_f1"),
        ("Strict JSON", "raw_diagnostics", "raw_json_parse_rate"),
        ("Raw A0 Schema", "raw_diagnostics", "raw_schema_valid_rate"),
    ):
        values = [summary["metrics"][name][section][key] for name in ("e01_b_full", "control", "candidate")]
        report.append(f"| {title} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    report.extend([
        "",
        "Candidate-128 versus Control-128 is the primary comparison; E01-B Full is the secondary same-ID reference.",
        "Conversation-cluster intervals and preregistered gates are in comparison.json. Results do not authorize expansion.",
        "The full 1,833-point Dev set was not regenerated, and official Val was not read.",
    ])
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_json(output_root / "manifest.json", {
        "run_id": run_id,
        "status": "completed",
        "created_at_utc": summary["created_at_utc"],
        "comparison_sha256": sha256_file(output_root / "comparison.json"),
        "cases_sha256": sha256_file(output_root / "comparison_cases.jsonl"),
        "report_sha256": sha256_file(output_root / "report.md"),
    })
    return {"status": "completed", "decision": summary["stop_gates"]["decision"], "output_dir": str(output_root)}


def _load_eval_run(config: dict[str, Any], root: Path, expected_group: str, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_relative_to(_resolve(config["evaluation"]["run_root"])):
        raise ValueError("comparison inputs must remain under outputs/evaluation/e03_a3/short")
    manifest = _json(root / "run_manifest.json")
    if (
        manifest.get("status") != "scored"
        or manifest.get("group") != expected_group
        or manifest.get("dataset", {}).get("ids_sha256") != config["identity"]["expected_frozen_eval_ids_sha256"]
        or manifest.get("config_sha256") != sha256_file(config_path)
        or manifest.get("source_code_sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError(f"{expected_group} A3 eval Run is incomplete or has a changed identity")
    for name, digest in manifest.get("artifacts", {}).items():
        _verify_hash(root / name, digest, name)
    _verify_hash(root / "inference_config.json", manifest["inference_config_sha256"], "inference config")
    _verify_hash(root / "evaluation_config.yaml", manifest["evaluation_config_sha256"], "evaluation config")
    if len(read_jsonl(root / "cases.jsonl")) != int(config["evaluation"]["frozen_eval_points"]):
        raise ValueError(f"{expected_group} eval case count differs from the frozen set")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/e03_a3_short.yaml"))
    parser.add_argument("--mode", choices=("run", "compare"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--training-run-root", type=Path)
    parser.add_argument("--control-run-root", type=Path)
    parser.add_argument("--candidate-run-root", type=Path)
    parser.add_argument("--hardware-label", default="local-user-gpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    config = _config(config_path)
    if args.mode == "run":
        if args.training_run_root is None:
            parser.error("run mode requires --training-run-root")
        result = execute(
            config, config_path, args.training_run_root, args.run_id,
            hardware_label=args.hardware_label, resume=args.resume,
        )
    else:
        if args.control_run_root is None or args.candidate_run_root is None:
            parser.error("compare mode requires --control-run-root and --candidate-run-root")
        if args.resume:
            parser.error("compare mode does not support --resume")
        result = compare(config, config_path, args.control_run_root, args.candidate_run_root, args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"scored", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
