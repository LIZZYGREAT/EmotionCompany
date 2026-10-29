#!/usr/bin/env python3
"""Read-only aggregation of existing E01-B Full training and Dev artifacts.

This command never loads a model, starts inference, recomputes official/local
scores, or changes source artifacts. It verifies recorded identities and adds
descriptive aggregates that are not present in the versioned comparison report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import read_jsonl, schema_enums, sha256_file, strict_json_loads


TRAIN_RUN_ID = "E01-B-FULL-20261028-133805"
BASE_RUN_ID = "E00.3-BASE-20261028-223105"
ADAPTER_RUN_ID = "E01-B-EVAL-20261028-223105"
PAIR_RUN_ID = "full-20261029-062541"
PILOT_PAIR_RUN_ID = "pilot-20261028-101136"
DIMENSIONS = ("personality_traits", "interests", "style")
RAW_FIELDS = {"id", "raw_text", "status", "prompt_tokens", "generated_tokens", "stop_reason", "latency_ms", "removed_history_messages", "backend", "error_type"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required {description} is missing: {path}")


def _check_sha(path: Path, expected: str | None, description: str) -> str:
    _require_file(path, description)
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(f"{description} SHA-256 mismatch: {path}")
    return actual


def _require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description} mismatch: expected={expected!r}, actual={actual!r}")


def _final_reload_check(training_root: Path) -> dict[str, Any]:
    path = training_root / "reload_check_standalone.json"
    _require_file(path, "final standalone Adapter reload check")
    record = _json(path)
    _require_equal(record.get("status"), "passed", "final Adapter reload status")
    _require_equal(record.get("token_ids_exact_match"), True, "final reload Token ID check")
    _require_equal(record.get("base_weights_unchanged"), True, "final reload Base weight check")
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "status": "passed",
        "token_ids_exact_match": True,
        "base_weights_unchanged": True,
        "base_weight_sha256": record.get("base_weight_sha256_after"),
    }


def _require_count(actual: int, expected: int, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description} count mismatch: expected={expected}, actual={actual}")


def _require_same_ids(actual: set[str], expected: set[str], description: str) -> None:
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(f"{description} ID sets differ: missing={missing}, extra={extra}")


def _unique(rows: list[dict[str, Any]], key: str, description: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{description} contains an empty/non-string {key}")
        if value in result:
            raise ValueError(f"{description} contains duplicate {key}: {value}")
        result[value] = row
    return result


def _fact(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    for item in manifest.get("outputs", []):
        if item.get("path") == path:
            return item
    raise ValueError(f"D0.3 manifest has no output fact for {path}")


def _finite_numbers(values: list[Any], description: str) -> list[float]:
    result = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if len(result) != len(values) or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{description} contains missing or non-finite values")
    return result


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _distribution(values: list[float | int]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "min": min(numbers),
        "p50": _quantile(numbers, 0.50),
        "p90": _quantile(numbers, 0.90),
        "p95": _quantile(numbers, 0.95),
        "max": max(numbers),
    }


def _safe_parse(raw_text: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw_text, str):
        return None, "missing_or_non_string_raw_text"
    try:
        value = strict_json_loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"json_decode:{exc.msg}"
    except ValueError as exc:
        message = str(exc)
        if message.startswith("duplicate JSON object key:"):
            return None, "duplicate_json_key"
        if message.startswith("non-standard JSON constant:"):
            return None, "nonstandard_json_constant"
        return None, "strict_json_value_error"
    if not isinstance(value, dict):
        return None, "json_top_level_not_object"
    return value, None


def _raw_a0_reasons(value: dict[str, Any], enums: dict[str, list[str]]) -> list[str]:
    reasons: list[str] = []
    required_fields = {"emotion_label", "user_profile", "response_text", "memory_refs"}
    if set(value) != required_fields:
        reasons.append("fields_missing_or_extra")
    if value.get("emotion_label") not in enums["emotion_label"]:
        reasons.append("invalid_emotion")
    profile = value.get("user_profile")
    if not isinstance(profile, dict) or set(profile) != set(DIMENSIONS):
        reasons.append("invalid_profile_shape")
    else:
        profile_reasons = []
        for dimension in DIMENSIONS:
            labels = profile.get(dimension)
            if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
                profile_reasons.append("invalid_profile_type")
                continue
            if len(labels) != len(set(labels)):
                profile_reasons.append("duplicate_profile_label")
            if not set(labels) <= set(enums[dimension]):
                profile_reasons.append("profile_label_outside_enum")
        reasons.extend(sorted(set(profile_reasons)))
    response = value.get("response_text")
    if not isinstance(response, str) or not response.strip():
        reasons.append("empty_or_invalid_response")
    if value.get("memory_refs") != []:
        reasons.append("memory_refs_not_empty_array")
    return reasons


def _case_profile_totals(case: dict[str, Any]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for dimension in DIMENSIONS:
        values = case["profile"][dimension]
        tp += int(values["tp"])
        fp += int(values["fp"])
        fn += int(values["fn"])
    return tp, fp, fn


def _profile_error_kind(tp: int, fp: int, fn: int) -> str:
    if fp == 0 and fn == 0:
        return "exact"
    if tp == 0 and fp == 0 and fn > 0:
        return "all_gold_labels_missed"
    if fp == 0 and fn > 0:
        return "misses_only_partial"
    if fn == 0 and fp > 0:
        return "extra_labels_only"
    return "misses_and_extra_labels"


def _sample_errors(
    model_cases: dict[str, dict[str, Any]],
    gold_emotions: dict[str, str],
    valid_emotion_labels: set[str],
    raw_objects: dict[str, dict[str, Any] | None],
    raw_failures: dict[str, str | None],
    raw_reasons: dict[str, list[str]],
    raw_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels = sorted(set(gold_emotions.values()))
    by_gold = {label: {"support": 0, "correct": 0, "wrong": 0} for label in labels}
    error_sources: Counter[str] = Counter()
    profile_kinds: Counter[str] = Counter()
    dimensions: dict[str, Counter[str]] = {dimension: Counter() for dimension in DIMENSIONS}
    all_empty_pred = 0
    raw_failure_ids_by_kind: Counter[str] = Counter()
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "tokens": [], "latency_ms": [], "emotion_correct": 0, "profile_tp": 0, "profile_fp": 0, "profile_fn": 0})
    for eval_id, case in model_cases.items():
        emotion = case["emotion_correct"]
        gold = gold_emotions[eval_id]
        by_gold[gold]["support"] += 1
        by_gold[gold]["correct" if emotion else "wrong"] += 1
        parsed = raw_objects[eval_id]
        if not emotion:
            if raw_failures[eval_id] is not None:
                source = "raw_json_parse_failure"
            elif parsed is not None and parsed.get("emotion_label") not in valid_emotion_labels:
                source = "invalid_or_empty_emotion_label"
            elif parsed is not None:
                source = "legal_label_wrong_class"
            else:
                source = "unclassified_raw_failure"
            error_sources[source] += 1

        tp, fp, fn = _case_profile_totals(case)
        profile_kinds[_profile_error_kind(tp, fp, fn)] += 1
        profile = case["profile"]
        for dimension in DIMENSIONS:
            item = profile[dimension]
            dimensions[dimension]["gold_nonempty"] += bool(item["gold_nonempty"])
            dimensions[dimension]["pred_nonempty"] += bool(item["pred_nonempty"])
            dimensions[dimension]["nonempty_gold_hit"] += bool(item["nonempty_gold_hit"])
            dimensions[dimension]["false_empty"] += bool(item["gold_nonempty"] and not item["pred_nonempty"])
        all_empty_pred += all(not profile[d]["pred_nonempty"] for d in DIMENSIONS)

        raw_failure = raw_failures[eval_id]
        a0_reasons = raw_reasons[eval_id]
        raw_status = "strict_json_failure" if raw_failure else ("raw_a0_valid" if not a0_reasons else "json_parsed_a0_invalid")
        raw_failure_ids_by_kind[raw_status] += 1
        row = raw_rows[eval_id]
        group = grouped[raw_status]
        group["n"] += 1
        tokens = row.get("generated_tokens")
        latency = row.get("latency_ms")
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
            group["tokens"].append(tokens)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            group["latency_ms"].append(latency)
        group["emotion_correct"] += bool(emotion)
        group["profile_tp"] += tp
        group["profile_fp"] += fp
        group["profile_fn"] += fn

    dim_summary = {}
    for dimension, counts in dimensions.items():
        gold_nonempty = counts["gold_nonempty"]
        dim_summary[dimension] = {
            **counts,
            "gold_nonempty_rate": gold_nonempty / len(model_cases),
            "pred_nonempty_rate": counts["pred_nonempty"] / len(model_cases),
            "nonempty_gold_sample_recall": counts["nonempty_gold_hit"] / gold_nonempty if gold_nonempty else 0.0,
        }
    grouped_summary = {}
    for name, values in grouped.items():
        n = values["n"]
        grouped_summary[name] = {
            "sample_count": n,
            "mean_generated_tokens": statistics.fmean(values["tokens"]) if values["tokens"] else None,
            "mean_latency_ms": statistics.fmean(values["latency_ms"]) if values["latency_ms"] else None,
            "emotion_accuracy": values["emotion_correct"] / n if n else None,
            "profile_tp_fp_fn": {key: values[key] for key in ("profile_tp", "profile_fp", "profile_fn")},
        }
    return {
        "sample_count": len(model_cases),
        "emotion_by_gold_class": by_gold,
        "emotion_error_sources": dict(error_sources),
        "profile_error_categories": dict(profile_kinds),
        "profile_dimensions": dim_summary,
        "normalized_submission_all_three_profile_dimensions_empty_count": all_empty_pred,
        "raw_output_groups": grouped_summary,
        "raw_output_group_counts": dict(raw_failure_ids_by_kind),
    }


def _confusions(emotion_metrics: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    matrix = emotion_metrics["confusion_matrix"]
    labels = matrix["labels"]
    values = matrix["rows_gold_columns_predicted"]
    pairs = []
    for row_index, gold in enumerate(labels):
        for column_index, predicted in enumerate(labels):
            count = int(values[row_index][column_index])
            if gold != predicted and count:
                pairs.append({"gold": gold, "predicted": predicted, "count": count})
    return sorted(pairs, key=lambda row: (-row["count"], row["gold"], row["predicted"]))[:limit]


def _profile_label_table(
    labels: dict[str, list[str]],
    train_profiles: list[dict[str, list[str]]],
    dev_profiles: list[dict[str, list[str]]],
    adapter_profile_metrics: dict[str, Any],
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        train_counts = Counter(label for profile in train_profiles for label in profile[dimension])
        dev_counts = Counter(label for profile in dev_profiles for label in profile[dimension])
        metric = adapter_profile_metrics["per_dimension"][dimension]["per_label"]
        table[dimension] = {
            label: {
                "train_support": train_counts[label],
                "dev_support": dev_counts[label],
                **metric[label],
            }
            for label in labels[dimension]
        }
    return table


def _train_log_summary(run_root: Path, manifest: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    train_rows = read_jsonl(run_root / "train_log.jsonl")
    expected_steps = int(manifest["optimizer_steps"])
    steps = [row.get("optimizer_step") for row in train_rows]
    if steps != list(range(1, expected_steps + 1)):
        raise ValueError("Full B train_log optimizer_step sequence is not continuous and unique")
    losses = _finite_numbers([row.get("loss") for row in train_rows], "training loss")
    grad_norms = _finite_numbers([row.get("gradient_norm") for row in train_rows], "gradient norm")
    if not all(row.get("finite_gradients") is True for row in train_rows):
        raise ValueError("Full B log contains a failed gradient-finiteness check")
    update_checked_rows = [row for row in train_rows if row.get("parameter_update_checked") is True]
    if not update_checked_rows or any(
        int(row.get("updated_parameter_count", 0)) <= 0 or int(row.get("updated_element_count", 0)) <= 0
        for row in update_checked_rows
    ):
        raise ValueError("Full B sampled parameter-update checks are absent or show no LoRA update")
    eval_rows = read_jsonl(run_root / "eval_log.jsonl")
    eval_losses = _finite_numbers([row.get("sft_dev_loss") for row in eval_rows], "SFT Dev loss")
    best_index = min(range(len(eval_losses)), key=eval_losses.__getitem__)
    completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in train_rows)
    samples_seen = train_rows[-1].get("cumulative_samples_seen")
    if completion_tokens != evidence.get("consumed_completion_tokens"):
        raise ValueError("train_log completion token ledger differs from training_evidence")
    if samples_seen != manifest.get("train_rows"):
        raise ValueError("train_log cumulative sample count differs from Full B Train rows")
    windows: dict[str, list[float]] = defaultdict(list)
    for row, loss in zip(train_rows, losses):
        lower = ((int(row["optimizer_step"]) - 1) // 256) * 256 + 1
        upper = min(lower + 255, expected_steps)
        windows[f"{lower}-{upper}"].append(loss)
    eval_curve = [{"optimizer_step": row["optimizer_step"], "sft_dev_loss": row["sft_dev_loss"], "samples": row.get("eval_samples")} for row in eval_rows]
    memory_peaks: dict[str, float] = {}
    for row in train_rows:
        for key, value in row.get("cuda_memory", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                memory_peaks[key] = max(memory_peaks.get(key, float("-inf")), float(value))
    return {
        "log_rows": len(train_rows),
        "optimizer_steps_contiguous": True,
        "nonfinite_loss_or_gradient_count": 0,
        "finite_gradients_all_steps": True,
        "sampled_parameter_update_check_count": len(update_checked_rows),
        "sampled_parameter_update_checks_all_positive": True,
        "training_step_loss_mean": statistics.fmean(losses),
        "first_10_step_loss_mean_recomputed": statistics.fmean(losses[:10]),
        "last_10_step_loss_mean_recomputed": statistics.fmean(losses[-10:]),
        "manifest_first_last_10_step_loss": [manifest.get("training_loss_first_10"), manifest.get("training_loss_last_10")],
        "training_step_loss_windows_256": {key: {"steps": len(values), "mean_loss": statistics.fmean(values)} for key, values in windows.items()},
        "sft_dev_eval_count": len(eval_rows),
        "sft_dev_eval_rows": eval_curve,
        "best_sft_dev_loss": {"step": eval_rows[best_index]["optimizer_step"], "loss": eval_losses[best_index]},
        "final_sft_dev_loss": {"step": eval_rows[-1]["optimizer_step"], "loss": eval_losses[-1]},
        "sft_dev_rows": evidence.get("sft_dev_loss_rows"),
        "completion_tokens_sum_from_train_log": completion_tokens,
        "completion_tokens_recorded": evidence.get("consumed_completion_tokens"),
        "cumulative_samples_seen": samples_seen,
        "learning_rate_start_end": [train_rows[0].get("learning_rate"), train_rows[-1].get("learning_rate")],
        "gradient_norm_distribution": _distribution(grad_norms),
        "cuda_memory_step_log_peaks_bytes": memory_peaks,
    }


def _validate_inputs(training_root: Path, comparison_root: Path) -> dict[str, Any]:
    if training_root.name != TRAIN_RUN_ID:
        raise ValueError(f"unexpected training Run; expected {TRAIN_RUN_ID}")
    if comparison_root.name != PAIR_RUN_ID:
        raise ValueError(f"unexpected comparison Run; expected {PAIR_RUN_ID}")
    training_manifest_path = training_root / "manifest.json"
    training_evidence_path = training_root / "training_evidence.json"
    comparison_manifest_path = comparison_root / "manifest.json"
    comparison_summary_path = comparison_root / "comparison_summary.json"
    for path, description in (
        (training_manifest_path, "training manifest"), (training_evidence_path, "training evidence"),
        (comparison_manifest_path, "comparison manifest"), (comparison_summary_path, "comparison summary"),
    ):
        _require_file(path, description)
    training = _json(training_manifest_path)
    evidence = _json(training_evidence_path)
    comp_manifest = _json(comparison_manifest_path)
    summary = _json(comparison_summary_path)
    if training.get("status") != "passed" or training.get("stage") != "completed" or training.get("training_scope") != "full_train":
        raise ValueError("Full B training Run is not completed/passed")
    if training.get("train_rows") != 16192 or training.get("optimizer_steps") != 4048 or training.get("init_adapter") is not None:
        raise ValueError("Full B training identity/counts do not match the expected independent full Run")
    _check_sha(training_root / "config_snapshot.yaml", training.get("config_sha256"), "approved Full B config snapshot")
    if comp_manifest.get("status") != "completed" or summary.get("status") != "completed" or summary.get("scope") != "full-dev":
        raise ValueError("Full Dev paired comparison is not completed")
    _check_sha(comparison_summary_path, comp_manifest.get("summary_sha256"), "comparison summary")
    if summary.get("dataset", {}).get("sample_count") != 1833 or summary.get("paired", {}).get("sample_count") != 1833:
        raise ValueError("Full Dev comparison does not contain exactly 1,833 paired samples")

    d03_path = ROOT / "outputs/data_pipeline/v1/manifest.json"
    sft_path = ROOT / "outputs/sft_data/v1/manifest.json"
    d03 = _json(d03_path)
    sft = _json(sft_path)
    base_metrics_precheck = _json(comparison_root / "base_scored/metrics.json")
    d03_hash_from_score = base_metrics_precheck.get("hashes", {}).get("d03_manifest_sha256")
    if sha256_file(d03_path) != d03_hash_from_score:
        raise ValueError("D0.3 manifest identity differs from the paired evaluation")
    train_fact = _fact(d03, "data/processed/v1/train.jsonl")
    dev_fact = _fact(d03, "data/processed/v1/dev.jsonl")
    train_path = ROOT / train_fact["path"]
    dev_path = ROOT / dev_fact["path"]
    _check_sha(train_path, train_fact["sha256"], "D0.3 Train")
    _check_sha(dev_path, dev_fact["sha256"], "D0.3 Dev")
    if train_fact.get("records") != 16312 or dev_fact.get("records") != 1833:
        raise ValueError("D0.3 Train/Dev record counts differ from the expected data version")
    sft_files = sft.get("outputs", {}).get("files", {})
    sft_train_path = ROOT / sft_files["train.jsonl"]["path"]
    sft_dev_path = ROOT / sft_files["dev.jsonl"]["path"]
    train_rows = _unique(read_jsonl(train_path), "sample_id", "D0.3 Train")
    if len(train_rows) != 16312:
        raise ValueError("D0.3 Train file row count does not match its Manifest")
    _check_sha(sft_train_path, training.get("train_sha256"), "Full B SFT Train")
    _check_sha(sft_dev_path, training.get("sft_dev_sha256"), "Full B SFT Dev")
    if sft_files["train.jsonl"].get("sha256") != training.get("train_sha256") or sft_files["dev.jsonl"].get("sha256") != training.get("sft_dev_sha256"):
        raise ValueError("SFT data Manifest identities differ from the Full B training Run")

    dataset_root = _resolve(summary["dataset"]["root"])
    dataset_manifest_path = dataset_root / "manifest.json"
    if sha256_file(dataset_manifest_path) != summary["dataset"]["manifest_sha256"]:
        raise ValueError("paired Dev dataset manifest hash mismatch")
    dataset_manifest = _json(dataset_manifest_path)
    if (
        dataset_manifest.get("dataset_scope") != "D0.3_dev"
        or dataset_manifest.get("source", {}).get("dev_sha256") != dev_fact["sha256"]
        or dataset_manifest.get("source", {}).get("d03_manifest_sha256") != d03_hash_from_score
    ):
        raise ValueError("paired Dev dataset does not reference the verified D0.3 Dev source")
    for filename in (summary["dataset"]["input"], summary["dataset"]["references"], summary["dataset"]["id_map"]):
        expected = summary["dataset"].get(filename + "_sha256")
        _check_sha(dataset_root / filename, expected, "paired Dev " + filename)
    references = _unique(read_jsonl(dataset_root / summary["dataset"]["references"]), "eval_id", "Dev references")
    id_map = _unique(read_jsonl(dataset_root / summary["dataset"]["id_map"]), "eval_id", "Dev ID map")
    dev_rows = _unique(read_jsonl(dev_path), "sample_id", "D0.3 Dev")
    _require_same_ids(set(references), set(id_map), "Full Dev References and ID Map")
    _require_count(len(references), 1833, "Full Dev References")
    mapped_samples = {row.get("sample_id") for row in id_map.values()}
    _require_same_ids(mapped_samples, set(dev_rows), "Full Dev ID Map and D0.3 Dev")
    for eval_id, id_row in id_map.items():
        sample_id = id_row["sample_id"]
        if references[eval_id].get("sample_id") != sample_id or references[eval_id].get("target") != dev_rows[sample_id].get("target"):
            raise ValueError("Full Dev References target differs from its hash-verified D0.3 source row")
    inference_rows = read_jsonl(dataset_root / summary["dataset"]["input"])
    inference_ids = _unique(inference_rows, "id", "Full Dev inference input")
    _require_same_ids(set(inference_ids), set(references), "Full Dev inference input and References")
    ordered_id_bytes = json.dumps(list(inference_ids), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(ordered_id_bytes).hexdigest() != summary["dataset"].get("ids_sha256"):
        raise ValueError("Full Dev ordered ID digest is inconsistent with inference input")

    models: dict[str, Any] = {}
    raw_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    submission_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    metric_by_model: dict[str, dict[str, Any]] = {}
    for name, expected_run_id in (("base", BASE_RUN_ID), ("adapter", ADAPTER_RUN_ID)):
        run_root = _resolve(comp_manifest["base_run_root" if name == "base" else "adapter_run_root"])
        if run_root.name != expected_run_id:
            raise ValueError(f"paired comparison {name} Run ID mismatch")
        run_manifest = _json(run_root / "run_manifest.json")
        metrics_path = comparison_root / f"{name}_scored/metrics.json"
        metrics = _json(metrics_path)
        if run_manifest.get("status") != "scored" or run_manifest.get("scope") != "full-dev" or run_manifest.get("model") != name:
            raise ValueError(f"{name} evaluation Run is not a scored full-dev model inference")
        if run_manifest.get("references_read_by_generator") is not False:
            raise ValueError(f"{name} evaluation did not prove References were excluded from generation")
        if metrics.get("sample_count") != 1833 or metrics.get("metrics_scope") != "local_proxy":
            raise ValueError(f"{name} metrics identity/sample count mismatch")
        if metrics.get("runtime_compatibility", {}).get("status") != "verified":
            raise ValueError(f"{name} runtime compatibility is not verified")
        if metrics.get("source_verification", {}).get("status") != "verified":
            raise ValueError(f"{name} D0.3 source is not verified")
        run_data = run_manifest.get("dataset", {})
        if run_data.get("ids_sha256") != summary["dataset"].get("ids_sha256") or run_data.get("sample_count") != 1833:
            raise ValueError(f"{name} evaluation did not use the paired 1,833 ID set")
        hashes = metrics.get("hashes", {})
        if hashes.get("references_sha256") != summary["dataset"]["references.jsonl_sha256"] or hashes.get("inference_input_sha256") != summary["dataset"]["inference_input.jsonl_sha256"] or hashes.get("id_map_sha256") != summary["dataset"]["id_map.jsonl_sha256"]:
            raise ValueError(f"{name} source/References/input hashes differ from the paired comparison")
        raw_path = run_root / "raw_predictions.jsonl"
        submission_path = run_root / "runner/submission.jsonl"
        _check_sha(raw_path, hashes.get("raw_predictions_sha256"), f"{name} Raw predictions")
        _check_sha(submission_path, hashes.get("submission_sha256"), f"{name} Submission")
        raw_map = _unique(read_jsonl(raw_path), "id", f"{name} Raw predictions")
        submission_map = _unique(read_jsonl(submission_path), "id", f"{name} Submission")
        _require_same_ids(set(raw_map), set(references), f"{name} Raw predictions and References")
        _require_same_ids(set(submission_map), set(references), f"{name} Submission and References")
        if summary.get("metrics", {}).get(name, {}).get("run_id") != expected_run_id:
            raise ValueError(f"paired summary does not embed the expected {name} scorer result")
        for row in raw_map.values():
            if set(row) != RAW_FIELDS:
                raise ValueError(f"{name} Raw prediction fields do not match the recorded allowlist")
        raw_by_model[name] = raw_map
        submission_by_model[name] = submission_map
        metric_by_model[name] = metrics
        models[name] = {
            "run_id": expected_run_id,
            "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
            "metrics_sha256": sha256_file(metrics_path),
            "scorer_code_commit": metrics.get("scorer_code_commit"),
            "raw_sha256": hashes["raw_predictions_sha256"],
            "submission_sha256": hashes["submission_sha256"],
            "base_weight_sha256": metrics["model_identity_verification"]["observed_weight_hash"],
            "adapter_weight_sha256": metrics["model_identity_verification"].get("adapter_weight_sha256"),
            "revision_status": metrics["model_identity_verification"].get("revision_status"),
            "raw_map": raw_map,
            "submission_map": submission_map,
        }
    _require_equal(models["base"]["base_weight_sha256"], models["adapter"]["base_weight_sha256"], "Base model SHA across paired evaluations")
    _require_equal(models["base"]["base_weight_sha256"], training.get("base_weight_sha256"), "Base model SHA between training and evaluation")
    _require_equal(models["adapter"]["adapter_weight_sha256"], training.get("adapter_identity", {}).get("weight_sha256"), "Full Dev Adapter SHA")
    if metric_by_model["adapter"]["model_identity_verification"].get("adapter_config_sha256") != training.get("adapter_identity", {}).get("config_sha256"):
        raise ValueError("Full Dev Adapter config SHA differs from the trained Full B Adapter")
    if not (models["base"]["revision_status"] == models["adapter"]["revision_status"] == "unverified_revision"):
        raise ValueError("model revision verification status changed; inspect source identities")
    case_path = comparison_root / "comparison_cases.jsonl"
    _require_file(case_path, "paired comparison cases")
    cases = _unique(read_jsonl(case_path), "id", "paired comparison cases")
    _require_same_ids(set(cases), set(references), "paired comparison cases and References")
    for model in ("base", "adapter"):
        metrics = metric_by_model[model]
        emotion_correct = sum(bool(row[model]["emotion_correct"]) for row in cases.values())
        expected_correct = round(metrics["emotion"]["accuracy"] * 1833)
        if emotion_correct != expected_correct:
            raise ValueError(f"{model} per-ID emotion correctness differs from the already-scored aggregate")
        for label in set(row["target"]["emotion_label"] for row in references.values()):
            support = sum(row["target"]["emotion_label"] == label for row in references.values())
            if metrics["emotion"]["per_label"][label]["support"] != support:
                raise ValueError(f"{model} per-class emotion support differs from verified References: {label}")
        for dimension in DIMENSIONS:
            totals = Counter()
            exact = 0
            for row in cases.values():
                value = row[model]["profile"][dimension]
                for key in ("tp", "fp", "fn"):
                    totals[key] += int(value[key])
                exact += bool(value["exact"])
            stored = metrics["profile"]["per_dimension"][dimension]
            if any(totals[key] != stored[key] for key in ("tp", "fp", "fn")) or exact != stored["exact_match_count"]:
                raise ValueError(f"{model} per-ID profile counts differ from already-scored metrics: {dimension}")
        combined_exact = sum(all(row[model]["profile"][dimension]["exact"] for dimension in DIMENSIONS) for row in cases.values())
        if combined_exact != metrics["profile"]["all_three_dimensions_exact_match_count"]:
            raise ValueError(f"{model} per-ID three-dimension Exact Match differs from already-scored metrics")

    return {
        "training_root": training_root,
        "training_manifest_path": training_manifest_path,
        "training_evidence_path": training_evidence_path,
        "training": training,
        "evidence": evidence,
        "d03_manifest_path": d03_path,
        "d03_manifest": d03,
        "d03_train": train_rows,
        "d03_dev": dev_rows,
        "sft_manifest": sft,
        "sft_train_path": sft_train_path,
        "sft_dev_path": sft_dev_path,
        "dataset_root": dataset_root,
        "dataset_manifest_path": dataset_manifest_path,
        "dataset_manifest": dataset_manifest,
        "references": references,
        "id_map": id_map,
        "cases": cases,
        "models": models,
        "metric_by_model": metric_by_model,
        "raw_by_model": raw_by_model,
        "submission_by_model": submission_by_model,
        "comparison_root": comparison_root,
        "comparison_manifest_path": comparison_manifest_path,
        "comparison_summary_path": comparison_summary_path,
        "comparison_manifest": comp_manifest,
        "comparison_summary": summary,
    }


def analyze(training_root: Path, comparison_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    facts = _validate_inputs(training_root, comparison_root)
    schema = json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8"))
    enums = schema_enums(schema)
    train_profiles = [row["target"]["user_profile"] for row in facts["d03_train"].values()]
    dev_profiles = [row["target"]["user_profile"] for row in facts["d03_dev"].values()]
    train_emotions = Counter(row["target"]["emotion_label"] for row in facts["d03_train"].values())
    dev_emotions = Counter(row["target"]["emotion_label"] for row in facts["d03_dev"].values())
    if set(train_emotions) - set(enums["emotion_label"]):
        raise ValueError("D0.3 Train contains an emotion outside the official fixed enum")
    if set(dev_emotions) - set(enums["emotion_label"]):
        raise ValueError("D0.3 Dev contains an emotion outside the official fixed enum")
    for split_name, profiles in (("Train", train_profiles), ("Dev", dev_profiles)):
        for profile in profiles:
            if set(profile) != set(DIMENSIONS):
                raise ValueError(f"D0.3 {split_name} profile has an unexpected dimension set")
            for dimension in DIMENSIONS:
                labels = profile[dimension]
                if len(labels) != len(set(labels)) or not set(labels) <= set(enums[dimension]):
                    raise ValueError(f"D0.3 {split_name} contains malformed profile labels")

    gold_emotions = {eval_id: row["target"]["emotion_label"] for eval_id, row in facts["references"].items()}
    dev_profile_by_eval_id = {
        eval_id: facts["references"][eval_id]["target"]["user_profile"] for eval_id in facts["references"]
    }
    model_outputs: dict[str, Any] = {}
    for model in ("base", "adapter"):
        metric = facts["metric_by_model"][model]
        model_cases = {eval_id: cases[model] for eval_id, cases in facts["cases"].items()}
        raw_objects: dict[str, dict[str, Any] | None] = {}
        raw_failures: dict[str, str | None] = {}
        raw_reasons: dict[str, list[str]] = {}
        raw_latencies = []
        token_counts = []
        raw_response_char_lengths = []
        for eval_id, row in facts["raw_by_model"][model].items():
            value, failure = _safe_parse(row.get("raw_text"))
            raw_objects[eval_id] = value
            raw_failures[eval_id] = failure
            raw_reasons[eval_id] = _raw_a0_reasons(value, enums) if value is not None else []
            if isinstance(row.get("generated_tokens"), int):
                token_counts.append(row["generated_tokens"])
            if isinstance(row.get("latency_ms"), (int, float)):
                raw_latencies.append(float(row["latency_ms"]))
            if value is not None and isinstance(value.get("response_text"), str):
                raw_response_char_lengths.append(len(value["response_text"]))
        output_errors = _sample_errors(model_cases, gold_emotions, set(enums["emotion_label"]), raw_objects, raw_failures, raw_reasons, facts["raw_by_model"][model])
        generation_cap = int(metric["model"]["generation_config"]["max_new_tokens"])
        cap_count = sum(row.get("stop_reason") in {"max_new_tokens", "server_finish_reason:length", "server_finish_reason:max_tokens"} or row.get("generated_tokens") == generation_cap for row in facts["raw_by_model"][model].values())
        length_bins = []
        bins = ((0, 256), (257, 512), (513, 768), (769, 1199), (1200, generation_cap))
        for lower, upper in bins:
            rows = [row for row in facts["raw_by_model"][model].values() if isinstance(row.get("generated_tokens"), int) and lower <= row["generated_tokens"] <= upper]
            if rows:
                length_bins.append({
                    "token_range": [lower, upper],
                    "sample_count": len(rows),
                    "mean_latency_ms": statistics.fmean(float(row["latency_ms"]) for row in rows),
                    "p95_latency_ms": _quantile([float(row["latency_ms"]) for row in rows], 0.95),
                })
        raw_error_kinds = Counter(value for value in raw_failures.values() if value)
        invalid_a0_reasons = Counter(reason for eval_id, reasons in raw_reasons.items() if not raw_failures[eval_id] for reason in reasons)
        a0_valid_count = sum(not raw_failures[eval_id] and not raw_reasons[eval_id] for eval_id in facts["references"])
        parsed_count = sum(value is not None for value in raw_objects.values())
        expected_parsed = int(round(metric["raw_diagnostics"]["raw_json_parse_rate"] * len(facts["references"])))
        expected_a0_valid = int(round(metric["raw_diagnostics"]["raw_schema_valid_rate"] * len(facts["references"])))
        if parsed_count != expected_parsed or a0_valid_count != expected_a0_valid:
            raise ValueError(f"{model} raw-format diagnostic counts differ from the already-scored metric artifact")
        profile_rows = facts["metric_by_model"][model]["profile"]
        raw_a0_valid_profile_equal_to_submission = 0
        for eval_id, raw_object in raw_objects.items():
            if raw_object is not None and not raw_reasons[eval_id]:
                raw_profile = raw_object["user_profile"]
                submission_profile = facts["submission_by_model"][model][eval_id]["user_profile"]
                if all(set(raw_profile[d]) == set(submission_profile[d]) for d in DIMENSIONS):
                    raw_a0_valid_profile_equal_to_submission += 1
        model_outputs[model] = {
            "run_id": facts["models"][model]["run_id"],
            "emotion": {
                "accuracy": metric["emotion"]["accuracy"],
                "macro_f1": metric["emotion"]["macro_f1"],
                "invalid_or_empty_prediction_count": metric["emotion"]["invalid_empty_prediction_count"],
                "per_class": metric["emotion"]["per_label"],
                "top_confusions": _confusions(metric["emotion"]),
                "error_sources_among_incorrect_predictions": output_errors["emotion_error_sources"],
                "per_gold_class_correctness": output_errors["emotion_by_gold_class"],
            },
            "profile": {
                "overall_33_label_micro_f1": profile_rows["overall_33_label_micro_f1"],
                "overall_33_label_macro_f1": profile_rows["overall_33_label_macro_f1"],
                "all_three_dimensions_exact_match_count": profile_rows["all_three_dimensions_exact_match_count"],
                "all_three_dimensions_exact_match_rate": profile_rows["all_three_dimensions_exact_match"],
                "per_dimension_metrics": profile_rows["per_dimension"],
                "per_dimension_nonempty_and_recall": output_errors["profile_dimensions"],
                "sample_level_error_categories": output_errors["profile_error_categories"],
                "all_three_normalized_predictions_empty_count": output_errors["normalized_submission_all_three_profile_dimensions_empty_count"],
                "raw_output_groups": output_errors["raw_output_groups"],
                "raw_a0_valid_profile_matches_normalized_submission": raw_a0_valid_profile_equal_to_submission,
            },
            "raw_output": {
                "sample_count": len(facts["references"]),
                "raw_json_parse_count_from_existing_metric": int(round(metric["raw_diagnostics"]["raw_json_parse_rate"] * len(facts["references"]))),
                "raw_a0_schema_valid_count_from_existing_metric": int(round(metric["raw_diagnostics"]["raw_schema_valid_rate"] * len(facts["references"]))),
                "raw_json_parse_failure_kinds": dict(raw_error_kinds),
                "raw_a0_invalid_reasons": dict(invalid_a0_reasons),
                "raw_json_parse_rate": metric["raw_diagnostics"]["raw_json_parse_rate"],
                "raw_a0_schema_valid_rate": metric["raw_diagnostics"]["raw_schema_valid_rate"],
                "submission_schema_valid_rate": metric["raw_diagnostics"]["submission_schema_valid_rate"],
                "generation_cap": generation_cap,
                "generation_cap_count": cap_count,
                "generation_cap_rate": cap_count / len(facts["references"]),
                "generated_token_distribution": _distribution(token_counts),
                "latency_ms_distribution_from_all_samples": _distribution(raw_latencies),
                "generation_length_latency_bins": length_bins,
                "parsed_raw_response_character_length": _distribution(raw_response_char_lengths),
                "local_text_metrics_from_existing_scoring": {
                    "bleu4_percent": metric["response"]["local_corpus_bleu4"]["score_percent"],
                    "rouge_l_macro_f1": metric["response"]["local_rouge_l_macro_f1"]["value"],
                    "empty_response_count": metric["response"]["local_rouge_l_macro_f1"]["empty_prediction_count"],
                    "bert_score_status": metric["response"]["bert_score"]["status"],
                    "per_sample_text_metric_correlation": "unavailable_without_rescoring; intentionally not recomputed",
                },
            },
            "local_performance_report": {
                "rounds": metric["performance"]["official_performance_rounds"],
                "complete": metric["performance"]["official_performance_complete"],
                "average_latency_ms_100_round_report": metric["performance"]["official_average_latency_ms"],
                "median_latency_ms_100_round_report": metric["performance"]["official_median_latency_ms"],
                "p95_latency_ms_100_round_report": metric["performance"]["official_p95_latency_ms"],
                "classification": "local runner measurement; not official competition performance",
            },
        }

    # Compute the Gold all-empty reference baseline from the verified D0.3 Dev.
    gold_all_empty = sum(all(not profile[dimension] for dimension in DIMENSIONS) for profile in dev_profiles)
    adapter_cases = [row["adapter"]["profile"] for row in facts["cases"].values()]
    predicted_all_empty = sum(all(not profile[dimension]["pred_nonempty"] for dimension in DIMENSIONS) for profile in adapter_cases)
    gold_empty_pred_empty = sum(
        all(not profile[dimension]["gold_nonempty"] for dimension in DIMENSIONS)
        and all(not profile[dimension]["pred_nonempty"] for dimension in DIMENSIONS)
        for profile in adapter_cases
    )
    exact_all_dims = model_outputs["adapter"]["profile"]["all_three_dimensions_exact_match_count"]
    sft_dev_rows = read_jsonl(facts["sft_dev_path"])
    if len(sft_dev_rows) != 1821:
        raise ValueError("SFT Dev must contain 1,821 teacher-forced rows")

    # Verify and summarize the historical 160-row Pilot result separately.
    pilot_path = ROOT / "outputs/evaluation/e01_b/comparison" / PILOT_PAIR_RUN_ID
    pilot_summary: dict[str, Any] = {"status": "unavailable", "reason": "Pilot comparison artifacts not present"}
    if pilot_path.is_dir():
        pilot_manifest = _json(pilot_path / "manifest.json")
        pilot_summary_json = _json(pilot_path / "comparison_summary.json")
        pilot_cases = _unique(read_jsonl(pilot_path / "comparison_cases.jsonl"), "id", "Pilot comparison cases")
        if pilot_manifest.get("status") != "completed" or len(pilot_cases) != 160 or pilot_summary_json.get("scope") != "pilot-dev":
            raise ValueError("Pilot comparison identity/count mismatch")
        pilot_empty = {}
        for model in ("base", "adapter"):
            pilot_empty[model] = sum(all(not row[model]["profile"][dimension]["pred_nonempty"] for dimension in DIMENSIONS) for row in pilot_cases.values())
        pilot_summary = {
            "status": "verified_from_local_artifacts",
            "sample_count": len(pilot_cases),
            "all_three_normalized_profile_dimensions_empty": pilot_empty,
            "comparison_run_id": PILOT_PAIR_RUN_ID,
        }

    training_manifest = facts["training"]
    final_reload_check = _final_reload_check(training_root)

    per_class_support = {
        label: {"train_support": int(train_emotions[label]), "dev_support": int(dev_emotions[label])}
        for label in enums["emotion_label"]
    }
    adapter_per_class = model_outputs["adapter"]["emotion"]["per_class"]
    emotion_distribution = {
        label: {**per_class_support[label], **adapter_per_class[label]}
        for label in enums["emotion_label"]
    }
    per_gold_gain = {}
    for label in enums["emotion_label"]:
        base_row = model_outputs["base"]["emotion"]["per_gold_class_correctness"][label]
        adapter_row = model_outputs["adapter"]["emotion"]["per_gold_class_correctness"][label]
        per_gold_gain[label] = {
            "support": base_row["support"],
            "base_correct": base_row["correct"],
            "adapter_correct": adapter_row["correct"],
            "adapter_minus_base_correct": adapter_row["correct"] - base_row["correct"],
        }

    summary = {
        "analysis_scope": "read_only_existing_artifacts; no inference or rescoring",
        "evidence_status": "verified_from_local_artifacts",
        "sample_denominators": {"d03_train": 16312, "full_dev_generation_and_scoring": 1833, "sft_dev_teacher_forced_loss": 1821},
        "emotion_distribution_train_vs_dev": per_class_support,
        "emotion_adapter_per_class_with_train_dev_support": emotion_distribution,
        "emotion_per_gold_class_base_to_adapter": per_gold_gain,
        "adapter_full_profile_label_table_with_train_dev_support": _profile_label_table(enums, train_profiles, dev_profiles, facts["metric_by_model"]["adapter"]["profile"]),
        "gold_profile_baselines": {
            "three_dimensions_all_empty_gold_count": gold_all_empty,
            "three_dimensions_all_empty_gold_rate": gold_all_empty / len(dev_profiles),
            "full_adapter_all_three_dimensions_predicted_empty_count": predicted_all_empty,
            "gold_empty_and_adapter_predicted_empty_count": gold_empty_pred_empty,
            "gold_empty_but_adapter_predicted_nonempty_count": gold_all_empty - gold_empty_pred_empty,
            "gold_nonempty_but_adapter_predicted_empty_count": predicted_all_empty - gold_empty_pred_empty,
            "exact_match_on_gold_nonempty_profiles_count": exact_all_dims - gold_empty_pred_empty,
            "full_adapter_exact_match_count": exact_all_dims,
            "full_adapter_exact_match_rate": model_outputs["adapter"]["profile"]["all_three_dimensions_exact_match_rate"],
            "explanation": "Always-empty achieves the Gold all-empty count as exact matches; Exact Match misses partial label recovery and penalizes any extra label, so read alongside Micro/Macro-F1 and per-dimension recall.",
        },
        "base": model_outputs["base"],
        "adapter": model_outputs["adapter"],
        "pilot_160_row_comparison": pilot_summary,
        "training": {
            "run_id": training_manifest["run_id"],
            "status": training_manifest["status"],
            "stage": training_manifest["stage"],
            "manifest_sha256": sha256_file(facts["training_manifest_path"]),
            "training_evidence_sha256": sha256_file(facts["training_evidence_path"]),
            "config_sha256": training_manifest.get("config_sha256"),
            "preflight_sha256": training_manifest.get("preflight_sha256"),
            "training_code_commit_recorded": False,
            "training_code_commit_limitation": "Full B Run Manifest records config/preflight/data/model identities but no source Git commit or source-file digest set.",
            "base_weight_sha256": training_manifest.get("base_weight_sha256"),
            "adapter_identity": {
                key: training_manifest.get("adapter_identity", {}).get(key)
                for key in ("identity_status", "weight_sha256", "config_sha256", "config_summary", "files")
            },
            "train_rows": training_manifest.get("train_rows"),
            "optimizer_steps": training_manifest.get("optimizer_steps"),
            "training_seconds": training_manifest.get("training_seconds"),
            "consumed_input_tokens": facts["evidence"].get("consumed_input_tokens"),
            "consumed_completion_tokens": facts["evidence"].get("consumed_completion_tokens"),
            "training_log": _train_log_summary(training_root, training_manifest, facts["evidence"]),
            "checkpoints": [
                {"name": path.name, "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file())}
                for path in sorted((training_root / "trainer").glob("checkpoint-*")) if path.is_dir()
            ],
            "final_reload_check": final_reload_check,
            "reload_check_history_in_manifest": training_manifest.get("reload_check_history", []),
            "cuda": training_manifest.get("cuda"),
            "model_facts": {key: value for key, value in training_manifest.get("model_facts", {}).items() if key not in {"trainable_parameters"}},
        },
        "identity": {
            "comparison_status": facts["comparison_manifest"].get("status"),
            "comparison_manifest_sha256": sha256_file(facts["comparison_manifest_path"]),
            "comparison_summary_sha256": sha256_file(facts["comparison_summary_path"]),
            "dataset_manifest_sha256": sha256_file(facts["dataset_manifest_path"]),
            "d03_manifest_sha256": sha256_file(facts["d03_manifest_path"]),
            "d03_train_sha256": sha256_file(ROOT / _fact(facts["d03_manifest"], "data/processed/v1/train.jsonl")["path"]),
            "d03_dev_sha256": sha256_file(ROOT / _fact(facts["d03_manifest"], "data/processed/v1/dev.jsonl")["path"]),
            "sft_train_sha256": sha256_file(facts["sft_train_path"]),
            "sft_dev_sha256": sha256_file(facts["sft_dev_path"]),
            "dev_ids_sha256": facts["comparison_summary"]["dataset"]["ids_sha256"],
            "references_sha256": facts["comparison_summary"]["dataset"]["references.jsonl_sha256"],
            "input_sha256": facts["comparison_summary"]["dataset"]["inference_input.jsonl_sha256"],
            "base_weight_sha256": facts["models"]["base"]["base_weight_sha256"],
            "adapter_weight_sha256": facts["models"]["adapter"]["adapter_weight_sha256"],
            "model_revision_status": facts["models"]["adapter"]["revision_status"],
            "base_run_id": BASE_RUN_ID,
            "adapter_run_id": ADAPTER_RUN_ID,
            "training_run_id": TRAIN_RUN_ID,
            "base_scorer_code_commit": facts["models"]["base"]["scorer_code_commit"],
            "adapter_scorer_code_commit": facts["models"]["adapter"]["scorer_code_commit"],
            "training_code_commit_recorded": False,
            "diagnostic_code_commit_at_execution": _git_commit(),
        },
        "paired_statistics_from_existing_comparison_summary": facts["comparison_summary"]["paired"],
        "non_run_and_limitations": {
            "official_performance": "not_run",
            "bert_score": "not_run",
            "vllm_equivalence": "not_verified",
            "docker_offline_image": "not_verified",
            "manual_response_review": "not_run",
            "per_sample_bleu_rouge_relationship": "not_computed; existing scorer stores aggregate-only response metrics and this task forbids rescoring",
            "model_hub_revision": "unverified_revision; local weight file hashes were observed",
        },
    }
    inputs = {
        "training_run_root": str(training_root),
        "training_manifest_sha256": summary["training"]["manifest_sha256"],
        "training_evidence_sha256": summary["training"]["training_evidence_sha256"],
        "comparison_run_root": str(comparison_root),
        "comparison_manifest_sha256": summary["identity"]["comparison_manifest_sha256"],
        "comparison_summary_sha256": summary["identity"]["comparison_summary_sha256"],
        "dataset_manifest_sha256": summary["identity"]["dataset_manifest_sha256"],
        "d03_manifest_sha256": summary["identity"]["d03_manifest_sha256"],
        "base_metrics_sha256": facts["models"]["base"]["metrics_sha256"],
        "adapter_metrics_sha256": facts["models"]["adapter"]["metrics_sha256"],
        "comparison_cases_sha256": sha256_file(comparison_root / "comparison_cases.jsonl"),
        "d03_train_sha256": summary["identity"]["d03_train_sha256"],
        "d03_dev_sha256": summary["identity"]["d03_dev_sha256"],
    }
    return summary, inputs


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_outputs(output_dir: Path, summary: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    diagnostics_root = (ROOT / "outputs/evaluation/e01_b/diagnostics").resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(diagnostics_root):
        raise ValueError(f"diagnostic outputs must remain under {diagnostics_root}")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing diagnostics: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    summary_path = output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    manifest = {
        "status": "completed",
        "analysis_kind": "read_only_existing_artifacts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git_commit": _git_commit(),
        "inputs": inputs,
        "outputs": {"summary.json": {"sha256": sha256_file(summary_path), "size_bytes": summary_path.stat().st_size}},
        "limitations": "No model load, training, inference, model download, rescoring or raw generated text is performed/stored by this tool.",
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"status": "completed", "output_dir": str(output_dir), "summary_sha256": manifest["outputs"]["summary.json"]["sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--comparison-run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    training_root = _resolve(args.training_run_root)
    comparison_root = _resolve(args.comparison_run_root)
    output_dir = _resolve(args.output_dir)
    summary, inputs = analyze(training_root, comparison_root)
    print(json.dumps(write_outputs(output_dir, summary, inputs), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
