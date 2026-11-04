#!/usr/bin/env python3
"""Verify frozen E01-B/A1 artifacts and aggregate E03-A3 CPU diagnostics.

This CLI reads only the existing D0.3 Train/Dev, SFT Train, D0.4.2 Train/Dev
budget rows, and frozen evaluation artifacts. It never loads a model, performs
inference, or analyzes official Val records. Frozen sample-level inputs and
References are written only under Git-ignored ``outputs/diagnostics/e03_a3``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e01_b_diagnose
from src.diagnostics.e03_a3_profile import (
    DIMENSIONS,
    LENGTH_BUCKETS,
    clipping_breakdown,
    cluster_bootstrap_recall_gap,
    freeze_conversation_subset,
    length_breakdown,
    length_bucket,
    opportunity_space,
    retained_message_count,
    transition_counts,
    validate_history,
    validate_profile,
)
from src.evaluation import read_jsonl, schema_enums, sha256_file
from src.evaluation.metrics import profile_metrics


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("experiment") != "E03-A3":
        raise ValueError("A3 config must be an E03-A3 mapping")
    if not isinstance(value.get("inputs"), dict) or not isinstance(value.get("protocol"), dict):
        raise ValueError("A3 config is missing inputs or protocol")
    return value


def _file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def _assert_hash(path: Path, expected: str | None, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(f"{description} SHA-256 mismatch")
    return actual


def _unique(rows: list[dict[str, Any]], key: str, description: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{description} contains a missing/non-string {key}")
        if value in result:
            raise ValueError(f"{description} contains duplicate {key}")
        result[value] = row
    return result


def _official_data_enums(schema: dict[str, Any]) -> dict[str, list[str]]:
    definitions = schema.get("$defs", {})
    profile_properties = definitions.get("profile", {}).get("properties", {})
    emotion = definitions.get("emotion", {}).get("enum", [])
    result = {"emotion_label": list(emotion)}
    for dimension in DIMENSIONS:
        result[dimension] = list(profile_properties.get(dimension, {}).get("items", {}).get("enum", []))
    if any(not values for values in result.values()):
        raise ValueError("official Train schema does not define all fixed label enums")
    return result


def _candidate_budget_rows(path: Path, prompt_cap: int, sequence_cap: int) -> dict[str, dict[str, Any]]:
    """Read only candidate Train/Dev budget entries; Val entries are ignored."""
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") not in {"train", "dev"}:
                continue
            if row.get("max_prompt_tokens") != prompt_cap or row.get("max_train_seq_length") != sequence_cap:
                continue
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"D0.4.2 candidate row {line_number} has no sample_id")
            if sample_id in selected:
                raise ValueError("D0.4.2 candidate budget has duplicate sample IDs")
            selected[sample_id] = row
    return selected


def _metric_values(gold_rows: list[dict[str, set[str]]], pred_rows: list[dict[str, set[str]]], enums: dict[str, list[str]]) -> dict[str, Any]:
    return profile_metrics(
        [{dimension: sorted(row[dimension], key=enums[dimension].index) for dimension in DIMENSIONS} for row in gold_rows],
        [{dimension: sorted(row[dimension], key=enums[dimension].index) for dimension in DIMENSIONS} for row in pred_rows],
        enums,
    )


def _assert_metrics_match(actual: dict[str, Any], calculated: dict[str, Any], description: str) -> None:
    for key in (
        "overall_33_label_tp",
        "overall_33_label_fp",
        "overall_33_label_fn",
    ):
        if actual.get(key) != calculated.get(key):
            raise ValueError(f"{description} {key} differs from the frozen scorer result")
    if not math.isclose(
        float(actual.get("overall_33_label_micro_f1", -1)),
        float(calculated["overall_33_label_micro_f1"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{description} Micro-F1 differs from the frozen scorer result")


def _verify_a1(config_inputs: dict[str, Path], e01_summary: dict[str, Any], expected_eval_ids: set[str]) -> dict[str, Any]:
    comparison_root = config_inputs["a1_comparison_root"]
    eval_root = config_inputs["a1_eval_root"]
    manifest_path = comparison_root / "manifest.json"
    summary_path = comparison_root / "comparison_summary.json"
    cases_path = comparison_root / "comparison_cases.jsonl"
    manifest = _json(manifest_path)
    summary = _json(summary_path)
    if manifest.get("status") != "completed" or summary.get("status") != "completed" or summary.get("scope") != "full-dev":
        raise ValueError("E03-A1 paired comparison is not a completed Full Dev run")
    _assert_hash(summary_path, manifest.get("comparison_summary_sha256"), "E03-A1 comparison summary")
    if manifest.get("baseline_run_root") != str(config_inputs["e01_b_eval_root"]):
        if Path(str(manifest.get("baseline_run_root", ""))).resolve() != config_inputs["e01_b_eval_root"]:
            raise ValueError("E03-A1 comparison does not use the frozen E01-B evaluation Run")
    if Path(str(manifest.get("candidate_run_root", ""))).resolve() != eval_root:
        raise ValueError("E03-A1 comparison candidate Run path differs from its config")

    dataset = summary.get("dataset", {})
    e01_dataset = e01_summary.get("dataset", {})
    for key in ("sample_count", "ids_sha256", "inference_input.jsonl_sha256", "references.jsonl_sha256", "id_map.jsonl_sha256"):
        if dataset.get(key) != e01_dataset.get(key):
            raise ValueError(f"E03-A1 dataset identity differs from E01-B for {key}")
    if dataset.get("sample_count") != len(expected_eval_ids):
        raise ValueError("E03-A1 paired Dev has an unexpected denominator")
    a1_cases = _unique(read_jsonl(cases_path), "id", "E03-A1 comparison cases")
    if set(a1_cases) != expected_eval_ids:
        raise ValueError("E03-A1 paired comparison IDs differ from verified E01-B IDs")

    run_manifest_path = eval_root / "run_manifest.json"
    run_manifest = _json(run_manifest_path)
    if run_manifest.get("status") != "scored" or run_manifest.get("scope") != "full-dev":
        raise ValueError("E03-A1 candidate evaluation is not scored on Full Dev")
    for relative, expected in run_manifest.get("artifacts", {}).items():
        _assert_hash(eval_root / relative, expected, f"E03-A1 evaluation artifact {relative}")
    metric_path = eval_root / "scored/metrics.json"
    metrics = _json(metric_path)
    if metrics.get("sample_count") != len(expected_eval_ids):
        raise ValueError("E03-A1 candidate metric denominator differs")
    return {
        "manifest": manifest,
        "summary": summary,
        "cases": a1_cases,
        "run_manifest": run_manifest,
        "metrics": metrics,
        "paths": {"comparison_manifest": manifest_path, "comparison_summary": summary_path, "comparison_cases": cases_path, "run_manifest": run_manifest_path, "metrics": metric_path},
    }


def _load_profile_rows(path: Path, expected_ids: set[str], enums: dict[str, list[str]], description: str) -> dict[str, dict[str, set[str]]]:
    rows = _unique(read_jsonl(path), "id", description)
    if set(rows) != expected_ids:
        raise ValueError(f"{description} IDs differ from the verified Full Dev IDs")
    return {sample_id: validate_profile(row.get("user_profile"), enums, field=description) for sample_id, row in rows.items()}


def _assert_sample_profile_case(case: dict[str, Any], gold: dict[str, set[str]], pred: dict[str, set[str]], description: str) -> None:
    for dimension in DIMENSIONS:
        expected = {
            "tp": len(gold[dimension] & pred[dimension]),
            "fp": len(pred[dimension] - gold[dimension]),
            "fn": len(gold[dimension] - pred[dimension]),
        }
        actual = case["profile"][dimension]
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{description} per-ID {dimension} TP/FP/FN differs from mapped Gold and Submission")


def _build_diagnosis(config_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    input_paths = {name: _resolve(value) for name, value in config["inputs"].items()}
    protocol = config["protocol"]
    prompt_cap = int(protocol["max_prompt_tokens"])
    sequence_cap = int(protocol["max_train_sequence_tokens"])

    # The existing E01-B verifier checks the original D0.3/Dev mapping, model,
    # adapter, output files, scorer totals, and per-ID aggregate conservation.
    facts = e01_b_diagnose._validate_inputs(input_paths["training_run_root"], input_paths["e01_b_comparison_root"])
    e01_summary = facts["comparison_summary"]
    inference_config_path = input_paths["e01_b_eval_root"] / "inference_config.json"
    inference_config = _json(inference_config_path)
    official_data_schema = _json(input_paths["official_data_schema"])
    output_schema = _json(input_paths["output_schema"])
    official_enums = _official_data_enums(official_data_schema)
    enums = schema_enums(output_schema)
    if official_enums != enums:
        raise ValueError("official Train schema and output schema label enums/order differ")
    d03_manifest = _json(input_paths["d03_manifest"])
    sft_manifest = _json(input_paths["sft_manifest"])
    d042_manifest = _json(input_paths["d042_manifest"])
    budget_path = input_paths["d042_budget_rows"]

    generation_protocol = inference_config.get("generation", {})
    context_protocol = inference_config.get("context", {})
    if context_protocol.get("max_prompt_tokens") != prompt_cap:
        raise ValueError("E01-B inference P differs from the locked A3 prompt budget")
    if generation_protocol.get("max_new_tokens") != int(protocol["max_new_tokens"]):
        raise ValueError("E01-B inference G differs from the locked A3 generation budget")
    if generation_protocol.get("do_sample") is not False:
        raise ValueError("E01-B inference is not the locked greedy decoding protocol")

    d03_sha = sha256_file(input_paths["d03_manifest"])
    if sft_manifest.get("d03_manifest_sha256") != d03_sha:
        raise ValueError("SFT Manifest references a different D0.3 Manifest")
    train_schema_sha = sha256_file(input_paths["official_data_schema"])
    if sft_manifest.get("inputs", {}).get("train_schema", {}).get("sha256") != train_schema_sha:
        raise ValueError("SFT Manifest references a different official Train schema")
    if d042_manifest.get("input_hashes_unchanged_after_run", {}).get("train_schema") != train_schema_sha:
        raise ValueError("D0.4.2 Manifest references a different official Train schema")
    d042_sha = sha256_file(input_paths["d042_manifest"])
    if sft_manifest.get("inputs", {}).get("d042_manifest", {}).get("sha256") != d042_sha:
        raise ValueError("SFT Manifest references a different D0.4.2 Manifest")
    expected_budget_sha = sft_manifest.get("inputs", {}).get("d042_budget_rows", {}).get("sha256")
    _assert_hash(budget_path, expected_budget_sha, "D0.4.2 budget ledger")
    if d042_manifest.get("validation", {}).get("exit_code") != 0:
        raise ValueError("D0.4.2 budget audit validation did not pass")
    if sft_manifest.get("validation", {}).get("all_d03_rows_have_one_d042_match") is not True:
        raise ValueError("D0.4.3 SFT Manifest does not confirm one D0.4.2 match per D0.3 row")
    if sft_manifest.get("validation", {}).get("all_splits_conserve_counts") is not True:
        raise ValueError("D0.4.3 SFT Manifest count conservation did not pass")

    sft_train_path = input_paths["sft_train"]
    sft_facts = sft_manifest.get("outputs", {}).get("files", {}).get("train.jsonl", {})
    _assert_hash(sft_train_path, sft_facts.get("sha256"), "SFT Train data")
    train_manifest = facts["training"]
    if train_manifest.get("train_sha256") != sft_facts.get("sha256"):
        raise ValueError("Full B training Run did not use the current SFT Train file")

    budget_by_id = _candidate_budget_rows(budget_path, prompt_cap, sequence_cap)
    train_budget = {sample_id: row for sample_id, row in budget_by_id.items() if row.get("split") == "train"}
    dev_budget = {sample_id: row for sample_id, row in budget_by_id.items() if row.get("split") == "dev"}
    if len(train_budget) != int(protocol["expected_d03_train_rows"]) or len(dev_budget) != int(protocol["expected_dev_rows"]):
        raise ValueError("D0.4.2 candidate Train/Dev row counts differ from the locked data version")

    sft_train = _unique(read_jsonl(sft_train_path), "sample_id", "Full B SFT Train")
    if len(sft_train) != int(protocol["expected_sft_train_rows"]):
        raise ValueError("Full B SFT Train row count differs from the locked data version")
    d03_train = facts["d03_train"]
    if not set(sft_train) <= set(d03_train) or not set(sft_train) <= set(train_budget):
        raise ValueError("SFT Train is not a subset of verified D0.3 Train and D0.4.2 candidate rows")

    train_rows: list[dict[str, Any]] = []
    for sample_id, sft_row in sft_train.items():
        source = d03_train[sample_id]
        budget = train_budget[sample_id]
        history_length = validate_history(source.get("history"))
        removed = budget.get("removed_history_message_count")
        retained = retained_message_count(history_length, removed)
        if budget.get("original_history_message_count") != history_length:
            raise ValueError("D0.4.2 original history length differs from D0.3 Train")
        if budget.get("retained_history_message_count") != retained:
            raise ValueError("D0.4.2 retained history length is inconsistent")
        if sft_row.get("conversation_id") != source.get("conversation_id") or budget.get("conversation_id") != source.get("conversation_id"):
            raise ValueError("SFT/D0.4.2 conversation identity differs from D0.3 Train")
        if sft_row.get("source") != source.get("source") or budget.get("source") != source.get("source"):
            raise ValueError("SFT/D0.4.2 source position differs from D0.3 Train")
        if sft_row.get("prompt_length") != budget.get("selected_prompt_tokens"):
            raise ValueError("SFT prompt length differs from D0.4.2 selected prompt tokens")
        if len(sft_row.get("input_ids", [])) != budget.get("selected_full_sequence_tokens"):
            raise ValueError("SFT token sequence length differs from D0.4.2 selected sequence")
        gold = validate_profile(source.get("target", {}).get("user_profile"), enums, field="D0.3 Train Gold")
        if budget.get("profile_all_empty") != all(not gold[dimension] for dimension in DIMENSIONS):
            raise ValueError("D0.4.2 Train Gold empty flag differs from D0.3 target")
        turn_index = source.get("source", {}).get("target_turn_index")
        train_rows.append({
            "conversation_id": source["conversation_id"],
            "target_turn_index": turn_index,
            "history_length": history_length,
            "retained_history_length": retained,
            "removed_history_messages": removed,
            "prompt_tokens": budget["selected_prompt_tokens"],
            "gold": gold,
        })

    # Validate A1 against the exact E01-B Dev data and model-independent ID set.
    e01_eval_root = input_paths["e01_b_eval_root"]
    eval_ids = set(facts["references"])
    a1 = _verify_a1(input_paths, e01_summary, eval_ids)
    id_map = facts["id_map"]
    d03_dev = facts["d03_dev"]
    inference_path = facts["dataset_root"] / e01_summary["dataset"]["input"]
    inference_rows = _unique(read_jsonl(inference_path), "id", "Full Dev inference inputs")
    if set(inference_rows) != eval_ids:
        raise ValueError("Full Dev inference input IDs differ from verified references")

    adapter_submission = _load_profile_rows(e01_eval_root / "runner/submission.jsonl", eval_ids, enums, "E01-B Adapter Submission")
    a1_submission = _load_profile_rows(input_paths["a1_eval_root"] / "runner/submission.jsonl", eval_ids, enums, "E03-A1 Submission")
    raw_rows = _unique(read_jsonl(e01_eval_root / "raw_predictions.jsonl"), "id", "E01-B Adapter raw predictions")
    if set(raw_rows) != eval_ids:
        raise ValueError("E01-B raw output IDs differ from verified references")

    dev_rows: list[dict[str, Any]] = []
    eval_to_sample: dict[str, str] = {}
    raw_status_counts: Counter[str] = Counter()
    for eval_id in sorted(eval_ids):
        map_row = id_map[eval_id]
        sample_id = map_row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in d03_dev or sample_id not in dev_budget:
            raise ValueError("Full Dev ID Map points outside verified D0.3 Dev/D0.4.2 Dev")
        source = d03_dev[sample_id]
        budget = dev_budget[sample_id]
        raw = raw_rows[eval_id]
        if map_row.get("conversation_id") and map_row["conversation_id"] != source.get("conversation_id"):
            raise ValueError("Full Dev ID Map conversation identity differs from D0.3 Dev")
        if source.get("conversation_id") != budget.get("conversation_id"):
            raise ValueError("D0.4.2 conversation identity differs from D0.3 Dev")
        history_length = validate_history(source.get("history"))
        retained = retained_message_count(history_length, budget.get("removed_history_message_count"))
        if budget.get("original_history_message_count") != history_length or budget.get("retained_history_message_count") != retained:
            raise ValueError("D0.4.2 Dev history counts differ from the verified D0.3 source")
        input_row = inference_rows[eval_id]
        if input_row.get("history") != source.get("history"):
            raise ValueError("Full Dev inference history differs from the mapped D0.3 row")
        if budget.get("profile_all_empty") != all(
            not source["target"]["user_profile"][dimension] for dimension in DIMENSIONS
        ):
            raise ValueError("D0.4.2 Dev Gold empty flag differs from D0.3 target")
        if raw.get("removed_history_messages") != budget.get("removed_history_message_count"):
            raise ValueError("E01-B runtime T1 removal count differs from the D0.4.2 selection")
        if raw.get("prompt_tokens") != budget.get("selected_prompt_tokens"):
            raise ValueError("E01-B runtime prompt token count differs from D0.4.2 selection")
        gold = validate_profile(source.get("target", {}).get("user_profile"), enums, field="D0.3 Dev Gold")
        e01_case = facts["cases"][eval_id]["adapter"]
        _assert_sample_profile_case(e01_case, gold, adapter_submission[eval_id], "E01-B Adapter")
        _assert_sample_profile_case(a1["cases"][eval_id]["e01_b_baseline"], gold, adapter_submission[eval_id], "E03-A1 paired E01-B baseline")
        _assert_sample_profile_case(a1["cases"][eval_id]["e03_a1_candidate"], gold, a1_submission[eval_id], "E03-A1 candidate")
        raw_status_counts[str(raw.get("status", "missing"))] += 1
        turn_index = source.get("source", {}).get("target_turn_index")
        dev_rows.append({
            "eval_id": eval_id,
            "sample_id": sample_id,
            "history": source["history"],
            "conversation_id": source["conversation_id"],
            "target_turn_index": turn_index,
            "history_length": history_length,
            "retained_history_length": retained,
            "removed_history_messages": budget["removed_history_message_count"],
            "prompt_tokens": budget["selected_prompt_tokens"],
            "gold": gold,
            "pred": adapter_submission[eval_id],
            "a1_pred": a1_submission[eval_id],
            "raw_status": str(raw.get("status", "missing")),
        })
        eval_to_sample[eval_id] = sample_id

    # Recalculate all three frozen profile totals from normalized submissions.
    ordered_ids = sorted(eval_ids)
    dev_gold = [row["gold"] for row in dev_rows]
    e01_predictions = [row["pred"] for row in dev_rows]
    a1_predictions = [row["a1_pred"] for row in dev_rows]
    e01_metrics = _metric_values(dev_gold, e01_predictions, enums)
    a1_metrics = _metric_values(dev_gold, a1_predictions, enums)
    stored_e01_profile = facts["metric_by_model"]["adapter"]["profile"]
    stored_a1_profile = a1["metrics"]["profile"]
    _assert_metrics_match(stored_e01_profile, e01_metrics, "E01-B Adapter")
    _assert_metrics_match(stored_a1_profile, a1_metrics, "E03-A1 candidate")
    stored_a1_baseline = a1["summary"]["metrics"]["e01_b_baseline"]["profile"]
    _assert_metrics_match(stored_a1_baseline, e01_metrics, "E03-A1 paired E01-B baseline")

    raw_bucket = lambda row: length_bucket(row["history_length"])
    retained_bucket = lambda row: length_bucket(row["retained_history_length"])
    for row in dev_rows:
        row["raw_length_bucket"] = raw_bucket(row)
        row["retained_length_bucket"] = retained_bucket(row)
    transition_rows = [
        {**row, "before": row["pred"], "after": row["a1_pred"]}
        for row in dev_rows
    ]
    transitions_raw = transition_counts(transition_rows, enums, before_key="before", after_key="after", group_field="raw_length_bucket")
    transitions_t1 = transition_counts(transition_rows, enums, before_key="before", after_key="after", group_field="retained_length_bucket")

    raw_breakdown = length_breakdown(train_rows, dev_rows, enums, length_field="history_length")
    t1_breakdown = length_breakdown(train_rows, dev_rows, enums, length_field="retained_history_length")
    clipping = clipping_breakdown(train_rows, dev_rows, enums)
    opportunities = {
        "raw_history": {
            bucket: opportunity_space(
                total_tp=e01_metrics["overall_33_label_tp"],
                total_fp=e01_metrics["overall_33_label_fp"],
                total_fn=e01_metrics["overall_33_label_fn"],
                group_fn=raw_breakdown[bucket]["dev"]["profile_fn"],
            )
            for _, _, bucket in LENGTH_BUCKETS
        },
        "t1_retained_history": {
            bucket: opportunity_space(
                total_tp=e01_metrics["overall_33_label_tp"],
                total_fp=e01_metrics["overall_33_label_fp"],
                total_fn=e01_metrics["overall_33_label_fn"],
                group_fn=t1_breakdown[bucket]["dev"]["profile_fn"],
            )
            for _, _, bucket in LENGTH_BUCKETS
        },
    }
    short_history_ci = cluster_bootstrap_recall_gap(
        dev_rows,
        repetitions=int(config.get("bootstrap", {}).get("repetitions", 1000)),
        seed=int(config.get("bootstrap", {}).get("seed", 42)),
    )
    frozen_features, frozen_eval_summary = freeze_conversation_subset(
        [
            {
                "eval_id": row["eval_id"],
                "conversation_id": row["conversation_id"],
                "history_length": row["history_length"],
                "gold": row["gold"],
            }
            for row in dev_rows
        ],
        **{
            key: int(value)
            for key, value in config.get("frozen_eval", {}).items()
        },
    )
    frozen_ids = {row["eval_id"] for row in frozen_features}
    frozen_eval_rows = [row for row in dev_rows if row["eval_id"] in frozen_ids]
    if len(frozen_eval_rows) != len(frozen_features):
        raise ValueError("frozen evaluation IDs do not map back to the verified Dev rows")
    frozen_eval_summary["e01_b_baseline_profile_metrics"] = _metric_values(
        [row["gold"] for row in frozen_eval_rows],
        [row["pred"] for row in frozen_eval_rows],
        enums,
    )
    frozen_eval_summary["e01_b_baseline_raw_status_counts"] = dict(sorted(Counter(row["raw_status"] for row in frozen_eval_rows).items()))

    label_table = e01_metrics["per_dimension"]
    top_fn_labels = []
    for dimension in DIMENSIONS:
        for label, metric in label_table[dimension]["per_label"].items():
            train_support = sum(row["gold"][dimension].__contains__(label) for row in train_rows)
            top_fn_labels.append({
                "dimension": dimension,
                "label": label,
                "train_support": train_support,
                "dev_gold_support": metric["support"],
                "dev_predicted_count": metric["predicted_count"],
                "tp": metric["tp"],
                "fp": metric["fp"],
                "fn": metric["fn"],
                "recall": metric["recall"],
                "precision": metric["precision"],
                "f1": metric["f1"],
            })
    top_fn_labels.sort(key=lambda row: (-row["fn"], row["dimension"], row["label"]))

    identity_paths = {
        "d03_manifest": input_paths["d03_manifest"],
        "normalized_train": input_paths["normalized_train"],
        "normalized_dev": input_paths["normalized_dev"],
        "sft_manifest": input_paths["sft_manifest"],
        "sft_train": input_paths["sft_train"],
        "d042_manifest": input_paths["d042_manifest"],
        "d042_budget_rows": budget_path,
        "full_b_training_manifest": input_paths["training_run_root"] / "manifest.json",
        "e01_b_comparison_manifest": input_paths["e01_b_comparison_root"] / "manifest.json",
        "e01_b_comparison_summary": input_paths["e01_b_comparison_root"] / "comparison_summary.json",
        "e01_b_comparison_cases": input_paths["e01_b_comparison_root"] / "comparison_cases.jsonl",
        "e01_b_adapter_raw": e01_eval_root / "raw_predictions.jsonl",
        "e01_b_adapter_submission": e01_eval_root / "runner/submission.jsonl",
        "e01_b_inference_config": inference_config_path,
        "a1_comparison_manifest": a1["paths"]["comparison_manifest"],
        "a1_comparison_summary": a1["paths"]["comparison_summary"],
        "a1_comparison_cases": a1["paths"]["comparison_cases"],
        "a1_candidate_manifest": a1["paths"]["run_manifest"],
        "a1_candidate_metrics": a1["paths"]["metrics"],
        "a1_candidate_submission": input_paths["a1_eval_root"] / "runner/submission.jsonl",
        "official_data_schema": input_paths["official_data_schema"],
        "output_schema": input_paths["output_schema"],
        "prompt": input_paths["prompt"],
    }
    source_hashes = _file_hashes(identity_paths)
    if sft_manifest.get("p0_sha256") != source_hashes["prompt"]:
        raise ValueError("SFT Manifest P0 SHA differs from the current locked P0 file")

    actual_ids_hash = e01_summary.get("dataset", {}).get("ids_sha256")
    if actual_ids_hash != "739a88b6e47e053a1a358c1ac0f119de474522d0fa471bdbe4a8bd86bac1f322":
        raise ValueError("Full Dev ID digest differs from the locked 1,833-row E01-B evaluation")
    identity = {
        "dataset": {
            "d03_manifest_sha256": d03_sha,
            "d03_train_sha256": source_hashes["normalized_train"],
            "d03_dev_sha256": source_hashes["normalized_dev"],
            "sft_manifest_sha256": source_hashes["sft_manifest"],
            "sft_train_sha256": source_hashes["sft_train"],
            "d042_manifest_sha256": d042_sha,
            "d042_budget_rows_sha256": source_hashes["d042_budget_rows"],
            "d03_train_count": len(d03_train),
            "sft_train_count": len(train_rows),
            "dev_count": len(dev_rows),
            "unique_dev_conversations": len({row["conversation_id"] for row in dev_rows}),
            "eval_ids_sha256": actual_ids_hash,
            "official_val_rows_used": 0,
        },
        "model": {
            "full_b_training_run_id": train_manifest.get("run_id"),
            "full_b_training_manifest_sha256": source_hashes["full_b_training_manifest"],
            "base_weight_sha256": train_manifest.get("base_weight_sha256"),
            "adapter_weight_sha256": train_manifest.get("adapter_identity", {}).get("weight_sha256"),
            "adapter_file_sha256": train_manifest.get("adapter_identity", {}).get("files", [{}])[0].get("sha256"),
            "model_revision_status": facts["metric_by_model"]["adapter"].get("model_identity_verification", {}).get("revision_status"),
            "scorer_code_commit": facts["metric_by_model"]["adapter"].get("scorer_code_commit"),
            "prompt_sha256": source_hashes["prompt"],
            "inference_config_sha256": source_hashes["e01_b_inference_config"],
            "official_data_schema_sha256": source_hashes["official_data_schema"],
            "output_schema_sha256": source_hashes["output_schema"],
            "P_max_prompt_tokens": prompt_cap,
            "G_max_new_tokens": generation_protocol["max_new_tokens"],
            "do_sample": generation_protocol["do_sample"],
        },
        "evaluation": {
            "e01_b_comparison_summary_sha256": source_hashes["e01_b_comparison_summary"],
            "e01_b_comparison_cases_sha256": source_hashes["e01_b_comparison_cases"],
            "a1_comparison_summary_sha256": source_hashes["a1_comparison_summary"],
            "a1_comparison_cases_sha256": source_hashes["a1_comparison_cases"],
            "same_id_set_verified": True,
            "history_equals_mapped_d03_dev": True,
            "e01_b_runtime_t1_matches_d042": True,
            "a1_evaluation_status": a1["run_manifest"].get("status"),
        },
        "source_hashes": source_hashes,
    }
    summary = {
        "status": "completed",
        "analysis_kind": "read_only_existing_artifacts",
        "scope": "D0.3 SFT Train + Full Dev; E01-B and E03-A1 paired comparison",
        "identity": identity,
        "global": {
            "train": e01_b_diagnose_summary(train_rows, enums, predictions=False),
            "e01_b_full_dev": {
                "recomputed": e01_metrics,
                "stored_metrics_match": True,
                "raw_status_counts": dict(sorted(raw_status_counts.items())),
            },
            "e03_a1_full_dev": {
                "recomputed": a1_metrics,
                "stored_metrics_match": True,
            },
        },
        "by_original_history_length": raw_breakdown,
        "by_t1_retained_history_length": t1_breakdown,
        "by_t1_clipping": clipping,
        "e01_b_to_a1_transitions": {
            "by_original_history_length": transitions_raw,
            "by_t1_retained_history_length": transitions_t1,
        },
        "opportunity_space": opportunities,
        "short_vs_long_cluster_bootstrap": short_history_ci,
        "frozen_eval_selection": frozen_eval_summary,
        "full_dev_label_error_ranking": top_fn_labels,
        "limitations": [
            "Full Dev has been used for prior development and is a development_proxy, not an untouched test set.",
            "Length associations are descriptive; history length is confounded with evidence density, target turns, labels, and T1 clipping.",
            "Gold labels are supervision targets and do not establish ground-truth personal traits.",
            "Official Val rows were excluded and were not read into the analysis aggregates.",
            "No training, model load, inference, or generated predictions occurred.",
        ],
    }
    frozen_artifacts = {
        "frozen_eval_input.jsonl": [
            {"id": row["eval_id"], "history": row["history"]}
            for row in frozen_eval_rows
        ],
        "frozen_eval_references.jsonl": [
            {
                "eval_id": row["eval_id"],
                "sample_id": row["sample_id"],
                "target": facts["d03_dev"][row["sample_id"]]["target"],
            }
            for row in frozen_eval_rows
        ],
        "frozen_eval_id_map.jsonl": [
            {
                "eval_id": row["eval_id"],
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
            }
            for row in frozen_eval_rows
        ],
    }
    return summary, source_hashes, {"rows": frozen_artifacts, "summary": frozen_eval_summary}


def e01_b_diagnose_summary(rows: list[dict[str, Any]], enums: dict[str, list[str]], *, predictions: bool) -> dict[str, Any]:
    from src.diagnostics.e03_a3_profile import _profile_counts

    return _profile_counts(rows, enums, pred_key="pred" if predictions else None)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_outputs(
    output_root: Path,
    run_id: str,
    config_path: Path,
    summary: dict[str, Any],
    source_hashes: dict[str, str],
    frozen_eval: dict[str, Any],
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", run_id):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")
    diagnostics_root = (ROOT / "outputs/diagnostics/e03_a3").resolve()
    output_root = output_root.resolve()
    if output_root != diagnostics_root:
        raise ValueError("A3 output root must be outputs/diagnostics/e03_a3")
    run_dir = output_root / run_id
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing A3 Run: {run_id}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    summary_path = run_dir / "aggregate.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    output_records: dict[str, Any] = {
        "aggregate.json": {
            "sha256": sha256_file(summary_path),
            "size_bytes": summary_path.stat().st_size,
            "contains_sample_ids_or_text": False,
        },
    }
    for filename, rows in frozen_eval["rows"].items():
        path = run_dir / filename
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
                handle.write("\n")
        output_records[filename] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "record_count": len(rows),
            "contains_sample_ids_or_text": True,
            "contains_gold": filename == "frozen_eval_references.jsonl",
        }
    selection_path = run_dir / "frozen_eval_selection.json"
    with selection_path.open("x", encoding="utf-8") as handle:
        json.dump(frozen_eval["summary"], handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    output_records[selection_path.name] = {
        "sha256": sha256_file(selection_path),
        "size_bytes": selection_path.stat().st_size,
        "contains_sample_ids_or_text": False,
    }
    manifest = {
        "status": "completed",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "read_only_existing_artifacts",
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "working_tree_dirty": bool(_git_value("status", "--porcelain")),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "outputs": output_records,
        "frozen_eval_selection": frozen_eval["summary"],
        "official_val_rows_used": 0,
        "training_started": False,
        "model_loaded": False,
        "inference_started": False,
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {
        "status": "completed",
        "run_id": run_id,
        "output_dir": str(run_dir.relative_to(ROOT)),
        "aggregate_sha256": manifest["outputs"]["aggregate.json"]["sha256"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/diagnostics/e03_a3.yaml")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    output_root = _resolve(config.get("output_root", "outputs/diagnostics/e03_a3"))
    summary, source_hashes, frozen_eval = _build_diagnosis(config_path, config)
    result = write_outputs(output_root, args.run_id, config_path, summary, source_hashes, frozen_eval)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
