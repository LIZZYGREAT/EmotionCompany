"""Reconcile historical E01-B profile predictions against Raw prefixes (CPU only)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    load_submission_schema,
    read_jsonl,
    schema_enums,
    sha256_file,
    strict_json_loads,
)
from src.evaluation.profile_fast_eval import select_frozen_id_samples
from src.evaluation.profile_prefix import parse_profile_prefix
from src.evaluation.profile_reconcile import (
    PROFILE_DIMENSIONS,
    classify_historical_difference,
    compare_profile_lists,
    empty_profile,
    index_profile_prediction_rows,
    profile_difference_dimensions,
    validate_profile,
)
from src.evaluation.source_integrity import (
    audit_prediction_point_sources,
    canonical_sha256,
    unique_index,
)


DEFAULT_CANDIDATE_CONFIG = ROOT / "configs/evaluation/e03_profile_candidate.yaml"
DEFAULT_A3_CONFIG = ROOT / "configs/training/e03_a3_short.yaml"
DEFAULT_SCHEMA_SHA256 = "91c42d56ae433212aab7905eabdbb96cf8463c2f887dc46b3f31647ab3a5d591"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
    temporary.replace(path)


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs from its frozen manifest")
    return actual


def _validate_profile_rows(
    rows: dict[str, dict[str, Any]], ids: list[str], enums: dict[str, list[str]], label: str
) -> list[dict[str, list[str]]]:
    profiles: list[dict[str, list[str]]] = []
    for sample_id in ids:
        profile = rows[sample_id].get("user_profile")
        profiles.append(validate_profile(profile, enums, label=f"{label} {sample_id}"))
    return profiles


def _load_sources(candidate_config_path: Path, a3_config_path: Path) -> dict[str, Any]:
    candidate = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
    a3_config = yaml.safe_load(a3_config_path.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict) or not isinstance(a3_config, dict):
        raise ValueError("A4 candidate and A3 lock configs must be YAML objects")
    selection = candidate.get("selection", {})
    sources = candidate.get("sources", {})
    if selection.get("mode") != "frozen_ids":
        raise ValueError("A4 reconciliation requires the frozen_ids candidate config")

    input_path = _resolve(sources["input_path"])
    id_map_path = _resolve(sources["id_map_path"])
    normalized_path = _resolve(sources["normalized_dev_path"])
    references_path = _resolve(sources["references_path"])
    frozen_ids_path = _resolve(selection["frozen_ids_path"])
    diagnosis_root = frozen_ids_path.parent
    inference_config_path = _resolve(sources["inference_config_path"])
    eval_root = inference_config_path.parent
    training = a3_config["identity"]
    evaluation = a3_config["evaluation"]

    diagnosis_manifest_path = diagnosis_root / "manifest.json"
    selection_path = diagnosis_root / "frozen_eval_selection.json"
    diagnosis_manifest = _json_object(diagnosis_manifest_path)
    selection_manifest = _json_object(selection_path)
    if diagnosis_manifest.get("status") != "completed" or diagnosis_manifest.get("run_id") != diagnosis_root.name:
        raise ValueError("frozen A3 diagnosis Run is incomplete or misidentified")
    _require_hash(
        selection_path,
        training["expected_frozen_eval_selection_sha256"],
        "frozen evaluation selection",
    )
    expected_count = int(selection["expected_count"])
    expected_conversations = int(selection["expected_conversation_count"])
    expected_ids_sha256 = selection["expected_ids_sha256"]
    if (
        selection_manifest.get("status") != "frozen"
        or selection_manifest.get("selected_points") != expected_count
        or selection_manifest.get("selected_conversations") != expected_conversations
        or selection_manifest.get("ids_sha256") != expected_ids_sha256
        or selection_manifest.get("selection_uses_predictions") is not False
        or selection_manifest.get("conversation_groups_split") is not False
    ):
        raise ValueError("frozen A3 selection differs from the A4 candidate lock")

    diagnosis_files = {
        "frozen_eval_input.jsonl": frozen_ids_path,
        "frozen_eval_references.jsonl": references_path,
        "frozen_eval_id_map.jsonl": diagnosis_root / "frozen_eval_id_map.jsonl",
    }
    for name, path in diagnosis_files.items():
        expected = diagnosis_manifest.get("outputs", {}).get(name, {}).get("sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"A3 diagnosis Manifest does not bind {name}")

    input_rows = read_jsonl(input_path)
    all_input_index = unique_index(input_rows, "id", "full target-free input")
    original_id_map_rows = read_jsonl(id_map_path)
    normalized_rows = read_jsonl(normalized_path)
    normalized_index = unique_index(normalized_rows, "sample_id", "normalized Dev")
    enriched_id_map: list[dict[str, Any]] = []
    for row in original_id_map_rows:
        sample_id = row.get("sample_id")
        normalized = normalized_index.get(sample_id) if isinstance(sample_id, str) else None
        if normalized is None:
            raise ValueError("Dev ID map contains a sample missing from normalized Dev")
        conversation_id = normalized.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("normalized Dev row has no conversation_id")
        enriched_id_map.append({**row, "conversation_id": conversation_id})

    frozen_rows = read_jsonl(frozen_ids_path)
    selected_rows = select_frozen_id_samples(
        input_rows,
        enriched_id_map,
        frozen_rows,
        expected_ids_sha256=expected_ids_sha256,
        expected_count=expected_count,
        expected_conversation_count=expected_conversations,
    )
    ids = [row["id"] for row in selected_rows]
    id_set = set(ids)
    refs = unique_index(read_jsonl(references_path), "eval_id", "frozen References")
    frozen_map = unique_index(
        read_jsonl(diagnosis_root / "frozen_eval_id_map.jsonl"), "eval_id", "frozen ID map"
    )
    if set(refs) != id_set or set(frozen_map) != id_set:
        raise ValueError("frozen input, References, and ID map IDs differ")

    for eval_id in ids:
        reference = refs[eval_id]
        mapping = frozen_map[eval_id]
        if reference.get("sample_id") != mapping.get("sample_id"):
            raise ValueError(f"frozen References and ID map sample IDs differ for {eval_id}")
        normalized = normalized_index.get(reference.get("sample_id"))
        if not isinstance(normalized, dict):
            raise ValueError(f"normalized Dev lacks sample for {eval_id}")
        if mapping.get("conversation_id") != normalized.get("conversation_id"):
            raise ValueError(f"frozen conversation mapping differs from normalized Dev for {eval_id}")
        if selected_rows[ids.index(eval_id)]["history"] != normalized.get("history"):
            raise ValueError(f"frozen input history differs from normalized Dev for {eval_id}")

    selected_normalized = [normalized_index[refs[eval_id]["sample_id"]] for eval_id in ids]
    source_lines = {
        row.get("source", {}).get("source_line")
        for row in selected_normalized
        if isinstance(row.get("source"), dict)
    }
    if len(source_lines) != expected_conversations or any(
        isinstance(line, bool) or not isinstance(line, int) or line < 1 for line in source_lines
    ):
        raise ValueError("frozen source references do not map to the expected conversation rows")
    raw_source_path = _resolve(sources["raw_source_path"])
    raw_source_rows: dict[int, dict[str, Any]] = {}
    with raw_source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in source_lines:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"raw Train source line {line_number} is not an object")
                raw_source_rows[line_number] = value
    if set(raw_source_rows) != source_lines:
        raise ValueError("one or more frozen sessions reference missing raw Train rows")
    history_audit = audit_prediction_point_sources(selected_normalized, raw_source_rows)
    if history_audit["point_count"] != expected_count or history_audit["conversation_count"] != expected_conversations:
        raise ValueError("source history audit did not cover the frozen 224-point cohort")

    eval_manifest_path = eval_root / "run_manifest.json"
    expected_eval_manifest_sha = evaluation["e01_b_full_eval_manifest_sha256"]
    _require_hash(eval_manifest_path, expected_eval_manifest_sha, "E01-B Full evaluation Manifest")
    eval_manifest = _json_object(eval_manifest_path)
    if (
        eval_manifest.get("status") != "scored"
        or eval_manifest.get("scope") != "full-dev"
        or eval_manifest.get("model") != "adapter"
        or eval_manifest.get("run_id") != eval_root.name
        or eval_manifest.get("training_run", {}).get("manifest_sha256")
        != training["expected_e01_b_manifest_sha256"]
    ):
        raise ValueError("E01-B reference Run is not the locked Full Adapter evaluation")
    training_manifest_path = _resolve(training["e01_b_training_run_root"]) / "manifest.json"
    _require_hash(
        training_manifest_path,
        training["expected_e01_b_manifest_sha256"],
        "E01-B Full training Manifest",
    )
    submission_path = eval_root / "runner/submission.jsonl"
    raw_path = eval_root / "raw_predictions.jsonl"
    for relative, path in (("runner/submission.jsonl", submission_path), ("raw_predictions.jsonl", raw_path)):
        expected = eval_manifest.get("artifacts", {}).get(relative)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"E01-B evaluation Manifest does not bind {relative}")
    performance_path = eval_root / "runner/performance_report.json"
    performance = _json_object(performance_path)
    adapter_identity = performance.get("adapter_identity", {})
    base_weight_identity = performance.get("model_weight_identity", {})
    expected_model = a3_config["base_model"]
    if (
        performance.get("run_id") != eval_root.name
        or performance.get("submission_sha256") != sha256_file(submission_path)
        or performance.get("raw_predictions_sha256") != sha256_file(raw_path)
        or performance.get("model_id") != expected_model["model_id"]
        or performance.get("model_revision") != expected_model["revision"]
        or base_weight_identity.get("observed_weight_hash") != training["expected_base_weight_sha256"]
        or adapter_identity.get("identity_status") != "observed"
        or adapter_identity.get("weight_sha256") != training["expected_initial_adapter_weight_sha256"]
        or adapter_identity.get("config_sha256") != training["expected_initial_adapter_config_sha256"]
        or performance.get("quantization", {}).get("real_nf4_verified") is not True
    ):
        raise ValueError("E01-B performance report does not bind the expected model-inference outputs")
    if sha256_file(inference_config_path) != eval_manifest.get("inference_config_sha256"):
        raise ValueError("E01-B inference config differs from its Run Manifest")

    output_schema_path = ROOT / "participant/templates/submission_schema.json"
    if sha256_file(output_schema_path) != a3_config["identity"]["expected_output_schema_sha256"]:
        raise ValueError("official output Schema differs from its frozen A3 fingerprint")
    if sha256_file(output_schema_path) != DEFAULT_SCHEMA_SHA256:
        raise ValueError("official output Schema differs from the competition boundary fingerprint")
    prompt_path = _resolve(a3_config["protocol"]["prompt_path"])
    prompt_sha256 = sha256_file(prompt_path)
    if prompt_sha256 != candidate["protocol"]["prompt_sha256"]:
        raise ValueError("P0 prompt hash differs from the frozen fast-eval protocol")

    source_paths = {
        "generation_input": input_path,
        "full_dev_id_map": id_map_path,
        "normalized_dev": normalized_path,
        "frozen_input": frozen_ids_path,
        "frozen_references": references_path,
        "frozen_id_map": diagnosis_root / "frozen_eval_id_map.jsonl",
        "frozen_selection": selection_path,
        "diagnosis_manifest": diagnosis_manifest_path,
        "raw_train_source": raw_source_path,
        "e01_b_submission": submission_path,
        "e01_b_raw_predictions": raw_path,
        "e01_b_performance_report": performance_path,
        "e01_b_eval_manifest": eval_manifest_path,
        "e01_b_training_manifest": training_manifest_path,
        "e01_b_inference_config": inference_config_path,
        "official_output_schema": output_schema_path,
        "p0_prompt": prompt_path,
        "a4_candidate_config": candidate_config_path,
        "a3_lock_config": a3_config_path,
        "reconcile_script": Path(__file__).resolve(),
        "profile_prefix_parser": ROOT / "src/evaluation/profile_prefix.py",
        "profile_metric_implementation": ROOT / "src/evaluation/metrics.py",
    }
    source_facts = {
        name: {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    return {
        "candidate_config": candidate,
        "a3_config": a3_config,
        "diagnosis_root": diagnosis_root,
        "eval_root": eval_root,
        "ids": ids,
        "all_dev_ids": set(all_input_index),
        "refs": refs,
        "input_index": {row["id"]: row for row in selected_rows},
        "normalized_index": normalized_index,
        "source_integrity": history_audit,
        "source_facts": source_facts,
        "eval_manifest": eval_manifest,
        "performance": performance,
        "submission_rows": read_jsonl(submission_path),
        "raw_rows": read_jsonl(raw_path),
        "enums": schema_enums(load_submission_schema(ROOT)),
        "output_schema_sha256": sha256_file(output_schema_path),
        "prompt_sha256": prompt_sha256,
        "training_manifest_sha256": sha256_file(training_manifest_path),
    }


def _full_json_status(raw_text: Any) -> str:
    if not isinstance(raw_text, str):
        return "missing_raw_text"
    try:
        value = strict_json_loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return "invalid_json"
    return "valid_object" if isinstance(value, dict) else "non_object_json"


def _brief_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": metrics["per_dimension"][PROFILE_DIMENSIONS[0]]["sample_count"],
        "overall_33_label_micro_precision": metrics["overall_33_label_micro_precision"],
        "overall_33_label_micro_recall": metrics["overall_33_label_micro_recall"],
        "overall_33_label_micro_f1": metrics["overall_33_label_micro_f1"],
        "overall_33_label_macro_f1": metrics["overall_33_label_macro_f1"],
        "overall_33_label_tp": metrics["overall_33_label_tp"],
        "overall_33_label_fp": metrics["overall_33_label_fp"],
        "overall_33_label_fn": metrics["overall_33_label_fn"],
        "per_dimension": {
            dimension: {
                key: metrics["per_dimension"][dimension][key]
                for key in ("local_micro_precision", "local_micro_recall", "local_micro_f1", "tp", "fp", "fn")
            }
            for dimension in PROFILE_DIMENSIONS
        },
    }


def _brief_pairwise(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "left_name": result["left_name"],
        "right_name": result["right_name"],
        "sample_count": result["sample_count"],
        "different_sample_count": result["different_sample_count"],
        "different_sample_ids": result["different_sample_ids"],
        "different_sample_count_by_dimension": result["different_sample_count_by_dimension"],
        "left_metrics": _brief_metrics(result["left_metrics"]),
        "right_metrics": _brief_metrics(result["right_metrics"]),
        "right_minus_left": result["right_minus_left"],
        "per_dimension_right_minus_left": result["per_dimension_right_minus_left"],
    }


def _load_fast_only_predictions(
    fast_report_path: Path,
    expected_ids: list[str],
    candidate: dict[str, Any],
    a3_config: dict[str, Any],
    source_facts: dict[str, Any],
    eval_manifest: dict[str, Any],
    enums: dict[str, list[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = _json_object(fast_report_path)
    artifact = report.get("profile_predictions", {})
    run_identity = report.get("run_identity", {})
    generation = report.get("generation", {})
    model_facts = run_identity.get("model_facts", {})
    tokenizer_facts = run_identity.get("tokenizer_facts", {})
    expected_ids_sha256 = candidate["selection"]["expected_ids_sha256"]
    prediction_path_value = artifact.get("path")
    if not isinstance(prediction_path_value, str):
        raise ValueError("Fast-only report has no profile prediction artifact path")
    prediction_path = _resolve(prediction_path_value)
    if prediction_path != fast_report_path.parent.resolve() / "profile_predictions.jsonl":
        raise ValueError("Fast-only predictions must be adjacent to their bound report")
    expected_adapter = a3_config["identity"]
    expected_protocol = a3_config["protocol"]
    if (
        report.get("status") != "passed"
        or report.get("gpu_used") is not True
        or report.get("references_read_by_generator") is not False
        or report.get("sample_count") != len(expected_ids)
        or report.get("valid_count") != len(expected_ids)
        or report.get("attempted_count") != len(expected_ids)
        or generation.get("model_generate_call_count") != len(expected_ids)
        or generation.get("full_generation_performed") is not False
        or generation.get("response_text_generated_or_saved") is not False
        or artifact.get("rows") != len(expected_ids)
        or artifact.get("expected_rows") != len(expected_ids)
        or artifact.get("all_selected_rows_valid") is not True
        or artifact.get("strict_quality_comparison_ready") is not True
        or artifact.get("ids_sha256") != expected_ids_sha256
        or artifact.get("sha256") != sha256_file(prediction_path)
        or run_identity.get("config_sha256") != source_facts["a4_candidate_config"]["sha256"]
        or run_identity.get("inference_config_sha256") != eval_manifest.get("inference_config_sha256")
        or run_identity.get("base_weight_sha256") != expected_adapter["expected_base_weight_sha256"]
        or run_identity.get("adapter_weight_sha256") != expected_adapter["expected_initial_adapter_weight_sha256"]
        or run_identity.get("adapter_config_sha256") != expected_adapter["expected_initial_adapter_config_sha256"]
        or run_identity.get("prompt_facts", {}).get("sha256") != candidate["protocol"]["prompt_sha256"]
        or run_identity.get("generation", {}).get("do_sample") is not False
        or run_identity.get("generation", {}).get("max_new_tokens") != candidate["generation"]["max_new_tokens"]
        or tokenizer_facts.get("signature_sha256") != expected_protocol["tokenizer_signature_sha256"]
        or tokenizer_facts.get("template_sha256") != expected_protocol["chat_template_sha256"]
        or model_facts.get("quantization", {}).get("real_nf4_verified") is not True
    ):
        raise ValueError("Fast-only report is incomplete or model/protocol identity differs from E01-B")
    rows = read_jsonl(prediction_path)
    indexed, status_counts = index_profile_prediction_rows(
        rows,
        expected_ids,
        enums,
        require_all_valid=True,
    )
    if status_counts.get("valid") != len(expected_ids):
        raise ValueError("strict A4 comparison requires 224 valid Fast-only profiles")
    source_selection = report.get("source_manifest", {}).get("selection", {})
    if (
        source_selection.get("selected_ids_sha256") != expected_ids_sha256
        or source_selection.get("expected_ids_sha256") != expected_ids_sha256
    ):
        raise ValueError("Fast-only source selection does not match the frozen ID set")
    return indexed, {
        "run_id": fast_report_path.parent.name,
        "report_path": str(fast_report_path),
        "report_sha256": sha256_file(fast_report_path),
        "predictions_path": str(prediction_path),
        "predictions_sha256": artifact["sha256"],
        "predictions_rows": len(rows),
        "predictions_ids_sha256": artifact["ids_sha256"],
        "status_counts": status_counts,
        "elapsed_seconds": report.get("timing", {}).get("elapsed_seconds_including_model_load"),
        "generation_seconds": generation.get("generation_seconds"),
        "model_load_seconds": report.get("timing", {}).get("model_load_seconds"),
        "base_weight_sha256": run_identity.get("base_weight_sha256"),
        "adapter_weight_sha256": run_identity.get("adapter_weight_sha256"),
        "adapter_config_sha256": run_identity.get("adapter_config_sha256"),
        "tokenizer_signature_sha256": tokenizer_facts.get("signature_sha256"),
        "chat_template_sha256": tokenizer_facts.get("template_sha256"),
        "fast_alignment_status": "pending_pairwise_comparison",
    }


def reconcile(
    run_id: str,
    candidate_config_path: Path,
    a3_config_path: Path,
    output_root: Path,
    fast_report_path: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
        raise ValueError("run ID contains unsupported path characters")
    run_root = output_root / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing A4 reconciliation Run: {run_root}")

    loaded = _load_sources(candidate_config_path, a3_config_path)
    ids = loaded["ids"]
    id_set = set(ids)
    submissions = unique_index(loaded["submission_rows"], "id", "E01-B Full submission")
    raw_predictions = unique_index(loaded["raw_rows"], "id", "E01-B Full Raw")
    full_dev_ids = loaded["all_dev_ids"]
    if set(submissions) != full_dev_ids or set(raw_predictions) != full_dev_ids:
        raise ValueError("E01-B submission/Raw IDs do not exactly cover the full target-free Dev input")
    if not id_set <= set(submissions) or not id_set <= set(raw_predictions):
        raise ValueError("E01-B output IDs do not cover the frozen 224-point cohort")

    selected_submission = {sample_id: submissions[sample_id] for sample_id in ids}
    selected_raw = {sample_id: raw_predictions[sample_id] for sample_id in ids}
    gold_profiles = [loaded["refs"][sample_id]["target"]["user_profile"] for sample_id in ids]
    submission_profiles = _validate_profile_rows(
        selected_submission, ids, loaded["enums"], "E01-B Full submission"
    )

    prefix_profiles: list[dict[str, list[str]]] = []
    prefix_results: dict[str, Any] = {}
    full_statuses: dict[str, str] = {}
    for sample_id in ids:
        raw_row = selected_raw[sample_id]
        parsed = parse_profile_prefix(raw_row.get("raw_text"), loaded["enums"])
        prefix_results[sample_id] = parsed
        prefix_profiles.append(
            validate_profile(parsed.user_profile, loaded["enums"], label=f"Raw prefix {sample_id}")
            if parsed.valid and parsed.user_profile is not None
            else empty_profile()
        )
        full_statuses[sample_id] = _full_json_status(raw_row.get("raw_text"))

    comparison = compare_profile_lists(
        ids=ids,
        gold_profiles=gold_profiles,
        left_profiles=submission_profiles,
        right_profiles=prefix_profiles,
        enums=loaded["enums"],
        left_name="S_submission",
        right_name="R_raw_profile_prefix",
    )
    fast_index: dict[str, dict[str, Any]] | None = None
    fast_metadata: dict[str, Any] | None = None
    fast_pairs: dict[str, dict[str, Any]] = {}
    emotion_alignment: dict[str, Any] | None = None
    if fast_report_path is not None:
        fast_index, fast_metadata = _load_fast_only_predictions(
            fast_report_path,
            ids,
            loaded["candidate_config"],
            loaded["a3_config"],
            loaded["source_facts"],
            loaded["eval_manifest"],
            loaded["enums"],
        )
        fast_profiles = [fast_index[sample_id]["user_profile"] for sample_id in ids]
        fast_pairs = {
            "S_submission_vs_R_raw_prefix": _brief_pairwise(comparison),
            "R_raw_prefix_vs_F_fast_only": _brief_pairwise(compare_profile_lists(
                ids=ids,
                gold_profiles=gold_profiles,
                left_profiles=prefix_profiles,
                right_profiles=fast_profiles,
                enums=loaded["enums"],
                left_name="R_raw_profile_prefix",
                right_name="F_fast_only",
            )),
            "S_submission_vs_F_fast_only": _brief_pairwise(compare_profile_lists(
                ids=ids,
                gold_profiles=gold_profiles,
                left_profiles=submission_profiles,
                right_profiles=fast_profiles,
                enums=loaded["enums"],
                left_name="S_submission",
                right_name="F_fast_only",
            )),
        }
        emotion_alignment = {
            "submission_vs_raw_prefix_different_ids": [
                sample_id
                for sample_id in ids
                if selected_submission[sample_id].get("emotion_label") != prefix_results[sample_id].emotion_label
            ],
            "raw_prefix_vs_fast_only_different_ids": [
                sample_id
                for sample_id in ids
                if prefix_results[sample_id].emotion_label != fast_index[sample_id].get("emotion_label")
            ],
            "submission_vs_fast_only_different_ids": [
                sample_id
                for sample_id in ids
                if selected_submission[sample_id].get("emotion_label") != fast_index[sample_id].get("emotion_label")
            ],
            "raw_prefix_vs_fast_only_different_count": sum(
                prefix_results[sample_id].emotion_label != fast_index[sample_id].get("emotion_label")
                for sample_id in ids
            ),
            "submission_vs_fast_only_different_count": sum(
                selected_submission[sample_id].get("emotion_label") != fast_index[sample_id].get("emotion_label")
                for sample_id in ids
            ),
            "submission_vs_fast_only_raw_status_counts": dict(sorted(Counter(
                selected_raw[sample_id].get("status", "missing_status")
                for sample_id in ids
                if selected_submission[sample_id].get("emotion_label") != fast_index[sample_id].get("emotion_label")
            ).items())),
            "submission_vs_fast_only_full_json_status_counts": dict(sorted(Counter(
                full_statuses[sample_id]
                for sample_id in ids
                if selected_submission[sample_id].get("emotion_label") != fast_index[sample_id].get("emotion_label")
            ).items())),
        }
        fast_metadata["fast_alignment_status"] = (
            "reconciled"
            if fast_pairs["R_raw_prefix_vs_F_fast_only"]["different_sample_count"] == 0
            and emotion_alignment["raw_prefix_vs_fast_only_different_count"] == 0
            else "unresolved"
        )
        fast_metadata["legacy_full_run_token_ids_available"] = False
    raw_status_counts = Counter(selected_raw[sample_id].get("status", "missing_status") for sample_id in ids)
    prefix_status_counts = Counter(prefix_results[sample_id].status for sample_id in ids)
    full_json_status_counts = Counter(full_statuses.values())

    cases: list[dict[str, Any]] = []
    for sample_id, submission_profile, prefix_profile in zip(ids, submission_profiles, prefix_profiles):
        parsed = prefix_results[sample_id]
        different_dimensions = profile_difference_dimensions(submission_profile, prefix_profile)
        case = {
            "id": sample_id,
            "raw_status": selected_raw[sample_id].get("status"),
            "prefix_status": parsed.status,
            "prefix_reason": parsed.reason,
            "full_json_status": full_statuses[sample_id],
            "S_submission_profile": submission_profile,
            "R_prefix_profile": prefix_profile,
            "different_dimensions": different_dimensions,
            "difference_type": classify_historical_difference(
                submission_profile=submission_profile,
                prefix_profile=prefix_profile,
                prefix_status=parsed.status,
                full_json_status="valid" if full_statuses[sample_id] == "valid_object" else full_statuses[sample_id],
            ),
        }
        if fast_index is not None:
            fast_prediction = fast_index[sample_id]
            fast_profile = fast_prediction["user_profile"]
            case.update({
                "S_emotion_label": selected_submission[sample_id].get("emotion_label"),
                "R_emotion_label": parsed.emotion_label,
                "F_status": fast_prediction["status"],
                "F_reason": fast_prediction["reason"],
                "F_emotion_label": fast_prediction["emotion_label"],
                "F_profile": fast_profile,
                "R_F_different_dimensions": profile_difference_dimensions(prefix_profile, fast_profile),
                "S_F_different_dimensions": profile_difference_dimensions(submission_profile, fast_profile),
                "R_F_emotion_match": parsed.emotion_label == fast_prediction["emotion_label"],
                "S_F_emotion_match": selected_submission[sample_id].get("emotion_label") == fast_prediction["emotion_label"],
            })
        cases.append(case)

    source_file_facts = dict(loaded["source_facts"])
    output_descriptions = {
        "historical_comparison.json": "S and R-prefix profile metrics, statuses, and pairwise deltas",
        "historical_cases.jsonl": "ID and minimal profile-only alignment facts; no dialogue or response text",
    }
    if fast_metadata is not None:
        source_file_facts["fast_only_report"] = {
            "path": fast_metadata["report_path"],
            "sha256": fast_metadata["report_sha256"],
        }
        source_file_facts["fast_only_profile_predictions"] = {
            "path": fast_metadata["predictions_path"],
            "sha256": fast_metadata["predictions_sha256"],
        }
        output_descriptions["historical_cases.jsonl"] = "Per-ID S/R/F profiles and differences; no dialogue or response text"
        output_descriptions["fast_alignment.json"] = "S/R/F pairwise profile and emotion alignment metrics"

    source_manifest = {
        "run_id": run_id,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "historical_submission_raw_prefix_and_optional_fast_only_reconciliation",
        "cohort": {
            "count": len(ids),
            "conversation_count": loaded["source_integrity"]["conversation_count"],
            "ids_sha256": canonical_sha256(sorted(ids)),
            "expected_ids_sha256": loaded["candidate_config"]["selection"]["expected_ids_sha256"],
            "membership_frozen": True,
            "gold_used_for_membership": False,
        },
        "source_integrity": loaded["source_integrity"],
        "protocol": {
            "prompt_version": loaded["candidate_config"]["protocol"]["prompt_version"],
            "prompt_sha256": loaded["prompt_sha256"],
            "format_version": loaded["candidate_config"]["protocol"]["format_version"],
            "truncation_version": loaded["candidate_config"]["protocol"]["truncation_version"],
            "generation_input_target_free": True,
        },
        "model_identity": {
            "e01_b_eval_run_id": loaded["eval_manifest"]["run_id"],
            "e01_b_eval_manifest_sha256": loaded["source_facts"]["e01_b_eval_manifest"]["sha256"],
            "e01_b_training_manifest_sha256": loaded["training_manifest_sha256"],
            "legacy_runner_source_sha256": loaded["eval_manifest"]
            .get("source_code", {})
            .get("files", {})
            .get("participant/run_inference.py"),
            "base_model_id": loaded["performance"].get("model_id"),
            "base_model_revision": loaded["performance"].get("model_revision"),
            "base_weight_sha256": loaded["performance"].get("model_weight_identity", {}).get("observed_weight_hash"),
            "adapter_identity": loaded["performance"].get("adapter_identity"),
            "quantization": loaded["performance"].get("quantization"),
        },
        "source_files": source_file_facts,
        "fast_only_run": fast_metadata,
        "outputs": output_descriptions,
    }
    historical_comparison = {
        "status": "completed",
        "run_id": run_id,
        "sample_count": len(ids),
        "source_submission": _brief_metrics(comparison["left_metrics"]),
        "raw_prefix_fixed_denominator": _brief_metrics(comparison["right_metrics"]),
        "submission_minus_raw_prefix": {
            "different_sample_count": comparison["different_sample_count"],
            "different_sample_count_by_dimension": comparison["different_sample_count_by_dimension"],
            "different_sample_ids": comparison["different_sample_ids"],
            "raw_prefix_minus_submission": comparison["right_minus_left"],
            "per_dimension_raw_prefix_minus_submission": comparison["per_dimension_right_minus_left"],
        },
        "raw_runner_status_counts": dict(sorted(raw_status_counts.items())),
        "raw_prefix_parse_status_counts": dict(sorted(prefix_status_counts.items())),
        "full_raw_json_status_counts": dict(sorted(full_json_status_counts.items())),
        "raw_prefix_failed_rows_scored_as_empty": sum(
            prefix_results[sample_id].status != "valid" for sample_id in ids
        ),
        "interpretation": "S is the historical normalized Submission; R is the parsed emotion/profile prefix from the same Raw generation. Full response content does not contribute to R profile scoring.",
    }
    if fast_metadata is not None:
        historical_comparison["fast_only_profile_metrics"] = fast_pairs["S_submission_vs_F_fast_only"]["right_metrics"]
        historical_comparison["fast_only_status_counts"] = fast_metadata["status_counts"]
        historical_comparison["pairwise_comparisons"] = fast_pairs
        historical_comparison["emotion_label_alignment"] = emotion_alignment
        historical_comparison["fast_alignment_status"] = fast_metadata["fast_alignment_status"]
    report = _render_report(source_manifest, historical_comparison)

    run_root.mkdir(parents=True)
    _write_json(run_root / "historical_comparison.json", historical_comparison)
    _write_jsonl(run_root / "historical_cases.jsonl", cases)
    if fast_metadata is not None:
        _write_json(run_root / "fast_alignment.json", {
            "status": fast_metadata["fast_alignment_status"],
            "sample_count": len(ids),
            "pairwise_comparisons": fast_pairs,
            "emotion_label_alignment": emotion_alignment,
            "legacy_full_run_token_ids_available": False,
        })
    (run_root / "report.md").write_text(report, encoding="utf-8")
    source_manifest["output_artifacts"] = {
        name: {"sha256": sha256_file(run_root / name), "row_count": len(cases) if name.endswith(".jsonl") else None}
        for name in (
            "historical_comparison.json",
            "historical_cases.jsonl",
            *( ("fast_alignment.json",) if fast_metadata is not None else () ),
            "report.md",
        )
    }
    _write_json(run_root / "source_manifest.json", source_manifest)
    return historical_comparison


def _render_report(manifest: dict[str, Any], result: dict[str, Any]) -> str:
    s = result["source_submission"]
    r = result["raw_prefix_fixed_denominator"]
    diff = result["submission_minus_raw_prefix"]
    lines = [
        "# E03-A4-1 历史画像评测口径核对",
        "",
        f"- Run：`{manifest['run_id']}`；状态：`{result['status']}`。",
        f"- 冻结 Cohort：{manifest['cohort']['count']} 点 / {manifest['cohort']['conversation_count']} 段完整会话；ID SHA-256 `{manifest['cohort']['ids_sha256']}`。",
        f"- 源历史核验：{manifest['source_integrity']['status']}；逐点核验当前 user turn、角色、内容与顺序。",
        "- 输入仅为 target-free `{id, history}`；报告不包含源对话或模型回复文本。",
        "",
        "## 同一模型、同一 ID 上的画像分数",
        "",
        "| 口径 | TP | FP | FN | Micro-F1 | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| S：历史 Submission | {s['overall_33_label_tp']} | {s['overall_33_label_fp']} | {s['overall_33_label_fn']} | {s['overall_33_label_micro_f1']:.6f} | {s['overall_33_label_macro_f1']:.6f} |",
        f"| R：Raw 的已闭合画像前缀（固定 224 分母） | {r['overall_33_label_tp']} | {r['overall_33_label_fp']} | {r['overall_33_label_fn']} | {r['overall_33_label_micro_f1']:.6f} | {r['overall_33_label_macro_f1']:.6f} |",
    ]
    if "fast_only_profile_metrics" in result:
        f = result["fast_only_profile_metrics"]
        rf = result["pairwise_comparisons"]["R_raw_prefix_vs_F_fast_only"]
        emotion = result["emotion_label_alignment"]
        lines.extend([
            f"| F：新 Fast-only 逐 ID 结果（固定 224 分母） | {f['overall_33_label_tp']} | {f['overall_33_label_fp']} | {f['overall_33_label_fn']} | {f['overall_33_label_micro_f1']:.6f} | {f['overall_33_label_macro_f1']:.6f} |",
            "",
            f"- Fast-only Run：`{manifest['fast_only_run']['run_id']}`；状态 `passed` 表示覆盖门槛通过；本次另核实 224/224 有效并完成严格 ID 对齐。",
            f"- R↔F 画像不一致：{rf['different_sample_count']} 条；维度计数 `{json.dumps(rf['different_sample_count_by_dimension'], ensure_ascii=False, sort_keys=True)}`；R↔F 情绪标签差异 `{emotion['raw_prefix_vs_fast_only_different_count']}` 条。",
            f"- S↔F 情绪标签差异 `{emotion['submission_vs_fast_only_different_count']}` 条，ID `{json.dumps(emotion['submission_vs_fast_only_different_ids'], ensure_ascii=False)}`；这些行的 Raw 状态 `{json.dumps(emotion['submission_vs_fast_only_raw_status_counts'], ensure_ascii=False, sort_keys=True)}`、完整 JSON 状态 `{json.dumps(emotion['submission_vs_fast_only_full_json_status_counts'], ensure_ascii=False, sort_keys=True)}`。R/F 情绪一致，差异在历史 Submission 的整段输出归一化结果中。",
            f"- 配对统计见 `fast_alignment.json`；逐 ID S/R/F 画像与情绪见 `historical_cases.jsonl`。",
            f"- 语义对齐状态：`{result['fast_alignment_status']}`。旧完整 Run 未保留逐 Token IDs，因此本次不作历史 Token-ID 逐项相等结论。",
        ])
    lines.extend([
        "",
        f"- R 对 S 的画像差异：{diff['different_sample_count']} 条；按维度 `{json.dumps(diff['different_sample_count_by_dimension'], ensure_ascii=False, sort_keys=True)}`。",
        f"- Raw Runner 状态：`{json.dumps(result['raw_runner_status_counts'], ensure_ascii=False, sort_keys=True)}`；前缀解析：`{json.dumps(result['raw_prefix_parse_status_counts'], ensure_ascii=False, sort_keys=True)}`；完整 Raw JSON：`{json.dumps(result['full_raw_json_status_counts'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 归因与边界",
        "",
        "- 逐 ID 对齐表只保存 ID、状态、画像标签及差异类型，不保存会话、Gold 原始行或回复文本。",
        "- S、R、F 使用同一 `profile_metrics()`；失败项按空画像进入固定 224 分母。严格比较拒绝缺 ID、重复 ID 或非 224 行预测文件。",
    ])
    if "fast_only_profile_metrics" not in result:
        lines.append("- 本 Run 没有 F 的逐 ID 预测；不能据此声称旧 Fast-only 聚合分数与 R 逐条一致。")
    lines.extend([
        "- 后续候选筛选使用已核验的 E01-B Fast-only 逐 ID 基线；历史完整 Dev E01-B 保留为正式指标参照。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-config", default=str(DEFAULT_CANDIDATE_CONFIG.relative_to(ROOT)))
    parser.add_argument("--a3-config", default=str(DEFAULT_A3_CONFIG.relative_to(ROOT)))
    parser.add_argument("--output-root", default="outputs/diagnostics/e03_a4/reconcile")
    parser.add_argument("--fast-report", help="strictly validate and pair a completed Fast-only run with S/R")
    args = parser.parse_args()
    try:
        result = reconcile(
            args.run_id,
            _resolve(args.candidate_config),
            _resolve(args.a3_config),
            _resolve(args.output_root),
            _resolve(args.fast_report) if args.fast_report else None,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "preflight_failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "sample_count": result["sample_count"],
                "S_micro_f1": result["source_submission"]["overall_33_label_micro_f1"],
                "R_prefix_micro_f1": result["raw_prefix_fixed_denominator"]["overall_33_label_micro_f1"],
                "different_sample_count": result["submission_minus_raw_prefix"]["different_sample_count"],
                "fast_alignment_status": result.get("fast_alignment_status"),
                "F_fast_micro_f1": result.get("fast_only_profile_metrics", {}).get("overall_33_label_micro_f1"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
