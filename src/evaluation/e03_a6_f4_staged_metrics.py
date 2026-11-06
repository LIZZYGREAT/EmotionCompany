"""CPU cohort identity and fixed-denominator metrics for E03-A6 F4 staged Fast-only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.f3_pilot import selected_ids_sha256, validate_reference_coverage
from src.evaluation.metrics import emotion_metrics, profile_metrics
from src.evaluation.source_integrity import unique_index, validate_target_free_inputs


DIMENSIONS = ("personality_traits", "interests", "style")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def response_body_token_count(token_ids: Sequence[int], decode) -> int:
    """Count emitted tokens after the Response marker, including a marker/body token."""
    decoded = decode(token_ids)
    marker = decoded.find("\nResponse=")
    if marker < 0:
        return 0
    marker_end = marker + len("\nResponse=")
    if not decoded[marker_end:]:
        return 0
    marker_token_count = next(
        index for index in range(1, len(token_ids) + 1)
        if "\nResponse=" in decode(token_ids[:index])
    )
    marker_token_text = decode(token_ids[:marker_token_count])
    token_suffix = marker_token_text[marker_token_text.find("\nResponse=") + len("\nResponse="):]
    return max(1, len(token_ids) - marker_token_count + int(bool(token_suffix)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _conversation_id(mapping: Mapping[str, Any]) -> str:
    conversation_id = mapping.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id:
        return conversation_id
    sample_id = mapping.get("sample_id")
    if not isinstance(sample_id, str):
        raise ValueError("frozen cohort ID map has no conversation identity")
    parts = sample_id.split(":")
    if len(parts) < 3 or not parts[1]:
        raise ValueError(f"sample_id does not encode a D0.3 conversation: {sample_id!r}")
    return parts[1]


def load_target_free_cohorts(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Read only target-free inputs and ID maps; References remain unopened."""
    cohorts: dict[str, Any] = {}
    for name, source in config["cohorts"].items():
        input_path = (project_root / source["input_path"]).resolve()
        map_path = (project_root / source["id_map_path"]).resolve()
        if sha256_file(input_path) != source["expected_input_sha256"]:
            raise ValueError(f"{name} target-free input SHA differs from its frozen source")
        if sha256_file(map_path) != source["expected_id_map_sha256"]:
            raise ValueError(f"{name} ID Map SHA differs from its frozen source")
        input_rows = _read_jsonl(input_path)
        input_index = validate_target_free_inputs(input_rows)
        mapping_rows = _read_jsonl(map_path)
        mapping = unique_index(mapping_rows, "eval_id", f"{name} ID Map")
        if set(input_index) != set(mapping):
            raise ValueError(f"{name} target-free input and ID Map IDs differ")
        ids = [str(row["id"]) for row in input_rows]
        if len(ids) != int(source["expected_count"]) or selected_ids_sha256(ids) != source["expected_ids_sha256"]:
            raise ValueError(f"{name} frozen ID count or SHA differs")
        conversations = {_conversation_id(mapping[eval_id]) for eval_id in ids}
        expected_conversations = source.get("expected_conversation_count")
        if expected_conversations is not None and len(conversations) != int(expected_conversations):
            raise ValueError(f"{name} conversation count differs from the frozen cohort")
        cohorts[name] = {
            "name": name,
            "input_path": input_path,
            "id_map_path": map_path,
            "input_rows": input_rows,
            "input_index": input_index,
            "mapping": mapping,
            "ids": ids,
            "conversation_count": len(conversations),
            "input_sha256": sha256_file(input_path),
            "id_map_sha256": sha256_file(map_path),
            "ids_sha256": selected_ids_sha256(ids),
        }
    first = cohorts["pilot160"]
    second = cohorts["dev224"]
    overlap = set(first["ids"]) & set(second["ids"])
    for eval_id in overlap:
        if first["mapping"][eval_id].get("sample_id") != second["mapping"][eval_id].get("sample_id"):
            raise ValueError(f"overlapping F4 cohort ID maps to different source rows: {eval_id}")
        if first["input_index"][eval_id] != second["input_index"][eval_id]:
            raise ValueError(f"overlapping F4 cohort has different target-free input: {eval_id}")
    ordered_union = list(first["ids"]) + [value for value in second["ids"] if value not in set(first["ids"])]
    union_index = {**first["input_index"], **second["input_index"]}
    return {
        "cohorts": cohorts,
        "overlap_count": len(overlap),
        "union_count": len(ordered_union),
        "union_ids": ordered_union,
        "union_ids_sha256": selected_ids_sha256(ordered_union),
        "union_inputs": [dict(union_index[eval_id]) for eval_id in ordered_union],
        "references_read": False,
    }


def load_gold_references(
    config: Mapping[str, Any],
    project_root: Path,
    cohorts: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Open Gold only after prediction JSONL hash is frozen by the caller."""
    gold: dict[str, dict[str, dict[str, Any]]] = {}
    for name, cohort in cohorts.items():
        source = config["cohorts"][name]
        reference_path = (project_root / source["references_path"]).resolve()
        if sha256_file(reference_path) != source["expected_references_sha256"]:
            raise ValueError(f"{name} References SHA differs from its frozen source")
        rows = _read_jsonl(reference_path)
        indexed = unique_index(rows, "eval_id", f"{name} References")
        validate_reference_coverage(list(indexed), cohort["ids"])
        if len(indexed) != int(source["expected_count"]):
            raise ValueError(f"{name} References row count differs from its fixed denominator")
        for eval_id in cohort["ids"]:
            if indexed[eval_id].get("sample_id") != cohort["mapping"][eval_id].get("sample_id"):
                raise ValueError(f"{name} Gold Reference and frozen ID Map disagree for {eval_id}")
            target = indexed[eval_id].get("target")
            if not isinstance(target, Mapping) or not isinstance(target.get("user_profile"), Mapping):
                raise ValueError(f"{name} Gold target lacks fixed F4 emotion/profile fields")
        gold[name] = indexed
    return gold


def score_cohort(
    cohort: Mapping[str, Any],
    reference_index: Mapping[str, Mapping[str, Any]],
    prediction_index: Mapping[str, Mapping[str, Any]],
    enums: Mapping[str, list[str]],
) -> dict[str, Any]:
    ids = list(cohort["ids"])
    if set(ids) - set(reference_index) or set(ids) - set(prediction_index):
        raise ValueError("fixed cohort lacks References or generated predictions")
    gold_targets = [reference_index[eval_id]["target"] for eval_id in ids]
    predictions = [prediction_index[eval_id] for eval_id in ids]
    valid = [row.get("status") == "valid" and isinstance(row.get("parsed"), Mapping) for row in predictions]
    pred_emotions = [row["parsed"]["emotion_label"] if is_valid else "" for row, is_valid in zip(predictions, valid)]
    pred_profiles = []
    gold_profiles = []
    reasons = Counter()
    status_counts = Counter()
    prefix_lengths = []
    for target, prediction, is_valid in zip(gold_targets, predictions, valid):
        profile = prediction.get("parsed", {}).get("user_profile") if is_valid else None
        if not isinstance(profile, Mapping) or any(not isinstance(profile.get(dimension), list) for dimension in DIMENSIONS):
            profile = {dimension: [] for dimension in DIMENSIONS}
        pred_profiles.append({dimension: list(profile[dimension]) for dimension in DIMENSIONS})
        gold_profiles.append({dimension: list(target["user_profile"][dimension]) for dimension in DIMENSIONS})
        status_counts[str(prediction.get("status", "missing"))] += 1
        if prediction.get("reason"):
            reasons[str(prediction["reason"])] += 1
        prefix_lengths.append(int(prediction.get("generated_token_count", 0)))
    emotions = emotion_metrics(
        [str(target["emotion_label"]) for target in gold_targets],
        pred_emotions,
        list(enums["emotion_label"]),
    )
    profiles = profile_metrics(gold_profiles, pred_profiles, dict(enums))
    prediction_counts = Counter(value for value, is_valid in zip(pred_emotions, valid) if is_valid)
    valid_count = sum(valid)
    predicted_nonempty = [any(profile[dimension] for dimension in DIMENSIONS) for profile in pred_profiles]
    predicted_nonempty_by_dimension = {
        dimension: sum(bool(profile[dimension]) for profile in pred_profiles)
        for dimension in DIMENSIONS
    }
    empty_profile_count = sum(not value for value in predicted_nonempty)
    prediction_label_counts = [sum(len(profile[dimension]) for dimension in DIMENSIONS) for profile in pred_profiles]
    gold_nonempty = [any(profile[dimension] for dimension in DIMENSIONS) for profile in gold_profiles]
    gold_nonempty_count = sum(gold_nonempty)
    recall_when_gold_nonempty = sum(a and b for a, b in zip(gold_nonempty, predicted_nonempty)) / gold_nonempty_count if gold_nonempty_count else 0.0
    return {
        "cohort_name": cohort["name"],
        "fixed_denominator_count": len(ids),
        "ids_sha256": cohort["ids_sha256"],
        "conversation_count": cohort["conversation_count"],
        "fast_prefix_success_rate": valid_count / len(ids) if ids else 0.0,
        "strict_f4_field_parse_rate": valid_count / len(ids) if ids else 0.0,
        "valid_count": valid_count,
        "status_counts": dict(sorted(status_counts.items())),
        "error_reason_counts": dict(sorted(reasons.items())),
        "boundary_missing_or_truncated_count": sum(row.get("stop_reason") != "response_marker_closed" for row in predictions),
        "response_boundary_closed_count": sum(row.get("stop_reason") == "response_marker_closed" for row in predictions),
        "response_body_tokens_generated": sum(int(row.get("response_body_tokens_generated", 0)) for row in predictions),
        "generated_prefix_tokens": {
            "total": sum(prefix_lengths),
            "mean": (sum(prefix_lengths) / len(prefix_lengths)) if prefix_lengths else 0.0,
            "max": max(prefix_lengths, default=0),
        },
        "emotion": {
            **emotions,
            "prediction_counts_valid_only": dict(sorted(prediction_counts.items())),
            "max_single_class_share_valid_only": max(prediction_counts.values(), default=0) / valid_count if valid_count else 0.0,
        },
        "profile_33_label": profiles,
        "profile_dimensions": profiles["per_dimension"],
        "nonempty_profile_prediction_count": sum(predicted_nonempty),
        "predicted_nonempty_by_dimension": predicted_nonempty_by_dimension,
        "empty_profile_prediction_count": empty_profile_count,
        "empty_profile_prediction_rate": empty_profile_count / len(ids) if ids else 0.0,
        "predicted_profile_label_count_per_sample": {
            "total": sum(prediction_label_counts),
            "mean": (sum(prediction_label_counts) / len(prediction_label_counts)) if prediction_label_counts else 0.0,
            "max": max(prediction_label_counts, default=0),
        },
        "gold_nonempty_profile_count": gold_nonempty_count,
        "recall_when_gold_profile_nonempty": recall_when_gold_nonempty,
    }


def compare_same_ids(
    previous: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> dict[str, Any]:
    shared = [eval_id for eval_id in ids if eval_id in previous and eval_id in current]
    correct_before = 0
    correct_after = 0
    improved = 0
    regressed = 0
    changed_emotion = 0
    changed_profile = 0
    tp_delta = fp_delta = fn_delta = 0
    for eval_id in shared:
        target = references[eval_id]["target"]
        old = previous[eval_id]
        new = current[eval_id]
        old_parsed = old.get("parsed") if old.get("status") == "valid" else None
        new_parsed = new.get("parsed") if new.get("status") == "valid" else None
        old_emotion = old_parsed.get("emotion_label") if isinstance(old_parsed, Mapping) else None
        new_emotion = new_parsed.get("emotion_label") if isinstance(new_parsed, Mapping) else None
        gold_emotion = target["emotion_label"]
        old_correct = old_emotion == gold_emotion
        new_correct = new_emotion == gold_emotion
        correct_before += old_correct
        correct_after += new_correct
        improved += not old_correct and new_correct
        regressed += old_correct and not new_correct
        changed_emotion += old_emotion != new_emotion
        old_profile = old_parsed.get("user_profile", {}) if isinstance(old_parsed, Mapping) else {}
        new_profile = new_parsed.get("user_profile", {}) if isinstance(new_parsed, Mapping) else {}
        changed_profile += any(list(old_profile.get(dim, [])) != list(new_profile.get(dim, [])) for dim in DIMENSIONS)
        for dim in DIMENSIONS:
            gold = set(target["user_profile"][dim])
            old_pred = set(old_profile.get(dim, []))
            new_pred = set(new_profile.get(dim, []))
            tp_delta += len(new_pred & gold) - len(old_pred & gold)
            fp_delta += len(new_pred - gold) - len(old_pred - gold)
            fn_delta += len(old_pred & gold) - len(new_pred & gold)
    return {
        "shared_id_count": len(shared),
        "emotion_correct_before": correct_before,
        "emotion_correct_after": correct_after,
        "emotion_correct_to_wrong": regressed,
        "emotion_wrong_to_correct": improved,
        "emotion_prediction_changed_count": changed_emotion,
        "profile_changed_sample_count": changed_profile,
        "profile_tp_delta": tp_delta,
        "profile_fp_delta": fp_delta,
        "profile_fn_delta": fn_delta,
    }


def historical_baselines(project_root: Path) -> dict[str, Any]:
    pilot_path = project_root / "outputs/evaluation/e03_a6_f4_full_field_pilot/E03-A6-F4-PILOT-EXPANDED-20261106-01/fast-160/summary.json"
    full_b_path = project_root / "outputs/diagnostics/e03_engineering/profile_fast/E03-PROFILE-FASTONLY-224-20261105-R2/profile_fast_metrics.json"
    result: dict[str, Any] = {}
    if pilot_path.is_file():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        metric = pilot.get("f4", {})
        if pilot.get("cohort_ids_sha256") != "98f24f1c1b41e1676618f4e557daa402b3cbb80686916ca818e1bdc1b89ae49c" or pilot.get("cohort_count") != 160:
            raise ValueError("historical F4 256-step Pilot baseline does not match the frozen 160 cohort")
        profile = metric.get("fixed_denominator_profile_metrics", {})
        result["f4_pilot_256_step_160"] = {
            "run_id": "E03-A6-F4-PILOT-EXPANDED-20261106-01",
            "count": metric.get("fixed_denominator_count"),
            "ids_sha256": pilot["cohort_ids_sha256"],
            "emotion_accuracy": metric.get("emotion_accuracy_fixed_denominator"),
            "emotion_prediction_counts": metric.get("emotion_prediction_counts"),
            "profile_tp": profile.get("overall_33_label_tp"),
            "profile_fp": profile.get("overall_33_label_fp"),
            "profile_fn": profile.get("overall_33_label_fn"),
            "profile_micro_f1": profile.get("overall_33_label_micro_f1"),
            "nonempty_profile_prediction_count": metric.get("nonempty_profile_prediction_count"),
            "status_counts": metric.get("status_counts"),
        }
    if full_b_path.is_file():
        baseline = json.loads(full_b_path.read_text(encoding="utf-8"))
        identity = baseline.get("selection", {})
        metrics = baseline.get("fixed_denominator_profile_metrics", {})
        if identity.get("selected_ids_sha256") != "d3c84beb6227e87c0fd09e425619635cd9fc6bac5b087e3f1010fb089f4ff8d2" or identity.get("selected_count") != 224:
            raise ValueError("historical E01-B Fast baseline does not match the frozen 224 cohort")
        result["e01_b_full_224"] = {
            "run_id": "E03-PROFILE-FASTONLY-224-20261105-R2",
            "count": baseline.get("sample_count"),
            "ids_sha256": identity["selected_ids_sha256"],
            "emotion_accuracy": baseline.get("emotion_accuracy_fixed_denominator"),
            "profile_tp": metrics.get("overall_33_label_tp"),
            "profile_fp": metrics.get("overall_33_label_fp"),
            "profile_fn": metrics.get("overall_33_label_fn"),
            "profile_micro_precision": metrics.get("overall_33_label_micro_precision"),
            "profile_micro_recall": metrics.get("overall_33_label_micro_recall"),
            "profile_micro_f1": metrics.get("overall_33_label_micro_f1"),
            "emotion_prediction_counts": baseline.get("emotion_prediction_counts"),
            "source_adapter_weight_sha256": baseline.get("run_identity", {}).get("adapter_weight_sha256"),
        }
    return result
