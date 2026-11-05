"""CPU-only helpers for aligning profile predictions from separate pipelines."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.evaluation.metrics import profile_metrics
from src.evaluation.source_integrity import unique_index


PROFILE_DIMENSIONS = ("personality_traits", "interests", "style")
PROFILE_PREDICTION_STATUSES = {"valid", "invalid", "incomplete", "unsupported_order", "budget_exhausted"}


def empty_profile() -> dict[str, list[str]]:
    return {dimension: [] for dimension in PROFILE_DIMENSIONS}


def validate_profile(
    profile: Mapping[str, Any], enums: Mapping[str, Sequence[str]], *, label: str
) -> dict[str, list[str]]:
    if not isinstance(profile, Mapping) or set(profile) != set(PROFILE_DIMENSIONS):
        raise ValueError(f"{label} must contain exactly the three profile dimensions")
    normalized: dict[str, list[str]] = {}
    for dimension in PROFILE_DIMENSIONS:
        values = profile[dimension]
        allowed = enums.get(dimension)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or not isinstance(allowed, Sequence)
            or isinstance(allowed, (str, bytes))
            or not set(values) <= set(allowed)
        ):
            raise ValueError(f"{label} has invalid labels for {dimension}")
        normalized[dimension] = list(values)
    return normalized


def profile_difference_dimensions(
    left: Mapping[str, Sequence[str]], right: Mapping[str, Sequence[str]]
) -> list[str]:
    return [
        dimension
        for dimension in PROFILE_DIMENSIONS
        if set(left[dimension]) != set(right[dimension])
    ]


def classify_historical_difference(
    *,
    submission_profile: Mapping[str, Sequence[str]],
    prefix_profile: Mapping[str, Sequence[str]],
    prefix_status: str,
    full_json_status: str,
) -> str:
    """Classify only the observable S/R mismatch, without inspecting reply text."""
    if not profile_difference_dimensions(submission_profile, prefix_profile):
        return "profile_match"
    if prefix_status != "valid":
        return "prefix_unusable_submission_differs"
    if full_json_status != "valid":
        return "downstream_full_json_parse_failure"
    return "submission_normalization_or_mapping_difference"


def compare_profile_lists(
    *,
    ids: Sequence[str],
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    left_profiles: Sequence[Mapping[str, Sequence[str]]],
    right_profiles: Sequence[Mapping[str, Sequence[str]]],
    enums: dict[str, list[str]],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    if not (len(ids) == len(gold_profiles) == len(left_profiles) == len(right_profiles)):
        raise ValueError("IDs, Gold, and paired prediction counts must match")
    if len(ids) != len(set(ids)):
        raise ValueError("paired profile comparison contains duplicate IDs")

    gold = [validate_profile(profile, enums, label="Gold profile") for profile in gold_profiles]
    left = [validate_profile(profile, enums, label=f"{left_name} profile") for profile in left_profiles]
    right = [validate_profile(profile, enums, label=f"{right_name} profile") for profile in right_profiles]
    left_metrics = profile_metrics(gold, left, enums)
    right_metrics = profile_metrics(gold, right, enums)
    changed_by_dimension = {
        dimension: sum(
            set(a[dimension]) != set(b[dimension]) for a, b in zip(left, right)
        )
        for dimension in PROFILE_DIMENSIONS
    }
    changed_ids = [
        sample_id
        for sample_id, a, b in zip(ids, left, right)
        if profile_difference_dimensions(a, b)
    ]
    metric_keys = (
        "overall_33_label_micro_precision",
        "overall_33_label_micro_recall",
        "overall_33_label_micro_f1",
        "overall_33_label_macro_f1",
        "overall_33_label_tp",
        "overall_33_label_fp",
        "overall_33_label_fn",
    )
    deltas = {
        key: right_metrics[key] - left_metrics[key]
        for key in metric_keys
    }
    per_dimension_deltas = {
        dimension: {
            key: right_metrics["per_dimension"][dimension][key]
            - left_metrics["per_dimension"][dimension][key]
            for key in ("tp", "fp", "fn", "local_micro_f1")
        }
        for dimension in PROFILE_DIMENSIONS
    }
    return {
        "sample_count": len(ids),
        "different_sample_count": len(changed_ids),
        "different_sample_ids": changed_ids,
        "different_sample_count_by_dimension": changed_by_dimension,
        "left_name": left_name,
        "right_name": right_name,
        "left_metrics": left_metrics,
        "right_metrics": right_metrics,
        "right_minus_left": deltas,
        "per_dimension_right_minus_left": per_dimension_deltas,
    }


def index_profile_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
    enums: dict[str, list[str]],
    *,
    require_all_valid: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Validate a minimal Fast-only JSONL artifact and enforce exact ID coverage."""
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected Fast-only IDs contain duplicates")
    indexed = unique_index(rows, "id", "Fast-only profile predictions")
    if set(indexed) != set(expected_ids) or len(rows) != len(expected_ids):
        raise ValueError("Fast-only profile prediction IDs or row count differ from the frozen cohort")
    counts = {status: 0 for status in sorted(PROFILE_PREDICTION_STATUSES)}
    for sample_id, row in indexed.items():
        status = row.get("status")
        if status not in PROFILE_PREDICTION_STATUSES:
            raise ValueError(f"Fast-only prediction {sample_id} has unsupported status")
        counts[status] += 1
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"Fast-only prediction {sample_id} has no reason")
        if status == "valid":
            if set(row) != {"id", "status", "emotion_label", "user_profile", "reason"}:
                raise ValueError(f"valid Fast-only prediction {sample_id} has unexpected fields")
            emotion = row.get("emotion_label")
            if not isinstance(emotion, str) or emotion not in enums["emotion_label"]:
                raise ValueError(f"Fast-only prediction {sample_id} has an invalid emotion label")
            validate_profile(row.get("user_profile"), enums, label=f"Fast-only prediction {sample_id}")
        elif set(row) != {"id", "status", "reason"}:
            raise ValueError(f"failed Fast-only prediction {sample_id} must not contain a fabricated profile")
    if require_all_valid and counts["valid"] != len(expected_ids):
        raise ValueError("strict Fast-only quality comparison requires every frozen ID to be valid")
    return indexed, counts
