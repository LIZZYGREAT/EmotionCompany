"""Fixed-session A8 Replay validation splits and predeclared quality gates."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.e03_a8_sft import canonical_sha256
from src.evaluation.e03_a8_staged_metrics import score_a8_cohort
from src.shared.e03_a8_protocol import PROFILE_DIMS


def _id_hash(values: Sequence[str]) -> str:
    return canonical_sha256(list(values))


def freeze_val_splits(
    inputs: Sequence[Mapping[str, Any]],
    id_map: Mapping[str, Mapping[str, Any]],
    historical_conversations: set[str],
    *,
    seed: int = 42,
    gate_fraction: float = 0.25,
) -> dict[str, Any]:
    """Split by conversation IDs without consulting Val labels."""
    input_ids = [str(row.get("id", "")) for row in inputs]
    if not input_ids or len(input_ids) != len(set(input_ids)):
        raise ValueError("target-free Val inputs must have unique non-empty IDs")
    if set(input_ids) != set(id_map):
        raise ValueError("target-free Val IDs and frozen ID map differ")
    if not 0.0 < gate_fraction < 1.0:
        raise ValueError("gate_fraction must be strictly between zero and one")
    conversation_by_id: dict[str, str] = {}
    for eval_id in input_ids:
        mapping = id_map[eval_id]
        conversation_id = mapping.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"Val ID map has no conversation for {eval_id}")
        conversation_by_id[eval_id] = conversation_id
    observed_conversations = set(conversation_by_id.values())
    if not historical_conversations <= observed_conversations:
        raise ValueError("historical A2 conversation set includes an unknown Val conversation")
    remaining = observed_conversations - historical_conversations
    ordered = sorted(
        remaining,
        key=lambda conversation_id: hashlib.sha256(f"{seed}|{conversation_id}".encode("utf-8")).hexdigest(),
    )
    gate_count = max(1, min(len(ordered) - 1, math.ceil(len(ordered) * gate_fraction))) if len(ordered) > 1 else 0
    gate_conversations = set(ordered[:gate_count])
    confirm_conversations = set(ordered[gate_count:])
    if not gate_conversations or not confirm_conversations:
        raise ValueError("Val split cannot provide non-empty gate and confirm conversation sets")
    assignments = {
        "historical_exposed": historical_conversations,
        "val_gate": gate_conversations,
        "val_confirm": confirm_conversations,
    }
    cohorts = {}
    for name, conversations in assignments.items():
        ids = [eval_id for eval_id in input_ids if conversation_by_id[eval_id] in conversations]
        cohort_map = {eval_id: {"conversation_id": conversation_by_id[eval_id]} for eval_id in ids}
        cohorts[name] = {
            "name": name,
            "ids": ids,
            "ids_sha256": _id_hash(ids),
            "conversation_ids": sorted(conversations),
            "conversation_ids_sha256": _id_hash(sorted(conversations)),
            "conversation_count": len(conversations),
            "mapping": cohort_map,
        }
    if gate_conversations & confirm_conversations or historical_conversations & (gate_conversations | confirm_conversations):
        raise AssertionError("frozen Val conversation splits overlap")
    return {
        "version": "E03-A8-replay-val-splits-v1",
        "seed": seed,
        "gate_fraction": gate_fraction,
        "historical_conversation_count": len(historical_conversations),
        "historical_conversations_sha256": _id_hash(sorted(historical_conversations)),
        "target_free_ids_sha256": _id_hash(input_ids),
        "cohorts": cohorts,
    }


def serialize_val_splits(splits: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime-only maps while preserving frozen IDs and hashes."""
    cohorts = {}
    for name, cohort in splits["cohorts"].items():
        cohorts[name] = {key: value for key, value in cohort.items() if key != "mapping"}
    return {**{key: value for key, value in splits.items() if key != "cohorts"}, "cohorts": cohorts}


def gate_reasons(
    parent_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    minimum_micro_f1_delta: float = 0.003,
    maximum_precision_drop: float = 0.02,
    maximum_dimension_f1_drop: float = 0.02,
    minimum_strict_parse_rate: float = 0.99,
    minimum_eos_rate: float = 0.99,
    maximum_new_fp_ratio: float = 1.5,
    engineering_passed: bool = True,
) -> dict[str, Any]:
    parent = parent_metrics["profile_33_label"]
    candidate = candidate_metrics["profile_33_label"]
    delta_tp = int(candidate["overall_33_label_tp"]) - int(parent["overall_33_label_tp"])
    delta_fp = int(candidate["overall_33_label_fp"]) - int(parent["overall_33_label_fp"])
    micro_f1_delta = float(candidate["overall_33_label_micro_f1"]) - float(parent["overall_33_label_micro_f1"])
    precision_delta = float(candidate["overall_33_label_micro_precision"]) - float(parent["overall_33_label_micro_precision"])
    dimensions = {
        dimension: {
            "parent_f1": float(parent["per_dimension"][dimension]["local_micro_f1"]),
            "candidate_f1": float(candidate["per_dimension"][dimension]["local_micro_f1"]),
            "delta_f1": float(candidate["per_dimension"][dimension]["local_micro_f1"]) - float(parent["per_dimension"][dimension]["local_micro_f1"]),
        }
        for dimension in PROFILE_DIMS
    }
    reasons = []
    if not engineering_passed:
        reasons.append("engineering_gate_failed")
    if float(candidate_metrics.get("strict_parse_rate", 0.0)) < minimum_strict_parse_rate:
        reasons.append("strict_parse_rate_below_99_percent")
    if float(candidate_metrics.get("eos_completion_rate", 0.0)) < minimum_eos_rate:
        reasons.append("eos_rate_below_99_percent")
    if micro_f1_delta < minimum_micro_f1_delta:
        reasons.append("profile_micro_f1_gain_below_0_003")
    if delta_tp <= 0:
        reasons.append("true_positive_delta_not_positive")
    if precision_delta < -maximum_precision_drop:
        reasons.append("profile_precision_drop_exceeds_0_02")
    for dimension, facts in dimensions.items():
        if facts["delta_f1"] < -maximum_dimension_f1_drop:
            reasons.append(f"{dimension}_f1_drop_exceeds_0_02")
    if delta_fp > maximum_new_fp_ratio * max(delta_tp, 1):
        reasons.append("new_false_positive_ratio_exceeds_1_5")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "delta_tp": delta_tp,
        "delta_fp": delta_fp,
        "profile_micro_f1_delta": micro_f1_delta,
        "profile_precision_delta": precision_delta,
        "dimension_f1": dimensions,
        "strict_parse_rate": float(candidate_metrics.get("strict_parse_rate", 0.0)),
        "eos_completion_rate": float(candidate_metrics.get("eos_completion_rate", 0.0)),
        "thresholds": {
            "minimum_micro_f1_delta": minimum_micro_f1_delta,
            "maximum_precision_drop": maximum_precision_drop,
            "maximum_dimension_f1_drop": maximum_dimension_f1_drop,
            "minimum_strict_parse_rate": minimum_strict_parse_rate,
            "minimum_eos_rate": minimum_eos_rate,
            "maximum_new_fp_ratio": maximum_new_fp_ratio,
        },
    }


def score_profile_predictions(
    cohort: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    enums: Mapping[str, list[str]],
) -> dict[str, Any]:
    if set(cohort["ids"]) - set(references) or set(cohort["ids"]) - set(predictions):
        raise ValueError("Replay Val cohort is missing a frozen reference or prediction")
    return score_a8_cohort(cohort, references, predictions, dict(enums), arm="P")
