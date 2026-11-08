"""Pairing and fixed-cohort statistics for the E01-B/A8 final Val comparison."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from src.evaluation.metrics import profile_metrics

PROFILE_DIMS = ("personality_traits", "interests", "style")


def prefix_token_ids_match(full_token_ids: Sequence[int], prefix_token_ids: Sequence[int]) -> bool:
    return bool(prefix_token_ids) and list(full_token_ids[:len(prefix_token_ids)]) == list(prefix_token_ids)


def select_stable_ids(ids: Sequence[str], *, count: int, seed: int, purpose: str) -> list[str]:
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("sample IDs must be non-empty and unique")
    if count <= 0 or count > len(ids):
        raise ValueError("selection count is outside the available sample range")
    ranked = sorted(
        ids,
        key=lambda value: hashlib.sha256(f"{purpose}:{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    return ranked[:count]


def profile_hit_transitions(
    gold_by_id: Mapping[str, Mapping[str, Sequence[str]]],
    e01_by_id: Mapping[str, Mapping[str, Sequence[str]]],
    a8_by_id: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    dimensions: Sequence[str] = PROFILE_DIMS,
) -> dict[str, int]:
    ids = set(gold_by_id)
    if set(e01_by_id) != ids or set(a8_by_id) != ids:
        raise ValueError("paired hit comparison requires the same fixed ID set")
    e01_hits: set[tuple[str, str, str]] = set()
    a8_hits: set[tuple[str, str, str]] = set()
    gold_hits: set[tuple[str, str, str]] = set()
    for sample_id in ids:
        for dimension in dimensions:
            gold = set(gold_by_id[sample_id].get(dimension, []))
            e01 = set(e01_by_id[sample_id].get(dimension, []))
            a8 = set(a8_by_id[sample_id].get(dimension, []))
            gold_hits.update((sample_id, dimension, label) for label in gold)
            e01_hits.update((sample_id, dimension, label) for label in gold & e01)
            a8_hits.update((sample_id, dimension, label) for label in gold & a8)
    both = len(e01_hits & a8_hits)
    e01_only = len(e01_hits - a8_hits)
    a8_only = len(a8_hits - e01_hits)
    both_missed = len(gold_hits - (e01_hits | a8_hits))
    if both + e01_only + a8_only + both_missed != len(gold_hits):
        raise RuntimeError("per-sample label hit transitions do not conserve Gold positives")
    return {
        "e01_b_only_hits": e01_only,
        "a8_only_hits": a8_only,
        "both_models_hit": both,
        "both_models_missed": both_missed,
        "gold_positive_sample_label_pairs": len(gold_hits),
    }


def paired_conversation_bootstrap(
    gold_by_id: Mapping[str, Mapping[str, Sequence[str]]],
    e01_by_id: Mapping[str, Mapping[str, Sequence[str]]],
    a8_by_id: Mapping[str, Mapping[str, Sequence[str]]],
    conversation_by_id: Mapping[str, str],
    enums: Mapping[str, list[str]],
    *,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    ids = set(gold_by_id)
    if set(e01_by_id) != ids or set(a8_by_id) != ids or set(conversation_by_id) != ids:
        raise ValueError("conversation bootstrap requires complete paired ID coverage")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")

    by_conversation: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        conversation_id = conversation_by_id[sample_id]
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation bootstrap requires non-empty conversation IDs")
        by_conversation[conversation_id].append(sample_id)
    clusters = sorted(by_conversation)
    rng = random.Random(seed)
    e01_scores: list[float] = []
    a8_scores: list[float] = []
    deltas: list[float] = []

    def score(sampled_ids: list[str], predictions: Mapping[str, Mapping[str, Sequence[str]]]) -> float:
        gold_rows = []
        predicted_rows = []
        for sample_id in sampled_ids:
            gold_rows.append({dim: list(gold_by_id[sample_id].get(dim, [])) for dim in PROFILE_DIMS})
            predicted_rows.append({dim: list(predictions[sample_id].get(dim, [])) for dim in PROFILE_DIMS})
        return float(profile_metrics(gold_rows, predicted_rows, dict(enums))["overall_33_label_micro_f1"])

    for _ in range(replicates):
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        sampled_ids = [sample_id for cluster in sampled_clusters for sample_id in by_conversation[cluster]]
        e01_value = score(sampled_ids, e01_by_id)
        a8_value = score(sampled_ids, a8_by_id)
        e01_scores.append(e01_value)
        a8_scores.append(a8_value)
        deltas.append(e01_value - a8_value)

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [ordered[int(0.025 * len(ordered))], ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]]

    return {
        "method": "paired_conversation_cluster_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "conversation_clusters": len(clusters),
        "e01_b_micro_f1_95_percent_interval": interval(e01_scores),
        "a8_micro_f1_95_percent_interval": interval(a8_scores),
        "e01_b_minus_a8_micro_f1_95_percent_interval": interval(deltas),
        "interval_method": "empirical_percentiles_2_5_and_97_5",
        "exploratory_only": True,
    }
