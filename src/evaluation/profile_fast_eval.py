"""CPU helpers shared by the isolated E03 fast profile evaluator."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from src.evaluation.source_integrity import canonical_sha256, unique_index, validate_target_free_inputs


def select_unique_conversation_samples(
    input_rows: Sequence[Mapping[str, Any]],
    id_map_rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select one point from each session using only IDs and fixed seed."""
    if isinstance(count, bool) or count < 1 or count > 32:
        raise ValueError("fast profile selection count must be between 1 and 32")
    inputs = validate_target_free_inputs(input_rows)
    id_map = unique_index(id_map_rows, "eval_id", "ID map")
    if set(inputs) != set(id_map):
        raise ValueError("generation input and ID map ID sets differ")
    by_conversation: dict[str, list[str]] = defaultdict(list)
    for eval_id, mapping in id_map.items():
        conversation_id = mapping.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"ID map {eval_id} has no conversation_id")
        by_conversation[conversation_id].append(eval_id)
    if len(by_conversation) < count:
        raise ValueError("not enough unique conversations for the fixed fast-eval sample count")

    ranked: list[tuple[str, str, str]] = []
    for conversation_id, eval_ids in by_conversation.items():
        representative = min(
            eval_ids,
            key=lambda eval_id: hashlib.sha256(f"{seed}:point:{eval_id}".encode("utf-8")).hexdigest(),
        )
        rank = hashlib.sha256(f"{seed}:conversation:{conversation_id}".encode("utf-8")).hexdigest()
        ranked.append((rank, conversation_id, representative))
    ranked.sort()
    return [dict(inputs[eval_id]) for _rank, _conversation, eval_id in ranked[:count]]


def selected_id_sha256(eval_ids: Sequence[str]) -> str:
    """Hash a frozen membership set independently of its presentation order."""
    if any(not isinstance(eval_id, str) or not eval_id for eval_id in eval_ids):
        raise ValueError("frozen evaluation IDs must be non-empty strings")
    if len(eval_ids) != len(set(eval_ids)):
        raise ValueError("frozen evaluation IDs contain duplicates")
    return canonical_sha256(sorted(eval_ids))


def select_frozen_id_samples(
    input_rows: Sequence[Mapping[str, Any]],
    id_map_rows: Sequence[Mapping[str, Any]],
    frozen_selection_rows: Sequence[Mapping[str, Any]],
    *,
    expected_ids_sha256: str,
    expected_count: int,
    expected_conversation_count: int | None = None,
    require_complete_conversations: bool = True,
) -> list[dict[str, Any]]:
    """Select an immutable, target-free ID cohort and verify its session groups."""
    inputs = validate_target_free_inputs(input_rows)
    frozen_inputs = validate_target_free_inputs(frozen_selection_rows)
    id_map = unique_index(id_map_rows, "eval_id", "ID map")
    if set(inputs) != set(id_map):
        raise ValueError("generation input and ID map ID sets differ")
    selected_ids = list(frozen_inputs)
    if isinstance(expected_count, bool) or len(selected_ids) != expected_count or expected_count < 1:
        raise ValueError("frozen selection count differs from its expected count")
    actual_ids_sha256 = selected_id_sha256(selected_ids)
    if actual_ids_sha256 != expected_ids_sha256:
        raise ValueError("frozen selection ID SHA-256 differs from its pinned value")
    missing_ids = set(selected_ids) - set(inputs)
    if missing_ids:
        raise ValueError("frozen selection contains IDs absent from the generation input")
    for eval_id, frozen_row in frozen_inputs.items():
        if dict(inputs[eval_id]) != dict(frozen_row):
            raise ValueError("frozen selection input differs from the full target-free source")

    selected_conversations: set[str] = set()
    all_ids_by_conversation: dict[str, set[str]] = defaultdict(set)
    for eval_id, mapping in id_map.items():
        conversation_id = mapping.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"ID map {eval_id} has no conversation_id")
        all_ids_by_conversation[conversation_id].add(eval_id)
        if eval_id in frozen_inputs:
            selected_conversations.add(conversation_id)
    if expected_conversation_count is not None and len(selected_conversations) != expected_conversation_count:
        raise ValueError("frozen selection conversation count differs from its pinned value")
    if require_complete_conversations:
        incomplete = [
            conversation_id
            for conversation_id in selected_conversations
            if not all_ids_by_conversation[conversation_id] <= set(frozen_inputs)
        ]
        if incomplete:
            raise ValueError("frozen selection splits one or more conversation groups")
    return [dict(inputs[eval_id]) for eval_id in sorted(frozen_inputs)]


def run_profile_fast_only_attempts(
    samples: Sequence[Mapping[str, Any]],
    generate_candidate_once: Any,
    *,
    on_attempt: Any | None = None,
) -> list[dict[str, Any]]:
    """Run one candidate-generation callback per row; never request full output."""
    attempts: list[dict[str, Any]] = []

    def record(attempt: dict[str, Any]) -> None:
        attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(dict(attempt), len(attempts) - 1)

    for index, sample in enumerate(samples):
        try:
            result = generate_candidate_once(sample)
        except KeyboardInterrupt:
            result = {"status": "incomplete", "reason": "interrupted"}
            record({"id": sample.get("id"), **result})
            for pending in samples[index + 1:]:
                record({"id": pending.get("id"), "status": "incomplete", "reason": "interrupted_before_generation"})
            break
        if not isinstance(result, Mapping):
            result = {"status": "invalid", "reason": "invalid_generation_result"}
        record({"id": sample.get("id"), **dict(result)})
        if result.get("status") == "budget_exhausted":
            for pending in samples[index + 1:]:
                record({"id": pending.get("id"), "status": "budget_exhausted", "reason": "run_budget_exhausted"})
            break
        if result.get("status") == "incomplete" and str(result.get("reason", "")).startswith("generation_error:"):
            for pending in samples[index + 1:]:
                record({"id": pending.get("id"), "status": "incomplete", "reason": "run_stopped_after_generation_error"})
            break
    return attempts


def score_profile_attempts(
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    attempts: Sequence[Mapping[str, Any]],
    enums: dict[str, list[str]],
) -> dict[str, Any]:
    """Score every selected row, treating every unusable output as empty."""
    from src.evaluation.metrics import profile_metrics

    if len(gold_profiles) != len(attempts):
        raise ValueError("gold and fast-eval attempt row counts differ")
    dimensions = ("personality_traits", "interests", "style")
    statuses = {"valid": 0, "invalid": 0, "incomplete": 0, "unsupported_order": 0, "budget_exhausted": 0}
    primary_predictions: list[dict[str, list[str]]] = []
    valid_gold: list[dict[str, list[str]]] = []
    valid_predictions: list[dict[str, list[str]]] = []
    reasons: dict[str, int] = defaultdict(int)
    for gold, attempt in zip(gold_profiles, attempts):
        status = attempt.get("status")
        if status not in statuses:
            raise ValueError(f"unsupported fast-eval attempt status: {status!r}")
        statuses[str(status)] += 1
        profile = attempt.get("user_profile")
        if status == "valid":
            if not isinstance(profile, Mapping) or any(not isinstance(profile.get(dim), list) for dim in dimensions):
                raise ValueError("valid fast-eval attempt has no complete profile object")
            normalized = {dim: list(profile[dim]) for dim in dimensions}
            primary_predictions.append(normalized)
            valid_gold.append({dim: list(gold[dim]) for dim in dimensions})
            valid_predictions.append(normalized)
        else:
            primary_predictions.append({dim: [] for dim in dimensions})
        reason = attempt.get("reason")
        if isinstance(reason, str) and reason:
            reasons[reason] += 1
    all_gold = [{dim: list(gold[dim]) for dim in dimensions} for gold in gold_profiles]
    return {
        "score_population": "all_selected_rows_failures_as_empty_profile",
        "fixed_denominator_count": len(attempts),
        "status_counts": statuses,
        "failure_reason_counts": dict(sorted(reasons.items())),
        "profile_metrics": profile_metrics(all_gold, primary_predictions, enums),
        "valid_only_diagnostic": {
            "denominator_count": len(valid_predictions),
            "profile_metrics": profile_metrics(valid_gold, valid_predictions, enums) if valid_predictions else None,
        },
    }


def compare_generated_prefix(
    full_token_ids: Sequence[int],
    fast_token_ids: Sequence[int],
    full_emotion: str | None,
    fast_emotion: str | None,
    full_profile: Mapping[str, Sequence[str]] | None,
    fast_profile: Mapping[str, Sequence[str]] | None,
) -> dict[str, Any]:
    """Compare token IDs before comparing parsed semantic fields."""
    full_ids = [int(token) for token in full_token_ids]
    fast_ids = [int(token) for token in fast_token_ids]
    token_match = len(fast_ids) <= len(full_ids) and full_ids[:len(fast_ids)] == fast_ids
    profile_match = (
        full_profile is not None
        and fast_profile is not None
        and full_emotion == fast_emotion
        and all(list(full_profile.get(key, [])) == list(fast_profile.get(key, []))
                for key in ("personality_traits", "interests", "style"))
    )
    return {
        "token_prefix_match": token_match,
        "emotion_and_profile_match": profile_match,
        "full_generated_token_count": len(full_ids),
        "fast_generated_token_count": len(fast_ids),
        "full_prefix_token_ids_sha256": canonical_sha256(fast_ids) if token_match else None,
    }


def profile_nonempty_recall(
    gold_profiles: Sequence[Mapping[str, Sequence[str]]],
    predicted_profiles: Sequence[Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, int | float]]:
    if len(gold_profiles) != len(predicted_profiles):
        raise ValueError("gold and predicted profile row counts differ")
    result: dict[str, dict[str, int | float]] = {}
    for dimension in ("personality_traits", "interests", "style"):
        support = hits = 0
        for gold, predicted in zip(gold_profiles, predicted_profiles):
            is_nonempty = bool(gold[dimension])
            support += is_nonempty
            hits += is_nonempty and bool(predicted[dimension])
        result[dimension] = {
            "gold_nonempty_count": support,
            "predicted_nonempty_on_gold_nonempty_count": hits,
            "recall": hits / support if support else 0.0,
        }
    return result
