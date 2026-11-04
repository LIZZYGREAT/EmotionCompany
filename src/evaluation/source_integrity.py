"""CPU-only identity checks for target-free evaluation inputs and Dev splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def unique_index(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row {row_number} has a missing/non-string {key}")
        if value in indexed:
            raise ValueError(f"{label} contains a duplicate {key}")
        indexed[value] = row
    return indexed


def validate_target_free_inputs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Require the exact ``id``/``history`` shape; references cannot be nested."""
    indexed = unique_index(rows, "id", "generation input")
    for sample_id, row in indexed.items():
        if set(row) != {"id", "history"}:
            raise ValueError(f"generation input {sample_id} must contain only id and history")
        history = row["history"]
        if not isinstance(history, list) or not history:
            raise ValueError(f"generation input {sample_id} has an empty/invalid history")
        for index, message in enumerate(history):
            if (
                not isinstance(message, Mapping)
                or set(message) != {"role", "content"}
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(f"generation input {sample_id} history[{index}] is invalid")
        if history[-1]["role"] != "user":
            raise ValueError(f"generation input {sample_id} omits the current user turn")
    return indexed


def _typed_id(value: Any) -> str:
    return f"{type(value).__name__}:{json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"


def validate_history_matches_turns(
    source_turns: Sequence[Mapping[str, Any]],
    local_history: Sequence[Mapping[str, Any]],
    target_user_turn_id: Any,
) -> int:
    """Check exact role/content order through the referenced target user turn.

    Turn IDs are matched explicitly; no odd/even role inference is used.
    Returns the zero-based target position in ``source_turns``.
    """
    if not isinstance(source_turns, Sequence) or isinstance(source_turns, (str, bytes)):
        raise ValueError("source turns must be a sequence")
    if not isinstance(local_history, Sequence) or isinstance(local_history, (str, bytes)):
        raise ValueError("local history must be a sequence")
    matches = [
        index for index, turn in enumerate(source_turns)
        if isinstance(turn, Mapping) and "turn_id" in turn
        and _typed_id(turn["turn_id"]) == _typed_id(target_user_turn_id)
    ]
    if len(matches) != 1:
        raise ValueError("target_user_turn_id is missing or ambiguous")
    target_index = matches[0]
    target_turn = source_turns[target_index]
    if target_turn.get("role") != "user":
        raise ValueError("target_user_turn_id does not reference a user turn")
    expected = source_turns[:target_index + 1]
    if len(local_history) != len(expected):
        raise ValueError("local history omits or adds source turns before the target user")
    for index, (source, local) in enumerate(zip(expected, local_history)):
        if (
            not isinstance(source, Mapping)
            or not isinstance(local, Mapping)
            or source.get("role") != local.get("role")
            or source.get("content") != local.get("content")
            or not isinstance(source.get("content"), str)
        ):
            raise ValueError(f"local history differs from source role/content/order at turn {index}")
    if not local_history or local_history[-1].get("role") != "user":
        raise ValueError("local history does not end with the referenced current user")
    return target_index


def audit_prediction_point_sources(
    normalized_rows: Sequence[Mapping[str, Any]],
    raw_source_rows: Sequence[Mapping[str, Any]] | Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify D0.3 histories against referenced Train turns before generation."""
    indexed = unique_index(normalized_rows, "sample_id", "normalized prediction points")
    checked_lines: set[int] = set()
    conversation_ids: set[str] = set()
    for sample_id, row in indexed.items():
        source = row.get("source")
        if not isinstance(source, Mapping) or source.get("split") != "train":
            raise ValueError(f"prediction point {sample_id} lacks a Train source reference")
        source_line = source.get("source_line")
        point_index = source.get("prediction_point_index")
        if (
            isinstance(source_line, bool) or not isinstance(source_line, int) or source_line < 1
            or isinstance(point_index, bool) or not isinstance(point_index, int) or point_index < 0
        ):
            raise ValueError(f"prediction point {sample_id} has invalid source coordinates")
        if isinstance(raw_source_rows, Mapping):
            raw = raw_source_rows.get(source_line)
        else:
            raw = raw_source_rows[source_line - 1] if source_line <= len(raw_source_rows) else None
        if raw is None:
            raise ValueError(f"prediction point {sample_id} references a missing source line")
        if raw.get("conversation_id") != row.get("conversation_id"):
            raise ValueError(f"prediction point {sample_id} conversation ID differs from source")
        points = raw.get("prediction_points")
        turns = raw.get("turns")
        if not isinstance(points, list) or point_index >= len(points):
            raise ValueError(f"prediction point {sample_id} references a missing source point")
        point = points[point_index]
        if not isinstance(point, Mapping):
            raise ValueError(f"prediction point {sample_id} source point is not an object")
        target_turn_id = source.get("target_user_turn_id")
        if _typed_id(point.get("target_user_turn_id")) != _typed_id(target_turn_id):
            raise ValueError(f"prediction point {sample_id} target user turn ID differs from source")
        actual_target_index = validate_history_matches_turns(
            turns if isinstance(turns, Sequence) else [],
            row.get("history") if isinstance(row.get("history"), Sequence) else [],
            target_turn_id,
        )
        recorded_target_index = source.get("target_turn_index")
        if recorded_target_index != actual_target_index:
            raise ValueError(f"prediction point {sample_id} recorded target turn position differs")
        checked_lines.add(source_line)
        conversation_ids.add(str(row.get("conversation_id")))
    ids = sorted(indexed)
    return {
        "status": "passed",
        "point_count": len(indexed),
        "conversation_count": len(conversation_ids),
        "sample_ids_sha256": canonical_sha256(ids),
        "conversation_ids_sha256": canonical_sha256(sorted(conversation_ids)),
        "source_line_count": len(checked_lines),
        "source_line_min": min(checked_lines) if checked_lines else None,
        "source_line_max": max(checked_lines) if checked_lines else None,
        "current_user_checked_by_turn_id": True,
        "role_content_order_checked": True,
    }


def assign_dev_cluster_partitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    holdout_percent: int = 20,
) -> dict[str, str]:
    """Assign each whole conversation to ``dev`` or ``holdout`` deterministically."""
    if not 1 <= holdout_percent <= 99:
        raise ValueError("holdout_percent must be between 1 and 99")
    by_conversation: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        sample_id, conversation_id = row.get("sample_id"), row.get("conversation_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise ValueError(f"row {row_number} has a missing or duplicate sample_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"row {row_number} has a missing conversation_id")
        seen_ids.add(sample_id)
        by_conversation[conversation_id].append(sample_id)
    assignment = {
        conversation_id: (
            "holdout"
            if int(hashlib.sha256(f"{seed}:{conversation_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
            < holdout_percent
            else "dev"
        )
        for conversation_id in by_conversation
    }
    return {
        sample_id: assignment[conversation_id]
        for conversation_id, sample_ids in by_conversation.items()
        for sample_id in sample_ids
    }


def verify_prompt_token_ids(actual_ids: Sequence[int], expected_ids: Sequence[int]) -> str:
    """Require byte-for-byte-equivalent token ID sequences, not just equal lengths."""
    actual = [int(value) for value in actual_ids]
    expected = [int(value) for value in expected_ids]
    if len(actual) != len(expected):
        raise ValueError("prompt token count differs")
    if actual != expected:
        raise ValueError("prompt token IDs differ despite matching token counts")
    return canonical_sha256(actual)


def build_source_manifest(
    *,
    source_path: str,
    source_sha256: str,
    input_rows: Sequence[Mapping[str, Any]],
    id_map_rows: Sequence[Mapping[str, Any]],
    prompt_version: str,
    prompt_sha256: str,
    format_version: str,
    truncation_version: str,
    exporter_source_sha256: str,
    selection_method: str | None = None,
    selection_seed: int | None = None,
    repaired: bool = False,
) -> dict[str, Any]:
    inputs = validate_target_free_inputs(input_rows)
    id_map = unique_index(id_map_rows, "eval_id", "ID map")
    if set(inputs) != set(id_map):
        raise ValueError("generation input and ID map ID sets differ")
    selected_ids = sorted(inputs)
    conversation_ids = []
    message_counts = []
    for eval_id in selected_ids:
        mapping = id_map[eval_id]
        conversation_id = mapping.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"ID map {eval_id} lacks a conversation_id")
        conversation_ids.append(conversation_id)
        message_counts.append(len(inputs[eval_id]["history"]))
    return {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "selected_point_count": len(selected_ids),
        "selected_point_ids_sha256": canonical_sha256(selected_ids),
        "selected_conversation_count": len(set(conversation_ids)),
        "selected_conversation_ids_sha256": canonical_sha256(sorted(set(conversation_ids))),
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "format_version": format_version,
        "truncation_version": truncation_version,
        "selection_method": selection_method,
        "selection_seed": selection_seed,
        "selected_history_message_count_min": min(message_counts) if message_counts else None,
        "selected_history_message_count_max": max(message_counts) if message_counts else None,
        "history_repaired": bool(repaired),
        "exporter_source_sha256": exporter_source_sha256,
        "prompt_token_ids_sha256": None,
        "prompt_token_ids_sha256_status": "not_required_for_text_identity",
    }
