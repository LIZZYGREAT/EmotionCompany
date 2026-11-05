"""Strictly recognize the generated JSON prefix ending at ``user_profile``.

This module has no model, tokenizer, Torch, or project-data dependencies. It is
safe to use from a development-only generation stopper and from CPU audits of
already completed Raw outputs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


DIMENSIONS = ("personality_traits", "interests", "style")
EXPECTED_TOP_LEVEL = ("emotion_label", "user_profile")


@dataclass(frozen=True)
class PrefixResult:
    status: str
    reason: str
    boundary: int | None = None
    emotion_label: str | None = None
    user_profile: Mapping[str, list[str]] | None = None

    @property
    def valid(self) -> bool:
        return self.status == "valid"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=reject_constant,
    )


def _validate_profile_object(
    value: Any,
    enums: Mapping[str, list[str]],
) -> tuple[str | None, dict[str, list[str]] | None, str | None]:
    if not isinstance(value, dict):
        return None, None, "profile_not_object"
    if set(value) != set(DIMENSIONS):
        return None, None, "profile_dimensions_mismatch"
    profile: dict[str, list[str]] = {}
    for dimension in DIMENSIONS:
        labels = value[dimension]
        allowed = enums.get(dimension)
        if not isinstance(allowed, list):
            return None, None, f"missing_enum:{dimension}"
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            return None, None, f"profile_dimension_not_string_array:{dimension}"
        if len(labels) != len(set(labels)):
            return None, None, f"duplicate_profile_label:{dimension}"
        if not set(labels) <= set(allowed):
            return None, None, f"unknown_profile_label:{dimension}"
        profile[dimension] = labels
    return None, profile, None


def _validate_candidate(candidate: str, enums: Mapping[str, list[str]], boundary: int) -> PrefixResult:
    try:
        value = _strict_loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        return PrefixResult("invalid", f"invalid_candidate_json:{exc}", boundary=boundary)
    if not isinstance(value, dict) or tuple(value.keys()) != EXPECTED_TOP_LEVEL:
        return PrefixResult("unsupported_order", "candidate_top_level_order", boundary=boundary)
    emotion = value.get("emotion_label")
    allowed_emotions = enums.get("emotion_label")
    if not isinstance(emotion, str):
        return PrefixResult("invalid", "emotion_label_not_string", boundary=boundary)
    if not isinstance(allowed_emotions, list) or emotion not in allowed_emotions:
        return PrefixResult("invalid", "unknown_emotion_label", boundary=boundary)
    _unused, profile, reason = _validate_profile_object(value.get("user_profile"), enums)
    if reason:
        return PrefixResult("invalid", reason, boundary=boundary)
    return PrefixResult("valid", "profile_prefix_closed", boundary, emotion, profile)


def parse_profile_prefix(text: str, enums: Mapping[str, list[str]]) -> PrefixResult:
    """Return a valid parsed profile only at the closed second top-level field.

    The supported order is exactly ``emotion_label``, then ``user_profile``.
    The root object may continue after the returned boundary. Any text before
    the root, field reordering, malformed JSON structure, duplicate key, or
    invalid enum value is rejected instead of repaired.
    """
    if not isinstance(text, str):
        return PrefixResult("invalid", "text_not_string")

    length = len(text)
    root_start: int | None = None
    stack: list[str] = []
    state = "before_root"
    key_start: int | None = None
    active_key: str | None = None
    seen_keys: set[str] = set()
    top_level_keys: list[str] = []
    value_start: int | None = None
    string_mode: str | None = None
    string_start = 0
    in_string = False
    escaped = False
    index = 0

    def incomplete(reason: str = "truncated_prefix") -> PrefixResult:
        return PrefixResult("incomplete", reason)

    def finish_root_value(end: int) -> PrefixResult | None:
        nonlocal state, active_key, value_start
        assert active_key is not None and value_start is not None
        value_text = text[value_start:end]
        try:
            parsed = _strict_loads(value_text)
        except (json.JSONDecodeError, ValueError) as exc:
            return PrefixResult("invalid", f"invalid_top_level_value:{exc}", end)
        if active_key == "emotion_label":
            allowed = enums.get("emotion_label")
            if not isinstance(parsed, str):
                return PrefixResult("invalid", "emotion_label_not_string", end)
            if not isinstance(allowed, list) or parsed not in allowed:
                return PrefixResult("invalid", "unknown_emotion_label", end)
            state = "after_value"
            active_key = None
            value_start = None
            return None
        if active_key != "user_profile":
            return PrefixResult("unsupported_order", "unsupported_top_level_field", end)
        if not value_text.lstrip().startswith("{"):
            return PrefixResult("invalid", "profile_not_object", end)
        assert root_start is not None
        candidate = text[root_start:end] + "}"
        return _validate_candidate(candidate, enums, end)

    while index < length:
        char = text[index]
        if root_start is None:
            if char.isspace():
                index += 1
                continue
            if char != "{":
                return PrefixResult("invalid", "prefix_does_not_start_with_json_object", index)
            root_start = index
            stack.append("{")
            state = "key_or_end"
            index += 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if string_mode == "top_key":
                    key_start = string_start
                    try:
                        key = _strict_loads(text[string_start:index + 1])
                    except (json.JSONDecodeError, ValueError) as exc:
                        return PrefixResult("invalid", f"invalid_top_level_key:{exc}", index + 1)
                    if not isinstance(key, str):
                        return PrefixResult("invalid", "top_level_key_not_string", index + 1)
                    if key in seen_keys:
                        return PrefixResult("invalid", "duplicate_top_level_key", index + 1)
                    seen_keys.add(key)
                    expected_index = len(top_level_keys)
                    if expected_index >= len(EXPECTED_TOP_LEVEL):
                        return PrefixResult("unsupported_order", "extra_top_level_field", index + 1)
                    expected_key = EXPECTED_TOP_LEVEL[expected_index]
                    if key != expected_key:
                        if key in EXPECTED_TOP_LEVEL or expected_index < len(EXPECTED_TOP_LEVEL):
                            return PrefixResult("unsupported_order", "unsupported_top_level_order", index + 1)
                    top_level_keys.append(key)
                    active_key = key
                    state = "colon"
                elif string_mode == "root_string_value":
                    result = finish_root_value(index + 1)
                    if result is not None:
                        return result
                string_mode = None
            index += 1
            continue

        if len(stack) == 1:
            if state in {"key_or_end", "key"}:
                if char.isspace():
                    index += 1
                    continue
                if char == "}" and state == "key_or_end":
                    return incomplete("profile_field_missing")
                if char != '"':
                    return PrefixResult("invalid", "expected_top_level_key", index)
                in_string = True
                string_mode = "top_key"
                string_start = index
                index += 1
                continue
            if state == "colon":
                if char.isspace():
                    index += 1
                    continue
                if char != ":":
                    return PrefixResult("invalid", "expected_colon_after_top_level_key", index)
                state = "value"
                index += 1
                continue
            if state == "value":
                if char.isspace():
                    index += 1
                    continue
                value_start = index
                if char == '"':
                    in_string = True
                    string_mode = "root_string_value"
                    string_start = index
                    index += 1
                    continue
                if char in "{[":
                    stack.append(char)
                    state = "nested_value"
                    index += 1
                    continue
                if char in "]},":
                    return PrefixResult("invalid", "empty_or_missing_top_level_value", index)
                state = "scalar_value"
                index += 1
                continue
            if state == "scalar_value":
                if char in ",}":
                    result = finish_root_value(index)
                    if result is not None:
                        return result
                    continue
                index += 1
                continue
            if state == "after_value":
                if char.isspace():
                    index += 1
                    continue
                if char == ",":
                    state = "key"
                    index += 1
                    continue
                if char == "}":
                    return incomplete("profile_field_missing")
                return PrefixResult("invalid", "expected_comma_after_top_level_value", index)

        # Nested arrays and objects are tracked outside strings. A value is
        # complete when its matching delimiter returns the stack to the root.
        if char == '"':
            in_string = True
            string_mode = "nested"
            string_start = index
            index += 1
            continue
        if char in "{[":
            stack.append(char)
            index += 1
            continue
        if char in "}]":
            if not stack or (char == "}" and stack[-1] != "{") or (char == "]" and stack[-1] != "["):
                return PrefixResult("invalid", "mismatched_json_delimiter", index)
            stack.pop()
            index += 1
            if len(stack) == 1 and state == "nested_value":
                result = finish_root_value(index)
                if result is not None:
                    return result
            elif not stack:
                return incomplete("profile_field_missing")
            continue
        index += 1

    if in_string:
        return incomplete("truncated_string")
    if state == "scalar_value" and value_start is not None:
        result = finish_root_value(length)
        if result is not None:
            return result
    if len(top_level_keys) < len(EXPECTED_TOP_LEVEL):
        return incomplete("profile_field_missing")
    return incomplete()


class ProfilePrefixStopper:
    """Inspect actual cumulative generated token IDs and stop at a valid prefix.

    ``decode`` must use the same tokenizer and decoding options as the complete
    generation path. A deadline can stop a run, but never marks its prefix valid.
    """

    def __init__(
        self,
        decode: Callable[[Sequence[int]], str],
        enums: Mapping[str, list[str]],
        *,
        deadline: float | None = None,
        stop_on_rejection: bool = False,
    ) -> None:
        self.decode = decode
        self.enums = enums
        self.deadline = deadline
        self.stop_on_rejection = stop_on_rejection
        self.result: PrefixResult | None = None
        self.generated_token_ids: tuple[int, ...] | None = None
        self.stop_reason: str | None = None

    def inspect(self, generated_token_ids: Sequence[int]) -> bool:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop_reason = "budget_exhausted"
            self.result = PrefixResult("incomplete", "budget_exhausted")
            return True
        current_ids = tuple(int(value) for value in generated_token_ids)
        parsed = parse_profile_prefix(self.decode(current_ids), self.enums)
        if parsed.valid:
            self.generated_token_ids = current_ids
            self.stop_reason = "profile_prefix_closed"
            self.result = parsed
            return True
        if self.stop_on_rejection and parsed.status in {"invalid", "unsupported_order"}:
            self.generated_token_ids = current_ids
            self.stop_reason = "profile_prefix_rejected"
            self.result = parsed
            return True
        self.result = parsed
        return False

    def finish(self, stop_reason: str, generated_token_ids: Sequence[int]) -> PrefixResult:
        """Classify EOS/token-cap completion when the prefix stopper did not fire."""
        if self.stop_reason == "budget_exhausted":
            return self.result or PrefixResult("incomplete", "budget_exhausted")
        if self.result is not None and self.result.valid:
            return self.result
        if stop_reason == "max_new_tokens":
            return PrefixResult("incomplete", "token_limit")
        if stop_reason == "eos":
            return PrefixResult("incomplete", "early_eos")
        if not generated_token_ids:
            return PrefixResult("incomplete", "no_generated_tokens")
        if self.result is not None:
            return self.result
        return PrefixResult("incomplete", "generation_ended_before_profile")
