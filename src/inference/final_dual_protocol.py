"""Frozen prompt construction and output assembly for the final dual-Adapter run."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from src.data.e03_a8_sft import PromptFloorExceeded, select_common_history
from src.shared.e03_a8_protocol import A8ParseError, parse_specialist_output
from src.shared.prompting import (
    PROFILE_DIMS,
    build_messages,
    canonical_json,
    encode_inference_prompt,
    select_recent_complete_suffix,
)


class PromptBuildError(ValueError):
    """A target-free input could not fit the frozen complete-message protocol."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _specialist_schema(official_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project official submission enums into the existing A8 parser contract."""
    definitions = official_schema.get("$defs", {})
    if isinstance(definitions, Mapping) and "emotion" in definitions and "profile" in definitions:
        return dict(official_schema)
    properties = official_schema.get("properties", {})
    emotion_enum = properties.get("emotion_label", {}).get("enum", [])
    profile_properties = properties.get("user_profile", {}).get("properties", {})
    if not isinstance(emotion_enum, list) or not profile_properties:
        raise ValueError("official submission schema does not expose emotion/profile enums")
    return {
        "$defs": {
            "emotion": {"enum": [value for value in emotion_enum if value]},
            "profile": {"properties": profile_properties},
        }
    }


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = canonical_json([int(value) for value in token_ids]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def history_sha256(history: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(list(history)).encode("utf-8")).hexdigest()


def _validate_sample_history(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    sample_id = sample.get("id", "<missing-id>")
    history = sample.get("history")
    if not isinstance(history, list) or not history:
        raise PromptBuildError("invalid_history", f"{sample_id}: history must be a non-empty array")
    if any(not isinstance(message, Mapping) for message in history):
        raise PromptBuildError("invalid_history", f"{sample_id}: history messages must be objects")
    return history


def build_profile_prompt(
    sample: Mapping[str, Any],
    tokenizer: Any,
    *,
    selection_prompt: str,
    profile_prompt: str,
    profile_budget_prompt: str,
    prompt_cap: int = 1280,
) -> dict[str, Any]:
    """Apply P0/T1, then the trained A8 P/EP common-history budget."""
    sample_id = str(sample.get("id", "<missing-id>"))
    history = _validate_sample_history(sample)
    try:
        p0_selection = select_recent_complete_suffix(
            history, selection_prompt, prompt_cap, tokenizer
        )
    except Exception as exc:
        raise PromptBuildError("p0_selection_failed", f"{sample_id}: P0/T1 failed: {exc}") from exc
    if p0_selection["status"] != "ok":
        raise PromptBuildError(
            "p0_prompt_floor_exceeds_cap",
            f"{sample_id}: current-user-only P0 prompt exceeds P={prompt_cap}",
            {"prompt_tokens": p0_selection["prompt_tokens"]},
        )
    try:
        common = select_common_history(
            history,
            p_prompt=profile_prompt,
            ep_prompt=profile_budget_prompt,
            tokenizer=tokenizer,
            initial_start_index=int(p0_selection["selected_start_index"]),
            prompt_cap=prompt_cap,
        )
    except PromptFloorExceeded as exc:
        raise PromptBuildError(
            "a8_prompt_floor_exceeds_cap",
            f"{sample_id}: current-user-only A8 P/EP prompt exceeds P={prompt_cap}",
            {"prompt_lengths": exc.prompt_lengths},
        ) from exc
    except Exception as exc:
        raise PromptBuildError("a8_common_history_failed", f"{sample_id}: A8 history selection failed: {exc}") from exc

    ids = [int(value) for value in common["prompt_ids"]["P"]]
    selected_history = list(common["selected_history"])
    if not ids or len(ids) > prompt_cap:
        raise PromptBuildError("a8_prompt_limit", f"{sample_id}: A8 P prompt violates the frozen token cap")
    selected_turn_ids = [
        message.get("turn_id", index)
        for index, message in enumerate(selected_history)
    ]
    return {
        "messages": build_messages(selected_history, profile_prompt),
        "prompt_ids": ids,
        "prompt_tokens": len(ids),
        "prompt_token_ids_sha256": token_ids_sha256(ids),
        "p0_selected_start_index": int(p0_selection["selected_start_index"]),
        "selected_start_index": int(common["selected_start_index"]),
        "selected_message_ids": selected_turn_ids,
        "selected_history_sha256": common["selected_history_sha256"],
        "extra_removed_messages": int(common["extra_removed_messages"]),
        "p0_prompt_tokens": int(p0_selection["prompt_tokens"]),
        "ep_budget_prompt_tokens": int(common["prompt_lengths"]["EP"]),
    }


def build_emotion_response_prompt(
    sample: Mapping[str, Any],
    tokenizer: Any,
    *,
    selection_prompt: str,
    emotion_response_prompt: str,
    prompt_cap: int = 1280,
) -> dict[str, Any]:
    """Apply P0/T1 first, then recheck the final E03-B prompt with complete turns."""
    sample_id = str(sample.get("id", "<missing-id>"))
    history = _validate_sample_history(sample)
    try:
        p0_selection = select_recent_complete_suffix(
            history, selection_prompt, prompt_cap, tokenizer
        )
    except Exception as exc:
        raise PromptBuildError("p0_selection_failed", f"{sample_id}: P0/T1 failed: {exc}") from exc
    if p0_selection["status"] != "ok":
        raise PromptBuildError(
            "p0_prompt_floor_exceeds_cap",
            f"{sample_id}: current-user-only P0 prompt exceeds P={prompt_cap}",
            {"prompt_tokens": p0_selection["prompt_tokens"]},
        )

    p0_history = list(p0_selection["selected_history"])
    try:
        final_selection = select_recent_complete_suffix(
            p0_history, emotion_response_prompt, prompt_cap, tokenizer
        )
    except Exception as exc:
        raise PromptBuildError("e03b_selection_failed", f"{sample_id}: E03-B history selection failed: {exc}") from exc
    if final_selection["status"] != "ok":
        raise PromptBuildError(
            "e03b_prompt_floor_exceeds_cap",
            f"{sample_id}: current-user-only E03-B prompt exceeds P={prompt_cap}",
            {"prompt_tokens": final_selection["prompt_tokens"]},
        )
    ids = [int(value) for value in final_selection["prompt_ids"]]
    selected_history = list(final_selection["selected_history"])
    if not ids or len(ids) > prompt_cap:
        raise PromptBuildError("e03b_prompt_limit", f"{sample_id}: E03-B prompt violates the frozen token cap")
    selected_turn_ids = [
        message.get("turn_id", index)
        for index, message in enumerate(selected_history)
    ]
    return {
        "messages": build_messages(selected_history, emotion_response_prompt),
        "prompt_ids": ids,
        "prompt_tokens": len(ids),
        "prompt_token_ids_sha256": token_ids_sha256(ids),
        "p0_selected_start_index": int(p0_selection["selected_start_index"]),
        "selected_start_index": int(p0_selection["selected_start_index"])
        + int(final_selection["selected_start_index"]),
        "selected_message_ids": selected_turn_ids,
        "selected_history_sha256": history_sha256(selected_history),
        "extra_removed_messages": int(final_selection["selected_start_index"]),
        "p0_prompt_tokens": int(p0_selection["prompt_tokens"]),
    }


def parse_emotion_response_output(
    raw_text: str, official_schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Split at the first `;response=` and retain a valid body independently."""
    empty = {
        "emotion_label": "",
        "response_text": "",
        "emotion_error": None,
        "response_error": None,
        "format_error": None,
    }
    if not isinstance(raw_text, str):
        return {**empty, "format_error": "invalid_type"}
    if ";response=" not in raw_text:
        return {**empty, "format_error": "missing_response_separator"}

    prefix, response_text = raw_text.split(";response=", 1)
    match = re.fullmatch(r"emotion=([a-z_]+)", prefix)
    parsed = dict(empty)
    if match is None:
        parsed["emotion_error"] = "invalid_emotion_prefix"
    else:
        label = match.group(1)
        allowed = set(_specialist_schema(official_schema).get("$defs", {}).get("emotion", {}).get("enum", []))
        if label not in allowed:
            parsed["emotion_error"] = "invalid_emotion_label"
        else:
            parsed["emotion_label"] = label

    if response_text.strip():
        parsed["response_text"] = response_text
    else:
        parsed["response_error"] = "empty_response"
    return parsed


def assemble_submission(
    sample_id: str,
    profile_raw: str | None,
    emotion_response_raw: str | None,
    official_schema: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep valid fields from either route and enforce the official submission schema."""
    profile = {dimension: [] for dimension in PROFILE_DIMS}
    profile_error = None
    if profile_raw is None:
        profile_error = "generation_failed"
    else:
        try:
            profile = parse_specialist_output(
                profile_raw, _specialist_schema(official_schema), arm="P"
            )["user_profile"]
        except A8ParseError as exc:
            profile_error = exc.code
        except Exception as exc:
            profile_error = f"parser_error:{type(exc).__name__}"

    if emotion_response_raw is None:
        emotion = {
            "emotion_label": "",
            "response_text": "",
            "emotion_error": "generation_failed",
            "response_error": "generation_failed",
            "format_error": None,
        }
    else:
        emotion = parse_emotion_response_output(emotion_response_raw, official_schema)

    submission = {
        "id": sample_id,
        "response_text": emotion["response_text"],
        "emotion_label": emotion["emotion_label"],
        "user_profile": profile,
        "memory_refs": [],
    }
    Draft202012Validator(dict(official_schema)).validate(submission)
    diagnostics = {
        "profile_parse_error": profile_error,
        "emotion_parse_error": emotion["emotion_error"],
        "response_parse_error": emotion["response_error"],
        "emotion_response_format_error": emotion["format_error"],
    }
    return submission, diagnostics
