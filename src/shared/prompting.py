"""P0 message construction and Qwen chat-template tokenization helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator


PROFILE_DIMS = ("personality_traits", "interests", "style")
OUTPUT_FIELDS = ("emotion_label", "user_profile", "response_text", "memory_refs")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _schema_enums(official_schema: dict[str, Any]) -> dict[str, list[str]]:
    definitions = official_schema.get("$defs", {})
    profile_properties = definitions.get("profile", {}).get("properties", {})
    result = {
        "emotion_label": list(definitions.get("emotion", {}).get("enum", [])),
    }
    for dimension in PROFILE_DIMS:
        result[dimension] = list(
            profile_properties.get(dimension, {}).get("items", {}).get("enum", [])
        )
    if any(not values for values in result.values()):
        raise ValueError("Official data schema does not define all P0 label enumerations")
    return result


def validate_prompt_enum_order(prompt_text: str, official_schema: dict[str, Any]) -> dict[str, list[str]]:
    """Require the P0 label lists to exactly match the official data schema order."""
    lines = prompt_text.splitlines()
    extracted: dict[str, list[str]] = {}
    expected_headings = {
        "emotion_label": "合法情绪标签（恰好选择一个）：",
        "personality_traits": "personality_traits:",
        "interests": "interests:",
        "style": "style:",
    }
    for key, heading in expected_headings.items():
        if key == "emotion_label":
            try:
                index = lines.index(heading)
            except ValueError as exc:
                raise ValueError(f"P0 is missing label heading: {heading}") from exc
            value_line = next((line.strip() for line in lines[index + 1:] if line.strip()), "")
            extracted[key] = [item.strip() for item in value_line.split(",") if item.strip()]
        else:
            line = next((item for item in lines if item.startswith(heading)), None)
            if line is None:
                raise ValueError(f"P0 is missing label heading: {heading}")
            value = line[len(heading):]
            extracted[key] = [item.strip() for item in value.split(",") if item.strip()]

    official = _schema_enums(official_schema)
    for key, labels in official.items():
        if extracted[key] != labels:
            missing = [label for label in labels if label not in extracted[key]]
            extra = [label for label in extracted[key] if label not in labels]
            raise ValueError(
                f"P0 {key} labels/order do not match the official data schema "
                f"(missing={missing}, extra={extra})"
            )
        if len(extracted[key]) != len(set(extracted[key])):
            raise ValueError(f"P0 {key} contains duplicate labels")
    return official


def build_messages(history: Sequence[dict[str, Any]], prompt_text: str) -> list[dict[str, str]]:
    """Return one system message followed by the unchanged native history turns."""
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt_text must be a non-empty string")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("history must be a non-empty sequence of messages")
    messages = [{"role": "system", "content": prompt_text}]
    for index, turn in enumerate(history):
        if not isinstance(turn, Mapping):
            raise ValueError(f"history[{index}] must be an object")
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"}:
            raise ValueError(f"history[{index}].role must be user or assistant")
        if not isinstance(content, str):
            raise ValueError(f"history[{index}].content must be a string")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "user":
        raise ValueError("the last history message must be the current user message")
    return messages


def _target_schema(official_schema: dict[str, Any]) -> dict[str, Any]:
    if "target" not in official_schema.get("$defs", {}):
        raise ValueError("Official data schema does not define $defs.target")
    return {
        "$schema": official_schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$ref": "#/$defs/target",
        "$defs": official_schema["$defs"],
    }


def serialize_completion(target: dict[str, Any], official_schema: dict[str, Any]) -> str:
    """Validate and emit the fixed compact P0 answer, preserving response text."""
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    validator = Draft202012Validator(_target_schema(official_schema))
    errors = sorted(
        validator.iter_errors(target),
        key=lambda error: (list(map(str, error.absolute_path)), error.message),
    )
    if errors:
        error = errors[0]
        location = "/".join(map(str, error.absolute_path)) or "$"
        raise ValueError(f"target violates the official data schema at {location} ({error.validator})")

    enums = _schema_enums(official_schema)
    profile = target["user_profile"]
    ordered_profile = {
        dimension: sorted(profile[dimension], key=enums[dimension].index)
        for dimension in PROFILE_DIMS
    }
    completion = {
        "emotion_label": target["emotion_label"],
        "user_profile": ordered_profile,
        "response_text": target["response_text"],
        "memory_refs": [],
    }
    return json.dumps(completion, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _extract_ids(encoded: Any, *, batch: bool = False) -> list[int] | list[list[int]]:
    if isinstance(encoded, Mapping) or hasattr(encoded, "keys"):
        if "input_ids" not in encoded:
            raise ValueError("apply_chat_template result has no input_ids")
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list):
        ids = list(ids)
    if batch:
        if ids and not isinstance(ids[0], (list, tuple)):
            raise ValueError("batched chat template returned a flat input_ids sequence")
        return [list(map(int, row)) for row in ids]
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("single-conversation chat template returned more than one sequence")
        ids = list(ids[0])
    return list(map(int, ids))


def encode_inference_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Encode P0 + native history + the model's assistant generation prefix."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _extract_ids(encoded)


def encode_full_training_sequence(
    tokenizer: Any,
    messages: list[dict[str, str]],
    completion_json: str,
) -> list[int]:
    """Encode the same prompt followed by one complete assistant answer message."""
    if not isinstance(completion_json, str) or not completion_json:
        raise ValueError("completion_json must be a non-empty string")
    full_messages = [*messages, {"role": "assistant", "content": completion_json}]
    encoded = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return _extract_ids(encoded)


def apply_chat_template_batch(
    tokenizer: Any,
    batch_messages: list[list[dict[str, str]]],
    *,
    add_generation_prompt: bool,
) -> list[list[int]]:
    """Batch the exact same native template calls used by the scalar interfaces."""
    if not batch_messages:
        return []
    encoded = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    ids = _extract_ids(encoded, batch=True)
    if len(ids) != len(batch_messages):
        raise ValueError("batched chat template returned an unexpected number of sequences")
    return ids


def _template_bytes(tokenizer: Any) -> bytes:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return template.encode("utf-8")
    if isinstance(template, (dict, list)):
        return canonical_json(template).encode("utf-8")
    raise ValueError("Tokenizer has no usable chat_template")


def tokenizer_signature(
    tokenizer: Any,
    *,
    model_id: str,
    revision: str,
    tokenizer_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fingerprint the exact tokenizer/template source used by one audit run."""
    template_hash = hashlib.sha256(_template_bytes(tokenizer)).hexdigest()
    special_tokens = []
    for token in getattr(tokenizer, "all_special_tokens", []):
        token_id = tokenizer.convert_tokens_to_ids(token)
        special_tokens.append({"token": str(token), "id": int(token_id) if token_id is not None else None})
    special_tokens.sort(key=lambda item: (item["id"] if item["id"] is not None else -1, item["token"]))
    identity = {
        "model_id": model_id,
        "revision": revision,
        "template_sha256": template_hash,
        "tokenizer_files": dict(sorted((tokenizer_files or {}).items())),
    }
    signature = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        **identity,
        "signature_sha256": signature,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "special_tokens": special_tokens,
        "eos_token": str(getattr(tokenizer, "eos_token", None)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": str(getattr(tokenizer, "pad_token", None)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
