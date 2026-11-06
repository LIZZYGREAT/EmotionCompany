"""Strict, lossless serializer and parser for the isolated E03-A6 F4 format."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator

from src.shared.f3_protocol import submission_enums


F4_PROFILE_FIELDS = (
    ("personality_traits", "Personality"),
    ("interests", "Interests"),
    ("style", "Style"),
)
F4_FIELD_ORDER = ("Emotion", "Personality", "Interests", "Style", "Response")


class F4ParseError(ValueError):
    """A strict parse failure with a stable machine-readable category."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_f4_prompt_enum_order(
    prompt_text: str,
    enums: dict[str, list[str]],
) -> None:
    """Require all full-name field headings and canonical label lists."""
    expected_prefixes = {
        "emotion_label": "Emotion 标签（选一个）：",
        "personality_traits": "Personality 标签：",
        "interests": "Interests 标签：",
        "style": "Style 标签：",
    }
    lines = prompt_text.splitlines()
    for name, prefix in expected_prefixes.items():
        line = next((value for value in lines if value.startswith(prefix)), None)
        if line is None:
            raise ValueError(f"F4 prompt is missing label heading: {prefix}")
        labels = [value.strip() for value in line[len(prefix):].split(",") if value.strip()]
        if labels != enums[name] or len(labels) != len(set(labels)):
            raise ValueError(f"F4 prompt {name} labels/order do not match the official submission schema")


def _enums_from_schema(official_schema: dict[str, Any]) -> dict[str, list[str]]:
    definitions = official_schema.get("$defs", {})
    if "target" not in definitions:
        raise ValueError("official data schema does not define $defs.target")
    submission_schema = {
        "properties": {
            "emotion_label": {"enum": [v for v in definitions["emotion"]["enum"] if v]},
            "user_profile": {"properties": definitions["profile"]["properties"]},
        }
    }
    return submission_enums(submission_schema)


def _validate_target(target: dict[str, Any], official_schema: dict[str, Any]) -> None:
    if "target" not in official_schema.get("$defs", {}):
        raise ValueError("official data schema does not define $defs.target")
    schema = {
        "$schema": official_schema.get(
            "$schema", "https://json-schema.org/draft/2020-12/schema"
        ),
        "$ref": "#/$defs/target",
        "$defs": official_schema["$defs"],
    }
    Draft202012Validator(schema).validate(target)


def _ordered_labels(
    labels: Any,
    *,
    allowed: list[str],
    field_name: str,
) -> list[str]:
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise ValueError(f"{field_name} must be a list of strings")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field_name} contains duplicate labels")
    if not set(labels) <= set(allowed):
        raise ValueError(f"{field_name} contains an unknown label")
    return sorted(labels, key=allowed.index)


def serialize_f4(target: dict[str, Any], official_schema: dict[str, Any]) -> str:
    """Serialize an official target to canonical five-field full-name text."""
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    _validate_target(target, official_schema)
    enums = _enums_from_schema(official_schema)
    emotion = target["emotion_label"]
    if emotion not in enums["emotion_label"]:
        raise ValueError("emotion_label contains an unknown label")
    rows = [f"Emotion={emotion}"]
    profile = target["user_profile"]
    for name, marker in F4_PROFILE_FIELDS:
        labels = _ordered_labels(profile[name], allowed=enums[name], field_name=name)
        rows.append(f"{marker}={'|'.join(labels) if labels else '-'}")
    response = target["response_text"]
    if not isinstance(response, str):
        raise ValueError("response_text must be a string")
    rows.append(f"Response={response}")
    return "\n".join(rows)


def _parse_labels(value: str, *, allowed: list[str], field_name: str) -> list[str]:
    if value == "-":
        return []
    if not value:
        raise F4ParseError("invalid_profile_label", f"{field_name} cannot be empty; use '-'")
    labels = value.split("|")
    if any(not label or label not in allowed for label in labels):
        raise F4ParseError("invalid_profile_label", f"{field_name} contains an unknown label")
    if len(labels) != len(set(labels)):
        raise F4ParseError("duplicate_label", f"{field_name} contains duplicate labels")
    canonical = sorted(labels, key=allowed.index)
    if labels != canonical:
        raise F4ParseError("noncanonical_label_order", f"{field_name} is not in official enum order")
    return labels


def parse_f4(raw_text: str, enums: Mapping[str, list[str]]) -> dict[str, Any]:
    """Parse F4 without trimming or rewriting any character after ``Response=``."""
    if not isinstance(raw_text, str):
        raise F4ParseError("incomplete", "F4 output must be text")
    rows = raw_text.split("\n", 4)
    if len(rows) != 5:
        raise F4ParseError("incomplete", "F4 requires Emotion, Personality, Interests, Style, and Response rows")
    markers = ("Emotion=", "Personality=", "Interests=", "Style=", "Response=")
    if any(not row.startswith(marker) for row, marker in zip(rows, markers)):
        raise F4ParseError("unsupported_order", "F4 fields are missing or out of order")
    emotion = rows[0][len(markers[0]):]
    if emotion not in enums.get("emotion_label", []):
        raise F4ParseError("invalid_emotion", "Emotion field contains an unknown emotion label")
    profile = {
        name: _parse_labels(
            row[len(marker):],
            allowed=enums.get(name, []),
            field_name=name,
        )
        for row, marker, (name, _long_name) in zip(rows[1:4], markers[1:4], F4_PROFILE_FIELDS)
    }
    return {
        "emotion_label": emotion,
        "user_profile": profile,
        "response_text": rows[4][len(markers[4]):],
        "memory_refs": [],
    }
