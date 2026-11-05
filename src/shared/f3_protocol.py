"""Strict, lossless serializer and parser for the isolated E03-A6 F3 format."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator


PROFILE_FIELDS = (
    ("personality_traits", "P"),
    ("interests", "I"),
    ("style", "S"),
)
F3_FIELD_ORDER = ("E", "P", "I", "S", "R")


class F3ParseError(ValueError):
    """A strict parse failure with a stable machine-readable category."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def submission_enums(submission_schema: dict[str, Any]) -> dict[str, list[str]]:
    """Read the allowed labels in their official canonical order."""
    properties = submission_schema.get("properties", {})
    profile = properties.get("user_profile", {}).get("properties", {})
    emotions = properties.get("emotion_label", {}).get("enum", [])
    enums = {
        "emotion_label": [value for value in emotions if value],
        **{
            name: list(profile.get(name, {}).get("items", {}).get("enum", []))
            for name, _ in PROFILE_FIELDS
        },
    }
    if any(not isinstance(values, list) or not values for values in enums.values()):
        raise ValueError("submission schema must define all F3 label enumerations")
    return enums


def _validate_target(target: dict[str, Any], official_schema: dict[str, Any]) -> None:
    definitions = official_schema.get("$defs", {})
    if "target" not in definitions:
        raise ValueError("official data schema does not define $defs.target")
    target_schema = {
        "$schema": official_schema.get(
            "$schema", "https://json-schema.org/draft/2020-12/schema"
        ),
        "$ref": "#/$defs/target",
        "$defs": definitions,
    }
    Draft202012Validator(target_schema).validate(target)


def _canonical_labels(
    labels: Any,
    *,
    allowed: list[str],
    field_name: str,
    error_type: type[ValueError] = ValueError,
) -> list[str]:
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise error_type(f"{field_name} must be a list of strings")
    if len(labels) != len(set(labels)):
        raise error_type(f"{field_name} contains duplicate labels")
    if not set(labels) <= set(allowed):
        raise error_type(f"{field_name} contains an unknown label")
    return sorted(labels, key=allowed.index)


def serialize_f3(target: dict[str, Any], official_schema: dict[str, Any]) -> str:
    """Serialize an official normalized target to canonical five-field F3 text."""
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    _validate_target(target, official_schema)
    enums = submission_enums({
        "properties": {
            "emotion_label": {
                "enum": [value for value in official_schema["$defs"]["emotion"]["enum"] if value]
            },
            "user_profile": {"properties": official_schema["$defs"]["profile"]["properties"]},
        }
    })

    emotion = target["emotion_label"]
    if emotion not in enums["emotion_label"]:
        raise ValueError("emotion_label contains an unknown label")
    rows = [f"E={emotion}"]
    profile = target["user_profile"]
    for name, marker in PROFILE_FIELDS:
        labels = _canonical_labels(profile[name], allowed=enums[name], field_name=name)
        rows.append(f"{marker}={'|'.join(labels) if labels else '-'}")
    response = target["response_text"]
    if not isinstance(response, str):
        raise ValueError("response_text must be a string")
    rows.append(f"R={response}")
    return "\n".join(rows)


def _parse_label_row(
    value: str,
    *,
    allowed: list[str],
    field_name: str,
) -> list[str]:
    if value == "-":
        return []
    if not value:
        raise F3ParseError("invalid_profile_label", f"{field_name} cannot be empty; use '-'")
    labels = value.split("|")
    if any(not label or label not in allowed for label in labels):
        raise F3ParseError("invalid_profile_label", f"{field_name} contains an unknown label")
    if len(labels) != len(set(labels)):
        raise F3ParseError("duplicate_label", f"{field_name} contains duplicate labels")
    canonical = sorted(labels, key=allowed.index)
    if labels != canonical:
        raise F3ParseError("noncanonical_label_order", f"{field_name} is not in official enum order")
    return labels


def parse_f3(raw_text: str, enums: dict[str, list[str]]) -> dict[str, Any]:
    """Parse F3 without trimming or rewriting any character after the R= marker."""
    if not isinstance(raw_text, str):
        raise F3ParseError("incomplete", "F3 output must be text")
    rows = raw_text.split("\n", 4)
    if len(rows) != 5:
        raise F3ParseError("incomplete", "F3 requires E, P, I, S, and R rows")
    prefixes = ("E=", "P=", "I=", "S=", "R=")
    if any(not row.startswith(prefix) for row, prefix in zip(rows, prefixes)):
        if rows[0].startswith("E="):
            raise F3ParseError("unsupported_order", "F3 fields are missing or out of order")
        raise F3ParseError("unsupported_order", "F3 must begin with the E field")

    emotion = rows[0][2:]
    if emotion not in enums.get("emotion_label", []):
        raise F3ParseError("invalid_emotion", "E field contains an unknown emotion label")
    profile: dict[str, list[str]] = {}
    for row, (name, marker) in zip(rows[1:4], PROFILE_FIELDS):
        profile[name] = _parse_label_row(
            row[len(marker) + 1 :],
            allowed=enums.get(name, []),
            field_name=name,
        )
    return {
        "emotion_label": emotion,
        "user_profile": profile,
        "response_text": rows[4][2:],
        "memory_refs": [],
    }


def parse_f3_prediction(
    sample_id: str,
    raw_text: str,
    enums: dict[str, list[str]],
) -> dict[str, Any]:
    """Create the official five-field prediction object from F3 text."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    parsed = parse_f3(raw_text, enums)
    return {"id": sample_id, **parsed}
