"""Strict serializers and parsers for E03-A8 generated classification outputs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator

PROFILE_DIMS = ("personality_traits", "interests", "style")
FIELD_ORDER = {
    "P": ("Personality", "Interests", "Style"),
    "EP": ("Emotion", "Personality", "Interests", "Style"),
}
FIELD_TO_DIM = {
    "Personality": "personality_traits",
    "Interests": "interests",
    "Style": "style",
}
FIELD_TO_HEADING = {
    "Emotion": "Emotion 标签：",
    "Personality": "Personality 标签：",
    "Interests": "Interests 标签：",
    "Style": "Style 标签：",
}


class A8ParseError(ValueError):
    """Stable machine-readable failure emitted by the strict A8 output parser."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def enums_from_schema(official_schema: Mapping[str, Any]) -> dict[str, list[str]]:
    definitions = official_schema.get("$defs", {})
    emotion = definitions.get("emotion", {}).get("enum")
    profile = definitions.get("profile", {}).get("properties", {})
    if not isinstance(emotion, list):
        raise ValueError("official schema has no emotion enum")
    result: dict[str, list[str]] = {"emotion_label": [v for v in emotion if v]}
    for dim in PROFILE_DIMS:
        values = profile.get(dim, {}).get("items", {}).get("enum")
        if not isinstance(values, list):
            raise ValueError(f"official schema has no enum for {dim}")
        result[dim] = list(values)
    expected_counts = {"emotion_label": 16, "personality_traits": 10, "interests": 12, "style": 11}
    for name, count in expected_counts.items():
        values = result[name]
        if len(values) != count or len(values) != len(set(values)):
            raise ValueError(f"official schema enum {name} has an unexpected size or duplicates")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"official schema enum {name} contains an invalid value")
    return result


def validate_a8_prompt_enum_order(
    prompt_text: str, official_schema: Mapping[str, Any], arm: str
) -> dict[str, list[str]]:
    if arm not in FIELD_ORDER:
        raise ValueError(f"unsupported A8 arm: {arm!r}")
    enums = enums_from_schema(official_schema)
    lines = prompt_text.splitlines()
    required = ("emotion_label", *PROFILE_DIMS) if arm == "EP" else PROFILE_DIMS
    extracted: dict[str, list[str]] = {}
    for name in required:
        field = "Emotion" if name == "emotion_label" else next(
            key for key, value in FIELD_TO_DIM.items() if value == name
        )
        heading = FIELD_TO_HEADING[field]
        matches = [line for line in lines if line.startswith(heading)]
        if len(matches) != 1:
            raise ValueError(f"A8 {arm} prompt must contain exactly one {heading!r} heading")
        labels = [item.strip() for item in matches[0][len(heading):].split(",")]
        if labels != enums[name] or len(labels) != len(set(labels)):
            raise ValueError(f"A8 {arm} prompt enum/order differs from the official schema for {name}")
        extracted[name] = labels

    output_rows = {
        "Emotion": "Emotion=<情绪标签>",
        "Personality": "Personality=<标签或->",
        "Interests": "Interests=<标签或->",
        "Style": "Style=<标签或->",
    }
    expected_rows = [output_rows[field] for field in FIELD_ORDER[arm]]
    positions = [prompt_text.find(row) for row in expected_rows]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError(f"A8 {arm} prompt output fields/order differ from the frozen protocol")
    forbidden = "Emotion=<情绪标签>"
    if arm == "P" and forbidden in prompt_text:
        raise ValueError("A8 P prompt must not declare an Emotion output field")
    return enums


def _validate_gold_target(target: Mapping[str, Any], official_schema: Mapping[str, Any]) -> None:
    if "target" not in official_schema.get("$defs", {}):
        raise ValueError("official schema has no $defs.target")
    schema = {
        "$schema": official_schema.get(
            "$schema", "https://json-schema.org/draft/2020-12/schema"
        ),
        "$ref": "#/$defs/target",
        "$defs": official_schema["$defs"],
    }
    Draft202012Validator(schema).validate(dict(target))


def serialize_specialist_completion(
    target: Mapping[str, Any], official_schema: Mapping[str, Any], arm: str
) -> str:
    if arm not in FIELD_ORDER:
        raise ValueError(f"unsupported A8 arm: {arm!r}")
    _validate_gold_target(target, official_schema)
    enums = enums_from_schema(official_schema)
    profile = target["user_profile"]
    rows = []
    if arm == "EP":
        emotion = target["emotion_label"]
        if emotion not in enums["emotion_label"]:
            raise ValueError("Gold emotion_label is not in the official enum")
        rows.append(f"Emotion={emotion}")
    for field in FIELD_ORDER[arm][-3:]:
        dim = FIELD_TO_DIM[field]
        labels = profile[dim]
        if len(labels) != len(set(labels)) or any(label not in enums[dim] for label in labels):
            raise ValueError(f"Gold profile labels are invalid for {dim}")
        ordered = sorted(labels, key=enums[dim].index)
        rows.append(f"{field}={'|'.join(ordered) if ordered else '-'}")
    return "\n".join(rows)


def _parse_profile_labels(value: str, *, allowed: list[str], dim: str) -> list[str]:
    if value == "-":
        return []
    if not value:
        raise A8ParseError("empty_value", f"{dim} must use '-' when empty")
    if any(char.isspace() for char in value) or "," in value:
        raise A8ParseError("invalid_separator", f"{dim} labels contain unsupported whitespace or commas")
    labels = value.split("|")
    if any(not label for label in labels):
        raise A8ParseError("empty_label", f"{dim} contains an empty label")
    if len(labels) != len(set(labels)):
        raise A8ParseError("duplicate_label", f"{dim} contains duplicate labels")
    if any(label not in allowed for label in labels):
        raise A8ParseError("unknown_label", f"{dim} contains a label outside the official enum")
    if labels != sorted(labels, key=allowed.index):
        raise A8ParseError("noncanonical_label_order", f"{dim} labels are not in official enum order")
    return labels


def parse_specialist_output(
    raw_text: str, official_schema: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    if arm not in FIELD_ORDER:
        raise ValueError(f"unsupported A8 arm: {arm!r}")
    if not isinstance(raw_text, str):
        raise A8ParseError("invalid_type", "generated output must be text")
    if "\r" in raw_text:
        raise A8ParseError("invalid_line_ending", "only LF line endings are supported")
    if raw_text != raw_text.strip():
        raise A8ParseError("surrounding_whitespace", "leading or trailing whitespace is not permitted")
    rows = raw_text.split("\n")
    expected = FIELD_ORDER[arm]
    if len(rows) != len(expected):
        if len(rows) < len(expected):
            code = "incomplete"
        else:
            other_fields = set(FIELD_ORDER["EP"] if arm == "P" else FIELD_ORDER["P"]) - set(expected)
            row_keys = {row.split("=", 1)[0] for row in rows if "=" in row}
            code = "extra_field" if row_keys & other_fields else "extra_text"
        raise A8ParseError(code, f"A8 {arm} output requires exactly {len(expected)} rows")
    fields: dict[str, str] = {}
    for index, (row, field) in enumerate(zip(rows, expected)):
        if row.count("=") != 1:
            raise A8ParseError("invalid_assignment", f"row {index + 1} must contain one ASCII '='")
        key, value = row.split("=", 1)
        if key != field:
            if key in FIELD_ORDER[arm]:
                raise A8ParseError("unsupported_order", f"expected {field} on row {index + 1}")
            if key in {"Response", "response_text", "memory_refs"}:
                raise A8ParseError("extra_field", f"field {key} is forbidden in A8 output")
            raise A8ParseError("unknown_field", f"field {key!r} is not allowed on row {index + 1}")
        if not value:
            raise A8ParseError("empty_value", f"{field} value cannot be empty")
        fields[field] = value

    enums = enums_from_schema(official_schema)
    profile = {
        dim: _parse_profile_labels(fields[field], allowed=enums[dim], dim=dim)
        for field, dim in FIELD_TO_DIM.items()
    }
    result: dict[str, Any] = {"user_profile": profile}
    if arm == "EP":
        emotion = fields["Emotion"]
        if re.fullmatch(r"[a-z_]+", emotion) is None or emotion not in enums["emotion_label"]:
            raise A8ParseError("invalid_emotion", "Emotion must be exactly one official label")
        result["emotion_label"] = emotion
    return result


def validate_training_target_round_trip(
    target: Mapping[str, Any], official_schema: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    completion = serialize_specialist_completion(target, official_schema, arm)
    parsed = parse_specialist_output(completion, official_schema, arm)
    expected = {
        "user_profile": {
            dim: sorted(target["user_profile"][dim], key=enums_from_schema(official_schema)[dim].index)
            for dim in PROFILE_DIMS
        }
    }
    if arm == "EP":
        expected["emotion_label"] = target["emotion_label"]
    if parsed != expected:
        raise ValueError(f"A8 {arm} target round trip changed a supervised field")
    if "response_text" in parsed or "memory_refs" in parsed:
        raise AssertionError("A8 target round trip leaked a non-classification field")
    return parsed
