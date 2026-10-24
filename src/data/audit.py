"""Read-only, reproducible quality audit for the original D0.1 JSONL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator


AUDIT_VERSION = "D0.1"
SPLITS = ("train", "val")
FALLBACK_ENUMS = {
    "emotion_label": [
        "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
        "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
    ],
    "personality_traits": [
        "extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual",
        "agreeable", "assertive", "emotionally_stable", "sensitive",
    ],
    "interests": [
        "study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games",
        "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design",
    ],
    "style": [
        "brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational",
        "high_emotional_expression", "low_emotional_expression", "emoji_user",
    ],
}
PROFILE_DIMS = ("personality_traits", "interests", "style")
SEVERITY_RANK = {"ERROR": 0, "REVIEW": 1, "INFO": 2}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _typed_key(value: Any) -> tuple[str, str]:
    """Keep IDs with different JSON/Python types distinct, including bool and int."""
    return type(value).__name__, _canonical_json(value)


def _safe_id(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "value_sha256": _digest(value)}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()


def _percentile(values: list[int], percentile: int) -> int | None:
    """Nearest-rank percentile: sorted[ceil(p*n)-1], with p in 1..100."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil((percentile / 100) * len(ordered)) - 1)]


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None,
                "p95": None, "p99": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts)}


def _location(ref: dict[str, Any]) -> dict[str, Any]:
    return {key: ref.get(key) for key in ("split", "source_line", "conversation_id", "prediction_point_index")}


class Audit:
    def __init__(self, root: Path, config_path: Path, config: dict[str, Any]):
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.config = config
        self.issues: list[dict[str, Any]] = []
        self.file_stats: dict[str, dict[str, Any]] = {}
        self.schema_info: dict[str, dict[str, Any]] = {}
        self.validators: dict[str, Draft202012Validator | None] = {}
        self.enums: dict[str, list[str]] = dict(FALLBACK_ENUMS)
        self.enums_by_split: dict[str, dict[str, list[str]]] = {}
        self.refs_by_turn_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.refs_by_conversation_text_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.refs_by_exact_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.refs_by_normalized_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.refs_by_prediction_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.id_refs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.history_supervision: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.thresholds = config.get("thresholds", {})
        self.checks = config.get("checks", {})
        self.inputs = config.get("inputs", {})
        self.output = config.get("output", {})

    def add_issue(
        self,
        severity: str,
        code: str,
        split: str,
        source_line: int | None,
        conversation_id: Any = None,
        prediction_point_index: int | None = None,
        turn_index: int | None = None,
        field_path: str | None = None,
        observed: Any = None,
        message: str = "",
        proposed_action: str = "manual_review_in_D0_2",
    ) -> None:
        context_fingerprint = _digest({"turn_index": turn_index, "field_path": field_path, "observed": observed})[:10]
        issue = {
            "issue_id": f"{split}:{source_line if source_line is not None else 'file'}:"
                        f"{prediction_point_index if prediction_point_index is not None else '-'}:{code}"
                        + (f":{turn_index}" if turn_index is not None else "") + f":{context_fingerprint}",
            "severity": severity,
            "code": code,
            "split": split,
            "source_line": source_line,
            "conversation_id": _safe_id(conversation_id),
            "prediction_point_index": prediction_point_index,
            "turn_index": turn_index,
            "field_path": field_path,
            "observed": observed,
            "message": message,
            "proposed_action": proposed_action,
        }
        self.issues.append(issue)

    def load_schemas(self) -> None:
        official_enums: dict[str, list[str]] | None = None
        for split in SPLITS:
            input_path = self._input_path(split)
            schema_path = Path(str(input_path)[:-6] + ".schema.json") if str(input_path).endswith(".jsonl") else None
            if schema_path is None or not schema_path.is_file():
                self.validators[split] = None
                self.schema_info[split] = {"path": None, "available": False, "sha256": None}
                continue
            raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(raw_schema)
            self.validators[split] = Draft202012Validator(raw_schema)
            self.schema_info[split] = {
                "path": str(schema_path.relative_to(self.root)),
                "available": True,
                "sha256": _sha256_file(schema_path),
            }
            defs = raw_schema.get("$defs", {})
            profile = defs.get("profile", {}).get("properties", {})
            found_enums = {
                "emotion_label": list(defs.get("emotion", {}).get("enum", [])),
                **{dim: list(profile.get(dim, {}).get("items", {}).get("enum", [])) for dim in PROFILE_DIMS},
            }
            if all(found_enums.values()):
                self.enums_by_split[split] = found_enums
                if official_enums is None:
                    official_enums = found_enums
        if official_enums:
            self.enums = official_enums

    def _input_path(self, split: str) -> Path:
        value = self.inputs.get(split)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing inputs.{split} in audit config")
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Input path escapes project root: {value}")
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {value}")
        return path

    def _validate_schema(self, split: str, row: Any, line_no: int) -> list[tuple[str, str]]:
        validator = self.validators.get(split)
        if not self.checks.get("schema", True) or validator is None:
            return []
        errors = sorted(validator.iter_errors(row), key=lambda err: (list(map(str, err.absolute_path)), err.message))
        result_paths: list[tuple[str, str]] = []
        for error in errors:
            path = list(error.absolute_path)
            result_paths.append(("/".join(map(str, path)), str(error.validator)))
            code = self._schema_issue_code(path, error.validator, error.validator_value)
            pp_index = path[1] if len(path) > 1 and path[0] == "prediction_points" and isinstance(path[1], int) else None
            turn_index = path[1] if len(path) > 1 and path[0] == "turns" and isinstance(path[1], int) else None
            cid = row.get("conversation_id") if isinstance(row, dict) else None
            expected = error.validator_value if error.validator in {"type", "enum", "minLength", "minimum"} else None
            observed = {"validator": error.validator, "expected": expected}
            if error.validator == "required":
                observed["missing_fields"] = sorted(set(error.validator_value) - set(error.instance)) if isinstance(error.instance, dict) else list(error.validator_value)
            elif error.validator == "type":
                observed["observed_type"] = type(error.instance).__name__
            elif error.validator == "enum":
                observed["observed_value"] = {"type": type(error.instance).__name__, "length": len(error.instance) if isinstance(error.instance, (str, list, dict)) else None, "sha256": _digest(error.instance)}
            elif error.validator == "minLength":
                observed["observed_length"] = len(error.instance) if isinstance(error.instance, str) else None
            self.add_issue(
                "ERROR", code, split, line_no, cid, pp_index, turn_index,
                field_path="/".join(map(str, path)) or "$",
                observed=observed,
                message=f"Official schema violation ({error.validator})",
                proposed_action="review_schema_violation_in_D0_2",
            )
        return result_paths

    @staticmethod
    def _schema_issue_code(path: list[Any], validator: str, _expected: Any) -> str:
        if validator == "enum" and path and (path[-1] == "emotion_label" or (len(path) >= 2 and path[-2] in PROFILE_DIMS)):
            return "ENUM_CHECK"
        if not path:
            return "ROOT_SCHEMA"
        first = path[0]
        if first == "turns":
            return "TURN_SCHEMA"
        if first == "prediction_points":
            if len(path) >= 3 and path[2] == "target":
                if len(path) >= 4 and path[3] == "user_profile":
                    return "PROFILE_SCHEMA"
                if len(path) >= 4:
                    return "TARGET_SCHEMA"
                return "TARGET_SCHEMA"
            return "POINT_SCHEMA"
        return "ROOT_SCHEMA"

    def _read_split(self, split: str) -> dict[str, Any]:
        path = self._input_path(split)
        split_enums = self.enums_by_split.get(split, self.enums)
        before_sha = _sha256_file(path)
        counters: Counter[str] = Counter()
        turns_per_conversation: list[int] = []
        points_per_conversation: list[int] = []
        emotion_freq: Counter[str] = Counter()
        profile_freq = {dim: Counter() for dim in PROFILE_DIMS}
        profile_points = 0
        profile_unavailable = 0
        profile_dim_readable = Counter()
        profile_dim_empty = Counter()
        profile_points_empty_dimensions: list[int] = []
        memory_lengths: list[int] = []
        memory_empty = 0
        memory_nonempty = 0
        memory_unavailable = 0
        emotion_missing_or_invalid = 0
        emotion_readable = 0
        dimension_tag_counts = {dim: [] for dim in PROFILE_DIMS}
        cooccurrence = Counter()
        cross_dim_cooccurrence = Counter()
        response_hash_counts: Counter[str] = Counter()
        message_lengths: list[int] = []
        user_message_lengths: list[int] = []
        assistant_message_lengths: list[int] = []
        response_lengths: list[int] = []
        current_user_lengths: list[int] = []
        history_message_counts: list[int] = []
        history_char_lengths: list[int] = []
        history_unavailable = 0
        short_responses = 0
        long_responses = 0
        long_messages = 0
        replacement_char_count = 0
        invalid_char_count = 0
        rows_with_root_shape = 0
        parsed_objects = 0
        schema_invalid_rows = 0
        all_schema_paths: Counter[str] = Counter()
        all_schema_validators: Counter[str] = Counter()
        empty_content_by_source_role: Counter[str] = Counter()
        first_id_by_split: dict[tuple[str, str], dict[str, Any]] = {}

        with path.open("rb") as stream:
            for line_no, raw_line in enumerate(stream, start=1):
                counters["physical_line_count"] += 1
                content = raw_line.rstrip(b"\r\n")
                if not content.strip():
                    counters["blank_line_count"] += 1
                    continue
                counters["nonempty_line_count"] += 1
                try:
                    decoded = content.decode("utf-8")
                except UnicodeDecodeError as error:
                    counters["unparseable_line_count"] += 1
                    self.add_issue("ERROR", "FILE_PARSE", split, line_no, field_path="$", observed={"exception": "UnicodeDecodeError", "start": error.start, "end": error.end}, message="Line is not valid UTF-8", proposed_action="inspect_encoding_in_D0_2")
                    continue
                try:
                    row = json.loads(decoded, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant {value}")))
                except (json.JSONDecodeError, ValueError) as error:
                    counters["unparseable_line_count"] += 1
                    observed = {"exception": type(error).__name__}
                    if isinstance(error, json.JSONDecodeError):
                        observed.update({"column": error.colno, "message": error.msg})
                    self.add_issue("ERROR", "FILE_PARSE", split, line_no, field_path="$", observed=observed, message="Nonempty JSONL line is not valid strict JSON", proposed_action="inspect_source_line_in_D0_2")
                    continue
                counters["parseable_line_count"] += 1
                if isinstance(row, dict):
                    parsed_objects += 1
                schema_errors = self._validate_schema(split, row, line_no)
                all_schema_paths.update(field_path for field_path, _ in schema_errors)
                all_schema_validators.update(validator for _, validator in schema_errors)
                if schema_errors:
                    schema_invalid_rows += 1
                cid = row.get("conversation_id") if isinstance(row, dict) else None
                if isinstance(row, dict) and isinstance(row.get("turns"), list) and isinstance(row.get("prediction_points"), list):
                    rows_with_root_shape += 1
                if not isinstance(row, dict):
                    counters["non_object_root_count"] += 1
                    continue

                ref = {"split": split, "source_line": line_no, "conversation_id": _safe_id(cid), "prediction_point_index": None}
                if cid is not None:
                    key = _typed_key(cid)
                    all_refs = self.id_refs[key]
                    if any(existing["split"] == split for existing in all_refs):
                        first_same = next(existing for existing in all_refs if existing["split"] == split)
                        self.add_issue("REVIEW", "DUPLICATE_CONVERSATION_ID", split, line_no, cid, field_path="conversation_id", observed={"first_source_line": first_same["source_line"]}, message="Conversation ID repeats within a split", proposed_action="review_id_collision_in_D0_2")
                    elif all_refs:
                        self.add_issue("REVIEW", "CROSS_SPLIT_CONVERSATION_ID", split, line_no, cid, field_path="conversation_id", observed={"other_split": all_refs[0]["split"], "other_source_line": all_refs[0]["source_line"]}, message="Conversation ID appears in both official splits", proposed_action="review_split_identity_in_D0_2")
                    all_refs.append(ref)

                turns = row.get("turns")
                points = row.get("prediction_points")
                if not isinstance(turns, list):
                    turns = []
                    history_unavailable += 1
                if not isinstance(points, list):
                    points = []
                counters["turn_count"] += len(turns)
                counters["prediction_point_count"] += len(points)
                turns_per_conversation.append(len(turns))
                points_per_conversation.append(len(points))
                if not points:
                    counters["empty_points_conversation_count"] += 1
                    self.add_issue("INFO", "EMPTY_POINTS", split, line_no, cid, field_path="prediction_points", observed={"count": 0}, message="Conversation has no prediction points; retained as a legal data fact", proposed_action="record_only")

                id_to_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
                id_display: dict[tuple[str, str], Any] = {}
                for turn_index, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        counters["turn_non_object_count"] += 1
                        continue
                    if "turn_id" in turn:
                        turn_key = _typed_key(turn["turn_id"])
                        id_to_positions[turn_key].append(turn_index)
                        id_display[turn_key] = turn["turn_id"]
                    role = turn.get("role")
                    text = turn.get("content")
                    if isinstance(text, str):
                        size = len(text)
                        message_lengths.append(size)
                        if role == "user":
                            user_message_lengths.append(size)
                        elif role == "assistant":
                            assistant_message_lengths.append(size)
                        if size > int(self.thresholds.get("long_message_chars", 4000)):
                            long_messages += 1
                            self.add_issue("REVIEW", "LONG_MESSAGE_CANDIDATE", split, line_no, cid, turn_index=turn_index, field_path=f"turns[{turn_index}].content", observed={"unicode_characters": size, "threshold": int(self.thresholds.get("long_message_chars", 4000))}, message="Message exceeds the configured length review threshold")
                        if text.strip() == "":
                            empty_content_by_source_role[f"turn:{role if isinstance(role, str) else '<invalid_role>'}"] += 1
                            self.add_issue("REVIEW", "EMPTY_CONTENT", split, line_no, cid, turn_index=turn_index, field_path=f"turns[{turn_index}].content", observed={"role": role, "unicode_characters": size, "whitespace_only": bool(text)}, message="Message content is empty or contains only whitespace")
                        chars = self._text_anomalies(text)
                        if chars:
                            invalid_char_count += len(chars)
                            replacement_char_count += sum(c["codepoint"] == "U+FFFD" for c in chars)
                            self.add_issue("REVIEW", "INVALID_TEXT_CHAR", split, line_no, cid, turn_index=turn_index, field_path=f"turns[{turn_index}].content", observed={"characters": chars}, message="Message contains replacement, NUL, or control characters")
                    else:
                        counters["message_content_unavailable_count"] += 1
                    if turn_index and isinstance(turns[turn_index - 1], dict):
                        previous_role = turns[turn_index - 1].get("role")
                        if role in {"user", "assistant"} and previous_role == role:
                            self.add_issue("INFO", "CONSECUTIVE_SAME_ROLE", split, line_no, cid, turn_index=turn_index, field_path=f"turns[{turn_index}].role", observed={"role": role, "previous_turn_index": turn_index - 1}, message="Adjacent turns have the same role; retained as a descriptive warning", proposed_action="review_role_convention_if_relevant")
                for turn_key, positions in id_to_positions.items():
                    if len(positions) > 1:
                        self.add_issue("ERROR", "DUPLICATE_TURN_ID", split, line_no, cid, field_path="turns", observed={"turn_id": _safe_id(id_display[turn_key]), "turn_indices": positions}, message="Turn ID maps to more than one array position", proposed_action="resolve_ambiguous_turn_reference_in_D0_2")

                point_target_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
                valid_points_for_fingerprint: list[dict[str, Any]] = []
                for point_index, point in enumerate(points):
                    if not isinstance(point, dict):
                        profile_unavailable += 1
                        emotion_missing_or_invalid += 1
                        memory_unavailable += 1
                        counters["response_text_unavailable_count"] += 1
                        history_unavailable += 1
                        continue
                    point_ref = {"split": split, "source_line": line_no, "conversation_id": _safe_id(cid), "prediction_point_index": point_index}
                    target = point.get("target")
                    if not isinstance(target, dict):
                        profile_unavailable += 1
                        emotion_missing_or_invalid += 1
                        memory_unavailable += 1
                        counters["response_text_unavailable_count"] += 1
                        history_unavailable += 1
                        continue

                    emotion = target.get("emotion_label")
                    if isinstance(emotion, str) and emotion in split_enums["emotion_label"]:
                        emotion_freq[emotion] += 1
                        emotion_readable += 1
                    else:
                        emotion_missing_or_invalid += 1
                        if emotion is not None:
                            emotion_freq[f"<invalid_sha256:{hashlib.sha256(_canonical_json(emotion).encode('utf-8')).hexdigest()[:12]}>"] += 1
                    profile = target.get("user_profile")
                    profile_is_readable = isinstance(profile, dict) and all(isinstance(profile.get(dim), list) for dim in PROFILE_DIMS)
                    if profile_is_readable:
                        profile_points += 1
                        empty_dims = 0
                        for dim in PROFILE_DIMS:
                            values = profile[dim]
                            profile_dim_readable[dim] += 1
                            if not values:
                                profile_dim_empty[dim] += 1
                                empty_dims += 1
                            dimension_tag_counts[dim].append(len(values))
                            for item in values:
                                if isinstance(item, str) and item in split_enums[dim]:
                                    profile_freq[dim][item] += 1
                                else:
                                    profile_freq[dim][f"<invalid_sha256:{hashlib.sha256(_canonical_json(item).encode('utf-8')).hexdigest()[:12]}>"] += 1
                            for item in values:
                                if isinstance(item, str) and item in split_enums[dim]:
                                    cooccurrence[f"{dim}:{item}"] += 1
                        for left_index, left_dim in enumerate(PROFILE_DIMS):
                            for right_dim in PROFILE_DIMS[left_index + 1:]:
                                for left_label in profile[left_dim]:
                                    for right_label in profile[right_dim]:
                                        if isinstance(left_label, str) and left_label in split_enums[left_dim] and isinstance(right_label, str) and right_label in split_enums[right_dim]:
                                            cross_dim_cooccurrence[f"{left_dim}:{left_label}|{right_dim}:{right_label}"] += 1
                        profile_points_empty_dimensions.append(empty_dims)
                        if empty_dims == 3:
                            counters["profile_all_empty_points"] += 1
                        elif empty_dims:
                            counters["profile_partly_empty_points"] += 1
                    else:
                        profile_unavailable += 1
                        for dim in PROFILE_DIMS:
                            if not isinstance(profile, dict) or not isinstance(profile.get(dim), list):
                                profile_unavailable += 0

                    memory_refs = target.get("memory_refs")
                    if isinstance(memory_refs, list):
                        memory_lengths.append(len(memory_refs))
                        if memory_refs:
                            memory_nonempty += 1
                        else:
                            memory_empty += 1
                    else:
                        memory_unavailable += 1

                    response_text = target.get("response_text")
                    if isinstance(response_text, str):
                        response_lengths.append(len(response_text))
                        response_hash_counts[hashlib.sha256(response_text.encode("utf-8")).hexdigest()] += 1
                        if response_text.strip() == "":
                            empty_content_by_source_role["prediction_point:assistant_reference"] += 1
                            self.add_issue("REVIEW", "EMPTY_CONTENT", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].target.response_text", observed={"role": "assistant_reference", "unicode_characters": len(response_text), "whitespace_only": bool(response_text)}, message="Reference response is empty or contains only whitespace")
                        short_limit = int(self.thresholds.get("short_response_chars", 5))
                        long_limit = int(self.thresholds.get("long_response_chars", 2000))
                        if len(response_text) < short_limit:
                            short_responses += 1
                            self.add_issue("REVIEW", "SHORT_RESPONSE_CANDIDATE", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].target.response_text", observed={"unicode_characters": len(response_text), "threshold_exclusive": short_limit}, message="Reference response is shorter than the configured review threshold")
                        if len(response_text) > long_limit:
                            long_responses += 1
                            self.add_issue("REVIEW", "LONG_RESPONSE_CANDIDATE", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].target.response_text", observed={"unicode_characters": len(response_text), "threshold_exclusive": long_limit}, message="Reference response exceeds the configured review threshold")
                        chars = self._text_anomalies(response_text)
                        if chars:
                            invalid_char_count += len(chars)
                            replacement_char_count += sum(c["codepoint"] == "U+FFFD" for c in chars)
                            self.add_issue("REVIEW", "INVALID_TEXT_CHAR", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].target.response_text", observed={"characters": chars}, message="Reference response contains replacement, NUL, or control characters")
                    else:
                        counters["response_text_unavailable_count"] += 1

                    target_id = point.get("target_user_turn_id")
                    response_id = point.get("assistant_response_turn_id")
                    target_positions = id_to_positions.get(_typed_key(target_id), []) if "target_user_turn_id" in point else []
                    response_positions = id_to_positions.get(_typed_key(response_id), []) if "assistant_response_turn_id" in point else []
                    target_index: int | None = None
                    response_index: int | None = None
                    if self.checks.get("turn_references", True):
                        if len(target_positions) == 0:
                            self.add_issue("ERROR", "TARGET_TURN_NOT_FOUND", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].target_user_turn_id", observed={"turn_id": _safe_id(target_id)}, message="Target user turn ID does not match any turn")
                        elif len(target_positions) > 1:
                            self.add_issue("ERROR", "TARGET_TURN_ID_AMBIGUOUS", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].target_user_turn_id", observed={"turn_id": _safe_id(target_id), "turn_indices": target_positions}, message="Target user turn ID matches multiple turns")
                        else:
                            target_index = target_positions[0]
                            target_turn = turns[target_index]
                            if target_turn.get("role") != "user":
                                self.add_issue("ERROR", "TARGET_TURN_NOT_USER", split, line_no, cid, point_index, turn_index=target_index, field_path=f"turns[{target_index}].role", observed={"role": target_turn.get("role")}, message="Target turn does not have role user")
                                target_index = None
                        if len(response_positions) == 0:
                            self.add_issue("ERROR", "RESPONSE_TURN_NOT_FOUND", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].assistant_response_turn_id", observed={"turn_id": _safe_id(response_id)}, message="Referenced assistant response turn ID does not match any turn")
                        elif len(response_positions) > 1:
                            self.add_issue("ERROR", "RESPONSE_TURN_ID_AMBIGUOUS", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].assistant_response_turn_id", observed={"turn_id": _safe_id(response_id), "turn_indices": response_positions}, message="Referenced assistant response turn ID matches multiple turns")
                        else:
                            response_index = response_positions[0]
                            response_turn = turns[response_index]
                            if response_turn.get("role") != "assistant":
                                self.add_issue("ERROR", "RESPONSE_TURN_NOT_ASSISTANT", split, line_no, cid, point_index, turn_index=response_index, field_path=f"turns[{response_index}].role", observed={"role": response_turn.get("role")}, message="Referenced response turn does not have role assistant")
                                response_index = None
                        if target_index is not None and response_index is not None:
                            if response_index <= target_index:
                                self.add_issue("ERROR", "RESPONSE_NOT_AFTER_TARGET", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].assistant_response_turn_id", observed={"target_index": target_index, "response_index": response_index}, message="Referenced assistant response is not after the target user turn")
                            else:
                                first_assistant = next((idx for idx in range(target_index + 1, len(turns)) if isinstance(turns[idx], dict) and turns[idx].get("role") == "assistant"), None)
                                if first_assistant is not None and response_index != first_assistant:
                                    self.add_issue("REVIEW", "RESPONSE_NOT_FIRST_ASSISTANT", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}].assistant_response_turn_id", observed={"target_index": target_index, "response_index": response_index, "first_assistant_index": first_assistant}, message="Referenced response is not the first assistant turn after the target; requires review")
                            if response_text is not None and isinstance(turns[response_index].get("content"), str):
                                turn_text = turns[response_index]["content"]
                                if response_text == turn_text:
                                    counters["response_text_exact_match_count"] += 1
                                elif _normalize_text(response_text) == _normalize_text(turn_text):
                                    counters["response_text_whitespace_only_mismatch_count"] += 1
                                    self.add_issue("REVIEW", "RESPONSE_TEXT_MISMATCH", split, line_no, cid, point_index, turn_index=response_index, field_path=f"prediction_points[{point_index}].target.response_text", observed={"comparison": "whitespace_only", "reference_length": len(response_text), "turn_length": len(turn_text)}, message="Reference response differs only after whitespace/newline normalization")
                                else:
                                    counters["response_text_other_mismatch_count"] += 1
                                    self.add_issue("REVIEW", "RESPONSE_TEXT_MISMATCH", split, line_no, cid, point_index, turn_index=response_index, field_path=f"prediction_points[{point_index}].target.response_text", observed={"comparison": "different_text", "reference_length": len(response_text), "turn_length": len(turn_text), "reference_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(), "turn_sha256": hashlib.sha256(turn_text.encode("utf-8")).hexdigest()}, message="Reference response text differs from the referenced assistant turn")
                    if target_index is not None:
                        target_text = turns[target_index].get("content")
                        if isinstance(target_text, str):
                            current_user_lengths.append(len(target_text))
                        prefix = turns[:target_index + 1]
                        prefix_text_turns = [{"role": item.get("role"), "content": item.get("content")} for item in prefix if isinstance(item, dict)]
                        history_message_counts.append(len(prefix_text_turns))
                        if len(prefix_text_turns) == len(prefix) and all(isinstance(item.get("content"), str) for item in prefix_text_turns):
                            history_char_lengths.append(sum(len(item["content"]) for item in prefix_text_turns))
                            exact_history_fp = _digest(prefix_text_turns)
                            normalized_turns = [{"role": item["role"], "content": _normalize_text(item["content"])} for item in prefix_text_turns]
                            normalized_history_fp = _digest(normalized_turns)
                            response_repr = response_text if isinstance(response_text, str) else None
                            label_repr = {
                                "emotion_label": emotion if isinstance(emotion, str) and emotion in split_enums["emotion_label"] else None,
                                "user_profile": {dim: profile[dim] for dim in PROFILE_DIMS} if profile_is_readable else None,
                            }
                            labels_readable = label_repr["emotion_label"] is not None and label_repr["user_profile"] is not None and all(all(isinstance(item, str) and item in split_enums[dim] for item in profile[dim]) for dim in PROFILE_DIMS)
                            history_ref = {**point_ref, "history_chars": sum(len(item["content"]) for item in prefix_text_turns), "history_turns": len(prefix_text_turns), "label_signature": _digest(label_repr) if labels_readable else None, "response_sha256": hashlib.sha256(response_repr.encode("utf-8")).hexdigest() if response_repr is not None else None, "exact_history_fp": exact_history_fp, "normalized_history_fp": normalized_history_fp}
                            self.refs_by_exact_history[exact_history_fp].append(history_ref)
                            self.refs_by_normalized_history[normalized_history_fp].append(history_ref)
                            self.history_supervision[exact_history_fp].append(history_ref)
                            if self.checks.get("duplicates", True):
                                prediction_payload = {"history": prefix_text_turns, "target": target}
                                pred_fp = _digest(prediction_payload)
                                if self.refs_by_prediction_fingerprint[pred_fp]:
                                    first = self.refs_by_prediction_fingerprint[pred_fp][0]
                                    if first["split"] == split:
                                        self.add_issue("REVIEW", "DUPLICATE_PREDICTION_POINT", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}]", observed={"first": _location(first), "fingerprint": pred_fp}, message="Prediction point repeats the same exact legal history and target")
                                    elif split == "val":
                                        self.add_issue("REVIEW", "CROSS_SPLIT_DUPLICATE_PREDICTION_POINT", split, line_no, cid, point_index, field_path=f"prediction_points[{point_index}]", observed={"train_example": _location(first), "fingerprint": pred_fp}, message="Validation prediction point exactly matches a training history and target")
                                self.refs_by_prediction_fingerprint[pred_fp].append(point_ref)
                        else:
                            history_unavailable += 1
                    else:
                        history_unavailable += 1

                    if "target_user_turn_id" in point and len(target_positions) == 1:
                        target_key = _typed_key(target_id)
                        point_target_groups[target_key].append({"ref": point_ref, "target": target, "target_id": target_id})

                    if target_index is not None and response_index is not None:
                        valid_points_for_fingerprint.append(point_ref)

                for target_key, grouped in point_target_groups.items():
                    if len(grouped) > 1:
                        labels = {_digest({"emotion_label": item["target"].get("emotion_label"), "user_profile": item["target"].get("user_profile")}) for item in grouped}
                        conflict = len(labels) > 1
                        first = grouped[0]["ref"]
                        self.add_issue("REVIEW", "DUPLICATE_TARGET_POINT", split, line_no, cid, field_path="prediction_points", observed={"target_user_turn_id": _safe_id(grouped[0]["target_id"]), "point_indices": [item["ref"]["prediction_point_index"] for item in grouped], "supervision_labels_conflict": conflict}, message="Multiple prediction points reference the same target user turn")
                        if conflict:
                            self.add_issue("REVIEW", "SUPERVISION_LABEL_CONFLICT", split, line_no, cid, field_path="prediction_points", observed={"point_indices": [item["ref"]["prediction_point_index"] for item in grouped]}, message="Prediction points for the same target turn have differing emotion/profile labels")

                if self.checks.get("duplicates", True) and isinstance(turns, list):
                    turns_fp = _digest(turns)
                    turn_text_length = sum(len(turn.get("content", "")) for turn in turns if isinstance(turn, dict) and isinstance(turn.get("content"), str))
                    duplicate_ref = {**ref, "turns_fp": turns_fp, "turn_text_length": turn_text_length, "turn_count": len(turns)}
                    existing = self.refs_by_turn_fingerprint[turns_fp]
                    if existing:
                        first = existing[0]
                        if first["split"] == split:
                            self.add_issue("REVIEW", "DUPLICATE_CONVERSATION_TURNS", split, line_no, cid, field_path="turns", observed={"first": _location(first), "fingerprint": turns_fp}, message="Conversation turns exactly match an earlier record in this split")
                    self.refs_by_turn_fingerprint[turns_fp].append(duplicate_ref)
                    if all(isinstance(turn, dict) and isinstance(turn.get("role"), str) and isinstance(turn.get("content"), str) for turn in turns):
                        text_turns = [{"role": turn["role"], "content": turn["content"]} for turn in turns]
                        text_fp = _digest(text_turns)
                        text_ref = {**duplicate_ref, "turns_fp": text_fp}
                        self.refs_by_conversation_text_fingerprint[text_fp].append(text_ref)

        after_sha = _sha256_file(path)
        if before_sha != after_sha:
            raise RuntimeError(f"Input changed during audit: {path.relative_to(self.root)}")
        counters["parsed_object_count"] = parsed_objects
        counters["root_shape_analyzable_conversation_count"] = rows_with_root_shape
        counters["schema_invalid_row_count"] = schema_invalid_rows
        counters["response_text_exact_match_count"] += 0
        counters["response_text_whitespace_only_mismatch_count"] += 0
        counters["response_text_other_mismatch_count"] += 0
        self.file_stats[split] = {
            "path": str(path.relative_to(self.root)),
            "sha256_before": before_sha,
            "sha256_after": after_sha,
            "size_bytes": path.stat().st_size,
            "counters": dict(counters),
            "turns_per_conversation": _distribution(turns_per_conversation),
            "turns_per_conversation_frequency": _histogram(turns_per_conversation),
            "prediction_points_per_conversation": _distribution(points_per_conversation),
            "prediction_points_per_conversation_frequency": _histogram(points_per_conversation),
            "emotion": {
                "denominator_valid_emotion_points": emotion_readable,
                "missing_or_invalid_count": emotion_missing_or_invalid,
                "official_label_set": split_enums["emotion_label"],
                "frequencies": {label: emotion_freq.get(label, 0) for label in split_enums["emotion_label"]} | {key: value for key, value in sorted(emotion_freq.items()) if key not in split_enums["emotion_label"]},
            },
            "profile": {
                "readable_points": profile_points,
                "unavailable_points": profile_unavailable,
                "all_three_empty_points": counters["profile_all_empty_points"],
                "partly_empty_points": counters["profile_partly_empty_points"],
                "all_three_empty_rate": round(counters["profile_all_empty_points"] / profile_points, 8) if profile_points else None,
                "dimensions": {
                    dim: {
                        "readable_points": profile_dim_readable[dim],
                        "empty_points": profile_dim_empty[dim],
                        "empty_rate": round(profile_dim_empty[dim] / profile_dim_readable[dim], 8) if profile_dim_readable[dim] else None,
                        "tags_per_point": _distribution(dimension_tag_counts[dim]),
                        "official_label_set": split_enums[dim],
                        "label_frequencies": {label: profile_freq[dim].get(label, 0) for label in split_enums[dim]} | {key: value for key, value in sorted(profile_freq[dim].items()) if key not in split_enums[dim]},
                    } for dim in PROFILE_DIMS
                },
                "empty_dimensions_per_readable_point": _histogram(profile_points_empty_dimensions),
                "label_occurrence_counts": dict(sorted(cooccurrence.items())),
                "cross_dimension_label_cooccurrence_counts": dict(sorted(cross_dim_cooccurrence.items())),
            },
            "memory_refs": {
                "readable_points": memory_empty + memory_nonempty,
                "unavailable_points": memory_unavailable,
                "empty_points": memory_empty,
                "nonempty_points": memory_nonempty,
                "empty_rate": round(memory_empty / (memory_empty + memory_nonempty), 8) if memory_empty + memory_nonempty else None,
                "nonempty_rate": round(memory_nonempty / (memory_empty + memory_nonempty), 8) if memory_empty + memory_nonempty else None,
                "list_length": _distribution(memory_lengths),
                "list_length_frequency": _histogram(memory_lengths),
            },
            "text_lengths_unicode_characters": {
                "all_turn_messages": _distribution(message_lengths),
                "user_messages": _distribution(user_message_lengths),
                "assistant_turn_messages": _distribution(assistant_message_lengths),
                "current_target_user_messages": _distribution(current_user_lengths),
                "reference_responses": _distribution(response_lengths),
                "reference_response_text_repeats": {
                    "distinct_text_count": len(response_hash_counts),
                    "repeated_text_count": sum(count > 1 for count in response_hash_counts.values()),
                    "repeated_occurrence_count": sum(count for count in response_hash_counts.values() if count > 1),
                    "maximum_repetition": max(response_hash_counts.values(), default=0),
                },
                "legal_history_messages": _distribution(history_message_counts),
                "legal_history_characters": _distribution(history_char_lengths),
                "history_unavailable_count": history_unavailable,
                "short_response_candidates": short_responses,
                "long_response_candidates": long_responses,
                "long_message_candidates": long_messages,
                "invalid_character_occurrences": invalid_char_count,
                "replacement_character_occurrences": replacement_char_count,
            },
            "official_schema": self.schema_info.get(split, {"path": None, "available": False, "sha256": None}),
            "schema_error_field_path_counts": dict(sorted(all_schema_paths.items())),
            "schema_error_validator_counts": dict(sorted(all_schema_validators.items())),
            "empty_content_by_source_role": dict(sorted(empty_content_by_source_role.items())),
        }
        return self.file_stats[split]

    @staticmethod
    def _text_anomalies(text: str) -> list[dict[str, Any]]:
        found = []
        for index, char in enumerate(text):
            cp = ord(char)
            if char == "\ufffd" or cp == 0 or (cp < 32 and char not in "\t\n\r") or cp == 127:
                found.append({"index": index, "codepoint": f"U+{cp:04X}"})
        return found

    def _finish_cross_checks(self) -> dict[str, Any]:
        duplicate_summary: dict[str, Any] = {
            "within_split_duplicate_conversation_ids": sum(1 for issue in self.issues if issue["code"] == "DUPLICATE_CONVERSATION_ID"),
            "cross_split_duplicate_conversation_ids": sum(1 for issue in self.issues if issue["code"] == "CROSS_SPLIT_CONVERSATION_ID"),
            "within_split_exact_duplicate_turn_groups": sum(1 for refs in self.refs_by_turn_fingerprint.values() for split in SPLITS if sum(ref["split"] == split for ref in refs) > 1),
            "within_split_exact_duplicate_turn_records": sum(max(0, sum(ref["split"] == split for ref in refs) - 1) for refs in self.refs_by_turn_fingerprint.values() for split in SPLITS),
            "within_split_exact_duplicate_prediction_issues": sum(1 for issue in self.issues if issue["code"] == "DUPLICATE_PREDICTION_POINT"),
            "cross_split_exact_duplicate_prediction_issues": sum(1 for issue in self.issues if issue["code"] == "CROSS_SPLIT_DUPLICATE_PREDICTION_POINT"),
            "same_history_label_conflict_groups_within_split": sum(1 for issue in self.issues if issue["code"] == "CONFLICTING_SUPERVISION"),
            "same_target_duplicate_groups": sum(1 for issue in self.issues if issue["code"] == "DUPLICATE_TARGET_POINT"),
            "same_target_label_conflict_groups": sum(1 for issue in self.issues if issue["code"] == "SUPERVISION_LABEL_CONFLICT"),
        }
        cross = {"exact_turn_overlap_groups": 0, "exact_turn_overlap_validation_conversations": 0,
                 "exact_turn_overlap_short_template_candidates": 0, "exact_turn_overlap_long_candidates": 0,
                 "exact_history_overlap_groups": 0, "exact_history_overlap_validation_points": 0,
                 "normalized_history_only_overlap_groups": 0, "normalized_history_only_overlap_validation_points": 0,
                 "history_overlap_short_template_candidates": 0, "history_overlap_long_candidates": 0,
                 "history_label_conflict_validation_points": 0}
        for history_fp, refs in sorted(self.history_supervision.items()):
            readable_refs = [ref for ref in refs if ref["label_signature"] is not None]
            distinct_labels = {ref["label_signature"] for ref in readable_refs}
            if len(distinct_labels) > 1:
                for split in SPLITS:
                    split_refs = [ref for ref in readable_refs if ref["split"] == split]
                    if len({ref["label_signature"] for ref in split_refs}) > 1:
                        first = split_refs[0]
                        self.add_issue("REVIEW", "CONFLICTING_SUPERVISION", split, first["source_line"], first["conversation_id"], first["prediction_point_index"], field_path="prediction_points[].target", observed={"locations": [_location(ref) for ref in split_refs], "history_fingerprint": history_fp}, message="Identical legal history has differing emotion/profile labels")

        if self.checks.get("cross_split", True):
            for fp, refs in sorted(self.refs_by_conversation_text_fingerprint.items()):
                train_refs = [ref for ref in refs if ref["split"] == "train"]
                val_refs = [ref for ref in refs if ref["split"] == "val"]
                if not train_refs or not val_refs:
                    continue
                cross["exact_turn_overlap_groups"] += 1
                for ref in val_refs:
                    short = ref["turn_text_length"] < int(self.thresholds.get("short_cross_split_chars", 100))
                    kind_key = "exact_turn_overlap_short_template_candidates" if short else "exact_turn_overlap_long_candidates"
                    cross[kind_key] += 1
                    self.add_issue("REVIEW", "TRAIN_VAL_TURNS_OVERLAP", "val", ref["source_line"], ref["conversation_id"], field_path="turns", observed={"train_matches": len(train_refs), "train_examples": [_location(item) for item in train_refs[:3]], "turn_count": ref["turn_count"], "text_characters": ref["turn_text_length"], "candidate_kind": "short_template" if short else "long_exact_overlap", "fingerprint": fp}, message="Validation conversation text exactly matches training text; this is a leakage-risk candidate, not proof", proposed_action="manual_leakage_review_in_D0_2")
                cross["exact_turn_overlap_validation_conversations"] += len(val_refs)

            exact_by_key: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
            for fp, refs in self.refs_by_exact_history.items():
                train_refs = [r for r in refs if r["split"] == "train"]
                val_refs = [r for r in refs if r["split"] == "val"]
                if train_refs and val_refs:
                    exact_by_key[fp] = (train_refs, val_refs)
                    cross["exact_history_overlap_groups"] += 1
                    cross["exact_history_overlap_validation_points"] += len(val_refs)
            exact_normalized_keys = {ref["normalized_history_fp"] for _, val_refs in exact_by_key.values() for ref in val_refs}
            for fp, refs in self.refs_by_normalized_history.items():
                train_refs = [r for r in refs if r["split"] == "train"]
                val_refs = [r for r in refs if r["split"] == "val"]
                if not train_refs or not val_refs:
                    continue
                exact_val_refs = [ref for ref in val_refs if ref["exact_history_fp"] in exact_by_key]
                normalized_only = [ref for ref in val_refs if ref["exact_history_fp"] not in exact_by_key]
                for ref in exact_val_refs:
                    train_exact = exact_by_key[ref["exact_history_fp"]][0]
                    short = ref["history_chars"] < int(self.thresholds.get("short_cross_split_chars", 100))
                    cross["history_overlap_short_template_candidates" if short else "history_overlap_long_candidates"] += 1
                    self.add_issue("REVIEW", "TRAIN_VAL_HISTORY_OVERLAP", "val", ref["source_line"], ref["conversation_id"], ref["prediction_point_index"], field_path="turns[:target_index+1]", observed={"match_kind": "exact", "train_matches": len(train_exact), "train_examples": [_location(item) for item in train_exact[:3]], "history_turns": ref["history_turns"], "history_characters": ref["history_chars"], "candidate_kind": "short_template" if ref["history_chars"] < int(self.thresholds.get("short_cross_split_chars", 100)) else "long_overlap", "fingerprint": ref["exact_history_fp"]}, message="Validation legal history exactly matches a training history; review for template versus leakage")
                if normalized_only:
                    train_norm = train_refs
                    cross["normalized_history_only_overlap_groups"] += 1
                    cross["normalized_history_only_overlap_validation_points"] += len(normalized_only)
                    for ref in normalized_only:
                        short = ref["history_chars"] < int(self.thresholds.get("short_cross_split_chars", 100))
                        cross["history_overlap_short_template_candidates" if short else "history_overlap_long_candidates"] += 1
                        self.add_issue("REVIEW", "TRAIN_VAL_HISTORY_OVERLAP", "val", ref["source_line"], ref["conversation_id"], ref["prediction_point_index"], field_path="turns[:target_index+1]", observed={"match_kind": "whitespace_normalized_only", "train_matches": len(train_norm), "train_examples": [_location(item) for item in train_norm[:3]], "history_turns": ref["history_turns"], "history_characters": ref["history_chars"], "candidate_kind": "short_template" if short else "long_overlap", "fingerprint": fp}, message="Validation legal history matches after whitespace normalization only")
            for exact_fp, refs in sorted(self.history_supervision.items()):
                train_refs = [ref for ref in refs if ref["split"] == "train"]
                val_refs = [ref for ref in refs if ref["split"] == "val"]
                if not train_refs or not val_refs:
                    continue
                label_signatures = {ref["label_signature"] for ref in refs if ref["label_signature"] is not None}
                if len(label_signatures) > 1:
                    cross["history_label_conflict_validation_points"] += len(val_refs)
                    for ref in val_refs:
                        self.add_issue("REVIEW", "CROSS_SPLIT_SUPERVISION_CONFLICT", "val", ref["source_line"], ref["conversation_id"], ref["prediction_point_index"], field_path="prediction_points[].target", observed={"train_matches": len(train_refs), "train_examples": [_location(item) for item in train_refs[:3]], "history_fingerprint": exact_fp}, message="Identical legal history has different emotion/profile labels across splits")
        duplicate_summary.update(cross)
        return duplicate_summary

    def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.load_schemas()
        for split in SPLITS:
            self._read_split(split)
        duplicate_summary = self._finish_cross_checks()
        self.issues.sort(key=lambda issue: (0 if issue["split"] == "train" else 1, issue["source_line"] or -1,
                                            issue["prediction_point_index"] if issue["prediction_point_index"] is not None else -1,
                                            issue["turn_index"] if issue["turn_index"] is not None else -1,
                                            issue["code"], issue["issue_id"]))
        issue_counts = Counter(issue["code"] for issue in self.issues)
        severity_counts = Counter(issue["severity"] for issue in self.issues)
        issue_by_split: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            split_issues = [issue for issue in self.issues if issue["split"] == split]
            by_code = Counter(issue["code"] for issue in split_issues)
            by_severity = Counter(issue["severity"] for issue in split_issues)
            by_code_severity: dict[str, dict[str, int]] = {}
            for issue in split_issues:
                by_code_severity.setdefault(issue["code"], Counter())[issue["severity"]] += 1
            affected_conversations = {(_canonical_json(issue["conversation_id"]) if issue["conversation_id"] is not None else None) for issue in split_issues}
            affected_points = {(issue["source_line"], issue["prediction_point_index"]) for issue in split_issues if issue["prediction_point_index"] is not None}
            affected_by_severity = {}
            for level in ("ERROR", "REVIEW", "INFO"):
                selected = [issue for issue in split_issues if issue["severity"] == level]
                affected_by_severity[level] = {
                    "issue_count": len(selected),
                    "conversation_count": len({(_canonical_json(issue["conversation_id"]) if issue["conversation_id"] is not None else None) for issue in selected} - {None}),
                    "prediction_point_count": len({(issue["source_line"], issue["prediction_point_index"]) for issue in selected if issue["prediction_point_index"] is not None}),
                }
            issue_by_split[split] = {
                "issue_count": len(split_issues),
                "by_severity": {level: by_severity.get(level, 0) for level in ("ERROR", "REVIEW", "INFO")},
                "by_code": dict(sorted(by_code.items())),
                "by_code_severity": {code: {level: counts.get(level, 0) for level in ("ERROR", "REVIEW", "INFO")} for code, counts in sorted(by_code_severity.items())},
                "affected_by_severity": affected_by_severity,
                "affected_conversation_count": len(affected_conversations - {None}),
                "affected_prediction_point_count": len(affected_points),
            }
            self.file_stats[split]["issues"] = issue_by_split[split]
        global_affected_by_severity = {}
        for level in ("ERROR", "REVIEW", "INFO"):
            selected = [issue for issue in self.issues if issue["severity"] == level]
            global_affected_by_severity[level] = {
                "issue_count": len(selected),
                "conversation_count": len({(issue["split"], _canonical_json(issue["conversation_id"])) for issue in selected if issue["conversation_id"] is not None}),
                "prediction_point_count": len({(issue["split"], issue["source_line"], issue["prediction_point_index"]) for issue in selected if issue["prediction_point_index"] is not None}),
            }
        issue_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for issue in self.issues:
            if len(issue_examples[issue["code"]]) < 3:
                issue_examples[issue["code"]].append({key: issue.get(key) for key in ("split", "source_line", "conversation_id", "prediction_point_index", "turn_index")})
        summary = {
            "audit_version": AUDIT_VERSION,
            "methodology": {
                "source_of_truth": "official JSONL and adjacent official JSON Schema",
                "turn_order": "turns array index; turn_id numeric ordering is never used",
                "valid_history": "turns[:target_index+1], roles/content only for history fingerprints",
                "text_length_unit": "Unicode code points (Python len on decoded strings)",
                "percentile_algorithm": "nearest-rank, sorted[ceil(p*n)-1]",
                "exact_fingerprint": "canonical JSON with sorted object keys and compact separators; source text unchanged",
                "normalized_fingerprint": "per-message whitespace collapsed to one space and trimmed; separate candidate-only comparison",
                "semantic_deduplication": "not performed; exact and light whitespace-normalized comparisons only",
                "tokenizer_audit": "not performed; belongs to D0.4",
                "thresholds": self.thresholds,
                "schema_enums_by_split": {split: self.enums_by_split.get(split, self.enums) for split in SPLITS},
            },
            "schema": {
                "all_splits_have_official_schema": all(self.schema_info.get(split, {}).get("available", False) for split in SPLITS),
                "files": self.schema_info,
                "a0_enum_comparison": self._compare_enums(),
                "cross_split_enum_differences": self._compare_split_enums(),
            },
            "splits": self.file_stats,
            "duplicates_and_cross_split": duplicate_summary,
            "issues": {
                "total_count": len(self.issues),
                "by_severity": {level: severity_counts.get(level, 0) for level in ("ERROR", "REVIEW", "INFO")},
                "by_code": dict(sorted(issue_counts.items())),
                "examples_by_code": {code: examples for code, examples in sorted(issue_examples.items())},
                "by_split": issue_by_split,
                "affected_by_severity": global_affected_by_severity,
                "affected_conversation_count": len({(issue["split"], _canonical_json(issue["conversation_id"])) for issue in self.issues if issue["conversation_id"] is not None}),
                "affected_prediction_point_count": len({(issue["split"], issue["source_line"], issue["prediction_point_index"]) for issue in self.issues if issue["prediction_point_index"] is not None}),
            },
            "conclusion_counts": {
                "deterministic_error_issues": severity_counts.get("ERROR", 0),
                "manual_review_issues": severity_counts.get("REVIEW", 0),
                "descriptive_info_issues": severity_counts.get("INFO", 0),
                "affected_conversations": len({(issue["split"], _canonical_json(issue["conversation_id"])) for issue in self.issues if issue["conversation_id"] is not None}),
                "affected_prediction_points": len({(issue["split"], issue["source_line"], issue["prediction_point_index"]) for issue in self.issues if issue["prediction_point_index"] is not None}),
            },
        }
        manifest = self._manifest(summary)
        return summary, manifest

    def _compare_enums(self) -> dict[str, Any]:
        comparison = {}
        for key, fallback in FALLBACK_ENUMS.items():
            actual = set(self.enums[key])
            expected = set(fallback)
            comparison[key] = {"matches_a0_reference": actual == expected, "only_in_official_schema": sorted(actual - expected), "only_in_a0_reference": sorted(expected - actual)}
        return comparison

    def _compare_split_enums(self) -> dict[str, Any]:
        train_enums = self.enums_by_split.get("train", self.enums)
        val_enums = self.enums_by_split.get("val", self.enums)
        return {
            key: {
                "matches": set(train_enums[key]) == set(val_enums[key]),
                "only_in_train_schema": sorted(set(train_enums[key]) - set(val_enums[key])),
                "only_in_val_schema": sorted(set(val_enums[key]) - set(train_enums[key])),
            }
            for key in FALLBACK_ENUMS
        }

    def _manifest(self, summary: dict[str, Any]) -> dict[str, Any]:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, check=True, text=True, capture_output=True).stdout.strip()
            dirty = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=self.root).returncode != 0
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        config_hash = _sha256_file(self.config_path)
        audit_path = Path(__file__).resolve()
        return {
            "audit_version": AUDIT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "project_root": ".",
            "git_commit": commit,
            "git_tracked_files_dirty": dirty,
            "untracked_files_in_dirty_check": False,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "audit_code_path": str(audit_path.relative_to(Path(__file__).resolve().parents[2])),
            "audit_code_sha256": _sha256_file(audit_path),
            "config_path": str(self.config_path.relative_to(self.root)),
            "config_sha256": config_hash,
            "inputs": {
                split: {
                    "path": self.file_stats[split]["path"],
                    "size_bytes": self.file_stats[split]["size_bytes"],
                    "sha256_before": self.file_stats[split]["sha256_before"],
                    "sha256_after": self.file_stats[split]["sha256_after"],
                    "physical_line_count": self.file_stats[split]["counters"].get("physical_line_count", 0),
                    "official_schema": self.schema_info.get(split),
                } for split in SPLITS
            },
            "checks_enabled": self.checks,
            "issue_counts": summary["issues"]["by_severity"],
        }


def _output_path(root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing output.{label} in audit config")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Output path escapes project root: {value}")
    return path


def _render_report(summary: dict[str, Any], manifest: dict[str, Any], command: str) -> str:
    def percentage(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "n/a"

    schema_text = (
        "Train/Val 相邻官方 Schema 均可用，本次据其进行字段、类型与枚举校验。"
        if summary["schema"]["all_splits_have_official_schema"]
        else "未能找到全部官方 Schema；结构检查依据实际数据及可用官方约定，标签集合仅作参考，未完成完整官方 Schema 校验。"
    )
    enum_status = "未能完成官方校验" if not summary["schema"]["all_splits_have_official_schema"] else (
        "一致" if all(item["matches_a0_reference"] for item in summary["schema"]["a0_enum_comparison"].values())
        and all(item["matches"] for item in summary["schema"]["cross_split_enum_differences"].values()) else "存在差异"
    )
    lines = [
        "# D0.1 原始数据质量审计报告", "",
        f"> 审计版本：{AUDIT_VERSION}　|　UTC：{manifest['created_at_utc']}　|　代码 Commit：`{manifest.get('git_commit') or 'unknown'}`",
        "",
        "## 1. 审计范围与依据", "",
        f"只读扫描官方 Train/Val JSONL；未改写原文、未生成清洗数据或训练样本。{schema_text} 引用时序严格按 `turns` 数组下标判定。",
        "",
        f"Schema 可用：{'是' if summary['schema']['all_splits_have_official_schema'] else '否'}。文件、SHA-256、行数、配置/代码指纹和运行信息见 `outputs/data_audit/d0_1/manifest.json`。",
        "",
        "| Split | 输入 | SHA-256 | Schema |",
        "|---|---|---|---|",
    ]
    for split in SPLITS:
        info = manifest["inputs"][split]
        schema = info.get("official_schema") or {}
        lines.append(f"| {split} | `{info['path']}` ({info['size_bytes']:,} bytes) | `{info['sha256_before']}` | {'可用' if schema.get('available') else '未找到'} |")
    lines += ["", "## 2. 基本规模", "", "| Split | 物理行 | 非空行 | 可解析 JSON | 顶层对象/对话行 | Turn 数 | Prediction Point 数 | 空预测点对话 | 不可解析行 | 根结构可分析对话 | Schema 不合规行 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        c = summary["splits"][split]["counters"]
        lines.append(f"| {split} | {c.get('physical_line_count', 0):,} | {c.get('nonempty_line_count', 0):,} | {c.get('parseable_line_count', 0):,} | {c.get('parsed_object_count', 0):,} | {c.get('turn_count', 0):,} | {c.get('prediction_point_count', 0):,} | {c.get('empty_points_conversation_count', 0):,} | {c.get('unparseable_line_count', 0):,} | {c.get('root_shape_analyzable_conversation_count', 0):,} | {c.get('schema_invalid_row_count', 0):,} |")
    for split in SPLITS:
        data = summary["splits"][split]
        counters = data["counters"]
        lines.append(f"{split} 不可统计项：非对象根记录 {counters.get('non_object_root_count', 0):,}；Schema 不合规行 {counters.get('schema_invalid_row_count', 0):,}；情绪缺失/非法 {data['emotion']['missing_or_invalid_count']:,}；画像不可读点 {data['profile']['unavailable_points']:,}；参考回复不可读点 {counters.get('response_text_unavailable_count', 0):,}；合法历史不可定位/计算点 {data['text_lengths_unicode_characters']['history_unavailable_count']:,}。")
        lines.append(f"{split} 每段对话 Turn 数分布 P50/P90/P95/P99/最大：{data['turns_per_conversation']['p50']}/{data['turns_per_conversation']['p90']}/{data['turns_per_conversation']['p95']}/{data['turns_per_conversation']['p99']}/{data['turns_per_conversation']['max']}；Prediction Point 数分布：{data['prediction_points_per_conversation']['p50']}/{data['prediction_points_per_conversation']['p90']}/{data['prediction_points_per_conversation']['p95']}/{data['prediction_points_per_conversation']['p99']}/{data['prediction_points_per_conversation']['max']}。")
    lines += ["", "历史参考为 Train 8,338 段/18,657 点、Val 900 段/2,354 点、Train 空预测点 5 段及 4 个回复时序异常。本次按可解析顶层对象计对话、按数组项计预测点；差异（如有）应结合文件版本和坏行计数解释。", "", "## 3. 结构与关联问题", "", "Issue 条数可能多于受影响样本数；一个预测点可以对应多个 Issue。`INFO` 仅描述现象，空预测点、连续同角色不据此判为脏数据。", "", "| Issue 编码 | 等级 | Train | Val | 示例定位（最多 3 个） | 含义 |", "|---|---|---:|---:|---|---|"]
    issue_desc = {
        "FILE_PARSE": "UTF-8/JSONL 行无法解析", "ROOT_SCHEMA": "顶层结构不符合官方 Schema", "TURN_SCHEMA": "Turn 字段或类型不合规",
        "POINT_SCHEMA": "Prediction Point 字段或类型不合规", "TARGET_SCHEMA": "Target 字段或类型不合规", "PROFILE_SCHEMA": "画像结构不合规",
        "ENUM_CHECK": "情绪/画像枚举不合规", "EMPTY_CONTENT": "空或纯空白消息/参考回复", "EMPTY_POINTS": "合法空预测点对话",
        "TARGET_TURN_NOT_FOUND": "目标用户轮次未找到", "TARGET_TURN_ID_AMBIGUOUS": "目标用户轮次 ID 多重匹配",
        "TARGET_TURN_NOT_USER": "目标轮次角色不是 user", "RESPONSE_TURN_NOT_FOUND": "回复轮次未找到", "RESPONSE_TURN_ID_AMBIGUOUS": "回复轮次 ID 多重匹配",
        "RESPONSE_TURN_NOT_ASSISTANT": "回复轮次角色不是 assistant", "RESPONSE_NOT_AFTER_TARGET": "回复在数组位置上未晚于目标", "RESPONSE_NOT_FIRST_ASSISTANT": "回复不是目标后的第一条 assistant（待复核）",
        "RESPONSE_TEXT_MISMATCH": "参考回复与指向 Turn 不一致", "DUPLICATE_TURN_ID": "对话内 Turn ID 重复", "DUPLICATE_TARGET_POINT": "同一目标用户轮次有多个监督点",
        "SUPERVISION_LABEL_CONFLICT": "相同目标轮次情绪/画像冲突", "DUPLICATE_CONVERSATION_ID": "集合内对话 ID 重复", "CROSS_SPLIT_CONVERSATION_ID": "Train/Val 对话 ID 重合",
        "DUPLICATE_CONVERSATION_TURNS": "集合内 turns 完全重复", "DUPLICATE_PREDICTION_POINT": "合法历史及监督内容完全重复", "CROSS_SPLIT_DUPLICATE_PREDICTION_POINT": "Train/Val 预测点完全重复候选", "CONFLICTING_SUPERVISION": "相同输入标签冲突",
        "TRAIN_VAL_TURNS_OVERLAP": "Train/Val 对话文本完全重合候选", "TRAIN_VAL_HISTORY_OVERLAP": "Train/Val 合法历史重合候选", "CROSS_SPLIT_SUPERVISION_CONFLICT": "Train/Val 相同历史标签冲突",
        "INVALID_TEXT_CHAR": "替换/NUL/控制字符", "SHORT_RESPONSE_CANDIDATE": "短回复复核候选", "LONG_RESPONSE_CANDIDATE": "长回复复核候选", "LONG_MESSAGE_CANDIDATE": "长消息复核候选", "CONSECUTIVE_SAME_ROLE": "连续相同角色（描述性）",
    }
    codes = set(summary["issues"]["by_code"])
    codes.update(issue_desc)
    for code in sorted(codes):
        train_count = summary["splits"]["train"]["issues"]["by_code"].get(code, 0)
        val_count = summary["splits"]["val"]["issues"]["by_code"].get(code, 0)
        if train_count or val_count:
            severities = set()
            for split in SPLITS:
                severities.update(level for level, count in summary["splits"][split]["issues"].get("by_code_severity", {}).get(code, {}).items() if count)
            examples = summary["issues"].get("examples_by_code", {}).get(code, [])
            locations = ", ".join(f"{item['split']}:{item['source_line']}:{item['conversation_id']}" + (f":pp{item['prediction_point_index']}" if item["prediction_point_index"] is not None else "") + (f":turn{item['turn_index']}" if item["turn_index"] is not None else "") for item in examples)
            lines.append(f"| `{code}` | {','.join(sorted(severities, key=lambda item: SEVERITY_RANK[item]))} | {train_count:,} | {val_count:,} | {locations or '—'} | {issue_desc.get(code, '详见 issues.jsonl')} |")
    lines += ["", f"本次空预测点：Train {summary['splits']['train']['counters'].get('empty_points_conversation_count', 0)} 段、Val {summary['splits']['val']['counters'].get('empty_points_conversation_count', 0)} 段；回复时序错误 `RESPONSE_NOT_AFTER_TARGET` 共 {summary['issues']['by_code'].get('RESPONSE_NOT_AFTER_TARGET', 0)} 个。", "", "## 4. 标签与画像分布", "", "情绪按 Prediction Point 计数。空画像合法，画像空值率分母为画像结构可读点数；记忆引用非空为合法数据，不作为清洗理由。", ""]
    for split in SPLITS:
        data = summary["splits"][split]
        emotion = data["emotion"]
        lines += [f"### {split}", "", f"有效情绪标签 {emotion['denominator_valid_emotion_points']:,}；缺失/非法 {emotion['missing_or_invalid_count']:,}。", "", "| 情绪 | 频次 | 占有效情绪标签比例 |", "|---|---:|---:|"]
        denominator = emotion["denominator_valid_emotion_points"]
        for label in emotion["official_label_set"]:
            count = emotion["frequencies"].get(label, 0)
            lines.append(f"| `{label}` | {count:,} | {count / denominator:.2%} |" if denominator else f"| `{label}` | {count:,} | n/a |")
        profile = data["profile"]
        all_empty_rate = f"{profile['all_three_empty_rate']:.2%}" if profile["all_three_empty_rate"] is not None else "n/a"
        lines += ["", f"画像可读点 {profile['readable_points']:,}/{data['counters'].get('prediction_point_count', 0):,}；三维全空 {profile['all_three_empty_points']:,}（可读画像分母 {profile['readable_points']:,}，比例 {all_empty_rate}）；部分维度为空 {profile['partly_empty_points']:,}；不可统计 {profile['unavailable_points']:,}。", "", "| 画像维度 | 可读点 | 空数组数 | 空数组率 | 标签频次（全部 Schema 标签） |", "|---|---:|---:|---:|---|"]
        for dim in PROFILE_DIMS:
            d = profile["dimensions"][dim]
            empty_rate = f"{d['empty_rate']:.2%}" if d["empty_rate"] is not None else "n/a"
            frequencies = ", ".join(f"{key}={value}" for key, value in d["label_frequencies"].items())
            lines.append(f"| `{dim}` | {d['readable_points']:,} | {d['empty_points']:,} | {empty_rate} | {frequencies or '—'} |")
        memory = data["memory_refs"]
        repeat = data["text_lengths_unicode_characters"]["reference_response_text_repeats"]
        top_cooccurrence = sorted(profile["cross_dimension_label_cooccurrence_counts"].items(), key=lambda item: (-item[1], item[0]))[:10]
        cooccurrence_text = ", ".join(f"{key}={value}" for key, value in top_cooccurrence) or "无"
        empty_roles = ", ".join(f"{key}={value}" for key, value in data["empty_content_by_source_role"].items()) or "无"
        lines += ["", f"画像跨维共现 Top 10：{cooccurrence_text}。", f"memory_refs：空 {memory['empty_points']:,}（{percentage(memory['empty_rate'])}）、非空 {memory['nonempty_points']:,}（{percentage(memory['nonempty_rate'])}）、不可统计 {memory['unavailable_points']:,}（按可读数组为分母；列表长度 P50/P90/P95/P99/最大值：{memory['list_length']['p50']}/{memory['list_length']['p90']}/{memory['list_length']['p95']}/{memory['list_length']['p99']}/{memory['list_length']['max']}）。非空引用保留为合法事实。", f"参考回复文本精确重复：不同文本 {repeat['distinct_text_count']:,}，重复文本种类 {repeat['repeated_text_count']:,}，重复覆盖出现次数 {repeat['repeated_occurrence_count']:,}，最高重复 {repeat['maximum_repetition']:,} 次。短回复/长回复/长消息候选分别 {data['text_lengths_unicode_characters']['short_response_candidates']:,}/{data['text_lengths_unicode_characters']['long_response_candidates']:,}/{data['text_lengths_unicode_characters']['long_message_candidates']:,}。", f"空/纯空白内容按来源与角色计数：{empty_roles}。", ""]
    lines += ["## 5. 重复与跨集合候选", "", "精确指纹保留原文，仅规范化对象键顺序和 JSON 序列化；轻度规范化仅折叠/去除空白，用于候选检索。报告不做语义去重，也不据此删除 Val。", "", "```json", json.dumps(summary["duplicates_and_cross_split"], ensure_ascii=False, indent=2, sort_keys=True), "```", "", "Train/Val 的短模板相同不等于泄漏；较长对话/合法历史重合也只是待核验候选，需结合来源与标注审核。", "", "## 6. 原始文本长度", "", "长度单位为 Unicode 字符数；分位数采用 nearest-rank。以下为每集合 Turn 消息、目标用户消息、参考回复及合法历史统计。未运行 Qwen Tokenizer；Token 预算审计属于 D0.4。", "", "| Split | 全部消息 P50/P90/P95/P99/最大 | 当前用户消息 P50/P90/P95/P99/最大 | 参考回复 P50/P90/P95/P99/最大 | 合法历史字符 P50/P90/P95/P99/最大 |", "|---|---|---|---|---|"]
    for split in SPLITS:
        lengths = summary["splits"][split]["text_lengths_unicode_characters"]
        def stat(key: str) -> str:
            d = lengths[key]
            return f"{d['p50']}/{d['p90']}/{d['p95']}/{d['p99']}/{d['max']}"
        lines.append(f"| {split} | {stat('all_turn_messages')} | {stat('current_target_user_messages')} | {stat('reference_responses')} | {stat('legal_history_characters')} |")
    thresholds = summary["methodology"]["thresholds"]
    lines += ["", f"候选阈值（仅人工复核提示）：回复短于 {thresholds.get('short_response_chars')} 字、长于 {thresholds.get('long_response_chars')} 字；消息长于 {thresholds.get('long_message_chars')} 字。", "", "## 7. 质量结论", "", "| 类别 | Issue 数 | 受影响对话数 | 受影响预测点数 |", "|---|---:|---:|---:|"]
    for level, name in (("ERROR", "确定结构/关联错误"), ("REVIEW", "需人工复核"), ("INFO", "描述性合法现象")):
        affected = summary["issues"]["affected_by_severity"][level]
        # Exact affected sample counts by severity are recomputed from issue records in the summary below when present.
        lines.append(f"| {name} | {affected['issue_count']:,} | {affected['conversation_count']:,} | {affected['prediction_point_count']:,} |")
    lines += ["", "Issue 条数不等于独立样本数。各 Issue 定位字段不包含对话正文；需人工复核时依据文件、行号、对话 ID 和数组索引从原始数据读取。", "", "## 8. D0.2 待决清单", "", "| 问题 | 数量/定位 | 建议讨论项（不是已批准规则） | 风险 | 优先级/确认方 |", "|---|---|---|---|---|",
              f"| 结构 Schema Issue | {summary['issues']['by_code'].get('ROOT_SCHEMA', 0) + summary['issues']['by_code'].get('TURN_SCHEMA', 0) + summary['issues']['by_code'].get('POINT_SCHEMA', 0) + summary['issues']['by_code'].get('TARGET_SCHEMA', 0) + summary['issues']['by_code'].get('PROFILE_SCHEMA', 0)} 条 | 决定逐字段隔离、修复授权及回溯策略 | 静默修复可能改变官方监督 | P0 / 数据负责人 |",
              f"| 回复轮次关联 | `RESPONSE_NOT_AFTER_TARGET` {summary['issues']['by_code'].get('RESPONSE_NOT_AFTER_TARGET', 0)}；`RESPONSE_NOT_FIRST_ASSISTANT` {summary['issues']['by_code'].get('RESPONSE_NOT_FIRST_ASSISTANT', 0)} | 确认严格回复时序规则及可接受例外 | 误关联会引入未来信息或错误监督 | P0 / 标注与建模负责人 |",
              f"| 标签集合口径 | 官方 Schema 与 A0 列表 {enum_status} | 如有差异，先确认唯一官方枚举版本；差异详情见 summary.json | 改标签会影响评测协议 | P0 / 赛题数据负责人 |",
              f"| Train/Val 重合候选 | 对话文本 {summary['duplicates_and_cross_split']['exact_turn_overlap_validation_conversations']} 个 Val 对话；历史 {summary['duplicates_and_cross_split']['exact_history_overlap_validation_points'] + summary['duplicates_and_cross_split']['normalized_history_only_overlap_validation_points']} 个 Val 点 | 结合长度、来源及短模板属性人工判断 | 错误删除 Val 会改变官方验证集 | P1 / 数据负责人 |",
              f"| 相同输入/目标标签冲突 | 同一历史 {summary['duplicates_and_cross_split']['same_history_label_conflict_groups_within_split']} 个集合内组；同一目标 {summary['duplicates_and_cross_split']['same_target_label_conflict_groups']} 组；跨集 {summary['duplicates_and_cross_split']['history_label_conflict_validation_points']} 个 Val 点 | 先核对标注口径，再决定是否隔离/保留多标签版本 | 不同合法回复本身不构成冲突 | P1 / 标注负责人 |",
              f"| 空预测点/空画像/非空记忆 | Train {summary['splits']['train']['counters'].get('empty_points_conversation_count', 0)} / {summary['splits']['train']['profile']['all_three_empty_points']} / {summary['splits']['train']['memory_refs']['nonempty_points']}；Val {summary['splits']['val']['counters'].get('empty_points_conversation_count', 0)} / {summary['splits']['val']['profile']['all_three_empty_points']} / {summary['splits']['val']['memory_refs']['nonempty_points']} | 明确是否保留无监督对话；空画像和 memory_refs 非空按既定协议统计 | 擅自删除改变样本分布 | P2 / 建模负责人 |",
              "", "## 9. 复现与完整性", "", f"命令：`{command}`", "", "合成测试：由 `tests/test_data_audit.py` 覆盖；测试命令及结果记录于交付说明。输入在读取前后分别计算 SHA-256；两个值一致方可产出。原始数据未由本工具写入。机器产物：`outputs/data_audit/d0_1/manifest.json`、`summary.json`、`issues.jsonl`。", "", "---", "数据事实、Schema/规则自动判定、以及需人工解释的候选项在本报告中分开表述；本阶段到此停止，不执行 D0.2 清洗。", ""]
    return "\n".join(lines)


def run_config(config_path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    audit_root = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = audit_root / resolved_config
    resolved_config = resolved_config.resolve()
    if not resolved_config.is_relative_to(audit_root):
        raise ValueError("Config path must be inside the project root")
    config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Audit config must be a YAML object")
    output_dir = _output_path(audit_root, config.get("output", {}).get("directory"), "directory")
    report_path = _output_path(audit_root, config.get("output", {}).get("report"), "report")
    # Validate all output destinations before creating directories or reading inputs.
    audit = Audit(audit_root, resolved_config, config)
    summary, manifest = audit.run()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "issues.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for issue in audit.issues:
            stream.write(json.dumps(issue, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    command = f"python -m src.data.audit --config {Path(config_path).as_posix()}"
    report_path.write_text(_render_report(summary, manifest, command), encoding="utf-8", newline="\n")
    print("D0.1 audit completed")
    for split in SPLITS:
        counters = summary["splits"][split]["counters"]
        issue_counts = summary["splits"][split]["issues"]["by_severity"]
        print(f"{split}: conversations={counters.get('parsed_object_count', 0)}, prediction_points={counters.get('prediction_point_count', 0)}, issues={summary['splits'][split]['issues']['issue_count']} (ERROR={issue_counts['ERROR']}, REVIEW={issue_counts['REVIEW']}, INFO={issue_counts['INFO']})")
    print(f"outputs: {output_dir.relative_to(audit_root)}")
    print(f"report: {report_path.relative_to(audit_root)}")
    print(f"input_sha256_unchanged: {all(value['sha256_before'] == value['sha256_after'] for value in manifest['inputs'].values())}")
    return {"summary": summary, "manifest": manifest, "issues": audit.issues}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a read-only D0.1 data quality audit")
    parser.add_argument("--config", required=True, help="YAML config path, relative to project root")
    args = parser.parse_args(argv)
    try:
        run_config(args.config)
    except Exception as error:  # The CLI should make operational failures explicit.
        print(f"D0.1 audit failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
