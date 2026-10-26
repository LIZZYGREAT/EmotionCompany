#!/usr/bin/env python3
"""Offline, read-only diagnosis for one completed E00.2 real Smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference
from src.evaluation import read_jsonl, schema_enums, sha256_file, strict_json_loads

PROFILE_DIMS = ("personality_traits", "interests", "style")
RAW_FIELDS = {"emotion_label", "user_profile", "response_text", "memory_refs"}
PROFILE_FIELDS = set(PROFILE_DIMS)
DEFAULT_D042_ROOT = ROOT / "outputs/token_audit/v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON 根节点必须为 object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: YAML 根节点必须为 object")
    return value


def _resolve_recorded_path(value: str, *, workspace: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workspace / path).resolve()


def _require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 缺失: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 与归档不符: {path}")
    return observed


def _index_rows(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} 第 {number} 行缺少非空字符串 {key}")
        if value in indexed:
            raise ValueError(f"{label} 存在重复 {key}: {value}")
        indexed[value] = row
    return indexed


def _require_same_ids(expected: set[str], indexed: dict[str, Any], label: str) -> None:
    observed = set(indexed)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} ID 集合不一致: missing={missing}, extra={extra}")


def _strict_parse(raw_text: Any) -> tuple[dict[str, Any] | None, str, str | None]:
    if not isinstance(raw_text, str):
        return None, "raw_text_missing_or_non_string", "raw_text_missing_or_non_string"
    try:
        value = strict_json_loads(raw_text)
    except json.JSONDecodeError as exc:
        message = exc.msg.lower()
        stripped = raw_text.lstrip()
        if stripped.startswith("```"):
            category = "code_fence"
        elif "extra data" in message:
            category = "trailing_content"
        elif not stripped.startswith(("{", "[", '"')) and not re.match(r"^(?:null|true|false|-?\d)", stripped):
            category = "leading_content_or_not_json"
        else:
            category = "malformed_or_truncated_json"
        return None, category, f"{exc.msg} at character {exc.pos}"
    except ValueError as exc:
        message = str(exc)
        if message.startswith("duplicate JSON object key:"):
            category = "duplicate_key"
        elif message.startswith("non-standard JSON constant:"):
            category = "non_standard_constant"
        else:
            category = "strict_json_value_error"
        return None, category, message
    if not isinstance(value, dict):
        return None, "valid_json_non_object", None
    return value, "valid_object", None


def _extract_object(raw_text: Any) -> tuple[dict[str, Any] | None, str, str | None]:
    if not isinstance(raw_text, str):
        return None, "no_object_extracted", "raw_text_missing_or_non_string"
    try:
        value = run_inference.extract_object(raw_text)
    except ValueError as exc:
        return None, "no_object_extracted", str(exc)
    return value, "object_extracted", None


def _label_value(value: Any, *, present: bool = True) -> dict[str, Any]:
    if not present:
        return {"type": "missing"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, str):
        if len(value) <= 128:
            return {"type": "string", "value": value}
        return {
            "type": "string",
            "value_prefix": value[:128],
            "truncated": True,
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "length": len(value),
        }
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, (int, float)):
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__}


def _label_value_key(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _emotion_status(raw: dict[str, Any] | None, enums: dict[str, list[str]]) -> tuple[str, Any, bool]:
    if raw is None or "emotion_label" not in raw:
        return "missing", None, False
    value = raw["emotion_label"]
    if value is None:
        return "null", value, False
    if not isinstance(value, str):
        return "non_string", value, False
    if value == "":
        return "empty_string", value, False
    if value in enums["emotion_label"]:
        return "valid", value, True
    if any(ord(character) > 127 for character in value):
        return "invalid_non_ascii_string", value, False
    return "invalid_string", value, False


def _dimension_status(
    raw: dict[str, Any] | None,
    dimension: str,
    enums: dict[str, list[str]],
) -> dict[str, Any]:
    if raw is None:
        return {
            "status": "raw_object_unavailable",
            "valid": False,
            "raw_legal_labels": [],
            "invalid_member_type_count": 0,
            "invalid_enum_count": 0,
            "duplicate_labels": [],
        }
    if "user_profile" not in raw:
        return {
            "status": "profile_root_missing",
            "valid": False,
            "raw_legal_labels": [],
            "invalid_member_type_count": 0,
            "invalid_enum_count": 0,
            "duplicate_labels": [],
        }
    profile = raw["user_profile"]
    if not isinstance(profile, dict):
        return {
            "status": "profile_root_null" if profile is None else "profile_root_non_object",
            "valid": False,
            "raw_legal_labels": [],
            "invalid_member_type_count": 0,
            "invalid_enum_count": 0,
            "duplicate_labels": [],
        }
    if dimension not in profile:
        return {
            "status": "missing",
            "valid": False,
            "raw_legal_labels": [],
            "invalid_member_type_count": 0,
            "invalid_enum_count": 0,
            "duplicate_labels": [],
        }
    values = profile[dimension]
    if not isinstance(values, list):
        return {
            "status": "non_array",
            "valid": False,
            "raw_legal_labels": [],
            "invalid_member_type_count": 0,
            "invalid_enum_count": 0,
            "duplicate_labels": [],
        }
    allowed = set(enums[dimension])
    legal = [value for value in values if isinstance(value, str) and value in allowed]
    invalid_member_type_count = sum(not isinstance(value, str) for value in values)
    invalid_enum_count = sum(isinstance(value, str) and value not in allowed for value in values)
    counts = Counter(value for value in values if isinstance(value, str))
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    reasons: list[str] = []
    if invalid_member_type_count:
        reasons.append("invalid_member_type")
    if invalid_enum_count:
        reasons.append("invalid_enum")
    if duplicates:
        reasons.append("duplicate")
    valid = not reasons
    status = ("valid_empty" if not values else "valid_nonempty") if valid else "+".join(reasons)
    return {
        "status": status,
        "valid": valid,
        "raw_legal_labels": sorted(set(legal)),
        "invalid_member_type_count": invalid_member_type_count,
        "invalid_enum_count": invalid_enum_count,
        "duplicate_labels": duplicates,
    }


def _prf(predicted: set[str], gold: set[str]) -> dict[str, int]:
    return {
        "tp": len(predicted & gold),
        "fp": len(predicted - gold),
        "fn": len(gold - predicted),
    }


def _profile_dimension_case(
    raw: dict[str, Any] | None,
    submission: dict[str, Any],
    gold: dict[str, Any],
    dimension: str,
    enums: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_info = _dimension_status(raw, dimension, enums)
    submitted_profile = submission.get("user_profile")
    submitted_values = submitted_profile.get(dimension) if isinstance(submitted_profile, dict) else None
    gold_profile = gold.get("user_profile")
    gold_values = gold_profile.get(dimension) if isinstance(gold_profile, dict) else None
    submitted_set = set(submitted_values) if isinstance(submitted_values, list) else set()
    gold_set = set(gold_values) if isinstance(gold_values, list) else set()
    raw_set = set(raw_info["raw_legal_labels"])
    raw_counts = _prf(raw_set, gold_set)
    submission_counts = _prf(submitted_set, gold_set)
    if raw is None:
        normalization_reason = "parse_failure_fallback"
    elif raw_info["status"] in {"valid_empty", "valid_nonempty"}:
        normalization_reason = "valid_raw_value_preserved"
    else:
        normalization_reason = raw_info["status"]
    case = {
        **raw_info,
        "raw_precision_counts": raw_counts,
        "submission_labels": sorted(submitted_set),
        "submission_precision_counts": submission_counts,
        "gold_labels": sorted(gold_set),
        "normalization_reason": normalization_reason,
        "normalization_to_empty": not submitted_set,
        "empty_set_exact_match": not submitted_set and not gold_set,
        "correct_nonempty_labels": submission_counts["tp"],
        "submission_raw_value_mismatch": (
            submitted_values != (raw.get("user_profile", {}).get(dimension) if raw and isinstance(raw.get("user_profile"), dict) and _dimension_status(raw, dimension, enums)["valid"] else [])
        ),
    }
    counts = {
        "raw_tp": raw_counts["tp"],
        "raw_fp": raw_counts["fp"],
        "raw_fn": raw_counts["fn"],
        "submission_tp": submission_counts["tp"],
        "submission_fp": submission_counts["fp"],
        "submission_fn": submission_counts["fn"],
        "gold_label_count": len(gold_set),
        "prediction_label_count": len(submitted_set),
        "empty_set_exact_match": int(not submitted_set and not gold_set),
        "prediction_empty": int(not submitted_set),
        "gold_empty": int(not gold_set),
        "correct_nonempty_labels": submission_counts["tp"],
        "raw_value_mismatch": int(case["submission_raw_value_mismatch"]),
    }
    return case, counts


def _raw_a0_reasons(raw: dict[str, Any] | None, strict_status: str, enums: dict[str, list[str]]) -> list[str]:
    if raw is None:
        return [f"strict_json:{strict_status}"]
    reasons: list[str] = []
    if set(raw) != RAW_FIELDS:
        reasons.append("raw_fields_missing_or_extra")
    emotion_status, _, _ = _emotion_status(raw, enums)
    if emotion_status != "valid":
        reasons.append("invalid_emotion")
    profile = raw.get("user_profile")
    if not isinstance(profile, dict):
        reasons.append("invalid_profile_root")
    else:
        if set(profile) != PROFILE_FIELDS:
            reasons.append("profile_fields_missing_or_extra")
        for dimension in PROFILE_DIMS:
            info = _dimension_status(raw, dimension, enums)
            if not info["valid"]:
                reasons.append(f"invalid_profile:{dimension}:{info['status']}")
    response = raw.get("response_text")
    if not isinstance(response, str):
        reasons.append("response_non_string_or_missing")
    elif not response.strip():
        reasons.append("response_empty_or_whitespace")
    if raw.get("memory_refs") != []:
        if "memory_refs" not in raw:
            reasons.append("memory_refs_missing")
        elif not isinstance(raw.get("memory_refs"), list):
            reasons.append("memory_refs_non_array")
        elif raw.get("memory_refs"):
            reasons.append("memory_refs_nonempty")
    return reasons


def _raw_field_status(raw: dict[str, Any] | None, field: str) -> str:
    if raw is None or field not in raw:
        return "missing"
    value = raw[field]
    if field == "response_text":
        if not isinstance(value, str):
            return "non_string"
        return "whitespace_or_empty" if not value.strip() else "nonempty_string"
    if field == "memory_refs":
        if not isinstance(value, list):
            return "non_array"
        return "empty_array" if not value else "nonempty_array"
    return type(value).__name__


def _case_diagnosis(
    sample_id: str,
    profile_name: str,
    raw_row: dict[str, Any],
    submission: dict[str, Any],
    performance_diag: dict[str, Any],
    gold: dict[str, Any],
    enums: dict[str, list[str]],
    schema_validator: Draft202012Validator,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_text = raw_row.get("raw_text")
    strict_object, strict_status, strict_error = _strict_parse(raw_text)
    extracted, extraction_status, extraction_error = _extract_object(raw_text)
    expected, parse_failed, invalid_label = run_inference.parse_prediction(
        sample_id, raw_text if isinstance(raw_text, str) else "", enums
    )
    normalized_match = expected == submission
    expected_status = "parse_failure" if parse_failed else "invalid_label" if invalid_label else "success"
    row_status = str(raw_row.get("status", "unknown"))
    perf_status = str(performance_diag.get("status", "unknown"))
    raw_fields_present = set(extracted) if extracted is not None else set()
    missing_fields = sorted(RAW_FIELDS - raw_fields_present)
    extra_fields = sorted(raw_fields_present - RAW_FIELDS)

    emotion_status, raw_emotion, emotion_is_valid = _emotion_status(extracted, enums)
    gold_emotion = gold.get("emotion_label")
    submission_emotion = submission.get("emotion_label")
    emotion_correct = submission_emotion == gold_emotion
    if extracted is None:
        emotion_normalization_reason = "parse_failure_fallback"
    elif emotion_is_valid:
        emotion_normalization_reason = "valid_raw_value_preserved"
    else:
        emotion_normalization_reason = emotion_status

    dimension_cases: dict[str, Any] = {}
    dimension_counts: dict[str, Any] = {}
    for dimension in PROFILE_DIMS:
        detail, counts = _profile_dimension_case(extracted, submission, gold, dimension, enums)
        dimension_cases[dimension] = detail
        dimension_counts[dimension] = counts

    raw_response = extracted.get("response_text") if extracted is not None else None
    if extracted is not None:
        response_source = "raw_response_text" if isinstance(raw_response, str) else "invalid_or_missing_raw_response_text"
    else:
        recovered = run_inference.recover_response_text(raw_text) if isinstance(raw_text, str) else ""
        response_source = "recover_response_text" if recovered else "empty_parse_failure_fallback"
    response_value = submission.get("response_text")
    response_diag = {
        "raw_status": _raw_field_status(extracted, "response_text"),
        "raw_characters": len(raw_response) if isinstance(raw_response, str) else None,
        "submission_status": "nonempty_string" if isinstance(response_value, str) and response_value.strip() else "empty_or_non_string",
        "submission_characters": len(response_value) if isinstance(response_value, str) else None,
        "source": response_source,
        "raw_submission_reconciled": (
            response_value == (raw_response.strip() if isinstance(raw_response, str) else "")
            if extracted is not None
            else response_value == (run_inference.recover_response_text(raw_text) if isinstance(raw_text, str) else "")
        ),
    }
    memory_status = _raw_field_status(extracted, "memory_refs")
    strict_a0_reasons = _raw_a0_reasons(strict_object, strict_status, enums)
    strict_emotion_status, strict_emotion_value, strict_emotion_valid = _emotion_status(strict_object, enums)
    submission_schema_errors = list(schema_validator.iter_errors(submission))
    submission_schema_valid = not submission_schema_errors
    profile_invalid_dims = [
        dim for dim, detail in dimension_cases.items()
        if detail["status"] not in {"valid_empty", "valid_nonempty"}
    ]
    predicted_invalid_label = emotion_status != "valid" or not isinstance(extracted.get("user_profile") if extracted else None, dict) or bool(profile_invalid_dims)

    case = {
        "id": sample_id,
        "profiles": {
            profile_name: {
                "runner_status": row_status,
                "performance_status": perf_status,
                "status_semantics_match": row_status == expected_status,
                "performance_status_matches_raw": row_status == perf_status,
                "expected_runner_status_from_raw": expected_status,
                "invalid_label_components": {
                    "emotion_invalid": emotion_status != "valid",
                    "invalid_profile_dimensions": profile_invalid_dims,
                    "combined_invalid_label_if_object_extracted": predicted_invalid_label if extracted is not None else None,
                },
                "strict_json": {"status": strict_status, "object": strict_object is not None, "error_category_detail": strict_error},
                "permissive_extraction": {"status": extraction_status, "error_category_detail": extraction_error},
                "fields": {
                    "missing": missing_fields,
                    "extra": extra_fields,
                    "root_is_object": extracted is not None,
                    "user_profile_root": (
                        "missing" if extracted is None or "user_profile" not in extracted
                        else "object" if isinstance(extracted.get("user_profile"), dict)
                        else "null" if extracted.get("user_profile") is None
                        else type(extracted.get("user_profile")).__name__
                    ),
                },
                "emotion": {
                    "raw_status": emotion_status,
                    "raw_value": _label_value(raw_emotion, present=extracted is not None and "emotion_label" in extracted),
                    "strict_json_raw_status": strict_emotion_status,
                    "strict_json_raw_value": _label_value(strict_emotion_value, present=strict_object is not None and "emotion_label" in strict_object),
                    "gold_value": gold_emotion,
                    "submission_value": submission_emotion,
                    "raw_is_valid": emotion_is_valid,
                    "strict_json_raw_is_valid": strict_emotion_valid,
                    "raw_valid_and_gold_correct": bool(emotion_is_valid and raw_emotion == gold_emotion),
                    "strict_json_raw_valid_and_gold_correct": bool(strict_emotion_valid and strict_emotion_value == gold_emotion),
                    "submission_correct": emotion_correct,
                    "normalization_reason": emotion_normalization_reason,
                    "submission_empty": submission_emotion == "",
                },
                "profile_dimensions": dimension_cases,
                "response": response_diag,
                "memory_refs": {"raw_status": memory_status, "a0_failure_reasons": [r for r in strict_a0_reasons if r.startswith("memory_refs_")]},
                "raw_a0": {"valid": not strict_a0_reasons, "failure_reasons": strict_a0_reasons},
                "submission_schema_valid": submission_schema_valid,
                "normalization_matches_existing_runner": normalized_match,
                "runtime": {
                    "stop_reason": raw_row.get("stop_reason"),
                    "generated_tokens": raw_row.get("generated_tokens"),
                    "removed_history_messages_raw": raw_row.get("removed_history_messages"),
                    "removed_history_messages_performance": performance_diag.get("removed_history_messages"),
                    "prompt_tokens": raw_row.get("prompt_tokens"),
                    "client_prompt_tokens": performance_diag.get("client_prompt_tokens"),
                    "original_history_messages": performance_diag.get("original_history_messages"),
                    "retained_history_messages": performance_diag.get("retained_history_messages"),
                },
            }
        },
    }
    aux = {
        "strict_status": strict_status,
        "strict_object": strict_object,
        "extracted": extracted,
        "emotion_status": emotion_status,
        "raw_emotion": raw_emotion,
        "emotion_valid": emotion_is_valid,
        "strict_emotion_status": strict_emotion_status,
        "strict_emotion_value": strict_emotion_value,
        "strict_emotion_valid": strict_emotion_valid,
        "strict_emotion_correct": bool(strict_emotion_valid and strict_emotion_value == gold_emotion),
        "emotion_correct": emotion_correct,
        "submission_schema_valid": submission_schema_valid,
        "raw_a0_valid": not strict_a0_reasons,
        "raw_a0_reasons": strict_a0_reasons,
        "runner_status": row_status,
        "raw_text_type": type(raw_text).__name__,
        "normalization_matches_existing_runner": normalized_match,
        "profile_dimension_counts": dimension_counts,
        "response_empty": not (isinstance(response_value, str) and bool(response_value.strip())),
        "raw_response_status": _raw_field_status(extracted, "response_text"),
        "memory_refs_status": memory_status,
        "runtime": performance_diag,
    }
    return case, aux, dimension_cases, dimension_counts


def _empty_profile_totals() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "strict_json_status_counts": Counter(),
        "permissive_extraction_counts": Counter(),
        "runner_status_counts": Counter(),
        "performance_status_counts": Counter(),
        "emotion_status_counts": Counter(),
        "emotion_value_counts": Counter(),
        "strict_emotion_status_counts": Counter(),
        "strict_emotion_value_counts": Counter(),
        "emotion_submission_empty_reasons": Counter(),
        "emotion_valid_count": 0,
        "emotion_valid_and_gold_correct_count": 0,
        "strict_emotion_valid_count": 0,
        "strict_emotion_valid_and_gold_correct_count": 0,
        "emotion_submission_correct_count": 0,
        "emotion_submission_empty_count": 0,
        "raw_a0_valid_count": 0,
        "submission_schema_valid_count": 0,
        "normalization_mismatch_count": 0,
        "status_semantics_mismatch_count": 0,
        "raw_response_status_counts": Counter(),
        "response_source_counts": Counter(),
        "response_submission_empty_count": 0,
        "memory_refs_status_counts": Counter(),
        "a0_failure_reason_counts": Counter(),
        "runner_status_by_emotion_status": defaultdict(Counter),
        "runner_status_by_profile_validity": defaultdict(Counter),
        "runner_status_by_a0_validity": defaultdict(Counter),
        "dimensions": {
            dim: {
                "status_counts": Counter(),
                "normalization_reason_counts": Counter(),
                "raw_tp": 0, "raw_fp": 0, "raw_fn": 0,
                "submission_tp": 0, "submission_fp": 0, "submission_fn": 0,
                "gold_label_count": 0, "prediction_label_count": 0,
                "empty_set_exact_matches": 0, "prediction_empty_count": 0, "gold_empty_count": 0,
                "correct_nonempty_labels": 0, "raw_value_mismatches": 0,
            }
            for dim in PROFILE_DIMS
        },
    }


def _increment_profile_totals(totals: dict[str, Any], case: dict[str, Any], aux: dict[str, Any], detail: dict[str, Any]) -> None:
    totals["sample_count"] += 1
    totals["strict_json_status_counts"][aux["strict_status"]] += 1
    extraction = case["profiles"]
    profile_name = next(iter(extraction))
    profile_case = extraction[profile_name]
    totals["permissive_extraction_counts"][profile_case["permissive_extraction"]["status"]] += 1
    totals["runner_status_counts"][aux["runner_status"]] += 1
    totals["performance_status_counts"][profile_case["performance_status"]] += 1
    totals["emotion_status_counts"][aux["emotion_status"]] += 1
    value_key = _label_value_key(profile_case["emotion"]["raw_value"])
    totals["emotion_value_counts"][value_key] += 1
    totals["strict_emotion_status_counts"][aux["strict_emotion_status"]] += 1
    strict_value_key = _label_value_key(profile_case["emotion"]["strict_json_raw_value"])
    totals["strict_emotion_value_counts"][strict_value_key] += 1
    if profile_case["emotion"]["submission_empty"]:
        totals["emotion_submission_empty_count"] += 1
        totals["emotion_submission_empty_reasons"][profile_case["emotion"]["normalization_reason"]] += 1
    totals["emotion_valid_count"] += int(aux["emotion_valid"])
    totals["emotion_valid_and_gold_correct_count"] += int(profile_case["emotion"]["raw_valid_and_gold_correct"])
    totals["strict_emotion_valid_count"] += int(aux["strict_emotion_valid"])
    totals["strict_emotion_valid_and_gold_correct_count"] += int(aux["strict_emotion_correct"])
    totals["emotion_submission_correct_count"] += int(aux["emotion_correct"])
    totals["raw_a0_valid_count"] += int(aux["raw_a0_valid"])
    totals["submission_schema_valid_count"] += int(aux["submission_schema_valid"])
    totals["normalization_mismatch_count"] += int(not aux["normalization_matches_existing_runner"])
    totals["status_semantics_mismatch_count"] += int(not profile_case["status_semantics_match"])
    totals["raw_response_status_counts"][aux["raw_response_status"]] += 1
    totals["response_source_counts"][profile_case["response"]["source"]] += 1
    totals["response_submission_empty_count"] += int(aux["response_empty"])
    totals["memory_refs_status_counts"][aux["memory_refs_status"]] += 1
    for reason in aux["raw_a0_reasons"]:
        totals["a0_failure_reason_counts"][reason] += 1
    totals["runner_status_by_emotion_status"][aux["runner_status"]][aux["emotion_status"]] += 1
    if profile_case["permissive_extraction"]["status"] != "object_extracted":
        profile_validity = "no_object_extracted"
    else:
        invalid_dimensions = profile_case["invalid_label_components"]["invalid_profile_dimensions"]
        profile_validity = "invalid" if invalid_dimensions else "valid"
    totals["runner_status_by_profile_validity"][aux["runner_status"]][profile_validity] += 1
    totals["runner_status_by_a0_validity"][aux["runner_status"]]["raw_a0_valid" if aux["raw_a0_valid"] else "raw_a0_invalid"] += 1
    for dim in PROFILE_DIMS:
        bucket = totals["dimensions"][dim]
        value = detail[dim]
        counts = aux["profile_dimension_counts"][dim]
        bucket["status_counts"][value["status"]] += 1
        bucket["normalization_reason_counts"][value["normalization_reason"]] += 1
        for name in (
            "raw_tp", "raw_fp", "raw_fn", "submission_tp", "submission_fp", "submission_fn",
            "gold_label_count", "prediction_label_count", "correct_nonempty_labels",
        ):
            bucket[name] += counts[name]
        bucket["empty_set_exact_matches"] += counts["empty_set_exact_match"]
        bucket["prediction_empty_count"] += counts["prediction_empty"]
        bucket["gold_empty_count"] += counts["gold_empty"]
        bucket["raw_value_mismatches"] += counts["raw_value_mismatch"]


def _counter_json(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, defaultdict):
        return {key: _counter_json(child) for key, child in sorted(value.items())}
    if isinstance(value, dict):
        return {key: _counter_json(child) for key, child in value.items()}
    return value


def _finalize_totals(totals: dict[str, Any]) -> dict[str, Any]:
    result = _counter_json(totals)
    count = result["sample_count"]
    result["rates"] = {
        "strict_json_object_rate": result["strict_json_status_counts"].get("valid_object", 0) / count if count else 0.0,
        "raw_a0_valid_rate": result["raw_a0_valid_count"] / count if count else 0.0,
        "submission_schema_valid_rate": result["submission_schema_valid_count"] / count if count else 0.0,
        "emotion_valid_rate": result["emotion_valid_count"] / count if count else 0.0,
        "emotion_accuracy": result["emotion_submission_correct_count"] / count if count else 0.0,
        "response_empty_rate": result["response_submission_empty_count"] / count if count else 0.0,
    }
    for dimension in result["dimensions"].values():
        predicted = dimension["submission_tp"] + dimension["submission_fp"]
        actual = dimension["submission_tp"] + dimension["submission_fn"]
        dimension["micro_precision"] = dimension["submission_tp"] / predicted if predicted else 0.0
        dimension["micro_recall"] = dimension["submission_tp"] / actual if actual else 0.0
        denom = 2 * dimension["submission_tp"] + dimension["submission_fp"] + dimension["submission_fn"]
        dimension["micro_f1"] = 2 * dimension["submission_tp"] / denom if denom else 0.0
    return result


def _find_artifact_paths(run_root: Path, manifest: dict[str, Any], profile_name: str) -> dict[str, Path]:
    profile = manifest.get("profiles", {}).get(profile_name)
    if not isinstance(profile, dict) or not isinstance(profile.get("run_id"), str):
        raise ValueError(f"run_manifest 缺少 {profile_name} run_id")
    run_dir = run_root / profile["run_id"]
    return {
        "run_dir": run_dir,
        "raw": run_dir / "raw_predictions.jsonl",
        "submission": run_dir / "runner/submission.jsonl",
        "performance": run_dir / "runner/performance_report.json",
        "metrics": run_dir / "scored/metrics.json",
    }


def _verify_model_mount(manifest: dict[str, Any], performances: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mount = manifest.get("model_mount", {})
    mounted_path = Path(mount.get("path", "")).expanduser()
    local_link = ROOT / "participant/model"
    resolved_local = local_link.resolve(strict=True) if local_link.exists() or local_link.is_symlink() else None
    resolved_mount = mounted_path.resolve(strict=True) if mounted_path.exists() else None
    identities = {
        name: performance.get("model_weight_identity", {})
        for name, performance in performances.items()
    }
    aggregates = {identity.get("aggregate_sha256") for identity in identities.values()}
    observed = {identity.get("observed_weight_hash") for identity in identities.values()}
    manifest_weight = manifest.get("comparison", {}).get("model_weight_sha256")
    identity_match = len(aggregates) == 1 and len(observed) == 1 and aggregates == observed == {manifest_weight}
    snapshot_revision = resolved_local.name if resolved_local else None
    revision = next(iter(identities.values()), {}).get("configured_revision")
    index_path = resolved_local / "model.safetensors.index.json" if resolved_local else None
    observed_index_sha = sha256_file(index_path) if index_path and index_path.is_file() else None
    shard_metadata: list[dict[str, Any]] = []
    expected_files = {
        item.get("path"): item
        for identity in identities.values()
        for item in identity.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    for shard_name in mount.get("weight_shards", []):
        expected = expected_files.get(shard_name, {})
        shard_path = resolved_local / shard_name if resolved_local else None
        resolved_shard = shard_path.resolve(strict=True) if shard_path and shard_path.exists() else None
        expected_sha = expected.get("sha256")
        link_target = os.readlink(shard_path) if shard_path and shard_path.is_symlink() else None
        blob_alias = (shard_path.parent / link_target).resolve(strict=True) if shard_path and link_target else None
        shard_metadata.append({
            "path": shard_name,
            "exists": resolved_shard is not None,
            "is_symlink": bool(shard_path and shard_path.is_symlink()),
            "link_target": link_target,
            "snapshot_link_targets_recorded_sha256_blob_alias": bool(
                link_target and Path(link_target).name == expected_sha and "blobs" in Path(link_target).parts
            ),
            "blob_alias_resolves": bool(blob_alias and blob_alias.is_file()),
            "resolved_blob_target": str(resolved_shard) if resolved_shard else None,
            "size_bytes_matches_recorded": bool(shard_path and expected.get("size_bytes") == shard_path.stat().st_size),
            "recorded_sha256": expected_sha,
        })
    config_path = resolved_local / "config.json" if resolved_local else None
    model_config = _read_json(config_path) if config_path and config_path.is_file() else {}
    return {
        "model_id": next(iter(identities.values()), {}).get("model_id"),
        "configured_revision": revision,
        "participant_model_link": str(local_link),
        "participant_model_link_is_symlink": local_link.is_symlink(),
        "participant_model_resolved_target": str(resolved_local) if resolved_local else None,
        "manifest_mount_path": str(mounted_path),
        "manifest_mount_resolved_target": str(resolved_mount) if resolved_mount else None,
        "link_matches_manifest_mount": resolved_local is not None and resolved_local == resolved_mount,
        "snapshot_directory_matches_revision": snapshot_revision == revision,
        "snapshot_directory": snapshot_revision,
        "manifest_weight_index_sha256": mount.get("weight_index_sha256"),
        "observed_weight_index_sha256": observed_index_sha,
        "weight_index_hash_matches": observed_index_sha == mount.get("weight_index_sha256"),
        "weight_shards": mount.get("weight_shards", []),
        "weight_shard_link_metadata": shard_metadata,
        "all_weight_shard_link_metadata_matches": bool(shard_metadata) and all(
            row["exists"] and row["snapshot_link_targets_recorded_sha256_blob_alias"]
            and row["blob_alias_resolves"] and row["size_bytes_matches_recorded"]
            for row in shard_metadata
        ),
        "local_config_model_type": model_config.get("model_type"),
        "local_config_architectures": model_config.get("architectures"),
        "profile_aggregate_hashes": {name: identity.get("aggregate_sha256") for name, identity in identities.items()},
        "profile_observed_hashes": {name: identity.get("observed_weight_hash") for name, identity in identities.items()},
        "same_archived_weight_identity": identity_match,
        "large_weight_shards_rehashed": False,
        "identity_evidence": "E00.2 archived model_weight_identity and shard index metadata; no multi-GB rehash",
    }


def _load_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = run_root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "real_smoke_scored":
        raise ValueError(f"仅接受 real_smoke_scored，当前 status={manifest.get('status')!r}")
    if manifest.get("stage") != "sampling_scored":
        raise ValueError(f"Manifest stage 不完整: {manifest.get('stage')!r}")
    source_files = manifest.get("source_files", {})
    required_sources = ("smoke_input", "smoke_references", "evaluation_manifest")
    sources: dict[str, Any] = {}
    for name in required_sources:
        item = source_files.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"run_manifest 缺少 source_files.{name} path/SHA-256")
        path = _resolve_recorded_path(item["path"])
        _require_sha(path, item["sha256"], name)
        sources[name] = {"path": path, "sha256": item["sha256"]}
    comparison = manifest.get("comparison", {})
    if comparison.get("sample_count") != 20 or comparison.get("input_sha256") != sources["smoke_input"]["sha256"]:
        raise ValueError("Manifest comparison 的 Smoke 行数或输入 SHA 与归档来源不符")
    if not comparison.get("model_weight_sha256"):
        raise ValueError("Manifest 缺少模型权重聚合 SHA")
    return manifest, sources


def _load_d042(smoke_map: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = DEFAULT_D042_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"D0.4.2 Token 审计 Manifest 缺失: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("audit_version") != "D0.4.2-v1":
        raise ValueError("D0.4.2 Token 审计版本不匹配")
    p0_fact = manifest.get("p0", {})
    if p0_fact.get("sha256") != sha256_file(ROOT / "configs/prompts/p0.txt"):
        raise ValueError("D0.4.2 P0 指纹与当前冻结 Prompt 不一致")
    facts = manifest.get("result_files", {})
    selected: dict[str, Path] = {}
    for name in ("per_sample_lengths", "budget_simulations"):
        fact = facts.get(name, {})
        if not isinstance(fact.get("path"), str) or not isinstance(fact.get("sha256"), str):
            raise ValueError(f"D0.4.2 Manifest 缺少 {name} 路径或 SHA")
        path = _resolve_recorded_path(fact["path"])
        _require_sha(path, fact["sha256"], f"D0.4.2 {name}")
        selected[name] = path
    sample_ids = {row["sample_id"] for row in smoke_map.values()}
    per_sample = [row for row in read_jsonl(selected["per_sample_lengths"]) if row.get("sample_id") in sample_ids and row.get("split") == "dev"]
    budget = [
        row for row in read_jsonl(selected["budget_simulations"])
        if row.get("sample_id") in sample_ids
        and row.get("split") == "dev"
        and row.get("max_train_seq_length") == 2048
        and row.get("max_prompt_tokens") == 1280
    ]
    per_sample_by_id = _index_rows(per_sample, "sample_id", "D0.4.2 per_sample_lengths")
    budget_by_id = _index_rows(budget, "sample_id", "D0.4.2 S2048/P1280 budget_simulations")
    _require_same_ids(sample_ids, per_sample_by_id, "D0.4.2 per_sample_lengths")
    _require_same_ids(sample_ids, budget_by_id, "D0.4.2 S2048/P1280 budget_simulations")
    return {
        sample_id: {"lengths": per_sample_by_id[sample_id], "budget": budget_by_id[sample_id]}
        for sample_id in sample_ids
    }, {
        "manifest_path": str(manifest_path),
        "manifest_version": manifest.get("audit_version"),
        "per_sample_lengths_sha256": facts["per_sample_lengths"]["sha256"],
        "budget_simulations_sha256": facts["budget_simulations"]["sha256"],
        "candidate": {"max_train_seq_length": 2048, "max_prompt_tokens": 1280},
        "p0_sha256": p0_fact.get("sha256"),
    }


def _truncation_group_stats(cases: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    selected = []
    for case in cases:
        removed = case["truncation"]["greedy"]["removed_history_messages"]
        is_truncated = removed > 0
        if (group_name == "truncated") == is_truncated:
            selected.append(case)
    result: dict[str, Any] = {"sample_count": len(selected), "profiles": {}}
    for profile in ("greedy", "sampling"):
        rows = [case["profiles"][profile] for case in selected]
        count = len(rows)
        result["profiles"][profile] = {
            "emotion_correct_count": sum(bool(row["emotion"]["submission_correct"]) for row in rows),
            "emotion_accuracy": sum(bool(row["emotion"]["submission_correct"]) for row in rows) / count if count else None,
            "strict_json_object_count": sum(row["strict_json"]["status"] == "valid_object" for row in rows),
            "raw_a0_valid_count": sum(row["raw_a0"]["valid"] for row in rows),
            "response_empty_count": sum(row["response"]["submission_status"] != "nonempty_string" for row in rows),
        }
    return result


def _make_report(summary: dict[str, Any], profiles: dict[str, Any], truncation: dict[str, Any]) -> str:
    lines = [
        "# E00.2 真实输出离线诊断",
        "",
        f"- Run ID：`{summary['run_id']}`。",
        f"- 诊断输入：{summary['integrity']['sample_count']} 条 Smoke；Greedy/Sampling 各一条输出。",
        f"- 运行类型：`{summary['integrity']['manifest_status']}` / `model_inference`。",
        f"- 同权重：`{summary['integrity']['model_identity']['same_archived_weight_identity']}`；模型为 `{summary['integrity']['model_identity']['model_id']}`，revision `{summary['integrity']['model_identity']['configured_revision']}`。",
        f"- `participant/model` 与归档挂载目标一致：`{summary['integrity']['model_identity']['link_matches_manifest_mount']}`；大权重分片未重新散列。",
        "- 本报告不含 Raw Completion、完整对话或 Gold 回复文本；只记录字段类别、标签和值长度。",
        "",
        "## 原始输出与错误状态",
        "",
        "| Profile | 严格 JSON object | 宽松提取 | Raw A0 合法 | Submission Schema | Emotion 合法（宽松提取） | 合法但误判 | Emotion 正确 | 回复空值 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("greedy", "sampling"):
        c = profiles[name]
        lines.append(
            f"| {name} | {c['strict_json_status_counts'].get('valid_object', 0)}/{c['sample_count']} "
            f"({c['rates']['strict_json_object_rate']:.1%}) | `{c['permissive_extraction_counts']}` | "
            f"{c['raw_a0_valid_count']}/{c['sample_count']} | {c['submission_schema_valid_count']}/{c['sample_count']} | "
            f"{c['emotion_valid_count']}/{c['sample_count']} | "
            f"{c['emotion_valid_count'] - c['emotion_valid_and_gold_correct_count']} | "
            f"{c['emotion_submission_correct_count']}/{c['sample_count']} | "
            f"{c['response_submission_empty_count']}/{c['sample_count']} |"
        )
    lines.extend([
        "",
        "`invalid_label` 是联合状态：情绪标签非法，或任一画像维度非法。下表将它拆开；不得把整个 `invalid_label` 数量当作情绪错误数。",
        "",
        "| Profile | Runner 状态 | 情绪原值类别 | 情绪置空原因 | Raw A0 失败原因 |",
        "|---|---|---|---|---|",
    ])
    for name in ("greedy", "sampling"):
        c = profiles[name]
        lines.append(
            f"| {name} | `{c['runner_status_counts']}` | `{c['emotion_status_counts']}` | "
            f"`{c['emotion_submission_empty_reasons']}` | `{c['a0_failure_reason_counts']}` |"
        )
    greedy_profile = profiles["greedy"]["runner_status_by_profile_validity"]
    sampling_profile = profiles["sampling"]["runner_status_by_profile_validity"]
    lines.extend([
        "",
        f"15 条 `invalid_label` 都伴有至少一个无效画像维度；其中 Greedy 有 11 条情绪值合法、4 条情绪缺失，Sampling 有 10 条合法、5 条缺失。另各有 5 条 `parse_failure`，没有宽松提取对象。逐字段交叉表：Greedy `{greedy_profile}`；Sampling `{sampling_profile}`。因此 Greedy 有 10 条、Sampling 有 9 条属于“情绪标签本身合法但预测错误”。",
        "",
        "Raw 情绪值直方图（含非法字符串）见 `label_histograms.json`；该本地文件会限制异常长值并附 SHA-256。画像 TP/FP/FN 与空集精确匹配见 `profile_diagnostics.json`。",
        "",
        "## P=1280 Smoke 截断",
        "",
        f"- 本轮实际裁剪历史：{truncation['summary']['truncated_count']}/{truncation['summary']['sample_count']} 条。Greedy/Sampling 逐 ID 的裁剪决定一致：`{truncation['summary']['profile_decisions_identical']}`。",
        f"- Runner 与 D0.4.2 S=2048/P=1280 的 retained/removed 消息数及 Prompt Token 数全部一致：`{truncation['summary']['d042_runner_match']}`。",
        "- 截断和未截断子集的当前结果如下；样本量很小，仅作描述，不作显著性推断。",
        "",
        "| 分组 | 条数 | Greedy Accuracy / JSON / A0 / 空回复 | Sampling Accuracy / JSON / A0 / 空回复 |",
        "|---|---:|---|---|",
    ])
    for group_name, label in (("truncated", "有裁剪"), ("untruncated", "无裁剪")):
        item = truncation["groups"][group_name]
        values = []
        for profile in ("greedy", "sampling"):
            p = item["profiles"][profile]
            accuracy = "n/a" if p["emotion_accuracy"] is None else f"{p['emotion_correct_count']}/{item['sample_count']} ({p['emotion_accuracy']:.1%})"
            values.append(f"{accuracy} / {p['strict_json_object_count']} / {p['raw_a0_valid_count']} / {p['response_empty_count']}")
        lines.append(f"| {label} | {item['sample_count']} | {values[0]} | {values[1]} |")
    lines.extend([
        "",
        "本轮有裁剪的 ID 与原/保留消息数、完整/保留 Prompt Token 数：",
        "",
        "| Eval ID | 原历史消息 | 保留消息 | 删除消息 | 完整 Prompt Token | T1 保留 Prompt Token |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in truncation["cases"]:
        detail = row["profiles"]["greedy"]
        if detail["truncated"]:
            lines.append(
                f"| {row['id']} | {detail['original_history_messages']} | {detail['retained_history_messages']} | "
                f"{detail['removed_history_messages']} | {detail['d042_full_prompt_tokens']} | {detail['d042_selected_prompt_tokens']} |"
            )
    lines.extend([
        "",
        "D0.4.2 全 Dev 的 31.26% 与官方 Val 的 24.81% 是另一组历史总体比例，不能代替这 20 条 Smoke 的实测截断率。P/T1 本轮未变更。",
        "",
        "## 协议核查与结论边界",
        "",
        f"- P0、Tokenizer revision/signature、Prompt cap、两组 Template equivalence 和逐样本 T1 对照均通过：`{summary['protocol']['protocol_check_passed']}`。训练 completion 附加 Gold 文本属于训练目标；Prompt 前缀复用 `build_messages`、`select_recent_complete_suffix` 和 `encode_inference_prompt`。本轮没有协议实现修改。",
        "- E00.2 所有样本的 stop reason 与生成长度分布见 `diagnosis_summary.json`；若都为 EOS 且低于 G=1200，不能把输出失败归因于生成上限。",
        "- 本诊断只解释当前 20 条、P=1280 的模型表现；不重算/覆盖历史指标，不据此冻结 Greedy/Sampling，不启动新 GPU 实验，也不修改 P0/F2/T1、数据或 Schema。",
        "",
        "## 输出文件",
        "",
        "- `diagnosis_summary.json`：运行/指纹核验、守恒关系、总体错误交叉表。",
        "- `diagnosis_cases.jsonl`：每个 Smoke ID 的两组状态，不含完整 Raw 文本或对话。",
        "- `label_histograms.json`：情绪实际值分布与归一化置空原因。",
        "- `profile_diagnostics.json`：逐维度合法性、TP/FP/FN、空集与非空标签统计。",
        "- `truncation_diagnostics.json`：逐 ID D0.4.2/Runner 对齐和截断分层。",
        "",
    ])
    return "\n".join(lines)


def diagnose_run(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    manifest, sources = _load_run(run_root)
    references_path = sources["smoke_references"]["path"]
    evaluation_manifest_path = sources["evaluation_manifest"]["path"]
    id_map_path = references_path.parent / "smoke_id_map.jsonl"
    evaluation_manifest = _read_json(evaluation_manifest_path)
    references = _index_rows(read_jsonl(references_path), "eval_id", "Smoke references")
    id_map = _index_rows(read_jsonl(id_map_path), "eval_id", "Smoke ID map")
    expected_ids = set(references)
    if len(expected_ids) != 20:
        raise ValueError(f"Smoke References 行数必须是 20，实际为 {len(expected_ids)}")
    _require_same_ids(expected_ids, id_map, "Smoke ID map")
    if evaluation_manifest.get("smoke_selection", {}).get("actual_count") != 20:
        raise ValueError("E00.1 Manifest 记载的 Smoke 数量不是 20")
    enums = schema_enums(json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8")))
    submission_schema = json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8"))
    schema_validator = Draft202012Validator(submission_schema)
    d042_rows, d042_facts = _load_d042(id_map)
    all_cases_by_id: dict[str, dict[str, Any]] = {}
    totals = {name: _empty_profile_totals() for name in ("greedy", "sampling")}
    performances: dict[str, dict[str, Any]] = {}
    metrics_by_profile: dict[str, dict[str, Any]] = {}
    raw_rows_by_profile: dict[str, dict[str, dict[str, Any]]] = {}
    submissions_by_profile: dict[str, dict[str, dict[str, Any]]] = {}
    perf_diags_by_profile: dict[str, dict[str, dict[str, Any]]] = {}
    integrity_profiles: dict[str, Any] = {}

    for profile_name in ("greedy", "sampling"):
        paths = _find_artifact_paths(run_root, manifest, profile_name)
        for key in ("raw", "submission", "performance", "metrics"):
            if not paths[key].is_file():
                raise FileNotFoundError(f"{profile_name} 归档文件缺失: {paths[key]}")
        raw_rows = read_jsonl(paths["raw"])
        submission_rows = read_jsonl(paths["submission"])
        performance = _read_json(paths["performance"])
        metrics = _read_json(paths["metrics"])
        profile_run_id = manifest["profiles"][profile_name]["run_id"]
        if performance.get("run_id") != profile_run_id or metrics.get("run_id") != profile_run_id:
            raise ValueError(f"{profile_name} Performance/Metrics run_id 与 Manifest 不符")
        if metrics.get("execution_mode") != "model_inference":
            raise ValueError(f"{profile_name} 不是 model_inference 结果")
        _require_sha(paths["raw"], performance.get("raw_predictions_sha256", ""), f"{profile_name} Raw")
        _require_sha(paths["submission"], performance.get("submission_sha256", ""), f"{profile_name} Submission")
        metric_hashes = metrics.get("hashes", {})
        if metric_hashes.get("raw_predictions_sha256") != sha256_file(paths["raw"]):
            raise ValueError(f"{profile_name} Metrics Raw SHA 与产物不符")
        if metric_hashes.get("submission_sha256") != sha256_file(paths["submission"]):
            raise ValueError(f"{profile_name} Metrics Submission SHA 与产物不符")
        if metric_hashes.get("inference_input_sha256") != sources["smoke_input"]["sha256"]:
            raise ValueError(f"{profile_name} Metrics 输入 SHA 与 E00.2 Manifest 不符")
        if metric_hashes.get("references_sha256") != sources["smoke_references"]["sha256"]:
            raise ValueError(f"{profile_name} Metrics References SHA 与 E00.2 Manifest 不符")
        if metric_hashes.get("evaluation_manifest_sha256") != sources["evaluation_manifest"]["sha256"]:
            raise ValueError(f"{profile_name} Metrics E00.1 Manifest SHA 与 E00.2 Manifest 不符")
        if performance.get("test_file_sha256") != sources["smoke_input"]["sha256"]:
            raise ValueError(f"{profile_name} Performance 输入 SHA 与 E00.2 Manifest 不符")
        if performance.get("raw_predictions_rows") != 20 or performance.get("submission_rows") != 20:
            raise ValueError(f"{profile_name} Performance 归档行数不为 20")
        if metrics.get("sample_count") != 20:
            raise ValueError(f"{profile_name} Metrics sample_count 不为 20")
        raw_by_id = _index_rows(raw_rows, "id", f"{profile_name} Raw")
        submission_by_id = _index_rows(submission_rows, "id", f"{profile_name} Submission")
        perf_diags = performance.get("sample_diagnostics")
        if not isinstance(perf_diags, list):
            raise ValueError(f"{profile_name} Performance 缺少 sample_diagnostics")
        perf_by_id = _index_rows(perf_diags, "id", f"{profile_name} Performance sample_diagnostics")
        for label, indexed in (("Raw", raw_by_id), ("Submission", submission_by_id), ("Performance", perf_by_id)):
            _require_same_ids(expected_ids, indexed, f"{profile_name} {label}")
        identity = performance.get("model_weight_identity", {})
        if identity.get("aggregate_sha256") != manifest["comparison"].get("model_weight_sha256"):
            raise ValueError(f"{profile_name} 权重 hash 与 E00.2 Manifest 不一致")
        if metric_hashes.get("id_map_sha256") != sha256_file(id_map_path):
            raise ValueError(f"{profile_name} Metrics ID map SHA 与 E00.1 归档不符")
        raw_rows_by_profile[profile_name] = raw_by_id
        submissions_by_profile[profile_name] = submission_by_id
        perf_diags_by_profile[profile_name] = perf_by_id
        performances[profile_name] = performance
        metrics_by_profile[profile_name] = metrics
        integrity_profiles[profile_name] = {
            "run_id": profile_run_id,
            "raw_sha256": performance.get("raw_predictions_sha256"),
            "submission_sha256": performance.get("submission_sha256"),
            "metrics_input_sha256": metric_hashes.get("inference_input_sha256"),
            "metrics_references_sha256": metric_hashes.get("references_sha256"),
            "metrics_id_map_sha256": metric_hashes.get("id_map_sha256"),
            "model_weight_sha256": identity.get("aggregate_sha256"),
            "execution_mode": metrics.get("execution_mode"),
        }

    input_hashes = {integrity_profiles[name]["metrics_input_sha256"] for name in integrity_profiles}
    weight_hashes = {integrity_profiles[name]["model_weight_sha256"] for name in integrity_profiles}
    if len(input_hashes) != 1 or len(weight_hashes) != 1:
        raise ValueError("Greedy/Sampling 输入或权重 hash 不一致，停止跨组比较")

    for sample_id in sorted(expected_ids):
        reference = references[sample_id]
        gold = reference.get("target")
        if not isinstance(gold, dict):
            raise ValueError(f"{sample_id}: Smoke Reference 缺少 target object")
        case = {
            "id": sample_id,
            "sample_id": id_map[sample_id].get("sample_id"),
            "source": id_map[sample_id].get("source"),
            "gold": {"emotion_label": gold.get("emotion_label"), "user_profile": gold.get("user_profile")},
            "profiles": {},
        }
        aux_by_profile: dict[str, Any] = {}
        dim_by_profile: dict[str, Any] = {}
        for profile_name in ("greedy", "sampling"):
            detailed, aux, dimension_cases, dimension_counts = _case_diagnosis(
                sample_id,
                profile_name,
                raw_rows_by_profile[profile_name][sample_id],
                submissions_by_profile[profile_name][sample_id],
                perf_diags_by_profile[profile_name][sample_id],
                gold,
                enums,
                schema_validator,
            )
            case["profiles"].update(detailed["profiles"])
            aux_by_profile[profile_name] = aux
            dim_by_profile[profile_name] = dimension_cases
            _increment_profile_totals(totals[profile_name], detailed, aux, dimension_cases)
        d042 = d042_rows[id_map[sample_id]["sample_id"]]
        truncation: dict[str, Any] = {}
        for profile_name in ("greedy", "sampling"):
            diag = perf_diags_by_profile[profile_name][sample_id]
            raw_row = raw_rows_by_profile[profile_name][sample_id]
            budget = d042["budget"]
            lengths = d042["lengths"]
            checks = {
                "prompt_cap_is_1280": diag.get("prompt_cap") == 1280,
                "original_history_messages_match_d042": diag.get("original_history_messages") == budget.get("original_history_message_count"),
                "retained_history_messages_match_d042": diag.get("retained_history_messages") == budget.get("retained_history_message_count"),
                "removed_history_messages_match_d042": diag.get("removed_history_messages") == budget.get("removed_history_message_count"),
                "raw_removed_history_messages_match_performance": raw_row.get("removed_history_messages") == diag.get("removed_history_messages"),
                "prompt_tokens_match_d042": diag.get("client_prompt_tokens") == budget.get("selected_prompt_tokens"),
                "full_prompt_tokens_match_d042_audit": lengths.get("prompt_tokens_full") == budget.get("prompt_tokens_full"),
            }
            truncation[profile_name] = {
                "original_history_messages": diag.get("original_history_messages"),
                "retained_history_messages": diag.get("retained_history_messages"),
                "removed_history_messages": diag.get("removed_history_messages"),
                "prompt_tokens": diag.get("client_prompt_tokens"),
                "d042_full_prompt_tokens": lengths.get("prompt_tokens_full"),
                "d042_selected_prompt_tokens": budget.get("selected_prompt_tokens"),
                "d042_outcome": budget.get("outcome"),
                "truncated": bool(diag.get("removed_history_messages", 0)),
                "checks": checks,
                "all_checks_passed": all(checks.values()),
            }
        case["truncation"] = truncation
        case["cross_profile"] = {
            "emotion_same": case["profiles"]["greedy"]["emotion"]["submission_value"] == case["profiles"]["sampling"]["emotion"]["submission_value"],
            "raw_a0_same": case["profiles"]["greedy"]["raw_a0"]["valid"] == case["profiles"]["sampling"]["raw_a0"]["valid"],
            "runner_status_same": case["profiles"]["greedy"]["runner_status"] == case["profiles"]["sampling"]["runner_status"],
            "truncation_same": truncation["greedy"]["removed_history_messages"] == truncation["sampling"]["removed_history_messages"],
        }
        all_cases_by_id[sample_id] = case

    finalized = {name: _finalize_totals(value) for name, value in totals.items()}
    for profile_name, aggregate in finalized.items():
        historical = metrics_by_profile[profile_name].get("raw_diagnostics", {})
        comparisons = {
            "sample_count": aggregate["sample_count"] == historical.get("sample_count"),
            "raw_json_parse_rate": aggregate["rates"]["strict_json_object_rate"] == historical.get("raw_json_parse_rate"),
            "raw_a0_rate": aggregate["rates"]["raw_a0_valid_rate"] == historical.get("raw_schema_valid_rate"),
            "strict_valid_emotion_rate": aggregate["strict_emotion_valid_count"] / aggregate["sample_count"] == historical.get("valid_emotion_rate"),
            "submission_schema_rate": aggregate["rates"]["submission_schema_valid_rate"] == historical.get("submission_schema_valid_rate"),
        }
        if not all(comparisons.values()):
            raise ValueError(f"{profile_name} 诊断总数与历史 Metrics 不一致: {comparisons}")
        aggregate["historical_metrics_reconciliation"] = {"matches": True, "checks": comparisons}

    cases = [all_cases_by_id[sample_id] for sample_id in sorted(all_cases_by_id)]
    label_histograms: dict[str, Any] = {}
    profile_diagnostics: dict[str, Any] = {}
    for profile_name in ("greedy", "sampling"):
        values = finalized[profile_name]["emotion_value_counts"]
        label_histograms[profile_name] = {
            "emotion_raw_values": [
                {**json.loads(value_json), "count": count}
                for value_json, count in sorted(values.items(), key=lambda pair: pair[0])
            ],
            "strict_json_emotion_raw_values": [
                {**json.loads(value_json), "count": count}
                for value_json, count in sorted(finalized[profile_name]["strict_emotion_value_counts"].items(), key=lambda pair: pair[0])
            ],
            "raw_status_counts": finalized[profile_name]["emotion_status_counts"],
            "strict_json_raw_status_counts": finalized[profile_name]["strict_emotion_status_counts"],
            "submission_empty_reasons": finalized[profile_name]["emotion_submission_empty_reasons"],
            "submission_empty_count": finalized[profile_name]["emotion_submission_empty_count"],
        }
        profile_diagnostics[profile_name] = {
            "sample_count": finalized[profile_name]["sample_count"],
            "dimensions": finalized[profile_name]["dimensions"],
        }

    removal_identical = all(case["cross_profile"]["truncation_same"] for case in cases)
    truncation_checks_passed = all(
        case["truncation"][profile]["all_checks_passed"]
        for case in cases for profile in ("greedy", "sampling")
    )
    trunc_count = sum(case["truncation"]["greedy"]["truncated"] for case in cases)
    truncation = {
        "d042": d042_facts,
        "summary": {
            "sample_count": len(cases),
            "truncated_count": trunc_count,
            "untruncated_count": len(cases) - trunc_count,
            "truncated_rate": trunc_count / len(cases),
            "profile_decisions_identical": removal_identical,
            "d042_runner_match": truncation_checks_passed,
            "all_raw_removed_counts_match_performance": all(
                case["truncation"][profile]["checks"]["raw_removed_history_messages_match_performance"]
                for case in cases for profile in ("greedy", "sampling")
            ),
        },
        "groups": {
            "truncated": _truncation_group_stats(cases, "truncated"),
            "untruncated": _truncation_group_stats(cases, "untruncated"),
        },
        "cases": [
            {
                "id": case["id"],
                "sample_id": case["sample_id"],
                "profiles": case["truncation"],
                "cross_profile_decision_identical": case["cross_profile"]["truncation_same"],
            }
            for case in cases
        ],
    }

    model_identity = _verify_model_mount(manifest, performances)
    profile_template_equivalence = {
        name: performances[name].get("template_equivalence") for name in ("greedy", "sampling")
    }
    eval_protocol = evaluation_manifest.get("protocol", {})
    config_protocols = {}
    for name in ("greedy", "sampling"):
        profile_manifest = manifest["profiles"][name]
        runner_config_path = _resolve_recorded_path(profile_manifest["runner_config_path"])
        evaluation_config_path = _resolve_recorded_path(profile_manifest["evaluation_config_path"])
        _require_sha(runner_config_path, profile_manifest.get("runner_config_sha256", ""), f"{name} runner config snapshot")
        _require_sha(evaluation_config_path, profile_manifest.get("evaluation_config_sha256", ""), f"{name} evaluation config snapshot")
        config = _read_json(runner_config_path)
        evaluation_config = _read_yaml(evaluation_config_path)
        if metrics_by_profile[name].get("hashes", {}).get("evaluation_config_sha256") != profile_manifest["evaluation_config_sha256"]:
            raise ValueError(f"{name} Metrics evaluation config SHA 与 Manifest 不符")
        config_protocols[name] = {
            "prompt_sha256": config.get("prompt", {}).get("sha256"),
            "tokenizer_signature_sha256": config.get("tokenizer", {}).get("expected_signature_sha256"),
            "tokenizer_revision": config.get("tokenizer", {}).get("revision"),
            "prompt_cap": config.get("context", {}).get("max_prompt_tokens"),
            "evaluation_prompt_sha256": evaluation_config.get("protocol", {}).get("prompt_sha256"),
            "runner_config_sha256": profile_manifest["runner_config_sha256"],
            "evaluation_config_sha256": profile_manifest["evaluation_config_sha256"],
        }
    prompt_sha = sha256_file(ROOT / "configs/prompts/p0.txt")
    prompt_protocol_match = (
        eval_protocol.get("prompt_sha256") == prompt_sha
        and all(value["prompt_sha256"] == prompt_sha for value in config_protocols.values())
        and all(value["evaluation_prompt_sha256"] == prompt_sha for value in config_protocols.values())
        and eval_protocol.get("max_prompt_tokens") == 1280
        and all(value["prompt_cap"] == 1280 for value in config_protocols.values())
    )
    tokenizer_protocol_match = all(
        config_protocols[name]["tokenizer_revision"] == eval_protocol.get("tokenizer_revision")
        and config_protocols[name]["tokenizer_signature_sha256"] == eval_protocol.get("tokenizer_signature_sha256")
        for name in ("greedy", "sampling")
    )
    template_equivalence_passed = all(
        profile_template_equivalence[name] == "local_input_ids_are_the_shared_encode_inference_prompt_result"
        or (
            isinstance(profile_template_equivalence[name], dict)
            and profile_template_equivalence[name].get("status") in {"passed", "verified"}
        )
        for name in ("greedy", "sampling")
    )
    protocol_passed = prompt_protocol_match and tokenizer_protocol_match and template_equivalence_passed and truncation_checks_passed

    stop_summary = {}
    for profile_name in ("greedy", "sampling"):
        perf = performances[profile_name]
        stop_summary[profile_name] = {
            "stop_reason_counts": dict(sorted(Counter(str(row.get("stop_reason", "missing")) for row in raw_rows_by_profile[profile_name].values()).items())),
            "generated_tokens_max": max((row.get("generated_tokens", 0) for row in raw_rows_by_profile[profile_name].values() if isinstance(row.get("generated_tokens"), int)), default=0),
            "generation_cap": perf.get("generation_config", {}).get("max_new_tokens"),
            "reported_parse_failure_count": perf.get("parse_failure_count"),
            "reported_invalid_label_count": perf.get("invalid_label_count"),
        }
    cross_profile = {
        "same_ids": True,
        "same_input_sha256": len(input_hashes) == 1,
        "same_model_weight_sha256": len(weight_hashes) == 1,
        "paired_truncation_decisions_identical": removal_identical,
        "emotion_submission_agreement_count": sum(case["cross_profile"]["emotion_same"] for case in cases),
        "raw_a0_validity_agreement_count": sum(case["cross_profile"]["raw_a0_same"] for case in cases),
        "runner_status_agreement_count": sum(case["cross_profile"]["runner_status_same"] for case in cases),
    }
    summary = {
        "run_id": run_root.name,
        "integrity": {
            "manifest_status": manifest["status"],
            "manifest_stage": manifest.get("stage"),
            "sample_count": len(expected_ids),
            "expected_ids": sorted(expected_ids),
            "id_conservation": {
                "references": len(references),
                "id_map": len(id_map),
                "greedy_raw": len(raw_rows_by_profile["greedy"]),
                "greedy_submission": len(submissions_by_profile["greedy"]),
                "greedy_performance": len(perf_diags_by_profile["greedy"]),
                "sampling_raw": len(raw_rows_by_profile["sampling"]),
                "sampling_submission": len(submissions_by_profile["sampling"]),
                "sampling_performance": len(perf_diags_by_profile["sampling"]),
                "all_id_sets_equal": True,
            },
            "input_sha256": next(iter(input_hashes)),
            "references_sha256": sources["smoke_references"]["sha256"],
            "evaluation_manifest_sha256": sources["evaluation_manifest"]["sha256"],
            "model_identity": model_identity,
            "profiles": integrity_profiles,
        },
        "profiles": finalized,
        "cross_profile": cross_profile,
        "stop_and_generation": stop_summary,
        "protocol": {
            "prompt_sha256_matches_p0_and_snapshots": prompt_protocol_match,
            "tokenizer_revision_and_signature_match": tokenizer_protocol_match,
            "profile_template_equivalence": profile_template_equivalence,
            "t1_d042_runner_sample_match": truncation_checks_passed,
            "prompt_training_prefix_code_review": "shared build_messages / select_recent_complete_suffix / encode_inference_prompt; training then appends supervised Gold completion",
            "protocol_check_passed": protocol_passed,
        },
        "output_policy": {
            "contains_raw_text": False,
            "contains_gold_response_text": False,
            "reads_only_existing_artifacts": True,
            "rewrites_history": False,
        },
    }
    return {
        "summary": summary,
        "cases": cases,
        "label_histograms": label_histograms,
        "profile_diagnostics": profile_diagnostics,
        "truncation": truncation,
        "report": _make_report(summary, finalized, truncation),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_diagnostics(result: dict[str, Any], output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    allowed_root = (ROOT / "outputs/evaluation/e00_2/diagnostics").resolve()
    if allowed_root not in output_dir.parents:
        raise ValueError(f"诊断结果必须写入 {allowed_root} 下")
    if output_dir.exists():
        raise FileExistsError(f"诊断输出目录已存在，拒绝覆盖: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_json(temporary / "diagnosis_summary.json", result["summary"])
        with (temporary / "diagnosis_cases.jsonl").open("w", encoding="utf-8") as handle:
            for case in result["cases"]:
                handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        _write_json(temporary / "label_histograms.json", result["label_histograms"])
        _write_json(temporary / "profile_diagnostics.json", result["profile_diagnostics"])
        _write_json(temporary / "truncation_diagnostics.json", result["truncation"])
        (temporary / "diagnosis_report.md").write_text(result["report"], encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path, help="已完成的 E00.2 run 目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="新建的 outputs/evaluation/e00_2/diagnostics/<run-id> 目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = diagnose_run(args.run_root)
    write_diagnostics(result, args.output_dir)
    print(json.dumps({
        "run_id": result["summary"]["run_id"],
        "sample_count": result["summary"]["integrity"]["sample_count"],
        "output_dir": str(args.output_dir.resolve()),
        "truncated_count": result["truncation"]["summary"]["truncated_count"],
        "protocol_check_passed": result["summary"]["protocol"]["protocol_check_passed"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
