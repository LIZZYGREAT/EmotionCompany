from __future__ import annotations

import json

import pytest

from participant import run_inference
from scripts import e00_2_diagnose as diagnose
from src.evaluation import load_submission_schema, schema_enums


ROOT = diagnose.ROOT
ENUMS = schema_enums(load_submission_schema(ROOT))


def _completion(emotion: object = "joy", profile: object | None = None, **extra: object) -> dict:
    return {
        "emotion_label": emotion,
        "user_profile": profile if profile is not None else {dimension: [] for dimension in diagnose.PROFILE_DIMS},
        "response_text": "A private response that should not appear in diagnostics.",
        "memory_refs": [],
        **extra,
    }


@pytest.mark.parametrize(
    ("raw_text", "strict_status", "extraction_status"),
    [
        (json.dumps(_completion(), ensure_ascii=False), "valid_object", "object_extracted"),
        ("```json\n" + json.dumps(_completion()) + "\n```", "code_fence", "object_extracted"),
        ("Result: " + json.dumps(_completion()) + " trailing note", "leading_content_or_not_json", "object_extracted"),
        (json.dumps(_completion()) + " trailing note", "trailing_content", "object_extracted"),
        ('{"emotion_label":"joy","emotion_label":"sadness"}', "duplicate_key", "object_extracted"),
        ('{"emotion_label":NaN}', "non_standard_constant", "object_extracted"),
        ('{"emotion_label":', "malformed_or_truncated_json", "no_object_extracted"),
        ("<think>unclosed reasoning\n" + json.dumps(_completion()), "leading_content_or_not_json", "object_extracted"),
        ("<think>unclosed reasoning without an object", "leading_content_or_not_json", "no_object_extracted"),
    ],
)
def test_strict_json_and_permissive_extraction_are_reported_separately(raw_text, strict_status, extraction_status):
    assert diagnose._strict_parse(raw_text)[1] == strict_status
    assert diagnose._extract_object(raw_text)[1] == extraction_status


def test_strict_json_scalar_is_not_a_valid_completion_object():
    assert diagnose._strict_parse("null")[1] == "valid_json_non_object"


@pytest.mark.parametrize(
    ("raw", "present", "expected"),
    [
        ("joy", True, "valid"),
        ("happiness", True, "invalid_string"),
        ("快乐", True, "invalid_non_ascii_string"),
        ("", True, "empty_string"),
        (None, True, "null"),
        (17, True, "non_string"),
        (None, False, "missing"),
    ],
)
def test_emotion_value_categories_preserve_actual_invalid_sources(raw, present, expected):
    source = {"emotion_label": raw} if present else {}
    assert diagnose._emotion_status(source, ENUMS)[0] == expected


def test_invalid_emotion_normalizes_to_empty_without_using_gold_labels():
    prediction = run_inference.normalize_prediction("sample-1", _completion(emotion="快乐"), ENUMS)
    assert prediction["emotion_label"] == ""


def test_profile_dimension_categories_and_whole_dimension_normalization():
    dimension = "personality_traits"
    allowed = ENUMS[dimension]
    valid_label = allowed[0]
    assert diagnose._dimension_status({"user_profile": {dimension: []}}, dimension, ENUMS)["status"] == "valid_empty"
    assert diagnose._dimension_status({"user_profile": {dimension: [valid_label]}}, dimension, ENUMS)["status"] == "valid_nonempty"
    assert diagnose._dimension_status({"user_profile": {dimension: valid_label}}, dimension, ENUMS)["status"] == "non_array"
    invalid_enum = diagnose._dimension_status({"user_profile": {dimension: [valid_label, "not-an-enum"]}}, dimension, ENUMS)
    assert invalid_enum["status"] == "invalid_enum"
    assert invalid_enum["raw_legal_labels"] == [valid_label]
    assert diagnose._dimension_status({"user_profile": {dimension: [valid_label, valid_label]}}, dimension, ENUMS)["status"] == "duplicate"
    assert diagnose._dimension_status({"user_profile": {dimension: [None]}}, dimension, ENUMS)["status"] == "invalid_member_type"
    assert diagnose._dimension_status({}, dimension, ENUMS)["status"] == "profile_root_missing"
    assert diagnose._dimension_status({"user_profile": None}, dimension, ENUMS)["status"] == "profile_root_null"

    profile = {name: [] for name in diagnose.PROFILE_DIMS}
    profile[dimension] = [valid_label, "not-an-enum"]
    prediction = run_inference.normalize_prediction("sample-1", _completion(profile=profile), ENUMS)
    assert prediction["user_profile"][dimension] == []


def test_raw_a0_failure_does_not_misclassify_a_valid_emotion_as_invalid():
    raw = _completion(profile={"personality_traits": [], "interests": ["invalid"], "style": []})
    reasons = diagnose._raw_a0_reasons(raw, "valid_object", ENUMS)
    assert "invalid_emotion" not in reasons
    assert "invalid_profile:interests:invalid_enum" in reasons


def test_parse_failure_is_not_labeled_as_an_invalid_label_prediction():
    raw_text = "<think>unclosed output without a JSON object"
    prediction, parse_failed, invalid_label = run_inference.parse_prediction("sample-1", raw_text, ENUMS)
    assert parse_failed is True
    assert invalid_label is False
    assert prediction["emotion_label"] == ""


def test_id_alignment_rejects_duplicate_and_missing_ids():
    with pytest.raises(ValueError, match="重复 id"):
        diagnose._index_rows([{"id": "one"}, {"id": "one"}], "id", "fixture")
    indexed = diagnose._index_rows([{"id": "one"}], "id", "fixture")
    with pytest.raises(ValueError, match="ID 集合不一致"):
        diagnose._require_same_ids({"one", "two"}, indexed, "fixture")


def test_case_report_does_not_copy_raw_or_gold_response_text():
    raw_text = json.dumps(_completion(), ensure_ascii=False)
    raw_row = {"id": "sample-1", "raw_text": raw_text, "status": "success", "stop_reason": "eos", "generated_tokens": 12}
    submission = run_inference.normalize_prediction("sample-1", _completion(), ENUMS)
    perf = {"id": "sample-1", "status": "success", "original_history_messages": 1, "retained_history_messages": 1, "removed_history_messages": 0}
    schema = json.loads((ROOT / "participant/templates/submission_schema.json").read_text(encoding="utf-8"))
    detail, _, _, _ = diagnose._case_diagnosis(
        "sample-1", "greedy", raw_row, submission, perf, _completion(), ENUMS,
        diagnose.Draft202012Validator(schema),
    )
    serialized = json.dumps(detail, ensure_ascii=False)
    assert "A private response" not in serialized
    assert "response_characters" not in serialized
    assert detail["profiles"]["greedy"]["response"]["raw_characters"] > 0


def test_output_writer_refuses_existing_paths_and_non_diagnostics_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "ROOT", tmp_path)
    outside = tmp_path / "elsewhere"
    with pytest.raises(ValueError, match="必须写入"):
        diagnose.write_diagnostics({}, outside)

    existing = tmp_path / "outputs/evaluation/e00_2/diagnostics/existing"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        diagnose.write_diagnostics({}, existing)
    assert marker.read_text(encoding="utf-8") == "preserve"
