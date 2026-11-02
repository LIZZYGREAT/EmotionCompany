"""CPU-only checks for the read-only Full B artifact diagnostics."""
from __future__ import annotations

import json

import pytest

from scripts import e01_b_diagnose as diagnose


def test_profile_error_categories_partition_common_outcomes():
    cases = [
        ((1, 0, 0), "exact"),
        ((0, 0, 2), "all_gold_labels_missed"),
        ((1, 0, 2), "misses_only_partial"),
        ((0, 2, 0), "extra_labels_only"),
        ((2, 1, 1), "misses_and_extra_labels"),
    ]
    assert [diagnose._profile_error_kind(*values) for values, _ in cases] == [expected for _, expected in cases]


def test_distribution_uses_nearest_rank_percentiles():
    summary = diagnose._distribution([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert summary == {
        "count": 10,
        "mean": 5.5,
        "min": 1.0,
        "p50": 5.0,
        "p90": 9.0,
        "p95": 10.0,
        "max": 10.0,
    }


def test_safe_parse_returns_only_sanitized_failure_category():
    parsed, failure = diagnose._safe_parse('{"safe":"value"} trailing')
    assert parsed is None
    assert failure == "json_decode:Extra data"
    assert "trailing" not in failure
    parsed, failure = diagnose._safe_parse('{"k":1,"k":2}')
    assert parsed is None
    assert failure == "duplicate_json_key"


def test_unique_rows_reject_missing_and_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate id"):
        diagnose._unique([{"id": "a"}, {"id": "a"}], "id", "fixture")
    with pytest.raises(ValueError, match="empty/non-string id"):
        diagnose._unique([{"id": ""}], "id", "fixture")


def test_input_set_count_and_identity_mismatches_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="ID sets differ"):
        diagnose._require_same_ids({"a", "b"}, {"a", "c"}, "fixture")
    with pytest.raises(ValueError, match="count mismatch"):
        diagnose._require_count(2, 3, "fixture")
    with pytest.raises(ValueError, match="Base SHA mismatch"):
        diagnose._require_equal("base-sha-a", "base-sha-b", "Base SHA")


def test_final_reload_summary_uses_passed_standalone_check(tmp_path):
    check = {
        "status": "passed",
        "token_ids_exact_match": True,
        "base_weights_unchanged": True,
        "base_weight_sha256_after": "base-sha",
    }
    (tmp_path / "reload_check_standalone.json").write_text(json.dumps(check), encoding="utf-8")
    summary = diagnose._final_reload_check(tmp_path)
    assert summary["file"] == "reload_check_standalone.json"
    assert summary["status"] == "passed"
    assert summary["token_ids_exact_match"] is True
    assert summary["base_weights_unchanged"] is True
    assert summary["base_weight_sha256"] == "base-sha"


def test_changed_input_artifact_fails_sha_verification(tmp_path):
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text('{"id":"a"}\n', encoding="utf-8")
    expected_sha = diagnose.sha256_file(artifact)
    artifact.write_text('{"id":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        diagnose._check_sha(artifact, expected_sha, "fixture artifact")


def test_raw_a0_validation_reports_out_of_enum_without_raw_text():
    enums = {
        "emotion_label": ["anxiety"],
        "personality_traits": ["sensitive"],
        "interests": ["music"],
        "style": ["rational"],
    }
    raw = {
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": ["sensitive"], "interests": ["not-a-label"], "style": ["rational"]},
        "response_text": "synthetic response",
        "memory_refs": [],
    }
    assert diagnose._raw_a0_reasons(raw, enums) == ["profile_label_outside_enum"]


def test_output_writer_is_confined_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "ROOT", tmp_path)
    output = tmp_path / "outputs/evaluation/e01_b/diagnostics/test-run"
    result = diagnose.write_outputs(output, {"aggregate": 3}, {"input_sha256": "abc"})
    assert result["status"] == "completed"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_kind"] == "read_only_existing_artifacts"
    assert manifest["outputs"]["summary.json"]["sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnose.write_outputs(output, {"aggregate": 4}, {"input_sha256": "changed"})
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == {"aggregate": 3}


def test_output_writer_rejects_paths_outside_ignored_diagnostics_root(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="must remain under"):
        diagnose.write_outputs(tmp_path / "reports/diagnostics", {}, {})
    assert not (tmp_path / "reports").exists()
