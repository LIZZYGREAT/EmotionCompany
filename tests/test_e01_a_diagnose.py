"""CPU-only regression coverage for E01-A closeout diagnostics."""

from __future__ import annotations

import json

import pytest

from scripts import e01_a_diagnose as diagnose


def _train_row(step: int, *, loss: float = 0.5) -> dict:
    return {
        "optimizer_step": step,
        "loss": loss,
        "learning_rate": 1e-4,
        "gradient_norm": 0.25,
        "finite_gradients": True,
    }


def test_validate_train_log_requires_unique_continuous_steps_and_finite_losses():
    rows = [_train_row(2), _train_row(1)]
    assert [row["optimizer_step"] for row in diagnose.validate_train_log(rows, expected_steps=2)] == [1, 2]

    with pytest.raises(ValueError, match="duplicate optimizer_step"):
        diagnose.validate_train_log([_train_row(1), _train_row(1)], expected_steps=2)
    with pytest.raises(ValueError, match="not continuous"):
        diagnose.validate_train_log([_train_row(1), _train_row(3)], expected_steps=2)
    with pytest.raises(ValueError, match="non-finite or missing loss"):
        diagnose.validate_train_log([_train_row(1, loss=float("nan"))], expected_steps=1)


def test_classify_raw_json_failure_uses_only_strict_parser_category():
    assert diagnose.classify_raw_json("Sure, here is the answer") == "not_json_object_prefix"
    assert diagnose.classify_raw_json('{"emotion_label":"joy"} extra') == "trailing_content_after_json"
    assert diagnose.classify_raw_json('["emotion"] extra') == "trailing_content_after_json"
    assert diagnose.classify_raw_json('{"response_text":"line\nbreak"}') == "unescaped_control_character"
    assert diagnose.classify_raw_json('{"emotion_label":"joy"}') == "strict_json_object"


def test_closeout_bundle_refuses_overwrite(tmp_path):
    output_dir = tmp_path / "closeout-v1"
    manifest = {"status": "passed"}
    metrics = {"schema_version": "test"}
    diagnose._write_bundle(output_dir, manifest, metrics, "# report\n", "# cases\n")
    saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics
    assert manifest["outputs"]["report.md"]["size_bytes"] > 0

    with pytest.raises(diagnose.OutputAlreadyExistsError, match="choose a new version"):
        diagnose._write_bundle(output_dir, manifest, metrics, "replacement\n", "replacement\n")
    assert (output_dir / "report.md").read_text(encoding="utf-8") == "# report\n"
