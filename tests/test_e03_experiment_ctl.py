from __future__ import annotations

import pytest

from scripts.e03_experiment_ctl import preflight, status


def test_status_keeps_missing_run_as_unknown(tmp_path):
    result = status([tmp_path / "missing-run"])
    assert result["status"] == "read_only_status"
    assert result["runs"][0]["status"] == "missing_run_root"
    assert result["runs"][0]["training_seconds"] is None


def test_preflight_cannot_enable_training(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "version: E03-experiment-spec-v1\nexperiment_id: E03-test-v1\nexecution:\n  enabled: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation-only"):
        preflight(spec)
