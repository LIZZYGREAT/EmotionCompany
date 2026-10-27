"""CPU-only regression coverage for E01 evaluation attempt isolation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import e01_evaluate as evaluation


def test_e01_attempt_ids_ignore_legacy_root_and_refuse_overwrite(tmp_path):
    run_root = tmp_path / "E01-A-test"
    legacy_root = run_root / "evaluation" / "dev_smoke"
    legacy_root.mkdir(parents=True)
    legacy_config = legacy_root / "evaluation_config.yaml"
    legacy_config.write_text("legacy snapshot\n", encoding="utf-8")
    (legacy_root / "runner").mkdir()

    first = evaluation._evaluation_attempt_root(run_root, "retry-001")
    second = evaluation._evaluation_attempt_root(run_root, "retry-002")
    assert first != second
    assert first == legacy_root / "retry-001"
    assert legacy_config.read_text(encoding="utf-8") == "legacy snapshot\n"

    first.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite Dev Smoke attempt"):
        evaluation._evaluation_attempt_root(run_root, "retry-001")
    with pytest.raises(ValueError, match="--attempt-id"):
        evaluation._evaluation_attempt_root(run_root, "../outside")


def test_e01_dev_smoke_records_running_before_subprocess_and_scores_attempt(tmp_path, monkeypatch):
    run_root = tmp_path / "E01-A-test"
    run_root.mkdir()
    legacy_root = run_root / "evaluation" / "dev_smoke"
    legacy_root.mkdir(parents=True)
    legacy_config = legacy_root / "evaluation_config.yaml"
    legacy_config.write_text("legacy snapshot\n", encoding="utf-8")

    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir()
    for name in ("smoke_input.jsonl", "smoke_references.jsonl", "manifest.json", "smoke_id_map.jsonl"):
        (smoke_root / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "SMOKE_ROOT", smoke_root)

    identity = {"weight_sha256": "a" * 64, "config_sha256": "b" * 64}
    run_manifest = {"experiment_id": "E01-A", "status": "passed", "base_weight_sha256": "c" * 64}
    monkeypatch.setattr(evaluation, "_load_run", lambda _root: (run_root, run_manifest, identity))
    monkeypatch.setattr(
        evaluation.run_inference,
        "adapter_weight_identity",
        lambda _path: (identity, {"base_model_name_or_path": "Qwen/Qwen3-1.7B"}),
    )

    def fake_runner(*_args, **_kwargs):
        attempt_root = run_root / "evaluation" / "dev_smoke" / "retry-003"
        record = json.loads((attempt_root / "run_manifest.json").read_text(encoding="utf-8"))
        assert record["status"] == "running"
        return SimpleNamespace(returncode=0, stdout="runner ok", stderr="")

    monkeypatch.setattr(evaluation.subprocess, "run", fake_runner)

    def fake_score_files(*, output_dir, **_kwargs):
        output_dir.mkdir(parents=True)
        (output_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
        return {"model_identity_verification": {"status": "passed"}}

    monkeypatch.setattr(evaluation, "score_files", fake_score_files)

    result = evaluation.run_dev_smoke(run_root, attempt_id="retry-003")

    attempt_root = legacy_root / "retry-003"
    saved_manifest = json.loads((attempt_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["status"] == saved_manifest["status"] == "scored"
    assert saved_manifest["return_code"] == 0
    assert saved_manifest["score_status"] == "passed"
    assert (attempt_root / "scored" / "metrics.json").is_file()
    assert legacy_config.read_text(encoding="utf-8") == "legacy snapshot\n"
