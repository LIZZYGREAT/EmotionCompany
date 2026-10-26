from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mock_acceptance_cli_builds_and_scores_an_explicit_synthetic_fixture(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    scored_dir = tmp_path / "scored"
    create = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/e00_mock_acceptance.py"),
            "--config", str(ROOT / "configs/evaluation/e00.yaml"),
            "--output-dir", str(fixture_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create.returncode == 0, create.stdout + create.stderr
    assert "dataset_scope=synthetic_fixture backend=mock rows=4" in create.stdout

    score = subprocess.run(
        [
            sys.executable,
            "-m", "src.evaluation.score",
            "--config", "configs/evaluation/e00.yaml",
            "--references", str(fixture_dir / "references.jsonl"),
            "--submission", str(fixture_dir / "submission.jsonl"),
            "--raw", str(fixture_dir / "raw_predictions.jsonl"),
            "--output-dir", str(scored_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert score.returncode == 0, score.stdout + score.stderr
    metrics = json.loads((scored_dir / "metrics.json").read_text(encoding="utf-8"))
    report = (scored_dir / "report.md").read_text(encoding="utf-8")
    assert metrics["dataset_scope"] == "synthetic_fixture"
    assert metrics["model"]["model_weight_identity"]["fingerprint_status"] == "not_applicable_synthetic_fixture"
    assert metrics["emotion"]["accuracy"] == 0.75
    assert metrics["raw_diagnostics"]["raw_json_parse_rate"] == 0.25
    assert metrics["raw_diagnostics"]["submission_schema_valid_rate"] == 1.0
    assert metrics["response"]["bert_score"]["status"] == "not_run"
    assert metrics["response"]["bert_score"]["f1"] is None
    assert "0.750000" in report
    assert "not_run" in report

