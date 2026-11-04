from __future__ import annotations

import json

from scripts.e03_pipeline_audit import analyze_runner_report, analyze_training_run, build_audit


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_training_audit_reports_step_distribution_and_missing_phases_as_unknown(tmp_path):
    run = tmp_path / "Run-A"
    _json(run / "manifest.json", {
        "run_id": "Run-A", "experiment_id": "E03-test", "status": "interrupted",
        "training_seconds": 15.0, "optimizer_steps": 3,
    })
    _jsonl(run / "train_log.jsonl", [
        {"optimizer_step": 1, "optimizer_step_seconds": 1.0},
        {"optimizer_step": 2, "optimizer_step_seconds": 2.0},
        {"optimizer_step": 4, "optimizer_step_seconds": 3.0},
    ])
    _jsonl(run / "eval_log.jsonl", [{"optimizer_step": 2, "eval_samples": 10}])
    result = analyze_training_run(run)
    assert result["identity_status"] == "verified_run_id_path"
    assert result["manifest_status"] == "interrupted"
    assert result["optimizer_steps_logged"] == 3
    assert result["training_step_timing"]["p50"] == 2.0
    assert result["training_step_timing"]["gap_count"] == 1
    assert result["dev_evaluation"]["optimizer_steps"] == [2]
    assert result["isolated_phase_timings"]["gradient_audit"] == "unknown"
    assert result["resume_history"] == "unknown"


def test_missing_or_mixed_run_artifacts_are_not_combined(tmp_path):
    missing = analyze_training_run(tmp_path / "missing")
    assert missing["status"] == "missing_run_root"
    assert missing["training_seconds"] is None
    run = tmp_path / "Run-B"
    _json(run / "manifest.json", {"run_id": "Run-A", "status": "partial"})
    result = analyze_training_run(run)
    assert result["identity_status"] == "run_id_path_mismatch"
    assert result["training_log_status"] == "not_read_identity_mismatch"
    assert result["training_step_timing"]["status"] == "unknown"


def test_runner_latency_is_summarized_only_when_present(tmp_path):
    report = tmp_path / "performance_report.json"
    _json(report, {
        "run_id": "Eval-A", "samples": 3, "success_count": 3,
        "latencies_ms": [10, 20, 40], "model_load_seconds": 5.0,
        "cuda_memory_after_model_load": {"allocator_peak_allocated_bytes": 100},
    })
    result = analyze_runner_report(report)
    assert result["run_id"] == "Eval-A"
    assert result["latency_ms"]["p50"] == 20
    assert result["latency_ms"]["p90"] == 40
    assert result["cuda_peak_allocated_bytes_reported"] == 100
    missing = analyze_runner_report(tmp_path / "missing.json")
    assert missing["latency_ms"]["status"] == "unknown"


def test_audit_explicitly_states_that_it_did_not_train_or_load_models(tmp_path):
    summary = build_audit([], [])
    assert summary["status"] == "read_only_audit_completed"
    assert summary["conclusions"] == {
        "isolated_phase_timing_claims": "none",
        "model_loading_performed": False,
        "training_performed": False,
        "raw_predictions_read": False,
    }
