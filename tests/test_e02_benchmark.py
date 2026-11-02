from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import e02_benchmark as benchmark
from src.evaluation import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "outputs/e02"
    monkeypatch.setattr(benchmark, "E02_OUTPUT_ROOT", output_root.resolve())
    selection_root = output_root / "candidates"
    selection_root.mkdir(parents=True)
    config_path = ROOT / "configs/deployment/e02_vllm_benchmark.json"
    rows = [
        {
            "id": f"test_{900001 + index:06d}_t0",
            "history": [{"role": "user", "content": f"synthetic request {index}"}],
        }
        for index in range(100)
    ]
    candidate_path = selection_root / "candidate_input.jsonl"
    candidate_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    selection_manifest = {
        "schema_version": "e02_vllm_benchmark_selection_v2",
        "run_id": "candidate-run",
        "status": "passed_candidate_audit",
        "selection": {"selection_rule_sha256": "a" * 64},
        "source": {
            "raw_input_sha256": "b" * 64,
            "excluded_smoke_input_sha256": "c" * 64,
            "id_map_sha256": "d" * 64,
            "smoke_eval_id_count": 10,
            "mapped_sample_id_count": 10,
            "excluded_smoke_sample_id_overlap": ["source-smoke"],
        },
        "protocol": {
            "config_sha256": sha256_file(config_path),
            "selector_script_sha256": "e" * 64,
        },
        "candidate_input": {
            "path": str(candidate_path.resolve()),
            "rows": 100,
            "sha256": sha256_file(candidate_path),
            "smoke_overlap_count": 0,
        },
        "samples": [
            {
                "id": row["id"],
                "source_sample_id": f"source-{index}",
                "prompt_tokens": 400 + index,
                "stratum": "short",
            }
            for index, row in enumerate(rows)
        ],
    }
    (selection_root / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_root, selection_root, config_path, rows


def _fake_command_runner(*, failure_index: int | None = None, interrupted_after: int | None = None):
    def invoke(command, *, cwd, env, check):
        assert command[1].endswith("participant/start.sh")
        assert env["EVALUATION_BACKEND"] == "openai_compatible"
        assert env["VLLM_HOST"] == "127.0.0.1"
        assert env["VLLM_PORT"] == "8000"
        assert env.get("VLLM_GPU_MEMORY_UTILIZATION") is None
        input_path = Path(command[2])
        result_dir = Path(command[3])
        result_dir.mkdir(parents=True)
        rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
        count = interrupted_after if interrupted_after is not None else len(rows)
        raw_rows = []
        submission_rows = []
        valid_raw = {
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "response_text": "Synthetic response.",
            "memory_refs": [],
        }
        for index, row in enumerate(rows[:count]):
            failed = index == failure_index
            cap_hit = index == 0
            raw_rows.append({
                "id": row["id"],
                "raw_text": None if failed else json.dumps(valid_raw, separators=(",", ":")),
                "status": "generation_failure" if failed else "success",
                "prompt_tokens": 100,
                "generated_tokens": 0 if failed else (1200 if cap_hit else 20),
                "stop_reason": "not_generated_generation_failure" if failed else (
                    "server_finish_reason:length" if cap_hit else "server_finish_reason:stop"
                ),
                "latency_ms": float(100 + index),
                "error_type": "RuntimeError" if failed else None,
                "failure_category": "timeout" if failed else None,
            })
            submission_rows.append({
                "id": row["id"],
                "response_text": "" if failed else "Synthetic response.",
                "emotion_label": "" if failed else "anxiety",
                "user_profile": {"personality_traits": [], "interests": [], "style": []},
                "memory_refs": [],
            })
        raw_path = Path(env["EVALUATION_RAW_OUTPUT_PATH"])
        raw_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in raw_rows),
            encoding="utf-8",
        )
        submission_path = result_dir / "submission.jsonl"
        submission_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in submission_rows),
            encoding="utf-8",
        )
        snapshot_config = Path(env["EVALUATION_INFERENCE_CONFIG"])
        report = {
            "complete": interrupted_after is None,
            "run_id": env["EVALUATION_RUN_ID"],
            "test_file_sha256": sha256_file(input_path),
            "configuration": {"inference_config_sha256": sha256_file(snapshot_config)},
            "backend": "openai_compatible",
            "api_model": "e01-full",
            "generation_config": json.loads(snapshot_config.read_text(encoding="utf-8"))["generation"],
            "submission_schema_validated": True,
            "resumption": {"performance_is_single_uninterrupted_run": interrupted_after is None},
            "samples": count,
            "submission_rows": count,
            "required_rounds": 100,
            "rounds": count,
            "success_count": count - int(failure_index is not None),
            "failure_counts": {"generation_failure": 1} if failure_index is not None else {"success": count},
            "submission_sha256": sha256_file(submission_path),
            "raw_predictions_sha256": sha256_file(raw_path),
        }
        (result_dir / "performance_report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
        if interrupted_after is not None:
            (result_dir / "run_progress.json").write_text(
                json.dumps({"completed_samples": interrupted_after}) + "\n", encoding="utf-8"
            )
            raise KeyboardInterrupt
        (Path(env["VLLM_LOG_FILE"])).write_text(
            "INFO GPU KV cache size: 2,560 tokens\nWARNING request preemption recompute\n",
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0})()

    return invoke


def test_benchmark_config_is_independent_and_keeps_the_smoke_protocol_identity():
    config = benchmark.validate_benchmark_config(benchmark.DEFAULT_CONFIG)
    smoke = json.loads((ROOT / "configs/deployment/e02_vllm_smoke.json").read_text(encoding="utf-8"))

    assert config["experiment_id"] != smoke["experiment_id"]
    assert config["performance_rounds"] == 100
    assert config["generation"] == smoke["generation"]
    assert config["tokenizer"] == smoke["tokenizer"]
    assert config["adapter"] == smoke["adapter"]
    assert config["deployment"] == smoke["deployment"]


def test_gpu_memory_sampler_keeps_peak_across_repeated_observations():
    observations = iter([
        {
            "status": "observed",
            "devices": [{"name": "GPU", "memory_used_mib_at_observation": 4096, "memory_total_mib": 24576}],
        },
        {
            "status": "observed",
            "devices": [{"name": "GPU", "memory_used_mib_at_observation": 3072, "memory_total_mib": 24576}],
        },
    ])
    sampler = benchmark.GpuMemorySampler(provider=lambda: next(observations))

    sampler._record()
    sampler._record()
    result = sampler.stop()

    assert result["sample_count"] == 2
    assert result["query_error_count"] == 0
    assert result["peak_observed_by_device"] == [{
        "index": 0,
        "name": "GPU",
        "memory_total_mib": 24576,
        "peak_memory_used_mib_observed": 4096,
    }]


def test_benchmark_completes_100_and_reports_metrics_from_mock_runner(tmp_path, monkeypatch):
    output_root, selection_root, config_path, rows = _fixture(tmp_path, monkeypatch)
    run_root = output_root / "benchmark-complete"
    start_script = ROOT / "participant/start.sh"
    monkeypatch.setenv("EVALUATION_BACKEND", "transformers")
    monkeypatch.setenv("EVALUATION_MAX_PROMPT_TOKENS", "1")
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.6")

    result = benchmark.run_benchmark(
        selection_root=selection_root,
        config_path=config_path,
        run_root=run_root,
        start_script=start_script,
        command_runner=_fake_command_runner(),
        gpu_memory_provider=lambda: {"status": "observed", "devices": []},
    )

    summary = json.loads((run_root / "performance_summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["completed_samples"] == len(rows) == 100
    assert summary["runner"]["raw_strict_json_object_count"] == 100
    assert summary["runner"]["raw_a0_schema_valid_count"] == 100
    assert summary["runner"]["submission_schema_valid_count"] == 100
    assert summary["generation"]["token_cap_hit_count"] == 1
    assert summary["generation"]["length_stop_count"] == 1
    assert summary["generation"]["total_generated_tokens"] == 1200 + 99 * 20
    assert summary["generation"]["tokens_per_second"] > 0
    assert summary["generation"]["whole_process_tokens_per_second"] > 0
    assert summary["generation"]["whole_process_requests_per_second"] > 0
    assert summary["runtime_observations"]["kv_cache"]["gpu_kv_cache_token_capacity_observations"] == [2560]
    assert summary["runtime_observations"]["preemption_or_recompute_log_line_count"] == 1


def test_benchmark_records_a_completed_run_with_one_partial_request_failure(tmp_path, monkeypatch):
    output_root, selection_root, config_path, _rows = _fixture(tmp_path, monkeypatch)
    run_root = output_root / "benchmark-one-timeout"

    result = benchmark.run_benchmark(
        selection_root=selection_root,
        config_path=config_path,
        run_root=run_root,
        start_script=ROOT / "participant/start.sh",
        command_runner=_fake_command_runner(failure_index=17),
        gpu_memory_provider=lambda: {"status": "observed", "devices": []},
    )

    summary = json.loads((run_root / "performance_summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed_with_request_failures"
    assert result["completed_samples"] == 100
    assert summary["failures"]["generation_failure_count"] == 1
    assert summary["failures"]["timeout_count"] == 1
    assert summary["failures"]["non_success_status_count"] == 1
    assert summary["runner"]["raw_strict_json_object_count"] == 99
    assert summary["runner"]["submission_schema_valid_count"] == 100


def test_benchmark_preserves_interrupted_progress_without_marking_complete(tmp_path, monkeypatch):
    output_root, selection_root, config_path, _rows = _fixture(tmp_path, monkeypatch)
    run_root = output_root / "benchmark-interrupted"

    result = benchmark.run_benchmark(
        selection_root=selection_root,
        config_path=config_path,
        run_root=run_root,
        start_script=ROOT / "participant/start.sh",
        command_runner=_fake_command_runner(interrupted_after=7),
        gpu_memory_provider=lambda: {"status": "observed", "devices": []},
    )

    assert result["status"] == "interrupted"
    assert result["completed_samples"] == 7
    assert not (run_root / "performance_summary.json").exists()
    assert len((run_root / "raw_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 7


def test_benchmark_rejects_candidate_input_identity_change_before_run(tmp_path, monkeypatch):
    output_root, selection_root, config_path, _rows = _fixture(tmp_path, monkeypatch)
    candidate_path = selection_root / "candidate_input.jsonl"
    candidate_path.write_text(candidate_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    run_root = output_root / "benchmark-mutated-input"

    with pytest.raises(ValueError, match="SHA-256"):
        benchmark.run_benchmark(
            selection_root=selection_root,
            config_path=config_path,
            run_root=run_root,
            start_script=ROOT / "participant/start.sh",
            command_runner=_fake_command_runner(),
            gpu_memory_provider=lambda: {"status": "observed", "devices": []},
        )

    assert not run_root.exists()


def test_benchmark_refuses_to_overwrite_a_historical_run(tmp_path, monkeypatch):
    output_root, selection_root, config_path, _rows = _fixture(tmp_path, monkeypatch)
    run_root = output_root / "historical-run"
    run_root.mkdir()
    marker = run_root / "keep.txt"
    marker.write_text("immutable", encoding="utf-8")

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        benchmark.run_benchmark(
            selection_root=selection_root,
            config_path=config_path,
            run_root=run_root,
            start_script=ROOT / "participant/start.sh",
            command_runner=_fake_command_runner(),
            gpu_memory_provider=lambda: {"status": "observed", "devices": []},
        )

    assert marker.read_text(encoding="utf-8") == "immutable"
