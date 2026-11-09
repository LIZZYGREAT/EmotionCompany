from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import final_dual_docker_package as package


def test_rewritten_adapter_config_points_to_container_base_and_hash_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "adapter_config.json"
    source.write_text(
        json.dumps({"base_model_name_or_path": "/host/model", "r": 8}), encoding="utf-8"
    )

    payload = package._rewritten_er_config_bytes(source)
    rewritten = json.loads(payload)

    assert rewritten["base_model_name_or_path"] == "/root/participant/model"
    assert rewritten["r"] == 8
    assert hashlib.sha256(payload).hexdigest() == package.sha256_bytes(payload)
    assert package._contains_host_path(rewritten) is False


@pytest.mark.parametrize(
    "value",
    [
        {"path": "/home/developer/model"},
        {"nested": ["/mnt/x/models/base"]},
        {"path": "C:\\Users\\someone\\model"},
    ],
)
def test_host_specific_absolute_paths_are_rejected(value: dict) -> None:
    assert package._contains_host_path(value) is True


def test_prepare_context_refuses_paths_outside_ignored_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="below the ignored outputs"):
        package.prepare_context(output_dir=tmp_path / "context")


def test_frozen_input_sha256_reads_historical_gate_evidence() -> None:
    expected = "a" * 64
    gates = {"historical_asset_protocol_evidence": {"input_sha256": expected}}

    assert package._frozen_input_sha256(gates) == expected

    with pytest.raises(ValueError, match="historical asset/protocol gate evidence is missing"):
        package._frozen_input_sha256({"input_sha256": expected})
    with pytest.raises(ValueError, match="valid frozen input SHA-256"):
        package._frozen_input_sha256({"historical_asset_protocol_evidence": {"input_sha256": "invalid"}})


def test_dockerfile_pins_validated_build_images() -> None:
    dockerfile = (package.ROOT / package.DOCKERFILE).read_text(encoding="utf-8")

    assert f"FROM {package.CUDA_IMAGE}" in dockerfile
    assert f"FROM {package.MINIFORGE_IMAGE} AS miniforge" in dockerfile
    assert "COPY --from=miniforge /opt/conda /opt/conda" in dockerfile


def test_read_jsonl_requires_exact_official_input_keys_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(
        json.dumps({"id": "test_000001_t1", "history": []}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert package._read_jsonl(path, exact_input=True)[0]["id"] == "test_000001_t1"

    path.write_text(
        json.dumps({"id": "test_000001_t1", "history": [], "target": {}}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only id and history"):
        package._read_jsonl(path, exact_input=True)


def test_copy_adapter_rewrites_only_config_and_verifies_both_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/host/model", "r": 8}), encoding="utf-8"
    )
    weight = b"adapter weights"
    (source / "adapter_model.safetensors").write_bytes(weight)
    destination = tmp_path / "participant" / "adapters" / "emotion-response"

    facts = package._copy_adapter(source, destination, rewrite_base=True)

    assert facts["weight_sha256"] == hashlib.sha256(weight).hexdigest()
    assert facts["config_sha256"] == package.sha256_file(destination / "adapter_config.json")
    assert facts["source_config_sha256"] == package.sha256_file(source / "adapter_config.json")
    assert facts["base_path_rewritten"] is True
    assert facts["base_model_name_or_path"] == "/root/participant/model"


def test_reference_comparison_reports_online_response_variation() -> None:
    docker_rows = [
        {
            "id": "test_000001_t1",
            "response_text": "Docker response",
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "memory_refs": [],
        }
    ]
    host_rows = [{**docker_rows[0], "response_text": "Host response"}]
    report = {
        "format_success_count": {"profile": 1, "emotion_response": 1},
        "prompt_token_ids_exact_match_count": {"profile": 1, "emotion_response": 1},
        "route_failure_count": {"profile": 0, "emotion_response": 0},
        "generation_cap_hits": {"profile": 0, "emotion_response": 0},
    }
    host_report = {
        **report,
        "status": "passed",
        "complete": True,
    }

    result = package.compare_reference_outputs(docker_rows, host_rows, report, host_report)

    assert result["status"] == "stable_fields_and_metrics_match_response_text_varies"
    assert result["same_user_profile"] is True
    assert result["same_emotion_labels"] is True
    assert result["format_success_count_equal"] is True
    assert result["exact_submission_match"] is False
    assert result["response_text_exact_match_count"] == 0


def test_reference_comparison_rejects_stable_field_or_metric_mismatch() -> None:
    row = {
        "id": "test_000001_t1",
        "response_text": "Same response",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    }
    report = {
        "format_success_count": {"profile": 1, "emotion_response": 1},
        "prompt_token_ids_exact_match_count": {"profile": 1, "emotion_response": 1},
        "route_failure_count": {"profile": 0, "emotion_response": 0},
        "generation_cap_hits": {"profile": 0, "emotion_response": 0},
    }
    host_report = {**report, "status": "passed", "complete": True}

    different_label = {**row, "emotion_label": "sadness"}
    with pytest.raises(ValueError, match="stable submission fields differ"):
        package.compare_reference_outputs([row], [different_label], report, host_report)

    different_format_report = {**host_report, "format_success_count": {"profile": 1, "emotion_response": 0}}
    with pytest.raises(ValueError, match="route or parse metrics differ"):
        package.compare_reference_outputs([row], [row], report, different_format_report)


def test_reference_comparison_allows_observed_cap_and_retry_count_variation() -> None:
    row = {
        "id": "test_000001_t1",
        "response_text": "A stable response",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    }
    host_report = {
        "status": "passed",
        "complete": True,
        "format_success_count": {"profile": 20, "emotion_response": 20},
        "prompt_token_ids_exact_match_count": {"profile": 20, "emotion_response": 20},
        "prompt_token_ids_check_denominator": {"profile": 20, "emotion_response": 20},
        "route_failure_count": {"profile": 0, "emotion_response": 0},
        "generation_cap_hits": {"profile": 0, "emotion_response": 0},
        "initial_cap_hits": {"profile": 0, "emotion_response": 0},
        "retry_attempts": {"profile": 0, "emotion_response": 0},
        "retry_resolved": {"profile": 0, "emotion_response": 0},
        "unresolved_cap_hits": {"profile": 0, "emotion_response": 0},
        "degraded_responses": {"count": 0, "sample_ids": []},
    }
    docker_report = {
        **host_report,
        "prompt_token_ids_exact_match_count": {"profile": 20, "emotion_response": 21},
        "prompt_token_ids_check_denominator": {"profile": 20, "emotion_response": 21},
        "server": {"requested_alias_count": {"profile": 20, "emotion-response": 21}},
        "generation_cap_hits": {"profile": 0, "emotion_response": 1},
        "initial_cap_hits": {"profile": 0, "emotion_response": 1},
        "retry_attempts": {"profile": 0, "emotion_response": 1},
        "retry_resolved": {"profile": 0, "emotion_response": 1},
        "unresolved_cap_hits": {"profile": 0, "emotion_response": 0},
        "degraded_responses": {"count": 0, "sample_ids": []},
    }

    result = package.compare_reference_outputs([row] * 20, [row] * 20, docker_report, host_report)

    assert result["status"] == "stable_fields_match_runtime_metrics_vary"
    assert result["generation_cap_hits_equal"] is False
    assert result["both_prompt_checks_pass"] is True
    assert result["runtime_statistics"]["retry_attempts"]["docker"]["emotion_response"] == 1
    assert result["runtime_statistics"]["retry_attempts"]["host"]["emotion_response"] == 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def test_verify_output_accepts_one_retry_and_one_resolved_cap(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    context = tmp_path / "run" / "docker" / "context"
    result_dir = tmp_path / "run" / "docker" / "smoke"
    for path in (repo / "configs/deployment", repo / "participant/templates", context, result_dir):
        path.mkdir(parents=True, exist_ok=True)
    config_path = repo / package.DOCKER_CONFIG
    config_path.write_text("{}\n", encoding="utf-8")
    schema_path = repo / package.SCHEMA_PATH
    schema_path.write_text((package.ROOT / package.SCHEMA_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    model_lock_path = context / "model_lock.json"
    model_lock_path.write_text("{}\n", encoding="utf-8")
    input_rows = [
        {"id": f"test_{index:06d}_t1", "history": [{"role": "user", "content": f"message {index}"}]}
        for index in range(1, 21)
    ]
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, input_rows)
    submissions = [
        {
            "id": row["id"],
            "response_text": "A response",
            "emotion_label": "anxiety",
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "memory_refs": [],
        }
        for row in input_rows
    ]
    _write_jsonl(result_dir / "submission.jsonl", submissions)
    run_dir = result_dir / ".emotionalcompany_runs" / "run-1"
    run_dir.mkdir(parents=True)
    raw_path = run_dir / "raw_outputs.jsonl"
    _write_jsonl(raw_path, [{"id": row["id"]} for row in input_rows])
    report = {
        "version": "final-dual-official-run-v1.2",
        "status": "passed",
        "complete": True,
        "backend": "vllm",
        "samples": 20,
        "submission_rows": 20,
        "submission_schema_validated_count": 20,
        "submission_schema_validated": True,
        "model_weight_identity": {"observed_weight_hash": "base-sha"},
        "configuration": {
            "sha256": package.sha256_file(config_path),
            "model_lock_sha256": package.sha256_file(model_lock_path),
        },
        "initialization_seconds": 1,
        "server": {
            "alive_before_intentional_shutdown": True,
            "registered_model_ids": ["qwen-base", "profile", "emotion-response"],
            "runtime": {"version": "0.11.1"},
            "requested_alias_count": {"profile": 20, "emotion-response": 21},
        },
        "retry_attempts": {"profile": 0, "emotion_response": 1},
        "prompt_token_ids_exact_match_count": {"profile": 20, "emotion_response": 21},
        "prompt_token_ids_check_denominator": {"profile": 20, "emotion_response": 21},
        "format_success_count": {"profile": 20, "emotion_response": 20},
        "route_failure_count": {"profile": 0, "emotion_response": 0},
        "generation_cap_hits": {"profile": 0, "emotion_response": 1},
        "unresolved_cap_hits": {"profile": 0, "emotion_response": 0},
        "emotion_response_severe_anomaly_count": 0,
        "runtime_code_identity": {
            "version": "E03-v1.2-readme-v3-contract",
            "git_head": "fake-head",
            "file_sha256": {"participant/final_dual_inference.py": "code-sha"},
        },
        "adapter_identities": {
            "profile": {"weight_sha256": "profile-sha", "config_sha256": "profile-config"},
            "emotion_response": {"weight_sha256": "er-sha", "config_sha256": "er-config"},
        },
        "gpu_memory": {"devices": [{"peak_memory_used_mib": 1200}]},
        "outputs": {
            "submission.jsonl": {
                "path": "submission.jsonl",
                "sha256": package.sha256_file(result_dir / "submission.jsonl"),
                "rows": 20,
            },
            "raw_outputs.jsonl": {
                "path": ".emotionalcompany_runs/run-1/raw_outputs.jsonl",
                "sha256": package.sha256_file(raw_path),
                "rows": 20,
            },
            "run_progress.json": {"path": ".emotionalcompany_runs/run-1/run_progress.json"},
        },
    }
    report_path = result_dir / "performance_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "run_progress.json").write_text(json.dumps({
        "status": "complete", "completed_samples": 20,
        "test_file_sha256": package.sha256_file(test_file), "sample_count": 20,
    }), encoding="utf-8")
    manifest = {
        "runtime_behavior_version": "E03-v1.2-readme-v3-contract",
        "runtime_code_identity": report["runtime_code_identity"],
        "input": {"sha256": package.sha256_file(test_file)},
    }
    (context / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(package, "validate_static_config", lambda *args: {
        "base_weight_sha256": "base-sha",
        "adapter_weight_sha256": {"profile": "profile-sha", "emotion_response": "er-sha"},
        "adapter_config_sha256": {"profile": "profile-config", "emotion_response": "er-config"},
    })

    result = package.verify_output(test_file=test_file, result_dir=result_dir, repo_root=repo)

    assert result["status"] == "passed_offline_dual_adapter_docker_smoke"
    assert result["requested_alias_count"] == {"profile": 20, "emotion-response": 21}
    assert result["retry_attempts"]["emotion_response"] == 1
    assert result["generation_cap_hits"]["emotion_response"] == 1
