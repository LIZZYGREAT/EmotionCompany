from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

import participant.final_dual_inference as final_dual
from participant.final_dual_inference import (
    _recover_emotion_response,
    _resume_identity,
    read_test_samples,
)


SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "participant/templates/submission_schema.json").read_text(encoding="utf-8")
)
PROFILE_TEXT = "Personality=introverted\nInterests=music\nStyle=colloquial"


def _er_result(finish_reason: str, text: str = "emotion=anxiety;response=A measured reply.", *, tokens: int = 12) -> dict:
    return {
        "text": text,
        "response_id": f"request-{finish_reason}-{tokens}",
        "finish_reason": finish_reason,
        "usage_completion_tokens": tokens,
        "requested_max_tokens": 1200,
        "generation_seconds": 0.25,
        "prompt_token_ids_exact_match": True,
        "prompt_token_count_expected": 8,
        "prompt_token_count_observed": 8,
        "prompt_token_ids_expected_sha256": "expected-prompt-hash",
        "prompt_token_ids_observed_sha256": "expected-prompt-hash",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    content = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(content)
    return content


def _samples() -> list[dict]:
    return [
        {"id": "test_000001_t1", "history": [{"role": "user", "content": "one"}]},
        {"id": "test_000002_t2", "history": [{"role": "user", "content": "two"}]},
    ]


def test_read_test_samples_accepts_only_official_target_free_input(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(_samples()[0]) + "\n", encoding="utf-8")
    assert read_test_samples(path) == _samples()[:1]

    path.write_text(json.dumps({**_samples()[0], "target": {}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only id and history"):
        read_test_samples(path)


@pytest.mark.parametrize("sample_id", ["val:1", "test_1_t1", "test_000001_t1"])
def test_read_test_samples_rejects_bad_or_duplicate_ids(tmp_path: Path, sample_id: str) -> None:
    path = tmp_path / "input.jsonl"
    row = {"id": sample_id, "history": [{"role": "user", "content": "one"}]}
    if sample_id == "test_000001_t1":
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid or duplicate"):
            read_test_samples(path)
    else:
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid or duplicate"):
            read_test_samples(path)


def test_resume_trims_one_file_ahead_to_last_fsynced_progress(tmp_path: Path, monkeypatch) -> None:
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, _samples())
    config_path = tmp_path / "config.json"
    config_path.write_text('{"backend":"vllm"}\n', encoding="utf-8")
    lock_path = tmp_path / "model_lock.json"
    lock_path.write_text('{"version":"lock"}\n', encoding="utf-8")
    result_dir = tmp_path / "result"
    samples = _samples()

    completed, _, _, progress = _resume_identity(
        result_dir=result_dir,
        test_file=test_file,
        config_path=config_path,
        lock_path=lock_path,
        samples=samples,
        backend="vllm",
        resume_enabled=False,
    )
    assert completed == 0
    assert progress["status"] == "running"

    committed_submission = {"id": samples[0]["id"], "value": "first"}
    committed_raw = {"id": samples[0]["id"], "sample_end_to_end_ms": 12.5}
    submission_prefix = _write_jsonl(result_dir / "submission.jsonl", [committed_submission])
    raw_prefix = _write_jsonl(result_dir / "raw_outputs.jsonl", [committed_raw])
    progress.update(
        completed_samples=1,
        submission_prefix_sha256=hashlib.sha256(submission_prefix).hexdigest(),
        raw_prefix_sha256=hashlib.sha256(raw_prefix).hexdigest(),
    )
    (result_dir / "run_progress.json").write_text(json.dumps(progress), encoding="utf-8")
    # Simulate interruption after the submission append but before the raw append/progress update.
    with (result_dir / "submission.jsonl").open("ab") as handle:
        handle.write(_write_jsonl(tmp_path / "extra.jsonl", [{"id": samples[1]["id"]}]))

    monkeypatch.delenv("EVALUATION_RUN_ID", raising=False)
    completed, submissions, raw_rows, resumed = _resume_identity(
        result_dir=result_dir,
        test_file=test_file,
        config_path=config_path,
        lock_path=lock_path,
        samples=samples,
        backend="vllm",
        resume_enabled=True,
    )

    assert completed == 1
    assert [row["id"] for row in submissions] == [samples[0]["id"]]
    assert [row["id"] for row in raw_rows] == [samples[0]["id"]]
    assert resumed["attempt"] == 2
    assert len((result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert [row["id"] for row in map(json.loads, (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines())] == [samples[0]["id"]]


def test_resume_rejects_changed_input_identity(tmp_path: Path) -> None:
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, _samples())
    config_path = tmp_path / "config.json"
    config_path.write_text('{"backend":"vllm"}\n', encoding="utf-8")
    lock_path = tmp_path / "model_lock.json"
    lock_path.write_text('{"version":"lock"}\n', encoding="utf-8")
    result_dir = tmp_path / "result"
    _resume_identity(
        result_dir=result_dir,
        test_file=test_file,
        config_path=config_path,
        lock_path=lock_path,
        samples=_samples(),
        backend="vllm",
        resume_enabled=False,
    )
    test_file.write_text(test_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input.*changed"):
        _resume_identity(
            result_dir=result_dir,
            test_file=test_file,
            config_path=config_path,
            lock_path=lock_path,
            samples=_samples(),
            backend="vllm",
            resume_enabled=True,
        )


def test_runtime_git_identity_uses_container_manifest_without_git_metadata(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "package_manifest.json").write_text(
        json.dumps({"source_commit": "docker-source-commit"}), encoding="utf-8"
    )
    monkeypatch.setattr(final_dual, "ROOT", tmp_path)
    monkeypatch.delenv("EVALUATION_GIT_HEAD", raising=False)

    assert final_dual._execution_git_head() == "docker-source-commit"


def test_runtime_git_identity_falls_back_when_git_executable_is_missing(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "package_manifest.json").write_text(
        json.dumps({"source_commit": "docker-source-commit"}), encoding="utf-8"
    )
    monkeypatch.setattr(final_dual, "ROOT", tmp_path)
    monkeypatch.delenv("EVALUATION_GIT_HEAD", raising=False)

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(final_dual.subprocess, "run", missing_git)
    assert final_dual._execution_git_head() == "docker-source-commit"


def test_v1_1_runtime_code_identity_is_locked_and_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runtime.py"
    source.write_text("stable runtime\n", encoding="utf-8")
    monkeypatch.setattr(final_dual, "RUNTIME_CODE_FILES", (Path("runtime.py"),))
    monkeypatch.setattr(final_dual, "_execution_git_head", lambda: "locked-head")
    config = {
        "runtime_behavior": {
            "version": "E03-v1.2-readme-v2-contract",
            "max_emotion_response_retries": 1,
            "retry_only_on_finish_reason": "length",
        }
    }
    hashes = final_dual.runtime_code_hashes(tmp_path)
    lock = {"runtime_code_identity": {
        "version": config["runtime_behavior"]["version"],
        "git_head": "locked-head",
        "file_sha256": hashes,
    }}

    assert final_dual._verify_runtime_code_identity(config, lock, root=tmp_path)["file_sha256"] == hashes
    source.write_text("changed runtime\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="code hashes differ"):
        final_dual._verify_runtime_code_identity(config, lock, root=tmp_path)


def test_stop_at_1200_tokens_is_not_a_confirmed_cap_hit() -> None:
    result = _recover_emotion_response(
        _er_result("stop", tokens=1200),
        None,
        sample_id="test_000001_t1",
        retry_attempted=False,
        schema=SCHEMA,
        profile_raw=PROFILE_TEXT,
    )

    assert result["outcome"] == "completed_stop"
    assert result["selected"]["finish_reason"] == "stop"
    assert result["counters"]["initial_cap_hit"] == 0
    assert result["attempts"][0]["completion_tokens_at_budget"] is True


def test_length_then_stop_selects_retry_and_records_both_attempts() -> None:
    result = _recover_emotion_response(
        _er_result("length", tokens=1200),
        _er_result("stop", tokens=33),
        sample_id="test_000001_t1",
        retry_attempted=True,
        schema=SCHEMA,
        profile_raw=PROFILE_TEXT,
    )

    assert result["outcome"] == "completed_stop"
    assert result["selected_attempt_index"] == 2
    assert result["counters"] == {
        "initial_cap_hit": 1,
        "retry_attempts": 1,
        "retry_resolved": 1,
        "unresolved_cap_hit": 0,
        "degraded_response": False,
    }
    assert [item["request_id"] for item in result["attempts"]] == [
        "request-length-1200", "request-stop-33"
    ]
    assert all(item["raw_text_sha256"] for item in result["attempts"])
    assert all(item["prompt_token_ids_exact_match"] is True for item in result["attempts"])


def test_two_length_attempts_keep_valid_output_as_unresolved_warning() -> None:
    result = _recover_emotion_response(
        _er_result("length", "emotion=anxiety;response=First valid answer.", tokens=1200),
        _er_result("length", "emotion=joy;response=Second valid answer.", tokens=1200),
        sample_id="test_000001_t1",
        retry_attempted=True,
        schema=SCHEMA,
        profile_raw=PROFILE_TEXT,
    )

    assert result["outcome"] == "length_limited_unresolved"
    assert result["selected_attempt_index"] == 1
    assert result["selected"]["text"].endswith("First valid answer.")
    assert result["counters"]["unresolved_cap_hit"] == 1
    assert result["counters"]["degraded_response"] is True


def test_invalid_retry_preserves_first_legal_cap_limited_output() -> None:
    result = _recover_emotion_response(
        _er_result("length", "emotion=anxiety;response=First valid answer.", tokens=1200),
        _er_result("stop", "not a structured response", tokens=20),
        sample_id="test_000001_t1",
        retry_attempted=True,
        schema=SCHEMA,
        profile_raw=PROFILE_TEXT,
    )

    assert result["outcome"] == "length_limited_unresolved"
    assert result["selected_attempt_index"] == 1
    assert result["selected"]["text"].endswith("First valid answer.")
    assert result["counters"]["retry_resolved"] == 0


def test_two_invalid_attempts_use_existing_field_degradation_without_inventing_values() -> None:
    result = _recover_emotion_response(
        _er_result("length", "not structured", tokens=1200),
        _er_result("stop", "also not structured", tokens=20),
        sample_id="test_000001_t1",
        retry_attempted=True,
        schema=SCHEMA,
        profile_raw=None,
    )
    submission, diagnostics = final_dual.assemble_submission(
        "test_000001_t1",
        None,
        None if result["selected"] is None else result["selected"]["text"],
        SCHEMA,
    )

    assert result["outcome"] == "invalid_output"
    assert result["selected"] is None
    assert result["counters"]["unresolved_cap_hit"] == 1
    assert submission["emotion_label"] == ""
    assert submission["response_text"] == ""
    assert submission["user_profile"] == {"personality_traits": [], "interests": [], "style": []}
    assert diagnostics["emotion_parse_error"] == "generation_failed"


def test_protocol_mismatch_is_not_retried_and_is_a_hard_identity_failure() -> None:
    result = _recover_emotion_response(
        {**_er_result("length", tokens=1200), "prompt_token_ids_exact_match": False},
        None,
        sample_id="test_000001_t1",
        retry_attempted=False,
        schema=SCHEMA,
        profile_raw=PROFILE_TEXT,
        protocol_mismatch=True,
    )

    assert result["outcome"] == "asset_or_protocol_mismatch"
    assert result["selected"] is None
    assert result["counters"]["retry_attempts"] == 0


def test_severe_repeated_response_fragment_is_not_selected() -> None:
    repeated = "emotion=anxiety;response=" + ("This exact sentence loops forever. " * 20)
    result = _recover_emotion_response(
        _er_result("length", repeated, tokens=1200),
        None,
        sample_id="test_000001_t1",
        retry_attempted=False,
        schema=SCHEMA,
        profile_raw=PROFILE_TEXT,
    )

    assert result["outcome"] == "invalid_output"
    assert result["selected"] is None
    assert result["attempts"][0]["severe_text_anomalies"]


def _run_mocked_official_run(
    tmp_path: Path,
    monkeypatch,
    *,
    scenario: str,
    sample_count: int,
    preexisting_results: bool = False,
) -> tuple[int, Path, list[tuple[str, list[dict], list[int], int]], Path]:
    result_dir = tmp_path / "result"
    if preexisting_results:
        result_dir.mkdir()
        (result_dir / "submission.jsonl").write_bytes(b"old submission\n")
        (result_dir / "performance_report.json").write_bytes(b"old report\n")
        (result_dir / "keep.txt").write_bytes(b"keep these exact bytes\x00\xff")
    prompt_paths = {}
    prompt_configs = {}
    for name in ("selection", "profile", "profile_history_budget", "emotion_response"):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"frozen {name}\n", encoding="utf-8")
        prompt_paths[name] = path
        prompt_configs[name] = {"path": str(path), "sha256": f"{name}-sha"}
    adapter_paths = {}
    for name in ("profile", "emotion_response"):
        adapter = tmp_path / name
        adapter.mkdir()
        adapter_paths[name] = adapter
    lock_path = tmp_path / "model_lock.json"
    lock_path.write_text(json.dumps({
        "deployment_adapters": {
            "profile": {"path": str(adapter_paths["profile"])},
            "emotion_response": {"path": str(adapter_paths["emotion_response"])},
        }
    }), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "mode": "dual_adapter",
        "backend": "vllm",
        "model_id": "fake-model",
        "model_revision": "fake-revision",
        "model_path": str(tmp_path / "model"),
        "model_lock_path": str(lock_path),
        "prompts": prompt_configs,
        "context": {"max_prompt_tokens": 1280},
        "generation": {
            "profile": {"max_new_tokens": 256},
            "emotion_response": {"max_new_tokens": 1200},
            "eos_token_id": 151645,
            "pad_token_id": 151643,
        },
        "deployment": {"vllm": {"request_timeout_seconds": 900}},
    }), encoding="utf-8")
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, [
        {"id": f"test_{index:06d}_t1", "history": [{"role": "user", "content": f"user {index}"}]}
        for index in range(1, sample_count + 1)
    ])

    asset_identity = {
        "tokenizer": {"signature_sha256": "tokenizer-sha"},
        "base": {"observed_weight_hash": "base-sha"},
        "adapters": {
            "profile": {"weight_sha256": "profile-sha", "config_sha256": "profile-config-sha"},
            "emotion_response": {"weight_sha256": "er-sha", "config_sha256": "er-config-sha"},
        },
        "prompts": {name: {"path": str(path), "sha256": f"{name}-sha"} for name, path in prompt_paths.items()},
        "model_lock": {"path": str(lock_path), "sha256": "lock-sha"},
    }
    monkeypatch.setattr(final_dual, "_verify_assets", lambda *args: (object(), asset_identity))
    monkeypatch.setattr(final_dual, "build_profile_prompt", lambda *args, **kwargs: {
        "messages": [{"role": "user", "content": "same-profile-prompt"}], "prompt_ids": [11, 12],
        "prompt_tokens": 2, "prompt_token_ids_sha256": "profile-prompt-sha", "selected_message_ids": [0],
        "selected_history_sha256": "profile-history-sha",
    })
    monkeypatch.setattr(final_dual, "build_emotion_response_prompt", lambda *args, **kwargs: {
        "messages": [{"role": "user", "content": "same-er-prompt"}], "prompt_ids": [21, 22],
        "prompt_tokens": 2, "prompt_token_ids_sha256": "er-prompt-sha", "selected_message_ids": [0],
        "selected_history_sha256": "er-history-sha",
    })

    calls: list[tuple[str, list[dict], list[int], int]] = []
    er_calls = 0

    class FakeMonitor:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return None

        def stop(self):
            return {"devices": [{"peak_memory_used_mib": 512}]}

    class FakeService:
        def __init__(self, **kwargs):
            self.alive = True

        def start(self):
            return {"startup_seconds": 0.01, "version": "0.11.1"}

        def require_adapters(self):
            return ["qwen-base", "profile", "emotion-response"]

        def is_alive(self):
            return self.alive

        def chat_completion(self, *, model_alias, messages, expected_prompt_token_ids, max_tokens):
            nonlocal er_calls
            calls.append((model_alias, messages, list(expected_prompt_token_ids), max_tokens))
            if model_alias == "profile":
                if scenario == "profile_error":
                    raise RuntimeError("A8 route request failed")
                if scenario == "three_profile_errors" and sum(call[0] == "profile" for call in calls) <= 3:
                    raise RuntimeError("intermittent A8 route request failed")
                if scenario == "service_exit":
                    self.alive = False
                exact = scenario != "prompt_mismatch"
                return {
                    "text": PROFILE_TEXT,
                    "response_id": "profile-1",
                    "finish_reason": "stop",
                    "usage_completion_tokens": 8,
                    "generation_seconds": 0.001,
                    "prompt_token_ids_exact_match": exact,
                    "prompt_token_count_expected": 2,
                    "prompt_token_count_observed": 2,
                    "prompt_token_ids_expected_sha256": "profile-prompt-sha",
                    "prompt_token_ids_observed_sha256": "profile-prompt-sha" if exact else "wrong-sha",
                }
            er_calls += 1
            if scenario == "three_format_errors" and er_calls <= 3:
                finish = "stop"
                text = "unrecoverable model text"
                tokens = 8
            elif scenario in {"resolved", "unresolved"} and er_calls == 1:
                finish = "length"
                text = "emotion=anxiety;response=First valid answer."
                tokens = 1200
            elif scenario == "unresolved" and er_calls == 2:
                finish = "length"
                text = "emotion=joy;response=Second valid answer."
                tokens = 1200
            elif scenario == "resolved" and er_calls == 2:
                time.sleep(0.025)
                finish = "stop"
                text = "emotion=anxiety;response=Recovered valid answer."
                tokens = 30
            else:
                finish = "stop"
                text = "emotion=anxiety;response=Normal valid answer."
                tokens = 30
            return {
                "text": text,
                "response_id": f"er-{er_calls}",
                "finish_reason": finish,
                "usage_completion_tokens": tokens,
                "generation_seconds": 0.02 if er_calls == 2 and scenario == "resolved" else 0.001,
                "prompt_token_ids_exact_match": True,
                "prompt_token_count_expected": 2,
                "prompt_token_count_observed": 2,
                "prompt_token_ids_expected_sha256": "er-prompt-sha",
                "prompt_token_ids_observed_sha256": "er-prompt-sha",
            }

        def close(self):
            return -15

    monkeypatch.setattr(final_dual, "GpuMemoryMonitor", FakeMonitor)
    monkeypatch.setattr(final_dual, "DualLoRAVLLMService", FakeService)
    monkeypatch.setenv("EVALUATION_INFERENCE_CONFIG", str(config_path))
    monkeypatch.setenv("EVALUATION_BACKEND", "vllm")
    monkeypatch.setenv("EVALUATION_RUN_ID", f"test-{scenario}-{sample_count}")
    monkeypatch.setenv("EVALUATION_RESUME", "false")
    monkeypatch.setattr(sys, "argv", ["final_dual_inference.py", str(test_file), str(result_dir)])

    return final_dual.main(), result_dir, calls, test_file


def test_mocked_twenty_row_run_counts_one_retry_and_keeps_one_row_per_input(tmp_path: Path, monkeypatch) -> None:
    exit_code, result_dir, calls, _ = _run_mocked_official_run(
        tmp_path, monkeypatch, scenario="resolved", sample_count=20
    )
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    submission = [json.loads(line) for line in (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    raw_path = result_dir / report["outputs"]["raw_outputs.jsonl"]["path"]
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]

    assert exit_code == 0
    assert report["complete"] is True and report["status"] == "passed"
    assert len(submission) == len(raw) == 20
    assert report["server"]["requested_alias_count"] == {"profile": 20, "emotion-response": 21}
    assert report["prompt_token_ids_exact_match_count"] == {"profile": 20, "emotion_response": 21}
    assert report["prompt_token_ids_check_denominator"] == {"profile": 20, "emotion_response": 21}
    assert report["initial_cap_hits"]["emotion_response"] == 1
    assert report["retry_attempts"]["emotion_response"] == 1
    assert report["retry_resolved"]["emotion_response"] == 1
    assert report["unresolved_cap_hits"]["emotion_response"] == 0
    assert report["performance"]["per_route_generation_seconds"]["emotion_response"]["count"] == 21
    assert raw[0]["sample_end_to_end_ms"] >= 15
    assert [call[0] for call in calls].count("profile") == 20
    er_calls = [call for call in calls if call[0] == "emotion-response"]
    assert len(er_calls) == 21
    assert er_calls[0][1:3] == er_calls[1][1:3]
    assert er_calls[0][3] == er_calls[1][3] == 1200
    assert raw[0]["emotion_response_recovery"]["attempts"][0]["finish_reason"] == "length"
    assert raw[0]["emotion_response_recovery"]["attempts"][1]["finish_reason"] == "stop"


def test_three_bad_model_formats_do_not_fail_complete_schema_valid_run(tmp_path: Path, monkeypatch) -> None:
    exit_code, result_dir, _, _ = _run_mocked_official_run(
        tmp_path, monkeypatch, scenario="three_format_errors", sample_count=20
    )
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    submissions = [
        json.loads(line)
        for line in (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert exit_code == 0
    assert report["inference_success"] is True
    assert report["diagnostic_quality_gate"] is False
    assert report["format_success_count"]["emotion_response"] == 17
    assert len(submissions) == 20
    assert report["submission_schema_validated_count"] == 20
    assert all(row["response_text"] == "" for row in submissions[:3])


def test_three_ordinary_route_errors_do_not_fail_other_completed_rows(tmp_path: Path, monkeypatch) -> None:
    exit_code, result_dir, _, _ = _run_mocked_official_run(
        tmp_path, monkeypatch, scenario="three_profile_errors", sample_count=20
    )
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["inference_success"] is True
    assert report["diagnostic_quality_gate"] is False
    assert report["route_failure_count"]["profile"] == 3
    assert report["submission_schema_validated_count"] == 20


def test_official_outputs_overwrite_only_the_two_contract_files(tmp_path: Path, monkeypatch) -> None:
    exit_code, result_dir, _, _ = _run_mocked_official_run(
        tmp_path, monkeypatch, scenario="normal", sample_count=2, preexisting_results=True
    )
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    keep_path = result_dir / "keep.txt"

    assert exit_code == 0
    assert (result_dir / "submission.jsonl").read_bytes() != b"old submission\n"
    assert (result_dir / "performance_report.json").read_bytes() != b"old report\n"
    assert keep_path.read_bytes() == b"keep these exact bytes\x00\xff"
    assert (result_dir / report["outputs"]["raw_outputs.jsonl"]["path"]).is_file()
    assert (result_dir / report["outputs"]["run_progress.json"]["path"]).is_file()


@pytest.mark.parametrize("sample_count", [100, 500, 1000])
def test_mocked_long_runs_keep_all_rows_and_bound_performance_samples(
    tmp_path: Path, monkeypatch, sample_count: int
) -> None:
    exit_code, result_dir, _, _ = _run_mocked_official_run(
        tmp_path, monkeypatch, scenario="normal", sample_count=sample_count
    )
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    submissions = (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()
    raw_path = result_dir / report["outputs"]["raw_outputs.jsonl"]["path"]
    raw_rows = raw_path.read_text(encoding="utf-8").splitlines()

    assert exit_code == 0
    assert len(submissions) == len(raw_rows) == sample_count
    assert report["performance"]["measured_samples"] == min(100, sample_count)
    assert report["io_checkpointing"]["checkpoint_interval_samples"] == 10


def test_mocked_unresolved_length_cap_does_not_fail_a_complete_schema_valid_run(tmp_path: Path, monkeypatch) -> None:
    exit_code, result_dir, _, _ = _run_mocked_official_run(
        tmp_path, monkeypatch, scenario="unresolved", sample_count=1
    )
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["status"] == "passed" and report["complete"] is True
    assert report["unresolved_cap_hits"]["emotion_response"] == 1
    assert report["emotion_response_outcome_count"]["length_limited_unresolved"] == 1
    assert report["quality_warnings"]


@pytest.mark.parametrize(
    ("scenario", "expected_hard_kind"),
    [("profile_error", None), ("prompt_mismatch", "asset_or_protocol_mismatch"), ("service_exit", "service_exit")],
)
def test_mocked_route_protocol_and_service_failures_are_distinguished(
    tmp_path: Path, monkeypatch, scenario: str, expected_hard_kind: str | None
) -> None:
    exit_code, result_dir, _, _ = _run_mocked_official_run(
        tmp_path, monkeypatch, scenario=scenario, sample_count=1
    )
    report_path = next((result_dir / ".emotionalcompany_runs").glob("*/performance_report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["status"] == "failed"
    if expected_hard_kind is None:
        assert report["route_failure_count"]["profile"] == 1
        assert report["hard_failures"] == []
        assert report["server"]["alive_before_intentional_shutdown"] is True
    else:
        assert any(item["kind"] == expected_hard_kind for item in report["hard_failures"])
