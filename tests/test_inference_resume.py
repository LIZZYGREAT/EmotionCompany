"""Opt-in inference recovery must preserve committed predictions and identity."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from participant import run_inference as runner


def _prefix(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    inputs = tmp_path / "input.jsonl"
    samples = [{"id": f"test_{index:06d}_t0", "history": [{"role": "user", "content": "synthetic"}]} for index in (1, 2)]
    inputs.write_text("".join(json.dumps(row) + "\n" for row in samples))
    kwargs = dict(result_dir=result, submission_path=result / "submission.jsonl", raw_path=result / "raw.jsonl", samples=samples, test_file=inputs, config={"model": "fixture"}, run_id="fixture-run", experiment_id="fixture")
    return kwargs


def test_resume_trims_only_uncommitted_tail_and_checks_prefix_hash(tmp_path):
    kwargs = _prefix(tmp_path)
    _, _, _, state = runner._prepare_resume_prefix(**kwargs)
    first = b'{"id":"test_000001_t0"}\n'
    kwargs["submission_path"].write_bytes(first + b'{"id":"test_000002_t0"}\n')
    kwargs["raw_path"].write_bytes(first + b'{"partial"')
    state.update(completed_samples=1, submission_prefix_sha256=hashlib.sha256(first).hexdigest(), raw_prefix_sha256=hashlib.sha256(first).hexdigest())
    runner._write_json_atomic(kwargs["result_dir"] / "run_progress.json", state)
    completed, submissions, raw, _state = runner._prepare_resume_prefix(**kwargs)
    assert completed == len(submissions) == len(raw) == 1
    assert kwargs["submission_path"].read_bytes() == kwargs["raw_path"].read_bytes() == first
    kwargs["raw_path"].write_bytes(first.replace(b"000001", b"000002"))
    with pytest.raises(ValueError, match="IDs"):
        runner._prepare_resume_prefix(**kwargs)


def test_resume_rejects_unknown_files_and_changed_run_or_config(tmp_path):
    kwargs = _prefix(tmp_path)
    kwargs["submission_path"].write_text('{"id":"existing"}\n')
    with pytest.raises(FileExistsError, match="no resumable progress"):
        runner._prepare_resume_prefix(**kwargs)
    kwargs["submission_path"].unlink()
    runner._prepare_resume_prefix(**kwargs)
    kwargs["config"] = {"model": "different"}
    with pytest.raises(ValueError, match="differs"):
        runner._prepare_resume_prefix(**kwargs)


def test_resume_runtime_rejects_actual_base_or_adapter_change(tmp_path):
    progress = tmp_path / "progress.json"
    state = runner._bind_resume_runtime(progress, {}, {"model_weight_identity": {"observed_weight_hash": "a"}}, {"signature_sha256": "t"})
    with pytest.raises(ValueError, match="actual Base"):
        runner._bind_resume_runtime(progress, state, {"model_weight_identity": {"observed_weight_hash": "b"}}, {"signature_sha256": "t"})


def test_mock_runner_interrupt_then_resume_generates_only_remaining_id(tmp_path, monkeypatch):
    kwargs = _prefix(tmp_path)
    monkeypatch.setattr(runner, "WORKSPACE_ROOT", tmp_path)
    raw_path = tmp_path / "outputs/raw.jsonl"
    monkeypatch.setenv("EVALUATION_RAW_OUTPUT_PATH", str(raw_path))
    monkeypatch.setenv("EVALUATION_RUN_ID", "resume-fixture")
    monkeypatch.setenv("EVALUATION_RESUME", "1")
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(test_file=kwargs["test_file"], result_dir=kwargs["result_dir"]))
    config = {"backend": "openai_compatible", "experiment_id": "resume-fixture", "api_model": "fixture", "base_url": "http://localhost", "performance_rounds": 100, "hardware_label": "synthetic-cpu", "model_id": "fixture", "tokenizer": {}, "generation": {"do_sample": False, "max_new_tokens": 20}, "context": {"max_prompt_tokens": 1280, "budget_profile": "synthetic-P1280", "serving_context_limit": None}}
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "run_configuration_provenance", lambda: {})
    schema = runner.load_submission_schema()
    monkeypatch.setattr(runner, "load_runtime_protocol", lambda cfg: ("fixture", schema, runner.submission_enums(schema), {"sha256": "p0"}))
    monkeypatch.setattr(runner, "load_runtime_tokenizer", lambda *args: (object(), {"signature_sha256": "tokenizer", "template_sha256": "template"}))
    monkeypatch.setattr(runner, "prepare_sample_messages", lambda sample, *args: ({"prompt_tokens": 7, "selected_history": sample["history"], "floor_exceeds_cap": False, "prompt_ids": [1]}, [{"id": sample["id"]}]))
    calls = []
    completion = json.dumps({"emotion_label": "anxiety", "user_profile": {"personality_traits": [], "interests": [], "style": []}, "response_text": "合成测试回复", "memory_refs": []}, ensure_ascii=False)
    def request(cfg, messages, generation):
        calls.append(messages[0]["id"])
        if len(calls) == 2:
            raise KeyboardInterrupt("synthetic interruption")
        return completion, {"server_prompt_tokens": 7, "server_completion_tokens": 15, "finish_reason": "stop"}
    monkeypatch.setattr(runner, "_request_chat_completion_details", request)
    with pytest.raises(KeyboardInterrupt):
        runner.main()
    assert json.loads((kwargs["result_dir"] / "run_progress.json").read_text())["completed_samples"] == 1
    runner.main()
    assert calls == ["test_000001_t0", "test_000002_t0", "test_000002_t0"]
    report = json.loads((kwargs["result_dir"] / "performance_report.json").read_text())
    predictions = [json.loads(line) for line in kwargs["submission_path"].read_text().splitlines()]
    assert [row["id"] for row in predictions] == [row["id"] for row in kwargs["samples"]]
    assert report["samples"] == report["raw_predictions_rows"] == 2
    assert report["complete"] is False
    assert report["resumption"]["reused_samples"] == 1
    assert report["submission_sha256"] == runner.sha256(kwargs["submission_path"])
    assert report["raw_predictions_sha256"] == runner.sha256(raw_path)
