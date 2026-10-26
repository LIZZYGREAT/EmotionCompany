from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from participant import run_inference as runner


ROOT = Path(__file__).resolve().parents[1]


def _completion(response: str = "听起来你最近有些紧张。") -> str:
    return json.dumps({
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "response_text": response,
        "memory_refs": [],
    }, ensure_ascii=False)


def _api_config() -> dict:
    return {
        "backend": "openai_compatible",
        "api_model": "mock-model",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "",
        "request_timeout_seconds": 10,
        "api_options": {},
        "openai_compatible": {"pass_reasoning_content_field": False},
        "model_id": "Qwen/Qwen3-1.7B",
        "model_revision": None,
        "experiment_id": "E00.2",
        "generation": {"max_new_tokens": 20, "do_sample": False},
        "performance_rounds": 100,
        "hardware_label": "mock-cpu",
        "prompt": {"version": "P0"},
        "tokenizer": {},
        "context": {
            "max_prompt_tokens": 12,
            "budget_profile": "mock-P12",
            "serving_context_limit": None,
        },
    }


def _run_mock_api(tmp_path: Path, monkeypatch, raw_path: Path | None) -> tuple[Path, list[dict]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner, "WORKSPACE_ROOT", tmp_path)
    input_path = tmp_path / "input.jsonl"
    samples = [
        {"id": "test_000001_t0", "history": [{"role": "user", "content": "短消息🙂"}]},
        {"id": "test_000002_t0", "history": [{"role": "user", "content": "第二条"}]},
    ]
    input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    result_dir = tmp_path / "result"
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(test_file=input_path, result_dir=result_dir))
    monkeypatch.setattr(runner, "load_config", _api_config)
    schema = runner.load_submission_schema()
    enums = runner.submission_enums(schema)
    prompt = (ROOT / "configs/prompts/p0.txt").read_text(encoding="utf-8")
    facts = {"sha256": "mock-p0"}
    monkeypatch.setattr(runner, "load_runtime_protocol", lambda _config: (prompt, schema, enums, facts))
    monkeypatch.setattr(runner, "load_runtime_tokenizer", lambda *_args: (object(), {"signature_sha256": "mock"}))
    monkeypatch.setattr(
        runner,
        "prepare_sample_messages",
        lambda sample, *_args: (
            {"prompt_tokens": 7, "selected_history": sample["history"], "floor_exceeds_cap": False, "prompt_ids": [1]},
            [{"role": "user", "content": sample["history"][-1]["content"]}],
        ),
    )

    def mock_request(_config, messages, _generation):
        if messages[0]["content"] == "第二条":
            raise RuntimeError("mock backend unavailable")
        return _completion(), {
            "server_prompt_tokens": 7,
            "server_completion_tokens": None,
            "finish_reason": None,
        }

    monkeypatch.setattr(runner, "_request_chat_completion_details", mock_request)
    if raw_path is None:
        monkeypatch.delenv("EVALUATION_RAW_OUTPUT_PATH", raising=False)
    else:
        monkeypatch.setenv("EVALUATION_RAW_OUTPUT_PATH", str(raw_path))
    runner.main()
    predictions = [json.loads(line) for line in (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    return result_dir, predictions


def test_opt_in_raw_log_uses_same_generation_and_records_failures(tmp_path, monkeypatch):
    raw_path = tmp_path / "outputs" / "evaluation" / "raw_predictions.jsonl"
    result_dir, predictions = _run_mock_api(tmp_path, monkeypatch, raw_path)
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))

    assert [row["id"] for row in predictions] == [row["id"] for row in raw]
    assert raw[0]["raw_text"] == _completion()
    assert raw[0]["status"] == "success"
    assert raw[0]["generated_tokens"] is None
    assert raw[0]["stop_reason"] == "unknown_server_finish_reason"
    assert raw[1]["raw_text"] is None
    assert raw[1]["status"] == "generation_failure"
    assert raw[1]["error_type"] == "RuntimeError"
    assert predictions[1]["emotion_label"] == ""
    assert report["raw_predictions_sha256"] == runner.sha256(raw_path)
    assert report["test_file_sha256"] == runner.sha256(tmp_path / "input.jsonl")
    assert report["run_id"]
    assert report["experiment_id"] == "E00.2"
    assert report["raw_predictions_rows"] == 2
    assert report["submission_schema_validated"] is True


def test_raw_log_is_disabled_by_default_and_does_not_change_official_files(tmp_path, monkeypatch):
    result_dir, predictions = _run_mock_api(tmp_path, monkeypatch, None)
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    assert len(predictions) == 2
    assert sorted(path.name for path in result_dir.iterdir()) == ["performance_report.json", "submission.jsonl"]
    assert report["raw_predictions_sha256"] is None
    assert report["raw_predictions_path"] is None


def test_opt_in_raw_log_does_not_change_official_submission(tmp_path, monkeypatch):
    default_dir, default_predictions = _run_mock_api(tmp_path / "default", monkeypatch, None)
    raw_path = tmp_path / "with_raw" / "outputs" / "evaluation" / "raw_predictions.jsonl"
    raw_dir, raw_predictions = _run_mock_api(tmp_path / "with_raw", monkeypatch, raw_path)

    assert default_predictions == raw_predictions
    assert sorted(path.name for path in default_dir.iterdir()) == ["performance_report.json", "submission.jsonl"]
    assert sorted(path.name for path in raw_dir.iterdir()) == ["performance_report.json", "submission.jsonl"]


def test_raw_log_cannot_be_written_outside_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATION_RAW_OUTPUT_PATH", str(tmp_path / "raw.jsonl"))
    with pytest.raises(ValueError, match="只能写入 outputs"):
        runner.resolve_raw_output_path(tmp_path / "result", tmp_path / "result/submission.jsonl")


def test_nf4_configuration_is_explicit_and_opt_in():
    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    torch_stub = SimpleNamespace(bfloat16=object())
    assert runner.build_quantization_config({"quantization": {"enabled": False}}, torch_stub, FakeBitsAndBytesConfig) is None
    result = runner.build_quantization_config({
        "quantization": {"enabled": True, "use_double_quant": False},
    }, torch_stub, FakeBitsAndBytesConfig)
    assert result.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch_stub.bfloat16,
        "bnb_4bit_use_double_quant": False,
    }


def test_transformers_timer_synchronizes_before_and_after_the_measured_region(tmp_path, monkeypatch):
    import torch
    import transformers

    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps({
        "id": "test_000001_t0", "history": [{"role": "user", "content": "一条消息"}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    result_dir = tmp_path / "result"
    config = _api_config()
    config.update({"backend": "transformers", "model_path": str(model_dir), "device_map": "cpu", "torch_dtype": "auto", "trust_remote_code": False})
    events: list[str] = []

    class FakeAutoConfig:
        model_type = "mock-causal"

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class FakeModel:
        def eval(self):
            return self

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(test_file=input_path, result_dir=result_dir))
    monkeypatch.setattr(runner, "load_config", lambda: config)
    schema = runner.load_submission_schema()
    enums = runner.submission_enums(schema)
    prompt = (ROOT / "configs/prompts/p0.txt").read_text(encoding="utf-8")
    facts = {"sha256": "mock-p0"}
    monkeypatch.setattr(runner, "load_runtime_protocol", lambda _config: (prompt, schema, enums, facts))
    monkeypatch.setattr(runner, "model_path", lambda _config: model_dir)
    monkeypatch.setattr(runner, "load_runtime_tokenizer", lambda *_args: (object(), {"signature_sha256": "mock"}))
    monkeypatch.setattr(runner, "_weight_identity", lambda *_args: {"fingerprint_status": "mock"})
    monkeypatch.setattr(runner, "prepare_sample_messages", lambda *_args: (
        {"prompt_tokens": 3, "selected_history": [{"role": "user", "content": "一条消息"}], "floor_exceeds_cap": False, "prompt_ids": [1, 2, 3]},
        [{"role": "user", "content": "一条消息"}],
    ))
    monkeypatch.setattr(runner, "synchronize", lambda _torch: events.append("sync"))
    clock_values = iter([10.0, 10.25])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: (events.append("clock"), next(clock_values))[1])
    monkeypatch.setattr(runner, "generate_transformers_sample", lambda *_args: (events.append("generate") or _completion(), {"generated_tokens": 2, "stop_reason": "eos"}))
    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeModel)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", FakeModel)
    monkeypatch.setattr(
        transformers.models.auto.modeling_auto,
        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
        {},
    )
    monkeypatch.delenv("EVALUATION_RAW_OUTPUT_PATH", raising=False)

    runner.main()

    assert events == ["sync", "clock", "generate", "sync", "clock"]
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    assert report["average_latency_ms"] == 250.0
    assert "CUDA synchronized before start and after validation" in report["timing_scope"]
