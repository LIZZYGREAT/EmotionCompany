from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import participant.final_dual_inference as final_dual


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sample(index: int, *, content: str = "hello") -> dict[str, Any]:
    return {
        "id": f"test_{index:06d}_t1",
        "history": [{"role": "user", "content": content}],
    }


def test_read_test_samples_ignores_extra_metadata_and_filters_history_fields(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    row = {
        "id": "unconstrained-id-kept-as-is",
        "history": [{
            "role": "user", "content": "hello", "turn_id": 2,
            "private_score": 99, "target": "never pass to Prompt",
        }],
        "target": {"emotion": "anxiety"},
        "metadata": {"source": "test"},
    }
    _write_jsonl(path, [row])

    assert final_dual.read_test_samples(path) == [{
        "id": "unconstrained-id-kept-as-is",
        "history": [{"role": "user", "content": "hello", "turn_id": 2}],
    }]


@pytest.mark.parametrize("row", [{"history": []}, {"id": "x"}])
def test_read_test_samples_requires_id_and_history(tmp_path: Path, row: dict[str, Any]) -> None:
    path = tmp_path / "input.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="provide id and history"):
        final_dual.read_test_samples(path)


def test_read_test_samples_reports_invalid_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text('{"id":"test_000001_t1","history":[]}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"input.jsonl:2: invalid JSON"):
        final_dual.read_test_samples(path)


def _run_mocked_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
    *,
    request_error_at: int | None = None,
    preserve_existing: bool = False,
) -> tuple[int, Path, list[tuple[str, list[dict[str, Any]], int]], dict[str, Any]]:
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, rows)
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    keep_content = b"keep this file\x00\xff"
    if preserve_existing:
        (result_dir / "submission.jsonl").write_bytes(b"old submission\n")
        (result_dir / "performance_report.json").write_bytes(b"old report\n")
        (result_dir / "keep.txt").write_bytes(keep_content)

    config_path = tmp_path / "deployment.json"
    config_path.write_text(json.dumps({
        "mode": "dual_adapter",
        "backend": "vllm",
        "model_path": str(tmp_path / "base"),
        "model_lock_path": str(tmp_path / "missing-model-lock.json"),
        "prompts": {},
        "context": {"max_prompt_tokens": 1280},
        "generation": {
            "profile": {"max_new_tokens": 256},
            "emotion_response": {"max_new_tokens": 1200},
            "eos_token_id": 151645,
            "pad_token_id": 151643,
        },
        "deployment": {"vllm": {"request_timeout_seconds": 900}},
    }), encoding="utf-8")

    prompts = {
        "selection": "selection",
        "profile": "profile",
        "profile_history_budget": "profile history budget",
        "emotion_response": "emotion response",
    }
    adapter_paths = {"profile": tmp_path / "profile", "emotion_response": tmp_path / "er"}
    monkeypatch.setattr(final_dual, "load_runtime_assets", lambda _config: (object(), prompts, adapter_paths))
    prompt_samples: list[dict[str, Any]] = []

    def build_prompt(sample: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        prompt_samples.append(sample)
        if any(not message.get("content") for message in sample["history"]):
            raise final_dual.PromptBuildError("invalid_history", "history content is missing")
        return {"messages": [{"role": "user", "content": sample["id"]}], "prompt_ids": [1]}

    monkeypatch.setattr(final_dual, "build_profile_prompt", build_prompt)
    monkeypatch.setattr(final_dual, "build_emotion_response_prompt", build_prompt)
    calls: list[tuple[str, list[dict[str, Any]], int]] = []
    service_state = {"closed": False}

    class FakeService:
        def __init__(self, **_kwargs: Any):
            pass

        def start(self) -> dict[str, Any]:
            return {"startup_seconds": 0.01, "models": []}

        def chat_completion(
            self, *, model_alias: str, messages: list[dict[str, Any]], max_tokens: int
        ) -> dict[str, Any]:
            calls.append((model_alias, messages, max_tokens))
            if request_error_at is not None and len(calls) == request_error_at:
                raise RuntimeError("simulated HTTP request failure")
            if model_alias == "profile":
                return {"text": "malformed profile output"}
            sample_id = messages[0]["content"]
            text = "emotion=invalid_label;response=Kept valid response text." if sample_id == "test_000001_t1" else (
                "emotion=anxiety;response=A normal response."
            )
            return {"text": text}

        def close(self) -> int:
            service_state["closed"] = True
            return -15

    monkeypatch.setattr(final_dual, "DualLoRAVLLMService", FakeService)
    monkeypatch.setenv("EVALUATION_INFERENCE_CONFIG", str(config_path))
    monkeypatch.setenv("EVALUATION_BACKEND", "vllm")
    monkeypatch.delenv("EVALUATION_RUN_ID", raising=False)
    monkeypatch.delenv("EVALUATION_RESUME", raising=False)
    monkeypatch.setattr(sys, "argv", ["final_dual_inference.py", str(test_file), str(result_dir)])

    return final_dual.main(), result_dir, calls, {
        "prompt_samples": prompt_samples,
        "service_state": service_state,
        "keep_content": keep_content,
    }


def test_main_falls_back_per_field_continues_and_overwrites_only_official_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            **_sample(1),
            "target": {"emotion": "must not enter the prompt"},
            "history": [{"role": "user", "content": "hello", "turn_id": 0, "extra": "ignored"}],
        },
        {"id": "test_000002_t1", "history": ["malformed history message"]},
        _sample(3),
    ]
    code, result_dir, calls, facts = _run_mocked_main(
        tmp_path, monkeypatch, rows, preserve_existing=True
    )

    assert code == 0
    assert [call[0] for call in calls] == ["profile", "emotion-response", "profile", "emotion-response"]
    assert all(call[2] in {256, 1200} for call in calls)
    assert facts["service_state"]["closed"] is True
    assert facts["prompt_samples"][0] == {
        "id": "test_000001_t1",
        "history": [{"role": "user", "content": "hello", "turn_id": 0}],
    }

    submissions = [json.loads(line) for line in (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in submissions] == [row["id"] for row in rows]
    schema = json.loads(final_dual.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    assert all(validator.is_valid(row) for row in submissions)
    assert submissions[0]["emotion_label"] == ""
    assert submissions[0]["response_text"] == "Kept valid response text."
    assert submissions[1]["user_profile"] == {
        "personality_traits": [], "interests": [], "style": []
    }

    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    assert stat.S_IMODE((result_dir / "submission.jsonl").stat().st_mode) == 0o644
    assert stat.S_IMODE((result_dir / "performance_report.json").stat().st_mode) == 0o644
    assert report["status"] == "passed"
    assert report["complete"] is True
    assert report["samples"] == report["submission_rows"] == 3
    assert report["field_fallback_count"]["emotion_label"] == 2
    assert report["field_fallback_count"]["response_text"] == 1
    assert report["performance"]["measured_samples"] == 3
    assert (result_dir / "keep.txt").read_bytes() == facts["keep_content"]
    assert {path.name for path in result_dir.iterdir()} == {
        "keep.txt", "submission.jsonl", "performance_report.json"
    }


def test_main_reports_input_line_error_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    test_file = tmp_path / "input.jsonl"
    test_file.write_text('{"id":"test_000001_t1","history":[]}\nnot-json\n', encoding="utf-8")
    config_path = tmp_path / "deployment.json"
    config_path.write_text('{"mode":"dual_adapter","backend":"vllm"}', encoding="utf-8")
    monkeypatch.setenv("EVALUATION_INFERENCE_CONFIG", str(config_path))
    monkeypatch.setattr(sys, "argv", ["final_dual_inference.py", str(test_file), str(tmp_path / "result")])

    assert final_dual.main() == 1
    assert "input.jsonl:2: invalid JSON" in capsys.readouterr().err


def test_service_failure_returns_nonzero_preserves_old_outputs_and_closes_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, result_dir, _calls, facts = _run_mocked_main(
        tmp_path, monkeypatch, [_sample(1)], request_error_at=1, preserve_existing=True
    )

    assert code == 1
    assert facts["service_state"]["closed"] is True
    assert (result_dir / "submission.jsonl").read_bytes() == b"old submission\n"
    assert (result_dir / "performance_report.json").read_bytes() == b"old report\n"
    assert (result_dir / "keep.txt").read_bytes() == facts["keep_content"]
    assert sorted(path.name for path in result_dir.iterdir()) == [
        "keep.txt", "performance_report.json", "submission.jsonl"
    ]


def test_main_measures_only_the_first_100_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_sample(index) for index in range(1, 102)]
    code, result_dir, calls, _facts = _run_mocked_main(tmp_path, monkeypatch, rows)

    assert code == 0
    assert len(calls) == 2 * len(rows)
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    assert report["performance"]["measurement_limit"] == 100
    assert report["performance"]["measured_samples"] == 100
    assert len(report["performance"]["latencies_ms"]) == 100
    assert len((result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()) == 101
