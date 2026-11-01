from __future__ import annotations

import io
import json
import urllib.error

import pytest

from scripts import e02_token_equivalence as diagnostic


def test_token_id_comparison_requires_exact_values_not_just_equal_lengths():
    exact = diagnostic.compare_token_ids([10, 20, 30], [10, 20, 30])
    same_length_difference = diagnostic.compare_token_ids([10, 20, 30], [10, 21, 30])

    assert exact["exact_match"] is True
    assert same_length_difference["exact_match"] is False
    assert same_length_difference["first_mismatch_index"] == 1
    assert same_length_difference["first_mismatch_window"] == {
        "start_index": 0,
        "expected_ids": [10, 20, 30],
        "observed_ids": [10, 21, 30],
    }


def test_token_id_comparison_reports_length_difference_and_rejects_bool_ids():
    comparison = diagnostic.compare_token_ids([10, 20, 30], [10, 20])
    assert comparison["exact_match"] is False
    assert comparison["first_mismatch_index"] == 2
    with pytest.raises(ValueError, match="整数数组"):
        diagnostic.compare_token_ids([True], [1])


def test_diagnostic_payload_requests_frozen_lora_and_token_ids_with_thinking_disabled():
    config = {
        "api_model": "e01-full",
        "api_options": {},
        "openai_compatible": {"pass_reasoning_content_field": False},
    }
    messages = [
        {"role": "system", "content": "synthetic prompt"},
        {"role": "assistant", "content": "Marker </think> is literal.", "reasoning_content": ""},
        {"role": "user", "content": "synthetic current user"},
    ]

    payload, reasoning_extension_used = diagnostic.build_diagnostic_payload(config, messages)

    assert payload["model"] == "e01-full"
    assert payload["max_tokens"] == 1
    assert payload["return_token_ids"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][1]["reasoning_content"] == ""
    assert reasoning_extension_used is True
    assert config["openai_compatible"]["pass_reasoning_content_field"] is False


def test_diagnostic_payload_omits_nonportable_reasoning_field_without_literal_marker():
    config = {
        "api_model": "e01-full",
        "api_options": {},
        "openai_compatible": {"pass_reasoning_content_field": False},
    }
    messages = [
        {"role": "system", "content": "synthetic prompt"},
        {"role": "assistant", "content": "ordinary text", "reasoning_content": ""},
        {"role": "user", "content": "synthetic current user"},
    ]

    payload, reasoning_extension_used = diagnostic.build_diagnostic_payload(config, messages)

    assert "reasoning_content" not in payload["messages"][1]
    assert reasoning_extension_used is False


def test_frozen_route_rejects_base_model_name_instead_of_lora_alias():
    with pytest.raises(ValueError, match="冻结 LoRA 别名"):
        diagnostic.validate_frozen_lora_route({
            "api_model": "qwen3-base",
            "adapter": {"enabled": True, "expected_server_model_id": "e01-full"},
        })


def test_local_expected_prompt_uses_shared_t1_recent_contiguous_suffix():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return {"input_ids": list(range(2 + len(messages) - 1))}

    sample = {
        "id": "synthetic-t1",
        "history": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ],
    }

    selection, messages = diagnostic.prepare_local_expected_prompt(
        sample, "P0", FakeTokenizer(), prompt_cap=5
    )

    assert selection["selected_start_index"] == 2
    assert selection["selected_history"] == sample["history"][2:]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert len(selection["prompt_ids"]) == selection["prompt_tokens"] <= 5


def test_unlabelled_input_rejects_target_extra_fields_and_duplicate_ids(tmp_path):
    path = tmp_path / "input.jsonl"
    row = {"id": "sample-1", "history": [{"role": "user", "content": "hello"}]}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复"):
        diagnostic.read_unlabelled_input(path)

    row["target"] = {"emotion_label": "joy"}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="恰好包含 id 与 history"):
        diagnostic.read_unlabelled_input(path)


def test_two_synthetic_think_samples_are_appended_without_mutating_input():
    rows = [{"id": "real-1", "history": [{"role": "user", "content": "hello"}]}]

    samples = diagnostic.diagnostic_samples(rows)

    assert len(samples) == 3
    assert [row["id"] for row in samples[1:]] == [
        "e02_synthetic_think_01",
        "e02_synthetic_think_02",
    ]
    assert "</think>" in samples[1]["history"][1]["content"]
    assert len(rows) == 1


def test_diagnostic_refuses_to_reuse_existing_run_directory(tmp_path):
    run_root = tmp_path / "already-used"
    run_root.mkdir()
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        diagnostic.run_diagnostic(
            config_path=tmp_path / "unused-config.json",
            input_path=tmp_path / "unused-input.jsonl",
            run_root=run_root,
            vllm_bin="vllm",
        )


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def _request_config():
    return {
        "api_model": "e01-full",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "",
        "api_options": {},
        "openai_compatible": {"pass_reasoning_content_field": False},
    }


def test_request_compares_server_prompt_ids_and_does_not_return_text_or_full_ids(monkeypatch):
    monkeypatch.setattr(
        diagnostic.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse({
            "model": "e01-full",
            "prompt_token_ids": [11, 22, 33],
            "usage": {"prompt_tokens": 3},
            "choices": [{"message": {"content": "private completion"}}],
        }),
    )
    result = diagnostic.request_prompt_token_ids(
        _request_config(),
        [{"role": "system", "content": "private message"}, {"role": "user", "content": "hello"}],
        [11, 22, 33],
        timeout=1,
    )

    assert result["status"] == "exact_match"
    assert result["exact_match"] is True
    assert result["server_usage_prompt_tokens"] == 3
    assert "prompt_token_ids" not in result
    assert "private completion" not in json.dumps(result)
    assert "private message" not in json.dumps(result)


def test_request_marks_missing_server_ids_not_observable(monkeypatch):
    monkeypatch.setattr(
        diagnostic.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse({
            "model": "e01-full",
            "usage": {"prompt_tokens": 3},
            "choices": [{"message": {"content": "not retained"}}],
        }),
    )
    result = diagnostic.request_prompt_token_ids(
        _request_config(),
        [{"role": "user", "content": "hello"}],
        [11, 22, 33],
        timeout=1,
    )
    assert result["status"] == "not_observable_missing_prompt_token_ids"
    assert result["response_top_level_keys"] == ["choices", "model", "usage"]


def test_request_summarizes_http_error_without_retaining_error_body(monkeypatch):
    error_body = json.dumps({"error": {"type": "BadRequestError", "code": "invalid_request", "message": "private prompt text"}}).encode()

    def raise_http_error(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:8000/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(error_body),
        )

    monkeypatch.setattr(diagnostic.urllib.request, "urlopen", raise_http_error)
    result = diagnostic.request_prompt_token_ids(
        _request_config(),
        [{"role": "user", "content": "private prompt text"}],
        [11],
        timeout=1,
    )
    assert result["status"] == "http_error"
    assert result["http_status"] == 400
    assert result["error_type"] == "BadRequestError"
    assert "private prompt text" not in json.dumps(result)
