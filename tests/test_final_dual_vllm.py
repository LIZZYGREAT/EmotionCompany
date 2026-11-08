from __future__ import annotations

from pathlib import Path

import pytest

import src.inference.final_dual_vllm as dual_vllm
from src.inference.final_dual_vllm import DualLoRAVLLMService, VLLMServiceError


def _service(tmp_path: Path) -> DualLoRAVLLMService:
    base = tmp_path / "base"
    profile = tmp_path / "profile"
    er = tmp_path / "er"
    for path in (base, profile, er):
        path.mkdir()
    config = {
        "model_path": str(base),
        "deployment": {
            "vllm": {
                "host": "127.0.0.1",
                "port": 8000,
                "base_url": "http://127.0.0.1:8000/v1",
                "served_model_name": "qwen-base",
                "dtype": "bfloat16",
                "max_model_len": 2560,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 2560,
                "max_lora_rank": 8,
                "max_loras": 1,
                "max_cpu_loras": 2,
                "gpu_memory_utilization": 0.8,
                "expected_version": "0.11.1",
            }
        },
    }
    lock = {"deployment_adapters": {"profile": {"path": str(profile)}, "emotion_response": {"path": str(er)}}}
    return DualLoRAVLLMService(config=config, model_lock=lock, root=tmp_path, log_path=tmp_path / "server.log", vllm_bin="vllm")


def test_build_command_registers_both_locked_loras_with_serial_gpu_limits(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.resolved_binary = "/opt/conda/bin/vllm"

    command = service.build_command()

    assert command[0:3] == ["/opt/conda/bin/vllm", "serve", str(tmp_path / "base")]
    assert "--enable-lora" in command
    assert command[command.index("--max-loras") + 1] == "1"
    assert command[command.index("--max-cpu-loras") + 1] == "2"
    lora_index = command.index("--lora-modules")
    assert command[lora_index + 1] == f"profile={tmp_path / 'profile'}"
    assert command[lora_index + 2] == f"emotion-response={tmp_path / 'er'}"


def test_chat_completion_sends_frozen_greedy_protocol_and_checks_exact_prompt_ids(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    seen = {}

    def fake_http_json(url, *, timeout, payload=None):
        seen.update(url=url, timeout=timeout, payload=payload)
        return {
            "id": "chatcmpl-test",
            "model": "profile",
            "choices": [{"message": {"content": "Personality=introverted"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            "prompt_token_ids": [11, 12],
        }

    monkeypatch.setattr(dual_vllm, "_http_json", fake_http_json)
    result = service.chat_completion(
        model_alias="profile",
        messages=[{"role": "user", "content": "hello"}],
        expected_prompt_token_ids=[11, 12],
        max_tokens=256,
    )

    assert seen["url"].endswith("/chat/completions")
    assert seen["payload"]["model"] == "profile"
    assert seen["payload"]["temperature"] == 0
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["payload"]["return_token_ids"] is True
    assert result["text"] == "Personality=introverted"
    assert result["prompt_token_ids_exact_match"] is True
    assert result["usage_completion_tokens"] == 3
    assert result["requested_model_alias"] == "profile"


def test_chat_completion_reports_token_mismatch_without_hiding_generation(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(dual_vllm, "_http_json", lambda *args, **kwargs: {
        "choices": [{"message": {"content": "emotion=neutral;response=ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        "prompt_token_ids": [11, 99],
    })

    result = service.chat_completion(
        model_alias="emotion-response",
        messages=[{"role": "user", "content": "hello"}],
        expected_prompt_token_ids=[11, 12],
        max_tokens=1200,
    )

    assert result["text"] == "emotion=neutral;response=ok"
    assert result["prompt_token_ids_exact_match"] is False
    assert result["prompt_token_count_expected"] == result["prompt_token_count_observed"] == 2


def test_chat_completion_fails_closed_when_server_token_ids_are_missing(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(dual_vllm, "_http_json", lambda *args, **kwargs: {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })

    with pytest.raises(VLLMServiceError, match="prompt_token_ids"):
        service.chat_completion(
            model_alias="profile",
            messages=[{"role": "user", "content": "hello"}],
            expected_prompt_token_ids=[11],
            max_tokens=256,
        )


def test_model_list_gate_requires_base_and_both_adapter_aliases(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.list_models = lambda: ["qwen-base", "profile", "emotion-response"]
    assert set(service.require_adapters()) == {"qwen-base", "profile", "emotion-response"}

    service.list_models = lambda: ["qwen-base", "profile"]
    with pytest.raises(VLLMServiceError, match="emotion-response"):
        service.require_adapters()
