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
        "adapters": {
            "profile": {"path": str(profile)},
            "emotion_response": {"path": str(er)},
        },
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
                "expected_version": "not-a-runtime-gate",
            }
        },
    }
    return DualLoRAVLLMService(
        config=config,
        root=tmp_path,
        log_path=tmp_path / "server.log",
        vllm_bin="vllm",
    )


def test_build_command_uses_configured_base_and_both_loras(tmp_path: Path) -> None:
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


def test_chat_completion_sends_required_payload_without_token_id_extension(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    seen = {}

    def fake_http_json(url, *, timeout, payload=None):
        seen.update(url=url, timeout=timeout, payload=payload)
        return {
            "id": "chatcmpl-test",
            "model": "profile",
            "choices": [{"message": {"content": "Personality=introverted"}, "finish_reason": "stop"}],
        }

    monkeypatch.setattr(dual_vllm, "_http_json", fake_http_json)
    result = service.chat_completion(
        model_alias="profile",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=256,
    )

    assert seen["url"].endswith("/chat/completions")
    assert seen["payload"]["model"] == "profile"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["payload"]["max_tokens"] == 256
    assert seen["payload"]["temperature"] == 0
    assert seen["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "return_token_ids" not in seen["payload"]
    assert result["text"] == "Personality=introverted"
    assert result["requested_model_alias"] == "profile"
    assert result["usage_prompt_tokens"] is None
    assert result["usage_completion_tokens"] is None


def test_chat_completion_treats_empty_content_as_field_fallback_and_metadata_as_optional(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(dual_vllm, "_http_json", lambda *args, **kwargs: {
        "choices": [{"message": {"content": None}}],
    })

    result = service.chat_completion(
        model_alias="emotion-response",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1200,
    )

    assert result["text"] == ""
    assert result["finish_reason"] is None
    assert result["response_id"] is None
    assert result["usage_completion_tokens"] is None


def test_chat_completion_treats_unreadable_content_as_empty_field_fallback(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(dual_vllm, "_http_json", lambda *args, **kwargs: {
        "choices": [{"message": {"content": {"unexpected": "object"}}}],
    })

    result = service.chat_completion(
        model_alias="emotion-response",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1200,
    )

    assert result["text"] == ""


def test_chat_completion_requires_a_usable_choices_message(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(dual_vllm, "_http_json", lambda *args, **kwargs: {"usage": {}})

    with pytest.raises(VLLMServiceError, match=r"choices\[0\].message.content"):
        service.chat_completion(
            model_alias="profile",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
        )


def test_start_uses_models_endpoint_only_for_readiness_without_alias_or_version_gate(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(service, "_resolve_binary", lambda: "/vllm")
    monkeypatch.setattr(service, "_port_available", lambda: True)
    monkeypatch.setattr(dual_vllm.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(dual_vllm, "_http_json", lambda *args, **kwargs: {"data": [{"id": "server-model"}]})

    facts = service.start()

    assert facts["status"] == "ready"
    assert facts["models"] == ["server-model"]
    assert "version" not in facts
    assert not hasattr(service, "_check_version")
    service.close()
