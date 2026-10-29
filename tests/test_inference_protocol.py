"""Runner contract checks with the pinned local Qwen tokenizer and mock backends."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from participant import run_inference as runner
from src.shared.prompting import (
    build_messages,
    encode_inference_prompt,
    select_recent_complete_suffix as shared_select_recent_complete_suffix,
    serialize_completion,
)
from src.data.token_stats import select_recent_complete_suffix as audit_select_recent_complete_suffix


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def protocol():
    config = runner.load_config()
    prompt, schema, enums, facts = runner.load_runtime_protocol(config)
    tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(config)
    return config, prompt, schema, enums, facts, tokenizer, tokenizer_facts


def _history(*, assistant_text: str = "之前的回答", current_text: str = "现在我有点焦虑"):
    return [
        {"role": "user", "content": "很早以前的问题"},
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": current_text},
    ]


def _valid_completion() -> str:
    return json.dumps({
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "response_text": "听起来你最近承受了不少。",
        "memory_refs": [],
    }, ensure_ascii=False)


def test_i01_uses_p0_and_native_messages_without_id(protocol):
    _, prompt, _, _, facts, tokenizer, _ = protocol
    sample = {"id": "test_000001_t0", "history": _history()}
    selection, messages = runner.prepare_sample_messages(sample, prompt, tokenizer, 1280)
    assert messages[0] == {"role": "system", "content": prompt}
    assert [item["role"] for item in messages[1:]] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == sample["history"][-1]["content"]
    assert all(sample["id"] not in item["content"] for item in messages)
    assert facts["sha256"] == "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2"
    assert selection["prompt_tokens"] == len(selection["prompt_ids"])


def test_i02_t1_keeps_recent_contiguous_complete_suffix(protocol):
    _, prompt, _, _, _, tokenizer, _ = protocol
    history = _history(assistant_text="长历史内容 " * 800, current_text="短的当前消息")
    baseline_ids = encode_inference_prompt(tokenizer, build_messages([history[-1]], prompt))
    selection = shared_select_recent_complete_suffix(history, prompt, len(baseline_ids), tokenizer)
    assert selection["status"] == "ok"
    assert selection["prompt_tokens"] <= len(baseline_ids)
    assert selection["selected_history"] == [history[-1]]
    assert history[-1] == {"role": "user", "content": "短的当前消息"}
    assert selection["selected_start_index"] == 2


def test_i03_prompt_floor_overflow_is_explicit_and_schema_valid(protocol):
    _, prompt, schema, enums, _, tokenizer, _ = protocol
    history = [{"role": "user", "content": "不能截断的当前消息"}]
    floor = encode_inference_prompt(tokenizer, build_messages(history, prompt))
    selection = shared_select_recent_complete_suffix(history, prompt, len(floor) - 1, tokenizer)
    fallback = runner.empty_prediction("test_000002_t0", enums)
    assert selection["status"] == "prompt_floor_exceeds_cap"
    assert selection["selected_history"] == history
    assert selection["prompt_tokens"] == len(floor)
    runner.validate_submission_prediction(fallback, schema)
    assert fallback["emotion_label"] == ""
    assert fallback["user_profile"] == {"personality_traits": [], "interests": [], "style": []}


def test_i04_runner_and_audit_share_exact_tokenizer_selection(protocol):
    _, prompt, _, _, facts, tokenizer, tokenizer_facts = protocol
    history = _history(assistant_text="包含换行\n🙂 的先前回复")
    sample = {"id": "test_000003_t0", "history": history}
    runner_selection, messages = runner.prepare_sample_messages(sample, prompt, tokenizer, 1280)
    audit_selection = audit_select_recent_complete_suffix(history, prompt, 1280, tokenizer)
    assert runner_selection == audit_selection
    assert runner_selection["prompt_ids"] == encode_inference_prompt(tokenizer, messages)
    assert facts["sha256"] == "d427e4996d0075a3adfe03ae25881d093687a6565f5b037f605accecc84c70b2"
    assert tokenizer_facts["signature_sha256"] == "1662c9e99c6185441026a168b2dd6b09af103dc6dc3563b72f598411f14d1913"
    assert tokenizer_facts["template_sha256"] == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


def test_i05_literal_think_text_is_preserved_or_explicitly_blocked():
    messages = [
        {"role": "user", "content": "字面 <think> 不改写\n换行🙂"},
        {"role": "assistant", "content": "原文 </think> 保留", "reasoning_content": ""},
        {"role": "user", "content": "继续"},
    ]
    blocked_config = {"openai_compatible": {"pass_reasoning_content_field": False}}
    with pytest.raises(runner.TemplateEquivalenceError, match="literal </think>"):
        runner.api_message_payload(messages, blocked_config)

    enabled_config = {"openai_compatible": {"pass_reasoning_content_field": True}}
    payload = runner.api_message_payload(messages, enabled_config)
    assert payload[0]["content"] == messages[0]["content"]
    assert payload[1]["content"] == messages[1]["content"]
    assert payload[1]["reasoning_content"] == ""


def test_i06_transformers_mock_decodes_only_new_tokens(protocol):
    _, _, schema, enums, _, _, _ = protocol
    completion = _valid_completion()

    class ToyTokenizer:
        decoded_ids = None

        def decode(self, ids, *, skip_special_tokens):
            self.decoded_ids = ids.tolist()
            assert skip_special_tokens is True
            return completion

    class ToyModel:
        captured = None

        def get_input_embeddings(self):
            return type("Embedding", (), {"weight": torch.zeros(1)})()

        def generate(self, *, input_ids, attention_mask, **kwargs):
            self.captured = (input_ids.detach().cpu().tolist(), attention_mask.detach().cpu().tolist(), kwargs)
            return torch.cat([input_ids, torch.tensor([[92, 93]], device=input_ids.device)], dim=1)

    model = ToyModel()
    tokenizer = ToyTokenizer()
    decoded, details = runner.generate_transformers_sample(
        model, torch, tokenizer, [11, 12, 13], {"max_new_tokens": 8, "do_sample": False}
    )
    prediction, parse_failed, invalid_label = runner.parse_prediction("test_000004_t0", decoded, enums)
    runner.validate_submission_prediction(prediction, schema)
    assert model.captured[0] == [[11, 12, 13]]
    assert model.captured[1] == [[1, 1, 1]]
    assert tokenizer.decoded_ids == [92, 93]
    assert details["generated_tokens"] == 2
    assert not parse_failed and not invalid_label


def test_i07_openai_payload_and_mock_http_preserve_contract(monkeypatch, protocol):
    config, _, _, _, _, _, _ = protocol
    api_config = copy.deepcopy(config)
    api_config.update({"api_model": "qwen-local", "base_url": "http://127.0.0.1:8000/v1", "api_key": "secret"})
    api_config["api_options"] = {"response_format": {"type": "json_object"}}
    messages = [
        {"role": "system", "content": "P0"},
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "回答"},
        {"role": "user", "content": "当前问题"},
    ]
    generation = {"max_new_tokens": 1200, "do_sample": False, "top_p": 1.0}
    payload = runner.build_api_payload(api_config, messages, generation)
    assert [message["role"] for message in payload["messages"]] == ["system", "user", "assistant", "user"]
    assert payload["model"] == "qwen-local"
    assert payload["max_tokens"] == 1200
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"] == {"type": "json_object"}
    assert "reasoning_content" not in payload["messages"][2]

    observed = {}

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": _valid_completion()}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 23, "completion_tokens": 31},
            }).encode()

    def mock_urlopen(request, timeout):
        observed["request"] = request
        observed["timeout"] = timeout
        observed["payload"] = json.loads(request.data)
        return MockResponse()

    monkeypatch.setattr(runner.urllib.request, "urlopen", mock_urlopen)
    content, details = runner._request_chat_completion_details(api_config, messages, generation)
    assert json.loads(content)["emotion_label"] == "anxiety"
    assert details == {"server_prompt_tokens": 23, "server_completion_tokens": 31, "finish_reason": "stop"}
    assert observed["timeout"] == api_config["request_timeout_seconds"]
    assert observed["request"].headers["Authorization"] == "Bearer secret"
    assert observed["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_i08_only_sixteen_emotion_labels_are_normal_and_empty_is_fallback(protocol):
    _, _, schema, enums, _, _, _ = protocol
    assert len(enums["emotion_label"]) == 16
    assert "" not in enums["emotion_label"]
    prediction, parse_failed, invalid_label = runner.parse_prediction(
        "test_000005_t0",
        json.dumps({"response_text": "回答", "emotion_label": "not-a-label", "user_profile": {}}),
        enums,
    )
    runner.validate_submission_prediction(prediction, schema)
    assert not parse_failed and invalid_label
    assert prediction["emotion_label"] == ""


def test_i09_completion_order_and_submission_schema_are_distinct(protocol):
    _, _, schema, enums, _, _, _ = protocol
    raw = json.dumps({
        "response_text": "合法回复",
        "user_profile": {"style": [], "interests": [], "personality_traits": []},
        "emotion_label": "joy",
        "memory_refs": [],
    }, ensure_ascii=False)
    prediction, parse_failed, invalid_label = runner.parse_prediction("test_000006_t0", raw, enums)
    runner.validate_submission_prediction(prediction, schema)
    assert not parse_failed and not invalid_label
    assert list(prediction) == ["id", "response_text", "emotion_label", "user_profile", "memory_refs"]

    official_schema = json.loads((ROOT / "data/train/train_public.schema.json").read_text(encoding="utf-8"))
    target = {
        "emotion_label": "joy",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "response_text": "合法回复",
        "memory_refs": ["mem_000001"],
    }
    completion = serialize_completion(target, official_schema)
    assert list(json.loads(completion)) == ["emotion_label", "user_profile", "response_text", "memory_refs"]


def test_i10_adapter_is_disabled_by_default_and_rejects_invalid_declarations(protocol, tmp_path):
    config, *_ = protocol
    assert runner.validate_adapter_config(config.get("adapter"), "transformers") == {"enabled": False}
    with pytest.raises(ValueError, match="expected_base_weight_sha256"):
        runner.validate_adapter_config({"enabled": True, "path": "outputs/training/a/adapter"}, "transformers")
    valid_shape = {
        "enabled": True,
        "path": "outputs/training/a/adapter",
        "expected_base_weight_sha256": "a" * 64,
        "expected_adapter_weight_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="仅支持 backend=transformers"):
        runner.validate_adapter_config(valid_shape, "openai_compatible")
    with pytest.raises(ValueError, match="64 位 SHA-256"):
        runner.validate_adapter_config({**valid_shape, "expected_adapter_weight_sha256": "wrong"}, "transformers")


def test_i11_adapter_identity_is_separate_and_base_reference_is_checked(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_path = adapter_dir / "adapter_config.json"
    config_path.write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-1.7B"}), encoding="utf-8")
    weights = adapter_dir / "adapter_model.safetensors"
    weights.write_bytes(b"synthetic adapter weights")
    identity, adapter_config = runner.adapter_weight_identity(adapter_dir)
    assert identity["identity_status"] == "observed"
    assert identity["weight_sha256"]
    assert identity["config_sha256"] == runner.sha256(config_path)
    assert runner.validate_adapter_base_reference(adapter_config, "Qwen/Qwen3-1.7B", tmp_path / "base") == "Qwen/Qwen3-1.7B"
    with pytest.raises(ValueError, match="does not match configured model"):
        runner.validate_adapter_base_reference({"base_model_name_or_path": "Qwen/other"}, "Qwen/Qwen3-1.7B", tmp_path / "base")


def test_i11_quantized_adapter_load_prepares_base_before_loading(monkeypatch, tmp_path):
    events = []

    class BaseModel:
        prepared = False

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, **kwargs):
            events.append(("adapter", model.prepared, adapter_path, kwargs))
            return {"base": model, "adapter_path": adapter_path}

    def prepare_model_for_kbit_training(model, *, use_gradient_checkpointing):
        events.append(("prepare", use_gradient_checkpointing))
        model.prepared = True
        return model

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=FakePeftModel,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
    ))
    base = BaseModel()
    loaded = runner.load_peft_adapter(
        base,
        tmp_path / "adapter",
        prepare_quantized_base=True,
        is_trainable=False,
        local_files_only=True,
    )

    assert events == [
        ("prepare", False),
        ("adapter", True, str(tmp_path / "adapter"), {"is_trainable": False, "local_files_only": True}),
    ]
    assert loaded["base"].prepared is True


def test_i11_unquantized_adapter_load_skips_kbit_preparation(monkeypatch, tmp_path):
    events = []

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, **kwargs):
            events.append((model, adapter_path, kwargs))
            return model

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("unquantized adapter loading must not run k-bit preparation")

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=FakePeftModel,
        prepare_model_for_kbit_training=unexpected_prepare,
    ))
    base = object()
    result = runner.load_peft_adapter(base, tmp_path / "adapter", prepare_quantized_base=False)

    assert result is base
    assert events == [(base, str(tmp_path / "adapter"), {"is_trainable": False, "local_files_only": True})]


def test_i10_budget_preflight_rejects_p_plus_g_over_k(tmp_path, monkeypatch):
    config = json.loads((ROOT / "participant/configs/inference_config.json").read_text(encoding="utf-8"))
    config["context"]["serving_context_limit"] = 2000
    config_path = tmp_path / "inference.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(runner, "CONFIG_PATH", config_path)
    with pytest.raises(ValueError, match=r"P\(1280\) \+ G\(1200\).*K\(2000\)"):
        runner.load_config()

    config["context"]["serving_context_limit"] = None
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert runner.load_config()["context"]["serving_context_limit"] is None


def test_i11_runner_imports_from_participant_working_directory(tmp_path):
    runner_script = ROOT / "participant/run_inference.py"
    result = subprocess.run(
        [sys.executable, str(runner_script), "--help"],
        cwd=runner_script.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "test_file result_dir" in result.stdout
    assert "本地模型单轮推理" in result.stdout


def test_i12_mock_api_run_emits_one_valid_row_per_sample(tmp_path, monkeypatch, protocol, capsys):
    config, prompt, schema, enums, _, tokenizer, _ = protocol
    api_config = copy.deepcopy(config)
    api_config.update({
        "backend": "openai_compatible",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_model": "qwen-local",
        "api_key": "",
        "hardware_label": "mock-test",
        "performance_rounds": 1,
    })
    api_config["openai_compatible"] = {"pass_reasoning_content_field": True}
    config_path = tmp_path / "inference.json"
    config_path.write_text(json.dumps(api_config), encoding="utf-8")
    monkeypatch.setattr(runner, "CONFIG_PATH", config_path)

    test_file = tmp_path / "test.jsonl"
    rows = [
        {"id": "test_000007_t0", "history": _history()},
        {"id": "test_000008_t0", "history": _history(assistant_text="字面 </think> 与🙂 保持原样")},
        {"id": "test_000009_t0", "history": [{"role": "user", "content": "超长但必须原样保留的当前消息。" * 1200}]},
    ]
    test_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    result_dir = tmp_path / "result"

    def mock_request(config, messages, generation):
        payload = runner.build_api_payload(config, messages, generation)
        assert [message["role"] for message in payload["messages"]][-1] == "user"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        if "</think>" in messages[1].get("content", "") or any(
            "</think>" in item.get("content", "") for item in messages
        ):
            assert any(item.get("reasoning_content") == "" for item in payload["messages"] if item["role"] == "assistant")
        prompt_tokens = len(encode_inference_prompt(tokenizer, messages))
        return _valid_completion(), {
            "server_prompt_tokens": prompt_tokens,
            "server_completion_tokens": 25,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(runner, "_request_chat_completion_details", mock_request)
    monkeypatch.setattr(sys, "argv", [str(ROOT / "participant/run_inference.py"), str(test_file), str(result_dir)])
    runner.main()
    capsys.readouterr()

    output_rows = [json.loads(line) for line in (result_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((result_dir / "performance_report.json").read_text(encoding="utf-8"))
    for row in output_rows:
        runner.validate_submission_prediction(row, schema)
    assert [row["id"] for row in output_rows] == [row["id"] for row in rows]
    assert report["samples"] == report["submission_rows"] == len(rows)
    assert report["complete"] is True
    assert report["rounds"] == 1
    assert report["success_count"] == 2
    assert report["prompt_floor_exceeds_cap_count"] == 1
    assert report["failure_counts"] == {"prompt_floor_exceeds_cap": 1, "success": 2}
    assert report["server_prompt_token_counts_match_client_count"] == 2
    assert report["server_exact_token_id_equivalence"] == "BLOCKED_NOT_RUN_no_pinned_vllm_token_id_diagnostic"
    assert report["budget"]["compatibility_status"] == "not_verified"
    assert report["protocol"]["id_in_prompt"] is False

