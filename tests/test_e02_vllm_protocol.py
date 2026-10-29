from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import e02_vllm_probe as probe


def _synthetic_artifacts(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"synthetic base weights")

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": sorted(probe.LORA_TARGET_MODULES),
        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
    }
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter weights")
    base_identity = probe.base_weight_identity(base)
    adapter_identity, _ = probe.adapter_identity(adapter)
    return base, adapter, base_identity, adapter_identity


def test_e02_preflight_verifies_frozen_synthetic_adapter_identity(tmp_path):
    base, adapter, base_identity, adapter_identity = _synthetic_artifacts(tmp_path)
    report = probe.validate_artifact_identity(
        base_path=base,
        adapter_path=adapter,
        model_id="Qwen/Qwen3-1.7B",
        alias="e01-full",
        base_alias="qwen3-base",
        max_lora_rank=8,
        expected_base_sha256=base_identity["observed_weight_hash"],
        expected_adapter_sha256=adapter_identity["weight_sha256"],
        expected_adapter_config_sha256=adapter_identity["config_sha256"],
    )
    assert report["status"] == "passed_local_artifact_preflight"
    assert report["adapter"]["config_summary"]["r"] == 8
    assert report["server_weight_identity"] == "unverified_until_vllm_runtime"


def test_e02_preflight_rejects_rank_alias_and_frozen_hash_mismatches(tmp_path):
    base, adapter, base_identity, adapter_identity = _synthetic_artifacts(tmp_path)
    kwargs = {
        "base_path": base,
        "adapter_path": adapter,
        "model_id": "Qwen/Qwen3-1.7B",
        "alias": "e01-full",
        "base_alias": "qwen3-base",
        "max_lora_rank": 4,
        "expected_base_sha256": base_identity["observed_weight_hash"],
        "expected_adapter_sha256": adapter_identity["weight_sha256"],
        "expected_adapter_config_sha256": adapter_identity["config_sha256"],
    }
    with pytest.raises(ValueError, match="超过 max_lora_rank"):
        probe.validate_artifact_identity(**kwargs)
    with pytest.raises(ValueError, match="Base served-model-name"):
        probe.validate_artifact_identity(**{**kwargs, "alias": "qwen3-base", "max_lora_rank": 8})
    with pytest.raises(ValueError, match="Base 权重 SHA-256"):
        probe.validate_artifact_identity(**{**kwargs, "expected_base_sha256": "0" * 64, "max_lora_rank": 8})


def test_e02_vllm_version_parser_requires_an_exact_release(monkeypatch, tmp_path):
    binary = tmp_path / "vllm"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    def mock_run(arguments, **kwargs):
        if arguments[-1] == "--version":
            return type("Result", (), {"returncode": 0, "stdout": "vllm 0.11.1\n", "stderr": ""})()
        return type("Result", (), {
            "returncode": 0,
            "stdout": "--enable-lora --lora-modules --max-lora-rank --dtype --max-model-len --max-num-seqs --max-num-batched-tokens",
            "stderr": "",
        })()

    monkeypatch.setattr(probe.subprocess, "run", mock_run)
    assert probe.inspect_vllm_cli(str(binary), "0.11.1")["version"] == "0.11.1"

    def dev_version(arguments, **kwargs):
        if arguments[-1] == "--version":
            return type("Result", (), {"returncode": 0, "stdout": "0.11.1.dev0\n", "stderr": ""})()
        return mock_run(arguments, **kwargs)

    monkeypatch.setattr(probe.subprocess, "run", dev_version)
    with pytest.raises(RuntimeError, match="不是精确的 release"):
        probe.inspect_vllm_cli(str(binary), "0.11.1")


def test_e02_server_model_list_rejects_base_only_and_returns_advertised_alias():
    base_only = {"object": "list", "data": [{"id": "qwen3-base"}]}
    with pytest.raises(ValueError, match="未出现在服务模型列表"):
        probe.verify_requested_model(base_only, "e01-full")

    both = {"object": "list", "data": [{"id": "qwen3-base"}, {"id": "e01-full"}]}
    assert probe.verify_requested_model(both, "e01-full") == ["qwen3-base", "e01-full"]


def test_e02_adapter_config_and_api_payload_preserve_expected_lora_route(tmp_path):
    from participant import run_inference as runner

    config = runner.validate_adapter_config(
        {
            "enabled": True,
            "path": str(tmp_path / "adapter"),
            "expected_base_weight_sha256": "a" * 64,
            "expected_adapter_weight_sha256": "b" * 64,
            "expected_adapter_config_sha256": "c" * 64,
            "expected_server_model_id": "e01-full",
            "expected_rank": 8,
        },
        "openai_compatible",
    )
    payload = runner.build_api_payload(
        {"api_model": config["expected_server_model_id"]},
        [{"role": "user", "content": "synthetic input"}],
        {"max_new_tokens": 5, "do_sample": False},
    )
    assert payload["model"] == "e01-full"
    with pytest.raises(ValueError, match="必须提供冻结的"):
        runner.validate_adapter_config(
            {
                "enabled": True,
                "path": str(tmp_path / "adapter"),
                "expected_base_weight_sha256": "a" * 64,
                "expected_adapter_weight_sha256": "b" * 64,
            },
            "openai_compatible",
        )
    with pytest.raises(RuntimeError, match="冻结的 LoRA 别名"):
        runner.validate_api_adapter_route(config, "qwen3-base", ["qwen3-base", "e01-full"])
    with pytest.raises(RuntimeError, match="包含请求别名"):
        runner.validate_api_adapter_route(config, "e01-full", ["qwen3-base"])
    assert runner.validate_api_adapter_route(config, "e01-full", ["qwen3-base", "e01-full"])["requested_model_id"] == "e01-full"


def test_e02_deployment_config_freezes_full_b_identity_and_smoke_budget():
    config_path = Path(__file__).resolve().parents[1] / "configs/deployment/e02_vllm_smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["adapter"]["expected_base_weight_sha256"] == "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
    assert config["adapter"]["expected_adapter_weight_sha256"] == "77b61e8a4caf46f54a8c813e98def488c7a8493fe891966e47df1cdac6df7d67"
    assert config["adapter"]["expected_adapter_config_sha256"] == "05d578eb242868385839b2bc779c5cb88911973db766e2a793cb361b42aafafd"
    assert config["api_model"] == config["adapter"]["expected_server_model_id"] == "e01-full"
    assert config["context"]["max_prompt_tokens"] + config["generation"]["max_new_tokens"] <= config["context"]["serving_context_limit"]
    assert config["performance_rounds"] == 10
    assert config["deployment"]["vllm"]["expected_version"] == "0.11.1"


def test_e02_sample_selector_uses_only_unlabelled_input_and_covers_length_range(tmp_path):
    source = tmp_path / "unlabelled.jsonl"
    rows = [
        {"id": f"test_{index:06d}_t0", "history": [{"role": "user", "content": "x" * (index + 1)}]}
        for index in range(20)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "sample.jsonl"
    result = probe.write_unlabelled_sample(source, output, 5)
    sampled = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["sample_count"] == 5
    assert result["history_character_length_min_max"] == [1, 20]
    assert all(set(row) == {"id", "history"} for row in sampled)


def test_e02_sample_selector_rejects_gold_bearing_input(tmp_path):
    source = tmp_path / "gold.jsonl"
    source.write_text(
        json.dumps({"id": "test_1", "history": [], "target": {"emotion_label": "joy"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"id\+history-only"):
        probe.write_unlabelled_sample(source, tmp_path / "sample.jsonl", 1)


def test_e02_manifest_records_observed_route_but_not_server_weight_identity(tmp_path):
    run_root = tmp_path / "run"
    runner_dir = run_root / "runner"
    runner_dir.mkdir(parents=True)
    (run_root / "vllm_environment_freeze.txt").write_text("vllm==0.11.1\n", encoding="utf-8")
    test_file = run_root / "test.jsonl"
    test_file.write_text(json.dumps({"id": "test_000001_t0", "history": []}) + "\n", encoding="utf-8")
    config_path = run_root / "inference.json"
    config_path.write_text(json.dumps({
        "adapter": {
            "expected_base_weight_sha256": "a" * 64,
            "expected_adapter_weight_sha256": "b" * 64,
            "expected_adapter_config_sha256": "c" * 64,
        },
        "deployment": {"vllm": {"expected_version": "0.11.1"}},
    }), encoding="utf-8")
    (runner_dir / "submission.jsonl").write_text("{}\n", encoding="utf-8")
    (runner_dir / "performance_report.json").write_text(json.dumps({
        "run_id": "E02.1-test",
        "backend": "openai_compatible",
        "complete": True,
        "submission_schema_validated": True,
        "samples": 1,
        "submission_rows": 1,
        "success_count": 1,
        "failure_counts": {},
        "api_model": "e01-full",
        "api_server_runtime": {"vllm_version": "0.11.1"},
        "model_weight_identity": {"local_reference_weight_sha256": "a" * 64},
        "adapter_identity": {
            "weight_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "locally_verified_adapter_file": {"rank": 8, "target_modules": ["q_proj"]},
            "server_route_observed": {"requested_model_id": "e01-full"},
        },
        "protocol": {"prompt": {}, "tokenizer": {}},
        "generation_config": {"max_new_tokens": 1200},
        "budget": {"P_plus_G": 2480},
        "server_exact_token_id_equivalence": "not_verified",
        "raw_predictions_sha256": "d" * 64,
    }), encoding="utf-8")
    result = probe.write_smoke_manifest(
        run_root=run_root,
        runner_dir=runner_dir,
        test_file=test_file,
        config_path=config_path,
        vllm_version="0.11.1",
        server_model_ids=["qwen3-base", "e01-full"],
        requested_model="e01-full",
        gpu_facts={"status": "observed", "devices": []},
    )
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert manifest["execution"]["server_weight_identity"] == "unverified"
    assert manifest["checks"]["requested_route_advertised"] is True
