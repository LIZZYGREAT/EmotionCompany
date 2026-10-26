from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from participant import run_inference as runner


ROOT = Path(__file__).resolve().parents[1]


def _clear_evaluation_overrides(monkeypatch):
    for name in (
        "EVALUATION_INFERENCE_CONFIG", "EVALUATION_EVALUATION_CONFIG", "EVALUATION_BACKEND",
        "EVALUATION_BASE_URL", "EVALUATION_API_MODEL", "EVALUATION_API_KEY",
        "EVALUATION_EXPERIMENT_ID", "EVALUATION_MODEL_PATH", "EVALUATION_HARDWARE_LABEL",
        "EVALUATION_TOKENIZER_PATH", "EVALUATION_ENABLE_NF4", "EVALUATION_MAX_PROMPT_TOKENS",
        "EVALUATION_PROMPT_BUDGET_PROFILE", "EVALUATION_SERVING_CONTEXT_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_load_config_default_is_unchanged_and_explicit_override_is_validated(tmp_path, monkeypatch):
    _clear_evaluation_overrides(monkeypatch)
    expected = json.loads((ROOT / "participant/configs/inference_config.json").read_text(encoding="utf-8"))
    assert runner.load_config() == expected

    override = copy.deepcopy(expected)
    override["hardware_label"] = "e00-2-test-override"
    override["generation"] = {
        "max_new_tokens": 1200, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    }
    override["sampling"] = {"base_seed": 42}
    config_path = tmp_path / "inference.json"
    config_path.write_text(json.dumps(override), encoding="utf-8")
    monkeypatch.setenv("EVALUATION_INFERENCE_CONFIG", str(config_path))
    loaded = runner.load_config()
    assert loaded["hardware_label"] == "e00-2-test-override"
    assert loaded["sampling"] == {"base_seed": 42}

    override.pop("sampling")
    config_path.write_text(json.dumps(override), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling.base_seed"):
        runner.load_config()


def test_sampling_seed_is_stable_per_id_and_sets_cpu_and_cuda_rng():
    first = runner.derive_sample_seed(42, "test_000001_t0")
    assert first == runner.derive_sample_seed(42, "test_000001_t0")
    assert first != runner.derive_sample_seed(42, "test_000002_t0")
    assert 0 <= first < 2**64

    calls = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("cpu", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
    )
    runner.set_sampling_seed(fake_torch, first)
    assert calls == [("cpu", first), ("cuda", first)]


def test_nf4_observation_uses_loaded_modules_dtype_and_device_map():
    class FakeLinear4bit:
        def __init__(self, dtype="torch.bfloat16", quant_type="nf4", double_quant=True):
            self.compute_dtype = dtype
            self.weight = SimpleNamespace(
                quant_type=quant_type,
                quant_state=SimpleNamespace(nested=object() if double_quant else None),
            )

    class FakeModel:
        hf_device_map = {"model": "cuda:0"}

        def __init__(self, modules):
            self._modules = modules

        def modules(self):
            return iter(self._modules)

    config = {"quantization": {"enabled": True, "use_double_quant": True}}
    torch_cuda = SimpleNamespace(is_available=lambda: True)
    torch_stub = SimpleNamespace(cuda=torch_cuda)
    observed = runner.inspect_quantization_runtime(
        FakeModel([FakeLinear4bit(), object(), FakeLinear4bit()]),
        config,
        torch_stub,
        FakeLinear4bit,
    )
    assert observed["real_nf4_verified"] is True
    assert observed["linear4bit_module_count"] == 2
    assert observed["compute_dtype"] == "bfloat16"
    assert observed["quant_type"] == "nf4"
    assert observed["double_quant_runtime_observed"] is True
    assert observed["offload_devices"] == []

    cpu_offload_model = FakeModel([FakeLinear4bit()])
    cpu_offload_model.hf_device_map = {"model": "cuda:0", "offload": "cpu"}
    offloaded = runner.inspect_quantization_runtime(cpu_offload_model, config, torch_stub, FakeLinear4bit)
    assert offloaded["real_nf4_verified"] is False
    assert offloaded["offload_devices"] == ["cpu"]

    wrong_dtype = runner.inspect_quantization_runtime(
        FakeModel([FakeLinear4bit(dtype="torch.float16")]), config, torch_stub, FakeLinear4bit
    )
    assert wrong_dtype["real_nf4_verified"] is False


def test_nf4_does_not_fall_back_when_cuda_is_unavailable():
    with pytest.raises(RuntimeError, match="拒绝降级到 CPU"):
        runner.require_cuda_for_quantization(
            {"quantization": {"enabled": True}},
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        )
    runner.require_cuda_for_quantization(
        {"quantization": {"enabled": False}},
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )


def test_runner_configuration_provenance_never_records_api_key(tmp_path, monkeypatch):
    _clear_evaluation_overrides(monkeypatch)
    evaluation_config = tmp_path / "eval.yaml"
    evaluation_config.write_text("experiment_id: E00.2\n", encoding="utf-8")
    monkeypatch.setenv("EVALUATION_EVALUATION_CONFIG", str(evaluation_config))
    monkeypatch.setenv("EVALUATION_API_KEY", "secret-must-never-appear")
    monkeypatch.setenv("EVALUATION_HARDWARE_LABEL", "test-hardware")
    provenance = runner.run_configuration_provenance()
    serialized = json.dumps(provenance)
    assert provenance["evaluation_config_sha256"] == runner.sha256(evaluation_config)
    assert "secret-must-never-appear" not in serialized
    assert "EVALUATION_API_KEY" not in provenance["environment_override_names"]
    assert "EVALUATION_HARDWARE_LABEL" in provenance["environment_override_names"]


def test_weight_identity_uses_only_indexed_safetensors_shards(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "used-a.safetensors").write_bytes(b"a")
    (model / "used-b.safetensors").write_bytes(b"b")
    (model / "unreferenced.safetensors").write_bytes(b"must not enter the aggregate")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.a": "used-a.safetensors", "layer.b": "used-b.safetensors"},
    }), encoding="utf-8")

    identity = runner._weight_identity(model, {"model_id": "mock", "model_revision": None})
    assert identity["fingerprint_status"] == "indexed_safetensors_shards_sha256"
    assert [item["path"] for item in identity["files"]] == ["used-a.safetensors", "used-b.safetensors"]


def test_weight_identity_rejects_missing_indexed_shard(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.a": "missing.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing or empty"):
        runner._weight_identity(model, {"model_id": "mock"})
