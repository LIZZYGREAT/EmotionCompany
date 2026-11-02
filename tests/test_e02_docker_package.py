from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import e02_docker_package as docker_package
from scripts import e02_vllm_probe as probe


def _write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _synthetic_package_sources(tmp_path: Path):
    repo = tmp_path / "repo"
    real_config = docker_package.load_config()
    real_smoke_config = docker_package.load_json(docker_package.REPO_ROOT / "configs/deployment/e02_vllm_smoke.json")
    real_prompt = (docker_package.REPO_ROOT / docker_package.PROMPT_PATH).read_bytes()
    for relative in docker_package.COPY_FILES:
        if relative == docker_package.DOCKER_CONFIG:
            continue
        if relative == docker_package.PROMPT_PATH:
            content = real_prompt
        elif relative.suffix == ".json":
            content = "{}"
        else:
            content = f"fixture:{relative.as_posix()}"
        _write(repo / relative, content)
    _write(repo / "src/shared/__init__.py", "")
    _write(repo / "src/shared/prompting.py", "# synthetic source\n")

    base = tmp_path / "base-source"
    _write(base / "config.json", "{}")
    _write(base / "model-00001-of-00001.safetensors", b"synthetic base")
    _write(
        base / "model.safetensors.index.json",
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
    )
    expected_tokenizer_files = real_config["tokenizer"]["files_sha256"]
    for name in expected_tokenizer_files:
        _write(base / name, f"tokenizer:{name}")

    adapter = tmp_path / "adapter-source"
    _write(
        adapter / "adapter_config.json",
        json.dumps({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": sorted(probe.LORA_TARGET_MODULES),
            "base_model_name_or_path": "Qwen/Qwen3-1.7B",
        }),
    )
    _write(adapter / "adapter_model.safetensors", b"synthetic adapter")

    config = real_config
    config["tokenizer"]["files_sha256"] = {
        name: hashlib.sha256((f"tokenizer:{name}").encode()).hexdigest()
        for name in expected_tokenizer_files
    }
    config["prompt"]["sha256"] = hashlib.sha256(real_prompt).hexdigest()
    base_identity = probe.base_weight_identity(base)
    adapter_identity, _ = probe.adapter_identity(adapter)
    config["adapter"]["expected_base_weight_sha256"] = base_identity["observed_weight_hash"]
    config["adapter"]["expected_adapter_weight_sha256"] = adapter_identity["weight_sha256"]
    config["adapter"]["expected_adapter_config_sha256"] = adapter_identity["config_sha256"]
    _write(repo / docker_package.DOCKER_CONFIG, json.dumps(config, indent=2))
    real_smoke_config["tokenizer"]["files_sha256"] = config["tokenizer"]["files_sha256"]
    for key in (
        "expected_base_weight_sha256",
        "expected_adapter_weight_sha256",
        "expected_adapter_config_sha256",
    ):
        real_smoke_config["adapter"][key] = config["adapter"][key]
    _write(repo / "configs/deployment/e02_vllm_smoke.json", json.dumps(real_smoke_config, indent=2))
    return repo, base, adapter, config


def test_static_docker_preflight_uses_locked_e02_runtime():
    result = docker_package.check_static()
    assert result["status"] == "ready_for_asset_packaging"
    assert result["facts"]["vllm_version"] == "0.11.1"
    assert result["facts"]["prompt_tokens"] + result["facts"]["generation_tokens"] == 2480
    assert result["facts"]["serving_context_limit"] == 2560
    assert result["pip_lock_package_count"] > 100


def test_prepare_context_copies_only_runtime_code_and_verified_assets(tmp_path):
    repo, base, adapter, config = _synthetic_package_sources(tmp_path)
    output = repo / "outputs/deployment/e02/docker-context"

    result = docker_package.prepare_context(
        base_model_dir=base,
        adapter_dir=adapter,
        output_dir=output,
        repo_root=repo,
    )

    assert result["status"] == "prepared_for_image_build"
    assert (output / "participant/model/model-00001-of-00001.safetensors").read_bytes() == b"synthetic base"
    assert (output / "participant/adapter/adapter_model.safetensors").read_bytes() == b"synthetic adapter"
    assert not any((output / forbidden).exists() for forbidden in (".git", ".venv", "data", "outputs"))
    manifest_text = (output / "package_manifest.json").read_text(encoding="utf-8")
    assert str(base) not in manifest_text
    assert str(adapter) not in manifest_text
    assert json.loads(manifest_text)["configuration"]["adapter_weight_sha256"] == config["adapter"]["expected_adapter_weight_sha256"]
    with pytest.raises(FileExistsError):
        docker_package.prepare_context(
            base_model_dir=base,
            adapter_dir=adapter,
            output_dir=output,
            repo_root=repo,
        )


def test_prepare_context_rejects_wrong_frozen_hash_before_creating_context(tmp_path):
    repo, base, adapter, config = _synthetic_package_sources(tmp_path)
    config["adapter"]["expected_base_weight_sha256"] = "0" * 64
    _write(repo / docker_package.DOCKER_CONFIG, json.dumps(config))
    baseline = docker_package.load_json(repo / "configs/deployment/e02_vllm_smoke.json")
    baseline["adapter"]["expected_base_weight_sha256"] = "0" * 64
    _write(repo / "configs/deployment/e02_vllm_smoke.json", json.dumps(baseline))
    output = repo / "outputs/docker-context"

    with pytest.raises(ValueError, match="Base 权重 SHA-256"):
        docker_package.prepare_context(
            base_model_dir=base,
            adapter_dir=adapter,
            output_dir=output,
            repo_root=repo,
        )
    assert not output.exists()


def test_verify_output_checks_schema_ids_and_frozen_runtime_identity(tmp_path):
    config = docker_package.load_config()
    expected = docker_package.validate_static_config(config)
    schema = json.loads((docker_package.REPO_ROOT / docker_package.SCHEMA_PATH).read_text(encoding="utf-8"))
    test_file = tmp_path / "input.jsonl"
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    row_id = "test_000001_t0"
    _write(test_file, json.dumps({"id": row_id, "history": []}) + "\n")
    prediction = {
        "id": row_id,
        "response_text": "fixture",
        "emotion_label": schema["properties"]["emotion_label"]["enum"][0],
        "user_profile": {
            dimension: [] for dimension in ("personality_traits", "interests", "style")
        },
        "memory_refs": [],
    }
    _write(result_dir / "submission.jsonl", json.dumps(prediction) + "\n")
    report = {
        "complete": True,
        "submission_rows": 1,
        "required_rounds": 1,
        "submission_schema_validated": True,
        "backend": "openai_compatible",
        "api_model": expected["adapter_model_id"],
        "success_count": 1,
        "parse_failure_count": 0,
        "invalid_label_count": 0,
        "adapter_identity": {
            "weight_sha256": expected["adapter_weight_sha256"],
            "config_sha256": expected["adapter_config_sha256"],
            "server_weight_identity": "unverified",
        },
        "model_weight_identity": {"local_reference_weight_sha256": expected["base_weight_sha256"]},
        "api_server_runtime": {"vllm_version": expected["vllm_version"]},
    }
    _write(result_dir / "performance_report.json", json.dumps(report))

    result = docker_package.verify_output(test_file, result_dir)

    assert result["status"] == "passed_output_contract_and_local_identity"
    assert result["input_rows"] == result["schema_valid_rows"] == 1
    assert result["server_weight_identity"] == "unverified"


def test_verify_output_rejects_input_with_extra_fields(tmp_path):
    test_file = tmp_path / "input.jsonl"
    _write(test_file, json.dumps({"id": "test_000001_t0", "history": [], "target": "forbidden"}) + "\n")
    with pytest.raises(ValueError, match="only id and history"):
        docker_package._read_jsonl(test_file, exact_input=True)


def test_container_entrypoint_rejects_existing_run_outputs(tmp_path):
    test_file = tmp_path / "input.jsonl"
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    _write(test_file, "{}\n")
    _write(result_dir / "performance_report.json", "{}")

    result = subprocess.run(
        ["bash", str(docker_package.REPO_ROOT / "docker/e02/entrypoint.sh"), str(test_file), str(result_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "use a new run directory" in result.stderr
