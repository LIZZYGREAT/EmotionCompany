from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import final_dual_docker_package as package


def test_rewritten_adapter_config_points_to_container_base_and_hash_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "adapter_config.json"
    source.write_text(
        json.dumps({"base_model_name_or_path": "/host/model", "r": 8}), encoding="utf-8"
    )

    payload = package._rewritten_er_config_bytes(source)
    rewritten = json.loads(payload)

    assert rewritten["base_model_name_or_path"] == "/root/participant/model"
    assert rewritten["r"] == 8
    assert hashlib.sha256(payload).hexdigest() == package.sha256_bytes(payload)
    assert package._contains_host_path(rewritten) is False


@pytest.mark.parametrize(
    "value",
    [
        {"path": "/home/developer/model"},
        {"nested": ["/mnt/x/models/base"]},
        {"path": "C:\\Users\\someone\\model"},
    ],
)
def test_host_specific_absolute_paths_are_rejected(value: dict) -> None:
    assert package._contains_host_path(value) is True


def test_prepare_context_refuses_paths_outside_ignored_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="below the ignored outputs"):
        package.prepare_context(output_dir=tmp_path / "context")


def test_dockerfile_pins_validated_build_images() -> None:
    dockerfile = (package.ROOT / package.DOCKERFILE).read_text(encoding="utf-8")

    assert f"FROM {package.CUDA_IMAGE}" in dockerfile
    assert f"FROM {package.MINIFORGE_IMAGE} AS miniforge" in dockerfile
    assert "COPY --from=miniforge /opt/conda /opt/conda" in dockerfile


def test_read_jsonl_requires_exact_official_input_keys_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(
        json.dumps({"id": "test_000001_t1", "history": []}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert package._read_jsonl(path, exact_input=True)[0]["id"] == "test_000001_t1"

    path.write_text(
        json.dumps({"id": "test_000001_t1", "history": [], "target": {}}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only id and history"):
        package._read_jsonl(path, exact_input=True)


def test_copy_adapter_rewrites_only_config_and_verifies_both_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/host/model", "r": 8}), encoding="utf-8"
    )
    weight = b"adapter weights"
    (source / "adapter_model.safetensors").write_bytes(weight)
    destination = tmp_path / "participant" / "adapters" / "emotion-response"

    facts = package._copy_adapter(source, destination, rewrite_base=True)

    assert facts["weight_sha256"] == hashlib.sha256(weight).hexdigest()
    assert facts["config_sha256"] == package.sha256_file(destination / "adapter_config.json")
    assert facts["source_config_sha256"] == package.sha256_file(source / "adapter_config.json")
    assert facts["base_path_rewritten"] is True
    assert facts["base_model_name_or_path"] == "/root/participant/model"
