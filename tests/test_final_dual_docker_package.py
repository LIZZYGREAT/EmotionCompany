from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import final_dual_docker_package as package


@pytest.mark.parametrize(
    "value",
    [
        {"path": "/home/developer/model"},
        {"nested": ["/mnt/x/models/base"]},
        {"path": "C:\\Users\\someone\\model"},
    ],
)
def test_host_specific_absolute_paths_are_rejected(value: dict[str, Any]) -> None:
    assert package._contains_host_path(value) is True


def test_prepare_context_refuses_paths_outside_ignored_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="below the ignored outputs"):
        package.prepare_context(output_dir=tmp_path / "context")


def test_dockerfile_pins_validated_build_images_and_minimal_runtime_entrypoint() -> None:
    dockerfile = (package.ROOT / package.DOCKERFILE).read_text(encoding="utf-8")
    entrypoint = (package.ROOT / package.ENTRYPOINT).read_text(encoding="utf-8")

    assert f"FROM {package.CUDA_IMAGE}" in dockerfile
    assert f"FROM {package.MINIFORGE_IMAGE} AS miniforge" in dockerfile
    assert "COPY --from=miniforge /opt/conda /opt/conda" in dockerfile
    assert "nvidia-smi" not in entrypoint
    assert "exec sleep infinity" in entrypoint
    assert 'exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"' in entrypoint
    assert "participant/run_inference.py" not in dockerfile
    assert "scripts/e02_vllm_probe.py" not in dockerfile


def test_static_package_preflight_accepts_simplified_runtime_contract() -> None:
    result = package.check_static()

    assert result["status"] == "ready_for_asset_packaging"
    assert result["package_preflight"]["status"] == "passed"


def test_read_jsonl_requires_id_and_history_but_ignores_additional_metadata(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    row = {
        "id": "id-preserved-as-is",
        "history": [],
        "target": {"emotion": "ignored"},
        "metadata": {"split": "test"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert package._read_jsonl(path, exact_input=True) == [row]

    path.write_text(json.dumps({"id": "missing-history"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="id and history"):
        package._read_jsonl(path, exact_input=True)


def test_read_jsonl_reports_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text('{"id":"x","history":[]}\ninvalid\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"input.jsonl:2: invalid JSON"):
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _valid_submission(sample_id: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "response_text": "A response",
        "emotion_label": "anxiety",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "memory_refs": [],
    }


def _verify_fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    repo = tmp_path / "repo"
    result_dir = tmp_path / "result"
    (repo / package.SCHEMA_PATH).parent.mkdir(parents=True)
    result_dir.mkdir()
    schema_source = package.ROOT / package.SCHEMA_PATH
    (repo / package.SCHEMA_PATH).write_bytes(schema_source.read_bytes())

    input_rows = [
        {
            "id": f"test_{index:06d}_t1",
            "history": [{"role": "user", "content": f"message {index}"}],
            "platform_metadata": {"ignored": True},
        }
        for index in range(1, 4)
    ]
    test_file = tmp_path / "input.jsonl"
    _write_jsonl(test_file, input_rows)
    outputs = [_valid_submission(row["id"]) for row in input_rows]
    _write_jsonl(result_dir / "submission.jsonl", outputs)
    report = {
        "version": "final-dual-official-run-v2",
        "status": "passed",
        "complete": True,
        "backend": "vllm",
        "samples": 3,
        "submission_rows": 3,
        "performance": {"measurement_limit": 100, "measured_samples": 3},
        "outputs": {
            "submission.jsonl": {"path": "submission.jsonl", "rows": 3},
            "performance_report.json": {"path": "performance_report.json"},
        },
    }
    (result_dir / "performance_report.json").write_text(json.dumps(report), encoding="utf-8")
    return repo, test_file, result_dir, input_rows, outputs


def test_verify_output_checks_only_official_files_ids_order_and_schema(tmp_path: Path) -> None:
    repo, test_file, result_dir, _input_rows, _outputs = _verify_fixture(tmp_path)

    result = package.verify_output(
        test_file=test_file,
        result_dir=result_dir,
        repo_root=repo,
        expected_rows=3,
    )

    assert result["status"] == "passed_interface_output"
    assert result["input_rows"] == result["submission_rows"] == 3
    assert result["schema_valid_rows"] == 3
    assert result["performance_measured_samples"] == 3


@pytest.mark.parametrize("failure", ["id_order", "schema", "missing_report"])
def test_verify_output_rejects_broken_official_contract(tmp_path: Path, failure: str) -> None:
    repo, test_file, result_dir, _input_rows, outputs = _verify_fixture(tmp_path)
    if failure == "id_order":
        _write_jsonl(result_dir / "submission.jsonl", list(reversed(outputs)))
    elif failure == "schema":
        invalid = {**outputs[0], "extra": "not allowed"}
        _write_jsonl(result_dir / "submission.jsonl", [invalid, *outputs[1:]])
    else:
        (result_dir / "performance_report.json").unlink()

    with pytest.raises((ValueError, FileNotFoundError)):
        package.verify_output(
            test_file=test_file,
            result_dir=result_dir,
            repo_root=repo,
            expected_rows=3,
        )
