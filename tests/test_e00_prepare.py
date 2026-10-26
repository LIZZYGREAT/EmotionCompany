from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation import load_submission_schema, schema_enums
from src.evaluation.prepare_dev import (
    prepare_records,
    select_smoke_indexes,
    validate_d03_source,
    validate_dev_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(sample_id: str, *, emotion: str = "joy", history: list[dict] | None = None, response: str = "你好🙂"):
    return {
        "sample_id": sample_id,
        "conversation_id": f"conv-{sample_id}",
        "source": {
            "split": "train",
            "source_line": 1,
            "prediction_point_index": 0,
            "target_user_turn_id": 1,
            "assistant_response_turn_id": 2,
        },
        "history": history or [{"role": "user", "content": "最近有点紧张。"}],
        "target": {
            "emotion_label": emotion,
            "user_profile": {"personality_traits": [], "interests": [], "style": []},
            "response_text": response,
            "memory_refs": ["mem_000001"],
        },
    }


def _config(tmp_path: Path, count: int) -> dict:
    config = yaml.safe_load((ROOT / "configs/evaluation/e00.yaml").read_text(encoding="utf-8"))
    config["data"]["expected_dev_count"] = count
    config["data"]["dev_path"] = "data/processed/v1/dev.jsonl"
    return config


def test_dev_adapter_keeps_target_out_of_input_and_preserves_references(tmp_path):
    rows = [_row("train:conv-a:pp0"), _row("train:conv-b:pp0", emotion="anxiety", response="可以慢慢来。")]
    target_before = copy.deepcopy([row["target"] for row in rows])
    config = _config(tmp_path, len(rows))
    d03_manifest_path = tmp_path / "d03-manifest.json"
    d03_manifest_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "prepared"
    manifest = prepare_records(
        rows,
        enums=schema_enums(load_submission_schema(ROOT)),
        d03_manifest={"pipeline_version": "D0.3-v1"},
        d03_manifest_path=d03_manifest_path,
        source_sha256="a" * 64,
        config=config,
        config_sha256="b" * 64,
        output_dir=output_dir,
    )

    inputs = [json.loads(line) for line in (output_dir / "inference_input.jsonl").read_text(encoding="utf-8").splitlines()]
    references = [json.loads(line) for line in (output_dir / "references.jsonl").read_text(encoding="utf-8").splitlines()]
    id_map = [json.loads(line) for line in (output_dir / "id_map.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in inputs] == ["test_000001_t0", "test_000002_t0"]
    assert all(set(row) == {"id", "history"} for row in inputs)
    assert all("target" not in row and "sample_id" not in row for row in inputs)
    assert [row["target"] for row in references] == target_before
    assert [row["sample_id"] for row in id_map] == [row["sample_id"] for row in rows]
    assert [row["eval_id"] for row in references] == [row["id"] for row in inputs]
    assert manifest["source"]["dev_rows"] == 2
    assert manifest["outputs"]["inference_input.jsonl"]["rows"] == 2
    assert manifest["outputs"]["references.jsonl"]["sha256"] == hashlib.sha256(
        (output_dir / "references.jsonl").read_bytes()
    ).hexdigest()
    assert all(item["source"]["split"] == "train" for item in id_map)


def test_dev_adapter_rerun_is_byte_deterministic(tmp_path):
    rows = [_row(f"train:conv-{index}:pp0", emotion="anxiety" if index % 2 else "joy") for index in range(30)]
    config = _config(tmp_path, len(rows))
    d03_manifest_path = tmp_path / "d03-manifest.json"
    d03_manifest_path.write_text('{"stable":true}\n', encoding="utf-8")
    output_dir = tmp_path / "prepared"
    kwargs = dict(
        rows=rows,
        enums=schema_enums(load_submission_schema(ROOT)),
        d03_manifest={"pipeline_version": "D0.3-v1"},
        d03_manifest_path=d03_manifest_path,
        source_sha256="c" * 64,
        config=config,
        config_sha256="d" * 64,
        output_dir=output_dir,
    )
    prepare_records(**kwargs)
    first = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    prepare_records(**kwargs)
    second = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert first == second
    smoke_inputs = [json.loads(line) for line in (output_dir / "smoke_input.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(smoke_inputs) == 20
    assert all(set(row) == {"id", "history"} for row in smoke_inputs)


def test_dev_contract_rejects_duplicate_ids_bad_source_target_and_last_role():
    enums = schema_enums(load_submission_schema(ROOT))
    rows = [_row("train:duplicate:pp0"), _row("train:duplicate:pp0")]
    with pytest.raises(ValueError, match="重复"):
        validate_dev_rows(rows, 2, enums)

    bad_source = _row("train:conv:pp0")
    bad_source["source"]["split"] = "val"
    with pytest.raises(ValueError, match="source.split"):
        validate_dev_rows([bad_source], 1, enums)

    bad_target = _row("train:conv:pp0")
    bad_target["target"]["emotion_label"] = ""
    with pytest.raises(ValueError, match="target"):
        validate_dev_rows([bad_target], 1, enums)

    bad_history = _row("train:conv:pp0", history=[{"role": "assistant", "content": "no current user"}])
    with pytest.raises(ValueError, match="最后一条 history"):
        validate_dev_rows([bad_history], 1, enums)


def test_dev_contract_rejects_count_mismatch_and_bad_manifest_hash(tmp_path):
    enums = schema_enums(load_submission_schema(ROOT))
    with pytest.raises(ValueError, match="数量不符"):
        validate_dev_rows([_row("train:conv:pp0")], 2, enums)

    source = tmp_path / "dev.jsonl"
    source.write_text('{"x":1}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="必须且只能包含输出"):
        validate_d03_source(source, manifest, {"outputs": []}, expected_count=1)

    bad_manifest = {"pipeline_version": "D0.3-v1", "outputs": [{
        "path": "data/processed/v1/dev.jsonl", "records": 1, "sha256": "0" * 64,
    }]}
    with pytest.raises(ValueError, match="SHA-256"):
        validate_d03_source(source, manifest, bad_manifest, expected_count=1)


def test_smoke_selection_is_seeded_and_reports_uncovered_rare_features():
    rows = [
        _row("train:a:pp0", emotion="joy"),
        _row("train:b:pp0", emotion="anxiety", history=[{"role": "user", "content": "x" * 900}]),
        _row("train:c:pp0", emotion="fear", history=[{"role": "user", "content": "表情🙂\n</think>"}]),
    ]
    enums = schema_enums(load_submission_schema(ROOT))
    selected_a, info_a = select_smoke_indexes(rows, enums, seed=9, count=2)
    selected_b, info_b = select_smoke_indexes(rows, enums, seed=9, count=2)
    assert selected_a == selected_b
    assert info_a == info_b
    assert info_a["actual_count"] == 2
    assert info_a["uncovered_available_features"]
