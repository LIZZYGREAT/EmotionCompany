from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation import jsonl_bytes, load_submission_schema, schema_enums, sha256_file
from src.evaluation.prepare_dev import prepare_records
from src.evaluation.score import _load_train_majority, score_files
from scripts import e00_mock_smoke


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCHEMA = ROOT / "data/train/train_public.schema.json"


def _target(emotion: str, response: str) -> dict:
    return {
        "emotion_label": emotion,
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "response_text": response,
        "memory_refs": [],
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_d03_shaped_fixture(tmp_path: Path) -> tuple[dict, str, Path]:
    config_bytes = (ROOT / "configs/evaluation/e00.yaml").read_bytes()
    config = yaml.safe_load(config_bytes)
    config["data"]["expected_dev_count"] = 6
    config["data"]["train_path"] = str(tmp_path / "missing-train.jsonl")
    config["smoke"]["count"] = 3
    enums = schema_enums(load_submission_schema(ROOT))
    rows = []
    for index in range(6):
        emotion = enums["emotion_label"][index % len(enums["emotion_label"])]
        sample_id = f"train:mock_conversation_{index + 1:03d}:pp0"
        history = [{"role": "user", "content": f"合成验证消息 {index + 1}🙂"}]
        rows.append({
            "sample_id": sample_id,
            "conversation_id": f"mock_conversation_{index + 1:03d}",
            "source": {"split": "train", "source_line": index + 1, "prediction_point_index": 0},
            "history": history,
            "target": _target(emotion, f"合成参考回复 {index + 1}。"),
        })

    fixture_dir = tmp_path / "d03-shaped"
    fixture_dir.mkdir()
    d03_dev_bytes = jsonl_bytes(rows)
    d03_dev_path = fixture_dir / "dev.jsonl"
    d03_dev_path.write_bytes(d03_dev_bytes)
    schema_sha = sha256_file(TRAIN_SCHEMA)
    d03_manifest = {
        "pipeline_version": "D0.3-v1",
        "inputs": {"train": {"schema_path": "data/train/train_public.schema.json", "schema_sha256": schema_sha}},
        "outputs": [{
            "path": "data/processed/v1/dev.jsonl",
            "sha256": hashlib.sha256(d03_dev_bytes).hexdigest(),
            "records": len(rows),
        }],
    }
    d03_manifest_path = fixture_dir / "d03-manifest.json"
    _write_json(d03_manifest_path, d03_manifest)
    target_schema_facts = {"path": "data/train/train_public.schema.json", "sha256": schema_sha}
    prepare_records(
        rows,
        enums=enums,
        d03_manifest=d03_manifest,
        d03_manifest_path=d03_manifest_path,
        source_sha256=hashlib.sha256(d03_dev_bytes).hexdigest(),
        config=config,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        output_dir=fixture_dir,
        target_schema_facts=target_schema_facts,
    )

    for collection in ("full", "smoke"):
        suffix = "" if collection == "full" else "smoke_"
        references_name = f"{suffix}references.jsonl"
        inputs_name = "inference_input.jsonl" if collection == "full" else "smoke_input.jsonl"
        id_map_name = "id_map.jsonl" if collection == "full" else "smoke_id_map.jsonl"
        references = [json.loads(line) for line in (fixture_dir / references_name).read_text(encoding="utf-8").splitlines()]
        predictions = []
        raw_rows = []
        for reference in references:
            eval_id = reference["eval_id"]
            target = reference["target"]
            predictions.append({"id": eval_id, **target})
            raw_rows.append({
                "id": eval_id,
                "raw_text": json.dumps({key: target[key] for key in ("emotion_label", "user_profile", "response_text", "memory_refs")}, ensure_ascii=False),
                "status": "success",
                "prompt_tokens": 10,
                "generated_tokens": 20,
                "stop_reason": "eos",
                "latency_ms": 12.0,
                "removed_history_messages": 0,
                "backend": "mock",
                "error_type": None,
            })
        submission_path = fixture_dir / f"{collection}-submission.jsonl"
        raw_path = fixture_dir / f"{collection}-raw.jsonl"
        submission_path.write_bytes(jsonl_bytes(predictions))
        raw_path.write_bytes(jsonl_bytes(raw_rows))
        input_path = fixture_dir / inputs_name
        report = {
            "run_id": f"mock-{collection}-run",
            "experiment_id": "E00.1-mock-fixture",
            "backend": "mock",
            "execution_mode": "mock_fixture",
            "hardware_label": "cpu-test-fixture",
            "timing_scope": "synthetic test fixture",
            "test_file_sha256": sha256_file(input_path),
            "submission_sha256": sha256_file(submission_path),
            "raw_predictions_sha256": sha256_file(raw_path),
            "samples": len(predictions),
            "submission_rows": len(predictions),
            "model_id": config["model"]["model_id"],
            "model_revision": None,
            "model_weight_identity": {"identity_status": "unverified", "fingerprint_status": "mock"},
            "generation_config": copy.deepcopy(config["generation"]),
            "budget": {"P_max_prompt_tokens": config["protocol"]["max_prompt_tokens"]},
            "protocol": {
                "prompt": {"sha256": config["protocol"]["prompt_sha256"]},
                "tokenizer": {
                    "signature_sha256": config["protocol"]["tokenizer_signature_sha256"],
                    "template_sha256": config["protocol"]["chat_template_sha256"],
                },
                "history_strategy": "T1_recent_contiguous_complete_message_suffix",
                "completion_field_order": ["emotion_label", "user_profile", "response_text", "memory_refs"],
                "id_in_prompt": False,
            },
        }
        report_path = fixture_dir / f"{collection}-performance_report.json"
        _write_json(report_path, report)

    return config, hashlib.sha256(config_bytes).hexdigest(), fixture_dir


def _score_fixture(config: dict, config_sha: str, fixture_dir: Path, collection: str, output_name: str, **kwargs):
    suffix = "" if collection == "full" else "smoke_"
    return score_files(
        config=config,
        config_sha256=config_sha,
        references_path=fixture_dir / f"{suffix}references.jsonl",
        submission_path=kwargs.pop("submission_path", fixture_dir / f"{collection}-submission.jsonl"),
        raw_path=kwargs.pop("raw_path", fixture_dir / f"{collection}-raw.jsonl"),
        run_metadata_path=kwargs.pop("run_metadata_path", fixture_dir / f"{collection}-performance_report.json"),
        output_dir=fixture_dir / output_name,
        **kwargs,
    )


def test_d03_shape_smoke_scores_with_auto_selected_id_map_and_full_set_stays_exact(tmp_path):
    config, config_sha, fixture_dir = _build_d03_shaped_fixture(tmp_path)

    smoke = _score_fixture(config, config_sha, fixture_dir, "smoke", "scored-smoke")
    full = _score_fixture(config, config_sha, fixture_dir, "full", "scored-full")

    assert smoke["sample_count"] == 3
    assert smoke["dataset_scope"] == "D0.3_dev"
    assert smoke["execution_mode"] == "mock_fixture"
    assert "not model quality evidence" in smoke["limitations"][-1]
    assert smoke["model_identity_verification"]["status"] == "not_applicable_mock"
    assert smoke["simple_majority_emotion_baseline"] == {
        "status": "not_run",
        "reason": "D0.3 Train file missing",
        "emotion_label": None,
        "accuracy": None,
    }
    assert full["sample_count"] == 6
    assert full["simple_majority_emotion_baseline"]["accuracy"] is None


def test_mock_smoke_builder_only_needs_target_free_input_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(e00_mock_smoke, "ROOT", tmp_path)
    input_path = tmp_path / "smoke_input.jsonl"
    input_path.write_bytes(jsonl_bytes([
        {"id": "test_000001_t0", "history": [{"role": "user", "content": "消息一"}]},
        {"id": "test_000002_t0", "history": [{"role": "user", "content": "消息二"}]},
    ]))
    output_dir = tmp_path / "outputs/evaluation/mock-smoke"
    report = e00_mock_smoke.build_mock_run(
        input_path,
        output_dir,
        "test-mock-run",
        ROOT / "configs/evaluation/e00.yaml",
    )

    submission = [json.loads(line) for line in (output_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in submission] == ["test_000001_t0", "test_000002_t0"]
    assert report["execution_mode"] == "mock_fixture"
    assert report["test_file_sha256"] == sha256_file(input_path)
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        e00_mock_smoke.build_mock_run(
            input_path,
            output_dir,
            "test-mock-run-2",
            ROOT / "configs/evaluation/e00.yaml",
        )


def test_smoke_scoring_rejects_missing_extra_and_wrong_id_map(tmp_path):
    config, config_sha, fixture_dir = _build_d03_shaped_fixture(tmp_path)
    original_submission = (fixture_dir / "smoke-submission.jsonl").read_bytes()
    rows = [json.loads(line) for line in original_submission.decode("utf-8").splitlines()]

    (fixture_dir / "smoke-submission.jsonl").write_bytes(jsonl_bytes(rows[:-1]))
    report_path = fixture_dir / "smoke-performance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["submission_sha256"] = sha256_file(fixture_dir / "smoke-submission.jsonl")
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="Submission/Reference ID 集合不一致"):
        _score_fixture(config, config_sha, fixture_dir, "smoke", "missing-row")

    full_inputs = [json.loads(line) for line in (fixture_dir / "inference_input.jsonl").read_text(encoding="utf-8").splitlines()]
    smoke_ids = set(json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))["smoke_selection"]["eval_ids"])
    extra_id = next(row["id"] for row in full_inputs if row["id"] not in smoke_ids)
    rows.append({"id": extra_id, "response_text": "多余", "emotion_label": "joy", "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []})
    (fixture_dir / "smoke-submission.jsonl").write_bytes(jsonl_bytes(rows))
    with pytest.raises(ValueError, match="Submission/Reference ID 集合不一致"):
        _score_fixture(config, config_sha, fixture_dir, "smoke", "extra-row")

    (fixture_dir / "smoke-submission.jsonl").write_bytes(original_submission)
    with pytest.raises(ValueError, match="ID map/Reference ID 集合不一致"):
        _score_fixture(
            config, config_sha, fixture_dir, "smoke", "wrong-map",
            id_map_path=fixture_dir / "id_map.jsonl",
        )


def test_smoke_scoring_rejects_manifest_membership_and_changed_adapter_conditions(tmp_path):
    config, config_sha, fixture_dir = _build_d03_shaped_fixture(tmp_path)
    manifest_path = fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["smoke_selection"]["eval_ids"] = manifest["smoke_selection"]["eval_ids"][:-1]
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Smoke References ID 集合与 Manifest"):
        _score_fixture(config, config_sha, fixture_dir, "smoke", "bad-manifest")

    # Restore the selection from the immutable references so this next check
    # isolates a changed adapter seed rather than the previous manifest error.
    references = [json.loads(line) for line in (fixture_dir / "smoke_references.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest["smoke_selection"]["eval_ids"] = [row["eval_id"] for row in references]
    _write_json(manifest_path, manifest)
    changed_config = copy.deepcopy(config)
    changed_config["smoke"]["seed"] += 1
    with pytest.raises(ValueError, match="适配条件与当前配置不一致"):
        _score_fixture(changed_config, config_sha, fixture_dir, "smoke", "changed-seed")


def test_scoring_config_can_change_without_rebuilding_dev_but_stale_runtime_is_rejected(tmp_path):
    config, config_sha, fixture_dir = _build_d03_shaped_fixture(tmp_path)
    changed = copy.deepcopy(config)
    changed["generation"]["do_sample"] = True
    changed["quantization"]["planned"]["load_in_4bit"] = False
    changed["scoring"]["bert_score"]["enabled"] = True
    changed_sha = hashlib.sha256(b"changed scoring and runtime candidates").hexdigest()
    report_path = fixture_dir / "smoke-performance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["generation_config"] = copy.deepcopy(changed["generation"])
    _write_json(report_path, report)

    scored = _score_fixture(changed, changed_sha, fixture_dir, "smoke", "changed-runtime")
    assert scored["sample_count"] == 3
    assert scored["response"]["bert_score"]["status"] == "not_run"
    assert scored["hashes"]["evaluation_config_sha256"] == changed_sha

    report["generation_config"] = copy.deepcopy(config["generation"])
    stale_runtime_report = copy.deepcopy(report)
    stale_runtime_report["backend"] = "openai_compatible"
    stale_runtime_report.pop("execution_mode", None)
    stale_runtime_report["run_id"] = "real-backend-stale-runtime"
    stale_runtime_path = fixture_dir / "real-backend-stale-report.json"
    _write_json(stale_runtime_path, stale_runtime_report)
    with pytest.raises(ValueError, match="generation_settings"):
        _score_fixture(
            changed, changed_sha, fixture_dir, "smoke", "stale-runtime",
            run_metadata_path=stale_runtime_path,
        )


def test_full_dev_rejects_short_reference_set_even_when_artifact_hashes_match(tmp_path):
    config, config_sha, fixture_dir = _build_d03_shaped_fixture(tmp_path)
    manifest_path = fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in ("references.jsonl", "id_map.jsonl", "inference_input.jsonl"):
        path = fixture_dir / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_bytes(jsonl_bytes(rows[:-1]))
        manifest["outputs"][filename]["sha256"] = sha256_file(path)
        manifest["outputs"][filename]["rows"] = len(rows) - 1
    _write_json(manifest_path, manifest)
    for filename in ("full-submission.jsonl", "full-raw.jsonl"):
        path = fixture_dir / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_bytes(jsonl_bytes(rows[:-1]))
    report_path = fixture_dir / "full-performance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["test_file_sha256"] = sha256_file(fixture_dir / "inference_input.jsonl")
    report["submission_sha256"] = sha256_file(fixture_dir / "full-submission.jsonl")
    report["raw_predictions_sha256"] = sha256_file(fixture_dir / "full-raw.jsonl")
    report["samples"] = 5
    report["submission_rows"] = 5
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="完整 D0.3 Dev 数量必须为 6"):
        _score_fixture(config, config_sha, fixture_dir, "full", "short-full")


def test_train_majority_is_null_when_missing_computed_from_train_or_rejects_bad_hash(tmp_path):
    labels = schema_enums(load_submission_schema(ROOT))["emotion_label"]
    missing_config = {"data": {"train_path": str(tmp_path / "missing.jsonl")}}
    missing = _load_train_majority(missing_config, tmp_path / "unused.json", labels)
    assert missing["status"] == "not_run"
    assert missing["emotion_label"] is None
    assert missing["accuracy"] is None

    train_path = tmp_path / "train.jsonl"
    train_path.write_bytes(jsonl_bytes([
        {"target": {"emotion_label": labels[0]}},
        {"target": {"emotion_label": labels[0]}},
        {"target": {"emotion_label": labels[1]}},
    ]))
    d03_manifest = {
        "outputs": [{"path": "data/processed/v1/train.jsonl", "sha256": sha256_file(train_path), "records": 3}]
    }
    d03_manifest_path = tmp_path / "train-manifest.json"
    _write_json(d03_manifest_path, d03_manifest)
    config = {"data": {"train_path": str(train_path)}}
    computed = _load_train_majority(config, d03_manifest_path, labels)
    assert computed["status"] == "computed"
    assert computed["emotion_label"] == labels[0]
    assert computed["train_sample_count"] == 3

    train_path.write_bytes(jsonl_bytes([{"target": {"emotion_label": labels[1]}}]))
    with pytest.raises(ValueError, match="Train SHA-256"):
        _load_train_majority(config, d03_manifest_path, labels)
