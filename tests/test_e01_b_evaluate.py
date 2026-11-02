"""CPU synthetic checks for complete, isolated and paired B evaluation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import e01_b_evaluate as evaluation
from src.evaluation import jsonl_bytes, sha256_file
from src.evaluation.prepare_dev import adapter_config_identity, source_identity_sha256


def _dataset_fixture(tmp_path, monkeypatch, count=1833):
    config = evaluation._config("full-dev")
    config["data"]["output_dir"] = str(tmp_path)
    ids = [f"test_{index:06d}_t0" for index in range(1, count + 1)]
    inputs = [{"id": value, "history": [{"role": "user", "content": "synthetic fixture"}]} for value in ids]
    maps = [{"eval_id": value, "sample_id": f"train:c{index}:pp0"} for index, value in enumerate(ids)]
    references = [{"eval_id": value, "sample_id": f"train:c{index}:pp0", "target": {}} for index, value in enumerate(ids)]
    for name, rows in (("inference_input.jsonl", inputs), ("id_map.jsonl", maps), ("references.jsonl", references)):
        (tmp_path / name).write_bytes(jsonl_bytes(rows))
    adapter, adapter_sha = adapter_config_identity(config)
    source = {"dev_rows": 1833, "dev_sha256": "synthetic-fixture-not-model-evidence"}
    manifest = {"dataset_scope": "D0.3_dev", "adapter_config": adapter, "adapter_config_sha256": adapter_sha, "source": source, "source_identity_sha256": source_identity_sha256(source), "outputs": {name: {"path": name, "sha256": sha256_file(tmp_path / name), "rows": count} for name in ("inference_input.jsonl", "references.jsonl", "id_map.jsonl")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    # Real D0.3 source verification is covered by the unchanged E00 scorer suite;
    # this fixture isolates B's additional membership and leakage checks.
    monkeypatch.setattr(evaluation, "_validate_d03_manifest", lambda *args: {"status": "verified"})
    return config, manifest


def test_full_dataset_rejects_partial_20_row_baseline(tmp_path, monkeypatch):
    config, _ = _dataset_fixture(tmp_path, monkeypatch, count=20)
    with pytest.raises(ValueError, match="exactly 1833"):
        evaluation._dataset(config, "full-dev")


def test_full_dataset_rejects_gold_leakage_and_wrong_id_map(tmp_path, monkeypatch):
    config, manifest = _dataset_fixture(tmp_path, monkeypatch)
    assert evaluation._dataset(config, "full-dev")["sample_count"] == 1833
    path = tmp_path / "inference_input.jsonl"
    rows = evaluation.read_jsonl(path)
    rows[0]["target"] = {"emotion_label": "anxiety"}
    path.write_bytes(jsonl_bytes(rows))
    manifest["outputs"][path.name]["sha256"] = sha256_file(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="输入|History|history|target|字段"):
        evaluation._dataset(config, "full-dev")
    rows[0].pop("target")
    path.write_bytes(jsonl_bytes(rows))
    manifest["outputs"][path.name]["sha256"] = sha256_file(path)
    maps = evaluation.read_jsonl(tmp_path / "id_map.jsonl")
    maps[0]["eval_id"] = "test_999999_t0"
    (tmp_path / "id_map.jsonl").write_bytes(jsonl_bytes(maps))
    manifest["outputs"]["id_map.jsonl"]["sha256"] = sha256_file(tmp_path / "id_map.jsonl")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ID map sets differ"):
        evaluation._dataset(config, "full-dev")


def test_training_run_rejects_wrong_adapter_and_pilot_as_full(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    root = tmp_path / "outputs/training/pilot"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3-1.7B"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter weights")
    (root / "config_snapshot.yaml").write_text("synthetic: config\n")
    identity, _ = evaluation.runner.adapter_weight_identity(adapter)
    manifest = {"run_id": root.name, "status": "passed", "experiment_id": "E01-B-PILOT", "training_scope": "pilot", "train_rows": 512, "init_adapter": None, "base_weights_unchanged": True, "base_weight_sha256": "b" * 64, "reload_check_status": "passed", "config_sha256": sha256_file(root / "config_snapshot.yaml"), "adapter_identity": identity}
    (root / "manifest.json").write_text(json.dumps(manifest))
    evaluation._training_run(root, "pilot-dev", "b" * 64)
    with pytest.raises(ValueError, match="matching B Run"):
        evaluation._training_run(root, "full-dev", "b" * 64)
    (adapter / "adapter_model.safetensors").write_bytes(b"another adapter")
    with pytest.raises(ValueError, match="Adapter weight/config"):
        evaluation._training_run(root, "pilot-dev", "b" * 64)


def test_full_dev_requires_separate_resource_permission_before_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="allow-full-dev"):
        evaluation.execute("full-dev", "base", "base-test", training_root=None, hardware_label="fixture", allow_full_dev=False)
    assert not (tmp_path / "outputs").exists()


def test_evaluation_records_progress_failure_and_same_run_resume(tmp_path, monkeypatch):
    config = evaluation._config("full-dev")
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "_config", lambda scope: copy.deepcopy(config))
    monkeypatch.setattr(evaluation, "_dataset", lambda cfg, scope: {"root": "synthetic", "scope": scope, "sample_count": 1833})
    monkeypatch.setattr(evaluation, "_snapshots", lambda *args: ({"fixture": True}, {"fixture": True}, None))
    monkeypatch.setattr(evaluation, "_code_identity", lambda: {"fixture": "cpu"})
    attempts = []
    def fake_runner(root, record):
        assert json.loads((root / "run_manifest.json").read_text())["status"] == "running"
        attempts.append(record["run_id"])
        if len(attempts) == 1:
            return 2
        (root / "runner").mkdir()
        for name in ("runner/submission.jsonl", "runner/performance_report.json", "raw_predictions.jsonl"):
            (root / name).write_text("{}\n")
        return 0
    def fake_score(root, record, output):
        output.mkdir()
        (output / "metrics.json").write_text("{}\n")
        return {}
    monkeypatch.setattr(evaluation, "_invoke_runner", fake_runner)
    monkeypatch.setattr(evaluation, "_score", fake_score)
    opts = dict(training_root=None, hardware_label="fixture", allow_full_dev=True)
    first = evaluation.execute("full-dev", "base", "base-test", **opts)
    assert first["status"] == "inference_failed"
    with pytest.raises(FileExistsError):
        evaluation.execute("full-dev", "base", "base-test", **opts)
    done = evaluation.execute("full-dev", "base", "base-test", resume=True, **opts)
    assert done["status"] == "scored"
    assert attempts == ["base-test", "base-test"]
    with pytest.raises(FileExistsError, match="scored"):
        evaluation.execute("full-dev", "base", "base-test", resume=True, **opts)


def test_compare_refuses_mismatched_scope_and_generation_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    output = tmp_path / "outputs/evaluation/comparison/new"
    records = {"base": {"dataset": {"hash": "full"}, "scope": "full-dev"}, "adapter": {"dataset": {"hash": "smoke"}, "scope": "pilot-dev"}}
    monkeypatch.setattr(evaluation, "_load_completed", lambda root, model: records[model])
    with pytest.raises(ValueError, match="identical complete Dev"):
        evaluation.compare(base, adapter, output)
    assert not output.exists()
    records["adapter"] = copy.deepcopy(records["base"])
    for root, sample in ((base, False), (adapter, True)):
        (root / "inference_config.json").write_text(json.dumps({"generation": {"do_sample": sample}}))
    with pytest.raises(ValueError, match="settings differ"):
        evaluation.compare(base, adapter, output)
    assert not output.exists()


def test_paired_accuracy_uses_same_id_conversation_clusters_and_fixed_seed():
    cases = [{"id": str(index), "base": {"emotion_correct": base}, "adapter": {"emotion_correct": adapter}} for index, (base, adapter) in enumerate(((False, True), (False, True), (True, True), (True, False)))]
    groups = {"0": "conversation-a", "1": "conversation-a", "2": "conversation-b", "3": "conversation-b"}
    stats = evaluation.paired_statistics(cases, groups, bootstrap_samples=200)
    assert stats == evaluation.paired_statistics(cases, groups, bootstrap_samples=200)
    assert stats["emotion_accuracy_delta"] == 0.25
    assert stats["cluster_count"] == 2
    assert stats["emotion_transitions"]["base_wrong_adapter_correct"] == 2
    assert stats["emotion_transitions"]["base_correct_adapter_wrong"] == 1


def test_runner_child_environment_clears_external_model_and_protocol_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATION_MODEL_PATH", "/wrong/model")
    monkeypatch.setenv("EVALUATION_MAX_PROMPT_TOKENS", "12")
    captured = {}
    class Process:
        stdout = ["synthetic runner progress\n"]
        def __init__(self, command, **kwargs):
            captured.update(kwargs["env"])
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(evaluation.subprocess, "Popen", Process)
    record = {"run_id": "synthetic-test", "dataset": {"root": str(tmp_path), "input": "input.jsonl"}}
    assert evaluation._invoke_runner(tmp_path, record) == 0
    assert "EVALUATION_MODEL_PATH" not in captured
    assert "EVALUATION_MAX_PROMPT_TOKENS" not in captured
    assert captured["EVALUATION_RESUME"] == "1"
    assert captured["HF_HUB_OFFLINE"] == "1"


def test_full_pair_runs_unchanged_scorer_on_all_1833_synthetic_ids(tmp_path, monkeypatch):
    """Fabricated runtime facts exercise validation only; no Qwen/GPU run occurs."""
    import hashlib
    import yaml
    from src.evaluation.prepare_dev import prepare_records
    official_root = evaluation.ROOT
    schema = evaluation.load_submission_schema(official_root)
    enums = evaluation.schema_enums(schema)
    config = evaluation._config("full-dev")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    config["data"]["output_dir"] = str(dataset_root)
    config["data"]["train_path"] = str(tmp_path / "missing-train.jsonl")
    rows = [{"sample_id": f"train:synthetic-{index}:pp0", "conversation_id": f"synthetic-{index}", "source": {"split": "train", "source_line": index + 1, "prediction_point_index": 0}, "history": [{"role": "user", "content": "合成测试"}], "target": {"emotion_label": enums["emotion_label"][index % 16], "user_profile": {dim: [] for dim in evaluation.DIMENSIONS}, "response_text": "合成回复", "memory_refs": []}} for index in range(1833)]
    source_bytes = jsonl_bytes(rows)
    schema_sha = sha256_file(official_root / "data/train/train_public.schema.json")
    d03_manifest = {"pipeline_version": "D0.3-v1", "inputs": {"train": {"schema_path": "data/train/train_public.schema.json", "schema_sha256": schema_sha}}, "outputs": [{"path": "data/processed/v1/dev.jsonl", "sha256": hashlib.sha256(source_bytes).hexdigest(), "records": 1833}]}
    d03_path = tmp_path / "d03-manifest.json"
    d03_path.write_text(json.dumps(d03_manifest))
    prepare_records(rows, enums=enums, d03_manifest=d03_manifest, d03_manifest_path=d03_path, source_sha256=hashlib.sha256(source_bytes).hexdigest(), config=config, config_sha256="f" * 64, output_dir=dataset_root, target_schema_facts={"path": "data/train/train_public.schema.json", "sha256": schema_sha})
    files = [{"path": "synthetic-model.safetensors", "size_bytes": 1, "sha256": "a" * 64}]
    weight_sha = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config["execution"]["expected_base_weight_sha256"] = weight_sha
    dataset = evaluation._dataset(config, "full-dev")
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "_code_identity", lambda: {"synthetic_runtime_fixture": True})
    monkeypatch.setattr(evaluation, "load_submission_schema", lambda root: schema)
    roots = {}
    references = evaluation.read_jsonl(dataset_root / "references.jsonl")
    for model in ("base", "adapter"):
        root = tmp_path / "outputs/evaluation/e01_b/full" / f"synthetic-{model}"
        roots[model] = root
        (root / "runner").mkdir(parents=True)
        cfg = copy.deepcopy(config)
        adapter = {"enabled": model == "adapter"}
        if model == "adapter":
            adapter.update(expected_base_weight_sha256=weight_sha, expected_adapter_weight_sha256="b" * 64, expected_adapter_config_sha256="c" * 64)
        cfg["model"]["adapter"] = adapter
        (root / "evaluation_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (root / "inference_config.json").write_text(json.dumps({"adapter": adapter, "generation": cfg["generation"], "expected_weight_sha256": weight_sha}))
        predictions = [{"id": ref["eval_id"], **ref["target"]} for ref in references]
        raw = [{"id": ref["eval_id"], "raw_text": json.dumps(ref["target"], ensure_ascii=False), "status": "success", "prompt_tokens": 10, "generated_tokens": 20, "stop_reason": "eos", "latency_ms": 1.0, "removed_history_messages": 0, "backend": "transformers", "error_type": None} for ref in references]
        if model == "base":
            predictions[0]["emotion_label"] = ""
        (root / "runner/submission.jsonl").write_bytes(jsonl_bytes(predictions))
        (root / "raw_predictions.jsonl").write_bytes(jsonl_bytes(raw))
        observed_adapter = {"enabled": False} if model == "base" else {"enabled": True, "base_weight_sha256": weight_sha, "weight_sha256": "b" * 64, "config_sha256": "c" * 64}
        report = {"run_id": root.name, "experiment_id": "synthetic-protocol-test", "backend": "transformers", "execution_mode": "model_inference", "samples": 1833, "test_file_sha256": sha256_file(dataset_root / "inference_input.jsonl"), "submission_sha256": sha256_file(root / "runner/submission.jsonl"), "raw_predictions_sha256": sha256_file(root / "raw_predictions.jsonl"), "model_id": cfg["model"]["model_id"], "model_revision": cfg["model"]["weight_revision"], "model_weight_identity": {"identity_status": "observed", "observed_weight_hash": weight_sha, "files": files}, "adapter_identity": observed_adapter, "generation_config": cfg["generation"], "budget": {"P_max_prompt_tokens": 1280}, "quantization": {"applied": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "use_double_quant": True, "real_nf4_verified": True, "linear4bit_module_count": 1, "device_map_observed": True, "placement_verified": True, "offload_devices": []}, "protocol": {"prompt": {"sha256": cfg["protocol"]["prompt_sha256"]}, "tokenizer": {"signature_sha256": cfg["protocol"]["tokenizer_signature_sha256"], "template_sha256": cfg["protocol"]["chat_template_sha256"]}, "history_strategy": "T1_recent_contiguous_complete_message_suffix", "completion_field_order": ["emotion_label", "user_profile", "response_text", "memory_refs"], "id_in_prompt": False}}
        (root / "runner/performance_report.json").write_text(json.dumps(report))
        record = {"status": "scored", "run_id": root.name, "model": model, "scope": "full-dev", "dataset": dataset, "source_code": evaluation._code_identity(), "inference_config_sha256": sha256_file(root / "inference_config.json"), "evaluation_config_sha256": sha256_file(root / "evaluation_config.yaml")}
        evaluation._score(root, record, root / "scored")
        record["artifacts"] = {name: sha256_file(root / name) for name in ("runner/submission.jsonl", "runner/performance_report.json", "raw_predictions.jsonl", "scored/metrics.json")}
        (root / "run_manifest.json").write_text(json.dumps(record))
    result = evaluation.compare(roots["base"], roots["adapter"], tmp_path / "outputs/evaluation/comparison/synthetic")
    summary = json.loads((Path(result["output_dir"]) / "comparison_summary.json").read_text())
    assert summary["paired"]["sample_count"] == 1833
    assert summary["paired"]["emotion_transitions"]["base_wrong_adapter_correct"] == 1
    assert summary["paired"]["emotion_accuracy_delta"] == pytest.approx(1 / 1833)
    assert summary["metrics"]["base"]["model_identity_verification"]["status"] == "verified"
    assert summary["metrics"]["adapter"]["model_identity_verification"]["status"] == "verified_base_and_adapter"
    assert summary["official_performance"] == summary["bert_score"] == "not_run"
