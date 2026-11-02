from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation import jsonl_bytes, load_submission_schema, schema_enums, sha256_file
from src.evaluation.metrics import (
    bert_score_optional,
    corpus_bleu4_char,
    emotion_metrics,
    profile_metrics,
    raw_diagnostics,
    response_metrics,
    rouge_l_f1,
)
from src.evaluation.score import _model_identity_verification, _validate_runtime_compatibility, score_files
from participant.run_inference import _weight_identity


ROOT = Path(__file__).resolve().parents[1]


def _target(emotion="joy", *, profile=None, response="你好🙂", memory_refs=None):
    return {
        "emotion_label": emotion,
        "user_profile": profile or {"personality_traits": [], "interests": [], "style": []},
        "response_text": response,
        "memory_refs": memory_refs or [],
    }


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _build_fixture(tmp_path: Path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    config_path = ROOT / "configs/evaluation/e00.yaml"
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    inputs = [
        {"id": "test_000001_t0", "history": [{"role": "user", "content": "你好🙂"}]},
        {"id": "test_000002_t0", "history": [{"role": "user", "content": "我有点害怕。"}]},
        {"id": "test_000003_t0", "history": [{"role": "user", "content": "最近状态不错"}]},
    ]
    references = [
        {"eval_id": inputs[0]["id"], "sample_id": "fixture:a", "target": _target(response="你好🙂")},
        {"eval_id": inputs[1]["id"], "sample_id": "fixture:b", "target": _target("fear", profile={"personality_traits": [], "interests": ["music"], "style": []}, response="我会陪着你。", memory_refs=["mem_000001"])},
        {"eval_id": inputs[2]["id"], "sample_id": "fixture:c", "target": _target("joy", profile={"personality_traits": ["open"], "interests": [], "style": []}, response="那真不错。")},
    ]
    id_map = [
        {"eval_id": row["eval_id"], "sample_id": row["sample_id"], "source": {"split": "synthetic_fixture"}}
        for row in references
    ]
    submission = [
        {"id": inputs[0]["id"], "response_text": "你好🙂", "emotion_label": "joy", "user_profile": {"personality_traits": [], "interests": [], "style": []}, "memory_refs": []},
        {"id": inputs[1]["id"], "response_text": "我会陪着你。", "emotion_label": "fear", "user_profile": {"personality_traits": [], "interests": ["music"], "style": []}, "memory_refs": []},
        {"id": inputs[2]["id"], "response_text": "那真不错。", "emotion_label": "", "user_profile": {"personality_traits": ["open"], "interests": [], "style": []}, "memory_refs": []},
    ]
    valid_raw = json.dumps({
        "emotion_label": "joy",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "response_text": "你好🙂",
        "memory_refs": [],
    }, ensure_ascii=False)
    fenced_raw = "```json\n" + json.dumps({
        "emotion_label": "fear",
        "user_profile": {"personality_traits": [], "interests": ["music"], "style": []},
        "response_text": "我会陪着你。",
        "memory_refs": [],
    }, ensure_ascii=False) + "\n```"
    trailing_text_raw = json.dumps({
        "emotion_label": "joy",
        "user_profile": {"personality_traits": ["open"], "interests": [], "style": []},
        "response_text": "那真不错。",
        "memory_refs": [],
    }, ensure_ascii=False) + " 以上为结果。"
    raw_rows = [
        {"id": inputs[0]["id"], "raw_text": valid_raw, "status": "success", "prompt_tokens": 9, "generated_tokens": 12, "stop_reason": "eos", "latency_ms": 10.0, "removed_history_messages": 0, "backend": "mock", "error_type": None},
        {"id": inputs[1]["id"], "raw_text": fenced_raw, "status": "success", "prompt_tokens": 9, "generated_tokens": 13, "stop_reason": "eos", "latency_ms": 20.0, "removed_history_messages": 0, "backend": "mock", "error_type": None},
        {"id": inputs[2]["id"], "raw_text": trailing_text_raw, "status": "success", "prompt_tokens": 9, "generated_tokens": 14, "stop_reason": "max_new_tokens", "latency_ms": 30.0, "removed_history_messages": 0, "backend": "mock", "error_type": None},
    ]
    names_rows = {
        "inference_input.jsonl": inputs,
        "references.jsonl": references,
        "id_map.jsonl": id_map,
    }
    for name, rows in names_rows.items():
        _write(fixture_dir / name, jsonl_bytes(rows))
    _write(fixture_dir / "submission.jsonl", jsonl_bytes(submission))
    _write(fixture_dir / "raw_predictions.jsonl", jsonl_bytes(raw_rows))
    output_facts = {
        name: {"path": name, "sha256": sha256_file(fixture_dir / name), "rows": len(rows)}
        for name, rows in names_rows.items()
    }
    manifest = {
        "manifest_version": "synthetic-e00-fixture-v1",
        "experiment_id": "E00.1",
        "dataset_scope": "synthetic_fixture",
        "source": {"fixture": "tests only"},
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "outputs": output_facts,
    }
    (fixture_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    run_metadata = {
        "run_id": "synthetic-e00-1",
        "experiment_id": "E00.1",
        "backend": "mock",
        "hardware_label": "synthetic-fixture",
        "timing_scope": "mock only",
        "test_file_sha256": sha256_file(fixture_dir / "inference_input.jsonl"),
        "submission_sha256": sha256_file(fixture_dir / "submission.jsonl"),
        "raw_predictions_sha256": sha256_file(fixture_dir / "raw_predictions.jsonl"),
        "samples": len(inputs),
        "submission_rows": len(inputs),
        "rounds": 3,
        "complete": False,
        "latencies_ms": [10.0, 20.0, 30.0],
        "average_latency_ms": 20.0,
        "median_latency_ms": 20.0,
        "p95_latency_ms": 30.0,
        "model_id": "mock-model",
        "model_revision": "fixture-only",
        "model_weight_identity": {"fingerprint_status": "mock"},
        "quantization": {"applied": False},
        "generation_config": {"max_new_tokens": 1200},
        "protocol": {"prompt": {"sha256": "mock"}},
    }
    (fixture_dir / "performance_report.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    return config, hashlib.sha256(config_bytes).hexdigest(), fixture_dir, references, submission, raw_rows


def test_emotion_metrics_keep_invalid_predictions_in_fixed_denominator():
    labels = schema_enums(load_submission_schema(ROOT))["emotion_label"]
    gold = list(labels)
    predicted = list(labels)
    predicted[1] = ""
    metric = emotion_metrics(gold, predicted, labels)
    assert len(labels) == 16
    assert metric["sample_count"] == 16
    assert metric["accuracy"] == pytest.approx(15 / 16)
    assert metric["invalid_empty_prediction_count"] == 1
    assert metric["per_label"][labels[1]]["support"] == 1
    assert metric["per_label"][labels[1]]["fn"] == 1
    assert metric["diagnostic_confusion_matrix"]["labels"][-1] == "__invalid_or_empty_prediction__"
    assert metric["macro_f1"] == pytest.approx(15 / 16)


def test_profile_metrics_cover_exact_empty_partial_and_fixed_enum_macro():
    enums = schema_enums(load_submission_schema(ROOT))
    gold = [
        {"personality_traits": [], "interests": [], "style": []},
        {"personality_traits": ["open"], "interests": ["music", "games"], "style": []},
    ]
    predicted = [
        {"personality_traits": [], "interests": [], "style": []},
        {"personality_traits": ["open"], "interests": ["music"], "style": ["brief"]},
    ]
    metric = profile_metrics(gold, predicted, enums)
    interests = metric["per_dimension"]["interests"]
    assert interests["local_exact_match"] == 0.5
    assert interests["tp"] == 1 and interests["fn"] == 1 and interests["fp"] == 0
    assert interests["gold_empty_rate"] == 0.5
    assert interests["pred_empty_rate"] == 0.5
    assert metric["all_three_dimensions_exact_match"] == 0.5
    assert metric["overall_33_label_macro_f1"] == pytest.approx(
        sum(metric["per_dimension"][dimension]["per_label"][label]["f1"]
            for dimension in ("personality_traits", "interests", "style")
            for label in enums[dimension]) / 33
    )


def test_character_rouge_l_unicode_whitespace_and_empty_prediction():
    assert rouge_l_f1("你 好🙂", "你好🙂") == 1.0
    assert rouge_l_f1("ABC", "AC") == pytest.approx(0.8)
    assert rouge_l_f1("参考", "") == 0.0
    assert rouge_l_f1("", "") == 0.0


def test_corpus_bleu_uses_pinned_sacrebleu_character_tokenizer_and_keeps_empty_rows():
    metric = corpus_bleu4_char(["你好🙂", "参考"], ["你好🙂", ""])
    assert metric["implementation"] == "sacrebleu"
    assert metric["version"] == "2.5.1"
    assert metric["tokenize"] == "char"
    assert metric["effective_order"] is True
    assert metric["sample_count"] == 2
    assert metric["empty_prediction_count"] == 1
    assert metric["score_percent"] == pytest.approx(51.341711903259224)


def test_bert_score_not_run_is_explicit_and_stub_records_metadata(tmp_path):
    config = {"enabled": False, "local_model_path": None, "model_id": None, "revision": None, "language": "zh"}
    result = bert_score_optional(["ref"], ["pred"], config, workspace_root=tmp_path)
    assert result["status"] == "not_run"
    assert result["reason"] == "disabled_by_config"
    assert result["f1"] is None

    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    config.update({"enabled": True, "local_model_path": str(model_dir), "model_id": "local-zh", "revision": "a" * 40})
    called = []

    def stub(predictions, references):
        called.append((predictions, references))
        return [0.7], [0.8], [0.75]

    result = bert_score_optional(["ref"], ["pred"], config, scorer=stub, workspace_root=tmp_path)
    assert called == [(["pred"], ["ref"])]
    assert result["status"] == "completed"
    assert result["model_revision"] == "a" * 40
    assert result["f1"] == pytest.approx(0.75)


def test_runtime_fingerprint_blocks_protocol_or_generation_mismatch():
    config = yaml.safe_load((ROOT / "configs/evaluation/e00.yaml").read_text(encoding="utf-8"))
    run = {
        "model_id": config["model"]["model_id"],
        "backend": "openai_compatible",
        "generation_config": dict(config["generation"]),
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
        "budget": {"P_max_prompt_tokens": config["protocol"]["max_prompt_tokens"]},
    }
    result = _validate_runtime_compatibility(run, config, "D0.3_dev")
    assert result["status"] == "verified"
    run["generation_config"]["do_sample"] = True
    with pytest.raises(ValueError, match="generation_settings"):
        _validate_runtime_compatibility(run, config, "D0.3_dev")


def test_transformers_runtime_requires_recomputed_weight_file_hashes(tmp_path):
    config = yaml.safe_load((ROOT / "configs/evaluation/e00.yaml").read_text(encoding="utf-8"))
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"tiny weight fixture")
    identity = _weight_identity(tmp_path, {"model_id": config["model"]["model_id"], "model_revision": "c" * 40})
    run = {
        "model_id": config["model"]["model_id"],
        "backend": "transformers",
        "model_weight_identity": identity,
        "quantization": {
            "applied": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "use_double_quant": config["quantization"]["planned"]["use_double_quant"],
            "real_nf4_verified": True,
            "linear4bit_module_count": 2,
            "device_map_observed": True,
            "placement_verified": True,
            "offload_devices": [],
        },
        "generation_config": dict(config["generation"]),
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
        "budget": {"P_max_prompt_tokens": config["protocol"]["max_prompt_tokens"]},
    }

    assert _validate_runtime_compatibility(run, config, "D0.3_dev")["status"] == "verified"
    assert _model_identity_verification(run, "D0.3_dev")["status"] == "verified"
    assert _model_identity_verification(run, "D0.3_dev")["revision_status"] == "unverified_revision"

    run["quantization"]["real_nf4_verified"] = False
    with pytest.raises(ValueError, match="transformers_nf4_applied"):
        _validate_runtime_compatibility(run, config, "D0.3_dev")
    run["quantization"]["real_nf4_verified"] = True
    run["quantization"]["placement_verified"] = False
    with pytest.raises(ValueError, match="transformers_nf4_applied"):
        _validate_runtime_compatibility(run, config, "D0.3_dev")
    run["quantization"]["placement_verified"] = True

    run["model_weight_identity"]["observed_weight_hash"] = "0" * 64
    with pytest.raises(ValueError, match="observed_model_weight_hash"):
        _validate_runtime_compatibility(run, config, "D0.3_dev")


def test_runtime_and_score_identity_bind_adapter_to_verified_base(tmp_path):
    config = yaml.safe_load((ROOT / "configs/evaluation/e00.yaml").read_text(encoding="utf-8"))
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"tiny weight fixture")
    base_identity = _weight_identity(tmp_path, {"model_id": config["model"]["model_id"]})
    adapter_identity = {
        "enabled": True,
        "status": "verified_local_adapter",
        "base_model_id": config["model"]["model_id"],
        "base_weight_sha256": base_identity["observed_weight_hash"],
        "weight_sha256": "b" * 64,
        "config_sha256": "c" * 64,
    }
    config["model"]["adapter"] = {
        "enabled": True,
        "expected_base_weight_sha256": base_identity["observed_weight_hash"],
        "expected_adapter_weight_sha256": "b" * 64,
        "expected_adapter_config_sha256": "c" * 64,
    }
    run = {
        "model_id": config["model"]["model_id"],
        "backend": "transformers",
        "model_weight_identity": base_identity,
        "adapter_identity": adapter_identity,
        "quantization": {
            "applied": True, "quant_type": "nf4", "compute_dtype": "bfloat16",
            "use_double_quant": config["quantization"]["planned"]["use_double_quant"],
            "real_nf4_verified": True, "linear4bit_module_count": 1,
            "device_map_observed": True, "placement_verified": True, "offload_devices": [],
        },
        "generation_config": dict(config["generation"]),
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
        "budget": {"P_max_prompt_tokens": config["protocol"]["max_prompt_tokens"]},
    }
    assert _validate_runtime_compatibility(run, config, "D0.3_dev")["status"] == "verified"
    identity = _model_identity_verification(run, "D0.3_dev", config)
    assert identity["status"] == "verified_base_and_adapter"
    assert identity["adapter_weight_sha256"] == "b" * 64

    run["adapter_identity"]["base_weight_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter_base_hash_matches"):
        _validate_runtime_compatibility(run, config, "D0.3_dev")


def test_raw_diagnostics_separate_strict_json_from_runner_repair():
    enums = schema_enums(load_submission_schema(ROOT))
    good = json.dumps({
        "emotion_label": "joy",
        "user_profile": {"personality_traits": [], "interests": [], "style": []},
        "response_text": "可以的。",
        "memory_refs": [],
    }, ensure_ascii=False)
    raw = [
        {"raw_text": good, "status": "success", "stop_reason": "eos", "generated_tokens": 5},
        {"raw_text": f"```json\n{good}\n```", "status": "success", "stop_reason": "eos", "generated_tokens": 5},
        {"raw_text": good + " explanation", "status": "parse_failure", "stop_reason": "max_new_tokens", "generated_tokens": 10},
        {"raw_text": None, "status": "generation_failure", "stop_reason": "not_generated", "generated_tokens": 0},
    ]
    submissions = [{"id": f"test_{index:06d}_t0"} for index in range(1, 5)]
    result = raw_diagnostics(raw, enums, submissions, [True] * 4, generation_cap=10)
    assert result["raw_json_parse_rate"] == 0.25
    assert result["raw_schema_valid_rate"] == 0.25
    assert result["valid_emotion_rate"] == 0.25
    assert result["valid_profile_rate"] == 0.25
    assert result["nonempty_response_rate"] == 0.25
    assert result["submission_schema_valid_rate"] == 1.0
    assert result["error_case_counts"]["strict_json_parse_failure"] == 3
    assert result["error_case_counts"]["reached_generation_token_cap"] == 1

    duplicate_key = good.replace('"emotion_label": "joy"', '"emotion_label": "joy", "emotion_label": "fear"')
    duplicate = raw_diagnostics(
        [{"raw_text": duplicate_key, "status": "success", "stop_reason": "unknown", "generated_tokens": 1}],
        enums,
        [{"id": "test_000001_t0"}],
        [True],
        generation_cap=10,
    )
    assert duplicate["raw_json_parse_rate"] == 0.0
    assert duplicate["raw_schema_valid_rate"] == 0.0


def test_score_cli_core_aligns_by_id_and_records_known_fixture_values(tmp_path):
    config, config_sha, fixture_dir, references, submission, _raw = _build_fixture(tmp_path)
    # The files are deliberately in reverse order; ID joins, not row position, define alignment.
    submissions_path = fixture_dir / "submission.jsonl"
    reordered = list(reversed([json.loads(line) for line in submissions_path.read_text(encoding="utf-8").splitlines()]))
    submissions_path.write_bytes(jsonl_bytes(reordered))
    report_path = fixture_dir / "performance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["submission_sha256"] = sha256_file(submissions_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    result = score_files(
        config=config,
        config_sha256=config_sha,
        references_path=fixture_dir / "references.jsonl",
        submission_path=submissions_path,
        raw_path=fixture_dir / "raw_predictions.jsonl",
        output_dir=fixture_dir / "scored",
    )
    assert result["dataset_scope"] == "synthetic_fixture"
    assert result["execution_mode"] == "mock_fixture"
    assert result["model_identity_verification"]["status"] == "not_applicable_synthetic_fixture"
    assert result["sample_count"] == 3
    assert result["emotion"]["accuracy"] == pytest.approx(2 / 3)
    assert result["emotion"]["invalid_empty_prediction_count"] == 1
    assert result["profile"]["per_dimension"]["interests"]["local_exact_match"] == pytest.approx(1.0)
    assert result["raw_diagnostics"]["raw_json_parse_rate"] == pytest.approx(1 / 3)
    assert result["raw_diagnostics"]["raw_schema_valid_rate"] == pytest.approx(1 / 3)
    assert result["raw_diagnostics"]["submission_schema_valid_rate"] == 1.0
    assert result["response"]["bert_score"]["status"] == "not_run"
    assert result["response"]["bert_score"]["f1"] is None
    assert result["performance"]["all_scored_samples_average_latency_ms"] == 20.0
    assert (fixture_dir / "scored/metrics.json").is_file()
    assert (fixture_dir / "scored/report.md").is_file()
    assert len((fixture_dir / "scored/case_errors.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_score_rejects_missing_predictions_duplicates_raw_leak_and_bad_schema(tmp_path):
    config, config_sha, fixture_dir, _references, _submission, _raw = _build_fixture(tmp_path)
    submission_path = fixture_dir / "submission.jsonl"
    original = submission_path.read_bytes()
    rows = [json.loads(line) for line in original.decode().splitlines()]

    submission_path.write_bytes(jsonl_bytes(rows[:-1]))
    with pytest.raises(ValueError, match="ID 集合不一致"):
        score_files(config=config, config_sha256=config_sha, references_path=fixture_dir / "references.jsonl", submission_path=submission_path, raw_path=None, output_dir=fixture_dir / "bad1")

    submission_path.write_bytes(jsonl_bytes([rows[0], rows[0], rows[2]]))
    with pytest.raises(ValueError, match="重复 id"):
        score_files(config=config, config_sha256=config_sha, references_path=fixture_dir / "references.jsonl", submission_path=submission_path, raw_path=None, output_dir=fixture_dir / "bad2")
    submission_path.write_bytes(original)

    raw_path = fixture_dir / "raw_predictions.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    raw_rows[0]["target"] = {"leak": True}
    raw_path.write_bytes(jsonl_bytes(raw_rows))
    with pytest.raises(ValueError, match="字段不符合 allowlist"):
        score_files(config=config, config_sha256=config_sha, references_path=fixture_dir / "references.jsonl", submission_path=submission_path, raw_path=raw_path, output_dir=fixture_dir / "bad3")
    raw_rows[0].pop("target")
    raw_path.write_bytes(jsonl_bytes(raw_rows))

    invalid = copy.deepcopy(rows)
    invalid[0]["emotion_label"] = "not-a-label"
    submission_path.write_bytes(jsonl_bytes(invalid))
    report_path = fixture_dir / "performance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["submission_sha256"] = sha256_file(submission_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不符合官方 Schema"):
        score_files(config=config, config_sha256=config_sha, references_path=fixture_dir / "references.jsonl", submission_path=submission_path, raw_path=None, output_dir=fixture_dir / "bad4")


def test_score_rejects_input_hash_mismatch_before_computing_metrics(tmp_path):
    config, config_sha, fixture_dir, *_ = _build_fixture(tmp_path)
    report_path = fixture_dir / "performance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["test_file_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="输入 hash"):
        score_files(config=config, config_sha256=config_sha, references_path=fixture_dir / "references.jsonl", submission_path=fixture_dir / "submission.jsonl", raw_path=None, output_dir=fixture_dir / "bad")


def test_score_rejects_leaked_target_even_when_all_hashes_are_consistent(tmp_path):
    config, config_sha, fixture_dir, *_ = _build_fixture(tmp_path)
    input_path = fixture_dir / "inference_input.jsonl"
    inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    inputs[0]["target"] = {"emotion_label": "joy"}
    input_path.write_bytes(jsonl_bytes(inputs))
    manifest_path = fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["inference_input.jsonl"]["sha256"] = sha256_file(input_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    run_path = fixture_dir / "performance_report.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["test_file_sha256"] = sha256_file(input_path)
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="可能泄漏真值"):
        score_files(config=config, config_sha256=config_sha, references_path=fixture_dir / "references.jsonl", submission_path=fixture_dir / "submission.jsonl", raw_path=None, output_dir=fixture_dir / "bad")
