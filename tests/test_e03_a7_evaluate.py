import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from scripts import e03_a7_evaluate
from scripts.e03_a7_evaluate import _pair_report, build_parser
from src.data.e03_a5_labels import sha256_file
from src.evaluation.e03_a7_classification_metrics import (
    average_precision_by_label,
    choose_calibration_threshold,
    compare_arm_profile_predictions,
    profile_from_probabilities,
    summarize_profile_pair,
)
from src.evaluation.profile_reconcile import index_profile_prediction_rows


def _enums():
    return {
        "personality_traits": [f"p{i}" for i in range(10)],
        "interests": [f"i{i}" for i in range(12)],
        "style": [f"s{i}" for i in range(11)],
    }


def test_a7_evaluation_sources_use_frozen_a5_asset_keys(monkeypatch):
    input_path = Path("frozen-input.jsonl")
    id_map_path = Path("frozen-id-map.jsonl")
    assets = {
        "source_hashes": {"frozen_input": "input-sha", "references": "references-sha", "id_map": "id-map-sha"},
        "frozen_ids": ["eval-1"],
        "source_paths": {"frozen_input": input_path, "references": Path("references.jsonl"), "id_map": id_map_path},
    }
    config = {
        "identity": {
            "frozen_dev_input_sha256": "input-sha",
            "frozen_dev_references_sha256": "references-sha",
            "frozen_dev_id_map_sha256": "id-map-sha",
            "frozen_dev_ids_sha256": "ids-sha",
        },
        "data": {"expected_dev_rows": 1},
    }

    monkeypatch.setattr(e03_a7_evaluate.e03_a5_evaluate, "_load_eval_assets", lambda _config: assets)
    monkeypatch.setattr(e03_a7_evaluate, "selected_id_sha256", lambda _ids: "ids-sha")

    def fake_iter_jsonl(path):
        if path == input_path:
            yield 1, {"id": "eval-1"}, "line-sha"
        elif path == id_map_path:
            yield 1, {"eval_id": "eval-1", "conversation_id": "conversation-1"}, "line-sha"
        else:
            raise AssertionError(f"unexpected frozen Dev asset path: {path}")

    monkeypatch.setattr(e03_a7_evaluate, "iter_jsonl", fake_iter_jsonl)
    monkeypatch.setattr(e03_a7_evaluate, "validate_target_free_inputs", lambda rows: {row["id"]: row for row in rows})

    loaded = e03_a7_evaluate._load_evaluation_sources(config, {})

    assert loaded["input_index"]["eval-1"] == {"id": "eval-1"}
    assert loaded["id_map"]["eval-1"]["conversation_id"] == "conversation-1"


def test_a7_baseline_reconciliation_enum_includes_emotion_label():
    enums = _enums()
    prediction = {
        "id": "eval-1",
        "status": "valid",
        "emotion_label": "neutral",
        "user_profile": {"personality_traits": ["p0"], "interests": [], "style": []},
        "reason": "valid",
    }

    indexed, statuses = index_profile_prediction_rows(
        [prediction], ["eval-1"],
        e03_a7_evaluate._baseline_reconciliation_enums(enums, ["neutral", "sad"]),
        require_all_valid=True,
    )

    assert indexed["eval-1"] == prediction
    assert statuses["valid"] == 1


def test_a7_scoring_records_gold_access_before_reading_references(monkeypatch, tmp_path):
    eval_dir = tmp_path / "step-0512"
    eval_dir.mkdir()
    predictions_path = eval_dir / "predictions.jsonl"
    predictions_path.write_text('{"id":"eval-1"}\n', encoding="utf-8")
    manifest_path = eval_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps({
        "status": "dev_predictions_frozen_gold_not_read",
        "gold_read": False,
        "predictions_sha256": sha256_file(predictions_path),
    }), encoding="utf-8")
    references_path = Path("frozen-references.jsonl")

    def assert_lock_then_fail(path):
        assert path == references_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "scoring_started_after_prediction_lock"
        assert manifest["gold_read"] is True
        assert manifest["gold_read_after_predictions_locked"] is True
        raise RuntimeError("stop after verifying scoring audit ordering")

    monkeypatch.setattr(e03_a7_evaluate, "iter_jsonl", assert_lock_then_fail)
    with pytest.raises(RuntimeError, match="audit ordering"):
        e03_a7_evaluate._score_after_prediction_freeze(
            config={}, arm="P", step=512, eval_dir=eval_dir, eval_manifest_path=manifest_path,
            prediction_rows=[], calibration={}, sources={}, a5_config={}, supervision={}, splits={},
            evaluation_assets={"source_paths": {"references": references_path}},
            start_time=0.0, training_code_sha256="training-code",
        )


def test_a7_eval_checkpoint_reload_restores_fp32_lora_dtype(monkeypatch, tmp_path):
    calls = []
    run_id = "pair-P"
    run_root = tmp_path / "training" / run_id
    checkpoint_dir = run_root / "trainer" / "checkpoint-512"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest_path.write_text("{}\n", encoding="utf-8")
    run_manifest = {
        "run_id": run_id,
        "arm": "P",
        "config_sha256": "config-sha",
        "code_sha256": "training-code-sha",
        "optimizer_step": 512,
        "checkpoint_identity": {},
        "schedule_sha256": "schedule-sha",
        "nodes": {"512": {"training_status": "passed", "checkpoint_manifest_sha256": "checkpoint-sha"}},
    }
    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, adapter_path, **kwargs):
            calls.append((base, adapter_path, kwargs))
            return SimpleNamespace(config=SimpleNamespace(hidden_size=4), eval=lambda: None)

    class FakeHead:
        def load_state_dict(self, _state, strict):
            assert strict is True

    class FakeHeads:
        profile_head = FakeHead()

        def to(self, _device):
            return self

        def eval(self):
            return None

    peft = SimpleNamespace(PeftModel=FakePeftModel)
    safetensors = ModuleType("safetensors")
    safetensors.__path__ = []
    safetensors_torch = ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0), device=lambda *_args: "cuda:0"),
    )
    monkeypatch.setattr(e03_a7_evaluate.e03_a7_train, "_load_base", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(e03_a7_evaluate.e03_a7_train, "build_heads", lambda *_args, **_kwargs: FakeHeads())
    monkeypatch.setattr(e03_a7_evaluate, "verify_checkpoint_manifest", lambda *_args, **_kwargs: {"status": "passed"})
    monkeypatch.setattr(e03_a7_evaluate, "sha256_file", lambda _path: "checkpoint-sha")

    loaded = e03_a7_evaluate._load_run_assets(
        {
            "training": {"pair_id": "pair"},
            "outputs": {"training_root": str(tmp_path / "training")},
            "model": {"heads": {"profile_seed": 1, "emotion_seed": 2}},
        },
        "config-sha",
        "training-code-sha",
        "P",
        512,
    )

    assert loaded["run_id"] == run_id
    assert calls[0][2]["autocast_adapter_dtype"] is True


def test_a7_eval_inference_keeps_single_row_logits_and_row_order(monkeypatch):
    import torch as real_torch

    import src.training.e03_a7_classifier as classifier

    calls = []

    def fake_forward(_model, input_ids, attention_mask, prompt_lengths):
        calls.append((input_ids.cpu().tolist(), attention_mask.cpu().tolist(), prompt_lengths.cpu().tolist()))
        return prompt_lengths.float().unsqueeze(1)

    class FakeHeads:
        def eval(self):
            return None

        def __call__(self, hidden):
            return {"profile_logits": hidden.expand(-1, 33)}

    class FakeModel:
        def eval(self):
            return None

    device_factory = real_torch.device
    monkeypatch.setattr(real_torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(real_torch, "device", lambda *_args: device_factory("cpu"))
    monkeypatch.setattr(classifier, "forward_prompt_end", fake_forward)

    rows = [
        {"sample_id": "a", "input_ids": [2, 3, 4]},
        {"sample_id": "b", "input_ids": [5]},
        {"sample_id": "c", "input_ids": [6, 7]},
        {"sample_id": "d", "input_ids": [8, 9, 10, 11]},
        {"sample_id": "e", "input_ids": [12, 13]},
    ]
    predictions = e03_a7_evaluate._infer_probabilities(
        FakeModel(), FakeHeads(), "P", rows, pad_token_id=0,
    )

    assert [row["id"] for row in predictions] == [row["sample_id"] for row in rows]
    assert [len(call[0]) for call in calls] == [1] * len(rows)
    assert [call[1][0] for call in calls] == [[1, 1, 1], [1], [1, 1], [1, 1, 1, 1], [1, 1]]
    assert [call[2] for call in calls] == [[3], [1], [2], [4], [2]]
    assert predictions[1]["profile_probabilities"][0] == float(real_torch.sigmoid(real_torch.tensor(1.0)))


def test_a7_threshold_is_chosen_on_calibration_with_precision_constraint():
    probabilities = [[0.9] + [0.01] * 32, [0.8] + [0.01] * 32, [0.2] + [0.01] * 32]
    targets = [[1] + [0] * 32, [1] + [0] * 32, [0] * 33]
    result = choose_calibration_threshold(
        probabilities,
        targets,
        thresholds=[0.2, 0.5, 0.8],
        minimum_precision=0.75,
    )
    assert result["status"] == "calibration_threshold_selected"
    assert result["threshold"] == 0.8
    failed = choose_calibration_threshold(
        [[0.1] * 33], [[0] * 33], thresholds=[0.5], minimum_precision=0.75
    )
    assert failed["status"] == "threshold_constraint_unsatisfied"
    assert failed["threshold"] is None


def test_a7_prediction_mapping_and_average_precision_are_fixed_to_33_labels():
    enums = _enums()
    probabilities = [[0.9] + [0.01] * 32, [0.1] * 33]
    profile = profile_from_probabilities(probabilities, threshold=0.5, enums=enums)
    assert profile[0]["personality_traits"] == ["p0"]
    assert profile[1] == {"personality_traits": [], "interests": [], "style": []}
    targets = [[1] + [0] * 32, [0] * 33]
    labels = [f"{dimension}:{label}" for dimension in enums for label in enums[dimension]]
    result = average_precision_by_label(probabilities, targets, labels)
    assert result["labels"][labels[0]]["average_precision"] == 1.0
    assert result["labels"][labels[1]]["status"] == "insufficient_support"
    assert result["micro_average_precision"] is not None


def test_a7_pairing_reports_added_true_positives_and_false_positives():
    enums = _enums()
    gold = [{"personality_traits": ["p0"], "interests": [], "style": []}]
    p = [{"personality_traits": [], "interests": [], "style": []}]
    ep = [{"personality_traits": ["p0", "p1"], "interests": [], "style": []}]
    result = compare_arm_profile_predictions(ids=["x"], gold=gold, p_predictions=p, ep_predictions=ep, enums=enums)
    assert result["ep_minus_p"]["tp"] == 1
    assert result["ep_minus_p"]["fp"] == 1
    assert result["ep_correct_p_wrong_label_counts"]["personality_traits:p0"] == 1


def test_a7_conversation_cluster_bootstrap_uses_fixed_aligned_population():
    enums = _enums()
    empty = {"personality_traits": [], "interests": [], "style": []}
    positive = {"personality_traits": ["p0"], "interests": [], "style": []}
    report = summarize_profile_pair(
        ids=["a", "b"], conversation_ids=["c1", "c2"],
        gold=[positive, empty], baseline=[empty, empty], candidate=[positive, empty],
        enums=enums, bootstrap_repetitions=10,
    )
    assert report["sample_count"] == 2
    assert report["cluster_bootstrap"]["unit"] == "complete_conversation"
    assert report["candidate_minus_baseline"]["tp"] == 1


def test_a7_single_arm_review_is_explicitly_unpaired_and_bound_to_artifacts(tmp_path):
    evaluation_root = tmp_path / "evaluation"
    arm_root = evaluation_root / "pair-P" / "step-0512"
    arm_root.mkdir(parents=True)
    report_path = arm_root / "evaluation_report.json"
    manifest_path = arm_root / "evaluation_manifest.json"
    predictions_path = arm_root / "predictions.jsonl"
    predictions_path.write_text("{}\n", encoding="utf-8")
    predictions_sha = sha256_file(predictions_path)
    report_path.write_text(json.dumps({
        "status": "scored", "arm": "P", "run_id": "pair-P", "checkpoint_step": 512,
        "sample_count": 224, "frozen_ids_sha256": "frozen-ids", "locked_predictions_sha256": predictions_sha,
        "gold_read_after_predictions_locked": True,
    }), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "status": "scored", "prediction_count": 224, "predictions_sha256": predictions_sha,
    }), encoding="utf-8")
    config = {
        "training": {"pair_id": "pair"},
        "outputs": {"evaluation_root": str(evaluation_root)},
        "data": {"expected_dev_rows": 224},
        "identity": {"frozen_dev_ids_sha256": "frozen-ids"},
    }

    result = _pair_report(config, 512)

    assert result["status"] == "awaiting_manual_review_missing_matched_arm"
    assert result["available_arm"] == "P"
    assert result["missing_arm"] == "EP"
    assert result["p_vs_ep"] == "not_applicable_missing_matched_arm"
    assert result["evaluation_report_sha256"] == sha256_file(report_path)
    assert build_parser().parse_args(["--mode", "evaluate-arm", "--arm", "P", "--checkpoint-step", "512"]).arm == "P"
