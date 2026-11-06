import json

from scripts.e03_a7_evaluate import _pair_report, build_parser
from src.data.e03_a5_labels import sha256_file
from src.evaluation.e03_a7_classification_metrics import (
    average_precision_by_label,
    choose_calibration_threshold,
    compare_arm_profile_predictions,
    profile_from_probabilities,
    summarize_profile_pair,
)


def _enums():
    return {
        "personality_traits": [f"p{i}" for i in range(10)],
        "interests": [f"i{i}" for i in range(12)],
        "style": [f"s{i}" for i in range(11)],
    }


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
