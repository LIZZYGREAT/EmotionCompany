from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import e03_a5_evaluate as evaluate
from src.data.e03_a5_labels import sha256_file
from src.evaluation.metrics import profile_metrics
from src.evaluation.profile_fast_eval import selected_id_sha256


PROFILE_ENUMS = {
    "personality_traits": ["trait_a", "trait_b"],
    "interests": ["interest_a"],
    "style": ["style_a"],
}
EMOTION_LABELS = ["calm", "sad"]


def _profile(*, trait: str | None = None, interest: str | None = None, style: str | None = None):
    return {
        "personality_traits": [trait] if trait else [],
        "interests": [interest] if interest else [],
        "style": [style] if style else [],
    }


def test_three_pairwise_reports_use_the_same_gold_for_each_comparison():
    ids = ["p1", "p2", "p3", "p4"]
    gold = [_profile(trait="trait_a"), _profile(), _profile(trait="trait_b"), _profile(style="style_a")]
    baseline = [_profile(), _profile(), _profile(trait="trait_b"), _profile(style="style_a")]
    control = [_profile(trait="trait_a"), _profile(), _profile(), _profile(style="style_a")]
    candidate = [_profile(trait="trait_a"), _profile(), _profile(trait="trait_b"), _profile()]
    gold_emotions = ["calm", "sad", "calm", "sad"]
    baseline_emotions = ["sad", "sad", "calm", "sad"]
    control_emotions = ["calm", "sad", "sad", "sad"]
    candidate_emotions = ["calm", "sad", "calm", "calm"]

    result = evaluate._pairwise_comparisons(
        ids=ids,
        conversations=["c1", "c1", "c2", "c2"],
        history_lengths=[1, 2, 5, 6],
        gold_profiles=gold,
        baseline_profiles=baseline,
        control_profiles=control,
        candidate_profiles=candidate,
        gold_emotions=gold_emotions,
        baseline_emotions=baseline_emotions,
        control_emotions=control_emotions,
        candidate_emotions=candidate_emotions,
        profile_enums=PROFILE_ENUMS,
        emotion_labels=EMOTION_LABELS,
        seed=7,
        repetitions=10,
        short_history_max=2,
    )

    expected_baseline = profile_metrics(gold, baseline, PROFILE_ENUMS)
    expected_control = profile_metrics(gold, control, PROFILE_ENUMS)
    observed = result["Control-E01-B"]
    assert observed["baseline"]["profile"]["overall_33_label_micro_f1"] == expected_baseline["overall_33_label_micro_f1"]
    assert observed["candidate"]["profile"]["overall_33_label_micro_f1"] == expected_control["overall_33_label_micro_f1"]
    assert observed["baseline"]["emotion_accuracy_fixed_denominator"] == 0.75
    assert observed["candidate"]["emotion_accuracy_fixed_denominator"] == 0.75
    assert set(result) == {"Candidate-Control", "Candidate-E01-B", "Control-E01-B"}


def test_load_fast_run_requires_a_passed_complete_frozen_prediction_artifact(tmp_path, monkeypatch):
    eval_root = tmp_path / "outputs" / "evaluation" / "e03_a5"
    fast_root = eval_root / "fast_only" / "FAST-01"
    fast_root.mkdir(parents=True)
    monkeypatch.setattr(evaluate, "_eval_root", lambda _config: eval_root)
    monkeypatch.setattr(evaluate.probe_runner, "_resolve", lambda path: Path(path))
    monkeypatch.setattr(evaluate, "schema_enums", lambda _schema: {**PROFILE_ENUMS, "emotion_label": EMOTION_LABELS})

    frozen_ids = [f"eval-{index:03d}" for index in range(224)]
    frozen_input = tmp_path / "frozen_input.jsonl"
    frozen_input.write_text("".join(json.dumps({"id": sample_id}) + "\n" for sample_id in frozen_ids), encoding="utf-8")
    predictions = [
        {
            "id": sample_id,
            "status": "valid",
            "emotion_label": "calm",
            "user_profile": _profile(trait="trait_a"),
            "reason": "valid_profile_prefix",
        }
        for sample_id in frozen_ids
    ]
    predictions_path = fast_root / "profile_predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    report = {
        "status": "passed",
        "run_id": "FAST-01",
        "sample_count": 224,
        "accounted_count": 224,
        "attempted_count": 224,
        "references_read_by_generator": False,
        "response_text_generated_or_saved": False,
        "full_generation_performed": False,
        "profile_predictions": {
            "rows": 224,
            "expected_rows": 224,
            "sha256": sha256_file(predictions_path),
            "ids_sha256": selected_id_sha256(frozen_ids),
            "response_text_saved": False,
            "complete_id_coverage": True,
            "all_selected_rows_valid": True,
            "strict_quality_comparison_ready": True,
        },
        "run_identity": {
            "config_sha256": "a" * 64,
            "inference_config_sha256": "b" * 64,
            "adapter_weight_sha256": "c" * 64,
            "adapter_config_sha256": "d" * 64,
            "base_weight_sha256": "e" * 64,
            "source_code_hashes": evaluate._fast_evaluator_source_hashes(),
        },
        "generation": {
            "full_generation_performed": False,
            "response_text_generated_or_saved": False,
        },
        "source_manifest": {
            "selection": {
                "selected_ids_sha256": selected_id_sha256(frozen_ids),
                "selected_count": 224,
                "selected_conversation_count": 101,
            }
        },
    }
    report_path = fast_root / "profile_fast_metrics.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    code_hashes = evaluate._fast_evaluator_source_hashes()
    plan = {
        "fast_run_id": "FAST-01",
        "pair_id": "PAIR-01",
        "group": "candidate",
        "training_run_id": "TRAIN-01",
        "training_manifest_sha256": "1" * 64,
        "config_sha256": "f" * 64,
        "audit_manifest_sha256": "2" * 64,
        "frozen_ids_sha256": selected_id_sha256(frozen_ids),
        "fast_config_sha256": "a" * 64,
        "inference_config_sha256": "b" * 64,
        "baseline_fast_predictions_sha256": "3" * 64,
        "generation_protocol": "P0_F2_T1_greedy_G1200",
        "fast_evaluator_source_hashes": code_hashes,
        "frozen_input_path": str(frozen_input),
        "frozen_input_sha256": sha256_file(frozen_input),
    }
    plan_dir = eval_root / "pairs" / "PAIR-01" / "plans" / "TRAIN-01"
    plan_dir.mkdir(parents=True)
    plan_path = plan_dir / "evaluation_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    ticket = {
        "version": "E03-A5-run-ticket-v1",
        "approved": True,
        "approved_by": "reviewer",
        "experiment_id": "E03-A5-BCE-AUX",
        "stage": "candidate_generation",
        "scope": "a5_frozen_candidate_generation",
        "run_id": "FAST-01",
        "config_sha256": plan["config_sha256"],
        "audit_manifest_sha256": plan["audit_manifest_sha256"],
        "budget": {"max_seconds": 900, "max_sample_count": 224},
        "pair_id": plan["pair_id"],
        "group": plan["group"],
        "training_run_id": plan["training_run_id"],
        "training_manifest_sha256": plan["training_manifest_sha256"],
        "evaluation_plan_sha256": sha256_file(plan_path),
        "fast_config_sha256": plan["fast_config_sha256"],
        "inference_config_sha256": plan["inference_config_sha256"],
        "frozen_ids_sha256": plan["frozen_ids_sha256"],
        "baseline_fast_predictions_sha256": plan["baseline_fast_predictions_sha256"],
        "generation_protocol": plan["generation_protocol"],
        "fast_evaluator_source_hashes": code_hashes,
    }
    ticket_path = tmp_path / "ticket.json"
    ticket_path.write_text(json.dumps(ticket), encoding="utf-8")
    wrapper = {
        "status": "generation_wrapper_completed",
        "run_id": "FAST-01",
        "pair_id": plan["pair_id"],
        "group": plan["group"],
        "training_run_id": plan["training_run_id"],
        "ticket_path": str(ticket_path),
        "ticket_sha256": sha256_file(ticket_path),
        "evaluation_plan_sha256": sha256_file(plan_path),
        "subprocess_exit_code": 0,
        "fast_report_sha256": sha256_file(report_path),
        "fast_report_status": "passed",
        "response_text_generated_or_saved": False,
        "full_generation_performed": False,
    }
    (fast_root / "a5_generation_wrapper.json").write_text(json.dumps(wrapper), encoding="utf-8")
    training_manifest = {"adapter_identity": {"weight_sha256": "c" * 64, "config_sha256": "d" * 64}}
    config = {"identity": {"base_weight_sha256": "e" * 64}, "data": {"official_output_schema": str(tmp_path / "schema.json")}}
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    loaded_report, loaded_rows, indexed, counts, _path = evaluate._load_fast_run(config, plan, training_manifest)
    assert loaded_report["status"] == "passed"
    assert len(loaded_rows) == 224
    assert len(indexed) == 224
    assert counts == {
        "budget_exhausted": 0,
        "incomplete": 0,
        "invalid": 0,
        "unsupported_order": 0,
        "valid": 224,
    }

    invalid_predictions = [
        {"id": sample_id, "status": "incomplete", "reason": "cuda_unavailable"}
        for sample_id in frozen_ids
    ]
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in invalid_predictions), encoding="utf-8")
    report["status"] = "blocked_environment"
    report["profile_predictions"].update({
        "sha256": sha256_file(predictions_path),
        "complete_id_coverage": True,
        "all_selected_rows_valid": False,
        "strict_quality_comparison_ready": False,
        "status_counts": {"incomplete": 224},
    })
    report_path.write_text(json.dumps(report), encoding="utf-8")
    wrapper["fast_report_sha256"] = sha256_file(report_path)
    wrapper["fast_report_status"] = "blocked_environment"
    (fast_root / "a5_generation_wrapper.json").write_text(json.dumps(wrapper), encoding="utf-8")
    _report, _rows, _indexed, failed_counts, _path = evaluate._load_fast_run(config, plan, training_manifest)
    assert _report["status"] == "blocked_environment"
    assert failed_counts["incomplete"] == 224
    assert failed_counts["valid"] == 0

    report["status"] = "passed"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    wrapper["fast_report_sha256"] = sha256_file(report_path)
    wrapper["fast_report_status"] = "passed"
    (fast_root / "a5_generation_wrapper.json").write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="status summary"):
        evaluate._load_fast_run(config, plan, training_manifest)


def test_label_count_summary_is_per_model_and_deterministic():
    result = evaluate._label_count_summary({
        "A": [_profile(trait="trait_a", style="style_a"), _profile()],
        "B": [_profile(interest="interest_a")],
    })
    assert result["A"]["histogram"] == {"0": 1, "2": 1}
    assert result["A"]["mean_labels_per_sample"] == 1.0
    assert result["B"]["histogram"] == {"1": 1}
