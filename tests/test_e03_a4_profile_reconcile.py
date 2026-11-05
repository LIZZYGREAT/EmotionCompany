from __future__ import annotations

import json

import pytest

from scripts.e03_a4_profile_reconcile import _load_fast_only_predictions
from src.evaluation import sha256_file
from src.evaluation.profile_reconcile import (
    classify_historical_difference,
    compare_profile_lists,
    empty_profile,
    index_profile_prediction_rows,
    profile_difference_dimensions,
    validate_profile,
)
from src.evaluation.source_integrity import canonical_sha256


ENUMS = {
    "emotion_label": ["neutral"],
    "personality_traits": ["open", "sensitive"],
    "interests": ["games"],
    "style": ["direct"],
}


def _profile(*, personality=(), interests=(), style=()):
    return {
        "personality_traits": list(personality),
        "interests": list(interests),
        "style": list(style),
    }


def test_pairwise_profile_metrics_keep_fixed_denominator_and_report_tp_fp_fn_deltas():
    gold = [_profile(personality=("open",), interests=("games",)), _profile(style=("direct",))]
    submission = [empty_profile(), _profile(style=("direct",))]
    raw_prefix = [_profile(personality=("open",), interests=("games",)), _profile()]

    result = compare_profile_lists(
        ids=["a", "b"],
        gold_profiles=gold,
        left_profiles=submission,
        right_profiles=raw_prefix,
        enums=ENUMS,
        left_name="S",
        right_name="R",
    )

    assert result["sample_count"] == 2
    assert result["different_sample_count"] == 2
    assert result["different_sample_count_by_dimension"] == {
        "personality_traits": 1,
        "interests": 1,
        "style": 1,
    }
    assert result["left_metrics"]["per_dimension"]["style"]["sample_count"] == 2
    assert result["left_metrics"]["overall_33_label_fn"] == 2
    assert result["right_metrics"]["overall_33_label_tp"] == 2
    assert result["right_minus_left"]["overall_33_label_tp"] == 1
    assert result["right_minus_left"]["overall_33_label_fn"] == -1


def test_profile_set_comparison_rejects_id_count_mapping_errors():
    with pytest.raises(ValueError, match="counts must match"):
        compare_profile_lists(
            ids=["a", "b"],
            gold_profiles=[_profile()],
            left_profiles=[_profile()],
            right_profiles=[_profile()],
            enums=ENUMS,
            left_name="S",
            right_name="R",
        )

    with pytest.raises(ValueError, match="duplicate IDs"):
        compare_profile_lists(
            ids=["a", "a"],
            gold_profiles=[_profile(), _profile()],
            left_profiles=[_profile(), _profile()],
            right_profiles=[_profile(), _profile()],
            enums=ENUMS,
            left_name="S",
            right_name="R",
        )


def test_profile_rows_reject_duplicate_and_unknown_labels():
    with pytest.raises(ValueError, match="invalid labels for interests"):
        validate_profile(_profile(interests=("games", "games")), ENUMS, label="fixture")
    with pytest.raises(ValueError, match="invalid labels for style"):
        validate_profile(_profile(style=("unknown",)), ENUMS, label="fixture")


def test_historical_attribution_separates_downstream_json_failure_from_prefix_failure():
    empty = empty_profile()
    predicted = _profile(personality=("open",))
    assert classify_historical_difference(
        submission_profile=empty,
        prefix_profile=predicted,
        prefix_status="valid",
        full_json_status="invalid_json",
    ) == "downstream_full_json_parse_failure"
    assert classify_historical_difference(
        submission_profile=predicted,
        prefix_profile=empty,
        prefix_status="incomplete",
        full_json_status="valid",
    ) == "prefix_unusable_submission_differs"
    assert classify_historical_difference(
        submission_profile=empty,
        prefix_profile=predicted,
        prefix_status="valid",
        full_json_status="valid",
    ) == "submission_normalization_or_mapping_difference"
    assert classify_historical_difference(
        submission_profile=predicted,
        prefix_profile=predicted,
        prefix_status="valid",
        full_json_status="valid",
    ) == "profile_match"


def test_profile_dimension_comparison_ignores_label_order():
    left = _profile(personality=("open", "sensitive"))
    right = _profile(personality=("sensitive", "open"))
    assert profile_difference_dimensions(left, right) == []


def test_minimal_fast_prediction_index_accepts_exact_ids_and_failure_placeholders():
    rows = [
        {
            "id": "a",
            "status": "valid",
            "emotion_label": "neutral",
            "user_profile": _profile(personality=("open",)),
            "reason": "profile_prefix_closed",
        },
        {"id": "b", "status": "incomplete", "reason": "interrupted_before_generation"},
    ]

    indexed, counts = index_profile_prediction_rows(rows, ["a", "b"], ENUMS)

    assert set(indexed) == {"a", "b"}
    assert counts == {
        "valid": 1,
        "invalid": 0,
        "incomplete": 1,
        "unsupported_order": 0,
        "budget_exhausted": 0,
    }


def test_minimal_fast_prediction_index_rejects_missing_duplicate_and_fake_failure_rows():
    valid = {
        "id": "a",
        "status": "valid",
        "emotion_label": "neutral",
        "user_profile": _profile(),
        "reason": "profile_prefix_closed",
    }
    with pytest.raises(ValueError, match="IDs or row count differ"):
        index_profile_prediction_rows([valid], ["a", "b"], ENUMS)
    with pytest.raises(ValueError, match="duplicate id"):
        index_profile_prediction_rows([valid, valid], ["a", "b"], ENUMS)
    with pytest.raises(ValueError, match="fabricated profile"):
        index_profile_prediction_rows(
            [valid, {"id": "b", "status": "invalid", "reason": "bad_prefix", "user_profile": _profile()}],
            ["a", "b"],
            ENUMS,
        )


def test_strict_fast_quality_comparison_requires_full_valid_cohort():
    rows = [
        {
            "id": "a",
            "status": "valid",
            "emotion_label": "neutral",
            "user_profile": _profile(),
            "reason": "profile_prefix_closed",
        },
        {"id": "b", "status": "budget_exhausted", "reason": "run_budget_exhausted"},
    ]
    with pytest.raises(ValueError, match="every frozen ID to be valid"):
        index_profile_prediction_rows(rows, ["a", "b"], ENUMS, require_all_valid=True)


def test_fast_only_loader_binds_224_style_facts_before_pairing(tmp_path):
    expected_ids = ["a", "b"]
    ids_sha256 = canonical_sha256(sorted(expected_ids))
    run_root = tmp_path / "fast-run"
    run_root.mkdir()
    prediction_path = run_root / "profile_predictions.jsonl"
    predictions = [
        {
            "id": sample_id,
            "status": "valid",
            "emotion_label": "neutral",
            "user_profile": _profile(personality=("open",) if sample_id == "a" else ()),
            "reason": "profile_prefix_closed",
        }
        for sample_id in expected_ids
    ]
    prediction_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in predictions),
        encoding="utf-8",
    )
    candidate = {
        "selection": {"expected_ids_sha256": ids_sha256},
        "protocol": {"prompt_sha256": "prompt-sha"},
        "generation": {"max_new_tokens": 1200},
    }
    a3_config = {
        "identity": {
            "expected_base_weight_sha256": "base-sha",
            "expected_initial_adapter_weight_sha256": "adapter-sha",
            "expected_initial_adapter_config_sha256": "adapter-config-sha",
        },
        "protocol": {
            "tokenizer_signature_sha256": "tokenizer-sha",
            "chat_template_sha256": "template-sha",
        },
    }
    source_facts = {"a4_candidate_config": {"sha256": "candidate-config-sha"}}
    eval_manifest = {"inference_config_sha256": "inference-config-sha"}
    report = {
        "status": "passed",
        "gpu_used": True,
        "references_read_by_generator": False,
        "sample_count": 2,
        "valid_count": 2,
        "attempted_count": 2,
        "generation": {
            "model_generate_call_count": 2,
            "full_generation_performed": False,
            "response_text_generated_or_saved": False,
        },
        "profile_predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "rows": 2,
            "expected_rows": 2,
            "ids_sha256": ids_sha256,
            "all_selected_rows_valid": True,
            "strict_quality_comparison_ready": True,
        },
        "run_identity": {
            "config_sha256": "candidate-config-sha",
            "inference_config_sha256": "inference-config-sha",
            "base_weight_sha256": "base-sha",
            "adapter_weight_sha256": "adapter-sha",
            "adapter_config_sha256": "adapter-config-sha",
            "prompt_facts": {"sha256": "prompt-sha"},
            "generation": {"do_sample": False, "max_new_tokens": 1200},
            "tokenizer_facts": {
                "signature_sha256": "tokenizer-sha",
                "template_sha256": "template-sha",
            },
            "model_facts": {"quantization": {"real_nf4_verified": True}},
        },
        "source_manifest": {"selection": {"ids_sha256": ids_sha256}},
        "timing": {"elapsed_seconds_including_model_load": 10.0, "model_load_seconds": 1.0},
    }
    report_path = run_root / "profile_fast_metrics.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    indexed, metadata = _load_fast_only_predictions(
        report_path,
        expected_ids,
        candidate,
        a3_config,
        source_facts,
        eval_manifest,
        ENUMS,
    )

    assert list(indexed) == expected_ids
    assert metadata["predictions_rows"] == 2
    assert metadata["run_id"] == "fast-run"
    assert metadata["status_counts"]["valid"] == 2


def test_fast_only_loader_rejects_report_identity_or_prediction_hash_mismatch(tmp_path):
    run_root = tmp_path / "fast-run"
    run_root.mkdir()
    prediction_path = run_root / "profile_predictions.jsonl"
    prediction_path.write_text("{}\n", encoding="utf-8")
    report_path = run_root / "profile_fast_metrics.json"
    report_path.write_text(
        json.dumps({"profile_predictions": {"path": str(prediction_path), "sha256": "wrong"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity differs"):
        _load_fast_only_predictions(
            report_path,
            ["a"],
            {"selection": {"expected_ids_sha256": "ids"}, "protocol": {"prompt_sha256": "p"}, "generation": {"max_new_tokens": 1}},
            {"identity": {}, "protocol": {}},
            {"a4_candidate_config": {"sha256": "c"}},
            {"inference_config_sha256": "i"},
            ENUMS,
        )
