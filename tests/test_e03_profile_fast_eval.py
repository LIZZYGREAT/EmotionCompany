from __future__ import annotations

import json

import pytest

from src.evaluation.profile_fast_eval import (
    compare_generated_prefix,
    profile_nonempty_recall,
    run_profile_fast_only_attempts,
    score_profile_attempts,
    selected_id_sha256,
    select_frozen_id_samples,
    select_unique_conversation_samples,
)
from src.evaluation.source_integrity import (
    audit_prediction_point_sources,
    assign_dev_cluster_partitions,
    build_source_manifest,
    validate_history_matches_turns,
    validate_target_free_inputs,
    verify_prompt_token_ids,
)
from scripts import e03_profile_fast_eval as fast_eval_cli


def _inputs():
    return [
        {"id": "p1", "history": [{"role": "user", "content": "one"}]},
        {"id": "p2", "history": [{"role": "user", "content": "two"}]},
        {"id": "p3", "history": [{"role": "user", "content": "three"}]},
        {"id": "p4", "history": [{"role": "user", "content": "four"}]},
    ]


def _id_map():
    return [
        {"eval_id": "p1", "conversation_id": "c1", "sample_id": "s1"},
        {"eval_id": "p2", "conversation_id": "c1", "sample_id": "s2"},
        {"eval_id": "p3", "conversation_id": "c2", "sample_id": "s3"},
        {"eval_id": "p4", "conversation_id": "c3", "sample_id": "s4"},
    ]


def test_fixed_selection_is_repeatable_session_unique_and_target_free():
    selected = select_unique_conversation_samples(_inputs(), _id_map(), count=3, seed=42)
    repeated = select_unique_conversation_samples(_inputs(), _id_map(), count=3, seed=42)
    assert selected == repeated
    assert len({row["id"] for row in selected}) == 3
    assert all(set(row) == {"id", "history"} for row in selected)


def test_selection_rejects_gold_leak_duplicates_and_invalid_budgets():
    with pytest.raises(ValueError, match="only id and history"):
        validate_target_free_inputs([{"id": "p1", "history": _inputs()[0]["history"], "target": {}}])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_target_free_inputs([_inputs()[0], _inputs()[0]])
    with pytest.raises(ValueError, match="between 1 and 32"):
        select_unique_conversation_samples(_inputs(), _id_map(), count=33, seed=42)


def test_current_user_is_verified_against_explicit_source_turn_id():
    turns = [
        {"turn_id": 10, "role": "user", "content": "prior"},
        {"turn_id": 11, "role": "assistant", "content": "answer"},
        {"turn_id": 20, "role": "user", "content": "current"},
        {"turn_id": 21, "role": "assistant", "content": "target response"},
    ]
    history = [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "current"},
    ]
    assert validate_history_matches_turns(turns, history, 20) == 2
    with pytest.raises(ValueError, match="omits or adds"):
        validate_history_matches_turns(turns, history[:-1], 20)
    with pytest.raises(ValueError, match="does not reference a user"):
        validate_history_matches_turns(turns, history, 11)


def test_prediction_point_preflight_blocks_missing_current_user_before_generation():
    raw = [{
        "conversation_id": "c1",
        "turns": [
            {"turn_id": 1, "role": "assistant", "content": "opening"},
            {"turn_id": 2, "role": "user", "content": "current"},
            {"turn_id": 3, "role": "assistant", "content": "response"},
        ],
        "prediction_points": [{"target_user_turn_id": 2}],
    }]
    normalized = [{
        "sample_id": "s1",
        "conversation_id": "c1",
        "source": {
            "split": "train", "source_line": 1, "prediction_point_index": 0,
            "target_user_turn_id": 2, "target_turn_index": 1,
        },
        "history": [{"role": "assistant", "content": "opening"}],
    }]
    with pytest.raises(ValueError, match="omits or adds"):
        audit_prediction_point_sources(normalized, raw)


def test_duplicate_sample_id_and_session_grouping_are_checked():
    rows = [
        {"sample_id": "s1", "conversation_id": "c1"},
        {"sample_id": "s2", "conversation_id": "c1"},
        {"sample_id": "s3", "conversation_id": "c2"},
    ]
    assignment = assign_dev_cluster_partitions(rows, seed=7, holdout_percent=35)
    assert assignment["s1"] == assignment["s2"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        assign_dev_cluster_partitions([rows[0], rows[0]], seed=7)


def test_equal_prompt_token_counts_do_not_imply_token_identity():
    assert verify_prompt_token_ids([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="despite matching token counts"):
        verify_prompt_token_ids([1, 2, 3], [3, 2, 1])


def test_source_manifest_contains_hashes_and_explicit_non_repair_status():
    manifest = build_source_manifest(
        source_path="outputs/dev.jsonl",
        source_sha256="a" * 64,
        input_rows=_inputs()[:2],
        id_map_rows=_id_map()[:2],
        prompt_version="P0",
        prompt_sha256="b" * 64,
        format_version="F2",
        truncation_version="T1",
        exporter_source_sha256="c" * 64,
    )
    assert manifest["selected_point_count"] == 2
    assert manifest["selected_conversation_count"] == 1
    assert manifest["history_repaired"] is False
    assert manifest["prompt_token_ids_sha256"] is None


def test_prefix_equivalence_requires_exact_token_ids_and_profile_values():
    same = compare_generated_prefix(
        [1, 2, 3, 4], [1, 2], "joy", "joy",
        {"personality_traits": [], "interests": ["music"], "style": []},
        {"personality_traits": [], "interests": ["music"], "style": []},
    )
    assert same["token_prefix_match"]
    assert same["emotion_and_profile_match"]
    changed = compare_generated_prefix(
        [1, 2, 3, 4], [2, 1], "joy", "joy",
        {"personality_traits": [], "interests": [], "style": []},
        {"personality_traits": [], "interests": [], "style": []},
    )
    assert not changed["token_prefix_match"]


def test_nonempty_dimension_recall_keeps_three_dimensions_separate():
    gold = [
        {"personality_traits": ["kind"], "interests": [], "style": ["warm"]},
        {"personality_traits": [], "interests": ["music"], "style": []},
    ]
    predicted = [
        {"personality_traits": [], "interests": [], "style": ["warm"]},
        {"personality_traits": [], "interests": ["music"], "style": []},
    ]
    recall = profile_nonempty_recall(gold, predicted)
    assert recall["personality_traits"]["recall"] == 0.0
    assert recall["interests"]["recall"] == 1.0
    assert recall["style"]["recall"] == 1.0


def test_frozen_selection_pins_id_membership_and_complete_conversation_groups():
    frozen = [_inputs()[0], _inputs()[1], _inputs()[2]]
    selected = select_frozen_id_samples(
        _inputs(),
        _id_map(),
        frozen,
        expected_ids_sha256=selected_id_sha256([row["id"] for row in frozen]),
        expected_count=3,
        expected_conversation_count=2,
    )
    assert [row["id"] for row in selected] == ["p1", "p2", "p3"]
    with pytest.raises(ValueError, match="splits one or more conversation groups"):
        select_frozen_id_samples(
            _inputs(),
            _id_map(),
            [_inputs()[0], _inputs()[2]],
            expected_ids_sha256=selected_id_sha256(["p1", "p3"]),
            expected_count=2,
            expected_conversation_count=2,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        select_frozen_id_samples(
            _inputs(),
            _id_map(),
            frozen,
            expected_ids_sha256="0" * 64,
            expected_count=3,
            expected_conversation_count=2,
        )


def test_fast_only_runner_calls_one_candidate_generation_per_sample():
    calls = []

    def generate_candidate_once(sample):
        calls.append(sample["id"])
        return {"status": "valid", "user_profile": {
            "personality_traits": [], "interests": [], "style": [],
        }}

    attempts = run_profile_fast_only_attempts(_inputs()[:3], generate_candidate_once)
    assert calls == ["p1", "p2", "p3"]
    assert [attempt["status"] for attempt in attempts] == ["valid", "valid", "valid"]
    assert [attempt["id"] for attempt in attempts] == calls


def test_fast_only_budget_termination_marks_unstarted_rows_without_calling_them():
    calls = []

    def generate_candidate_once(sample):
        calls.append(sample["id"])
        if sample["id"] == "p2":
            return {"status": "budget_exhausted", "reason": "deadline"}
        return {"status": "incomplete", "reason": "early_eos"}

    attempts = run_profile_fast_only_attempts(_inputs(), generate_candidate_once)
    assert calls == ["p1", "p2"]
    assert [attempt["status"] for attempt in attempts] == [
        "incomplete", "budget_exhausted", "budget_exhausted", "budget_exhausted",
    ]


def test_fast_only_generation_error_marks_remaining_rows_incomplete():
    calls = []

    def generate_candidate_once(sample):
        calls.append(sample["id"])
        return {"status": "incomplete", "reason": "generation_error:RuntimeError:oom"}

    attempts = run_profile_fast_only_attempts(_inputs(), generate_candidate_once)
    assert calls == ["p1"]
    assert [attempt["status"] for attempt in attempts] == ["incomplete"] * 4
    assert attempts[1]["reason"] == "run_stopped_after_generation_error"


def test_fixed_denominator_keeps_invalid_incomplete_and_unsupported_rows():
    enums = {
        "personality_traits": ["kind"],
        "interests": ["music"],
        "style": ["warm"],
    }
    gold = [
        {"personality_traits": ["kind"], "interests": [], "style": []},
        {"personality_traits": [], "interests": ["music"], "style": []},
        {"personality_traits": [], "interests": [], "style": ["warm"]},
        {"personality_traits": [], "interests": [], "style": []},
    ]
    attempts = [
        {"status": "valid", "user_profile": gold[0]},
        {"status": "invalid", "reason": "unknown_profile_label"},
        {"status": "incomplete", "reason": "early_eos"},
        {"status": "unsupported_order", "reason": "candidate_top_level_order"},
    ]
    score = score_profile_attempts(gold, attempts, enums)
    assert score["fixed_denominator_count"] == 4
    assert score["status_counts"] == {
        "valid": 1, "invalid": 1, "incomplete": 1,
        "unsupported_order": 1, "budget_exhausted": 0,
    }
    assert score["profile_metrics"]["overall_33_label_tp"] == 1
    assert score["profile_metrics"]["overall_33_label_fn"] == 2
    assert score["valid_only_diagnostic"]["denominator_count"] == 1
    assert score["valid_only_diagnostic"]["profile_metrics"]["overall_33_label_micro_f1"] == 1.0
    assert score["score_population"] == "all_selected_rows_failures_as_empty_profile"


@pytest.mark.parametrize(
    ("mode", "status", "default_exit", "strict_exit"),
    [
        ("parse-raw", "cpu_raw_parse_completed", 0, 1),
        ("audit-source", "source_preflight_passed", 0, 1),
        ("gpu-fast-only", "passed", 0, 0),
        ("gpu-fast-only", "not_adopted", 0, 1),
        ("gpu-fast-only", "budget_exhausted", 0, 1),
        ("gpu-fast-only", "blocked_environment", 0, 1),
        ("gpu-fast-only", "incomplete", 0, 1),
        ("gpu-fast-only", "preflight_failed", 1, 1),
        ("gpu-fast-only", "unknown", 1, 1),
    ],
)
def test_cli_report_json_and_optional_strict_exit_status(
    tmp_path, monkeypatch, capsys, mode, status, default_exit, strict_exit,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test config", encoding="utf-8")
    base_config = {
        "version": "E03-profile-fast-v1",
        "experiment_version": "E03-profile-fast-test-v1",
        "protocol": {"format_version": "F2", "truncation_version": "T1"},
        "generation": {"do_sample": False},
        "gpu_budget": {"max_seconds": 30},
        "selection": {"mode": "mechanism_smoke", "count": 16, "seed": 42},
        "output_root": str(tmp_path / "reports"),
    }
    monkeypatch.setattr(fast_eval_cli, "_load_config", lambda _path: base_config)
    monkeypatch.setattr(
        fast_eval_cli,
        "run_gpu_fast_only",
        lambda *_args: {"status": status, "sample_count": 16, "valid_count": 12},
    )
    monkeypatch.setattr(
        fast_eval_cli,
        "parse_existing_raw",
        lambda *_args: {"status": status, "sample_count": 16, "valid_fast_count": 12},
    )
    monkeypatch.setattr(
        fast_eval_cli,
        "run_source_audit",
        lambda *_args: {"status": status, "sample_count": 16, "valid_fast_count": 12},
    )

    def invoke(strict):
        run_id = f"status-{status.replace('_', '-')}-{'strict' if strict else 'default'}"
        args = ["--mode", mode, "--config", str(config_path), "--run-id", run_id]
        if mode == "parse-raw":
            args.extend(["--raw", str(tmp_path / "raw.jsonl")])
        if strict:
            args.append("--require-passed")
        result = fast_eval_cli.main(args)
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == status
        assert output["sample_count"] == 16
        assert output["valid_fast_count"] == 12
        assert output["output"].endswith(f"/{run_id}/profile_fast_metrics.json")
        return result

    assert invoke(False) == default_exit
    assert invoke(True) == strict_exit
