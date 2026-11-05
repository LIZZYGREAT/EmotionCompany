from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import e03_a3_short_train
from scripts import e03_a3_short_evaluate
from scripts.e03_a3_short_evaluate import paired_cluster_bootstrap
from src.training.e03_a3_sampling import (
    NATURAL,
    SHORT_EMPTY,
    SHORT_NONEMPTY,
    build_schedule,
    materialize_schedule,
    prepare_training_records,
    schedule_summary,
)


EOS = 151645
ENUMS = {
    "emotion_label": ["neutral"],
    "personality_traits": ["sensitive"],
    "interests": ["music"],
    "style": ["direct"],
}


def _gold(nonempty: bool) -> dict[str, list[str]]:
    return {
        "personality_traits": ["sensitive"] if nonempty else [],
        "interests": [],
        "style": [],
    }


def _source(index: int, *, history_length: int, nonempty: bool) -> dict:
    history = []
    first_role = "assistant" if history_length % 2 == 0 else "user"
    for position in range(history_length):
        role = first_role if position % 2 == 0 else ("user" if first_role == "assistant" else "assistant")
        history.append({"role": role, "content": f"message {index}-{position}"})
    return {
        "sample_id": f"train:sample-{index}",
        "conversation_id": f"conversation-{index // 2}",
        "history": history,
        "target": {"user_profile": _gold(nonempty)},
    }


def _encoded(source: dict) -> dict:
    return {
        "sample_id": source["sample_id"],
        "conversation_id": source["conversation_id"],
        "source_split": "train",
        "source": {},
        "prompt_length": 2,
        "completion_length": 3,
        "input_ids": [11, 12, 13, EOS, 198],
        "labels": [-100, -100, 13, EOS, 198],
    }


def _records() -> list[dict]:
    sources = []
    encoded = []
    for index in range(16):
        sources.append(_source(
            index,
            history_length=2 if index < 8 else 6,
            nonempty=4 <= index < 8 or index >= 8,
        ))
        encoded.append(_encoded(sources[-1]))
    return prepare_training_records(
        encoded,
        sources,
        ENUMS,
        expected_rows=16,
        eos_token_id=EOS,
        max_sequence_length=32,
    )


def test_prepare_training_records_checks_join_mask_eos_and_history() -> None:
    records = _records()

    assert len(records) == 16
    assert records[0]["raw_history_length"] == 2
    assert records[0]["gold_profile_empty"] is True
    assert records[4]["gold_profile_empty"] is False
    assert records[0]["encoded"]["labels"][-2] == EOS


def test_prepare_training_records_rejects_masked_completion_eos() -> None:
    sources = [_source(0, history_length=2, nonempty=False)]
    row = _encoded(sources[0])
    row["labels"][3] = -100

    with pytest.raises(ValueError, match="completion labels"):
        prepare_training_records(
            [row], sources, ENUMS, expected_rows=1, eos_token_id=EOS, max_sequence_length=32,
        )


def test_control_and_candidate_schedules_are_deterministic_and_balanced() -> None:
    records = _records()
    kwargs = {
        "seed": 42,
        "per_device_micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
    }
    control = build_schedule(records, group="control", **kwargs)
    candidate = build_schedule(records, group="candidate", **kwargs)

    assert control == build_schedule(records, group="control", **kwargs)
    assert candidate == build_schedule(records, group="candidate", **kwargs)
    control_rows = materialize_schedule(records, control)
    candidate_rows = materialize_schedule(records, candidate)
    control_summary = schedule_summary(control_rows, optimizer_steps=4, accumulation_steps=4)
    candidate_summary = schedule_summary(candidate_rows, optimizer_steps=4, accumulation_steps=4)

    assert control_summary["consumed_unique_sample_ids"] == 16
    assert control_summary["consumed_repeated_draws"] == 0
    assert candidate_summary["consumed_component_counts"] == {
        NATURAL: 8,
        SHORT_EMPTY: 4,
        SHORT_NONEMPTY: 4,
    }
    for start in range(0, 16, 4):
        components = [row["sampling_component"] for row in candidate_rows[start:start + 4]]
        assert components.count(NATURAL) == 2
        assert components.count(SHORT_EMPTY) == 1
        assert components.count(SHORT_NONEMPTY) == 1


def test_candidate_schedule_requires_both_short_history_strata() -> None:
    records = _records()
    for row in records:
        row["gold_profile_empty"] = True

    with pytest.raises(ValueError, match="both short-history"):
        build_schedule(
            records,
            group="candidate",
            seed=42,
            per_device_micro_batch_size=1,
            gradient_accumulation_steps=4,
        )


def test_gpu_preflight_rejects_invalid_run_id_before_gpu(monkeypatch) -> None:
    monkeypatch.setattr(e03_a3_short_train, "_resolve", lambda path: path)
    monkeypatch.setattr(e03_a3_short_train, "_config", lambda path: {"data": {"output_root": "unused"}})
    monkeypatch.setattr(e03_a3_short_train, "_verify_static_assets", lambda config: {"status": "passed"})
    monkeypatch.setattr(
        e03_a3_short_train,
        "_run_gpu_preflight",
        lambda *_args: pytest.fail("invalid Run ID must be rejected before GPU preflight"),
    )

    with pytest.raises(ValueError, match="valid unique --run-id"):
        e03_a3_short_train.main(["--mode", "gpu-preflight", "--run-id", "!"])


def test_run_manifest_writer_serializes_yaml_date_values(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    e03_a3_short_train._write_json(
        manifest_path,
        {"authorization_receipt": {"authorized_on": date(2026, 11, 4)}},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["authorization_receipt"]["authorized_on"] == "2026-11-04"


def test_evaluation_authorization_receipt_matches_json_serialized_yaml_date() -> None:
    receipt = {"scope": "E03-A3", "authorized_on": date(2026, 11, 4)}

    assert e03_a3_short_evaluate._json_compatible(receipt) == {
        "scope": "E03-A3", "authorized_on": "2026-11-04",
    }


def test_reload_recovery_requires_complete_finite_training_trace() -> None:
    step_rows = [
        {
            "optimizer_step": step,
            "microbatch_count": 1,
            "microbatch_losses": [0.5],
            "finite_gradient_nonzero_checked": True,
            "parameter_update_checked": step in {1, 64, 128},
        }
        for step in range(1, 129)
    ]
    trainer_rows = [
        {"optimizer_step": step, "loss": 0.5, "grad_norm": 1.0, "learning_rate": 1e-4}
        for step in range(1, 129)
    ]

    e03_a3_short_train._validate_recovery_optimizer_trace(step_rows, trainer_rows, 128)

    step_rows[80]["microbatch_losses"] = [float("nan")]
    with pytest.raises(ValueError, match="invalid loss/gradient evidence"):
        e03_a3_short_train._validate_recovery_optimizer_trace(step_rows, trainer_rows, 128)


def test_reload_recovery_rejects_missing_parameter_update_audit() -> None:
    step_rows = [
        {
            "optimizer_step": step,
            "microbatch_count": 1,
            "microbatch_losses": [0.5],
            "finite_gradient_nonzero_checked": True,
            "parameter_update_checked": step in {1, 64},
        }
        for step in range(1, 129)
    ]
    trainer_rows = [
        {"optimizer_step": step, "loss": 0.5, "grad_norm": 1.0, "learning_rate": 1e-4}
        for step in range(1, 129)
    ]

    with pytest.raises(ValueError, match="parameter-update evidence"):
        e03_a3_short_train._validate_recovery_optimizer_trace(step_rows, trainer_rows, 128)


def _fake_torch_for_gpu_gate(total_bytes: int, free_bytes: int):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda _device: SimpleNamespace(total_memory=total_bytes),
        mem_get_info=lambda _device: (free_bytes, total_bytes),
        get_device_name=lambda _device: "synthetic GPU",
    )
    return SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="synthetic"))


def test_gpu_gate_allows_small_total_memory_reporting_shortfall() -> None:
    minimum_total = 8 * 1024**3
    reported_total = minimum_total - 4_718_592
    minimum_free = 6 * 1024**3
    torch = _fake_torch_for_gpu_gate(reported_total, minimum_free)

    facts = e03_a3_short_train._gpu_gate(
        {"training": {"min_total_vram_gb": 8, "min_free_vram_gb": 6}},
        torch,
    )

    assert facts["total_vram_bytes"] == reported_total
    assert facts["minimum_total_vram_bytes"] == minimum_total
    assert facts["total_vram_reporting_tolerance_bytes"] == 16 * 1024**2
    assert facts["effective_minimum_total_vram_bytes"] == minimum_total - 16 * 1024**2
    assert facts["free_vram_bytes_before_load"] == minimum_free


def test_gpu_gate_still_rejects_total_memory_below_reporting_tolerance() -> None:
    minimum_total = 8 * 1024**3
    reported_total = minimum_total - 16 * 1024**2 - 1
    torch = _fake_torch_for_gpu_gate(reported_total, 6 * 1024**3)

    with pytest.raises(RuntimeError, match="blocked_total_vram_below_gate"):
        e03_a3_short_train._gpu_gate(
            {"training": {"min_total_vram_gb": 8, "min_free_vram_gb": 6}},
            torch,
        )


def test_gpu_gate_keeps_minimum_free_memory_requirement() -> None:
    minimum_total = 8 * 1024**3
    total = minimum_total - 4_718_592
    minimum_free = 6 * 1024**3
    torch = _fake_torch_for_gpu_gate(total, minimum_free - 1)

    with pytest.raises(RuntimeError, match="blocked_free_vram_below_gate"):
        e03_a3_short_train._gpu_gate(
            {"training": {"min_total_vram_gb": 8, "min_free_vram_gb": 6}},
            torch,
        )


def _case(index: int, cluster: str, candidate_tp: int, control_tp: int) -> dict:
    def model(tp: int) -> dict:
        return {
            "profile_counts": {
                "personality_traits": {"tp": tp, "fp": 0, "fn": 1 - tp},
                "interests": {"tp": 0, "fp": 0, "fn": 0},
                "style": {"tp": 0, "fp": 0, "fn": 0},
            },
            "emotion_correct": bool(tp),
            "raw_json_valid": True,
            "raw_a0_schema_valid": True,
        }

    return {"id": f"eval-{index}", "conversation_id": cluster, "candidate": model(candidate_tp), "control": model(control_tp)}


def test_paired_cluster_bootstrap_is_deterministic_and_reports_delta_ci() -> None:
    cases = [
        _case(1, "c1", 1, 0),
        _case(2, "c1", 1, 0),
        _case(3, "c2", 0, 0),
        _case(4, "c3", 1, 0),
    ]
    groups = {case["id"]: case["conversation_id"] for case in cases}
    first = paired_cluster_bootstrap(cases, groups, "candidate", "control", repetitions=100)
    second = paired_cluster_bootstrap(cases, groups, "candidate", "control", repetitions=100)

    assert first == second
    assert first["cluster_count"] == 3
    assert first["candidate_minus_control"]["profile_micro_f1"] > 0
    low, high = first["cluster_bootstrap_95_percent_intervals"]["profile_micro_f1"]
    assert low <= high
