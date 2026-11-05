from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.training.e03_a6_contract import (
    MICROFIT_EXPERIMENT_ID,
    PILOT_EXPERIMENT_ID,
    canonical_sha256,
    f3_prefix_through_r_marker,
    validate_historical_pilot_selection,
    validate_microfit_config,
    validate_pilot_config,
    validate_pilot_resume_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_microfit_config_freezes_bounded_f3_identity_and_budget():
    config = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_microfit.yaml").read_text())
    validate_microfit_config(config)
    assert config["experiment_id"] == MICROFIT_EXPERIMENT_ID
    assert config["protocol"]["format_version"] == "F3"
    assert config["training"]["max_optimizer_steps"] == 32
    assert config["training"]["sample_count"] == 4
    assert config["training"]["max_training_seconds"] == 1200


@pytest.mark.parametrize("mutation", [
    {"experiment_id": "E01-B-PILOT"},
    {"training_scope": "full"},
    {"training.max_optimizer_steps": 4048},
    {"protocol.format_version": "F2"},
    {"evaluation.autocast_adapter_dtype": False},
])
def test_microfit_rejects_other_scope_protocol_or_budget(mutation):
    config = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_microfit.yaml").read_text())
    changed = copy.deepcopy(config)
    if "." in next(iter(mutation)):
        path, value = next(iter(mutation.items()))
        parent, key = path.split(".")
        changed[parent][key] = value
    else:
        changed.update(mutation)
    with pytest.raises(ValueError):
        validate_microfit_config(changed)


def test_historical_pilot_selection_checks_ordered_rows_and_sha():
    rows = [{
        "sample_id": f"train:conv_{index:04d}:pp0",
        "conversation_id": f"conv_{index:04d}",
    } for index in range(512)]
    selection = {
        "selection_rule": (
            "seeded_length_deciles_v1; train-only; deterministic seed hash within deciles; "
            "all Train emotion labels, both profile-empty states, and min/max sequence lengths included"
        ),
        "rows": rows,
        "sha256": canonical_sha256(rows),
    }
    assert validate_historical_pilot_selection(selection) == [row["sample_id"] for row in rows]

    changed = copy.deepcopy(selection)
    changed["rows"][1], changed["rows"][2] = changed["rows"][2], changed["rows"][1]
    with pytest.raises(ValueError, match="SHA"):
        validate_historical_pilot_selection(changed)


def test_microfit_cli_has_no_full_training_mode():
    from scripts.e03_a6_train import _build_parser, main

    parser = _build_parser()
    assert parser._option_string_actions["--mode"].choices == (
        "microfit", "pilot-dry-step", "pilot-train", "resume", "reload-check",
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "full-train", "--run-id", "E03-A6-F3-FULL-UNAPPROVED"])
    with pytest.raises(SystemExit) as error:
        main(["--mode", "pilot-train", "--run-id", "E03-A6-F3-PILOT-UNAPPROVED"])
    assert error.value.code == 2


def test_pilot_config_freezes_historical_ids_optimizer_and_128_step_budget():
    config = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_pilot.yaml").read_text())
    validate_pilot_config(config)
    assert config["experiment_id"] == PILOT_EXPERIMENT_ID
    assert config["training"]["sample_selection"]["count"] == 512
    assert config["training"]["pause_after_step"] == 32
    assert config["training"]["max_optimizer_steps"] == 128
    assert config["evaluation"]["pilot_dev_generation_rows"] == 160


@pytest.mark.parametrize("mutation", [
    {"experiment_id": "E01-B-PILOT"},
    {"training.max_optimizer_steps": 4048},
    {"training.pause_after_step": 64},
    {"evaluation.autocast_adapter_dtype": False},
    {"data.expected_historical_selection_sha256": "0" * 64},
])
def test_pilot_config_rejects_scope_budget_dtype_or_sample_drift(mutation):
    config = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_pilot.yaml").read_text())
    changed = copy.deepcopy(config)
    if "." in next(iter(mutation)):
        path, value = next(iter(mutation.items()))
        parent, key = path.split(".")
        changed[parent][key] = value
    else:
        changed.update(mutation)
    with pytest.raises(ValueError):
        validate_pilot_config(changed)


def test_pilot_resume_requires_its_own_exact_checkpoint_32_manifest():
    manifest = {
        "run_id": "E03-A6-F3-PILOT-20261105-01",
        "experiment_id": PILOT_EXPERIMENT_ID,
        "training_scope": "pilot",
        "status": "interrupted",
        "stage": "checkpoint_saved",
        "optimizer_steps": 32,
        "max_optimizer_steps": 128,
        "stop_reason": "requested_checkpoint_pause",
        "last_checkpoint": "outputs/training/E03-A6-F3-PILOT-20261105-01/trainer/checkpoint-32",
        "full_training_status": "not_authorized_not_run",
    }
    validate_pilot_resume_manifest(
        manifest,
        run_id=manifest["run_id"],
        checkpoint_name="checkpoint-32",
    )
    changed = copy.deepcopy(manifest)
    changed["last_checkpoint"] = "outputs/training/E01-B-PILOT/trainer/checkpoint-32"
    with pytest.raises(ValueError):
        validate_pilot_resume_manifest(
            changed,
            run_id=manifest["run_id"],
            checkpoint_name="checkpoint-32",
        )


def test_f3_protocol_reload_prefix_stops_after_actual_r_marker_tokens():
    first = list(map(ord, "E=sadness\nP=-\nI=-\nS=brief\nR=回答一"))
    second = list(map(ord, "E=sadness\nP=-\nI=-\nS=brief\nR=回答二"))
    decode = lambda ids: "".join(map(chr, ids))
    first_prefix, first_text = f3_prefix_through_r_marker(first, decode)
    second_prefix, second_text = f3_prefix_through_r_marker(second, decode)
    assert first_prefix == second_prefix
    assert first_text == second_text == "E=sadness\nP=-\nI=-\nS=brief\nR="
    with pytest.raises(ValueError, match="R= boundary"):
        f3_prefix_through_r_marker(list(map(ord, "E=sadness\nP=-\nI=-\nS=brief")), decode)
