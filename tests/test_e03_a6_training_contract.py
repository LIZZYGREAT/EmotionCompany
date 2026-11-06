from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.training.e03_a6_contract import (
    MICROFIT_EXPERIMENT_ID,
    PILOT_EXPERIMENT_ID,
    canonical_sha256,
    validate_expanded_pilot_config,
    validate_expanded_pilot_resume_manifest,
    f3_prefix_through_r_marker,
    validate_historical_pilot_selection,
    validate_microfit_config,
    validate_pilot_config,
    validate_pilot_resume_manifest,
)
from src.training.e03_a6_pilot import (
    PRECHECKPOINT_PILOT_MODULE_SHA256,
    PRECHECKPOINT_TRAINING_CODE_SHA256,
    RECOVERABLE_RESUME_FAILURE,
    _adopt_checkpoint_compatible_resume_code,
    _recover_pretrain_resume_failure,
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


def test_expanded_pilot_config_freezes_nested_1024_rows_and_256_steps():
    config = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_expanded_pilot.yaml").read_text())
    validate_expanded_pilot_config(config)
    from scripts.e03_a6_f3_pilot_evaluate import _validate_training_config

    _validate_training_config(config)
    assert config["experiment_id"] == "E03-A6-F3-PILOT-EXPANDED"
    assert config["training_scope"] == "supplemental_pilot"
    assert config["training"]["sample_selection"]["count"] == 1024
    assert config["training"]["sample_selection"]["base_count"] == 512
    assert config["training"]["max_optimizer_steps"] == 256
    assert config["training"]["pause_after_step"] == 64


def test_a6_f3_adapter_dtype_is_preserved_for_both_pilot_scopes_only():
    from src.training.training_sft import _uses_a6_f3_adapter_dtype

    baseline = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_pilot.yaml").read_text())
    expanded = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_expanded_pilot.yaml").read_text())
    full_field = yaml.safe_load((ROOT / "configs/training/e03_a6_f4_full_field_pilot.yaml").read_text())
    assert _uses_a6_f3_adapter_dtype(baseline)
    assert _uses_a6_f3_adapter_dtype(expanded)
    assert _uses_a6_f3_adapter_dtype(full_field)
    assert not _uses_a6_f3_adapter_dtype({"experiment_id": "E01-B-PILOT", "training_scope": "pilot"})


@pytest.mark.parametrize("mutation", [
    {"training.max_optimizer_steps": 4048},
    {"training.sample_selection.count": 2048},
    {"training.pause_after_step": 128},
    {"training_scope": "full"},
])
def test_expanded_pilot_rejects_scope_or_budget_drift(mutation):
    config = yaml.safe_load((ROOT / "configs/training/e03_a6_f3_expanded_pilot.yaml").read_text())
    changed = copy.deepcopy(config)
    for path, value in mutation.items():
        if "." in path:
            parent_path, key = path.rsplit(".", 1)
            parent = changed
            for component in parent_path.split("."):
                parent = parent[component]
            parent[key] = value
        else:
            changed[path] = value
    with pytest.raises(ValueError):
        validate_expanded_pilot_config(changed)


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


def test_expanded_pilot_resume_requires_its_own_checkpoint_64():
    run_id = "E03-A6-F3-PILOT-EXPANDED-20261106-01"
    manifest = {
        "run_id": run_id,
        "experiment_id": "E03-A6-F3-PILOT-EXPANDED",
        "training_scope": "supplemental_pilot",
        "status": "interrupted",
        "stage": "checkpoint_saved",
        "optimizer_steps": 64,
        "max_optimizer_steps": 256,
        "stop_reason": "requested_checkpoint_pause",
        "last_checkpoint": f"outputs/training/{run_id}/trainer/checkpoint-64",
        "full_training_status": "not_authorized_not_run",
    }
    validate_expanded_pilot_resume_manifest(
        manifest, run_id=run_id, checkpoint_name="checkpoint-64",
    )
    changed = copy.deepcopy(manifest)
    changed["last_checkpoint"] = "outputs/training/other-run/trainer/checkpoint-64"
    with pytest.raises(ValueError):
        validate_expanded_pilot_resume_manifest(
            changed, run_id=run_id, checkpoint_name="checkpoint-64",
        )


def test_pilot_resume_records_only_the_frozen_checkpoint_compatible_code_repair():
    old_files = {
        "scripts/e03_a6_train.py": "1" * 64,
        "src/training/e03_a6_pilot.py": PRECHECKPOINT_PILOT_MODULE_SHA256,
        "src/training/e03_a6_contract.py": "2" * 64,
    }
    current_files = {**old_files, "src/training/e03_a6_pilot.py": "3" * 64}
    source_identity = {
        "training_code_files": current_files,
        "training_code_sha256": canonical_sha256(current_files),
    }
    manifest = {
        "run_id": "E03-A6-F3-PILOT-20261106-01",
        "training_code_files": old_files,
        "training_code_sha256": PRECHECKPOINT_TRAINING_CODE_SHA256,
        "checkpoint_identity": {
            "training_code_sha256": PRECHECKPOINT_TRAINING_CODE_SHA256,
        },
    }
    assert _adopt_checkpoint_compatible_resume_code(source_identity, manifest) == PRECHECKPOINT_TRAINING_CODE_SHA256
    assert manifest["training_code_at_checkpoint"]["sha256"] == PRECHECKPOINT_TRAINING_CODE_SHA256
    assert manifest["resume_code_compatibility"]["changed_files"] == ["src/training/e03_a6_pilot.py"]
    assert manifest["resume_code_compatibility"]["checkpoint_identity_preserved"] is True
    assert manifest["training_code_files"] == current_files
    assert _adopt_checkpoint_compatible_resume_code(source_identity, manifest) == PRECHECKPOINT_TRAINING_CODE_SHA256

    unsafe_manifest = {
        "run_id": "E03-A6-F3-PILOT-20261106-01",
        "training_code_files": old_files,
        "training_code_sha256": PRECHECKPOINT_TRAINING_CODE_SHA256,
        "checkpoint_identity": {"training_code_sha256": PRECHECKPOINT_TRAINING_CODE_SHA256},
    }
    unsafe_files = {**current_files, "src/training/training_sft.py": "4" * 64}
    with pytest.raises(ValueError, match="compatibility repair"):
        _adopt_checkpoint_compatible_resume_code(
            {"training_code_files": unsafe_files, "training_code_sha256": canonical_sha256(unsafe_files)},
            unsafe_manifest,
        )


def test_pilot_resume_recovers_only_pretrain_failure_with_verified_checkpoint32(tmp_path):
    from src.training.training_sft import _write_checkpoint_manifest

    run_id = "E03-A6-F3-PILOT-20261106-01"
    run_root = tmp_path / run_id
    checkpoint = run_root / "trainer" / "checkpoint-32"
    checkpoint.mkdir(parents=True)
    for name in (
        "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json",
        "adapter_model.safetensors",
    ):
        (checkpoint / name).write_bytes(b"verified checkpoint fixture")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 32}))
    checkpoint_identity = {"run_id": run_id, "training_code_sha256": PRECHECKPOINT_TRAINING_CODE_SHA256}
    _write_checkpoint_manifest(
        checkpoint,
        {**checkpoint_identity, "optimizer_step": 32},
    )
    rows = [{
        "optimizer_step": step,
        "loss": 1.0,
        "finite_gradients": True,
        "nonzero_gradient_parameter_count": 1,
        "parameter_update_checked": True,
        "updated_parameter_count": 1,
    } for step in range(1, 33)]
    (run_root / "train_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = {
        "run_id": run_id,
        "status": "failed",
        "stage": "training_failed",
        "optimizer_steps": 0,
        "error": RECOVERABLE_RESUME_FAILURE,
        "last_checkpoint": str(checkpoint.resolve()),
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checkpoint_identity": checkpoint_identity,
        "training_attempts": [{"ending_optimizer_step": 32}],
    }
    assert _recover_pretrain_resume_failure(run_root, manifest) is True
    assert manifest["status"] == "interrupted"
    assert manifest["optimizer_steps"] == 32
    assert manifest["recovered_pretrain_resume_failure"]["optimizer_steps_after_checkpoint"] == 0
    assert manifest["recovered_pretrain_resume_failure"]["verified_log_rows"] == 32

    manifest.update(
        status="failed",
        stage="training_failed",
        optimizer_steps=0,
        error="RuntimeError: partial optimizer update",
    )
    with pytest.raises(ValueError, match="recoverable pre-train"):
        _recover_pretrain_resume_failure(run_root, manifest)


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
