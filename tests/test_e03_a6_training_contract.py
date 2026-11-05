from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.training.e03_a6_contract import (
    MICROFIT_EXPERIMENT_ID,
    canonical_sha256,
    validate_historical_pilot_selection,
    validate_microfit_config,
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
            "all Train emotion labels, both profile-empty states, and min/max sequence lengths"
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
    from scripts.e03_a6_train import _build_parser

    parser = _build_parser()
    assert parser._option_string_actions["--mode"].choices == ("microfit",)
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "full-train", "--run-id", "E03-A6-F3-FULL-UNAPPROVED"])
