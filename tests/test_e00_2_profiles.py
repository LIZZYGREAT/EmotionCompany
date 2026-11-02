from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import e00_2_smoke
from src.evaluation.prepare_dev import adapter_config_identity


ROOT = Path(__file__).resolve().parents[1]


def test_profile_snapshots_keep_protocol_and_dataset_identity_and_match_generation(tmp_path):
    rendered = e00_2_smoke.render_profiles("test-profile-render", output_root=tmp_path)
    base_evaluation = yaml.safe_load((ROOT / "configs/evaluation/e00.yaml").read_text(encoding="utf-8"))
    rendered_evaluations = {}
    for name, profile in rendered["profiles"].items():
        runner_config = json.loads(Path(profile["runner_config_path"]).read_text(encoding="utf-8"))
        evaluation_config = yaml.safe_load(Path(profile["evaluation_config_path"]).read_text(encoding="utf-8"))
        rendered_evaluations[name] = evaluation_config

        assert runner_config["generation"] == evaluation_config["generation"]
        assert runner_config["experiment_id"] == evaluation_config["experiment_id"] == "E00.2"
        assert runner_config["quantization"]["enabled"] is True
        assert evaluation_config["quantization"]["planned"] == {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "use_double_quant": True,
        }
        assert evaluation_config["protocol"] == base_evaluation["protocol"]
        assert evaluation_config["data"] == base_evaluation["data"]
        assert evaluation_config["scoring"] == base_evaluation["scoring"]
        assert profile["runner_config_sha256"] == e00_2_smoke._sha256_file(Path(profile["runner_config_path"]))
        assert profile["evaluation_config_sha256"] == e00_2_smoke._sha256_file(Path(profile["evaluation_config_path"]))

    greedy = rendered_evaluations["greedy"]["generation"]
    sampling = rendered_evaluations["sampling"]["generation"]
    assert greedy == {"max_new_tokens": 1200, "do_sample": False, "temperature": 1.0, "top_p": 1.0}
    assert sampling == {
        "max_new_tokens": 1200,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    assert rendered["profiles"]["sampling"]["sampling_base_seed"] == 42
    assert rendered["profiles"]["greedy"]["sampling_base_seed"] is None

    base_identity = adapter_config_identity(base_evaluation)[1]
    assert all(adapter_config_identity(value)[1] == base_identity for value in rendered_evaluations.values())


def test_profile_renderer_refuses_existing_directory_and_unsafe_run_ids(tmp_path):
    e00_2_smoke.render_profiles("safe-id", output_root=tmp_path)
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        e00_2_smoke.render_profiles("safe-id", output_root=tmp_path)
    for run_id in ("../escape", ".hidden", "contains space", ""):
        with pytest.raises(ValueError, match="run-id"):
            e00_2_smoke.render_profiles(run_id, output_root=tmp_path)


def test_local_model_mount_gitignore_matches_model_path():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "participant/model"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
