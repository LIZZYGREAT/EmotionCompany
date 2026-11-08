#!/usr/bin/env python3
"""Verify frozen final-dual assets and create a hash-locked deployment copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}
SOURCE_BRANCH = "origin/chore/e03-b-p2-twofield"
REPLAY_REPORT = "e03-b/results/replay/E03-B_replay_难例回放实验报告.md"
RUN_NOTE = "e03-b/results/replay/E03B-RUN-NOTE.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def adapter_facts(config: dict[str, Any], name: str, adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file() or weights_path.stat().st_size <= 0:
        raise FileNotFoundError(f"{name} Adapter files are missing or empty under {adapter_dir}")
    actual_weight_sha = sha256_file(weights_path)
    actual_config_sha = sha256_file(config_path)
    if actual_weight_sha != config["expected_weight_sha256"]:
        raise ValueError(f"{name} Adapter weight SHA-256 differs from the frozen deployment config")
    if actual_config_sha != config["expected_config_sha256"]:
        raise ValueError(f"{name} Adapter config SHA-256 differs from the frozen deployment config")

    adapter_config = json_file(config_path)
    modules = adapter_config.get("target_modules")
    if not isinstance(modules, list) or set(modules) != EXPECTED_TARGETS or len(modules) != len(EXPECTED_TARGETS):
        raise ValueError(f"{name} Adapter target modules are not the frozen seven-module LoRA set")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"{name} Adapter is not a causal-LM LoRA")
    if adapter_config.get("r") != 8 or adapter_config.get("lora_alpha") != 16:
        raise ValueError(f"{name} Adapter must have r=8 and alpha=16")
    if adapter_config.get("lora_dropout") != 0.0 or adapter_config.get("bias") != "none":
        raise ValueError(f"{name} Adapter dropout/bias differs from the frozen training protocol")
    if adapter_config.get("modules_to_save") or adapter_config.get("trainable_token_indices"):
        raise ValueError(f"{name} Adapter contains saved modules or trainable token indices")
    if adapter_config.get("use_dora") or adapter_config.get("use_rslora") or adapter_config.get("use_qalora"):
        raise ValueError(f"{name} Adapter enables a non-frozen LoRA variant")

    return {
        "adapter_name": name,
        "adapter_dir": str(adapter_dir),
        "weight_sha256": actual_weight_sha,
        "config_sha256": actual_config_sha,
        "weight_size_bytes": weights_path.stat().st_size,
        "source_base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        "peft": {
            key: adapter_config.get(key)
            for key in (
                "peft_type", "task_type", "r", "lora_alpha", "lora_dropout",
                "target_modules", "bias", "modules_to_save", "trainable_token_indices",
            )
        },
    }


def copy_deployment_adapter(
    source_dir: Path, destination: Path, *, base_model_path: Path, rewrite_base_path: bool
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"deployment Adapter output already exists: {destination}")
    destination.mkdir(parents=True)
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(source_dir / filename, destination / filename)
    config_path = destination / "adapter_config.json"
    config = json_file(config_path)
    if rewrite_base_path:
        config["base_model_name_or_path"] = str(base_model_path.resolve())
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(destination.relative_to(ROOT)),
        "weight_sha256": sha256_file(destination / "adapter_model.safetensors"),
        "config_sha256": sha256_file(config_path),
        "base_path_rewritten": rewrite_base_path,
        "base_model_name_or_path": config.get("base_model_name_or_path"),
    }


def e03b_source_facts() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", SOURCE_BRANCH], cwd=ROOT, text=True
    ).strip()
    report_path = ROOT / REPLAY_REPORT
    note_path = ROOT / RUN_NOTE
    training_config = ROOT / "configs/training/e03_b_p2_twofield.yaml"
    report = report_path.read_text(encoding="utf-8")
    note = note_path.read_text(encoding="utf-8")
    config_text = training_config.read_text(encoding="utf-8")
    if "1152" not in report or "选 1152" not in report:
        raise ValueError("E03-B replay report does not identify Replay-1152 as the selected checkpoint")
    if "从 4665 adapter 续训" not in note or "选 1152" not in note:
        raise ValueError("E03-B run note does not verify Replay-1152 as a continuation from 4665")
    prompt_sha_match = re.search(r"(?m)^\s*prompt_sha256:\s*([0-9a-f]{64})\s*$", config_text)
    if prompt_sha_match is None:
        raise ValueError("E03-B training config is missing its locked prompt SHA-256")
    inference_script = subprocess.check_output(
        ["git", "show", f"{SOURCE_BRANCH}:scripts/e03b/infer_209.py"], cwd=ROOT
    )
    return {
        "source_branch": SOURCE_BRANCH,
        "source_commit": commit,
        "training_config": {
            "path": str(training_config.relative_to(ROOT)),
            "sha256": sha256_file(training_config),
            "prompt_sha256": prompt_sha_match.group(1),
        },
        "replay_report": {"path": str(report_path.relative_to(ROOT)), "sha256": sha256_file(report_path)},
        "run_note": {"path": str(note_path.relative_to(ROOT)), "sha256": sha256_file(note_path)},
        "infer_209_source": {
            "path": "scripts/e03b/infer_209.py",
            "sha256": hashlib.sha256(inference_script).hexdigest(),
            "history_policy": "P0+T1 selection then E03-B prompt construction",
        },
    }


def make_preflight(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json_file(config_path)
    if config.get("mode") != "dual_adapter":
        raise ValueError("deployment config mode must be dual_adapter")
    if output_dir.exists():
        raise FileExistsError(f"Run output already exists; refusing to reuse it: {output_dir}")
    output_dir.mkdir(parents=True)
    facts: dict[str, Any] = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_porcelain_sha256": hashlib.sha256(
            subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT)
        ).hexdigest(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "candidate_root": str(resolve_path(config["candidates_root"])),
        "checks": {},
    }
    try:
        model_path = resolve_path(config["model_path"])
        if not model_path.is_dir():
            raise FileNotFoundError(f"locked Base model directory is missing: {model_path}")

        from participant import run_inference as runner

        base_config = {
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "model_path": "model",
            "tokenizer": {
                **config["tokenizer"],
                "local_path": "model",
                "local_files_only": True,
            },
        }
        base_identity = runner._weight_identity(model_path, base_config)
        if base_identity.get("observed_weight_hash") != config["expected_base_weight_sha256"]:
            raise ValueError("local Base weight SHA-256 differs from the frozen revision identity")
        tokenizer, tokenizer_identity = runner.load_runtime_tokenizer(base_config)
        if tokenizer_identity.get("signature_sha256") != config["tokenizer"]["expected_signature_sha256"]:
            raise ValueError("local Tokenizer Signature differs from the frozen A8/E03-B training identity")
        if tokenizer_identity.get("template_sha256") != config["tokenizer"]["expected_template_sha256"]:
            raise ValueError("local Qwen Chat Template differs from the frozen training identity")
        if tokenizer.eos_token_id != config["tokenizer"]["eos_token_id"]:
            raise ValueError("local Tokenizer EOS token differs from the frozen training identity")
        facts["base"] = base_identity
        facts["tokenizer"] = tokenizer_identity
        facts["checks"]["base_and_tokenizer"] = "passed"

        prompt_facts = {}
        for key, prompt_info in config["prompts"].items():
            prompt_path = resolve_path(prompt_info["path"])
            actual = sha256_file(prompt_path)
            if actual != prompt_info["sha256"]:
                raise ValueError(f"{key} Prompt SHA-256 differs from the frozen lock")
            prompt_facts[key] = {"path": str(prompt_path.relative_to(ROOT)), "sha256": actual}
        e03b_source = e03b_source_facts()
        if e03b_source["training_config"]["prompt_sha256"] != prompt_facts["emotion_response"]["sha256"]:
            raise ValueError("E03-B source training config and deployed Prompt hash differ")
        facts["prompts"] = prompt_facts
        facts["e03b_source"] = e03b_source
        facts["checks"]["prompt_and_training_provenance"] = "passed"

        profile_config = config["adapters"]["profile"]
        profile_dir = resolve_path(profile_config["path"])
        profile = adapter_facts(profile_config, "profile", profile_dir)
        candidate_manifest_path = resolve_path("outputs/final_candidates/A8-P4504/source_manifest.json")
        candidate_manifest = json_file(candidate_manifest_path)
        source_lock = candidate_manifest.get("source_lock", {})
        if source_lock.get("run_id") != "E03-A8-DEVFT-20261108-02-PDEV-1EPOCH":
            raise ValueError("A8 source archive does not identify the frozen P4504 run")
        if source_lock.get("optimizer_step") != 4504:
            raise ValueError("A8 source archive does not identify optimizer step 4504")
        if source_lock.get("base_weight_sha256") != config["expected_base_weight_sha256"]:
            raise ValueError("A8 source archive Base hash differs from the locked Base")
        manifest_files = {item["path"]: item["sha256"] for item in candidate_manifest.get("files", [])}
        for relative, actual in (
            ("adapter/adapter_model.safetensors", profile["weight_sha256"]),
            ("adapter/adapter_config.json", profile["config_sha256"]),
        ):
            if manifest_files.get(relative) != actual:
                raise ValueError(f"A8 archive manifest hash mismatch for {relative}")
        profile["archive_manifest_sha256"] = sha256_file(candidate_manifest_path)
        profile["archive_source_lock"] = source_lock

        er_config = config["adapters"]["emotion_response"]
        er_dir = resolve_path(er_config["path"])
        er_checkpoint = resolve_path(er_config["checkpoint_path"])
        emotion_response = adapter_facts(er_config, "emotion_response", er_dir)
        checkpoint_state = json_file(er_checkpoint / "trainer_state.json")
        if checkpoint_state.get("global_step") != 1152 or checkpoint_state.get("max_steps") != 2219:
            raise ValueError("A8-ER checkpoint trainer state does not match replay-local step 1152/2219")
        checkpoint_weight = er_checkpoint / "adapter_model.safetensors"
        checkpoint_config_path = er_checkpoint / "adapter_config.json"
        if sha256_file(checkpoint_weight) != emotion_response["weight_sha256"]:
            raise ValueError("A8-ER adapter/ checkpoint-1152 weights are not identical")
        if sha256_file(checkpoint_config_path) != emotion_response["config_sha256"]:
            raise ValueError("A8-ER adapter/ checkpoint-1152 configs are not identical")
        emotion_response["checkpoint"] = {
            "path": str(er_checkpoint),
            "trainer_state_sha256": sha256_file(er_checkpoint / "trainer_state.json"),
            "trainer_state_global_step": checkpoint_state["global_step"],
            "trainer_state_max_steps": checkpoint_state["max_steps"],
            "trainer_state_epoch": checkpoint_state.get("epoch"),
            "trainer_state_mtime_utc": datetime.fromtimestamp(
                (er_checkpoint / "trainer_state.json").stat().st_mtime, timezone.utc
            ).isoformat(),
            "adapter_weights_match_candidate_root": True,
        }
        facts["adapters"] = {"profile": profile, "emotion_response": emotion_response}
        facts["checks"]["adapter_files_and_provenance"] = "passed"

        deployment_root = output_dir / "deployment/adapters"
        deployment_root.mkdir(parents=True)
        deployed = {
            "profile": copy_deployment_adapter(
                profile_dir, deployment_root / "profile", base_model_path=model_path, rewrite_base_path=False
            ),
            "emotion_response": copy_deployment_adapter(
                er_dir,
                deployment_root / "emotion-response",
                base_model_path=model_path,
                rewrite_base_path=True,
            ),
        }
        facts["deployment_adapters"] = deployed
        facts["checks"]["deployment_copy"] = "passed"

        model_lock = {
            "version": "final-dual-model-lock-v1",
            "run_id": config["run_id"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": facts["git_head"],
            "deployment_config": {"path": str(config_path.relative_to(ROOT)), "sha256": facts["config_sha256"]},
            "base": {
                "model_id": config["model_id"],
                "revision": config["model_revision"],
                "expected_weight_sha256": config["expected_base_weight_sha256"],
                "observed": base_identity,
            },
            "tokenizer": tokenizer_identity,
            "chat_template_sha256": tokenizer_identity["template_sha256"],
            "prompts": prompt_facts,
            "adapters": facts["adapters"],
            "deployment_adapters": deployed,
            "e03b_provenance": e03b_source,
            "generation": config["generation"],
            "context": config["context"],
            "quantization": config["quantization"],
            "asset_root_user_confirmed": "outputs/final_candidates/A8-P4504 and outputs/final_candidates/A8-ER",
            "original_adapter_assets_modified": False,
        }
        lock_path = output_dir / "model_lock.json"
        lock_path.write_text(json.dumps(model_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        facts["model_lock"] = {
            "path": str(lock_path.relative_to(ROOT)),
            "sha256": sha256_file(lock_path),
        }
        facts["status"] = "passed"
        facts["failure"] = None
        (output_dir / "asset_preflight.json").write_text(
            json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return facts
    except Exception as exc:
        facts["status"] = "failed"
        facts["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        (output_dir / "asset_preflight.json").write_text(
            json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deployment/final_dual.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir or (resolve_path(json_file(config_path)["model_lock_path"]).parent)
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    try:
        facts = make_preflight(config_path, output_dir)
    except Exception as exc:
        print(f"FAILED asset preflight: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"status": facts["status"], "model_lock": facts["model_lock"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
