#!/usr/bin/env python3
"""Prepare and run the isolated, paired E03-A5 SFT/BCE experiment.

CPU schedule preparation is safe by itself. Every GPU backward or optimizer
run requires a fresh stage/group-specific ticket and a reviewed A5.1 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e03_a5_probe as probe_runner
from src.data.e03_a5_labels import canonical_sha256, iter_jsonl, sha256_file, validate_sft_row
from src.training.e03_a5_contract import (
    RunAuthorizationError,
    validate_paired_group_contract,
    verify_lambda_freeze,
    verify_probe_gate,
    verify_run_ticket,
)
from src.training.e03_a5_multitask import (
    FinalHiddenCapture,
    assert_finite_nonzero_gradients,
    build_a5_train_schedule,
    combine_sft_bce_losses,
    completion_only_sft_loss,
    forward_with_prompt_end_hidden,
    masked_bce_with_logits,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_facts() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _run_root(config: Mapping[str, Any]) -> Path:
    root = probe_runner._resolve(config["outputs"]["training_root"])
    if not root.is_relative_to(probe_runner._resolve("outputs")):
        raise ValueError("A5 training artifacts must remain under ignored outputs/")
    return root


def _validate_id(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ValueError(f"{name} must be a simple filesystem-safe Run ID")
    return value


def _audit(config: Mapping[str, Any], config_sha256: str, audit_run_id: str) -> dict[str, Any]:
    _validate_id(audit_run_id, "audit_run_id")
    return probe_runner._verify_audit_run(config, config_sha256, audit_run_id)


def _pair_paths(config: Mapping[str, Any], pair_id: str) -> dict[str, Path]:
    _validate_id(pair_id, "pair_id")
    root = _run_root(config) / "pairs" / pair_id
    return {"root": root, "schedule": root / "schedule.json", "manifest": root / "pair_manifest.json"}


def _load_pair(
    config: Mapping[str, Any], config_sha256: str, audit: Mapping[str, Any], pair_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    paths = _pair_paths(config, pair_id)
    manifest = probe_runner._json(paths["manifest"])
    if manifest.get("status") != "paired_schedule_prepared" or manifest.get("pair_id") != pair_id:
        raise ValueError("A5 pair manifest is not a prepared immutable schedule")
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("A5 pair manifest uses a different frozen config")
    if manifest.get("audit_manifest_sha256") != audit["manifest_sha256"]:
        raise ValueError("A5 pair manifest does not bind the passed A5.0 audit")
    if manifest.get("sft_train_sha256") != config["identity"]["sft_train_sha256"]:
        raise ValueError("A5 pair manifest does not bind the frozen SFT Train source")
    if sha256_file(paths["schedule"]) != manifest.get("schedule_sha256"):
        raise ValueError("A5 pair schedule SHA differs from its immutable manifest")
    schedule = probe_runner._json(paths["schedule"])
    if (
        schedule.get("source_split") != "train_fit"
        or schedule.get("unique_draws") is not True
        or schedule.get("draw_count") != int(config["training_proposal"]["sample_draw_count"])
        or schedule.get("schedule_sha256") != manifest.get("train_schedule_sha256")
        or schedule.get("sample_ids_sha256") != manifest.get("sample_ids_sha256")
    ):
        raise ValueError("A5 pair schedule violates the frozen Fit-only draw contract")
    return manifest, schedule, paths["root"]


def _prepare_pair(args: argparse.Namespace) -> int:
    config, config_sha = probe_runner._load_config(probe_runner._resolve(args.config))
    audit = _audit(config, config_sha, args.audit_run_id)
    supervision = probe_runner._read_supervision(audit["supervision"])
    split_ids = probe_runner._read_splits(audit["splits"])
    fit_ids = split_ids["fit"]
    proposal = config["training_proposal"]
    if int(proposal["per_device_micro_batch_size"]) != 1:
        raise ValueError("A5 v1 currently freezes micro_batch_size=1")
    schedule = build_a5_train_schedule(
        fit_ids,
        optimizer_steps=int(proposal["max_optimizer_steps"]),
        micro_batch_size=int(proposal["per_device_micro_batch_size"]),
        gradient_accumulation_steps=int(proposal["gradient_accumulation_steps"]),
        seed=int(proposal["seed"]),
    )
    if schedule["draw_count"] != int(proposal["sample_draw_count"]):
        raise ValueError("A5 frozen schedule draw count differs from the versioned config")
    schedule_ids = set(schedule["sample_ids"])
    if not schedule_ids.issubset(set(fit_ids)) or schedule_ids - set(supervision):
        raise ValueError("A5 schedule contains an ID outside supervised Train Fit")

    sft_path = probe_runner._resolve(config["data"]["sft_train"])
    if sha256_file(sft_path) != config["identity"]["sft_train_sha256"]:
        raise ValueError("SFT Train hash differs from the frozen A5 config")
    seen: set[str] = set()
    lengths: dict[str, int] = {}
    row_count = 0
    for line_number, row, _row_sha in iter_jsonl(sft_path):
        row_count += 1
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"SFT row {line_number} has a missing or duplicate sample_id")
        if row.get("source_split") != "train":
            raise ValueError(f"{sample_id}: only Train SFT rows may enter the schedule")
        if sample_id in schedule_ids:
            facts = validate_sft_row(row, max_sequence_length=int(config["identity"]["max_train_sequence_length"]))
            if int(facts["prompt_length"]) > int(config["identity"]["max_prompt_tokens"]):
                raise ValueError(f"{sample_id}: Prompt exceeds the frozen P=1280 budget")
            lengths[sample_id] = int(facts["sequence_length"])
        seen.add(sample_id)
    if row_count != int(config["data"]["expected_sft_train_rows"]) or seen != set(supervision):
        raise ValueError("SFT Train IDs do not exactly match the frozen A5.0 supervision set")
    if set(lengths) != schedule_ids:
        raise ValueError("one or more frozen schedule IDs are absent from the SFT Train source")
    dry_sample_id = min(schedule_ids, key=lambda sample_id: (-lengths[sample_id], sample_id))
    if any(length > int(config["identity"]["max_train_sequence_length"]) for length in lengths.values()):
        raise ValueError("A5 schedule contains a sequence above the frozen S=2048 ceiling")

    contract_common = {
        "source_commit": config["identity"]["source_commit"],
        "config_sha256": config_sha,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "supervision_sha256": audit["manifest_value"]["outputs"]["supervision.jsonl"]["sha256"],
        "split_manifest_sha256": audit["manifest_value"]["outputs"]["conversation_splits.json"]["sha256"],
        "base_weight_sha256": config["identity"]["base_weight_sha256"],
        "initial_adapter_weight_sha256": config["identity"]["initial_adapter_weight_identity_sha256"],
        "initial_adapter_config_sha256": config["identity"]["initial_adapter_config_sha256"],
        "train_schedule_sha256": schedule["schedule_sha256"],
        "sample_ids_sha256": schedule["sample_ids_sha256"],
        "optimizer": proposal["optimizer"],
        "learning_rate": float(proposal["learning_rate"]),
        "scheduler": proposal["scheduler"],
        "weight_decay": float(proposal["weight_decay"]),
        "optimizer_steps": int(proposal["max_optimizer_steps"]),
        "micro_batch_size": int(proposal["per_device_micro_batch_size"]),
        "gradient_accumulation_steps": int(proposal["gradient_accumulation_steps"]),
        "seed": int(proposal["seed"]),
        "protocol_sha256": canonical_sha256(config["identity"]["protocol"]),
    }
    control = {**contract_common, "group": "control_sft", "bce_lambda": 0.0}
    candidate = {
        **contract_common,
        "group": "candidate_sft_bce",
        "bce_lambda": float(proposal["candidate_lambda"]),
    }
    pair_contract = validate_paired_group_contract(control, candidate)
    paths = _pair_paths(config, args.pair_id)
    paths["root"].mkdir(parents=True, exist_ok=False)
    _write_once(paths["schedule"], schedule)
    implementation_paths = (
        Path(__file__).resolve(),
        ROOT / "src/training/e03_a5_multitask.py",
        ROOT / "src/training/e03_a5_contract.py",
    )
    manifest = {
        "status": "paired_schedule_prepared",
        "experiment_id": config["experiment_id"],
        "pair_id": args.pair_id,
        "created_at_utc": _now(),
        "git": _git_facts(),
        "config_sha256": config_sha,
        "audit_run_id": args.audit_run_id,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "supervision_sha256": contract_common["supervision_sha256"],
        "split_manifest_sha256": contract_common["split_manifest_sha256"],
        "sft_train_sha256": config["identity"]["sft_train_sha256"],
        "source_commit": config["identity"]["source_commit"],
        "train_schedule_sha256": schedule["schedule_sha256"],
        "sample_ids_sha256": schedule["sample_ids_sha256"],
        "draw_count": schedule["draw_count"],
        "fit_sample_count": len(fit_ids),
        "sequence_length_summary": {
            "scheduled_max": max(lengths.values()),
            "scheduled_min": min(lengths.values()),
            "dry_step_sample_id": dry_sample_id,
            "dry_step_source_length": lengths[dry_sample_id],
            "dry_step_padded_length": int(config["identity"]["max_train_sequence_length"]),
        },
        "candidate_lambda": float(proposal["candidate_lambda"]),
        "candidate_lambda_status": proposal["candidate_lambda_status"],
        "group_contracts": {"control": control, "candidate": candidate},
        "paired_contract": pair_contract,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in implementation_paths
        },
        "schedule_sha256": sha256_file(paths["schedule"]),
        "gpu_run_started": False,
    }
    _write_once(paths["manifest"], manifest)
    print(json.dumps({
        "status": manifest["status"],
        "pair_id": args.pair_id,
        "draw_count": schedule["draw_count"],
        "schedule_sha256": schedule["schedule_sha256"],
        "sample_ids_sha256": schedule["sample_ids_sha256"],
        "dry_step_source_length": lengths[dry_sample_id],
        "dry_step_padded_length": int(config["identity"]["max_train_sequence_length"]),
        "candidate_lambda_status": proposal["candidate_lambda_status"],
        "gpu_run_started": False,
        "output_dir": str(paths["root"]),
    }, ensure_ascii=False))
    return 0


def _load_gate(
    config: Mapping[str, Any], config_sha256: str, audit: Mapping[str, Any], probe_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _validate_id(probe_run_id, "probe_run_id")
    root = probe_runner._resolve(config["outputs"]["probe_root"]) / probe_run_id
    report_path, manifest_path = root / "probe_report.json", root / "run_manifest.json"
    gate_path, gate_manifest_path = root / "probe_gate.json", root / "probe_gate_manifest.json"
    report = probe_runner._json(report_path)
    probe_manifest = probe_runner._json(manifest_path)
    gate = probe_runner._json(gate_path)
    gate_manifest = probe_runner._json(gate_manifest_path)
    if (
        gate_manifest.get("status") != "probe_gate_frozen"
        or gate_manifest.get("probe_run_id") != probe_run_id
        or gate_manifest.get("decision") != gate.get("decision")
        or gate.get("probe_run_id") != probe_run_id
        or gate_manifest.get("config_sha256") != config_sha256
        or gate_manifest.get("audit_manifest_sha256") != audit["manifest_sha256"]
    ):
        raise RunAuthorizationError("A5.1 has no frozen, report-bound manual gate")
    expected_outputs = gate_manifest.get("outputs", {})
    if (
        probe_manifest.get("outputs", {}).get("probe_report.json") != sha256_file(report_path)
        or probe_manifest.get("config_sha256") != config_sha256
        or probe_manifest.get("a5_audit_manifest_sha256") != audit["manifest_sha256"]
        or expected_outputs.get("probe_gate.json") != sha256_file(gate_path)
        or gate_manifest.get("probe_report_sha256") != sha256_file(report_path)
        or gate_manifest.get("probe_run_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RunAuthorizationError("A5.1 gate or evidence report changed after review")
    gate_facts = verify_probe_gate(
        report, probe_manifest, gate,
        config_sha256=config_sha256,
        audit_manifest_sha256=audit["manifest_sha256"],
        probe_report_sha256=sha256_file(report_path),
        probe_manifest_sha256=sha256_file(manifest_path),
    )
    gate_facts.update({
        "probe_run_id": probe_run_id,
        "probe_run_manifest_sha256": sha256_file(manifest_path),
        "probe_report_sha256": sha256_file(report_path),
        "gate_manifest_sha256": sha256_file(gate_manifest_path),
    })
    return gate, gate_manifest, gate_facts


def _load_scheduled_rows(
    config: Mapping[str, Any], audit: Mapping[str, Any], schedule: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    sft_path = probe_runner._resolve(config["data"]["sft_train"])
    if sha256_file(sft_path) != config["identity"]["sft_train_sha256"]:
        raise ValueError("SFT Train hash changed after the A5 schedule was prepared")
    expected_ids = set(probe_runner._read_supervision(audit["supervision"]))
    probe_runner._validate_sft_ids(
        sft_path,
        expected_ids,
        expected_rows=int(config["data"]["expected_sft_train_rows"]),
    )
    schedule_ids = set(schedule["sample_ids"])
    rows: dict[str, dict[str, Any]] = {}
    for _line_number, row, _row_sha in iter_jsonl(sft_path):
        sample_id = row["sample_id"]
        if sample_id in schedule_ids:
            facts = validate_sft_row(row, max_sequence_length=int(config["identity"]["max_train_sequence_length"]))
            if int(facts["prompt_length"]) > int(config["identity"]["max_prompt_tokens"]):
                raise ValueError(f"{sample_id}: Prompt exceeds the frozen P=1280 budget")
            rows[sample_id] = row
    if set(rows) != schedule_ids:
        raise ValueError("SFT rows do not exactly contain every scheduled sample ID")
    supervision = probe_runner._read_supervision(audit["supervision"])
    return rows, supervision


def _gpu_gate(torch: Any) -> dict[str, Any]:
    facts = probe_runner._gpu_preflight(torch)
    device = facts["device"]
    facts["mem_get_info_total_vram_bytes"] = int(torch.cuda.mem_get_info(device)[1])
    return facts


def _seed_everything(seed: int, torch: Any) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_trainable_model(config: Mapping[str, Any], torch: Any) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from participant import run_inference as inference

    assets, identity = config["model_assets"], config["identity"]
    base_path = probe_runner._resolve(assets["base_model_path"])
    source_adapter = probe_runner._resolve(assets["source_adapter_path"])
    base_before = inference._weight_identity(base_path, {
        "model_id": assets["base_model_id"], "model_revision": assets["base_model_revision"],
    })
    if base_before.get("observed_weight_hash") != identity["base_weight_sha256"]:
        raise ValueError("Base weight SHA differs from the frozen E01-B/A4 identity")
    adapter_before, adapter_config = inference.adapter_weight_identity(source_adapter)
    if (
        adapter_before.get("weight_sha256") != identity["initial_adapter_weight_identity_sha256"]
        or adapter_before.get("config_sha256") != identity["initial_adapter_config_sha256"]
        or sha256_file(source_adapter / assets["source_adapter_file"]) != identity["initial_adapter_file_sha256"]
    ):
        raise ValueError("source E01-B Adapter identity differs from the frozen A5 config")
    if (
        adapter_config.get("r") != 8
        or adapter_config.get("lora_alpha") != 16
        or set(adapter_config.get("target_modules", [])) != {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        }
    ):
        raise ValueError("source Adapter changed frozen rank, alpha, or target modules")
    source_tensors = load_file(str(source_adapter / assets["source_adapter_file"]), device="cpu")
    if {str(value.dtype).removeprefix("torch.") for value in source_tensors.values()} != {assets["source_adapter_parameter_dtype"]}:
        raise ValueError("source Adapter tensor dtype differs from the frozen A5 config")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), local_files_only=True, trust_remote_code=False,
        device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization,
    )
    model.config.use_cache = False
    import inspect
    prepare_kwargs: dict[str, Any] = {"use_gradient_checkpointing": True}
    if "gradient_checkpointing_kwargs" in inspect.signature(prepare_model_for_kbit_training).parameters:
        prepare_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    model = prepare_model_for_kbit_training(model, **prepare_kwargs)
    model = PeftModel.from_pretrained(
        model, str(source_adapter), is_trainable=True, local_files_only=True,
        autocast_adapter_dtype=False,
    )
    if not getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(model, "_require_grads_hook", None) is None:
        model.enable_input_require_grads()
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    if set(loaded) != set(source_tensors) or any(
        not torch.equal(loaded[name].detach().cpu().to(dtype=value.dtype), value)
        for name, value in source_tensors.items()
    ):
        raise ValueError("loaded trainable Adapter tensors differ from the read-only E01-B source")
    lora_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not lora_parameters or any(".lora_A." not in name and ".lora_B." not in name for name, _ in lora_parameters):
        raise ValueError("only LoRA A/B tensors may be trainable in the causal model")
    quant_facts = inference.inspect_quantization_runtime(
        model, {"quantization": {"enabled": True, "use_double_quant": True}}, torch
    )
    if not quant_facts.get("real_nf4_verified") or quant_facts.get("offload_devices"):
        raise RuntimeError("A5 requires verified CUDA NF4/BF16 with no CPU/disk offload")
    model.train()
    return model, base_before, adapter_before, {
        "quantization": quant_facts,
        "source_adapter_tensor_exact_match": True,
        "source_adapter_autocast_adapter_dtype": False,
        "trainable_lora_parameter_count": sum(parameter.numel() for _, parameter in lora_parameters),
        "trainable_lora_parameter_names_sha256": canonical_sha256([name for name, _ in lora_parameters]),
        "lora_parameters": lora_parameters,
    }


def _new_head(hidden_size: int, seed: int, device: Any, *, trainable: bool) -> Any:
    import torch
    from torch import nn

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = nn.Linear(hidden_size, 33, bias=True, dtype=torch.float32, device="cpu")
    head.requires_grad_(trainable)
    return head.to(device)


def _head_state_sha256(head: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(bytes(contiguous.untyped_storage()))
    return digest.hexdigest()


def _run_identity_facts(
    config: Mapping[str, Any], config_sha256: str, audit: Mapping[str, Any],
    pair: Mapping[str, Any], pair_sha256: str, gate_facts: Mapping[str, Any],
    group: str, bce_lambda: float, run_id: str,
) -> dict[str, Any]:
    return {
        "created_at_utc": _now(),
        "experiment_id": config["experiment_id"],
        "run_id": run_id,
        "group": group,
        "bce_lambda": float(bce_lambda),
        "config_sha256": config_sha256,
        "a5_audit_manifest_sha256": audit["manifest_sha256"],
        "pair_id": pair["pair_id"],
        "pair_manifest_sha256": pair_sha256,
        "probe_gate": dict(gate_facts),
        "train_schedule_sha256": pair["train_schedule_sha256"],
        "sample_ids_sha256": pair["sample_ids_sha256"],
        "source_commit": config["identity"]["source_commit"],
        "git": _git_facts(),
        "optimizer_state_resume_enabled": False,
        "base_weight_sha256": config["identity"]["base_weight_sha256"],
        "initial_adapter_weight_sha256": config["identity"]["initial_adapter_weight_identity_sha256"],
        "initial_adapter_config_sha256": config["identity"]["initial_adapter_config_sha256"],
        "protocol_sha256": canonical_sha256(config["identity"]["protocol"]),
    }


def _verify_ticket_bindings(
    ticket: Mapping[str, Any], *, pair: Mapping[str, Any], pair_sha256: str,
    gate_facts: Mapping[str, Any], group: str, bce_lambda: float,
    dry_manifest_sha256: str | None = None,
    lambda_freeze_manifest_sha256: str | None = None,
) -> None:
    expected = {
        "pair_id": pair["pair_id"],
        "pair_manifest_sha256": pair_sha256,
        "probe_gate_manifest_sha256": gate_facts["gate_manifest_sha256"],
        "probe_run_manifest_sha256": gate_facts["probe_run_manifest_sha256"],
        "group": group,
        "bce_lambda": float(bce_lambda),
    }
    if dry_manifest_sha256 is not None:
        expected["dry_step_manifest_sha256"] = dry_manifest_sha256
    if lambda_freeze_manifest_sha256 is not None:
        expected["lambda_freeze_manifest_sha256"] = lambda_freeze_manifest_sha256
    mismatches = [name for name, value in expected.items() if ticket.get(name) != value]
    if mismatches:
        raise RunAuthorizationError(f"Run ticket does not bind A5 training facts: {mismatches}")


def _freeze_lambda(args: argparse.Namespace) -> int:
    _validate_id(args.run_id, "lambda_freeze_run_id")
    if not args.approved_by.strip() or not args.rationale.strip():
        raise ValueError("freeze-lambda requires a non-empty reviewer and rationale")
    config, config_sha = probe_runner._load_config(probe_runner._resolve(args.config))
    audit = _audit(config, config_sha, args.audit_run_id)
    pair, _schedule, _pair_dir = _load_pair(config, config_sha, audit, args.pair_id)
    pair_manifest_sha = sha256_file(_pair_paths(config, args.pair_id)["manifest"])
    _gate, _gate_manifest, gate_facts = _load_gate(config, config_sha, audit, args.probe_run_id)
    expected_lambda = float(config["training_proposal"]["candidate_lambda"])
    if int(config["training_proposal"]["per_device_micro_batch_size"]) != 1:
        raise ValueError("A5 v1 lambda freeze requires its locked micro_batch_size=1")
    dry_manifests: dict[str, tuple[dict[str, Any], str]] = {}
    for group, run_id in (
        ("control_sft", args.control_dry_step_run_id),
        ("candidate_sft_bce", args.candidate_dry_step_run_id),
    ):
        _validate_id(run_id, f"{group}_dry_step_run_id")
        manifest_path = _run_root(config) / run_id / "run_manifest.json"
        manifest = probe_runner._json(manifest_path)
        expected_lambda_for_group = 0.0 if group == "control_sft" else expected_lambda
        if (
            manifest.get("status") != "gpu_dry_step_passed"
            or manifest.get("group") != group
            or manifest.get("pair_id") != args.pair_id
            or manifest.get("pair_manifest_sha256") != pair_manifest_sha
            or manifest.get("probe_gate_manifest_sha256") != gate_facts["gate_manifest_sha256"]
            or manifest.get("config_sha256") != config_sha
            or manifest.get("a5_audit_manifest_sha256") != audit["manifest_sha256"]
            or manifest.get("bce_lambda") != expected_lambda_for_group
            or manifest.get("train_schedule_sha256") != pair["train_schedule_sha256"]
            or manifest.get("probe_run_manifest_sha256") != gate_facts["probe_run_manifest_sha256"]
            or manifest.get("optimizer_updates") != 0
        ):
            raise RunAuthorizationError(f"{group} has no matching passed no-update dry-step")
        dry_manifests[group] = (manifest, sha256_file(manifest_path))
    output_root = _run_root(config) / "lambda_freezes" / args.run_id
    output_root.mkdir(parents=True, exist_ok=False)
    freeze = {
        "status": "candidate_lambda_frozen",
        "experiment_id": config["experiment_id"],
        "run_id": args.run_id,
        "created_at_utc": _now(),
        "approved": True,
        "approved_by": args.approved_by.strip(),
        "rationale": args.rationale.strip(),
        "candidate_lambda": expected_lambda,
        "config_sha256": config_sha,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "pair_id": args.pair_id,
        "pair_manifest_sha256": pair_manifest_sha,
        "probe_gate_manifest_sha256": gate_facts["gate_manifest_sha256"],
        "probe_run_manifest_sha256": gate_facts["probe_run_manifest_sha256"],
        "control_dry_step_run_id": args.control_dry_step_run_id,
        "control_dry_step_manifest_sha256": dry_manifests["control_sft"][1],
        "candidate_dry_step_run_id": args.candidate_dry_step_run_id,
        "candidate_dry_step_manifest_sha256": dry_manifests["candidate_sft_bce"][1],
        "both_dry_steps_passed_without_optimizer_updates": True,
        "gpu_training_started": False,
    }
    freeze_path = output_root / "lambda_freeze.json"
    _write_once(freeze_path, freeze)
    manifest = {
        "status": "candidate_lambda_frozen",
        "run_id": args.run_id,
        "config_sha256": config_sha,
        "pair_manifest_sha256": pair_manifest_sha,
        "probe_gate_manifest_sha256": gate_facts["gate_manifest_sha256"],
        "outputs": {"lambda_freeze.json": sha256_file(freeze_path)},
        "created_at_utc": freeze["created_at_utc"],
    }
    _write_once(output_root / "lambda_freeze_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "run_id": args.run_id,
        "candidate_lambda": expected_lambda,
        "control_dry_step_passed": True, "candidate_dry_step_passed": True,
        "gpu_training_started": False, "output_dir": str(output_root),
    }, ensure_ascii=False))
    return 0


def _load_lambda_freeze(config: Mapping[str, Any], run_id: str) -> tuple[dict[str, Any], str]:
    _validate_id(run_id, "lambda_freeze_run_id")
    root = _run_root(config) / "lambda_freezes" / run_id
    freeze_path = root / "lambda_freeze.json"
    manifest_path = root / "lambda_freeze_manifest.json"
    freeze = probe_runner._json(freeze_path)
    manifest = probe_runner._json(manifest_path)
    if (
        manifest.get("status") != "candidate_lambda_frozen"
        or manifest.get("run_id") != run_id
        or freeze.get("run_id") != run_id
        or freeze.get("status") != "candidate_lambda_frozen"
        or manifest.get("outputs", {}).get("lambda_freeze.json") != sha256_file(freeze_path)
    ):
        raise RunAuthorizationError("A5 candidate lambda freeze artifact is missing or changed")
    return freeze, sha256_file(manifest_path)


def _gpu_run(args: argparse.Namespace, *, train: bool) -> int:
    _validate_id(args.run_id, "run_id")
    _validate_id(args.probe_run_id, "probe_run_id")
    if train:
        _validate_id(args.dry_step_run_id, "dry_step_run_id")
        _validate_id(args.lambda_freeze_run_id, "lambda_freeze_run_id")
    config, config_sha = probe_runner._load_config(probe_runner._resolve(args.config))
    audit = _audit(config, config_sha, args.audit_run_id)
    pair, schedule, _pair_dir = _load_pair(config, config_sha, audit, args.pair_id)
    pair_path = _pair_paths(config, args.pair_id)["manifest"]
    pair_sha = sha256_file(pair_path)
    gate, gate_manifest, gate_facts = _load_gate(config, config_sha, audit, args.probe_run_id)
    bce_lambda = 0.0 if args.group == "control_sft" else float(config["training_proposal"]["candidate_lambda"])
    if args.group not in {"control_sft", "candidate_sft_bce"}:
        raise ValueError("A5 group must be control_sft or candidate_sft_bce")
    if train:
        dry_dir = _run_root(config) / args.dry_step_run_id
        dry_manifest_path = dry_dir / "run_manifest.json"
        dry_manifest = probe_runner._json(dry_manifest_path)
        if (
            dry_manifest.get("status") != "gpu_dry_step_passed"
            or dry_manifest.get("group") != args.group
            or dry_manifest.get("pair_manifest_sha256") != pair_sha
            or dry_manifest.get("probe_gate_manifest_sha256") != gate_facts["gate_manifest_sha256"]
            or dry_manifest.get("config_sha256") != config_sha
            or dry_manifest.get("a5_audit_manifest_sha256") != audit["manifest_sha256"]
            or dry_manifest.get("train_schedule_sha256") != pair["train_schedule_sha256"]
            or dry_manifest.get("bce_lambda") != bce_lambda
            or dry_manifest.get("optimizer_updates") != 0
        ):
            raise RunAuthorizationError("A5 training requires its own matching passed no-update GPU dry-step")
        dry_sha = sha256_file(dry_manifest_path)
        lambda_freeze, lambda_freeze_manifest_sha = _load_lambda_freeze(config, args.lambda_freeze_run_id)
        _validate_id(lambda_freeze.get("control_dry_step_run_id"), "control_dry_step_run_id")
        _validate_id(lambda_freeze.get("candidate_dry_step_run_id"), "candidate_dry_step_run_id")
        control_dry_path = _run_root(config) / str(lambda_freeze.get("control_dry_step_run_id", "")) / "run_manifest.json"
        candidate_dry_path = _run_root(config) / str(lambda_freeze.get("candidate_dry_step_run_id", "")) / "run_manifest.json"
        lambda_facts = verify_lambda_freeze(
            config,
            lambda_freeze,
            config_sha256=config_sha,
            audit_manifest_sha256=audit["manifest_sha256"],
            pair_id=args.pair_id,
            pair_manifest_sha256=pair_sha,
            probe_gate_manifest_sha256=gate_facts["gate_manifest_sha256"],
            control_dry_step_manifest_sha256=sha256_file(control_dry_path),
            candidate_dry_step_manifest_sha256=sha256_file(candidate_dry_path),
        )
        expected_dry_run_id = (
            lambda_freeze["control_dry_step_run_id"]
            if args.group == "control_sft"
            else lambda_freeze["candidate_dry_step_run_id"]
        )
        expected_dry_sha = (
            lambda_freeze["control_dry_step_manifest_sha256"]
            if args.group == "control_sft"
            else lambda_freeze["candidate_dry_step_manifest_sha256"]
        )
        if args.dry_step_run_id != expected_dry_run_id or dry_sha != expected_dry_sha:
            raise RunAuthorizationError("A5 training dry-step differs from the reviewed lambda freeze")
        if args.group == "candidate_sft_bce" and bce_lambda != lambda_facts["candidate_lambda"]:
            raise RunAuthorizationError("Candidate lambda differs from its reviewed dry-step freeze")
    else:
        dry_manifest = None
        dry_sha = None
        lambda_freeze = None
        lambda_freeze_manifest_sha = None
        lambda_facts = None

    ticket = probe_runner._json(probe_runner._resolve(args.ticket))
    stage = "gpu_training" if train else "gpu_dry_step"
    verify_run_ticket(
        ticket,
        stage=stage,
        run_id=args.run_id,
        config_sha256=config_sha,
        audit_manifest_sha256=audit["manifest_sha256"],
        max_sample_count=int(config["training_proposal"]["sample_draw_count"]) if train else 1,
        max_optimizer_steps=int(config["training_proposal"]["max_optimizer_steps"]) if train else None,
    )
    _verify_ticket_bindings(
        ticket, pair=pair, pair_sha256=pair_sha, gate_facts=gate_facts,
        group=args.group, bce_lambda=bce_lambda, dry_manifest_sha256=dry_sha,
        lambda_freeze_manifest_sha256=lambda_freeze_manifest_sha,
    )
    training_root = _run_root(config).resolve()
    output_dir = (training_root / args.run_id).resolve()
    if not output_dir.is_relative_to(training_root):
        raise ValueError("A5 Run ID escaped the immutable training output root")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run_identity = _run_identity_facts(
        config, config_sha, audit, pair, pair_sha, gate_facts,
        args.group, bce_lambda, args.run_id,
    )
    run_identity.update({
        "status": "gpu_dry_step_started" if not train else "training_started",
        "ticket_sha256": sha256_file(probe_runner._resolve(args.ticket)),
        "dry_step_manifest_sha256": dry_sha,
        "lambda_freeze": lambda_facts,
        "lambda_freeze_manifest_sha256": lambda_freeze_manifest_sha,
        "candidate_lambda_status": "candidate_lambda_frozen" if train else config["training_proposal"]["candidate_lambda_status"],
        "pair_schedule_sha256": pair["schedule_sha256"],
        "gpu_updates_allowed": bool(train),
    })
    _write_once(output_dir / "run_identity.json", run_identity)
    model = optimizer = scheduler = head = None
    completed_steps = 0
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from src.data.sft_collator import SFTDataCollator
        from participant import run_inference as inference

        proposal = config["training_proposal"]
        _seed_everything(int(proposal["seed"]), torch)
        gpu = _gpu_gate(torch)
        rows, supervision = _load_scheduled_rows(config, audit, schedule)
        model, base_before, adapter_before, model_facts = _load_trainable_model(config, torch)
        device = model.get_input_embeddings().weight.device
        lora_parameters = model_facts.pop("lora_parameters")
        hidden_size = int(getattr(model.config, "hidden_size"))
        head = _new_head(hidden_size, int(proposal["seed"]), device, trainable=args.group == "candidate_sft_bce")
        head_sha = _head_state_sha256(head)
        sample_ids = list(schedule["sample_ids"])
        dry_sample_id = pair["sequence_length_summary"]["dry_step_sample_id"]
        if dry_sample_id not in rows or dry_sample_id not in sample_ids:
            raise ValueError("frozen dry-step sample is not in the paired Fit schedule")
        collator = SFTDataCollator(
            int(config["identity"]["pad_token_id"]),
            max_sequence_length=int(config["identity"]["max_train_sequence_length"]),
            pad_to_multiple_of=int(config["identity"]["max_train_sequence_length"]) if not train else None,
        )
        parameters = [parameter for _name, parameter in lora_parameters]
        head_parameters = list(head.parameters()) if args.group == "candidate_sft_bce" else []
        optimizer_parameters = parameters + head_parameters
        if train:
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=float(proposal["learning_rate"]),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(proposal["weight_decay"]),
            )
            if optimizer.state:
                raise RuntimeError("A5 optimizer must start with fresh, empty state")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
            torch.cuda.reset_peak_memory_stats()

        if not train:
            row = rows[dry_sample_id]
            batch = collator([row])
            if batch["input_ids"].shape[1] != int(config["identity"]["max_train_sequence_length"]):
                raise RuntimeError("A5 GPU dry-step must execute the full frozen S=2048 sequence shape")
            model_inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            prompt_lengths = torch.tensor([int(row["prompt_length"])], dtype=torch.long, device=device)
            targets = torch.tensor([supervision[dry_sample_id]["y"]], dtype=torch.float32, device=device)
            masks = torch.tensor([supervision[dry_sample_id]["mask"]], dtype=torch.float32, device=device)
            before = [value.detach().cpu().clone() for value in optimizer_parameters]
            torch.cuda.reset_peak_memory_stats()
            with FinalHiddenCapture(model) as capture, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, prompt_hidden = forward_with_prompt_end_hidden(
                    model, capture,
                    model_inputs["input_ids"], model_inputs["attention_mask"], prompt_lengths,
                )
                sft_loss = completion_only_sft_loss(outputs.logits, batch["labels"].to(device))
                bce_loss = None
                if args.group == "candidate_sft_bce":
                    bce_loss = masked_bce_with_logits(head(prompt_hidden.float()), targets, masks)
                total_loss = combine_sft_bce_losses(sft_loss, bce_loss, bce_lambda)
            if not bool(torch.isfinite(total_loss.detach()).item()):
                raise FloatingPointError("A5 GPU dry-step loss is non-finite")
            total_loss.backward()
            lora_grad = assert_finite_nonzero_gradients(parameters, description="A5 LoRA")
            head_grad = (
                assert_finite_nonzero_gradients(head_parameters, description="A5 auxiliary head")
                if head_parameters else None
            )
            if any(not torch.equal(previous, current.detach().cpu()) for previous, current in zip(before, optimizer_parameters)):
                raise RuntimeError("A5 no-update GPU dry-step changed a trainable parameter")
            source_adapter_after, _ = inference.adapter_weight_identity(
                probe_runner._resolve(config["model_assets"]["source_adapter_path"])
            )
            base_after = inference._weight_identity(
                probe_runner._resolve(config["model_assets"]["base_model_path"]),
                {
                    "model_id": config["model_assets"]["base_model_id"],
                    "model_revision": config["model_assets"]["base_model_revision"],
                },
            )
            if any(source_adapter_after.get(key) != adapter_before.get(key) for key in ("weight_sha256", "config_sha256")):
                raise RuntimeError("A5 dry-step modified the read-only source Adapter")
            if base_after.get("observed_weight_hash") != base_before.get("observed_weight_hash"):
                raise RuntimeError("A5 dry-step modified the frozen Base weights")
            run_manifest = {
                "status": "gpu_dry_step_passed",
                "created_at_utc": run_identity["created_at_utc"],
                "ended_at_utc": _now(),
                "experiment_id": config["experiment_id"],
                "run_id": args.run_id,
                "group": args.group,
                "bce_lambda": bce_lambda,
                "config_sha256": config_sha,
                "a5_audit_manifest_sha256": audit["manifest_sha256"],
                "pair_id": args.pair_id,
                "pair_manifest_sha256": pair_sha,
                "probe_gate_manifest_sha256": gate_facts["gate_manifest_sha256"],
                "probe_run_manifest_sha256": gate_facts["probe_run_manifest_sha256"],
                "train_schedule_sha256": pair["train_schedule_sha256"],
                "sample_id": dry_sample_id,
                "source_sequence_length": len(row["input_ids"]),
                "padded_sequence_length": int(batch["input_ids"].shape[1]),
                "sft_loss": float(sft_loss.detach().float().item()),
                "bce_loss": float(bce_loss.detach().float().item()) if bce_loss is not None else None,
                "total_loss": float(total_loss.detach().float().item()),
                "lora_gradient": lora_grad,
                "head_gradient": head_grad,
                "head_initialization_sha256": head_sha,
                "optimizer_updates": 0,
                "stage_wall_seconds": time.monotonic() - started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "gpu": gpu,
                "model_facts": model_facts,
                "source_adapter_unchanged": True,
                "base_weights_unchanged": True,
                "candidate_training_allowed": False,
                "run_identity_sha256": sha256_file(output_dir / "run_identity.json"),
            }
            if run_manifest["stage_wall_seconds"] > float(ticket["budget"]["max_seconds"]):
                raise TimeoutError("A5 GPU dry-step exceeded its ticket wall-time budget")
            _write_once(output_dir / "run_manifest.json", run_manifest)
            print(json.dumps({
                "status": run_manifest["status"], "run_id": args.run_id,
                "group": args.group, "padded_sequence_length": run_manifest["padded_sequence_length"],
                "optimizer_updates": 0, "peak_allocated_bytes": run_manifest["peak_allocated_bytes"],
                "output_dir": str(output_dir),
            }, ensure_ascii=False))
            return 0

        # Training runs use the same frozen schedule in both arms. The only
        # objective difference is lambda=0 versus the frozen positive BCE lambda.
        accum = int(proposal["gradient_accumulation_steps"])
        total_steps = int(proposal["max_optimizer_steps"])
        expected_draws = int(proposal["sample_draw_count"])
        if len(sample_ids) != total_steps * accum or len(sample_ids) != expected_draws:
            raise ValueError("A5 schedule does not match the exact optimizer/microbatch budget")
        if not isinstance(dry_manifest, Mapping):
            raise AssertionError("training path lost its passed dry-step Manifest")
        step_start = time.monotonic()
        max_seconds = float(ticket["budget"]["max_seconds"])
        log_path = output_dir / "optimizer_steps.jsonl"
        for step in range(1, total_steps + 1):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("A5 training exceeded its ticket wall-time budget")
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[dict[str, float | None]] = []
            step_ids: list[str] = []
            last_prompt_features = None
            for micro_index in range(accum):
                sample_id = schedule["schedule"][step - 1]["microbatches"][micro_index][0]
                row = rows[sample_id]
                step_ids.append(sample_id)
                batch = collator([row])
                model_inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                prompt_lengths = torch.tensor([int(row["prompt_length"])], dtype=torch.long, device=device)
                with FinalHiddenCapture(model) as capture, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs, prompt_hidden = forward_with_prompt_end_hidden(
                        model, capture,
                        model_inputs["input_ids"], model_inputs["attention_mask"], prompt_lengths,
                    )
                    sft_loss = completion_only_sft_loss(outputs.logits, batch["labels"].to(device))
                    bce_loss = None
                    if args.group == "candidate_sft_bce":
                        targets = torch.tensor([supervision[sample_id]["y"]], dtype=torch.float32, device=device)
                        masks = torch.tensor([supervision[sample_id]["mask"]], dtype=torch.float32, device=device)
                        bce_loss = masked_bce_with_logits(head(prompt_hidden.float()), targets, masks)
                    total_loss = combine_sft_bce_losses(sft_loss, bce_loss, bce_lambda)
                if not bool(torch.isfinite(total_loss.detach()).item()):
                    raise FloatingPointError(f"A5 non-finite loss at optimizer step {step}")
                micro_losses.append({
                    "sft_loss": float(sft_loss.detach().float().item()),
                    "bce_loss": float(bce_loss.detach().float().item()) if bce_loss is not None else None,
                    "total_loss": float(total_loss.detach().float().item()),
                })
                (total_loss / accum).backward()
                last_prompt_features = prompt_hidden.detach().float().cpu()
            lora_grad = assert_finite_nonzero_gradients(parameters, description="A5 LoRA")
            head_grad = (
                assert_finite_nonzero_gradients(head_parameters, description="A5 auxiliary head")
                if head_parameters else None
            )
            check_update = step in {1, total_steps // 2, total_steps}
            watched = parameters + head_parameters
            before = [parameter.detach().cpu().clone() for parameter in watched] if check_update else []
            optimizer.step()
            scheduler.step()
            if float(optimizer.param_groups[0]["lr"]) != float(proposal["learning_rate"]):
                raise RuntimeError("A5 constant scheduler changed the frozen learning rate")
            changed = any(not torch.equal(previous, parameter.detach().cpu()) for previous, parameter in zip(before, watched)) if check_update else None
            if check_update and not changed:
                raise RuntimeError(f"A5 optimizer did not update any trainable tensor at step {step}")
            completed_steps = step
            elapsed = time.monotonic() - step_start
            _append_jsonl(log_path, {
                "optimizer_step": step,
                "sample_ids": step_ids,
                "micro_losses": micro_losses,
                "mean_sft_loss": sum(float(item["sft_loss"]) for item in micro_losses) / accum,
                "mean_bce_loss": (
                    sum(float(item["bce_loss"]) for item in micro_losses if item["bce_loss"] is not None) / accum
                    if args.group == "candidate_sft_bce" else None
                ),
                "mean_total_loss": sum(float(item["total_loss"]) for item in micro_losses) / accum,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "lora_gradient": lora_grad,
                "head_gradient": head_grad,
                "parameters_changed": changed,
                "elapsed_training_seconds": elapsed,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            })
            if elapsed > max_seconds:
                raise TimeoutError("A5 training exceeded its ticket wall-time budget")
        if completed_steps != total_steps:
            raise RuntimeError("A5 did not complete the exact frozen optimizer step count")
        training_seconds = time.monotonic() - step_start
        training_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        training_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

        adapter_dir = output_dir / "adapter"
        model.save_pretrained(str(adapter_dir), safe_serialization=True)
        saved_identity, saved_adapter_config = inference.adapter_weight_identity(adapter_dir)
        if (
            saved_adapter_config.get("r") != 8
            or saved_adapter_config.get("lora_alpha") != 16
            or set(saved_adapter_config.get("target_modules", [])) != {
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            }
        ):
            raise RuntimeError("saved A5 Adapter changed the frozen LoRA structure")
        head_facts = None
        if args.group == "candidate_sft_bce":
            head_path = output_dir / "classifier_head.safetensors"
            saved_head = {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}
            save_file(saved_head, str(head_path))
            reloaded_head_tensors = load_file(str(head_path), device="cpu")
            roundtrip = torch.nn.Linear(hidden_size, 33, bias=True, dtype=torch.float32, device="cpu")
            roundtrip.load_state_dict(reloaded_head_tensors)
            if any(not torch.equal(saved_head[name], reloaded_head_tensors[name]) for name in saved_head):
                raise RuntimeError("saved classifier-head tensor reload changed values")
            if last_prompt_features is None:
                raise RuntimeError("no Prompt-end feature is available for the head reload check")
            expected_logits = head(last_prompt_features.to(device)).detach().cpu()
            observed_logits = roundtrip(last_prompt_features).detach().cpu()
            if not torch.equal(expected_logits, observed_logits):
                raise RuntimeError("saved classifier head changed logits after independent CPU reload")
            head_facts = {
                "path": str(head_path),
                "sha256": sha256_file(head_path),
                "initialization_sha256": head_sha,
                "saved_tensor_reload_exact": True,
                "logits_reload_exact": True,
                "dtype": "float32",
                "hidden_size": hidden_size,
                "label_count": 33,
            }

        final_trainable_state = optimizer.state
        state_steps = [
            int(state["step"].item()) if hasattr(state.get("step"), "item") else int(state["step"])
            for state in final_trainable_state.values() if state.get("step") is not None
        ]
        if len(final_trainable_state) != len(optimizer_parameters) or not state_steps or min(state_steps) != total_steps or max(state_steps) != total_steps:
            raise RuntimeError("A5 fresh AdamW state does not prove the exact optimizer step count")
        initial_adapter_after, _ = inference.adapter_weight_identity(probe_runner._resolve(config["model_assets"]["source_adapter_path"]))
        if any(initial_adapter_after.get(key) != adapter_before.get(key) for key in ("weight_sha256", "config_sha256")):
            raise RuntimeError("A5 modified the read-only E01-B Adapter")
        del model, optimizer, scheduler
        model = optimizer = scheduler = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Independently attach the saved adapter to a fresh frozen NF4 Base and
        # compare every persisted LoRA tensor with the serialized artifact.
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from participant import run_inference as inference

        base_path = probe_runner._resolve(config["model_assets"]["base_model_path"])
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        reloaded_base = AutoModelForCausalLM.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False,
            device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization,
        )
        reloaded = inference.load_peft_adapter(
            reloaded_base, adapter_dir, prepare_quantized_base=True, is_trainable=False,
            local_files_only=True, autocast_adapter_dtype=False,
        )
        saved_tensors = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
        reloaded_tensors = get_peft_model_state_dict(reloaded, adapter_name="default")
        adapter_reload_exact = set(saved_tensors) == set(reloaded_tensors) and all(
            torch.equal(reloaded_tensors[name].detach().cpu().to(dtype=value.dtype), value)
            for name, value in saved_tensors.items()
        )
        if not adapter_reload_exact:
            raise RuntimeError("independent A5 Adapter reload changed a saved LoRA tensor")
        del reloaded, reloaded_base
        gc.collect()
        torch.cuda.empty_cache()
        base_after = inference._weight_identity(base_path, {
            "model_id": config["model_assets"]["base_model_id"],
            "model_revision": config["model_assets"]["base_model_revision"],
        })
        if base_after.get("observed_weight_hash") != base_before.get("observed_weight_hash"):
            raise RuntimeError("A5 Base weight identity changed during training or reload")
        elapsed_total = time.monotonic() - started
        if elapsed_total > max_seconds:
            raise TimeoutError("A5 training and standalone reload exceeded its ticket wall-time budget")
        run_manifest = {
            "status": "trained_pending_eval",
            "created_at_utc": run_identity["created_at_utc"],
            "ended_at_utc": _now(),
            "experiment_id": config["experiment_id"],
            "run_id": args.run_id,
            "group": args.group,
            "bce_lambda": bce_lambda,
            "candidate_lambda_status": "candidate_lambda_frozen",
            "config_sha256": config_sha,
            "a5_audit_manifest_sha256": audit["manifest_sha256"],
            "pair_id": args.pair_id,
            "pair_manifest_sha256": pair_sha,
            "probe_gate_manifest_sha256": gate_facts["gate_manifest_sha256"],
            "probe_run_manifest_sha256": gate_facts["probe_run_manifest_sha256"],
            "dry_step_manifest_sha256": dry_sha,
            "lambda_freeze_manifest_sha256": lambda_freeze_manifest_sha,
            "train_schedule_sha256": pair["train_schedule_sha256"],
            "sample_ids_sha256": pair["sample_ids_sha256"],
            "optimizer": {
                "name": "AdamW",
                "learning_rate": float(proposal["learning_rate"]),
                "scheduler": proposal["scheduler"],
                "weight_decay": float(proposal["weight_decay"]),
                "optimizer_steps": completed_steps,
                "micro_batch_size": int(proposal["per_device_micro_batch_size"]),
                "gradient_accumulation_steps": accum,
                "draw_count": len(sample_ids),
                "fresh_state_verified": True,
                "optimizer_state_resume_enabled": False,
                "state_step_min": min(state_steps),
                "state_step_max": max(state_steps),
            },
            "training_seconds": training_seconds,
            "stage_wall_seconds": elapsed_total,
            "max_ticket_seconds": max_seconds,
            "ticket_sha256": run_identity["ticket_sha256"],
            "gpu": gpu,
            "model_facts": model_facts,
            "base_weight_sha256_before": base_before.get("observed_weight_hash"),
            "base_weight_sha256_after": base_after.get("observed_weight_hash"),
            "base_weights_unchanged": True,
            "initial_adapter_unchanged": True,
            "adapter_identity": saved_identity,
            "adapter_reload_exact": adapter_reload_exact,
            "classifier_head": head_facts,
            "optimizer_step_ledger_sha256": sha256_file(log_path),
            "optimizer_step_count": completed_steps,
            "peak_allocated_bytes": training_peak_allocated_bytes,
            "peak_reserved_bytes": training_peak_reserved_bytes,
            "candidate_scoring_allowed": False,
            "next_stage": "A5.3 separately authorized Fast-only paired generation",
        }
        _write_once(output_dir / "run_manifest.json", run_manifest)
        print(json.dumps({
            "status": run_manifest["status"], "run_id": args.run_id,
            "group": args.group, "optimizer_steps": completed_steps,
            "adapter_reload_exact": adapter_reload_exact, "output_dir": str(output_dir),
        }, ensure_ascii=False))
        return 0
    except BaseException as exc:
        failure = {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "budget_exhausted" if isinstance(exc, TimeoutError) else "failed",
            "ended_at_utc": _now(),
            "experiment_id": config["experiment_id"],
            "run_id": args.run_id,
            "group": args.group,
            "bce_lambda": bce_lambda,
            "config_sha256": config_sha,
            "pair_manifest_sha256": pair_sha,
            "probe_gate_manifest_sha256": gate_facts["gate_manifest_sha256"],
            "dry_step_manifest_sha256": dry_sha,
            "optimizer_steps_completed": completed_steps,
            "error": f"{type(exc).__name__}: {exc}",
            "candidate_scoring_allowed": False,
        }
        if not (output_dir / "run_manifest.json").exists():
            _write_once(output_dir / "run_manifest.json", failure)
        raise
    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        if head is not None:
            del head
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _group_args(parser: argparse.ArgumentParser, *, requires_dry: bool = False) -> None:
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--probe-run-id", required=True)
    parser.add_argument("--group", required=True, choices=("control_sft", "candidate_sft_bce"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ticket", type=Path, required=True)
    if requires_dry:
        parser.add_argument("--dry-step-run-id", required=True)
        parser.add_argument("--lambda-freeze-run-id", required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/e03_a5.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-pair")
    prepare.add_argument("--audit-run-id", required=True)
    prepare.add_argument("--pair-id", required=True)
    freeze = commands.add_parser("freeze-lambda")
    freeze.add_argument("--audit-run-id", required=True)
    freeze.add_argument("--pair-id", required=True)
    freeze.add_argument("--probe-run-id", required=True)
    freeze.add_argument("--control-dry-step-run-id", required=True)
    freeze.add_argument("--candidate-dry-step-run-id", required=True)
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--approved-by", required=True)
    freeze.add_argument("--rationale", required=True)
    dry = commands.add_parser("dry-step")
    _group_args(dry)
    train_parser = commands.add_parser("train-group")
    _group_args(train_parser, requires_dry=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-pair":
            return _prepare_pair(args)
        if args.command == "freeze-lambda":
            return _freeze_lambda(args)
        return _gpu_run(args, train=args.command == "train-group")
    except RunAuthorizationError as exc:
        print(json.dumps({"status": "blocked_run_authorization", "error": str(exc)}, ensure_ascii=False))
        return 3
    except Exception as exc:
        print(json.dumps({
            "status": f"{args.command}_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gpu_run_started": False if args.command == "prepare-pair" else None,
        }, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
