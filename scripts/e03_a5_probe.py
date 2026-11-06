#!/usr/bin/env python3
"""E03-A5 prompt-only feature extraction and CPU linear-probe runner.

Real Base/Adapter loading is reachable only through ``extract-train`` after an
explicit, stage-specific Run ticket has been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.e03_a5_labels import (
    canonical_sha256,
    iter_jsonl,
    schema_label_order,
    sha256_file,
)
from src.training.e03_a5_contract import (
    RunAuthorizationError,
    validate_a5_config,
    verify_run_ticket,
)
from src.training.e03_a5_probe import fit_and_evaluate_probe, iter_prompt_features


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("A5 config must be a YAML object")
    validate_a5_config(config)
    return config, sha256_file(path)


def _audit_paths(config: Mapping[str, Any], audit_run_id: str) -> dict[str, Path]:
    root = _resolve(config["outputs"]["audit_root"]) / audit_run_id
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "report": root / "audit_report.json",
        "supervision": root / "supervision.jsonl",
        "splits": root / "conversation_splits.json",
    }


def _verify_audit_run(config: Mapping[str, Any], config_sha256: str, audit_run_id: str) -> dict[str, Any]:
    paths = _audit_paths(config, audit_run_id)
    manifest = _json(paths["manifest"])
    if manifest.get("status") != "supervision_audit_passed" or manifest.get("run_id") != audit_run_id:
        raise ValueError("A5.0 audit Manifest is not a passed Run")
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError("A5.0 audit Manifest was generated from a different A5 config")
    for artifact_name, path in (
        ("supervision.jsonl", paths["supervision"]),
        ("conversation_splits.json", paths["splits"]),
        ("audit_report.json", paths["report"]),
    ):
        record = manifest.get("outputs", {}).get(artifact_name, {})
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"A5.0 audit artifact changed or is missing: {artifact_name}")
    report = _json(paths["report"])
    if report.get("status") != "supervision_audit_passed" or report.get("gpu_probe_status") != "not_run":
        raise ValueError("A5.0 audit report status is invalid")
    if manifest.get("outputs", {}).get("supervision.jsonl", {}).get("rows") != config["data"]["expected_sft_train_rows"]:
        raise ValueError("A5.0 supervision row count differs from the frozen SFT Train contract")
    return {**paths, "manifest_value": manifest, "report_value": report, "manifest_sha256": sha256_file(paths["manifest"])}


def _load_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    for _line_number, row, _row_sha in iter_jsonl(path):
        yield row


def _read_supervision(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for _line_number, row, _row_sha in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in by_id:
            raise ValueError("A5.0 supervision contains a missing or duplicate sample_id")
        by_id[sample_id] = row
    return by_id


def _read_splits(path: Path) -> dict[str, list[str]]:
    value = _json(path)
    if value.get("version") != "e03_a5_conversation_group_stratified_v1":
        raise ValueError("A5.0 split version differs from the frozen A5 config")
    splits = value.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("A5.0 split artifact lacks split rows")
    return {
        name: list(splits[name]["sample_ids"])
        for name in ("fit", "calibration", "internal_holdout")
    }


def _validate_sft_ids(path: Path, expected_ids: set[str], *, expected_rows: int) -> None:
    seen: set[str] = set()
    count = 0
    for line_number, row, _row_sha in iter_jsonl(path):
        count += 1
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"SFT Train row {line_number} has a missing or duplicate sample_id")
        if row.get("source_split") != "train":
            raise ValueError(f"{sample_id}: only source_split=train rows may be encoded")
        seen.add(sample_id)
    if count != expected_rows or seen != expected_ids:
        raise ValueError("SFT Train IDs do not exactly match the frozen A5.0 supervision set")


def _gpu_preflight(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("blocked_gpu_unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_bf16_unavailable")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    tolerance = 16 * 1024 * 1024
    minimum_total = 8 * 1024**3
    minimum_free = 6 * 1024**3
    if props.total_memory < minimum_total - tolerance:
        raise RuntimeError(f"blocked_total_vram_below_gate:{props.total_memory}")
    if free_bytes < minimum_free:
        raise RuntimeError(f"blocked_free_vram_below_gate:{free_bytes}")
    return {
        "device": device,
        "device_name": props.name,
        "total_vram_bytes": int(props.total_memory),
        "free_vram_bytes_before_load": int(free_bytes),
        "minimum_total_vram_bytes": minimum_total,
        "reporting_tolerance_bytes": tolerance,
        "minimum_free_vram_bytes": minimum_free,
        "bf16_supported": True,
        "cuda_version": torch.version.cuda,
    }


def _load_frozen_model(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from participant import run_inference as inference

    assets = config["model_assets"]
    identity = config["identity"]
    base_path = _resolve(assets["base_model_path"])
    adapter_path = _resolve(assets["source_adapter_path"])
    base_identity = inference._weight_identity(base_path, {
        "model_id": assets["base_model_id"],
        "model_revision": assets["base_model_revision"],
    })
    if base_identity.get("observed_weight_hash") != identity["base_weight_sha256"]:
        raise ValueError("local Base weight SHA differs from frozen E01-B/A4")
    adapter_identity, adapter_config = inference.adapter_weight_identity(adapter_path)
    if (
        adapter_identity.get("weight_sha256") != identity["initial_adapter_weight_identity_sha256"]
        or adapter_identity.get("config_sha256") != identity["initial_adapter_config_sha256"]
    ):
        raise ValueError("local E01-B Adapter SHA differs from frozen A5 identity")
    weight_file = adapter_path / assets["source_adapter_file"]
    if sha256_file(weight_file) != identity["initial_adapter_file_sha256"]:
        raise ValueError("local E01-B Adapter file SHA differs from frozen A5 identity")
    observed_dtype = adapter_config.get("base_model_name_or_path")
    if observed_dtype != assets["base_model_id"]:
        raise ValueError("E01-B Adapter declares a different Base model")
    if (
        adapter_config.get("r") != 8
        or adapter_config.get("lora_alpha") != 16
        or set(adapter_config.get("target_modules", [])) != {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        }
    ):
        raise ValueError("E01-B Adapter LoRA identity differs from A4")
    source_tensors = load_file(str(weight_file), device="cpu")
    source_dtypes = {str(tensor.dtype).removeprefix("torch.") for tensor in source_tensors.values()}
    if source_dtypes != {assets["source_adapter_parameter_dtype"]}:
        raise ValueError("E01-B Adapter parameter dtype differs from the frozen BF16 identity")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
    )
    model.config.use_cache = False
    model = inference.load_peft_adapter(
        model,
        adapter_path,
        prepare_quantized_base=True,
        is_trainable=False,
        local_files_only=True,
        autocast_adapter_dtype=False,
    )
    model.eval()
    loaded_tensors = get_peft_model_state_dict(model, adapter_name="default")
    if set(loaded_tensors) != set(source_tensors):
        raise ValueError("loaded Adapter tensor keys differ from the frozen E01-B Adapter")
    for name, source_tensor in source_tensors.items():
        observed = loaded_tensors[name].detach().to(device="cpu", dtype=source_tensor.dtype)
        if not torch.equal(observed, source_tensor):
            raise ValueError(f"E01-B Adapter tensor changed on load: {name}")
    quant_facts = inference.inspect_quantization_runtime(
        model,
        {"quantization": {"enabled": True, "use_double_quant": True}},
        torch,
    )
    if not quant_facts.get("real_nf4_verified") or quant_facts.get("offload_devices"):
        raise RuntimeError("A5 requires verified CUDA NF4/BF16 placement without CPU/disk offload")
    runtime_config = _json(_resolve(config["data"]["runtime_inference_config"]))
    tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runtime_config, local_source=base_path)
    if (
        tokenizer_facts.get("signature_sha256") != identity["tokenizer_signature_sha256"]
        or tokenizer_facts.get("template_sha256") != identity["chat_template_sha256"]
        or tokenizer_facts.get("eos_token_id") != identity["eos_token_id"]
        or getattr(tokenizer, "pad_token_id", None) != identity["pad_token_id"]
    ):
        raise ValueError("runtime Tokenizer facts differ from frozen D0.4.2/A4 protocol")
    return model, tokenizer, base_identity, adapter_identity, {
        "quantization": quant_facts,
        "tokenizer": tokenizer_facts,
        "adapter_dtype_autocast": False,
        "loaded_adapter_tensor_exact_match": True,
    }


def _extract_train(args: argparse.Namespace) -> int:
    config_path = _resolve(args.config)
    config, config_sha = _load_config(config_path)
    audit = _verify_audit_run(config, config_sha, args.audit_run_id)
    ticket = _json(_resolve(args.ticket))
    expected_ids = set(_read_supervision(audit["supervision"]))
    split_ids = _read_splits(audit["splits"])
    all_split_ids = [sample_id for name in split_ids for sample_id in split_ids[name]]
    if len(all_split_ids) != len(set(all_split_ids)) or set(all_split_ids) != expected_ids:
        raise ValueError("A5.0 split IDs do not exactly partition the supervision IDs")
    sft_path = _resolve(config["data"]["sft_train"])
    if sha256_file(sft_path) != config["identity"]["sft_train_sha256"]:
        raise ValueError("D0.4.3 SFT Train hash changed after A5.0")
    _validate_sft_ids(sft_path, expected_ids, expected_rows=int(config["data"]["expected_sft_train_rows"]))
    ticket_facts = verify_run_ticket(
        ticket,
        stage="feature_extraction",
        run_id=args.run_id,
        config_sha256=config_sha,
        audit_manifest_sha256=audit["manifest_sha256"],
        max_sample_count=len(expected_ids),
    )
    output_root = _resolve(config["outputs"]["probe_root"])
    output_dir = output_root / args.run_id
    if not output_dir.is_relative_to(_resolve("outputs")):
        raise ValueError("feature cache must be written under ignored outputs/")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    run_started = time.monotonic()
    started_at = _now()
    gpu_facts: dict[str, Any] = {}
    model_facts: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}
    try:
        import torch
        from safetensors.torch import save_file

        gpu_facts = _gpu_preflight(torch)
        model, tokenizer, base_before, adapter_before, model_facts = _load_frozen_model(config)
        id_order: list[str] = []
        feature_batches = []
        feature_dtype: str | None = None
        batch_size = int(config["probe"]["feature_batch_size"])
        for ids, features, dtype in iter_prompt_features(
            model,
            _load_jsonl_rows(sft_path),
            pad_token_id=int(tokenizer.pad_token_id),
            batch_size=batch_size,
            max_prompt_tokens=int(config["identity"]["max_prompt_tokens"]),
        ):
            if time.monotonic() - run_started > ticket_facts["max_seconds"]:
                raise TimeoutError("authorized feature extraction time budget exceeded")
            id_order.extend(ids)
            feature_batches.append(features)
            if feature_dtype is None:
                feature_dtype = dtype
            elif feature_dtype != dtype:
                raise ValueError("feature dtype changed across extraction batches")
        if len(id_order) != len(expected_ids) or len(set(id_order)) != len(id_order) or set(id_order) != expected_ids:
            raise ValueError("Prompt feature IDs do not exactly match the frozen Train supervision IDs")
        features = torch.cat(feature_batches, dim=0).contiguous()
        if features.shape[0] != len(expected_ids) or features.dtype not in {
            torch.float16, torch.bfloat16, torch.float32,
        }:
            raise ValueError("feature shape or dtype is not supported by the A5 cache contract")
        feature_path = output_dir / "features.safetensors"
        save_file({"features": features}, str(feature_path))
        ids_path = output_dir / "sample_ids.json"
        _write_json(ids_path, id_order)
        output_hashes = {
            "features.safetensors": sha256_file(feature_path),
            "sample_ids.json": sha256_file(ids_path),
        }
        from participant import run_inference as inference

        base_after = inference._weight_identity(_resolve(config["model_assets"]["base_model_path"]), {
            "model_id": config["model_assets"]["base_model_id"],
            "model_revision": config["model_assets"]["base_model_revision"],
        })
        adapter_after, _ = inference.adapter_weight_identity(_resolve(config["model_assets"]["source_adapter_path"]))
        if (
            base_after.get("observed_weight_hash") != base_before.get("observed_weight_hash")
            or adapter_after.get("weight_sha256") != adapter_before.get("weight_sha256")
            or adapter_after.get("config_sha256") != adapter_before.get("config_sha256")
        ):
            raise ValueError("read-only Base or E01-B Adapter identity changed during feature extraction")
        gpu_facts.update(
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
        manifest = {
            "status": "gpu_feature_extraction_passed",
            "experiment_id": config["experiment_id"],
            "run_id": args.run_id,
            "stage": "frozen_prompt_feature_extraction",
            "config_sha256": config_sha,
            "a5_audit_run_id": args.audit_run_id,
            "a5_audit_manifest_sha256": audit["manifest_sha256"],
            "supervision_sha256": audit["manifest_value"]["outputs"]["supervision.jsonl"]["sha256"],
            "split_manifest_sha256": audit["manifest_value"]["outputs"]["conversation_splits.json"]["sha256"],
            "source_hashes": audit["manifest_value"]["source_hashes"],
            "ticket_sha256": sha256_file(_resolve(args.ticket)),
            "ticket_facts": ticket_facts,
            "feature_contract": {
                "input_source": "D0.4.3 input_ids[:prompt_length] only",
                "position": "prompt_length_minus_one",
                "layer": "final_hidden_state",
                "batch_size": batch_size,
                "feature_dtype": feature_dtype,
                "head_compute_dtype": "float32",
                "row_count": int(features.shape[0]),
                "hidden_size": int(features.shape[1]),
                "sample_ids_sha256": canonical_sha256(id_order),
                "gold_completion_passed_to_model": False,
                "dev_rows_read": 0,
                "val_rows_read": 0,
            },
            "model_identity": {
                "base_weight_sha256": base_after["observed_weight_hash"],
                "adapter_weight_sha256": adapter_after["weight_sha256"],
                "adapter_config_sha256": adapter_after["config_sha256"],
                "base_weights_unchanged": True,
                "source_adapter_unchanged": True,
            },
            "model_facts": model_facts,
            "gpu": gpu_facts,
            "outputs": output_hashes,
            "started_at_utc": started_at,
            "ended_at_utc": _now(),
            "elapsed_seconds": time.monotonic() - run_started,
        }
        _write_json(output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        failure = {
            "status": "gpu_feature_extraction_failed",
            "experiment_id": config["experiment_id"],
            "run_id": args.run_id,
            "stage": "frozen_prompt_feature_extraction",
            "config_sha256": config_sha,
            "a5_audit_run_id": args.audit_run_id,
            "a5_audit_manifest_sha256": audit["manifest_sha256"],
            "ticket_sha256": sha256_file(_resolve(args.ticket)),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gpu": gpu_facts,
            "partial_outputs": output_hashes,
            "started_at_utc": started_at,
            "ended_at_utc": _now(),
            "elapsed_seconds": time.monotonic() - run_started,
        }
        _write_json(output_dir / "run_manifest.json", failure)
        raise
    print(json.dumps({
        "status": manifest["status"],
        "run_id": args.run_id,
        "row_count": manifest["feature_contract"]["row_count"],
        "feature_dtype": manifest["feature_contract"]["feature_dtype"],
        "hidden_size": manifest["feature_contract"]["hidden_size"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0


def _fit_cpu(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import load_file, save_file

    config_path = _resolve(args.config)
    config, config_sha = _load_config(config_path)
    audit = _verify_audit_run(config, config_sha, args.audit_run_id)
    feature_dir = _resolve(config["outputs"]["probe_root"]) / args.feature_run_id
    feature_manifest = _json(feature_dir / "run_manifest.json")
    if feature_manifest.get("status") != "gpu_feature_extraction_passed":
        raise ValueError("probe fitting requires one passed, frozen A5 feature extraction Run")
    if feature_manifest.get("config_sha256") != config_sha:
        raise ValueError("feature extraction Run uses a different A5 config")
    if feature_manifest.get("a5_audit_manifest_sha256") != audit["manifest_sha256"]:
        raise ValueError("feature extraction Run does not bind the requested A5.0 audit")
    feature_path = feature_dir / "features.safetensors"
    ids_path = feature_dir / "sample_ids.json"
    if sha256_file(feature_path) != feature_manifest.get("outputs", {}).get("features.safetensors"):
        raise ValueError("feature cache SHA differs from the feature Run Manifest")
    if sha256_file(ids_path) != feature_manifest.get("outputs", {}).get("sample_ids.json"):
        raise ValueError("feature ID list SHA differs from the feature Run Manifest")
    feature_ids = _json(ids_path)
    if not isinstance(feature_ids, list) or len(feature_ids) != len(set(feature_ids)):
        raise ValueError("feature ID order must be a unique JSON list")
    split_ids = _read_splits(audit["splits"])
    expected_ids = {sample_id for values in split_ids.values() for sample_id in values}
    if set(feature_ids) != expected_ids:
        raise ValueError("feature cache ID set differs from the frozen Train partitions")
    supervision = _read_supervision(audit["supervision"])
    enums_schema = _json(_resolve(config["data"]["official_train_schema"]))
    output_schema = _json(_resolve(config["data"]["official_output_schema"]))
    _label_order, enums = schema_label_order(enums_schema, output_schema)
    feature_tensors = load_file(str(feature_path), device="cpu")
    features = feature_tensors.get("features")
    if not isinstance(features, torch.Tensor) or features.shape[0] != len(feature_ids):
        raise ValueError("feature tensor row count differs from feature_ids.json")
    feature_dtype = str(features.dtype).removeprefix("torch.")
    if feature_dtype != feature_manifest.get("feature_contract", {}).get("feature_dtype"):
        raise ValueError("feature tensor dtype differs from extraction Manifest")
    features_by_id = {sample_id: features[index] for index, sample_id in enumerate(feature_ids)}
    report_head, report = fit_and_evaluate_probe(
        features_by_id,
        supervision,
        split_ids,
        enums,
        seed=int(config["probe"]["seed"]),
        shuffled_seed=int(config["probe"]["shuffled_label_seed"]),
        epochs=int(config["probe"]["epochs"]),
        batch_size=512,
        learning_rate=float(config["probe"]["learning_rate"]),
    )
    output_root = _resolve(config["outputs"]["probe_root"])
    output_dir = output_root / args.run_id
    if not output_dir.is_relative_to(_resolve("outputs")):
        raise ValueError("probe outputs must remain under ignored outputs/")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    head_path = output_dir / "probe_head.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in report_head.state_dict().items()}, str(head_path))
    report_path = output_dir / "probe_report.json"
    report.update({
        "run_id": args.run_id,
        "experiment_id": config["experiment_id"],
        "feature_run_id": args.feature_run_id,
        "feature_run_manifest_sha256": sha256_file(feature_dir / "run_manifest.json"),
        "a5_audit_run_id": args.audit_run_id,
        "a5_audit_manifest_sha256": audit["manifest_sha256"],
        "feature_cache_sha256": sha256_file(feature_path),
        "feature_dtype": feature_dtype,
        "dev_224_status": "not_run",
        "created_at_utc": _now(),
    })
    _write_json(report_path, report)
    manifest = {
        "status": report["status"],
        "run_id": args.run_id,
        "config_sha256": config_sha,
        "feature_run_manifest_sha256": sha256_file(feature_dir / "run_manifest.json"),
        "a5_audit_manifest_sha256": audit["manifest_sha256"],
        "outputs": {
            "probe_head.safetensors": sha256_file(head_path),
            "probe_report.json": sha256_file(report_path),
        },
        "created_at_utc": report["created_at_utc"],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "run_id": args.run_id,
        "feature_run_id": args.feature_run_id,
        "holdout_linear_f1": report["holdout"]["linear_head_calibrated"]["metrics"]["overall_33_label_micro_f1"],
        "probe_decision": report["probe_decision"],
        "dev_224_status": report["dev_224_status"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0


def _cpu_preflight(args: argparse.Namespace) -> int:
    config, config_sha = _load_config(_resolve(args.config))
    audit = _verify_audit_run(config, config_sha, args.audit_run_id)
    report = audit["report_value"]
    print(json.dumps({
        "status": "cpu_preflight_passed",
        "config_sha256": config_sha,
        "audit_run_id": args.audit_run_id,
        "audit_manifest_sha256": audit["manifest_sha256"],
        "supervision_rows": report["join"]["joined_rows"],
        "fit_rows": report["partition_counts"]["fit"]["sample_count"],
        "calibration_rows": report["partition_counts"]["calibration"]["sample_count"],
        "internal_holdout_rows": report["partition_counts"]["internal_holdout"]["sample_count"],
        "gpu_run_started": False,
    }, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/e03_a5.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("cpu-preflight")
    preflight.add_argument("--audit-run-id", required=True)
    extract = subparsers.add_parser("extract-train")
    extract.add_argument("--audit-run-id", required=True)
    extract.add_argument("--run-id", required=True)
    extract.add_argument("--ticket", type=Path, required=True)
    fit = subparsers.add_parser("fit-cpu")
    fit.add_argument("--audit-run-id", required=True)
    fit.add_argument("--feature-run-id", required=True)
    fit.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "cpu-preflight":
            return _cpu_preflight(args)
        if args.command == "extract-train":
            return _extract_train(args)
        if args.command == "fit-cpu":
            return _fit_cpu(args)
    except RunAuthorizationError as exc:
        print(json.dumps({"status": "blocked_run_authorization", "error": str(exc)}, ensure_ascii=False))
        return 3
    except Exception as exc:
        print(json.dumps({"status": f"{args.command}_failed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
