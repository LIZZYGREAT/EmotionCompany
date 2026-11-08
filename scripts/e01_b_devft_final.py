#!/usr/bin/env python3
"""Archive E01-B/A8, restore the E01-B Trainer state, and train on Dev once."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from src.data.e01_b_devft_final import (
    build_dev_schedule,
    canonical_sha256,
    validate_dev_schedule,
    validate_optimizer_state_layout,
)
from src.data.sft_collator import SFTDataCollator
from src.training import training_sft


DEFAULT_CONFIG = ROOT / "configs/training/e01_b_devft_final.yaml"
BASE_SHA256 = "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5"
PARENT_RUN_ID = "E01-B-FULL-20261028-133805"
A8_RUN_ID = "E03-A8-DEVFT-20261108-02-PDEV-1EPOCH"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path, *, allow_external_symlink: bool = False) -> Path:
    path = Path(value).expanduser()
    lexical = Path(os.path.abspath(path if path.is_absolute() else ROOT / path))
    if not lexical.is_relative_to(ROOT.resolve()):
        raise ValueError(f"configured path escapes the repository: {value}")
    resolved = lexical.resolve()
    if not allow_external_symlink and not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"configured path resolves outside the repository: {value}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"{path}:{line_number}: blank or incomplete JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _checkpoint_facts(checkpoint: Path) -> list[dict[str, Any]]:
    facts = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint must not contain symlinks: {path}")
        if path.is_file() and path.name != "checkpoint_manifest.json":
            facts.append({
                "path": path.relative_to(checkpoint).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return facts


def _verify_checkpoint(checkpoint: Path, *, expected_run_id: str, expected_step: int) -> dict[str, Any]:
    manifest = _read_json(checkpoint / "checkpoint_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("run_id") != expected_run_id:
        raise ValueError(f"checkpoint identity/status mismatch: {checkpoint}")
    step = manifest.get("optimizer_step", manifest.get("global_optimizer_step"))
    if step != expected_step:
        raise ValueError(f"checkpoint step mismatch: expected={expected_step}, observed={step}")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("files_sha256") != canonical_sha256(files):
        raise ValueError(f"checkpoint file manifest hash is invalid: {checkpoint}")
    actual = _checkpoint_facts(checkpoint)
    expected = sorted(files, key=lambda item: item.get("path", ""))
    if actual != expected:
        raise ValueError(f"checkpoint file SHA/size mismatch: {checkpoint}")
    required = {"adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"}
    if not required <= {item["path"] for item in actual}:
        raise ValueError(f"checkpoint is missing recoverable state files: {checkpoint}")
    if not any(item["path"].startswith("adapter_model") for item in actual):
        raise ValueError(f"checkpoint is missing Adapter weights: {checkpoint}")
    return manifest


def _copytree_checked(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"archive source contains a symlink: {source}")
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def _archive_facts(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"source_manifest.json", "SHA256SUMS.txt"}
    ]


def _verify_archive(path: Path, source_lock: dict[str, Any]) -> dict[str, Any]:
    source_manifest = _read_json(path / "source_manifest.json")
    if source_manifest.get("source_lock") != source_lock:
        raise ValueError(f"existing final-candidate archive has a different source identity: {path}")
    actual = _archive_facts(path)
    if actual != source_manifest.get("files"):
        raise ValueError(f"final-candidate archive hash mismatch: {path}")
    sums_path = path / "SHA256SUMS.txt"
    expected_sums = "".join(f"{item['sha256']}  {item['path']}\n" for item in actual)
    if not sums_path.is_file() or sums_path.read_text(encoding="utf-8") != expected_sums:
        raise ValueError(f"final-candidate SHA256SUMS mismatch: {path}")
    return source_manifest


def _archive_candidate(
    archive_root: Path,
    name: str,
    source_lock: dict[str, Any],
    copy_items: list[tuple[str, Path]],
) -> dict[str, Any]:
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / name
    if destination.exists():
        return _verify_archive(destination, source_lock)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=archive_root))
    try:
        for relative, source in copy_items:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _copytree_checked(source, target)
        files = _archive_facts(temporary)
        source_manifest = {
            "version": "E01-B-final-candidate-archive-v1",
            "candidate": name,
            "source_lock": source_lock,
            "files": files,
            "files_sha256": canonical_sha256(files),
            "created_at_utc": _now(),
        }
        _write_json(temporary / "source_manifest.json", source_manifest, exclusive=True)
        sums_path = temporary / "SHA256SUMS.txt"
        sums_path.write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in files), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
        return _verify_archive(destination, source_lock)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _unique_by(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{label} has a missing or duplicate {field}")
        result[value] = row
    return result


def _verify_data(config: dict[str, Any], original_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_path, dev_path, _normalized_train, manifests = training_sft._training_rows(original_config)
    sft_manifest = manifests["sft"]
    d03_manifest = manifests["d03"]
    d043_path = _resolve(config["data"]["sft_manifest"])
    d03_path = _resolve(config["data"]["d03_manifest"])
    d042_path = _resolve(config["data"]["d042_manifest"])
    if _sha256(d03_path) != sft_manifest.get("d03_manifest_sha256"):
        raise ValueError("D0.4.3 points to a different D0.3 Manifest")
    if _sha256(d042_path) != sft_manifest.get("d042_manifest_sha256"):
        raise ValueError("D0.4.3 points to a different D0.4.2 Manifest")
    if _sha256(_resolve(config["protocol"]["prompt_path"])) != config["protocol"]["prompt_sha256"]:
        raise ValueError("protected P0 hash differs from the approved E01-B input")

    normalized_dev_path = _resolve(sft_manifest["inputs"]["normalized_dev"]["path"])
    expected_dev_sha = sft_manifest["inputs"]["normalized_dev"]["sha256"]
    if _sha256(normalized_dev_path) != expected_dev_sha:
        raise ValueError("normalized D0.3 Dev hash differs from the D0.4.3 source Manifest")
    d03_output = next((row for row in d03_manifest.get("outputs", []) if row.get("path") == normalized_dev_path.relative_to(ROOT).as_posix()), None)
    if not d03_output or d03_output.get("sha256") != expected_dev_sha or d03_output.get("records") != 1833:
        raise ValueError("normalized Dev is not the exact 1,833-row D0.3 source")

    budget_path = _resolve(sft_manifest["inputs"]["d042_budget_rows"]["path"])
    if _sha256(budget_path) != sft_manifest["inputs"]["d042_budget_rows"]["sha256"]:
        raise ValueError("D0.4.2 budget audit changed after D0.4.3 preparation")
    normalized_rows = _unique_by(_read_jsonl(normalized_dev_path), "sample_id", "D0.3 Dev")
    sft_rows = _unique_by(_read_jsonl(dev_path), "sample_id", "D0.4.3 Dev")
    budget_rows = {}
    for row in _read_jsonl(budget_path):
        if row.get("split") != "dev" or row.get("max_prompt_tokens") != 1280 or row.get("max_train_seq_length") != 2048:
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in budget_rows:
            raise ValueError("D0.4.2 has a missing or duplicate Dev budget ID")
        budget_rows[sample_id] = row
    accepted = {sample_id for sample_id, row in budget_rows.items() if row.get("outcome") in {"full_fit", "trimmed_fit"}}
    expected_count = int(config["data"]["allowed_rows"])
    if len(normalized_rows) != 1833 or len(sft_rows) != expected_count or len(budget_rows) != 1833:
        raise ValueError("D0.3/D0.4.2/D0.4.3 Dev counts differ from the frozen source counts")
    if accepted != set(sft_rows) or set(sft_rows) - set(normalized_rows):
        raise ValueError("D0.4.3 rows do not exactly match all accepted 1280/2048 Dev IDs")
    if len(normalized_rows) - len(accepted) != int(config["data"]["excluded_rows"]):
        raise ValueError("D0.4.2 Dev exclusion count differs from the task plan")

    for sample_id, row in sft_rows.items():
        source = normalized_rows[sample_id]
        if row.get("source_split") != "dev" or row.get("conversation_id") != source.get("conversation_id"):
            raise ValueError(f"D0.4.3 Dev identity differs from D0.3 for sample {sample_id}")
        training_sft.validate_train_row(
            row,
            expected_eos_token_id=int(original_config["protocol"]["eos_token_id"]),
            expected_split="dev",
        )

    ids = list(sft_rows)
    schedule = build_dev_schedule(ids, seed=int(config["schedule"]["seed"]), group_size=int(config["schedule"]["group_size"]))
    exclusion_counts: dict[str, int] = {}
    for sample_id, row in budget_rows.items():
        if sample_id not in accepted:
            outcome = str(row.get("outcome", "missing_outcome"))
            exclusion_counts[outcome] = exclusion_counts.get(outcome, 0) + 1
    facts = {
        "d03_manifest_sha256": _sha256(d03_path),
        "d043_manifest_sha256": _sha256(d043_path),
        "d042_manifest_sha256": _sha256(d042_path),
        "d03_normalized_dev_sha256": _sha256(normalized_dev_path),
        "sft_dev_sha256": _sha256(dev_path),
        "d042_budget_rows_sha256": _sha256(budget_path),
        "sft_dev_rows": len(sft_rows),
        "normalized_dev_rows": len(normalized_rows),
        "excluded_dev_rows": len(normalized_rows) - len(accepted),
        "excluded_outcomes": dict(sorted(exclusion_counts.items())),
        "sample_ids_sha256": canonical_sha256(ids),
        "sample_order_sha256": canonical_sha256(schedule),
        "schedule_steps": len(schedule),
        "last_group_size": len(schedule[-1]["sample_ids"]),
    }
    return list(sft_rows.values()), schedule, facts


def _verify_a8_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    a8_root = _resolve(config["a8_candidate"]["run_root"])
    checkpoint = _resolve(config["a8_candidate"]["checkpoint"])
    eval_root = _resolve(config["a8_candidate"]["evaluation_root"])
    a8_manifest = _read_json(a8_root / "manifest.json")
    if (
        _sha256(a8_root / "manifest.json") != config["a8_candidate"]["expected_run_manifest_sha256"]
        or a8_manifest.get("run_id") != A8_RUN_ID
        or a8_manifest.get("status") != "scored_local_proxy"
        or a8_manifest.get("optimizer_steps") != 4504
    ):
        raise ValueError("A8 P4504 Run is not the expected completed candidate")
    checkpoint_manifest = _verify_checkpoint(checkpoint, expected_run_id=A8_RUN_ID, expected_step=4504)
    if _sha256(checkpoint / "checkpoint_manifest.json") != config["a8_candidate"]["expected_checkpoint_manifest_sha256"]:
        raise ValueError("A8 P4504 checkpoint Manifest hash differs from the frozen identity")
    eval_generation = _read_json(eval_root / "generation_manifest.json")
    eval_scoring_manifest = _read_json(eval_root / "scoring_manifest.json")
    val_input_manifest = _read_json(eval_root / "val_input_manifest.json")
    if eval_generation.get("status") != "predictions_frozen" or eval_generation.get("run_id") != A8_RUN_ID:
        raise ValueError("A8 Val predictions are not frozen under the expected Run")
    if eval_scoring_manifest.get("status") != "scored_after_prediction_freeze" or eval_scoring_manifest.get("run_id") != A8_RUN_ID:
        raise ValueError("A8 Val score was not produced after its prediction freeze")
    if val_input_manifest.get("status") != "target_free_val_frozen" or val_input_manifest.get("run_id") != A8_RUN_ID:
        raise ValueError("A8 target-free Val Manifest is not frozen under the expected Run")
    for filename, expected_hash in val_input_manifest.get("files", {}).items():
        if _sha256(eval_root / filename) != expected_hash:
            raise ValueError(f"A8 frozen Val input changed: {filename}")
    if (
        _sha256(eval_root / "generation_manifest.json") != config["a8_candidate"]["expected_generation_manifest_sha256"]
        or _sha256(eval_root / "predictions.jsonl") != eval_generation.get("prediction_sha256")
        or _sha256(eval_root / "predictions.jsonl") != eval_scoring_manifest.get("prediction_sha256")
    ):
        raise ValueError("A8 prediction file or generation identity changed")
    d03_path = _resolve(config["data"]["d03_manifest"])
    if _sha256(d03_path) != val_input_manifest.get("source", {}).get("d03_manifest_sha256"):
        raise ValueError("A8 Val input points to a different D0.3 Manifest")
    target_free = _read_jsonl(eval_root / "val_target_free.jsonl")
    id_map = _read_jsonl(eval_root / "val_id_map.jsonl")
    if len(target_free) != 2354 or len(id_map) != 2354:
        raise ValueError("A8 frozen Val input/ID map must each contain 2,354 rows")
    input_by_id = _unique_by(target_free, "id", "A8 target-free Val")
    map_by_id = _unique_by(id_map, "eval_id", "A8 Val ID map")
    if set(input_by_id) != set(map_by_id):
        raise ValueError("A8 target-free Val and ID map ID sets differ")
    for row in target_free:
        if set(row) != {"id", "history"} or not isinstance(row.get("history"), list) or not row["history"]:
            raise ValueError("A8 frozen Val input is not target-free id+history data")
    predictions = _read_jsonl(eval_root / "predictions.jsonl")
    if len(predictions) != 2354 or set(_unique_by(predictions, "id", "A8 predictions")) != set(input_by_id):
        raise ValueError("A8 P4504 predictions do not cover the same 2,354 Val IDs")
    return {
        "run_manifest_sha256": _sha256(a8_root / "manifest.json"),
        "checkpoint_manifest_sha256": _sha256(checkpoint / "checkpoint_manifest.json"),
        "generation_manifest_sha256": _sha256(eval_root / "generation_manifest.json"),
        "scoring_manifest_sha256": _sha256(eval_root / "scoring_manifest.json"),
        "val_input_manifest_sha256": _sha256(eval_root / "val_input_manifest.json"),
        "prediction_sha256": _sha256(eval_root / "predictions.jsonl"),
        "val_input_sha256": _sha256(eval_root / "val_target_free.jsonl"),
        "val_id_map_sha256": _sha256(eval_root / "val_id_map.jsonl"),
        "target_free_count": len(target_free),
        "prediction_count": len(predictions),
    }, checkpoint_manifest, target_free, id_map


def _load_runtime_inference_config(adapter_path: Path, base_path: Path, expected_adapter_sha: str, expected_adapter_config_sha: str) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    raw = _read_json(ROOT / "participant/configs/inference_config.json")
    raw["model_path"] = str(base_path)
    raw["format_version"] = "F2"
    raw["expected_weight_sha256"] = BASE_SHA256
    raw["quantization"]["enabled"] = True
    raw["context"]["max_prompt_tokens"] = 1280
    raw["generation"] = {"max_new_tokens": 32, "do_sample": False, "temperature": 1.0, "top_p": 1.0}
    raw["adapter"] = {
        "enabled": True,
        "path": str(adapter_path),
        "expected_base_weight_sha256": BASE_SHA256,
        "expected_adapter_weight_sha256": expected_adapter_sha,
        "expected_adapter_config_sha256": expected_adapter_config_sha,
    }
    prompt, schema, enums, prompt_facts = inference.load_runtime_protocol(raw)
    return raw, prompt, schema, enums, prompt_facts


def _verify_archive_model_load(adapter_path: Path, expected_adapter_sha: str, expected_config_sha: str, sample: dict[str, Any]) -> dict[str, Any]:
    import torch
    from scripts.e03_profile_fast_eval import _load_model

    base_path = _resolve("participant/model", allow_external_symlink=True)
    config, prompt, _schema, _enums, prompt_facts = _load_runtime_inference_config(
        adapter_path, base_path, expected_adapter_sha, expected_config_sha,
    )
    model, tokenizer, _model_config, facts = _load_model(config, inference, torch)
    selection, _messages = inference.prepare_sample_messages(
        sample, prompt, tokenizer, int(config["context"]["max_prompt_tokens"]),
    )
    if selection.get("floor_exceeds_cap"):
        raise ValueError("fixed target-free reload probe exceeds the locked P0 prompt cap")
    inference.synchronize(torch)
    raw, generation = inference.generate_transformers_sample(
        model, torch, tokenizer, list(selection["prompt_ids"]), config["generation"],
    )
    inference.synchronize(torch)
    result = {
        "status": "passed",
        "adapter_path": str(adapter_path.relative_to(ROOT)),
        "adapter_weight_sha256": facts.get("adapter_weight_sha256"),
        "adapter_config_sha256": facts.get("adapter_config_sha256"),
        "base_weight_sha256": facts.get("model_weight_identity_sha256"),
        "tokenizer_signature_sha256": prompt_facts.get("tokenizer_signature_sha256"),
        "probe_id": sample["id"],
        "probe_prompt_ids_sha256": canonical_sha256(selection["prompt_ids"]),
        "probe_generated_text_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "probe_generated_tokens": generation["generated_tokens"],
        "quantization": facts.get("quantization"),
        "model_load_seconds": facts.get("model_load_seconds"),
        "cuda_device": facts.get("cuda_device"),
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _archive_sources(config: dict[str, Any], parent_manifest: dict[str, Any], parent_checkpoint_manifest: dict[str, Any], a8_facts: dict[str, Any], a8_checkpoint_manifest: dict[str, Any]) -> dict[str, Any]:
    archive_root = _resolve(config["archive_root"])
    parent_root = _resolve(config["parent"]["run_root"])
    parent_ckpt = _resolve(config["parent"]["checkpoint"])
    a8_root = _resolve(config["a8_candidate"]["run_root"])
    a8_ckpt = _resolve(config["a8_candidate"]["checkpoint"])
    parent_adapter_identity, _ = inference.adapter_weight_identity(parent_root / "adapter")
    checkpoint_adapter_identity, _ = inference.adapter_weight_identity(parent_ckpt)
    if parent_adapter_identity["weight_sha256"] != checkpoint_adapter_identity["weight_sha256"]:
        raise ValueError("E01-B top-level Adapter and checkpoint Adapter differ")
    parent_lock = {
        "run_id": PARENT_RUN_ID,
        "optimizer_step": 4048,
        "run_manifest_sha256": _sha256(parent_root / "manifest.json"),
        "checkpoint_manifest_sha256": _sha256(parent_ckpt / "checkpoint_manifest.json"),
        "adapter_weight_sha256": parent_adapter_identity["weight_sha256"],
        "adapter_config_sha256": parent_adapter_identity["config_sha256"],
        "base_weight_sha256": BASE_SHA256,
    }
    a8_adapter_identity, _ = inference.adapter_weight_identity(a8_ckpt)
    a8_lock = {
        "run_id": A8_RUN_ID,
        "optimizer_step": 4504,
        "run_manifest_sha256": _sha256(a8_root / "manifest.json"),
        "checkpoint_manifest_sha256": _sha256(a8_ckpt / "checkpoint_manifest.json"),
        "adapter_weight_sha256": a8_adapter_identity["weight_sha256"],
        "adapter_config_sha256": a8_adapter_identity["config_sha256"],
        "base_weight_sha256": BASE_SHA256,
        "evaluation_prediction_sha256": a8_facts["prediction_sha256"],
    }
    e01_archive = _archive_candidate(
        archive_root,
        "E01-B-TRAIN4048",
        parent_lock,
        [("adapter", parent_root / "adapter"), ("checkpoint-4048", parent_ckpt)],
    )
    adapter_dir = archive_root / "A8-P4504" / "adapter"
    if not (archive_root / "A8-P4504").exists():
        temporary_parent = Path(tempfile.mkdtemp(prefix=".A8-P4504.", dir=archive_root))
        try:
            temporary_adapter = temporary_parent / "adapter"
            temporary_adapter.mkdir()
            for path in sorted(a8_ckpt.iterdir()):
                if path.is_file() and (path.name == "adapter_config.json" or path.name.startswith("adapter_model") or path.name == "README.md"):
                    shutil.copy2(path, temporary_adapter / path.name)
            _copytree_checked(a8_ckpt, temporary_parent / "checkpoint-0456")
            files = _archive_facts(temporary_parent)
            a8_source_manifest = {
                "version": "E01-B-final-candidate-archive-v1",
                "candidate": "A8-P4504",
                "source_lock": a8_lock,
                "files": files,
                "files_sha256": canonical_sha256(files),
                "created_at_utc": _now(),
            }
            _write_json(temporary_parent / "source_manifest.json", a8_source_manifest, exclusive=True)
            (temporary_parent / "SHA256SUMS.txt").write_text(
                "".join(f"{item['sha256']}  {item['path']}\n" for item in files), encoding="utf-8",
            )
            destination = archive_root / "A8-P4504"
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(temporary_parent, destination)
        finally:
            if temporary_parent.exists():
                shutil.rmtree(temporary_parent)
    a8_archive = _verify_archive(archive_root / "A8-P4504", a8_lock)
    return {
        "e01_b_train4048": {
            "path": str((archive_root / "E01-B-TRAIN4048").relative_to(ROOT)),
            "source_manifest_sha256": _sha256(archive_root / "E01-B-TRAIN4048/source_manifest.json"),
            "files_sha256": e01_archive["files_sha256"],
            "adapter_weight_sha256": parent_adapter_identity["weight_sha256"],
        },
        "a8_p4504": {
            "path": str((archive_root / "A8-P4504").relative_to(ROOT)),
            "source_manifest_sha256": _sha256(archive_root / "A8-P4504/source_manifest.json"),
            "files_sha256": a8_archive["files_sha256"],
            "adapter_weight_sha256": a8_adapter_identity["weight_sha256"],
        },
    }


def _parent_facts(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_root = _resolve(config["parent"]["run_root"])
    parent_ckpt = _resolve(config["parent"]["checkpoint"])
    parent_manifest = _read_json(parent_root / "manifest.json")
    if (
        _sha256(parent_root / "manifest.json") != config["parent"]["expected_run_manifest_sha256"]
        or parent_manifest.get("run_id") != PARENT_RUN_ID
        or parent_manifest.get("status") != "passed"
        or parent_manifest.get("training_scope") != "full_train"
        or parent_manifest.get("optimizer_steps") != 4048
        or parent_manifest.get("train_rows") != 16192
        or parent_manifest.get("base_weight_sha256") != BASE_SHA256
    ):
        raise ValueError("E01-B Full Run identity, status, data count, or Base hash differs")
    checkpoint_manifest = _verify_checkpoint(parent_ckpt, expected_run_id=PARENT_RUN_ID, expected_step=4048)
    if _sha256(parent_ckpt / "checkpoint_manifest.json") != config["parent"]["expected_checkpoint_manifest_sha256"]:
        raise ValueError("E01-B checkpoint Manifest hash differs from the frozen identity")
    if checkpoint_manifest.get("config_sha256") != parent_manifest.get("config_sha256"):
        raise ValueError("E01-B checkpoint and Run config identity differ")
    trainer_state = _read_json(parent_ckpt / "trainer_state.json")
    if int(trainer_state.get("global_step", -1)) != 4048:
        raise ValueError("E01-B Trainer state is not at global_step 4048")
    evidence = _read_json(parent_root / "training_evidence.json")
    standalone = _read_json(parent_root / "reload_check_standalone.json")
    if (
        evidence.get("optimizer_steps") != 4048
        or parent_manifest.get("base_weights_unchanged") is not True
        or parent_manifest.get("reload_check_status") != "passed"
        or standalone.get("status") != "passed"
        or standalone.get("token_ids_exact_match") is not True
    ):
        raise ValueError("E01-B training/reload evidence does not satisfy the frozen parent gate")
    adapter_identity, _adapter_config = inference.adapter_weight_identity(parent_root / "adapter")
    checkpoint_adapter_identity, _checkpoint_config = inference.adapter_weight_identity(parent_ckpt)
    expected_adapter = parent_manifest.get("adapter_identity", {})
    if (
        adapter_identity["weight_sha256"] != expected_adapter.get("weight_sha256")
        or adapter_identity["config_sha256"] != expected_adapter.get("config_sha256")
        or adapter_identity["weight_sha256"] != checkpoint_adapter_identity["weight_sha256"]
    ):
        raise ValueError("E01-B adapter directory and checkpoint do not match the Run Manifest")
    parent_config_path = parent_root / "config_snapshot.yaml"
    if _sha256(parent_config_path) != parent_manifest.get("config_sha256"):
        raise ValueError("E01-B approved config snapshot hash differs from the Run Manifest")
    original_config = yaml.safe_load(parent_config_path.read_text(encoding="utf-8"))
    if original_config.get("optimizer", {}).get("name") != "AdamW":
        raise ValueError("parent optimizer is not the declared AdamW")
    base_path = _resolve(original_config["base_model"]["local_path"], allow_external_symlink=True)
    base_identity = inference._weight_identity(base_path, {
        "model_id": original_config["base_model"]["model_id"],
        "model_revision": original_config["base_model"]["revision"],
    })
    if base_identity.get("observed_weight_hash") != BASE_SHA256:
        raise ValueError("local Base model files differ from the verified E01-B identity")
    return parent_manifest, checkpoint_manifest, original_config, {
        "run_manifest_sha256": _sha256(parent_root / "manifest.json"),
        "checkpoint_manifest_sha256": _sha256(parent_ckpt / "checkpoint_manifest.json"),
        "config_snapshot_sha256": _sha256(parent_config_path),
        "adapter_weight_sha256": adapter_identity["weight_sha256"],
        "adapter_config_sha256": adapter_identity["config_sha256"],
        "base_weight_sha256": base_identity["observed_weight_hash"],
        "base_file_count": len(base_identity.get("files", [])),
        "checkpoint_optimizer_step": int(trainer_state["global_step"]),
    }


def prepare_run(config_path: Path, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    config = _load_config(config_path)
    selected_run_id = run_id or config["run_id"]
    run_root = _resolve(config["output_root"]) / selected_run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing DevFT Run: {run_root}")

    parent_manifest, parent_checkpoint_manifest, original_config, parent_facts = _parent_facts(config)
    dev_rows, schedule, data_facts = _verify_data(config, original_config)
    a8_facts, a8_checkpoint_manifest, target_free, id_map = _verify_a8_inputs(config)
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": "E01-B-DEVFT-Final-run-v1",
        "experiment_id": "E01-B-DEVFT-FINAL",
        "run_id": selected_run_id,
        "status": "preparing",
        "stage": "archive_and_independent_load_checks",
        "parent": parent_facts,
        "a8_source": a8_facts,
        "data": data_facts,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "training_config_sha256": parent_manifest.get("config_sha256"),
        "checkpoint_manifest_sha256": parent_facts["checkpoint_manifest_sha256"],
        "parent_checkpoint_files_sha256": parent_checkpoint_manifest["files_sha256"],
        "a8_checkpoint_files_sha256": a8_checkpoint_manifest["files_sha256"],
        "official_val_read": False,
        "created_at_utc": _now(),
    }
    _write_json(run_root / "manifest.json", manifest, exclusive=True)
    try:
        _write_json(run_root / "source_manifest.json", {
            "parent_run": parent_facts,
            "a8_run": a8_facts,
            "dev_data": data_facts,
            "target_free_val": {
                "rows": len(target_free),
                "ids_sha256": canonical_sha256([row["id"] for row in target_free]),
                "id_map_sha256": _sha256(_resolve(config["a8_candidate"]["evaluation_root"]) / "val_id_map.jsonl"),
            },
        }, exclusive=True)
        (run_root / "sample_order.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in schedule),
            encoding="utf-8",
        )
        shutil.copy2(config_path, run_root / "config_snapshot.yaml")
        parent_config_path = _resolve(config["parent"]["run_root"]) / "config_snapshot.yaml"
        shutil.copy2(parent_config_path, run_root / "parent_config_snapshot.yaml")
        shutil.copy2(_resolve(config["parent"]["checkpoint"]) / "training_args.bin", run_root / "parent_training_args.bin")
        archive_facts = _archive_sources(
            config, parent_manifest, parent_checkpoint_manifest, a8_facts, a8_checkpoint_manifest,
        )
        manifest.update({
            "archive": archive_facts,
            "sample_order_sha256": _sha256(run_root / "sample_order.jsonl"),
            "target_free_val_count": len(target_free),
            "target_free_val_ids_sha256": canonical_sha256([row["id"] for row in target_free]),
        })

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the approved execution environment")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the approved E01-B BF16 compute mode is unavailable")
        base_path = _resolve(original_config["base_model"]["local_path"], allow_external_symlink=True)
        eval_root = _resolve(config["a8_candidate"]["evaluation_root"])
        first_val = target_free[0]
        a8_adapter = _resolve(config["archive_root"]) / "A8-P4504/adapter"
        a8_identity, _ = inference.adapter_weight_identity(a8_adapter)
        a8_reload = _verify_archive_model_load(
            a8_adapter, a8_identity["weight_sha256"], a8_identity["config_sha256"], first_val,
        )
        e01_adapter = _resolve(config["archive_root"]) / "E01-B-TRAIN4048/adapter"
        e01_identity, _ = inference.adapter_weight_identity(e01_adapter)
        e01_reload = _verify_archive_model_load(
            e01_adapter, e01_identity["weight_sha256"], e01_identity["config_sha256"], first_val,
        )
        if e01_reload["base_weight_sha256"] != BASE_SHA256 or a8_reload["base_weight_sha256"] != BASE_SHA256:
            raise ValueError("independent candidate load did not use the locked Base model")
        manifest["archive_independent_load"] = {"a8_p4504": a8_reload, "e01_b_train4048": e01_reload}
        manifest["status"] = "ready_for_training"
        manifest["stage"] = "preflight_passed"
        manifest["training_sample_count"] = len(dev_rows)
        manifest["schedule_steps"] = len(schedule)
        manifest["last_group_size"] = len(schedule[-1]["sample_ids"])
        manifest["updated_at_utc"] = _now()
        _write_json(run_root / "manifest.json", manifest)
        return config, run_root
    except BaseException as exc:
        manifest.update({
            "status": "failed",
            "stage": "preflight_failed",
            "stop_reason": type(exc).__name__,
            "error_summary": str(exc)[:1200],
            "updated_at_utc": _now(),
        })
        _write_json(run_root / "manifest.json", manifest)
        raise


def _load_rng_state(path: Path, torch: Any) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("parent RNG state is not a mapping")
    return state


def _capture_rng_state(torch: Any) -> dict[str, Any]:
    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any], torch: Any) -> None:
    import numpy as np

    python_state = state.get("python")
    numpy_state = state.get("numpy")
    cpu_state = state.get("cpu", state.get("torch"))
    cuda_state = state.get("cuda", state.get("torch_cuda_all", []))
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if cpu_state is not None:
        torch.set_rng_state(cpu_state.cpu())
    if cuda_state is not None and torch.cuda.is_available():
        if torch.is_tensor(cuda_state):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=0)
        elif len(cuda_state):
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def _verify_restored_rng_state(state: dict[str, Any], observed: dict[str, Any], torch: Any) -> bool:
    import pickle
    import numpy as np

    expected_cpu = state.get("cpu", state.get("torch_cpu"))
    if expected_cpu is None or not torch.equal(expected_cpu.cpu(), observed["cpu"].cpu()):
        return False
    if pickle.dumps(state.get("python")) != pickle.dumps(observed.get("python")):
        return False
    expected_numpy = state.get("numpy")
    observed_numpy = observed.get("numpy")
    if expected_numpy is not None and observed_numpy is not None:
        if expected_numpy[0] != observed_numpy[0] or expected_numpy[2:] != observed_numpy[2:]:
            return False
        if not np.array_equal(expected_numpy[1], observed_numpy[1]):
            return False
    if torch.cuda.is_available():
        expected_cuda = state.get("cuda", state.get("torch_cuda_all", []))
        observed_cuda = observed.get("cuda", [])
        if torch.is_tensor(expected_cuda):
            expected_cuda = [expected_cuda]
        if expected_cuda and len(expected_cuda) == 1 and len(observed_cuda) == 1:
            if not torch.equal(expected_cuda[0].cpu(), observed_cuda[0].cpu()):
                return False
        elif expected_cuda and len(expected_cuda) != len(observed_cuda):
            return False
    return True


def _seed_independent_dev_epoch(seed: int, torch: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cpu_optimizer_state(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_optimizer_state(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_optimizer_state(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_optimizer_state(item, torch) for item in value)
    return value


def _load_full_optimizer(model: Any, parent_checkpoint: Path, run_root: Path, torch: Any) -> tuple[Any, Any, dict[str, Any], Any]:
    from transformers import Trainer

    parent_args = torch.load(parent_checkpoint / "training_args.bin", map_location="cpu", weights_only=False)
    if (
        int(parent_args.max_steps) != 4048
        or int(parent_args.gradient_accumulation_steps) != 4
        or float(parent_args.max_grad_norm) != 1.0
        or float(parent_args.learning_rate) != 1e-4
        or float(parent_args.weight_decay) != 0.0
        or str(parent_args.lr_scheduler_type).split(".")[-1].lower() != "constant"
    ):
        raise ValueError("saved Trainer arguments differ from the E01-B recovery contract")
    factory_args = copy.copy(parent_args)
    factory_args.output_dir = str(run_root / "optimizer_factory")
    # This Trainer exists only to reproduce the parent's optimizer parameter
    # groups. Its saved training arguments may request periodic evaluation,
    # but no evaluation dataset is needed (or permitted) for this DevFT run.
    factory_args.eval_strategy = "no"
    factory = Trainer(model=model, args=factory_args)
    optimizer = factory.create_optimizer()

    name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    named_groups: list[list[tuple[str, Any]]] = []
    for group in optimizer.param_groups:
        named_group = []
        for parameter in group["params"]:
            name = name_by_parameter.get(id(parameter))
            if name is None:
                raise ValueError("reconstructed optimizer contains a parameter outside the model")
            named_group.append((name, parameter))
        named_groups.append(named_group)
    saved_state = torch.load(parent_checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    layout_facts = validate_optimizer_state_layout(saved_state, named_groups, expected_step=4048)
    if layout_facts["parameters_per_group"] != [392, 0]:
        raise ValueError(f"recreated Trainer group layout differs from E01-B: {layout_facts['parameters_per_group']}")
    if len(optimizer.param_groups) != 2:
        raise ValueError("recreated Trainer did not produce both original AdamW parameter groups")
    for index, group in enumerate(saved_state["param_groups"]):
        if float(group["lr"]) != 1e-4 or float(group["weight_decay"]) != 0.0:
            raise ValueError(f"saved AdamW group {index} hyperparameters differ from E01-B")
        if tuple(group.get("betas", ())) != (0.9, 0.999) or float(group.get("eps", -1)) != 1e-8:
            raise ValueError(f"saved AdamW group {index} beta/epsilon differs from E01-B")
    optimizer.load_state_dict(saved_state)
    scheduler = factory.create_scheduler(num_training_steps=4048, optimizer=optimizer)
    scheduler_state = torch.load(parent_checkpoint / "scheduler.pt", map_location="cpu", weights_only=False)
    if int(scheduler_state.get("last_epoch", -1)) != 4048 or len(scheduler_state.get("base_lrs", [])) != 2:
        raise ValueError("saved constant scheduler is not at Step 4048 with both original groups")
    scheduler.load_state_dict(scheduler_state)
    if int(scheduler.last_epoch) != 4048 or len(scheduler.base_lrs) != 2:
        raise ValueError("constant scheduler failed to restore at Step 4048")
    observed_steps = set()
    for parameter, state in optimizer.state.items():
        if tuple(state["exp_avg"].shape) != tuple(parameter.shape) or tuple(state["exp_avg_sq"].shape) != tuple(parameter.shape):
            raise ValueError("restored AdamW moment shape differs from its LoRA parameter")
        value = state["step"]
        observed_steps.add(int(value.item()) if hasattr(value, "item") else int(value))
    if observed_steps != {4048} or len(optimizer.state) != 392:
        raise ValueError("restored AdamW moment step/count differs from the E01-B checkpoint")
    return optimizer, scheduler, {
        **layout_facts,
        "optimizer_class": type(optimizer).__name__,
        "scheduler_class": type(scheduler).__name__,
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "beta1": float(optimizer.param_groups[0]["betas"][0]),
        "beta2": float(optimizer.param_groups[0]["betas"][1]),
        "epsilon": float(optimizer.param_groups[0]["eps"]),
        "max_grad_norm": float(parent_args.max_grad_norm),
        "training_args_sha256": _sha256(parent_checkpoint / "training_args.bin"),
    }, parent_args


def _lora_parameters(model: Any) -> list[tuple[str, Any]]:
    values = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(values) != 392 or any(".lora_A." not in name and ".lora_B." not in name for name, _ in values):
        raise ValueError("trainable parameter set differs from the 392 E01-B LoRA tensors")
    if any(parameter.dtype != __import__("torch").bfloat16 for _name, parameter in values):
        raise ValueError("restored E01-B LoRA tensor dtype is not BF16")
    return values


def _tensor_fingerprint(parameters: list[tuple[str, Any]], torch: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode("utf-8"))
        tensor = parameter.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fixed_prompt_probe(model: Any, tokenizer: Any, row: dict[str, Any], torch: Any) -> list[int]:
    prompt_ids = row["input_ids"][:row["prompt_length"]]
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    previous_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
            )
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
    return [int(value) for value in output[0, input_ids.shape[1]:].detach().cpu().tolist()]


def _checkpoint_file_facts(checkpoint: Path) -> list[dict[str, Any]]:
    return _checkpoint_facts(checkpoint)


def _save_checkpoint(
    *,
    checkpoint: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    parent_state: dict[str, Any],
    run_id: str,
    local_step: int,
    global_step: int,
    consumed_ids: list[str],
    schedule_sha256: str,
    loss_mean: float,
    training_seconds: float,
) -> dict[str, Any]:
    checkpoint.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    torch.save(_cpu_optimizer_state(optimizer.state_dict(), torch), checkpoint / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(_capture_rng_state(torch), checkpoint / "rng_state.pth")
    state = copy.deepcopy(parent_state)
    state.update({
        "global_step": global_step,
        "max_steps": 4504,
        "epoch": 1.0 + local_step / 456.0,
        "devft_step": local_step,
        "sample_cursor": len(consumed_ids),
        "consumed_sample_ids_sha256": canonical_sha256(consumed_ids),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "last_loss_mean": float(loss_mean),
        "devft_elapsed_seconds": float(training_seconds),
        "rng_state_saved": True,
    })
    history = list(state.get("log_history", []))
    history.append({
        "loss": float(loss_mean),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "step": global_step,
        "devft_step": local_step,
    })
    state["log_history"] = history
    _write_json(checkpoint / "trainer_state.json", state, exclusive=True)
    facts = _checkpoint_file_facts(checkpoint)
    record = {
        "version": "E01-B-DEVFT-Final-checkpoint-v1",
        "experiment_id": "E01-B-DEVFT-FINAL",
        "run_id": run_id,
        "optimizer_step": global_step,
        "devft_step": local_step,
        "sample_cursor": len(consumed_ids),
        "consumed_sample_ids_sha256": canonical_sha256(consumed_ids),
        "sample_order_sha256": schedule_sha256,
        "parent_checkpoint_manifest_sha256": _sha256(_resolve("outputs/final_candidates/E01-B-TRAIN4048/checkpoint-4048/checkpoint_manifest.json")),
        "base_weight_sha256": BASE_SHA256,
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "trainer_state_global_step": global_step,
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "loss_mean_through_step": float(loss_mean),
        "training_seconds": float(training_seconds),
        "files": facts,
        "files_sha256": canonical_sha256(facts),
        "status": "complete",
        "created_at_utc": _now(),
    }
    _write_json(checkpoint / "checkpoint_manifest.json", record, exclusive=True)
    _verify_checkpoint(checkpoint, expected_run_id=run_id, expected_step=global_step)
    return record


def _base_identity(config: dict[str, Any]) -> dict[str, Any]:
    base_path = _resolve("participant/model", allow_external_symlink=True)
    return inference._weight_identity(base_path, {
        "model_id": config["base_model"]["model_id"],
        "model_revision": config["base_model"]["revision"],
    })


def train_run(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "ready_for_training":
        raise ValueError("training requires a Run that passed archive and independent-load preflight")

    import torch
    import numpy as np
    from transformers import AutoTokenizer

    parent_checkpoint = _resolve("outputs/final_candidates/E01-B-TRAIN4048/checkpoint-4048")
    parent_run_root = _resolve(config["parent"]["run_root"])
    original_config = yaml.safe_load((run_root / "parent_config_snapshot.yaml").read_text(encoding="utf-8"))
    dev_rows, schedule, data_facts = _verify_data(config, original_config)
    schedule_bytes = (run_root / "sample_order.jsonl").read_bytes()
    if hashlib.sha256(schedule_bytes).hexdigest() != manifest.get("sample_order_sha256"):
        raise ValueError("frozen Dev schedule hash differs from the prepared Run")
    rows_by_id = {row["sample_id"]: row for row in dev_rows}
    schedule = _read_jsonl(run_root / "sample_order.jsonl")
    allowed_ids = list(rows_by_id)
    validate_dev_schedule(schedule, allowed_ids, group_size=4, parent_global_step=4048)
    if data_facts["sft_dev_sha256"] != manifest["data"]["sft_dev_sha256"]:
        raise ValueError("SFT Dev rows changed after preflight")
    disk = shutil.disk_usage(run_root)
    if disk.free < int(config["minimum_free_disk_gb"]) * 1024**3:
        raise OSError("free disk fell below the configured checkpoint reserve")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("training requires the verified CUDA BF16 environment")

    run_id = str(manifest["run_id"])
    manifest.update({"status": "running", "stage": "loading_parent_training_state", "updated_at_utc": _now()})
    _write_json(manifest_path, manifest)
    ledger_path = run_root / "step_ledger.jsonl"
    ledger_path.touch(exist_ok=False)
    tokenizer = model = None
    start_time = time.monotonic()
    try:
        inference_config = _read_json(ROOT / "participant/configs/inference_config.json")
        tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(
            inference_config, _resolve("participant/model", allow_external_symlink=True),
        )
        model, torch, model_facts = training_sft._load_train_model(
            original_config,
            _resolve(original_config["base_model"]["local_path"], allow_external_symlink=True),
            adapter_path=parent_checkpoint,
            is_trainable_adapter=True,
            autocast_adapter_dtype=False,
        )
        trainable = _lora_parameters(model)
        trainable_names_hash = canonical_sha256([name for name, _ in trainable])
        if not model_facts.get("real_nf4_verified") or model_facts.get("offload_devices"):
            raise RuntimeError("restored E01-B training model failed NF4/BF16/CUDA/no-offload verification")
        parent_tensor_sha = _tensor_fingerprint(trainable, torch)
        parent_rng_path = parent_checkpoint / "rng_state.pth"
        parent_rng = _load_rng_state(parent_rng_path, torch)
        _restore_rng_state(parent_rng, torch)
        parent_rng_restored = _capture_rng_state(torch)
        parent_rng_readable = _verify_restored_rng_state(parent_rng, parent_rng_restored, torch)
        if not parent_rng_readable:
            raise ValueError("parent Python/NumPy/Torch/CUDA RNG states did not restore exactly")

        optimizer, scheduler, optimizer_facts, parent_args = _load_full_optimizer(
            model, parent_checkpoint, run_root, torch,
        )
        _seed_independent_dev_epoch(int(config["schedule"]["seed"]), torch)
        base_before = _base_identity(original_config)
        if base_before.get("observed_weight_hash") != BASE_SHA256:
            raise ValueError("Base model identity changed before the first Dev optimizer step")
        tokenizer_config = inference_config["tokenizer"]
        if tokenizer_facts.get("signature_sha256") != tokenizer_config["expected_signature_sha256"]:
            raise ValueError("locked E01-B tokenizer signature differs from the inference runtime")
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        max_grad_norm = float(parent_args.max_grad_norm)
        checkpoints = list(config["schedule"]["checkpoint_dev_steps"])
        if checkpoints != [128, 256, 384, 456] or len(schedule) != 456:
            raise ValueError("checkpoint or optimizer-step schedule differs from the approved task plan")
        consumed_ids: list[str] = []
        losses_all: list[float] = []
        previous_trainer_state = _read_json(parent_checkpoint / "trainer_state.json")
        checkpoint_records: list[dict[str, Any]] = []
        schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
        manifest.update({
            "stage": "devft_training",
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "optimizer_reconstruction": optimizer_facts,
            "parent_rng_state_sha256": _sha256(parent_rng_path),
            "parent_rng_state_readable_and_restored": parent_rng_readable,
            "dev_epoch_rng_policy": "independent_python_numpy_torch_cuda_seed_42_after_parent_rng_restore",
            "initial_adapter_tensor_sha256": parent_tensor_sha,
            "trainable_parameter_names_sha256": trainable_names_hash,
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "trainable_tensor_count": len(trainable),
            "trainable_dtype": str(trainable[0][1].dtype),
            "max_grad_norm": max_grad_norm,
            "base_weight_sha256_before_training": base_before["observed_weight_hash"],
            "tokenizer_facts": tokenizer_facts,
            "model_facts": model_facts,
            "updated_at_utc": _now(),
        })
        _write_json(manifest_path, manifest)
        model.train()
        for group in schedule:
            local_step = int(group["dev_local_step"])
            global_step = int(group["global_step"])
            sample_ids = list(group["sample_ids"])
            actual_group_size = len(sample_ids)
            if not 1 <= actual_group_size <= 4:
                raise ValueError(f"invalid actual Dev accumulation group size at step {local_step}")
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            token_counts = {"prompt": 0, "completion": 0, "sequence": 0}
            torch.cuda.synchronize()
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            host_start = time.monotonic()
            for sample_id in sample_ids:
                row = rows_by_id[sample_id]
                batch = collator([row])
                device = model.get_input_embeddings().weight.device
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                loss = output.loss
                if loss is None or not bool(torch.isfinite(loss.detach()).all().item()):
                    raise FloatingPointError(f"non-finite E01-B completion loss at Dev-local step {local_step}")
                micro_losses.append(float(loss.detach().float().cpu().item()))
                token_counts["prompt"] += int(row["prompt_length"])
                token_counts["completion"] += int(row["completion_length"])
                token_counts["sequence"] += len(row["input_ids"])
                (loss / actual_group_size).backward()
                consumed_ids.append(sample_id)
            gradients = [parameter.grad for _, parameter in trainable]
            if any(gradient is None for gradient in gradients):
                raise RuntimeError(f"a LoRA parameter has no gradient at Dev-local step {local_step}")
            gradient_vector = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
            gradients_finite = bool(torch.isfinite(gradient_vector).all().item())
            gradient_nonzero_count = int(torch.count_nonzero(gradient_vector).item())
            if not gradients_finite or gradient_nonzero_count == 0:
                raise FloatingPointError(f"LoRA gradients are non-finite or all zero at Dev-local step {local_step}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], max_grad_norm, error_if_nonfinite=True,
            )
            before_update = torch.cat([parameter.detach().reshape(-1) for _, parameter in trainable]).clone()
            optimizer.step()
            scheduler.step()
            after_update = torch.cat([parameter.detach().reshape(-1) for _, parameter in trainable])
            changed_numel = int(torch.count_nonzero(after_update != before_update).item())
            update_norm = float((after_update.float() - before_update.float()).norm().item())
            model.zero_grad(set_to_none=True)
            event_end.record()
            torch.cuda.synchronize()
            gpu_seconds = float(event_start.elapsed_time(event_end) / 1000.0)
            host_seconds = time.monotonic() - host_start
            step_loss = sum(micro_losses) / actual_group_size
            losses_all.append(step_loss)
            record = {
                "run_id": run_id,
                "dev_local_step": local_step,
                "global_step": global_step,
                "sample_ids": sample_ids,
                "sample_ids_sha256": canonical_sha256(sample_ids),
                "actual_group_size": actual_group_size,
                "micro_sample_losses": micro_losses,
                "mean_completion_loss": step_loss,
                "token_counts": token_counts,
                "gradient_finite": gradients_finite,
                "nonzero_gradient_elements": gradient_nonzero_count,
                "preclip_gradient_norm": float(gradient_norm.detach().float().cpu().item()),
                "lora_parameters_changed": changed_numel > 0,
                "changed_lora_elements": changed_numel,
                "lora_update_norm": update_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "scheduler_last_epoch": int(scheduler.last_epoch),
                "gpu_step_seconds": gpu_seconds,
                "host_step_seconds": host_seconds,
                "created_at_utc": _now(),
            }
            if changed_numel <= 0 or not math.isfinite(update_norm):
                raise RuntimeError(f"optimizer did not produce a finite LoRA update at Dev-local step {local_step}")
            _append_jsonl(ledger_path, record)

            if local_step in checkpoints:
                elapsed = time.monotonic() - start_time
                checkpoint = run_root / "trainer" / f"checkpoint-{local_step:04d}"
                checkpoint_record = _save_checkpoint(
                    checkpoint=checkpoint,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    torch=torch,
                    parent_state=previous_trainer_state,
                    run_id=run_id,
                    local_step=local_step,
                    global_step=global_step,
                    consumed_ids=consumed_ids,
                    schedule_sha256=schedule_sha,
                    loss_mean=sum(losses_all) / len(losses_all),
                    training_seconds=elapsed,
                )
                checkpoint_records.append({
                    "dev_local_step": local_step,
                    "global_step": global_step,
                    "checkpoint_path": str(checkpoint.relative_to(run_root)),
                    "checkpoint_manifest_sha256": _sha256(checkpoint / "checkpoint_manifest.json"),
                    "files_sha256": checkpoint_record["files_sha256"],
                    "consumed_rows": len(consumed_ids),
                    "consumed_sample_ids_sha256": canonical_sha256(consumed_ids),
                })
                manifest.update({
                    "stage": f"checkpoint_saved_{local_step}",
                    "devft_steps": local_step,
                    "optimizer_steps": global_step,
                    "consumed_rows": len(consumed_ids),
                    "consumed_sample_ids_sha256": canonical_sha256(consumed_ids),
                    "checkpoints": list(checkpoint_records),
                    "updated_at_utc": _now(),
                })
                _write_json(manifest_path, manifest)
                print(
                    f"[E01-B DevFT] step={local_step}/456 global={global_step} "
                    f"rows={len(consumed_ids)}/1821 loss={step_loss:.5f} "
                    f"gpu_step={gpu_seconds:.2f}s checkpoint=complete",
                    flush=True,
                )
                if elapsed > int(config["optimizer"]["max_training_seconds"]):
                    raise TimeoutError("E01-B DevFT exceeded its declared training time limit")

        if len(consumed_ids) != 1821 or len(set(consumed_ids)) != 1821 or set(consumed_ids) != set(allowed_ids):
            raise RuntimeError("DevFT did not consume all 1,821 unique allowed Dev IDs exactly once")
        if int(scheduler.last_epoch) != 4504 or len(checkpoint_records) != 4:
            raise RuntimeError("final scheduler or checkpoint state differs from the 4,504-step contract")
        ledger_rows = _read_jsonl(ledger_path)
        if len(ledger_rows) != 456 or [row["global_step"] for row in ledger_rows] != list(range(4049, 4505)):
            raise RuntimeError("optimizer step ledger is incomplete or non-contiguous")
        if [sample_id for row in ledger_rows for sample_id in row["sample_ids"]] != consumed_ids:
            raise RuntimeError("optimizer step ledger does not match the consumed-ID sequence")

        model.eval()
        final_step = run_root / "trainer/checkpoint-0456"
        probe_row = dev_rows[0]
        token_ids_before_reload = _fixed_prompt_probe(model, tokenizer, probe_row, torch)
        tensor_sha_before_reload = _tensor_fingerprint(trainable, torch)
        final_adapter = run_root / "adapter"
        final_adapter.mkdir(exist_ok=False)
        for path in final_step.iterdir():
            if path.is_file() and (path.name == "adapter_config.json" or path.name.startswith("adapter_model")):
                shutil.copy2(path, final_adapter / path.name)
        if _sha256(final_adapter / "adapter_model.safetensors") != _sha256(final_step / "adapter_model.safetensors"):
            raise RuntimeError("final inference Adapter copy differs from checkpoint-0456")
        reload_model, _reload_torch, reload_facts = training_sft._load_train_model(
            original_config,
            _resolve(original_config["base_model"]["local_path"], allow_external_symlink=True),
            adapter_path=final_adapter,
            is_trainable_adapter=False,
            autocast_adapter_dtype=False,
        )
        reload_model.eval()
        token_ids_after_reload = _fixed_prompt_probe(reload_model, tokenizer, probe_row, torch)
        if token_ids_before_reload != token_ids_after_reload:
            raise RuntimeError("final Adapter independent reload changed the fixed Prompt Token IDs")
        if _tensor_fingerprint([(name, parameter) for name, parameter in reload_model.named_parameters() if ".lora_A." in name or ".lora_B." in name], torch) != tensor_sha_before_reload:
            raise RuntimeError("final Adapter independent reload changed the saved LoRA tensors")
        del reload_model
        gc.collect()
        torch.cuda.empty_cache()
        base_after = _base_identity(original_config)
        if base_after.get("observed_weight_hash") != base_before.get("observed_weight_hash"):
            raise RuntimeError("frozen Base model files changed during DevFT")
        elapsed = time.monotonic() - start_time
        step_times = [float(row["gpu_step_seconds"]) for row in ledger_rows]
        final_summary = {
            "status": "passed",
            "run_id": run_id,
            "parent_run_id": PARENT_RUN_ID,
            "parent_global_step": 4048,
            "global_step": 4504,
            "dev_local_steps": 456,
            "rows_consumed": len(consumed_ids),
            "rows_unique": len(set(consumed_ids)),
            "consumed_sample_ids_sha256": canonical_sha256(consumed_ids),
            "schedule_sha256": schedule_sha,
            "mean_completion_loss": sum(losses_all) / len(losses_all),
            "last_step_loss": losses_all[-1],
            "training_seconds": elapsed,
            "gpu_step_seconds": {
                "mean": sum(step_times) / len(step_times),
                "p50": sorted(step_times)[(len(step_times) - 1) // 2],
                "p90": sorted(step_times)[math.ceil(0.9 * len(step_times)) - 1],
            },
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "scheduler_last_epoch": int(scheduler.last_epoch),
            "optimizer_reconstruction": optimizer_facts,
            "initial_adapter_tensor_sha256": parent_tensor_sha,
            "final_adapter_tensor_sha256": tensor_sha_before_reload,
            "final_adapter_weight_sha256": inference.adapter_weight_identity(final_adapter)[0]["weight_sha256"],
            "fixed_prompt_probe": {
                "sample_id": probe_row["sample_id"],
                "prompt_ids_sha256": canonical_sha256(probe_row["input_ids"][:probe_row["prompt_length"]]),
                "generated_token_ids_sha256": canonical_sha256(token_ids_before_reload),
                "token_ids_exact_match_after_independent_reload": token_ids_before_reload == token_ids_after_reload,
            },
            "independent_reload": {"status": "passed", "model_facts": reload_facts},
            "base_weights_unchanged": True,
            "base_weight_sha256": base_before["observed_weight_hash"],
            "base_file_count": len(base_before.get("files", [])),
            "trainable_parameter_names_sha256": trainable_names_hash,
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "trainable_tensor_count": len(trainable),
            "trainable_dtype": str(trainable[0][1].dtype),
            "all_trainable_parameters_are_lora": True,
            "checkpoint_manifests": checkpoint_records,
            "tokenizer_facts": tokenizer_facts,
        }
        _write_json(run_root / "training_summary.json", final_summary, exclusive=True)
        manifest.update({
            "status": "completed",
            "stage": "training_complete_awaiting_val_generation",
            "optimizer_steps": 4504,
            "devft_steps": 456,
            "rows_consumed": 1821,
            "rows_unique": 1821,
            "consumed_sample_ids_sha256": canonical_sha256(consumed_ids),
            "step_ledger_sha256": _sha256(ledger_path),
            "training_summary_sha256": _sha256(run_root / "training_summary.json"),
            "final_adapter_weight_sha256": final_summary["final_adapter_weight_sha256"],
            "final_adapter_tensor_sha256": tensor_sha_before_reload,
            "base_weights_unchanged": True,
            "official_val_read": False,
            "checkpoints": checkpoint_records,
            "ended_at_utc": _now(),
            "updated_at_utc": _now(),
        })
        _write_json(manifest_path, manifest)
        return final_summary
    except BaseException as exc:
        manifest = _read_json(manifest_path)
        manifest.update({
            "status": "failed",
            "stage": "devft_training_failed",
            "stop_reason": type(exc).__name__,
            "error_summary": str(exc)[:1200],
            "optimizer_steps": max(4048, int(manifest.get("optimizer_steps", 4048))),
            "consumed_rows": len(locals().get("consumed_ids", [])),
            "consumed_sample_ids_sha256": canonical_sha256(locals().get("consumed_ids", [])),
            "step_ledger_sha256": _sha256(ledger_path) if ledger_path.exists() else None,
            "updated_at_utc": _now(),
        })
        _write_json(manifest_path, manifest)
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--phase", choices=("run", "prepare", "train"), default="run")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.phase == "train":
        config = _load_config(config_path)
        run_root = _resolve(config["output_root"]) / (args.run_id or config["run_id"])
        result = train_run(config, run_root)
    else:
        config, run_root = prepare_run(config_path, args.run_id)
        if args.phase == "prepare":
            result = _read_json(run_root / "manifest.json")
        else:
            result = train_run(config, run_root)
    print(json.dumps({"run_root": str(run_root.relative_to(ROOT)), "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
