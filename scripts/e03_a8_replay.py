#!/usr/bin/env python3
"""Run the versioned E03-A8 hard-sample Replay experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from src.data.e03_a8_replay import (
    assert_safe_replay_checkpoint_path as _assert_safe_checkpoint_output,
    build_replay_schedule,
    canonical_json_sha256,
    profile_label_keys,
    score_candidate_difficulty,
    set_replay_learning_rate as _set_replay_learning_rate,
    select_candidates,
    validate_encoded_gold_completion,
)
from src.data.e03_a8_sft import canonical_sha256
from src.data.sft_collator import SFTDataCollator
from src.evaluation.e03_a6_f4_staged_metrics import _conversation_id
from src.evaluation.e03_a8_replay_metrics import (
    freeze_val_splits,
    gate_reasons,
    score_profile_predictions,
    serialize_val_splits,
)
from src.evaluation.e03_a8_staged_metrics import score_a8_cohort
from src.shared.e03_a8_protocol import (
    A8ParseError,
    PROFILE_DIMS,
    parse_specialist_output,
    serialize_specialist_completion,
    validate_a8_prompt_enum_order,
)
from src.shared.prompting import select_recent_complete_suffix
from src.training.e03_a8_staged import (
    JsonlById,
    _base_identity,
    _checkpoint_files,
    _lora_parameters,
    _lora_tensor_sha256,
    _next_token_id,
    _package_versions,
    _read_json,
    _read_jsonl,
    _restore_rng_state,
    _rng_state,
    _row_contract,
    _runtime_tokenizer,
    _set_seed,
    _write_json,
    sha256_file,
)
from src.training.e03_a8_completion_loss import completion_only_forward_loss
from src.training.e03_a8_v2_staged import _gradient_summary
from src.training.training_sft import _load_train_model


DEFAULT_CONFIG = ROOT / "configs/experiments/e03_a8_replay.yaml"
CODE_PATHS = (
    "configs/experiments/e03_a8_replay.yaml",
    "configs/training/e03_a8_v2_p_staged.yaml",
    "configs/training/e03_a8_devft_p.yaml",
    "scripts/e03_a8_replay.py",
    "src/data/e03_a8_replay.py",
    "src/data/e03_a8_sft.py",
    "src/data/sft_collator.py",
    "src/evaluation/e03_a8_replay_metrics.py",
    "src/evaluation/e03_a8_staged_metrics.py",
    "src/shared/e03_a8_protocol.py",
    "src/shared/prompting.py",
    "src/training/e03_a8_completion_loss.py",
    "src/training/e03_a8_staged.py",
    "src/training/e03_a8_v2_staged.py",
    "src/training/training_sft.py",
    "participant/run_inference.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_config(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != "E03-A8-hard-replay-v1":
        raise ValueError(f"unsupported E03-A8 Replay config: {resolved}")
    return value, resolved


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Replay project path escapes the repository: {value}")
    return resolved


def _archive_path(value: str | Path) -> Path:
    """Resolve an explicitly configured archive path without the A8 _resolve guard."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (rendered + "\n").encode("utf-8")


def _write_exclusive(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = _json_bytes(value, pretty=pretty)
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    payload = _json_bytes(value, pretty=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        for row in rows:
            payload = _json_bytes(row)
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    _write_jsonl_exclusive(temporary, rows)
    os.replace(temporary, path)


def _append_jsonl_fsync(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("ab") as handle:
        for row in rows:
            handle.write(_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def _file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"archive/checkpoint contains a symlink: {path}")
        if path.is_file():
            records.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return records


def _verify_records(root: Path, expected: list[dict[str, Any]]) -> None:
    actual = _file_records(root)
    if actual != expected:
        raise ValueError(f"archive file identity mismatch at {root}")


def _chmod_readonly(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def _copy_tree_exclusive(source: Path, destination: Path, *, run_tag: str) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial-{run_tag}")
    if temporary.exists():
        raise FileExistsError(f"stale partial archive exists; preserving it: {temporary}")
    shutil.copytree(source, temporary, symlinks=False, copy_function=shutil.copy2)
    if any(path.is_symlink() for path in temporary.rglob("*")):
        raise ValueError(f"copied archive contains a symlink: {temporary}")
    temporary.rename(destination)


def _checkpoint_file_facts(checkpoint: Path) -> list[dict[str, Any]]:
    return [record for record in _file_records(checkpoint) if record["path"] != "checkpoint_manifest.json"]


def _git_facts() -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return {"head": head, "dirty": bool(status.strip()), "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest()}


def _base_config(config: dict[str, Any]) -> dict[str, Any]:
    path = _project_path(config["model_config"])
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("arm") != "P" or value.get("base_model", {}).get("model_id") != "Qwen/Qwen3-1.7B":
        raise ValueError("Replay model config is not the frozen A8 P Qwen3-1.7B configuration")
    if value.get("protocol", {}).get("prompt_path") != "configs/prompts/p_a8_v2_profile.txt":
        raise ValueError("Replay P prompt differs from A8-V2")
    if value.get("base_model", {}).get("expected_weight_sha256") != config["parent"]["base_weight_sha256"]:
        raise ValueError("Replay Base SHA config differs from the P4504 task identity")
    return value


def _validate_parent(config: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent"]
    checkpoint = _project_path(parent["checkpoint_path"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"P4504 parent checkpoint is missing: {checkpoint}")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if sha256_file(manifest_path) != parent["checkpoint_manifest_sha256"]:
        raise ValueError("P4504 parent checkpoint manifest SHA differs from the frozen plan")
    record = _read_json(manifest_path)
    expected_fields = {
        "status": "complete",
        "run_id": parent["run_id"],
        "arm": "P",
        "devft_step": parent["devft_step"],
        "global_optimizer_step": parent["global_optimizer_step"],
        "sample_cursor": parent["sample_cursor"],
        "base_model_weight_sha256": parent["base_weight_sha256"],
    }
    for key, expected in expected_fields.items():
        if record.get(key) != expected:
            raise ValueError(f"P4504 checkpoint {key} differs: {record.get(key)!r} != {expected!r}")
    if record.get("adapter_identity", {}).get("weight_sha256") != parent["adapter_weight_sha256"]:
        raise ValueError("P4504 Adapter weight identity differs")
    if record.get("files_sha256") != canonical_sha256(record.get("files")):
        raise ValueError("P4504 checkpoint embedded file manifest hash is invalid")
    if _checkpoint_file_facts(checkpoint) != record.get("files"):
        raise ValueError("P4504 checkpoint files or hashes differ from its embedded manifest")
    state = _read_json(checkpoint / "trainer_state.json")
    if (
        state.get("run_id") != parent["run_id"]
        or state.get("arm") != "P"
        or state.get("devft_step") != parent["devft_step"]
        or state.get("global_optimizer_step") != parent["global_optimizer_step"]
        or state.get("sample_cursor") != parent["sample_cursor"]
        or state.get("scheduler_last_epoch") != parent["global_optimizer_step"]
    ):
        raise ValueError("P4504 Trainer state or sample cursor differs")
    run_root = checkpoint.parent.parent
    run_manifest = _read_json(run_root / "manifest.json")
    node = run_manifest.get("nodes", {}).get(str(parent["devft_step"]), {})
    if (
        run_manifest.get("run_id") != parent["run_id"]
        or run_manifest.get("arm") != "P"
        or run_manifest.get("optimizer_steps") != parent["global_optimizer_step"]
        or run_manifest.get("last_checkpoint_manifest_sha256") != parent["checkpoint_manifest_sha256"]
        or node.get("training_status") != "passed"
        or node.get("checkpoint_manifest_sha256") != parent["checkpoint_manifest_sha256"]
    ):
        raise ValueError("P4504 parent Run manifest does not confirm the final checkpoint")

    import torch

    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer_steps = []
    for state_item in optimizer.get("state", {}).values():
        step_value = state_item.get("step")
        if step_value is not None:
            optimizer_steps.append(int(step_value.item()) if hasattr(step_value, "item") else int(step_value))
    if not optimizer_steps or set(optimizer_steps) != {parent["global_optimizer_step"]}:
        raise ValueError("P4504 AdamW moment step count differs from 4,504")
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False)
    if int(scheduler.get("last_epoch", -1)) != parent["global_optimizer_step"]:
        raise ValueError("P4504 scheduler state is not at step 4,504")
    if not (checkpoint / "rng_state.pth").is_file():
        raise ValueError("P4504 RNG recovery state is missing")

    base_source = (ROOT / model_config["base_model"]["local_path"]).resolve()
    base_before = inference._weight_identity(base_source, {
        "model_id": model_config["base_model"]["model_id"],
        "model_revision": model_config["base_model"]["revision"],
    })
    if base_before.get("observed_weight_hash") != parent["base_weight_sha256"]:
        raise ValueError("P4504 Base model SHA differs")
    adapter_identity, adapter_cfg = inference.adapter_weight_identity(checkpoint)
    if adapter_identity.get("weight_sha256") != parent["adapter_weight_sha256"]:
        raise ValueError("P4504 Adapter files do not match its declared weight identity")
    if adapter_cfg.get("r") != 8 or adapter_cfg.get("lora_alpha") != 16:
        raise ValueError("P4504 LoRA configuration differs from r=8/alpha=16")
    tokenizer, tokenizer_facts = _runtime_tokenizer(model_config)
    if int(tokenizer.eos_token_id) != int(model_config["protocol"]["eos_token_id"]):
        raise ValueError("A8 tokenizer EOS differs from the frozen config")
    p_prompt = _project_path(model_config["protocol"]["prompt_path"])
    p0_prompt = _project_path(model_config["protocol"]["selection_prompt_path"])
    if sha256_file(p_prompt) != model_config["protocol"]["prompt_sha256"] or sha256_file(p0_prompt) != model_config["protocol"]["p0_sha256"]:
        raise ValueError("A8 P/P0 Prompt hashes differ")
    del tokenizer
    return {
        "checkpoint_path": checkpoint,
        "checkpoint_manifest": record,
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_files": _checkpoint_file_facts(checkpoint),
        "run_manifest": run_manifest,
        "base_source_path": base_source,
        "base_identity": base_before,
        "adapter_identity": adapter_identity,
        "adapter_config": adapter_cfg,
        "tokenizer_facts": tokenizer_facts,
        "prompt_sha256": model_config["protocol"]["prompt_sha256"],
        "p0_sha256": model_config["protocol"]["p0_sha256"],
        "scheduler_state": scheduler,
        "optimizer_step_values": optimizer_steps,
    }


def _base_archive_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "archive_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Base archive has no manifest: {root}")
    manifest = _read_json(manifest_path)
    model_path = root / "model"
    if manifest.get("status") != "complete" or manifest.get("base_weight_sha256") != "005d20550c04f4e246e5b434448786b44a0196b47d758d0acc77aba279e5bbc5":
        raise ValueError(f"Base archive identity differs: {root}")
    _verify_records(model_path, manifest.get("files", []))
    return manifest


def _ensure_base_archive(
    config: dict[str, Any],
    model_config: dict[str, Any],
    parent_identity: dict[str, Any],
    *,
    run_tag: str,
) -> tuple[Path, dict[str, Any]]:
    archive_root = _archive_path(config["outputs"]["archive_root"])
    name = f"Qwen3-1.7B-{config['parent']['base_weight_sha256'][:12]}"
    destination = archive_root / "base" / name
    model_path = destination / "model"
    if destination.exists():
        manifest = _base_archive_manifest(destination)
        identity = inference._weight_identity(model_path, {
            "model_id": model_config["base_model"]["model_id"],
            "model_revision": model_config["base_model"]["revision"],
        })
        if identity.get("observed_weight_hash") != config["parent"]["base_weight_sha256"]:
            raise ValueError("existing Base archive is incomplete or has a different weight identity")
        for record in manifest["files"]:
            source_stat = (parent_identity["base_source_path"] / record["path"]).stat()
            archive_stat = (model_path / record["path"]).stat()
            if (source_stat.st_dev, source_stat.st_ino) == (archive_stat.st_dev, archive_stat.st_ino):
                raise ValueError("existing Base archive shares an inode with its source cache")
        return model_path, manifest
    source = parent_identity["base_source_path"]
    source_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    free = shutil.disk_usage(archive_root if archive_root.exists() else archive_root.parent if archive_root.parent.exists() else Path.home()).free
    if free < source_bytes + 5 * 1024**3:
        raise OSError(f"insufficient archive disk space: {free} bytes free, {source_bytes} required plus 5 GiB margin")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial-{run_tag}")
    if temporary.exists():
        raise FileExistsError(f"stale Base archive partial exists; preserving it: {temporary}")
    temporary.mkdir(parents=False)
    shutil.copytree(source, temporary / "model", symlinks=False, copy_function=shutil.copy2)
    files = _file_records(temporary / "model")
    identity = inference._weight_identity(temporary / "model", {
        "model_id": model_config["base_model"]["model_id"],
        "model_revision": model_config["base_model"]["revision"],
    })
    if identity.get("observed_weight_hash") != config["parent"]["base_weight_sha256"]:
        raise ValueError("copied Base archive does not match the frozen weight SHA")
    for record in files:
        source_stat = (source / record["path"]).stat()
        archive_stat = (temporary / "model" / record["path"]).stat()
        if (source_stat.st_dev, source_stat.st_ino) == (archive_stat.st_dev, archive_stat.st_ino):
            raise ValueError("Base archive copy shares an inode with its source cache")
    manifest = {
        "status": "complete",
        "version": "E03-A8-REPLAY-base-archive-v1",
        "model_id": model_config["base_model"]["model_id"],
        "revision": model_config["base_model"]["revision"],
        "base_weight_sha256": identity["observed_weight_hash"],
        "base_identity": identity,
        "files": files,
        "files_sha256": canonical_sha256(files),
        "created_at_utc": _now(),
    }
    _write_exclusive(temporary / "archive_manifest.json", manifest)
    (temporary / "archive_manifest.json").chmod(0o444)
    for path in (temporary / "model").rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in (temporary / "model").rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    (temporary / "model").chmod(0o555)
    temporary.chmod(0o555)
    temporary.rename(destination)
    return destination / "model", manifest


def _archive_identity_paths(config: dict[str, Any], parent_identity: dict[str, Any]) -> tuple[Path, Path]:
    prefix = config["parent"]["checkpoint_manifest_sha256"][:12]
    archive = _archive_path(config["outputs"]["archive_root"])
    external = archive / "E03-A8" / f"P4504-{prefix}"
    local = _project_path(config["outputs"]["training_root"]) / "baselines" / f"P4504-{prefix}"
    return external, local


def _copy_parent_snapshot(
    source: Path,
    destination: Path,
    *,
    expected_files: list[dict[str, Any]],
    parent_manifest_sha: str,
    run_tag: str,
    identity: dict[str, Any],
) -> Path:
    checkpoint_dest = destination / "checkpoint"
    if destination.exists():
        if not checkpoint_dest.is_dir() or not (destination / "identity.json").is_file():
            raise ValueError(f"partial parent archive exists and will not be overwritten: {destination}")
        archive_identity = _read_json(destination / "identity.json")
        if archive_identity.get("parent_checkpoint_manifest_sha256") != parent_manifest_sha:
            raise ValueError(f"existing parent archive is bound to another checkpoint: {destination}")
        _verify_records(checkpoint_dest, expected_files)
        return checkpoint_dest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial-{run_tag}")
    if temporary.exists():
        raise FileExistsError(f"stale P4504 archive partial exists; preserving it: {temporary}")
    temporary.mkdir(parents=False)
    shutil.copytree(source, temporary / "checkpoint", symlinks=False, copy_function=shutil.copy2)
    _verify_records(temporary / "checkpoint", expected_files)
    _write_exclusive(temporary / "identity.json", identity)
    restore_md = (
        "# Restore P4504\n\n"
        "This immutable archive contains the verified parent Adapter and full optimizer state.\n\n"
        "1. Use the paired Base snapshot recorded in `identity.json` and the Adapter in `checkpoint/`.\n"
        "2. From the repository root, run `.venv/bin/python scripts/e03_a8_replay.py restore-check --manifest "
        f"'{destination}/checkpoint/checkpoint_manifest.json'`.\n"
        "3. The restore check loads the local archived Base and Adapter, verifies the file manifest, LoRA tensors, and five fixed next-token probes.\n"
        "4. To resume training, load `checkpoint/optimizer.pt`, `scheduler.pt`, `rng_state.pth`, and `trainer_state.json`; the global optimizer step is 4,504 and the DevFT sample cursor is 1,821.\n"
    )
    (temporary / "RESTORE.md").write_text(restore_md, encoding="utf-8")
    for path in (temporary / "checkpoint").rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in (temporary / "checkpoint").rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    (temporary / "checkpoint").chmod(0o555)
    (temporary / "identity.json").chmod(0o444)
    (temporary / "RESTORE.md").chmod(0o444)
    temporary.chmod(0o555)
    temporary.rename(destination)
    return destination / "checkpoint"


def _assert_distinct_copies(source: Path, copies: list[Path], expected: list[dict[str, Any]]) -> None:
    for copy in copies:
        _verify_records(copy, expected)
    for record in expected:
        source_stat = (source / record["path"]).stat()
        for copy in copies:
            copied = (copy / record["path"]).stat()
            if source_stat.st_dev == copied.st_dev and source_stat.st_ino == copied.st_ino:
                raise ValueError("archive copy shares an inode with its source")
        for left_index, left in enumerate(copies):
            for right in copies[left_index + 1:]:
                lstat, rstat = (left / record["path"]).stat(), (right / record["path"]).stat()
                if lstat.st_dev == rstat.st_dev and lstat.st_ino == rstat.st_ino:
                    raise ValueError("parent backups are hard-linked; independent copies are required")


def _fixed_probe_prompts(config: dict[str, Any], count: int = 5) -> list[tuple[str, list[int]]]:
    encoded = _project_path(config["data"]["train_encoded"])
    probes = []
    with encoded.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt_ids = list(row["input_ids"][:row["prompt_length"]])
            if prompt_ids:
                probes.append((str(row["sample_id"]), prompt_ids))
            if len(probes) == count:
                break
    if len(probes) != count:
        raise ValueError("could not construct five fixed prompt-only recovery probes")
    return probes


def _probe_logits(model: Any, torch: Any, prompt_ids: list[int]) -> Any:
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.inference_mode():
        values = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].detach().float().cpu()
    return values


def _verify_restored_models(
    config: dict[str, Any],
    model_config: dict[str, Any],
    parent_identity: dict[str, Any],
    archive_base: Path,
    archived_adapter: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_gpu_unavailable_or_bf16_unsupported: P4504 restore gate requires CUDA BF16")
    probes = _fixed_probe_prompts(config)
    before_base = inference._weight_identity(parent_identity["base_source_path"], {
        "model_id": model_config["base_model"]["model_id"],
        "model_revision": model_config["base_model"]["revision"],
    })
    original_model = archived_model = None
    try:
        original_model, torch, original_facts = _load_train_model(
            model_config,
            parent_identity["base_source_path"],
            adapter_path=parent_identity["checkpoint_path"],
            autocast_adapter_dtype=True,
        )
        expected_tensor_sha = parent_identity["run_manifest"].get("final_adapter_tensor_sha256")
        actual_tensor_sha = _lora_tensor_sha256(original_model, torch)
        if actual_tensor_sha != expected_tensor_sha:
            raise ValueError("original P4504 adapter tensors differ from the verified DevFT Run")
        source_logits = [(sample_id, prompt_ids, _probe_logits(original_model, torch, prompt_ids)) for sample_id, prompt_ids in probes]
        del original_model
        original_model = None
        gc.collect()
        torch.cuda.empty_cache()
        archived_model, torch, archived_facts = _load_train_model(
            model_config,
            archive_base,
            adapter_path=archived_adapter,
            autocast_adapter_dtype=True,
        )
        archived_tensor_sha = _lora_tensor_sha256(archived_model, torch)
        if archived_tensor_sha != expected_tensor_sha:
            raise ValueError("archived P4504 adapter tensors differ from the original")
        probes_result = []
        for sample_id, prompt_ids, expected_logits in source_logits:
            actual_logits = _probe_logits(archived_model, torch, prompt_ids)
            max_abs = float((expected_logits - actual_logits).abs().max().item())
            source_argmax = int(expected_logits.argmax().item())
            archived_argmax = int(actual_logits.argmax().item())
            if source_argmax != archived_argmax or max_abs > 1e-3:
                raise ValueError(f"archive next-token probe differs for {sample_id}: max_abs={max_abs}")
            probes_result.append({
                "sample_id": sample_id,
                "prompt_tokens": len(prompt_ids),
                "next_token_argmax": source_argmax,
                "next_token_argmax_equal": True,
                "logit_max_abs_diff": max_abs,
                "tolerance": 1e-3,
            })
        after_base = inference._weight_identity(parent_identity["base_source_path"], {
            "model_id": model_config["base_model"]["model_id"],
            "model_revision": model_config["base_model"]["revision"],
        })
        archive_base_identity = inference._weight_identity(archive_base, {
            "model_id": model_config["base_model"]["model_id"],
            "model_revision": model_config["base_model"]["revision"],
        })
        if before_base.get("observed_weight_hash") != after_base.get("observed_weight_hash") or archive_base_identity.get("observed_weight_hash") != config["parent"]["base_weight_sha256"]:
            raise ValueError("P4504 restore gate found a changed source or archive Base")
        return {
            "status": "passed",
            "baseline_backup_gate": "passed",
            "run_id": run_id,
            "parent_global_step": config["parent"]["global_optimizer_step"],
            "original_adapter_tensor_sha256": actual_tensor_sha,
            "archived_adapter_tensor_sha256": archived_tensor_sha,
            "source_base_weight_sha256": before_base.get("observed_weight_hash"),
            "archived_base_weight_sha256": archive_base_identity.get("observed_weight_hash"),
            "original_model_facts": original_facts,
            "archived_model_facts": archived_facts,
            "probes": probes_result,
            "created_at_utc": _now(),
        }
    finally:
        if original_model is not None:
            del original_model
        if archived_model is not None:
            del archived_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ensure_baseline_backup(
    config: dict[str, Any],
    model_config: dict[str, Any],
    parent_identity: dict[str, Any],
    run_root: Path,
    *,
    run_id: str,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    parent = config["parent"]
    parent_manifest = parent_identity["checkpoint_manifest"]
    parent_size = sum(item["size_bytes"] for item in parent_identity["checkpoint_files"]) + (parent_identity["checkpoint_path"] / "checkpoint_manifest.json").stat().st_size
    external_root, local_root = _archive_identity_paths(config, parent_identity)
    for destination in (external_root, local_root):
        if not destination.exists():
            probe = destination.parent
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            free = shutil.disk_usage(probe).free
            if free < parent_size + 512 * 1024**2:
                raise OSError(f"insufficient space for independent P4504 snapshot at {destination}: {free} free, {parent_size} required plus 512 MiB margin")
    archive_base, base_archive_manifest = _ensure_base_archive(config, model_config, parent_identity, run_tag=run_id)
    identity_payload = {
        "version": "E03-A8-REPLAY-P4504-identity-v1",
        "parent_run_id": parent["run_id"],
        "parent_checkpoint_path": str(parent_identity["checkpoint_path"]),
        "parent_checkpoint_manifest_sha256": parent_identity["checkpoint_manifest_sha256"],
        "parent_checkpoint_file_sha256": parent_identity["checkpoint_files"],
        "parent_global_optimizer_step": parent["global_optimizer_step"],
        "devft_step": parent["devft_step"],
        "sample_cursor": parent["sample_cursor"],
        "adapter_weight_sha256": parent["adapter_weight_sha256"],
        "adapter_tensor_sha256": parent_identity["run_manifest"].get("final_adapter_tensor_sha256"),
        "base_model_id": model_config["base_model"]["model_id"],
        "base_revision": model_config["base_model"]["revision"],
        "base_archive_path": str(archive_base),
        "base_weight_sha256": parent["base_weight_sha256"],
        "base_archive_files_sha256": base_archive_manifest["files_sha256"],
        "prompt_sha256": parent_identity["prompt_sha256"],
        "p0_sha256": parent_identity["p0_sha256"],
        "tokenizer_facts": parent_identity["tokenizer_facts"],
        "parent_run_manifest_sha256": sha256_file(parent_identity["checkpoint_path"].parent.parent / "manifest.json"),
        "parent_manifest_record_sha256": canonical_sha256(parent_manifest),
        "code_git": _git_facts(),
        "runtime_packages": _package_versions(),
        "created_at_utc": _now(),
    }
    expected = parent_identity["checkpoint_files"] + [{
        "path": "checkpoint_manifest.json",
        "size_bytes": (parent_identity["checkpoint_path"] / "checkpoint_manifest.json").stat().st_size,
        "sha256": parent_identity["checkpoint_manifest_sha256"],
    }]
    # The checkpoint manifest is sorted among all other files in its source.
    expected.sort(key=lambda row: row["path"])
    archive_cp = _copy_parent_snapshot(
        parent_identity["checkpoint_path"],
        external_root,
        expected_files=expected,
        parent_manifest_sha=parent_identity["checkpoint_manifest_sha256"],
        run_tag=run_id,
        identity=identity_payload,
    )
    local_identity = {**identity_payload, "archive_root": str(local_root), "external_archive_root": str(external_root)}
    local_cp = _copy_parent_snapshot(
        parent_identity["checkpoint_path"],
        local_root,
        expected_files=expected,
        parent_manifest_sha=parent_identity["checkpoint_manifest_sha256"],
        run_tag=run_id,
        identity=local_identity,
    )
    _assert_distinct_copies(parent_identity["checkpoint_path"], [archive_cp, local_cp], expected)
    baseline_gate_path = run_root / "baseline_backup_gate.json"
    if baseline_gate_path.is_file():
        gate = _read_json(baseline_gate_path)
        if gate.get("status") != "passed" or gate.get("parent_checkpoint_manifest_sha256") != parent_identity["checkpoint_manifest_sha256"]:
            raise ValueError("existing P4504 baseline backup gate does not match the frozen parent")
    else:
        gate = _verify_restored_models(config, model_config, parent_identity, archive_base, archive_cp, run_id=run_id)
        gate["parent_checkpoint_manifest_sha256"] = parent_identity["checkpoint_manifest_sha256"]
        gate["external_archive_path"] = str(external_root)
        gate["local_snapshot_path"] = str(local_root)
        gate["base_archive_path"] = str(archive_base)
        _write_exclusive(baseline_gate_path, gate)
    return archive_cp, local_cp, archive_base, gate


def _source_compact_records(config: dict[str, Any], model_config: dict[str, Any], tokenizer: Any, enums: dict[str, list[str]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    paths = {
        "train_encoded": _project_path(config["data"]["train_encoded"]),
        "dev_encoded": _project_path(config["data"]["dev_encoded"]),
        "train_normalized": _project_path(config["data"]["train_normalized"]),
        "dev_normalized": _project_path(config["data"]["dev_normalized"]),
    }
    for key, sha_key in (("train_encoded", "train_encoded_sha256"), ("dev_encoded", "dev_encoded_sha256"), ("train_normalized", "train_normalized_sha256"), ("dev_normalized", "dev_normalized_sha256")):
        if sha256_file(paths[key]) != config["data"][sha_key]:
            raise ValueError(f"frozen A8 Replay data source hash changed: {key}")
    for source in ("train", "dev"):
        manifest_key = f"{source}_manifest"
        manifest_path = _project_path(config["data"][manifest_key])
        if sha256_file(manifest_path) != config["data"][f"{manifest_key}_sha256"]:
            raise ValueError(f"Replay source manifest SHA differs: {manifest_key}")
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "cpu_preparation_passed":
            raise ValueError(f"Replay source manifest is not passed: {manifest_key}")
    normalized: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "dev": {}}
    for source in ("train", "dev"):
        path = paths[f"{source}_normalized"]
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or sample_id in normalized[source]:
                    raise ValueError(f"{path}:{number}: duplicate or missing normalized sample ID")
                profile = row.get("target", {}).get("user_profile")
                profile_label_keys(profile)
                normalized[source][sample_id] = {
                    "sample_id": sample_id,
                    "conversation_id": row.get("conversation_id"),
                    "source": source,
                    "target": row["target"],
                    "history_message_count": len(row.get("history", [])),
                }
    if set(normalized["train"]) & set(normalized["dev"]):
        raise ValueError("A8 Train and Dev normalized IDs overlap")
    train_conversations = {str(row["conversation_id"]) for row in normalized["train"].values()}
    dev_conversations = {str(row["conversation_id"]) for row in normalized["dev"].values()}
    if train_conversations & dev_conversations:
        raise ValueError("A8 Train and Dev conversation IDs overlap")
    result: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    total_rows = 0
    for source in ("train", "dev"):
        path = paths[f"{source}_encoded"]
        seen: set[str] = set()
        expected_split = "train" if source == "train" else "dev_ft"
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                total_rows += 1
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or sample_id in seen:
                    raise ValueError(f"{path}:{number}: duplicate or missing encoded sample ID")
                seen.add(sample_id)
                gold_row = normalized[source].get(sample_id)
                if gold_row is None or row.get("conversation_id") != gold_row["conversation_id"]:
                    raise ValueError(f"{sample_id}: encoded row is not paired with its Train/Dev Gold source")
                if row.get("source_split") != expected_split:
                    raise ValueError(f"{sample_id}: encoded source split {row.get('source_split')!r} differs")
                legacy = dict(row)
                legacy["source_split"] = "train"
                _row_contract(
                    legacy,
                    eos_token_id=int(model_config["protocol"]["eos_token_id"]),
                    prompt_cap=int(model_config["protocol"]["max_prompt_tokens"]),
                    sequence_cap=int(model_config["protocol"]["max_train_seq_length"]),
                )
                expected_completion = serialize_specialist_completion(
                    gold_row["target"],
                    _schema_cache["schema"],
                    "P",
                )
                parsed = validate_encoded_gold_completion(
                    tokenizer,
                    row["input_ids"],
                    int(row["prompt_length"]),
                    expected_completion,
                    _schema_cache["schema"],
                    sample_id=sample_id,
                )
                expected_profile = {
                    dim: sorted(gold_row["target"]["user_profile"][dim], key=enums[dim].index)
                    for dim in PROFILE_DIMS
                }
                if parsed.get("user_profile") != expected_profile:
                    raise ValueError(f"{sample_id}: encoded Completion does not round-trip to Train/Dev Gold")
                result[source].append({
                    "sample_id": sample_id,
                    "source": source,
                    "conversation_id": gold_row["conversation_id"],
                    "target": gold_row["target"],
                    "prompt_length": int(row["prompt_length"]),
                    "history_message_count": int(gold_row["history_message_count"]),
                })
    expected_counts = {"train": int(config["data"]["train_count"]), "dev": int(config["data"]["dev_count"])}
    if {key: len(value) for key, value in result.items()} != expected_counts or total_rows != int(config["data"]["expected_total_count"]):
        raise ValueError("A8 Replay encoded Train/Dev row counts differ from the frozen inputs")
    facts = {
        "train_encoded_sha256": sha256_file(paths["train_encoded"]),
        "dev_encoded_sha256": sha256_file(paths["dev_encoded"]),
        "train_normalized_sha256": sha256_file(paths["train_normalized"]),
        "dev_normalized_sha256": sha256_file(paths["dev_normalized"]),
        "train_manifest_sha256": sha256_file(_project_path(config["data"]["train_manifest"])),
        "dev_manifest_sha256": sha256_file(_project_path(config["data"]["dev_manifest"])),
        "train_conversation_count": len(train_conversations),
        "dev_conversation_count": len(dev_conversations),
        "train_count": len(result["train"]),
        "dev_count": len(result["dev"]),
        "total_count": total_rows,
        "completion_gold_round_trip": "passed",
        "candidate_label_frequency_sha256": canonical_sha256({
            f"{dim}:{label}": count
            for dim, label, count in sorted(
                (dim, label, sum(label in row["target"]["user_profile"][dim] for rows in result.values() for row in rows))
                for dim in PROFILE_DIMS for label in enums[dim]
            )
        }),
    }
    return result, facts


_schema_cache: dict[str, Any] = {}


def _read_schema(config: dict[str, Any], model_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    path = _project_path(config["data"]["official_schema"])
    if sha256_file(path) != config["data"]["official_schema_sha256"]:
        raise ValueError("official Train Schema differs from the frozen Replay source")
    schema = _read_json(path)
    p_prompt_path = _project_path(model_config["protocol"]["prompt_path"])
    enums = validate_a8_prompt_enum_order(p_prompt_path.read_text(encoding="utf-8"), schema, "P")
    _schema_cache["schema"] = schema
    _schema_cache["enums"] = enums
    return schema, enums


def _candidate_artifacts(
    config: dict[str, Any],
    model_config: dict[str, Any],
    run_root: Path,
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mine_root = run_root / "mining"
    ids_path = mine_root / "candidate_ids.jsonl"
    audit_path = mine_root / "candidate_audit.json"
    tokenizer, _tokenizer_facts = _runtime_tokenizer(model_config)
    try:
        selection_inputs, source_facts = _source_compact_records(config, model_config, tokenizer, _schema_cache["enums"])
    finally:
        del tokenizer
        gc.collect()
    selected, selection_audit = select_candidates(
        selection_inputs,
        total_count=int(config["candidate_selection"]["count"]),
        seed=int(config["seed"]),
        source_counts={key: int(value) for key, value in config["candidate_selection"]["source_counts"].items()},
        per_conversation_cap=int(config["candidate_selection"]["max_samples_per_conversation"]),
        rare_label_cap=int(config["candidate_selection"]["rare_label_sample_cap"]),
    )
    candidate_rows = [
        {
            "sample_id": row["sample_id"],
            "source": row["source"],
            "conversation_id": row["conversation_id"],
            "prompt_length": row["prompt_length"],
            "history_message_count": row["history_message_count"],
            "gold_nonempty": row["gold_nonempty"],
            "gold_profile": row["gold_profile"],
            "candidate_tie_sha256": row["candidate_tie_sha256"],
        }
        for row in selected
    ]
    public_ids = [
        {key: row[key] for key in ("sample_id", "source", "conversation_id", "candidate_tie_sha256")}
        for row in candidate_rows
    ]
    audit = {
        "status": "frozen_before_model_prediction",
        "version": "E03-A8-REPLAY-mining-v1",
        "run_id": run_root.name,
        "seed": config["seed"],
        "selection": selection_audit,
        "source_facts": source_facts,
        "source_data_sha256": {key: value for key, value in source_facts.items() if key.endswith("sha256")},
        "candidate_ids_sha256": canonical_sha256([row["sample_id"] for row in public_ids]),
        "candidate_id_file_sha256": canonical_sha256(public_ids),
        "algorithm_sha256": {path: sha256_file(ROOT / path) for path in ("src/data/e03_a8_replay.py", "scripts/e03_a8_replay.py")},
        "created_at_utc": _now(),
    }
    mine_root.mkdir(parents=True, exist_ok=True)
    if ids_path.exists() or audit_path.exists():
        if not resume or not ids_path.is_file() or not audit_path.is_file():
            raise FileExistsError("candidate artifacts already exist and cannot be overwritten")
        existing_ids = _read_jsonl(ids_path)
        existing_audit = _read_json(audit_path)
        if (
            existing_ids != public_ids
            or existing_audit.get("candidate_ids_sha256") != audit["candidate_ids_sha256"]
            or existing_audit.get("candidate_ids_file_sha256") != sha256_file(ids_path)
            or existing_audit.get("source_facts") != audit["source_facts"]
            or existing_audit.get("algorithm_sha256") != audit["algorithm_sha256"]
        ):
            raise ValueError("existing frozen candidate IDs or audit differ from the current source/config")
        public_ids = existing_ids
        audit = existing_audit
    else:
        _write_jsonl_exclusive(ids_path, public_ids)
        audit["candidate_ids_file_sha256"] = sha256_file(ids_path)
        _write_exclusive(audit_path, audit)
    # The raw source Gold and candidate Gold stay out of the inference stage.
    del selection_inputs, candidate_rows
    return public_ids, audit


def _code_identity() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in CODE_PATHS}


def _run_identity(config: dict[str, Any], config_path: Path, model_config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "version": "E03-A8-REPLAY-run-v1",
        "experiment_id": "E03-A8-REPLAY",
        "arm": "P",
        "run_id": run_id,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": canonical_sha256(config),
        "config_file_sha256": sha256_file(config_path),
        "model_config_sha256": canonical_sha256(model_config),
        "parent_run_id": config["parent"]["run_id"],
        "parent_checkpoint_manifest_sha256": config["parent"]["checkpoint_manifest_sha256"],
        "parent_adapter_weight_sha256": config["parent"]["adapter_weight_sha256"],
        "base_weight_sha256": config["parent"]["base_weight_sha256"],
        "code_sha256": _code_identity(),
        "git": _git_facts(),
        "created_at_utc": _now(),
    }


def _initialize_run(
    config: dict[str, Any], config_path: Path, model_config: dict[str, Any], run_id: str
) -> tuple[Path, dict[str, Any]]:
    root = _project_path(config["outputs"]["training_root"])
    if not run_id or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for char in run_id):
        raise ValueError("run-id must use ASCII letters, digits, dot, underscore, or hyphen")
    run_root = root / run_id
    identity = _run_identity(config, config_path, model_config, run_id)
    manifest_path = run_root / "manifest.json"
    if run_root.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"existing Replay Run has no manifest; preserving it: {run_root}")
        manifest = _read_json(manifest_path)
        for key in (
            "run_id", "config_sha256", "config_file_sha256", "model_config_sha256",
            "parent_checkpoint_manifest_sha256", "code_sha256",
        ):
            if manifest.get(key) != identity.get(key):
                raise ValueError(f"existing Replay Run {key} differs; refusing to resume")
        if manifest.get("status") in {"complete", "mining_insufficient", "validation_inadequate", "failed"}:
            raise FileExistsError(f"Replay Run is already terminal: {run_root}")
        return run_root, manifest
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "trainer").mkdir()
    manifest = {
        **identity,
        "status": "running",
        "stage": "created",
        "baseline_backup_gate": "not_run",
        "mining_status": "not_run",
        "schedule_status": "not_run",
        "pilot_status": "not_run",
        "extension_status": "not_run",
        "final_selection": None,
        "stages": {},
        "updated_at_utc": _now(),
    }
    _write_exclusive(manifest_path, manifest)
    return run_root, manifest


def _save_run_manifest(run_root: Path, manifest: dict[str, Any], **updates: Any) -> dict[str, Any]:
    manifest.update(updates)
    manifest["updated_at_utc"] = _now()
    _write_atomic(run_root / "manifest.json", manifest)
    return manifest


def _write_stage(manifest: dict[str, Any], run_root: Path, name: str, facts: dict[str, Any]) -> dict[str, Any]:
    stages = dict(manifest.get("stages", {}))
    stages[name] = facts
    return _save_run_manifest(run_root, manifest, stages=stages, stage=name)


def _load_val_facts(config: dict[str, Any], run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    validation = config["validation"]
    paths = {
        "input": _project_path(validation["input_path"]),
        "id_map": _project_path(validation["id_map_path"]),
        "historical": _project_path(validation["historical_exposure_path"]),
        "parent_predictions": _project_path(validation["parent_predictions_path"]),
    }
    for key, sha_key in (
        ("input", "input_sha256"),
        ("id_map", "id_map_sha256"),
        ("historical", "historical_exposure_sha256"),
        ("parent_predictions", "parent_predictions_sha256"),
    ):
        if sha256_file(paths[key]) != validation[sha_key]:
            raise ValueError(f"frozen Val source identity differs: {key}")
    inputs = _read_jsonl(paths["input"])
    id_rows = _read_jsonl(paths["id_map"])
    id_map = {}
    for row in id_rows:
        eval_id = row.get("eval_id")
        if not isinstance(eval_id, str) or eval_id in id_map:
            raise ValueError("frozen Val ID map contains a duplicate or missing eval_id")
        id_map[eval_id] = row
    historical = _read_json(paths["historical"])
    historical_ids = historical.get("ids")
    if not isinstance(historical_ids, list) or len(historical_ids) != 209 or len(set(historical_ids)) != 209:
        raise ValueError("frozen A2 Val exposure list does not contain 209 unique IDs")
    if set(historical_ids) - set(id_map):
        raise ValueError("frozen A2 exposure includes IDs outside the target-free Val set")
    historical_conversations = {
        str(id_map[sample_id]["conversation_id"])
        for sample_id in historical_ids
        if sample_id in id_map
    }
    if len(historical_conversations) != 78 or len(historical_conversations) != len({str(id_map[sid]["conversation_id"]) for sid in historical_ids if sid in id_map}):
        raise ValueError("A2 exact point exposure does not map to 78 Val conversations")
    splits = freeze_val_splits(
        inputs,
        id_map,
        historical_conversations,
        seed=int(config["seed"]),
        gate_fraction=float(validation["gate_fraction_by_conversation"]),
    )
    for cohort_name in ("val_gate", "val_confirm"):
        if len(splits["cohorts"][cohort_name]["ids"]) < int(validation["minimum_cohort_points"]):
            raise ValueError(f"frozen Val {cohort_name} has too few prediction points")
    split_root = run_root / "validation"
    split_root.mkdir(parents=True, exist_ok=True)
    split_path = split_root / "val_splits.json"
    public_splits = serialize_val_splits(splits)
    if split_path.exists():
        if _read_json(split_path) != public_splits:
            raise ValueError("existing frozen Val split differs; refusing to overwrite")
    else:
        _write_exclusive(split_path, public_splits)

    parent_predictions = _read_jsonl(paths["parent_predictions"])
    parent_index = {str(row.get("id", "")): row for row in parent_predictions}
    if len(parent_index) != len(parent_predictions) or set(parent_index) != set(id_map):
        raise ValueError("frozen P4504 Val predictions do not cover the exact target-free Val ID set")
    subset_paths = {}
    for name in ("historical_exposed", "val_gate", "val_confirm"):
        cohort = splits["cohorts"][name]
        subset = [parent_index[eval_id] for eval_id in cohort["ids"]]
        path = split_root / f"parent_predictions_{name}.jsonl"
        if path.exists():
            if [row.get("id") for row in _read_jsonl(path)] != cohort["ids"]:
                raise ValueError(f"existing frozen P4504 {name} predictions differ")
        else:
            _write_jsonl_exclusive(path, subset)
        subset_paths[name] = {
            "path": str(path.relative_to(ROOT)),
            "prediction_count": len(subset),
            "prediction_sha256": sha256_file(path),
            "ids_sha256": cohort["ids_sha256"],
        }
    parent_manifest_path = split_root / "parent_predictions_manifest.json"
    parent_proof = {
        "status": "predictions_frozen",
        "source_prediction_sha256": sha256_file(paths["parent_predictions"]),
        "source_id_sha256": canonical_sha256(list(parent_index)),
        "cohorts": subset_paths,
        "references_read": False,
    }
    if parent_manifest_path.exists():
        if _read_json(parent_manifest_path) != parent_proof:
            raise ValueError("existing P4504 Val subset manifest differs")
    else:
        _write_exclusive(parent_manifest_path, parent_proof)
    return splits, inputs, id_map, {
        "source_paths": {key: str(value.relative_to(ROOT)) for key, value in paths.items()},
        "source_sha256": {key: sha256_file(value) for key, value in paths.items()},
        "split_path": str(split_path.relative_to(ROOT)),
        "split_sha256": sha256_file(split_path),
        "parent_predictions": parent_index,
        "parent_subset_paths": subset_paths,
        "parent_subset_manifest_sha256": sha256_file(parent_manifest_path),
        "historical_conversation_count": len(historical_conversations),
        "val_input_count": len(inputs),
    }


def _load_val_references(config: dict[str, Any], splits: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validation = config["validation"]
    path = _project_path(validation["normalized_gold_path"])
    if sha256_file(path) != validation["normalized_gold_sha256"]:
        raise ValueError("frozen D0.3 Val Gold hash differs")
    wanted = set().union(*(set(splits["cohorts"][name]["ids"]) for name in ("val_gate", "val_confirm")))
    references: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            eval_id = row.get("sample_id")
            if eval_id not in wanted:
                continue
            target = row.get("target")
            profile = target.get("user_profile") if isinstance(target, dict) else None
            if not isinstance(profile, dict):
                raise ValueError(f"Val Gold row {number} has no profile")
            references[eval_id] = {"target": {"emotion_label": target.get("emotion_label", ""), "user_profile": {dim: list(profile[dim]) for dim in PROFILE_DIMS}}}
    if set(references) != wanted:
        raise ValueError("frozen Val Gold does not cover all fixed split IDs")
    return references


def _freeze_parent_metrics(
    config: dict[str, Any],
    run_root: Path,
    splits: dict[str, Any],
    val_facts: dict[str, Any],
    enums: dict[str, list[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    parent_predictions = val_facts["parent_predictions"]
    references = _load_val_references(config, splits)
    metrics = {}
    cohort_readiness = {}
    for name in ("val_gate", "val_confirm"):
        cohort = splits["cohorts"][name]
        path = _project_path(val_facts["parent_subset_paths"][name]["path"])
        if sha256_file(path) != val_facts["parent_subset_paths"][name]["prediction_sha256"]:
            raise ValueError(f"frozen P4504 {name} predictions changed before references were loaded")
        subset_predictions = {eval_id: parent_predictions[eval_id] for eval_id in cohort["ids"]}
        metrics[name] = score_a8_cohort(cohort, references, subset_predictions, enums, arm="P")
        profile = metrics[name]["profile_33_label"]
        positive_by_dimension = {
            dimension: sum(profile["per_dimension"][dimension]["per_label"][label]["support"] for label in enums[dimension])
            for dimension in PROFILE_DIMS
        }
        cohort_readiness[name] = {
            "point_count": len(cohort["ids"]),
            "conversation_count": cohort["conversation_count"],
            "gold_positive_label_count_by_dimension": positive_by_dimension,
            "gold_positive_label_count_total": sum(positive_by_dimension.values()),
            "status": "passed" if min(positive_by_dimension.values(), default=0) >= 10 and sum(positive_by_dimension.values()) >= 50 else "failed",
        }
    failed = [name for name, item in cohort_readiness.items() if item["status"] != "passed"]
    result = {
        "status": "passed" if not failed else "failed",
        "quality_scope": "local_proxy",
        "references_opened_after_parent_predictions_frozen": True,
        "cohort_readiness": cohort_readiness,
        "failed_cohorts": failed,
        "parent_metrics": metrics,
        "source_gold_sha256": config["validation"]["normalized_gold_sha256"],
        "created_at_utc": _now(),
    }
    path = run_root / "validation" / "parent_val_metrics.json"
    if path.exists():
        if _read_json(path) != result:
            # Timestamp differences are expected only after a crash before write;
            # all substantive fields must remain identical.
            existing = _read_json(path)
            existing.pop("created_at_utc", None)
            comparable = dict(result)
            comparable.pop("created_at_utc", None)
            if existing != comparable:
                raise ValueError("frozen parent Val metrics changed")
            result = _read_json(path)
    else:
        _write_exclusive(path, result)
    return references, result


def _prediction_manifest_path(path: Path) -> Path:
    return path.with_name(path.stem + ".manifest.json")


def _verify_prediction_file(path: Path, manifest_path: Path, expected_ids: list[str], identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"frozen prediction file/manifest is missing: {path}")
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(path)
    ids = [str(row.get("id", row.get("sample_id", ""))) for row in rows]
    if (
        manifest.get("status") != "predictions_frozen"
        or manifest.get("prediction_sha256") != sha256_file(path)
        or manifest.get("prediction_count") != len(expected_ids)
        or manifest.get("ids_sha256") != canonical_sha256(expected_ids)
        or ids != expected_ids
    ):
        raise ValueError(f"frozen predictions differ from expected IDs or hash: {path}")
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"prediction manifest identity differs for {key}: {path}")
    return rows


def _generate_one(
    model: Any,
    tokenizer: Any,
    torch: Any,
    *,
    sample_id: str,
    conversation_id: str,
    prompt_ids: list[int],
    schema: dict[str, Any],
    max_new_tokens: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not prompt_ids or len(prompt_ids) > 1280:
        raise ValueError(f"{sample_id}: frozen A8 P Prompt length is invalid")
    device = model.get_input_embeddings().weight.device
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(prompt_tensor)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt_tensor,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    generated_ids = [int(value) for value in output[0, prompt_tensor.shape[1]:].detach().cpu().tolist()]
    eos_positions = [index for index, token in enumerate(generated_ids) if token == int(tokenizer.eos_token_id)]
    eos_generated = bool(eos_positions)
    reason = None
    visible_ids = generated_ids
    if eos_positions:
        eos_position = eos_positions[0]
        if eos_position != len(generated_ids) - 1:
            reason = "tokens_after_eos"
        visible_ids = generated_ids[:eos_position]
    else:
        reason = "missing_eos_or_truncated"
    special_ids = {int(token) for token in getattr(tokenizer, "all_special_ids", [])}
    raw_text = tokenizer.decode(visible_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if reason is None and any(token in special_ids for token in visible_ids):
        reason = "unexpected_special_token"
    parsed = None
    status = "invalid"
    if reason is None:
        try:
            parsed = parse_specialist_output(raw_text, schema, "P")
            status = "valid"
        except A8ParseError as error:
            reason = error.code
    result = {
        "id": sample_id,
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "status": status,
        "reason": reason,
        "parsed": parsed,
        "raw_text": raw_text,
        "generated_token_count": len(generated_ids),
        "eos_generated": eos_generated,
        "stop_reason": "eos" if eos_generated else "token_limit_or_generation_end",
        "generation_seconds": elapsed,
    }
    if metadata:
        result.update(metadata)
    return result


def _generate_to_frozen_file(
    *,
    output_path: Path,
    input_ids: list[str],
    get_prompt: Any,
    model: Any,
    tokenizer: Any,
    torch: Any,
    schema: dict[str, Any],
    run_id: str,
    model_identity: dict[str, Any],
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    manifest_path = _prediction_manifest_path(output_path)
    if output_path.exists():
        _verify_prediction_file(output_path, manifest_path, input_ids, {"run_id": run_id, **model_identity})
        return _read_json(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    prior_rows = _read_jsonl(partial_path) if partial_path.exists() else []
    prior_ids = [str(row.get("id", "")) for row in prior_rows]
    if prior_ids != input_ids[:len(prior_ids)]:
        raise ValueError(f"partial prediction ledger is not a prefix of the frozen IDs: {partial_path}")
    if len(prior_rows) > len(input_ids):
        raise ValueError("partial prediction ledger exceeds its frozen input count")
    if not partial_path.exists():
        fd = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
    with partial_path.open("ab") as handle:
        pending = []
        for index in range(len(prior_rows), len(input_ids)):
            sample_id = input_ids[index]
            row = get_prompt(index)
            prediction = _generate_one(
                model,
                tokenizer,
                torch,
                sample_id=sample_id,
                conversation_id=str(row["conversation_id"]),
                prompt_ids=list(row["prompt_ids"]),
                schema=schema,
                max_new_tokens=max_new_tokens,
                metadata=row.get("metadata"),
            )
            pending.append(prediction)
            if len(pending) >= 32 or index + 1 == len(input_ids):
                for item in pending:
                    handle.write(_json_bytes(item))
                handle.flush()
                os.fsync(handle.fileno())
                pending.clear()
                complete = index + 1
                valid_count = sum(item.get("status") == "valid" for item in _read_jsonl(partial_path))
                print(f"[E03-A8 Replay {run_id}] generated {complete}/{len(input_ids)} valid={valid_count}", flush=True)
    os.replace(partial_path, output_path)
    rows = _read_jsonl(output_path)
    if [row.get("id") for row in rows] != input_ids:
        raise RuntimeError("generated Replay predictions do not cover the frozen ID order")
    manifest = {
        "status": "predictions_frozen",
        "run_id": run_id,
        **model_identity,
        "prediction_count": len(rows),
        "ids_sha256": canonical_sha256(input_ids),
        "prediction_sha256": sha256_file(output_path),
        "valid_count": sum(row.get("status") == "valid" for row in rows),
        "strict_parse_rate": sum(row.get("status") == "valid" for row in rows) / len(rows) if rows else 0.0,
        "eos_completion_count": sum(bool(row.get("eos_generated")) for row in rows),
        "eos_completion_rate": sum(bool(row.get("eos_generated")) for row in rows) / len(rows) if rows else 0.0,
        "generation_seconds": sum(float(row.get("generation_seconds", 0.0)) for row in rows),
        "references_read": False,
        "created_at_utc": _now(),
    }
    _write_exclusive(manifest_path, manifest)
    return manifest


def _generate_candidate_predictions(
    config: dict[str, Any],
    model_config: dict[str, Any],
    schema: dict[str, Any],
    candidate_ids: list[dict[str, Any]],
    run_root: Path,
    archive_base: Path,
    archive_checkpoint: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    output_path = run_root / "mining" / "candidate_predictions.jsonl"
    id_order = [row["sample_id"] for row in candidate_ids]
    generation_identity = {
        "stage": "candidate-parent-greedy-v1",
        "parent_checkpoint_manifest_sha256": config["parent"]["checkpoint_manifest_sha256"],
        "candidate_ids_sha256": canonical_sha256(id_order),
    }
    manifest_path = _prediction_manifest_path(output_path)
    if output_path.exists():
        _verify_prediction_file(output_path, manifest_path, id_order, {"run_id": run_id, **generation_identity})
        return _read_json(manifest_path)
    train_index = dev_index = None
    tokenizer = model = None
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(model_config)
        model, torch, model_facts = _load_train_model(
            model_config,
            archive_base,
            adapter_path=archive_checkpoint,
            autocast_adapter_dtype=True,
        )
        model.eval()
        p_path = _project_path(model_config["protocol"]["prompt_path"])
        if sha256_file(p_path) != model_config["protocol"]["prompt_sha256"]:
            raise ValueError("A8 P Prompt changed before candidate prediction")
        train_index = JsonlById(_project_path(config["data"]["train_encoded"]))
        dev_index = JsonlById(_project_path(config["data"]["dev_encoded"]))

        def prompt_at(index: int) -> dict[str, Any]:
            candidate = candidate_ids[index]
            source_index = train_index if candidate["source"] == "train" else dev_index
            row = source_index.get(candidate["sample_id"])
            if row.get("conversation_id") != candidate["conversation_id"]:
                raise ValueError(f"{candidate['sample_id']}: candidate conversation identity changed")
            prompt_ids = list(row["input_ids"][:row["prompt_length"]])
            if len(prompt_ids) != int(row["prompt_length"]):
                raise ValueError(f"{candidate['sample_id']}: candidate Prompt contains non-encoded completion tokens")
            return {
                "conversation_id": row["conversation_id"],
                "prompt_ids": prompt_ids,
                "metadata": {
                    "prompt_length": int(row["prompt_length"]),
                    "selected_history_sha256": row["common_history_sha256"],
                    "source": candidate["source"],
                },
            }

        facts = _generate_to_frozen_file(
            output_path=output_path,
            input_ids=id_order,
            get_prompt=prompt_at,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            schema=schema,
            run_id=run_id,
            model_identity=generation_identity,
        )
        facts = {**facts, "tokenizer_facts": tokenizer_facts, "model_facts": model_facts}
        enriched_manifest_path = output_path.parent / "candidate_prediction_model.json"
        if enriched_manifest_path.exists():
            if _read_json(enriched_manifest_path).get("prediction_sha256") != facts["prediction_sha256"]:
                raise ValueError("candidate prediction model facts are bound to another output")
        else:
            _write_exclusive(enriched_manifest_path, facts)
        return facts
    finally:
        if train_index is not None:
            train_index.close()
        if dev_index is not None:
            dev_index.close()
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _generate_val_predictions(
    config: dict[str, Any],
    model_config: dict[str, Any],
    schema: dict[str, Any],
    split_name: str,
    splits: dict[str, Any],
    val_inputs: list[dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    run_root: Path,
    archive_base: Path,
    adapter_checkpoint: Path,
    *,
    run_id: str,
    checkpoint_manifest_sha256: str,
    replay_step: int,
) -> dict[str, Any]:
    cohort = splits["cohorts"][split_name]
    id_set = set(cohort["ids"])
    inputs = [row for row in val_inputs if row["id"] in id_set]
    if [row["id"] for row in inputs] != cohort["ids"]:
        raise ValueError(f"frozen target-free Val input order differs for {split_name}")
    parent_prediction_path = run_root / "validation" / f"parent_predictions_{split_name}.jsonl"
    parent_predictions = {row["id"]: row for row in _read_jsonl(parent_prediction_path)}
    output_path = _project_path(config["outputs"]["evaluation_root"]) / run_id / f"{split_name}-replay-{replay_step:04d}" / "predictions.jsonl"
    identity = {
        "stage": f"{split_name}-replay-{replay_step:04d}",
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "cohort_ids_sha256": cohort["ids_sha256"],
        "val_split_sha256": sha256_file(run_root / "validation" / "val_splits.json"),
    }
    manifest_path = _prediction_manifest_path(output_path)
    if output_path.exists():
        rows = _verify_prediction_file(output_path, manifest_path, cohort["ids"], {"run_id": run_id, **identity})
        for row in rows:
            baseline = parent_predictions[row["id"]]
            if row.get("selected_history_sha256") != baseline.get("selected_history_sha256"):
                raise ValueError(f"{row['id']}: Replay Val prompt history differs from frozen P4504")
        return _read_json(manifest_path)

    p_prompt = _project_path(model_config["protocol"]["prompt_path"]).read_text(encoding="utf-8")
    ep_prompt_path = ROOT / "configs/prompts/p_a8_v2_emotion_profile.txt"
    ep_prompt = ep_prompt_path.read_text(encoding="utf-8")
    p0_prompt = _project_path(model_config["protocol"]["selection_prompt_path"]).read_text(encoding="utf-8")
    eval_config = yaml.safe_load((ROOT / "configs/evaluation/e03_a8_staged_fast.yaml").read_text(encoding="utf-8"))
    tokenizer = model = None
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(model_config)
        model, torch, model_facts = _load_train_model(
            model_config,
            archive_base,
            adapter_path=adapter_checkpoint,
            autocast_adapter_dtype=True,
        )
        model.eval()

        def prompt_at(index: int) -> dict[str, Any]:
            item = inputs[index]
            eval_id = str(item["id"])
            p0_selection = select_recent_complete_suffix(
                item.get("history"),
                p0_prompt,
                int(eval_config["protocol"]["max_prompt_tokens"]),
                tokenizer,
            )
            if p0_selection.get("status") != "ok":
                raise ValueError(f"{eval_id}: P0/T1 current-user floor exceeds the frozen prompt cap")
            from src.data.e03_a8_sft import select_common_history

            common = select_common_history(
                item.get("history"),
                p_prompt=p_prompt,
                ep_prompt=ep_prompt,
                tokenizer=tokenizer,
                initial_start_index=int(p0_selection["selected_start_index"]),
                prompt_cap=int(eval_config["protocol"]["max_prompt_tokens"]),
            )
            mapping = id_map[eval_id]
            baseline = parent_predictions[eval_id]
            if common["selected_history_sha256"] != baseline.get("selected_history_sha256"):
                raise ValueError(f"{eval_id}: reconstructed Replay history differs from frozen P4504")
            metadata = {
                "selected_start_index": common["selected_start_index"],
                "p0_selected_start_index": p0_selection["selected_start_index"],
                "extra_removed_messages": common["extra_removed_messages"],
                "selected_history_sha256": common["selected_history_sha256"],
                "p_prompt_tokens": common["prompt_lengths"]["P"],
                "ep_prompt_tokens": common["prompt_lengths"]["EP"],
            }
            return {
                "conversation_id": _conversation_id(mapping),
                "prompt_ids": list(common["prompt_ids"]["P"]),
                "metadata": metadata,
            }

        facts = _generate_to_frozen_file(
            output_path=output_path,
            input_ids=cohort["ids"],
            get_prompt=prompt_at,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            schema=schema,
            run_id=run_id,
            model_identity=identity,
        )
        rows = _read_jsonl(output_path)
        for row in rows:
            baseline = parent_predictions[row["id"]]
            if row.get("selected_history_sha256") != baseline.get("selected_history_sha256"):
                raise ValueError(f"{row['id']}: frozen prompt identity differs from P4504")
        facts = {**facts, "tokenizer_facts": tokenizer_facts, "model_facts": model_facts}
        facts_path = output_path.parent / "model_facts.json"
        if not facts_path.exists():
            _write_exclusive(facts_path, facts)
        return facts
    finally:
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _replay_checkpoint_path(run_root: Path, replay_step: int) -> Path:
    return run_root / "trainer" / f"checkpoint-replay-{replay_step:04d}"


def _optimizer_steps(optimizer_state: dict[str, Any]) -> list[int]:
    values = []
    for state in optimizer_state.get("state", {}).values():
        step_value = state.get("step")
        if step_value is not None:
            values.append(int(step_value.item()) if hasattr(step_value, "item") else int(step_value))
    return values


def _verify_replay_checkpoint(
    checkpoint: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    data_facts: dict[str, Any],
    mining_facts: dict[str, Any],
    schedule_sha256: str,
) -> dict[str, Any]:
    if not checkpoint.is_dir() or not (checkpoint / "checkpoint_manifest.json").is_file():
        raise FileNotFoundError(f"complete Replay checkpoint is missing: {checkpoint}")
    record = _read_json(checkpoint / "checkpoint_manifest.json")
    replay_step = int(record.get("replay_step", -1))
    expected = {
        "status": "complete",
        "run_id": run_id,
        "arm": "P",
        "replay_step": replay_step,
        "optimizer_step": replay_step,
        "global_optimizer_step": int(config["parent"]["global_optimizer_step"]) + replay_step,
        "parent_global_step": int(config["parent"]["global_optimizer_step"]),
        "parent_checkpoint_manifest_sha256": config["parent"]["checkpoint_manifest_sha256"],
        "base_model_weight_sha256": config["parent"]["base_weight_sha256"],
        "schedule_sha256": schedule_sha256,
        "candidate_ids_sha256": mining_facts["candidate_ids_sha256"],
        "train_encoded_sha256": data_facts["train_encoded_sha256"],
        "dev_encoded_sha256": data_facts["dev_encoded_sha256"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Replay checkpoint {key} differs: {record.get(key)!r} != {value!r}")
    if record.get("files_sha256") != canonical_sha256(record.get("files")):
        raise ValueError("Replay checkpoint file manifest hash is invalid")
    actual_files = _checkpoint_file_facts(checkpoint)
    if actual_files != record.get("files"):
        raise ValueError("Replay checkpoint Adapter/Optimizer/Scheduler/RNG file SHA differs")
    state = _read_json(checkpoint / "trainer_state.json")
    if (
        state.get("run_id") != run_id
        or state.get("arm") != "P"
        or state.get("global_step") != replay_step
        or state.get("replay_step") != replay_step
        or state.get("global_optimizer_step") != int(config["parent"]["global_optimizer_step"]) + replay_step
        or state.get("sample_cursor") != replay_step * int(config["schedule"]["gradient_accumulation_steps"])
        or state.get("scheduler_last_epoch") != int(config["parent"]["global_optimizer_step"]) + replay_step
        or state.get("rng_state_saved") is not True
    ):
        raise ValueError("Replay checkpoint Trainer state, step cursor, or RNG identity differs")
    import torch

    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    global_step = int(config["parent"]["global_optimizer_step"]) + replay_step
    if set(_optimizer_steps(optimizer)) != {global_step}:
        raise ValueError("Replay checkpoint AdamW moments do not match the global optimizer step")
    for group in optimizer.get("param_groups", []):
        if not math.isclose(float(group["lr"]), float(config["training"]["learning_rate"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Replay checkpoint AdamW LR differs from 2e-5")
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False)
    if int(scheduler.get("last_epoch", -1)) != global_step or any(
        not math.isclose(float(value), float(config["training"]["learning_rate"]), rel_tol=0.0, abs_tol=1e-12)
        for value in scheduler.get("base_lrs", [])
    ):
        raise ValueError("Replay checkpoint scheduler step or base LR differs")
    adapter_identity, adapter_config = inference.adapter_weight_identity(checkpoint)
    if adapter_config.get("r") != 8 or adapter_config.get("lora_alpha") != 16:
        raise ValueError("Replay checkpoint LoRA config differs from the frozen A8 config")
    if adapter_identity.get("weight_sha256") != record.get("adapter_identity", {}).get("weight_sha256"):
        raise ValueError("Replay checkpoint Adapter weight identity differs")
    return {"record": record, "trainer_state": state, "adapter_identity": adapter_identity}


def _save_replay_checkpoint(
    *,
    config: dict[str, Any],
    model_config: dict[str, Any],
    run_root: Path,
    run_id: str,
    replay_step: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    torch: Any,
    data_facts: dict[str, Any],
    mining_facts: dict[str, Any],
    schedule_sha256: str,
    code_identity: dict[str, str],
    elapsed_training_seconds: float,
    last_loss: float,
) -> tuple[Path, dict[str, Any]]:
    final_path = _replay_checkpoint_path(run_root, replay_step)
    _assert_safe_checkpoint_output(
        final_path,
        parent_checkpoint=_project_path(config["parent"]["checkpoint_path"]),
        local_baseline=_project_path(config["outputs"]["training_root"]) / "baselines" / f"P4504-{config['parent']['checkpoint_manifest_sha256'][:12]}",
        archive_root=_archive_path(config["outputs"]["archive_root"]),
    )
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite complete Replay checkpoint: {final_path}")
    temporary = final_path.with_name(final_path.name + f".partial-{run_id}")
    partial_attempt = 1
    while temporary.exists():
        temporary = final_path.with_name(final_path.name + f".partial-{run_id}-attempt-{partial_attempt:02d}")
        partial_attempt += 1
    temporary.mkdir(parents=False)
    model.save_pretrained(temporary, safe_serialization=True)
    adapter_config_path = temporary / "adapter_config.json"
    adapter_config = _read_json(adapter_config_path)
    adapter_config["base_model_name_or_path"] = model_config["base_model"]["model_id"]
    _write_atomic(adapter_config_path, adapter_config)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    torch.save(_rng_state(torch), temporary / "rng_state.pth")
    global_optimizer_step = int(config["parent"]["global_optimizer_step"]) + replay_step
    state = {
        "global_step": replay_step,
        "replay_step": replay_step,
        "global_optimizer_step": global_optimizer_step,
        "sample_cursor": replay_step * int(config["schedule"]["gradient_accumulation_steps"]),
        "run_id": run_id,
        "arm": "P",
        "parent_global_step": int(config["parent"]["global_optimizer_step"]),
        "last_loss": float(last_loss),
        "training_seconds": float(elapsed_training_seconds),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "rng_state_saved": True,
        "created_at_utc": _now(),
    }
    _write_exclusive(temporary / "trainer_state.json", state)
    adapter_identity, restored_config = inference.adapter_weight_identity(temporary)
    if restored_config.get("r") != 8 or restored_config.get("lora_alpha") != 16:
        raise ValueError("Replay checkpoint LoRA r/alpha identity changed during save")
    tensor_sha = _lora_tensor_sha256(model, torch)
    files = _checkpoint_file_facts(temporary)
    record = {
        "status": "complete",
        "version": "E03-A8-REPLAY-checkpoint-v1",
        "experiment_id": "E03-A8-REPLAY",
        "arm": "P",
        "run_id": run_id,
        "replay_step": replay_step,
        "optimizer_step": replay_step,
        "global_optimizer_step": global_optimizer_step,
        "parent_global_step": int(config["parent"]["global_optimizer_step"]),
        "sample_cursor": state["sample_cursor"],
        "parent_run_id": config["parent"]["run_id"],
        "parent_checkpoint_manifest_sha256": config["parent"]["checkpoint_manifest_sha256"],
        "base_model_id": model_config["base_model"]["model_id"],
        "base_revision": model_config["base_model"]["revision"],
        "base_archive_path": str(_archive_path(config["outputs"]["archive_root"]) / "base" / f"Qwen3-1.7B-{config['parent']['base_weight_sha256'][:12]}" / "model"),
        "base_model_weight_sha256": config["parent"]["base_weight_sha256"],
        "adapter_identity": adapter_identity,
        "adapter_tensor_sha256": tensor_sha,
        "config_sha256": canonical_sha256(config),
        "model_config_sha256": canonical_sha256(model_config),
        "train_encoded_sha256": data_facts["train_encoded_sha256"],
        "dev_encoded_sha256": data_facts["dev_encoded_sha256"],
        "train_normalized_sha256": data_facts["train_normalized_sha256"],
        "dev_normalized_sha256": data_facts["dev_normalized_sha256"],
        "candidate_ids_sha256": mining_facts["candidate_ids_sha256"],
        "candidate_predictions_sha256": mining_facts["candidate_predictions_sha256"],
        "hard_samples_sha256": mining_facts["hard_samples_sha256"],
        "schedule_sha256": schedule_sha256,
        "training_code_sha256": code_identity,
        "training_parameters": {
            "loss": "src.training.e03_a8_completion_loss.completion_only_forward_loss",
            "collator": "SFTDataCollator",
            "microbatch": 1,
            "gradient_accumulation": 4,
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "scheduler": "constant",
            "quantization": model_config["quantization"],
            "lora": model_config["lora"],
            "prompt_sha256": model_config["protocol"]["prompt_sha256"],
            "eos_token_id": model_config["protocol"]["eos_token_id"],
        },
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "cumulative_training_seconds": elapsed_training_seconds,
        "last_loss": float(last_loss),
        "files": files,
        "files_sha256": canonical_sha256(files),
        "created_at_utc": _now(),
    }
    _write_exclusive(temporary / "checkpoint_manifest.json", record)
    temporary.rename(final_path)
    return final_path, record


def _validate_replay_row(row: dict[str, Any], config: dict[str, Any], model_config: dict[str, Any]) -> None:
    legacy = dict(row)
    legacy["source_split"] = "train"
    _row_contract(
        legacy,
        eos_token_id=int(model_config["protocol"]["eos_token_id"]),
        prompt_cap=int(model_config["protocol"]["max_prompt_tokens"]),
        sequence_cap=int(model_config["protocol"]["max_train_seq_length"]),
    )
    if row["source_split"] not in {"train", "dev_ft"}:
        raise ValueError(f"{row['sample_id']}: Replay source must be Train or DevFT")


def _dry_step(
    *,
    config: dict[str, Any],
    model_config: dict[str, Any],
    run_root: Path,
    run_id: str,
    model: Any,
    tokenizer: Any,
    torch: Any,
    train_index: JsonlById,
    dev_index: JsonlById,
    schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    path = run_root / "trainer" / "dry_step.json"
    identity = {
        "run_id": run_id,
        "parent_checkpoint_manifest_sha256": config["parent"]["checkpoint_manifest_sha256"],
        "schedule_prefix_sha256": canonical_sha256(schedule[:4]),
    }
    if path.is_file():
        prior = _read_json(path)
        if prior.get("status") != "passed" or any(prior.get(key) != value for key, value in identity.items()):
            raise ValueError("existing Replay GPU dry-step gate differs")
        return prior
    collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=int(model_config["protocol"]["max_train_seq_length"]))
    rng_before = _rng_state(torch)
    before_sha = _lora_tensor_sha256(model, torch)
    trainable = _lora_parameters(model)
    losses = []
    completion_tokens = 0
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        for entry in schedule[:4]:
            index = train_index if entry["source"] == "train" else dev_index
            row = index.get(entry["sample_id"])
            _validate_replay_row(row, config, model_config)
            batch = collator([row])
            device = model.get_input_embeddings().weight.device
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = completion_only_forward_loss(model, batch, prompt_length=int(row["prompt_length"]), sequence_length=len(row["input_ids"]))
            if not bool(torch.isfinite(loss.detach()).item()):
                raise FloatingPointError("Replay GPU dry-step produced a non-finite Completion loss")
            losses.append(float(loss.detach().float().cpu().item()))
            completion_tokens += int(row["completion_length"])
            (loss / 4).backward()
        gradients = _gradient_summary(trainable, torch, full=True, loss_tensors=[torch.tensor(value, device=model.get_input_embeddings().weight.device) for value in losses])
        model.zero_grad(set_to_none=True)
        after_sha = _lora_tensor_sha256(model, torch)
        if before_sha != after_sha:
            raise RuntimeError("Replay GPU dry-step changed Adapter parameters without optimizer.step")
        lr = float(config["training"]["learning_rate"])
        result = {
            "status": "passed",
            **identity,
            "optimizer_update_performed": False,
            "initial_adapter_tensor_sha256": before_sha,
            "post_backward_adapter_tensor_sha256": after_sha,
            "sample_ids": [entry["sample_id"] for entry in schedule[:4]],
            "completion_tokens": completion_tokens,
            "mean_completion_loss": sum(losses) / len(losses),
            "gradient_check": gradients,
            "first_optimizer_step_learning_rate": lr,
            "cuda_memory": inference.cuda_memory_facts(torch),
            "created_at_utc": _now(),
        }
        _write_exclusive(path, result)
        return result
    finally:
        model.zero_grad(set_to_none=True)
        _restore_rng_state(rng_before, torch)


def _run_training_segment(
    *,
    config: dict[str, Any],
    model_config: dict[str, Any],
    run_root: Path,
    run_id: str,
    start_step: int,
    stop_step: int,
    previous_checkpoint: Path,
    previous_is_parent: bool,
    local_parent_checkpoint: Path,
    parent_adapter_tensor_sha256: str,
    archive_base: Path,
    schedule: list[dict[str, Any]],
    schedule_sha256: str,
    data_facts: dict[str, Any],
    mining_facts: dict[str, Any],
    candidate_ids: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_gpu_unavailable_or_bf16_unsupported: Replay training requires CUDA BF16")
    checkpoint = _replay_checkpoint_path(run_root, stop_step)
    if checkpoint.exists():
        checked = _verify_replay_checkpoint(
            checkpoint,
            config=config,
            run_id=run_id,
            data_facts=data_facts,
            mining_facts=mining_facts,
            schedule_sha256=schedule_sha256,
        )
        training_summary_path = run_root / "steps" / f"replay-{stop_step:04d}.json"
        summary = _read_json(training_summary_path) if training_summary_path.is_file() else {
            "status": "checkpoint_verified_after_resume",
            "replay_step": stop_step,
            "global_optimizer_step": int(config["parent"]["global_optimizer_step"]) + stop_step,
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
        }
        check = _independent_reload(
            config,
            model_config,
            run_root,
            checkpoint,
            checked,
            archive_base,
            candidate_ids,
            data_facts,
            mining_facts,
            schedule_sha256,
            run_id=run_id,
            expected_tensor_sha=summary.get("adapter_tensor_sha256_before_reload"),
            expected_next_token=summary.get("next_token_before_reload"),
        )
        summary.update({
            "status": "passed",
            "independent_reload": check,
            "independent_reload_status": check["status"],
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_manifest_sha256": check["checkpoint_manifest_sha256"],
        })
        training_summary_path.parent.mkdir(parents=True, exist_ok=True)
        if training_summary_path.is_file():
            _write_atomic(training_summary_path, summary)
        else:
            _write_exclusive(training_summary_path, summary)
        nodes = dict(manifest.get("nodes", {}))
        nodes[str(stop_step)] = {"training_status": "passed", "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"), "independent_reload_status": check["status"]}
        _save_run_manifest(run_root, manifest, nodes=nodes, latest_checkpoint=str(checkpoint.relative_to(ROOT)), stage=f"checkpoint_{stop_step:04d}")
        return summary, checked
    if start_step < 0 or stop_step not in {128, 256, 384} or start_step >= stop_step:
        raise ValueError("Replay segment step boundaries are invalid")
    if previous_is_parent and start_step != 0:
        raise ValueError("initial P4504 restore must begin at Replay step zero")
    if not previous_is_parent and start_step not in {128, 256}:
        raise ValueError("Replay continuation must start from a passed checkpoint")
    from src.training.e03_a8_v2_staged import restore_recovery_bundle

    tokenizer = model = optimizer = scheduler = None
    train_index = dev_index = None
    attempt = 1
    segment_stem = f"step_ledger_replay_{start_step:04d}_{stop_step:04d}_attempt-"
    existing_attempts = sorted(run_root.glob(f"{segment_stem}*.jsonl"))
    error_attempts = sorted((run_root / "training_errors").glob(f"replay-{start_step:04d}-{stop_step:04d}-attempt-*.json")) if (run_root / "training_errors").is_dir() else []
    all_attempt_numbers = [int(path.stem.rsplit("attempt-", 1)[-1]) for path in existing_attempts]
    all_attempt_numbers.extend(int(path.stem.rsplit("attempt-", 1)[-1]) for path in error_attempts)
    if all_attempt_numbers:
        attempt = max(all_attempt_numbers) + 1
    ledger_path = run_root / f"{segment_stem}{attempt:02d}.jsonl"
    if ledger_path.exists():
        raise FileExistsError(f"Replay training ledger already exists: {ledger_path}")
    segment_start = time.monotonic()
    losses_segment: list[float] = []
    gpu_seconds_segment: list[float] = []
    pending: list[dict[str, Any]] = []
    pending_loss_tensors = []
    previous_training_seconds = 0.0
    lr_reset_facts = None
    model_facts = None
    start_tensor_sha = None
    try:
        tokenizer, tokenizer_facts = _runtime_tokenizer(model_config)
        model, torch, model_facts = _load_train_model(
            model_config,
            archive_base,
            adapter_path=previous_checkpoint,
            is_trainable_adapter=True,
            autocast_adapter_dtype=True,
        )
        trainable = _lora_parameters(model)
        start_tensor_sha = _lora_tensor_sha256(model, torch)
        optimizer = torch.optim.AdamW(
            [parameter for _name, parameter in trainable],
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        if previous_is_parent:
            if start_tensor_sha != parent_adapter_tensor_sha256:
                raise ValueError("P4504 Adapter tensor SHA differs before Replay training")
            optimizer.load_state_dict(torch.load(previous_checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
            scheduler.load_state_dict(torch.load(previous_checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
            _restore_rng_state(torch.load(previous_checkpoint / "rng_state.pth", map_location="cpu", weights_only=False), torch)
            if scheduler.last_epoch != int(config["parent"]["global_optimizer_step"]) or set(_optimizer_steps(optimizer.state_dict())) != {int(config["parent"]["global_optimizer_step"])}:
                raise ValueError("P4504 AdamW or Scheduler state did not restore at global step 4,504")
        else:
            checked_parent = _verify_replay_checkpoint(
                previous_checkpoint,
                config=config,
                run_id=run_id,
                data_facts=data_facts,
                mining_facts=mining_facts,
                schedule_sha256=schedule_sha256,
            )
            if start_tensor_sha != checked_parent["record"].get("adapter_tensor_sha256"):
                raise ValueError("Replay continuation Adapter tensor SHA differs from its previous checkpoint")
            restored_state = restore_recovery_bundle(
                previous_checkpoint,
                optimizer=optimizer,
                scheduler=scheduler,
                torch=torch,
                expected_run_id=run_id,
                expected_arm="P",
                accumulation_steps=int(config["schedule"]["gradient_accumulation_steps"]),
            )
            if restored_state.get("global_step") != start_step or restored_state.get("sample_cursor") != start_step * 4:
                raise ValueError("Replay continuation cursor differs from its previous checkpoint")
            previous_training_seconds = float(restored_state.get("training_seconds", 0.0))
            if scheduler.last_epoch != int(config["parent"]["global_optimizer_step"]) + start_step:
                raise ValueError("Replay continuation Scheduler step differs from its checkpoint")
        lr_reset_facts = _set_replay_learning_rate(optimizer, scheduler, float(config["training"]["learning_rate"]))
        if any(not math.isclose(float(group["lr"]), 2e-5, rel_tol=0.0, abs_tol=1e-12) for group in optimizer.param_groups):
            raise ValueError("first Replay optimizer step would not use exactly 2e-5")
        if len(optimizer.state) != len(trainable):
            raise ValueError("Replay must restore the complete AdamW moment state")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_index = JsonlById(_project_path(config["data"]["train_encoded"]))
        dev_index = JsonlById(_project_path(config["data"]["dev_encoded"]))
        dry_facts = _dry_step(
            config=config,
            model_config=model_config,
            run_root=run_root,
            run_id=run_id,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            train_index=train_index,
            dev_index=dev_index,
            schedule=schedule,
        )
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=int(model_config["protocol"]["max_train_seq_length"]))
        for replay_step in range(start_step + 1, stop_step + 1):
            if replay_step != schedule[(replay_step - 1) * 4]["optimizer_step"]:
                raise ValueError(f"Replay schedule step cursor does not match {replay_step}")
            entries = schedule[(replay_step - 1) * 4:replay_step * 4]
            if len(entries) != 4 or len({row["sample_id"] for row in entries}) != 4:
                raise ValueError(f"Replay optimizer step {replay_step} has invalid microbatch entries")
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            completion_tokens = prompt_tokens = sequence_tokens = 0
            step_start = time.monotonic()
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            for entry in entries:
                index = train_index if entry["source"] == "train" else dev_index
                row = index.get(entry["sample_id"])
                if row.get("conversation_id") != entry["conversation_id"]:
                    raise ValueError(f"{entry['sample_id']}: schedule conversation identity changed")
                _validate_replay_row(row, config, model_config)
                batch = collator([row])
                device = model.get_input_embeddings().weight.device
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = completion_only_forward_loss(
                    model,
                    batch,
                    prompt_length=int(row["prompt_length"]),
                    sequence_length=len(row["input_ids"]),
                )
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError(f"Replay produced non-finite Completion loss at step {replay_step}")
                step_losses.append(loss.detach().float())
                completion_tokens += int(row["completion_length"])
                prompt_tokens += int(row["prompt_length"])
                sequence_tokens += len(row["input_ids"])
                (loss / 4).backward()
            diagnostic = replay_step == start_step + 1 or replay_step % 32 == 0 or replay_step == stop_step
            gradient = _gradient_summary(trainable, torch, full=True, loss_tensors=step_losses) if diagnostic else None
            lr_before_step = [float(group["lr"]) for group in optimizer.param_groups]
            if any(not math.isclose(value, 2e-5, rel_tol=0.0, abs_tol=1e-12) for value in lr_before_step):
                raise RuntimeError(f"Replay optimizer LR changed before optimizer.step at {replay_step}: {lr_before_step}")
            optimizer.step()
            scheduler.step()
            model.zero_grad(set_to_none=True)
            event_end.record()
            pending_loss_tensors.extend(step_losses)
            pending.append({
                "replay_step": replay_step,
                "global_optimizer_step": int(config["parent"]["global_optimizer_step"]) + replay_step,
                "sample_ids": [row["sample_id"] for row in entries],
                "source_splits": [row["source"] for row in entries],
                "buckets": [row["bucket"] for row in entries],
                "gold_nonempty": [bool(row["gold_nonempty"]) for row in entries],
                "profile_dimensions": [row["profile_dimensions"] for row in entries],
                "history_message_counts": [row["history_message_count"] for row in entries],
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "sequence_tokens": sequence_tokens,
                "loss_tensors": step_losses,
                "lr_before_optimizer_step": lr_before_step,
                "gradient": gradient,
                "event_start": event_start,
                "event_end": event_end,
                "host_step_seconds": time.monotonic() - step_start,
            })
            if len(pending) >= int(config["training"]["ledger_flush_steps"]) or replay_step == stop_step:
                torch.cuda.synchronize()
                flat_losses = torch.stack(pending_loss_tensors).detach().float().cpu().tolist()
                cursor = 0
                ledger_rows = []
                for item in pending:
                    values = [float(value) for value in flat_losses[cursor:cursor + 4]]
                    cursor += 4
                    mean_loss = sum(values) / len(values)
                    gpu_seconds = float(item["event_start"].elapsed_time(item["event_end"]) / 1000.0)
                    losses_segment.append(mean_loss)
                    gpu_seconds_segment.append(gpu_seconds)
                    ledger_rows.append({
                        "run_id": run_id,
                        "arm": "P",
                        "replay_step": item["replay_step"],
                        "global_optimizer_step": item["global_optimizer_step"],
                        "sample_ids": item["sample_ids"],
                        "sample_ids_sha256": canonical_sha256(item["sample_ids"]),
                        "source_splits": item["source_splits"],
                        "buckets": item["buckets"],
                        "gold_nonempty": item["gold_nonempty"],
                        "profile_dimensions": item["profile_dimensions"],
                        "history_message_counts": item["history_message_counts"],
                        "completion_tokens": item["completion_tokens"],
                        "prompt_tokens": item["prompt_tokens"],
                        "sequence_tokens": item["sequence_tokens"],
                        "micro_sample_losses": values,
                        "mean_completion_loss": mean_loss,
                        "lr_before_optimizer_step": item["lr_before_optimizer_step"],
                        "gradient": item["gradient"],
                        "gpu_step_seconds": gpu_seconds,
                        "host_step_seconds": item["host_step_seconds"],
                        "created_at_utc": _now(),
                    })
                _append_jsonl_fsync(ledger_path, ledger_rows)
                pending.clear()
                pending_loss_tensors.clear()
                memory = inference.cuda_memory_facts(torch)
                if memory.get("cuda_available") and int(memory["allocator_peak_allocated_bytes"]) > int(memory["total_memory_bytes"]) * float(config["training"]["max_cuda_memory_fraction"]):
                    raise RuntimeError("Replay training exceeded the frozen 92% CUDA allocator budget")
                elapsed = previous_training_seconds + (time.monotonic() - segment_start)
                if elapsed > float(config["training"]["max_training_seconds"]):
                    raise TimeoutError("Replay exceeded its configured total training-time limit")
                print(
                    f"[E03-A8 Replay P] step={replay_step}/{stop_step} global={int(config['parent']['global_optimizer_step']) + replay_step} "
                    f"loss={sum(losses_segment[-len(ledger_rows):]) / len(ledger_rows):.5f} "
                    f"gpu_p50={statistics.median(gpu_seconds_segment):.2f}s elapsed={elapsed:.0f}s",
                    flush=True,
                )
        elapsed_training = previous_training_seconds + (time.monotonic() - segment_start)
        if not losses_segment or not gpu_seconds_segment:
            # A resumed stage can consist only of an already completed checkpoint;
            # this path is guarded earlier and cannot silently fabricate a node.
            raise RuntimeError("Replay training segment recorded no optimization steps")
        checkpoint_path, checkpoint_record = _save_replay_checkpoint(
            config=config,
            model_config=model_config,
            run_root=run_root,
            run_id=run_id,
            replay_step=stop_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            torch=torch,
            data_facts=data_facts,
            mining_facts=mining_facts,
            schedule_sha256=schedule_sha256,
            code_identity=_code_identity(),
            elapsed_training_seconds=elapsed_training,
            last_loss=losses_segment[-1],
        )
        # Keep only the prompt tokens for a fixed, target-free reload probe.
        probe = candidate_ids[0]
        probe_index = train_index if probe["source"] == "train" else dev_index
        probe_row = probe_index.get(probe["sample_id"])
        probe_prompt = list(probe_row["input_ids"][:probe_row["prompt_length"]])
        before_tensor_sha = _lora_tensor_sha256(model, torch)
        before_next_token = _next_token_id(model, torch, probe_prompt)
        if before_tensor_sha != checkpoint_record["adapter_tensor_sha256"]:
            raise ValueError("saved Replay checkpoint LoRA tensor SHA differs from the in-memory model")
        memory = inference.cuda_memory_facts(torch)
        summary = {
            "status": "checkpoint_saved_pending_independent_reload",
            "run_id": run_id,
            "replay_step": stop_step,
            "global_optimizer_step": int(config["parent"]["global_optimizer_step"]) + stop_step,
            "parent_global_step": int(config["parent"]["global_optimizer_step"]),
            "sample_cursor": stop_step * 4,
            "segment_start_step_exclusive": start_step,
            "segment_stop_step_inclusive": stop_step,
            "segment_steps": stop_step - start_step,
            "segment_training_seconds": time.monotonic() - segment_start,
            "cumulative_replay_training_seconds": elapsed_training,
            "loss_mean": sum(losses_segment) / len(losses_segment),
            "last_loss": losses_segment[-1],
            "gpu_step_seconds": {
                "mean": statistics.fmean(gpu_seconds_segment),
                "p50": statistics.median(gpu_seconds_segment),
                "p90": sorted(gpu_seconds_segment)[math.ceil(0.90 * len(gpu_seconds_segment)) - 1],
                "count": len(gpu_seconds_segment),
            },
            "source_exposure_count": dict(Counter(source for row in schedule[start_step * 4:stop_step * 4] for source in [row["source"]])),
            "bucket_exposure_count": dict(Counter(row["bucket"] for row in schedule[start_step * 4:stop_step * 4])),
            "gold_nonempty_exposure_count": sum(bool(row["gold_nonempty"]) for row in schedule[start_step * 4:stop_step * 4]),
            "dimension_exposure_count": {dim: sum(dim in row["profile_dimensions"] for row in schedule[start_step * 4:stop_step * 4]) for dim in PROFILE_DIMS},
            "first_optimizer_step_lr": lr_reset_facts,
            "dry_step": dry_facts,
            "model_facts": model_facts,
            "tokenizer_facts": tokenizer_facts,
            "cuda_memory": memory,
            "ledger_path": str(ledger_path.relative_to(ROOT)),
            "ledger_sha256": sha256_file(ledger_path),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_path / "checkpoint_manifest.json"),
            "adapter_tensor_sha256_before_reload": before_tensor_sha,
            "next_token_before_reload": before_next_token,
            "probe_sample_id": probe["sample_id"],
            "probe_prompt_sha256": canonical_sha256(probe_prompt),
            "created_at_utc": _now(),
        }
        steps_root = run_root / "steps"
        steps_root.mkdir(exist_ok=True)
        summary_path = steps_root / f"replay-{stop_step:04d}.json"
        if summary_path.exists():
            raise FileExistsError(f"Replay step summary already exists: {summary_path}")
        _write_exclusive(summary_path, summary)
        del model, optimizer, scheduler
        model = optimizer = scheduler = None
        gc.collect()
        torch.cuda.empty_cache()
        checked = _verify_replay_checkpoint(
            checkpoint_path,
            config=config,
            run_id=run_id,
            data_facts=data_facts,
            mining_facts=mining_facts,
            schedule_sha256=schedule_sha256,
        )
        reload = _independent_reload(
            config,
            model_config,
            run_root,
            checkpoint_path,
            checked,
            archive_base,
            candidate_ids,
            data_facts,
            mining_facts,
            schedule_sha256,
            run_id=run_id,
            expected_tensor_sha=before_tensor_sha,
            expected_next_token=before_next_token,
        )
        summary.update({
            "status": "passed",
            "independent_reload": reload,
            "independent_reload_status": reload["status"],
        })
        _write_atomic(summary_path, summary)
        nodes = dict(manifest.get("nodes", {}))
        nodes[str(stop_step)] = {
            "training_status": "passed",
            "checkpoint_manifest_sha256": checked["record"]["_manifest_file_sha256"],
            "independent_reload_status": reload["status"],
            "training_summary_path": str(summary_path.relative_to(ROOT)),
            "training_summary_sha256": sha256_file(summary_path),
        }
        _save_run_manifest(
            run_root,
            manifest,
            nodes=nodes,
            latest_checkpoint=str(checkpoint_path.relative_to(ROOT)),
            latest_checkpoint_manifest_sha256=checked["record"]["_manifest_file_sha256"],
            replay_steps_completed=stop_step,
        )
        return summary, checked
    except BaseException as exc:
        stage_path = run_root / "training_errors"
        stage_path.mkdir(exist_ok=True)
        error_path = stage_path / f"replay-{start_step:04d}-{stop_step:04d}-attempt-{attempt:02d}.json"
        if not error_path.exists():
            _write_exclusive(error_path, {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": f"{type(exc).__name__}: {exc}", "last_successful_replay_step": start_step, "created_at_utc": _now()})
        raise
    finally:
        if train_index is not None:
            train_index.close()
        if dev_index is not None:
            dev_index.close()
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _independent_reload(
    config: dict[str, Any],
    model_config: dict[str, Any],
    run_root: Path,
    checkpoint: Path,
    checked: dict[str, Any],
    archive_base: Path,
    candidate_ids: list[dict[str, Any]],
    data_facts: dict[str, Any],
    mining_facts: dict[str, Any],
    schedule_sha256: str,
    *,
    run_id: str,
    expected_tensor_sha: str | None = None,
    expected_next_token: int | None = None,
) -> dict[str, Any]:
    import torch

    replay_step = int(checked["record"]["replay_step"])
    check_path = run_root / "checkpoint_checks" / f"replay-{replay_step:04d}.json"
    if check_path.is_file():
        prior = _read_json(check_path)
        if prior.get("status") != "passed" or prior.get("checkpoint_manifest_sha256") != sha256_file(checkpoint / "checkpoint_manifest.json"):
            raise ValueError("existing Replay independent reload record differs")
        if expected_tensor_sha is not None and prior.get("adapter_tensor_sha256") != expected_tensor_sha:
            raise ValueError("existing Replay reload tensor SHA differs from the pre-save model")
        if expected_next_token is not None and prior.get("next_token_id") != expected_next_token:
            raise ValueError("existing Replay reload next token differs from the pre-save model")
        return prior
    probe = candidate_ids[0]
    index = JsonlById(_project_path(config["data"]["train_encoded"] if probe["source"] == "train" else config["data"]["dev_encoded"]))
    try:
        row = index.get(probe["sample_id"])
        prompt_ids = list(row["input_ids"][:row["prompt_length"]])
    finally:
        index.close()
    model = None
    try:
        model, torch, model_facts = _load_train_model(
            model_config,
            archive_base,
            adapter_path=checkpoint,
            autocast_adapter_dtype=True,
        )
        model.eval()
        tensor_sha = _lora_tensor_sha256(model, torch)
        next_token = _next_token_id(model, torch, prompt_ids)
        expected_tensor_sha = expected_tensor_sha or str(checked["record"]["adapter_tensor_sha256"])
        if expected_next_token is None:
            del model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            second_model, torch, second_facts = _load_train_model(
                model_config,
                archive_base,
                adapter_path=checkpoint,
                autocast_adapter_dtype=True,
            )
            try:
                second_model.eval()
                second_tensor_sha = _lora_tensor_sha256(second_model, torch)
                second_next_token = _next_token_id(second_model, torch, prompt_ids)
            finally:
                del second_model
                gc.collect()
                torch.cuda.empty_cache()
            if second_tensor_sha != tensor_sha or second_next_token != next_token:
                raise ValueError("two independent Replay checkpoint reloads disagree")
            expected_next_token = second_next_token
            second_reload_facts = second_facts
        else:
            second_reload_facts = None
        if tensor_sha != expected_tensor_sha or next_token != expected_next_token:
            raise ValueError("Replay independent checkpoint reload changed LoRA tensors or fixed-Prompt next token")
        base_identity = inference._weight_identity(archive_base, {
            "model_id": model_config["base_model"]["model_id"],
            "model_revision": model_config["base_model"]["revision"],
        })
        if base_identity.get("observed_weight_hash") != config["parent"]["base_weight_sha256"]:
            raise ValueError("Replay independent reload Base identity differs")
        result = {
            "status": "passed",
            "run_id": run_id,
            "replay_step": replay_step,
            "global_optimizer_step": int(checked["record"]["global_optimizer_step"]),
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
            "adapter_tensor_sha256": tensor_sha,
            "next_token_id": next_token,
            "probe_sample_id": probe["sample_id"],
            "probe_prompt_tokens": len(prompt_ids),
            "probe_prompt_sha256": canonical_sha256(prompt_ids),
            "base_weight_sha256": base_identity.get("observed_weight_hash"),
            "model_facts": model_facts,
            "second_independent_model_facts": second_reload_facts,
            "optimizer_state_sha256": sha256_file(checkpoint / "optimizer.pt"),
            "scheduler_state_sha256": sha256_file(checkpoint / "scheduler.pt"),
            "rng_state_sha256": sha256_file(checkpoint / "rng_state.pth"),
            "created_at_utc": _now(),
        }
        check_path.parent.mkdir(parents=True, exist_ok=True)
        _write_exclusive(check_path, result)
        checked["record"]["_manifest_file_sha256"] = result["checkpoint_manifest_sha256"]
        return result
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_candidate_gold_and_rows(
    config: dict[str, Any],
    candidate_ids: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted = {str(row["sample_id"]): row for row in candidate_ids}
    profiles: dict[str, dict[str, list[str]]] = {}
    history_counts: dict[str, int] = {}
    for source in ("train", "dev"):
        path = _project_path(config["data"][f"{source}_normalized"])
        expected_sha = config["data"][f"{source}_normalized_sha256"]
        if sha256_file(path) != expected_sha:
            raise ValueError(f"candidate Gold source changed after prediction freeze: {source}")
        source_ids = {sample_id for sample_id, entry in wanted.items() if entry["source"] == source}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = row.get("sample_id")
                if sample_id not in source_ids:
                    continue
                if row.get("conversation_id") != wanted[sample_id]["conversation_id"]:
                    raise ValueError(f"{sample_id}: frozen candidate Gold conversation identity differs")
                profile = row.get("target", {}).get("user_profile")
                profile_label_keys(profile)
                profiles[sample_id] = {dim: list(profile[dim]) for dim in PROFILE_DIMS}
                history_counts[sample_id] = len(row.get("history", []))
    if set(profiles) != set(wanted):
        raise ValueError("candidate Gold reread does not exactly cover the frozen candidate IDs")

    indexes = {}
    try:
        for source in ("train", "dev"):
            indexes[source] = JsonlById(_project_path(config["data"][f"{source}_encoded"]))
        result = []
        for item in candidate_ids:
            source = str(item["source"])
            row = indexes[source].get(item["sample_id"])
            if row.get("conversation_id") != item["conversation_id"]:
                raise ValueError(f"{item['sample_id']}: encoded candidate source differs")
            result.append({
                **item,
                "prompt_length": int(row["prompt_length"]),
                "history_message_count": history_counts[item["sample_id"]],
                "gold_profile": profiles[item["sample_id"]],
                "gold_nonempty": bool(profile_label_keys(profiles[item["sample_id"]])),
            })
    finally:
        for index in indexes.values():
            index.close()
    return result


def _freeze_hard_samples(
    config: dict[str, Any],
    run_root: Path,
    candidate_ids: list[dict[str, Any]],
    candidate_audit: dict[str, Any],
    prediction_manifest: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prediction_path = run_root / "mining" / "candidate_predictions.jsonl"
    prediction_manifest_path = _prediction_manifest_path(prediction_path)
    expected_ids = [row["sample_id"] for row in candidate_ids]
    predictions = _verify_prediction_file(
        prediction_path,
        prediction_manifest_path,
        expected_ids,
        {
            "run_id": candidate_audit["run_id"],
            "stage": "candidate-parent-greedy-v1",
            "parent_checkpoint_manifest_sha256": config["parent"]["checkpoint_manifest_sha256"],
            "candidate_ids_sha256": canonical_sha256(expected_ids),
        },
    )
    prediction_index = {row["id"]: row for row in predictions}
    if len(prediction_index) != len(predictions):
        raise ValueError("candidate prediction IDs contain duplicates")
    # Gold is reread only after the candidate prediction file is frozen and hashed.
    enriched = _load_candidate_gold_and_rows(config, candidate_ids)
    frequency = candidate_audit["selection"]["train_dev_label_frequency"]
    hard_rows, stats = score_candidate_difficulty(enriched, prediction_index, frequency)
    hard_path = run_root / "mining" / "hard_samples.jsonl"
    hard_audit_path = run_root / "mining" / "hard_sample_audit.json"
    hard_payload = [{**row, "source_data_sha256": candidate_audit["source_data_sha256"]} for row in hard_rows]
    hard_sha = canonical_sha256(hard_payload)
    hard_audit = {
        "status": "gold_scored_after_prediction_freeze",
        "version": "E03-A8-REPLAY-hardness-v1",
        "candidate_ids_sha256": canonical_sha256(expected_ids),
        "candidate_prediction_sha256": sha256_file(prediction_path),
        "candidate_prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "source_data_sha256": candidate_audit["source_data_sha256"],
        "algorithm_sha256": sha256_file(ROOT / "src/data/e03_a8_replay.py"),
        "hard_samples_sha256": hard_sha,
        "statistics": stats,
        "created_at_utc": _now(),
    }
    if hard_path.exists() or hard_audit_path.exists():
        if not resume or not hard_path.is_file() or not hard_audit_path.is_file():
            raise FileExistsError("hard-sample files already exist and cannot be overwritten")
        if sha256_file(hard_path) != _read_json(hard_audit_path).get("hard_samples_file_sha256"):
            raise ValueError("existing hard-sample file hash differs")
        existing_rows = _read_jsonl(hard_path)
        existing_audit = _read_json(hard_audit_path)
        if (
            existing_rows != hard_payload
            or existing_audit.get("hard_samples_sha256") != hard_sha
            or existing_audit.get("candidate_prediction_sha256") != hard_audit["candidate_prediction_sha256"]
        ):
            raise ValueError("existing frozen hard samples differ from predictions or Train/Dev Gold")
        hard_audit = existing_audit
        hard_rows = existing_rows
    else:
        _write_jsonl_exclusive(hard_path, hard_payload)
        hard_audit["hard_samples_file_sha256"] = sha256_file(hard_path)
        _write_exclusive(hard_audit_path, hard_audit)
    # The schedule consumes only compact metadata; full source histories and Gold
    # texts are not written to the Git working tree.
    return hard_rows, enriched, {
        "group_counts": stats["group_counts"],
        "candidate_count": stats["candidate_count"],
        "valid_nonempty_prediction_count": stats["valid_nonempty_prediction_count"],
        "empty_gold_false_positive_label_count": stats["empty_gold_false_positive_label_count"],
        "distinct_conversations": stats["distinct_conversations"],
        "top_label_errors": stats["top_label_errors"],
        "hard_samples_sha256": hard_sha,
        "hard_samples_file_sha256": sha256_file(hard_path),
        "candidate_ids_sha256": canonical_sha256(expected_ids),
        "candidate_predictions_sha256": sha256_file(prediction_path),
        "prediction_manifest": prediction_manifest,
        "hard_audit_path": str(hard_audit_path.relative_to(ROOT)),
    }


def _freeze_schedule(
    config: dict[str, Any],
    run_root: Path,
    hard_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    mining_facts: dict[str, Any],
    *,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_counts = mining_facts["group_counts"]
    has_fn = int(group_counts.get("miss", 0)) + int(group_counts.get("mixed", 0)) > 0
    has_fp = int(group_counts.get("extra", 0)) + int(group_counts.get("mixed", 0)) > 0
    if not has_fn or not has_fp:
        raise ValueError("mining_insufficient: the frozen candidate set does not contain both missed and extra labels")
    schedule, schedule_audit = build_replay_schedule(
        hard_rows,
        candidates,
        seed=int(config["seed"]),
        pilot_steps=int(config["schedule"]["pilot_steps"]),
        extension_steps=int(config["schedule"]["extension_steps"]),
        accumulation=int(config["schedule"]["gradient_accumulation_steps"]),
        dev_fraction_cap=float(config["schedule"]["dev_exposure_fraction_max"]),
        pilot_repeat_fraction=float(config["schedule"]["pilot_hard_repeat_fraction_min"]),
    )
    path = run_root / "mining" / "replay_schedule.jsonl"
    audit_path = run_root / "mining" / "replay_schedule_audit.json"
    schedule_sha = canonical_sha256(schedule)
    audit = {
        "status": "frozen",
        "version": "E03-A8-REPLAY-schedule-v1",
        "candidate_ids_sha256": mining_facts["candidate_ids_sha256"],
        "hard_samples_sha256": mining_facts["hard_samples_sha256"],
        "train_encoded_sha256": config["data"]["train_encoded_sha256"],
        "dev_encoded_sha256": config["data"]["dev_encoded_sha256"],
        "seed": config["seed"],
        "schedule_sha256": schedule_sha,
        "schedule_file_sha256": None,
        "audit": schedule_audit,
        "created_at_utc": _now(),
    }
    if path.exists() or audit_path.exists():
        if not resume or not path.is_file() or not audit_path.is_file():
            raise FileExistsError("Replay schedule already exists and cannot be overwritten")
        prior_rows = _read_jsonl(path)
        prior_audit = _read_json(audit_path)
        if prior_rows != schedule or prior_audit.get("schedule_sha256") != schedule_sha:
            raise ValueError("existing frozen Replay schedule differs from the current source/algorithm")
        if prior_audit.get("schedule_file_sha256") != sha256_file(path):
            raise ValueError("existing Replay schedule file hash differs")
        audit = prior_audit
        schedule = prior_rows
    else:
        _write_jsonl_exclusive(path, schedule)
        audit["schedule_file_sha256"] = sha256_file(path)
        _write_exclusive(audit_path, audit)
    mining_facts = {
        **mining_facts,
        "hard_samples_sha256": mining_facts["hard_samples_sha256"],
        "schedule_sha256": schedule_sha,
        "schedule_file_sha256": sha256_file(path),
    }
    return schedule, {**audit, "schedule_audit": schedule_audit, "path": str(path.relative_to(ROOT)), "file_sha256": sha256_file(path)}


def _selection_record(
    run_root: Path,
    *,
    run_id: str,
    candidate: str,
    reason: str,
    parent_identity: dict[str, Any],
    checkpoint: Path | None,
    checkpoint_sha256: str | None,
    gates: dict[str, Any],
) -> dict[str, Any]:
    path = run_root / "selection.json"
    record = {
        "status": "complete",
        "version": "E03-A8-REPLAY-selection-v1",
        "run_id": run_id,
        "selected_candidate": candidate,
        "decision_reason": reason,
        "selected_checkpoint_path": str(checkpoint.relative_to(ROOT)) if checkpoint else str(parent_identity["checkpoint_path"].relative_to(ROOT)),
        "selected_checkpoint_manifest_sha256": checkpoint_sha256 or parent_identity["checkpoint_manifest_sha256"],
        "parent_checkpoint_manifest_sha256": parent_identity["checkpoint_manifest_sha256"],
        "parent_adapter_weight_sha256": parent_identity["adapter_identity"]["weight_sha256"],
        "gates": gates,
        "official_submission_entry_changed": False,
        "created_at_utc": _now(),
    }
    if path.exists():
        existing = _read_json(path)
        existing.pop("created_at_utc", None)
        comparable = dict(record)
        comparable.pop("created_at_utc", None)
        if existing != comparable:
            raise ValueError("existing immutable selection.json differs from the final decision")
        return _read_json(path)
    _write_exclusive(path, record)
    return record


def _render_experiment_report(report: dict[str, Any]) -> str:
    mining = report.get("mining") or {}
    schedule = report.get("schedule") or {}
    validation = report.get("validation") or {}

    def fmt(value: Any) -> str:
        if value is None:
            return "not_run"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# E03-A8 Replay 实验结果",
        "",
        f"- Run：`{report['run_id']}`；结果范围：`local_proxy`。",
        f"- 最终选择：`{report['selection'].get('selected_candidate', report['selection'].get('fallback_candidate', 'P4504'))}`；原因：{report['selection']['decision_reason']}。",
        f"- P4504 Checkpoint Manifest SHA-256：`{report['parent']['checkpoint_manifest_sha256']}`。",
        f"- Replay Schedule SHA-256：`{schedule.get('schedule_sha256', 'not_run')}`。",
        "",
        "## 阶段与检查点",
        "",
        "| 阶段 | 状态 | 全局优化步 | Checkpoint SHA-256 | 累计训练秒 | GPU step P50/P90 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for step in (128, 256, 384):
        item = report.get("training_nodes", {}).get(str(step))
        if not item:
            lines.append(f"| Replay{step} | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` |")
            continue
        performance = item.get("gpu_step_seconds", {})
        lines.append(
            f"| Replay{step} | `{item.get('status', 'unknown')}` | {item.get('global_optimizer_step', 'not_run')} | "
            f"`{item.get('checkpoint_manifest_sha256', 'not_run')}` | {fmt(item.get('cumulative_replay_training_seconds'))} | "
            f"{fmt(performance.get('p50'))} / {fmt(performance.get('p90'))} |"
        )
    lines += [
        "",
        "## 困难样本与曝光",
        "",
        f"- 候选：{mining.get('candidate_count', 'not_run')} 条；错误类型：`{json.dumps(mining.get('group_counts', {}), ensure_ascii=False, sort_keys=True)}`。",
        f"- 空 Gold 画像上的误报标签：{mining.get('empty_gold_false_positive_label_count', 'not_run')}；有效非空预测：{mining.get('valid_nonempty_prediction_count', 'not_run')}。",
        f"- 重复曝光：Pilot 第二次出现 {schedule.get('pilot_second_exposures', 'not_run')} / {schedule.get('pilot_hard_exposures', 'not_run')} 个困难曝光；全程最大单样本曝光 {schedule.get('max_sample_exposures', 'not_run')}。",
        f"- 来源曝光：`{json.dumps(schedule.get('source_exposure_counts', {}), ensure_ascii=False, sort_keys=True)}`；桶曝光：`{json.dumps(schedule.get('bucket_exposure_counts', {}), ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 固定会话分组配对结果",
        "",
        "| 集合 | 模型 | 预测点 | TP / FP / FN | Micro-F1 | Precision | Recall | Macro-F1 | 严格解析率 | EOS 率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, stage_name in (("val_gate", "replay128"), ("val_confirm", "replay384")):
        parent = validation.get("parent_metrics", {}).get(split_name)
        candidate = validation.get("candidate_metrics", {}).get(stage_name)
        for label, item in (("P4504", parent), (stage_name, candidate)):
            if not item:
                lines.append(f"| {split_name} | {label} | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` |")
                continue
            profile = item["profile_33_label"]
            lines.append(
                f"| {split_name} | {label} | {item['fixed_denominator_count']} | "
                f"{profile['overall_33_label_tp']} / {profile['overall_33_label_fp']} / {profile['overall_33_label_fn']} | "
                f"{fmt(profile['overall_33_label_micro_f1'])} | {fmt(profile['overall_33_label_micro_precision'])} | "
                f"{fmt(profile['overall_33_label_micro_recall'])} | {fmt(profile['overall_33_label_macro_f1'])} | "
                f"{fmt(item['strict_parse_rate'])} | {fmt(item['eos_completion_rate'])} |"
            )
    lines += ["", "评分仅是项目内本地代理结果，不等同于官方赛事成绩；val_gate 与 val_confirm 按会话互斥，历史 A2 曝光会话单独冻结。", ""]
    return "\n".join(lines)


def _verify_backup_bundle(
    config: dict[str, Any],
    model_config: dict[str, Any],
    parent_identity: dict[str, Any],
    archive_checkpoint: Path,
    local_checkpoint: Path,
    archive_base: Path,
) -> dict[str, Any]:
    parent_files = parent_identity["checkpoint_files"] + [{
        "path": "checkpoint_manifest.json",
        "size_bytes": (parent_identity["checkpoint_path"] / "checkpoint_manifest.json").stat().st_size,
        "sha256": parent_identity["checkpoint_manifest_sha256"],
    }]
    parent_files.sort(key=lambda row: row["path"])
    _verify_records(parent_identity["checkpoint_path"], parent_files)
    _verify_records(archive_checkpoint, parent_files)
    _verify_records(local_checkpoint, parent_files)
    _assert_distinct_copies(parent_identity["checkpoint_path"], [archive_checkpoint, local_checkpoint], parent_files)
    base_destination = archive_base.parent
    base_manifest = _base_archive_manifest(base_destination)
    base_identity = inference._weight_identity(archive_base, {
        "model_id": model_config["base_model"]["model_id"],
        "model_revision": model_config["base_model"]["revision"],
    })
    if base_identity.get("observed_weight_hash") != config["parent"]["base_weight_sha256"]:
        raise ValueError("final Base archive identity differs from the frozen P4504 Base")
    for path in (archive_checkpoint, local_checkpoint, archive_base):
        for item in [path, *path.rglob("*")]:
            if item.is_symlink():
                raise ValueError(f"verified backup contains a symbolic link: {item}")
            if item.is_file() and item.stat().st_mode & 0o222:
                raise ValueError(f"verified backup contains a writable file: {item}")
    return {
        "status": "passed",
        "parent_checkpoint_files_sha256": canonical_sha256(parent_files),
        "parent_checkpoint_manifest_sha256": parent_identity["checkpoint_manifest_sha256"],
        "external_checkpoint_path": str(archive_checkpoint),
        "local_checkpoint_path": str(local_checkpoint),
        "base_archive_path": str(archive_base),
        "base_archive_manifest_sha256": sha256_file(base_destination / "archive_manifest.json"),
        "base_archive_files_sha256": base_manifest["files_sha256"],
        "base_weight_sha256": base_identity["observed_weight_hash"],
        "copies_are_independent": True,
        "archives_read_only": True,
        "created_at_utc": _now(),
    }


def _evaluate_replay_checkpoint(
    *,
    config: dict[str, Any],
    model_config: dict[str, Any],
    schema: dict[str, Any],
    split_name: str,
    replay_step: int,
    checkpoint: Path,
    checkpoint_facts: dict[str, Any],
    splits: dict[str, Any],
    val_inputs: list[dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
    references: dict[str, dict[str, Any]],
    parent_metrics: dict[str, Any],
    run_root: Path,
    archive_base: Path,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_sha = sha256_file(checkpoint / "checkpoint_manifest.json")
    prediction_manifest = _generate_val_predictions(
        config,
        model_config,
        schema,
        split_name,
        splits,
        val_inputs,
        id_map,
        run_root,
        archive_base,
        checkpoint,
        run_id=run_id,
        checkpoint_manifest_sha256=checkpoint_sha,
        replay_step=replay_step,
    )
    cohort = splits["cohorts"][split_name]
    prediction_path = _project_path(config["outputs"]["evaluation_root"]) / run_id / f"{split_name}-replay-{replay_step:04d}" / "predictions.jsonl"
    rows = _verify_prediction_file(
        prediction_path,
        _prediction_manifest_path(prediction_path),
        cohort["ids"],
        {
            "run_id": run_id,
            "stage": f"{split_name}-replay-{replay_step:04d}",
            "checkpoint_manifest_sha256": checkpoint_sha,
            "cohort_ids_sha256": cohort["ids_sha256"],
            "val_split_sha256": sha256_file(run_root / "validation" / "val_splits.json"),
        },
    )
    predictions = {row["id"]: row for row in rows}
    if len(predictions) != len(rows):
        raise ValueError(f"Replay{replay_step} Val predictions contain duplicate IDs")
    metrics = score_profile_predictions(cohort, references, predictions, _schema_cache["enums"])
    validation_config = config["validation"]
    parent_item = parent_metrics[split_name]
    engineering_passed = (
        checkpoint_facts["record"].get("status") == "complete"
        and checkpoint_facts["record"].get("replay_step") == replay_step
        and checkpoint_facts.get("independent_reload_status", "passed") == "passed"
        and prediction_manifest.get("references_read") is False
    )
    gates = gate_reasons(
        parent_item,
        metrics,
        minimum_micro_f1_delta=float(validation_config["minimum_micro_f1_delta"]),
        maximum_precision_drop=float(validation_config["maximum_precision_drop"]),
        maximum_dimension_f1_drop=float(validation_config["maximum_dimension_f1_drop"]),
        minimum_strict_parse_rate=float(validation_config["minimum_strict_parse_rate"]),
        minimum_eos_rate=float(validation_config["minimum_eos_rate"]),
        maximum_new_fp_ratio=float(validation_config["maximum_new_fp_ratio"]),
        engineering_passed=engineering_passed,
    )
    result = {
        "status": "passed" if gates["passed"] else "failed",
        "run_id": run_id,
        "replay_step": replay_step,
        "global_optimizer_step": int(config["parent"]["global_optimizer_step"]) + replay_step,
        "cohort": split_name,
        "cohort_ids_sha256": cohort["ids_sha256"],
        "checkpoint_manifest_sha256": checkpoint_sha,
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_manifest_sha256": sha256_file(_prediction_manifest_path(prediction_path)),
        "metrics": metrics,
        "parent_metrics": parent_item,
        "gates": gates,
        "created_at_utc": _now(),
    }
    stage_filename = "pilot_eval.json" if replay_step == 128 and split_name == "val_gate" else (
        "final_eval.json" if replay_step == 384 and split_name == "val_confirm" else f"replay-{replay_step:04d}-{split_name}.json"
    )
    stage_path = run_root / "validation" / stage_filename
    if stage_path.exists():
        old = _read_json(stage_path)
        comparable = dict(result)
        old.pop("created_at_utc", None)
        comparable.pop("created_at_utc", None)
        if old != comparable:
            raise ValueError(f"existing Replay{replay_step} evaluation differs from frozen inputs")
        result = _read_json(stage_path)
    else:
        _write_exclusive(stage_path, result)
    return result, prediction_manifest


def _run_all(config_path: Path, run_id: str) -> dict[str, Any]:
    config, config_path = _read_config(config_path)
    model_config = _base_config(config)
    parent_identity = _validate_parent(config, model_config)
    run_root, manifest = _initialize_run(config, config_path, model_config, run_id)
    report: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "quality_scope": "local_proxy",
        "parent": {
            "run_id": config["parent"]["run_id"],
            "checkpoint_path": str(parent_identity["checkpoint_path"].relative_to(ROOT)),
            "checkpoint_manifest_sha256": parent_identity["checkpoint_manifest_sha256"],
            "adapter_weight_sha256": parent_identity["adapter_identity"]["weight_sha256"],
            "adapter_tensor_sha256": parent_identity["run_manifest"].get("final_adapter_tensor_sha256"),
            "base_weight_sha256": config["parent"]["base_weight_sha256"],
        },
        "source_data": {},
        "baseline_backup_gate": None,
        "mining": None,
        "schedule": None,
        "validation": {"parent_metrics": {}, "candidate_metrics": {}, "gates": {}},
        "training_nodes": {},
        "backup_integrity": None,
    }

    def persist(final_status: str, selection_candidate: str, reason: str, gates: dict[str, Any]) -> dict[str, Any]:
        nonlocal manifest, selected_checkpoint, selected_checkpoint_sha
        if archive_checkpoint is None or local_checkpoint is None or archive_base is None:
            backup = None
        else:
            try:
                backup = _verify_backup_bundle(
                    config,
                    model_config,
                    parent_identity,
                    archive_checkpoint,
                    local_checkpoint,
                    archive_base,
                )
            except Exception as backup_error:
                backup = {"status": "failed", "error": f"{type(backup_error).__name__}: {backup_error}"}
        report["backup_integrity"] = backup
        if backup is None or backup.get("status") != "passed":
            selection_candidate = "P4504"
            selected_checkpoint = None
            selected_checkpoint_sha = None
            reason = "backup_integrity_failed; retain the original P4504 checkpoint"
        if final_status == "interrupted":
            selection = {
                "status": "pending_after_recoverable_execution_error",
                "run_id": run_id,
                "fallback_candidate": "P4504",
                "decision_reason": reason,
                "selection_file_written": False,
            }
        else:
            selection = _selection_record(
                run_root,
                run_id=run_id,
                candidate=selection_candidate,
                reason=reason,
                parent_identity=parent_identity,
                checkpoint=selected_checkpoint,
                checkpoint_sha256=selected_checkpoint_sha,
                gates=gates,
            )
        report["selection"] = selection
        report["status"] = final_status
        report["execution_error"] = error_record
        report["runtime"] = {"python": sys.version, "packages": _package_versions(), "git": _git_facts()}
        report_path = run_root / "report.json"
        if report_path.exists():
            existing = _read_json(report_path)
            if existing != report:
                _write_atomic(report_path, report)
        else:
            _write_exclusive(report_path, report)
        markdown = _render_experiment_report(report)
        run_report = run_root / "report.md"
        if run_report.exists():
            if run_report.read_text(encoding="utf-8") != markdown:
                temporary = run_report.with_name(f"{run_report.name}.tmp-{os.getpid()}-{time.time_ns()}")
                temporary.write_text(markdown, encoding="utf-8")
                os.replace(temporary, run_report)
        else:
            run_report.write_text(markdown, encoding="utf-8")
        manifest = _save_run_manifest(
            run_root,
            manifest,
            status=final_status,
            stage="complete" if final_status in {"complete", "mining_insufficient", "validation_inadequate"} else final_status,
            final_selection=selection,
            backup_integrity_status=backup.get("status") if backup else "not_verified",
            report_path=str(report_path.relative_to(ROOT)),
            report_sha256=sha256_file(report_path),
            completed_at_utc=_now(),
        )
        report["run_manifest_sha256"] = sha256_file(run_root / "manifest.json")
        if report_path.is_file():
            # Report JSON does not embed its own or the mutable Run manifest hash.
            pass
        return report

    archive_checkpoint: Path | None = None
    local_checkpoint: Path | None = None
    archive_base: Path | None = None
    selected_checkpoint: Path | None = None
    selected_checkpoint_sha: str | None = None
    error_record: dict[str, Any] | None = None
    gates_for_selection: dict[str, Any] = {}
    try:
        archive_checkpoint, local_checkpoint, archive_base, backup_gate = _ensure_baseline_backup(
            config, model_config, parent_identity, run_root, run_id=run_id
        )
        if backup_gate.get("baseline_backup_gate") != "passed":
            raise RuntimeError("P4504 baseline backup gate did not pass")
        report["baseline_backup_gate"] = backup_gate
        manifest = _save_run_manifest(
            run_root,
            manifest,
            baseline_backup_gate="passed",
            stage="baseline_backup_passed",
            archive_checkpoint_path=str(archive_checkpoint),
            local_checkpoint_path=str(local_checkpoint),
            base_archive_path=str(archive_base),
        )

        schema, enums = _read_schema(config, model_config)
        candidate_ids, candidate_audit = _candidate_artifacts(config, model_config, run_root, resume=True)
        candidate_ids_path = run_root / "mining" / "candidate_ids.jsonl"
        if _read_jsonl(candidate_ids_path) != candidate_ids:
            raise ValueError("candidate ID file differs from its deterministic selection")
        if len(candidate_ids) != int(config["candidate_selection"]["count"]):
            raise ValueError("candidate ID set is not the frozen 3,000 rows")
        source_facts = candidate_audit["source_facts"]
        report["source_data"] = source_facts
        report["candidate_selection"] = candidate_audit["selection"]
        manifest = _save_run_manifest(
            run_root,
            manifest,
            mining_status="candidate_ids_frozen",
            source_data_facts=source_facts,
            candidate_ids_sha256=candidate_audit["candidate_ids_sha256"],
            stage="candidate_ids_frozen",
        )

        candidate_predictions = _generate_candidate_predictions(
            config,
            model_config,
            schema,
            candidate_ids,
            run_root,
            archive_base,
            archive_checkpoint,
            run_id=run_id,
        )
        manifest = _save_run_manifest(
            run_root,
            manifest,
            mining_status="candidate_predictions_frozen",
            candidate_predictions_sha256=candidate_predictions["prediction_sha256"],
            stage="candidate_predictions_frozen",
        )
        hard_rows, enriched_candidates, mining_facts = _freeze_hard_samples(
            config,
            run_root,
            candidate_ids,
            candidate_audit,
            candidate_predictions,
            resume=True,
        )
        report["mining"] = mining_facts
        manifest = _save_run_manifest(
            run_root,
            manifest,
            mining_status="hard_samples_frozen",
            mining_facts=mining_facts,
            stage="hard_samples_frozen",
        )
        try:
            schedule, schedule_facts = _freeze_schedule(
                config,
                run_root,
                hard_rows,
                enriched_candidates,
                mining_facts,
                resume=True,
            )
        except ValueError as exc:
            if "mining_insufficient" not in str(exc):
                raise
            report["mining"]["schedule_stop_reason"] = str(exc)
            report["schedule"] = None
            manifest = _save_run_manifest(run_root, manifest, mining_status="mining_insufficient", schedule_status="not_run")
            selected_checkpoint = None
            selected_checkpoint_sha = None
            del enriched_candidates
            return persist("mining_insufficient", "P4504", str(exc), {})
        del enriched_candidates
        report["schedule"] = {
            "schedule_sha256": schedule_facts["schedule_sha256"],
            "schedule_file_sha256": schedule_facts["file_sha256"],
            **schedule_facts["schedule_audit"],
        }
        manifest = _save_run_manifest(
            run_root,
            manifest,
            schedule_status="frozen",
            schedule_sha256=schedule_facts["schedule_sha256"],
            schedule_file_sha256=schedule_facts["file_sha256"],
            stage="schedule_frozen",
        )

        splits, val_inputs, id_map, val_facts = _load_val_facts(config, run_root)
        references, parent_val_facts = _freeze_parent_metrics(config, run_root, splits, val_facts, enums)
        report["validation"].update({
            "source_facts": {key: value for key, value in val_facts.items() if key not in {"parent_predictions"}},
            "split_facts": serialize_val_splits(splits),
            "parent_metrics": parent_val_facts["parent_metrics"],
            "cohort_readiness": parent_val_facts["cohort_readiness"],
        })
        if parent_val_facts["status"] != "passed":
            report["validation"]["stop_reason"] = "fixed_val_cohort_gold_adequacy_failed"
            manifest = _save_run_manifest(run_root, manifest, pilot_status="not_run", extension_status="not_run")
            return persist("validation_inadequate", "P4504", "fixed Val cohort Gold adequacy failed", {})

        training_nodes = report["training_nodes"]
        replay128_path = _replay_checkpoint_path(run_root, 128)
        summary128, checked128 = _run_training_segment(
            config=config,
            model_config=model_config,
            run_root=run_root,
            run_id=run_id,
            start_step=0,
            stop_step=128,
            previous_checkpoint=local_checkpoint,
            previous_is_parent=True,
            local_parent_checkpoint=local_checkpoint,
            parent_adapter_tensor_sha256=str(parent_identity["run_manifest"].get("final_adapter_tensor_sha256")),
            archive_base=archive_base,
            schedule=schedule,
            schedule_sha256=schedule_facts["schedule_sha256"],
            data_facts=source_facts,
            mining_facts=mining_facts,
            candidate_ids=candidate_ids,
            manifest=manifest,
        )
        summary128["independent_reload_status"] = "passed"
        training_nodes["128"] = summary128
        manifest = _save_run_manifest(run_root, manifest, pilot_status="training_passed", stage="replay_0128_passed")
        pilot_eval, pilot_predictions = _evaluate_replay_checkpoint(
            config=config,
            model_config=model_config,
            schema=schema,
            split_name="val_gate",
            replay_step=128,
            checkpoint=replay128_path,
            checkpoint_facts={**checked128, "independent_reload_status": "passed"},
            splits=splits,
            val_inputs=val_inputs,
            id_map=id_map,
            references=references,
            parent_metrics=parent_val_facts["parent_metrics"],
            run_root=run_root,
            archive_base=archive_base,
            run_id=run_id,
        )
        report["validation"]["candidate_metrics"]["replay128"] = pilot_eval["metrics"]
        report["validation"]["gates"]["pilot"] = pilot_eval["gates"]
        report["validation"]["prediction_manifests"] = {"replay128": pilot_predictions}
        manifest = _save_run_manifest(
            run_root,
            manifest,
            pilot_status="passed" if pilot_eval["gates"]["passed"] else "failed",
            pilot_gate=pilot_eval["gates"],
            stage="pilot_gate_passed" if pilot_eval["gates"]["passed"] else "pilot_gate_failed",
        )
        if not pilot_eval["gates"]["passed"]:
            gates_for_selection = {"pilot": pilot_eval["gates"]}
            return persist("complete", "P4504", "pilot_gate_failed:" + ",".join(pilot_eval["gates"]["reasons"]), gates_for_selection)

        replay256_path = _replay_checkpoint_path(run_root, 256)
        summary256, checked256 = _run_training_segment(
            config=config,
            model_config=model_config,
            run_root=run_root,
            run_id=run_id,
            start_step=128,
            stop_step=256,
            previous_checkpoint=replay128_path,
            previous_is_parent=False,
            local_parent_checkpoint=local_checkpoint,
            parent_adapter_tensor_sha256=str(parent_identity["run_manifest"].get("final_adapter_tensor_sha256")),
            archive_base=archive_base,
            schedule=schedule,
            schedule_sha256=schedule_facts["schedule_sha256"],
            data_facts=source_facts,
            mining_facts=mining_facts,
            candidate_ids=candidate_ids,
            manifest=manifest,
        )
        training_nodes["256"] = summary256
        manifest = _save_run_manifest(run_root, manifest, extension_status="step_256_passed", stage="replay_0256_passed")

        replay384_path = _replay_checkpoint_path(run_root, 384)
        summary384, checked384 = _run_training_segment(
            config=config,
            model_config=model_config,
            run_root=run_root,
            run_id=run_id,
            start_step=256,
            stop_step=384,
            previous_checkpoint=replay256_path,
            previous_is_parent=False,
            local_parent_checkpoint=local_checkpoint,
            parent_adapter_tensor_sha256=str(parent_identity["run_manifest"].get("final_adapter_tensor_sha256")),
            archive_base=archive_base,
            schedule=schedule,
            schedule_sha256=schedule_facts["schedule_sha256"],
            data_facts=source_facts,
            mining_facts=mining_facts,
            candidate_ids=candidate_ids,
            manifest=manifest,
        )
        training_nodes["384"] = summary384
        manifest = _save_run_manifest(run_root, manifest, extension_status="step_384_passed", stage="replay_0384_passed")
        confirm_eval, confirm_predictions = _evaluate_replay_checkpoint(
            config=config,
            model_config=model_config,
            schema=schema,
            split_name="val_confirm",
            replay_step=384,
            checkpoint=replay384_path,
            checkpoint_facts={**checked384, "independent_reload_status": "passed"},
            splits=splits,
            val_inputs=val_inputs,
            id_map=id_map,
            references=references,
            parent_metrics=parent_val_facts["parent_metrics"],
            run_root=run_root,
            archive_base=archive_base,
            run_id=run_id,
        )
        report["validation"]["candidate_metrics"]["replay384"] = confirm_eval["metrics"]
        report["validation"]["gates"]["confirm"] = confirm_eval["gates"]
        report["validation"]["prediction_manifests"]["replay384"] = confirm_predictions
        gates_for_selection = {"pilot": pilot_eval["gates"], "confirm": confirm_eval["gates"]}
        if confirm_eval["gates"]["passed"]:
            selected_checkpoint = replay384_path
            selected_checkpoint_sha = sha256_file(replay384_path / "checkpoint_manifest.json")
            selection_candidate = "Replay384"
            reason = "pilot and val_confirm quality gates passed"
        else:
            selected_checkpoint = None
            selected_checkpoint_sha = None
            selection_candidate = "P4504"
            reason = "confirm_gate_failed:" + ",".join(confirm_eval["gates"]["reasons"])
        return persist("complete", selection_candidate, reason, gates_for_selection)
    except BaseException as exc:
        error_record = {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "type": type(exc).__name__, "message": str(exc), "created_at_utc": _now()}
        report["execution_error"] = error_record
        try:
            return persist("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "P4504", f"execution_failed:{type(exc).__name__}:{exc}", gates_for_selection)
        finally:
            if not isinstance(exc, KeyboardInterrupt):
                raise


def _restore_check(config_path: Path, manifest_path: Path) -> dict[str, Any]:
    import torch

    config, _ = _read_config(config_path)
    model_config = _base_config(config)
    manifest_path = manifest_path.expanduser().resolve()
    checkpoint = manifest_path.parent
    if not manifest_path.is_file() or manifest_path.name != "checkpoint_manifest.json":
        raise ValueError("restore-check requires a checkpoint_manifest.json path")
    record = _read_json(manifest_path)
    if record.get("status") != "complete":
        raise ValueError("restore-check checkpoint manifest is not complete")
    expected_files = record.get("files")
    if record.get("files_sha256") != canonical_sha256(expected_files):
        raise ValueError("restore-check embedded file manifest hash is invalid")
    expected_files = list(expected_files) + [{
        "path": "checkpoint_manifest.json",
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }]
    expected_files.sort(key=lambda row: row["path"])
    _verify_records(checkpoint, expected_files)
    parent_archive_identity_path = checkpoint.parent / "identity.json"
    parent_archive_identity = _read_json(parent_archive_identity_path) if parent_archive_identity_path.is_file() else None
    if record.get("run_id") == config["parent"]["run_id"]:
        if parent_archive_identity is None:
            raise ValueError("P4504 restore-check requires the archived identity.json")
        base_path_value = parent_archive_identity.get("base_archive_path")
        parent_identity = _validate_parent(config, model_config)
        result = _verify_restored_models(
            config,
            model_config,
            parent_identity,
            Path(base_path_value).expanduser().resolve(),
            checkpoint,
            run_id="restore-check",
        )
        result["checkpoint_manifest_path"] = str(manifest_path)
        result["checkpoint_manifest_sha256"] = sha256_file(manifest_path)
        return result
    base_path_value = record.get("base_archive_path") or (parent_archive_identity or {}).get("base_archive_path")
    base_path = Path(base_path_value or "").expanduser().resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"checkpoint Base archive is missing: {base_path}")
    base_identity = inference._weight_identity(base_path, {
        "model_id": model_config["base_model"]["model_id"],
        "model_revision": model_config["base_model"]["revision"],
    })
    if base_identity.get("observed_weight_hash") != config["parent"]["base_weight_sha256"]:
        raise ValueError("restore-check Base weight identity differs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("blocked_gpu_unavailable_or_bf16_unsupported: restore-check requires CUDA BF16")
    model = None
    try:
        model, torch, model_facts = _load_train_model(
            model_config,
            base_path,
            adapter_path=checkpoint,
            autocast_adapter_dtype=True,
        )
        model.eval()
        actual_tensor_sha = _lora_tensor_sha256(model, torch)
        expected_tensor_sha = record.get("adapter_tensor_sha256")
        if expected_tensor_sha and actual_tensor_sha != expected_tensor_sha:
            raise ValueError("restore-check Adapter tensor SHA differs")
        probes = []
        for sample_id, prompt_ids in _fixed_probe_prompts(config):
            next_token = _next_token_id(model, torch, prompt_ids)
            probes.append({"sample_id": sample_id, "prompt_tokens": len(prompt_ids), "next_token_id": next_token})
        return {
            "status": "passed",
            "checkpoint_manifest_path": str(manifest_path),
            "checkpoint_manifest_sha256": sha256_file(manifest_path),
            "run_id": record.get("run_id"),
            "replay_step": record.get("replay_step"),
            "global_optimizer_step": record.get("global_optimizer_step"),
            "adapter_tensor_sha256": actual_tensor_sha,
            "base_weight_sha256": base_identity.get("observed_weight_hash"),
            "model_facts": model_facts,
            "prompt_only_probes": probes,
            "created_at_utc": _now(),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run-all", help="mine, train, gate, and select a complete Replay Run")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--run-id", required=True)
    restore_parser = commands.add_parser("restore-check", help="validate and load a complete P4504 or Replay checkpoint")
    restore_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    restore_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run-all":
        report = _run_all(args.config, args.run_id)
        print(json.dumps({"status": report["status"], "run_id": report["run_id"], "selection": report.get("selection"), "execution_error": report.get("execution_error")}, ensure_ascii=False, indent=2))
        return 1 if report["status"] in {"failed", "interrupted"} else 0
    result = _restore_check(args.config, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
