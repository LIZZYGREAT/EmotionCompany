"""Stage receipts and full-state checkpoint helpers for E03-A7."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE_STEPS = (512, 1024, 2048, 4048)
INITIAL_AUTHORIZATION_TEXT = (
    "按 docs/task/E0.3-A7_分类解耦双路线分阶段训练执行方案.md 执行首轮；"
    "P 与 EP 各训练到 Step 512 并完成配对评测，之后逐节点等待用户明确批准。"
)


class StageAuthorizationError(PermissionError):
    """Raised when a staged A7 training transition has no matching receipt."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage_transition(from_step: int | None, to_step: int) -> None:
    allowed = {None: 512, 512: 1024, 1024: 2048, 2048: 4048}
    if allowed.get(from_step) != to_step:
        raise ValueError(f"invalid A7 staged transition: {from_step!r} -> {to_step}; allowed={allowed}")


def capture_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required <= set(state):
        raise ValueError("A7 checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("A7 checkpoint CUDA RNG state does not match the current device topology")
        torch.cuda.set_rng_state_all(cuda_states)


def save_training_state(
    path: Path,
    *,
    identity: Mapping[str, Any],
    arm: str,
    run_id: str,
    optimizer_step: int,
    schedule_sha256: str,
    consumed_ids: Sequence[str],
    model_state: Mapping[str, Any],
    head_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist all mutable optimization and sample-order state atomically."""
    import torch

    if arm not in {"P", "EP"} or not run_id or isinstance(optimizer_step, bool) or optimizer_step < 1:
        raise ValueError("A7 checkpoint requires a valid arm, Run ID, and positive optimizer step")
    if len(consumed_ids) != optimizer_step * 4:
        raise ValueError("A7 checkpoint consumed-ID count does not equal four samples per step")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    state = {
        "version": "E03-A7-full-training-state-v1",
        "identity": dict(identity),
        "arm": arm,
        "run_id": run_id,
        "optimizer_step": optimizer_step,
        "schedule_sha256": schedule_sha256,
        "sample_cursor": optimizer_step * 4,
        "consumed_ids": list(consumed_ids),
        "consumed_ids_sha256": canonical_sha256(list(consumed_ids)),
        "consumed_ids_set_sha256": canonical_sha256(sorted(set(consumed_ids))),
        "model_state": dict(model_state),
        "head_state": dict(head_state),
        "optimizer_state": dict(optimizer_state),
        "scheduler_state": dict(scheduler_state),
        "rng_state": dict(rng_state or capture_rng_state()),
    }
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "status": "full_state_saved",
        "arm": arm,
        "run_id": run_id,
        "optimizer_step": optimizer_step,
        "sample_cursor": optimizer_step * 4,
        "consumed_ids_sha256": state["consumed_ids_sha256"],
        "consumed_ids_set_sha256": state["consumed_ids_set_sha256"],
        "training_state_sha256": sha256_file(path),
    }


def load_training_state(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_arm: str,
    expected_run_id: str,
    expected_step: int,
    expected_schedule_sha256: str,
    expected_consumed_ids: Sequence[str],
) -> dict[str, Any]:
    """Verify a full-state checkpoint before the caller restores model/optimizer state."""
    import torch

    if not path.is_file():
        raise FileNotFoundError("A7 same-Run full-state checkpoint is missing")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or state.get("version") != "E03-A7-full-training-state-v1"
        or state.get("identity") != dict(expected_identity)
        or state.get("arm") != expected_arm
        or state.get("run_id") != expected_run_id
        or state.get("optimizer_step") != expected_step
        or state.get("sample_cursor") != expected_step * 4
        or state.get("schedule_sha256") != expected_schedule_sha256
        or state.get("consumed_ids") != list(expected_consumed_ids)
        or state.get("consumed_ids_sha256") != canonical_sha256(list(expected_consumed_ids))
        or len(expected_consumed_ids) != expected_step * 4
    ):
        raise StageAuthorizationError("A7 checkpoint identity, Arm, Step, schedule, or sample cursor differs")
    for key in ("model_state", "head_state", "optimizer_state", "scheduler_state", "rng_state"):
        if not isinstance(state.get(key), dict) or not state[key]:
            raise ValueError(f"A7 checkpoint is missing its {key}")
    return state


def write_manifest(path: Path, manifest: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def create_start_ticket(
    path: Path,
    *,
    run_id: str,
    arm: str,
    config_sha256: str,
    code_sha256: str,
    data_sha256: str,
    init_sha256: str,
    dry_run_manifest_sha256: str,
    user_authorization_text: str = INITIAL_AUTHORIZATION_TEXT,
) -> dict[str, Any]:
    """Write the single-use, 0→512 authorization after both dry-runs passed."""
    if arm not in {"P", "EP"} or not all((run_id, config_sha256, code_sha256, data_sha256, init_sha256, dry_run_manifest_sha256)):
        raise ValueError("A7 start ticket bindings must be complete")
    if not isinstance(user_authorization_text, str) or not user_authorization_text.strip():
        raise ValueError("A7 first-stage ticket must cite the explicit user authorization")
    ticket = {
        "version": "E03-A7-start-ticket-v1",
        "status": "unused",
        "run_id": run_id,
        "arm": arm,
        "from_step": 0,
        "to_step": 512,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "data_sha256": data_sha256,
        "init_sha256": init_sha256,
        "dry_run_manifest_sha256": dry_run_manifest_sha256,
        "authorization_source": "explicit_user_request_for_E03_A7",
        "authorization_text": user_authorization_text,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_use": True,
    }
    write_manifest(path, ticket)
    return ticket


def verify_start_ticket(path: Path, *, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise StageAuthorizationError("A7 initial training requires the bound Step 512 start ticket")
    ticket = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if ticket.get(key) != value:
            raise StageAuthorizationError(f"A7 start ticket {key} does not bind this requested Run")
    if ticket.get("version") != "E03-A7-start-ticket-v1" or ticket.get("status") != "unused" or ticket.get("single_use") is not True:
        raise StageAuthorizationError("A7 start ticket is invalid, already used, or not single-use")
    return ticket


def write_continuation_receipt(
    path: Path,
    *,
    run_id: str,
    arm: str,
    from_step: int,
    to_step: int,
    checkpoint_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    config_sha256: str,
    code_sha256: str,
    user_approval_text: str,
) -> dict[str, Any]:
    """Record a user's explicit node-specific approval; the caller verifies evidence first."""
    validate_stage_transition(from_step, to_step)
    if from_step is None or not user_approval_text.strip():
        raise ValueError("A7 continuation receipt requires an explicit user approval statement")
    receipt = {
        "version": "E03-A7-continuation-receipt-v1",
        "status": "approved",
        "run_id": run_id,
        "arm": arm,
        "from_step": from_step,
        "to_step": to_step,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "authorization_source": "explicit_user_message_after_node_review",
        "authorization_text": user_approval_text.strip(),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_use": True,
    }
    write_manifest(path, receipt)
    return receipt


def verify_continuation_receipt(path: Path, *, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise StageAuthorizationError("A7 resume requires a node-specific user approval receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise StageAuthorizationError(f"A7 continuation receipt {key} binding differs")
    if (
        receipt.get("version") != "E03-A7-continuation-receipt-v1"
        or receipt.get("status") != "approved"
        or receipt.get("single_use") is not True
        or not isinstance(receipt.get("authorization_text"), str)
        or not receipt["authorization_text"].strip()
    ):
        raise StageAuthorizationError("A7 continuation receipt is invalid or already consumed")
    return receipt


def mark_receipt_consumed(path: Path, *, run_id: str, arm: str, from_step: int) -> None:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("run_id") != run_id or receipt.get("arm") != arm or receipt.get("from_step") != from_step or receipt.get("status") != "approved":
        raise StageAuthorizationError("A7 receipt cannot be consumed by this Run/node")
    receipt["status"] = "consumed"
    receipt["consumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(path, receipt, overwrite=True)


def checkpoint_manifest(
    checkpoint_path: Path,
    *,
    arm: str,
    run_id: str,
    optimizer_step: int,
    identity: Mapping[str, Any],
    schedule_sha256: str,
    consumed_ids_sha256: str,
    sample_count: int,
    adapter_sha256: str,
    profile_head_sha256: str,
    emotion_head_sha256: str | None,
) -> dict[str, Any]:
    if optimizer_step not in {*STAGE_STEPS, 2826} or sample_count != optimizer_step * 4:
        raise ValueError("A7 checkpoint manifest has an invalid review/epoch milestone or consumed sample count")
    return {
        "version": "E03-A7-checkpoint-manifest-v1",
        "status": "checkpoint_saved",
        "run_id": run_id,
        "arm": arm,
        "optimizer_step": optimizer_step,
        "sample_count": sample_count,
        "sample_cursor": sample_count,
        "identity": dict(identity),
        "schedule_sha256": schedule_sha256,
        "consumed_ids_sha256": consumed_ids_sha256,
        "training_state_file": checkpoint_path.name,
        "training_state_sha256": sha256_file(checkpoint_path),
        "adapter_sha256": adapter_sha256,
        "profile_head_sha256": profile_head_sha256,
        "emotion_head_sha256": emotion_head_sha256,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def verify_checkpoint_manifest(
    manifest_path: Path,
    *,
    expected_run_id: str,
    expected_arm: str,
    expected_step: int,
    expected_identity: Mapping[str, Any],
    expected_schedule_sha256: str,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError("A7 checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_path = manifest_path.parent / manifest.get("training_state_file", "")
    if (
        manifest.get("version") != "E03-A7-checkpoint-manifest-v1"
        or manifest.get("status") != "checkpoint_saved"
        or manifest.get("run_id") != expected_run_id
        or manifest.get("arm") != expected_arm
        or manifest.get("optimizer_step") != expected_step
        or manifest.get("sample_count") != expected_step * 4
        or manifest.get("identity") != dict(expected_identity)
        or manifest.get("schedule_sha256") != expected_schedule_sha256
        or not state_path.is_file()
        or sha256_file(state_path) != manifest.get("training_state_sha256")
    ):
        raise StageAuthorizationError("A7 checkpoint Manifest is invalid or differs from the requested Run")
    return manifest
