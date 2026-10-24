"""D0.4.2 exact Qwen token-length audit and T1 context-budget simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tokenizers
import transformers
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.shared.prompting import (
    apply_chat_template_batch,
    build_messages,
    canonical_json,
    encode_full_training_sequence,
    encode_inference_prompt,
    serialize_completion,
    select_recent_complete_suffix,
    tokenizer_signature,
    validate_completion_suffix,
    validate_prompt_enum_order,
)


AUDIT_VERSION = "D0.4.2-v1"
SPLITS = ("train", "dev", "val")
PROFILE_DIMS = ("personality_traits", "interests", "style")
LENGTH_FIELDS = (
    "prompt_tokens_full",
    "completion_tokens_full",
    "train_sequence_tokens_full",
    "current_user_only_prompt_tokens",
    "current_user_only_completion_tokens",
    "current_user_only_sequence_tokens",
    "current_user_text_tokens",
    "history_message_count",
    "history_text_characters",
    "response_text_characters",
)
HISTOGRAM_BINS = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def _safe_path(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} path escapes project root: {value}")
    return resolved


def _reject_path_overlaps(inputs: dict[str, Path], outputs: dict[str, Path]) -> None:
    for output_name, output_path in outputs.items():
        for input_name, input_path in inputs.items():
            if output_path == input_path or output_path.is_relative_to(input_path) or input_path.is_relative_to(output_path):
                raise ValueError(f"Output {output_name} overlaps input {input_name}")
    items = list(outputs.items())
    for index, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[index + 1:]:
            if left_path == right_path or left_path.is_relative_to(right_path) or right_path.is_relative_to(left_path):
                raise ValueError(f"Outputs {left_name} and {right_name} overlap")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _publish_transaction(items: list[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for stage, final in items:
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists() or final.is_symlink():
                backup = final.with_name(f".{final.name}.backup-{uuid.uuid4().hex}")
                os.replace(final, backup)
                backups.append((final, backup))
            os.replace(stage, final)
            installed.append(final)
    except Exception:
        for final in reversed(installed):
            _remove_path(final)
        for final, backup in reversed(backups):
            if backup.exists() or backup.is_symlink():
                os.replace(backup, final)
        raise
    for _, backup in backups:
        _remove_path(backup)


def _history_key(history: list[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple((turn["role"], turn["content"]) for turn in history)


def _history_from_key(key: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in key]


def _extract_ids(encoded: Any) -> list[list[int]]:
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list):
        ids = list(ids)
    if ids and not isinstance(ids[0], (list, tuple)):
        if len(ids) == 1:
            return [list(map(int, ids))]
        raise ValueError("Tokenizer batch returned a flat sequence")
    return [list(map(int, row)) for row in ids]


def _batch_text_token_ids(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list):
        ids = list(ids)
    if ids and not isinstance(ids[0], (list, tuple)):
        if len(texts) == 1:
            return [list(map(int, ids))]
        raise ValueError("Tokenizer text batch returned a flat sequence")
    return [list(map(int, row)) for row in ids]


def _validate_completion_suffix(
    tokenizer: Any,
    completion_json: str,
    prompt_ids: list[int],
    full_ids: list[int],
    sample_id: str,
) -> list[int]:
    return validate_completion_suffix(
        tokenizer, completion_json, prompt_ids, full_ids, sample_id=sample_id
    )


def _target_enums(schema: dict[str, Any]) -> dict[str, list[str]]:
    defs = schema.get("$defs", {})
    props = defs.get("profile", {}).get("properties", {})
    result = {"emotion_label": list(defs.get("emotion", {}).get("enum", []))}
    for dim in PROFILE_DIMS:
        result[dim] = list(props.get(dim, {}).get("items", {}).get("enum", []))
    if any(not values for values in result.values()):
        raise ValueError("Official data schema is missing a label enumeration")
    return result


def _submission_enums(schema: dict[str, Any]) -> dict[str, list[str]]:
    props = schema.get("properties", {})
    profile = props.get("user_profile", {}).get("properties", {})
    result = {"emotion_label": list(props.get("emotion_label", {}).get("enum", []))}
    for dim in PROFILE_DIMS:
        result[dim] = list(profile.get(dim, {}).get("items", {}).get("enum", []))
    if any(not values for values in result.values()):
        raise ValueError("Submission schema is missing a label enumeration")
    return result


def _load_schema(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    schema = json.loads(raw.decode("utf-8"))
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(schema)
    return schema, _sha256_bytes(raw)


def _load_normalized_inputs(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Path]]:
    inputs_cfg = config["inputs"]
    normalized_dir = _safe_path(root, inputs_cfg["normalized_dir"], "inputs.normalized_dir")
    pipeline_manifest_path = _safe_path(root, inputs_cfg["pipeline_manifest"], "inputs.pipeline_manifest")
    if not pipeline_manifest_path.is_file():
        raise FileNotFoundError(f"D0.3 manifest not found: {pipeline_manifest_path.relative_to(root)}")
    d03_manifest_raw = pipeline_manifest_path.read_bytes()
    d03_manifest = json.loads(d03_manifest_raw.decode("utf-8"))
    if d03_manifest.get("pipeline_version") != "D0.3-v1":
        raise ValueError("Expected a D0.3-v1 pipeline manifest")
    expected_outputs = {item["path"]: item for item in d03_manifest.get("outputs", [])}

    samples: list[dict[str, Any]] = []
    input_hashes: dict[str, Any] = {}
    input_paths: dict[str, Path] = {"d03_manifest": pipeline_manifest_path}
    seen_ids: set[str] = set()
    for split in SPLITS:
        path = normalized_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Normalized {split} JSONL not found: {path.relative_to(root)}")
        input_paths[f"{split}_jsonl"] = path
        before_hash = _sha256_file(path)
        relative_path = str(path.relative_to(root))
        expected = expected_outputs.get(relative_path)
        if expected is None:
            raise ValueError(f"D0.3 manifest has no output record for {relative_path}")
        if expected.get("sha256") != before_hash:
            raise ValueError(f"{split} normalized input hash differs from the D0.3 manifest")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"{relative_path}:{line_number}: blank JSONL record")
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{relative_path}:{line_number}: invalid JSON at column {exc.colno}") from exc
                sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f"{relative_path}:{line_number}: missing sample_id")
                if sample_id in seen_ids:
                    raise ValueError(f"{relative_path}:{line_number}: duplicate sample_id {sample_id}")
                seen_ids.add(sample_id)
                expected_source_split = "val" if split == "val" else "train"
                if sample.get("source", {}).get("split") != expected_source_split:
                    raise ValueError(f"{sample_id}: source.split violates the D0.3 source contract")
                history = sample.get("history")
                if not isinstance(history, list) or not history:
                    raise ValueError(f"{sample_id}: history must be a non-empty array")
                for turn_index, turn in enumerate(history):
                    if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"} or not isinstance(turn.get("content"), str):
                        raise ValueError(f"{sample_id}: history[{turn_index}] violates the D0.3 role/content contract")
                if history[-1]["role"] != "user":
                    raise ValueError(f"{sample_id}: final history message must be user")
                if not isinstance(sample.get("target"), dict):
                    raise ValueError(f"{sample_id}: missing target object")
                item = {"split": split, "sample": sample, "sample_id": sample_id}
                rows.append(item)
                samples.append(item)
        if expected.get("records") != len(rows):
            raise ValueError(f"{split} record count differs from the D0.3 manifest")
        input_hashes[split] = {
            "path": relative_path,
            "records": len(rows),
            "sha256_before": before_hash,
            "sha256_manifest": expected.get("sha256"),
            "matches_d03_manifest": True,
        }
    input_hashes["d03_manifest"] = {
        "path": str(pipeline_manifest_path.relative_to(root)),
        "sha256": _sha256_bytes(d03_manifest_raw),
        "pipeline_version": d03_manifest["pipeline_version"],
        "policy_version": d03_manifest.get("policy_version"),
    }
    return samples, input_hashes, input_paths


def _load_tokenizer(config: dict[str, Any]) -> tuple[Any, Path, dict[str, Any]]:
    model_config = config["model"]
    model_id = model_config["tokenizer_id"]
    revision = model_config.get("tokenizer_revision")
    local_only = bool(model_config.get("local_files_only", True))
    try:
        tokenizer_config_file = Path(hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
            revision=revision,
            local_files_only=local_only,
        ))
        # Keep the snapshot symlink path. Resolving it would point into the shared
        # blobs directory and lose the pinned revision name used for provenance.
        snapshot_path = tokenizer_config_file.parent.absolute()
    except Exception as exc:
        source_mode = "local cache only" if local_only else "configured repository access"
        raise RuntimeError(f"Cannot resolve pinned tokenizer ({source_mode}): {type(exc).__name__}: {exc}") from exc
    resolved_revision = snapshot_path.name
    if revision and resolved_revision != revision:
        raise RuntimeError(f"Resolved tokenizer revision {resolved_revision} differs from configured {revision}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Cannot load tokenizer from {snapshot_path}: {type(exc).__name__}: {exc}") from exc
    tokenizer_files: dict[str, str] = {}
    for name in (
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json",
    ):
        path = snapshot_path / name
        if path.is_file():
            tokenizer_files[name] = _sha256_file(path)
    signature = tokenizer_signature(
        tokenizer,
        model_id=model_id,
        revision=resolved_revision,
        tokenizer_files=tokenizer_files,
    )
    signature.update({
        "snapshot_path": str(snapshot_path),
        "configured_revision": revision,
        "resolved_revision": resolved_revision,
        "local_files_only": local_only,
    })
    return tokenizer, snapshot_path, signature


def _ensure_input_paths_stable(paths: dict[str, Path], before: dict[str, str]) -> dict[str, str]:
    after = {key: _sha256_file(path) for key, path in paths.items()}
    changed = [key for key in before if before[key] != after[key]]
    if changed:
        raise RuntimeError(f"Read-only D0.3 inputs changed during token audit: {changed}")
    return after


def _git_facts(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
            check=False,
        ).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_files_dirty": None}
    return {"commit": commit, "tracked_files_dirty": dirty}


def _percentile(values: list[float | int], percentile: int) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered) / 100) - 1)]


def summarize_values(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "mean": None, "min": None, "p50": None, "p90": None,
            "p95": None, "p99": None, "max": None,
        }
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "min": min(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def _schema_checks(root: Path, config: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    inputs = config["inputs"]
    train_path = _safe_path(root, inputs["train_schema"], "inputs.train_schema")
    val_path = _safe_path(root, inputs["val_schema"], "inputs.val_schema")
    submission_path = _safe_path(root, inputs["submission_schema"], "inputs.submission_schema")
    train_schema, train_hash = _load_schema(train_path, "Train")
    val_schema, val_hash = _load_schema(val_path, "Val")
    submission_schema, submission_hash = _load_schema(submission_path, "submission")
    train_enums = _target_enums(train_schema)
    val_enums = _target_enums(val_schema)
    if train_enums != val_enums:
        raise ValueError("Train and Val official target label enumerations differ")
    prompt_enums = validate_prompt_enum_order(prompt_text, train_schema)
    submission_enums = _submission_enums(submission_schema)
    for key, prompt_labels in prompt_enums.items():
        if not set(prompt_labels).issubset(set(submission_enums[key])):
            raise ValueError(f"P0 {key} contains labels absent from the submission schema")
    return {
        "train_schema": {"path": str(train_path.relative_to(root)), "sha256": train_hash},
        "val_schema": {"path": str(val_path.relative_to(root)), "sha256": val_hash},
        "submission_schema": {"path": str(submission_path.relative_to(root)), "sha256": submission_hash},
        "dataset_target_enums_match_train_val": True,
        "p0_matches_dataset_target_enums_and_order": True,
        "submission_only_emotion_values": [
            item for item in submission_enums["emotion_label"] if item not in train_enums["emotion_label"]
        ],
        "prompt_label_sets_are_accepted_by_submission_schema": True,
        "target_schema_source": "D0.1 Train/Val data schema ($defs.target)",
        "submission_schema_note": "The submission schema includes a blank emotion sentinel; labeled Train/Val targets do not. P0 is validated against the data target schema and requires one actual label.",
    }, train_schema


def _measure_full_batch(
    tokenizer: Any,
    prompt_text: str,
    batch: list[tuple[list[dict[str, str]], str, str]],
    batch_size: int,
) -> list[dict[str, int]]:
    """Measure prompt and full assistant sequence together and check exact token prefixes."""
    measured: list[dict[str, int]] = []
    for offset in range(0, len(batch), batch_size):
        current = batch[offset:offset + batch_size]
        prompt_messages = [messages for messages, _, _ in current]
        full_messages = [
            [*messages, {"role": "assistant", "content": completion, "reasoning_content": ""}]
            for messages, completion, _ in current
        ]
        prompt_ids_batch = apply_chat_template_batch(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        full_ids_batch = apply_chat_template_batch(
            tokenizer, full_messages, add_generation_prompt=False
        )
        for (messages, completion_json, sample_id), prompt_ids, full_ids in zip(
            current, prompt_ids_batch, full_ids_batch
        ):
            completion_ids = _validate_completion_suffix(
                tokenizer, completion_json, prompt_ids, full_ids, sample_id
            )
            measured.append({
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "full_tokens": len(full_ids),
            })
    return measured


def _measure_baseline(
    records: list[dict[str, Any]],
    tokenizer: Any,
    prompt_text: str,
    official_schema: dict[str, Any],
    batch_size: int,
    scalar_verify_count: int,
) -> tuple[list[dict[str, Any]], dict[Any, dict[str, int]], dict[str, Any]]:
    """Measure untruncated, current-user-only, and direct user-text lengths."""
    full_metrics_cache: dict[tuple[Any, str], dict[str, int]] = {}
    prompt_length_cache: dict[Any, int] = {}
    scalar_checks = 0
    output_rows: list[dict[str, Any]] = []

    for offset in range(0, len(records), batch_size):
        current = records[offset:offset + batch_size]
        prepared = []
        for item in current:
            sample = item["sample"]
            sample_id = item["sample_id"]
            try:
                completion_json = serialize_completion(sample["target"], official_schema)
            except ValueError as exc:
                raise ValueError(f"{sample_id}: cannot serialize target: {exc}") from exc
            history = sample["history"]
            messages = build_messages(history, prompt_text)
            current_messages = build_messages([history[-1]], prompt_text)
            prepared.append({
                "item": item,
                "sample": sample,
                "completion_json": completion_json,
                "messages": messages,
                "current_messages": current_messages,
                "history_key": _history_key(history),
                "current_key": _history_key([history[-1]]),
            })

        base_prompt_ids = apply_chat_template_batch(
            tokenizer,
            [entry["messages"] for entry in prepared],
            add_generation_prompt=True,
        )
        base_full_ids = apply_chat_template_batch(
            tokenizer,
            [[*entry["messages"], {"role": "assistant", "content": entry["completion_json"], "reasoning_content": ""}] for entry in prepared],
            add_generation_prompt=False,
        )
        current_prompt_ids = apply_chat_template_batch(
            tokenizer,
            [entry["current_messages"] for entry in prepared],
            add_generation_prompt=True,
        )
        current_full_ids = apply_chat_template_batch(
            tokenizer,
            [[*entry["current_messages"], {"role": "assistant", "content": entry["completion_json"], "reasoning_content": ""}] for entry in prepared],
            add_generation_prompt=False,
        )
        current_user_ids = _batch_text_token_ids(
            tokenizer,
            [entry["sample"]["history"][-1]["content"] for entry in prepared],
        )

        for local_index, entry in enumerate(prepared):
            sample = entry["sample"]
            item = entry["item"]
            sample_id = item["sample_id"]
            base_comp_ids = _validate_completion_suffix(
                tokenizer,
                entry["completion_json"],
                base_prompt_ids[local_index],
                base_full_ids[local_index],
                sample_id,
            )
            current_comp_ids = _validate_completion_suffix(
                tokenizer,
                entry["completion_json"],
                current_prompt_ids[local_index],
                current_full_ids[local_index],
                sample_id,
            )
            if scalar_checks < scalar_verify_count:
                direct_prompt = encode_inference_prompt(tokenizer, entry["messages"])
                direct_full = encode_full_training_sequence(
                    tokenizer, entry["messages"], entry["completion_json"]
                )
                if direct_prompt != base_prompt_ids[local_index] or direct_full != base_full_ids[local_index]:
                    raise ValueError(f"{sample_id}: batched Token IDs differ from direct template calls")
                scalar_checks += 1

            base_metrics = {
                "prompt_tokens": len(base_prompt_ids[local_index]),
                "completion_tokens": len(base_comp_ids),
                "full_tokens": len(base_full_ids[local_index]),
            }
            current_metrics = {
                "prompt_tokens": len(current_prompt_ids[local_index]),
                "completion_tokens": len(current_comp_ids),
                "full_tokens": len(current_full_ids[local_index]),
            }
            history_key = entry["history_key"]
            current_key = entry["current_key"]
            prompt_length_cache[history_key] = base_metrics["prompt_tokens"]
            prompt_length_cache[current_key] = current_metrics["prompt_tokens"]
            full_metrics_cache[(history_key, entry["completion_json"])] = base_metrics
            full_metrics_cache[(current_key, entry["completion_json"])] = current_metrics

            history = sample["history"]
            profile = sample["target"]["user_profile"]
            output = {
                "split": item["split"],
                "sample_id": sample_id,
                "conversation_id": sample["conversation_id"],
                "source": sample["source"],
                "prompt_tokens_full": base_metrics["prompt_tokens"],
                "completion_tokens_full": base_metrics["completion_tokens"],
                "train_sequence_tokens_full": base_metrics["full_tokens"],
                "current_user_only_prompt_tokens": current_metrics["prompt_tokens"],
                "current_user_only_completion_tokens": current_metrics["completion_tokens"],
                "current_user_only_sequence_tokens": current_metrics["full_tokens"],
                "current_user_text_tokens": len(current_user_ids[local_index]),
                "history_message_count": len(history),
                "history_text_characters": sum(len(turn["content"]) for turn in history),
                "response_text_characters": len(sample["target"]["response_text"]),
                "emotion_label": sample["target"]["emotion_label"],
                "profile_all_empty": all(not profile[dim] for dim in PROFILE_DIMS),
                "profile_nonempty_dimensions": [dim for dim in PROFILE_DIMS if profile[dim]],
            }
            item["completion_json"] = entry["completion_json"]
            item["history_key"] = history_key
            item["base_metrics"] = base_metrics
            item["current_key"] = current_key
            item["current_metrics"] = current_metrics
            item["length_row"] = output
            output_rows.append(output)

    p0_text_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    baseline_facts = {
        "p0_text_tokens_independent_reference": len(p0_text_ids),
        "scalar_direct_template_checks": scalar_checks,
        "length_fields": list(LENGTH_FIELDS),
    }
    return output_rows, full_metrics_cache, {
        "prompt_length_cache": prompt_length_cache,
        "full_metrics_cache": full_metrics_cache,
        "baseline_facts": baseline_facts,
    }


def _plan_t1_selections(
    records: list[dict[str, Any]],
    prompt_text: str,
    prompt_caps: list[int],
    tokenizer: Any,
    prompt_length_cache: dict[Any, int],
    batch_size: int,
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, int]]:
    """Compute all T1 suffix choices using cached exact chat-template encodings."""
    choices: dict[str, dict[int, dict[str, Any]]] = {}
    states: list[dict[str, Any]] = []
    for item in records:
        sample = item["sample"]
        history = sample["history"]
        current_key = item["current_key"]
        current_tokens = prompt_length_cache[current_key]
        sample_choices: dict[int, dict[str, Any]] = {}
        pending_caps: set[int] = set()
        for cap in prompt_caps:
            floor_exceeds = current_tokens > cap
            sample_choices[cap] = {
                "selected_start_index": len(history) - 1,
                "prompt_tokens": current_tokens,
                "floor_exceeds_cap": floor_exceeds,
            }
            if not floor_exceeds:
                pending_caps.add(cap)
        choices[item["sample_id"]] = sample_choices
        states.append({
            "item": item,
            "cursor": len(history) - 1,
            "pending_caps": pending_caps,
        })

    prompt_encodings = 0
    while True:
        jobs: list[tuple[dict[str, Any], int, Any]] = []
        missing_order: list[Any] = []
        missing_messages: dict[Any, list[dict[str, str]]] = {}
        for state in states:
            if not state["pending_caps"]:
                continue
            candidate_start = state["cursor"] - 1
            if candidate_start < 0:
                state["pending_caps"].clear()
                continue
            history = state["item"]["sample"]["history"]
            candidate_history = history[candidate_start:]
            key = _history_key(candidate_history)
            jobs.append((state, candidate_start, key))
            if key not in prompt_length_cache and key not in missing_messages:
                missing_order.append(key)
                missing_messages[key] = build_messages(candidate_history, prompt_text)

        if not jobs:
            break
        for offset in range(0, len(missing_order), batch_size):
            keys = missing_order[offset:offset + batch_size]
            encoded = apply_chat_template_batch(
                tokenizer,
                [missing_messages[key] for key in keys],
                add_generation_prompt=True,
            )
            prompt_encodings += len(keys)
            for key, ids in zip(keys, encoded):
                prompt_length_cache[key] = len(ids)

        for state, candidate_start, key in jobs:
            prompt_tokens = prompt_length_cache[key]
            selections = choices[state["item"]["sample_id"]]
            for cap in list(state["pending_caps"]):
                if prompt_tokens <= cap:
                    selections[cap] = {
                        "selected_start_index": candidate_start,
                        "prompt_tokens": prompt_tokens,
                        "floor_exceeds_cap": False,
                    }
                else:
                    # T1 stops at the first failing older message for each cap.
                    state["pending_caps"].remove(cap)
            state["cursor"] = candidate_start
            if candidate_start == 0:
                state["pending_caps"].clear()
    return choices, {"additional_exact_prompt_encodings": prompt_encodings}


def _ensure_selected_full_metrics(
    records: list[dict[str, Any]],
    choices: dict[str, dict[int, dict[str, Any]]],
    prompt_caps: list[int],
    prompt_text: str,
    tokenizer: Any,
    prompt_length_cache: dict[Any, int],
    full_metrics_cache: dict[tuple[Any, str], dict[str, int]],
    batch_size: int,
) -> dict[str, dict[int, dict[str, int]]]:
    """Re-encode every selected full sequence, reusing only exact history+target keys."""
    missing: dict[tuple[Any, str], tuple[list[dict[str, str]], str, str]] = {}
    for item in records:
        history = item["sample"]["history"]
        for cap in prompt_caps:
            choice = choices[item["sample_id"]][cap]
            selected_history = history[choice["selected_start_index"]:]
            key = (_history_key(selected_history), item["completion_json"])
            if key not in full_metrics_cache and key not in missing:
                messages = build_messages(selected_history, prompt_text)
                missing[key] = (messages, item["completion_json"], item["sample_id"])

    missing_keys = list(missing)
    for offset in range(0, len(missing_keys), batch_size):
        keys = missing_keys[offset:offset + batch_size]
        metrics = _measure_full_batch(
            tokenizer,
            prompt_text,
            [missing[key] for key in keys],
            batch_size,
        )
        for key, value in zip(keys, metrics):
            cached_prompt_length = prompt_length_cache.get(key[0])
            if cached_prompt_length is None:
                raise RuntimeError("Selected T1 suffix has no exact prompt length cache entry")
            if value["prompt_tokens"] != cached_prompt_length:
                raise RuntimeError("Selected T1 suffix cached prompt length differs from direct full check")
            full_metrics_cache[key] = value

    result: dict[str, dict[int, dict[str, int]]] = {}
    for item in records:
        sample = item["sample"]
        selected_by_cap = {}
        for cap in prompt_caps:
            choice = choices[item["sample_id"]][cap]
            selected_history = sample["history"][choice["selected_start_index"]:]
            key = (_history_key(selected_history), item["completion_json"])
            metrics = full_metrics_cache.get(key)
            if metrics is None:
                raise RuntimeError(f"Missing selected full-sequence measurement for {item['sample_id']} P={cap}")
            if metrics["prompt_tokens"] != choice["prompt_tokens"]:
                raise RuntimeError(f"{item['sample_id']}: selected prompt length changed after full-sequence encoding")
            selected_by_cap[cap] = metrics
        result[item["sample_id"]] = selected_by_cap
    return result


def _candidate_grid(config: dict[str, Any]) -> list[dict[str, int]]:
    candidates = config.get("simulation", {}).get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("simulation.candidates must be a non-empty list")
    result: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"simulation.candidates[{candidate_index}] must be an object")
        seq = candidate.get("max_train_seq_length")
        prompts = candidate.get("max_prompt_tokens")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
            raise ValueError(f"simulation.candidates[{candidate_index}].max_train_seq_length must be positive")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"simulation.candidates[{candidate_index}].max_prompt_tokens must be a non-empty list")
        for prompt in prompts:
            if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt <= 0 or prompt >= seq:
                raise ValueError(f"Candidate P must be a positive integer smaller than S (S={seq}, P={prompt})")
            pair = (seq, prompt)
            if pair in seen:
                raise ValueError(f"Duplicate S/P candidate: S={seq}, P={prompt}")
            seen.add(pair)
            result.append({"max_train_seq_length": seq, "max_prompt_tokens": prompt})
    return result


def _response_char_buckets(records: list[dict[str, Any]], percentiles: list[int]) -> dict[str, Any]:
    train_lengths = [
        len(item["sample"]["target"]["response_text"])
        for item in records if item["split"] == "train"
    ]
    thresholds = {f"p{p}": _percentile(train_lengths, p) for p in percentiles}
    p50 = thresholds.get("p50")
    p90 = thresholds.get("p90")

    def bucket(length: int) -> str:
        if p50 is None or p90 is None:
            return "unavailable"
        if length <= p50:
            return "short"
        if length <= p90:
            return "medium"
        return "long"

    return {"thresholds": thresholds, "bucket": bucket, "method": "nearest_rank; short <= Train P50, medium <= Train P90, long > Train P90"}


def _length_summaries(length_rows: list[dict[str, Any]], response_bucket: dict[str, Any]) -> dict[str, Any]:
    metric_names = [
        "prompt_tokens_full", "completion_tokens_full", "train_sequence_tokens_full",
        "current_user_only_prompt_tokens", "current_user_only_completion_tokens",
        "current_user_only_sequence_tokens", "current_user_text_tokens",
        "history_message_count", "history_text_characters", "response_text_characters",
    ]
    split_summary: dict[str, Any] = {}
    for split in (*SPLITS, "overall"):
        rows = [row for row in length_rows if split == "overall" or row["split"] == split]
        split_summary[split] = {
            "n": len(rows),
            "metrics": {metric: summarize_values([row[metric] for row in rows]) for metric in metric_names},
        }

    groups: dict[str, Any] = {}
    dimensions: dict[str, Any] = {
        "emotion_label": sorted({row["emotion_label"] for row in length_rows}),
        "profile_all_empty": [True, False],
        "response_length_bucket": ["short", "medium", "long"],
    }
    for dimension, values in dimensions.items():
        groups[dimension] = {}
        for value in values:
            if dimension == "response_length_bucket":
                rows = [row for row in length_rows if response_bucket["bucket"](row["response_text_characters"]) == value]
            else:
                rows = [row for row in length_rows if row[dimension] == value]
            groups[dimension][str(value).lower() if isinstance(value, bool) else str(value)] = {
                "n": len(rows),
                "small_sample_warning": len(rows) < 30,
                "metrics": {
                    metric: summarize_values([row[metric] for row in rows])
                    for metric in ("prompt_tokens_full", "completion_tokens_full", "train_sequence_tokens_full", "response_text_characters")
                },
            }
    return {"by_split": split_summary, "by_group": groups}


def _classify_budget_rows(
    records: list[dict[str, Any]],
    length_by_id: dict[str, dict[str, Any]],
    candidates: list[dict[str, int]],
    choices: dict[str, dict[int, dict[str, Any]]],
    selected_metrics: dict[str, dict[int, dict[str, int]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    simulation_rows: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for candidate in candidates:
        seq = candidate["max_train_seq_length"]
        cap = candidate["max_prompt_tokens"]
        key = f"S{seq}_P{cap}"
        rows: list[dict[str, Any]] = []
        for item in records:
            sample = item["sample"]
            sample_id = item["sample_id"]
            baseline = length_by_id[sample_id]
            choice = choices[sample_id][cap]
            metrics = selected_metrics[sample_id][cap]
            start = choice["selected_start_index"]
            original_history = sample["history"]
            selected_history = original_history[start:]
            original_messages = len(original_history)
            retained_messages = len(selected_history)
            original_characters = baseline["history_text_characters"]
            retained_characters = sum(len(turn["content"]) for turn in selected_history)
            removed_messages = original_messages - retained_messages
            floor_exceeds = bool(choice["floor_exceeds_cap"])
            if floor_exceeds:
                outcome = "prompt_floor_exceeds_cap"
                reason = "CURRENT_MESSAGE_WITH_SYSTEM_EXCEEDS_PROMPT_CAP"
            elif metrics["full_tokens"] > seq:
                outcome = "completion_overflows_train_budget"
                reason = (
                    "CURRENT_MESSAGE_AND_COMPLETE_COMPLETION_EXCEED_TRAIN_BUDGET"
                    if metrics["completion_tokens"] > seq - baseline["current_user_only_prompt_tokens"]
                    else "SELECTED_PROMPT_LEAVES_INSUFFICIENT_TRAIN_BUDGET_FOR_COMPLETION"
                )
            elif removed_messages:
                outcome = "trimmed_fit"
                reason = None
            else:
                outcome = "full_fit"
                reason = None
            raw_fits = (
                baseline["prompt_tokens_full"] <= cap
                and baseline["train_sequence_tokens_full"] <= seq
            )
            row = {
                "split": item["split"],
                "sample_id": sample_id,
                "conversation_id": sample["conversation_id"],
                "source": sample["source"],
                "max_train_seq_length": seq,
                "max_prompt_tokens": cap,
                "outcome": outcome,
                "failure_reason": reason,
                "raw_full_fit": raw_fits,
                "prompt_tokens_full": baseline["prompt_tokens_full"],
                "selected_prompt_tokens": metrics["prompt_tokens"],
                "completion_tokens_full": metrics["completion_tokens"],
                "selected_full_sequence_tokens": metrics["full_tokens"],
                "current_user_only_prompt_tokens": baseline["current_user_only_prompt_tokens"],
                "current_user_only_sequence_tokens": baseline["current_user_only_sequence_tokens"],
                "original_history_message_count": original_messages,
                "retained_history_message_count": retained_messages,
                "removed_history_message_count": removed_messages,
                "original_history_text_characters": original_characters,
                "retained_history_text_characters": retained_characters,
                "message_retention_ratio": retained_messages / original_messages if original_messages else None,
                "text_character_retention_ratio": retained_characters / original_characters if original_characters else None,
                "emotion_label": sample["target"]["emotion_label"],
                "profile_all_empty": baseline["profile_all_empty"],
            }
            rows.append(row)
            if reason:
                exceptions.append({
                    "split": row["split"],
                    "sample_id": sample_id,
                    "conversation_id": sample["conversation_id"],
                    "source": sample["source"],
                    "max_train_seq_length": seq,
                    "max_prompt_tokens": cap,
                    "outcome": outcome,
                    "reason": reason,
                    "prompt_tokens_full": baseline["prompt_tokens_full"],
                    "selected_prompt_tokens": metrics["prompt_tokens"],
                    "completion_tokens_full": metrics["completion_tokens"],
                    "selected_full_sequence_tokens": metrics["full_tokens"],
                    "current_user_only_prompt_tokens": baseline["current_user_only_prompt_tokens"],
                    "current_user_only_sequence_tokens": baseline["current_user_only_sequence_tokens"],
                })
            simulation_rows.append(row)

        count_by_outcome = Counter(row["outcome"] for row in rows)
        n = len(rows)
        if sum(count_by_outcome.values()) != n:
            raise RuntimeError(f"Budget outcome counts do not conserve samples for {key}")
        usable = count_by_outcome["full_fit"] + count_by_outcome["trimmed_fit"]
        deleted = [row["removed_history_message_count"] for row in rows]
        raw_fit_count = sum(row["raw_full_fit"] for row in rows)
        summary[key] = {
            "max_train_seq_length": seq,
            "max_prompt_tokens": cap,
            "n_total": n,
            "n_full_fit": count_by_outcome["full_fit"],
            "n_trimmed_fit": count_by_outcome["trimmed_fit"],
            "n_prompt_floor_exceeds_cap": count_by_outcome["prompt_floor_exceeds_cap"],
            "n_completion_overflows_train_budget": count_by_outcome["completion_overflows_train_budget"],
            "raw_full_fit_count": raw_fit_count,
            "raw_full_fit_rate": raw_fit_count / n if n else None,
            "usable_count": usable,
            "usable_rate": usable / n if n else None,
            "truncated_count": sum(value > 0 for value in deleted),
            "truncation_rate": sum(value > 0 for value in deleted) / n if n else None,
            "unusable_count": n - usable,
            "unusable_rate": (n - usable) / n if n else None,
            "removed_history_message_count": summarize_values(deleted),
            "retained_history_message_count": summarize_values([row["retained_history_message_count"] for row in rows]),
            "mean_message_retention_ratio_proxy": round(statistics.mean([row["message_retention_ratio"] for row in rows]), 6) if rows else None,
            "mean_text_character_retention_ratio_proxy": round(statistics.mean([row["text_character_retention_ratio"] for row in rows if row["text_character_retention_ratio"] is not None]), 6) if any(row["text_character_retention_ratio"] is not None for row in rows) else None,
            "selected_prompt_tokens": summarize_values([row["selected_prompt_tokens"] for row in rows if not row["failure_reason"] or row["outcome"] == "completion_overflows_train_budget"]),
            "selected_full_sequence_tokens_all_encodable": summarize_values([row["selected_full_sequence_tokens"] for row in rows]),
            "failure_groups": _failure_groups(rows),
            "counts_conserve": sum(count_by_outcome.values()) == n,
        }
        by_split = {}
        for split in SPLITS:
            split_rows = [row for row in rows if row["split"] == split]
            split_counts = Counter(row["outcome"] for row in split_rows)
            split_n = len(split_rows)
            split_usable = split_counts["full_fit"] + split_counts["trimmed_fit"]
            split_raw_fit = sum(row["raw_full_fit"] for row in split_rows)
            split_truncated = sum(row["removed_history_message_count"] > 0 for row in split_rows)
            by_split[split] = {
                "n_total": split_n,
                "n_full_fit": split_counts["full_fit"],
                "n_trimmed_fit": split_counts["trimmed_fit"],
                "n_prompt_floor_exceeds_cap": split_counts["prompt_floor_exceeds_cap"],
                "n_completion_overflows_train_budget": split_counts["completion_overflows_train_budget"],
                "raw_full_fit_count": split_raw_fit,
                "raw_full_fit_rate": split_raw_fit / split_n if split_n else None,
                "usable_rate": split_usable / split_n if split_n else None,
                "truncation_rate": split_truncated / split_n if split_n else None,
                "unusable_rate": (split_n - split_usable) / split_n if split_n else None,
                "mean_message_retention_ratio_proxy": round(statistics.mean([row["message_retention_ratio"] for row in split_rows]), 6) if split_rows else None,
                "mean_text_character_retention_ratio_proxy": round(statistics.mean([row["text_character_retention_ratio"] for row in split_rows if row["text_character_retention_ratio"] is not None]), 6) if any(row["text_character_retention_ratio"] is not None for row in split_rows) else None,
                "counts_conserve": sum(split_counts.values()) == split_n,
            }
        summary[key]["by_split"] = by_split
    return simulation_rows, exceptions, summary


def _frontier_comparisons(
    simulation_rows: list[dict[str, Any]], candidates: list[dict[str, int]]
) -> dict[str, Any]:
    by_sample_candidate = {
        (row["sample_id"], row["max_train_seq_length"], row["max_prompt_tokens"]): row
        for row in simulation_rows
    }
    candidate_pairs: list[tuple[str, tuple[int, int], tuple[int, int], bool]] = []
    available = {(item["max_train_seq_length"], item["max_prompt_tokens"]) for item in candidates}
    if (2048, 1536) in available and (4096, 3584) in available:
        candidate_pairs.append(("default_joint_frontier_S2048_P1536_vs_S4096_P3584", (2048, 1536), (4096, 3584), False))
    for prompt in (1024, 1280, 1536):
        if (2048, prompt) in available and (4096, prompt) in available:
            candidate_pairs.append((f"matched_P{prompt}_S2048_vs_S4096", (2048, prompt), (4096, prompt), True))

    comparisons = {}
    for label, lower, upper, matched_prompt in candidate_pairs:
        lower_seq, lower_prompt = lower
        upper_seq, upper_prompt = upper
        retained_deltas = []
        affected = []
        nonempty_count = 0
        nonempty_affected = 0
        for (sample_id, _, _), low_row in by_sample_candidate.items():
            if (low_row["max_train_seq_length"], low_row["max_prompt_tokens"]) != lower:
                continue
            high_row = by_sample_candidate[(sample_id, upper_seq, upper_prompt)]
            delta = high_row["retained_history_message_count"] - low_row["retained_history_message_count"]
            retained_deltas.append(delta)
            affected.append(delta > 0)
            if not low_row["profile_all_empty"]:
                nonempty_count += 1
                nonempty_affected += delta > 0
            if matched_prompt and (
                delta != 0 or low_row["selected_prompt_tokens"] != high_row["selected_prompt_tokens"]
            ):
                raise RuntimeError(f"T1 selection differs across matched-P sensitivity pair {label}: {sample_id}")
        comparisons[label] = {
            "lower_candidate": {"S": lower_seq, "P": lower_prompt},
            "upper_candidate": {"S": upper_seq, "P": upper_prompt},
            "same_P_control": matched_prompt,
            "mean_additional_retained_history_messages": round(statistics.mean(retained_deltas), 6) if retained_deltas else None,
            "n_samples_retaining_additional_history": sum(affected),
            "rate_samples_retaining_additional_history": sum(affected) / len(affected) if affected else None,
            "n_nonempty_profile_samples": nonempty_count,
            "n_nonempty_profile_samples_retaining_additional_history": nonempty_affected,
            "rate_nonempty_profile_samples_retaining_additional_history": nonempty_affected / nonempty_count if nonempty_count else None,
            "nonempty_profile_share_among_affected_samples": nonempty_affected / sum(affected) if sum(affected) else None,
            "interpretation": "history retention proxy only; no model quality inference",
        }
    return comparisons


def _failure_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("emotion_label", "profile_all_empty"):
        result[field] = {}
        categories = sorted({str(row[field]) for row in rows})
        for category in categories:
            subset = [row for row in rows if str(row[field]) == category]
            failures = [row for row in subset if row["outcome"] in {"prompt_floor_exceeds_cap", "completion_overflows_train_budget"}]
            result[field][category] = {
                "n_total": len(subset),
                "n_failed": len(failures),
                "failure_rate": len(failures) / len(subset) if subset else None,
                "small_sample_warning": len(subset) < 30,
                "by_outcome": dict(sorted(Counter(row["outcome"] for row in failures).items())),
            }
    return result


def _deployment_check(root: Path, config: dict[str, Any], candidates: list[dict[str, int]]) -> dict[str, Any]:
    deployment = config.get("deployment", {})
    reference_path = _safe_path(root, deployment["reference_inference_config"], "deployment.reference_inference_config")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_g = reference.get("generation", {}).get("max_new_tokens")
    generation_candidates = deployment.get("additional_generation_candidates", [])
    generations = []
    if isinstance(reference_g, int) and not isinstance(reference_g, bool) and reference_g > 0:
        generations.append({"G": reference_g, "source": "reference_inference_config (reference only)"})
    for value in generation_candidates:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("deployment.additional_generation_candidates must contain positive integers")
        generations.append({"G": value, "source": "configured candidate"})
    limits = deployment.get("serving_context_limits", [])
    for value in limits:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("deployment.serving_context_limits must contain positive integers")
    checks = []
    for candidate in candidates:
        for generation in generations:
            for limit in limits:
                required = candidate["max_prompt_tokens"] + generation["G"]
                checks.append({
                    "S": candidate["max_train_seq_length"], "P": candidate["max_prompt_tokens"],
                    "G": generation["G"], "G_source": generation["source"], "K": limit,
                    "required_P_plus_G": required, "compatible": required <= limit,
                })
    return {
        "reference_config_path": str(reference_path.relative_to(root)),
        "reference_max_new_tokens": reference_g,
        "reference_G_is_not_final": True,
        "serving_context_limits_K": limits,
        "checks": checks,
        "status": "not_assessable_K_unconfigured" if not limits else "checked_for_configured_candidates",
        "constraint": "P + G <= K; S is a separate training budget",
    }


def _svg_histogram(path: Path, title: str, values_by_label: dict[str, list[int]]) -> None:
    width, height = 960, 500
    left, right, top, bottom = 74, 28, 58, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    bins = HISTOGRAM_BINS
    labels = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)] + [f">{bins[-1]}"]
    series = []
    maximum = 1
    for label, values in values_by_label.items():
        counts = [sum(bins[i] <= value < bins[i + 1] for value in values) for i in range(len(bins) - 1)]
        counts.append(sum(value >= bins[-1] for value in values))
        series.append((label, counts))
        maximum = max(maximum, max(counts, default=0))
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed"]
    group_w = plot_w / max(1, len(labels))
    bar_w = group_w / (len(series) + 1)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="#ffffff"/>',
              f'<text x="{left}" y="32" font-family="sans-serif" font-size="21" font-weight="600">{title}</text>']
    for tick in range(5):
        value = maximum * tick / 4
        y = top + plot_h * (1 - tick / 4)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{value:.0f}</text>')
    for index, bin_label in enumerate(labels):
        center = left + group_w * (index + 0.5)
        pieces.append(f'<text x="{center:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#475569">{bin_label}</text>')
        for series_index, (name, counts) in enumerate(series):
            count = counts[index]
            bar_h = count / maximum * plot_h
            x = center - (len(series) * bar_w / 2) + series_index * bar_w
            y = top + plot_h - bar_h
            pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2, bar_w-3):.1f}" height="{bar_h:.1f}" fill="{colors[series_index % len(colors)]}" opacity="0.88"/>')
    legend_y = height - 35
    for index, (name, _) in enumerate(series):
        x = left + index * 115
        pieces.append(f'<rect x="{x}" y="{legend_y-12}" width="12" height="12" fill="{colors[index % len(colors)]}"/>')
        pieces.append(f'<text x="{x+18}" y="{legend_y-2}" font-family="sans-serif" font-size="12" fill="#334155">{name}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _svg_retention(path: Path, budget_summary: dict[str, Any]) -> None:
    width, height = 1100, 520
    left, right, top, bottom = 80, 35, 70, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    entries = list(budget_summary.items())
    group_w = plot_w / max(1, len(entries))
    series = [("消息数比例", "mean_message_retention_ratio_proxy", "#2563eb"), ("正文字符比例", "mean_text_character_retention_ratio_proxy", "#16a34a")]
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="#ffffff"/>',
              f'<text x="{left}" y="34" font-family="sans-serif" font-size="21" font-weight="600">T1 平均历史保留代理比例（消息数 / 正文字符）</text>']
    for tick in range(5):
        fraction = tick / 4
        y = top + plot_h * (1 - fraction)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        pieces.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{fraction:.0%}</text>')
    for index, (label, metrics) in enumerate(entries):
        center = left + group_w * (index + 0.5)
        for series_index, (legend, field, color) in enumerate(series):
            value = metrics[field]
            if value is None:
                continue
            bar_w = group_w * 0.30
            x = center - group_w * 0.30 + series_index * group_w * 0.31
            bar_h = max(0, min(1, value)) * plot_h
            y = top + plot_h - bar_h
            pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            pieces.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#334155">{value:.1%}</text>')
        pieces.append(f'<text x="{center:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#475569">{label}</text>')
    for index, (legend, field, color) in enumerate(series):
        x = left + index * 160
        y = height - 35
        pieces.append(f'<rect x="{x}" y="{y-12}" width="12" height="12" fill="{color}"/>')
        pieces.append(f'<text x="{x+18}" y="{y-2}" font-family="sans-serif" font-size="12" fill="#334155">{legend}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _report_markdown(
    config: dict[str, Any],
    input_hashes: dict[str, Any],
    schema_facts: dict[str, Any],
    tokenizer_signature_data: dict[str, Any],
    prompt_sha: str,
    length_summary: dict[str, Any],
    budget_summary: dict[str, Any],
    comparisons: dict[str, Any],
    response_buckets: dict[str, Any],
    deployment: dict[str, Any],
    counts: dict[str, int],
    output_paths: dict[str, str],
    validation_result: dict[str, Any] | None,
) -> str:
    overall = length_summary["by_split"]["overall"]["metrics"]
    lines = [
        "# D0.4.2 Token 长度审计与上下文预算模拟报告", "",
        "> 本报告只提供精确长度事实与固定候选预算模拟，不冻结最终 S/P/G/K，不进入训练或 D0.4.3。", "",
        "## 1. 数据、Prompt 与 Tokenizer", "",
        f"- D0.3 版本：`D0.3-v1`；样本数：Train {counts['train']:,} / Dev {counts['dev']:,} / Val {counts['val']:,}，合计 {sum(counts.values()):,}。",
        f"- D0.3 三份规范化文件均与 Manifest SHA-256 一致：{all(input_hashes[split]['matches_d03_manifest'] for split in SPLITS)}。",
        f"- P0：`{config['prompt']['version']}`，SHA-256 `{prompt_sha}`；标签枚举与 Train/Val 官方 target schema 及顺序一致。",
        f"- Tokenizer：`{tokenizer_signature_data['model_id']}` revision `{tokenizer_signature_data['resolved_revision']}`，类 `{tokenizer_signature_data['tokenizer_class']}`；Chat Template SHA-256 `{tokenizer_signature_data['template_sha256']}`。",
        f"- 模板参数：F2 原生消息，`enable_thinking=False`；前缀门禁已在逐样本统计和 {config['tokenizer']['verify_batch_with_scalar_samples']} 条标量/批量对照中检查。",
        "- Qwen 模板会解析 assistant 文本中的字面 `</think>`；历史 assistant 消息和目标 JSON 显式带空 `reasoning_content` 字段，防止模板改写原文，字段本身不进入序列。",
        f"- 数据 Schema 检查：Train/Val 目标枚举一致；提交 Schema 额外允许空情绪哨兵 `{schema_facts['submission_only_emotion_values']}`，P0 仍要求一个实际情绪标签。",
        "- 分位数采用 nearest-rank：排序后索引 `ceil(p*n)-1`；所有长度均为真实 Tokenizer Token 数，字符数只用于明确标注的历史保留代理指标。", "",
        "## 2. 未截断长度分布", "",
        "| Split | n | Prompt P50/P90/P95/P99/Max | Completion P50/P90/P95/P99/Max | Full sequence P50/P90/P95/P99/Max |",
        "|---|---:|---|---|---|",
    ]
    for split in (*SPLITS, "overall"):
        group = length_summary["by_split"][split]
        metrics = group["metrics"]
        def p_summary(name: str) -> str:
            stat = metrics[name]
            return "/".join(str(stat[field]) for field in ("p50", "p90", "p95", "p99", "max"))
        lines.append(f"| {split} | {group['n']:,} | {p_summary('prompt_tokens_full')} | {p_summary('completion_tokens_full')} | {p_summary('train_sequence_tokens_full')} |")
    lines.extend(["", "单位依次为 Token；表中列顺序为 P50 / P90 / P95 / P99 / Max。回复字符短/中/长阈值取 Train 字符长度 nearest-rank P50=`{}`、P90=`{}`（短 ≤P50，中 ≤P90，长 >P90）。".format(response_buckets['thresholds'].get('p50'), response_buckets['thresholds'].get('p90')), ""])
    lines.extend([
        "## 3. S/P 候选预算模拟", "",
        "四类结果互斥：`full_fit`、`trimmed_fit`、`prompt_floor_exceeds_cap`、`completion_overflows_train_budget`。raw full-fit 同时要求未裁剪 Prompt ≤ P 且未裁剪完整序列 ≤ S；可用率只计前两类；截断率按有删除至少一条历史消息的样本计。", "",
        "| S | P | full_fit | trimmed_fit | Prompt floor 超限 | 完整监督超 S | raw full-fit | 可用率 | 截断率 | 不可用率 | 平均消息保留代理 | 平均字符保留代理 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for entry in budget_summary.values():
        lines.append(
            f"| {entry['max_train_seq_length']} | {entry['max_prompt_tokens']} | {entry['n_full_fit']:,} | {entry['n_trimmed_fit']:,} | {entry['n_prompt_floor_exceeds_cap']:,} | {entry['n_completion_overflows_train_budget']:,} | {entry['raw_full_fit_rate']:.2%} | {entry['usable_rate']:.2%} | {entry['truncation_rate']:.2%} | {entry['unusable_rate']:.2%} | {entry['mean_message_retention_ratio_proxy']:.2%} | {entry['mean_text_character_retention_ratio_proxy'] if entry['mean_text_character_retention_ratio_proxy'] is not None else 0:.2%} |"
        )
    lines.extend([
        "",
        "### 按 Split 的候选统计",
        "",
        "下表各行独立以该 Split 的全部样本为 N；四类数量之和必须为 N。",
        "",
        "| Split | S | P | N | full_fit | trimmed_fit | Prompt floor 超限 | Completion 超 S | raw-fit | 可用率 | 截断率 | 不可用率 | 平均消息保留代理 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for entry in budget_summary.values():
        for split in SPLITS:
            part = entry["by_split"][split]
            lines.append(
                f"| {split} | {entry['max_train_seq_length']} | {entry['max_prompt_tokens']} | {part['n_total']:,} | {part['n_full_fit']:,} | {part['n_trimmed_fit']:,} | {part['n_prompt_floor_exceeds_cap']:,} | {part['n_completion_overflows_train_budget']:,} | {part['raw_full_fit_rate']:.2%} | {part['usable_rate']:.2%} | {part['truncation_rate']:.2%} | {part['unusable_rate']:.2%} | {part['mean_message_retention_ratio_proxy']:.2%} |"
            )
    lines.extend([
        "",
        "指标分母统一为对应 Split（overall 时为全部样本）N：`usable_rate=(full_fit+trimmed_fit)/N`；`unusable_rate=(prompt_floor_exceeds_cap+completion_overflows_train_budget)/N`；`raw_full_fit_rate=n_raw_full_fit/N`；`truncation_rate=count(removed_history_message_count>0)/N`。最后一项包含不可用但已删除历史的样本，不等同于 `n_trimmed_fit/N`。",
        "",
        "历史消息数与正文字符保留率是代理指标，不等于 Token 保留率；Completion 从不裁剪，失败样本和源定位见 `length_exceptions.jsonl`。按情绪标签与空/非空画像分层的失败数量、比例及小样本警示写入 `token_stats.json`。",
        "",
        "### 候选权衡（非最终配置）",
        "",
        "| S/P | overall 可用率 | 平均消息保留代理 | 候选定位 |",
        "|---|---:|---:|---|",
    ])
    candidate_notes = {
        "S2048_P1024": "监督覆盖较高，较长历史裁剪更多",
        "S2048_P1280": "本地折中候选，可优先做后续训练可行性验证",
        "S2048_P1536": "更重视历史保留，可用率略低",
        "S4096_P2048": "更长训练序列候选，显存/延迟未测",
        "S4096_P3072": "更高历史保留候选，需检查 P+G 与 K",
    }
    for key, note in candidate_notes.items():
        entry = budget_summary.get(key)
        if entry:
            lines.append(f"| {entry['max_train_seq_length']}/{entry['max_prompt_tokens']} | {entry['usable_rate']:.2%} | {entry['mean_message_retention_ratio_proxy']:.2%} | {note} |")
    lines.extend([
        "",
        "假设示例（不是实际服务验证）：若 K=4096 且参考 G=1200，则 P=1280、2048 满足 P+G≤K，P=3072、3584 不满足；最终 K 必须从实际服务配置并经部署验证确认。",
        "",
    ])
    lines.extend([
        "## 4. 关键比较与限制", "",
        "- 默认探索前沿 S2048/P1536 与 S4096/P3584 同时改变了 S 和 P，只能作为联合预算对照，不可解释成 S 的单独因果效应。",
    ])
    frontier_a = budget_summary.get("S2048_P1536")
    frontier_b = budget_summary.get("S4096_P3584")
    if frontier_a and frontier_b:
        delta = frontier_b["mean_message_retention_ratio_proxy"] - frontier_a["mean_message_retention_ratio_proxy"]
        lines.append(f"- 联合前沿的平均消息保留代理比例变化：{delta:+.2%}；这里只描述 T1 历史保留，不表示分类/回复质量提升。")
        lines.append(f"- 两个前沿可用率分别为 {frontier_a['usable_rate']:.2%} 和 {frontier_b['usable_rate']:.2%}。")
    lines.append("- 同 P 的 S2048 与 S4096 对照用于隔离训练总预算影响；固定 P 下 T1 选择相同，主要差异应体现在完整监督是否容纳。")
    for label, comparison in comparisons.items():
        if comparison["same_P_control"]:
            lines.append(f"- {label}：T1 选择已逐样本验证相同；平均额外保留历史消息 {comparison['mean_additional_retained_history_messages']:+.3f}，额外保留历史样本 {comparison['n_samples_retaining_additional_history']}。")
        else:
            profile_rate = comparison["rate_nonempty_profile_samples_retaining_additional_history"]
            profile_text = "无非空画像样本" if profile_rate is None else f"非空画像样本中 {profile_rate:.2%} 保留了更多历史"
            lines.append(f"- {label}：平均额外保留历史消息 {comparison['mean_additional_retained_history_messages']:+.3f}，{comparison['n_samples_retaining_additional_history']} 条样本保留更多历史；{profile_text}。这是同时改变 S/P 的联合预算差异。")
    lines.append(f"- 推理约束为 P+G≤K。当前参考配置 max_new_tokens={deployment['reference_max_new_tokens']} 仅作 G 参考；K={deployment['serving_context_limits_K']}，状态：`{deployment['status']}`。S 与 K 不等同。")
    lines.extend([
        "- 本阶段没有模型权重加载、训练、Loss Mask、Padding 张量、质量对照或硬件吞吐/显存结论。", "",
        "## 5. 图表与机器产物", "",
        f"- 完整训练序列长度分布：[{Path(output_paths['full_histogram']).name}]({output_paths['full_histogram']})",
        f"- Completion 长度分布：[{Path(output_paths['completion_histogram']).name}]({output_paths['completion_histogram']})",
        f"- T1 历史保留代理比例：[{Path(output_paths['retention_chart']).name}]({output_paths['retention_chart']})",
        "- 机器统计、逐样本长度、预算逐样本模拟、不可用样本定位和运行 Manifest 均位于 `outputs/token_audit/v1/`。", "",
        "## 6. 复现、测试与交接", "",
        "```bash", "python -m src.data.token_stats --config configs/data_token_audit.yaml", "```", "",
    ])
    if validation_result:
        lines.append(f"- 回归命令：`{validation_result.get('command')}`；退出码 `{validation_result.get('exit_code')}`；摘要：`{validation_result.get('summary')}`。")
    else:
        lines.append("- 当前 Manifest 未提供执行中的回归测试摘要；请用配置中的 validation.unit_test_command 完成回归。")
    lines.extend([
        "- D0.4.3 交接：结合这些长度分布与失败定位，再决定训练 S/P、监督边界及需处理的极端样本；仍需独立实测 G/K 兼容性与硬件可行性。", "",
        "### 复现所用输入 SHA-256", "",
    ])
    for split in SPLITS:
        item = input_hashes[split]
        lines.append(f"- `{item['path']}`: `{item['sha256_before']}`（Manifest 匹配：{item['matches_d03_manifest']}）")
    lines.append("")
    return "\n".join(lines)


def _run_validation(root: Path, command: str) -> dict[str, Any]:
    # Execute the configured module list with the current interpreter, without a shell.
    prefix = "python -m pytest "
    if not command.startswith(prefix):
        raise ValueError("validation.unit_test_command must use `python -m pytest ...`")
    test_paths = [part for part in command[len(prefix):].split() if part != "-q"]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *test_paths, "-q"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    summary = next((line.strip() for line in reversed(output.splitlines()) if " passed" in line or " failed" in line or " error" in line), output[-500:])
    return {
        "command": f"{Path(sys.executable).name} -m pytest {' '.join(test_paths)} -q",
        "exit_code": completed.returncode,
        "summary": summary,
    }


def run_audit(
    root: Path,
    config_path: Path,
    *,
    validation_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config_raw = config_path.read_bytes()
    config = yaml.safe_load(config_raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("version") != AUDIT_VERSION:
        raise ValueError(f"Expected configuration version {AUDIT_VERSION}")
    candidates = _candidate_grid(config)
    batch_size = config.get("tokenizer", {}).get("batch_size")
    scalar_verify_count = config.get("tokenizer", {}).get("verify_batch_with_scalar_samples")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("tokenizer.batch_size must be a positive integer")
    if not isinstance(scalar_verify_count, int) or scalar_verify_count < 0:
        raise ValueError("tokenizer.verify_batch_with_scalar_samples must be a non-negative integer")

    prompt_path = _safe_path(root, config["prompt"]["text_path"], "prompt.text_path")
    prompt_bytes = prompt_path.read_bytes()
    prompt_text = prompt_bytes.decode("utf-8")
    prompt_sha = _sha256_bytes(prompt_bytes)
    schema_facts, official_schema = _schema_checks(root, config, prompt_text)
    records, input_hashes, input_paths = _load_normalized_inputs(root, config)
    reference_config_path = _safe_path(root, config["deployment"]["reference_inference_config"], "deployment.reference_inference_config")
    protected_input_paths = {
        **input_paths,
        "audit_config": config_path,
        "p0": prompt_path,
        "train_schema": _safe_path(root, config["inputs"]["train_schema"], "inputs.train_schema"),
        "val_schema": _safe_path(root, config["inputs"]["val_schema"], "inputs.val_schema"),
        "submission_schema": _safe_path(root, config["inputs"]["submission_schema"], "inputs.submission_schema"),
        "reference_inference_config": reference_config_path,
    }
    input_hashes_before = {key: _sha256_file(path) for key, path in protected_input_paths.items()}
    tokenizer, tokenizer_path, tokenizer_facts = _load_tokenizer(config)
    if bool(config["model"].get("enable_thinking", False)):
        raise ValueError("D0.4.2 requires model.enable_thinking=false")
    if config["prompt"].get("history_format") != "native_chat_template":
        raise ValueError("Only native_chat_template is supported in D0.4.2")

    length_rows, _, cache_bundle = _measure_baseline(
        records, tokenizer, prompt_text, official_schema, batch_size, scalar_verify_count
    )
    prompt_cache = cache_bundle["prompt_length_cache"]
    full_cache = cache_bundle["full_metrics_cache"]
    prompt_caps = list(dict.fromkeys(candidate["max_prompt_tokens"] for candidate in candidates))
    choices, t1_facts = _plan_t1_selections(
        records, prompt_text, prompt_caps, tokenizer, prompt_cache, batch_size
    )
    selected_metrics = _ensure_selected_full_metrics(
        records, choices, prompt_caps, prompt_text, tokenizer, prompt_cache, full_cache, batch_size
    )
    length_by_id = {row["sample_id"]: row for row in length_rows}
    simulation_rows, exceptions, budget_summary = _classify_budget_rows(
        records, length_by_id, candidates, choices, selected_metrics
    )
    comparisons = _frontier_comparisons(simulation_rows, candidates)
    response_buckets = _response_char_buckets(records, config["simulation"].get("response_length_train_percentiles", [50, 90]))
    length_summary = _length_summaries(length_rows, response_buckets)
    deployment = _deployment_check(root, config, candidates)
    verified_hashes = _ensure_input_paths_stable(protected_input_paths, input_hashes_before)
    for key, path in protected_input_paths.items():
        if verified_hashes[key] != input_hashes_before[key]:
            raise RuntimeError(f"Input {key} changed during audit")

    outputs_cfg = config["outputs"]
    output_dir = _safe_path(root, outputs_cfg["directory"], "outputs.directory")
    report_path = _safe_path(root, outputs_cfg["report"], "outputs.report")
    final_files = {
        "manifest": output_dir / "manifest.json",
        "token_stats": output_dir / "token_stats.json",
        "per_sample_lengths": output_dir / "per_sample_lengths.jsonl",
        "budget_simulations": output_dir / "budget_simulations.jsonl",
        "length_exceptions": output_dir / "length_exceptions.jsonl",
        "full_histogram": output_dir / "full_sequence_length_by_split.svg",
        "completion_histogram": output_dir / "completion_length_by_split.svg",
        "retention_chart": output_dir / "history_retention_by_budget.svg",
    }
    output_paths = {key: f"../../{path.relative_to(root)}" for key, path in final_files.items() if key != "manifest"}
    protected_outputs = {**{name: path for name, path in final_files.items()}, "report": report_path}
    _reject_path_overlaps(protected_input_paths, protected_outputs)
    if report_path == output_dir or report_path.is_relative_to(output_dir) or output_dir.is_relative_to(report_path):
        raise ValueError("Report path and output directory must not overlap")
    if config_path in protected_outputs.values() or prompt_path in protected_outputs.values():
        raise ValueError("Output paths must not overwrite the config or P0 input")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir.with_name(f".{output_dir.name}.stage-{uuid.uuid4().hex}")
    stage_report = report_path.with_name(f".{report_path.name}.stage-{uuid.uuid4().hex}")
    stage_dir.mkdir()
    staged = {name: stage_dir / path.name for name, path in final_files.items()}
    try:
        with staged["per_sample_lengths"].open("wb") as stream:
            for row in length_rows:
                stream.write(_json_bytes(row))
        with staged["budget_simulations"].open("wb") as stream:
            for row in simulation_rows:
                stream.write(_json_bytes(row))
        with staged["length_exceptions"].open("wb") as stream:
            for row in exceptions:
                stream.write(_json_bytes(row))

        _svg_histogram(staged["full_histogram"], "完整训练序列 Token 长度分布", {
            split: [row["train_sequence_tokens_full"] for row in length_rows if row["split"] == split]
            for split in SPLITS
        })
        _svg_histogram(staged["completion_histogram"], "完整 Completion Token 长度分布", {
            split: [row["completion_tokens_full"] for row in length_rows if row["split"] == split]
            for split in SPLITS
        })
        _svg_retention(staged["retention_chart"], budget_summary)

        stats = {
            "audit_version": AUDIT_VERSION,
            "input_data": input_hashes,
            "counts": {split: sum(item["split"] == split for item in records) for split in SPLITS},
            "prompt": {
                "version": config["prompt"]["version"], "path": str(prompt_path.relative_to(root)),
                "sha256": prompt_sha, "history_format": config["prompt"]["history_format"],
                "completion_json": config["prompt"]["completion_json"],
                "memory_refs_policy": config["prompt"]["memory_refs_policy"],
            },
            "schema_validation": schema_facts,
            "tokenizer": {key: value for key, value in tokenizer_facts.items() if key != "snapshot_path"},
            "environment": {
                "python": platform.python_version(), "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
            },
            "length_statistics": length_summary,
            "response_length_buckets": {key: value for key, value in response_buckets.items() if key != "bucket"},
            "budget_simulations": budget_summary,
            "frontier_comparisons": comparisons,
            "deployment_budget_check": deployment,
            "tokenization_validation": {
                "scalar_direct_template_checks": cache_bundle["baseline_facts"]["scalar_direct_template_checks"],
                "additional_exact_t1_prompt_encodings": t1_facts["additional_exact_prompt_encodings"],
                "p0_text_tokens_independent_reference": cache_bundle["baseline_facts"]["p0_text_tokens_independent_reference"],
                "length_fields": LENGTH_FIELDS,
                "completion_token_boundary": "Exact full_ids prompt prefix; JSON body followed by tokenizer EOS and template whitespace only.",
                "qwen_literal_thinking_marker_policy": "Assistant history and target messages set reasoning_content to an empty string so literal <think>/</think> text stays ordinary content and is not parsed by the template.",
            },
            "resource_feasibility": "not_measured_no_model_weights_or_training_loaded",
        }
        staged["token_stats"].write_bytes(_json_bytes(stats, pretty=True))
        report = _report_markdown(
            config, input_hashes, schema_facts, tokenizer_facts, prompt_sha,
            length_summary, budget_summary, comparisons, response_buckets, deployment,
            stats["counts"], output_paths, validation_result,
        )
        stage_report.write_text(report, encoding="utf-8", newline="\n")
        source_status = _git_facts(root)
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        status_lines = [line for line in status_result.stdout.splitlines() if line]
        source_status["working_tree_dirty"] = bool(status_lines)
        source_status["untracked_file_count"] = sum(line.startswith("??") for line in status_lines)
        result_hashes = {
            key: {"path": str(final_files[key].relative_to(root)), "sha256": _sha256_file(staged[key])}
            for key in staged if key != "manifest"
        }
        manifest = {
            "audit_version": AUDIT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": source_status,
            "config": {"path": str(config_path.relative_to(root)), "sha256": _sha256_bytes(config_raw)},
            "p0": {"path": str(prompt_path.relative_to(root)), "sha256": prompt_sha},
            "d03_manifest": input_hashes["d03_manifest"],
            "normalized_inputs": {split: input_hashes[split] for split in SPLITS},
            "tokenizer": tokenizer_facts,
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            },
            "counts": stats["counts"],
            "candidate_grid": candidates,
            "result_files": result_hashes,
            "report": {"path": str(report_path.relative_to(root)), "sha256": _sha256_file(stage_report)},
            "validation": validation_result or {"status": "not_run_by_programmatic_api"},
            "input_hashes_unchanged_after_run": verified_hashes,
        }
        staged["manifest"].write_bytes(_json_bytes(manifest, pretty=True))
        _publish_transaction([(stage_dir, output_dir), (stage_report, report_path)])
    except Exception:
        _remove_path(stage_dir)
        stage_report.unlink(missing_ok=True)
        raise
    return {
        "counts": stats["counts"],
        "candidate_count": len(candidates),
        "output_dir": str(output_dir.relative_to(root)),
        "report": str(report_path.relative_to(root)),
        "budget_summary": budget_summary,
        "input_hashes_unchanged": True,
        "tokenizer": tokenizer_facts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the exact Qwen Token length audit and T1 budget simulation.")
    parser.add_argument("--config", default="configs/data_token_audit.yaml", help="YAML audit configuration")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the configured regression suite and record this fact")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = _safe_path(root, args.config, "--config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validation_result = None
    if args.skip_validation:
        validation_result = {"status": "skipped_by_cli_flag", "exit_code": None}
    else:
        validation_result = _run_validation(root, config["validation"]["unit_test_command"])
        if validation_result["exit_code"] != 0:
            print(f"Validation failed: {validation_result['summary']}", file=sys.stderr)
            return validation_result["exit_code"] or 1
    try:
        result = run_audit(root, config_path, validation_result=validation_result)
    except Exception as exc:
        print(f"D0.4.2 audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "counts": result["counts"], "candidate_count": result["candidate_count"],
        "output_dir": result["output_dir"], "report": result["report"],
        "input_hashes_unchanged": result["input_hashes_unchanged"],
        "tokenizer_revision": result["tokenizer"]["resolved_revision"],
        "budget_summary": {
            key: {
                "full_fit": value["n_full_fit"],
                "trimmed_fit": value["n_trimmed_fit"],
                "prompt_floor_exceeds_cap": value["n_prompt_floor_exceeds_cap"],
                "completion_overflows_train_budget": value["n_completion_overflows_train_budget"],
                "usable_rate": value["usable_rate"],
            }
            for key, value in result["budget_summary"].items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

