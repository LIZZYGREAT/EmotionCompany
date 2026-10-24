"""Build deterministic, pre-tokenized Completion-only SFT datasets for D0.4.3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml
from jsonschema import Draft202012Validator

from participant.run_inference import load_runtime_tokenizer
from src.shared.prompting import (
    build_messages,
    encode_full_training_sequence,
    select_recent_complete_suffix,
    serialize_completion,
    validate_completion_suffix,
    validate_prompt_enum_order,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OUTCOMES = {
    "full_fit",
    "trimmed_fit",
    "prompt_floor_exceeds_cap",
    "completion_overflows_train_budget",
}
TRAIN_SPLITS = ("train", "dev")
ALL_SPLITS = ("train", "dev", "val")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=pretty,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _safe_path(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} path escapes project root: {value}")
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    resolved = _safe_path(PROJECT_ROOT, str(path), "--config")
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "D0.4.3-v1":
        raise ValueError("config.version must be D0.4.3-v1")
    for name in ("inputs", "profile", "outputs", "validation"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"config.{name} must be an object")
    profile = config["profile"]
    for field in ("name", "max_train_seq_length", "max_prompt_tokens"):
        if field not in profile:
            raise ValueError(f"config.profile.{field} is required")
    seq = profile["max_train_seq_length"]
    prompt = profile["max_prompt_tokens"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
        raise ValueError("profile.max_train_seq_length must be a positive integer")
    if not isinstance(prompt, int) or isinstance(prompt, bool) or not 0 < prompt < seq:
        raise ValueError("profile.max_prompt_tokens must be a positive integer smaller than S")
    if not isinstance(profile["name"], str) or not profile["name"].strip():
        raise ValueError("profile.name must be a non-empty string")
    if profile.get("history_policy") != "T1_recent_contiguous_complete_message_suffix":
        raise ValueError("only the shared D0.4.2 T1 history policy is supported")
    if profile.get("completion_policy") != "complete_or_exclude":
        raise ValueError("completion_policy must be complete_or_exclude")
    if profile.get("packing") is not False:
        raise ValueError("packing must remain false in D0.4.3")
    if profile.get("padding_side") != "right":
        raise ValueError("padding_side must be right")
    multiple = profile.get("pad_to_multiple_of")
    if multiple is not None and (not isinstance(multiple, int) or isinstance(multiple, bool) or multiple <= 0):
        raise ValueError("pad_to_multiple_of must be null or a positive integer")
    batch_sizes = config["validation"].get("batch_sizes")
    if not isinstance(batch_sizes, list) or not batch_sizes or any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in batch_sizes
    ):
        raise ValueError("validation.batch_sizes must contain positive integers")
    required_inputs = {
        "normalized_dir", "d03_manifest", "d042_manifest", "d042_budget_rows",
        "train_schema", "prompt_path", "tokenizer_config", "runner_config",
    }
    if not required_inputs <= set(config["inputs"]):
        raise ValueError(f"config.inputs lacks keys: {sorted(required_inputs - set(config['inputs']))}")
    required_outputs = {"dataset_dir", "artifacts_dir", "report"}
    if not required_outputs <= set(config["outputs"]):
        raise ValueError(f"config.outputs lacks keys: {sorted(required_outputs - set(config['outputs']))}")
    return config


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON at column {exc.colno}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield line_number, value


def _manifest_output_record(manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    entries = [entry for entry in manifest.get("outputs", []) if entry.get("path") == relative_path]
    if len(entries) != 1:
        raise ValueError(f"D0.3 manifest must contain exactly one output record for {relative_path}")
    return entries[0]


def _load_pinned_context(
    root: Path,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any], Any, dict[str, Any], dict[str, Path]]:
    paths = {
        name: _safe_path(root, config["inputs"][name], f"inputs.{name}")
        for name in (
            "d03_manifest", "d042_manifest", "d042_budget_rows", "train_schema",
            "prompt_path", "tokenizer_config", "runner_config",
        )
    }
    normalized_dir = _safe_path(root, config["inputs"]["normalized_dir"], "inputs.normalized_dir")
    paths["normalized_train"] = normalized_dir / "train.jsonl"
    paths["normalized_dev"] = normalized_dir / "dev.jsonl"
    paths["normalized_val"] = normalized_dir / "val.jsonl"
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} input: {path}")

    d03 = json.loads(paths["d03_manifest"].read_text(encoding="utf-8"))
    d042 = json.loads(paths["d042_manifest"].read_text(encoding="utf-8"))
    if d03.get("pipeline_version") != "D0.3-v1" or d042.get("audit_version") != "D0.4.2-v1":
        raise ValueError("D0.3/D0.4.2 manifest versions do not match the configured v1 data")
    if d042.get("d03_manifest", {}).get("sha256") != sha256_file(paths["d03_manifest"]):
        raise ValueError("D0.4.2 manifest was produced from a different D0.3 manifest")

    schema_bytes = paths["train_schema"].read_bytes()
    schema = json.loads(schema_bytes.decode("utf-8"))
    Draft202012Validator.check_schema(schema)
    target_schema = {
        "$schema": schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$ref": "#/$defs/target",
        "$defs": schema["$defs"],
    }
    Draft202012Validator.check_schema(target_schema)
    schema_validator = Draft202012Validator(target_schema)
    prompt_bytes = paths["prompt_path"].read_bytes()
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    prompt = prompt_bytes.decode("utf-8")
    if prompt_sha != d042.get("p0", {}).get("sha256"):
        raise ValueError("P0 SHA-256 does not match the D0.4.2 manifest")
    validate_prompt_enum_order(prompt, schema)

    audit_config = yaml.safe_load(paths["tokenizer_config"].read_text(encoding="utf-8"))
    audit_tokenizer_config = audit_config.get("model", {})
    if (
        audit_tokenizer_config.get("tokenizer_id") != d042.get("tokenizer", {}).get("model_id")
        or audit_tokenizer_config.get("tokenizer_revision") != d042.get("tokenizer", {}).get("resolved_revision")
        or audit_tokenizer_config.get("enable_thinking") is not False
    ):
        raise ValueError("D0.4.2 tokenizer config does not match the frozen model/revision/non-thinking protocol")
    if sha256_file(paths["tokenizer_config"]) != d042.get("config", {}).get("sha256"):
        raise ValueError("D0.4.2 audit config SHA-256 differs from its manifest")
    participant_config = json.loads(paths["runner_config"].read_text(encoding="utf-8"))
    if participant_config.get("prompt", {}).get("sha256") != prompt_sha:
        raise ValueError("participant runner P0 fingerprint differs from the SFT P0")
    tokenizer, tokenizer_facts = load_runtime_tokenizer(participant_config)
    audit_tokenizer = d042.get("tokenizer", {})
    for field, config_field in (
        ("model_id", "model_id"),
        ("revision", "revision"),
        ("signature_sha256", "expected_signature_sha256"),
        ("template_sha256", "expected_template_sha256"),
        ("eos_token_id", "eos_token_id"),
    ):
        actual = tokenizer_facts.get(field)
        expected = audit_tokenizer.get(field)
        configured = participant_config["tokenizer"].get(config_field)
        if actual != expected or actual != configured:
            raise ValueError(f"locked Tokenizer field {field} differs between D0.4.2, runner, and runtime")

    hashes = {name: sha256_file(path) for name, path in paths.items()}
    d03_outputs = {}
    for split in ALL_SPLITS:
        rel_path = f"{config['inputs']['normalized_dir'].rstrip('/')}/{split}.jsonl"
        output = _manifest_output_record(d03, rel_path)
        d03_outputs[split] = output
        d042_input = d042.get("normalized_inputs", {}).get(split, {})
        key = f"{split}_jsonl"
        if hashes[f"normalized_{split}"] != output.get("sha256"):
            raise ValueError(f"{split} JSONL SHA-256 differs from the D0.3 manifest")
        if hashes[f"normalized_{split}"] != d042_input.get("sha256_before"):
            raise ValueError(f"{split} JSONL SHA-256 differs from the D0.4.2 audit")
        if d042.get("input_hashes_unchanged_after_run", {}).get(key) != hashes[f"normalized_{split}"]:
            raise ValueError(f"{split} JSONL SHA-256 differs from D0.4.2 unchanged-input evidence")
    if hashes["prompt_path"] != d042.get("input_hashes_unchanged_after_run", {}).get("p0"):
        raise ValueError("P0 SHA-256 differs from D0.4.2 unchanged-input evidence")
    if hashes["train_schema"] != d042.get("input_hashes_unchanged_after_run", {}).get("train_schema"):
        raise ValueError("Train schema SHA-256 differs from D0.4.2 unchanged-input evidence")
    if hashes["runner_config"] != d042.get("input_hashes_unchanged_after_run", {}).get("reference_inference_config"):
        raise ValueError("participant runner config SHA-256 differs from D0.4.2 unchanged-input evidence")
    if hashes["tokenizer_config"] != d042.get("input_hashes_unchanged_after_run", {}).get("audit_config"):
        raise ValueError("D0.4.2 config SHA-256 differs from unchanged-input evidence")
    if hashes["d03_manifest"] != d042.get("input_hashes_unchanged_after_run", {}).get("d03_manifest"):
        raise ValueError("D0.3 manifest SHA-256 differs from D0.4.2 unchanged-input evidence")

    budget_path = paths["d042_budget_rows"]
    budget_meta = d042.get("result_files", {}).get("budget_simulations", {})
    if sha256_file(budget_path) != budget_meta.get("sha256"):
        raise ValueError("D0.4.2 budget simulation file hash does not match its manifest")
    stats_meta = d042.get("result_files", {}).get("token_stats", {})
    stats_path = _safe_path(root, stats_meta.get("path", ""), "D0.4.2 token_stats")
    if not stats_path.is_file() or sha256_file(stats_path) != stats_meta.get("sha256"):
        raise ValueError("D0.4.2 token_stats file hash does not match its manifest")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return prompt, schema, tokenizer, {
        "schema_validator": schema_validator,
        "d03_manifest": d03,
        "d042_manifest": d042,
        "d042_stats": stats,
        "tokenizer_facts": tokenizer_facts,
        "input_hashes": hashes,
        "d03_outputs": d03_outputs,
    }, paths


def _load_budget_index(path: Path, seq: int, prompt_cap: int) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, row in iter_jsonl(path):
        if row.get("max_train_seq_length") != seq or row.get("max_prompt_tokens") != prompt_cap:
            continue
        key = (row.get("split"), row.get("sample_id"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError(f"{path}:{line_number}: candidate row lacks split/sample_id")
        if key in selected:
            raise ValueError(f"D0.4.2 contains duplicate budget row for {key}")
        if row.get("outcome") not in EXPECTED_OUTCOMES:
            raise ValueError(f"D0.4.2 has unknown budget outcome for {key}: {row.get('outcome')}")
        selected[key] = row
    if not selected:
        raise ValueError(f"D0.4.2 contains no candidate rows for S={seq}, P={prompt_cap}")
    return selected


def _derive_outcome(selection: dict[str, Any], full_length: int, seq: int) -> str:
    if selection["floor_exceeds_cap"]:
        return "prompt_floor_exceeds_cap"
    if full_length > seq:
        return "completion_overflows_train_budget"
    if selection["selected_start_index"] > 0:
        return "trimmed_fit"
    return "full_fit"


def build_training_item(
    sample: dict[str, Any],
    split: str,
    prompt_text: str,
    prompt_cap: int,
    sequence_cap: int,
    tokenizer: Any,
    official_schema: dict[str, Any],
    schema_validator: Draft202012Validator,
    d042_row: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{split}: normalized sample has no sample_id")
    source_split = sample.get("source", {}).get("split")
    expected_source_split = "val" if split == "val" else "train"
    if source_split != expected_source_split or not sample_id.startswith(f"{expected_source_split}:"):
        raise ValueError(f"{sample_id}: D0.3 source provenance does not match the normalized {split} partition")
    history = sample.get("history")
    target = sample.get("target")
    if not isinstance(history, list) or not history or not isinstance(target, dict):
        raise ValueError(f"{sample_id}: history/target violate the D0.3 normalized contract")
    schema_validator.validate(target)
    target_before = copy.deepcopy(target)

    selection = select_recent_complete_suffix(history, prompt_text, prompt_cap, tokenizer)
    messages = build_messages(selection["selected_history"], prompt_text)
    completion_json = serialize_completion(target, official_schema)
    serialized_target = json.loads(completion_json)
    if serialized_target["response_text"] != target["response_text"]:
        raise ValueError(f"{sample_id}: serialized response_text changed from D0.3 target")
    if serialized_target["memory_refs"] != []:
        raise ValueError(f"{sample_id}: A0 completion memory_refs must be empty")
    full_ids = encode_full_training_sequence(tokenizer, messages, completion_json)
    completion_ids = validate_completion_suffix(
        tokenizer, completion_json, selection["prompt_ids"], full_ids, sample_id=sample_id
    )
    if target != target_before:
        raise RuntimeError(f"{sample_id}: completion serialization mutated the D0.3 target")

    prompt_length = len(selection["prompt_ids"])
    full_length = len(full_ids)
    completion_length = len(completion_ids)
    removed_messages = selection["selected_start_index"]
    retained_messages = len(history) - removed_messages
    outcome = _derive_outcome(selection, full_length, sequence_cap)

    comparisons = {
        "outcome": outcome,
        "selected_prompt_tokens": prompt_length,
        "completion_tokens_full": completion_length,
        "selected_full_sequence_tokens": full_length,
        "removed_history_message_count": removed_messages,
        "retained_history_message_count": retained_messages,
        "source": sample["source"],
    }
    for field, actual in comparisons.items():
        if d042_row.get(field) != actual:
            raise ValueError(
                f"{sample_id}: D0.4.2 row mismatch for {field}: "
                f"builder={actual!r}, audit={d042_row.get(field)!r}"
            )
    if len(selection["prompt_ids"]) > prompt_cap and outcome != "prompt_floor_exceeds_cap":
        raise RuntimeError(f"{sample_id}: prompt exceeds P but was not classified as a floor failure")
    if len(full_ids) <= sequence_cap and outcome in {"full_fit", "trimmed_fit"}:
        if len(full_ids) > sequence_cap:
            raise RuntimeError(f"{sample_id}: usable sample exceeds S")
    if len(completion_ids) == 0 or all(token == -100 for token in completion_ids):
        raise RuntimeError(f"{sample_id}: usable completion has no supervised token")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None or int(eos_id) not in completion_ids:
        raise ValueError(f"{sample_id}: final assistant EOS is missing from the supervised completion")
    terminal_eos_offset = max(index for index, token in enumerate(completion_ids) if token == int(eos_id))
    labels = [-100] * prompt_length + completion_ids
    if labels[prompt_length] != completion_ids[0] or labels[prompt_length + terminal_eos_offset] != int(eos_id):
        raise RuntimeError(f"{sample_id}: first completion token or terminal EOS is not supervised")
    if len(labels) != full_length or labels[:prompt_length] != [-100] * prompt_length:
        raise RuntimeError(f"{sample_id}: Completion-only label boundaries are invalid")

    diagnostic = {
        "split": split,
        "sample_id": sample_id,
        "outcome": outcome,
        "prompt_length": prompt_length,
        "completion_length": completion_length,
        "full_length": full_length,
        "selected_start_index": selection["selected_start_index"],
        "removed_history_message_count": removed_messages,
    }
    if outcome in {"prompt_floor_exceeds_cap", "completion_overflows_train_budget"}:
        excluded = {
            "split": split,
            "sample_id": sample_id,
            "S": sequence_cap,
            "P": prompt_cap,
            "reason": outcome,
            "prompt_tokens": prompt_length,
            "completion_tokens": completion_length,
            "full_tokens": full_length,
            "source": copy.deepcopy(sample["source"]),
        }
        return None, excluded, diagnostic

    item = {
        "sample_id": sample_id,
        "conversation_id": sample["conversation_id"],
        "source_split": split,
        "source": copy.deepcopy(sample["source"]),
        "prompt_length": prompt_length,
        "completion_length": completion_length,
        "input_ids": full_ids,
        "labels": labels,
    }
    if len(item["input_ids"]) != len(item["labels"]) or len(item["input_ids"]) > sequence_cap:
        raise RuntimeError(f"{sample_id}: constructed training item violates sequence/label length invariants")
    return item, None, diagnostic


def _nearest_rank(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (percentile * len(ordered) + 99) // 100 - 1
    return ordered[max(0, index)]


def _length_summary(items: list[int]) -> dict[str, int | None]:
    return {
        "count": len(items),
        "p50": _nearest_rank(items, 50),
        "p90": _nearest_rank(items, 90),
        "p95": _nearest_rank(items, 95),
        "p99": _nearest_rank(items, 99),
        "max": max(items) if items else None,
    }


def _git_facts(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "working_tree_dirty": bool(status),
        "status_entry_count": len(status),
    }


def _publish_directories(items: list[tuple[Path, Path]]) -> None:
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for staged, final in items:
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                backup = final.with_name(f".{final.name}.backup-{uuid.uuid4().hex}")
                os.replace(final, backup)
                backups.append((final, backup))
            os.replace(staged, final)
            installed.append(final)
    except Exception:
        for final in reversed(installed):
            shutil.rmtree(final, ignore_errors=True)
        for final, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, final)
        raise
    for _, backup in backups:
        shutil.rmtree(backup, ignore_errors=True)


def build_sft_data(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    profile = config["profile"]
    seq = profile["max_train_seq_length"]
    prompt_cap = profile["max_prompt_tokens"]
    prompt, schema, tokenizer, context, paths = _load_pinned_context(root, config)
    d03 = context["d03_manifest"]
    d042 = context["d042_manifest"]
    audit_stats = context["d042_stats"]
    candidate_key = f"S{seq}_P{prompt_cap}"
    audit_candidate = audit_stats.get("budget_simulations", {}).get(candidate_key)
    if not isinstance(audit_candidate, dict):
        raise ValueError(f"D0.4.2 token_stats lacks candidate {candidate_key}")
    budget_rows = _load_budget_index(paths["d042_budget_rows"], seq, prompt_cap)

    output_dir = _safe_path(root, config["outputs"]["dataset_dir"], "outputs.dataset_dir")
    artifact_dir = _safe_path(root, config["outputs"]["artifacts_dir"], "outputs.artifacts_dir")
    report_path = _safe_path(root, config["outputs"]["report"], "outputs.report")
    input_paths = set(paths.values())
    input_paths.add(_safe_path(root, config["inputs"]["normalized_dir"], "inputs.normalized_dir"))
    for output_name, final in (("dataset_dir", output_dir), ("artifacts_dir", artifact_dir)):
        if any(final == path or final.is_relative_to(path) or path.is_relative_to(final) for path in input_paths):
            raise ValueError(f"outputs.{output_name} overlaps an input path")
    if output_dir == artifact_dir or output_dir.is_relative_to(artifact_dir) or artifact_dir.is_relative_to(output_dir):
        raise ValueError("outputs.dataset_dir and outputs.artifacts_dir must be disjoint")
    if report_path in input_paths:
        raise ValueError("outputs.report overlaps an input file")

    with tempfile.TemporaryDirectory(prefix=".d043-sft-", dir=root / "outputs") as temporary:
        stage_root = Path(temporary)
        stage_dataset = stage_root / "dataset"
        stage_artifacts = stage_root / "artifacts"
        stage_dataset.mkdir()
        stage_artifacts.mkdir()
        excluded_path = stage_artifacts / "excluded_samples.jsonl"
        excluded_handle = excluded_path.open("wb")
        split_summary: dict[str, Any] = {}
        audit_match_counts = Counter()
        all_ids: set[str] = set()
        output_counts = Counter()

        try:
            for split in ALL_SPLITS:
                input_path = paths[f"normalized_{split}"]
                expected_count = context["d03_outputs"][split]["records"]
                outcome_counts: Counter[str] = Counter()
                included_lengths: dict[str, list[int]] = {
                    "prompt": [], "completion": [], "full": [],
                }
                destination = stage_dataset / f"{split}.jsonl" if split in TRAIN_SPLITS else None
                dataset_handle = destination.open("wb") if destination is not None else None
                seen_split = 0
                try:
                    for _, sample in iter_jsonl(input_path):
                        seen_split += 1
                        sample_id = sample.get("sample_id")
                        if not isinstance(sample_id, str) or sample_id in all_ids:
                            raise ValueError(f"{split}: missing or duplicate sample_id {sample_id!r}")
                        all_ids.add(sample_id)
                        audit_row = budget_rows.get((split, sample_id))
                        if audit_row is None:
                            raise ValueError(f"{sample_id}: no matching D0.4.2 row for {candidate_key}")
                        item, excluded, diagnostic = build_training_item(
                            sample, split, prompt, prompt_cap, seq, tokenizer, schema,
                            context["schema_validator"], audit_row,
                        )
                        outcome_counts[diagnostic["outcome"]] += 1
                        audit_match_counts[split] += 1
                        if excluded is not None:
                            if split in TRAIN_SPLITS:
                                excluded_handle.write(_json_bytes(excluded))
                        elif item is not None:
                            if dataset_handle is not None:
                                dataset_handle.write(_json_bytes(item))
                                output_counts[split] += 1
                            included_lengths["prompt"].append(item["prompt_length"])
                            included_lengths["completion"].append(item["completion_length"])
                            included_lengths["full"].append(len(item["input_ids"]))
                        else:
                            raise RuntimeError(f"{sample_id}: no item or exclusion disposition")
                finally:
                    if dataset_handle is not None:
                        dataset_handle.close()
                if seen_split != expected_count:
                    raise ValueError(f"{split}: row count {seen_split} differs from D0.3 manifest {expected_count}")
                if sum(outcome_counts.values()) != expected_count:
                    raise RuntimeError(f"{split}: D0.4.2 outcome counts do not conserve input rows")
                expected_audit = audit_candidate["by_split"][split]
                for outcome, audit_field in (
                    ("full_fit", "n_full_fit"),
                    ("trimmed_fit", "n_trimmed_fit"),
                    ("prompt_floor_exceeds_cap", "n_prompt_floor_exceeds_cap"),
                    ("completion_overflows_train_budget", "n_completion_overflows_train_budget"),
                ):
                    if outcome_counts[outcome] != expected_audit[audit_field]:
                        raise ValueError(
                            f"{split}: {outcome} count {outcome_counts[outcome]} differs from D0.4.2 "
                            f"{expected_audit[audit_field]}"
                        )
                usable_count = outcome_counts["full_fit"] + outcome_counts["trimmed_fit"]
                if split in TRAIN_SPLITS and output_counts[split] != usable_count:
                    raise RuntimeError(f"{split}: output row count differs from usable outcome count")
                split_summary[split] = {
                    "input_count": seen_split,
                    "outcomes": {outcome: outcome_counts[outcome] for outcome in sorted(EXPECTED_OUTCOMES)},
                    "sft_usable_count": usable_count,
                    "excluded_count": outcome_counts["prompt_floor_exceeds_cap"] + outcome_counts["completion_overflows_train_budget"],
                    "output_count": output_counts[split] if split in TRAIN_SPLITS else 0,
                    "included_length_scope": "SFT-usable samples only" if split in TRAIN_SPLITS else "audit-only; Val not emitted",
                    "prompt_length_tokens": _length_summary(included_lengths["prompt"]),
                    "completion_length_tokens": _length_summary(included_lengths["completion"]),
                    "full_length_tokens": _length_summary(included_lengths["full"]),
                    "counts_conserve": sum(outcome_counts.values()) == seen_split,
                    "d042_rows_compared": audit_match_counts[split],
                    "d042_rows_match": audit_match_counts[split] == seen_split,
                }
        finally:
            excluded_handle.close()

        consumed_ids = set(all_ids)
        if consumed_ids != {sample_id for _, sample_id in budget_rows}:
            raise ValueError("D0.4.2 selected candidate rows do not match D0.3 sample IDs")
        expected_total = sum(context["d042_manifest"]["counts"].values())
        if len(budget_rows) != expected_total:
            raise ValueError(f"D0.4.2 candidate rows {len(budget_rows)} != expected total {expected_total}")

        input_hashes_after = {name: sha256_file(path) for name, path in paths.items()}
        if context["input_hashes"] != input_hashes_after:
            raise RuntimeError("one or more locked inputs changed during SFT construction")
        summary = {
            "version": "D0.4.3-v1",
            "profile": copy.deepcopy(profile),
            "candidate": {"S": seq, "P": prompt_cap, "candidate_only": True},
            "input_counts": {split: context["d03_outputs"][split]["records"] for split in ALL_SPLITS},
            "split_summary": split_summary,
            "total_input_count": sum(item["input_count"] for item in split_summary.values()),
            "train_output_count": output_counts["train"],
            "dev_output_count": output_counts["dev"],
            "val_output_count": 0,
            "val_used_for_training": False,
            "d042_candidate": candidate_key,
            "d042_rows_compared": len(budget_rows),
            "d042_rows_match": len(budget_rows) == expected_total,
            "d03_input_hashes": {
                split: context["input_hashes"][f"normalized_{split}"] for split in ALL_SPLITS
            },
            "locked_p0_sha256": context["input_hashes"]["prompt_path"],
            "tokenizer": context["tokenizer_facts"],
            "schema_sha256": context["input_hashes"]["train_schema"],
            "count_conservation": all(item["counts_conserve"] for item in split_summary.values()),
        }
        (stage_artifacts / "summary.json").write_bytes(_json_bytes(summary, pretty=True))

        staged_files = {
            "train.jsonl": stage_dataset / "train.jsonl",
            "dev.jsonl": stage_dataset / "dev.jsonl",
            "excluded_samples.jsonl": excluded_path,
            "summary.json": stage_artifacts / "summary.json",
        }
        hashes = {name: sha256_file(path) for name, path in staged_files.items()}
        output_dir_hash = hashlib.sha256(
            "\n".join(f"{name}:{hashes[name]}" for name in sorted(hashes) if name in {"train.jsonl", "dev.jsonl"}).encode()
        ).hexdigest()
        manifest = {
            "version": "D0.4.3-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": _git_facts(root),
            "config": {
                "path": str(config.get("_config_path", "configs/sft_data.yaml")),
                "sha256": config.get("_config_sha256"),
                "profile": copy.deepcopy(profile),
            },
            "inputs": {
                name: {"path": str(path.relative_to(root)), "sha256": context["input_hashes"][name]}
                for name, path in paths.items()
            },
            "d03_manifest_sha256": context["input_hashes"]["d03_manifest"],
            "d042_manifest_sha256": context["input_hashes"]["d042_manifest"],
            "p0_sha256": summary["locked_p0_sha256"],
            "tokenizer": context["tokenizer_facts"],
            "environment": {
                "python": platform.python_version(),
                "transformers": importlib.metadata.version("transformers"),
                "trl": importlib.metadata.version("trl"),
                "datasets": importlib.metadata.version("datasets"),
            },
            "outputs": {
                "dataset_dir": str(output_dir.relative_to(root)),
                "artifacts_dir": str(artifact_dir.relative_to(root)),
                "dataset_content_sha256": output_dir_hash,
                "files": {
                    name: {"path": str((output_dir / name).relative_to(root)) if name.endswith(".jsonl") and name in {"train.jsonl", "dev.jsonl"} else str((artifact_dir / name).relative_to(root)), "sha256": digest}
                    for name, digest in hashes.items()
                },
            },
            "validation": {
                "input_hashes_unchanged": True,
                "all_d03_rows_have_one_d042_match": len(budget_rows) == expected_total,
                "all_splits_conserve_counts": summary["count_conservation"],
                "trainer_batch_path": "pending_C_D_validation",
            },
        }
        (stage_artifacts / "manifest.json").write_bytes(_json_bytes(manifest, pretty=True))
        report_path.parent.mkdir(parents=True, exist_ok=True)

        _publish_directories([(stage_dataset, output_dir), (stage_artifacts, artifact_dir)])
        report = {
            "dataset_dir": str(output_dir),
            "artifacts_dir": str(artifact_dir),
            "summary": summary,
            "manifest": manifest,
        }
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify D0.4.3 pre-tokenized Completion-only SFT data")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_data.yaml"))
    parser.add_argument("--profile-name")
    parser.add_argument("--max-train-seq-length", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--report")
    return parser.parse_args(argv)


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    overrides = {
        "name": args.profile_name,
        "max_train_seq_length": args.max_train_seq_length,
        "max_prompt_tokens": args.max_prompt_tokens,
    }
    changed = False
    for key, value in overrides.items():
        if value is not None:
            changed = changed or value != effective["profile"].get(key)
            effective["profile"][key] = value
    if changed and (args.dataset_dir is None or args.artifacts_dir is None):
        raise ValueError("changing profile/S/P requires explicit --dataset-dir and --artifacts-dir")
    if args.dataset_dir is not None:
        effective["outputs"]["dataset_dir"] = args.dataset_dir
    if args.artifacts_dir is not None:
        effective["outputs"]["artifacts_dir"] = args.artifacts_dir
    if args.report is not None:
        effective["outputs"]["report"] = args.report
    seq = effective["profile"]["max_train_seq_length"]
    prompt = effective["profile"]["max_prompt_tokens"]
    if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
        raise ValueError("max_train_seq_length must be a positive integer")
    if not isinstance(prompt, int) or isinstance(prompt, bool) or not 0 < prompt < seq:
        raise ValueError("max_prompt_tokens must be a positive integer smaller than S")
    if not isinstance(effective["profile"]["name"], str) or not effective["profile"]["name"].strip():
        raise ValueError("profile name must be a non-empty string")
    return effective


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config_path = _safe_path(PROJECT_ROOT, str(args.config), "--config")
        config = load_config(config_path)
        config["_config_path"] = str(config_path.relative_to(PROJECT_ROOT))
        config["_config_sha256"] = sha256_file(config_path)
        config = _apply_overrides(config, args)
        result = build_sft_data(PROJECT_ROOT, config)
    except Exception as exc:
        print(f"D0.4.3 SFT build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "dataset_dir": result["dataset_dir"],
        "artifacts_dir": result["artifacts_dir"],
        "train_output_count": result["summary"]["train_output_count"],
        "dev_output_count": result["summary"]["dev_output_count"],
        "d042_rows_compared": result["summary"]["d042_rows_compared"],
        "d042_rows_match": result["summary"]["d042_rows_match"],
        "counts": result["summary"]["input_counts"],
        "dataset_content_sha256": result["manifest"]["outputs"]["dataset_content_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
