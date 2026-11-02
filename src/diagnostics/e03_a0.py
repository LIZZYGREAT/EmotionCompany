"""Read-only E03-A0 supervision and teacher-forced loss diagnostics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from participant import run_inference as inference
from src.shared.prompting import (
    build_messages,
    encode_full_training_sequence,
    encode_inference_prompt,
    select_recent_complete_suffix,
    serialize_completion,
)
from src.training.training_sft import validate_train_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "E03-A0-v1"
TOP_REGIONS = (
    "emotion_label",
    "user_profile",
    "response_text",
    "memory_refs",
    "public_structure",
    "eos",
    "boundary_mixed",
)
PROFILE_REGIONS = (
    "personality_traits",
    "interests",
    "style",
    "profile_structure",
    "boundary_mixed",
)
PROFILE_DIMS = ("personality_traits", "interests", "style")
SELECTION_STATUSES = ("full_fit", "trimmed_fit")


class AlignmentError(ValueError):
    """The source sequence cannot be assigned to fields without guessing."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, int, dict[str, Any]]]:
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: empty JSONL line")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield line_number, offset, value


def validate_sample_cap(mode: str, sample_cap: int, max_gpu_sample_cap: int) -> None:
    if mode not in {"token", "ce"}:
        raise ValueError("mode must be token or ce")
    if sample_cap < 0:
        raise ValueError("sample-cap must be zero or positive")
    if mode == "ce" and sample_cap == 0:
        raise ValueError("CE mode refuses sample-cap=0 to prevent full-train inference")
    if mode == "ce" and sample_cap > max_gpu_sample_cap:
        raise ValueError("CE sample-cap exceeds the configured GPU safety limit")


def create_run_dir(output_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id contains unsupported characters")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _jsonl_count(path: Path) -> int:
    return sum(1 for _ in _iter_jsonl(path))


def _read_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.readline()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("indexed JSONL line is not an object")
    return value


def _manifest_output(manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    outputs = manifest.get("outputs", [])
    if isinstance(outputs, dict):
        outputs = outputs.get("files", {}).values()
    matches = [item for item in outputs if item.get("path") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"manifest must have one output record for {relative_path}")
    return matches[0]


def _git_facts(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "head": run("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "working_tree_status_entries": len(status.splitlines()) if status else 0,
    }


def _package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "peft", "bitsandbytes", "tokenizers", "PyYAML")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return value


def load_config(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load_yaml(config_path)
    if config.get("version") != VERSION:
        raise ValueError(f"diagnostic config version must be {VERSION}")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("config.inputs must be a mapping")
    paths = {name: _resolve(root, value) for name, value in inputs.items() if name != "output_root"}
    paths["output_root"] = _resolve(root, inputs["output_root"])
    required = {
        "normalized_train", "sft_train", "d03_manifest", "sft_manifest", "d042_manifest",
        "sft_summary", "sft_excluded", "d042_budget_rows", "train_schema", "prompt", "tokenizer_config", "runner_config",
        "training_config", "training_run_root", "output_root",
    }
    if not required <= set(paths):
        raise ValueError(f"config inputs missing: {sorted(required - set(paths))}")
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(f"configured input missing: {name}={path}")
    return config, paths


def verify_source_identity(
    root: Path, config: dict[str, Any], paths: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify D0.3, D0.4.2, D0.4.3, and Full B identity before reading rows."""
    d03 = json.loads(paths["d03_manifest"].read_text(encoding="utf-8"))
    sft = json.loads(paths["sft_manifest"].read_text(encoding="utf-8"))
    d042 = json.loads(paths["d042_manifest"].read_text(encoding="utf-8"))
    training = json.loads((paths["training_run_root"] / "manifest.json").read_text(encoding="utf-8"))
    training_config = _load_yaml(paths["training_config"])
    sft_summary_meta = sft.get("outputs", {}).get("files", {}).get("summary.json", {})
    sft_summary_path = _resolve(root, sft_summary_meta.get("path", ""))
    if sft_summary_path != paths["sft_summary"]:
        raise ValueError("configured D0.4.3 summary path differs from its Manifest")
    if not sft_summary_path.is_file() or sha256_file(sft_summary_path) != sft_summary_meta.get("sha256"):
        raise ValueError("D0.4.3 summary is missing or differs from its Manifest SHA-256")
    sft_summary = json.loads(sft_summary_path.read_text(encoding="utf-8"))
    excluded_meta = sft.get("outputs", {}).get("files", {}).get("excluded_samples.jsonl", {})
    excluded_path = _resolve(root, excluded_meta.get("path", ""))
    if excluded_path != paths["sft_excluded"]:
        raise ValueError("configured D0.4.3 exclusion ledger path differs from its Manifest")
    if not excluded_path.is_file() or sha256_file(excluded_path) != excluded_meta.get("sha256"):
        raise ValueError("D0.4.3 exclusion ledger is missing or differs from its Manifest SHA-256")
    excluded_counts: Counter[str] = Counter()
    excluded_train_count = 0
    for _line, _offset, excluded in _iter_jsonl(excluded_path):
        if excluded.get("split") == "train":
            excluded_train_count += 1
            excluded_counts[str(excluded.get("reason", "missing_reason"))] += 1

    hashes = {
        name: sha256_file(paths[name])
        for name in (
            "normalized_train", "sft_train", "d03_manifest", "sft_manifest", "d042_manifest",
            "d042_budget_rows", "train_schema", "prompt", "tokenizer_config", "runner_config",
            "training_config", "sft_summary", "sft_excluded",
        )
    }
    hashes["training_run_manifest"] = sha256_file(paths["training_run_root"] / "manifest.json")

    if d03.get("pipeline_version") != "D0.3-v1" or d03.get("conservation_valid") is not True:
        raise ValueError("D0.3 manifest is not the accepted v1 pipeline output")
    d03_train = _manifest_output(d03, "data/processed/v1/train.jsonl")
    if hashes["normalized_train"] != d03_train.get("sha256"):
        raise ValueError("normalized D0.3 Train SHA-256 differs from its Manifest")
    if hashes["d03_manifest"] != sft.get("d03_manifest_sha256"):
        raise ValueError("D0.4.3 Manifest refers to another D0.3 Manifest")
    if sft.get("version") != "D0.4.3-v1":
        raise ValueError("D0.4.3 Manifest version differs from expected v1")
    sft_file = sft.get("outputs", {}).get("files", {}).get("train.jsonl", {})
    if hashes["sft_train"] != sft_file.get("sha256"):
        raise ValueError("SFT Train SHA-256 differs from its D0.4.3 Manifest")
    if hashes["sft_train"] != training.get("train_sha256"):
        raise ValueError("Full B Run used a different SFT Train file")
    if training.get("run_id") != config["model"]["training_run_id"]:
        raise ValueError("Full B Run ID differs from the diagnostic config")
    if training.get("status") != "passed" or training.get("stage") != "completed":
        raise ValueError("Full B Run is not a completed, passed Run")
    if training.get("training_scope") != "full_train":
        raise ValueError("configured Full B Run is not full_train")
    if training.get("model_facts", {}).get("load_in_4bit") is not True:
        raise ValueError("Full B Run Manifest does not confirm NF4/4-bit loading")
    if training.get("model_facts", {}).get("quant_type") != "nf4":
        raise ValueError("Full B Run Manifest does not confirm NF4")
    if training.get("model_facts", {}).get("compute_dtype") != "bfloat16":
        raise ValueError("Full B Run Manifest does not confirm BF16 compute")
    if hashes["prompt"] != sft.get("p0_sha256"):
        raise ValueError("P0 SHA-256 differs from the D0.4.3 Manifest")
    tokenizer = sft.get("tokenizer", {})
    runner = json.loads(paths["runner_config"].read_text(encoding="utf-8"))
    d042_tokenizer = d042.get("tokenizer", {})
    build_runner_sha = sft.get("inputs", {}).get("runner_config", {}).get("sha256")
    if build_runner_sha != d042.get("input_hashes_unchanged_after_run", {}).get("reference_inference_config"):
        raise ValueError("D0.4.2/D0.4.3 historical runner config hashes differ")
    if tokenizer.get("signature_sha256") != runner.get("tokenizer", {}).get("expected_signature_sha256"):
        raise ValueError("D0.4.3 and runner Tokenizer signatures differ")
    if tokenizer.get("template_sha256") != runner.get("tokenizer", {}).get("expected_template_sha256"):
        raise ValueError("D0.4.3 and runner Chat Template signatures differ")
    for field_name, expected in (
        ("model_id", tokenizer.get("model_id")),
        ("revision", tokenizer.get("revision")),
        ("local_files_only", True),
        ("eos_token_id", tokenizer.get("eos_token_id")),
    ):
        if runner.get("tokenizer", {}).get(field_name) != expected:
            raise ValueError(f"current runner Tokenizer {field_name} differs from D0.4.3")
    if runner.get("tokenizer", {}).get("files_sha256") != tokenizer.get("tokenizer_files"):
        raise ValueError("current runner Tokenizer file hashes differ from D0.4.3")
    if runner.get("prompt", {}).get("sha256") != hashes["prompt"]:
        raise ValueError("current runner P0 identity differs from the protected prompt")
    if tokenizer.get("signature_sha256") != d042_tokenizer.get("signature_sha256"):
        raise ValueError("D0.4.2/D0.4.3 Tokenizer signatures differ")
    if hashes["train_schema"] != d042.get("input_hashes_unchanged_after_run", {}).get("train_schema"):
        raise ValueError("Train schema SHA-256 differs from D0.4.2 unchanged-input evidence")
    if hashes["prompt"] != d042.get("input_hashes_unchanged_after_run", {}).get("p0"):
        raise ValueError("P0 SHA-256 differs from D0.4.2 unchanged-input evidence")
    if hashes["tokenizer_config"] != d042.get("input_hashes_unchanged_after_run", {}).get("audit_config"):
        raise ValueError("Tokenizer config SHA-256 differs from D0.4.2 unchanged-input evidence")
    if hashes["d03_manifest"] != d042.get("input_hashes_unchanged_after_run", {}).get("d03_manifest"):
        raise ValueError("D0.3 Manifest SHA-256 differs from D0.4.2 unchanged-input evidence")
    if d042.get("result_files", {}).get("budget_simulations", {}).get("sha256") != hashes["d042_budget_rows"]:
        raise ValueError("D0.4.2 budget simulation SHA-256 differs from its Manifest")
    expected_adapter = training.get("adapter_identity", {})
    if expected_adapter.get("weight_sha256") != config["model"].get("expected_adapter_weight_sha256"):
        raise ValueError("Full B Adapter identity differs from the diagnostic config")
    if expected_adapter.get("config_sha256") != config["model"].get("expected_adapter_config_sha256"):
        raise ValueError("Full B Adapter config identity differs from the diagnostic config")
    if training.get("base_weight_sha256") != config["model"].get("expected_base_weight_sha256"):
        raise ValueError("Full B Base identity differs from the diagnostic config")
    if training_config.get("base_model", {}).get("expected_weight_sha256") != training.get("base_weight_sha256"):
        raise ValueError("Full B training config Base identity differs from its Run Manifest")

    expected_counts = config.get("expected_counts", {})
    d03_count = int(d03_train.get("records", -1))
    sft_count = int(sft_summary.get("train_output_count", -1))
    if d03_count != expected_counts.get("d03_train_rows"):
        raise ValueError(f"D0.3 Train count {d03_count} differs from configured expectation")
    if sft_count != expected_counts.get("sft_train_rows"):
        raise ValueError(f"SFT Train count {sft_count} differs from configured expectation")
    if excluded_train_count != d03_count - sft_count:
        raise ValueError("D0.4.3 Train exclusions do not conserve D0.3-to-SFT counts")
    if excluded_counts != Counter({"completion_overflows_train_budget": d03_count - sft_count}):
        raise ValueError("D0.4.3 Train exclusions differ from the recorded completion budget reason")
    if training.get("training_rows") != sft_count:
        raise ValueError("Full B Run training row count differs from D0.4.3 SFT Train")

    facts = {
        "d03_version": d03.get("pipeline_version"),
        "d03_manifest_sha256": hashes["d03_manifest"],
        "d03_train_sha256": hashes["normalized_train"],
        "d03_train_rows": d03_count,
        "sft_version": sft.get("version"),
        "sft_manifest_sha256": hashes["sft_manifest"],
        "sft_train_sha256": hashes["sft_train"],
        "sft_train_rows": sft_count,
        "sft_summary_sha256": sft_summary_meta.get("sha256"),
        "excluded_d03_train_rows": d03_count - sft_count,
        "excluded_train_reason_counts": dict(sorted(excluded_counts.items())),
        "excluded_train_reason_counts": dict(sorted(excluded_counts.items())),
        "excluded_ledger_sha256": excluded_meta.get("sha256"),
        "d042_manifest_sha256": hashes["d042_manifest"],
        "d042_budget_sha256": hashes["d042_budget_rows"],
        "p0_sha256": hashes["prompt"],
        "train_schema_sha256": hashes["train_schema"],
        "tokenizer_config_sha256": hashes["tokenizer_config"],
        "runner_config_sha256": hashes["runner_config"],
        "runner_config_sha256_at_sft_build": build_runner_sha,
        "runner_config_current_hash_matches_build": hashes["runner_config"] == build_runner_sha,
        "runner_config_locked_protocol_fields_match": True,
        "tokenizer_signature_sha256": tokenizer.get("signature_sha256"),
        "chat_template_sha256": tokenizer.get("template_sha256"),
        "tokenizer_model_id": tokenizer.get("model_id"),
        "tokenizer_revision": tokenizer.get("revision"),
        "eos_token_id": tokenizer.get("eos_token_id"),
        "training_run_id": training.get("run_id"),
        "training_run_manifest_sha256": hashes["training_run_manifest"],
        "training_config_sha256": training.get("config_sha256"),
        "base_weight_sha256": training.get("base_weight_sha256"),
        "adapter_weight_sha256": expected_adapter.get("weight_sha256"),
        "adapter_config_sha256": expected_adapter.get("config_sha256"),
        "protocol_version": "P0/F2/T1",
        "max_prompt_tokens": int(sft.get("config", {}).get("profile", {}).get("max_prompt_tokens", 0)),
        "max_train_sequence_length": int(sft.get("config", {}).get("profile", {}).get("max_train_seq_length", 0)),
    }
    if facts["max_prompt_tokens"] <= 0 or facts["max_train_sequence_length"] <= 0:
        raise ValueError("D0.4.3 Manifest is missing the locked S/P token budgets")
    if facts["max_prompt_tokens"] != config["protocol"]["max_prompt_tokens"]:
        raise ValueError("configured Prompt budget differs from D0.4.3")
    if facts["max_train_sequence_length"] != config["protocol"]["max_train_sequence_length"]:
        raise ValueError("configured sequence budget differs from D0.4.3")
    return facts, {"d03": d03, "sft": sft, "d042": d042, "training": training, "training_config": training_config, "hashes": hashes}


def _build_line_index(path: Path, expected_rows: int, id_name: str = "sample_id") -> dict[str, int]:
    index: dict[str, int] = {}
    count = 0
    for _line, offset, row in _iter_jsonl(path):
        sample_id = row.get(id_name)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}: row {count + 1} has no valid sample_id")
        if sample_id in index:
            raise ValueError(f"{path}: duplicate sample_id at row {count + 1}")
        index[sample_id] = offset
        count += 1
    if count != expected_rows:
        raise ValueError(f"{path}: row count {count} differs from verified Manifest {expected_rows}")
    return index


def _load_budget_index(path: Path, train_rows: int, sequence_cap: int, prompt_cap: int) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for _line, _offset, row in _iter_jsonl(path):
        if (
            row.get("split") != "train"
            or row.get("max_train_seq_length") != sequence_cap
            or row.get("max_prompt_tokens") != prompt_cap
        ):
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise ValueError("D0.4.2 has a missing or duplicate Train sample in the selected S/P profile")
        by_id[sample_id] = row.get("outcome")
    if len(by_id) != train_rows:
        raise ValueError(f"D0.4.2 S/P index has {len(by_id)} Train rows; expected {train_rows}")
    return by_id


def _candidate_metadata(
    paths: dict[str, Path],
    identity: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    d03_index = _build_line_index(paths["normalized_train"], identity["d03_train_rows"])
    budget = _load_budget_index(
        paths["d042_budget_rows"],
        identity["d03_train_rows"],
        identity["max_train_sequence_length"],
        identity["max_prompt_tokens"],
    )
    records: list[dict[str, Any]] = []
    sft_seen: set[str] = set()
    for _line, _offset, sft_row in _iter_jsonl(paths["sft_train"]):
        sample_id = sft_row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in sft_seen:
            raise ValueError("SFT Train has a missing or duplicate sample_id")
        sft_seen.add(sample_id)
        d03_offset = d03_index.get(sample_id)
        if d03_offset is None:
            raise ValueError("SFT Train sample does not map to D0.3 Train")
        source = _read_at(paths["normalized_train"], d03_offset)
        if source.get("conversation_id") != sft_row.get("conversation_id"):
            raise ValueError("SFT/D0.3 conversation_id differs for a matched sample")
        target = source.get("target")
        if not isinstance(target, dict):
            raise ValueError("D0.3 target is not an object")
        profile = target.get("user_profile")
        if not isinstance(profile, dict):
            raise ValueError("D0.3 user_profile is not an object")
        dim_nonempty = {dim: bool(profile.get(dim)) for dim in PROFILE_DIMS}
        outcome = budget.get(sample_id)
        if outcome not in SELECTION_STATUSES:
            raise ValueError("valid SFT sample has no full_fit/trimmed_fit D0.4.2 outcome")
        records.append({
            "sample_id": sample_id,
            "sample_id_offset": _offset,
            "normalized_offset": d03_offset,
            "conversation_id": sft_row.get("conversation_id"),
            "source_split": sft_row.get("source_split"),
            "prompt_length": sft_row.get("prompt_length"),
            "completion_length": sft_row.get("completion_length"),
            "sequence_length": len(sft_row.get("input_ids", [])),
            "profile_all_empty": not any(dim_nonempty.values()),
            "dim_nonempty": dim_nonempty,
            "fit_outcome": outcome,
        })
    if len(records) != identity["sft_train_rows"]:
        raise ValueError("SFT Train candidate count differs from verified Manifest")
    if len(sft_seen) != len(d03_index) - identity["excluded_d03_train_rows"]:
        raise ValueError("D0.3/SFT Train matched row count differs from expected exclusions")
    # The index only stores IDs and offsets. Drop it before model/tokenizer work.
    del d03_index
    return records, budget, {}


def _length_median(records: list[dict[str, Any]]) -> float:
    ordered = sorted(int(row["completion_length"]) for row in records)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _selection_memberships(row: dict[str, Any], length_median: float) -> tuple[str, ...]:
    groups = ["profile_all_empty" if row["profile_all_empty"] else "profile_any_nonempty"]
    for dim in PROFILE_DIMS:
        groups.append(f"{dim}_gold_nonempty" if row["dim_nonempty"][dim] else f"{dim}_gold_empty")
    groups.append(row["fit_outcome"])
    groups.append("short_completion" if int(row["completion_length"]) <= length_median else "long_completion")
    return tuple(groups)


def deterministic_select(
    records: list[dict[str, Any]], *, cap: int, seed: int, stratified: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cap <= 0:
        raise ValueError("sample cap must be positive for selection")
    if not records:
        raise ValueError("cannot select from an empty candidate set")
    cap = min(cap, len(records))
    order_key = lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode("utf-8")).hexdigest()
    ordered = sorted(records, key=order_key)
    if not stratified:
        selected = ordered[:cap]
    else:
        median = _length_median(records)
        memberships = {row["sample_id"]: set(_selection_memberships(row, median)) for row in records}
        universe = set().union(*memberships.values())
        population = Counter(group for value in memberships.values() for group in value)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        covered: set[str] = set()
        # Greedy set cover gives scarce profile/fit/length strata a chance to appear.
        while len(selected) < cap and covered != universe:
            choices = [row for row in ordered if row["sample_id"] not in selected_ids]
            best = max(
                choices,
                key=lambda row: (
                    sum(1.0 / population[group] for group in memberships[row["sample_id"]] - covered),
                    sum(group not in covered for group in memberships[row["sample_id"]]),
                    tuple(-int(char, 16) for char in order_key(row)[:8]),
                ),
                default=None,
            )
            if best is None or not (memberships[best["sample_id"]] - covered):
                break
            selected.append(best)
            selected_ids.add(best["sample_id"])
            covered.update(memberships[best["sample_id"]])
        for row in ordered:
            if len(selected) >= cap:
                break
            if row["sample_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["sample_id"])
    selected_ids = {row["sample_id"] for row in selected}
    median = _length_median(records)
    selected_memberships = Counter(
        group
        for row in selected
        for group in _selection_memberships(row, median)
    )
    available_memberships = Counter(
        group for row in records for group in _selection_memberships(row, median)
    )
    summary = {
        "selection_rule": "sha256_seeded_greedy_strata_then_sha256_fill_v1" if stratified else "sha256_seeded_sample_id_order_v1",
        "seed": seed,
        "candidate_rows": len(records),
        "selected_rows": len(selected),
        "selected_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in selected]),
        "selection_is_population_representative": False,
        "length_median_tokens": _length_median(records),
        "strata": {
            key: {
                "available": available_memberships[key],
                "selected": selected_memberships[key],
            }
            for key in sorted(available_memberships)
        },
    }
    return selected, summary


def _skip_json_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _parse_json_string(text: str, index: int) -> tuple[str, int, int]:
    if index >= len(text) or text[index] != '"':
        raise AlignmentError("JSON string key/value expected at parsed offset")
    start = index
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
        elif char == '"':
            end = index + 1
            try:
                value = json.loads(text[start:end])
            except json.JSONDecodeError as exc:
                raise AlignmentError("invalid JSON string in serialized completion") from exc
            return value, start, end
        else:
            index += 1
    raise AlignmentError("unterminated JSON string in serialized completion")


def _parse_json_node(text: str, index: int) -> tuple[dict[str, Any], int]:
    index = _skip_json_ws(text, index)
    if index >= len(text):
        raise AlignmentError("unexpected end while parsing serialized completion")
    start = index
    char = text[index]
    if char == '"':
        value, _key_start, end = _parse_json_string(text, index)
        return {"kind": "string", "start": start, "end": end, "value": value}, end
    if char == "{":
        index += 1
        members: dict[str, dict[str, Any]] = {}
        index = _skip_json_ws(text, index)
        while index < len(text) and text[index] != "}":
            key, key_start, key_end = _parse_json_string(text, index)
            if key in members:
                raise AlignmentError("duplicate JSON object key in serialized completion")
            index = _skip_json_ws(text, key_end)
            if index >= len(text) or text[index] != ":":
                raise AlignmentError("missing colon in serialized completion")
            value_node, value_end = _parse_json_node(text, index + 1)
            members[key] = {
                "key_start": key_start,
                "key_end": key_end,
                "entry_end": value_end,
                "value": value_node,
            }
            index = _skip_json_ws(text, value_end)
            if index < len(text) and text[index] == ",":
                index = _skip_json_ws(text, index + 1)
            elif index < len(text) and text[index] == "}":
                break
            else:
                raise AlignmentError("invalid JSON object separator")
        if index >= len(text) or text[index] != "}":
            raise AlignmentError("unterminated JSON object in serialized completion")
        end = index + 1
        return {"kind": "object", "start": start, "end": end, "members": members}, end
    if char == "[":
        index += 1
        items: list[dict[str, Any]] = []
        index = _skip_json_ws(text, index)
        while index < len(text) and text[index] != "]":
            node, value_end = _parse_json_node(text, index)
            items.append(node)
            index = _skip_json_ws(text, value_end)
            if index < len(text) and text[index] == ",":
                index = _skip_json_ws(text, index + 1)
            elif index < len(text) and text[index] == "]":
                break
            else:
                raise AlignmentError("invalid JSON array separator")
        if index >= len(text) or text[index] != "]":
            raise AlignmentError("unterminated JSON array in serialized completion")
        end = index + 1
        return {"kind": "array", "start": start, "end": end, "items": items}, end
    end = index
    while end < len(text) and text[end] not in ",]} \t\r\n":
        end += 1
    if end == index:
        raise AlignmentError("invalid JSON scalar in serialized completion")
    try:
        value = json.loads(text[index:end])
    except json.JSONDecodeError as exc:
        raise AlignmentError("invalid JSON scalar in serialized completion") from exc
    return {"kind": "scalar", "start": start, "end": end, "value": value}, end


def parse_completion_regions(completion: str) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Parse exact character spans; never search for value strings with find()."""
    node, end = _parse_json_node(completion, 0)
    if _skip_json_ws(completion, end) != len(completion) or node.get("kind") != "object":
        raise AlignmentError("serialized completion is not one complete JSON object")
    members = node["members"]
    expected_top = {"emotion_label", "user_profile", "response_text", "memory_refs"}
    if set(members) != expected_top:
        raise AlignmentError("serialized completion fields differ from the fixed A0 schema")
    primary = {
        key: (members[key]["value"]["start"], members[key]["value"]["end"])
        for key in expected_top
    }
    profile_node = members["user_profile"]["value"]
    if profile_node.get("kind") != "object":
        raise AlignmentError("serialized user_profile is not an object")
    profile_members = profile_node.get("members", {})
    if set(profile_members) != set(PROFILE_DIMS):
        raise AlignmentError("serialized profile dimensions differ from P0 contract")
    dimensions = {
        dim: (profile_members[dim]["key_start"], profile_members[dim]["entry_end"])
        for dim in PROFILE_DIMS
    }
    return primary, dimensions


def _paint(target: list[str], start: int, end: int, value: str) -> None:
    if start < 0 or end < start or end > len(target):
        raise AlignmentError("parsed JSON character span lies outside rendered sequence")
    for position in range(start, end):
        target[position] = value


def assign_token_regions(
    *,
    rendered_text: str,
    completion: str,
    completion_start: int,
    offsets: list[tuple[int, int]],
    token_ids: list[int],
    prompt_length: int,
    labels: list[int],
    eos_token_id: int,
    special_token_ids: set[int],
) -> tuple[list[str], list[str | None], dict[str, int]]:
    if len(offsets) != len(token_ids) or len(labels) != len(token_ids):
        raise AlignmentError("Offset/input/label lengths differ")
    completion_end = completion_start + len(completion)
    if completion_end > len(rendered_text):
        raise AlignmentError("completion character span exceeds full rendered sequence")
    json.loads(completion)
    top_spans, dim_spans = parse_completion_regions(completion)
    primary_chars = ["prompt_context"] * len(rendered_text)
    profile_chars: list[str | None] = [None] * len(rendered_text)
    _paint(primary_chars, completion_start, completion_end, "public_structure")
    for key, (start, end) in top_spans.items():
        _paint(primary_chars, completion_start + start, completion_start + end, key)
    profile_start, profile_end = top_spans["user_profile"]
    _paint(profile_chars, completion_start + profile_start, completion_start + profile_end, "profile_structure")
    for dim, (start, end) in dim_spans.items():
        _paint(profile_chars, completion_start + start, completion_start + end, dim)
    # Chat-template text after the JSON body (EOS spelling and trailing newline) is structure.
    _paint(primary_chars, completion_end, len(rendered_text), "public_structure")

    eos_positions = [index for index in range(prompt_length, len(token_ids)) if token_ids[index] == eos_token_id]
    if not eos_positions:
        raise AlignmentError("supervised Completion has no locked EOS token")
    terminal_eos = eos_positions[-1]
    regions: list[str] = []
    profile_regions: list[str | None] = []
    active_count = 0
    for index, (start, end) in enumerate(offsets):
        if labels[index] == -100:
            regions.append("ignored_prompt")
            profile_regions.append(None)
            continue
        active_count += 1
        if index == terminal_eos:
            regions.append("eos")
            profile_regions.append(None)
            continue
        if start == end:
            if token_ids[index] not in special_token_ids:
                raise AlignmentError("active non-special token has an empty Offset span")
            regions.append("public_structure")
            profile_regions.append(None)
            continue
        if start < 0 or end > len(rendered_text) or end < start:
            raise AlignmentError("Tokenizer returned an Offset outside the rendered sequence")
        primary_values = set(primary_chars[start:end])
        primary_values.discard("prompt_context")
        if "prompt_context" in primary_chars[start:end] and primary_values:
            regions.append("boundary_mixed")
            profile_regions.append(None)
            continue
        if not primary_values:
            raise AlignmentError("supervised token maps only to Prompt characters")
        if len(primary_values) != 1:
            regions.append("boundary_mixed")
            profile_regions.append(None)
            continue
        primary = next(iter(primary_values))
        if primary not in TOP_REGIONS:
            raise AlignmentError("supervised token has an unrecognized primary region")
        regions.append(primary)
        if primary == "user_profile":
            subvalues = set(profile_chars[start:end])
            subvalues.discard(None)
            if len(subvalues) != 1:
                profile_regions.append("boundary_mixed")
            else:
                subregion = next(iter(subvalues))
                if subregion not in PROFILE_REGIONS:
                    raise AlignmentError("profile token has an unrecognized dimension region")
                profile_regions.append(subregion)
        else:
            profile_regions.append(None)
    if active_count != len(regions) - sum(label == -100 for label in labels):
        raise AlignmentError("active Token assignment count differs from supervised Labels")
    counts = Counter(region for region, label in zip(regions, labels) if label != -100)
    profile_counts = Counter(
        region for region, primary, label in zip(profile_regions, regions, labels)
        if label != -100 and primary == "user_profile" and region is not None
    )
    if sum(counts.get(region, 0) for region in TOP_REGIONS) != active_count:
        raise AlignmentError("field/structure/EOS/boundary Token counts do not conserve supervision")
    if sum(profile_counts.get(region, 0) for region in PROFILE_REGIONS) != counts.get("user_profile", 0):
        raise AlignmentError("profile dimension/structure/boundary Tokens do not conserve user_profile")
    return regions, profile_regions, {region: counts.get(region, 0) for region in TOP_REGIONS}


def _nearest_rank(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (percentile * len(ordered) + 99) // 100 - 1
    return ordered[max(0, index)]


def _distribution(values: list[int], total_tokens: int) -> dict[str, Any]:
    if not values:
        return {"sum": 0, "mean": None, "median": None, "p90": None, "share": {"numerator": 0, "denominator": total_tokens, "ratio": None}}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    total = sum(values)
    return {
        "sum": total,
        "mean": total / len(values),
        "median": median,
        "p90": _nearest_rank(values, 90),
        "share": {
            "numerator": total,
            "denominator": total_tokens,
            "ratio": total / total_tokens if total_tokens else None,
        },
    }


@dataclass
class TokenGroup:
    samples: int = 0
    input_tokens: int = 0
    supervised_tokens: int = 0
    fields: dict[str, list[int]] = field(default_factory=lambda: {key: [] for key in TOP_REGIONS})
    profile: dict[str, list[int]] = field(default_factory=lambda: {key: [] for key in PROFILE_REGIONS})

    def add(self, token_counts: dict[str, int], profile_counts: dict[str, int], input_tokens: int) -> None:
        self.samples += 1
        self.input_tokens += input_tokens
        self.supervised_tokens += sum(token_counts.values())
        for key in TOP_REGIONS:
            self.fields[key].append(token_counts.get(key, 0))
        for key in PROFILE_REGIONS:
            self.profile[key].append(profile_counts.get(key, 0))

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": self.samples,
            "input_token_sum": self.input_tokens,
            "supervised_token_sum": self.supervised_tokens,
            "top_level_regions": {
                key: _distribution(values, self.supervised_tokens)
                for key, values in self.fields.items()
            },
            "profile_subregions": {
                key: _distribution(values, sum(self.fields["user_profile"]))
                for key, values in self.profile.items()
            },
        }


class TokenAccumulator:
    def __init__(self) -> None:
        self.groups: dict[str, TokenGroup] = {"all": TokenGroup()}
        self.input_tokens = 0
        self.supervised_tokens = 0
        self.rows_verified = 0
        self.token_length_max = 0

    def add(self, metadata: dict[str, Any], counts: dict[str, int], profile_counts: dict[str, int], input_tokens: int) -> None:
        self.rows_verified += 1
        self.input_tokens += input_tokens
        self.supervised_tokens += sum(counts.values())
        self.token_length_max = max(self.token_length_max, input_tokens)
        groups = ["all"]
        groups.append("profile_all_empty" if metadata["profile_all_empty"] else "profile_any_nonempty")
        for dim in PROFILE_DIMS:
            groups.append(f"{dim}_gold_nonempty" if metadata["dim_nonempty"][dim] else f"{dim}_gold_empty")
        groups.append(metadata["fit_outcome"])
        groups.append("short_completion" if metadata["short_completion"] else "long_completion")
        for group_name in groups:
            group = self.groups.setdefault(group_name, TokenGroup())
            group.add(counts, profile_counts, input_tokens)

    def summary(self, *, input_rows: int, cap: int, seed: int, selection: dict[str, Any]) -> dict[str, Any]:
        if self.rows_verified != input_rows:
            raise AlignmentError("verified rows do not equal diagnostic input rows")
        return {
            "status": "completed",
            "population": "D0.4.3 SFT Train only; no Dev or official Val rows",
            "sample_cap": cap,
            "seed": seed,
            "input_rows": input_rows,
            "verified_rows": self.rows_verified,
            "unverified_rows": input_rows - self.rows_verified,
            "input_token_sum": self.input_tokens,
            "supervised_token_sum": self.supervised_tokens,
            "max_sequence_tokens": self.token_length_max,
            "classification_conservation": {
                "classified_supervised_token_sum": self.supervised_tokens,
                "unclassified_supervised_token_sum": 0,
                "passed": True,
            },
            "selection": selection,
            "groups": {key: value.summary() for key, value in sorted(self.groups.items())},
        }


def _profile_counts(regions: list[str], profile_regions: list[str | None], labels: list[int]) -> dict[str, int]:
    result = {key: 0 for key in PROFILE_REGIONS}
    for primary, subregion, label in zip(regions, profile_regions, labels):
        if label != -100 and primary == "user_profile" and subregion is not None:
            result[subregion] += 1
    return result


def _load_tokenizer_context(root: Path, paths: dict[str, Path]) -> tuple[Any, str, dict[str, Any], dict[str, Any]]:
    runner = json.loads(paths["runner_config"].read_text(encoding="utf-8"))
    schema = json.loads(paths["train_schema"].read_text(encoding="utf-8"))
    prompt = paths["prompt"].read_text(encoding="utf-8")
    model_config = _load_yaml(paths["training_config"])
    base_path = _resolve(root, model_config["base_model"]["local_path"])
    tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runner, base_path)
    return tokenizer, prompt, schema, {"runner": runner, "tokenizer": tokenizer_facts, "base_path": base_path}


def _build_sample_mapping(
    *,
    source: dict[str, Any],
    sft_row: dict[str, Any],
    tokenizer: Any,
    prompt_text: str,
    schema: dict[str, Any],
    max_prompt_tokens: int,
) -> tuple[
    str,
    str,
    list[tuple[int, int]],
    list[str],
    list[str | None],
    dict[str, int],
    dict[str, int],
    dict[str, Any],
]:
    try:
        validate_train_row(sft_row, expected_eos_token_id=int(tokenizer.eos_token_id), expected_split="train")
    except (TypeError, ValueError) as exc:
        raise AlignmentError("D0.4.3 SFT row violates stored input/label invariants") from exc
    if source.get("sample_id") != sft_row.get("sample_id"):
        raise AlignmentError("D0.3/SFT sample IDs do not match")
    if source.get("conversation_id") != sft_row.get("conversation_id"):
        raise AlignmentError("D0.3/SFT conversation IDs do not match")
    if source.get("source", {}).get("split") != "train":
        raise AlignmentError("matched D0.3 sample is not from Train")
    history, target = source.get("history"), source.get("target")
    if not isinstance(history, list) or not isinstance(target, dict):
        raise AlignmentError("D0.3 source sample has invalid history or target")
    try:
        selection = select_recent_complete_suffix(history, prompt_text, max_prompt_tokens, tokenizer)
        messages = build_messages(selection["selected_history"], prompt_text)
        completion = serialize_completion(target, schema)
        prefix_ids = encode_inference_prompt(tokenizer, messages)
        training_ids = encode_full_training_sequence(tokenizer, messages, completion)
    except Exception as exc:
        raise AlignmentError("existing P0/F2/T1 serializer failed to rebuild the training sequence") from exc
    input_ids = list(map(int, sft_row["input_ids"]))
    labels = list(map(int, sft_row["labels"]))
    if len(selection["prompt_ids"]) != sft_row["prompt_length"] or prefix_ids != input_ids[:sft_row["prompt_length"]]:
        raise AlignmentError("reconstructed Prompt IDs differ from D0.4.3 prefix")
    if training_ids != input_ids:
        raise AlignmentError("reconstructed complete training Token IDs differ from D0.4.3")
    if labels[:sft_row["prompt_length"]] != [-100] * sft_row["prompt_length"]:
        raise AlignmentError("Prompt Labels are not fully masked")
    if labels[sft_row["prompt_length"]:] != input_ids[sft_row["prompt_length"]:]:
        raise AlignmentError("Completion Labels differ from saved input IDs")
    if len(labels) != len(input_ids) or len(input_ids) - sft_row["prompt_length"] != sft_row["completion_length"]:
        raise AlignmentError("stored lengths disagree with the saved sequence")
    if int(tokenizer.eos_token_id) not in input_ids[sft_row["prompt_length"]:]:
        raise AlignmentError("Completion is missing the locked EOS Token ID")
    full_messages = [*messages, {"role": "assistant", "content": completion, "reasoning_content": ""}]
    try:
        rendered = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        rendered_prefix = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    except Exception as exc:
        raise AlignmentError("Tokenizer cannot return verified full-sequence Offset mappings") from exc
    offset_ids = encoded.get("input_ids")
    offsets_value = encoded.get("offset_mapping")
    if hasattr(offset_ids, "tolist"):
        offset_ids = offset_ids.tolist()
    if hasattr(offsets_value, "tolist"):
        offsets_value = offsets_value.tolist()
    if not isinstance(rendered, str) or not isinstance(offset_ids, list) or not isinstance(offsets_value, list):
        raise AlignmentError("Tokenizer Offset result is missing or has an unsupported shape")
    if offset_ids != input_ids:
        raise AlignmentError("rendered-text Offset Token IDs differ from the saved D0.4.3 input_ids")
    if len(offsets_value) != len(input_ids):
        raise AlignmentError("Tokenizer Offset count differs from the saved D0.4.3 input_ids")
    if not rendered.startswith(rendered_prefix):
        raise AlignmentError("full Chat Template does not begin with the exact training Prompt render")
    start = len(rendered_prefix)
    if rendered[start:start + len(completion)] != completion:
        raise AlignmentError("serialized JSON is not at the expected assistant completion character boundary")
    normalized_offsets = [(int(item[0]), int(item[1])) for item in offsets_value]
    regions, profile_regions, counts = assign_token_regions(
        rendered_text=rendered,
        completion=completion,
        completion_start=start,
        offsets=normalized_offsets,
        token_ids=input_ids,
        prompt_length=int(sft_row["prompt_length"]),
        labels=labels,
        eos_token_id=int(tokenizer.eos_token_id),
        special_token_ids=set(map(int, tokenizer.all_special_ids)),
    )
    p_counts = _profile_counts(regions, profile_regions, labels)
    if sum(counts.values()) != sft_row["completion_length"]:
        raise AlignmentError("all field and structure buckets do not sum to Completion Length")
    profile = target.get("user_profile", {})
    dim_nonempty = {dim: bool(profile.get(dim)) for dim in PROFILE_DIMS}
    metadata = {
        "profile_all_empty": not any(dim_nonempty.values()),
        "dim_nonempty": dim_nonempty,
        "fit_outcome": "unassigned",
    }
    return completion, rendered, normalized_offsets, regions, profile_regions, counts, p_counts, metadata


def _tokenize_one(
    *,
    source: dict[str, Any],
    sft_row: dict[str, Any],
    tokenizer: Any,
    prompt_text: str,
    schema: dict[str, Any],
    max_prompt_tokens: int,
) -> tuple[list[str], list[str | None], dict[str, int], dict[str, int]]:
    _completion, _rendered, _offsets, regions, profile_regions, counts, profile_counts, _metadata = _build_sample_mapping(
        source=source,
        sft_row=sft_row,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        schema=schema,
        max_prompt_tokens=max_prompt_tokens,
    )
    return regions, profile_regions, counts, profile_counts


def _source_hashes_after(paths: dict[str, Path]) -> dict[str, str]:
    result = {}
    for name in (
        "normalized_train", "sft_train", "d03_manifest", "sft_manifest", "d042_manifest",
        "sft_summary", "sft_excluded", "d042_budget_rows", "train_schema", "prompt", "tokenizer_config", "runner_config",
    ):
        result[name] = sha256_file(paths[name])
    return result


def _model_identity(root: Path, paths: dict[str, Path], config: dict[str, Any], training_manifest: dict[str, Any]) -> dict[str, Any]:
    training_config = _load_yaml(paths["training_config"])
    base_path = _resolve(root, training_config["base_model"]["local_path"])
    adapter_path = paths["training_run_root"] / "adapter"
    if not base_path.is_dir() or not adapter_path.is_dir():
        return {"status": "missing", "base_path_exists": base_path.is_dir(), "adapter_path_exists": adapter_path.is_dir()}
    base = inference._weight_identity(base_path, {
        "model_id": training_config["base_model"]["model_id"],
        "model_revision": training_config["base_model"].get("revision"),
    })
    adapter, adapter_config = inference.adapter_weight_identity(adapter_path)
    try:
        inference.validate_adapter_base_reference(adapter_config, training_config["base_model"]["model_id"], base_path)
    except (TypeError, ValueError) as exc:
        raise ValueError("Full B Adapter declares a different Base model") from exc
    expected_base = training_manifest.get("base_weight_sha256")
    expected_adapter = training_manifest.get("adapter_identity", {})
    matched = (
        base.get("observed_weight_hash") == expected_base
        and adapter.get("weight_sha256") == expected_adapter.get("weight_sha256")
        and adapter.get("config_sha256") == expected_adapter.get("config_sha256")
    )
    return {
        "status": "verified" if matched else "mismatch",
        "base": {
            "identity_status": base.get("identity_status"),
            "fingerprint_status": base.get("fingerprint_status"),
            "observed_weight_sha256": base.get("observed_weight_hash"),
            "expected_weight_sha256": expected_base,
            "files": base.get("files", []),
            "revision_status": base.get("revision_status"),
        },
        "adapter": {
            "observed_weight_sha256": adapter.get("weight_sha256"),
            "expected_weight_sha256": expected_adapter.get("weight_sha256"),
            "observed_config_sha256": adapter.get("config_sha256"),
            "expected_config_sha256": expected_adapter.get("config_sha256"),
            "files": adapter.get("files", []),
            "config_summary": adapter.get("config_summary", {}),
        },
        "matched_historical_full_b_identity": matched,
    }


def run_token_mode(
    *, root: Path, paths: dict[str, Path], config: dict[str, Any], identity: dict[str, Any],
    tokenizer: Any, prompt_text: str, schema: dict[str, Any], cap: int, seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    population, _budget, _unused = _candidate_metadata(paths, identity, config)
    if cap == 0:
        records = population
        selection = {
            "selection_rule": "full_population_no_sampling",
            "seed": seed,
            "candidate_rows": len(records),
            "selected_rows": len(records),
            "selected_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in records]),
            "selection_is_population_representative": True,
            "strata": {},
        }
    else:
        records, selection = deterministic_select(population, cap=cap, seed=seed, stratified=False)
    median_length = _length_median(population)
    accumulator = TokenAccumulator()
    processed = 0
    with paths["normalized_train"].open("rb") as normalized_handle, paths["sft_train"].open("rb") as sft_handle:
      for row in records:
        normalized_handle.seek(row["normalized_offset"])
        source = json.loads(normalized_handle.readline())
        sft_handle.seek(row["sample_id_offset"])
        sft_row = json.loads(sft_handle.readline())
        regions, profile_regions, counts, profile_counts = _tokenize_one(
            source=source,
            sft_row=sft_row,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            schema=schema,
            max_prompt_tokens=identity["max_prompt_tokens"],
        )
        outcome = row["fit_outcome"]
        if outcome not in SELECTION_STATUSES:
            raise AlignmentError("SFT row has no valid D0.4.2 fit outcome")
        target = source["target"]
        dim_nonempty = {dim: bool(target["user_profile"].get(dim)) for dim in PROFILE_DIMS}
        metadata = {
            "profile_all_empty": not any(dim_nonempty.values()),
            "dim_nonempty": dim_nonempty,
            "fit_outcome": outcome,
            "short_completion": int(sft_row["completion_length"]) <= median_length,
        }
        accumulator.add(metadata, counts, profile_counts, len(sft_row["input_ids"]))
        processed += 1
    token_summary = accumulator.summary(input_rows=processed, cap=cap, seed=seed, selection=selection)
    _write_json(run_dir / "token_summary.json", token_summary)
    _write_json(run_dir / "sample_selection.json", selection)
    alignment = {
        "status": "passed",
        "requested_rows": processed,
        "verified_rows": processed,
        "failed_rows": 0,
        "saved_input_ids_equal_reconstructed_ids": True,
        "offset_retokenized_ids_equal_saved_input_ids": True,
        "prompt_labels_all_masked": True,
        "completion_labels_equal_saved_input_ids": True,
        "completion_contains_locked_eos": True,
        "classification_conservation": True,
        "supervised_token_sum": accumulator.supervised_tokens,
        "classified_token_sum": accumulator.supervised_tokens,
        "boundary_mixed_token_sum": sum(
            group.fields["boundary_mixed"][i]
            for group in [accumulator.groups["all"]]
            for i in range(len(group.fields["boundary_mixed"]))
        ),
        "field_mapping_method": "whole rendered Chat Template + fast Tokenizer character Offsets; exact ID equality verified per row",
        "field_mapping_version": VERSION,
    }
    _write_json(run_dir / "alignment_summary.json", alignment)
    return {"status": "completed", "rows": processed, "selection": selection, "token_summary": token_summary, "alignment": alignment}


def _verify_prior_token_run(output_root: Path, run_id: str, identity: dict[str, Any]) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("token-run-id contains unsupported characters")
    run_dir = output_root / run_id
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "token_summary.json"
    alignment_path = run_dir / "alignment_summary.json"
    if not all(path.is_file() for path in (manifest_path, summary_path, alignment_path)):
        raise FileNotFoundError("a completed full CPU Token run is required before CE mode")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("mode") != "token" or manifest.get("status") != "completed":
        raise ValueError("prior Token run has the wrong version, mode, or status")
    if summary.get("sample_cap") != 0 or summary.get("verified_rows") != identity["sft_train_rows"]:
        raise ValueError("CE requires a completed full-population CPU Token run")
    if alignment.get("status") != "passed" or alignment.get("classification_conservation") is not True:
        raise ValueError("prior full CPU Token run did not pass alignment and conservation")
    if manifest.get("source_identity", {}).get("sft_train_sha256") != identity["sft_train_sha256"]:
        raise ValueError("prior Token run used another SFT Train file")
    return {
        "run_id": run_id,
        "manifest_sha256": sha256_file(manifest_path),
        "token_summary_sha256": sha256_file(summary_path),
        "alignment_summary_sha256": sha256_file(alignment_path),
        "verified_rows": summary["verified_rows"],
    }


def _ce_for_logits(torch: Any, logits: Any, labels: Any, regions: list[str], profile_regions: list[str | None], chunk_tokens: int) -> tuple[dict[str, float], dict[str, float], float, int]:
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    active_positions = (shifted_labels[0] != -100).nonzero(as_tuple=False).flatten()
    if active_positions.numel() == 0:
        raise ValueError("selected sample has no active shifted labels")
    values: dict[str, float] = {name: 0.0 for name in TOP_REGIONS}
    profile_values: dict[str, float] = {name: 0.0 for name in PROFILE_REGIONS}
    valid_sum = 0.0
    valid_count = int(active_positions.numel())
    for start in range(0, shifted_logits.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, shifted_logits.shape[1])
        chunk_labels = shifted_labels[:, start:end]
        if not bool((chunk_labels != -100).any().item()):
            continue
        chunk_logits = shifted_logits[:, start:end, :]
        ce = functional.cross_entropy(
            chunk_logits.float().reshape(-1, chunk_logits.shape[-1]),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(chunk_labels)[0]
        for local_pos in (chunk_labels[0] != -100).nonzero(as_tuple=False).flatten().tolist():
            seq_pos = start + local_pos + 1
            loss = float(ce[local_pos].item())
            region = regions[seq_pos]
            values[region] += loss
            if region == "user_profile":
                subregion = profile_regions[seq_pos]
                if subregion is None:
                    raise AlignmentError("profile CE Token has no dimension/structure subregion")
                profile_values[subregion] += loss
            valid_sum += loss
        del ce, chunk_logits
    return values, profile_values, valid_sum, valid_count


def _load_inference_model(
    root: Path, paths: dict[str, Path], training_config: dict[str, Any], adapter_path: Path
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_blocked: torch.cuda.is_available() is false")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("gpu_blocked: CUDA device does not support BF16")
    base_path = _resolve(root, training_config["base_model"]["local_path"])
    quantization = training_config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization["quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bool(quantization["use_double_quant"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        device_map=quantization.get("device_map", "auto"),
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model = inference.load_peft_adapter(
        model,
        adapter_path,
        prepare_quantized_base=True,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    if model.training:
        raise RuntimeError("loaded Full B model did not enter eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen Full B model unexpectedly contains trainable parameters")
    inspect_config = {
        "quantization": {
            "load_in_4bit": True,
            "quant_type": quantization["quant_type"],
            "compute_dtype": "bfloat16",
            "use_double_quant": bool(quantization["use_double_quant"]),
        }
    }
    facts = inference.inspect_quantization_runtime(model, inspect_config, torch)
    if not facts.get("real_nf4_verified") or facts.get("offload_devices"):
        raise RuntimeError("loaded model failed NF4/CUDA placement verification")
    return model, torch, facts


def _aggregate_ce(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    total_tokens = sum(row["valid_token_count"] for row in rows)
    total_ce = sum(row["total_ce_sum"] for row in rows)
    region_names = TOP_REGIONS
    result: dict[str, Any] = {
        "sample_count": samples,
        "supervised_token_count": total_tokens,
        "total_ce_sum": total_ce,
        "native_model_loss_mean": sum(row["native_loss"] for row in rows) / samples if samples else None,
        "reduced_manual_loss_mean": total_ce / total_tokens if total_tokens else None,
        "regions": {},
    }
    for region in region_names:
        token_count = sum(row["token_counts"][region] for row in rows)
        ce_sum = sum(row["ce_sums"][region] for row in rows)
        per_sample = [
            row["ce_sums"][region] / row["token_counts"][region]
            for row in rows if row["token_counts"][region] > 0
        ]
        result["regions"][region] = {
            "token_count": token_count,
            "token_share": {"numerator": token_count, "denominator": total_tokens, "ratio": token_count / total_tokens if total_tokens else None},
            "ce_sum": ce_sum,
            "pooled_mean_token_ce": ce_sum / token_count if token_count else None,
            "macro_by_sample_mean_token_ce": sum(per_sample) / len(per_sample) if per_sample else None,
            "macro_sample_count": len(per_sample),
            "not_applicable_sample_count": samples - len(per_sample),
            "ce_contribution": {"numerator": ce_sum, "denominator": total_ce, "ratio": ce_sum / total_ce if total_ce else None},
        }
    result["profile_subregions"] = {}
    profile_tokens = sum(row["token_counts"]["user_profile"] for row in rows)
    for region in PROFILE_REGIONS:
        token_count = sum(row["profile_token_counts"][region] for row in rows)
        ce_sum = sum(row["profile_ce_sums"][region] for row in rows)
        per_sample = [
            row["profile_ce_sums"][region] / row["profile_token_counts"][region]
            for row in rows if row["profile_token_counts"][region] > 0
        ]
        result["profile_subregions"][region] = {
            "token_count": token_count,
            "token_share_within_user_profile": {"numerator": token_count, "denominator": profile_tokens, "ratio": token_count / profile_tokens if profile_tokens else None},
            "ce_sum": ce_sum,
            "pooled_mean_token_ce": ce_sum / token_count if token_count else None,
            "macro_by_sample_mean_token_ce": sum(per_sample) / len(per_sample) if per_sample else None,
            "macro_sample_count": len(per_sample),
            "not_applicable_sample_count": samples - len(per_sample),
            "ce_contribution_to_all": {"numerator": ce_sum, "denominator": total_ce, "ratio": ce_sum / total_ce if total_ce else None},
        }
    result["conservation"] = {
        "classified_token_count": sum(item["token_count"] for item in result["regions"].values()),
        "valid_token_count": total_tokens,
        "classified_ce_sum": sum(item["ce_sum"] for item in result["regions"].values()),
        "total_ce_sum": total_ce,
        "passed": (
            sum(item["token_count"] for item in result["regions"].values()) == total_tokens
            and math.isclose(sum(item["ce_sum"] for item in result["regions"].values()), total_ce, rel_tol=1e-10, abs_tol=1e-8)
            and sum(item["token_count"] for item in result["profile_subregions"].values()) == profile_tokens
        ),
    }
    return result


def run_ce_mode(
    *, root: Path, paths: dict[str, Path], config: dict[str, Any], identity: dict[str, Any],
    tokenizer: Any, prompt_text: str, schema: dict[str, Any], cap: int, seed: int,
    run_dir: Path, token_run_id: str,
) -> dict[str, Any]:
    token_run = _verify_prior_token_run(paths["output_root"], token_run_id, identity)
    records, _budget, _unused = _candidate_metadata(paths, identity, config)
    selected, selection = deterministic_select(records, cap=cap, seed=seed, stratified=True)
    if len(selected) > int(config["gpu"]["max_sample_cap"]):
        raise ValueError("GPU selected sample count exceeds configured safety cap")
    _write_json(run_dir / "sample_selection.json", {**selection, "source_token_run": token_run})
    selected_mapping: list[dict[str, Any]] = []
    alignment_failures = 0
    with paths["normalized_train"].open("rb") as normalized_handle, paths["sft_train"].open("rb") as sft_handle:
      for row in selected:
        normalized_handle.seek(row["normalized_offset"])
        source = json.loads(normalized_handle.readline())
        sft_handle.seek(row["sample_id_offset"])
        sft_row = json.loads(sft_handle.readline())
        try:
            result = _build_sample_mapping(
                source=source,
                sft_row=sft_row,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                schema=schema,
                max_prompt_tokens=identity["max_prompt_tokens"],
            )
        except AlignmentError:
            alignment_failures += 1
            continue
        completion, rendered, offsets, regions, profile_regions, counts, profile_counts, metadata = result
        del completion, rendered, offsets
        metadata.update({
            "fit_outcome": row["fit_outcome"],
            "short_completion": int(row["completion_length"]) <= selection["length_median_tokens"],
        })
        selected_mapping.append({
            "row": row,
            "source": source,
            "sft_row": sft_row,
            "regions": regions,
            "profile_regions": profile_regions,
            "token_counts": counts,
            "profile_token_counts": profile_counts,
            "metadata": metadata,
        })
    alignment_summary = {
        "status": "passed" if alignment_failures == 0 and len(selected_mapping) == len(selected) else "alignment_blocked",
        "requested_rows": len(selected),
        "verified_rows": len(selected_mapping),
        "failed_rows": alignment_failures,
        "source_token_run_id": token_run_id,
        "source_token_run_manifest_sha256": token_run["manifest_sha256"],
        "mapping_scope": "selected CE samples only; previous full CPU Token run verified",
        "offset_retokenized_ids_equal_saved_input_ids": alignment_failures == 0,
        "classification_conservation": alignment_failures == 0 and len(selected_mapping) == len(selected),
    }
    _write_json(run_dir / "alignment_summary.json", alignment_summary)
    if alignment_summary["status"] != "passed":
        blocked = {"status": "alignment_blocked", "sample_count": len(selected), "gpu_model_loaded": False}
        _write_json(run_dir / "ce_summary.json", blocked)
        return {"status": "alignment_blocked", "alignment": alignment_summary, "ce": blocked}

    model_identity = _model_identity(root, paths, config, json.loads((paths["training_run_root"] / "manifest.json").read_text(encoding="utf-8")))
    if model_identity.get("status") != "verified":
        blocked = {"status": "gpu_blocked", "reason": "local Full B Base or Adapter SHA differs from historical Run", "model_identity": model_identity, "gpu_model_loaded": False}
        _write_json(run_dir / "ce_summary.json", blocked)
        return {"status": "gpu_blocked", "alignment": alignment_summary, "ce": blocked}
    try:
        model, torch, runtime_facts = _load_inference_model(
            root,
            paths,
            _load_yaml(paths["training_config"]),
            paths["training_run_root"] / "adapter",
        )
    except RuntimeError as exc:
        if "gpu_blocked" in str(exc) or "CUDA" in str(exc) or "BF16" in str(exc):
            blocked = {
                "status": "gpu_blocked",
                "reason": str(exc),
                "sample_count": len(selected),
                "sample_ids_sha256": selection["selected_sample_ids_sha256"],
                "model_identity": model_identity,
                "gpu_model_loaded": False,
            }
            _write_json(run_dir / "ce_summary.json", blocked)
            return {"status": "gpu_blocked", "alignment": alignment_summary, "ce": blocked}
        raise

    chunk_tokens = int(config["gpu"]["logit_chunk_tokens"])
    if chunk_tokens <= 0:
        raise ValueError("gpu.logit_chunk_tokens must be positive")
    abs_tol = float(config["gpu"]["native_loss_abs_tolerance"])
    repeat_tol = float(config["gpu"]["repeat_abs_tolerance"])
    per_sample: list[dict[str, Any]] = []
    max_native_error = 0.0
    repeat_error: float | None = None
    repeat_checked = False
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for index_in_selection, item in enumerate(selected_mapping):
                row = item["sft_row"]
                input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda:0")
                labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda:0")
                output = model(input_ids=input_ids, labels=labels, use_cache=False)
                native_loss = float(output.loss.float().item())
                logits = output.logits
                ce_sums, profile_ce_sums, total_ce_sum, valid_count = _ce_for_logits(
                    torch,
                    logits,
                    labels,
                    item["regions"],
                    item["profile_regions"],
                    chunk_tokens,
                )
                manual_mean = total_ce_sum / valid_count
                native_error = abs(manual_mean - native_loss)
                max_native_error = max(max_native_error, native_error)
                if index_in_selection == 0:
                    del logits, output
                    repeated = model(input_ids=input_ids, labels=labels, use_cache=False)
                    repeat_error = abs(float(repeated.loss.float().item()) - native_loss)
                    repeat_checked = True
                    del repeated
                else:
                    del logits, output
                row_counts = item["token_counts"]
                row_profile_counts = item["profile_token_counts"]
                if sum(row_counts.values()) != valid_count:
                    raise AlignmentError("CE valid-token denominator differs from mapped field tokens")
                if not math.isclose(sum(ce_sums.values()), total_ce_sum, rel_tol=1e-10, abs_tol=1e-8):
                    raise AlignmentError("per-field CE sums do not conserve total CE")
                if not math.isclose(sum(profile_ce_sums.values()), ce_sums["user_profile"], rel_tol=1e-10, abs_tol=1e-8):
                    raise AlignmentError("profile dimension CE sums do not conserve profile CE")
                per_sample.append({
                    "groups": _selection_memberships(item["row"], selection["length_median_tokens"]),
                    "token_counts": row_counts,
                    "profile_token_counts": row_profile_counts,
                    "ce_sums": ce_sums,
                    "profile_ce_sums": profile_ce_sums,
                    "total_ce_sum": total_ce_sum,
                    "valid_token_count": valid_count,
                    "native_loss": native_loss,
                    "native_abs_error": native_error,
                    "metadata": item["metadata"],
                })
                del input_ids, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as exc:
        del model
        torch.cuda.empty_cache()
        blocked = {
            "status": "gpu_blocked",
            "reason": "CUDA OOM during bounded per-sample forward",
            "failed_sample_scale": len(selected_mapping),
            "completed_before_oom": len(per_sample),
            "sample_ids_sha256": selection["selected_sample_ids_sha256"],
            "model_identity": model_identity,
            "runtime_facts": runtime_facts,
            "gpu_model_loaded": True,
        }
        _write_json(run_dir / "ce_summary.json", blocked)
        return {"status": "gpu_blocked", "alignment": alignment_summary, "ce": blocked}
    finally:
        elapsed = time.monotonic() - started

    grouped: dict[str, list[dict[str, Any]]] = {"all": per_sample}
    for row in per_sample:
        for group in row["groups"]:
            grouped.setdefault(group, []).append(row)
    aggregates = {key: _aggregate_ce(value) for key, value in sorted(grouped.items())}
    native_passed = max_native_error <= abs_tol
    repeat_passed = repeat_checked and repeat_error is not None and repeat_error <= repeat_tol
    conservation_passed = all(value["conservation"]["passed"] for value in aggregates.values())
    ce_summary = {
        "status": "completed" if native_passed and repeat_passed and conservation_passed else "consistency_check_failed",
        "measurement": "frozen Full B Adapter teacher-forced CE on selected D0.4.3 Train samples",
        "sample_count": len(per_sample),
        "sample_ids_sha256": selection["selected_sample_ids_sha256"],
        "selection": selection,
        "model_identity": model_identity,
        "runtime_facts": runtime_facts,
        "dtype": "NF4 weights with BF16 compute; logits converted to FP32 in time chunks for CE",
        "logit_chunk_tokens": chunk_tokens,
        "native_loss_check": {
            "max_absolute_difference": max_native_error,
            "absolute_tolerance": abs_tol,
            "passed": native_passed,
            "denominator": "all non -100 labels after causal shift",
        },
        "repeat_forward_check": {
            "sample_count": 1 if repeat_checked else 0,
            "absolute_loss_difference": repeat_error,
            "absolute_tolerance": repeat_tol,
            "model_eval": True,
            "inference_mode": True,
            "passed": repeat_passed,
        },
        "conservation_passed": conservation_passed,
        "elapsed_seconds": elapsed,
        "groups": aggregates,
    }
    _write_json(run_dir / "ce_summary.json", ce_summary)
    del model
    return {"status": ce_summary["status"], "alignment": alignment_summary, "ce": ce_summary}


def run_diagnosis(
    *, config_path: Path, run_id: str, mode: str, sample_cap: int | None,
    seed: int | None, token_run_id: str | None,
) -> Path:
    root = PROJECT_ROOT
    config_path = config_path.resolve()
    config, paths = load_config(root, config_path)
    if mode not in {"token", "ce"}:
        raise ValueError("mode must be token or ce")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id contains unsupported characters")
    default_cap = int(config["selection"].get("token_default_sample_cap", 0)) if mode == "token" else None
    if sample_cap is None:
        sample_cap = default_cap
    if sample_cap is None:
        raise ValueError("CE mode requires an explicit finite --sample-cap")
    sample_cap = int(sample_cap)
    seed = int(config["selection"].get("seed", 42) if seed is None else seed)
    validate_sample_cap(mode, sample_cap, int(config["gpu"]["max_sample_cap"]))
    if mode == "ce" and not token_run_id:
        raise ValueError("CE mode requires --token-run-id for a passed full CPU alignment Run")
    run_dir = create_run_dir(paths["output_root"], run_id)
    started = time.monotonic()
    import torch

    manifest: dict[str, Any] = {
        "version": VERSION,
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_facts(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "selection": {"seed": seed, "sample_cap": sample_cap},
        "population": "D0.4.3 SFT Train only; no Dev or official Val rows",
    }
    _write_json(run_dir / "manifest.json", manifest)
    try:
        identity, manifests = verify_source_identity(root, config, paths)
        manifest["source_identity"] = identity
        manifest["environment"].update({"packages": _package_versions()})
        _write_json(run_dir / "manifest.json", manifest)
        tokenizer, prompt_text, schema, tokenizer_facts = _load_tokenizer_context(root, paths)
        if tokenizer_facts["tokenizer"].get("signature_sha256") != identity["tokenizer_signature_sha256"]:
            raise AlignmentError("runtime Tokenizer signature differs from the accepted D0.4.3 identity")
        manifest["runtime_tokenizer"] = {
            "signature_sha256": tokenizer_facts["tokenizer"].get("signature_sha256"),
            "template_sha256": tokenizer_facts["tokenizer"].get("template_sha256"),
            "source": tokenizer_facts["tokenizer"].get("source"),
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        }
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise AlignmentError("Tokenizer has no verified fast Offset implementation")
        _write_json(run_dir / "manifest.json", manifest)
        if mode == "token":
            result = run_token_mode(
                root=root, paths=paths, config=config, identity=identity,
                tokenizer=tokenizer, prompt_text=prompt_text, schema=schema,
                cap=sample_cap, seed=seed, run_dir=run_dir,
            )
            if result["status"] == "completed":
                try:
                    model_identity = _model_identity(root, paths, config, manifests["training"])
                except Exception as exc:
                    model_identity = {
                        "status": "unverified",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                manifest["model_identity"] = model_identity
                source_after = _source_hashes_after(paths)
                manifest["source_hashes_after"] = source_after
                manifest["source_hashes_unchanged"] = all(
                    source_after.get(key) == value for key, value in manifests["hashes"].items()
                    if key in source_after
                )
                if not manifest["source_hashes_unchanged"]:
                    raise RuntimeError("a protected source file changed during token diagnosis")
                manifest.update(status="completed", result_status=result["status"])
            else:
                manifest.update(status=result["status"], result_status=result["status"])
        else:
            prior = _verify_prior_token_run(paths["output_root"], str(token_run_id), identity)
            manifest["prior_full_token_run"] = prior
            result = run_ce_mode(
                root=root, paths=paths, config=config, identity=identity,
                tokenizer=tokenizer, prompt_text=prompt_text, schema=schema,
                cap=sample_cap, seed=seed, run_dir=run_dir, token_run_id=str(token_run_id),
            )
            source_after = _source_hashes_after(paths)
            manifest["source_hashes_after"] = source_after
            manifest["source_hashes_unchanged"] = all(
                source_after.get(key) == value for key, value in manifests["hashes"].items()
                if key in source_after
            )
            if not manifest["source_hashes_unchanged"]:
                raise RuntimeError("a protected source file changed during CE diagnosis")
            manifest.update(status=result["status"], result_status=result["status"])
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
        return run_dir
    except BaseException as exc:
        manifest.update(
            status="alignment_blocked" if isinstance(exc, AlignmentError) else "blocked",
            error_type=type(exc).__name__,
            error_message=str(exc),
            elapsed_seconds=time.monotonic() - started,
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(run_dir / "manifest.json", manifest)
        if isinstance(exc, AlignmentError):
            _write_json(run_dir / "alignment_summary.json", {
                "status": "alignment_blocked",
                "verified_rows_before_failure": 0,
                "failed_rows": 1,
                "reason": str(exc),
                "field_quantitative_conclusions_generated": False,
            })
        raise
