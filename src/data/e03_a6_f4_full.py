"""Build the full F4 training cohort without changing the frozen F2/P0/T1 lineage."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.data.e03_a6_f4_sft import build_f4_training_item
from src.data.e03_a6_sft import _length_summary, _lineage_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the project root: {value}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _ordered_subset(reference_ids: list[str], selected_ids: list[str]) -> None:
    positions = {sample_id: index for index, sample_id in enumerate(reference_ids)}
    if len(positions) != len(reference_ids):
        raise ValueError("D0.3 Train contains duplicate sample IDs")
    last = -1
    for sample_id in selected_ids:
        position = positions.get(sample_id)
        if position is None:
            raise ValueError(f"F2 sample is absent from D0.3 Train: {sample_id}")
        if position <= last:
            raise ValueError("F2 Train IDs are not an ordered D0.3 subsequence")
        last = position


def _load_config(config_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = _resolve(root, str(config_path))
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E03-A6-F4-data-v1":
        raise ValueError("unsupported F4 full-data config")
    if config.get("format_version") != "F4":
        raise ValueError("full data config must explicitly select F4")
    profile = config.get("profile", {})
    if profile != {
        "max_prompt_tokens": 1280,
        "max_train_seq_length": 2048,
        "history_policy": "P0_T1_same_selected_history_then_F4_prompt",
        "completion_policy": "complete_or_block",
        "padding_side": "right",
        "packing": False,
        "padding_free": False,
    }:
        raise ValueError("F4 full-data protocol or fixed S/P budget changed")
    if config.get("expected_counts") != {
        "source_train": 16312, "effective_train": 16192, "historical_excluded": 120,
    }:
        raise ValueError("F4 full data must retain the exact D0.4.3 Train cohort and exclusions")
    paths = {
        name: _resolve(root, value)
        for name, value in config["inputs"].items()
    }
    paths["dataset_dir"] = _resolve(root, config["outputs"]["dataset_dir"])
    paths["run_root"] = _resolve(root, config["outputs"]["run_root"])
    for name, source in paths.items():
        if name not in {"dataset_dir", "run_root"} and not source.is_file():
            raise FileNotFoundError(f"missing F4 full-data input {name}: {source}")
    if paths["dataset_dir"].exists():
        raise FileExistsError(f"refusing to overwrite F4 full data: {paths['dataset_dir']}")
    return config, paths


def _verify_sources(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    d03 = _read_json(paths["d03_manifest"])
    d042 = _read_json(paths["d042_manifest"])
    f2_manifest = _read_json(paths["f2_manifest"])
    if d03.get("pipeline_version") != "D0.3-v1" or d042.get("audit_version") != "D0.4.2-v1":
        raise ValueError("D0.3 or D0.4.2 source version changed")
    if f2_manifest.get("version") != "D0.4.3-v1":
        raise ValueError("D0.4.3 source version changed")
    schema_hash = sha256_file(paths["train_schema"])
    if (
        f2_manifest.get("inputs", {}).get("train_schema", {}).get("path") != "data/train/train_public.schema.json"
        or f2_manifest.get("inputs", {}).get("train_schema", {}).get("sha256") != schema_hash
        or d042.get("input_hashes_unchanged_after_run", {}).get("train_schema") != schema_hash
    ):
        raise ValueError("official Train Schema differs from the D0.4.2/D0.4.3 source identity")
    if d042.get("d03_manifest", {}).get("sha256") != sha256_file(paths["d03_manifest"]):
        raise ValueError("D0.4.2 was built from another D0.3 manifest")
    if f2_manifest.get("d042_manifest_sha256") != sha256_file(paths["d042_manifest"]):
        raise ValueError("D0.4.3 was built from another D0.4.2 manifest")
    budget_record = d042.get("result_files", {}).get("budget_simulations", {})
    if (
        budget_record.get("path") != "outputs/token_audit/v1/budget_simulations.jsonl"
        or budget_record.get("sha256") != sha256_file(paths["d042_budget_rows"])
    ):
        raise ValueError("D0.4.2 budget simulation rows differ from their Manifest")
    for label, key, manifest in (
        ("normalized_train", "normalized_train", d03),
        ("f2_train", "train.jsonl", f2_manifest),
    ):
        path = paths[label]
        if label == "normalized_train":
            records = [row for row in manifest.get("outputs", []) if row.get("path") == "data/processed/v1/train.jsonl"]
            if len(records) != 1 or records[0].get("sha256") != sha256_file(path):
                raise ValueError("normalized D0.3 Train differs from its Manifest")
        else:
            record = manifest.get("outputs", {}).get("files", {}).get(key, {})
            if record.get("path") != "data/processed/sft/v1/train.jsonl" or record.get("sha256") != sha256_file(path):
                raise ValueError("F2 Train differs from its D0.4.3 Manifest")
    budget_index: dict[tuple[str, str], dict[str, Any]] = {}
    with paths["d042_budget_rows"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("split") != "train" or row.get("max_prompt_tokens") != 1280 or row.get("max_train_seq_length") != 2048:
                continue
            key = ("train", row.get("sample_id"))
            if not isinstance(key[1], str) or key in budget_index:
                raise ValueError(f"invalid/duplicate D0.4.2 budget row at line {line_number}")
            budget_index[key] = row
    return d03, f2_manifest, budget_index


def prepare_f4_full_dataset(config_path: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config, paths = _load_config(config_path, root)
    d03, f2_manifest, budget_index = _verify_sources(paths)
    normalized = _read_jsonl(paths["normalized_train"])
    f2_rows = _read_jsonl(paths["f2_train"])
    expected = config["expected_counts"]
    if len(normalized) != expected["source_train"] or len(f2_rows) != expected["effective_train"]:
        raise ValueError(f"D0.3/F2 counts differ: source={len(normalized)}, F2={len(f2_rows)}")
    normalized_ids = [row.get("sample_id") for row in normalized]
    f2_ids = [row.get("sample_id") for row in f2_rows]
    if any(not isinstance(value, str) or not value for value in normalized_ids + f2_ids):
        raise ValueError("D0.3/F2 has a missing sample ID")
    if len(f2_ids) != len(set(f2_ids)):
        raise ValueError("F2 Train contains duplicate IDs")
    _ordered_subset(normalized_ids, f2_ids)
    d03_by_id = {row["sample_id"]: row for row in normalized}
    f2_by_id = {row["sample_id"]: row for row in f2_rows}
    if not set(f2_by_id) <= set(d03_by_id):
        raise ValueError("F2 Train includes an ID absent from D0.3 Train")
    excluded_ids = [sample_id for sample_id in normalized_ids if sample_id not in f2_by_id]
    if len(excluded_ids) != expected["historical_excluded"]:
        raise ValueError("the 120 historical D0.4.3 exclusions changed")

    runner_config = _read_json(paths["inference_config"])
    from participant.run_inference import load_runtime_tokenizer

    tokenizer, tokenizer_facts = load_runtime_tokenizer(runner_config)
    expected_tokenizer = config["tokenizer"]
    observed = {
        "model_id": tokenizer_facts.get("model_id"),
        "revision": tokenizer_facts.get("revision"),
        "signature_sha256": tokenizer_facts.get("signature_sha256"),
        "chat_template_sha256": tokenizer_facts.get("template_sha256"),
        "eos_token_id": int(tokenizer.eos_token_id),
    }
    if observed != expected_tokenizer:
        raise ValueError(f"runtime Tokenizer identity differs from the locked F2 profile: {observed}")

    p0_prompt = paths["p0_prompt"].read_text(encoding="utf-8")
    f4_prompt = paths["f4_prompt"].read_text(encoding="utf-8")
    official_schema = _read_json(paths["train_schema"])
    built_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, f2_item in enumerate(f2_rows, 1):
        sample_id = f2_item["sample_id"]
        audit = budget_index.get(("train", sample_id))
        if not audit or audit.get("outcome") not in {"full_fit", "trimmed_fit"}:
            raise ValueError(f"{sample_id}: missing or unusable frozen S/P budget row")
        start = audit.get("removed_history_message_count")
        item, diagnostic = build_f4_training_item(
            d03_by_id[sample_id], f2_item,
            p0_prompt=p0_prompt, f4_prompt=f4_prompt,
            prompt_cap=1280, sequence_cap=2048, tokenizer=tokenizer,
            official_schema=official_schema, selected_start_index=start,
        )
        if item["source"] != audit.get("source"):
            raise ValueError(f"{sample_id}: D0.3 source differs from the D0.4.2 audited source")
        diagnostic.update(
            history_sha256=_lineage_hash(d03_by_id[sample_id]["history"]),
            selected_history_sha256=_lineage_hash(d03_by_id[sample_id]["history"][start:]),
            f2_sequence_tokens=len(f2_item["input_ids"]),
            f4_sequence_tokens=len(item["input_ids"]),
            supervised_tokens=diagnostic["f4_completion_tokens"],
        )
        built_rows.append(item)
        diagnostics.append(diagnostic)
        if index % 512 == 0 or index == len(f2_rows):
            print(f"[F4 Train] encoded {index}/{len(f2_rows)}", flush=True)
    if [row["sample_id"] for row in built_rows] != f2_ids:
        raise RuntimeError("F4 Train IDs or order differ from the frozen F2 cohort")

    from src.data.sft_collator import validate_dataset_batches

    parent = paths["dataset_dir"].parent
    parent.mkdir(parents=True, exist_ok=True)
    original_hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    with tempfile.TemporaryDirectory(prefix=".e03-a6-f4-full-", dir=parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        for name, rows in (("train.jsonl", built_rows), ("audit_train.jsonl", diagnostics)):
            with (stage / name).open("wb") as handle:
                for row in rows:
                    handle.write(_json_bytes(row))
        batch_validation = validate_dataset_batches(
            stage, split_names=("train",), batch_sizes=(1, 2, 4),
            pad_token_id=int(tokenizer.pad_token_id), eos_token_id=int(tokenizer.eos_token_id),
            max_sequence_length=2048, pad_to_multiple_of=None,
        )
        hashes = {name: sha256_file(stage / name) for name in ("train.jsonl", "audit_train.jsonl")}
        if original_hashes != {name: sha256_file(path) for name, path in paths.items() if path.is_file()}:
            raise RuntimeError("one or more protected F4 data inputs changed during construction")
        summaries = {
            "count": len(diagnostics),
            "ids_sha256_ordered": _lineage_hash([row["sample_id"] for row in diagnostics]),
            "ids_sha256_set": hashlib.sha256("\n".join(sorted(row["sample_id"] for row in diagnostics)).encode()).hexdigest(),
            "prompt_tokens_p0": _length_summary([row["p0_prompt_tokens"] for row in diagnostics]),
            "prompt_tokens_f4": _length_summary([row["f4_prompt_tokens"] for row in diagnostics]),
            "completion_tokens_f2": _length_summary([row["f2_completion_tokens"] for row in diagnostics]),
            "completion_tokens_f4": _length_summary([row["f4_completion_tokens"] for row in diagnostics]),
            "sequence_tokens_f4": _length_summary([row["f4_sequence_tokens"] for row in diagnostics]),
            "supervised_tokens_f4": sum(row["supervised_tokens"] for row in diagnostics),
            "prompt_token_delta_f4_minus_p0": _length_summary([row["f4_prompt_tokens"] - row["p0_prompt_tokens"] for row in diagnostics]),
            "completion_token_delta_f4_minus_f2": sum(row["f4_completion_tokens"] - row["f2_completion_tokens"] for row in diagnostics),
            "selected_start_index_counts": dict(sorted(Counter(row["selected_start_index"] for row in diagnostics).items())),
            "p0_selected_history_matches_f2": all(row["p0_selected_history_matches_f2"] for row in diagnostics),
            "completion_round_trip_lossless": all(row["completion_round_trip_lossless"] for row in diagnostics),
            "prompt_labels_masked": all(row["prompt_labels_masked"] for row in diagnostics),
            "terminal_eos_supervised": all(row["terminal_eos_supervised"] for row in diagnostics),
            "over_budget_ids": [],
            "excluded_historical_ids_sha256": _lineage_hash(excluded_ids),
            "excluded_historical_count": len(excluded_ids),
        }
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        report = {
            "version": "E03-A6-F4-data-v1",
            "status": "cpu_preparation_passed",
            "format_version": "F4",
            "history_policy": "reuse exact F2/P0/T1 selected history; replace only prompt and completion fields",
            "profile": config["profile"],
            "source_versions": {"d03": d03["pipeline_version"], "d043": f2_manifest["version"]},
            "inputs": {name: {"path": str(path.relative_to(root)), "sha256": original_hashes[name]} for name, path in paths.items() if name not in {"dataset_dir", "run_root"}},
            "prompts": {"p0_sha256": sha256_file(paths["p0_prompt"]), "f4_sha256": sha256_file(paths["f4_prompt"])},
            "tokenizer": observed,
            "git": {"branch": branch, "head": head},
            "train_summary": summaries,
            "batch_validation": batch_validation,
            "outputs": {"dataset_dir": str(paths["dataset_dir"].relative_to(root)), "files": hashes, "train_count": len(built_rows), "val_emitted": False},
            "model_training": "not_run",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(stage, paths["dataset_dir"])
    return report
