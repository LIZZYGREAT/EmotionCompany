"""Build an isolated F3 SFT dataset from the frozen D0.3/F2 cohort."""

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
from jsonschema import Draft202012Validator

from src.shared.f3_protocol import parse_f3, serialize_f3
from src.shared.prompting import (
    build_messages,
    encode_full_training_sequence,
    encode_inference_prompt,
    validate_completion_suffix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SPLITS = ("train", "dev")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _safe_path(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} path escapes project root: {value}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _ordered_subset(reference_ids: list[str], selected_ids: list[str], split: str) -> None:
    positions = {sample_id: index for index, sample_id in enumerate(reference_ids)}
    if len(positions) != len(reference_ids):
        raise ValueError(f"{split}: normalized source contains duplicate sample IDs")
    last_position = -1
    for sample_id in selected_ids:
        position = positions.get(sample_id)
        if position is None:
            raise ValueError(f"{split}: F2 sample {sample_id} is absent from normalized D0.3")
        if position <= last_position:
            raise ValueError(f"{split}: F2 sample IDs are not an ordered D0.3 subsequence")
        last_position = position


def _lineage_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_f3_training_item(
    sample: dict[str, Any],
    f2_item: dict[str, Any],
    *,
    p0_prompt: str,
    f3_prompt: str,
    prompt_cap: int,
    sequence_cap: int,
    tokenizer: Any,
    official_schema: dict[str, Any],
    selected_start_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode one F3 completion while proving exact F2/P0 T1 prefix identity."""
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id or f2_item.get("sample_id") != sample_id:
        raise ValueError("normalized and F2 rows must have the same non-empty sample_id")
    history = sample.get("history")
    target = sample.get("target")
    if not isinstance(history, list) or not history or not isinstance(target, dict):
        raise ValueError(f"{sample_id}: normalized history/target contract is invalid")
    target_schema = {
        "$schema": official_schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$ref": "#/$defs/target",
        "$defs": official_schema["$defs"],
    }
    Draft202012Validator(target_schema).validate(target)

    if selected_start_index is None:
        from src.shared.prompting import select_recent_complete_suffix

        p0_selection = select_recent_complete_suffix(history, p0_prompt, prompt_cap, tokenizer)
        selected_start_index = p0_selection["selected_start_index"]
        selected_history = p0_selection["selected_history"]
        p0_prompt_ids = p0_selection["prompt_ids"]
        floor_exceeds_cap = p0_selection["floor_exceeds_cap"]
    else:
        if isinstance(selected_start_index, bool) or not isinstance(selected_start_index, int):
            raise ValueError(f"{sample_id}: audited selected_start_index must be an integer")
        if not 0 <= selected_start_index < len(history):
            raise ValueError(f"{sample_id}: audited selected_start_index is out of range")
        selected_history = history[selected_start_index:]
        p0_prompt_ids = encode_inference_prompt(
            tokenizer, build_messages(selected_history, p0_prompt)
        )
        floor_exceeds_cap = len(p0_prompt_ids) > prompt_cap
    old_prompt_length = f2_item.get("prompt_length")
    old_ids = f2_item.get("input_ids")
    if (
        not isinstance(old_prompt_length, int)
        or not isinstance(old_ids, list)
        or old_prompt_length != len(p0_prompt_ids)
        or old_ids[:old_prompt_length] != p0_prompt_ids
    ):
        raise ValueError(f"{sample_id}: recomputed P0/T1 Token IDs differ from frozen F2 dataset")
    if floor_exceeds_cap:
        raise ValueError(f"{sample_id}: frozen F2 train sample has a P0/T1 prompt-floor overflow")

    messages = build_messages(selected_history, f3_prompt)
    f3_prompt_ids = encode_inference_prompt(tokenizer, messages)
    if len(f3_prompt_ids) > prompt_cap:
        raise ValueError(
            f"{sample_id}: F3 prompt exceeds P={prompt_cap} using the exact P0/T1 selected history "
            f"({len(f3_prompt_ids)} tokens)"
        )
    completion = serialize_f3(target, official_schema)
    enums = _f3_enums(official_schema)
    parsed = parse_f3(completion, enums)
    expected_profile = {
        name: sorted(target["user_profile"][name], key=enums[name].index)
        for name in ("personality_traits", "interests", "style")
    }
    if parsed["emotion_label"] != target["emotion_label"]:
        raise ValueError(f"{sample_id}: F3 round trip changed emotion_label")
    if parsed["user_profile"] != expected_profile:
        raise ValueError(f"{sample_id}: F3 round trip changed canonical user_profile")
    if parsed["response_text"] != target["response_text"]:
        raise ValueError(f"{sample_id}: F3 round trip changed response_text")
    if parsed["memory_refs"] != []:
        raise ValueError(f"{sample_id}: F3 parser did not produce empty memory_refs")

    full_ids = encode_full_training_sequence(tokenizer, messages, completion)
    completion_ids = validate_completion_suffix(
        tokenizer, completion, f3_prompt_ids, full_ids, sample_id=sample_id
    )
    if len(full_ids) > sequence_cap:
        raise ValueError(
            f"{sample_id}: F3 sequence exceeds S={sequence_cap} ({len(full_ids)} tokens); "
            "the frozen cohort cannot be silently reduced"
        )
    labels = [-100] * len(f3_prompt_ids) + completion_ids
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None or not completion_ids or int(eos_id) not in completion_ids:
        raise ValueError(f"{sample_id}: template EOS is missing from the supervised completion")
    if labels[:len(f3_prompt_ids)] != [-100] * len(f3_prompt_ids):
        raise RuntimeError(f"{sample_id}: F3 Prompt tokens are not fully masked")
    if labels[len(f3_prompt_ids):] != full_ids[len(f3_prompt_ids):]:
        raise RuntimeError(f"{sample_id}: F3 completion tokens are not fully supervised")

    item = {
        "sample_id": sample_id,
        "conversation_id": sample["conversation_id"],
        "source_split": f2_item["source_split"],
        "source": sample["source"],
        "prompt_length": len(f3_prompt_ids),
        "completion_length": len(completion_ids),
        "input_ids": full_ids,
        "labels": labels,
    }
    diagnostic = {
        "sample_id": sample_id,
        "source": sample["source"],
        "history_sha256": _lineage_hash(history),
        "selected_history_sha256": _lineage_hash(selected_history),
        "source_sha256": _lineage_hash(sample["source"]),
        "selected_start_index": selected_start_index,
        "retained_history_message_count": len(selected_history),
        "p0_prompt_tokens": len(p0_prompt_ids),
        "f3_prompt_tokens": len(f3_prompt_ids),
        "f2_completion_tokens": int(f2_item["completion_length"]),
        "f3_completion_tokens": len(completion_ids),
        "f2_sequence_tokens": len(old_ids),
        "f3_sequence_tokens": len(full_ids),
        "supervised_tokens": len(completion_ids),
        "p0_selected_history_matches_f2": True,
        "completion_round_trip_lossless": True,
        "prompt_labels_masked": True,
        "terminal_eos_supervised": int(eos_id) in labels[len(f3_prompt_ids):],
    }
    return item, diagnostic


def _f3_enums(official_schema: dict[str, Any]) -> dict[str, list[str]]:
    definitions = official_schema["$defs"]
    return {
        "emotion_label": [value for value in definitions["emotion"]["enum"] if value],
        **{
            name: list(definitions["profile"]["properties"][name]["items"]["enum"])
            for name in ("personality_traits", "interests", "style")
        },
    }


def _nearest_rank(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, (percentile * len(ordered) + 99) // 100 - 1)]


def _length_summary(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "total": sum(values),
        "p50": _nearest_rank(values, 50),
        "p90": _nearest_rank(values, 90),
        "p95": _nearest_rank(values, 95),
        "p99": _nearest_rank(values, 99),
        "max": max(values) if values else None,
    }


def _load_config(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E03-A6-F3-data-v1":
        raise ValueError("unsupported E03-A6 F3 data config")
    if config.get("format_version") != "F3":
        raise ValueError("data config must select format_version=F3 explicitly")
    profile = config.get("profile", {})
    if profile.get("packing") is not False or profile.get("padding_free") is not False:
        raise ValueError("F3 data requires packing=false and padding_free=false")
    if profile.get("padding_side") != "right" or profile.get("completion_policy") != "complete_or_block":
        raise ValueError("F3 data requires right padding and complete_or_block")
    paths = {
        "normalized_train": _safe_path(root, config["inputs"]["normalized_dir"] + "/train.jsonl", "normalized_train"),
        "normalized_dev": _safe_path(root, config["inputs"]["normalized_dir"] + "/dev.jsonl", "normalized_dev"),
        "f2_train": _safe_path(root, config["inputs"]["f2_dataset_dir"] + "/train.jsonl", "f2_train"),
        "f2_dev": _safe_path(root, config["inputs"]["f2_dataset_dir"] + "/dev.jsonl", "f2_dev"),
        **{
            name: _safe_path(root, config["inputs"][name], name)
            for name in (
                "f2_manifest", "d03_manifest", "d042_manifest", "d042_budget_rows",
                "train_schema", "p0_prompt", "f3_prompt", "inference_config",
            )
        },
        "dataset_dir": _safe_path(root, config["outputs"]["dataset_dir"], "outputs.dataset_dir"),
        "run_dir": _safe_path(root, config["outputs"]["run_dir"], "outputs.run_dir"),
    }
    for label, path in paths.items():
        if label not in {"dataset_dir", "run_dir"} and not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if paths["dataset_dir"].exists() or paths["run_dir"].exists():
        raise FileExistsError("refusing to overwrite existing E03-A6 F3 data or preparation run")
    if paths["dataset_dir"].is_relative_to(paths["run_dir"]) or paths["run_dir"].is_relative_to(paths["dataset_dir"]):
        raise ValueError("dataset_dir and run_dir must be disjoint")
    return config, paths


def _verify_manifest_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    d03 = _read_json(paths["d03_manifest"])
    d042 = _read_json(paths["d042_manifest"])
    f2 = _read_json(paths["f2_manifest"])
    if (
        d03.get("pipeline_version") != "D0.3-v1"
        or d042.get("audit_version") != "D0.4.2-v1"
        or f2.get("version") != "D0.4.3-v1"
    ):
        raise ValueError("D0.3/F2 manifest versions do not match the locked source datasets")
    if d042.get("d03_manifest", {}).get("sha256") != sha256_file(paths["d03_manifest"]):
        raise ValueError("D0.4.2 budget audit was produced from another D0.3 manifest")
    if f2.get("d042_manifest_sha256") != sha256_file(paths["d042_manifest"]):
        raise ValueError("F2 manifest references a different D0.4.2 budget audit")
    audit_record = d042.get("result_files", {}).get("budget_simulations", {})
    if (
        audit_record.get("path") != paths["d042_budget_rows"].relative_to(PROJECT_ROOT).as_posix()
        or audit_record.get("sha256") != sha256_file(paths["d042_budget_rows"])
    ):
        raise ValueError("D0.4.2 budget rows differ from their frozen manifest")
    for split in DATA_SPLITS:
        normalized_path = paths[f"normalized_{split}"].relative_to(PROJECT_ROOT).as_posix()
        d03_records = [entry for entry in d03.get("outputs", []) if entry.get("path") == normalized_path]
        if len(d03_records) != 1 or d03_records[0].get("sha256") != sha256_file(paths[f"normalized_{split}"]):
            raise ValueError(f"D0.3 {split} hash does not match the source manifest")
        f2_relative = paths[f"f2_{split}"].relative_to(PROJECT_ROOT).as_posix()
        f2_record = f2.get("outputs", {}).get("files", {}).get(f"{split}.jsonl", {})
        if f2_record.get("path") != f2_relative or f2_record.get("sha256") != sha256_file(paths[f"f2_{split}"]):
            raise ValueError(f"F2 {split} hash does not match the source manifest")
    return {"d03": d03, "d042": d042, "f2": f2}


def _load_d042_index(
    path: Path, prompt_cap: int, sequence_cap: int
) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("max_train_seq_length") != sequence_cap or row.get("max_prompt_tokens") != prompt_cap:
                continue
            key = (row.get("split"), row.get("sample_id"))
            if not all(isinstance(value, str) and value for value in key):
                raise ValueError(f"D0.4.2 row {line_number} lacks split/sample_id")
            if key in selected:
                raise ValueError(f"D0.4.2 contains a duplicate S={sequence_cap}/P={prompt_cap} row for {key}")
            selected[key] = row
    if not selected:
        raise ValueError(f"D0.4.2 contains no S={sequence_cap}/P={prompt_cap} rows")
    return selected


def prepare_f3_dataset(root: Path, config: dict[str, Any], paths: dict[str, Path], tokenizer: Any) -> dict[str, Any]:
    manifests = _verify_manifest_inputs(paths)
    expected_counts = config["expected_counts"]
    p0_prompt = paths["p0_prompt"].read_text(encoding="utf-8")
    f3_prompt = paths["f3_prompt"].read_text(encoding="utf-8")
    p0_bytes = paths["p0_prompt"].read_bytes()
    f3_bytes = paths["f3_prompt"].read_bytes()
    official_schema = _read_json(paths["train_schema"])
    Draft202012Validator.check_schema(official_schema)
    profile = config["profile"]
    prompt_cap = profile["max_prompt_tokens"]
    sequence_cap = profile["max_train_seq_length"]
    d042_index = _load_d042_index(paths["d042_budget_rows"], prompt_cap, sequence_cap)

    normalized = {split: _read_jsonl(paths[f"normalized_{split}"]) for split in DATA_SPLITS}
    f2_rows = {split: _read_jsonl(paths[f"f2_{split}"]) for split in DATA_SPLITS}
    for split in DATA_SPLITS:
        if len(f2_rows[split]) != expected_counts[split]:
            raise ValueError(
                f"{split}: frozen F2 count {len(f2_rows[split])} != expected {expected_counts[split]}"
            )
        normalized_ids = [row.get("sample_id") for row in normalized[split]]
        f2_ids = [row.get("sample_id") for row in f2_rows[split]]
        if any(not isinstance(item, str) or not item for item in normalized_ids + f2_ids):
            raise ValueError(f"{split}: one or more normalized/F2 sample IDs are missing")
        _ordered_subset(normalized_ids, f2_ids, split)
        if len(f2_ids) != len(set(f2_ids)):
            raise ValueError(f"{split}: F2 source cohort contains duplicate sample IDs")

    built_splits: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    for split in DATA_SPLITS:
        source_by_id = {row["sample_id"]: row for row in normalized[split]}
        built_splits[split] = []
        diagnostics[split] = []
        for row_index, f2_item in enumerate(f2_rows[split], 1):
            sample = source_by_id[f2_item["sample_id"]]
            audit_row = d042_index.get((split, f2_item["sample_id"]))
            if audit_row is None:
                raise ValueError(f"{split}/{f2_item['sample_id']}: missing frozen D0.4.2 budget row")
            if audit_row.get("outcome") not in {"full_fit", "trimmed_fit"}:
                raise ValueError(f"{split}/{f2_item['sample_id']}: F2 cohort is not usable under its frozen audit")
            item, diagnostic = build_f3_training_item(
                sample,
                f2_item,
                p0_prompt=p0_prompt,
                f3_prompt=f3_prompt,
                prompt_cap=prompt_cap,
                sequence_cap=sequence_cap,
                tokenizer=tokenizer,
                official_schema=official_schema,
                selected_start_index=audit_row.get("removed_history_message_count"),
            )
            if item["source"] != audit_row.get("source"):
                raise ValueError(f"{split}/{f2_item['sample_id']}: normalized source differs from D0.4.2 audit")
            built_splits[split].append(item)
            diagnostics[split].append(diagnostic)
            if row_index % 256 == 0 or row_index == len(f2_rows[split]):
                print(f"[{split}] prepared {row_index}/{len(f2_rows[split])} F3 samples", flush=True)
        built_ids = [row["sample_id"] for row in built_splits[split]]
        if built_ids != [row["sample_id"] for row in f2_rows[split]]:
            raise RuntimeError(f"{split}: F3 construction changed F2 row identity or order")

    original_hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    parent = paths["dataset_dir"].parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e03-a6-f3-", dir=parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        for split in DATA_SPLITS:
            with (stage / f"{split}.jsonl").open("wb") as handle:
                for row in built_splits[split]:
                    handle.write(_json_bytes(row))
            with (stage / f"audit_{split}.jsonl").open("wb") as handle:
                for row in diagnostics[split]:
                    handle.write(_json_bytes(row))

        from src.data.sft_collator import validate_dataset_batches

        batch_validation = validate_dataset_batches(
            stage,
            split_names=DATA_SPLITS,
            batch_sizes=(1, 2, 4),
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            max_sequence_length=sequence_cap,
            pad_to_multiple_of=None,
        )
        file_hashes = {
            name: sha256_file(stage / name)
            for name in (
                "train.jsonl", "dev.jsonl", "audit_train.jsonl", "audit_dev.jsonl"
            )
        }
        split_summary: dict[str, Any] = {}
        for split in DATA_SPLITS:
            rows = diagnostics[split]
            split_summary[split] = {
                "count": len(rows),
                "ids_sha256": _lineage_hash([row["sample_id"] for row in rows]),
                "id_order_matches_f2": True,
                "prompt_tokens_p0": _length_summary([row["p0_prompt_tokens"] for row in rows]),
                "p0_selected_history_matches_f2_count": sum(row["p0_selected_history_matches_f2"] for row in rows),
                "completion_round_trip_lossless_count": sum(row["completion_round_trip_lossless"] for row in rows),
                "prompt_labels_masked_count": sum(row["prompt_labels_masked"] for row in rows),
                "terminal_eos_supervised_count": sum(row["terminal_eos_supervised"] for row in rows),
                "prompt_tokens": _length_summary([row["f3_prompt_tokens"] for row in rows]),
                "prompt_token_delta_f3_minus_p0": _length_summary([
                    row["f3_prompt_tokens"] - row["p0_prompt_tokens"] for row in rows
                ]),
                "completion_tokens_f2": _length_summary([row["f2_completion_tokens"] for row in rows]),
                "completion_tokens_f3": _length_summary([row["f3_completion_tokens"] for row in rows]),
                "sequence_tokens_f3": _length_summary([row["f3_sequence_tokens"] for row in rows]),
                "supervised_tokens_f3": sum(row["supervised_tokens"] for row in rows),
                "total_f2_completion_tokens": sum(row["f2_completion_tokens"] for row in rows),
                "total_f3_completion_tokens": sum(row["f3_completion_tokens"] for row in rows),
                "completion_token_delta_f3_minus_f2": sum(row["f3_completion_tokens"] - row["f2_completion_tokens"] for row in rows),
                "selected_start_index_counts": dict(sorted(Counter(row["selected_start_index"] for row in rows).items())),
            }

        input_hashes_after = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
        if original_hashes != input_hashes_after:
            raise RuntimeError("one or more locked F3 preparation inputs changed during construction")
        tokenizer_facts = {
            "model_id": config["tokenizer"]["model_id"],
            "revision": config["tokenizer"]["revision"],
            "signature_sha256": config["tokenizer"]["signature_sha256"],
            "chat_template_sha256": config["tokenizer"]["chat_template_sha256"],
            "eos_token_id": int(tokenizer.eos_token_id),
        }
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        code_paths = (
            PROJECT_ROOT / "src/shared/f3_protocol.py",
            PROJECT_ROOT / "src/shared/prompting.py",
            PROJECT_ROOT / "src/data/e03_a6_sft.py",
            PROJECT_ROOT / "scripts/e03_a6_prepare.py",
        )
        report = {
            "version": "E03-A6-F3-data-v1",
            "status": "cpu_preparation_passed",
            "format_version": "F3",
            "history_policy": "select P0/T1 history first, then retokenize that exact history with P_A6_F3",
            "profile": profile,
            "split_summary": split_summary,
            "inputs": {
                name: {"path": str(path.relative_to(root)), "sha256": original_hashes[name]}
                for name, path in paths.items() if name not in {"dataset_dir", "run_dir"}
            },
            "manifests": {
                "d03_sha256": sha256_file(paths["d03_manifest"]),
                "f2_sft_sha256": sha256_file(paths["f2_manifest"]),
                "d03_version": manifests["d03"]["pipeline_version"],
                "f2_version": manifests["f2"]["version"],
            },
            "prompts": {
                "p0_sha256": hashlib.sha256(p0_bytes).hexdigest(),
                "f3_sha256": hashlib.sha256(f3_bytes).hexdigest(),
            },
            "tokenizer": tokenizer_facts,
            "git": {"branch": git_branch, "head": git_head},
            "config_sha256": config.get("_config_sha256"),
            "code_sha256": {
                path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
                for path in code_paths
            },
            "batch_validation": batch_validation,
            "outputs": {
                "dataset_dir": str(paths["dataset_dir"].relative_to(root)),
                "files": file_hashes,
                "train_count": len(built_splits["train"]),
                "dev_count": len(built_splits["dev"]),
                "val_emitted": False,
            },
            "model_training": "not_run",
            "pilot": "not_run",
            "full_training": "not_run_user_approval_required",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (stage / "manifest.json").write_bytes(
            (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        )
        os.replace(stage, paths["dataset_dir"])

    paths["run_dir"].mkdir(parents=True, exist_ok=False)
    run_report = paths["run_dir"] / "prepare_summary.json"
    run_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def load_and_prepare(config_path: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config_path = _safe_path(root, str(config_path), "--config")
    config, paths = _load_config(root, config_path)
    config["_config_sha256"] = sha256_file(config_path)
    from participant.run_inference import load_runtime_tokenizer

    runner_config = _read_json(paths["inference_config"])
    tokenizer, facts = load_runtime_tokenizer(runner_config)
    expected = config.get("tokenizer", {})
    for field, key in (
        ("model_id", "model_id"),
        ("revision", "revision"),
        ("signature_sha256", "signature_sha256"),
        ("template_sha256", "chat_template_sha256"),
        ("eos_token_id", "eos_token_id"),
    ):
        expected_value = expected.get(key)
        actual_value = facts.get(field)
        if expected_value is not None and expected_value != actual_value:
            raise ValueError(f"tokenizer {field} differs from the frozen F2 protocol")
    config["tokenizer"] = {
        "model_id": facts["model_id"],
        "revision": facts["revision"],
        "signature_sha256": facts["signature_sha256"],
        "chat_template_sha256": facts["template_sha256"],
    }
    return prepare_f3_dataset(root, config, paths, tokenizer)
