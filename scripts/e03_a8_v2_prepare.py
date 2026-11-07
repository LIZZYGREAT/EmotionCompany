#!/usr/bin/env python3
"""Prepare isolated A8-V2 paired Train data and a stratified first-window schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from scripts.e03_a8_prepare import _read_json, _read_jsonl, _resolve, _verify_sources, sha256_file
from src.data.e03_a8_schedule import build_a8_v2_schedule
from src.data.e03_a8_sft import build_a8_training_item, canonical_sha256
from src.shared.e03_a8_protocol import validate_a8_prompt_enum_order


def _json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot summarize an empty token-length group")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999)))
    return ordered[index]


def _reorder_jsonl(path: Path, ids: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".ordered")
    offsets: dict[str, int] = {}
    with path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in offsets:
                raise ValueError(f"{path}: missing or duplicate sample_id while reordering")
            offsets[sample_id] = offset
        if set(offsets) != set(ids) or len(offsets) != len(ids):
            raise ValueError(f"{path}: output IDs differ from the frozen A8-V2 schedule")
        source.seek(0)
        with temporary.open("wb") as destination:
            for sample_id in ids:
                source.seek(offsets[sample_id])
                line = source.readline()
                if json.loads(line).get("sample_id") != sample_id:
                    raise RuntimeError("A8-V2 JSONL byte index resolved to a different ID")
                destination.write(line)
            destination.flush()
            os.fsync(destination.fileno())
    os.replace(temporary, path)


def _load_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E03-A8-V2-data-v1" or config.get("experiment_id") != "E03-A8-V2":
        raise ValueError("unsupported E03-A8-V2 data configuration")
    if config.get("expected_counts") != {"source_train": 16312, "effective_train": 16192, "historical_excluded": 120}:
        raise ValueError("A8-V2 must retain the frozen D0.4.3 Train counts")
    if config.get("profile") != {
        "max_prompt_tokens": 1280,
        "max_train_seq_length": 2048,
        "history_policy": "P0_T1_same_selected_history_then_common_P_EP_budget",
        "completion_policy": "complete_classification_only_or_block",
        "padding_side": "right",
        "packing": False,
        "padding_free": False,
    }:
        raise ValueError("A8-V2 changed the frozen P/S, history, or Completion contract")
    schedule = config.get("schedule", {})
    if schedule != {
        "algorithm": "a8-v2-greedy-stratified-75-25-v1",
        "seed": 42,
        "gradient_accumulation_steps": 4,
        "planned_steps": 4048,
        "first_window_rows": 2048,
        "first_window_nonempty_rows": 1536,
        "first_window_all_empty_rows": 512,
    }:
        raise ValueError("A8-V2 first-window schedule differs from the approved 75/25 plan")
    input_paths = {name: _resolve(value) for name, value in config["inputs"].items() if isinstance(value, str) and (name.endswith("_train") or name.endswith("_manifest") or name.endswith("_rows") or name in {"train_schema", "p0_prompt", "inference_config"})}
    output_path = _resolve(config["outputs"]["dataset_dir"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite A8-V2 prepared data: {output_path}")
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing A8-V2 input {name}: {path}")
    return config, {**input_paths, "dataset_dir": output_path, "config": config_path}


def prepare_a8_v2_dataset(config_path: Path) -> dict[str, Any]:
    config, paths = _load_inputs(config_path)
    original_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name not in {"dataset_dir", "config"}
    }
    _d03, _f2_manifest, budget_index = _verify_sources(config, paths)
    normalized = _read_jsonl(paths["normalized_train"])
    f2_rows = _read_jsonl(paths["f2_train"])
    expected = config["expected_counts"]
    if len(normalized) != expected["source_train"] or len(f2_rows) != expected["effective_train"]:
        raise ValueError(f"A8-V2 source counts differ: normalized={len(normalized)}, F2={len(f2_rows)}")
    normalized_ids = [row.get("sample_id") for row in normalized]
    f2_ids = [row.get("sample_id") for row in f2_rows]
    if any(not isinstance(value, str) or not value for value in normalized_ids + f2_ids) or len(set(normalized_ids)) != len(normalized_ids) or len(set(f2_ids)) != len(f2_ids):
        raise ValueError("A8-V2 source Train IDs must be non-empty and unique")
    normalized_positions = {sample_id: index for index, sample_id in enumerate(normalized_ids)}
    if any(sample_id not in normalized_positions for sample_id in f2_ids):
        raise ValueError("F2 includes a Train ID absent from normalized D0.3 Train")
    if [normalized_positions[sample_id] for sample_id in f2_ids] != sorted(normalized_positions[sample_id] for sample_id in f2_ids):
        raise ValueError("F2 IDs are not an ordered D0.3 Train subsequence")
    if len(set(normalized_ids) - set(f2_ids)) != expected["historical_excluded"]:
        raise ValueError("the frozen 120-row D0.4.3 exclusion count changed")
    normalized_by_id = {row["sample_id"]: row for row in normalized}

    runner_config = _read_json(paths["inference_config"])
    tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runner_config)
    expected_tokenizer = {
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": runner_config.get("model", {}).get("revision", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
        "signature_sha256": config["inputs"]["tokenizer_signature_sha256"],
        "template_sha256": config["inputs"]["chat_template_sha256"],
        "eos_token_id": int(config["inputs"]["eos_token_id"]),
    }
    observed_tokenizer = {
        "model_id": tokenizer_facts.get("model_id"),
        "revision": tokenizer_facts.get("revision"),
        "signature_sha256": tokenizer_facts.get("signature_sha256"),
        "template_sha256": tokenizer_facts.get("template_sha256"),
        "eos_token_id": int(tokenizer.eos_token_id),
    }
    if observed_tokenizer != expected_tokenizer:
        raise ValueError(f"A8-V2 runtime tokenizer identity differs from D0.4.3: {observed_tokenizer}")
    official_schema = _read_json(paths["train_schema"])
    p0_prompt = paths["p0_prompt"].read_text(encoding="utf-8")
    prompts = {arm: _resolve(path).read_text(encoding="utf-8") for arm, path in config["prompts"].items()}
    enum_map = {
        "P": validate_a8_prompt_enum_order(prompts["P"], official_schema, "P"),
        "EP": validate_a8_prompt_enum_order(prompts["EP"], official_schema, "EP"),
    }
    if {tuple(enum_map["P"][dim]) for dim in ("personality_traits", "interests", "style")} != {tuple(enum_map["EP"][dim]) for dim in ("personality_traits", "interests", "style")}:
        raise ValueError("A8-V2 P and EP profile enums differ")
    prompt_hashes = {arm: hashlib.sha256(text.encode("utf-8")).hexdigest() for arm, text in prompts.items()}
    data_dir = paths["dataset_dir"]
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".e03-a8-v2-build-", dir=data_dir.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        handles = {arm: (stage / f"{arm.lower()}_train.jsonl").open("wb") for arm in ("P", "EP")}
        audit_handle = (stage / "common_history_audit.jsonl").open("wb")
        candidates: list[dict[str, Any]] = []
        history_hashes_by_id: dict[str, str] = {}
        prompt_lengths = {"P": [], "EP": []}
        completion_lengths = {"P": [], "EP": []}
        sequence_lengths = {"P": [], "EP": []}
        try:
            for index, f2_item in enumerate(f2_rows, 1):
                sample_id = f2_item["sample_id"]
                sample = normalized_by_id[sample_id]
                if sample.get("conversation_id") != f2_item.get("conversation_id"):
                    raise ValueError(f"{sample_id}: D0.3/F2 conversation identity differs")
                budget = budget_index.get(("train", sample_id))
                if not budget:
                    raise ValueError(f"{sample_id}: missing frozen D0.4.2 P=1280/S=2048 audit row")
                rows, audit = build_a8_training_item(
                    sample,
                    f2_item,
                    budget,
                    p0_prompt=p0_prompt,
                    p_prompt=prompts["P"],
                    ep_prompt=prompts["EP"],
                    prompt_cap=int(config["profile"]["max_prompt_tokens"]),
                    sequence_cap=int(config["profile"]["max_train_seq_length"]),
                    tokenizer=tokenizer,
                    official_schema=official_schema,
                )
                if sample.get("source") != budget.get("source"):
                    raise ValueError(f"{sample_id}: D0.3 source differs from the D0.4.2 audit")
                for arm in ("P", "EP"):
                    row = rows[arm]
                    handles[arm].write(_json_line(row))
                    prompt_lengths[arm].append(row["prompt_length"])
                    completion_lengths[arm].append(row["completion_length"])
                    sequence_lengths[arm].append(len(row["input_ids"]))
                audit_handle.write(_json_line(audit))
                history_hashes_by_id[sample_id] = audit["selected_history_sha256"]
                candidates.append({
                    "sample_id": sample_id,
                    "conversation_id": sample["conversation_id"],
                    "profile": {dim: list(sample["target"]["user_profile"][dim]) for dim in ("personality_traits", "interests", "style")},
                    "history_message_count": len(sample["history"]) - int(audit["selected_start_index"]),
                    "prompt_tokens": {arm: int(rows[arm]["prompt_length"]) for arm in ("P", "EP")},
                    "completion_tokens": {arm: int(rows[arm]["completion_length"]) for arm in ("P", "EP")},
                    "sequence_tokens": {arm: len(rows[arm]["input_ids"]) for arm in ("P", "EP")},
                })
                if index % 512 == 0 or index == len(f2_rows):
                    print(f"[E03-A8-V2 data] encoded {index}/{len(f2_rows)}", flush=True)
        finally:
            for handle in handles.values():
                handle.close()
            audit_handle.close()

        if [row["sample_id"] for row in candidates] != f2_ids:
            raise RuntimeError("A8-V2 encoded IDs differ from the frozen 16,192-row Train cohort")
        schedule, schedule_audit = build_a8_v2_schedule(
            candidates,
            seed=int(config["schedule"]["seed"]),
            accumulation_steps=int(config["schedule"]["gradient_accumulation_steps"]),
            first_window_rows=int(config["schedule"]["first_window_rows"]),
            first_window_nonempty=int(config["schedule"]["first_window_nonempty_rows"]),
        )
        observed_first_window_labels = schedule_audit["first_window_profile_label_counts"]
        schedule_audit["first_window_profile_label_counts"] = {
            label: int(observed_first_window_labels.get(label, 0))
            for dimension in ("personality_traits", "interests", "style")
            for label in enum_map["P"][dimension]
        }
        if len(schedule) != int(config["schedule"]["planned_steps"]):
            raise RuntimeError("A8-V2 schedule does not yield exactly 4,048 optimizer steps")
        ordered_ids = [sample_id for step in schedule for sample_id in step["sample_ids"]]
        if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != set(f2_ids):
            raise RuntimeError("A8-V2 schedule repeats, omits, or adds a frozen Train ID")
        if schedule_audit["first_window_nonempty_rows"] != 1536 or schedule_audit["first_window_all_empty_rows"] != 512:
            raise RuntimeError("A8-V2 first window failed the 1,536/512 gold composition gate")
        _reorder_jsonl(stage / "p_train.jsonl", ordered_ids)
        _reorder_jsonl(stage / "ep_train.jsonl", ordered_ids)
        _reorder_jsonl(stage / "common_history_audit.jsonl", ordered_ids)
        with (stage / "sample_order.jsonl").open("wb") as handle:
            for row in schedule:
                handle.write(_json_line(row))
            handle.flush()
            os.fsync(handle.fileno())

        first_ids = set(ordered_ids[:int(config["schedule"]["first_window_rows"])])
        first_candidates = [row for row in candidates if row["sample_id"] in first_ids]
        remainder_candidates = [row for row in candidates if row["sample_id"] not in first_ids]
        token_audit = {
            subset: {
                arm: {
                    field: {"p50": _percentile([int(row[field][arm]) for row in sample_rows], 0.50), "p90": _percentile([int(row[field][arm]) for row in sample_rows], 0.90)}
                    for field in ("prompt_tokens", "completion_tokens")
                }
                for arm in ("P", "EP")
            }
            for subset, sample_rows in (("first_window", first_candidates), ("remainder", remainder_candidates))
        }
        history_length_summary = {
            "first_window": {
                "min": min(row["history_message_count"] for row in first_candidates),
                "p50": _percentile([int(row["history_message_count"]) for row in first_candidates], 0.50),
                "p90": _percentile([int(row["history_message_count"]) for row in first_candidates], 0.90),
                "max": max(row["history_message_count"] for row in first_candidates),
            },
            "remainder": {
                "min": min(row["history_message_count"] for row in remainder_candidates),
                "p50": _percentile([int(row["history_message_count"]) for row in remainder_candidates], 0.50),
                "p90": _percentile([int(row["history_message_count"]) for row in remainder_candidates], 0.90),
                "max": max(row["history_message_count"] for row in remainder_candidates),
            },
        }
        file_names = ("p_train.jsonl", "ep_train.jsonl", "common_history_audit.jsonl", "sample_order.jsonl")
        file_hashes = {name: sha256_file(stage / name) for name in file_names}
        code_paths = (
            "configs/data/e03_a8_v2.yaml",
            "configs/prompts/p_a8_v2_profile.txt",
            "configs/prompts/p_a8_v2_emotion_profile.txt",
            "scripts/e03_a8_v2_prepare.py",
            "scripts/e03_a8_prepare.py",
            "src/data/e03_a8_schedule.py",
            "src/data/e03_a8_sft.py",
            "src/shared/e03_a8_protocol.py",
            "src/shared/prompting.py",
        )
        manifest = {
            "version": "E03-A8-V2-data-v1",
            "experiment_id": "E03-A8-V2",
            "status": "cpu_preparation_passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(paths["config"]),
            "inputs": {
                name: {"path": str(path.relative_to(ROOT)), "sha256": original_hashes[name]}
                for name, path in paths.items() if name not in {"dataset_dir", "config"}
            },
            "official_schema_sha256": sha256_file(paths["train_schema"]),
            "p0_prompt_sha256": sha256_file(paths["p0_prompt"]),
            "prompts": {arm: {"path": config["prompts"][arm], "sha256": prompt_hashes[arm]} for arm in ("P", "EP")},
            "code_sha256": {name: sha256_file(_resolve(name)) for name in code_paths},
            "tokenizer": observed_tokenizer,
            "counts": {
                "normalized_source_train": len(normalized),
                "effective_train": len(ordered_ids),
                "historical_excluded": len(set(normalized_ids) - set(f2_ids)),
                "common_history_sha256_ordered": canonical_sha256([history_hashes_by_id[sample_id] for sample_id in ordered_ids]),
            },
            "sample_ids_ordered_sha256": canonical_sha256(ordered_ids),
            "sample_ids_set_sha256": hashlib.sha256("\n".join(sorted(ordered_ids)).encode("utf-8")).hexdigest(),
            "schedule": {
                "algorithm": config["schedule"]["algorithm"],
                "seed": int(config["schedule"]["seed"]),
                "steps": len(schedule),
                "accumulation_steps": int(config["schedule"]["gradient_accumulation_steps"]),
                "micro_batches": len(ordered_ids),
                "ordered_ids_sha256": canonical_sha256(ordered_ids),
                "schedule_sha256": canonical_sha256(schedule),
                "audit": schedule_audit,
            },
            "lengths": {
                arm: {
                    "prompt_min": min(prompt_lengths[arm]),
                    "prompt_max": max(prompt_lengths[arm]),
                    "prompt_mean": sum(prompt_lengths[arm]) / len(prompt_lengths[arm]),
                    "completion_min": min(completion_lengths[arm]),
                    "completion_max": max(completion_lengths[arm]),
                    "completion_mean": sum(completion_lengths[arm]) / len(completion_lengths[arm]),
                    "sequence_min": min(sequence_lengths[arm]),
                    "sequence_max": max(sequence_lengths[arm]),
                    "sequence_mean": sum(sequence_lengths[arm]) / len(sequence_lengths[arm]),
                } for arm in ("P", "EP")
            },
            "first_window_history_message_count": history_length_summary,
            "first_window_and_remainder_token_length_p50_p90": token_audit,
            "files": file_hashes,
            "git_committed_data": False,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if {name: sha256_file(path) for name, path in paths.items() if name not in {"dataset_dir", "config"}} != original_hashes:
            raise RuntimeError("an A8-V2 protected/source input changed during dataset construction")
        shutil.move(str(stage), str(data_dir))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build paired A8-V2 classification Train data and schedule.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/e03_a8_v2.yaml")
    args = parser.parse_args(argv)
    manifest = prepare_a8_v2_dataset(args.config)
    print(json.dumps({
        "status": manifest["status"],
        "rows": manifest["counts"]["effective_train"],
        "schedule_steps": manifest["schedule"]["steps"],
        "schedule_audit": manifest["schedule"]["audit"],
        "files": manifest["files"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
