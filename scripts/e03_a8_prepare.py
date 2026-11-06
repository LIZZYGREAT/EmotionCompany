#!/usr/bin/env python3
"""Prepare paired, common-history, completion-only E03-A8 training data."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from src.data.e03_a8_sft import build_a8_training_item, canonical_sha256, make_sample_order
from src.shared.e03_a8_protocol import validate_a8_prompt_enum_order


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"A8 path escapes the project root: {value}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            rows.append(value)
    return rows


def _json_line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _load_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = _resolve(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E03-A8-data-v1" or config.get("experiment_id") != "E03-A8":
        raise ValueError("unsupported E03-A8 data configuration")
    if config.get("expected_counts") != {"source_train": 16312, "effective_train": 16192, "historical_excluded": 120}:
        raise ValueError("A8 must retain the frozen D0.4.3 Train counts")
    if config.get("profile") != {
        "max_prompt_tokens": 1280,
        "max_train_seq_length": 2048,
        "history_policy": "P0_T1_same_selected_history_then_common_P_EP_budget",
        "completion_policy": "complete_classification_only_or_block",
        "padding_side": "right",
        "packing": False,
        "padding_free": False,
    }:
        raise ValueError("A8 profile/S/P contract changed")
    paths = {name: _resolve(value) for name, value in config["inputs"].items() if name.endswith("_train") or name.endswith("_manifest") or name.endswith("_rows") or name in {"train_schema", "p0_prompt", "inference_config"}}
    paths["dataset_dir"] = _resolve(config["outputs"]["dataset_dir"])
    paths["run_root"] = _resolve(config["outputs"]["run_root"])
    for name, file_path in paths.items():
        if name not in {"dataset_dir", "run_root"} and not file_path.is_file():
            raise FileNotFoundError(f"missing A8 input {name}: {file_path}")
    if paths["dataset_dir"].exists():
        raise FileExistsError(f"refusing to overwrite prepared A8 data: {paths['dataset_dir']}")
    return config, paths


def _verify_sources(config: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    d03 = _read_json(paths["d03_manifest"])
    d042 = _read_json(paths["d042_manifest"])
    f2_manifest = _read_json(paths["f2_manifest"])
    if d03.get("pipeline_version") != "D0.3-v1" or d042.get("audit_version") != "D0.4.2-v1" or f2_manifest.get("version") != "D0.4.3-v1":
        raise ValueError("A8 source pipeline manifests differ from the frozen D0.3/D0.4 versions")
    schema_sha = sha256_file(paths["train_schema"])
    if (
        f2_manifest.get("inputs", {}).get("train_schema", {}).get("sha256") != schema_sha
        or d042.get("input_hashes_unchanged_after_run", {}).get("train_schema") != schema_sha
        or d042.get("d03_manifest", {}).get("sha256") != sha256_file(paths["d03_manifest"])
        or f2_manifest.get("d042_manifest_sha256") != sha256_file(paths["d042_manifest"])
    ):
        raise ValueError("A8 official schema/D0.3/D0.4 source lineage has changed")
    if sha256_file(paths["p0_prompt"]) != config["inputs"]["p0_sha256"]:
        raise ValueError("protected P0 prompt SHA differs from its frozen D0.4.3 identity")
    if sha256_file(paths["d042_budget_rows"]) != d042.get("result_files", {}).get("budget_simulations", {}).get("sha256"):
        raise ValueError("D0.4.2 budget rows do not match their frozen Manifest")
    d03_outputs = [row for row in d03.get("outputs", []) if row.get("path") == "data/processed/v1/train.jsonl"]
    if len(d03_outputs) != 1 or d03_outputs[0].get("sha256") != sha256_file(paths["normalized_train"]):
        raise ValueError("normalized D0.3 Train does not match its source Manifest")
    f2_train_record = f2_manifest.get("outputs", {}).get("files", {}).get("train.jsonl", {})
    if f2_train_record.get("path") != "data/processed/sft/v1/train.jsonl" or f2_train_record.get("sha256") != sha256_file(paths["f2_train"]):
        raise ValueError("frozen D0.4.3 Train does not match its source Manifest")

    budget_index: dict[tuple[str, str], dict[str, Any]] = {}
    with paths["d042_budget_rows"].open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("split") != "train" or row.get("max_prompt_tokens") != 1280 or row.get("max_train_seq_length") != 2048:
                continue
            key = ("train", row.get("sample_id"))
            if not isinstance(key[1], str) or key in budget_index:
                raise ValueError(f"D0.4.2 budget row {line_no} has a missing or duplicate ID")
            budget_index[key] = row
    return d03, f2_manifest, budget_index


def prepare_a8_dataset(config_path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    config, paths = _load_config(config_path)
    original_hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    _d03, _f2_manifest, budget_index = _verify_sources(config, paths)
    normalized = _read_jsonl(paths["normalized_train"])
    f2_rows = _read_jsonl(paths["f2_train"])
    counts = config["expected_counts"]
    if len(normalized) != counts["source_train"] or len(f2_rows) != counts["effective_train"]:
        raise ValueError(f"A8 source counts differ: normalized={len(normalized)}, F2={len(f2_rows)}")
    normalized_ids = [row.get("sample_id") for row in normalized]
    f2_ids = [row.get("sample_id") for row in f2_rows]
    if any(not isinstance(value, str) or not value for value in normalized_ids + f2_ids) or len(set(normalized_ids)) != len(normalized_ids) or len(set(f2_ids)) != len(f2_ids):
        raise ValueError("A8 source Train IDs must be non-empty and unique")
    normalized_positions = {sample_id: index for index, sample_id in enumerate(normalized_ids)}
    if any(sample_id not in normalized_positions for sample_id in f2_ids):
        raise ValueError("F2 includes a Train ID absent from normalized D0.3 Train")
    positions = [normalized_positions[sample_id] for sample_id in f2_ids]
    if positions != sorted(positions):
        raise ValueError("F2 IDs are not an ordered D0.3 Train subsequence")
    if len(set(normalized_ids) - set(f2_ids)) != counts["historical_excluded"]:
        raise ValueError("the frozen 120-row D0.4.3 exclusion count changed")
    normalized_by_id = {row["sample_id"]: row for row in normalized}

    runner_config = _read_json(paths["inference_config"])
    base_path = _resolve(runner_config["model"]["local_path"] if isinstance(runner_config.get("model"), dict) else "participant/model")
    tokenizer, tokenizer_facts = inference.load_runtime_tokenizer(runner_config, base_path)
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
        raise ValueError(f"A8 runtime tokenizer identity differs from D0.4.3: {observed_tokenizer}")
    official_schema = _read_json(paths["train_schema"])
    p0_prompt = paths["p0_prompt"].read_text(encoding="utf-8")
    p_prompt = _resolve("configs/prompts/p_a8_profile.txt").read_text(encoding="utf-8")
    ep_prompt = _resolve("configs/prompts/p_a8_emotion_profile.txt").read_text(encoding="utf-8")
    validate_a8_prompt_enum_order(p_prompt, official_schema, "P")
    validate_a8_prompt_enum_order(ep_prompt, official_schema, "EP")
    prompts = {"P": p_prompt, "EP": ep_prompt}
    prompt_hashes = {arm: hashlib.sha256(value.encode("utf-8")).hexdigest() for arm, value in prompts.items()}

    dataset_dir = paths["dataset_dir"]
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e03-a8-build-", dir=dataset_dir.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        output_handles = {arm: (stage / f"{arm.lower()}_train.jsonl").open("wb") for arm in ("P", "EP")}
        audit_handle = (stage / "common_history_audit.jsonl").open("wb")
        history_hashes: list[str] = []
        prompt_lengths = {"P": [], "EP": []}
        completion_lengths = {"P": [], "EP": []}
        sequence_lengths = {"P": [], "EP": []}
        all_ids: list[str] = []
        try:
            for index, f2_item in enumerate(f2_rows, 1):
                sample_id = f2_item["sample_id"]
                sample = normalized_by_id[sample_id]
                if sample.get("conversation_id") != f2_item.get("conversation_id"):
                    raise ValueError(f"{sample_id}: D0.3/F2 conversation identity differs")
                budget = budget_index.get(("train", sample_id))
                if not budget:
                    raise ValueError(f"{sample_id}: missing D0.4.2 P=1280/S=2048 audit row")
                rows, audit = build_a8_training_item(
                    sample,
                    f2_item,
                    budget,
                    p0_prompt=p0_prompt,
                    p_prompt=p_prompt,
                    ep_prompt=ep_prompt,
                    prompt_cap=int(config["profile"]["max_prompt_tokens"]),
                    sequence_cap=int(config["profile"]["max_train_seq_length"]),
                    tokenizer=tokenizer,
                    official_schema=official_schema,
                )
                if sample.get("source") != budget.get("source"):
                    raise ValueError(f"{sample_id}: D0.3 source differs from the D0.4.2 audit")
                for arm in ("P", "EP"):
                    row = rows[arm]
                    output_handles[arm].write(_json_line(row))
                    prompt_lengths[arm].append(row["prompt_length"])
                    completion_lengths[arm].append(row["completion_length"])
                    sequence_lengths[arm].append(len(row["input_ids"]))
                audit_handle.write(_json_line(audit))
                history_hashes.append(audit["selected_history_sha256"])
                all_ids.append(sample_id)
                if index % 512 == 0 or index == len(f2_rows):
                    print(f"[E03-A8 data] encoded {index}/{len(f2_rows)}", flush=True)
        finally:
            for handle in output_handles.values():
                handle.close()
            audit_handle.close()

        if all_ids != f2_ids:
            raise RuntimeError("A8 P/EP row IDs or order differ from the frozen 16,192-row Train cohort")
        history_hash_sha = canonical_sha256(history_hashes)
        schedule = make_sample_order(
            all_ids,
            seed=int(config["schedule"]["seed"]),
            accumulation_steps=int(config["schedule"]["gradient_accumulation_steps"]),
        )
        if len(schedule) != int(config["schedule"]["planned_steps"]):
            raise RuntimeError("A8 sample order does not yield exactly 4,048 frozen optimizer steps")
        with (stage / "sample_order.jsonl").open("wb") as handle:
            for row in schedule:
                handle.write(_json_line(row))
        schedule_ids = [sample_id for row in schedule for sample_id in row["sample_ids"]]
        if len(schedule_ids) != len(set(schedule_ids)) or set(schedule_ids) != set(all_ids):
            raise RuntimeError("A8 schedule repeats or omits a frozen Train ID")
        file_names = ("p_train.jsonl", "ep_train.jsonl", "common_history_audit.jsonl", "sample_order.jsonl")
        output_hashes = {name: sha256_file(stage / name) for name in file_names}
        manifest = {
            "version": "E03-A8-data-v1",
            "experiment_id": "E03-A8",
            "status": "cpu_preparation_passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(_resolve(config_path)),
            "inputs": {
                name: {"path": str(path.relative_to(ROOT)), "sha256": digest}
                for name, path in paths.items() if path.is_file()
                for digest in [original_hashes[name]]
            },
            "official_schema_sha256": sha256_file(paths["train_schema"]),
            "p0_prompt_sha256": sha256_file(paths["p0_prompt"]),
            "prompts": {arm: {"path": f"configs/prompts/{'p_a8_profile.txt' if arm == 'P' else 'p_a8_emotion_profile.txt'}", "sha256": prompt_hashes[arm]} for arm in ("P", "EP")},
            "code_sha256": {
                name: sha256_file(_resolve(name))
                for name in (
                    "configs/data/e03_a8.yaml",
                    "configs/prompts/p_a8_profile.txt",
                    "configs/prompts/p_a8_emotion_profile.txt",
                    "scripts/e03_a8_prepare.py",
                    "src/data/e03_a8_sft.py",
                    "src/shared/e03_a8_protocol.py",
                    "src/shared/prompting.py",
                )
            },
            "tokenizer": observed_tokenizer,
            "counts": {"normalized_source_train": len(normalized), "effective_train": len(all_ids), "historical_excluded": len(set(normalized_ids) - set(all_ids)), "common_history_sha256_ordered": history_hash_sha},
            "sample_ids_ordered_sha256": canonical_sha256(all_ids),
            "sample_ids_set_sha256": hashlib.sha256("\n".join(sorted(all_ids)).encode("utf-8")).hexdigest(),
            "schedule": {"seed": int(config["schedule"]["seed"]), "steps": len(schedule), "accumulation_steps": int(config["schedule"]["gradient_accumulation_steps"]), "micro_batches": len(schedule_ids), "ordered_ids_sha256": canonical_sha256(schedule_ids), "schedule_sha256": canonical_sha256(schedule)},
            "lengths": {
                arm: {
                    "prompt_min": min(prompt_lengths[arm]), "prompt_max": max(prompt_lengths[arm]), "prompt_mean": sum(prompt_lengths[arm]) / len(prompt_lengths[arm]),
                    "completion_min": min(completion_lengths[arm]), "completion_max": max(completion_lengths[arm]), "completion_mean": sum(completion_lengths[arm]) / len(completion_lengths[arm]),
                    "sequence_min": min(sequence_lengths[arm]), "sequence_max": max(sequence_lengths[arm]), "sequence_mean": sum(sequence_lengths[arm]) / len(sequence_lengths[arm]),
                } for arm in ("P", "EP")
            },
            "files": output_hashes,
            "git_committed_data": False,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if {name: sha256_file(path) for name, path in paths.items() if path.is_file()} != original_hashes:
            raise RuntimeError("an A8 source input changed during dataset construction")
        shutil.move(str(stage), str(dataset_dir))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build frozen paired E03-A8 completion-only Train data and sample schedule.")
    parser.add_argument("--mode", choices=("build",), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/e03_a8.yaml")
    args = parser.parse_args(argv)
    manifest = prepare_a8_dataset(args.config)
    print(json.dumps({"status": manifest["status"], "rows": manifest["counts"]["effective_train"], "schedule_steps": manifest["schedule"]["steps"], "files": manifest["files"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
