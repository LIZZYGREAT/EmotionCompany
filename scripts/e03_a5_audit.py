#!/usr/bin/env python3
"""Run the CPU-only E03-A5 supervision, provenance, and split audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.e03_a5_labels import (
    DIMENSIONS,
    LABEL_ORDER_VERSION,
    TARGET_POLICY,
    build_supervision,
    canonical_sha256,
    freeze_conversation_splits,
    iter_jsonl,
    schema_label_order,
    sha256_file,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def _verify(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{description} SHA-256 differs from the pinned A5 identity")
    return observed


def _manifest_output(manifest: Mapping[str, Any], relative_path: str) -> Mapping[str, Any]:
    matches = [item for item in manifest.get("outputs", []) if item.get("path") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"D0.3 Manifest must bind exactly one {relative_path} output")
    return matches[0]


def verify_locked_sources(config: Mapping[str, Any]) -> dict[str, str]:
    identity, data = config["identity"], config["data"]
    expected_paths = {
        "normalized_train": (data["normalized_train"], identity["normalized_train_sha256"]),
        "sft_train": (data["sft_train"], identity["sft_train_sha256"]),
        "d03_manifest": (data["d03_manifest"], identity["d03_manifest_sha256"]),
        "d042_manifest": (data["d042_manifest"], identity["d042_manifest_sha256"]),
        "d043_manifest": (data["d043_manifest"], identity["d043_manifest_sha256"]),
        "excluded_ledger": (data["excluded_ledger"], identity["excluded_ledger_sha256"]),
        "official_train_schema": (data["official_train_schema"], identity["official_train_schema_sha256"]),
        "official_output_schema": (data["official_output_schema"], identity["official_output_schema_sha256"]),
        "p0": (data["prompt"], identity["p0_sha256"]),
        "runtime_inference_config": (
            data["runtime_inference_config"], identity["runtime_inference_config_sha256"]
        ),
        "e01_b_manifest": (data["e01_b_manifest"], identity["e01_b_manifest_sha256"]),
    }
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, (value, expected) in expected_paths.items():
        path = _resolve(value)
        paths[name] = path
        hashes[name] = _verify(path, expected, name)

    d03 = _json(paths["d03_manifest"])
    d042 = _json(paths["d042_manifest"])
    d043 = _json(paths["d043_manifest"])
    e01 = _json(paths["e01_b_manifest"])
    if d03.get("pipeline_version") != "D0.3-v1":
        raise ValueError("D0.3 Manifest version is not D0.3-v1")
    d03_train = _manifest_output(d03, data["normalized_train"])
    if (
        d03_train.get("sha256") != identity["normalized_train_sha256"]
        or d03_train.get("records") != data["expected_d03_train_rows"]
    ):
        raise ValueError("D0.3 Manifest does not bind the frozen Train row count and SHA")
    if d042.get("audit_version") != "D0.4.2-v1":
        raise ValueError("D0.4.2 Manifest version is not D0.4.2-v1")
    if d042.get("d03_manifest", {}).get("sha256") != identity["d03_manifest_sha256"]:
        raise ValueError("D0.4.2 Manifest does not bind the frozen D0.3 Manifest")
    if d042.get("p0", {}).get("sha256") != identity["p0_sha256"]:
        raise ValueError("D0.4.2 Manifest does not bind the frozen P0")
    if d042.get("normalized_inputs", {}).get("train", {}).get("sha256_manifest") != identity["normalized_train_sha256"]:
        raise ValueError("D0.4.2 Manifest does not bind the normalized Train SHA")
    tokenizer = d042.get("tokenizer", {})
    expected_tokenizer = {
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": identity["tokenizer_revision"],
        "resolved_revision": identity["tokenizer_revision"],
        "signature_sha256": identity["tokenizer_signature_sha256"],
        "template_sha256": identity["chat_template_sha256"],
        "eos_token_id": identity["eos_token_id"],
    }
    if any(tokenizer.get(key) != value for key, value in expected_tokenizer.items()):
        raise ValueError("D0.4.2 tokenizer identity differs from the A4-frozen Tokenizer")
    runtime_config = _json(paths["runtime_inference_config"])
    if (
        runtime_config.get("prompt", {}).get("sha256") != identity["p0_sha256"]
        or runtime_config.get("context", {}).get("max_prompt_tokens") != config["identity"]["max_prompt_tokens"]
        or runtime_config.get("generation", {}).get("max_new_tokens") != config["identity"]["max_new_tokens"]
        or runtime_config.get("generation", {}).get("do_sample") is not False
        or runtime_config.get("tokenizer", {}).get("model_id") != "Qwen/Qwen3-1.7B"
        or runtime_config.get("tokenizer", {}).get("revision") != identity["tokenizer_revision"]
        or runtime_config.get("tokenizer", {}).get("expected_signature_sha256") != identity["tokenizer_signature_sha256"]
        or runtime_config.get("tokenizer", {}).get("expected_template_sha256") != identity["chat_template_sha256"]
        or runtime_config.get("tokenizer", {}).get("eos_token_id") != identity["eos_token_id"]
    ):
        raise ValueError("current inference config differs from the frozen P0/F2/T1/Greedy protocol")
    if d043.get("d03_manifest_sha256") != identity["d03_manifest_sha256"]:
        raise ValueError("D0.4.3 Manifest does not bind the frozen D0.3 Manifest")
    if d043.get("d042_manifest_sha256") != identity["d042_manifest_sha256"]:
        raise ValueError("D0.4.3 Manifest does not bind the frozen D0.4.2 Manifest")
    if d043.get("p0_sha256") != identity["p0_sha256"]:
        raise ValueError("D0.4.3 Manifest does not bind the frozen P0")
    sft_output = d043.get("outputs", {}).get("files", {}).get("train.jsonl", {})
    excluded_output = d043.get("outputs", {}).get("files", {}).get("excluded_samples.jsonl", {})
    if (
        sft_output.get("sha256") != identity["sft_train_sha256"]
        or excluded_output.get("sha256") != identity["excluded_ledger_sha256"]
    ):
        raise ValueError("D0.4.3 outputs do not bind the frozen SFT Train and exclusion ledger")
    if (
        e01.get("status") != "passed"
        or e01.get("experiment_id") != "E01-B"
        or e01.get("training_scope") != "full_train"
        or e01.get("base_weights_unchanged") is not True
        or e01.get("train_sha256") != identity["sft_train_sha256"]
        or e01.get("base_weight_sha256") != identity["base_weight_sha256"]
        or e01.get("adapter_identity", {}).get("weight_sha256") != identity["initial_adapter_weight_identity_sha256"]
        or e01.get("adapter_identity", {}).get("config_sha256") != identity["initial_adapter_config_sha256"]
    ):
        raise ValueError("E01-B Full Manifest does not match the A5 frozen model baseline")
    return hashes


def audit_exclusion_ledger(
    path: Path, joined_ids: set[str], *, expected_train_exclusions: int,
) -> dict[str, Any]:
    train_rows: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    total_rows = 0
    for line_number, row, _record_sha in iter_jsonl(path):
        total_rows += 1
        split = row.get("split")
        outcome = row.get("reason")
        sample_id = row.get("sample_id")
        if not isinstance(split, str) or not isinstance(outcome, str) or not isinstance(sample_id, str):
            raise ValueError(f"{path}:{line_number}: exclusion ledger row is malformed")
        split_counts[split] += 1
        outcome_counts[outcome] += 1
        if split == "train":
            if sample_id in train_rows:
                raise ValueError(f"duplicate Train exclusion sample_id: {sample_id}")
            train_rows[sample_id] = outcome
    if set(train_rows) & joined_ids:
        raise ValueError("D0.4.3 marks a sample both usable and excluded")
    if len(train_rows) != expected_train_exclusions:
        raise ValueError("D0.4.3 Train exclusion count differs from the pinned 120-row ledger")
    if any(outcome != "completion_overflows_train_budget" for outcome in train_rows.values()):
        raise ValueError("unexpected D0.4.3 Train exclusion reason; A5 will not regenerate excluded rows")
    return {
        "ledger_rows_total": total_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "train_excluded_count": len(train_rows),
        "train_exclusion_reasons": dict(sorted(Counter(train_rows.values()).items())),
        "all_train_exclusions_are_completion_budget_overflow": True,
        "train_excluded_sample_ids": sorted(train_rows),
    }


def _split_support_named(split: Mapping[str, Any], label_order: list[str]) -> dict[str, Any]:
    support = {}
    for split_name, details in split["support"].items():
        positive_counts = details["positive_counts"]
        support[split_name] = {
            "sample_count": details["sample_count"],
            "conversation_count": details["conversation_count"],
            "labels": [
                {
                    "label": label_order[index],
                    "positive_count": count,
                    "status": "supported" if count else "insufficient_support",
                }
                for index, count in enumerate(positive_counts)
            ],
        }
    return support


def run_audit(config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != "E03-A5-v1":
        raise ValueError("A5 requires the versioned E03-A5-v1 config")
    if config.get("experiment_id") != "E03-A5-BCE-AUX" or config.get("authorization", {}).get("execution_enabled") is not False:
        raise ValueError("A5 CPU audit config identity or GPU authorization gate is invalid")
    if config.get("target", {}).get("target_policy") != TARGET_POLICY:
        raise ValueError("A5 must explicitly select closed_world_gold_v1")
    if config.get("target", {}).get("label_order_version") != LABEL_ORDER_VERSION:
        raise ValueError("A5 label order version differs from the implementation")
    source_hashes = verify_locked_sources(config)
    data = config["data"]
    official_train_schema = _json(_resolve(data["official_train_schema"]))
    official_output_schema = _json(_resolve(data["official_output_schema"]))
    label_order, enums = schema_label_order(official_train_schema, official_output_schema)
    rows, audit = build_supervision(
        _resolve(data["normalized_train"]),
        _resolve(data["sft_train"]),
        label_order,
        enums,
        max_sequence_length=int(data["max_sequence_length"]),
    )
    join = audit["join"]
    if (
        join["normalized_train_rows"] != data["expected_d03_train_rows"]
        or join["sft_train_rows"] != data["expected_sft_train_rows"]
        or join["joined_rows"] != data["expected_sft_train_rows"]
        or join["excluded_normalized_rows"] != data["expected_sft_excluded_train_rows"]
        or not join["join_exact"]
    ):
        raise ValueError("A5.0 Train row counts or sample_id join differ from the frozen D0.4.3 contract")
    exclusion = audit_exclusion_ledger(
        _resolve(data["excluded_ledger"]),
        {row["sample_id"] for row in rows},
        expected_train_exclusions=int(data["expected_sft_excluded_train_rows"]),
    )
    if set(exclusion["train_excluded_sample_ids"]) != set(join["excluded_sample_ids"]):
        raise ValueError("D0.4.3 exclusion IDs do not equal normalized Train minus usable SFT Train")
    if any(sum(row["mask"]) != 33 for row in rows):
        raise ValueError("closed-world A5 supervision requires all 33 known mask positions")
    if not any(any(row["y"]) for row in rows) or not any(not any(row["y"]) for row in rows):
        raise ValueError("Train supervision must include both positive and all-empty instances")

    split_config = config["split"]
    split = freeze_conversation_splits(
        rows,
        seed=int(split_config["seed"]),
        fit_fraction=float(split_config["fit_fraction"]),
        calibration_fraction=float(split_config["calibration_fraction"]),
    )
    if split["version"] != split_config.get("version"):
        raise ValueError("conversation split implementation differs from the frozen config version")
    named_support = _split_support_named(split, label_order)
    split["support_by_label"] = named_support
    output_dir = _resolve(config["outputs"]["audit_root"]) / run_id
    if not output_dir.is_relative_to((_resolve("outputs")).resolve()):
        raise ValueError("A5 audit outputs must remain under ignored outputs/")
    output_dir.mkdir(parents=True, exist_ok=False)
    supervision_path = output_dir / "supervision.jsonl"
    split_path = output_dir / "conversation_splits.json"
    _write_jsonl(supervision_path, rows)
    _write_json(split_path, split)
    data_hashes = {
        "supervision_jsonl": sha256_file(supervision_path),
        "conversation_splits_json": sha256_file(split_path),
    }
    report = {
        "status": "supervision_audit_passed",
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "output_dir": str(output_dir),
        "target_policy": TARGET_POLICY,
        "label_order_version": LABEL_ORDER_VERSION,
        "source_hashes": source_hashes,
        "runtime_protocol_crosscheck": {
            "current_inference_config_sha256": source_hashes["runtime_inference_config"],
            "d042_historical_inference_config_sha256": _json(_resolve(data["d042_manifest"]))
            .get("input_hashes_unchanged_after_run", {}).get("reference_inference_config"),
            "protocol_fields_match_a4": True,
        },
        "source_files": {key: config["data"][key] for key in (
            "normalized_train", "sft_train", "d03_manifest", "d042_manifest",
            "d043_manifest", "excluded_ledger", "official_train_schema",
            "official_output_schema", "prompt", "e01_b_manifest",
            "runtime_inference_config",
        )},
        "label_order": label_order,
        "join": join,
        "exclusion_audit": exclusion,
        "target": audit["target"],
        "annotation_evolution": audit["annotation_evolution"],
        "conversation_split": {
            key: value for key, value in split.items()
            if key not in {"splits", "support", "support_by_label"}
        },
        "partition_counts": {
            name: {
                "sample_count": details["sample_count"],
                "conversation_count": details["conversation_count"],
                "sample_ids_sha256": split["splits"][name]["sample_ids_sha256"],
                "conversation_ids_sha256": split["splits"][name]["conversation_ids_sha256"],
            }
            for name, details in split["support"].items()
        },
        "partition_label_support": named_support,
        "artifacts": {"supervision_jsonl": data_hashes["supervision_jsonl"], "conversation_splits_json": data_hashes["conversation_splits_json"]},
        "gpu_probe_status": "not_run",
        "joint_training_status": "not_run",
        "candidate_generation_status": "not_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "audit_report.json"
    _write_json(report_path, report)
    manifest = {
        "status": report["status"],
        "run_id": run_id,
        "config_path": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_hashes": source_hashes,
        "outputs": {
            "supervision.jsonl": {"sha256": data_hashes["supervision_jsonl"], "rows": len(rows)},
            "conversation_splits.json": {"sha256": data_hashes["conversation_splits_json"]},
            "audit_report.json": {"sha256": sha256_file(report_path)},
        },
        "created_at_utc": report["created_at_utc"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/e03_a5.yaml")
    parser.add_argument("--run-id", required=True, help="Immutable output run ID; existing paths are rejected")
    args = parser.parse_args()
    try:
        report = run_audit(_resolve(args.config), args.run_id)
    except Exception as exc:
        print(json.dumps({"status": "supervision_audit_failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": report["status"],
        "run_id": report["run_id"],
        "joined_rows": report["join"]["joined_rows"],
        "excluded_rows": report["join"]["excluded_normalized_rows"],
        "conversation_count": report["target"]["profile_stats"]["conversation_count"],
        "all_empty_rows": report["target"]["profile_stats"]["all_three_dimensions_empty_count"],
        "output_dir": report["output_dir"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
