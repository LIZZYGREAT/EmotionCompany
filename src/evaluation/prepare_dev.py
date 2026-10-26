"""Build target-free E00 inputs and physically separate reference files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.evaluation import (
    jsonl_bytes,
    load_d03_target_validator,
    load_submission_schema,
    read_jsonl,
    schema_enums,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_NAMES = (
    "inference_input.jsonl",
    "references.jsonl",
    "id_map.jsonl",
    "smoke_input.jsonl",
    "smoke_references.jsonl",
    "smoke_id_map.jsonl",
    "manifest.json",
)


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else base / value).resolve()


def _schema_target_errors(target: Any, enums: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    if not isinstance(target, dict) or set(target) != {
        "emotion_label", "user_profile", "response_text", "memory_refs"
    }:
        return ["target_fields"]
    if target["emotion_label"] not in enums["emotion_label"]:
        errors.append("emotion_label")
    profile = target["user_profile"]
    if not isinstance(profile, dict) or set(profile) != {"personality_traits", "interests", "style"}:
        errors.append("user_profile_fields")
    else:
        for dimension in ("personality_traits", "interests", "style"):
            values = profile[dimension]
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
                or not set(values) <= set(enums[dimension])
            ):
                errors.append(f"user_profile.{dimension}")
    if not isinstance(target["response_text"], str):
        errors.append("response_text")
    memory_refs = target["memory_refs"]
    if (
        not isinstance(memory_refs, list)
        or any(not isinstance(value, str) for value in memory_refs)
        or len(memory_refs) != len(set(memory_refs))
    ):
        errors.append("memory_refs")
    return errors


def validate_dev_rows(
    rows: list[dict[str, Any]],
    expected_count: int,
    enums: dict[str, list[str]],
    target_validator: Any | None = None,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"D0.3 Dev 数量不符: expected={expected_count}, actual={len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Dev 第 {index} 行 sample_id 缺失或重复: {sample_id!r}")
        seen.add(sample_id)
        if set(row) != {"sample_id", "conversation_id", "source", "history", "target"}:
            raise ValueError(f"Dev 第 {index} 行字段不符合 D0.3 契约: {sample_id}")
        source = row["source"]
        if not isinstance(source, dict) or source.get("split") != "train":
            raise ValueError(f"Dev 第 {index} 行 source.split 必须为原始 train: {sample_id}")
        if not isinstance(row["conversation_id"], str) or not row["conversation_id"]:
            raise ValueError(f"Dev 第 {index} 行 conversation_id 无效: {sample_id}")
        history = row["history"]
        if not isinstance(history, list) or not history:
            raise ValueError(f"Dev 第 {index} 行 history 必须非空: {sample_id}")
        for message in history:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(f"Dev 第 {index} 行 history 消息无效: {sample_id}")
        if history[-1]["role"] != "user":
            raise ValueError(f"Dev 第 {index} 行最后一条 history 必须为 user: {sample_id}")
        target_errors = _schema_target_errors(row["target"], enums)
        if target_validator is not None:
            target_errors.extend(
                f"official_schema:{error.message}"
                for error in target_validator.iter_errors(row["target"])
            )
        if target_errors:
            raise ValueError(f"Dev 第 {index} 行 target 不符合 D0.3 Schema: {sample_id}: {target_errors}")


def _find_manifest_output(d03_manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    matches = [item for item in d03_manifest.get("outputs", []) if item.get("path") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"D0.3 Manifest 必须且只能包含输出 {relative_path}")
    return matches[0]


def validate_d03_source(
    dev_path: Path,
    d03_manifest_path: Path,
    d03_manifest: dict[str, Any],
    expected_count: int,
) -> tuple[str, dict[str, Any]]:
    output = _find_manifest_output(d03_manifest, "data/processed/v1/dev.jsonl")
    actual_sha = sha256_file(dev_path)
    if actual_sha != output.get("sha256"):
        raise ValueError(
            f"D0.3 Dev SHA-256 与 Manifest 不符: expected={output.get('sha256')}, actual={actual_sha}"
        )
    if output.get("records") != expected_count:
        raise ValueError(
            f"D0.3 Manifest Dev 数量不符: expected={expected_count}, actual={output.get('records')}"
        )
    if d03_manifest.get("pipeline_version") != "D0.3-v1":
        raise ValueError(f"不支持的 D0.3 pipeline_version: {d03_manifest.get('pipeline_version')!r}")
    manifest_facts = {
        "path": str(d03_manifest_path.relative_to(ROOT)) if d03_manifest_path.is_relative_to(ROOT) else str(d03_manifest_path),
        "sha256": sha256_file(d03_manifest_path),
        "pipeline_version": d03_manifest["pipeline_version"],
        "dev_sha256": actual_sha,
        "dev_rows": output["records"],
    }
    return actual_sha, manifest_facts


def _sample_features(row: dict[str, Any], emotion: str) -> set[str]:
    history = row["history"]
    target = row["target"]
    profile = target["user_profile"]
    features = {f"emotion:{emotion}"}
    is_empty = all(not profile[dimension] for dimension in ("personality_traits", "interests", "style"))
    features.add("profile:empty" if is_empty else "profile:nonempty")
    history_chars = sum(len(message["content"]) for message in history)
    features.add("history:short" if history_chars < 500 else "history:long")
    all_text = "\n".join(message["content"] for message in history)
    if any(ord(char) > 127 for char in all_text):
        features.add("text:unicode")
    if "\n" in all_text or "\t" in all_text:
        features.add("text:whitespace_control")
    if any(marker in all_text for marker in ("</think>", "<think>", "```")):
        features.add("text:special_marker")
    return features


def select_smoke_indexes(rows: list[dict[str, Any]], enums: dict[str, list[str]], seed: int, count: int) -> tuple[list[int], dict[str, Any]]:
    if count <= 0:
        raise ValueError("smoke.count 必须为正整数")
    requested = min(count, len(rows))
    feature_rows = [_sample_features(row, row["target"]["emotion_label"]) for row in rows]
    uncovered = set().union(*feature_rows) if feature_rows else set()
    tie_order = list(range(len(rows)))
    random.Random(seed).shuffle(tie_order)
    rank = {index: position for position, index in enumerate(tie_order)}
    chosen: list[int] = []
    while uncovered and len(chosen) < requested:
        candidates = [index for index in range(len(rows)) if index not in chosen]
        best = max(candidates, key=lambda index: (len(feature_rows[index] & uncovered), -rank[index]))
        if not feature_rows[best] & uncovered:
            break
        chosen.append(best)
        uncovered -= feature_rows[best]
    while len(chosen) < requested:
        candidates = [index for index in range(len(rows)) if index not in chosen]
        chosen.append(min(candidates, key=lambda index: rank[index]))
    chosen_set = set(chosen)
    chosen_in_source_order = [index for index in range(len(rows)) if index in chosen_set]
    covered = set().union(*(feature_rows[index] for index in chosen_in_source_order)) if chosen_in_source_order else set()
    all_features = set().union(*feature_rows) if feature_rows else set()
    return chosen_in_source_order, {
        "seed": seed,
        "requested_count": count,
        "actual_count": len(chosen_in_source_order),
        "selection_method": "greedy_feature_coverage_then_seeded_order_v1",
        "covered_features": sorted(covered),
        "uncovered_available_features": sorted(all_features - covered),
        "available_features": sorted(all_features),
    }


def _commit_id() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_write_bundle(output_dir: Path, payloads: dict[str, bytes], source_sha: str, config_sha: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in ARTIFACT_NAMES if (output_dir / name).exists()]
    manifest_path = output_dir / "manifest.json"
    if existing and not manifest_path.is_file():
        raise FileExistsError(f"输出目录已有文件但没有本工具 Manifest，拒绝覆盖: {output_dir}")
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("source", {}).get("dev_sha256") != source_sha or previous.get("config_sha256") != config_sha:
            raise FileExistsError(f"输出目录属于不同输入/配置，拒绝覆盖: {output_dir}")
    temporary: dict[str, Path] = {}
    try:
        for name, data in payloads.items():
            with tempfile.NamedTemporaryFile("wb", dir=output_dir, prefix=f".{name}.", suffix=".tmp", delete=False) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                temporary[name] = Path(handle.name)
        for name in ARTIFACT_NAMES:
            if name == "manifest.json":
                continue
            if name in temporary:
                os.replace(temporary.pop(name), output_dir / name)
        if "manifest.json" in temporary:
            os.replace(temporary.pop("manifest.json"), manifest_path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def prepare_records(
    rows: list[dict[str, Any]],
    *,
    enums: dict[str, list[str]],
    d03_manifest: dict[str, Any],
    d03_manifest_path: Path,
    source_sha256: str,
    config: dict[str, Any],
    config_sha256: str,
    output_dir: Path,
    target_schema_facts: dict[str, Any] | None = None,
    target_validator: Any | None = None,
) -> dict[str, Any]:
    expected_count = int(config["data"]["expected_dev_count"])
    validate_dev_rows(rows, expected_count, enums, target_validator)
    source = {
        "split": config["data"]["split"],
        "dev_path": str(_resolve(config["data"]["dev_path"]).relative_to(ROOT)),
        "dev_sha256": source_sha256,
        "dev_rows": len(rows),
        "d03_manifest_path": str(d03_manifest_path.relative_to(ROOT)) if d03_manifest_path.is_relative_to(ROOT) else str(d03_manifest_path),
        "d03_manifest_sha256": sha256_file(d03_manifest_path),
        "d03_pipeline_version": d03_manifest.get("pipeline_version"),
        "target_schema": target_schema_facts,
    }
    inference_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    id_map_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        eval_id = f"test_{index:06d}_t0"
        inference_rows.append({"id": eval_id, "history": row["history"]})
        reference_rows.append({"eval_id": eval_id, "sample_id": row["sample_id"], "target": row["target"]})
        id_map_rows.append({"eval_id": eval_id, "sample_id": row["sample_id"], "source": row["source"]})

    smoke_indices, smoke_facts = select_smoke_indexes(
        rows, enums, int(config["smoke"]["seed"]), int(config["smoke"]["count"])
    )
    smoke_facts["eval_ids"] = [inference_rows[index]["id"] for index in smoke_indices]
    smoke_facts["sample_ids"] = [rows[index]["sample_id"] for index in smoke_indices]
    smoke_facts["history_length_basis"] = "sum of Python Unicode code points in message content; 500-character boundary; descriptive proxy, not tokenizer length"
    output_rows = {
        "inference_input.jsonl": inference_rows,
        "references.jsonl": reference_rows,
        "id_map.jsonl": id_map_rows,
        "smoke_input.jsonl": [inference_rows[index] for index in smoke_indices],
        "smoke_references.jsonl": [reference_rows[index] for index in smoke_indices],
        "smoke_id_map.jsonl": [id_map_rows[index] for index in smoke_indices],
    }
    files: dict[str, bytes] = {name: jsonl_bytes(value) for name, value in output_rows.items()}
    output_facts = {
        name: {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(output_rows[name]),
        }
        for name, data in files.items()
    }
    manifest = {
        "manifest_version": "E00.1-dev-adapter-v1",
        "experiment_id": config["experiment_id"],
        "dataset_scope": "D0.3_dev",
        "source": source,
        "config_sha256": config_sha256,
        "protocol": config["protocol"],
        "planned_config": {
            "model": config["model"],
            "quantization": config["quantization"]["planned"],
            "generation": config["generation"],
        },
        "observed_runtime": None,
        "smoke_selection": smoke_facts,
        "outputs": output_facts,
        "git_commit": _commit_id(),
    }
    files["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bundle(output_dir, files, source_sha256, config_sha256)
    return manifest


def prepare_from_paths(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError(f"配置必须是 YAML object: {config_path}")
    dev_path = _resolve(config["data"]["dev_path"])
    manifest_path = _resolve(config["data"]["d03_manifest_path"])
    output_dir = output_override.resolve() if output_override else _resolve(config["data"]["output_dir"])
    if not dev_path.is_file():
        raise FileNotFoundError(f"D0.3 Dev 文件缺失: {dev_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"D0.3 Manifest 缺失: {manifest_path}")
    d03_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha, _ = validate_d03_source(dev_path, manifest_path, d03_manifest, int(config["data"]["expected_dev_count"]))
    target_validator, target_schema_facts = load_d03_target_validator(manifest_path, ROOT)
    rows = read_jsonl(dev_path)
    schema = load_submission_schema(ROOT)
    enums = schema_enums(schema)
    prompt_path = _resolve(config["protocol"]["prompt_path"])
    if not prompt_path.is_file() or sha256_file(prompt_path) != config["protocol"]["prompt_sha256"]:
        raise ValueError("P0 文本缺失或 SHA-256 与冻结协议不符")
    return prepare_records(
        rows,
        enums=enums,
        d03_manifest=d03_manifest,
        d03_manifest_path=manifest_path,
        source_sha256=source_sha,
        config=config,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        output_dir=output_dir,
        target_schema_facts=target_schema_facts,
        target_validator=target_validator,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造无标签泄漏的 E00 Dev 评测输入")
    parser.add_argument("--config", required=True, type=Path, help="E00 YAML 配置")
    parser.add_argument("--output-dir", type=Path, help="可选覆盖输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_from_paths(args.config, args.output_dir)
    print(
        f"PASS: prepared {manifest['source']['dev_rows']} D0.3 Dev rows; "
        f"smoke={manifest['smoke_selection']['actual_count']}; "
        f"output={args.output_dir or yaml.safe_load(args.config.read_text(encoding='utf-8'))['data']['output_dir']}"
    )


if __name__ == "__main__":
    main()
