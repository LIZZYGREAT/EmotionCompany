"""Deterministic D0.3 normalization, point disposition, and split pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from src.data.split import assign_train_dev, build_conversation_groups, canonical_json, fingerprint


PIPELINE_VERSION = "D0.3-v1"
PROFILE_DIMS = ("personality_traits", "interests", "style")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} path escapes project root: {value}")
    return resolved


def _reject_overlapping_paths(inputs: dict[str, Path], outputs: dict[str, Path]) -> None:
    for output_name, output_path in outputs.items():
        for input_name, input_path in inputs.items():
            if output_path == input_path or output_path.is_relative_to(input_path) or input_path.is_relative_to(output_path):
                raise ValueError(f"Output {output_name} overlaps input {input_name}")
    output_items = list(outputs.items())
    for index, (left_name, left_path) in enumerate(output_items):
        for right_name, right_path in output_items[index + 1:]:
            if left_path == right_path or left_path.is_relative_to(right_path) or right_path.is_relative_to(left_path):
                raise ValueError(f"Outputs {left_name} and {right_name} overlap")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _publish_transaction(items: list[tuple[Path, Path]]) -> None:
    """Replace completed outputs as a rollback-capable group of renames."""
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


def _typed_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, canonical_json(value)


def _role_content(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": turn["role"], "content": turn["content"]} for turn in turns]


def _point_ref(split: str, line: int, conversation_id: str, point_index: int) -> dict[str, Any]:
    return {
        "split": split,
        "source_line": line,
        "conversation_id": conversation_id,
        "prediction_point_index": point_index,
    }


def _read_split(path: Path, schema_path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes.decode("utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    rows: list[dict[str, Any]] = []
    conversation_ids: set[str] = set()
    point_count = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{split} line {line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{split} line {line_number}: invalid JSON at column {exc.colno}") from exc
            errors = sorted(
                validator.iter_errors(row),
                key=lambda error: (list(map(str, error.absolute_path)), error.message),
            )
            if errors:
                location = "/".join(map(str, errors[0].absolute_path)) or "$"
                raise ValueError(
                    f"{split} line {line_number}: official schema violation at {location} "
                    f"({errors[0].validator})"
                )
            conversation_id = row["conversation_id"]
            if conversation_id in conversation_ids:
                raise ValueError(f"{split} line {line_number}: duplicate conversation_id {conversation_id}")
            conversation_ids.add(conversation_id)
            point_count += len(row["prediction_points"])
            rows.append({"source_line": line_number, "row": row})
    stats = {
        "path": path,
        "schema_path": schema_path,
        "schema_sha256": _sha256_bytes(schema_bytes),
        "line_count": len(rows),
        "conversation_count": len(conversation_ids),
        "prediction_point_count": point_count,
    }
    return rows, stats


def _normalize_point(
    split: str,
    source_line: int,
    row: dict[str, Any],
    point_index: int,
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, str]] | None]:
    point = row["prediction_points"][point_index]
    turns = row["turns"]
    id_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, turn in enumerate(turns):
        id_positions[_typed_key(turn["turn_id"])].append(index)

    target_positions = id_positions.get(_typed_key(point["target_user_turn_id"]), [])
    response_positions = id_positions.get(_typed_key(point["assistant_response_turn_id"]), [])
    if not target_positions:
        return None, "TARGET_TURN_NOT_FOUND", None
    if len(target_positions) != 1:
        return None, "TARGET_TURN_ID_AMBIGUOUS", None
    target_index = target_positions[0]
    if turns[target_index]["role"] != "user":
        return None, "TARGET_TURN_NOT_USER", None

    history = _role_content(turns[:target_index + 1])
    if not response_positions:
        return None, "RESPONSE_TURN_NOT_FOUND", history
    if len(response_positions) != 1:
        return None, "RESPONSE_TURN_ID_AMBIGUOUS", history
    response_index = response_positions[0]
    if turns[response_index]["role"] != "assistant":
        return None, "RESPONSE_TURN_NOT_ASSISTANT", history
    if response_index <= target_index:
        return None, "RESPONSE_NOT_AFTER_TARGET", history
    if point["target"]["response_text"] != turns[response_index]["content"]:
        raise ValueError(
            f"{split} line {source_line} conversation {row['conversation_id']} "
            f"point {point_index}: response_text differs from referenced assistant turn"
        )

    sample_id = f"{split}:{row['conversation_id']}:pp{point_index}"
    sample = {
        "sample_id": sample_id,
        "conversation_id": row["conversation_id"],
        "source": {
            "split": split,
            "source_line": source_line,
            "prediction_point_index": point_index,
            "target_user_turn_id": point["target_user_turn_id"],
            "assistant_response_turn_id": point["assistant_response_turn_id"],
            "target_turn_index": target_index,
            "response_turn_index": response_index,
        },
        "history": history,
        "target": point["target"],
    }
    return sample, None, history


def _distribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    emotions: Counter[str] = Counter()
    profile_tags = {dim: Counter() for dim in PROFILE_DIMS}
    profile_all_empty = 0
    profile_partly_empty = 0
    memory_nonempty = 0
    memory_empty = 0
    short_response_count = 0
    for sample in samples:
        target = sample["target"]
        emotions[target["emotion_label"]] += 1
        profile = target["user_profile"]
        empty_dimensions = 0
        for dim in PROFILE_DIMS:
            profile_tags[dim].update(profile[dim])
            empty_dimensions += not profile[dim]
        if empty_dimensions == len(PROFILE_DIMS):
            profile_all_empty += 1
        elif empty_dimensions:
            profile_partly_empty += 1
        memory_nonempty += bool(target["memory_refs"])
        memory_empty += not target["memory_refs"]
        short_response_count += len(target["response_text"]) < 5
    return {
        "sample_count": len(samples),
        "emotion_counts": dict(sorted(emotions.items())),
        "profile_tag_counts": {
            dim: dict(sorted(counts.items())) for dim, counts in profile_tags.items()
        },
        "profile_all_empty_count": profile_all_empty,
        "profile_partly_empty_count": profile_partly_empty,
        "memory_refs_nonempty_count": memory_nonempty,
        "memory_refs_empty_count": memory_empty,
        "short_response_under_5_count": short_response_count,
    }


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


def _file_record(stage_path: Path, final_path: Path, root: Path, *, records: int | None = None) -> dict[str, Any]:
    result = {
        "path": str(final_path.relative_to(root)),
        "size_bytes": stage_path.stat().st_size,
        "sha256": _sha256_file(stage_path),
    }
    if records is not None:
        result["records"] = records
    return result


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("wb") as stream:
        for item in items:
            stream.write(_json_bytes(item))


def _validate_jsonl_matches(path: Path, expected: list[dict[str, Any]], split: str) -> None:
    seen_ids: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for count, line in enumerate(stream, start=1):
            try:
                actual = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Staged {split} JSONL is invalid at record {count}") from exc
            if count > len(expected) or actual != expected[count - 1]:
                raise RuntimeError(f"Staged {split} JSONL differs from its normalized source record {count}")
            if set(actual) != {"sample_id", "conversation_id", "source", "history", "target"}:
                raise RuntimeError(f"Staged {split} sample has unexpected top-level fields")
            if not actual["history"] or actual["history"][-1]["role"] != "user":
                raise RuntimeError(f"Staged {split} sample history does not end at its target user turn")
            if any(set(turn) != {"role", "content"} for turn in actual["history"]):
                raise RuntimeError(f"Staged {split} history contains fields beyond role/content")
            expected_source_split = "val" if split == "val" else "train"
            if actual["source"]["split"] != expected_source_split:
                raise RuntimeError(f"Staged {split} sample source split changed")
            sample_id = actual["sample_id"]
            if sample_id in seen_ids:
                raise RuntimeError(f"Staged {split} contains duplicate sample_id {sample_id}")
            seen_ids.add(sample_id)
    if count != len(expected):
        raise RuntimeError(f"Staged {split} record count {count} differs from expected {len(expected)}")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value, pretty=True))


def _render_report(
    summary: dict[str, Any],
    split_manifest: dict[str, Any],
    manifest_core: dict[str, Any],
    test_result: str | None,
    full_repeat_result: str | None,
) -> str:
    train = summary["train"]
    val = summary["val"]
    outcomes = train["outcomes"]
    train_points = summary["distributions"]
    labels = sorted(set(train_points["raw_train"]["emotion_counts"]) | set(train_points["unique_train"]["emotion_counts"]))
    label_rows = [
        f"| `{label}` | {train_points['raw_train']['emotion_counts'].get(label, 0)} | "
        f"{train_points['unique_train']['emotion_counts'].get(label, 0)} | "
        f"{train_points['train']['emotion_counts'].get(label, 0)} | "
        f"{train_points['dev']['emotion_counts'].get(label, 0)} | "
        f"{train_points['val']['emotion_counts'].get(label, 0)} |"
        for label in labels
    ]
    dimension_rows = []
    for dim in PROFILE_DIMS:
        counts = train_points["raw_train"]["profile_tag_counts"][dim]
        after = train_points["unique_train"]["profile_tag_counts"][dim]
        tags = sorted(set(counts) | set(after))
        dimension_rows.append(
            f"| `{dim}` | {sum(counts.values())} | {sum(after.values())} | "
            f"{sum(train_points['val']['profile_tag_counts'][dim].values())} | {len(tags)} 个标签 |"
        )
    hash_lines = [
        f"| {split} | `{item['path']}` | {item['line_count']} | {item['prediction_point_count']} | "
        f"`{item['sha256_before']}` | `{item['schema_sha256']}` |"
        for split, item in summary["inputs"].items()
    ]
    tests = summary.get("verification", {})
    test_text = test_result if test_result else "未随本次构建提供测试运行摘要。"
    return "\n".join([
        "# D0.3 数据构建报告",
        "",
        f"> 管线版本：`{PIPELINE_VERSION}`　|　政策版本：`{summary['policy_version']}`　|　配置 SHA-256：`{manifest_core['config_sha256']}`",
        "",
        f"实现 Git commit：`{manifest_core['git']['commit'] or 'unavailable'}`；tracked dirty：`{manifest_core['git']['tracked_files_dirty']}`。代码 SHA-256：`prepare.py` `{manifest_core['source_code_sha256']['src/data/prepare.py']}`，`split.py` `{manifest_core['source_code_sha256']['src/data/split.py']}`。",
        "",
        "## 1. 输入版本与完整性",
        "",
        "| Split | 输入 | 行/对话数 | 原始预测点 | 输入 SHA-256 | Schema SHA-256 |",
        "|---|---|---:|---:|---|---|",
        *hash_lines,
        "",
        f"处理前后输入 SHA-256 一致：**{str(summary['input_sha256_unchanged']).lower()}**。Schema 严格校验启用；本次无 Schema 错误。原始输入只读。",
        "",
        "## 2. 预测点数量守恒",
        "",
        "| Split / 阶段 | 数量 | 说明 |",
        "|---|---:|---|",
        f"| Train 原始 | {train['raw_prediction_points']} | 8,338 个原始对话中的预测点 |",
        f"| Train 隔离 | {outcomes['quarantined']} | 确定性引用错误；仅隔离单点 |",
        f"| Train 去重移除 | {outcomes['deduplicated']} | 合法历史与完整 Target 精确相同的后续副本 |",
        f"| Train + Dev 唯一保留 | {outcomes['retained']} | 原始 - 隔离 - 去重 |",
        f"| 最终 Train | {outcomes['train']} | 按关联组分配 |",
        f"| 最终 Dev | {outcomes['dev']} | 按关联组分配 |",
        f"| Val 原始 / 最终 | {val['raw_prediction_points']} / {val['final_prediction_points']} | Val 不去重、不参与划分 |",
        "",
        f"守恒：`{train['raw_prediction_points']} = {outcomes['quarantined']} + {outcomes['deduplicated']} + {outcomes['train']} + {outcomes['dev']}`；Val `{val['raw_prediction_points']} = {val['final_prediction_points']}`。",
        "",
        "## 3. 去重与 Train/Dev 关联组",
        "",
        f"Train 完整 turns（按 role/content 投影）的重复组：{summary['duplicate_stats']['full_turn_duplicate_groups']} 组、{summary['duplicate_stats']['full_turn_duplicate_extra_conversations']} 条额外对话记录。预测点精确监督组：{summary['duplicate_stats']['exact_supervision_duplicate_groups']} 组，实际移除 {outcomes['deduplicated']} 点。二者是不同口径，不相加作为删除数。",
        "",
        f"分组规则：同一完整 turns role/content 或同一合法历史 role/content 的记录进入同一连通分量；不做模糊/语义匹配。共 {split_manifest['group_count']} 组，最大 {split_manifest['max_group_conversation_count']} 个对话、{split_manifest['max_group_sample_count']} 个唯一监督点；超过配置阈值的大组 {summary['group_stats']['large_group_count']} 组。seed={split_manifest['seed']}，Dev 对话比例目标 {split_manifest['dev_ratio_target']:.0%}。Dev 有 {split_manifest['dev_group_count']} 组、{split_manifest['actual_dev_conversation_count']} 个对话（{split_manifest['actual_dev_conversation_fraction']:.2%}）及 {split_manifest['actual_dev_sample_count']} 个点（{split_manifest['actual_dev_sample_fraction']:.2%}）。",
        "",
        f"相同合法历史对应不同完整 Target 的组：{summary['duplicate_stats']['same_history_different_target_groups']}；其中情绪/画像标注冲突组：{summary['duplicate_stats']['same_history_label_conflict_groups']}。若出现则保留监督并写入复核日志，不自动选取。",
        "",
        f"分组间 Train/Dev 相同对话、完整 turns 或合法历史重叠：{summary['leakage_checks']['conversation_id_overlap']} / {summary['leakage_checks']['full_turn_overlap']} / {summary['leakage_checks']['legal_history_overlap']}。",
        "",
        "## 4. 情绪、画像与保留内容",
        "",
        "| 情绪 | Train 原始 | Train/Dev 去重后合计 | Train | Dev | Val |",
        "|---|---:|---:|---:|---:|---:|",
        *label_rows,
        "",
        "| 画像维度 | Train 原始标签赋值数 | Train/Dev 去重后标签赋值数 | Val 标签赋值数 | Train 标签种类 |",
        "|---|---:|---:|---:|---:|",
        *dimension_rows,
        "",
        f"Train 全空画像：原始 {train_points['raw_train']['profile_all_empty_count']}，去重后 {train_points['unique_train']['profile_all_empty_count']}；Val 全空画像 {train_points['val']['profile_all_empty_count']}。非空 memory_refs：Train 原始 {train_points['raw_train']['memory_refs_nonempty_count']}、去重后 {train_points['unique_train']['memory_refs_nonempty_count']}；Val {train_points['val']['memory_refs_nonempty_count']}。短回复（<5 Unicode 字符）：Train 原始 {train_points['raw_train']['short_response_under_5_count']}、去重后保留 {train_points['unique_train']['short_response_under_5_count']}；Val {train_points['val']['short_response_under_5_count']}。空画像、短回复和非空记忆均不做填补或过滤。",
        "",
        "## 5. 自动验证与复现",
        "",
        f"测试命令：`{tests.get('unit_test_command', 'python -m pytest tests/test_data_audit.py tests/test_data_pipeline.py -q')}`。",
        f"本次测试摘要：{test_text}",
        "",
        f"有效原始 Train 点恰好落入隔离、去重、Train 或 Dev 之一：**{str(summary['conservation_valid']).lower()}**。Val 样本数符合基线：**{str(summary['val_count_valid']).lower()}**。输出生成前校验输入指纹：**{str(summary['input_sha256_unchanged']).lower()}**。",
        "T12 自动用例在合成 JSONL 上逐字节比较 Train/Dev/Val JSONL、清洗日志和 split manifest。",
        f"官方全量数据二次构建检查：{full_repeat_result or '未提供重复构建结果'}",
        "具体输出指纹见 `outputs/data_pipeline/v1/manifest.json`。",
        "",
        "## 6. 风险与 D0.4 交接",
        "",
        "未实施 Tokenizer、Chat Template、token 截断、system prompt、SFT mask 或训练。D0.4 应以本阶段 `history`（含目标 user 轮）和 `target.response_text` 为输入，固定 tokenizer/model revision 与模板后再统计 token 长度和截断风险；不得把本报告中的 Unicode 字符长度当成 token 数。D0.1 合法历史字符长度 Train P99/最大为 9,705/40,003，仅提示后续需评估上下文预算，不代表 Token 统计。",
        "",
        "样本正文仅存于本地被忽略的数据产物；Git 报告只包含数量、指纹和规则，不复制私人对话。",
        "",
    ])


def run_pipeline(
    root: Path,
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    test_result: str | None = None,
    full_repeat_result: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config = copy.deepcopy(config)
    input_paths: dict[str, Path] = {}
    schema_paths: dict[str, Path] = {}
    input_configs = config.get("inputs", {})
    for split in ("train", "val"):
        split_config = input_configs.get(split, {})
        input_paths[split] = _safe_path(root, split_config.get("path", ""), f"inputs.{split}")
        schema_paths[split] = _safe_path(root, split_config.get("schema", ""), f"inputs.{split}.schema")
        if not input_paths[split].is_file():
            raise FileNotFoundError(f"Input file not found: {input_paths[split].relative_to(root)}")
        if not schema_paths[split].is_file():
            raise FileNotFoundError(f"Schema file not found: {schema_paths[split].relative_to(root)}")

    output_config = config.get("outputs", {})
    output_paths = {
        "normalized": _safe_path(root, output_config.get("normalized_dir", ""), "outputs.normalized_dir"),
        "artifacts": _safe_path(root, output_config.get("artifacts_dir", ""), "outputs.artifacts_dir"),
        "report": _safe_path(root, output_config.get("report", ""), "outputs.report"),
    }
    _reject_overlapping_paths(input_paths, output_paths)

    input_before: dict[str, str] = {}
    schema_before: dict[str, str] = {}
    for split in ("train", "val"):
        input_before[split] = _sha256_file(input_paths[split])
        schema_before[split] = _sha256_file(schema_paths[split])
        split_config = input_configs[split]
        expected = split_config.get("expected_sha256")
        if expected and input_before[split] != expected:
            raise ValueError(f"{split} input SHA-256 differs from the D0.1 baseline")
        expected_schema = split_config.get("expected_schema_sha256")
        if expected_schema and schema_before[split] != expected_schema:
            raise ValueError(f"{split} schema SHA-256 differs from the D0.1 baseline")

    raw: dict[str, list[dict[str, Any]]] = {}
    input_stats: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        rows, stats = _read_split(input_paths[split], schema_paths[split], split)
        raw[split] = rows
        input_stats[split] = stats
        split_config = input_configs[split]
        for key, observed_key in (
            ("expected_line_count", "line_count"),
            ("expected_conversation_count", "conversation_count"),
            ("expected_prediction_points", "prediction_point_count"),
        ):
            expected = split_config.get(key)
            if expected is not None and stats[observed_key] != expected:
                raise ValueError(
                    f"{split} {observed_key} {stats[observed_key]} differs from configured baseline {expected}"
                )

    source_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    candidates: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    quarantine: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    review_log: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for entry in raw[split]:
            source_line, row = entry["source_line"], entry["row"]
            legal_histories: list[list[dict[str, str]]] = []
            target_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
            for index, turn in enumerate(row["turns"]):
                target_positions[_typed_key(turn["turn_id"])].append(index)
            for point_index, point in enumerate(row["prediction_points"]):
                positions = target_positions.get(_typed_key(point["target_user_turn_id"]), [])
                if len(positions) == 1 and row["turns"][positions[0]]["role"] == "user":
                    legal_histories.append(_role_content(row["turns"][:positions[0] + 1]))
                sample, reason, history = _normalize_point(split, source_line, row, point_index)
                if reason is not None:
                    ref = _point_ref(split, source_line, row["conversation_id"], point_index)
                    item = {
                        **ref,
                        "sample_id": f"{split}:{row['conversation_id']}:pp{point_index}",
                        "action": "quarantined",
                        "reason": reason,
                    }
                    if split == "val":
                        raise ValueError(
                            f"Val point failed deterministic reference validation at line {source_line}, "
                            f"conversation {row['conversation_id']}, point {point_index}: {reason}"
                        )
                    quarantine[split].append(item)
                    continue
                assert sample is not None and history is not None
                candidate = {
                    "sample": sample,
                    "history": history,
                    "history_fingerprint": fingerprint(history),
                    "canonical_target": canonical_json(sample["target"]),
                    "split": split,
                }
                candidates[split].append(candidate)
                if len(sample["target"]["response_text"]) < int(config.get("policy", {}).get("short_response_chars", 5)):
                    review_log.append({
                        **_point_ref(split, source_line, row["conversation_id"], point_index),
                        "sample_id": sample["sample_id"],
                        "action": "review_retained",
                        "reason": "SHORT_RESPONSE_CANDIDATE",
                        "kept_as": sample["sample_id"],
                    })
            source_records[split].append({
                "conversation_id": row["conversation_id"],
                "turns": row["turns"],
                "legal_histories": legal_histories,
            })

    train_full_turns = {
        fingerprint(_role_content(entry["row"]["turns"])) for entry in raw["train"]
    }
    val_full_turns = {
        fingerprint(_role_content(entry["row"]["turns"])) for entry in raw["val"]
    }
    train_histories = {
        fingerprint(history)
        for record in source_records["train"] for history in record["legal_histories"]
    }
    val_histories = {
        fingerprint(history)
        for record in source_records["val"] for history in record["legal_histories"]
    }
    train_ids = {record["conversation_id"] for record in source_records["train"]}
    val_ids = {record["conversation_id"] for record in source_records["val"]}
    leakage = {
        "conversation_id_overlap": len(train_ids & val_ids),
        "full_turn_overlap": len(train_full_turns & val_full_turns),
        "legal_history_overlap": len(train_histories & val_histories),
    }
    if any(leakage.values()):
        raise ValueError(f"Unexpected Train/Val overlap; automatic build stopped: {leakage}")

    # Val is normalized but never deduplicated or assigned into Train/Dev.
    val_samples = [candidate["sample"] for candidate in candidates["val"]]
    if len(val_samples) != input_stats["val"]["prediction_point_count"]:
        raise ValueError("Val point count changed during normalization")

    # Keep the first exact history + complete Target in source order.
    dedupe_seen: dict[str, tuple[str, dict[str, Any]]] = {}
    train_unique: list[dict[str, Any]] = []
    dedup_log: list[dict[str, Any]] = []
    duplicate_group_members: Counter[str] = Counter()
    for candidate in candidates["train"]:
        key_payload = {"history": candidate["history"], "target": candidate["sample"]["target"]}
        key_fingerprint = fingerprint(key_payload)
        key_canonical = canonical_json(key_payload)
        existing = dedupe_seen.get(key_fingerprint)
        if existing is None:
            dedupe_seen[key_fingerprint] = (key_canonical, candidate)
            train_unique.append(candidate)
            continue
        first_canonical, first_candidate = existing
        if first_canonical != key_canonical:
            raise RuntimeError("SHA-256 collision while comparing exact supervision values")
        duplicate_group_members[key_fingerprint] += 1
        sample = candidate["sample"]
        kept_as = first_candidate["sample"]["sample_id"]
        dedup_log.append({
            **_point_ref(
                "train",
                sample["source"]["source_line"],
                sample["conversation_id"],
                sample["source"]["prediction_point_index"],
            ),
            "sample_id": sample["sample_id"],
            "action": "deduplicated",
            "reason": "EXACT_HISTORY_AND_COMPLETE_TARGET",
            "kept_as": kept_as,
        })

    # Distinct targets for an exact history are preserved and explicitly logged.
    history_targets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for candidate in train_unique:
        history_targets[candidate["history_fingerprint"]][candidate["canonical_target"]].append(candidate)
    same_history_target_groups = [targets for targets in history_targets.values() if len(targets) > 1]
    same_history_label_conflict_groups = 0
    for targets in same_history_target_groups:
        label_signatures = {
            canonical_json({
                "emotion_label": group[0]["sample"]["target"]["emotion_label"],
                "user_profile": group[0]["sample"]["target"]["user_profile"],
            })
            for group in targets.values()
        }
        same_history_label_conflict_groups += len(label_signatures) > 1
        for target_group in targets.values():
            candidate = target_group[0]
            sample = candidate["sample"]
            review_log.append({
                **_point_ref(
                    "train",
                    sample["source"]["source_line"],
                    sample["conversation_id"],
                    sample["source"]["prediction_point_index"],
                ),
                "sample_id": sample["sample_id"],
                "action": "review_retained",
                "reason": "SAME_HISTORY_DIFFERENT_TARGET",
                "kept_as": sample["sample_id"],
            })

    # Build conversation components using every legal target history, before split assignment.
    groups = build_conversation_groups(source_records["train"])
    conversation_sample_counts: Counter[str] = Counter()
    conversation_emotion_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in train_unique:
        sample = candidate["sample"]
        cid = sample["conversation_id"]
        conversation_sample_counts[cid] += 1
        conversation_emotion_counts[cid][sample["target"]["emotion_label"]] += 1

    split_config = config.get("split", {})
    seed = int(split_config.get("seed", 42))
    dev_ratio = float(split_config.get("dev_ratio", 0.10))
    preserve_emotions = list(split_config.get("preserve_emotions", []))
    conversation_to_split, split_manifest = assign_train_dev(
        groups,
        dict(conversation_sample_counts),
        conversation_emotion_counts,
        dev_ratio=dev_ratio,
        seed=seed,
        preserve_emotions=preserve_emotions,
    )

    large_group_fraction = float(split_config.get("large_group_fraction", 0.01))
    large_threshold = max(1, int(len(source_records["train"]) * large_group_fraction))
    split_manifest["large_group_threshold_conversations"] = large_threshold
    split_manifest["large_groups"] = [
        group["group_id"] for group in groups if group["conversation_count"] >= large_threshold
    ]
    split_deviation = abs(split_manifest["actual_dev_conversation_fraction"] - dev_ratio)
    max_dev_deviation = float(split_config.get("max_dev_ratio_deviation", 0.02))
    split_manifest["max_allowed_dev_ratio_deviation"] = max_dev_deviation
    split_manifest["dev_ratio_deviation"] = split_deviation
    if split_deviation > max_dev_deviation:
        raise ValueError(
            "Connected groups prevent the configured Dev ratio: "
            f"target={dev_ratio:.4f}, actual={split_manifest['actual_dev_conversation_fraction']:.4f}, "
            f"max_group={split_manifest['max_group_conversation_count']}"
        )

    train_samples: list[dict[str, Any]] = []
    dev_samples: list[dict[str, Any]] = []
    for candidate in train_unique:
        split = conversation_to_split[candidate["sample"]["conversation_id"]]
        (dev_samples if split == "dev" else train_samples).append(candidate["sample"])
    # Source-order list construction above is stable; no filesystem/hash iteration determines output order.

    # Verify that the component assignment actually keeps all projected inputs together.
    train_dev_full_overlap: set[str] = set()
    # Full-turn and history fingerprints are separated for clearer validation counts.
    turn_fps_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    history_fps_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    for record in source_records["train"]:
        split = conversation_to_split[record["conversation_id"]]
        turn_fps_by_split[split].add(fingerprint(_role_content(record["turns"])))
        history_fps_by_split[split].update(fingerprint(history) for history in record["legal_histories"])
    train_dev_full_overlap = turn_fps_by_split["train"] & turn_fps_by_split["dev"]
    train_dev_history_overlap = history_fps_by_split["train"] & history_fps_by_split["dev"]
    train_conversation_ids = {cid for cid, split in conversation_to_split.items() if split == "train"}
    dev_conversation_ids = {cid for cid, split in conversation_to_split.items() if split == "dev"}
    train_dev_conversation_overlap = train_conversation_ids & dev_conversation_ids
    if train_dev_full_overlap or train_dev_history_overlap or train_dev_conversation_overlap:
        raise RuntimeError("Train/Dev leakage detected after grouped assignment")
    leakage.update({
        "train_dev_conversation_id_overlap": len(train_dev_conversation_overlap),
        "train_dev_full_turn_overlap": len(train_dev_full_overlap),
        "train_dev_legal_history_overlap": len(train_dev_history_overlap),
    })

    raw_train_samples = [
        {"target": point["target"]}
        for entry in raw["train"] for point in entry["row"]["prediction_points"]
    ]
    distributions = {
        "raw_train": _distribution(raw_train_samples),
        "unique_train": _distribution([candidate["sample"] for candidate in train_unique]),
        "train": _distribution(train_samples),
        "dev": _distribution(dev_samples),
        "val": _distribution(val_samples),
    }
    quarantine_count = len(quarantine["train"])
    deduplicated_count = len(dedup_log)
    retained_count = len(train_unique)
    raw_train_point_count = input_stats["train"]["prediction_point_count"]
    raw_val_point_count = input_stats["val"]["prediction_point_count"]
    outcomes_valid = raw_train_point_count == quarantine_count + deduplicated_count + len(train_samples) + len(dev_samples)
    if not outcomes_valid:
        raise RuntimeError("Train prediction-point conservation failed")
    val_expected = input_configs["val"].get("expected_prediction_points")
    val_count_valid = val_expected is None or len(val_samples) == val_expected
    if not val_count_valid:
        raise RuntimeError("Val sample count differs from the configured official baseline")

    # A second outcome check keyed by every original point proves there are no omissions or double dispositions.
    outcome_by_sample: dict[str, str] = {}
    for item in quarantine["train"]:
        outcome_by_sample[item["sample_id"]] = "quarantined"
    for item in dedup_log:
        outcome_by_sample[item["sample_id"]] = "deduplicated"
    for sample in train_samples:
        outcome_by_sample[sample["sample_id"]] = "train"
    for sample in dev_samples:
        outcome_by_sample[sample["sample_id"]] = "dev"
    if len(outcome_by_sample) != raw_train_point_count:
        raise RuntimeError(
            f"Train outcome ledger has {len(outcome_by_sample)} unique points; expected {raw_train_point_count}"
        )

    before_after_hashes = {
        split: _sha256_file(input_paths[split]) for split in ("train", "val")
    }
    input_unchanged = before_after_hashes == input_before
    if config.get("validation", {}).get("assert_input_unchanged", True) and not input_unchanged:
        raise RuntimeError("An original input changed during pipeline execution")

    full_turn_ref_counts: dict[str, set[str]] = defaultdict(set)
    for record in source_records["train"]:
        full_turn_ref_counts[fingerprint(_role_content(record["turns"]))].add(record["conversation_id"])
    duplicate_stats = {
        "full_turn_duplicate_groups": sum(len(ids) > 1 for ids in full_turn_ref_counts.values()),
        "full_turn_duplicate_extra_conversations": sum(max(0, len(ids) - 1) for ids in full_turn_ref_counts.values()),
        "exact_supervision_duplicate_groups": len(duplicate_group_members),
        "exact_supervision_duplicate_occurrences": deduplicated_count,
        "same_history_different_target_groups": len(same_history_target_groups),
        "same_history_label_conflict_groups": same_history_label_conflict_groups,
    }

    input_summary = {}
    for split in ("train", "val"):
        stats = input_stats[split]
        input_summary[split] = {
            "path": str(input_paths[split].relative_to(root)),
            "schema_path": str(schema_paths[split].relative_to(root)),
            "line_count": stats["line_count"],
            "conversation_count": stats["conversation_count"],
            "prediction_point_count": stats["prediction_point_count"],
            "size_bytes": input_paths[split].stat().st_size,
            "sha256_before": input_before[split],
            "sha256_after": before_after_hashes[split],
            "schema_sha256": stats["schema_sha256"],
        }
    summary: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "policy_version": config.get("policy", {}).get("version", "D0.2-v1"),
        "inputs": input_summary,
        "train": {
            "raw_conversations": input_stats["train"]["conversation_count"],
            "raw_prediction_points": raw_train_point_count,
            "empty_prediction_point_conversations": sum(
                not entry["row"]["prediction_points"] for entry in raw["train"]
            ),
            "outcomes": {
                "quarantined": quarantine_count,
                "deduplicated": deduplicated_count,
                "retained": retained_count,
                "train": len(train_samples),
                "dev": len(dev_samples),
            },
        },
        "val": {
            "raw_conversations": input_stats["val"]["conversation_count"],
            "raw_prediction_points": raw_val_point_count,
            "final_prediction_points": len(val_samples),
        },
        "duplicate_stats": duplicate_stats,
        "group_stats": {
            "group_count": split_manifest["group_count"],
            "group_size_histogram": dict(sorted(Counter(
                group["conversation_count"] for group in groups
            ).items())),
            "max_group_conversation_count": split_manifest["max_group_conversation_count"],
            "max_group_sample_count": split_manifest["max_group_sample_count"],
            "large_group_count": len(split_manifest["large_groups"]),
        },
        "leakage_checks": leakage,
        "distributions": distributions,
        "input_sha256_unchanged": input_unchanged,
        "conservation_valid": outcomes_valid and len(outcome_by_sample) == raw_train_point_count,
        "val_count_valid": val_count_valid,
        "verification": {
            "unit_test_command": config.get("validation", {}).get(
                "unit_test_command", "python -m pytest tests/test_data_audit.py tests/test_data_pipeline.py -q"
            ),
            "unit_test_result": test_result,
            "full_dataset_repeat_result": full_repeat_result,
        },
    }

    # Add stable review/disposition entries. No input text or target content is written to the log.
    cleaning_log = quarantine["train"] + dedup_log + review_log
    action_order = {"quarantined": 0, "deduplicated": 1, "review_retained": 2}
    cleaning_log.sort(key=lambda item: (
        0 if item["split"] == "train" else 1,
        item["source_line"],
        item["prediction_point_index"],
        action_order.get(item["action"], 9),
        item["reason"],
    ))

    # Prepare each deliverable beside its final path so rename publication is atomic per filesystem.
    stage_dirs: dict[str, Path] = {}
    stage_report: Path | None = None
    try:
        for key in ("normalized", "artifacts"):
            final = output_paths[key]
            final.parent.mkdir(parents=True, exist_ok=True)
            stage_dirs[key] = Path(tempfile.mkdtemp(prefix=f".{final.name}.stage-", dir=final.parent))
        output_jsonl = {
            "train": (stage_dirs["normalized"] / "train.jsonl", output_paths["normalized"] / "train.jsonl", train_samples),
            "dev": (stage_dirs["normalized"] / "dev.jsonl", output_paths["normalized"] / "dev.jsonl", dev_samples),
            "val": (stage_dirs["normalized"] / "val.jsonl", output_paths["normalized"] / "val.jsonl", val_samples),
        }
        output_records: list[dict[str, Any]] = []
        for split, (stage_path, final_path, items) in output_jsonl.items():
            _write_jsonl(stage_path, items)
            _validate_jsonl_matches(stage_path, items, split)
            output_records.append(_file_record(stage_path, final_path, root, records=len(items)))

        cleaning_stage = stage_dirs["artifacts"] / "cleaning_log.jsonl"
        _write_jsonl(cleaning_stage, cleaning_log)
        if len(cleaning_stage.read_text(encoding="utf-8").splitlines()) != len(cleaning_log):
            raise RuntimeError("Staged cleaning log record count mismatch")
        split_stage = stage_dirs["artifacts"] / "split_manifest.json"
        _write_json(split_stage, split_manifest)
        summary_stage = stage_dirs["artifacts"] / "summary.json"
        _write_json(summary_stage, summary)
        output_records.extend([
            _file_record(cleaning_stage, output_paths["artifacts"] / "cleaning_log.jsonl", root, records=len(cleaning_log)),
            _file_record(split_stage, output_paths["artifacts"] / "split_manifest.json", root),
            _file_record(summary_stage, output_paths["artifacts"] / "summary.json", root),
        ])

        config_sha = _sha256_bytes(_json_bytes(config))
        config_file_sha = _sha256_file(config_path) if config_path and config_path.is_file() else None
        code_info = {}
        for relative in ("src/data/prepare.py", "src/data/split.py"):
            code_path = root / relative
            if not code_path.is_file():
                code_path = PROJECT_ROOT / relative
            code_info[relative] = _sha256_file(code_path)
        manifest_core = {
            "pipeline_version": PIPELINE_VERSION,
            "policy_version": summary["policy_version"],
            "effective_config": config,
            "config_sha256": config_sha,
            "config_file_sha256": config_file_sha,
            "inputs": input_summary,
            "seed": seed,
            "dev_ratio": dev_ratio,
            "group_count": split_manifest["group_count"],
            "max_group_conversation_count": split_manifest["max_group_conversation_count"],
            "source_code_sha256": code_info,
            "git": _git_facts(root),
            "python_version": platform.python_version(),
            "platform": sys.platform,
        }
        report_text = _render_report(summary, split_manifest, manifest_core, test_result, full_repeat_result)
        report_final = output_paths["report"]
        report_final.parent.mkdir(parents=True, exist_ok=True)
        fd, report_temp_name = tempfile.mkstemp(prefix=f".{report_final.name}.stage-", dir=report_final.parent)
        stage_report = Path(report_temp_name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(report_text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        output_records.append(_file_record(stage_report, report_final, root))

        manifest = {
            **manifest_core,
            "input_sha256_unchanged": input_unchanged,
            "conservation_valid": summary["conservation_valid"],
            "val_count_valid": val_count_valid,
            "cleaning_log_counts": dict(sorted(Counter(item["action"] for item in cleaning_log).items())),
            "duplicate_stats": duplicate_stats,
            "group_stats": summary["group_stats"],
            "split_summary": {
                key: value for key, value in split_manifest.items()
                if key not in {"groups", "conversation_to_split"}
            },
            "outputs": output_records,
        }
        manifest_stage = stage_dirs["artifacts"] / "manifest.json"
        _write_json(manifest_stage, manifest)

        # Re-check originals immediately before publishing completed outputs.
        after_publish_guard = {split: _sha256_file(input_paths[split]) for split in ("train", "val")}
        if config.get("validation", {}).get("assert_input_unchanged", True) and after_publish_guard != input_before:
            raise RuntimeError("An original input changed before output publication")

        _publish_transaction([
            (stage_dirs["normalized"], output_paths["normalized"]),
            (stage_dirs["artifacts"], output_paths["artifacts"]),
            (stage_report, output_paths["report"]),
        ])
        stage_dirs.clear()
        stage_report = None
    finally:
        for stage in stage_dirs.values():
            _remove_path(stage)
        if stage_report is not None:
            _remove_path(stage_report)

    return {
        "pipeline_version": PIPELINE_VERSION,
        "train": summary["train"],
        "val": summary["val"],
        "duplicate_stats": duplicate_stats,
        "group_stats": summary["group_stats"],
        "dev_conversations": split_manifest["actual_dev_conversation_count"],
        "dev_conversation_fraction": split_manifest["actual_dev_conversation_fraction"],
        "dev_sample_fraction": split_manifest["actual_dev_sample_fraction"],
        "leakage_checks": leakage,
        "input_sha256_unchanged": input_unchanged,
        "outputs": {key: str(path.relative_to(root)) for key, path in output_paths.items()},
        "output_sha256": {record["path"]: record["sha256"] for record in output_records},
    }


def load_config(root: Path, config_path: Path) -> tuple[dict[str, Any], Path]:
    root = root.resolve()
    path = config_path if config_path.is_absolute() else root / config_path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Config path must be inside the project root")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be a YAML object")
    return config, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic D0.3 normalized datasets")
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument(
        "--test-result",
        default=None,
        help="Optional observed test summary to include in the generated build report",
    )
    parser.add_argument(
        "--full-repeat-result",
        default=None,
        help="Optional observed full-data byte-repeatability result for the build report",
    )
    args = parser.parse_args(argv)
    try:
        config, config_path = load_config(PROJECT_ROOT, Path(args.config))
        result = run_pipeline(
            PROJECT_ROOT,
            config,
            config_path=config_path,
            test_result=args.test_result,
            full_repeat_result=args.full_repeat_result,
        )
    except Exception as exc:
        print(f"D0.3 build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
