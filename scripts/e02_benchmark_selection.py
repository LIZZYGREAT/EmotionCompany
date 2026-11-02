#!/usr/bin/env python3
"""Create a deterministic, target-free 100-row E02.2-B candidate set.

The selector uses only sample IDs and conversation histories from the source
JSONL. It measures prompt lengths through the shared P0/F2/T1 Runner path,
then samples deterministically from short, medium, and long prompt-length
tertiles. Candidate inputs and manifests are written under ignored outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
E02_OUTPUT_ROOT = (WORKSPACE_ROOT / "outputs/evaluation/e02").resolve()
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_SEED = 20261101
DEFAULT_QUOTAS = {"short": 34, "medium": 33, "long": 33}
BENCHMARK_ID_START = 900001
SELECTION_RULE = {
    "version": "e02.2-b-selection-v2",
    "source_identity": "D0.3 sample_id",
    "smoke_identity": "local eval_id mapped to D0.3 sample_id through id_map.jsonl",
    "smoke_exclusion": "exclude exact mapped D0.3 sample_id overlap",
    "length": "effective prompt token count from shared P0/F2/T1 path",
    "strata": "sort by prompt token count then id; split into equal-count tertiles",
    "sampling": "sha256 seeded rank within each stratum",
    "request_order": "sha256 seeded rank over selected rows",
    "seed": DEFAULT_SEED,
    "sample_count": DEFAULT_SAMPLE_COUNT,
    "stratum_quotas": DEFAULT_QUOTAS,
    "benchmark_id_namespace": "synthetic test_900001_t0 through test_900100_t0; source sample_id is manifest-only",
    "target_fields_used": False,
}

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from participant import run_inference as runner  # noqa: E402
from scripts import e02_token_equivalence as token_equivalence  # noqa: E402
from scripts import e02_vllm_probe as probe  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _stable_rank(seed: int, domain: str, sample_id: str) -> str:
    material = f"e02.2-b-v1\0{seed}\0{domain}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def project_target_free_sample(row: Any, *, line_number: int) -> dict[str, Any]:
    """Project either processed-dev or target-free rows onto only id + history."""
    if not isinstance(row, dict):
        raise ValueError(f"输入第 {line_number} 行必须是 JSON object")
    sample_id = row.get("id")
    if sample_id is None:
        sample_id = row.get("sample_id")
    history = row.get("history")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"输入第 {line_number} 行缺少字符串 id/sample_id")
    if not isinstance(history, list) or not history:
        raise ValueError(f"输入第 {line_number} 行 history 必须是非空数组")
    projected_history: list[dict[str, str]] = []
    for message in history:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError(f"输入第 {line_number} 行含非标准 History message")
        projected_history.append({"role": message["role"], "content": message["content"]})
    if projected_history[-1]["role"] != "user":
        raise ValueError(f"输入第 {line_number} 行必须以当前 user 消息结束")
    return {"id": sample_id, "history": projected_history}


def read_target_free_projection(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read only id/history fields for selection and hash their canonical projection."""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    projection_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"输入第 {line_number} 行不是有效 UTF-8 JSON") from exc
            sample = project_target_free_sample(row, line_number=line_number)
            if sample["id"] in seen_ids:
                raise ValueError(f"输入第 {line_number} 行 ID 重复: {sample['id']}")
            seen_ids.add(sample["id"])
            projection_digest.update(_canonical_json(sample) + b"\n")
            rows.append(sample)
    if not rows:
        raise ValueError("候选输入为空")
    return rows, projection_digest.hexdigest()


def map_smoke_ids_to_sample_ids(
    smoke_input_path: Path, id_map_path: Path
) -> tuple[set[str], dict[str, Any]]:
    """Resolve local eval IDs to D0.3 sample IDs before overlap filtering."""
    smoke_rows = token_equivalence.read_unlabelled_input(smoke_input_path)
    eval_to_sample: dict[str, str] = {}
    sample_ids: set[str] = set()
    with id_map_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ID map 第 {line_number} 行不是有效 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"ID map 第 {line_number} 行必须是 JSON object")
            eval_id = row.get("eval_id")
            sample_id = row.get("sample_id")
            if not isinstance(eval_id, str) or not eval_id:
                raise ValueError(f"ID map 第 {line_number} 行缺少非空 eval_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"ID map 第 {line_number} 行缺少非空 sample_id")
            if eval_id in eval_to_sample:
                raise ValueError(f"ID map 存在重复 eval_id: {eval_id}")
            if sample_id in sample_ids:
                raise ValueError(f"ID map 存在重复 sample_id: {sample_id}")
            eval_to_sample[eval_id] = sample_id
            sample_ids.add(sample_id)

    smoke_eval_ids = {row["id"] for row in smoke_rows}
    unmapped = sorted(smoke_eval_ids - set(eval_to_sample))
    if unmapped:
        raise ValueError(f"Smoke input 中有 ID 未出现在 id_map.jsonl: {unmapped}")
    mapped_ids = {eval_to_sample[eval_id] for eval_id in smoke_eval_ids}
    return mapped_ids, {
        "smoke_eval_id_count": len(smoke_eval_ids),
        "mapped_sample_id_count": len(mapped_ids),
        "smoke_eval_ids": sorted(smoke_eval_ids),
        "mapped_sample_ids": sorted(mapped_ids),
        "id_map_sha256": token_equivalence.sha256_file(id_map_path),
    }


def exclude_sample_id_overlap(
    source_rows: list[dict[str, Any]], excluded_sample_ids: set[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    """Remove candidates only when their D0.3 sample_id is in the mapped set."""
    overlap = {row["id"] for row in source_rows} & excluded_sample_ids
    return [row for row in source_rows if row["id"] not in excluded_sample_ids], overlap


def assign_length_strata(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(candidates) < 3:
        raise ValueError("候选池至少需要 3 条样本才能分为短、中、长三层")
    ordered = sorted(candidates, key=lambda item: (item["prompt_tokens"], item["id"]))
    first_cut = len(ordered) // 3
    second_cut = (2 * len(ordered)) // 3
    return {
        "short": ordered[:first_cut],
        "medium": ordered[first_cut:second_cut],
        "long": ordered[second_cut:],
    }


def select_stratified_candidates(
    candidates: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    quotas: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    quotas = dict(quotas or DEFAULT_QUOTAS)
    if set(quotas) != {"short", "medium", "long"} or any(value < 0 for value in quotas.values()):
        raise ValueError("stratum quota 必须包含非负的 short/medium/long")
    strata = assign_length_strata(candidates)
    selected: list[dict[str, Any]] = []
    facts: dict[str, dict[str, Any]] = {}
    for name in ("short", "medium", "long"):
        values = strata[name]
        quota = quotas[name]
        if len(values) < quota:
            raise ValueError(f"{name} 候选不足: available={len(values)}, required={quota}")
        ranked = sorted(values, key=lambda item: (_stable_rank(seed, name, item["id"]), item["id"]))
        chosen = [dict(item, stratum=name) for item in ranked[:quota]]
        selected.extend(chosen)
        facts[name] = {
            "candidate_count": len(values),
            "selected_count": len(chosen),
            "prompt_token_min": min(item["prompt_tokens"] for item in values),
            "prompt_token_max": max(item["prompt_tokens"] for item in values),
        }
    selected.sort(key=lambda item: (_stable_rank(seed, "request-order", item["id"]), item["id"]))
    return selected, facts


def _write_jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json({"id": row["id"], "history": row["history"]}) + b"\n" for row in rows)


def assign_benchmark_input_ids(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use submission-schema-compatible synthetic IDs while retaining source IDs in metadata."""
    if len(selected) != DEFAULT_SAMPLE_COUNT:
        raise ValueError(f"E02.2-B 候选输入必须恰好为 {DEFAULT_SAMPLE_COUNT} 条")
    return [
        {
            **row,
            "source_sample_id": row["id"],
            "id": f"test_{BENCHMARK_ID_START + index:06d}_t0",
        }
        for index, row in enumerate(selected)
    ]


def run_selection(
    *,
    config_path: Path,
    source_path: Path,
    exclude_input_path: Path,
    exclude_id_map_path: Path,
    run_root: Path,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    source_path = source_path.expanduser().resolve()
    exclude_input_path = exclude_input_path.expanduser().resolve()
    exclude_id_map_path = exclude_id_map_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    if run_root.exists():
        raise FileExistsError(f"拒绝覆盖已有 E02.2-B 候选 Run: {run_root}")
    try:
        run_root.relative_to(E02_OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Run 目录必须位于 ignored 目录 {E02_OUTPUT_ROOT}") from exc
    if sample_count != DEFAULT_SAMPLE_COUNT:
        raise ValueError(f"本版选择协议固定抽取 {DEFAULT_SAMPLE_COUNT} 条")
    for path in (config_path, source_path, exclude_input_path, exclude_id_map_path):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")

    config = runner.load_config(config_path)
    prompt_cap = config["context"]["max_prompt_tokens"]
    if prompt_cap != 1280:
        raise ValueError(f"E02.2-B 冻结 Prompt 上限必须为 1280，当前为 {prompt_cap}")
    base_path = token_equivalence._resolve_base_path(config)
    if not base_path.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在: {base_path}")
    prompt_text, _schema, _enums, prompt_facts = runner.load_runtime_protocol(config)
    tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(config, base_path)

    source_rows, projection_sha = read_target_free_projection(source_path)
    excluded_sample_ids, mapping_facts = map_smoke_ids_to_sample_ids(
        exclude_input_path, exclude_id_map_path
    )
    rows, overlap_ids = exclude_sample_id_overlap(source_rows, excluded_sample_ids)

    measured: list[dict[str, Any]] = []
    rejected_floor_count = 0
    for row in rows:
        selection, _messages = runner.prepare_sample_messages(row, prompt_text, tokenizer, prompt_cap)
        prompt_tokens = selection["prompt_tokens"]
        if selection["floor_exceeds_cap"] or prompt_tokens > prompt_cap:
            rejected_floor_count += 1
            continue
        measured.append({**row, "prompt_tokens": prompt_tokens})
    if len(measured) < sample_count:
        raise ValueError(f"有效 Prompt 候选不足: eligible={len(measured)}, required={sample_count}")

    selected_sources, stratum_facts = select_stratified_candidates(measured, seed=seed)
    selected = assign_benchmark_input_ids(selected_sources)
    input_bytes = _write_jsonl_bytes(selected)
    input_sha = sha256_bytes(input_bytes)
    prompt_lengths = [item["prompt_tokens"] for item in selected]
    manifest = {
        "schema_version": "e02_vllm_benchmark_selection_v2",
        "run_id": run_root.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_candidate_audit",
        "selection": {
            "rule": "effective_prompt_tokens_sorted_into_equal_count_tertiles_then_sha256_ranked_within_each_stratum",
            "seed": seed,
            "sample_count": sample_count,
            "stratum_quotas": DEFAULT_QUOTAS,
            "strata": stratum_facts,
            "candidate_pool_count": len(measured),
            "excluded_smoke_id_count": len(overlap_ids),
            "mapped_smoke_sample_id_count": mapping_facts["mapped_sample_id_count"],
            "smoke_source_overlap_count": len(overlap_ids),
            "prompt_floor_rejected_count": rejected_floor_count,
            "effective_prompt_tokens_min_max": [min(prompt_lengths), max(prompt_lengths)],
            "target_fields_used": False,
            "selection_fields": ["id/sample_id", "history", "effective_prompt_token_count"],
            "selection_rule_sha256": sha256_bytes(
                json.dumps(SELECTION_RULE, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "selection_rule": SELECTION_RULE,
        },
        "source": {
            "path": str(source_path),
            "rows": len(source_rows),
            "raw_input_sha256": token_equivalence.sha256_file(source_path),
            "id_history_projection_sha256": projection_sha,
            "excluded_smoke_input_path": str(exclude_input_path),
            "excluded_smoke_input_sha256": token_equivalence.sha256_file(exclude_input_path),
            "excluded_smoke_id_map_path": str(exclude_id_map_path),
            **mapping_facts,
            "excluded_smoke_sample_id_overlap_count": len(overlap_ids),
            "excluded_smoke_sample_id_overlap": sorted(overlap_ids),
        },
        "protocol": {
            "config_path": str(config_path),
            "config_sha256": token_equivalence.sha256_file(config_path),
            "p0": prompt_facts,
            "tokenizer": tokenizer_facts,
            "prompt_token_cap": prompt_cap,
            "model_id": config["model_id"],
            "model_revision": config.get("model_revision"),
            "selector_script_sha256": token_equivalence.sha256_file(Path(__file__).resolve()),
        },
        "candidate_input": {
            "path": str(run_root / "candidate_input.jsonl"),
            "rows": len(selected),
            "sha256": input_sha,
            "fields": ["id", "history"],
            "id_namespace": "synthetic test_900001_t0 through test_900100_t0; D0.3 sample_id retained only in selection_manifest.samples",
            "smoke_overlap_count": 0,
        },
        "samples": [
            {
                "id": item["id"],
                "source_sample_id": item["source_sample_id"],
                "stratum": item["stratum"],
                "prompt_tokens": item["prompt_tokens"],
            }
            for item in selected
        ],
    }

    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "candidate_input.jsonl").write_bytes(input_bytes)
    (run_root / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select", help="审计并固定 100 条候选样本")
    select_parser.add_argument("--config", type=Path, required=True)
    select_parser.add_argument("--source", type=Path, required=True)
    select_parser.add_argument("--exclude-input", type=Path, required=True)
    select_parser.add_argument("--exclude-id-map", type=Path, required=True)
    select_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_selection(
            config_path=args.config,
            source_path=args.source,
            exclude_input_path=args.exclude_input,
            exclude_id_map_path=args.exclude_id_map,
            run_root=args.run_root,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed_preflight", "failure_type": type(exc).__name__, "failure_summary": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({
        "status": result["status"],
        "run_root": str(args.run_root.expanduser().resolve()),
        "candidate_input_sha256": result["candidate_input"]["sha256"],
        "sample_count": result["candidate_input"]["rows"],
        "strata": result["selection"]["strata"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
