#!/usr/bin/env python3
"""CPU-safe identity and route checks for the E02 vLLM LoRA smoke.

This tool never loads a model or starts a server. ``preflight`` checks local
Base/Adapter files and the installed vLLM CLI surface; ``models`` verifies a
running OpenAI-compatible server advertises the requested model ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LORA_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}
_ADAPTER_SUFFIXES = (".safetensors", ".bin")
_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    status = git("status", "--porcelain", "--untracked-files=all")
    files = {}
    for relative in (
        "participant/start.sh",
        "participant/run_inference.py",
        "scripts/e02_vllm_probe.py",
        "src/shared/prompting.py",
        "configs/deployment/e02_vllm_smoke.json",
        "configs/deployment/e02_vllm_requirements.txt",
    ):
        path = WORKSPACE_ROOT / relative
        if path.is_file():
            files[relative] = sha256_file(path)
    return {
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "worktree_clean": status == "" if status is not None else None,
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest() if status is not None else None,
        "critical_file_sha256": files,
        "critical_files_identity_sha256": _aggregate_identity([
            {"path": name, "sha256": digest} for name, digest in sorted(files.items())
        ]),
    }


def _aggregate_identity(files: list[dict[str, Any]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _weight_files(model_dir: Path) -> tuple[list[Path], str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Base model.safetensors.index.json 缺少有效 weight_map")
        names = sorted(set(weight_map.values()))
        candidates: list[Path] = []
        for name in names:
            if not isinstance(name, str):
                raise ValueError("Base weight_map 包含非字符串路径")
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Base 权重分片路径越界: {name}")
            candidates.append(model_dir / relative)
        mode = "indexed_safetensors_shards_sha256"
    else:
        candidates = sorted(
            path for path in model_dir.rglob("*")
            if path.is_file() and (
                path.name.endswith(".safetensors")
                or path.name.endswith(".gguf")
                or path.name == "pytorch_model.bin"
                or path.name.startswith("pytorch_model-") and path.name.endswith(".bin")
            )
        )
        mode = "all_supported_weight_files_sha256"
    if not candidates or any(not path.is_file() or path.stat().st_size <= 0 for path in candidates):
        raise FileNotFoundError(f"Base 权重分片缺失或为空: {model_dir}")
    return candidates, mode


def base_weight_identity(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Base 目录不存在或缺少 config.json: {model_dir}")
    candidates, mode = _weight_files(model_dir)
    files = [
        {
            "path": path.relative_to(model_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in candidates
    ]
    return {
        "local_path": str(model_dir),
        "identity_status": "observed",
        "fingerprint_status": mode,
        "observed_weight_hash": _aggregate_identity(files),
        "files": files,
    }


def adapter_identity(adapter_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter_dir = adapter_dir.expanduser().resolve()
    config_path = adapter_dir / "adapter_config.json"
    if not adapter_dir.is_dir() or not config_path.is_file():
        raise FileNotFoundError(f"Adapter 目录或 adapter_config.json 不存在: {adapter_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = sorted(
        path for path in adapter_dir.iterdir()
        if path.is_file() and (
            path.name == "adapter_model.safetensors"
            or path.name == "adapter_model.bin"
            or re.fullmatch(r"adapter_model-[0-9]+-of-[0-9]+\.safetensors", path.name)
        )
    )
    if not candidates or any(path.stat().st_size <= 0 for path in candidates):
        raise FileNotFoundError(f"Adapter 权重缺失或为空: {adapter_dir}")
    files = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in candidates
    ]
    return {
        "local_path": str(adapter_dir),
        "identity_status": "observed",
        "weight_sha256": _aggregate_identity(files),
        "config_sha256": sha256_file(config_path),
        "files": files,
    }, config


def validate_alias(alias: str, *, base_alias: str | None = None) -> str:
    if not isinstance(alias, str) or not _ALIAS_RE.fullmatch(alias):
        raise ValueError("LoRA 服务别名必须是非空、无空格的模型 ID")
    if base_alias is not None and alias == base_alias:
        raise ValueError("LoRA 别名必须与 Base served-model-name 不同")
    return alias


def validate_artifact_identity(
    *,
    base_path: Path,
    adapter_path: Path,
    model_id: str,
    alias: str,
    base_alias: str,
    max_lora_rank: int,
    expected_base_sha256: str,
    expected_adapter_sha256: str,
    expected_adapter_config_sha256: str,
) -> dict[str, Any]:
    validate_alias(alias, base_alias=base_alias)
    if not isinstance(max_lora_rank, int) or isinstance(max_lora_rank, bool) or max_lora_rank <= 0:
        raise ValueError("max_lora_rank 必须为正整数")
    for label, value in (
        ("Base", expected_base_sha256),
        ("Adapter 权重", expected_adapter_sha256),
        ("Adapter 配置", expected_adapter_config_sha256),
    ):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(f"expected {label} SHA-256 必须是 64 位十六进制")

    base = base_weight_identity(base_path)
    if base["observed_weight_hash"] != expected_base_sha256.lower():
        raise ValueError("本地 Base 权重 SHA-256 与冻结基线不匹配")
    adapter, config = adapter_identity(adapter_path)
    if adapter["weight_sha256"] != expected_adapter_sha256.lower():
        raise ValueError("本地 Adapter 权重 SHA-256 与冻结 Full B 不匹配")
    if adapter["config_sha256"] != expected_adapter_config_sha256.lower():
        raise ValueError("本地 Adapter config SHA-256 与冻结 Full B 不匹配")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Adapter 必须是 PEFT LoRA CAUSAL_LM")
    rank = config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("Adapter adapter_config.json 中的 r 必须为正整数")
    if rank > max_lora_rank:
        raise ValueError(f"Adapter rank={rank} 超过 max_lora_rank={max_lora_rank}")
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or set(target_modules) != LORA_TARGET_MODULES:
        raise ValueError("Adapter target_modules 与 Full B 冻结的 7 个 LoRA 模块不一致")
    declared_base = config.get("base_model_name_or_path")
    if not isinstance(declared_base, str) or not declared_base.strip():
        raise ValueError("Adapter 未声明 base_model_name_or_path")
    accepted_base_paths = {str(Path(base_path).expanduser().resolve())}
    if declared_base != model_id and str(Path(declared_base).expanduser().resolve()) not in accepted_base_paths:
        raise ValueError("Adapter 声明的 Base 模型与当前模型不匹配")
    return {
        "status": "passed_local_artifact_preflight",
        "base": base,
        "adapter": {
            **adapter,
            "config_summary": {
                key: config.get(key)
                for key in ("peft_type", "task_type", "r", "lora_alpha", "lora_dropout", "target_modules", "base_model_name_or_path")
            },
        },
        "requested_adapter_model_id": alias,
        "server_weight_identity": "unverified_until_vllm_runtime",
    }


def advertised_model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("/v1/models 响应缺少 data 列表")
    model_ids = []
    for item in payload["data"]:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
            model_ids.append(item["id"])
    if not model_ids:
        raise ValueError("/v1/models 没有返回有效模型 ID")
    return model_ids


def verify_requested_model(payload: Any, requested_model: str) -> list[str]:
    ids = advertised_model_ids(payload)
    if requested_model not in ids:
        raise ValueError(f"请求模型 {requested_model!r} 未出现在服务模型列表: {ids}")
    return ids


def nvidia_gpu_facts() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise FileNotFoundError("vLLM smoke 需要 nvidia-smi 读取 GPU 资源信息")
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,compute_cap,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"nvidia-smi GPU 查询失败: {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 5:
            raise ValueError("nvidia-smi GPU 查询返回了非预期字段")
        rows.append({
            "name": columns[0],
            "compute_capability": columns[1],
            "driver_version": columns[2],
            "memory_total_mib": columns[3],
            "memory_used_mib_at_observation": columns[4],
        })
    return {"status": "observed", "source": "nvidia-smi", "devices": rows}


def inspect_vllm_cli(binary: str, expected_version: str | None = None) -> dict[str, Any]:
    executable = shutil.which(binary)
    if executable is None:
        raise FileNotFoundError(f"找不到 vLLM 命令: {binary}")
    version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=30, check=False)
    help_result = subprocess.run([executable, "serve", "--help=all"], capture_output=True, text=True, timeout=60, check=False)
    help_text = help_result.stdout + "\n" + help_result.stderr
    if version.returncode != 0:
        raise RuntimeError(f"vLLM --version 失败: {version.stderr[-1000:]}")
    if help_result.returncode != 0:
        raise RuntimeError(f"vLLM serve --help 失败: {help_text[-1000:]}")
    required_flags = (
        "--enable-lora", "--lora-modules", "--max-lora-rank", "--dtype",
        "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
    )
    missing = [flag for flag in required_flags if flag not in help_text]
    if missing:
        raise RuntimeError(f"当前 vLLM CLI 缺少 LoRA 参数: {missing}")
    version_text = version.stdout + "\n" + version.stderr
    found_versions = [
        match.group(1)
        for line in version_text.splitlines()
        if (match := re.fullmatch(
            r"(?:vllm(?:,\s*version)?\s+)?v?(\d+\.\d+\.\d+)",
            line.strip(),
            flags=re.IGNORECASE,
        )) is not None
    ]
    if len(found_versions) != 1:
        raise RuntimeError(
            f"无法唯一确定 vLLM release 版本: {version_text[-1000:]!r}"
        )

    actual_version = found_versions[0]
    if expected_version and actual_version != expected_version:
        raise RuntimeError(
            f"vLLM 版本与 E02.1 已锁定版本不一致: expected={expected_version}, actual={actual_version}"
        )
    return {
        "executable": executable,
        "version": actual_version,
        "required_lora_flags": list(required_flags),
        "status": "passed_cli_surface_check",
    }


def resolve_gpu_memory_utilization(
    config_value: Any,
    override_value: str | None,
) -> dict[str, str | None]:
    """Resolve and validate the vLLM memory fraction without changing defaults."""
    configured = None if config_value is None else str(config_value).strip()
    override = None if override_value is None else override_value.strip()
    if override == "":
        override = None
    effective = override if override is not None else configured
    source = "environment_override" if override is not None else "config" if configured is not None else "vllm_default"
    if effective is not None:
        try:
            numeric = float(effective)
        except (TypeError, ValueError) as exc:
            raise ValueError("GPU 显存利用率必须是数字") from exc
        if not math.isfinite(numeric) or not 0.0 < numeric <= 1.0:
            raise ValueError("GPU 显存利用率必须在 (0, 1] 范围内")
    return {
        "config_value": configured,
        "override_value": override,
        "effective_value": effective,
        "source": source,
    }


def _request_models(base_url: str, api_key: str, timeout: float) -> Any:
    endpoint = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"读取服务模型列表失败: {exc}") from exc


def write_smoke_manifest(
    *,
    run_root: Path,
    runner_dir: Path,
    test_file: Path,
    config_path: Path,
    vllm_version: str,
    server_model_ids: list[str],
    requested_model: str,
    gpu_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a hash-only execution manifest after a successful runner process."""
    run_root = run_root.expanduser().resolve()
    runner_dir = runner_dir.expanduser().resolve()
    test_file = test_file.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    report_path = runner_dir / "performance_report.json"
    submission_path = runner_dir / "submission.jsonl"
    if not run_root.is_dir() or not runner_dir.is_dir() or not test_file.is_file() or not config_path.is_file():
        raise FileNotFoundError("E02.1 Run 根目录、Runner 结果、测试输入或配置缺失")
    if not report_path.is_file() or not submission_path.is_file():
        raise FileNotFoundError("Runner 未生成 performance_report.json 和 submission.jsonl")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_adapter = config.get("adapter", {})
    observed_adapter = report.get("adapter_identity", {})
    observed_base = report.get("model_weight_identity", {})
    observed_route = observed_adapter.get("server_route_observed", {})
    expected_vllm_version = config.get("deployment", {}).get("vllm", {}).get("expected_version")
    input_rows = sum(1 for line in test_file.open(encoding="utf-8") if line.strip())
    check_values = {
        "runner_complete": report.get("complete") is True,
        "submission_schema_validated": report.get("submission_schema_validated") is True,
        "sample_count_matches_input": report.get("samples") == input_rows,
        "requested_route_advertised": requested_model in server_model_ids,
        "runner_requested_expected_model": report.get("api_model") == requested_model,
        "runner_vllm_version_matches": report.get("api_server_runtime", {}).get("vllm_version") == vllm_version,
        "pinned_vllm_version_matches": not expected_vllm_version or vllm_version == expected_vllm_version,
        "gpu_runtime_observed": isinstance(gpu_facts, dict) and gpu_facts.get("status") == "observed",
        "vllm_environment_freeze_recorded": (run_root / "vllm_environment_freeze.txt").is_file(),
        "local_base_reference_matches": observed_base.get("local_reference_weight_sha256") == expected_adapter.get("expected_base_weight_sha256"),
        "local_adapter_weight_matches": observed_adapter.get("weight_sha256") == expected_adapter.get("expected_adapter_weight_sha256"),
        "local_adapter_config_matches": observed_adapter.get("config_sha256") == expected_adapter.get("expected_adapter_config_sha256"),
        "runner_route_matches": observed_route.get("requested_model_id") == requested_model,
    }
    manifest = {
        "schema_version": "e02_vllm_smoke_v1",
        "run_id": report.get("run_id"),
        "status": "completed" if all(check_values.values()) else "failed",
        "acceptance_status": (
            "partial_not_verified_server_token_ids"
            if str(report.get("server_exact_token_id_equivalence", "")).startswith("BLOCKED")
            else "review_required"
        ),
        "checks": check_values,
        "code_provenance": code_provenance(),
        "execution": {
            "backend": report.get("backend"),
            "vllm_version": vllm_version,
            "requested_model_id": requested_model,
            "advertised_model_ids": server_model_ids,
            "gpu_runtime": gpu_facts,
            "server_launch_configuration": report.get("api_server_runtime", {}).get("launch_configuration"),
            "server_weight_identity": "unverified",
            "local_base_reference_weight_sha256": observed_base.get("local_reference_weight_sha256"),
            "local_adapter_weight_sha256": observed_adapter.get("weight_sha256"),
            "local_adapter_config_sha256": observed_adapter.get("config_sha256"),
            "adapter_rank": (observed_adapter.get("locally_verified_adapter_file") or {}).get("rank"),
            "adapter_target_modules": (observed_adapter.get("locally_verified_adapter_file") or {}).get("target_modules"),
        },
        "protocol": {
            "prompt": report.get("protocol", {}).get("prompt"),
            "tokenizer": report.get("protocol", {}).get("tokenizer"),
            "generation_config": report.get("generation_config"),
            "budget": report.get("budget"),
            "server_exact_token_id_equivalence": report.get("server_exact_token_id_equivalence"),
        },
        "counts": {
            "input_rows": input_rows,
            "submission_rows": report.get("submission_rows"),
            "success_count": report.get("success_count"),
            "failure_counts": report.get("failure_counts"),
            "parse_failure_count": report.get("parse_failure_count"),
            "invalid_label_count": report.get("invalid_label_count"),
            "prompt_overflow_count": report.get("prompt_overflow_count"),
            "template_equivalence_failure_count": report.get("template_equivalence_failure_count"),
            "server_prompt_token_mismatch_count": report.get("server_prompt_token_mismatch_count"),
        },
        "hashes": {
            "inference_config_sha256": sha256_file(config_path),
            "test_file_sha256": sha256_file(test_file),
            "submission_sha256": sha256_file(submission_path),
            "performance_report_sha256": sha256_file(report_path),
            "raw_predictions_sha256": report.get("raw_predictions_sha256"),
            "vllm_environment_freeze_sha256": (
                sha256_file(run_root / "vllm_environment_freeze.txt")
                if (run_root / "vllm_environment_freeze.txt").is_file() else None
            ),
        },
    }
    manifest_path = run_root / "run_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {"status": manifest["status"], "run_manifest": str(manifest_path), "checks": check_values}


def write_unlabelled_sample(source_path: Path, output_path: Path, count: int = 10) -> dict[str, Any]:
    """Select evenly spaced short-to-long histories from id+history-only input."""
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"無标签 Dev 输入不存在: {source_path}")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("样本数必须为正整数")
    rows = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"id", "history"}:
                raise ValueError(f"样本来源第 {line_number} 行不是 id+history-only 输入")
            if not isinstance(row["id"], str) or not row["id"] or not isinstance(row["history"], list):
                raise ValueError(f"样本来源第 {line_number} 行 ID/History 无效")
            if any(
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in row["history"]
            ):
                raise ValueError(f"样本来源第 {line_number} 行含非标准 History message")
            rows.append(row)
    if not rows or count > len(rows):
        raise ValueError(f"请求样本数需在 1 到 {len(rows)} 之间")
    if count == 1:
        selected = [min(rows, key=lambda row: sum(len(message["content"]) for message in row["history"]))]
    else:
        ordered = sorted(rows, key=lambda row: sum(len(message["content"]) for message in row["history"]))
        indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
        selected = [ordered[index] for index in indices]
    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("分层抽样生成重复 ID")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    lengths = [sum(len(message["content"]) for message in row["history"]) for row in selected]
    return {
        "status": "created_unlabelled_sample",
        "sample_count": len(selected),
        "input_rows": len(rows),
        "history_character_length_min_max": [min(lengths), max(lengths)],
        "sample_sha256": sha256_file(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="本地权重身份与 vLLM CLI 检查，不加载模型")
    preflight.add_argument("--base-path", required=True, type=Path)
    preflight.add_argument("--adapter-path", required=True, type=Path)
    preflight.add_argument("--model-id", required=True)
    preflight.add_argument("--alias", required=True)
    preflight.add_argument("--base-alias", required=True)
    preflight.add_argument("--max-lora-rank", required=True, type=int)
    preflight.add_argument("--expected-base-sha256", required=True)
    preflight.add_argument("--expected-adapter-sha256", required=True)
    preflight.add_argument("--expected-adapter-config-sha256", required=True)
    preflight.add_argument("--vllm-bin", default="vllm")
    preflight.add_argument("--expected-vllm-version")

    models = subparsers.add_parser("models", help="验证 /v1/models 包含目标请求模型 ID")
    models.add_argument("--base-url", required=True, help="OpenAI API Base URL，通常以 /v1 结尾")
    models.add_argument("--requested-model", required=True)
    models.add_argument("--api-key", default="")
    models.add_argument("--timeout", type=float, default=5.0)

    manifest_parser = subparsers.add_parser("write-manifest", help="写入不含原始对话的 E02.1 运行指纹")
    manifest_parser.add_argument("--run-root", required=True, type=Path)
    manifest_parser.add_argument("--runner-dir", required=True, type=Path)
    manifest_parser.add_argument("--test-file", required=True, type=Path)
    manifest_parser.add_argument("--config", required=True, type=Path)
    manifest_parser.add_argument("--vllm-version", required=True)
    manifest_parser.add_argument("--server-model-ids-json", required=True)
    manifest_parser.add_argument("--requested-model", required=True)
    manifest_parser.add_argument("--gpu-facts-json", default="")

    sample_parser = subparsers.add_parser("make-sample", help="从无标签 id+history 输入抽取短到长样本")
    sample_parser.add_argument("--source", required=True, type=Path)
    sample_parser.add_argument("--output", required=True, type=Path)
    sample_parser.add_argument("--count", type=int, default=10)

    gpu_parser = subparsers.add_parser("gpu-facts", help="读取当前 NVIDIA GPU/显存观察值")

    memory_parser = subparsers.add_parser(
        "resolve-gpu-memory", help="验证并解析 vLLM GPU 显存利用率配置"
    )
    memory_parser.add_argument("--config-value", default="")
    memory_parser.add_argument("--override-value", default="")
    memory_parser.add_argument("--format", choices=("json", "nul"), default="json")

    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = validate_artifact_identity(
            base_path=args.base_path,
            adapter_path=args.adapter_path,
            model_id=args.model_id,
            alias=args.alias,
            base_alias=args.base_alias,
            max_lora_rank=args.max_lora_rank,
            expected_base_sha256=args.expected_base_sha256,
            expected_adapter_sha256=args.expected_adapter_sha256,
            expected_adapter_config_sha256=args.expected_adapter_config_sha256,
        )
        result["vllm_cli"] = inspect_vllm_cli(args.vllm_bin, args.expected_vllm_version)
    elif args.command == "models":
        payload = _request_models(args.base_url, args.api_key, args.timeout)
        result = {
            "status": "passed_requested_route_observed",
            "requested_model_id": args.requested_model,
            "server_model_ids": verify_requested_model(payload, args.requested_model),
            "server_weight_identity": "unverified",
        }
    elif args.command == "write-manifest":
        try:
            server_model_ids = json.loads(args.server_model_ids_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--server-model-ids-json 必须是 JSON 字符串数组") from exc
        if not isinstance(server_model_ids, list) or any(not isinstance(item, str) for item in server_model_ids):
            raise ValueError("--server-model-ids-json 必须是 JSON 字符串数组")
        try:
            gpu_facts = json.loads(args.gpu_facts_json) if args.gpu_facts_json else None
        except json.JSONDecodeError as exc:
            raise ValueError("--gpu-facts-json 必须为 JSON object") from exc
        if gpu_facts is not None and not isinstance(gpu_facts, dict):
            raise ValueError("--gpu-facts-json 必须为 JSON object")
        result = write_smoke_manifest(
            run_root=args.run_root,
            runner_dir=args.runner_dir,
            test_file=args.test_file,
            config_path=args.config,
            vllm_version=args.vllm_version,
            server_model_ids=server_model_ids,
            requested_model=args.requested_model,
            gpu_facts=gpu_facts,
        )
    elif args.command == "make-sample":
        result = write_unlabelled_sample(args.source, args.output, args.count)
    elif args.command == "resolve-gpu-memory":
        result = resolve_gpu_memory_utilization(args.config_value or None, args.override_value)
    else:
        result = nvidia_gpu_facts()
    if args.command == "resolve-gpu-memory" and args.format == "nul":
        for key in ("config_value", "override_value", "effective_value", "source"):
            sys.stdout.write((result[key] or "") + "\0")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
