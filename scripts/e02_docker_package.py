#!/usr/bin/env python3
"""Prepare and verify a self-contained E02.2-C Docker package.

``check`` is CPU-only and does not read model files. ``prepare`` deliberately
performs full Base/Adapter hashing and copies the large local assets into an
ignored outputs/ directory. The Docker image itself is built in a later GPU
resource step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

if __package__:
    from .e02_vllm_probe import (
        adapter_identity,
        base_weight_identity,
        sha256_file,
        validate_artifact_identity,
    )
else:
    from e02_vllm_probe import (
        adapter_identity,
        base_weight_identity,
        sha256_file,
        validate_artifact_identity,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_CONFIG = Path("configs/deployment/e02_vllm_docker.json")
PROMPT_PATH = Path("configs/prompts/p0.txt")
SCHEMA_PATH = Path("participant/templates/submission_schema.json")
COPY_FILES = (
    Path("participant/start.sh"),
    Path("participant/run_inference.py"),
    SCHEMA_PATH,
    DOCKER_CONFIG,
    PROMPT_PATH,
    Path("scripts/e02_vllm_probe.py"),
    Path("docker/e02/Dockerfile"),
    Path("docker/e02/entrypoint.sh"),
    Path("docker/e02/conda-linux-64.explicit.txt"),
    Path("docker/e02/requirements-linux-64.lock"),
)
COPY_DIRS = (Path("src/shared"),)
EXPECTED_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须为 object: {path}")
    return value


def load_config(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return load_json(repo_root / DOCKER_CONFIG)


def validate_static_config(config: dict[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    reference_path = repo_root / "configs/deployment/e02_vllm_smoke.json"
    if not reference_path.is_file():
        raise FileNotFoundError("缺少 E02.1 Smoke 基线配置，无法确认 Docker 配置未漂移")
    reference = load_json(reference_path)
    if config.get("backend") != "openai_compatible":
        raise ValueError("Docker 配置必须使用 openai_compatible 后端")
    if config.get("model_path") != "participant/model":
        raise ValueError("Docker Base 模型路径必须为镜像内 participant/model")
    if config.get("base_url") != "http://127.0.0.1:8000/v1":
        raise ValueError("Docker Runner 必须连接容器内的 vLLM loopback 服务")
    tokenizer = config.get("tokenizer", {})
    if tokenizer.get("local_path") != "model" or tokenizer.get("local_files_only") is not True:
        raise ValueError("Docker Tokenizer 必须从 participant/model 本地离线加载")
    prompt = config.get("prompt", {})
    prompt_file = repo_root / prompt.get("path", "")
    if prompt.get("path") != PROMPT_PATH.as_posix() or not prompt_file.is_file():
        raise ValueError("Docker 配置缺少固定 P0 文件")
    if sha256_file(prompt_file) != prompt.get("sha256"):
        raise ValueError("Docker 配置的 P0 SHA-256 与仓库文件不一致")

    context = config.get("context", {})
    generation = config.get("generation", {})
    vllm = config.get("deployment", {}).get("vllm", {})
    prompt_tokens = context.get("max_prompt_tokens")
    generation_tokens = generation.get("max_new_tokens")
    serving_limit = context.get("serving_context_limit")
    if (prompt_tokens, generation_tokens, serving_limit) != (1280, 1200, 2560):
        raise ValueError("Docker Smoke 必须保留冻结的 P=1280, G=1200, K=2560")
    if vllm.get("expected_version") != "0.11.1" or vllm.get("dtype") != "bfloat16":
        raise ValueError("Docker 配置必须锁定 vLLM 0.11.1 / BF16")
    if vllm.get("max_num_batched_tokens", 0) < prompt_tokens + generation_tokens:
        raise ValueError("vLLM max_num_batched_tokens 小于 P+G")
    if vllm.get("gpu_memory_utilization") != 0.8:
        raise ValueError("Docker Smoke 必须保留 gpu_memory_utilization=0.80")

    adapter = config.get("adapter", {})
    if adapter.get("enabled") is not True or adapter.get("path") != "participant/adapter":
        raise ValueError("Docker 配置必须加载镜像内 participant/adapter")
    if config.get("api_key", "") != "":
        raise ValueError("Docker Smoke 配置不得内置 API Key")

    for key in (
        "backend", "model_path", "model_id", "model_revision", "trust_remote_code",
        "device_map", "torch_dtype", "base_url", "api_model", "api_key",
        "request_timeout_seconds", "api_options", "openai_compatible", "prompt",
        "generation", "quantization", "performance_rounds",
    ):
        if config.get(key) != reference.get(key):
            raise ValueError(f"Docker 配置相对 E02.1 Smoke 基线发生漂移: {key}")
    docker_tokenizer = {key: value for key, value in tokenizer.items() if key != "local_path"}
    if docker_tokenizer != reference.get("tokenizer"):
        raise ValueError("Docker Tokenizer 配置相对 E02.1 Smoke 基线发生漂移")
    reference_context = reference.get("context", {})
    if any(context.get(key) != reference_context.get(key) for key in ("max_prompt_tokens", "serving_context_limit")):
        raise ValueError("Docker P/K 与 E02.1 Smoke 基线不一致")
    reference_adapter = reference.get("adapter", {})
    for key in (
        "enabled", "expected_base_weight_sha256", "expected_adapter_weight_sha256",
        "expected_adapter_config_sha256", "expected_server_model_id", "expected_rank",
        "expected_target_modules",
    ):
        if adapter.get(key) != reference_adapter.get(key):
            raise ValueError(f"Docker Adapter 配置相对 E02.1 Smoke 基线发生漂移: {key}")
    if vllm != reference.get("deployment", {}).get("vllm", {}):
        raise ValueError("Docker vLLM 参数相对 E02.1 Smoke 基线发生漂移")
    if adapter.get("expected_server_model_id") != config.get("api_model"):
        raise ValueError("Docker LoRA alias 与请求 model ID 不一致")
    if adapter.get("expected_rank") != 8 or set(adapter.get("expected_target_modules", [])) != EXPECTED_TARGET_MODULES:
        raise ValueError("Docker Adapter 配置与 Full B rank-8 七模块身份不一致")
    for key in (
        "expected_base_weight_sha256",
        "expected_adapter_weight_sha256",
        "expected_adapter_config_sha256",
    ):
        value = adapter.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"Docker Adapter 缺少有效冻结字段: {key}")

    return {
        "experiment_id": config.get("experiment_id"),
        "base_model_id": config.get("model_id"),
        "adapter_model_id": config.get("api_model"),
        "vllm_version": vllm["expected_version"],
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "serving_context_limit": serving_limit,
        "base_weight_sha256": adapter["expected_base_weight_sha256"],
        "adapter_weight_sha256": adapter["expected_adapter_weight_sha256"],
        "adapter_config_sha256": adapter["expected_adapter_config_sha256"],
    }


def check_static(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    config = load_config(repo_root)
    facts = validate_static_config(config, repo_root)
    dockerfile = (repo_root / "docker/e02/Dockerfile").read_text(encoding="utf-8")
    entrypoint = (repo_root / "docker/e02/entrypoint.sh").read_text(encoding="utf-8")
    if "FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04" not in dockerfile:
        raise ValueError("Dockerfile 必须固定 CUDA 12.8 runtime 基础镜像")
    if "ENV CONDA_DEFAULT_ENV=conda_env" not in dockerfile or "ENV VLLM_BIN=/opt/conda/envs/conda_env/bin/vllm" not in dockerfile:
        raise ValueError("Dockerfile 未在固定 conda_env 中运行 vLLM")
    if "HF_HUB_OFFLINE=1" not in entrypoint or "TRANSFORMERS_OFFLINE=1" not in entrypoint:
        raise ValueError("容器入口未强制启用模型离线模式")
    if "unset VLLM_API_KEY EVALUATION_API_KEY" not in entrypoint:
        raise ValueError("Docker 内部 loopback 服务不应继承外部 API Key")
    if 'mktemp "$RESULT_DIR/.docker-write-test.' not in entrypoint:
        raise ValueError("Docker 入口未在启动 vLLM 前检查结果目录可写性")
    if "bash /root/participant/start.sh \"$@\"" not in entrypoint:
        raise ValueError("容器入口必须把两个参数原样交给官方 start.sh")

    conda_lock = repo_root / "docker/e02/conda-linux-64.explicit.txt"
    pip_lock = repo_root / "docker/e02/requirements-linux-64.lock"
    if not conda_lock.is_file() or not pip_lock.is_file():
        raise FileNotFoundError("缺少 E02 Docker runtime lock 文件")
    conda_text = conda_lock.read_text(encoding="utf-8")
    pip_pins = _read_requirement_pins(pip_lock)
    if "@EXPLICIT" not in conda_text or not any("python-3.10.22-" in line for line in conda_text.splitlines()):
        raise ValueError("Conda lock 未固定到实测 Python 3.10.22 linux-64")
    expected_pins = {
        "vllm": "0.11.1",
        "torch": "2.9.0+cu128",
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "huggingface-hub": "0.36.2",
        "jsonschema": "4.26.0",
    }
    if any(pip_pins.get(name) != version for name, version in expected_pins.items()):
        raise ValueError("pip lock 与成功 E02 vLLM 环境的关键版本不一致")
    return {"status": "ready_for_asset_packaging", "facts": facts, "pip_lock_package_count": len(pip_pins)}


def _read_requirement_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version or "@" in version:
            raise ValueError(f"Runtime lock 只接受 name==version 固定项: {line}")
        pins[name.lower().replace("_", "-")] = version
    return pins


def _safe_relpath(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"资产相对路径越界: {relative}")
    return path


def _adapter_weight_paths(adapter_dir: Path) -> list[Path]:
    adapter_id, _ = adapter_identity(adapter_dir)
    result = []
    for item in adapter_id["files"]:
        relative = _safe_relpath(item["path"])
        result.append(relative)
    return result


def _copy_source_tree(repo_root: Path, stage: Path) -> None:
    for relative in COPY_FILES:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Docker context source missing: {relative}")
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in COPY_DIRS:
        source = repo_root / relative
        if not source.is_dir():
            raise FileNotFoundError(f"Docker context source missing: {relative}")
        shutil.copytree(source, stage / relative, symlinks=False)
    (stage / ".dockerignore").write_text(
        ".git\n.venv\noutputs\ndata\n.env\n**/__pycache__\n**/*.pyc\n",
        encoding="utf-8",
    )


def _git_commit(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _regular_files_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def prepare_context(
    *,
    base_model_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify frozen input identities and create one immutable Docker context."""
    repo_root = repo_root.resolve()
    output_dir = output_dir.expanduser().resolve()
    outputs_root = (repo_root / "outputs").resolve()
    if not output_dir.is_relative_to(outputs_root):
        raise ValueError("Docker context 必须写入 Git 忽略的仓库 outputs/ 路径")
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖现有 Docker context: {output_dir}")
    if output_dir == outputs_root:
        raise ValueError("Docker context 不能直接覆盖 outputs/ 根目录")

    config = load_config(repo_root)
    facts = validate_static_config(config, repo_root)
    base_model_dir = base_model_dir.expanduser().resolve()
    adapter_dir = adapter_dir.expanduser().resolve()
    adapter_cfg = config["adapter"]
    identities = validate_artifact_identity(
        base_path=base_model_dir,
        adapter_path=adapter_dir,
        model_id=config["model_id"],
        alias=config["api_model"],
        base_alias="model",
        max_lora_rank=adapter_cfg["expected_rank"],
        expected_base_sha256=adapter_cfg["expected_base_weight_sha256"],
        expected_adapter_sha256=adapter_cfg["expected_adapter_weight_sha256"],
        expected_adapter_config_sha256=adapter_cfg["expected_adapter_config_sha256"],
    )

    tokenizer_hashes = {
        name: sha256_file(base_model_dir / name)
        for name in config["tokenizer"]["files_sha256"]
        if (base_model_dir / name).is_file()
    }
    if tokenizer_hashes != config["tokenizer"]["files_sha256"]:
        raise ValueError("本地 Base Tokenizer 文件缺失或与冻结 SHA-256 不匹配")
    if sha256_file(repo_root / PROMPT_PATH) != config["prompt"]["sha256"]:
        raise ValueError("仓库 P0 文件与 Docker 配置冻结 SHA-256 不匹配")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _copy_source_tree(repo_root, temporary_dir)
        staged_base = temporary_dir / "participant/model"
        shutil.copytree(
            base_model_dir,
            staged_base,
            symlinks=False,
            ignore=shutil.ignore_patterns(".cache", "__pycache__"),
        )
        staged_adapter = temporary_dir / "participant/adapter"
        staged_adapter.mkdir(parents=True)
        shutil.copy2(adapter_dir / "adapter_config.json", staged_adapter / "adapter_config.json")
        for relative in _adapter_weight_paths(adapter_dir):
            source = adapter_dir / relative
            if not source.is_file():
                raise FileNotFoundError(f"Adapter 权重文件缺失: {relative}")
            shutil.copy2(source, staged_adapter / relative)

        staged_base_identity = base_weight_identity(staged_base)
        staged_adapter_identity, _ = adapter_identity(staged_adapter)
        if staged_base_identity["observed_weight_hash"] != identities["base"]["observed_weight_hash"]:
            raise RuntimeError("复制后的 Base 权重指纹变化，已取消 Docker context")
        if staged_adapter_identity["weight_sha256"] != identities["adapter"]["weight_sha256"]:
            raise RuntimeError("复制后的 Adapter 权重指纹变化，已取消 Docker context")
        if staged_adapter_identity["config_sha256"] != identities["adapter"]["config_sha256"]:
            raise RuntimeError("复制后的 Adapter 配置指纹变化，已取消 Docker context")

        code_files: dict[str, str] = {}
        for relative in COPY_FILES:
            code_files[relative.as_posix()] = sha256_file(temporary_dir / relative)
        for relative in COPY_DIRS:
            for path in sorted((temporary_dir / relative).rglob("*")):
                if path.is_file():
                    code_files[path.relative_to(temporary_dir).as_posix()] = sha256_file(path)
        manifest = {
            "schema_version": "e02_docker_package_v1",
            "status": "prepared_for_image_build",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _git_commit(repo_root),
            "configuration": facts,
            "runtime": {
                "python": "3.10.22",
                "vllm": "0.11.1",
                "torch": "2.9.0+cu128",
                "transformers": "4.57.6",
                "platform": "linux-64 / x86_64",
            },
            "base": {
                "weight_sha256": identities["base"]["observed_weight_hash"],
                "weight_files": identities["base"]["files"],
                "tokenizer_files_sha256": tokenizer_hashes,
            },
            "adapter": {
                "weight_sha256": identities["adapter"]["weight_sha256"],
                "config_sha256": identities["adapter"]["config_sha256"],
                "weight_files": identities["adapter"]["files"],
                "rank": identities["adapter"]["config_summary"]["r"],
                "target_modules": sorted(identities["adapter"]["config_summary"]["target_modules"]),
            },
            "prompt_sha256": config["prompt"]["sha256"],
            "source_files_sha256": code_files,
            "package_size_bytes_before_manifest": _regular_files_bytes(temporary_dir),
        }
        (temporary_dir / "package_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if any((temporary_dir / forbidden).exists() for forbidden in (".git", ".venv", "data", "outputs")):
            raise RuntimeError("Docker context 不得包含 Git、宿主环境、数据或 outputs 目录")
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    manifest_path = output_dir / "package_manifest.json"
    return {
        "status": "prepared_for_image_build",
        "context_dir": str(output_dir),
        "context_bytes": _regular_files_bytes(output_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "base_weight_sha256": identities["base"]["observed_weight_hash"],
        "adapter_weight_sha256": identities["adapter"]["weight_sha256"],
    }


def _read_jsonl(path: Path, *, exact_input: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            if exact_input and set(row) != {"id", "history"}:
                raise ValueError(f"{path}:{line_number}: input must contain only id and history")
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
            rows.append(row)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: empty rows or duplicate IDs")
    return rows


def verify_output(test_file: Path, result_dir: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Check Docker Smoke output contract and local artifact identity, without Gold."""
    config = load_config(repo_root)
    expected = validate_static_config(config, repo_root)
    input_rows = _read_jsonl(test_file, exact_input=True)
    submission_path = result_dir / "submission.jsonl"
    report_path = result_dir / "performance_report.json"
    if not submission_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("结果目录缺少 submission.jsonl 或 performance_report.json")
    predictions = _read_jsonl(submission_path)
    input_ids = [row["id"] for row in input_rows]
    prediction_ids = [row["id"] for row in predictions]
    if set(input_ids) != set(prediction_ids) or len(input_ids) != len(prediction_ids):
        raise ValueError("submission ID 集与输入 ID 集不一致")

    schema = load_json(repo_root / SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    schema_errors = []
    for index, row in enumerate(predictions):
        errors = list(validator.iter_errors(row))
        if errors:
            schema_errors.append({"row": index + 1, "message": errors[0].message})
    if schema_errors:
        raise ValueError(f"submission Schema 校验失败: {schema_errors[:3]}")

    report = load_json(report_path)
    if report.get("complete") is not True or report.get("submission_rows") != len(input_rows):
        raise ValueError("performance_report 未完整处理所有输入行")
    if report.get("required_rounds") != len(input_rows):
        raise ValueError("performance_report required_rounds 与 Docker Smoke 输入行数不一致")
    if report.get("submission_schema_validated") is not True:
        raise ValueError("performance_report 未确认 Runner Schema 校验")
    if report.get("backend") != "openai_compatible" or report.get("api_model") != expected["adapter_model_id"]:
        raise ValueError("performance_report 后端或请求模型 alias 不匹配")

    adapter_identity = report.get("adapter_identity", {})
    model_identity = report.get("model_weight_identity", {})
    runtime = report.get("api_server_runtime", {})
    if adapter_identity.get("weight_sha256") != expected["adapter_weight_sha256"]:
        raise ValueError("performance_report Adapter SHA-256 与冻结身份不一致")
    if adapter_identity.get("config_sha256") != expected["adapter_config_sha256"]:
        raise ValueError("performance_report Adapter config SHA-256 与冻结身份不一致")
    if model_identity.get("local_reference_weight_sha256") != expected["base_weight_sha256"]:
        raise ValueError("performance_report Base SHA-256 与冻结身份不一致")
    if runtime.get("vllm_version") != expected["vllm_version"]:
        raise ValueError("performance_report vLLM 版本与冻结身份不一致")

    return {
        "status": "passed_output_contract_and_local_identity",
        "input_rows": len(input_rows),
        "submission_rows": len(predictions),
        "schema_valid_rows": len(predictions),
        "runner_success_count": report.get("success_count"),
        "runner_parse_failure_count": report.get("parse_failure_count"),
        "runner_invalid_label_count": report.get("invalid_label_count"),
        "base_weight_sha256": expected["base_weight_sha256"],
        "adapter_weight_sha256": expected["adapter_weight_sha256"],
        "vllm_version": expected["vllm_version"],
        "server_weight_identity": adapter_identity.get("server_weight_identity", "unverified"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="CPU-only static Docker/config/lock preflight")
    prepare = subparsers.add_parser("prepare", help="hash and copy fixed local assets into ignored outputs/")
    prepare.add_argument("--base-model-dir", type=Path, default=Path("participant/model"))
    prepare.add_argument("--adapter-dir", type=Path, default=Path("outputs/training/E01-B-FULL-20261028-133805/adapter"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-output", help="validate a completed Docker smoke result without Gold")
    verify.add_argument("--test-file", type=Path, required=True)
    verify.add_argument("--result-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            result = check_static()
        elif args.command == "prepare":
            result = prepare_context(
                base_model_dir=args.base_model_dir,
                adapter_dir=args.adapter_dir,
                output_dir=args.output_dir,
            )
        else:
            result = verify_output(args.test_file, args.result_dir)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
