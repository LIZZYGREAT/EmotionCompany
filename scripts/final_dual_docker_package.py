#!/usr/bin/env python3
"""Build and verify the isolated E03 final dual-Adapter Docker package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_CONFIG = Path("configs/deployment/final_dual_v1_2.json")
DOCKER_CONFIG = Path("configs/deployment/final_dual_docker.json")
SCHEMA_PATH = Path("participant/templates/submission_schema.json")
PDF_PATH = Path("report/main.pdf")
CONDA_LOCK = Path("docker/e02/conda-linux-64.explicit.txt")
PIP_LOCK = Path("docker/e02/requirements-linux-64.lock")
DOCKERFILE = Path("docker/final_dual/Dockerfile")
ENTRYPOINT = Path("docker/final_dual/entrypoint.sh")
EXPECTED_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}
CONTAINER_BASE_PATH = "/root/participant/model"
RUNTIME_CODE_FILES = (
    Path("participant/start.sh"),
    Path("participant/run_inference.py"),
    Path("participant/final_dual_inference.py"),
    Path("scripts/e02_vllm_probe.py"),
    Path("docker/final_dual/entrypoint.sh"),
    Path("docker/final_dual/entrypoint_args.sh"),
    Path("src/inference/final_dual_protocol.py"),
    Path("src/inference/final_dual_runtime.py"),
    Path("src/inference/final_dual_vllm.py"),
)

COPY_FILES = (
    Path("participant/start.sh"),
    Path("participant/run_inference.py"),
    Path("participant/final_dual_inference.py"),
    Path("participant/configs/inference_config.json"),
    Path("scripts/e02_vllm_probe.py"),
    SCHEMA_PATH,
    DOCKER_CONFIG,
    Path("configs/prompts/p0.txt"),
    Path("configs/prompts/p_a8_v2_profile.txt"),
    Path("configs/prompts/p_a8_v2_emotion_profile.txt"),
    Path("configs/prompts/p_e03b.txt"),
    Path("src/data/__init__.py"),
    Path("src/data/e03_a8_sft.py"),
    Path("src/inference/__init__.py"),
    Path("src/inference/final_dual_protocol.py"),
    Path("src/inference/final_dual_runtime.py"),
    Path("src/inference/final_dual_vllm.py"),
    CONDA_LOCK,
    PIP_LOCK,
    DOCKERFILE,
    ENTRYPOINT,
    Path("docker/final_dual/entrypoint_args.sh"),
    PDF_PATH,
)
COPY_DIRS = (Path("src/shared"),)
MAX_IMAGE_BYTES = 50_000_000_000
CUDA_IMAGE = "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2934e1fa96152b14f78078bfbafd0f00f391df995dc6c641a720fce1202bb"
MINIFORGE_IMAGE = "condaforge/miniforge3:26.1.0-0@sha256:507f4365aba4825eb5541e00d57e515389053bcacaa46a83da4efaefce34874d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_asset_facts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report PDF is missing: {path}")
    size_bytes = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(5)
    if size_bytes <= 0 or header != b"%PDF-":
        raise ValueError(f"report PDF must be non-empty and begin with %PDF-: {path}")
    return {"path": PDF_PATH.as_posix(), "size_bytes": size_bytes, "sha256": sha256_file(path)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_repo_path(value: str | Path, repo_root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def _source_run_paths(config: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Path]:
    return {
        "model_lock": resolve_repo_path(config["model_lock_path"], repo_root),
    }


def _read_requirement_pins(path: Path) -> dict[str, str]:
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version or "@" in version:
            raise ValueError(f"runtime lock only accepts exact pins: {line}")
        pins[name.lower().replace("_", "-")] = version
    return pins


def _contains_host_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_host_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_host_path(item) for item in value)
    if isinstance(value, str):
        return bool(re.search(r"(?:^|[=:])(?:/home/|/mnt/[a-z]/|[A-Za-z]:\\)", value, flags=re.IGNORECASE))
    return False


def _sha_code_files(root: Path) -> dict[str, str]:
    facts = {path.as_posix(): sha256_file(root / path) for path in COPY_FILES}
    for directory in COPY_DIRS:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                facts[path.relative_to(root).as_posix()] = sha256_file(path)
    return facts


def _runtime_code_hashes(root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(root / path) for path in RUNTIME_CODE_FILES}



def _regular_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _git_head(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None



def validate_static_config(config: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Any]:
    source = load_json(repo_root / SOURCE_CONFIG)
    if config.get("version") != "E03-final-dual-docker-v3.0":
        raise ValueError("Docker config must declare the v3 portable runtime contract")
    if config.get("runtime_contract_version") != "E03-v3-portable-runtime":
        raise ValueError("Docker config must identify the portable inference entrypoint")
    if config.get("mode") != "dual_adapter" or config.get("backend") != "vllm":
        raise ValueError("Docker deployment config must use mode=dual_adapter and backend=vllm")
    if config.get("model_id") != source.get("model_id") or config.get("model_revision") != source.get("model_revision"):
        raise ValueError("Docker Base identity differs from the tested deployment config")
    if config.get("expected_base_weight_sha256") != source.get("expected_base_weight_sha256"):
        raise ValueError("Docker Base SHA-256 differs from the tested deployment config")
    if config.get("model_path") != "participant/model" or config.get("model_lock_path") != "model_lock.json":
        raise ValueError("Docker Base and lock paths must be container-relative")
    if config.get("tokenizer") != source.get("tokenizer"):
        raise ValueError("Docker Tokenizer lock differs from the tested deployment config")
    if config.get("prompts") != source.get("prompts"):
        raise ValueError("Docker Prompt paths or hashes differ from the tested deployment config")
    if config.get("context") != source.get("context") or config.get("generation") != source.get("generation"):
        raise ValueError("Docker generation/context protocol differs from the tested deployment config")
    if config.get("deployment", {}).get("vllm") != source.get("deployment", {}).get("vllm"):
        raise ValueError("Docker vLLM settings differ from the tested deployment config")
    if config.get("quantization", {}).get("load_in_4bit") is not False:
        raise ValueError("Docker vLLM runtime must use the tested BF16 non-quantized Base")
    if _contains_host_path(config):
        raise ValueError("Docker config contains a host-specific absolute path")

    docker_adapters = config.get("adapters", {})
    expected_er_container_config_sha = docker_adapters.get("emotion_response", {}).get("expected_config_sha256")
    if not isinstance(expected_er_container_config_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_er_container_config_sha):
        raise ValueError("Docker emotion_response Adapter config SHA-256 must be recorded")
    expected_adapters = {
        "profile": {
            "candidate": "A8-P4504",
            "path": "participant/adapters/profile",
            "expected_weight_sha256": source["adapters"]["profile"]["expected_weight_sha256"],
            "expected_config_sha256": source["adapters"]["profile"]["expected_config_sha256"],
            "server_model_id": "profile",
        },
        "emotion_response": {
            "candidate": "A8-ER",
            "path": "participant/adapters/emotion-response",
            "expected_weight_sha256": source["adapters"]["emotion_response"]["expected_weight_sha256"],
            "expected_config_sha256": expected_er_container_config_sha,
            "source_expected_config_sha256": source["adapters"]["emotion_response"]["expected_config_sha256"],
            "server_model_id": "emotion-response",
        },
    }
    adapters = config.get("adapters", {})
    for name, expected in expected_adapters.items():
        record = adapters.get(name, {})
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Docker {name} Adapter identity/path differs from the locked assets")
    vllm = config["deployment"]["vllm"]
    if vllm.get("expected_version") != "0.11.1" or vllm.get("dtype") not in {
        "auto", "bfloat16", "float16", "float32",
    }:
        raise ValueError("Docker runtime must use vLLM 0.11.1 with an environment-compatible dtype")
    if (config["context"]["max_prompt_tokens"] + config["generation"]["emotion_response"]["max_new_tokens"]
            > vllm["max_model_len"]):
        raise ValueError("Docker vLLM context is smaller than the frozen prompt+generation budget")
    if vllm.get("max_num_batched_tokens", 0) < config["context"]["max_prompt_tokens"] + config["generation"]["emotion_response"]["max_new_tokens"]:
        raise ValueError("Docker max_num_batched_tokens is smaller than P+G")
    if vllm.get("host") != "127.0.0.1" or vllm.get("port") != 8000:
        raise ValueError("Docker vLLM must bind only to container loopback:8000")
    if vllm.get("max_loras") != 1 or vllm.get("max_cpu_loras") < 2:
        raise ValueError("Docker must register both LoRAs with one active LoRA per serial request")
    return {
        "base_model_id": config["model_id"],
        "base_weight_sha256": config["expected_base_weight_sha256"],
        "adapter_weight_sha256": {key: value["expected_weight_sha256"] for key, value in adapters.items()},
        "adapter_config_sha256": {key: value["expected_config_sha256"] for key, value in adapters.items()},
        "model_aliases": ["qwen-base", "profile", "emotion-response"],
        "vllm_version": vllm["expected_version"],
        "runtime": {"python": "3.10.22", "torch": "2.9.0+cu128", "transformers": "4.57.6"},
        "build_images": {"cuda": CUDA_IMAGE, "miniforge": MINIFORGE_IMAGE},
    }


def check_static(repo_root: Path = ROOT) -> dict[str, Any]:
    config = load_json(repo_root / DOCKER_CONFIG)
    facts = validate_static_config(config, repo_root)
    dockerfile = (repo_root / DOCKERFILE).read_text(encoding="utf-8")
    entrypoint = (repo_root / ENTRYPOINT).read_text(encoding="utf-8")
    if f"FROM {CUDA_IMAGE}" not in dockerfile:
        raise ValueError("Dockerfile must pin the validated CUDA 12.8 runtime base digest")
    if f"FROM {MINIFORGE_IMAGE} AS miniforge" not in dockerfile or "COPY --from=miniforge /opt/conda /opt/conda" not in dockerfile:
        raise ValueError("Dockerfile must use the pinned conda-forge Miniforge 26.1.0-0 image")
    if "ENV CONDA_DEFAULT_ENV=conda_env" not in dockerfile or "ENV VLLM_BIN=/opt/conda/envs/conda_env/bin/vllm" not in dockerfile:
        raise ValueError("Dockerfile does not use the fixed conda_env vLLM runtime")
    if "HF_HUB_OFFLINE" not in entrypoint or "TRANSFORMERS_OFFLINE" not in entrypoint or "EVALUATION_BACKEND" not in entrypoint:
        raise ValueError("Docker entrypoint must force offline vLLM inference")
    if 'exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"' not in entrypoint:
        raise ValueError("Docker entrypoint must pass the two official arguments to start.sh")
    if "nvidia-smi" in entrypoint or "exec sleep infinity" not in entrypoint:
        raise ValueError("Docker entrypoint must not require nvidia-smi and must support idle containers")
    if "--network none" not in (repo_root / Path("docker/final_dual/README.md")).read_text(encoding="utf-8"):
        raise ValueError("Docker README must document network-isolated inference")
    conda_text = (repo_root / CONDA_LOCK).read_text(encoding="utf-8")
    pins = _read_requirement_pins(repo_root / PIP_LOCK)
    if "@EXPLICIT" not in conda_text or not any("python-3.10.22-" in line for line in conda_text.splitlines()):
        raise ValueError("E02 Conda lock does not pin linux-64 Python 3.10.22")
    required_pins = {
        "vllm": "0.11.1", "torch": "2.9.0+cu128", "transformers": "4.57.6",
        "tokenizers": "0.22.2", "huggingface-hub": "0.36.2", "jsonschema": "4.26.0",
    }
    if any(pins.get(name) != version for name, version in required_pins.items()):
        raise ValueError("E02 pip lock differs from the validated final dual-Adapter runtime")
    for relative in COPY_FILES:
        if not (repo_root / relative).is_file():
            raise FileNotFoundError(f"Docker package source is missing: {relative}")
    for relative in COPY_DIRS:
        if not (repo_root / relative).is_dir():
            raise FileNotFoundError(f"Docker package source directory is missing: {relative}")
    report_pdf = pdf_asset_facts(repo_root / PDF_PATH)
    return {
        "status": "ready_for_asset_packaging",
        "facts": facts,
        "package_preflight": {"status": "passed", "scope": "static config and build inputs"},
        "runtime_lock_package_count": len(pins),
        "image_size_limit_bytes": MAX_IMAGE_BYTES,
        "report_pdf": report_pdf,
    }


def _safe_copy_files(repo_root: Path, stage: Path) -> dict[str, str]:
    for relative in COPY_FILES:
        source = repo_root / relative
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative in COPY_DIRS:
        shutil.copytree(
            repo_root / relative,
            stage / relative,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return _sha_code_files(stage)


def _copy_adapter(source: Path, destination: Path, *, rewrite_base: bool) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    config_path = source / "adapter_config.json"
    weight_path = source / "adapter_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file() or weight_path.stat().st_size <= 0:
        raise FileNotFoundError(f"frozen Adapter must contain one config and one safetensors file: {source}")
    config = load_json(config_path)
    source_config_sha = sha256_file(config_path)
    if rewrite_base:
        config["base_model_name_or_path"] = CONTAINER_BASE_PATH
        payload = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        (destination / "adapter_config.json").write_bytes(payload)
    else:
        shutil.copy2(config_path, destination / "adapter_config.json")
    shutil.copy2(weight_path, destination / weight_path.name)
    return {
        "path": destination.relative_to(destination.parents[1]).as_posix(),
        "weight_sha256": sha256_file(destination / weight_path.name),
        "config_sha256": sha256_file(destination / "adapter_config.json"),
        "source_config_sha256": source_config_sha,
        "base_path_rewritten": rewrite_base,
        "base_model_name_or_path": config.get("base_model_name_or_path"),
    }


def _container_model_lock(
    *, config: dict[str, Any], config_sha256: str, base_identity: dict[str, Any],
    tokenizer_identity: dict[str, Any], prompt_records: dict[str, Any], adapters: dict[str, Any],
    runtime_code_identity: dict[str, Any],
) -> dict[str, Any]:
    base_observed = {
        key: base_identity[key]
        for key in ("identity_status", "fingerprint_status", "revision_status", "observed_weight_hash", "aggregate_sha256", "files")
        if key in base_identity
    }
    base_observed["local_path"] = "participant/model"
    tokenizer = {
        key: tokenizer_identity[key]
        for key in ("model_id", "revision", "signature_sha256", "template_sha256", "eos_token_id", "pad_token_id", "tokenizer_files")
        if key in tokenizer_identity
    }
    deployments = {
        "profile": {
            "path": "participant/adapters/profile",
            "weight_sha256": adapters["profile"]["weight_sha256"],
            "config_sha256": adapters["profile"]["config_sha256"],
            "base_path_rewritten": False,
            "base_model_name_or_path": adapters["profile"]["base_model_name_or_path"],
        },
        "emotion_response": {
            "path": "participant/adapters/emotion-response",
            "weight_sha256": adapters["emotion_response"]["weight_sha256"],
            "config_sha256": adapters["emotion_response"]["config_sha256"],
            "source_config_sha256": adapters["emotion_response"]["source_config_sha256"],
            "base_path_rewritten": True,
            "base_model_name_or_path": CONTAINER_BASE_PATH,
        },
    }
    return {
        "version": "final-dual-container-model-lock-v2",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_config": {"path": DOCKER_CONFIG.as_posix(), "sha256": config_sha256},
        "runtime_code_identity": runtime_code_identity,
        "base": {
            "model_id": config["model_id"],
            "revision": config["model_revision"],
            "expected_weight_sha256": config["expected_base_weight_sha256"],
            "observed": base_observed,
        },
        "tokenizer": tokenizer,
        "prompts": prompt_records,
        "deployment_adapters": deployments,
    }


def prepare_context(*, output_dir: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.expanduser().resolve()
    outputs_root = (repo_root / "outputs").resolve()
    if not output_dir.is_relative_to(outputs_root) or output_dir == outputs_root:
        raise ValueError("Docker context must be placed below the ignored outputs/ directory")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Docker context: {output_dir}")
    static = check_static(repo_root)
    if static.get("status") != "ready_for_asset_packaging":
        raise RuntimeError("refusing Docker packaging until static package checks pass")
    config = load_json(repo_root / DOCKER_CONFIG)
    source_config = load_json(repo_root / SOURCE_CONFIG)
    paths = _source_run_paths(source_config, repo_root)
    source_lock_path = paths["model_lock"]
    source_lock = load_json(source_lock_path)

    base_path = resolve_repo_path(source_config["model_path"], repo_root)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(base_path), local_files_only=True, trust_remote_code=False)
    verified = {"tokenizer": source_lock.get("tokenizer", config.get("tokenizer", {}))}
    for name, prompt_record in config["prompts"].items():
        prompt_path = resolve_repo_path(prompt_record["path"], repo_root)
        if sha256_file(prompt_path) != prompt_record["sha256"]:
            raise ValueError(f"Docker Prompt identity mismatch: {name}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        copied_source_hashes = _safe_copy_files(repo_root, temporary_dir)
        staged_base = temporary_dir / "participant/model"
        shutil.copytree(
            base_path, staged_base, symlinks=False,
            ignore=shutil.ignore_patterns(".cache", "__pycache__"),
        )
        staged_adapters = temporary_dir / "participant/adapters"
        staged_adapters.mkdir(parents=True)
        deployment = source_lock["deployment_adapters"]
        profile_source = resolve_repo_path(deployment["profile"]["path"], repo_root)
        er_source = resolve_repo_path(deployment["emotion_response"]["path"], repo_root)
        profile_record = _copy_adapter(profile_source, staged_adapters / "profile", rewrite_base=False)
        er_record = _copy_adapter(er_source, staged_adapters / "emotion-response", rewrite_base=True)
        adapters = {"profile": profile_record, "emotion_response": er_record}
        for key in ("profile", "emotion_response"):
            expected = config["adapters"][key]
            if adapters[key]["weight_sha256"] != expected["expected_weight_sha256"]:
                raise ValueError(f"copied {key} Adapter weight SHA-256 mismatch")
            if adapters[key]["config_sha256"] != expected["expected_config_sha256"]:
                raise ValueError(f"copied {key} Adapter config SHA-256 mismatch")

        from scripts.e02_vllm_probe import base_weight_identity
        staged_base_identity = base_weight_identity(staged_base)
        if staged_base_identity["observed_weight_hash"] != config["expected_base_weight_sha256"]:
            raise ValueError("copied Base weights differ from the locked fingerprint")
        tokenizer_hashes = {
            name: sha256_file(staged_base / name)
            for name in config["tokenizer"]["files_sha256"]
            if (staged_base / name).is_file()
        }
        if tokenizer_hashes != config["tokenizer"]["files_sha256"]:
            raise ValueError("copied Base Tokenizer files differ from the frozen hashes")

        docker_config_path = temporary_dir / DOCKER_CONFIG
        docker_config_sha = sha256_file(docker_config_path)
        prompt_records = {
            name: {"path": record["path"], "sha256": sha256_file(temporary_dir / record["path"])}
            for name, record in config["prompts"].items()
        }
        source_runtime_hashes = _runtime_code_hashes(repo_root)
        container_runtime_hashes = _runtime_code_hashes(temporary_dir)
        if source_runtime_hashes != container_runtime_hashes:
            raise ValueError("copied runtime code differs from the frozen source code hashes")
        runtime_code_identity = {
            "version": config["runtime_contract_version"],
            "git_head": _git_head(repo_root),
            "file_sha256": container_runtime_hashes,
        }
        container_lock = _container_model_lock(
            config=config,
            config_sha256=docker_config_sha,
            base_identity=staged_base_identity,
            tokenizer_identity=verified["tokenizer"],
            prompt_records=prompt_records,
            adapters=adapters,
            runtime_code_identity=runtime_code_identity,
        )
        lock_path = temporary_dir / "model_lock.json"
        lock_path.write_text(json.dumps(container_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        (temporary_dir / ".dockerignore").write_text(
            ".git\n.venv\noutputs\ndata\n.env\n**/__pycache__\n**/*.pyc\n",
            encoding="utf-8",
        )
        forbidden = (".git", ".venv", "outputs", "data", "checkpoints", "runs")
        if any((temporary_dir / path).exists() for path in forbidden):
            raise RuntimeError("Docker context contains a forbidden data, cache, Git, or training path")
        for path in temporary_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin"}:
                if path.name != "pytorch_model.bin":
                    raise RuntimeError(f"Docker context includes an unexpected checkpoint file: {path.relative_to(temporary_dir)}")
        code_hashes = _sha_code_files(temporary_dir)
        manifest = {
            "schema_version": "final_dual_docker_package_v2",
            "runtime_contract_version": config["runtime_contract_version"],
            "status": "prepared_for_image_build",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _git_head(repo_root),
            "source_model_lock_sha256": sha256_file(source_lock_path),
            "container_model_lock_sha256": sha256_file(lock_path),
            "runtime_code_identity": runtime_code_identity,
            "deployment_config": {"path": DOCKER_CONFIG.as_posix(), "sha256": docker_config_sha},
            "runtime": {"python": "3.10.22", "torch": "2.9.0+cu128", "vllm": "0.11.1", "transformers": "4.57.6", "cuda": "12.8", "conda": "26.1.0"},
            "build_images": {"cuda": CUDA_IMAGE, "miniforge": MINIFORGE_IMAGE},
            "base": {
                "model_id": config["model_id"],
                "revision": config["model_revision"],
                "weight_sha256": staged_base_identity["observed_weight_hash"],
                "weight_files": staged_base_identity["files"],
                "tokenizer_files_sha256": tokenizer_hashes,
                "tokenizer_signature_sha256": verified["tokenizer"]["signature_sha256"],
                "chat_template_sha256": verified["tokenizer"]["template_sha256"],
            },
            "adapters": {
                key: {
                    "path": "participant/adapters/profile" if key == "profile" else "participant/adapters/emotion-response",
                    "candidate": config["adapters"][key]["candidate"],
                    "weight_sha256": value["weight_sha256"],
                    "config_sha256": value["config_sha256"],
                    "source_config_sha256": value["source_config_sha256"],
                    "base_path_rewritten": value["base_path_rewritten"],
                    "rank": 8,
                    "target_modules": sorted(EXPECTED_MODULES),
                }
                for key, value in adapters.items()
            },
            "prompts": prompt_records,
            "package_preflight": static["package_preflight"],
            "source_file_sha256": copied_source_hashes,
            "container_file_sha256": code_hashes,
            "report_pdf": pdf_asset_facts(temporary_dir / PDF_PATH),
            "context_regular_file_bytes_before_manifest": _regular_file_bytes(temporary_dir),
        }
        manifest_path = temporary_dir / "package_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if _contains_host_path(manifest):
            raise RuntimeError("package manifest contains a host-specific absolute path")
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    package_manifest_path = output_dir / "package_manifest.json"
    return {
        "status": "prepared_for_image_build",
        "context_dir": str(output_dir),
        "context_regular_file_bytes": _regular_file_bytes(output_dir),
        "package_manifest_sha256": sha256_file(package_manifest_path),
        "model_lock_sha256": sha256_file(output_dir / "model_lock.json"),
        "base_weight_sha256": staged_base_identity["observed_weight_hash"],
        "adapter_weight_sha256": {key: value["weight_sha256"] for key, value in adapters.items()},
    }


def _read_jsonl(path: Path, *, exact_input: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if exact_input and ("id" not in row or "history" not in row):
                raise ValueError(f"{path}:{line_number}: input must provide id and history")
            if not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_number}: id must be a string")
            if exact_input and not isinstance(row.get("history"), list):
                raise ValueError(f"{path}:{line_number}: history must be an array")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty rows")
    return rows


def _report_output_file(report: dict[str, Any], result_dir: Path, name: str) -> Path:
    record = report.get("outputs", {}).get(name)
    value = record.get("path") if isinstance(record, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"performance_report does not name {name}")
    candidate = Path(value)
    resolved = (result_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(result_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"performance_report {name} path escapes RESULT_DIR") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Docker result missing {name}: {resolved}")
    return resolved



def verify_output(
    *, test_file: Path, result_dir: Path,
) -> dict[str, Any]:
    """Check both official files, row counts, and exact ID order."""
    submission_path = result_dir / "submission.jsonl"
    report_path = result_dir / "performance_report.json"
    for path in (submission_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"Docker result missing required output: {path.name}")

    inputs = _read_jsonl(test_file, exact_input=True)
    predictions = _read_jsonl(submission_path)
    input_ids = [row["id"] for row in inputs]
    output_ids = [row["id"] for row in predictions]
    row_count = len(inputs)
    if len(predictions) != row_count or output_ids != input_ids:
        raise ValueError("submission rows must preserve every input ID in order")
    schema = load_json(ROOT / SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    for row in predictions:
        errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
        if errors:
            raise ValueError(
                f"submission row does not satisfy the official Schema: "
                f"{row.get('id')}: {errors[0].message}"
            )
    load_json(report_path)

    return {
        "status": "passed_interface_output",
        "input_rows": row_count,
        "submission_rows": len(predictions),
        "ids_preserved_in_order": True,
        "submission_schema_valid": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="check the Docker runtime config and build inputs")
    prepare = sub.add_parser("prepare", help="create a verified, white-list Docker context under outputs/")
    prepare.add_argument("--config", type=Path, default=ROOT / DOCKER_CONFIG)
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = sub.add_parser("verify-output", help="check official files, row counts, and ID order")
    verify.add_argument("--test-file", type=Path, required=True)
    verify.add_argument("--result-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            result = check_static()
        elif args.command == "prepare":
            config_path = args.config.expanduser()
            if not config_path.is_absolute():
                config_path = ROOT / config_path
            if config_path.resolve() != (ROOT / DOCKER_CONFIG).resolve():
                raise ValueError("prepare only accepts the tracked final dual-Adapter Docker config")
            result = prepare_context(output_dir=args.output_dir)
        else:
            result = verify_output(
                test_file=args.test_file.expanduser().resolve(),
                result_dir=args.result_dir.expanduser().resolve(),
            )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
