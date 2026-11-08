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
SOURCE_CONFIG = Path("configs/deployment/final_dual.yaml")
DOCKER_CONFIG = Path("configs/deployment/final_dual_docker.json")
SCHEMA_PATH = Path("participant/templates/submission_schema.json")
CONDA_LOCK = Path("docker/e02/conda-linux-64.explicit.txt")
PIP_LOCK = Path("docker/e02/requirements-linux-64.lock")
DOCKERFILE = Path("docker/final_dual/Dockerfile")
ENTRYPOINT = Path("docker/final_dual/entrypoint.sh")
EXPECTED_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}
OFFICIAL_ID = re.compile(r"^test_[0-9]{6}_t[0-9]+$")
CONTAINER_BASE_PATH = "/root/participant/model"
RUNTIME_CODE_FILES = (
    Path("participant/start.sh"),
    Path("participant/final_dual_inference.py"),
    Path("participant/run_inference.py"),
    Path("src/inference/final_dual_protocol.py"),
    Path("src/inference/final_dual_runtime.py"),
    Path("src/inference/final_dual_vllm.py"),
)

COPY_FILES = (
    Path("participant/start.sh"),
    Path("participant/final_dual_inference.py"),
    Path("participant/run_inference.py"),
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
    Path("scripts/e02_vllm_probe.py"),
    CONDA_LOCK,
    PIP_LOCK,
    DOCKERFILE,
    ENTRYPOINT,
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_repo_path(value: str | Path, repo_root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def _source_run_paths(config: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Path]:
    run_id = config["run_id"]
    run_root = repo_root / "outputs/final_delivery" / run_id
    official_root = repo_root / "outputs/evaluation/final_dual" / f"{run_id}-official-start-vllm20"
    return {
        "run_root": run_root,
        "model_lock": resolve_repo_path(config["model_lock_path"], repo_root),
        "transformers_manifest": run_root / "transformers_smoke20/manifest.json",
        "input_manifest": run_root / "transformers_smoke20/input/manifest.json",
        "vllm_manifest": run_root / "vllm_smoke/manifest.json",
        "official_root": official_root,
        "official_report": official_root / "performance_report.json",
        "official_submission": official_root / "submission.jsonl",
        "official_raw": official_root / "raw_outputs.jsonl",
        "official_code_identity": official_root / "code_identity.json",
        "smoke_input": run_root / "transformers_smoke20/input/smoke20_input.jsonl",
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


def _rewritten_er_config_bytes(source_path: Path) -> bytes:
    adapter_config = load_json(source_path)
    adapter_config["base_model_name_or_path"] = CONTAINER_BASE_PATH
    return (json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


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


def verify_previous_gates(config: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Any]:
    source_config = load_json(repo_root / SOURCE_CONFIG)
    paths = _source_run_paths(source_config, repo_root)
    required = (
        paths["model_lock"], paths["transformers_manifest"], paths["input_manifest"],
        paths["vllm_manifest"], paths["official_report"], paths["official_submission"],
        paths["official_raw"], paths["official_code_identity"], paths["smoke_input"],
        paths["official_root"] / "run_progress.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prior gate evidence is incomplete: {missing}")
    source_config_path = repo_root / SOURCE_CONFIG
    source_lock = load_json(paths["model_lock"])
    if source_lock.get("deployment_config", {}).get("path") != SOURCE_CONFIG.as_posix():
        raise ValueError("source model lock points to a different deployment config")
    if source_lock.get("deployment_config", {}).get("sha256") != sha256_file(source_config_path):
        raise ValueError("source deployment config changed since asset preflight")

    input_manifest = load_json(paths["input_manifest"])
    frozen_input_sha = input_manifest.get("input", {}).get("sha256")
    if frozen_input_sha != sha256_file(paths["smoke_input"]):
        raise ValueError("frozen 20-row input hash differs from its manifest")
    transformers = load_json(paths["transformers_manifest"])
    vllm = load_json(paths["vllm_manifest"])
    official = load_json(paths["official_report"])
    official_progress = load_json(paths["official_root"] / "run_progress.json")
    code_identity = load_json(paths["official_code_identity"])
    if transformers.get("status") != "passed" or transformers.get("input_sha256") != frozen_input_sha:
        raise ValueError("Transformers gate is not passed for the Docker Smoke input")
    vllm_stages = {stage.get("name"): stage for stage in vllm.get("stages", [])}
    if vllm.get("status") != "passed" or any(vllm_stages.get(name, {}).get("status") != "passed" for name in ("smoke10", "smoke20")):
        raise ValueError("vLLM 10/20-row gate is not passed")
    if vllm.get("input", {}).get("sha256") != frozen_input_sha:
        raise ValueError("vLLM Smoke input identity differs from the Docker Smoke input")
    if (official.get("status") != "passed" or official_progress.get("test_file_sha256") != frozen_input_sha
            or official_progress.get("sample_count") != 20 or official_progress.get("status") != "complete"):
        raise ValueError("official start.sh gate is not passed on the Docker Smoke input")
    if official.get("submission_schema_validated_count") != 20 or official.get("submission_rows") != 20:
        raise ValueError("official start.sh gate does not contain 20 validated rows")
    if code_identity.get("execution_code_commit") != official.get("git_head") and not code_identity.get("code_files_sha256"):
        raise ValueError("official execution code identity supplement is incomplete")
    historical_evidence = {
        "status": "passed",
        "scope": "immutable v1 asset and historical prompt/protocol evidence",
        "runtime_behavior_is_current_v1_1_evidence": False,
        "run_id": source_config["run_id"],
        "input_sha256": frozen_input_sha,
        "model_lock_sha256": sha256_file(paths["model_lock"]),
        "transformers_status": transformers.get("status"),
        "vllm_status": vllm.get("status"),
        "historical_official_status": official.get("status"),
        "source_config_sha256": sha256_file(source_config_path),
        "source_git_head": code_identity.get("execution_code_commit"),
    }

    runtime_validation_path = repo_root / "outputs/final_delivery" / config["run_id"] / "runtime_validation.json"
    current_runtime_hashes = _sha_code_files(repo_root)
    runtime_validation: dict[str, Any] = {
        "status": "pending",
        "manifest_path": runtime_validation_path.relative_to(repo_root).as_posix(),
        "reason": "new v1.1 host smoke/stress and pre-build container evidence are required",
    }
    if runtime_validation_path.is_file():
        evidence = load_json(runtime_validation_path)
        if evidence.get("status") != "passed":
            raise ValueError("new v1.1 runtime validation manifest is not passed")
        if evidence.get("container_runtime_file_sha256") != current_runtime_hashes:
            raise ValueError("new v1.1 evidence code hashes differ from the current Docker runtime files")
        if evidence.get("docker_verifier_sha256") != sha256_file(repo_root / "scripts/final_dual_docker_package.py"):
            raise ValueError("new v1.1 evidence Docker verifier hash differs from the current verifier")
        if evidence.get("docker_config_sha256") != sha256_file(repo_root / DOCKER_CONFIG):
            raise ValueError("new v1.1 evidence Docker config hash differs from the current package config")
        host = evidence.get("host_validation", {})
        smoke_runs = host.get("smoke_runs", [])
        stress_run = host.get("stress_run")
        if host.get("status") != "passed" or len(smoke_runs) != 3 or not isinstance(stress_run, dict):
            raise ValueError("new v1.1 host evidence must include three 20-row runs and one fixed 100-row run")
        expected = validate_static_config(config, repo_root)
        host_config_path = repo_root / host.get("config_path", "")
        host_lock_path = repo_root / host.get("model_lock_path", "")
        if not host_config_path.is_file() or not host_lock_path.is_file():
            raise FileNotFoundError("new v1.1 host deployment config or model lock is missing")
        if sha256_file(host_config_path) != host.get("config_sha256") or sha256_file(host_lock_path) != host.get("model_lock_sha256"):
            raise ValueError("new v1.1 host deployment config/model lock identity differs from its evidence manifest")
        for record, expected_rows in [*((item, 20) for item in smoke_runs), (stress_run, 100)]:
            if record.get("samples") != expected_rows or record.get("status") != "passed":
                raise ValueError(f"new v1.1 host evidence is not passed for its {expected_rows}-row run")
            report_path = repo_root / record.get("report_path", "")
            if not report_path.is_file():
                raise FileNotFoundError(f"new v1.1 host report is missing: {report_path}")
            report = load_json(report_path)
            if report.get("status") != "passed" or report.get("complete") is not True or report.get("samples") != expected_rows:
                raise ValueError(f"new v1.1 host report is incomplete or failed: {report_path}")
            if report.get("configuration", {}).get("sha256") != host.get("config_sha256") or report.get("configuration", {}).get("model_lock_sha256") != host.get("model_lock_sha256"):
                raise ValueError(f"new v1.1 host report config/lock identity differs: {report_path}")
            if report.get("runtime_code_identity", {}).get("file_sha256") != _runtime_code_hashes(repo_root):
                raise ValueError(f"new v1.1 host runtime code hashes differ from the frozen source files: {report_path}")
            if report.get("model_weight_identity", {}).get("observed_weight_hash") != expected["base_weight_sha256"]:
                raise ValueError("new v1.1 host Base identity differs from the immutable asset lock")
            for route, identity_key in (("profile", "profile"), ("emotion_response", "emotion_response")):
                if report.get("adapter_identities", {}).get(identity_key, {}).get("weight_sha256") != expected["adapter_weight_sha256"][route]:
                    raise ValueError(f"new v1.1 host {route} Adapter identity differs from the immutable asset lock")
            if any(report.get("route_failure_count", {}).get(route, 0) for route in ("profile", "emotion_response")):
                raise ValueError(f"new v1.1 host report contains route errors: {report_path}")
            exact = report.get("prompt_token_ids_exact_match_count", {})
            denominator = report.get("prompt_token_ids_check_denominator", {})
            if any(exact.get(route) != denominator.get(route) for route in ("profile", "emotion_response")):
                raise ValueError(f"new v1.1 host report contains a Prompt Token ID mismatch: {report_path}")
            retry_count = report.get("retry_attempts", {}).get("emotion_response", 0)
            if report.get("server", {}).get("requested_alias_count") != {
                "profile": expected_rows,
                "emotion-response": expected_rows + retry_count,
            }:
                raise ValueError(f"new v1.1 host alias counts differ from samples plus retries: {report_path}")
        stress_report = load_json(repo_root / stress_run["report_path"])
        if stress_report.get("unresolved_cap_hits", {}).get("emotion_response", 0) > 2:
            raise ValueError("new v1.1 fixed 100-row host run exceeds the unresolved-cap threshold")
        if stress_report.get("emotion_response_severe_anomaly_count", 0) != 0 or stress_report.get("emotion_response_outcome_count", {}).get("invalid_output", 0) != 0:
            raise ValueError("new v1.1 fixed 100-row host run contains invalid or severely repeated responses")
        overlay = evidence.get("old_image_code_overlay", {})
        overlay_runs = overlay.get("runs", [])
        allowed_overlay_statuses = {"passed", "passed_equivalent_host"}
        if overlay.get("status") not in allowed_overlay_statuses or len(overlay_runs) < 2:
            raise ValueError("new v1.1 evidence must include two successful old-image code-overlay runs")
        if overlay.get("status") == "passed_equivalent_host" and not overlay.get("reason"):
            raise ValueError("host-equivalent Gate 2 evidence must explain why old-image overlay was unavailable")
        if any(item.get("container_runtime_file_sha256") != current_runtime_hashes for item in overlay_runs):
            raise ValueError("old-image overlay code hashes differ from the new Docker runtime files")
        if overlay.get("status") == "passed":
            for item in overlay_runs:
                overlay_report_path = repo_root / item.get("report_path", "")
                if not overlay_report_path.is_file():
                    raise FileNotFoundError(f"old-image overlay report is missing: {overlay_report_path}")
                overlay_report = load_json(overlay_report_path)
                if overlay_report.get("status") != "passed" or overlay_report.get("complete") is not True or overlay_report.get("samples") != 20:
                    raise ValueError(f"old-image overlay run is incomplete or failed: {overlay_report_path}")
        runtime_validation = {
            "status": "passed",
            "manifest_path": runtime_validation_path.relative_to(repo_root).as_posix(),
            "manifest_sha256": sha256_file(runtime_validation_path),
            "host_smoke_runs": len(smoke_runs),
            "host_stress_rows": 100,
            "old_image_code_overlay_runs": len(overlay_runs),
            "container_runtime_file_sha256": current_runtime_hashes,
        }
    return {
        "status": "passed" if runtime_validation["status"] == "passed" else "pending_new_v1_1_runtime_evidence",
        "historical_asset_protocol_evidence": historical_evidence,
        "new_v1_1_runtime_validation": runtime_validation,
    }


def validate_static_config(config: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Any]:
    source = load_json(repo_root / SOURCE_CONFIG)
    if config.get("version") != "E03-final-dual-docker-v1.1":
        raise ValueError("Docker config must declare the v1.1 bounded length-recovery behavior")
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
    runtime_behavior = config.get("runtime_behavior", {})
    if runtime_behavior != {
        "version": "E03-v1.1-length-recovery",
        "max_emotion_response_retries": 1,
        "retry_only_on_finish_reason": "length",
    }:
        raise ValueError("Docker config must lock one ER retry on finish_reason=length")
    if config.get("deployment", {}).get("vllm") != source.get("deployment", {}).get("vllm"):
        raise ValueError("Docker vLLM settings differ from the tested deployment config")
    if config.get("quantization", {}).get("load_in_4bit") is not False:
        raise ValueError("Docker vLLM runtime must use the tested BF16 non-quantized Base")
    if _contains_host_path(config):
        raise ValueError("Docker config contains a host-specific absolute path")

    source_lock_path = resolve_repo_path(source["model_lock_path"], repo_root)
    source_lock = load_json(source_lock_path)
    expected_er_container_config_sha = sha256_bytes(_rewritten_er_config_bytes(
        resolve_repo_path(source_lock["deployment_adapters"]["emotion_response"]["path"], repo_root) / "adapter_config.json"
    ))
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
    if vllm.get("expected_version") != "0.11.1" or vllm.get("dtype") != "bfloat16":
        raise ValueError("Docker runtime must use vLLM 0.11.1 BF16")
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
    if "HF_HUB_OFFLINE=1" not in entrypoint or "TRANSFORMERS_OFFLINE=1" not in entrypoint or "EVALUATION_BACKEND=vllm" not in entrypoint:
        raise ValueError("Docker entrypoint must force offline vLLM inference")
    if 'exec bash /root/participant/start.sh "$TEST_FILE" "$RESULT_DIR"' not in entrypoint:
        raise ValueError("Docker entrypoint must pass the two official arguments to start.sh")
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
    gates = verify_previous_gates(config, repo_root)
    package_ready = gates.get("status") == "passed"
    return {
        "status": "ready_for_asset_packaging" if package_ready else "historical_evidence_passed_runtime_validation_pending",
        "facts": facts,
        "previous_gates": gates,
        "runtime_lock_package_count": len(pins),
        "image_size_limit_bytes": MAX_IMAGE_BYTES,
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
        "version": "final-dual-container-model-lock-v1.1",
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
        raise RuntimeError("refusing Docker packaging until new v1.1 host and old-image overlay gates pass")
    config = load_json(repo_root / DOCKER_CONFIG)
    source_config = load_json(repo_root / SOURCE_CONFIG)
    paths = _source_run_paths(source_config, repo_root)
    source_lock_path = paths["model_lock"]
    source_lock = load_json(source_lock_path)

    from participant.final_dual_inference import _verify_assets
    _tokenizer, verified = _verify_assets(source_config, (repo_root / SOURCE_CONFIG).resolve(), source_lock_path)
    base_path = resolve_repo_path(source_config["model_path"], repo_root)
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
            "version": config["runtime_behavior"]["version"],
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
            "schema_version": "final_dual_docker_package_v1.1",
            "runtime_behavior_version": "E03-v1.1-length-recovery",
            "status": "prepared_for_image_build",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _git_head(repo_root),
            "source_model_lock_sha256": sha256_file(source_lock_path),
            "container_model_lock_sha256": sha256_file(lock_path),
            "runtime_code_identity": runtime_code_identity,
            "deployment_config": {"path": DOCKER_CONFIG.as_posix(), "sha256": docker_config_sha},
            "input": {
                "path": "transformers_smoke20/input/smoke20_input.jsonl",
                "sha256": static["previous_gates"]["input_sha256"],
                "rows": 20,
            },
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
            "previous_gates": static["previous_gates"],
            "source_file_sha256": copied_source_hashes,
            "container_file_sha256": code_hashes,
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
            if exact_input and set(row) != {"id", "history"}:
                raise ValueError(f"{path}:{line_number}: input must contain only id and history")
            if not isinstance(row.get("id"), str) or not OFFICIAL_ID.fullmatch(row["id"]):
                raise ValueError(f"{path}:{line_number}: invalid official test ID")
            rows.append(row)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: empty rows or duplicate IDs")
    return rows


def _container_log_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"checked": False, "status": "not_provided"}
    content = path.read_text(encoding="utf-8", errors="replace")
    lowered = content.lower()
    markers = ("out of memory", "cuda error: out of memory", "fatal error", "segmentation fault")
    found = [marker for marker in markers if marker in lowered]
    if found:
        raise ValueError(f"container log contains failure markers: {found}")
    if "docker_gpu_preflight=passed" not in lowered:
        raise ValueError("container log does not prove the in-container CUDA GPU preflight passed")
    return {"checked": True, "status": "passed", "sha256": sha256_file(path), "oom_or_crash_markers": []}


def compare_reference_outputs(
    predictions: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    report: dict[str, Any],
    reference_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare stable contract fields while measuring online response-text equality.

    vLLM's online server does not guarantee repeatable generated text. Keep the
    fixed-input comparison useful by requiring identical IDs, stable fields,
    per-run exact Prompt token checks, and acceptable route/format quality;
    report response and runtime-statistic differences as observations.
    """
    if len(predictions) != len(reference_rows):
        raise ValueError("Docker and host vLLM references have different row counts")
    if reference_report.get("status") != "passed" or reference_report.get("complete") is not True:
        raise ValueError("host vLLM reference report is not a passed complete run")

    same_ids = [row.get("id") for row in predictions] == [
        row.get("id") for row in reference_rows
    ]
    same_profiles = all(
        row.get("user_profile") == ref.get("user_profile")
        for row, ref in zip(predictions, reference_rows)
    )
    same_emotion_labels = all(
        row.get("emotion_label") == ref.get("emotion_label")
        for row, ref in zip(predictions, reference_rows)
    )
    same_memory_refs = all(
        row.get("memory_refs") == ref.get("memory_refs")
        for row, ref in zip(predictions, reference_rows)
    )
    def asset_facts(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "base_weight_sha256": value.get("model_weight_identity", {}).get("observed_weight_hash"),
            "tokenizer_signature_sha256": value.get("tokenizer_identity", {}).get("signature_sha256"),
            "tokenizer_template_sha256": value.get("tokenizer_identity", {}).get("template_sha256"),
            "adapter_weight_sha256": {
                route: value.get("adapter_identities", {}).get(key, {}).get("weight_sha256")
                for route, key in (("profile", "profile"), ("emotion_response", "emotion_response"))
            },
            "prompt_sha256": {
                name: record.get("sha256")
                for name, record in value.get("prompt_identities", {}).items()
            },
            "runtime_code_sha256": value.get("runtime_code_identity", {}).get("file_sha256"),
        }

    same_asset_identity = asset_facts(report) == asset_facts(reference_report)
    if not same_ids or not same_profiles or not same_emotion_labels or not same_memory_refs:
        raise ValueError("Docker and host vLLM stable submission fields differ")
    if not same_asset_identity:
        raise ValueError("Docker and host Base, Adapter, Tokenizer, Prompt, or runtime-code identities differ")

    row_count = len(predictions)
    format_threshold = int(row_count * 0.95 + 0.999999)
    format_counts = report.get("format_success_count", {})
    reference_format_counts = reference_report.get("format_success_count", {})
    both_format_rates_pass = all(
        int(record.get(route, 0)) >= format_threshold
        for record in (format_counts, reference_format_counts)
        for route in ("profile", "emotion_response")
    )
    route_failures = report.get("route_failure_count", {})
    reference_route_failures = reference_report.get("route_failure_count", {})
    both_route_error_free = all(
        int(record.get(route, 0)) == 0
        for record in (route_failures, reference_route_failures)
        for route in ("profile", "emotion_response")
    )

    def prompt_facts(value: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
        exact = value.get("prompt_token_ids_exact_match_count") or {}
        denominator = value.get("prompt_token_ids_check_denominator")
        if not isinstance(denominator, dict):
            aliases = value.get("server", {}).get("requested_alias_count", {})
            denominator = {
                "profile": int(aliases.get("profile", row_count)),
                "emotion_response": int(aliases.get("emotion-response", row_count)),
            }
        normalized_exact = {
            route: int(exact.get(route, 0)) for route in ("profile", "emotion_response")
        }
        normalized_denominator = {
            route: int(denominator.get(route, 0)) for route in ("profile", "emotion_response")
        }
        return normalized_exact, normalized_denominator

    prompt_exact, prompt_denominator = prompt_facts(report)
    reference_prompt_exact, reference_prompt_denominator = prompt_facts(reference_report)
    both_prompt_checks_pass = all(
        exact[route] == denominator[route] and denominator[route] > 0
        for exact, denominator in (
            (prompt_exact, prompt_denominator),
            (reference_prompt_exact, reference_prompt_denominator),
        )
        for route in ("profile", "emotion_response")
    )
    if not both_format_rates_pass or not both_route_error_free or not both_prompt_checks_pass:
        raise ValueError("Docker and host route or parse metrics differ outside accepted quality gates")

    same_format_counts = format_counts == reference_format_counts
    same_prompt_counts = prompt_exact == reference_prompt_exact and prompt_denominator == reference_prompt_denominator
    same_route_failures = route_failures == reference_route_failures
    same_generation_cap_hits = report.get("generation_cap_hits") == reference_report.get("generation_cap_hits")
    runtime_statistics = {
        key: {
            "docker": report.get(key, {"profile": 0, "emotion_response": 0} if key != "degraded_responses" else 0),
            "host": reference_report.get(key, {"profile": 0, "emotion_response": 0} if key != "degraded_responses" else 0),
        }
        for key in ("generation_cap_hits", "initial_cap_hits", "retry_attempts", "retry_resolved", "unresolved_cap_hits", "degraded_responses")
    }

    response_text_exact_count = sum(
        row.get("response_text") == ref.get("response_text")
        for row, ref in zip(predictions, reference_rows)
    )
    exact_submission_match = predictions == reference_rows
    metrics_match = all((same_format_counts, same_prompt_counts, same_route_failures, same_generation_cap_hits)) and all(
        stats["docker"] == stats["host"] for stats in runtime_statistics.values()
    )
    return {
        "status": (
            "exact_submission_and_metrics_match"
            if exact_submission_match and metrics_match
            else "stable_fields_and_metrics_match_response_text_varies"
            if not exact_submission_match and metrics_match
            else "stable_fields_match_runtime_metrics_vary"
        ),
        "same_id_order": same_ids,
        "same_user_profile": same_profiles,
        "same_emotion_labels": same_emotion_labels,
        "same_memory_refs": same_memory_refs,
        "same_asset_identity": same_asset_identity,
        "format_success_count_equal": same_format_counts,
        "prompt_token_ids_exact_match_count_equal": same_prompt_counts,
        "route_failure_count_equal": same_route_failures,
        "generation_cap_hits_equal": same_generation_cap_hits,
        "prompt_token_ids_exact_match_count": prompt_exact,
        "prompt_token_ids_check_denominator": prompt_denominator,
        "reference_prompt_token_ids_exact_match_count": reference_prompt_exact,
        "reference_prompt_token_ids_check_denominator": reference_prompt_denominator,
        "both_format_rates_pass": both_format_rates_pass,
        "both_prompt_checks_pass": both_prompt_checks_pass,
        "both_route_error_free": both_route_error_free,
        "runtime_statistics": runtime_statistics,
        "exact_submission_match": exact_submission_match,
        "response_text_exact_match_count": response_text_exact_count,
        "response_text_rows": len(predictions),
    }


def verify_output(
    *, test_file: Path, result_dir: Path, repo_root: Path = ROOT,
    reference_dir: Path | None = None, container_log: Path | None = None,
    expected_rows: int = 20,
) -> dict[str, Any]:
    config = load_json(repo_root / DOCKER_CONFIG)
    expected = validate_static_config(config, repo_root)
    package_manifest_path = result_dir.parent / "context/package_manifest.json"
    if not package_manifest_path.is_file():
        package_manifest_path = result_dir.parent / "package_manifest.json"
    if not package_manifest_path.is_file():
        raise FileNotFoundError("prepared Docker context package_manifest.json is missing")
    package_manifest = load_json(package_manifest_path)
    package_model_lock_path = package_manifest_path.parent / "model_lock.json"
    if not package_model_lock_path.is_file():
        raise FileNotFoundError("prepared Docker context model_lock.json is missing")
    report_path = result_dir / "performance_report.json"
    submission_path = result_dir / "submission.jsonl"
    raw_path = result_dir / "raw_outputs.jsonl"
    for path in (report_path, submission_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"Docker result missing required output: {path.name}")
    inputs = _read_jsonl(test_file, exact_input=True)
    predictions = _read_jsonl(submission_path)
    raw_rows = _read_jsonl(raw_path)
    input_ids = [row["id"] for row in inputs]
    output_ids = [row["id"] for row in predictions]
    row_count = len(inputs)
    if row_count != expected_rows or output_ids != input_ids or [row["id"] for row in raw_rows] != input_ids:
        raise ValueError(f"Docker run must preserve all {expected_rows} input IDs in order in both output files")
    schema = load_json(repo_root / SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    errors = [index for index, row in enumerate(predictions, 1) if next(validator.iter_errors(row), None) is not None]
    if errors:
        raise ValueError(f"submission schema failed for rows {errors[:5]}")

    report = load_json(report_path)
    progress_path = result_dir / "run_progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError("Docker result missing run_progress.json")
    progress = load_json(progress_path)
    if report.get("status") != "passed" or report.get("complete") is not True:
        raise ValueError("performance_report does not show a passed complete run")
    if package_manifest.get("runtime_behavior_version") == "E03-v1.1-length-recovery" and report.get("version") != "final-dual-official-run-v1.1":
        raise ValueError("v1.1 Docker package was not run with the v1.1 official inference entrypoint")
    if package_manifest.get("runtime_behavior_version") == "E03-v1.1-length-recovery":
        runtime_code_identity = package_manifest.get("runtime_code_identity")
        if not isinstance(runtime_code_identity, dict) or not runtime_code_identity.get("file_sha256"):
            raise ValueError("prepared v1.1 package manifest lacks runtime code file hashes")
        if report.get("runtime_code_identity") != runtime_code_identity:
            raise ValueError("container runtime code identity differs from the prepared v1.1 package manifest")
    if report.get("backend") != "vllm" or report.get("samples") != row_count or report.get("submission_rows") != row_count:
        raise ValueError("performance_report backend/sample count differs from the requested Docker run")
    if report.get("submission_schema_validated_count") != row_count or report.get("submission_schema_validated") is not True:
        raise ValueError(f"performance_report does not certify {row_count} Schema-valid submissions")
    if progress.get("test_file_sha256") != sha256_file(test_file) or progress.get("sample_count") != row_count:
        raise ValueError("performance_report input SHA-256 differs from the Docker run input")
    if row_count == 20 and package_manifest.get("input", {}).get("sha256") != sha256_file(test_file):
        raise ValueError("Docker package manifest input SHA-256 differs from the Docker Smoke input")
    if report.get("model_weight_identity", {}).get("observed_weight_hash") != expected["base_weight_sha256"]:
        raise ValueError("container Base identity differs from the locked model")
    if report.get("configuration", {}).get("sha256") != sha256_file(repo_root / DOCKER_CONFIG):
        raise ValueError("container deployment config identity differs from the package config")
    if report.get("configuration", {}).get("model_lock_sha256") != sha256_file(package_model_lock_path):
        raise ValueError("container model_lock SHA-256 differs from the prepared package")
    if report.get("initialization_seconds") is None or report.get("server", {}).get("alive_before_intentional_shutdown") is not True:
        raise ValueError("performance_report lacks a live initialized vLLM server observation")

    registered = set(report.get("server", {}).get("registered_model_ids", []))
    if not {"qwen-base", "profile", "emotion-response"}.issubset(registered):
        raise ValueError(f"container vLLM did not register all locked aliases: {sorted(registered)}")
    if report.get("server", {}).get("runtime", {}).get("version") != "0.11.1":
        raise ValueError("container vLLM runtime is not version 0.11.1")
    retry_counts = report.get("retry_attempts")
    if not isinstance(retry_counts, dict):
        retry_counts = {"profile": 0, "emotion_response": 0}
    profile_retries = int(retry_counts.get("profile", 0))
    emotion_retries = int(retry_counts.get("emotion_response", 0))
    if profile_retries != 0:
        raise ValueError("the profile alias must not use length retries")
    expected_alias_counts = {
        "profile": row_count + profile_retries,
        "emotion-response": row_count + emotion_retries,
    }
    if report.get("server", {}).get("requested_alias_count") != expected_alias_counts:
        raise ValueError("LoRA alias request counts differ from completed samples plus retries")
    exact_counts = report.get("prompt_token_ids_exact_match_count", {})
    prompt_denominators = report.get("prompt_token_ids_check_denominator")
    if not isinstance(prompt_denominators, dict):
        prompt_denominators = {
            "profile": expected_alias_counts["profile"],
            "emotion_response": expected_alias_counts["emotion-response"],
        }
    if any(
        exact_counts.get(route) != expected_alias_counts[alias]
        or prompt_denominators.get(route) != expected_alias_counts[alias]
        for route, alias in (("profile", "profile"), ("emotion_response", "emotion-response"))
    ):
        raise ValueError("container vLLM Prompt Token IDs did not exactly match for every successful route attempt")
    format_threshold = int(row_count * 0.95 + 0.999999)
    if report.get("format_success_count", {}).get("profile", 0) < format_threshold or report.get("format_success_count", {}).get("emotion_response", 0) < format_threshold:
        raise ValueError("container strict format rate is below 95%")
    if any(report.get("route_failure_count", {}).get(route, 0) for route in ("profile", "emotion_response")):
        raise ValueError("container encountered a route failure")
    severe_anomalies = int(report.get("emotion_response_severe_anomaly_count", 0))
    if severe_anomalies:
        raise ValueError(f"container encountered {severe_anomalies} severe repeated-response anomalies")
    outcome_counts = report.get("emotion_response_outcome_count", {})
    unresolved_caps = report.get("unresolved_cap_hits", {}).get("emotion_response", 0)
    if row_count == 100 and (unresolved_caps > 2 or outcome_counts.get("invalid_output", 0) > 0):
        raise ValueError("100-row stress run exceeded the unresolved-cap or invalid-output quality threshold")
    for name, key in (("profile", "profile"), ("emotion_response", "emotion_response")):
        identity = report.get("adapter_identities", {}).get(key, {})
        if identity.get("weight_sha256") != expected["adapter_weight_sha256"][name]:
            raise ValueError(f"container {name} Adapter weight identity differs from the package lock")
        if identity.get("config_sha256") != expected["adapter_config_sha256"][name]:
            raise ValueError(f"container {name} Adapter config identity differs from the package lock")
    gpu_devices = report.get("gpu_memory", {}).get("devices", [])
    if not gpu_devices or max(int(device.get("peak_memory_used_mib", 0)) for device in gpu_devices) <= 0:
        raise ValueError("performance_report does not show in-container GPU memory observations")
    log_facts = _container_log_gate(container_log)

    comparison = {"status": "not_requested"}
    if reference_dir is not None:
        reference_submission = reference_dir / "submission.jsonl"
        reference_report_path = reference_dir / "performance_report.json"
        if not reference_submission.is_file() or not reference_report_path.is_file():
            raise FileNotFoundError("host reference directory lacks the official outputs")
        reference_rows = _read_jsonl(reference_submission)
        reference_report = load_json(reference_report_path)
        comparison = compare_reference_outputs(predictions, reference_rows, report, reference_report)
        comparison["reference_submission_sha256"] = sha256_file(reference_submission)
        comparison["docker_submission_sha256"] = sha256_file(submission_path)
        comparison["submission_sha256_equal"] = (
            sha256_file(reference_submission) == sha256_file(submission_path)
        )
    return {
        "status": "passed_offline_dual_adapter_docker_smoke",
        "input_rows": len(inputs),
        "submission_rows": len(predictions),
        "raw_output_rows": len(raw_rows),
        "schema_valid_rows": len(predictions),
        "base_weight_sha256": expected["base_weight_sha256"],
        "adapter_weight_sha256": expected["adapter_weight_sha256"],
        "adapter_config_sha256": expected["adapter_config_sha256"],
        "vllm_version": "0.11.1",
        "registered_model_ids": sorted(registered),
        "requested_alias_count": report["server"]["requested_alias_count"],
        "prompt_token_ids_exact_match_count": exact_counts,
        "prompt_token_ids_check_denominator": prompt_denominators,
        "retry_attempts": retry_counts,
        "generation_cap_hits": report.get("generation_cap_hits", {}),
        "unresolved_cap_hits": report.get("unresolved_cap_hits", {}),
        "quality_warnings": report.get("quality_warnings", []),
        "format_success_count": report["format_success_count"],
        "gpu_memory": gpu_devices,
        "container_log": log_facts,
        "comparison_to_host": comparison,
        "submission_sha256": sha256_file(submission_path),
        "raw_outputs_sha256": sha256_file(raw_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify earlier gates and static runtime locks")
    prepare = sub.add_parser("prepare", help="create a verified, white-list Docker context under outputs/")
    prepare.add_argument("--config", type=Path, default=ROOT / DOCKER_CONFIG)
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = sub.add_parser("verify-output", help="check offline Docker outputs and optional host equivalence")
    verify.add_argument("--test-file", type=Path, required=True)
    verify.add_argument("--result-dir", type=Path, required=True)
    verify.add_argument("--reference-dir", type=Path)
    verify.add_argument("--container-log", type=Path)
    verify.add_argument("--expected-rows", type=int, default=20, help="expected official input/output rows; use 100 for the fixed stress set")
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
                reference_dir=None if args.reference_dir is None else args.reference_dir.expanduser().resolve(),
                container_log=None if args.container_log is None else args.container_log.expanduser().resolve(),
                expected_rows=args.expected_rows,
            )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
