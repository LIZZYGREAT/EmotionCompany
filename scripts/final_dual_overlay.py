#!/usr/bin/env python3
"""Build a code-only overlay on a locally available final Dual image."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.final_dual_docker_package import PDF_PATH, pdf_asset_facts, verify_output  # noqa: E402

DEFAULT_FACTS = Path("outputs/deployment/final_overlay/base_image.json")
OVERLAY_FILES = (
    Path("participant/start_official.sh"),
    Path("participant/run_inference.py"),
    Path("participant/final_dual_inference.py"),
    Path("participant/configs/inference_config.json"),
    Path("scripts/e02_vllm_probe.py"),
    Path("configs/deployment/final_dual_docker.json"),
    Path("docker/final_dual/entrypoint.sh"),
    Path("docker/final_dual/entrypoint_args.sh"),
    Path("docker/final_dual/emotionalcompany_vllm_wrapper.py"),
    Path("src/inference/final_dual_protocol.py"),
    Path("src/inference/final_dual_vllm.py"),
    PDF_PATH,
)
LOCK_FILES = (
    Path("docker/e02/conda-linux-64.explicit.txt"),
    Path("docker/e02/requirements-linux-64.lock"),
)
MAX_IMAGE_BYTES = 50_000_000_000
FORBIDDEN_DOCKERFILE_TEXT = (
    "apt-get", "conda create", "pip install", "curl ", "wget ",
    "copy participant/model", "copy participant/adapters",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run_command(
    command: list[str], *, cwd: Path = ROOT, timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False,
        timeout=timeout, env=env,
    )


def inspect_image(image: str) -> dict[str, Any]:
    result = run_command(["docker", "image", "inspect", image])
    if result.returncode != 0:
        raise RuntimeError(f"docker image inspect failed for {image}: {result.stderr.strip()}")
    values = json.loads(result.stdout)
    if not isinstance(values, list) or not values or not isinstance(values[0], dict):
        raise ValueError(f"docker image inspect returned no image for {image}")
    return values[0]


def _runtime_probe(image: str) -> dict[str, Any]:
    probe = r"""set -euo pipefail
test -x /opt/conda/envs/conda_env/bin/python
test -x /opt/conda/envs/conda_env/bin/vllm
test -s /root/participant/model/config.json
test -s /root/participant/adapters/profile/adapter_model.safetensors
test -s /root/participant/adapters/emotion-response/adapter_model.safetensors
/opt/conda/envs/conda_env/bin/python - <<'PY'
import importlib.metadata
import json
import sys
print(json.dumps({
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "torch": importlib.metadata.version("torch"),
    "vllm": importlib.metadata.version("vllm"),
    "transformers": importlib.metadata.version("transformers"),
    "model_path": "/root/participant/model",
    "adapter_paths": [
        "/root/participant/adapters/profile",
        "/root/participant/adapters/emotion-response",
    ],
}, sort_keys=True))
PY"""
    command = [
        "docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/bash",
        image, "-lc", probe,
    ]
    result = run_command(command, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            "parent image does not contain the required conda_env, vLLM, Base, and both Adapters; "
            f"command={json.dumps(command)} stderr={result.stderr.strip()}"
        )
    try:
        facts = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read runtime versions from parent image: {result.stdout}") from exc
    return facts


def _repo_head() -> str | None:
    result = run_command(["git", "rev-parse", "HEAD"])
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_base(image: str, facts_file: Path) -> dict[str, Any]:
    facts_file = facts_file.expanduser().resolve()
    info = inspect_image(image)
    runtime = _runtime_probe(image)
    lock_hashes = {path.as_posix(): sha256_file(ROOT / path) for path in LOCK_FILES}
    facts = {
        "schema_version": "final-dual-overlay-base-v1",
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "docker_context": run_command(["docker", "context", "show"]).stdout.strip(),
        "image": image,
        "image_id": info.get("Id"),
        "repo_digests": info.get("RepoDigests") or [],
        "repo_tags": info.get("RepoTags") or [],
        "runtime": runtime,
        "rootfs_layers": info.get("RootFS", {}).get("Layers", []),
        "runtime_lock_sha256": lock_hashes,
        "source_git_head": _repo_head(),
        "model_assets": {
            "base": "/root/participant/model",
            "adapters": [
                "/root/participant/adapters/profile",
                "/root/participant/adapters/emotion-response",
            ],
        },
    }
    if not isinstance(facts["image_id"], str) or not facts["image_id"].startswith("sha256:"):
        raise ValueError(f"Docker did not return a stable image ID for {image}")
    facts_file.parent.mkdir(parents=True, exist_ok=True)
    if facts_file.exists():
        old = json_read(facts_file)
        if old.get("image_id") != facts["image_id"]:
            raise FileExistsError(f"refusing to replace facts for a different parent image: {facts_file}")
    facts_file.write_text(json.dumps(facts, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return facts


def _check_base_facts(image: str, facts_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    facts = json_read(facts_file.expanduser().resolve())
    if facts.get("image") != image:
        raise ValueError("base image tag differs from the inspected base_image.json")
    info = inspect_image(image)
    if info.get("Id") != facts.get("image_id"):
        raise ValueError("base image tag now resolves to a different image ID; refusing to build")
    current_lock_hashes = {path.as_posix(): sha256_file(ROOT / path) for path in LOCK_FILES}
    if current_lock_hashes != facts.get("runtime_lock_sha256"):
        raise ValueError("runtime dependency locks changed after parent inspection; this is not a code-only overlay")
    return facts, info


def validate_overlay_dockerfile(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    forbidden = [item for item in FORBIDDEN_DOCKERFILE_TEXT if item in lowered]
    if forbidden:
        raise ValueError(f"overlay Dockerfile contains forbidden rebuild/install instructions: {forbidden}")
    if "from ${base_image}" not in lowered or "copy report/main.pdf /root/report/main.pdf" not in lowered:
        raise ValueError("overlay Dockerfile must inherit BASE_IMAGE and copy report/main.pdf")


def prepare_code_context(
    *, image: str, tag: str, output_dir: Path, facts_file: Path,
) -> dict[str, Any]:
    if image == tag:
        raise ValueError("overlay must use a new tag and leave the parent image tag untouched")
    facts, _info = _check_base_facts(image, facts_file)
    dockerfile_source = ROOT / "docker/final_dual/Dockerfile.overlay"
    validate_overlay_dockerfile(dockerfile_source)
    pdf_facts = pdf_asset_facts(ROOT / PDF_PATH)
    sources = [ROOT / relative for relative in OVERLAY_FILES]
    for relative, source in zip(OVERLAY_FILES, sources, strict=True):
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"required overlay source missing or empty: {relative}")

    output_dir = output_dir.expanduser().resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if not output_dir.is_relative_to(outputs_root) or output_dir == outputs_root:
        raise ValueError("overlay context must be placed below ignored outputs/")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite overlay context: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        file_hashes: dict[str, str] = {}
        for relative, source in zip(OVERLAY_FILES, sources, strict=True):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            file_hashes[relative.as_posix()] = sha256_file(destination)
        shutil.copy2(dockerfile_source, temporary / "Dockerfile")
        (temporary / ".dockerignore").write_text("**/__pycache__\n**/*.pyc\n", encoding="utf-8")
        manifest = {
            "schema_version": "final-dual-code-overlay-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_git_head": _repo_head(),
            "parent_image": image,
            "parent_image_id": facts["image_id"],
            "new_image_tag": tag,
            "runtime": facts["runtime"],
            "runtime_lock_sha256": facts["runtime_lock_sha256"],
            "overlay_files_sha256": file_hashes,
            "report_pdf": pdf_facts,
            "context_file_count": len(file_hashes) + 3,
            "dependency_installation": False,
            "model_copy": False,
        }
        (temporary / "overlay_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "prepared",
        "context_dir": str(output_dir),
        "context_file_count": len(file_hashes) + 3,
        "context_regular_file_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
        "parent_image_id": facts["image_id"],
        "new_image_tag": tag,
        "report_pdf": pdf_facts,
    }


def _existing_tag(tag: str) -> bool:
    result = run_command(["docker", "image", "inspect", tag])
    if result.returncode == 0:
        return True
    if "No such image" in result.stderr or "not found" in result.stderr.lower():
        return False
    raise RuntimeError(f"could not check whether Docker tag already exists: {result.stderr.strip()}")


def build_overlay(*, image: str, tag: str, context_dir: Path, facts_file: Path, run_dir: Path) -> dict[str, Any]:
    facts, base_info = _check_base_facts(image, facts_file)
    context_dir = context_dir.expanduser().resolve()
    manifest = json_read(context_dir / "overlay_manifest.json")
    if manifest.get("parent_image_id") != facts["image_id"] or manifest.get("new_image_tag") != tag:
        raise ValueError("overlay context was not prepared for this parent image/tag")
    if _existing_tag(tag):
        raise FileExistsError(f"refusing to replace existing Docker image tag: {tag}")
    command = [
        "docker", "build", "--pull=false", "--network=none",
        "--build-arg", f"BASE_IMAGE={image}",
        "--build-arg", f"BASE_IMAGE_ID={facts['image_id']}",
        "-f", str(context_dir / "Dockerfile"), "-t", tag, str(context_dir),
    ]
    build_env = {**os.environ, "DOCKER_BUILDKIT": "1"}
    result = run_command(command, timeout=7200, env=build_env)
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"offline overlay Docker build failed (exit {result.returncode}); see {run_dir / 'build.log'}")
    child_info = inspect_image(tag)
    child_layers = child_info.get("RootFS", {}).get("Layers", [])
    base_layers = base_info.get("RootFS", {}).get("Layers", [])
    if child_layers[:len(base_layers)] != base_layers:
        raise ValueError("new image does not retain the inspected parent image layers")
    labels = child_info.get("Config", {}).get("Labels") or {}
    if labels.get("org.emotionalcompany.base-image-id") != facts["image_id"]:
        raise ValueError("new image label does not identify the inspected parent Image ID")
    result_data = {
        "status": "built",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "build_network": "none",
        "pull": False,
        "package_manager_invoked": False,
        "parent_image": image,
        "parent_image_id": facts["image_id"],
        "image": tag,
        "image_id": child_info.get("Id"),
        "rootfs_parent_layers_preserved": len(base_layers),
        "report_pdf": manifest.get("report_pdf"),
        "build_log": str(run_dir / "build.log"),
    }
    (run_dir / "build.json").write_text(json.dumps(result_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result_data


def make_smoke_rows(count: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("smoke count must be positive")
    rows = []
    for index in range(count):
        # Use IDs that satisfy the official submission Schema so the local
        # verifier can prove the platform-facing output contract.
        rows.append({
            "id": f"test_{index + 1:06d}_t0",
            "history": [{
                "role": "user",
                "content": "我今天有些疲惫，想找人聊几句。",
                "turn_id": index + 1,
                "smoke_metadata": "must be ignored",
            }],
            "platform_metadata": {"source": "overlay-smoke"},
        })
    return rows


def run_smoke(
    *, image: str, run_dir: Path, count: int = 2, mode: str = "direct", timeout: float = 7200,
) -> dict[str, Any]:
    if mode not in {"direct", "explicit-bash", "shell-wrapper", "bin-bash-command"}:
        raise ValueError("smoke mode must be direct, explicit-bash, shell-wrapper, or bin-bash-command")
    run_dir = run_dir.expanduser().resolve()
    if not (run_dir / "build.json").is_file():
        raise FileNotFoundError(f"overlay build evidence is missing: {run_dir / 'build.json'}")
    build = json_read(run_dir / "build.json")
    info = inspect_image(image)
    if info.get("Id") != build.get("image_id"):
        raise ValueError("smoke image ID differs from the image recorded by the build")
    pdf = pdf_asset_facts(ROOT / PDF_PATH)
    pdf_check = run_command([
        "docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/bash",
        image, "-lc", "test -s /root/report/main.pdf && sha256sum /root/report/main.pdf",
    ], timeout=120)
    if pdf_check.returncode != 0:
        raise RuntimeError(f"container PDF check failed: {pdf_check.stderr.strip()}")
    image_pdf_hash = pdf_check.stdout.split()[0] if pdf_check.stdout.split() else ""
    if image_pdf_hash != pdf["sha256"]:
        raise ValueError("PDF SHA-256 in the image differs from report/main.pdf")

    smoke_root = run_dir / f"smoke-{mode}-{count}"
    if smoke_root.exists():
        raise FileExistsError(f"refusing to overwrite smoke outputs: {smoke_root}")
    input_dir = smoke_root / "input"
    result_dir = smoke_root / "result"
    input_dir.mkdir(parents=True)
    result_dir.mkdir()
    test_file = input_dir / "test.jsonl"
    test_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in make_smoke_rows(count)),
        encoding="utf-8",
    )
    keep_path = result_dir / "keep.txt"
    keep_path.write_text("preserve this file\n", encoding="utf-8")
    container_command = [
        "docker", "run", "--rm", "--gpus", "all", "--network", "none",
        "-v", f"{test_file}:/input/test.jsonl:ro",
        "-v", f"{result_dir}:/output:rw",
    ]
    if mode == "explicit-bash":
        container_command.extend([image, "bash", "/root/participant/start.sh", "/input/test.jsonl", "/output"])
    elif mode == "shell-wrapper":
        container_command.extend([
            image, "bash", "-lc",
            "bash /root/participant/start.sh /input/test.jsonl /output",
        ])
    elif mode == "bin-bash-command":
        container_command.extend([
            image, "/bin/bash", "-c",
            "bash /root/participant/start.sh /input/test.jsonl /output",
        ])
    else:
        container_command.extend([image, "/input/test.jsonl", "/output"])
    started_at = datetime.now(timezone.utc).isoformat()
    result = run_command(container_command, timeout=timeout)
    (smoke_root / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (smoke_root / "stderr.log").write_text(result.stderr, encoding="utf-8")
    data: dict[str, Any] = {
        "status": "failed",
        "started_at_utc": started_at,
        "command": container_command,
        "exit_code": result.returncode,
        "network": "none",
        "gpu_requested": True,
        "input_rows": count,
        "image": image,
        "image_id": info.get("Id"),
        "pdf_sha256_local": pdf["sha256"],
        "pdf_sha256_container": image_pdf_hash,
        "stdout_path": str(smoke_root / "stdout.log"),
        "stderr_path": str(smoke_root / "stderr.log"),
    }
    if result.returncode == 0:
        verification = verify_output(test_file=test_file, result_dir=result_dir)
        if not keep_path.is_file() or keep_path.read_text(encoding="utf-8") != "preserve this file\n":
            raise ValueError("official inference removed or changed a non-official result-directory file")
        data["verification"] = verification
        data["submission_path"] = str(result_dir / "submission.jsonl")
        data["performance_report_path"] = str(result_dir / "performance_report.json")
        data["status"] = "passed"
    (smoke_root / "smoke.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Docker smoke exited {result.returncode}; see {smoke_root / 'stderr.log'}")
    return data


def export_image(*, image: str, run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    build_path = run_dir / "build.json"
    smoke_candidates = list(run_dir.glob("smoke-*/smoke.json"))
    if not build_path.is_file() or not smoke_candidates:
        raise FileNotFoundError("export requires a successful overlay build and smoke record")
    build = json_read(build_path)
    successful_smokes = [json_read(path) for path in smoke_candidates]
    matching_smokes = [
        item
        for item in successful_smokes
        if item.get("status") == "passed" and item.get("image_id") == build.get("image_id")
    ]
    if not matching_smokes:
        raise ValueError(
            "export is blocked until a Docker smoke for the exact built image ID passes"
        )
    info = inspect_image(image)
    if info.get("Id") != build.get("image_id"):
        raise ValueError("export image ID differs from the smoke-tested build")
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    tar_path = export_dir / "image.tar"
    if tar_path.exists():
        raise FileExistsError(f"refusing to replace existing image tar: {tar_path}")
    temporary_tar = export_dir / ".image.tar.partial"
    if temporary_tar.exists():
        raise FileExistsError(f"refusing to overwrite partial tar: {temporary_tar}")
    command = ["docker", "save", image, "-o", str(temporary_tar)]
    result = run_command(command, timeout=7200)
    if result.returncode != 0:
        temporary_tar.unlink(missing_ok=True)
        raise RuntimeError(f"docker save failed (exit {result.returncode}): {result.stderr.strip()}")
    size_bytes = temporary_tar.stat().st_size
    if size_bytes > MAX_IMAGE_BYTES:
        temporary_tar.unlink(missing_ok=True)
        raise ValueError(f"full image tar exceeds 50 GB: {size_bytes} bytes")
    digest = sha256_file(temporary_tar)
    os.replace(temporary_tar, tar_path)
    checksum_path = export_dir / "image.tar.sha256"
    checksum_path.write_text(f"{digest}  image.tar\n", encoding="utf-8")
    data = {
        "status": "exported",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "image": image,
        "image_id": info.get("Id"),
        "parent_image_id": build.get("parent_image_id"),
        "tar_path": str(tar_path),
        "tar_size_bytes": size_bytes,
        "tar_size_gib": round(size_bytes / (1024 ** 3), 3),
        "tar_sha256": digest,
        "checksum_path": str(checksum_path),
    }
    (export_dir / "export.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-base", help="inspect a local complete image without pulling")
    inspect.add_argument("--base-image", required=True)
    inspect.add_argument("--facts-file", type=Path, default=ROOT / DEFAULT_FACTS)

    prepare = subparsers.add_parser("prepare-code-context", help="make a small code-and-PDF overlay context")
    prepare.add_argument("--base-image", required=True)
    prepare.add_argument("--tag", required=True)
    prepare.add_argument("--facts-file", type=Path, default=ROOT / DEFAULT_FACTS)
    prepare.add_argument("--output-dir", type=Path, required=True)

    build = subparsers.add_parser("build", help="build an offline overlay from the inspected local image")
    build.add_argument("--base-image", required=True)
    build.add_argument("--tag", required=True)
    build.add_argument("--facts-file", type=Path, default=ROOT / DEFAULT_FACTS)
    build.add_argument("--context-dir", type=Path, required=True)
    build.add_argument("--run-dir", type=Path, required=True)

    smoke = subparsers.add_parser("smoke", help="run a small offline GPU smoke against the new image")
    smoke.add_argument("--image", required=True)
    smoke.add_argument("--run-dir", type=Path, required=True)
    smoke.add_argument("--count", type=int, default=2)
    smoke.add_argument("--mode", choices=("direct", "explicit-bash", "shell-wrapper", "bin-bash-command"), default="direct")
    smoke.add_argument("--timeout", type=float, default=7200)

    export = subparsers.add_parser("export", help="save one complete, smoke-tested image tar")
    export.add_argument("--image", required=True)
    export.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-base":
            result = inspect_base(args.base_image, args.facts_file)
        elif args.command == "prepare-code-context":
            result = prepare_code_context(
                image=args.base_image,
                tag=args.tag,
                output_dir=args.output_dir,
                facts_file=args.facts_file,
            )
        elif args.command == "build":
            result = build_overlay(
                image=args.base_image,
                tag=args.tag,
                context_dir=args.context_dir,
                facts_file=args.facts_file,
                run_dir=args.run_dir,
            )
        elif args.command == "smoke":
            result = run_smoke(
                image=args.image,
                run_dir=args.run_dir,
                count=args.count,
                mode=args.mode,
                timeout=args.timeout,
            )
        else:
            result = export_image(image=args.image, run_dir=args.run_dir)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
