"""Managed vLLM 0.11.1 service and strict OpenAI-compatible client."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


class VLLMServiceError(RuntimeError):
    """A service startup, routing, or response-contract failure."""


def _sha256_json(values: Sequence[int]) -> str:
    encoded = json.dumps([int(value) for value in values], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _response_content(payload: Mapping[str, Any]) -> str:
    try:
        value = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLLMServiceError("chat completion response has no choices[0].message.content") from exc
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item if isinstance(item, str) else item.get("text", "") for item in value if isinstance(item, (str, dict))]
        if all(isinstance(part, str) for part in parts) and parts:
            return "".join(parts)
    raise VLLMServiceError("chat completion content is not a string or text-part array")


def _http_json(url: str, *, timeout: float, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data is not None else {})},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            error_payload = json.loads(body.decode("utf-8"))
            detail = error_payload.get("error", {}) if isinstance(error_payload, dict) else {}
            code = detail.get("code") if isinstance(detail, dict) else None
            error_type = detail.get("type") if isinstance(detail, dict) else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            code = error_type = None
        raise VLLMServiceError(f"HTTP {exc.code} from vLLM (code={code!r}, type={error_type!r})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise VLLMServiceError(f"vLLM request failed: {type(exc).__name__}: {exc}") from exc
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VLLMServiceError(f"vLLM returned non-JSON response (HTTP {status}, {len(body)} bytes)") from exc
    if not isinstance(result, dict):
        raise VLLMServiceError("vLLM returned a non-object JSON response")
    return result


def gpu_snapshot() -> dict[str, Any]:
    """Read a privacy-safe GPU name and memory snapshot, when nvidia-smi is present."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {"status": "unavailable", "reason": "nvidia-smi_not_found", "devices": []}
    result = subprocess.run(
        [binary, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return {"status": "unavailable", "reason": "nvidia-smi_failed", "devices": []}
    devices = []
    for line in result.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) == 3:
            try:
                used, total = int(columns[1]), int(columns[2])
            except ValueError:
                continue
            devices.append({"name": columns[0], "memory_used_mib": used, "memory_total_mib": total})
    return {"status": "observed" if devices else "unavailable", "source": "nvidia-smi", "devices": devices}


class GpuMemoryMonitor:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds
        self.snapshots: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU monitor is already started")
        self._thread = threading.Thread(target=self._run, name="final-dual-gpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.snapshots.append({"observed_at_monotonic": time.monotonic(), **gpu_snapshot()})
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3))
        devices = {}
        for snapshot in self.snapshots:
            for device in snapshot.get("devices", []):
                name = device["name"]
                current = devices.setdefault(name, {"name": name, "peak_memory_used_mib": 0, "memory_total_mib": device["memory_total_mib"]})
                current["peak_memory_used_mib"] = max(current["peak_memory_used_mib"], device["memory_used_mib"])
        return {"sample_count": len(self.snapshots), "devices": list(devices.values()), "first": self.snapshots[0] if self.snapshots else None, "last": self.snapshots[-1] if self.snapshots else None}


class DualLoRAVLLMService:
    """Start one pinned vLLM server and route requests to its named LoRAs."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        model_lock: Mapping[str, Any],
        root: str | Path,
        log_path: str | Path,
        vllm_bin: str = "vllm",
        startup_timeout_seconds: float = 900.0,
        request_timeout_seconds: float | None = None,
    ):
        self.config = dict(config)
        self.lock = dict(model_lock)
        self.root = Path(root).resolve()
        self.log_path = Path(log_path).resolve()
        self.vllm_bin = vllm_bin
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        deployment = self.config.get("deployment", {}).get("vllm", {})
        self.vllm = dict(deployment)
        self.host = str(self.vllm.get("host", "127.0.0.1"))
        self.port = int(self.vllm.get("port", 8000))
        self.base_url = str(self.vllm.get("base_url", f"http://{self.host}:{self.port}/v1")).rstrip("/")
        self.timeout_seconds = float(request_timeout_seconds or self.vllm.get("request_timeout_seconds", 900))
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self.command: list[str] = []
        self.resolved_binary: str | None = None
        self.version: str | None = None

    def build_command(self) -> list[str]:
        base_path = _resolve_path(self.config["model_path"], self.root)
        adapters = self.lock["deployment_adapters"]
        profile_path = _resolve_path(adapters["profile"]["path"], self.root)
        emotion_path = _resolve_path(adapters["emotion_response"]["path"], self.root)
        for path in (base_path, profile_path, emotion_path):
            if not path.is_dir():
                raise FileNotFoundError(f"locked vLLM asset path not found: {path}")
        required = ("served_model_name", "dtype", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "max_lora_rank", "max_loras", "max_cpu_loras", "gpu_memory_utilization")
        missing = [name for name in required if name not in self.vllm]
        if missing:
            raise ValueError(f"dual vLLM config is missing parameters: {missing}")
        command = [
            self.resolved_binary or self.vllm_bin,
            "serve",
            str(base_path),
            "--host", self.host,
            "--port", str(self.port),
            "--served-model-name", str(self.vllm["served_model_name"]),
            "--dtype", str(self.vllm["dtype"]),
            "--max-model-len", str(self.vllm["max_model_len"]),
            "--max-num-seqs", str(self.vllm["max_num_seqs"]),
            "--max-num-batched-tokens", str(self.vllm["max_num_batched_tokens"]),
            "--gpu-memory-utilization", str(self.vllm["gpu_memory_utilization"]),
            "--enable-lora",
            "--max-lora-rank", str(self.vllm["max_lora_rank"]),
            "--max-loras", str(self.vllm["max_loras"]),
            "--max-cpu-loras", str(self.vllm["max_cpu_loras"]),
            "--lora-modules", f"profile={profile_path}", f"emotion-response={emotion_path}",
        ]
        return command

    def _resolve_binary(self) -> str:
        candidate = shutil.which(self.vllm_bin)
        if candidate is None:
            path = Path(self.vllm_bin).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                candidate = str(path.resolve())
        if candidate is None:
            raise FileNotFoundError(f"vLLM executable not found: {self.vllm_bin}")
        return candidate

    def _check_version(self) -> str:
        result = subprocess.run([self.resolved_binary, "--version"], capture_output=True, text=True, timeout=30, check=False)
        if result.returncode != 0:
            raise VLLMServiceError(f"vLLM --version failed: {result.stderr[-1000:]}")
        text = result.stdout + "\n" + result.stderr
        versions = re.findall(r"(?m)^\s*(?:vllm(?:,\s*version)?\s+)?v?(\d+\.\d+\.\d+)\s*$", text, flags=re.IGNORECASE)
        if len(versions) != 1:
            raise VLLMServiceError(f"cannot uniquely determine vLLM version from {text[-1000:]!r}")
        expected = str(self.vllm.get("expected_version", ""))
        if expected and versions[0] != expected:
            raise VLLMServiceError(f"vLLM version mismatch: expected {expected}, got {versions[0]}")
        return versions[0]

    def _port_available(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((self.host, self.port)) != 0

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise RuntimeError("vLLM service is already started")
        if urlsplit(self.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("dual LoRA service must bind to loopback only")
        self.resolved_binary = self._resolve_binary()
        self.version = self._check_version()
        if not self._port_available():
            raise VLLMServiceError(f"configured vLLM port is already occupied: {self.host}:{self.port}")
        self.command = self.build_command()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("wb")
        started = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            cwd=self.root,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )
        try:
            deadline = started + self.startup_timeout_seconds
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise VLLMServiceError(f"vLLM exited during startup with code {self.process.returncode}; see {self.log_path}")
                try:
                    models = self.list_models(timeout=5)
                    return {
                        "status": "ready",
                        "startup_seconds": time.monotonic() - started,
                        "version": self.version,
                        "command": self.command,
                        "models": models,
                        "log_path": str(self.log_path),
                    }
                except Exception as exc:
                    last_error = exc
                    time.sleep(2)
            raise VLLMServiceError(f"timed out waiting for vLLM readiness: {last_error}; see {self.log_path}")
        except Exception:
            self.close()
            raise

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def list_models(self, *, timeout: float = 10.0) -> list[str]:
        payload = _http_json(f"{self.base_url}/models", timeout=timeout)
        values = payload.get("data")
        if not isinstance(values, list):
            raise VLLMServiceError("/v1/models response has no data array")
        model_ids = [value.get("id") for value in values if isinstance(value, dict) and isinstance(value.get("id"), str)]
        if not model_ids:
            raise VLLMServiceError("/v1/models returned no model IDs")
        return model_ids

    def require_adapters(self) -> list[str]:
        model_ids = self.list_models()
        expected = {str(self.vllm.get("served_model_name", "qwen-base")), "profile", "emotion-response"}
        missing = expected - set(model_ids)
        if missing:
            raise VLLMServiceError(f"vLLM /v1/models misses registered names: {sorted(missing)}; observed={model_ids}")
        return model_ids

    def chat_completion(
        self,
        *,
        model_alias: str,
        messages: Sequence[Mapping[str, Any]],
        expected_prompt_token_ids: Sequence[int],
        max_tokens: int,
    ) -> dict[str, Any]:
        clean_messages = []
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ValueError("generation messages must have standard role/content string fields")
            clean_messages.append({"role": role, "content": content})
        payload = {
            "model": model_alias,
            "messages": clean_messages,
            "max_tokens": int(max_tokens),
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "return_token_ids": True,
        }
        started = time.monotonic()
        response = _http_json(f"{self.base_url}/chat/completions", timeout=self.timeout_seconds, payload=payload)
        elapsed = time.monotonic() - started
        observed = response.get("prompt_token_ids")
        if not isinstance(observed, list) or any(type(value) is not int for value in observed):
            raise VLLMServiceError("vLLM response omitted a valid prompt_token_ids array")
        expected = [int(value) for value in expected_prompt_token_ids]
        usage = response.get("usage")
        if not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int:
            raise VLLMServiceError("vLLM response omitted integer usage.completion_tokens")
        prompt_usage = usage.get("prompt_tokens")
        if type(prompt_usage) is not int:
            raise VLLMServiceError("vLLM response omitted integer usage.prompt_tokens")
        try:
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise VLLMServiceError("vLLM response omitted choices[0]") from exc
        response_model = response.get("model")
        return {
            "text": _response_content(response),
            "requested_model_alias": model_alias,
            "response_model": response_model if isinstance(response_model, str) else None,
            "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
            "prompt_token_count_expected": len(expected),
            "prompt_token_count_observed": len(observed),
            "prompt_token_ids_expected_sha256": _sha256_json(expected),
            "prompt_token_ids_observed_sha256": _sha256_json(observed),
            "prompt_token_ids_exact_match": expected == observed,
            "usage_prompt_tokens": prompt_usage,
            "usage_completion_tokens": usage["completion_tokens"],
            "generation_seconds": elapsed,
            "response_id": response.get("id") if isinstance(response.get("id"), str) else None,
        }

    def close(self, *, terminate_timeout_seconds: float = 30.0) -> int | None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=terminate_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        returncode = process.poll() if process is not None else None
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        return returncode
