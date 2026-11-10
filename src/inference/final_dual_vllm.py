"""Minimal managed vLLM service and OpenAI-compatible client."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


class VLLMServiceError(RuntimeError):
    """A service startup, routing, or response-contract failure."""


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _response_content(payload: Mapping[str, Any]) -> str:
    try:
        value = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLLMServiceError("chat completion response has no choices[0].message.content") from exc
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


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
    """Start one local vLLM server and send serial requests to its two LoRAs."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        root: str | Path,
        log_path: str | Path,
        vllm_bin: str = "vllm",
        startup_timeout_seconds: float = 900.0,
        request_timeout_seconds: float | None = None,
    ):
        self.config = dict(config)
        self.root = Path(root).resolve()
        self.log_path = Path(log_path).resolve()
        self.vllm_bin = vllm_bin
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.vllm = dict(self.config.get("deployment", {}).get("vllm", {}))
        self.host = str(os.environ.get("VLLM_HOST", self.vllm.get("host", "127.0.0.1")))
        self.port = int(os.environ.get("VLLM_PORT", self.vllm.get("port", 8000)))
        client_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        self.base_url = f"http://{client_host}:{self.port}/v1"
        self.timeout_seconds = float(request_timeout_seconds or self.vllm.get("request_timeout_seconds", 900))
        configured_memory = os.environ.get(
            "VLLM_GPU_MEMORY_UTILIZATION", self.vllm.get("gpu_memory_utilization", 0.8)
        )
        self.gpu_memory_utilization = float(configured_memory)
        self.enforce_eager = str(
            os.environ.get("VLLM_ENFORCE_EAGER", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self.command: list[str] = []
        self.resolved_binary: str | None = None

    def build_command(self) -> list[str]:
        base_path = _resolve_path(self.config["model_path"], self.root)
        adapters = self.config.get("adapters", {})
        try:
            profile_path = _resolve_path(adapters["profile"]["path"], self.root)
            emotion_path = _resolve_path(adapters["emotion_response"]["path"], self.root)
        except (KeyError, TypeError) as exc:
            raise ValueError("deployment config must define profile and emotion_response Adapter paths") from exc
        for label, path in (("Base", base_path), ("profile Adapter", profile_path), ("emotion-response Adapter", emotion_path)):
            if not path.is_dir():
                raise FileNotFoundError(f"{label} directory not found: {path}")
        required = (
            "served_model_name", "dtype", "max_model_len", "max_num_seqs",
            "max_num_batched_tokens", "max_lora_rank", "max_loras",
            "max_cpu_loras", "gpu_memory_utilization",
        )
        missing = [name for name in required if name not in self.vllm]
        if missing:
            raise ValueError(f"dual vLLM config is missing parameters: {missing}")
        return [
            self.resolved_binary or self.vllm_bin,
            "serve", str(base_path),
            "--host", self.host,
            "--port", str(self.port),
            "--served-model-name", str(self.vllm["served_model_name"]),
            "--dtype", str(self.vllm["dtype"]),
            "--max-model-len", str(self.vllm["max_model_len"]),
            "--max-num-seqs", str(self.vllm["max_num_seqs"]),
            "--max-num-batched-tokens", str(self.vllm["max_num_batched_tokens"]),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--enable-lora",
            "--max-lora-rank", str(self.vllm["max_lora_rank"]),
            "--max-loras", str(self.vllm["max_loras"]),
            "--max-cpu-loras", str(self.vllm["max_cpu_loras"]),
            "--lora-modules", f"profile={profile_path}", f"emotion-response={emotion_path}",
            *(["--enforce-eager"] if self.enforce_eager else []),
        ]

    def _resolve_binary(self) -> str:
        candidate = shutil.which(self.vllm_bin)
        if candidate is None:
            path = Path(self.vllm_bin).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                candidate = str(path.resolve())
        if candidate is None:
            raise FileNotFoundError(f"vLLM executable not found: {self.vllm_bin}")
        return candidate

    def _port_available(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((self.host, self.port)) != 0

    def _select_port(self) -> None:
        candidates = [self.port, *(port for port in range(8001, 8011) if port != self.port)]
        seen: set[int] = set()
        for port in candidates:
            if port in seen:
                continue
            seen.add(port)
            self.port = port
            client_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
            self.base_url = f"http://{client_host}:{port}/v1"
            if self._port_available():
                return
        raise VLLMServiceError(
            f"no free local vLLM port in {candidates}; host={self.host}"
        )

    def _log_tail(self, limit: int = 6000) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-limit:]

    @staticmethod
    def _looks_like_memory_error(text: str) -> bool:
        lowered = text.lower()
        return any(
            token in lowered
            for token in (
                "out of memory",
                "cuda out of memory",
                "no available memory",
                "not enough memory",
                "free memory",
                "gpu_memory_utilization",
                "kv cache",
                "oom",
            )
        )

    def _start_once(self, *, timeout: float) -> dict[str, Any]:
        self.command = self.build_command()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
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
            deadline = started + timeout
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise VLLMServiceError(
                        f"vLLM exited during startup with code {self.process.returncode}; see {self.log_path}"
                    )
                try:
                    models = self.list_models(timeout=5)
                    return {
                        "status": "ready",
                        "startup_seconds": time.monotonic() - started,
                        "models": models or None,
                        "port": self.port,
                        "gpu_memory_utilization": self.gpu_memory_utilization,
                        "enforce_eager": self.enforce_eager,
                    }
                except Exception as exc:
                    last_error = exc
                    time.sleep(2)
            raise VLLMServiceError(
                f"timed out waiting for vLLM readiness: {last_error}; see {self.log_path}"
            )
        except Exception:
            self.close()
            raise

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise RuntimeError("vLLM service is already started")
        self.resolved_binary = self._resolve_binary()
        self._select_port()

        configured_memory = self.gpu_memory_utilization
        memory_candidates = [configured_memory]
        for value in (0.7, 0.6, 0.5, 0.4):
            if value < configured_memory and value not in memory_candidates:
                memory_candidates.append(value)

        eager_candidates = [self.enforce_eager]
        if not self.enforce_eager:
            eager_candidates.append(True)

        attempts: list[str] = []
        overall_started = time.monotonic()
        for eager in eager_candidates:
            self.enforce_eager = eager
            for index, memory in enumerate(memory_candidates):
                self.gpu_memory_utilization = memory
                timeout = self.startup_timeout_seconds if not attempts else min(
                    self.startup_timeout_seconds, 300.0
                )
                try:
                    facts = self._start_once(timeout=timeout)
                    facts["startup_seconds"] = time.monotonic() - overall_started
                    facts["attempts"] = attempts
                    return facts
                except Exception as exc:
                    tail = self._log_tail()
                    attempts.append(
                        f"eager={eager} gpu_memory_utilization={memory} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    print(
                        f"WARNING: vLLM startup attempt failed "
                        f"(eager={eager}, gpu_memory_utilization={memory}): "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if index + 1 < len(memory_candidates) and self._looks_like_memory_error(tail):
                        continue
                    # Non-memory failures (driver, kernel, graph capture, ...)
                    # are retried once with --enforce-eager when available.
                    break

        last_tail = self._log_tail(limit=2500)
        raise VLLMServiceError(
            "vLLM failed to start after portable retries; "
            f"attempts={attempts}; log={self.log_path}; tail={last_tail}"
        )

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def list_models(self, *, timeout: float = 10.0) -> list[str]:
        payload = _http_json(f"{self.base_url}/models", timeout=timeout)
        values = payload.get("data")
        if not isinstance(values, list):
            return []
        return [value["id"] for value in values if isinstance(value, dict) and isinstance(value.get("id"), str)]

    def chat_completion(
        self,
        *,
        model_alias: str,
        messages: Sequence[Mapping[str, Any]],
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
        }
        started = time.monotonic()
        response = _http_json(f"{self.base_url}/chat/completions", timeout=self.timeout_seconds, payload=payload)
        elapsed = time.monotonic() - started
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return {
            "text": _response_content(response),
            "requested_model_alias": model_alias,
            "response_model": response.get("model") if isinstance(response.get("model"), str) else None,
            "finish_reason": choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
            "usage_prompt_tokens": prompt_tokens if type(prompt_tokens) is int else None,
            "usage_completion_tokens": completion_tokens if type(completion_tokens) is int else None,
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
