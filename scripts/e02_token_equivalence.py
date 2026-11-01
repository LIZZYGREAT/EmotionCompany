#!/usr/bin/env python3
"""Compare local P0/F2/T1 prompt IDs with IDs returned by pinned vLLM.

The diagnostic reads target-free ``id`` + ``history`` input, sends one-token
requests to an already running vLLM OpenAI-compatible server, and writes only
hashes, counts, comparison positions, and status metadata. It never saves
messages, generated text, or complete token ID arrays.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from participant import run_inference as runner  # noqa: E402
from scripts import e02_vllm_probe as probe  # noqa: E402


MAX_INPUT_SAMPLES = 20
SYNTHETIC_THINK_SAMPLES = (
    {
        "id": "e02_synthetic_think_01",
        "history": [
            {"role": "user", "content": "This is a synthetic protocol check."},
            {"role": "assistant", "content": "The literal marker </think> is ordinary history text."},
            {"role": "user", "content": "Continue the synthetic exchange."},
        ],
    },
    {
        "id": "e02_synthetic_think_02",
        "history": [
            {"role": "user", "content": "A second synthetic protocol check."},
            {"role": "assistant", "content": "</think> appears at the start of this literal sentence."},
            {"role": "user", "content": "Preserve that marker as message content."},
        ],
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def compare_token_ids(expected: list[int], observed: list[int], window: int = 3) -> dict[str, Any]:
    if not isinstance(expected, list) or not all(type(item) is int for item in expected):
        raise ValueError("本地 expected prompt IDs 必须是整数数组")
    if not isinstance(observed, list) or not all(type(item) is int for item in observed):
        raise ValueError("服务端 prompt_token_ids 必须是整数数组")
    shared_length = min(len(expected), len(observed))
    mismatch = next(
        (index for index in range(shared_length) if expected[index] != observed[index]),
        None,
    )
    if mismatch is None and len(expected) != len(observed):
        mismatch = shared_length
    exact = mismatch is None
    details: dict[str, Any] = {
        "exact_match": exact,
        "expected_token_count": len(expected),
        "observed_token_count": len(observed),
        "expected_token_ids_sha256": token_ids_sha256(expected),
        "observed_token_ids_sha256": token_ids_sha256(observed),
        "first_mismatch_index": mismatch,
    }
    if mismatch is not None:
        start = max(0, mismatch - window)
        end = min(max(len(expected), len(observed)), mismatch + window + 1)
        details["first_mismatch_window"] = {
            "start_index": start,
            "expected_ids": expected[start:end],
            "observed_ids": observed[start:end],
        }
    return details


def read_unlabelled_input(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"id", "history"}:
                raise ValueError(f"输入第 {line_number} 行必须恰好包含 id 与 history")
            sample_id = row.get("id")
            history = row.get("history")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
                raise ValueError(f"输入第 {line_number} 行 ID 缺失或重复")
            if not isinstance(history, list) or not history:
                raise ValueError(f"输入第 {line_number} 行 History 必须为非空数组")
            for message in history:
                if (
                    not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or message.get("role") not in {"user", "assistant"}
                    or not isinstance(message.get("content"), str)
                ):
                    raise ValueError(f"输入第 {line_number} 行含非标准 History message")
            if history[-1].get("role") != "user":
                raise ValueError(f"输入第 {line_number} 行必须以当前 user 消息结束")
            seen_ids.add(sample_id)
            rows.append(row)
    if not rows or len(rows) > MAX_INPUT_SAMPLES:
        raise ValueError(f"无标签输入样本数必须在 1 到 {MAX_INPUT_SAMPLES} 之间")
    return rows


def diagnostic_samples(
    input_rows: list[dict[str, Any]], include_think_synthetic: bool = True
) -> list[dict[str, Any]]:
    rows = list(input_rows)
    if include_think_synthetic:
        existing = {row["id"] for row in rows}
        collisions = existing & {row["id"] for row in SYNTHETIC_THINK_SAMPLES}
        if collisions:
            raise ValueError(f"输入 ID 与合成样本 ID 冲突: {sorted(collisions)}")
        rows.extend(copy.deepcopy(SYNTHETIC_THINK_SAMPLES))
    if len(rows) > MAX_INPUT_SAMPLES + len(SYNTHETIC_THINK_SAMPLES):
        raise ValueError("诊断样本数超出上限")
    return rows


def build_diagnostic_payload(
    config: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[dict[str, Any], bool]:
    contains_literal_marker = any(
        message.get("role") == "assistant" and "</think>" in message.get("content", "")
        for message in messages
    )
    payload_config = config
    if contains_literal_marker:
        payload_config = copy.deepcopy(config)
        payload_config.setdefault("openai_compatible", {})["pass_reasoning_content_field"] = True
    payload = runner.build_api_payload(
        payload_config,
        messages,
        {"max_new_tokens": 1, "do_sample": False},
    )
    payload["return_token_ids"] = True
    if payload.get("model") != config["api_model"] or payload.get("max_tokens") != 1:
        raise ValueError("Token ID 诊断请求未保持冻结模型别名或单 Token 上限")
    return payload, contains_literal_marker


def validate_frozen_lora_route(config: dict[str, Any]) -> str:
    adapter_config = config.get("adapter", {})
    expected_model = adapter_config.get("expected_server_model_id")
    if not adapter_config.get("enabled") or not isinstance(expected_model, str) or not expected_model:
        raise ValueError("E02.2-A 要求启用 Full B Adapter 并配置冻结服务别名")
    if config.get("api_model") != expected_model:
        raise ValueError("API 请求模型 ID 与 Full B 冻结 LoRA 别名不一致")
    return expected_model


def prepare_local_expected_prompt(
    sample: dict[str, Any],
    prompt_text: str,
    tokenizer: Any,
    prompt_cap: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Use the Runner's shared T1 and P0/F2 message construction unchanged."""
    return runner.prepare_sample_messages(sample, prompt_text, tokenizer, prompt_cap)


def _safe_base_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path.rstrip("/"), "", ""))


def _json_error_facts(body: bytes) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "body_bytes": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        facts["body_json"] = False
        return facts
    facts["body_json"] = True
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        for field in ("type", "code"):
            if isinstance(error.get(field), (str, int)):
                facts[f"error_{field}"] = error[field]
    return facts


def request_prompt_token_ids(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    expected_ids: list[int],
    *,
    timeout: float,
) -> dict[str, Any]:
    request_payload, reasoning_extension_used = build_diagnostic_payload(config, messages)
    request_bytes = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = config.get("api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        runner.api_endpoint(config["base_url"]),
        data=request_bytes,
        headers=headers,
        method="POST",
    )
    base_result: dict[str, Any] = {
        "requested_model_id": request_payload["model"],
        "return_token_ids_requested": request_payload["return_token_ids"],
        "reasoning_content_extension_used": reasoning_extension_used,
        "expected_token_count": len(expected_ids),
        "expected_token_ids_sha256": token_ids_sha256(expected_ids),
    }
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            http_status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return {
            **base_result,
            "status": "http_error",
            "http_status": exc.code,
            **_json_error_facts(body),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            **base_result,
            "status": "connection_or_timeout_error",
            "error_type": type(exc).__name__,
        }

    result: dict[str, Any] = {
        **base_result,
        "http_status": http_status,
        "response_body_bytes": len(response_body),
        "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
    }
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {**result, "status": "invalid_json_response"}
    if not isinstance(payload, dict):
        return {**result, "status": "invalid_response_shape"}
    result["response_top_level_keys"] = sorted(payload)
    response_model = payload.get("model")
    if isinstance(response_model, str):
        result["response_model_id_observed"] = response_model
    usage = payload.get("usage")
    if isinstance(usage, dict):
        count = usage.get("prompt_tokens")
        if type(count) is int and count >= 0:
            result["server_usage_prompt_tokens"] = count
    if "prompt_token_ids" not in payload:
        return {**result, "status": "not_observable_missing_prompt_token_ids"}
    observed_ids = payload["prompt_token_ids"]
    if not isinstance(observed_ids, list) or not all(type(item) is int for item in observed_ids):
        return {**result, "status": "invalid_prompt_token_ids_type"}
    comparison = compare_token_ids(expected_ids, observed_ids)
    result.update(comparison)
    result["status"] = "exact_match" if comparison["exact_match"] else "token_ids_mismatch"
    return result


def _resolve_adapter_path(config: dict[str, Any]) -> Path:
    path = Path(config["adapter"]["path"]).expanduser()
    return (path if path.is_absolute() else WORKSPACE_ROOT / path).resolve()


def run_diagnostic(
    *,
    config_path: Path,
    input_path: Path,
    run_root: Path,
    vllm_bin: str,
    base_url: str | None = None,
    api_key: str | None = None,
    include_think_synthetic: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    input_path = input_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    if run_root.exists():
        raise FileExistsError(f"拒绝覆盖已有 E02.2-A Run 目录: {run_root}")
    if not config_path.is_file() or not input_path.is_file():
        raise FileNotFoundError("配置或无标签输入文件不存在")
    input_rows = read_unlabelled_input(input_path)
    samples = diagnostic_samples(input_rows, include_think_synthetic)
    run_root.mkdir(parents=True, exist_ok=False)

    try:
        config = runner.load_config(config_path)
        adapter_config = config["adapter"]
        expected_model = validate_frozen_lora_route(config)
        config["api_model"] = expected_model
        if base_url:
            config["base_url"] = base_url
        if api_key is not None:
            config["api_key"] = api_key
        elif os.environ.get("VLLM_API_KEY"):
            config["api_key"] = os.environ["VLLM_API_KEY"]

        cli_facts = probe.inspect_vllm_cli(
            vllm_bin,
            config.get("deployment", {}).get("vllm", {}).get("expected_version"),
        )
        base_path = runner.model_path(config)
        local_identity = probe.validate_artifact_identity(
            base_path=base_path,
            adapter_path=_resolve_adapter_path(config),
            model_id=config["model_id"],
            alias=expected_model,
            base_alias=Path(config["model_path"]).name,
            max_lora_rank=adapter_config["expected_rank"],
            expected_base_sha256=adapter_config["expected_base_weight_sha256"],
            expected_adapter_sha256=adapter_config["expected_adapter_weight_sha256"],
            expected_adapter_config_sha256=adapter_config["expected_adapter_config_sha256"],
        )
        expected_base_model = Path(config["model_path"]).name
        model_payload = probe._request_models(
            config["base_url"], config.get("api_key", ""), min(timeout or config["request_timeout_seconds"], 15.0)
        )
        server_models = probe.verify_requested_model(model_payload, expected_model)
        if expected_base_model not in server_models:
            raise ValueError("vLLM /v1/models 未发布配置的 Base served-model-name")
        prompt_text, _schema, _enums, prompt_facts = runner.load_runtime_protocol(config)
        tokenizer, tokenizer_facts = runner.load_runtime_tokenizer(config, base_path)
        prompt_cap = config["context"]["max_prompt_tokens"]
        request_timeout = timeout or float(config["request_timeout_seconds"])
        sample_results = []
        for sample in samples:
            selection, messages = prepare_local_expected_prompt(
                sample, prompt_text, tokenizer, prompt_cap
            )
            sample_result = request_prompt_token_ids(
                config,
                messages,
                selection["prompt_ids"],
                timeout=request_timeout,
            )
            sample_result.update({
                "sample_id": sample["id"],
                "sample_origin": "synthetic_think_marker" if sample["id"].startswith("e02_synthetic_think_") else "input",
                "t1_selected_start_index": selection["selected_start_index"],
                "t1_selected_history_messages": len(selection["selected_history"]),
                "prompt_floor_exceeds_cap": selection["floor_exceeds_cap"],
            })
            sample_results.append(sample_result)

        exact_count = sum(item.get("status") == "exact_match" for item in sample_results)
        mismatch_count = sum(item.get("status") == "token_ids_mismatch" for item in sample_results)
        missing_count = sum(item.get("status") == "not_observable_missing_prompt_token_ids" for item in sample_results)
        failure_count = len(sample_results) - exact_count - mismatch_count
        if exact_count == len(sample_results):
            status = "passed_exact_server_prompt_ids"
        elif mismatch_count:
            status = "failed_prompt_token_id_mismatch"
        elif exact_count == 0 and missing_count == len(sample_results):
            status = "blocked_server_prompt_token_ids_not_observable"
        else:
            status = "partial_diagnostic_requests_failed"

        vllm_environment = None
        if Path(cli_facts["executable"]).is_file():
            vllm_environment = str(Path(cli_facts["executable"]).resolve())
        report = {
            "schema_version": "e02_token_equivalence_v1",
            "run_id": run_root.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "acceptance_status": status,
            "counts": {
                "input_rows": len(input_rows),
                "synthetic_think_rows": len(samples) - len(input_rows),
                "diagnostic_requests": len(sample_results),
                "exact_match": exact_count,
                "mismatch": mismatch_count,
                "not_observable": missing_count,
                "other_failures": failure_count - missing_count,
            },
            "inputs": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "id_sha256": hashlib.sha256(
                    json.dumps([row["id"] for row in input_rows], separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "gold_or_target_read": False,
            },
            "configuration": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "p0": prompt_facts,
                "tokenizer": tokenizer_facts,
                "prompt_token_cap": prompt_cap,
                "generation_tokens": 1,
                "requested_model_id": expected_model,
                "base_model_id": expected_base_model,
                "api_base_url": _safe_base_url(config["base_url"]),
                "vllm_cli": cli_facts,
                "vllm_environment_executable": vllm_environment,
                "base_weight_sha256": local_identity["base"]["observed_weight_hash"],
                "adapter_weight_sha256": local_identity["adapter"]["weight_sha256"],
                "adapter_config_sha256": local_identity["adapter"]["config_sha256"],
                "adapter_rank": local_identity["adapter"]["config_summary"]["r"],
                "adapter_target_modules": sorted(local_identity["adapter"]["config_summary"]["target_modules"]),
                "server_model_ids": server_models,
                "server_weight_identity": "unverified",
                "diagnostic_request": {
                    "return_token_ids": True,
                    "max_tokens": 1,
                    "enable_thinking": False,
                },
            },
            "samples": sample_results,
        }
    except Exception as exc:
        report = {
            "schema_version": "e02_token_equivalence_v1",
            "run_id": run_root.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed_preflight",
            "acceptance_status": "failed_preflight",
            "failure_type": type(exc).__name__,
            "failure_summary": str(exc),
            "inputs": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "gold_or_target_read": False,
            },
            "samples": [],
        }

    manifest_path = run_root / "run_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="对已运行的 vLLM 服务执行有限 Token ID 对照")
    run_parser.add_argument("--config", type=Path, default=WORKSPACE_ROOT / "configs/deployment/e02_vllm_smoke.json")
    run_parser.add_argument("--input", required=True, type=Path, help="只含 id 与 history 的 JSONL，最多 20 条")
    run_parser.add_argument("--run-root", required=True, type=Path, help="必须是尚不存在的新目录")
    run_parser.add_argument("--vllm-bin", default="vllm", help="需指向已锁定 vLLM 环境中的 CLI")
    run_parser.add_argument("--base-url", help="默认使用冻结配置中的 OpenAI API Base URL")
    run_parser.add_argument("--api-key", default=None, help="默认读取 VLLM_API_KEY 或冻结配置")
    run_parser.add_argument("--timeout", type=float, help="单请求超时；默认沿用冻结配置")
    run_parser.add_argument(
        "--include-think-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="附加两个不含真实对话的 </think> 合成协议用例",
    )
    args = parser.parse_args(argv)
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout 必须为正数")
    report = run_diagnostic(
        config_path=args.config,
        input_path=args.input,
        run_root=args.run_root,
        vllm_bin=args.vllm_bin,
        base_url=args.base_url,
        api_key=args.api_key,
        include_think_synthetic=args.include_think_synthetic,
        timeout=args.timeout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed_exact_server_prompt_ids" else 2


if __name__ == "__main__":
    raise SystemExit(main())
