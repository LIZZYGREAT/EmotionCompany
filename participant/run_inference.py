#!/usr/bin/env python3
"""Run local Transformers or OpenAI-compatible inference and timing in one pass.

Public CLI contract: python run_inference.py TEST_FILE RESULT_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "inference_config.json"


def _find_workspace_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "configs/prompts/p0.txt").is_file() and (candidate / "src/shared/prompting.py").is_file():
            return candidate
    raise RuntimeError(
        "项目运行包缺少 configs/prompts/p0.txt 或 src/shared/prompting.py；"
        "部署目录必须包含仓库共享协议文件"
    )


WORKSPACE_ROOT = _find_workspace_root(ROOT)
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.shared.prompting import (  # noqa: E402
    build_messages,
    select_recent_complete_suffix,
    tokenizer_signature,
    validate_prompt_enum_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地模型单轮推理并计时")
    parser.add_argument("test_file", type=Path, help="具体测试 JSONL 文件")
    parser.add_argument("result_dir", type=Path, help="结果输出目录")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: JSON 解析失败: {exc.msg}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("history"), list):
                raise ValueError(f"{path}:{number}: 必须包含字符串 id 和数组 history")
            rows.append(row)
    if not rows:
        raise ValueError(f"测试文件为空: {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("测试文件包含重复 id")
    return rows


def validate_test_file(test_file: Path) -> Path:
    if not test_file.is_file():
        raise ValueError(f"测试文件不存在或不是文件: {test_file}")
    if test_file.suffix.lower() != ".jsonl":
        raise ValueError(f"测试文件必须使用 .jsonl 扩展名: {test_file}")
    return test_file


def prepare_result_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"结果路径已存在但不是目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    environment_overrides = {
        "EVALUATION_BACKEND": "backend",
        "EVALUATION_BASE_URL": "base_url",
        "EVALUATION_API_MODEL": "api_model",
        "EVALUATION_API_KEY": "api_key",
        "EVALUATION_EXPERIMENT_ID": "experiment_id",
    }
    for environment_name, config_name in environment_overrides.items():
        if environment_name in os.environ and (
            config_name == "api_key" or os.environ[environment_name]
        ):
            config[config_name] = os.environ[environment_name]
    if os.environ.get("EVALUATION_MODEL_PATH"):
        config["model_path"] = os.environ["EVALUATION_MODEL_PATH"]
    if os.environ.get("EVALUATION_HARDWARE_LABEL"):
        config["hardware_label"] = os.environ["EVALUATION_HARDWARE_LABEL"]
    if os.environ.get("EVALUATION_TOKENIZER_PATH"):
        config.setdefault("tokenizer", {})["local_path"] = os.environ["EVALUATION_TOKENIZER_PATH"]
    if "EVALUATION_ENABLE_NF4" in os.environ:
        value = os.environ["EVALUATION_ENABLE_NF4"].strip().lower()
        if value not in {"1", "true", "yes", "0", "false", "no"}:
            raise ValueError("EVALUATION_ENABLE_NF4 必须为 true/false")
        config.setdefault("quantization", {})["enabled"] = value in {"1", "true", "yes"}
    if os.environ.get("EVALUATION_MAX_PROMPT_TOKENS"):
        config.setdefault("context", {})["max_prompt_tokens"] = int(os.environ["EVALUATION_MAX_PROMPT_TOKENS"])
    if os.environ.get("EVALUATION_PROMPT_BUDGET_PROFILE"):
        config.setdefault("context", {})["budget_profile"] = os.environ["EVALUATION_PROMPT_BUDGET_PROFILE"]
    if "EVALUATION_SERVING_CONTEXT_LIMIT" in os.environ:
        value = os.environ["EVALUATION_SERVING_CONTEXT_LIMIT"].strip()
        config.setdefault("context", {})["serving_context_limit"] = int(value) if value else None
    config.setdefault("backend", "transformers")
    required = {"backend", "generation", "performance_rounds", "hardware_label", "prompt", "tokenizer", "context"}
    if not required <= set(config):
        raise ValueError(f"推理配置缺少字段: {sorted(required - set(config))}")
    if config["backend"] not in {"transformers", "openai_compatible"}:
        raise ValueError("backend 必须为 transformers 或 openai_compatible")
    if config["backend"] == "transformers" and not config.get("model_path"):
        raise ValueError("transformers 模式必须配置 model_path")
    if config["backend"] == "openai_compatible":
        missing = [name for name in ("base_url", "api_model") if not config.get(name)]
        if missing:
            raise ValueError(f"openai_compatible 模式缺少字段: {missing}")
        api_endpoint(config["base_url"])
        timeout = config.get("request_timeout_seconds", 600)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("request_timeout_seconds 必须为正数")
        if not isinstance(config.get("api_options", {}), dict):
            raise ValueError("api_options 必须为 JSON object")
    quantization = config.get("quantization", {})
    if not isinstance(quantization, dict):
        raise ValueError("quantization 必须为 JSON object")
    if quantization.get("enabled", False):
        if quantization.get("load_in_4bit", True) is not True:
            raise ValueError("启用量化时 load_in_4bit 必须为 true")
        if quantization.get("quant_type", "nf4") != "nf4":
            raise ValueError("当前仅支持显式 NF4 候选")
        if quantization.get("compute_dtype", "bfloat16") != "bfloat16":
            raise ValueError("当前 NF4 候选要求 bfloat16 compute dtype")
        if not isinstance(quantization.get("use_double_quant", True), bool):
            raise ValueError("quantization.use_double_quant 必须为 boolean")
    if not isinstance(config["performance_rounds"], int) or config["performance_rounds"] <= 0:
        raise ValueError("performance_rounds 必须为正整数")
    context = config["context"]
    prompt_cap = context.get("max_prompt_tokens")
    if not isinstance(prompt_cap, int) or isinstance(prompt_cap, bool) or prompt_cap <= 0:
        raise ValueError("context.max_prompt_tokens 必须是显式正整数")
    if not isinstance(context.get("budget_profile"), str) or not context["budget_profile"].strip():
        raise ValueError("context.budget_profile 必须显式命名")
    serving_limit = context.get("serving_context_limit")
    if serving_limit is not None and (
        not isinstance(serving_limit, int) or isinstance(serving_limit, bool) or serving_limit <= 0
    ):
        raise ValueError("context.serving_context_limit 必须为正整数或 null")
    generation_cap = config["generation"].get("max_new_tokens")
    if not isinstance(generation_cap, int) or isinstance(generation_cap, bool) or generation_cap <= 0:
        raise ValueError("generation.max_new_tokens 必须为正整数")
    if serving_limit is not None and prompt_cap + generation_cap > serving_limit:
        raise ValueError(
            f"部署预算不兼容：P({prompt_cap}) + G({generation_cap}) = "
            f"{prompt_cap + generation_cap} > K({serving_limit})"
        )
    tokenizer_config = config["tokenizer"]
    for key in ("model_id", "revision", "expected_signature_sha256", "expected_template_sha256", "files_sha256"):
        if key not in tokenizer_config:
            raise ValueError(f"tokenizer.{key} 缺失；必须锁定审计 Tokenizer")
    if tokenizer_config.get("local_files_only") is not True:
        raise ValueError("tokenizer.local_files_only 必须为 true，禁止运行时隐式下载")
    return config


def build_quantization_config(
    config: dict[str, Any],
    torch_module: Any,
    bitsandbytes_config_cls: Any | None = None,
) -> Any | None:
    """Return a real BitsAndBytesConfig only when opt-in NF4 is enabled."""
    quantization = config.get("quantization", {})
    if not quantization.get("enabled", False):
        return None
    if bitsandbytes_config_cls is None:
        from transformers import BitsAndBytesConfig

        bitsandbytes_config_cls = BitsAndBytesConfig
    dtype = getattr(torch_module, "bfloat16", None)
    if dtype is None:
        raise ValueError("当前 PyTorch 缺少 bfloat16 dtype")
    return bitsandbytes_config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=quantization.get("use_double_quant", True),
    )


def _weight_identity(model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    configured_revision = config.get("model_revision")
    if isinstance(configured_revision, str) and configured_revision.strip():
        return {
            "model_id": config.get("model_id"),
            "configured_revision": configured_revision,
            "local_path": str(model_dir),
            "fingerprint_status": "revision_configured_file_hashes_not_computed",
            "files": [],
        }
    candidates = sorted(
        path for path in model_dir.rglob("*")
        if path.is_file() and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".gguf")
            or path.name == "pytorch_model.bin"
            or path.name.startswith("pytorch_model-") and path.name.endswith(".bin")
        )
    )
    if not candidates:
        return {
            "model_id": config.get("model_id"),
            "configured_revision": None,
            "local_path": str(model_dir),
            "fingerprint_status": "no_supported_weight_files_found",
            "files": [],
        }
    files = [{"path": path.relative_to(model_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in candidates]
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "model_id": config.get("model_id"),
        "configured_revision": None,
        "local_path": str(model_dir),
        "fingerprint_status": "all_supported_weight_files_sha256",
        "aggregate_sha256": fingerprint,
        "files": files,
    }


def resolve_raw_output_path(result_dir: Path, submission_path: Path) -> Path | None:
    configured = os.environ.get("EVALUATION_RAW_OUTPUT_PATH")
    if not configured:
        return None
    raw_path = Path(configured).expanduser().resolve()
    outputs_root = (WORKSPACE_ROOT / "outputs").resolve()
    if not raw_path.is_relative_to(outputs_root):
        raise ValueError(f"原始生成文本只能写入 outputs/ 下: {raw_path}")
    if raw_path in {submission_path.resolve(), (result_dir / "performance_report.json").resolve()}:
        raise ValueError("原始诊断路径不得覆盖官方提交或性能报告")
    if raw_path.exists():
        raise FileExistsError(f"原始诊断输出已存在，拒绝覆盖: {raw_path}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    return raw_path


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def model_path(config: dict[str, Any]) -> Path:
    path = Path(config["model_path"])
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"本地模型目录不存在: {resolved}")
    return resolved.resolve()


@lru_cache(maxsize=1)
def load_submission_schema() -> dict[str, Any]:
    schema_path = ROOT / "templates" / "submission_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def submission_enums(schema: dict[str, Any] | None = None) -> dict[str, list[str]]:
    schema = schema or load_submission_schema()
    properties = schema["properties"]
    profile_properties = properties["user_profile"]["properties"]
    emotions = [value for value in properties["emotion_label"]["enum"] if value]
    return {
        "emotion_label": emotions,
        **{
            dimension: list(profile_properties[dimension]["items"]["enum"])
            for dimension in ("personality_traits", "interests", "style")
        },
    }


def load_runtime_protocol(config: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    prompt_cfg = config["prompt"]
    prompt_path = (WORKSPACE_ROOT / prompt_cfg["path"]).resolve()
    if not prompt_path.is_relative_to(WORKSPACE_ROOT) or not prompt_path.is_file():
        raise FileNotFoundError(f"P0 文本缺失或路径越界: {prompt_path}")
    prompt_bytes = prompt_path.read_bytes()
    prompt_text = prompt_bytes.decode("utf-8")
    prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
    if prompt_sha != prompt_cfg["sha256"]:
        raise RuntimeError(f"P0 SHA-256 与冻结协议不一致: {prompt_sha}")

    schema = load_submission_schema()
    enums = submission_enums(schema)
    official_prompt_schema = {
        "$defs": {
            "emotion": {"enum": enums["emotion_label"]},
            "profile": {
                "properties": {
                    dimension: {"items": {"enum": enums[dimension]}}
                    for dimension in ("personality_traits", "interests", "style")
                }
            },
        }
    }
    validate_prompt_enum_order(prompt_text, official_prompt_schema)
    return prompt_text, schema, enums, {
        "path": str(prompt_path),
        "version": prompt_cfg["version"],
        "sha256": prompt_sha,
        "tokenizer_id": config["tokenizer"]["model_id"],
        "tokenizer_revision": config["tokenizer"]["revision"],
    }


def _tokenizer_files_for_path(path: Path, expected_files: dict[str, str]) -> dict[str, str]:
    actual = {}
    for name in expected_files:
        file_path = path / name
        if file_path.is_file():
            actual[name] = sha256(file_path)
    return actual


def load_runtime_tokenizer(config: dict[str, Any], local_source: Path | None = None) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer_config = config["tokenizer"]
    model_id = tokenizer_config["model_id"]
    revision = tokenizer_config["revision"]
    local_path_value = tokenizer_config.get("local_path")
    source_path = Path(local_path_value).expanduser() if local_path_value else local_source
    if source_path is not None and not source_path.is_absolute():
        source_path = ROOT / source_path
    if source_path is not None:
        source_path = source_path.resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Tokenizer 本地目录不存在: {source_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(source_path), local_files_only=True, trust_remote_code=False
        )
        file_hashes = _tokenizer_files_for_path(source_path, tokenizer_config["files_sha256"])
        if file_hashes != tokenizer_config["files_sha256"]:
            raise RuntimeError("本地模型 Tokenizer 文件指纹与 D0.4.2 锁定版本不一致")
        source_label = str(source_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        try:
            from huggingface_hub import hf_hub_download
            file_hashes = {
                name: sha256(Path(hf_hub_download(
                    repo_id=model_id,
                    filename=name,
                    revision=revision,
                    local_files_only=True,
                )))
                for name in tokenizer_config["files_sha256"]
            }
        except Exception as exc:
            raise RuntimeError(
                f"无法从本地缓存逐文件核验锁定 Tokenizer: {type(exc).__name__}: {exc}"
            ) from exc
        source_label = f"{model_id}@{revision} (local cache only)"

    signature = tokenizer_signature(
        tokenizer,
        model_id=model_id,
        revision=revision,
        tokenizer_files=file_hashes,
    )
    if signature["template_sha256"] != tokenizer_config["expected_template_sha256"]:
        raise RuntimeError("Tokenizer Chat Template SHA-256 与 D0.4.2 锁定版本不一致")
    if signature["signature_sha256"] != tokenizer_config["expected_signature_sha256"]:
        raise RuntimeError("Tokenizer 文件/模板签名与 D0.4.2 锁定版本不一致")
    if getattr(tokenizer, "eos_token_id", None) != tokenizer_config.get("eos_token_id"):
        raise RuntimeError("Tokenizer EOS Token ID 与 D0.4.2 锁定版本不一致")
    signature["source"] = source_label
    signature["local_files_only"] = True
    return tokenizer, signature


def api_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise ValueError("base_url 必须为字符串")
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url 必须是有效的 HTTP(S) URL: {base_url}")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def api_generation_options(generation: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "temperature", "top_p", "stop", "seed", "frequency_penalty",
        "presence_penalty", "min_p", "top_k",
    }
    options = {name: value for name, value in generation.items() if name in allowed}
    if "max_new_tokens" in generation:
        options["max_tokens"] = generation["max_new_tokens"]
    elif "max_tokens" in generation:
        options["max_tokens"] = generation["max_tokens"]
    if generation.get("do_sample") is False:
        options["temperature"] = 0
    return options


class TemplateEquivalenceError(RuntimeError):
    pass


def api_message_payload(messages: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, str]]:
    api_config = config.get("openai_compatible", {})
    allow_reasoning_field = bool(api_config.get("pass_reasoning_content_field", False))
    payload_messages = []
    for message in messages:
        api_message = {"role": message["role"], "content": message["content"]}
        if message["role"] == "assistant" and "reasoning_content" in message:
            if "</think>" in message["content"] and not allow_reasoning_field:
                raise TemplateEquivalenceError(
                    "assistant history contains literal </think>; API protocol does not permit "
                    "the Qwen reasoning_content control field, so server rendering is unverified"
                )
            if allow_reasoning_field:
                # This is a vLLM/Qwen template extension, not a portable OpenAI field.
                api_message["reasoning_content"] = message["reasoning_content"]
        payload_messages.append(api_message)
    return payload_messages


def build_api_payload(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> dict[str, Any]:
    api_options = dict(config.get("api_options", {}))
    reserved = {"model", "messages", "max_tokens", "chat_template_kwargs"}
    conflicts = reserved & set(api_options)
    if conflicts:
        expected_kwargs = {"enable_thinking": False}
        if conflicts == {"chat_template_kwargs"} and api_options["chat_template_kwargs"] == expected_kwargs:
            api_options.pop("chat_template_kwargs")
        else:
            raise ValueError(f"api_options 不得覆盖协议字段: {sorted(conflicts)}")
    payload = {
        "model": config["api_model"],
        "messages": api_message_payload(messages, config),
        **api_generation_options(generation),
        **api_options,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if payload.get("max_tokens") != generation.get("max_new_tokens"):
        raise ValueError("API max_tokens 必须与 generation.max_new_tokens 一致")
    return payload


def response_content(payload: Any) -> str:
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API 响应缺少 choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError("API 响应的 message.content 必须为字符串或文本片段数组")


def _request_chat_completion_details(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    payload = build_api_payload(config, messages, generation)
    headers = {"Content-Type": "application/json"}
    api_key = config.get("api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_endpoint(config["base_url"]),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=float(config.get("request_timeout_seconds", 600)),
        ) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API 请求失败: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接本地 Base URL: {exc.reason}") from exc
    try:
        response_payload = json.loads(raw_response)
        choice = response_payload.get("choices", [{}])[0]
        usage = response_payload.get("usage", {})
        details = {
            "server_prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            "server_completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        }
        return response_content(response_payload), details
    except json.JSONDecodeError as exc:
        raise ValueError("API 响应不是有效 JSON") from exc


def request_chat_completion(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> str:
    """Compatibility wrapper retaining the original content-only helper API."""
    return _request_chat_completion_details(config, messages, generation)[0]


def extract_object(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidate = fenced.group(1) if fenced else text
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            value, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("模型输出中没有可解析的 JSON object")


def recover_response_text(text: str) -> str:
    """Preserve useful model text when its surrounding JSON is malformed."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    match = re.search(
        r'"response_text"\s*:\s*"((?:\\.|[^"\\])*)',
        cleaned,
        flags=re.S,
    )
    if match:
        try:
            recovered = json.loads(f'"{match.group(1)}"')
            if isinstance(recovered, str):
                return recovered.strip()
        except json.JSONDecodeError:
            pass
    return cleaned


def has_invalid_label(raw: dict[str, Any], enums: dict[str, list[str]]) -> bool:
    emotion = raw.get("emotion_label")
    if not isinstance(emotion, str) or emotion not in enums["emotion_label"]:
        return True
    profile = raw.get("user_profile")
    if not isinstance(profile, dict):
        return True
    for dimension in ("personality_traits", "interests", "style"):
        values = profile.get(dimension)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or not set(values) <= set(enums[dimension])
        ):
            return True
    return False


def normalize_prediction(
    sample_id: str,
    raw: dict[str, Any],
    enums: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    enums = enums or submission_enums()
    response = raw.get("response_text")
    emotion = raw.get("emotion_label")
    profile = raw.get("user_profile")
    clean_response = response.strip() if isinstance(response, str) else ""
    clean_emotion = emotion if isinstance(emotion, str) and emotion in enums["emotion_label"] else ""
    profile = profile if isinstance(profile, dict) else {}
    clean_profile: dict[str, list[str]] = {}
    for name in ("personality_traits", "interests", "style"):
        allowed = set(enums[name])
        values = profile.get(name)
        valid = (
            isinstance(values, list)
            and all(isinstance(value, str) for value in values)
            and len(values) == len(set(values))
            and set(values) <= allowed
        )
        clean_profile[name] = values if valid else []
    return {
        "id": sample_id,
        "response_text": clean_response,
        "emotion_label": clean_emotion,
        "user_profile": clean_profile,
        "memory_refs": [],
    }


def parse_prediction(
    sample_id: str,
    decoded: str,
    enums: dict[str, list[str]] | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    enums = enums or submission_enums()
    try:
        raw_prediction = extract_object(decoded)
        parse_failed = False
    except ValueError:
        raw_prediction = {
            "response_text": recover_response_text(decoded),
            "emotion_label": "",
            "user_profile": {},
            "memory_refs": [],
        }
        parse_failed = True
    invalid_label = not parse_failed and has_invalid_label(raw_prediction, enums)
    return normalize_prediction(sample_id, raw_prediction, enums), parse_failed, invalid_label


def empty_prediction(sample_id: str, enums: dict[str, list[str]] | None = None) -> dict[str, Any]:
    return normalize_prediction(sample_id, {
        "response_text": "",
        "emotion_label": "",
        "user_profile": {},
        "memory_refs": [],
    }, enums)


def synchronize(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_sample_messages(
    sample: dict[str, Any],
    prompt_text: str,
    tokenizer: Any,
    prompt_cap: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    history = sample.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"样本 {sample.get('id', '<missing-id>')} 的 history 必须为非空数组")
    selection = select_recent_complete_suffix(history, prompt_text, prompt_cap, tokenizer)
    messages = build_messages(selection["selected_history"], prompt_text)
    return selection, messages


def generate_transformers_sample(
    model: Any,
    torch_module: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    generation: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not prompt_ids:
        raise ValueError("Transformers Prompt Token IDs 不能为空")
    if hasattr(model, "get_input_embeddings"):
        input_device = model.get_input_embeddings().weight.device
    else:
        input_device = next(model.parameters()).device
    input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=input_device)
    attention_mask = torch_module.ones_like(input_ids)
    with torch_module.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation,
        )
    new_ids = output_ids[0, input_ids.shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    contains_eos = False
    if eos_id is not None:
        contains_eos = bool((new_ids == eos_id).any().item())
    details = {
        "generated_tokens": int(new_ids.shape[-1]),
        "stop_reason": (
            "eos" if contains_eos
            else "max_new_tokens" if int(new_ids.shape[-1]) >= generation.get("max_new_tokens", 0)
            else "unknown"
        ),
    }
    return decoded, details


def validate_submission_prediction(prediction: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    schema = schema or load_submission_schema()
    Draft202012Validator(schema).validate(prediction)


def _prediction_status_counts(statuses: list[str]) -> dict[str, int]:
    return {status: statuses.count(status) for status in sorted(set(statuses))}


def main() -> None:
    args = parse_args()
    test_file = validate_test_file(args.test_file.resolve())
    result_dir = args.result_dir.resolve()
    prepare_result_dir(result_dir)
    submission_path = result_dir / "submission.jsonl"
    raw_output_path = resolve_raw_output_path(result_dir, submission_path)
    config = load_config()
    run_id = os.environ.get("EVALUATION_RUN_ID", "").strip() or uuid.uuid4().hex
    experiment_id = os.environ.get("EVALUATION_EXPERIMENT_ID", config.get("experiment_id", "unspecified"))
    backend = config["backend"]
    samples = read_jsonl(test_file)
    prompt_text, submission_schema, enums, prompt_facts = load_runtime_protocol(config)
    context = config["context"]
    prompt_cap = context["max_prompt_tokens"]
    generation = dict(config["generation"])
    serving_limit = context.get("serving_context_limit")
    budget_status = (
        "not_verified" if serving_limit is None
        else "configured_preflight_pass_not_server_verified"
    )

    if backend == "transformers":
        local_model = model_path(config)
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise RuntimeError("请先安装 participant/requirements.txt") from exc

        trust_remote_code = bool(config.get("trust_remote_code", False))
        model_config = AutoConfig.from_pretrained(
            local_model,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        model_loader = (
            AutoModelForImageTextToText
            if model_config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
            else AutoModelForCausalLM
        )
        tokenizer, tokenizer_facts = load_runtime_tokenizer(config, local_model)
        weight_identity = _weight_identity(local_model, config)
        model_load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": True,
            "device_map": config.get("device_map", "auto"),
            "torch_dtype": config.get("torch_dtype", "auto"),
        }
        quantization_config = build_quantization_config(config, torch)
        if quantization_config is not None:
            model_load_kwargs["quantization_config"] = quantization_config
        model = model_loader.from_pretrained(local_model, **model_load_kwargs)
        model.eval()
        quantization_facts = (
            {
                "applied": True,
                "load_in_4bit": True,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "use_double_quant": config.get("quantization", {}).get("use_double_quant", True),
            }
            if quantization_config is not None
            else {"applied": False, "method": "legacy default model loader arguments"}
        )
        backend_report = {
            "version": "transformers_p0_f2_t1_v3",
            "timing_scope": (
                "per-sample tokenize+generate+decode+parse+schema validation; CUDA synchronized "
                "before start and after validation; model/tokenizer loading, weight fingerprinting, "
                "initialization and warmup excluded"
            ),
            "model_path_name": local_model.name,
            "model_type": model_config.model_type,
            "model_loader": model_loader.__name__,
            "model_weight_identity": weight_identity,
            "quantization": quantization_facts,
            "template_equivalence": "local_input_ids_are_the_shared_encode_inference_prompt_result",
        }
    else:
        local_tokenizer_source = None
        configured_tokenizer_path = config["tokenizer"].get("local_path")
        if configured_tokenizer_path:
            local_tokenizer_source = Path(configured_tokenizer_path).expanduser()
            if not local_tokenizer_source.is_absolute():
                local_tokenizer_source = ROOT / local_tokenizer_source
        tokenizer, tokenizer_facts = load_runtime_tokenizer(config, local_tokenizer_source)
        backend_report = {
            "version": "openai_compatible_p0_f2_t1_v3",
            "timing_scope": "client prompt tokenization+HTTP request/server inference+response decode+parse+schema validation; server startup excluded",
            "api_model": config["api_model"],
            "base_url": config["base_url"],
            "model_weight_identity": {
                "model_id": config.get("model_id"),
                "configured_revision": config.get("model_revision"),
                "fingerprint_status": "remote_or_local_server_weight_identity_not_observable",
            },
            "quantization": {"applied": False, "method": "configured by external server; not observed"},
            "template_equivalence": "BLOCKED_NOT_RUN_exact_server_token_ids_not_observable",
        }

    latencies: list[float] = []
    prediction_count = 0
    parse_failure_ids: list[str] = []
    invalid_label_ids: list[str] = []
    prompt_overflow_ids: list[str] = []
    template_equivalence_failure_ids: list[str] = []
    server_prompt_token_mismatch_ids: list[str] = []
    sample_diagnostics: list[dict[str, Any]] = []
    statuses: list[str] = []
    raw_predictions: list[dict[str, Any]] = []

    with submission_path.open("w", encoding="utf-8", newline="\n") as submission_handle:
        for index, sample in enumerate(samples, 1):
            if backend == "transformers":
                synchronize(torch)
            started = time.perf_counter()
            selection, messages = prepare_sample_messages(sample, prompt_text, tokenizer, prompt_cap)
            prompt_tokens = selection["prompt_tokens"]
            removed_messages = len(sample["history"]) - len(selection["selected_history"])
            server_details: dict[str, Any] = {
                "server_prompt_tokens": None,
                "server_completion_tokens": None,
                "finish_reason": None,
            }
            generated_tokens = 0
            stop_reason = "not_generated"
            raw_text: str | None = None
            error_type: str | None = None
            prediction: dict[str, Any]

            if selection["floor_exceeds_cap"]:
                prediction = empty_prediction(sample["id"], enums)
                status = "prompt_floor_exceeds_cap"
                prompt_overflow_ids.append(sample["id"])
                stop_reason = "not_generated_prompt_floor_exceeds_cap"
            else:
                decoded: str | None = None
                can_parse_output = False
                try:
                    if backend == "transformers":
                        decoded, generation_details = generate_transformers_sample(
                            model, torch, tokenizer, selection["prompt_ids"], generation
                        )
                        generated_tokens = generation_details["generated_tokens"]
                        stop_reason = generation_details["stop_reason"]
                        can_parse_output = True
                    else:
                        decoded, server_details = _request_chat_completion_details(config, messages, generation)
                        generated_tokens = server_details.get("server_completion_tokens")
                        finish_reason = server_details.get("finish_reason")
                        stop_reason = (
                            f"server_finish_reason:{finish_reason}"
                            if isinstance(finish_reason, str) and finish_reason
                            else "unknown_server_finish_reason"
                        )
                        server_prompt_tokens = server_details.get("server_prompt_tokens")
                        if isinstance(server_prompt_tokens, int) and server_prompt_tokens != prompt_tokens:
                            prediction = empty_prediction(sample["id"], enums)
                            status = "server_prompt_token_mismatch"
                            server_prompt_token_mismatch_ids.append(sample["id"])
                        else:
                            can_parse_output = True
                except TemplateEquivalenceError:
                    prediction = empty_prediction(sample["id"], enums)
                    status = "template_equivalence_failure"
                    template_equivalence_failure_ids.append(sample["id"])
                    stop_reason = "not_generated_template_equivalence_failure"
                except Exception as exc:
                    prediction = empty_prediction(sample["id"], enums)
                    status = "generation_failure"
                    error_type = type(exc).__name__
                    stop_reason = "not_generated_generation_failure"
                raw_text = decoded

                if can_parse_output and decoded is not None:
                    prediction, parse_failed, invalid_label = parse_prediction(sample["id"], decoded, enums)
                    if parse_failed:
                        parse_failure_ids.append(sample["id"])
                    if invalid_label:
                        invalid_label_ids.append(sample["id"])
                    status = "parse_failure" if parse_failed else ("invalid_label" if invalid_label else "success")

            validate_submission_prediction(prediction, submission_schema)
            if backend == "transformers":
                synchronize(torch)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if index <= config["performance_rounds"]:
                latencies.append(elapsed_ms)
            prediction_count += 1
            statuses.append(status)
            sample_diagnostics.append({
                "id": sample["id"],
                "status": status,
                "client_prompt_tokens": prompt_tokens,
                "prompt_cap": prompt_cap,
                "original_history_messages": len(sample["history"]),
                "retained_history_messages": len(selection["selected_history"]),
                "removed_history_messages": removed_messages,
                "generated_tokens": generated_tokens,
                "stop_reason": stop_reason,
                "server_prompt_tokens": server_details.get("server_prompt_tokens"),
                "server_completion_tokens": server_details.get("server_completion_tokens"),
            })
            if raw_output_path is not None:
                raw_predictions.append({
                    "id": sample["id"],
                    "raw_text": raw_text,
                    "status": status,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "stop_reason": stop_reason,
                    "latency_ms": round(elapsed_ms, 6),
                    "removed_history_messages": removed_messages,
                    "backend": backend,
                    "error_type": error_type,
                })
            submission_handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"[{index}/{len(samples)}] {sample['id']}: {elapsed_ms:.3f} ms ({status})", flush=True)

    if prediction_count != len(samples):
        raise RuntimeError(f"提交行数不守恒: input={len(samples)}, output={prediction_count}")

    if raw_output_path is not None:
        _write_jsonl_atomic(raw_output_path, raw_predictions)

    report = {
        **backend_report,
        "backend": backend,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "complete": len(latencies) == config["performance_rounds"],
        "required_rounds": config["performance_rounds"],
        "rounds": len(latencies),
        "average_latency_ms": round(statistics.fmean(latencies), 6),
        "median_latency_ms": round(statistics.median(latencies), 6),
        "p95_latency_ms": round(percentile95(latencies), 6),
        "min_latency_ms": round(min(latencies), 6),
        "max_latency_ms": round(max(latencies), 6),
        "latencies_ms": [round(value, 6) for value in latencies],
        "hardware_label": config["hardware_label"],
        "test_file_sha256": sha256(test_file),
        "submission_sha256": sha256(submission_path),
        "raw_predictions_sha256": sha256(raw_output_path) if raw_output_path is not None else None,
        "raw_predictions_path": str(raw_output_path) if raw_output_path is not None else None,
        "model_id": config.get("model_id"),
        "model_revision": config.get("model_revision"),
        "generation_config": generation,
        "raw_predictions_rows": len(raw_predictions) if raw_output_path is not None else None,
        "samples": prediction_count,
        "submission_rows": prediction_count,
        "submission_schema_validated": True,
        "protocol": {
            "prompt": prompt_facts,
            "tokenizer": tokenizer_facts,
            "history_strategy": "T1_recent_contiguous_complete_message_suffix",
            "history_messages_are_native": True,
            "id_in_prompt": False,
            "completion_field_order": ["emotion_label", "user_profile", "response_text", "memory_refs"],
            "submission_key_order_is_scoring_semantics": False,
            "memory_refs_policy": "always_empty_array",
        },
        "budget": {
            "P_max_prompt_tokens": prompt_cap,
            "P_profile": context["budget_profile"],
            "P_source": context.get("budget_profile_source"),
            "G_max_new_tokens": generation["max_new_tokens"],
            "G_is_final": False,
            "K_serving_context_limit": serving_limit,
            "P_plus_G": prompt_cap + generation["max_new_tokens"],
            "compatibility_status": budget_status,
            "K_must_be_verified_against_actual_server": True,
        },
        "success_count": statuses.count("success"),
        "failure_counts": _prediction_status_counts(statuses),
        "parse_failure_count": len(parse_failure_ids),
        "parse_failure_ids": parse_failure_ids,
        "invalid_label_count": len(invalid_label_ids),
        "invalid_label_ids": invalid_label_ids,
        "prompt_overflow_count": len(prompt_overflow_ids),
        "prompt_floor_exceeds_cap_count": len(prompt_overflow_ids),
        "prompt_floor_exceeds_cap_ids": prompt_overflow_ids,
        "template_equivalence_failure_count": len(template_equivalence_failure_ids),
        "template_equivalence_failure_ids": template_equivalence_failure_ids,
        "server_prompt_token_mismatch_count": len(server_prompt_token_mismatch_ids),
        "server_prompt_token_mismatch_ids": server_prompt_token_mismatch_ids,
        "server_prompt_token_usage_reported_count": sum(
            item["server_prompt_tokens"] is not None for item in sample_diagnostics
        ),
        "server_prompt_token_counts_match_client_count": sum(
            item["server_prompt_tokens"] == item["client_prompt_tokens"]
            for item in sample_diagnostics if item["server_prompt_tokens"] is not None
        ),
        "server_exact_token_id_equivalence": (
            "not_applicable_local_transformers_exact_ids_used"
            if backend == "transformers" else "BLOCKED_NOT_RUN_no_pinned_vllm_token_id_diagnostic"
        ),
        "api_reasoning_content_field_enabled": bool(config.get("openai_compatible", {}).get("pass_reasoning_content_field", False)),
        "sample_diagnostics": sample_diagnostics,
    }
    (result_dir / "performance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {submission_path}")
    print(f"PASS: {result_dir / 'performance_report.json'}")


if __name__ == "__main__":
    main()
