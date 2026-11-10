#!/opt/conda/envs/conda_env/bin/python
"""Portable vLLM launcher used by the unmodified official start.sh.

The official start.sh starts vLLM with a fixed ``--tensor-parallel-size 2``
and no LoRA flags.  This wrapper keeps start.sh byte-identical to the
official package while adding the frozen dual-Adapter routing for the image:

* force a single-GPU launch unless VLLM_TENSOR_PARALLEL_SIZE says otherwise;
* add the context/memory budget used by the frozen prompts;
* register the ``profile`` and ``emotion-response`` LoRA modules when the
  adapter directories exist.

Everything else, including the server address and served model name chosen by
start.sh, is passed through unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REAL_VLLM = os.environ.get(
    "EMOTIONALCOMPANY_REAL_VLLM", "/opt/conda/envs/conda_env/bin/vllm"
)
PROFILE_ADAPTER = Path(
    os.environ.get(
        "EMOTIONALCOMPANY_PROFILE_ADAPTER",
        "/root/participant/adapters/profile",
    )
)
EMOTION_ADAPTER = Path(
    os.environ.get(
        "EMOTIONALCOMPANY_EMOTION_ADAPTER",
        "/root/participant/adapters/emotion-response",
    )
)

VALUE_OPTIONS = {
    "--tensor-parallel-size",
    "--dtype",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--gpu-memory-utilization",
    "--max-lora-rank",
    "--max-loras",
    "--max-cpu-loras",
}


def _strip_options(args: list[str]) -> list[str]:
    """Remove options that this wrapper owns, in both --opt value and --opt=value forms."""
    cleaned: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        name = argument.split("=", 1)[0]
        if name in VALUE_OPTIONS:
            index += 1 if "=" in argument else 2
            continue
        if argument in {"--enable-lora", "--enforce-eager"}:
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned


def _extra_arguments() -> list[str]:
    arguments = [
        "--tensor-parallel-size", os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"),
        "--dtype", os.environ.get("VLLM_DTYPE", "auto"),
        "--max-model-len", os.environ.get("VLLM_MAX_MODEL_LEN", "2560"),
        "--max-num-seqs", os.environ.get("VLLM_MAX_NUM_SEQS", "1"),
        "--max-num-batched-tokens", os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "2560"),
        "--gpu-memory-utilization", os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.7"),
    ]
    if str(os.environ.get("VLLM_ENFORCE_EAGER", "")).strip().lower() in {"1", "true", "yes", "on"}:
        arguments.append("--enforce-eager")
    if PROFILE_ADAPTER.is_dir() and EMOTION_ADAPTER.is_dir():
        arguments.extend([
            "--enable-lora",
            "--lora-modules",
            f"profile={PROFILE_ADAPTER}",
            f"emotion-response={EMOTION_ADAPTER}",
            "--max-lora-rank", os.environ.get("VLLM_MAX_LORA_RANK", "8"),
            "--max-loras", os.environ.get("VLLM_MAX_LORAS", "2"),
            "--max-cpu-loras", os.environ.get("VLLM_MAX_CPU_LORAS", "2"),
        ])
    return arguments


def main() -> None:
    arguments = sys.argv[1:]
    if not arguments or arguments[0] != "serve":
        os.execv(REAL_VLLM, [REAL_VLLM, *arguments])
    cleaned = _strip_options(arguments)
    extra = _extra_arguments()
    print(
        "emotionalcompany-vllm: launching real vLLM with "
        f"adapters={'yes' if '--enable-lora' in extra else 'no'}",
        file=sys.stderr,
        flush=True,
    )
    os.execv(REAL_VLLM, [REAL_VLLM, *cleaned, *extra])


if __name__ == "__main__":
    main()
