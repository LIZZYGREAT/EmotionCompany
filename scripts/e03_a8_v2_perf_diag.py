#!/usr/bin/env python3
"""Profile the old and A8-V2 training paths on identical local Train samples."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from participant import run_inference as inference
from src.data.sft_collator import SFTDataCollator
from src.training.e03_a8_staged import (
    JsonlById,
    _base_identity,
    _base_path,
    _gradient_facts,
    _lora_parameters,
    _lora_tensor_sha256,
    _runtime_tokenizer,
    _set_seed,
    _write_json,
    _row_contract,
    _append_ledger,
    sha256_file,
)
from src.training.e03_a8_completion_loss import completion_only_forward_loss
from src.training.e03_a8_v2_staged import (
    DEFAULT_CONFIGS,
    _gradient_summary,
    _load_config as _load_v2_config,
    _resolve,
    _validate_prepared_data,
)
from src.training.e03_a8_contract import canonical_sha256


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))]


def _gpu_utilization() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip().isdigit()]
        if values:
            return {"samples_percent": values, "mean_percent": sum(values) / len(values)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


class _GpuUtilizationSampler:
    """Collect one-second utilization samples for the duration of a benchmark."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.samples: list[int] = []
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "_GpuUtilizationSampler":
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi", "--id=0", "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits", "--loop-ms=1000",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.process = None
            return self
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()
        return self

    def _collect(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            try:
                value = int(line.strip())
            except ValueError:
                continue
            self.samples.append(value)

    def __exit__(self, *_exc: Any) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)

    def facts(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        return {
            "samples_percent": list(self.samples),
            "sample_count": len(self.samples),
            "mean_percent": statistics.mean(self.samples),
            "sampling_interval_ms": 1000,
        }


def _run_path(config: dict[str, Any], rows: list[dict[str, Any]], *, selective: bool, scratch_dir: Path) -> dict[str, Any]:
    torch = __import__("torch")
    tokenizer = model = None
    try:
        tokenizer, _tokenizer_facts = _runtime_tokenizer(config)
        _set_seed(int(config["training"]["seed"]), torch)
        from src.training.training_sft import _load_train_model

        model, torch, model_facts = _load_train_model(config, _base_path(config))
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A8-V2 performance benchmark requires CUDA and BF16")
        initial_lora_sha = _lora_tensor_sha256(model, torch)
        trainable = _lora_parameters(model)
        optimizer = torch.optim.AdamW(
            [parameter for _name, parameter in trainable],
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        collator = SFTDataCollator(tokenizer.pad_token_id, max_sequence_length=2048)
        path_name = "selective" if selective else "full_logits"
        ledger_path = scratch_dir / f"{config['arm']}-{path_name}.jsonl"
        ledger_rows = []
        step_rows = []
        peaks = []
        memory_samples = []
        torch.cuda.reset_peak_memory_stats()
        _base_identity(config)
        start_all = time.monotonic()
        with JsonlById(_resolve(config["data"]["train_path"])) as indexed:
            for step_index, step in enumerate(rows):
                torch.cuda.synchronize()
                step_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                losses: list[float] | list[Any] = []
                micro_timings = []
                timing = {
                    "data_fetch_seconds": 0.0,
                    "collate_seconds": 0.0,
                    "host_to_device_seconds": 0.0,
                    "forward_enqueue_seconds": 0.0,
                    "backward_enqueue_seconds": 0.0,
                    "memory_facts_seconds": 0.0,
                    "ledger_fsync_seconds": 0.0,
                }
                completion_tokens = sequence_tokens = prompt_tokens = 0
                for sample_id in step["sample_ids"]:
                    started = time.monotonic()
                    row = indexed.get(sample_id)
                    _row_contract(row, eos_token_id=int(config["protocol"]["eos_token_id"]), prompt_cap=1280, sequence_cap=2048)
                    timing["data_fetch_seconds"] += time.monotonic() - started
                    started = time.monotonic()
                    batch = collator([row])
                    timing["collate_seconds"] += time.monotonic() - started
                    started = time.monotonic()
                    device = model.get_input_embeddings().weight.device
                    batch = {key: value.to(device) for key, value in batch.items()}
                    timing["host_to_device_seconds"] += time.monotonic() - started
                    event_begin = torch.cuda.Event(enable_timing=True)
                    event_forward = torch.cuda.Event(enable_timing=True)
                    event_backward = torch.cuda.Event(enable_timing=True)
                    event_begin.record()
                    started = time.monotonic()
                    if selective:
                        loss = completion_only_forward_loss(
                            model, batch,
                            prompt_length=int(row["prompt_length"]),
                            sequence_length=len(row["input_ids"]),
                        )
                    else:
                        loss = model(**batch).loss
                        if not bool(torch.isfinite(loss).item()):
                            raise FloatingPointError("A8 reference benchmark produced a non-finite loss")
                        losses.append(float(loss.detach().float().item()))
                    timing["forward_enqueue_seconds"] += time.monotonic() - started
                    event_forward.record()
                    if selective:
                        losses.append(loss.detach().float())
                    completion_tokens += int(row["completion_length"])
                    sequence_tokens += len(row["input_ids"])
                    prompt_tokens += int(row["prompt_length"])
                    started = time.monotonic()
                    (loss / 4).backward()
                    timing["backward_enqueue_seconds"] += time.monotonic() - started
                    event_backward.record()
                    micro_timings.append((event_begin, event_forward, event_backward))

                grad_started = time.monotonic()
                if selective:
                    diagnostic = step_index < 4 or (step_index + 1) % 32 == 0
                    gradient = _gradient_summary(trainable, torch, full=diagnostic)
                else:
                    gradient = _gradient_facts(trainable, torch)
                gradient_seconds = time.monotonic() - grad_started
                optimizer_begin = torch.cuda.Event(enable_timing=True)
                optimizer_end = torch.cuda.Event(enable_timing=True)
                optimizer_begin.record()
                optimizer_started = time.monotonic()
                optimizer.step()
                scheduler.step()
                model.zero_grad(set_to_none=True)
                optimizer_seconds = time.monotonic() - optimizer_started
                optimizer_end.record()
                torch.cuda.synchronize()
                step_seconds = time.monotonic() - step_started
                event_facts = {
                    "forward_gpu_seconds": sum(event[0].elapsed_time(event[1]) for event in micro_timings) / 1000.0,
                    "backward_gpu_seconds": sum(event[1].elapsed_time(event[2]) for event in micro_timings) / 1000.0,
                    "optimizer_gpu_seconds": optimizer_begin.elapsed_time(optimizer_end) / 1000.0,
                }
                memory = None
                if not selective:
                    memory_started = time.monotonic()
                    memory = inference.cuda_memory_facts(torch)
                    timing["memory_facts_seconds"] = time.monotonic() - memory_started
                    memory_samples.append(memory)
                timing.update({
                    "gradient_statistics_seconds": gradient_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    **event_facts,
                })
                step_rows.append({"step": step_index + 1, "wall_seconds": step_seconds, **timing})
                ledger_rows.append({
                    "arm": config["arm"],
                    "optimizer_step": step_index + 1,
                    "sample_ids": step["sample_ids"],
                    "completion_tokens": completion_tokens,
                    "sequence_tokens": sequence_tokens,
                    "prompt_tokens": prompt_tokens,
                    "gradient": gradient,
                    "timing": timing,
                    "loss_tensors": list(losses) if selective else [],
                })
                if not selective:
                    log_started = time.monotonic()
                    _append_ledger(ledger_path, {**ledger_rows[-1], "mean_loss": sum(losses) / len(losses)})
                    timing["ledger_fsync_seconds"] = time.monotonic() - log_started
                    step_rows[-1]["ledger_fsync_seconds"] = timing["ledger_fsync_seconds"]
                peaks.append(torch.cuda.max_memory_allocated())
        if selective:
            losses_flat = torch.stack([loss for row in ledger_rows for loss in row.pop("loss_tensors")]).detach().cpu().tolist()
            cursor = 0
            for row in ledger_rows:
                step_losses = [float(value) for value in losses_flat[cursor:cursor + 4]]
                cursor += 4
                row["micro_batch_losses"] = step_losses
                row["mean_loss"] = sum(step_losses) / len(step_losses)
        torch.cuda.synchronize()
        total_compute_seconds = time.monotonic() - start_all
        logger_seconds = 0.0
        if selective:
            payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for row in ledger_rows)
            log_started = time.monotonic()
            fd = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                offset = 0
                while offset < len(payload):
                    offset += os.write(fd, payload[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
            logger_seconds = time.monotonic() - log_started
            amortized_fsync_seconds = logger_seconds / len(rows)
            for timing_row in step_rows:
                timing_row["ledger_fsync_seconds"] = amortized_fsync_seconds
                timing_row["memory_facts_seconds"] = 0.0
        total_seconds = total_compute_seconds + logger_seconds
        memory = inference.cuda_memory_facts(torch)
        peaks.append(int(memory.get("allocator_peak_allocated_bytes", 0)))
        final_lora_sha = _lora_tensor_sha256(model, torch)
        if final_lora_sha == initial_lora_sha:
            raise RuntimeError("A8-V2 benchmark did not execute optimizer updates")
        return {
            "path": path_name,
            "initial_lora_tensor_sha256": initial_lora_sha,
            "final_lora_tensor_sha256": final_lora_sha,
            "optimizer_steps": len(rows),
            "samples_seen": len(rows) * 4,
            "completion_tokens": sum(row["completion_tokens"] for row in ledger_rows),
            "sequence_tokens": sum(row["sequence_tokens"] for row in ledger_rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in ledger_rows),
            "compute_seconds": total_compute_seconds,
            "logger_seconds": logger_seconds,
            "total_seconds_including_logger": total_seconds,
            "mean_seconds_per_step_including_logger": total_seconds / len(rows),
            "step_wall_seconds": {
                "mean": statistics.mean(row["wall_seconds"] for row in step_rows),
                "p50": _percentile([row["wall_seconds"] for row in step_rows], 0.5),
                "p90": _percentile([row["wall_seconds"] for row in step_rows], 0.9),
                "max": max(row["wall_seconds"] for row in step_rows),
            },
            "step_timing_components_mean_seconds": {
                key: statistics.mean(row[key] for row in step_rows)
                for key in (
                    "data_fetch_seconds", "collate_seconds", "host_to_device_seconds",
                    "forward_enqueue_seconds", "backward_enqueue_seconds", "gradient_statistics_seconds",
                    "optimizer_seconds", "memory_facts_seconds", "ledger_fsync_seconds",
                    "forward_gpu_seconds", "backward_gpu_seconds", "optimizer_gpu_seconds",
                )
            },
            "memory": memory,
            "peak_allocated_bytes": max(peaks, default=0),
            "model_facts": model_facts,
            "step_rows": step_rows,
        }
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_benchmark(arm: str, run_id: str, config_path: Path | None = None) -> dict[str, Any]:
    if arm not in {"P", "EP"}:
        raise ValueError("arm must be P or EP")
    v2_config_path = config_path or DEFAULT_CONFIGS[arm]
    config, _ = _load_v2_config(v2_config_path, arm)
    data_facts = _validate_prepared_data(config)
    schedule = data_facts["schedule"]
    rows = schedule[:16]
    ids = [sid for step in rows for sid in step["sample_ids"]]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("A8-V2 benchmark requires 16 optimizer steps and 64 unique sample IDs")
    output_dir = _resolve(config["performance_diagnostics"]["benchmark_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{arm}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite performance benchmark: {output_path}")
    utilization_before = _gpu_utilization()
    with tempfile.TemporaryDirectory(prefix="e03-a8-v2-benchmark-") as temporary, _GpuUtilizationSampler() as utilization_sampler:
        scratch = Path(temporary)
        full = _run_path(config, rows, selective=False, scratch_dir=scratch)
        selective = _run_path(config, rows, selective=True, scratch_dir=scratch)
    utilization_during = utilization_sampler.facts()
    utilization_after = _gpu_utilization()
    speedup = full["total_seconds_including_logger"] / max(selective["total_seconds_including_logger"], 1e-12)
    total_memory_bytes = min(
        int(full["memory"].get("total_memory_bytes", 0)),
        int(selective["memory"].get("total_memory_bytes", 0)),
    )
    memory_limit_bytes = int(total_memory_bytes * float(config["quantization"]["max_cuda_memory_fraction"]))
    same_gpu = (
        full["memory"].get("gpu_name") == selective["memory"].get("gpu_name")
        and full["memory"].get("device_index") == selective["memory"].get("device_index")
        and full["memory"].get("total_memory_bytes") == selective["memory"].get("total_memory_bytes")
        and total_memory_bytes > 0
    )
    passed = (
        full["initial_lora_tensor_sha256"] == selective["initial_lora_tensor_sha256"]
        and full["samples_seen"] == selective["samples_seen"]
        and full["completion_tokens"] == selective["completion_tokens"]
        and full["sequence_tokens"] == selective["sequence_tokens"]
        and same_gpu
        and speedup >= float(config["performance_diagnostics"]["minimum_speedup"])
        and max(full["peak_allocated_bytes"], selective["peak_allocated_bytes"]) <= memory_limit_bytes
    )
    result = {
        "status": "passed" if passed else "failed",
        "experiment_id": "E03-A8-V2",
        "arm": arm,
        "run_id": run_id,
        "benchmark_steps_per_path": 16,
        "samples_per_path": 64,
        "sample_ids_sha256": canonical_sha256(ids),
        "sample_ids_first": ids,
        "old_full_logits_path": full,
        "selective_completion_path": selective,
        "selective_speedup": speedup,
        "minimum_required_speedup": float(config["performance_diagnostics"]["minimum_speedup"]),
        "gpu_utilization_before": utilization_before,
        "gpu_utilization_during": utilization_during,
        "gpu_utilization_after": utilization_after,
        "hardware_match": {
            "same_gpu_name": same_gpu,
            "gpu_name": full["memory"].get("gpu_name"),
            "total_memory_bytes": total_memory_bytes,
            "allocator_peak_limit_bytes": memory_limit_bytes,
        },
        "measurement_scope": "same 64 A8-V2 IDs and encoded rows, same optimizer settings, fresh same-seed LoRA initialization; only the Causal LM full-logit vs selective Completion projection path and associated sync/log policies differ; each step is synchronized for wall timing; full-logit path includes per-step complete gradient facts, CUDA memory facts and fsynced ledger; selective path batches scalar/device details and fsyncs once after 16 steps.",
        "limitations": "Benchmark consumes 16 optimizer steps only in ephemeral in-memory models; no adapter/checkpoint or official Run state is written.",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output_path, result, exclusive=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose and benchmark A8-V2 training performance.")
    parser.add_argument("--arm", choices=("P", "EP"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    result = run_benchmark(args.arm, args.run_id, args.config)
    print(json.dumps({
        "status": result["status"],
        "arm": result["arm"],
        "benchmark_steps_per_path": result["benchmark_steps_per_path"],
        "old_seconds_per_step": result["old_full_logits_path"]["mean_seconds_per_step_including_logger"],
        "selective_seconds_per_step": result["selective_completion_path"]["mean_seconds_per_step_including_logger"],
        "selective_speedup": result["selective_speedup"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
