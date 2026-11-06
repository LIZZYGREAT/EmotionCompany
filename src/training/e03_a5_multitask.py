"""Torch-only loss and Prompt-end hidden-state helpers for E03-A5."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


LABEL_COUNT = 33


def build_a5_train_schedule(
    fit_sample_ids: Sequence[str],
    *,
    optimizer_steps: int = 128,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Freeze a deterministic, Train-Fit-only, without-replacement paired schedule."""
    from src.data.e03_a5_labels import canonical_sha256

    if any(not isinstance(value, str) or not value for value in fit_sample_ids):
        raise ValueError("Fit sample IDs must be non-empty strings")
    if len(fit_sample_ids) != len(set(fit_sample_ids)):
        raise ValueError("Fit sample IDs must be unique before schedule construction")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (optimizer_steps, micro_batch_size, gradient_accumulation_steps)
    ):
        raise ValueError("schedule steps, microbatch, and accumulation must be positive integers")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("schedule seed must be an integer")
    draw_count = optimizer_steps * micro_batch_size * gradient_accumulation_steps
    if draw_count > len(fit_sample_ids):
        raise ValueError("Fit partition is too small for a no-replacement A5 schedule")
    ordered_fit_ids = sorted(fit_sample_ids)
    draws = random.Random(seed).sample(ordered_fit_ids, draw_count)
    schedule = []
    cursor = 0
    for step in range(1, optimizer_steps + 1):
        microbatches = []
        for _ in range(gradient_accumulation_steps):
            ids = draws[cursor:cursor + micro_batch_size]
            cursor += micro_batch_size
            microbatches.append(ids)
        schedule.append({"optimizer_step": step, "microbatches": microbatches})
    flattened = [sample_id for step in schedule for batch in step["microbatches"] for sample_id in batch]
    if len(flattened) != draw_count or len(flattened) != len(set(flattened)):
        raise AssertionError("internal A5 schedule contains a repeat or lost draw")
    return {
        "version": "e03_a5_fit_without_replacement_v1",
        "seed": seed,
        "optimizer_steps": optimizer_steps,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "draw_count": draw_count,
        "source_split": "train_fit",
        "sample_ids": flattened,
        "sample_ids_sha256": canonical_sha256(flattened),
        "schedule": schedule,
        "schedule_sha256": canonical_sha256(schedule),
        "unique_draws": True,
    }


def masked_bce_with_logits(logits: Any, targets: Any, mask: Any) -> Any:
    """Per-sample normalized BCE in float32; every sample must supervise a label."""
    import torch
    import torch.nn.functional as F

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != LABEL_COUNT:
        raise ValueError("BCE logits must have shape [batch, 33]")
    targets = torch.as_tensor(targets, device=logits.device)
    mask = torch.as_tensor(mask, device=logits.device)
    if targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("BCE target and mask shapes must match [batch, 33] logits")
    if not torch.isfinite(targets).all() or not torch.isfinite(mask).all():
        raise ValueError("BCE target and mask must be finite")
    if not torch.logical_or(targets == 0, targets == 1).all():
        raise ValueError("BCE targets must be binary")
    if not torch.logical_or(mask == 0, mask == 1).all():
        raise ValueError("BCE mask must be binary")
    mask32 = mask.to(dtype=torch.float32)
    target32 = targets.to(dtype=torch.float32)
    valid_counts = mask32.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("every BCE sample must have at least one supervised label")
    element_losses = F.binary_cross_entropy_with_logits(
        logits.to(dtype=torch.float32), target32, reduction="none"
    )
    per_sample = (element_losses * mask32).sum(dim=1) / valid_counts
    loss = per_sample.mean()
    if not torch.isfinite(loss):
        raise ValueError("BCE loss is non-finite")
    return loss


def completion_only_sft_loss(logits: Any, labels: Any) -> Any:
    """Standard shifted causal CE with the existing ``-100`` Prompt mask."""
    import torch
    import torch.nn.functional as F

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("causal LM logits must have shape [batch, sequence, vocabulary]")
    labels = torch.as_tensor(labels, device=logits.device)
    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError("SFT labels must align with the [batch, sequence] logits prefix")
    shifted_logits = logits[:, :-1, :].contiguous().to(dtype=torch.float32)
    shifted_labels = labels[:, 1:].contiguous()
    if not torch.any(shifted_labels != -100):
        raise ValueError("SFT batch contains no supervised Completion tokens")
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


def combine_sft_bce_losses(sft_loss: Any, bce_loss: Any, bce_lambda: float) -> Any:
    """Return SFT loss unchanged for the Control arm, otherwise add weighted BCE."""
    if (
        isinstance(bce_lambda, bool)
        or not isinstance(bce_lambda, (int, float))
        or not math.isfinite(float(bce_lambda))
        or bce_lambda < 0
    ):
        raise ValueError("BCE lambda must be a non-negative finite number")
    if bce_lambda == 0:
        return sft_loss
    if bce_loss is None:
        raise ValueError("candidate arm requires a computed BCE loss")
    return sft_loss + float(bce_lambda) * bce_loss


def gather_prompt_end_hidden(hidden: Any, prompt_lengths: Any, attention_mask: Any) -> Any:
    """Gather final hidden states at each row's last Prompt token (p-1)."""
    import torch

    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("hidden states must have shape [batch, sequence, hidden]")
    attention = torch.as_tensor(attention_mask, device=hidden.device)
    lengths = torch.as_tensor(prompt_lengths, device=hidden.device, dtype=torch.long)
    if attention.ndim != 2 or attention.shape != hidden.shape[:2]:
        raise ValueError("attention_mask must align with hidden-state batch and sequence")
    if lengths.ndim != 1 or lengths.shape[0] != hidden.shape[0]:
        raise ValueError("prompt_lengths must have one value per batch row")
    if torch.any(lengths <= 0) or torch.any(lengths > hidden.shape[1]):
        raise ValueError("prompt_length must identify a token inside the hidden-state sequence")
    positions = lengths - 1
    row_indices = torch.arange(hidden.shape[0], device=hidden.device)
    if torch.any(attention[row_indices, positions] != 1):
        raise ValueError("prompt_length - 1 points at padding or a masked token")
    sequence_positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    prompt_prefix = sequence_positions < lengths.unsqueeze(1)
    if torch.any((attention == 0) & prompt_prefix):
        raise ValueError("Prompt prefix must be contiguous and unpadded on the left")
    return hidden[row_indices, positions, :]


def find_final_norm_module(model: Any) -> tuple[str, Any]:
    """Find Qwen's final model norm so a hook captures only the final layer output."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = (
        ("model.norm", getattr(getattr(base, "model", None), "norm", None)),
        ("transformer.ln_f", getattr(getattr(base, "transformer", None), "ln_f", None)),
        ("model.final_layernorm", getattr(getattr(base, "model", None), "final_layernorm", None)),
    )
    for name, module in candidates:
        if module is not None:
            return name, module
    raise ValueError("unable to locate the final hidden-state normalization module")


@dataclass
class FinalHiddenCapture:
    """Forward hook that retains the final normalized hidden tensor and its graph."""

    model: Any

    def __post_init__(self) -> None:
        self.module_name, module = find_final_norm_module(self.model)
        self.hidden = None
        self.call_count = 0

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - construction requires Torch
                raise RuntimeError("Torch is required for final hidden capture") from exc
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError("final norm hook output must be [batch, sequence, hidden]")
            self.hidden = value
            self.call_count += 1

        self._hook = module.register_forward_hook(capture)

    def clear(self) -> None:
        self.hidden = None
        self.call_count = 0

    def take(self) -> Any:
        if self.hidden is None:
            raise RuntimeError("model forward did not invoke the captured final norm")
        if self.call_count != 1:
            raise RuntimeError(f"expected one final norm call per forward, observed {self.call_count}")
        value = self.hidden
        self.clear()
        return value

    def close(self) -> None:
        self._hook.remove()

    def __enter__(self) -> "FinalHiddenCapture":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def forward_with_prompt_end_hidden(
    model: Any,
    capture: FinalHiddenCapture,
    input_ids: Any,
    attention_mask: Any,
    prompt_lengths: Any,
    **model_kwargs: Any,
) -> tuple[Any, Any]:
    """Use one causal forward for LM logits and Prompt-end BCE features."""
    capture.clear()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
        **model_kwargs,
    )
    final_hidden = capture.take()
    prompt_hidden = gather_prompt_end_hidden(final_hidden, prompt_lengths, attention_mask)
    return outputs, prompt_hidden


def assert_finite_nonzero_gradients(parameters: Sequence[Any], *, description: str) -> dict[str, float | int]:
    """Summarize nonzero finite gradients for a trainable parameter group."""
    import torch

    seen = nonzero = 0
    squared_norm = 0.0
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        seen += 1
        gradient = parameter.grad
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise ValueError(f"{description} has a non-finite gradient")
        norm = float(torch.linalg.vector_norm(gradient.detach().to(dtype=torch.float32)).item())
        squared_norm += norm * norm
        if norm > 0:
            nonzero += 1
    if seen == 0 or nonzero == 0:
        raise ValueError(f"{description} has no nonzero trainable gradients")
    return {
        "trainable_parameter_count": seen,
        "parameters_with_nonzero_gradient": nonzero,
        "gradient_norm": squared_norm ** 0.5,
    }
