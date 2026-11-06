"""Frozen E01-B representation features and CPU 33-label linear probes."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from src.data.e03_a5_labels import DIMENSIONS
from src.training.e03_a5_multitask import FinalHiddenCapture, forward_with_prompt_end_hidden


def extract_prompt_features(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int,
    batch_size: int = 1,
    max_prompt_tokens: int = 1280,
) -> tuple[list[str], Any, str]:
    """Run prompt-only inputs and gather the last Prompt token, without Gold suffixes."""
    import torch

    sample_ids: list[str] = []
    collected: list[Any] = []
    observed_dtypes: set[str] = set()
    for batch_ids, batch_features, dtype in iter_prompt_features(
        model, rows, pad_token_id=pad_token_id, batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
    ):
        sample_ids.extend(batch_ids)
        collected.append(batch_features)
        observed_dtypes.add(dtype)
    if not collected:
        raise ValueError("feature extraction requires at least one row")
    if len(observed_dtypes) != 1:
        raise ValueError(f"model changed feature dtype between batches: {sorted(observed_dtypes)}")
    features = torch.cat(collected, dim=0)
    return sample_ids, features, next(iter(observed_dtypes))


def iter_prompt_features(
    model: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    pad_token_id: int,
    batch_size: int = 1,
    max_prompt_tokens: int = 1280,
) -> Iterable[tuple[list[str], Any, str]]:
    """Stream Prompt-only feature batches to keep tokenized Train rows out of memory."""
    import torch

    if batch_size < 1 or max_prompt_tokens < 1:
        raise ValueError("batch_size and max_prompt_tokens must be positive")
    sample_ids: list[str] = []
    prompt_sequences: list[list[int]] = []
    seen_ids: set[str] = set()

    def emit_batch(ids: list[str], sequences: list[list[int]], device: Any, capture: Any) -> tuple[list[str], Any, str]:
        lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
        padded_length = int(lengths.max().item())
        input_ids = torch.full((len(sequences), padded_length), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, :len(sequence)] = 1
        outputs, prompt_hidden = forward_with_prompt_end_hidden(
            model, capture, input_ids.to(device), attention_mask.to(device), lengths.to(device)
        )
        if not hasattr(outputs, "logits"):
            raise ValueError("causal model output must expose LM logits during feature extraction")
        features = prompt_hidden.detach().to(device="cpu").contiguous()
        return ids, features, str(features.dtype).removeprefix("torch.")

    device = next(model.parameters()).device
    model.eval()
    with FinalHiddenCapture(model) as capture, torch.inference_mode():
        for row in rows:
            sample_id = row.get("sample_id")
            input_ids = row.get("input_ids")
            prompt_length = row.get("prompt_length")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("feature row requires a non-empty sample_id")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate feature sample_id: {sample_id}")
            if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or not 0 < prompt_length <= max_prompt_tokens:
                raise ValueError(f"{sample_id}: invalid or over-budget prompt_length")
            if not isinstance(input_ids, list) or prompt_length >= len(input_ids):
                raise ValueError(f"{sample_id}: SFT input_ids do not contain a completion suffix")
            tokens = input_ids[:prompt_length]
            if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
                raise ValueError(f"{sample_id}: Prompt IDs must be non-negative integers")
            sample_ids.append(sample_id)
            prompt_sequences.append(tokens)
            seen_ids.add(sample_id)
            if len(sample_ids) == batch_size:
                yield emit_batch(sample_ids, prompt_sequences, device, capture)
                sample_ids, prompt_sequences = [], []
        if sample_ids:
            yield emit_batch(sample_ids, prompt_sequences, device, capture)


def _as_float_tensor(value: Any, *, name: str) -> Any:
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.ndim != 2 or tensor.shape[1] != 33:
        raise ValueError(f"{name} must have shape [rows, 33]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def validate_probe_arrays(features: Any, targets: Any, masks: Any) -> tuple[Any, Any, Any]:
    import torch

    x = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty [rows, hidden_size] matrix")
    if not torch.isfinite(x).all():
        raise ValueError("features contain non-finite values")
    y = _as_float_tensor(targets, name="targets")
    mask = _as_float_tensor(masks, name="masks")
    if y.shape != mask.shape or y.shape[0] != x.shape[0]:
        raise ValueError("features, targets, and masks must have the same row count")
    if not torch.logical_or(y == 0, y == 1).all():
        raise ValueError("targets must be binary")
    if not torch.logical_or(mask == 0, mask == 1).all():
        raise ValueError("masks must be binary")
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("each probe sample must have at least one supervised label")
    return x, y, mask


def fit_linear_probe(
    features: Any,
    targets: Any,
    masks: Any,
    *,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 0.01,
) -> tuple[Any, list[float]]:
    """Fit only a CPU FP32 linear head on the supplied Train-Fit rows."""
    import torch
    from torch import nn

    x, y, mask = validate_probe_arrays(features, targets, masks)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("epochs, batch_size, and learning_rate must be finite and positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = nn.Linear(x.shape[1], 33, bias=True, dtype=torch.float32, device="cpu")
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    losses: list[float] = []
    head.train()
    for _epoch in range(epochs):
        permutation = torch.randperm(x.shape[0], generator=generator)
        weighted_loss = 0.0
        seen = 0
        for start in range(0, x.shape[0], batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = head(x[indices])
            loss = _masked_bce(logits, y[indices], mask[indices])
            loss.backward()
            optimizer.step()
            count = int(indices.numel())
            weighted_loss += float(loss.detach().item()) * count
            seen += count
        losses.append(weighted_loss / seen)
    head.eval()
    return head, losses


def _masked_bce(logits: Any, targets: Any, masks: Any) -> Any:
    from src.training.e03_a5_multitask import masked_bce_with_logits

    return masked_bce_with_logits(logits, targets, masks)


def frequency_prior(targets: Any, masks: Any) -> Any:
    import torch

    y = _as_float_tensor(targets, name="targets")
    mask = _as_float_tensor(masks, name="masks")
    if y.shape != mask.shape:
        raise ValueError("frequency-prior target and mask shapes differ")
    support = mask.sum(dim=0)
    if torch.any(support <= 0):
        raise ValueError("frequency prior has a label with no supervised Fit rows")
    return (y * mask).sum(dim=0) / support


def choose_global_threshold(probabilities: Any, targets: Any, masks: Any) -> dict[str, float]:
    """Select one global threshold using only the supplied Train-Calibration rows."""
    import torch

    probs = _as_float_tensor(probabilities, name="probabilities")
    y = _as_float_tensor(targets, name="targets")
    mask = _as_float_tensor(masks, name="masks")
    if probs.shape != y.shape or y.shape != mask.shape:
        raise ValueError("threshold calibration arrays must share [rows, 33] shape")
    thresholds = [index / 100 for index in range(5, 96)]
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    best: tuple[float, float, float, float] | None = None
    best_result: dict[str, float] | None = None
    y_bool, mask_bool = y.bool(), mask.bool()
    total_positive = int(((y_bool & mask_bool).sum()).item())
    for threshold in thresholds:
        predicted = probs >= threshold
        tp = int((predicted & y_bool & mask_bool).sum().item())
        fp = int((predicted & ~y_bool & mask_bool).sum().item())
        fn = total_positive - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rank = (f1, precision, -abs(threshold - 0.5), threshold)
        if best is None or rank > best:
            best = rank
            best_result = {
                "threshold": threshold,
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
    if best_result is None:
        raise RuntimeError("threshold search produced no candidates")
    return best_result


def _profiles_from_bits(bits: Any, enums: Mapping[str, Sequence[str]]) -> list[dict[str, list[str]]]:
    if hasattr(bits, "detach"):
        bits = bits.detach().cpu().tolist()
    profiles = []
    for row in bits:
        if len(row) != 33:
            raise ValueError("profile prediction vector must contain 33 labels")
        profile: dict[str, list[str]] = {}
        cursor = 0
        for dimension in DIMENSIONS:
            labels = list(enums[dimension])
            profile[dimension] = [label for label, value in zip(labels, row[cursor:cursor + len(labels)]) if value]
            cursor += len(labels)
        profiles.append(profile)
    return profiles


def evaluate_probabilities(
    probabilities: Any,
    targets: Any,
    masks: Any,
    enums: Mapping[str, Sequence[str]],
    *,
    threshold: float,
) -> dict[str, Any]:
    import torch
    from src.evaluation.metrics import profile_metrics

    probs = _as_float_tensor(probabilities, name="probabilities")
    y = _as_float_tensor(targets, name="targets")
    mask = _as_float_tensor(masks, name="masks")
    if probs.shape != y.shape or y.shape != mask.shape:
        raise ValueError("evaluation arrays must share [rows, 33] shape")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    prediction_bits = ((probs >= threshold) & mask.bool()).to(dtype=torch.int32)
    gold_bits = (y * mask).to(dtype=torch.int32)
    metrics = profile_metrics(
        _profiles_from_bits(gold_bits, enums),
        _profiles_from_bits(prediction_bits, enums),
        {dimension: list(enums[dimension]) for dimension in DIMENSIONS},
    )
    per_label = []
    cursor = 0
    for dimension in DIMENSIONS:
        for label in enums[dimension]:
            gold = gold_bits[:, cursor].bool()
            predicted = prediction_bits[:, cursor].bool()
            tp = int((gold & predicted).sum().item())
            fp = int((~gold & predicted & mask[:, cursor].bool()).sum().item())
            fn = int((gold & ~predicted).sum().item())
            support = int(gold.sum().item())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_label.append({
                "label": f"{dimension}:{label}",
                "positive_support": support,
                "predicted_positive_count": int(predicted.sum().item()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "status": "supported" if support else "insufficient_support",
            })
            cursor += 1
    return {"threshold": threshold, "metrics": metrics, "per_label": per_label}


def fit_and_evaluate_probe(
    features_by_id: Mapping[str, Any],
    supervision_by_id: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, Sequence[str]],
    enums: Mapping[str, Sequence[str]],
    *,
    seed: int = 42,
    shuffled_seed: int = 314159,
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 0.01,
) -> tuple[Any, dict[str, Any]]:
    """Fit on Fit; calibrate thresholds on Calibration; evaluate on Holdout."""
    import torch

    names = ("fit", "calibration", "internal_holdout")
    for split_name in names:
        if split_name not in splits or not splits[split_name]:
            raise ValueError(f"missing or empty frozen split: {split_name}")
    all_ids = [sample_id for split_name in names for sample_id in splits[split_name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("frozen split sample IDs overlap")
    if set(all_ids) != set(features_by_id):
        raise ValueError("feature cache IDs do not exactly equal the frozen Train split IDs")
    if set(all_ids) - set(supervision_by_id):
        raise ValueError("frozen split contains IDs without A5 supervision")

    def arrays(split_name: str) -> tuple[Any, Any, Any]:
        ids = list(splits[split_name])
        feature_rows = torch.stack([torch.as_tensor(features_by_id[item], dtype=torch.float32) for item in ids])
        target_rows = [supervision_by_id[item]["y"] for item in ids]
        mask_rows = [supervision_by_id[item]["mask"] for item in ids]
        return validate_probe_arrays(feature_rows, target_rows, mask_rows)

    x_fit, y_fit, m_fit = arrays("fit")
    x_cal, y_cal, m_cal = arrays("calibration")
    x_holdout, y_holdout, m_holdout = arrays("internal_holdout")
    head, losses = fit_linear_probe(
        x_fit, y_fit, m_fit, seed=seed, epochs=epochs,
        batch_size=batch_size, learning_rate=learning_rate,
    )
    with torch.no_grad():
        cal_probabilities = head(x_cal).sigmoid()
        holdout_probabilities = head(x_holdout).sigmoid()
    calibrated = choose_global_threshold(cal_probabilities, y_cal, m_cal)
    fit_prior = frequency_prior(y_fit, m_fit)
    prior_probs = fit_prior.unsqueeze(0).expand(x_cal.shape[0], -1)
    prior_threshold = choose_global_threshold(prior_probs, y_cal, m_cal)
    prior_holdout_probs = fit_prior.unsqueeze(0).expand(x_holdout.shape[0], -1)

    shuffled_generator = torch.Generator(device="cpu")
    shuffled_generator.manual_seed(shuffled_seed)
    shuffle_indices = torch.randperm(x_fit.shape[0], generator=shuffled_generator)
    shuffled_head, shuffled_losses = fit_linear_probe(
        x_fit, y_fit[shuffle_indices], m_fit[shuffle_indices], seed=seed,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    )
    with torch.no_grad():
        shuffled_cal_probs = shuffled_head(x_cal).sigmoid()
        shuffled_holdout_probs = shuffled_head(x_holdout).sigmoid()
    shuffled_threshold = choose_global_threshold(shuffled_cal_probs, y_cal, m_cal)

    report = {
        "status": "probe_metrics_ready_for_gate_review",
        "fit_only_head_training": True,
        "calibration_only_threshold_selection": True,
        "internal_holdout_used_for_model_selection": False,
        "fit_sample_count": int(x_fit.shape[0]),
        "calibration_sample_count": int(x_cal.shape[0]),
        "internal_holdout_sample_count": int(x_holdout.shape[0]),
        "feature_hidden_size": int(x_fit.shape[1]),
        "head_parameter_dtype": "float32",
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "first_epoch_bce": losses[0],
            "last_epoch_bce": losses[-1],
            "loss_history": losses,
        },
        "thresholds": {
            "linear_head": calibrated,
            "frequency_prior": prior_threshold,
            "shuffled_label_control": shuffled_threshold,
        },
        "holdout": {
            "linear_head_threshold_0_5": evaluate_probabilities(
                holdout_probabilities, y_holdout, m_holdout, enums, threshold=0.5
            ),
            "linear_head_calibrated": evaluate_probabilities(
                holdout_probabilities, y_holdout, m_holdout, enums,
                threshold=calibrated["threshold"],
            ),
            "frequency_prior_threshold_0_5": evaluate_probabilities(
                prior_holdout_probs, y_holdout, m_holdout, enums, threshold=0.5
            ),
            "frequency_prior_calibrated": evaluate_probabilities(
                prior_holdout_probs, y_holdout, m_holdout, enums,
                threshold=prior_threshold["threshold"],
            ),
            "shuffled_label_control_calibrated": evaluate_probabilities(
                shuffled_holdout_probs, y_holdout, m_holdout, enums,
                threshold=shuffled_threshold["threshold"],
            ),
        },
        "shuffled_label_control": {
            "seed": shuffled_seed,
            "fit_labels_permuted_across_sample_ids": True,
            "first_epoch_bce": shuffled_losses[0],
            "last_epoch_bce": shuffled_losses[-1],
        },
        "dev_224_status": "not_run",
        "probe_decision": "requires_predeclared_gate_and_manual_review",
    }
    return head, report
