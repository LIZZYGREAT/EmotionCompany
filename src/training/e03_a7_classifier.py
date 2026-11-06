"""Prompt-end classifier heads, losses, and LoRA-backed backbone helpers for E03-A7."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


PROFILE_LABEL_COUNT = 33
EMOTION_LABEL_COUNT = 16
ARMS = ("P", "EP")


def validate_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("E03-A7 arm must be P or EP")
    return arm


def build_heads(hidden_size: int, arm: str, *, profile_seed: int = 42, emotion_seed: int = 43) -> Any:
    """Construct exact FP32 heads with deterministic shared profile initialization."""
    import torch
    from torch import nn

    validate_arm(arm)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(profile_seed)
        profile = nn.Linear(hidden_size, PROFILE_LABEL_COUNT, bias=True, dtype=torch.float32, device="cpu")
        emotion = None
        if arm == "EP":
            torch.manual_seed(emotion_seed)
            emotion = nn.Linear(hidden_size, EMOTION_LABEL_COUNT, bias=True, dtype=torch.float32, device="cpu")
    return ClassificationHeads(profile, emotion)


class ClassificationHeads:
    """Small module wrapper that avoids registering P's nonexistent emotion head."""

    def __new__(cls, profile_head: Any, emotion_head: Any | None):
        from torch import nn

        class _Heads(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.profile_head = profile_head
                if emotion_head is not None:
                    self.emotion_head = emotion_head

            def forward(self, hidden: Any) -> dict[str, Any]:
                result = {"profile_logits": self.profile_head(hidden.float())}
                if hasattr(self, "emotion_head"):
                    result["emotion_logits"] = self.emotion_head(hidden.float())
                return result

        return _Heads()


def masked_profile_bce(logits: Any, targets: Any, mask: Any) -> Any:
    from src.training.e03_a5_multitask import masked_bce_with_logits

    if logits.shape[-1] != PROFILE_LABEL_COUNT:
        raise ValueError("A7 Profile logits must have 33 labels")
    return masked_bce_with_logits(logits, targets, mask)


def emotion_cross_entropy(logits: Any, targets: Any) -> Any:
    import torch
    import torch.nn.functional as F

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != EMOTION_LABEL_COUNT:
        raise ValueError("Emotion logits must have shape [batch, 16]")
    target_tensor = torch.as_tensor(targets, device=logits.device, dtype=torch.long)
    if target_tensor.ndim != 1 or target_tensor.shape[0] != logits.shape[0]:
        raise ValueError("Emotion targets must contain one class index per example")
    if torch.any(target_tensor < 0) or torch.any(target_tensor >= EMOTION_LABEL_COUNT):
        raise ValueError("Emotion targets must be in [0, 15]")
    loss = F.cross_entropy(logits.float(), target_tensor, reduction="mean")
    if not torch.isfinite(loss):
        raise ValueError("Emotion CE is non-finite")
    return loss


def arm_loss(
    output: Mapping[str, Any],
    profile_targets: Any,
    profile_mask: Any,
    emotion_targets: Any | None,
    arm: str,
    *,
    lambda_emotion: float,
) -> dict[str, Any]:
    """Compute the sole frozen arm difference: EP adds weighted emotion CE."""
    import math
    import torch

    validate_arm(arm)
    if isinstance(lambda_emotion, bool) or not isinstance(lambda_emotion, (float, int)) or not math.isfinite(float(lambda_emotion)) or lambda_emotion <= 0:
        raise ValueError("A7 lambda_emotion must be a positive finite value")
    profile_logits = output.get("profile_logits")
    if not isinstance(profile_logits, torch.Tensor):
        raise ValueError("A7 output is missing Profile logits")
    profile = masked_profile_bce(profile_logits, profile_targets, profile_mask)
    if arm == "P":
        if "emotion_logits" in output or emotion_targets is not None:
            raise ValueError("P arm must not create an Emotion head or calculate Emotion loss")
        return {"profile_bce": profile, "emotion_ce": None, "total": profile}
    emotion_logits = output.get("emotion_logits")
    if not isinstance(emotion_logits, torch.Tensor) or emotion_targets is None:
        raise ValueError("EP arm requires Emotion logits and targets")
    emotion = emotion_cross_entropy(emotion_logits, emotion_targets)
    total = profile + float(lambda_emotion) * emotion
    if not torch.isfinite(total):
        raise ValueError("E03-A7 total loss is non-finite")
    return {"profile_bce": profile, "emotion_ce": emotion, "total": total}


def get_peft_backbone(model: Any) -> Any:
    """Resolve the Qwen base transformer under a PEFT CausalLM without LM-head projection."""
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(causal, "model", None)
    if backbone is None or not callable(backbone):
        raise ValueError("PEFT CausalLM does not expose a callable Qwen transformer backbone")
    return backbone


def forward_prompt_end(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    prompt_lengths: Any,
) -> Any:
    """Run only the Qwen backbone and gather final hidden state at each Prompt end."""
    import torch

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Prompt input IDs and attention mask must share [batch, sequence] shape")
    lengths = torch.as_tensor(prompt_lengths, device=input_ids.device, dtype=torch.long)
    if lengths.shape != (input_ids.shape[0],) or torch.any(lengths <= 0) or torch.any(lengths > input_ids.shape[1]):
        raise ValueError("Prompt lengths must identify one valid token per example")
    backbone = get_peft_backbone(model)
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("Qwen backbone must return last_hidden_state [batch, sequence, hidden]")
    positions = lengths - 1
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    if torch.any(attention_mask[rows, positions] != 1):
        raise ValueError("Prompt end points to a masked or padded token")
    sequence_positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    if torch.any((sequence_positions < lengths.unsqueeze(1)) & (attention_mask == 0)):
        raise ValueError("Prompt tokens must form an unpadded left-aligned prefix")
    return hidden[rows, positions, :]


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor keys, dtypes, shapes, and raw bytes in stable key order."""
    import torch

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("tensor state identity accepts only named tensors")
        tensor = value.detach().contiguous().cpu()
        header = json.dumps({"key": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_parameter_groups(model: Any, heads: Any, arm: str) -> dict[str, list[tuple[str, Any]]]:
    """Return the exact trainable whitelist and reject any unfrozen Base/LM parameters."""
    validate_arm(arm)
    lora = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"unexpected trainable Base/LM parameter: {name}")
            lora.append((name, parameter))
    profile = [(name, parameter) for name, parameter in heads.profile_head.named_parameters() if parameter.requires_grad]
    emotion = []
    if arm == "P":
        if hasattr(heads, "emotion_head"):
            raise ValueError("P must not contain an EmotionHead")
    else:
        if not hasattr(heads, "emotion_head"):
            raise ValueError("EP requires an EmotionHead")
        emotion = [(name, parameter) for name, parameter in heads.emotion_head.named_parameters() if parameter.requires_grad]
    if not lora or len(profile) != 2 or (arm == "EP" and len(emotion) != 2):
        raise ValueError("A7 trainable LoRA/head parameter groups are incomplete")
    return {"lora": lora, "profile_head": profile, "emotion_head": emotion}


def count_parameters(groups: Mapping[str, Sequence[tuple[str, Any]]]) -> dict[str, Any]:
    return {
        group: {
            "tensor_count": len(values),
            "parameter_count": sum(int(parameter.numel()) for _name, parameter in values),
            "names": [name for name, _parameter in values],
            "dtypes": sorted({str(parameter.dtype).removeprefix("torch.") for _name, parameter in values}),
        }
        for group, values in groups.items()
    }
