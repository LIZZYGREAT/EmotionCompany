"""Completion-only Causal LM loss that avoids projecting Prompt positions."""

from __future__ import annotations

from typing import Any


def completion_only_forward_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    prompt_length: int | None = None,
    sequence_length: int | None = None,
    return_facts: bool = False,
):
    """Run the adapted backbone and project only hidden states supervised by CE.

    This implementation intentionally accepts one right-padded (or unpadded)
    sequence, matching A8's per-device microbatch size of one. Prompt tokens
    remain in the backbone attention computation; only their vocabulary logits
    are skipped. The final target, including terminal EOS, remains supervised.
    """
    import torch
    import torch.nn.functional as F

    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("completion-only A8 loss requires input_ids with shape [1, sequence]")
    if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
        raise ValueError("completion-only A8 loss requires labels matching input_ids")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise ValueError("completion-only A8 loss requires attention_mask matching input_ids")

    valid_length = input_ids.shape[1] if sequence_length is None else sequence_length
    if not isinstance(valid_length, int) or isinstance(valid_length, bool) or not 2 <= valid_length <= input_ids.shape[1]:
        raise ValueError("A8 sequence_length must describe a valid unpadded token prefix")
    if prompt_length is None:
        # The diagnostic/test route performs detailed checks. Training supplies
        # prevalidated integer metadata from `_row_contract` to avoid device
        # synchronization while locating the Completion boundary every batch.
        valid_length = int(attention_mask[0].sum().detach().cpu())
        if valid_length < 2 or not bool((attention_mask[0, :valid_length] == 1).all()):
            raise ValueError("A8 attention mask must describe a non-empty right-padded sequence")
        if bool((attention_mask[0, valid_length:] != 0).any()):
            raise ValueError("A8 attention mask must use right padding only")
        supervised = torch.nonzero(labels[0, :valid_length] != -100, as_tuple=False).flatten()
        if supervised.numel() == 0:
            raise ValueError("A8 completion has no supervised tokens")
        prompt_length = int(supervised[0].detach().cpu())
        expected_positions = torch.arange(prompt_length, valid_length, device=supervised.device)
        if prompt_length <= 0 or supervised.numel() != expected_positions.numel() or not bool((supervised == expected_positions).all()):
            raise ValueError("A8 labels must mask the Prompt and supervise one complete suffix")
        if not torch.equal(labels[0, prompt_length:valid_length], input_ids[0, prompt_length:valid_length]):
            raise ValueError("A8 Completion labels must equal the input suffix")
    elif not isinstance(prompt_length, int) or isinstance(prompt_length, bool) or not 0 < prompt_length < valid_length:
        raise ValueError("A8 prompt_length must precede a non-empty Completion")

    # PeftModelForCausalLM.get_base_model() returns the actual wrapped causal
    # model. Its `.model` is the Qwen backbone whose linear modules carry LoRA.
    causal_lm = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    backbone = getattr(causal_lm, "model", None)
    output_embeddings = model.get_output_embeddings()
    if backbone is None or output_embeddings is None:
        raise TypeError("model must expose its adapted backbone and real output embeddings")
    outputs = backbone(
        input_ids=input_ids[:, :valid_length],
        attention_mask=attention_mask[:, :valid_length],
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise TypeError("adapted backbone did not return last_hidden_state")
    prediction_hidden = hidden[:, prompt_length - 1 : valid_length - 1, :]
    targets = labels[:, prompt_length:valid_length]
    logits = output_embeddings(prediction_hidden)
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    if return_facts:
        return loss, {
            "sequence_tokens": valid_length,
            "prompt_tokens": prompt_length,
            "completion_tokens": valid_length - prompt_length,
            "projected_positions": int(prediction_hidden.shape[1]),
            "logit_dtype": str(logits.dtype),
        }
    return loss
