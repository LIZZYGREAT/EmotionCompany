from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.training.e03_a8_completion_loss import completion_only_forward_loss


class _Backbone(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lora_A = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        hidden = self.embed(input_ids)
        hidden = hidden + (hidden @ self.lora_A) * 0.1
        hidden = self.proj(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Head(nn.Linear):
    def __init__(self, hidden_size: int = 12, vocab_size: int = 32):
        super().__init__(hidden_size, vocab_size, bias=False)
        self.projected_shape = None

    def forward(self, hidden):
        self.projected_shape = tuple(hidden.shape)
        return super().forward(hidden)


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lm_head = _Head()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, labels, attention_mask):
        hidden = self.model(input_ids, attention_mask, use_cache=False, return_dict=True).last_hidden_state
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        return SimpleNamespace(loss=loss, logits=logits)


class _PeftLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.causal_lm = _CausalLM()

    def get_base_model(self):
        return self.causal_lm

    def get_output_embeddings(self):
        return self.causal_lm.get_output_embeddings()

    def forward(self, **kwargs):
        return self.causal_lm(**kwargs)


def _batch(prompt_length: int = 5, completion_length: int = 4, padded_length: int = 0):
    real_length = prompt_length + completion_length
    ids = torch.arange(1, real_length + 1, dtype=torch.long) % 31
    ids[-1] = 31  # completion ends in the synthetic EOS token
    labels = torch.cat([torch.full((prompt_length,), -100), ids[prompt_length:]])
    mask = torch.ones(real_length, dtype=torch.long)
    if padded_length:
        ids = F.pad(ids, (0, padded_length), value=0)
        labels = F.pad(labels, (0, padded_length), value=-100)
        mask = F.pad(mask, (0, padded_length), value=0)
    return {"input_ids": ids.unsqueeze(0), "labels": labels.unsqueeze(0), "attention_mask": mask.unsqueeze(0)}


@pytest.mark.parametrize("prompt_length,completion_length", [(3, 2), (5, 4), (17, 7)])
def test_selective_loss_and_all_trainable_gradients_match_full_logits(prompt_length, completion_length):
    torch.manual_seed(7)
    model = _PeftLike()
    batch = _batch(prompt_length, completion_length)

    full_loss = model(**batch).loss
    full_loss.backward()
    full_grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    model.zero_grad(set_to_none=True)

    selective_loss = completion_only_forward_loss(model, batch)
    selective_loss.backward()
    selective_grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert torch.allclose(full_loss, selective_loss, atol=2e-6, rtol=2e-6)
    assert full_grads.keys() == selective_grads.keys()
    for name in full_grads:
        assert torch.allclose(full_grads[name], selective_grads[name], atol=2e-6, rtol=2e-6), name
    assert model.get_output_embeddings().projected_shape == (1, completion_length, 12)


def test_selective_loss_handles_right_padding_and_supervises_terminal_eos():
    model = _PeftLike()
    batch = _batch(prompt_length=6, completion_length=5, padded_length=3)
    loss, facts = completion_only_forward_loss(model, batch, return_facts=True)
    loss.backward()
    assert facts == {
        "sequence_tokens": 11,
        "prompt_tokens": 6,
        "completion_tokens": 5,
        "projected_positions": 5,
        "logit_dtype": "torch.float32",
    }
    assert model.get_output_embeddings().projected_shape == (1, 5, 12)
    assert batch["labels"][0, 10].item() == 31


def test_selective_loss_rejects_a_gap_in_the_supervised_suffix():
    model = _PeftLike()
    batch = _batch(prompt_length=4, completion_length=4)
    batch["labels"][0, 6] = -100
    with pytest.raises(ValueError, match="complete suffix"):
        completion_only_forward_loss(model, batch)
