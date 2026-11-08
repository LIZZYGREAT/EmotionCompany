from __future__ import annotations

import torch

from src.inference.final_dual_runtime import DualAdapterRuntime, cuda_memory_facts


class FakeTokenizer:
    def decode(self, token_ids, **kwargs):
        return ",".join(str(value) for value in token_ids)


class FakeModel:
    def __init__(self):
        self.active_adapter = None
        self.embedding = torch.nn.Embedding(4, 2)
        self.switches = []

    def set_adapter(self, name):
        self.active_adapter = name
        self.switches.append(name)

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, *, input_ids, **kwargs):
        token = 101 if self.active_adapter == "profile" else 202
        next_id = torch.tensor([[token]], device=input_ids.device, dtype=input_ids.dtype)
        return torch.cat((input_ids, next_id), dim=1)


def test_runtime_switches_adapter_explicitly_for_each_serial_generation() -> None:
    model = FakeModel()
    runtime = DualAdapterRuntime(model, FakeTokenizer(), torch)

    profile = runtime.generate("profile", [1, 2], max_new_tokens=256, eos_token_id=151645, pad_token_id=151643)
    emotion = runtime.generate("emotion-response", [3, 4], max_new_tokens=1200, eos_token_id=151645, pad_token_id=151643)
    profile_again = runtime.generate("profile", [1, 2], max_new_tokens=256, eos_token_id=151645, pad_token_id=151643)

    assert model.switches == ["profile", "emotion-response", "profile"]
    assert profile["token_ids"] == [101]
    assert emotion["token_ids"] == [202]
    assert profile_again["token_ids"] == profile["token_ids"]
    assert profile["text"] == "101"
    assert emotion["adapter_name"] == "emotion-response"


def test_cpu_memory_facts_are_explicit_when_cuda_is_unavailable() -> None:
    facts = cuda_memory_facts(torch)
    assert facts["available"] is torch.cuda.is_available()
    if not facts["available"]:
        assert facts["peak_allocated_bytes"] is None
        assert facts["peak_reserved_bytes"] is None
