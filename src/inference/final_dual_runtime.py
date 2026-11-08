"""Single-Base, explicitly switched two-LoRA Transformers runtime."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any


class DualAdapterRuntime:
    def __init__(self, model: Any, tokenizer: Any, torch_module: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module

    @classmethod
    def load(
        cls,
        *,
        base_model_path: str | Path,
        tokenizer: Any,
        adapters: dict[str, str | Path],
        torch_module: Any | None = None,
    ) -> "DualAdapterRuntime":
        if not adapters:
            raise ValueError("at least one named Adapter is required")
        if len(set(adapters)) != len(adapters):
            raise ValueError("Adapter names must be unique")
        if torch_module is None:
            import torch as torch_module
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_module.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            str(Path(base_model_path).resolve()),
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=quantization,
            torch_dtype=torch_module.bfloat16,
            device_map={"": 0},
        )
        base.config.use_cache = True
        if getattr(base, "is_gradient_checkpointing", False):
            base.gradient_checkpointing_disable()

        items = list(adapters.items())
        active_name, active_path = items[0]
        model = PeftModel.from_pretrained(
            base,
            str(active_path),
            adapter_name=active_name,
            is_trainable=False,
            autocast_adapter_dtype=True,
        )
        for name, adapter_path in items[1:]:
            model.load_adapter(
                str(adapter_path),
                adapter_name=name,
                is_trainable=False,
                autocast_adapter_dtype=True,
            )
        model.eval()
        model.config.use_cache = True
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = True
        model.set_adapter(active_name)
        return cls(model=model, tokenizer=tokenizer, torch_module=torch_module)

    def generate(
        self,
        adapter_name: str,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> dict[str, Any]:
        if not prompt_ids:
            raise ValueError("prompt token IDs must not be empty")
        self.model.set_adapter(adapter_name)
        if hasattr(self.model, "get_input_embeddings"):
            device = self.model.get_input_embeddings().weight.device
        else:
            device = next(self.model.parameters()).device
        input_ids = self.torch.tensor([prompt_ids], dtype=self.torch.long, device=device)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                eos_token_id=int(eos_token_id),
                pad_token_id=int(pad_token_id),
                use_cache=True,
            )
        generated_ids = [int(value) for value in output_ids[0, input_ids.shape[1]:].tolist()]
        return {
            "text": self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            "token_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "adapter_name": adapter_name,
        }

    def close(self) -> None:
        del self.model
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def cuda_memory_facts(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"available": False, "device_name": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "available": True,
        "device_name": torch_module.cuda.get_device_name(0),
        "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(0)),
    }
