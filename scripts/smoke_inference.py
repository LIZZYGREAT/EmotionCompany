import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen3-1.7B"

print("=" * 50)
print("1. Loading tokenizer...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Tokenizer loaded.")

print("\n2. Loading model in 4-bit NF4...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    dtype=torch.bfloat16,
)

model.eval()

print("Model loaded.")
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", model.device)

print(
    "GPU allocated memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB",
)

print("\n3. Preparing conversation...")

messages = [
    {
        "role": "system",
        "content": (
            "你是一名中文情感陪伴助手。"
            "请先理解用户的感受，"
            "再给予自然、简洁的回应。"
        ),
    },
    {
        "role": "user",
        "content": (
            "最近学习压力很大，"
            "总觉得自己付出了不少时间，"
            "却没有取得什么进步。"
        ),
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to("cuda")

print("Input tokens:", inputs["input_ids"].shape[-1])

print("\n4. Generating response...")

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

generated_ids = outputs[0][
    inputs["input_ids"].shape[-1]:
]

response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

print("\n" + "=" * 50)
print("MODEL RESPONSE")
print("=" * 50)
print(response)

print("\n" + "=" * 50)
print("PERFORMANCE")
print("=" * 50)

print("Generated tokens:", len(generated_ids))
print("Generation time:", round(elapsed, 2), "seconds")
print(
    "Peak GPU memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GiB",
)

assert response.strip(), "Model generated empty response"

print("\nQwen3-1.7B inference test PASSED")
