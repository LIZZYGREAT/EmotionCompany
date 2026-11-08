import argparse
import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys
sys.path.insert(0, "/mnt/c/Users/sumiha/work/match")
from src.shared.prompting import build_messages, encode_inference_prompt, select_recent_complete_suffix

MODEL = "/home/sumiha/projects/match/participant/model"
PROMPT = open("/mnt/c/Users/sumiha/work/match/configs/prompts/p_e03b.txt", encoding="utf-8").read()
SRC = "/mnt/c/Users/sumiha/work/match/test_inference_data.jsonl"
OUT = "/mnt/c/Users/sumiha/work/match/outputs/training/e03b-replay/test_dense_1152.jsonl"
ADAPTER = "/mnt/c/Users/sumiha/work/match/outputs/training/e03b-replay/checkpoint-1152"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                             device_map={"": 0}, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
start = time.time()
with open(OUT, "w", encoding="utf-8") as out:
    for row in rows:
        history = row["history"]
        if history[-1]["role"] != "user":
            raise ValueError(f"{row['id']}: history does not end with user")
        sel = select_recent_complete_suffix(history, PROMPT, 1280, tok)
        ids = encode_inference_prompt(tok, build_messages(sel["selected_history"], PROMPT))
        with torch.no_grad():
            gen = model.generate(input_ids=torch.tensor([ids], device="cuda"),
                                 max_new_tokens=1200, do_sample=False,
                                 eos_token_id=151645, pad_token_id=151643)
        raw = tok.decode(gen[0][len(ids):], skip_special_tokens=True)
        out.write(json.dumps({"id": row["id"], "raw": raw}, ensure_ascii=False) + "\n")
        out.flush()
        print("done", row["id"], flush=True)
print("total_sec", round(time.time() - start, 1))
