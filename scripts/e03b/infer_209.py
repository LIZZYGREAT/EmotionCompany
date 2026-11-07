import json
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "/mnt/c/Users/sumiha/work/match")
from src.shared.prompting import build_messages, encode_inference_prompt, select_recent_complete_suffix

MODEL = "/home/sumiha/projects/match/participant/model"
ADAPTER = "/mnt/c/Users/sumiha/work/match/outputs/training/e03b-run1536-4665/checkpoint-4665"
PROMPT = open("/mnt/c/Users/sumiha/work/match/configs/prompts/p_e03b.txt", encoding="utf-8").read()
PROMPT_P0 = open("/mnt/c/Users/sumiha/work/match/configs/prompts/p0.txt", encoding="utf-8").read()
SRC = "/mnt/c/Users/sumiha/work/match/data-incoming/e01-b-209/points_209.jsonl"
OUT = "/mnt/c/Users/sumiha/work/match/outputs/training/e03b-run1536-4665/pred_209.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                             device_map={"": 0}, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]

import gzip
val = {}
with gzip.open("/mnt/c/Users/sumiha/work/match/data-incoming/val_public.jsonl.gz", "rt", encoding="utf-8") as handle:
    for line in handle:
        conv = json.loads(line)
        val[conv["conversation_id"]] = conv
print("points", len(rows), flush=True)
start_all = time.time()
done_ids = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as prev:
        for line in prev:
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
print("resume_done", len(done_ids), flush=True)
with open(OUT, "a", encoding="utf-8") as out:
    for i, row in enumerate(rows):
        if row["id"] in done_ids:
            continue
        conv = val[row["conversation_id"]]
        tid = row["target_user_turn_id"]
        turns = [t for t in conv["turns"] if t["turn_id"] <= tid]
        sel = select_recent_complete_suffix(turns, PROMPT_P0, 1280, tok)
        messages = build_messages(sel["selected_history"], PROMPT)
        input_ids = encode_inference_prompt(tok, messages)
        ids = torch.tensor([input_ids], device="cuda")
        with torch.no_grad():
            gen = model.generate(input_ids=ids, max_new_tokens=1200, do_sample=False,
                                 eos_token_id=151645, pad_token_id=151643)
        gen_ids = gen[0][len(input_ids):]
        raw = tok.decode(gen_ids, skip_special_tokens=True)
        out.write(json.dumps({"id": row["id"], "raw": raw, "gen_tokens": len(gen_ids), "finish": "length" if len(gen_ids) >= 1200 else "eos"}, ensure_ascii=False) + "\n")
        out.flush()
        print("step", i, "sec", round(time.time() - start_all, 1), flush=True)
        if i % 10 == 0:
            print("done", i, flush=True)
print("finished", flush=True)
print("total_sec", round(time.time() - start_all, 1))
print("peak_alloc_gb", round(torch.cuda.max_memory_allocated() / 1e9, 3))
print("peak_reserved_gb", round(torch.cuda.max_memory_reserved() / 1e9, 3))
