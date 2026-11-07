import gzip
import json
import re
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "/mnt/c/Users/sumiha/work/match")
from src.shared.prompting import build_messages, encode_inference_prompt, select_recent_complete_suffix

MODEL = "/home/sumiha/projects/match/participant/model"
ADAPTER = "/mnt/c/Users/sumiha/work/match/outputs/training/e03b-run1536-1024/checkpoint-1024"
P0 = open("/mnt/c/Users/sumiha/work/match/configs/prompts/p0.txt", encoding="utf-8").read()
NEW = open("/mnt/c/Users/sumiha/work/match/configs/prompts/p_e03b.txt", encoding="utf-8").read()
IDS = [
    "val_conv_000136_9", "val_conv_000353_7", "val_conv_000353_9",
    "val_conv_000044_3", "val_conv_000154_5",
    "val_conv_000012_5", "val_conv_000099_1", "val_conv_000012_7",
    "val_conv_000024_5", "val_conv_000206_5",
]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
val = {}
with gzip.open("/mnt/c/Users/sumiha/work/match/data-incoming/val_public.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        val[c["conversation_id"]] = c

rows = [json.loads(l) for l in open("/mnt/c/Users/sumiha/work/match/data-incoming/e01-b-209/points_209.jsonl", encoding="utf-8")]
by_id = {r["id"]: r for r in rows}

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                             device_map={"": 0}, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

start = time.time()
out = []
for pid in IDS:
    row = by_id[pid]
    turns = [t for t in val[row["conversation_id"]]["turns"] if t["turn_id"] <= row["target_user_turn_id"]]
    sel = select_recent_complete_suffix(turns, P0, 1280, tok)
    ids = encode_inference_prompt(tok, build_messages(sel["selected_history"], NEW))
    with torch.no_grad():
        gen = model.generate(input_ids=torch.tensor([ids], device="cuda"),
                             max_new_tokens=1200, do_sample=False,
                             eos_token_id=151645, pad_token_id=151643)
    raw = tok.decode(gen[0][len(ids):], skip_special_tokens=True)
    resp = raw.split(";response=", 1)[-1] if ";response=" in raw else raw
    reps = len(re.findall(r"(.{8,})\1", resp))
    out.append({"id": pid, "chars": len(resp.strip()), "repeat": reps, "raw": raw})
    print(pid, "chars", len(resp.strip()), "repeat", reps, flush=True)

print("elapsed", round(time.time() - start, 1))
longs = [o for o in out if o["id"] in IDS[:5]]
print("long_over500", sum(1 for o in longs if o["chars"] > 500))
print("long_repeat", sum(1 for o in longs if o["repeat"] > 0))
with open("/mnt/c/Users/sumiha/work/match/outputs/training/e03b-run1536-1024/sample10.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
