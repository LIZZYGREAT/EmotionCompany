# -*- coding: utf-8 -*-
"""E03 P1 底模代理：一次加载模型，四轮 prompt 顺序跑。

用法（WSL，项目 .venv）：
  .venv/bin/python /mnt/c/Users/sumiha/work/ops/e03/run_base_four.py \
      --root ~/projects/match --out ~/projects/match/outputs/e03/p1-base-run1 \
      --prompts p0=/path/p0.txt,p1a=/path/p1a.txt --limit 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def load_points(root: Path, sample_ids: set[str] | None):
    path = root / "data" / "val" / "val_public.jsonl"
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = row["conversation_id"]
            by_id = {t["turn_id"]: t for t in row["turns"]}
            for point in row["prediction_points"]:
                tid = point["target_user_turn_id"]
                pid = f"{cid}_{tid}"
                if sample_ids is not None and pid not in sample_ids:
                    continue
                ids = sorted(t for t in by_id if t <= tid)
                history = [{"role": by_id[t]["role"], "content": by_id[t]["content"]} for t in ids]
                if not history or history[-1]["role"] != "user":
                    continue
                out.append({"id": pid, "history": history})
    return out


def extract_object(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True, help="name=path,name=path")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", type=int, default=1280)
    ap.add_argument("--max-new", type=int, default=1200)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.shared.prompting import build_messages, encode_inference_prompt, select_recent_complete_suffix

    prompts = []
    for item in args.prompts.split(","):
        name, _, path = item.partition("=")
        prompts.append((name, Path(path).expanduser().read_text(encoding="utf-8")))

    model_dir = root / "participant" / "model"
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    sample_ids = None
    ids_path = out_dir / "sample_ids.json"
    if ids_path.exists():
        sample_ids = set(json.loads(ids_path.read_text(encoding="utf-8"))["point_ids"])
    points = load_points(root, sample_ids)
    if args.limit:
        points = points[: args.limit]
    print(f"points {len(points)} rounds {len(prompts)}", flush=True)

    schema = json.loads((root / "data" / "val" / "val_public.schema.json").read_text(encoding="utf-8"))
    enums = {}
    for dim in ("personality_traits", "interests", "style"):
        pass

    for name, prompt_text in prompts:
        rows = []
        t_round = time.time()
        for n, pt in enumerate(points, 1):
            sel = select_recent_complete_suffix(pt["history"], prompt_text, args.cap, tok)
            input_ids = sel["prompt_ids"]
            tensor = torch.tensor([input_ids], device=model.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=tensor, max_new_tokens=args.max_new,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            new_ids = gen[0][len(input_ids):].tolist()
            text = tok.decode(new_ids, skip_special_tokens=True)
            rows.append({
                "id": pt["id"], "prompt": name,
                "prompt_tokens": len(input_ids), "raw": text,
                "truncated": len(new_ids) >= args.max_new,
                "floor_exceeds": sel["floor_exceeds_cap"],
            })
            if n % 20 == 0:
                print(f"  {name} {n}/{len(points)} {time.time()-t_round:.0f}s", flush=True)
        with (out_dir / "raw" / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()
        print(f"round {name} done in {time.time()-t_round:.0f}s", flush=True)

    facts = {
        "model_dir": str(model_dir),
        "prompt_cap": args.cap,
        "max_new_tokens": args.max_new,
        "decoding": "greedy",
        "quantization": "nf4-double-bf16",
        "torch": torch.__version__,
        "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
