# -*- coding: utf-8 -*-
"""E03 T1 裁剪统计：对 Val 每个预测点算真实 token、模拟 T1 截断。

用法（在 WSL 里）：
  .venv/bin/python /mnt/c/Users/sumiha/work/ops/e03/t1_stats.py --root ~/projects/match --out ~/projects/match/outputs/e03/t1-stats --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_points(root: Path):
    """把 val_public.jsonl 的 900 段对话展开成预测点。"""
    path = root / "data" / "val" / "val_public.jsonl"
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = row["conversation_id"]
            turns = row["turns"]
            by_id = {t["turn_id"]: t for t in turns}
            for idx, point in enumerate(row["prediction_points"]):
                target_tid = point["target_user_turn_id"]
                ids = sorted(tid for tid in by_id if tid <= target_tid)
                history = [
                    {"role": by_id[tid]["role"], "content": by_id[tid]["content"]}
                    for tid in ids
                ]
                if not history or history[-1]["role"] != "user":
                    continue
                out.append({
                    "point_id": f"{cid}_{target_tid}",
                    "conversation_id": cid,
                    "point_index": idx,
                    "target_turn_id": target_tid,
                    "history": history,
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="outputs/e03/t1-stats")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))

    from transformers import AutoTokenizer
    from src.shared.prompting import build_messages, encode_inference_prompt, select_recent_complete_suffix

    model_dir = root / "participant" / "model"
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt_text = (root / "configs" / "prompts" / "p0.txt").read_text(encoding="utf-8")
    cap = 1280

    points = load_points(root)
    if args.limit:
        points = points[: args.limit]
    print(f"points {len(points)} prompt_cap {cap}", flush=True)

    rows = []
    for n, pt in enumerate(points, 1):
        history = pt["history"]
        full_ids = encode_inference_prompt(tok, build_messages(history, prompt_text))
        sel = select_recent_complete_suffix(history, prompt_text, cap, tok)
        dropped_turns = len(history) - len(sel["selected_history"])
        rows.append({
            "point_id": pt["point_id"],
            "conversation_id": pt["conversation_id"],
            "point_index": pt["point_index"],
            "turns_total": len(history),
            "full_prompt_tokens": len(full_ids),
            "kept_prompt_tokens": sel["prompt_tokens"],
            "dropped_tokens": len(full_ids) - sel["prompt_tokens"],
            "dropped_turns": dropped_turns,
            "truncated": dropped_turns > 0,
            "floor_exceeds_cap": sel["floor_exceeds_cap"],
            "status": sel["status"],
        })
        if n % 200 == 0:
            print(f"  {n}/{len(points)}", flush=True)

    with (out_dir / "t1_points.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(rows)
    trunc = [r for r in rows if r["truncated"]]
    floor = [r for r in rows if r["floor_exceeds_cap"]]
    full_tok = sum(r["full_prompt_tokens"] for r in rows)
    kept_tok = sum(r["kept_prompt_tokens"] for r in rows)
    summary = {
        "points": total,
        "truncated_points": len(trunc),
        "truncated_ratio": round(len(trunc) / total, 4) if total else 0,
        "floor_exceeds_points": len(floor),
        "mean_full_tokens": round(full_tok / total, 1) if total else 0,
        "mean_kept_tokens": round(kept_tok / total, 1) if total else 0,
        "mean_dropped_tokens": round((full_tok - kept_tok) / total, 1) if total else 0,
        "mean_dropped_turns_on_truncated": round(
            sum(r["dropped_turns"] for r in trunc) / len(trunc), 1) if trunc else 0,
        "prompt_cap": cap,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
