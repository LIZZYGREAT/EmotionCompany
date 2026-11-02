# -*- coding: utf-8 -*-
"""E03：乐观可救样本数。

口径（2026-11-03 与白咲对齐）：
  1) 该点 Gold 画像非空；
  2) 按该维度对应的 Gold 标签找证据词；
  3) 证据只出现在被裁掉的段落里，保留段里找不到。
规则在 Train 上做、做完冻结；Val 只套不改。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RULES = {
    "personality_traits": {
        "extroverted": ["外向", "爱社交", "喜欢热闹", "聚会", "人多"],
        "introverted": ["内向", "不爱说话", "独处", "社恐", "安静"],
        "open": ["开放", "愿意尝试", "好奇", "新东西"],
        "conservative": ["保守", "传统", "稳妥", "不敢"],
        "high_conscientiousness": ["计划", "自律", "认真", "条理", "安排"],
        "casual": ["随意", "随便", "佛系", "无所谓"],
        "agreeable": ["随和", "好说话", "迁就", "顺着"],
        "assertive": ["直接", "坚持", "主张", "强势", "不肯让"],
        "emotionally_stable": ["冷静", "稳定", "不慌", "稳住"],
        "sensitive": ["敏感", "在意", "容易受伤", "想多"],
    },
    "interests": {
        "study_exam": ["考试", "复习", "考研", "学习", "成绩"],
        "programming_technology": ["编程", "代码", "程序", "算法", "技术"],
        "reading_writing": ["看书", "读书", "写作", "小说", "阅读"],
        "film_animation": ["电影", "动漫", "动画", "追番"],
        "music": ["音乐", "听歌", "唱歌", "歌"],
        "games": ["游戏", "打游戏", "网游", "开黑"],
        "sports_fitness": ["运动", "健身", "跑步", "打球", "球"],
        "travel_outdoor": ["旅行", "旅游", "爬山", "户外", "露营"],
        "pets": ["猫", "狗", "宠物", "铲屎"],
        "social": ["朋友", "聚会", "社交", "约"],
        "career_development": ["工作", "实习", "求职", "面试", "简历"],
        "art_design": ["设计", "画画", "美术", "绘画"],
    },
    "style": {
        "brief": ["简单说", "简短", "别啰嗦", "少说"],
        "detailed": ["详细", "具体", "展开说"],
        "colloquial": ["口语", "随便说", "白话"],
        "formal": ["正式", "书面", "规范"],
        "direct": ["直接", "直说", "别绕"],
        "indirect": ["委婉", "含蓄", "绕着说"],
        "humorous": ["幽默", "开玩笑", "搞笑", "逗"],
        "rational": ["理性", "逻辑", "分析", "客观"],
        "high_emotional_expression": ["崩溃", "激动", "受不了", "好难受"],
        "low_emotional_expression": ["平静", "克制", "还好", "没什么"],
        "emoji_user": ["表情", "emoji", "🤣", "😊", "😂"],
    },
}


def hits(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def load_train_coverage(root: Path):
    """看规则在 Train 上的召回，作为规则质量的体检，不用于调规则。"""
    path = root / "data" / "train" / "train_public.jsonl"
    stat = {d: {l: [0, 0] for l in RULES[d]} for d in RULES}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = " ".join(t.get("content", "") for t in row["turns"])
            for point in row["prediction_points"]:
                prof = point.get("target", {}).get("user_profile", {})
                for dim, labels in prof.items():
                    for lab in labels:
                        if lab not in stat.get(dim, {}):
                            continue
                        stat[dim][lab][0] += 1
                        if hits(text, RULES[dim][lab]):
                            stat[dim][lab][1] += 1
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--stats", required=True, help="t1_points.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) 规则体检（Train）
    cov = load_train_coverage(root)
    cov_rows = []
    for dim, labels in cov.items():
        for lab, (total, hit) in labels.items():
            cov_rows.append({"dimension": dim, "label": lab, "gold_total": total,
                             "keyword_hit": hit,
                             "hit_rate": round(hit / total, 3) if total else None})
    (out / "rule_coverage_train.json").write_text(
        json.dumps(cov_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) Val 上套规则
    stats = {}
    with open(args.stats, encoding="utf-8") as fh:
        for line in fh:
            stats[json.loads(line)["point_id"]] = json.loads(line)

    val_path = root / "data" / "val" / "val_public.jsonl"
    rows = []
    with val_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cid = row["conversation_id"]
            by_id = {t["turn_id"]: t for t in row["turns"]}
            for point in row["prediction_points"]:
                tid = point["target_user_turn_id"]
                pid = f"{cid}_{tid}"
                st = stats.get(pid)
                if not st or not st["truncated"]:
                    continue
                ids = sorted(t for t in by_id if t <= tid)
                dropped = ids[: len(ids) - (len(ids) - st["dropped_turns"])] if st["dropped_turns"] else []
                kept_ids = ids[st["dropped_turns"]:]
                dropped_text = " ".join(by_id[t]["content"] for t in dropped)
                kept_text = " ".join(by_id[t]["content"] for t in kept_ids)
                prof = point.get("target", {}).get("user_profile", {})
                for dim, labels in prof.items():
                    for lab in labels:
                        if lab not in RULES.get(dim, {}):
                            continue
                        if hits(dropped_text, RULES[dim][lab]) and not hits(kept_text, RULES[dim][lab]):
                            rows.append({"point_id": pid, "dimension": dim, "label": lab,
                                         "dropped_turns": st["dropped_turns"],
                                         "dropped_tokens": st["dropped_tokens"]})

    (out / "rescuable_evidence.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8")

    by_dim = {}
    for r in rows:
        by_dim[r["dimension"]] = by_dim.get(r["dimension"], 0) + 1
    points = {r["point_id"] for r in rows}
    summary = {
        "truncated_points": sum(1 for s in stats.values() if s["truncated"]),
        "rescuable_points": len(points),
        "rescuable_label_hits": len(rows),
        "by_dimension": by_dim,
        "note": "乐观估计：规则在 Train 上冻结，只统计证据只在被裁段出现的 Gold 标签；关键词会漏，实际可救数应低于此值。",
    }
    (out / "rescuable_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
