# -*- coding: utf-8 -*-
"""E03 P1 底模对照：从 raw 算指标（不读 submission）。"""
import argparse
import json
from pathlib import Path

EMO = ["", "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
       "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
       "shame", "helplessness"]
DIMS = {
    "personality_traits": ["extroverted", "introverted", "open", "conservative",
                           "high_conscientiousness", "casual", "agreeable",
                           "assertive", "emotionally_stable", "sensitive"],
    "interests": ["study_exam", "programming_technology", "reading_writing",
                  "film_animation", "music", "games", "sports_fitness",
                  "travel_outdoor", "pets", "social", "career_development",
                  "art_design"],
    "style": ["brief", "detailed", "colloquial", "formal", "direct", "indirect",
              "humorous", "rational", "high_emotional_expression",
              "low_emotional_expression", "emoji_user"],
}


def parse_obj(text: str):
    t = text.strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    s = t.find("{")
    if s < 0:
        return None
    depth = 0
    instr = False
    esc = False
    for i in range(s, len(t)):
        c = t[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(t[s:i + 1])
                    return v if isinstance(v, dict) else None
                except Exception:
                    return None
    return None


def check(obj):
    out = {"fields": False, "enum": False, "extra": False, "profile_nonempty": False,
           "empty_response": True, "chars": 0}
    if obj is None:
        return out
    out["fields"] = all(k in obj for k in ("emotion_label", "user_profile", "response_text", "memory_refs"))
    if not out["fields"]:
        return out
    rt = obj.get("response_text")
    if isinstance(rt, str):
        out["chars"] = len(rt)
        out["empty_response"] = not rt.strip()
    prof = obj.get("user_profile")
    ok = isinstance(prof, dict)
    extra = False
    nonempty = False
    if ok:
        for dim, allowed in DIMS.items():
            vals = prof.get(dim)
            if not isinstance(vals, list):
                ok = False
                continue
            if vals:
                nonempty = True
            for v in vals:
                if v not in allowed:
                    extra = True
                    ok = False
    emo = obj.get("emotion_label")
    if emo not in EMO:
        ok = False
        if isinstance(emo, str):
            extra = True
    out["enum"] = bool(ok)
    out["extra"] = extra
    out["profile_nonempty"] = nonempty
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run).expanduser().resolve()
    rows = []
    for f in sorted((run / "raw").glob("*.jsonl")):
        name = f.stem
        recs = [json.loads(l) for l in f.open(encoding="utf-8")]
        n = len(recs)
        agg = {k: 0 for k in ("json", "fields", "enum", "extra", "nonempty", "trunc", "empty")}
        chars = 0
        for r in recs:
            obj = parse_obj(r["raw"])
            c = check(obj)
            agg["json"] += obj is not None
            agg["fields"] += c["fields"]
            agg["enum"] += c["enum"]
            agg["extra"] += c["extra"]
            agg["nonempty"] += c["profile_nonempty"]
            agg["trunc"] += bool(r.get("truncated"))
            agg["empty"] += c["empty_response"]
            chars += c["chars"]
        rows.append({
            "prompt": name, "n": n,
            "json_object_rate": round(agg["json"] / n, 4),
            "required_fields_rate": round(agg["fields"] / n, 4),
            "enum_valid_rate": round(agg["enum"] / n, 4),
            "profile_nonempty_rate": round(agg["nonempty"] / n, 4),
            "schema_extra_label_rate": round(agg["extra"] / n, 4),
            "truncation_rate": round(agg["trunc"] / n, 4),
            "empty_response_rate": round(agg["empty"] / n, 4),
            "mean_response_chars": round(chars / n, 1),
        })
    import csv
    with (run / "metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
