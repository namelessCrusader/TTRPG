"""Every model's answer to a card, side by side, for judging case by case.

    lab/.venv/bin/python -m lab.review      # -> lab/results/review_v1.md

No key. Each card shows the state, what every decision model picked (with
its runner-up, because a close second is a different finding from a
confident miss), and what every voice freely did and said. The verdicts are
written into `lab/verdicts/<card set>.jsonl` by whoever reads it, one line a
(card, model), as ok / arguable / wrong with a reason.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
DECIDERS = ["laya", "von", "gliclass", "qwen"]
VOICES = ["qwen", "reyna"]


def _load(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if path.exists() else {}


def _pick(ans):
    a = ans.get("answer")
    if ans.get("p_true") is not None:
        return f"{a} (p={ans['p_true']:.2f})"
    return f"{a}" + (f" ({ans['p']:.2f})" if ans.get("p") is not None else "")


def main(cards_name="social_v1"):
    cards = [json.loads(l) for l in open(HERE / "cards" / f"{cards_name}.jsonl")]
    dec = {m: _load(HERE / "results" / f"{m}__{cards_name}.jsonl") for m in DECIDERS}
    free = {v: _load(HERE / "results" / f"free__{v}.jsonl") for v in VOICES}
    out = [f"# Review: {cards_name}\n"]
    for c in cards:
        out.append(f"## {c['id']} ({c['track']})\n")
        for k, v in c["state"].items():
            out.append(f"- **{k}**: {v}")
        out.append("")
        qs = list(c["questions"])
        out.append("| model | " + " | ".join(qs) + " |")
        out.append("|---|" + "---|" * len(qs))
        for m in DECIDERS:
            r = dec[m].get(c["id"])
            if r:
                out.append(f"| {m} | " + " | ".join(_pick(r["answers"].get(q, {}))
                                                    for q in qs) + " |")
        out.append("")
        for v in VOICES:
            r = free[v].get(c["id"])
            if r:
                out.append(f"- **{v}** DO: {r['do']} / SAY: {r['say']}"
                           if r["do"] or r["say"] else f"- **{v}** raw: {r['raw'][:200]}")
        out.append("")
    path = HERE / "results" / f"review_{cards_name.split('_')[-1]}.md"
    path.write_text("\n".join(out))
    print(path)


if __name__ == "__main__":
    main()
