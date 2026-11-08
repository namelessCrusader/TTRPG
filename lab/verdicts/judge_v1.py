"""Verdicts on social_v1, judged case by case by reading each answer against
the person and the scene (lab/results/review_v1.md) — not against the key.

o = a reasonable thing for this person here, a = defensible but not what the
scene most suggests, w = wrong for this person. READ is judged on intent and
target separately; MIND on the act (decision models) and on DO+SAY (voices).
Run to write judge_v1.jsonl and print the tally."""
import json
from collections import Counter
from pathlib import Path

M = ["laya", "von", "gliclass", "qwen"]
# READ: per model "intent target"
READ = {
 "r01": "ow oo wo aw", "r02": "ww ww oo ow", "r03": "ow oo wo oo", "r04": "oo ww ww wo",
 "r05": "ww wo wo ao", "r06": "wo oo wo oo", "r07": "wo oo ao wo", "r08": "ow oo oo ow",
 "r09": "oo wo wo wo", "r10": "oo wo wo ow", "r11": "oo wo wo aw", "r12": "oo oo oo oo",
 "r13": "wo wo ow ow", "r14": "wo ow wo oo", "r15": "oo oo wo wo", "r16": "wa wa wa oo",
 "r17": "ao wo ww aw", "r18": "ww ow ow ow", "r19": "oo ww wo ww", "r20": "oo wo oo wo",
 "r21": "ao aw aw wo", "r22": "ow oo wo wo", "r23": "oo wo wo oo", "r24": "wa oa wa ao",
 "r25": "oo wo wo oo", "r26": "oo ow ow oo", "r27": "oo wo wo ao", "r28": "oo oo oo oo",
 "r29": "wo ow ow aw", "r30": "oo oo oo ww",
}
# MIND: decision models' act, then the two voices (qwen, reyna)
MIND = {
 "m01": ("wwwa", "aw", "freezing is what a timid woman with a knife at her throat does; attack/refuse are not"),
 "m02": ("oooa", "ow", "reyna's line contradicts itself"),
 "m03": ("oaww", "ow", "a guard known to take coin, unwatched, takes it or asks more"),
 "m04": ("wwww", "oo", "every decider misread flattery of a vain host; both voices got it"),
 "m05": ("oaao", "ow", ""),
 "m06": ("awaw", "ow", "reyna speaks as the neighbour"),
 "m07": ("oooo", "aw", "all four deciders: haggle"),
 "m08": ("owwa", "ww", "qwen's goblin runs from a dead end it cannot leave"),
 "m09": ("wowa", "aw", ""),
 "m10": ("wwoa", "ow", "reyna's war chief apologises"),
 "m11": ("owow", "aw", ""),
 "m12": ("wwww", "oa", "every decider has a healer sworn to heal refuse to"),
 "m13": ("awow", "ow", ""),
 "m14": ("wwow", "aw", ""),
 "m15": ("wwaw", "aw", ""),
 "m16": ("aoow", "aw", ""),
 "m17": ("waow", "wa", "reyna answers but invents a road name"),
 "m18": ("wwww", "ww", "nobody hands the purse to three drawn bows"),
 "m19": ("oooo", "ww", "deciders all right; both voices hand the purse over or join the bandit"),
 "m20": ("ooaw", "aa", ""),
 "m21": ("wowa", "ow", "reyna takes the bribe the keeper would never take"),
 "m22": ("awoa", "wa", ""),
 "m23": ("awow", "ww", "reyna breaks character: 'as an AI language model'"),
 "m24": ("aaoa", "ow", ""),
 "m25": ("awwo", "ww", ""),
 "m26": ("aaaa", "ww", "a fair dealer short on salt should just take a fair trade"),
 "m27": ("ooaw", "oa", "a rival's loyal envoy refusing to kneel is also fair"),
 "m28": ("awoa", "aw", ""),
 "m29": ("owww", "ww", ""),
 "m30": ("oaow", "wo", "qwen answers as the adventurer"),
}


def main():
    rows, tally = [], {}
    for cid, s in READ.items():
        for m, v in zip(M, s.split()):
            rows.append({"id": cid, "model": m, "intent": v[0], "target": v[1]})
            tally.setdefault((m, "read intent"), Counter())[v[0]] += 1
            tally.setdefault((m, "read target"), Counter())[v[1]] += 1
    for cid, (acts, voices, note) in MIND.items():
        for m, v in zip(M, acts):
            rows.append({"id": cid, "model": m, "act": v, "note": note})
            tally.setdefault((m, "mind act"), Counter())[v] += 1
        for m, v in zip(["voice:qwen", "voice:reyna"], voices):
            rows.append({"id": cid, "model": m, "act": v, "note": note})
            tally.setdefault((m, "free do+say"), Counter())[v] += 1
    out = Path(__file__).with_suffix(".jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{'model':12} {'judged on':12} {'ok':>4} {'arg':>4} {'wrong':>5}  ok+arg")
    for (m, what), c in tally.items():
        n = sum(c.values())
        print(f"{m:12} {what:12} {c['o']:4} {c['a']:4} {c['w']:5}  {(c['o'] + c['a']) / n:.0%}")


if __name__ == "__main__":
    main()
