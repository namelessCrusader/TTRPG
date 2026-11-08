"""The decision is made; can a small model SAY it, and say only it?

    lab/.venv/bin/python -m lab.speak --voice qwen --reader laya

This is the combination the lab exists for. The division of labour we want in
play is: the sim (or an encoder) decides WHAT a person does, and a small
decoder turns that into words. Each half is cheap only if the other can be
trusted, so this checks the seam between them, three ways, per MIND card:

  1. SPEAK. The voice is told who they are, what happened, and what they have
     decided — the card's gold act and fear — and says one line.
  2. HEARD. A decision model is shown only the person and the LINE, and asked
     the same `reply` question. If it cannot recover the act from the words,
     the words did not carry it: a refusal that reads as haggling is a
     different game. (This is the "hearing a line is hearing what it said"
     rule, applied to our own mouths.)
  3. GROUNDED. GLiNER pulls every person, place, group and thing named in the
     line; any that the card never mentioned was invented. A voice that makes
     up a brother, a city or a sword has written facts into the world that the
     sim does not hold — the thing the whole design is built to stop.

The stages run one after another, each model freed before the next loads:
6 GB of VRAM holds any one of them, and the laptop's RAM holds one comfortably.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

from lab.cards.social_v1 import FEAR, REPLIES
from lab.models import MODELS, state_text

HERE = Path(__file__).parent

VOICES = {"qwen": "Qwen/Qwen3-0.6B",
          "reyna": "aloobun/Reyna-RP-Qwen1.5-0.5B-Chat-v0.1"}
GLINER = "urchade/gliner_medium-v2.1"
KINDS = ["person", "place", "organization", "object", "creature"]


def _free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def speak(cards, repo):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    lm = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    lines = []
    for c in cards:
        s, g = c["state"], c["gold"]
        msgs = [
            {"role": "system", "content":
             f"You are {s['person']}. Stay in character. Say ONE short line OUT "
             "LOUD to the person in front of you — the words themselves, not a "
             "description of what you do or feel. No narration, no quotation "
             "marks, no names or places that are not mentioned below."},
            {"role": "user", "content":
             f"Where you are: {s['situation']}\nWhat just happened: {s['what_happened']}\n"
             f"You have decided: you {REPLIES[g['reply']]}.\n"
             f"You feel: {FEAR[g['fear']]}.\nWhat do you say to them?"},
        ]
        kw = {"enable_thinking": False} if "Qwen3" in repo else {}
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, **kw)
        ids = tok(prompt, return_tensors="pt").to(dev)
        t = time.perf_counter()
        with torch.no_grad():
            out = lm.generate(**ids, max_new_tokens=48, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        ms = (time.perf_counter() - t) * 1000
        text = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        text = text.splitlines()[0].strip().strip('"') if text else ""
        lines.append({"id": c["id"], "line": text, "ms": ms,
                      "tokens": int(out.shape[1] - ids.input_ids.shape[1])})
    vram = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
    del lm
    return lines, vram


def hear(cards, lines, reader):
    m = MODELS[reader]()
    m.load()
    got = []
    for c, ln in zip(cards, lines):
        q = {"reply": c["questions"]["reply"], "fear": c["questions"]["fear"]}
        q = {"reply": dict(q["reply"], instructions="What is this person doing with these words?"),
             "fear": dict(q["fear"], instructions="How afraid does this person sound?")}
        a = m.decide(state_text({"person": c["state"]["person"], "says": ln["line"]}), q)
        got.append({"reply": a["reply"].get("answer"), "fear": a["fear"].get("answer")})
    vram = m.vram_mb()
    del m
    return got, vram


def _named(line, e):
    """Does this span carry a capital that is not just the start of a
    sentence or the word I?"""
    words = e["text"].split()
    for i, w in enumerate(words):
        before = line[:e["start"]].rstrip() if i == 0 else "x"
        sentence_start = i == 0 and (not before or before[-1] in ".!?\"")
        if w[:1].isupper() and w not in ("I", "I'm", "I'll") and not sentence_start:
            return True
    return False


def ground(cards, lines):
    from gliner import GLiNER as G
    m = G.from_pretrained(GLINER)
    try:
        import torch
        if torch.cuda.is_available():
            m = m.to("cuda")
    except Exception:
        pass
    out = []
    for c, ln in zip(cards, lines):
        known = " ".join(str(v) for v in c["state"].values()).lower()
        ents = m.predict_entities(ln["line"], KINDS, threshold=0.5)
        # A NAME, not a noun. "the guards" and "no one" are words anybody may
        # use; "Harrowgate" or "my brother Tomas" is a fact the scene must hold.
        # So only spans with a capitalised word past the line's first count.
        new = [e["text"] for e in ents
               if e["text"].lower() not in known and _named(ln["line"], e)]
        out.append({"entities": [(e["text"], e["label"]) for e in ents], "invented": new})
    del m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="qwen", choices=sorted(VOICES))
    ap.add_argument("--reader", default="laya", choices=sorted(MODELS))
    ap.add_argument("--cards", default=str(HERE / "cards" / "social_v1.jsonl"))
    args = ap.parse_args()

    cards = [c for c in map(json.loads, open(args.cards)) if c["track"] == "mind"]

    lines, v_speak = speak(cards, VOICES[args.voice])
    _free()
    heard, v_hear = hear(cards, lines, args.reader)
    _free()
    grounded = ground(cards, lines)

    rows = []
    for c, ln, h, g in zip(cards, lines, heard, grounded):
        rows.append({**ln, **g, "decided": c["gold"]["reply"], "heard": h["reply"],
                     "fear": c["gold"]["fear"], "heard_fear": h["fear"]})
    out = HERE / "results" / f"speak__{args.voice}__{args.reader}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    act = sum(r["heard"] == r["decided"] for r in rows) / n
    fear = sum(r["heard_fear"] is not None and abs(r["heard_fear"] - r["fear"]) <= 1
               for r in rows) / n
    inv = sum(bool(r["invented"]) for r in rows) / n
    print(f"voice {args.voice}: {sum(r['ms'] for r in rows) / n:.0f} ms a line, "
          f"{sum(r['tokens'] for r in rows) / n:.0f} tokens, peak VRAM {v_speak:.0f} MB")
    print(f"reader {args.reader}: peak VRAM {v_hear:.0f} MB")
    print(f"act carried by the words   {act:.0%}")
    print(f"fear carried (within one)  {fear:.0%}")
    print(f"lines inventing a name     {inv:.0%}")
    for r in rows:
        mark = "ok " if r["heard"] == r["decided"] else "-- "
        extra = f"  INVENTED {r['invented']}" if r["invented"] else ""
        print(f"{mark}{r['id']} {r['decided']:>9} -> {str(r['heard']):9} | {r['line'][:90]}{extra}")
    print(out)


if __name__ == "__main__":
    main()
