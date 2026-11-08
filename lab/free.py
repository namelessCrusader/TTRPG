"""Let the voices decide AND speak, with nothing decided for them.

    lab/.venv/bin/python -m lab.free --voice qwen

`speak.py` hands a voice the act and checks it was carried. This does not:
the voice gets the person, where they are and what just happened, and says
what they say. Whether that is a reasonable thing for THIS person to do here
is judged case by case afterwards (`review.py` lays every model's answer to a
card side by side for that), not against a key written in advance — a key
only measures agreement with whoever wrote it.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from lab.speak import VOICES

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="qwen", choices=sorted(VOICES))
    ap.add_argument("--cards", default=str(HERE / "cards" / "social_v1.jsonl"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = VOICES[args.voice]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    lm = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    rows = []
    for c in map(json.loads, open(args.cards)):
        if c["track"] != "mind":
            continue
        s = c["state"]
        msgs = [
            {"role": "system", "content":
             f"You are {s['person']}. Stay in character. Answer with what you do, "
             "in a few words, then what you say out loud, like this:\n"
             "DO: <what you do>\nSAY: <your words>\n"
             "Use no names or places that are not mentioned below."},
            {"role": "user", "content":
             f"Where you are: {s['situation']}\nWhat just happened: {s['what_happened']}"},
        ]
        kw = {"enable_thinking": False} if "Qwen3" in repo else {}
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True, **kw)
        ids = tok(prompt, return_tensors="pt").to(dev)
        t = time.perf_counter()
        with torch.no_grad():
            out = lm.generate(**ids, max_new_tokens=80, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        do = re.search(r"DO:\s*(.+)", text)
        say = re.search(r"SAY:\s*(.+)", text)
        rows.append({"id": c["id"], "raw": text,
                     "do": do.group(1).strip() if do else None,
                     "say": say.group(1).strip().strip('"') if say else None,
                     "ms": (time.perf_counter() - t) * 1000})

    out = HERE / "results" / f"free__{args.voice}.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = sum(bool(r["do"] and r["say"]) for r in rows)
    print(f"{args.voice}: {ok}/{len(rows)} kept the DO/SAY form, "
          f"{sum(r['ms'] for r in rows) / len(rows):.0f} ms a card -> {out}")


if __name__ == "__main__":
    main()
