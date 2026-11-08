"""Which zero-shot scorer points at the right thing in a long list?

    lab/.venv/bin/python -m lab.calib [--lm Qwen/Qwen3-0.6B]

The task (lab/listsize.py): one need, one right item, N-1 distractors, the
right one at a random place. "Kael needs to cut the rope" -> the knife.

Five scorers:

  words       content-word overlap between the need and the item: the control
  laya        Laya's own distribution over the options
  laya+cal    Laya calibrated: its log-probability for each option minus the
              log-probability it gives the SAME options when the need is
              replaced by "do something". Laya picked "the door" from a state
              that said "banana", and "stay" everywhere: that is a prior over
              the option strings, and this divides it out (Zhao et al. 2021,
              "Calibrate before use"; Holtzman et al. 2021, "Surface form
              competition").
  lm          a small causal LM scores each legal name as the continuation of
              "...Kael takes": the log-probability of the whole name. This is
              the grammar-constrained slot: only legal fillers are scored, a
              multi-word name is scored token by token, and nothing had to be
              embedded or stored.
  lm+cal      the same, minus the name's log-probability with the need removed
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch  # noqa: E402

torch.set_num_threads(4)
src = open(os.path.join(os.path.dirname(__file__), "listsize.py")).read()
exec(src.split("a = laya.load")[0].split("import laya")[1])        # NEEDS, JUNK

STOP = set("the a an of to in on at is it and or for with up from".split())


def words(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP}


def state(need):
    return f"Kael needs to {need}. He looks at what is in the storeroom."


NULL = "Kael needs to do something. He looks at what is in the storeroom."


class LM:
    def __init__(self, repo):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def scores(self, prompt, cands, bs=16):
        p = self.tok(prompt, add_special_tokens=False)["input_ids"]
        seqs = [p + self.tok(" " + c + ".", add_special_tokens=False)["input_ids"] for c in cands]
        out = []
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i + bs]
            L = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, s in enumerate(chunk):
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            # the output layer only where a candidate's tokens are predicted:
            # the whole vocabulary at every position of 16 sequences was 3 GB
            h = self.m.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            for j, s in enumerate(chunk):
                tgt = torch.tensor(s[len(p):]).cuda()
                pos = torch.arange(len(p) - 1, len(s) - 1).cuda()
                lp = torch.log_softmax(self.m.lm_head(h[j, pos]).float(), -1)
                out.append(float(lp[torch.arange(len(tgt)), tgt].sum()))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--sizes", default="4,16,32,64,80")
    ap.add_argument("--no-laya", action="store_true")
    args = ap.parse_args()
    laya_agent = None
    if not args.no_laya:
        import laya
        laya_agent = laya.load("convaiinnovations/laya")
    lm = LM(args.lm)
    print(f"{'N':>4} {'chance':>7} " + " ".join(f"{k:>9}" for k in
          ("words", "laya", "laya+cal", "lm", "lm+cal")), flush=True)
    for N in map(int, args.sizes.split(",")):
        rng = random.Random(N)
        hit = {k: 0 for k in ("words", "laya", "laya+cal", "lm", "lm+cal")}
        for need, item in NEEDS:
            opts = rng.sample([j for j in JUNK if j != item], N - 1) + [item]
            rng.shuffle(opts)
            w = {o: len(words(need) & words(o)) + rng.random() * 1e-3 for o in opts}
            hit["words"] += max(w, key=w.get) == item
            if laya_agent:
                q = {"q": {"type": "choice", "instructions": "What does Kael take?",
                           "criteria": {o: None for o in opts}}}
                p = laya_agent.predict(state(need), q)["answers"]["q"]["probabilities"]
                p0 = laya_agent.predict(NULL, q)["answers"]["q"]["probabilities"]
                hit["laya"] += max(p, key=p.get) == item
                cal = {o: math.log(p[o] + 1e-9) - math.log(p0[o] + 1e-9) for o in opts}
                hit["laya+cal"] += max(cal, key=cal.get) == item
            listing = "In the storeroom: " + ", ".join(opts) + "."
            s1 = lm.scores(f"{state(need)} {listing} To {need}, Kael takes", opts)
            s0 = lm.scores(f"{NULL} {listing} To do something, Kael takes", opts)
            hit["lm"] += opts[max(range(N), key=lambda i: s1[i])] == item
            hit["lm+cal"] += opts[max(range(N), key=lambda i: s1[i] - s0[i])] == item
        n = len(NEEDS)
        print(f"{N:>4} {1 / N:>7.0%} " + " ".join(f"{hit[k]:>6}/{n}" for k in hit), flush=True)


if __name__ == "__main__":
    main()
