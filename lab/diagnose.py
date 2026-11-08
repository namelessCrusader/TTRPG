"""WHY a decision model fails, one hypothesis at a time.

    lab/.venv/bin/python -m lab.diagnose                     # zero-shot Laya
    lab/.venv/bin/python -m lab.diagnose --ckpt lab/checkpoints/rooms_tuned.pt

A classifier's output cannot be read like generated text, so each possible
cause gets its own controlled test, where one thing changes and the rest is
held still. All of them run on states from the same randomised rooms
(lab/rooms.py), labelled by the same expert.

1. SWAP. Same menu, another room's state. If the pick does not move when the
   state is swapped, the model is scoring the options and ignoring the
   state. Compared with how often the EXPERT's answer moves under the same
   swap: that is how often it should.

2. OCCLUSION. Delete one line of the state at a time and measure how far the
   probability of the expert's answer moves. Which lines does the model lean
   on? The line naming the expert's current target should matter most.

3. MINIMAL PAIRS. Two states identical but for one fact, where the right
   answer flips with it. One suite per skill:
     direction  the target's bearing mirrored (3 west <-> 3 east)
     knowledge  "You know: the key is in X" present or not -> what to deal with
     holding    holding the note or not -> read it or go get it
   A skill is present when both halves of the pair come out right.

4. PROBES. Freeze the encoder, fit a linear readout from its hidden states to
   facts about the room (is the key's place known, is the key held, is the
   door locked, which way is the target, what kind of thing is it). Fitted
   on TRAIN-pool rooms, scored on TEST-pool rooms. If a fact reads out well
   and the decision that needs it still fails, the information is there and
   the head does not use it: a head-only fine-tune should fix it. If the fact
   does not read out, the encoder never represented it.
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict

import numpy as np
import torch

from lab.rooms import Room, STEPS

LIMBS = ("legs", "hands", "mouth")


def collect(pool, seeds, per_room=8, noise=0.3):
    """(room snapshot facts, state, questions, expert pick, target) from
    expert play with noise, sampled a few per room."""
    rng = random.Random(seeds[0])
    out = []
    for s in seeds:
        r = Room(s, pool)
        rows = []
        for _ in range(60):
            st, qs = r.state(), r.questions()
            best = r.expert()
            tq = r.target_question()
            t = r.target
            tpos = r.things[t]["pos"] if t in r.things else None
            facts = {
                "knows": any(r.key_box in k for k in r.known),
                "has_key": r.key in r.holding,
                "locked": r.things["the door"]["locked"],
                "target_kind": (r.things[t]["kind"] if t in r.things else "none"),
                "target_dir": _bearing(r.pos, tpos),
            }
            rows.append(dict(state=st, qs=qs, tq=tq, best=best, target=t, facts=facts,
                             where=(r.where(tpos) if tpos else None)))
            act = best if rng.random() > noise else {l: rng.choice(list(q["criteria"]))
                                                     for l, q in qs.items()}
            r.act(act)
            if r.out:
                break
        out += rng.sample(rows, min(per_room, len(rows)))
    return out


def _bearing(a, b):
    if b is None:
        return "none"
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west" if dx < 0 else "here"
    return "south" if dy > 0 else "north"


class Probe:
    def __init__(self, agent):
        from laya.common import build_sequence
        self.a, self.build = agent, build_sequence

    def probs(self, state, q):
        return self.a.predict(state, {"q": q})["answers"]["q"]["probabilities"]

    @torch.no_grad()
    def hidden(self, state, q, layers=(-1, 14)):
        from laya.common import collate_items
        cfg = self.a.cfg
        qi = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
        ids, marks = self.build(self.a.tok, state, qi, cfg["max_len"], cfg["head_max_len"])
        b = collate_items([[{"ids": ids, "markers": marks, "qtype": 0}]], self.a.tok.pad_token_id)
        out = self.a.model.encoder(input_ids=b["input_ids"].to(self.a.device),
                                   attention_mask=b["attention_mask"].to(self.a.device),
                                   output_hidden_states=True)
        # the STATE's tokens only: everything after the options' closing [SEP]
        sep = [i for i, t in enumerate(ids) if t == self.a.tok.sep_token_id]
        lo = sep[1] + 1 if len(sep) > 1 else 0
        return {L: out.hidden_states[L][0, lo:len(ids) - 1].float().mean(0).cpu().numpy()
                for L in layers}


# ------------------------------------------------------------------ 1. swap
def swap_test(P, rows, rng):
    """Legs: the menu is the same four steps and stay in almost every state,
    so a state from elsewhere can be put under the same question."""
    res = defaultdict(list)
    for limb in ("legs",):
        cand = [r for r in rows if len(r["qs"][limb]["criteria"]) >= 4]
        for r in cand:
            other = rng.choice([o for o in cand if o is not r
                                and set(o["qs"][limb]["criteria"]) == set(r["qs"][limb]["criteria"])] or [r])
            q = r["qs"][limb]
            p1, p2 = P.probs(r["state"], q), P.probs(other["state"], q)
            tv = 0.5 * sum(abs(p1[k] - p2[k]) for k in p1)
            res["tv"].append(tv)
            res["model_flips"].append(max(p1, key=p1.get) != max(p2, key=p2.get))
            res["expert_flips"].append(r["best"][limb] != other["best"][limb])
    return {k: float(np.mean(v)) for k, v in res.items()}


# ------------------------------------------------------------------ 2. occlusion
def _lines(state):
    head, around = state.split("Around you: ")
    parts = [l for l in head.strip().split("\n") if l]
    things = [t.strip() for t in around.rstrip(".").split("; ")]
    return parts, things


def _join(parts, things):
    return "\n".join(parts + ["Around you: " + "; ".join(things) + "."])


def occlusion(P, rows):
    """Mean drop in p(expert's answer) when a kind of line is removed."""
    drop = defaultdict(list)
    for r in rows:
        for limb, q in (("target", r["tq"]), ("legs", r["qs"]["legs"])):
            gold = r["target"] if limb == "target" else r["best"]["legs"]
            if gold not in q["criteria"]:
                continue
            base = P.probs(r["state"], q)[gold]
            parts, things = _lines(r["state"])
            for i, line in enumerate(parts):
                kind = ("goal" if line.startswith("You are locked") else
                        "holding" if line.startswith("You are holding") else
                        "known" if line.startswith("You know") else
                        "last" if line.startswith("A moment") else "other")
                s2 = _join(parts[:i] + parts[i + 1:], things)
                drop[(limb, kind)].append(base - P.probs(s2, q)[gold])
            for i, t in enumerate(things):
                kind = "TARGET's line" if r["target"] and t.startswith(r["target"]) else "other thing's line"
                s2 = _join(parts, things[:i] + things[i + 1:])
                drop[(limb, kind)].append(base - P.probs(s2, q)[gold])
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(drop.items())}


# ------------------------------------------------------------------ 3. minimal pairs
MIRROR = {"west": "east", "east": "west", "north": "south", "south": "north"}


def minimal_pairs(P, rows):
    out = {}
    # direction: the target's bearing mirrored on its own line, legs asked
    # while walking to it (the expert's step is along the dominant axis)
    hit = n = 0
    for r in rows:
        step = r["best"]["legs"]
        if not step.startswith("step ") or not r["target"] or not r["where"]:
            continue
        d = step[5:]
        m = MIRROR[d]
        q = r["qs"]["legs"]
        if f"step {m}" not in q["criteria"]:
            continue
        line_old = f"{r['target']}"
        flipped = re.sub(rf"(?<=: )(.*?)(?=;|\.$)",
                         lambda mo: re.sub(r"\b(north|south|east|west)\b",
                                           lambda w: MIRROR[w.group(1)], mo.group(1)),
                         r["state"].split("Around you: ")[1])
        # only the target's own line is mirrored
        things = r["state"].split("Around you: ")[1].rstrip(".").split("; ")
        things2 = [re.sub(r"\b(north|south|east|west)\b", lambda w: MIRROR[w.group(1)], t)
                   if t.startswith(line_old) else t for t in things]
        s2 = r["state"].split("Around you: ")[0] + "Around you: " + "; ".join(things2) + "."
        p1, p2 = P.probs(r["state"], q), P.probs(s2, q)
        ok = max(p1, key=p1.get) == f"step {d}" and max(p2, key=p2.get) == f"step {m}"
        hit += ok
        n += 1
    out["direction (target mirrored)"] = (hit, n, 1 / 25)
    # knowledge: add / remove "You know: the key is in X" -> target changes
    hit = n = 0
    for r in rows:
        f = r["facts"]
        if f["knows"] or f["has_key"] or r["target"] is None:
            continue
        box = next((t for t in r["tq"]["criteria"] if t in r["state"] and "(closed)" in
                    r["state"].split(t, 1)[1][:10]), None)
        if box is None:
            continue
        s2 = r["state"].replace("\nAround you:", f"\nYou know: The key is in {box}.\nAround you:", 1)
        p1, p2 = P.probs(r["state"], r["tq"]), P.probs(s2, r["tq"])
        ok = max(p1, key=p1.get) == r["target"] and max(p2, key=p2.get) == box
        hit += ok
        n += 1
    k = np.mean([len(r["tq"]["criteria"]) for r in rows])
    out["knowledge (told where the key is)"] = (hit, n, 1 / k ** 2)
    return out


# ------------------------------------------------------------------ 4. probes
def probes(P, train_rows, test_rows):
    facts = ["knows", "has_key", "locked", "target_kind", "target_dir"]
    q = lambda r: r["qs"]["legs"]
    Htr = [P.hidden(r["state"], q(r)) for r in train_rows]
    Hte = [P.hidden(r["state"], q(r)) for r in test_rows]
    res = {}
    for L in Htr[0]:
        Xtr = np.stack([h[L] for h in Htr])
        Xte = np.stack([h[L] for h in Hte])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        for f in facts:
            ytr = [str(r["facts"][f]) for r in train_rows]
            yte = [str(r["facts"][f]) for r in test_rows]
            classes = sorted(set(ytr))
            if len(classes) < 2:
                continue
            idx = {c: i for i, c in enumerate(classes)}
            Y = torch.tensor([idx[y] for y in ytr])
            X = torch.tensor(Xtr, dtype=torch.float32)
            W = torch.zeros(X.shape[1], len(classes), requires_grad=True)
            b = torch.zeros(len(classes), requires_grad=True)
            opt = torch.optim.LBFGS([W, b], max_iter=200)

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(X @ W + b, Y) + 1e-2 * (W ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            pred = (torch.tensor(Xte, dtype=torch.float32) @ W + b).argmax(1).numpy()
            keep = [i for i, y in enumerate(yte) if y in idx]
            acc = float(np.mean([pred[i] == idx[yte[i]] for i in keep]))
            maj = max(np.mean([y == c for y in yte]) for c in classes)
            res[(f, "last" if L == -1 else f"layer {L}")] = (acc, float(maj))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--rooms", type=int, default=40)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import laya
    agent = laya.load("convaiinnovations/laya", device=args.device)
    if args.ckpt:
        agent.model.load_state_dict(torch.load(args.ckpt, map_location=agent.device))
        agent.model.eval()
    P = Probe(agent)
    rng = random.Random(0)
    test = collect("test", list(range(60_000, 60_000 + args.rooms)))
    train = collect("train", list(range(61_000, 61_000 + args.rooms)))
    tag = "TUNED" if args.ckpt else "ZERO-SHOT"
    print(f"== {tag}: {len(test)} test-pool states, {len(train)} train-pool states", flush=True)

    sw = swap_test(P, test, rng)
    print(f"1. SWAP (legs, another room's state, same menu): model changes its pick "
          f"{sw['model_flips']:.0%} of the time, the expert {sw['expert_flips']:.0%}; "
          f"mean shift in probability {sw['tv']:.2f}", flush=True)

    occ = occlusion(P, test[:60])
    print("2. OCCLUSION: mean drop in p(expert's answer) when a line is removed", flush=True)
    for (limb, kind), (d, n) in occ.items():
        print(f"   {limb:7} {kind:20} {d:+.3f}  (n={n})", flush=True)

    mp = minimal_pairs(P, test)
    print("3. MINIMAL PAIRS: both halves right", flush=True)
    for k, (h, n, ch) in mp.items():
        print(f"   {k:36} {h}/{n}  (blind guessing ~{ch:.0%})", flush=True)

    pr = probes(P, train, test)
    print("4. PROBES: linear readout from the encoder, fit on train-pool rooms, scored on test-pool", flush=True)
    for (f, L), (acc, maj) in sorted(pr.items()):
        print(f"   {f:12} {L:9} {acc:.0%}  (majority {maj:.0%})", flush=True)


if __name__ == "__main__":
    main()
