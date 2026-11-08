"""Is escaping a room LEARNABLE by Laya? Fine-tune on an expert in TRAIN rooms,
then play rooms built from names it has never seen.

    lab/.venv/bin/python -m lab.tune --rooms 400 --epochs 2

Three things are measured, and the zero-shot model is measured the same way
first so there is something to compare against:

1. POINTER over the room: "of everything here, what do you deal with next?" —
   every thing in the room is an option, 8 to 15 of them.
2. Each limb, asked flat ("which way do you step now?") and CHAINED, told the
   target ("You are dealing with the wardrobe. Which way do you step now?").
3. Closed loop: the model plays whole rooms, and the milestones say how far.

The training rooms use one name pool and the test rooms another (lab/rooms.py):
no container, key, item, furniture or person is shared. Passing a test room
means the skill moved, not the vocabulary.

Only the top of the encoder and the decision head are trained — six of the
28 layers — which is what fits in 6 GB, and the rest stays what it was.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from lab.rooms import MILESTONES, Room, play

HERE = Path(__file__).parent
LIMBS = ("legs", "hands", "mouth")


def chained(q, target):
    return dict(q, instructions=f"You are dealing with {target}. {q['instructions']}")


# ------------------------------------------------------------------- data
def harvest(n_rooms, pool, seed0, noise=0.25):
    """Expert play with some random acts mixed in, labelled by the expert.

    The noise is what teaches recovery: an expert that never errs never
    visits the states a learner's own mistakes lead to, and a learner trained
    only on its path cannot get back to it (DAgger's point, in its cheapest
    form). The label is always what the expert would do from where it is."""
    rng = random.Random(seed0)
    rows = []
    for i in range(n_rooms):
        r = Room(seed0 + i, pool)
        for _ in range(60):
            st, qs = r.state(), r.questions()
            best = r.expert()
            tq = r.target_question()
            if r.target in tq["criteria"]:
                rows.append((st, tq, r.target))
            for l in LIMBS:
                if len(qs[l]["criteria"]) > 1:
                    rows.append((st, qs[l], best[l]))
                    if r.target:
                        rows.append((st, chained(qs[l], r.target), best[l]))
            act = best if rng.random() > noise else {
                l: rng.choice(list(q["criteria"])) for l, q in qs.items()}
            r.act(act)
            if r.out:
                break
    return rows


def encode(agent, rows, shuffle=True, rng=None):
    from laya.common import QTYPES, build_sequence
    cfg, items, dropped = agent.cfg, [], 0
    for st, q, gold in rows:
        keys = list(q["criteria"])
        order = list(range(len(keys)))
        if shuffle:
            rng.shuffle(order)          # position must not be the answer
        qi = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
        ids, marks = build_sequence(agent.tok, st, qi, cfg["max_len"], cfg["head_max_len"],
                                    option_order=order)
        if len(marks) != len(keys):
            dropped += 1
            continue
        items.append({"ids": ids, "markers": marks, "qtype": QTYPES["choice"],
                      "label": order.index(keys.index(gold))})
    return items, dropped


# ------------------------------------------------------------------- training
def trainable(model, top_layers):
    for p in model.parameters():
        p.requires_grad = False
    enc = model.encoder
    for layer in enc.layers[-top_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in enc.final_norm.parameters():
        p.requires_grad = True
    for mod in (model.head, model.type_emb, model.scorer):
        for p in mod.parameters():
            p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def train(agent, items, epochs, top_layers, bs=8, log=print):
    from laya.common import collate_items
    model, dev = agent.model, agent.device
    params = trainable(model, top_layers)
    head = [p for m in (model.head, model.type_emb, model.scorer) for p in m.parameters()]
    enc = [p for p in params if all(p is not h for h in head)]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2e-5}, {"params": head, "lr": 1e-4}],
                            weight_decay=0.01)
    steps = epochs * ((len(items) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-4], total_steps=steps,
                                                pct_start=0.06)
    model.train()
    rng, t0, k = random.Random(0), time.time(), 0
    for ep in range(epochs):
        rng.shuffle(items)
        run_loss, run_hit, n = 0.0, 0, 0
        for i in range(0, len(items), bs):
            b = collate_items([items[i:i + bs]], agent.tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                  b["marker_pos"].to(dev), b["marker_mask"].to(dev),
                                  b["qtype"].to(dev))
            y = b["label"].to(dev)
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            k += 1
            run_loss += loss.item() * len(y)
            run_hit += int((logits.argmax(-1) == y).sum())
            n += len(y)
            if k % 200 == 0:
                log(f"  epoch {ep} step {k}/{steps}  loss {run_loss / n:.3f}  "
                    f"train acc {run_hit / n:.2f}  {time.time() - t0:.0f}s")
                run_loss, run_hit, n = 0.0, 0, 0
    model.eval()


# ------------------------------------------------------------------- measuring
def accuracy(agent, rows):
    """Offline: of held-out expert decisions, how many does the model make."""
    by = {}
    for st, q, gold in rows:
        kind = ("target" if q["instructions"].startswith("Of everything")
                else ("chained " if q["instructions"].startswith("You are dealing") else "")
                + next(l for l in LIMBS if Room.QUESTIONS[l] in q["instructions"]))
        pick = agent.predict(st, {"q": q})["answers"]["q"]["choice"]
        hit, n = by.get(kind, (0, 0))
        by[kind] = (hit + (pick == gold), n + 1)
    return {k: (h / n, n) for k, (h, n) in sorted(by.items())}


def closed_loop(agent, pool, seeds, mode):
    counts = {m: 0 for m in MILESTONES}
    ticks = []

    def choose(room, st, qs):
        pick = {}
        target = None
        if mode == "chained":
            target = agent.predict(st, {"q": room.target_question()})["answers"]["q"]["choice"]
        for l, q in qs.items():
            if len(q["criteria"]) == 1:
                pick[l] = next(iter(q["criteria"]))
                continue
            qq = chained(q, target) if target else q
            pick[l] = agent.predict(st, {"q": qq})["answers"]["q"]["choice"]
        return pick

    for s in seeds:
        log = []
        r = play(Room(s, pool), choose, log=log)
        ticks.append(len(log))
        for m in MILESTONES:
            counts[m] += m in r.done
    return counts, ticks


def report(title, counts, n, out):
    line = f"{title:36} " + " | ".join(f"{m.split()[0] if m != 'out' else 'OUT'} {counts[m]}/{n}"
                                        for m in MILESTONES)
    print(line, flush=True)
    out.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--top-layers", type=int, default=6)
    ap.add_argument("--eval-rooms", type=int, default=40)
    ap.add_argument("--skip-zero-shot", action="store_true")
    args = ap.parse_args()

    import laya
    agent = laya.load("convaiinnovations/laya")
    rng = random.Random(0)
    out = []
    test_rows = harvest(30, "test", 50_000, noise=0.25)
    seen_rows = harvest(30, "train", 90_000, noise=0.25)
    test_seeds = list(range(70_000, 70_000 + args.eval_rooms))
    seen_seeds = list(range(80_000, 80_000 + args.eval_rooms))

    def measure(tag):
        for name, rows in (("unseen names", test_rows), ("train names, new rooms", seen_rows)):
            acc = accuracy(agent, rows)
            line = f"{tag} / {name}: " + ", ".join(f"{k} {a:.0%} (n={n})" for k, (a, n) in acc.items())
            print(line, flush=True)
            out.append(line)
        for pool, seeds, name in (("test", test_seeds, "unseen names"),
                                  ("train", seen_seeds, "train names, new rooms")):
            for mode in ("flat", "chained"):
                c, t = closed_loop(agent, pool, seeds, mode)
                report(f"{tag} {mode} / {name}", c, len(seeds), out)

    if not args.skip_zero_shot:
        measure("ZERO-SHOT")
    t0 = time.time()
    rows = harvest(args.rooms, "train", 0)
    items, dropped = encode(agent, rows, rng=rng)
    msg = f"training on {len(items)} decisions from {args.rooms} rooms ({dropped} dropped: menu over budget)"
    print(msg, flush=True)
    out.append(msg)
    train(agent, items, args.epochs, args.top_layers)
    msg = f"trained in {time.time() - t0:.0f}s"
    print(msg, flush=True)
    out.append(msg)
    ck = HERE / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save({k: v for k, v in agent.model.state_dict().items()}, ck / "rooms_tuned.pt")
    measure("TUNED")
    (HERE / "results" / "tune_rooms.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
