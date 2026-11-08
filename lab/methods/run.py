"""Run ONE scorer over the cases. One model on the GPU at a time.

    lab/.venv/bin/python -m lab.methods.run laya_en:all [--limit 30]

Before it loads anything it checks free GPU memory against the model's size,
and refuses rather than crash or fall back to the CPU. Each case's result is
appended as soon as it is done, and cases already in the output are skipped,
so a run the heat guard paused, or one that was stopped, simply continues.

Per case it records:
    ok     the pick is in the acceptable set
    mass   how much of the probability falls on the acceptable set
           (only for methods whose scores are probabilities over the options)
    first  the rank of the first acceptable option (1 = best)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from lab.methods.score import _softmax, make

HERE = Path(__file__).parent
OUT = HERE.parent / "results" / "methods"


def gpu_check(params):
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: refusing to run on the CPU")
    free, total = torch.cuda.mem_get_info()
    # bf16 weights, activations, slack. The 1.4 was too careful for a 2B model
    # (it asked 6.2 GB for one that runs in about 4.5): LAB_GPU_FACTOR lowers it.
    need = params * 2 * float(os.environ.get("LAB_GPU_FACTOR", "1.4")) + 0.6e9
    print(f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB; this model needs about "
          f"{need / 2**30:.1f} GiB", flush=True)
    if free < need:
        raise SystemExit("not enough GPU memory: is another job running?")


def invariance(s, c, k):
    """The same case asked in k orders. Picks are mapped back to the original
    positions, so an order-invariant model gives the same pick every time."""
    import random
    n, good = len(c["options"]), set(c["good"])
    rng = random.Random(c["id"])
    orders = [list(range(n)), list(range(n))[::-1]]
    while len(orders) < k:
        o = list(range(n))
        rng.shuffle(o)
        orders.append(o)
    picks, masses = [], []
    try:
        for od in orders[:k]:
            r = s.rank(c["situation"], c["question"], [c["options"][i] for i in od])
            picks.append(od[r["pick"]])
            if r.get("prob") and r["scores"] is not None:
                sc = r["scores"]
                p = sc if abs(sum(sc) - 1) < 1e-3 else _softmax(sc)
                masses.append(sum(p[j] for j, i in enumerate(od) if i in good))
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)[:200]}
    out = {"picks": picks, "same": len(set(picks)) == 1,
           "ok_orders": sum(p in good for p in picks)}
    if masses:
        out["mass_spread"] = max(masses) - min(masses)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scorer")
    ap.add_argument("--cases", default=str(HERE / "cases_things.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="cases per cell (0 = all)")
    ap.add_argument("--only", default="", help="factor[=value],... to run a subset")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. smoke)")
    ap.add_argument("--perms", type=int, default=0,
                    help="INVARIANCE TEST: ask each case in this many orders (original, "
                         "reversed, then shuffles) and record whether the pick changes")
    ap.add_argument("--keep-scores", action="store_true",
                    help="store every option's score (for short lists and the creativity report)")
    ap.add_argument("--capture", action="store_true",
                    help="also save what the model sees (GLiNER-Decide: each option's vector, the text's "
                         "vector, the raw logits) in <output>.cap/partN.npz; implies --keep-scores")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases)]
    if args.only:
        want = args.only.split(",")
        cases = [c for c in cases if any(
            w == c["factor"] or w == f"{c['factor']}={c['value']}" for w in want)]
    if args.limit:
        seen = {}
        keep = []
        for c in cases:
            k = (c["factor"], str(c["value"]))
            if seen.get(k, 0) < args.limit:
                keep.append(c)
                seen[k] = seen.get(k, 0) + 1
        cases = keep

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("inv" if args.perms else "")
    out = OUT / f"{args.scorer.replace(':', '__')}{'.' + tag if tag else ''}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out)}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{args.scorer}: {len(todo)} to do, {len(done)} already done", flush=True)
    if not todo:
        return

    s = make(args.scorer)
    if os.environ.get("LAB_DEVICE", "cuda") == "cuda":
        gpu_check(s.params)
    t0 = time.time()
    s.load()
    if args.capture:
        args.keep_scores = True
        s.capture = True
        cap_dir = out.with_suffix(".cap")
        cap_dir.mkdir(exist_ok=True)
        part = len(list(cap_dir.glob("part*.npz")))
        buf = []

        def flush():
            nonlocal part, buf
            if not buf:
                return
            import numpy as np
            np.savez_compressed(
                cap_dir / f"part{part}.npz",
                ids=np.array([b[0] for b in buf]),
                n=np.array([b[1]["opts"].shape[0] for b in buf]),
                opts=np.concatenate([b[1]["opts"].numpy() for b in buf]).astype("float16"),
                logits=np.concatenate([b[1]["logits"].numpy() for b in buf]).astype("float32"),
                text_first=np.stack([b[1]["text_first"].numpy() for b in buf]).astype("float16"),
                text_mean=np.stack([b[1]["text_mean"].numpy() for b in buf]).astype("float16"))
            part += 1
            buf = []
    on_gpu = os.environ.get("LAB_DEVICE", "cuda") == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    with out.open("a") as f:
        for n, c in enumerate(todo):
            t = time.perf_counter()
            rec = {"id": c["id"], "factor": c["factor"], "value": c["value"],
                   "n": len(c["options"]), "good": c["good"]}
            if args.perms:
                rec.update(invariance(s, c, args.perms))
                rec["ms"] = (time.perf_counter() - t) * 1000
                f.write(json.dumps(rec) + "\n")
                f.flush()
                continue
            try:
                r = s.rank(c["situation"], c["question"], c["options"])
                if args.capture and "capture" in r:
                    buf.append((c["id"], r.pop("capture")))
                good = set(c["good"])
                rec["pick"] = r["pick"]
                rec["ok"] = r["pick"] in good
                # None when no option is acceptable (a short list that lost them all)
                rec["first"] = next((i + 1 for i, j in enumerate(r["order"]) if j in good), None)
                if r.get("prob") and r["scores"] is not None:
                    sc = r["scores"]
                    p = sc if abs(sum(sc) - 1) < 1e-3 else _softmax(sc)
                    rec["mass"] = sum(p[i] for i in good)
                if (c.get("keep_scores") or args.keep_scores) and r["scores"] is not None:
                    # the creativity report draws from these (lab.methods.creative)
                    rec["scores"] = [float(x) for x in r["scores"]]
                    rec["prob"] = bool(r.get("prob"))
            except (ValueError, RuntimeError) as e:
                # Laya cannot fit very long lists: that is a result, not a crash
                rec["error"] = str(e)[:200]
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
            rec["ms"] = (time.perf_counter() - t) * 1000
            f.write(json.dumps(rec) + "\n")
            f.flush()
            # REST BETWEEN CASES. The laptop overshot to 100 C with a guard
            # that only pauses at a threshold; a steady duty cycle keeps it
            # cool instead. LAB_DUTY=1 turns the rest off.
            duty = float(os.environ.get("LAB_DUTY", "0.5"))
            if duty < 1:
                time.sleep(rec["ms"] / 1000 * (1 / duty - 1))
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
                if args.capture:
                    flush()
    if args.capture:
        flush()
    print(f"done; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if on_gpu else "done (CPU)",
          flush=True)


if __name__ == "__main__":
    main()
