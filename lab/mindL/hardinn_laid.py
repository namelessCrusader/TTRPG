"""Where the hard inn's hardship sits (2026-11-21, lane A, read only): per run of a setting, which scored nights the focal and the
partners were laid up (replayed from the night seeds: the episodes depend on seeds only), and the focal's drive error on those nights
against the others, for each chooser. Prints, 95% t intervals over seeds 1-10.

    lab/.venv/bin/python -m lab.mindL.hardinn_laid RUNS_DIR
"""
from __future__ import annotations

import sys

import numpy as np

from lab.mindL import step1_report as S1
from lab.mindL.tournament import WORLDS
from lab.world2.lifetime import night_seed
from lab.world2.world import World

NAMES = ("Tom", "Sir Ada", "Oren", "Fenn", "Marta")


def laid_nights(seed, nights):
    """per night: the set of people laid up (carry only the episodes, as the tournament loop does)."""
    flags = tuple(sorted(WORLDS["all6h"]))
    w, carry, out = World(seed=0, flags=flags), None, []
    for n in range(nights):
        w.laid = {}
        w._episodes(night_seed(seed, n), carry)
        out.append(set(w.laid))
        carry = {k: {"ill": (max(0, w.laid.get(k, 0) - 1), w.people[k].ail)} for k in NAMES}
    return out


def main():
    R = S1.load(sys.argv[1])
    rows = {c: dict(laid=[], ok=[], gap_laid=[], gap_ok=[]) for c in S1.CH}
    share = dict(focal=[], Fenn=[], Oren=[], anyone_else=[])
    for s in range(1, 11):
        if ("rules", s) not in R:
            continue
        r0 = R[("rules", s)]
        focal, w0, nn = r0["focal"], r0["win0"], len(r0["err"])
        L = laid_nights(s, nn)
        sc = range(w0, nn)
        share["focal"].append(np.mean([focal in L[n] for n in sc]))
        share["Fenn"].append(np.mean(["Fenn" in L[n] for n in sc]))
        share["Oren"].append(np.mean(["Oren" in L[n] for n in sc]))
        share["anyone_else"].append(np.mean([bool(L[n] - {focal}) for n in sc]))
        lay, ok = [n for n in sc if focal in L[n]], [n for n in sc if focal not in L[n]]
        for c in S1.CH:
            r = R.get((c, s))
            if not r:
                continue
            e = r["err"]
            rows[c]["laid"].append(np.mean([e[n] for n in lay]) if lay else np.nan)
            rows[c]["ok"].append(np.mean([e[n] for n in ok]))
            if c != "rules":
                rows[c]["gap_laid"].append(np.mean([R[("rules", s)]["err"][n] - e[n] for n in lay]) if lay else np.nan)
                rows[c]["gap_ok"].append(np.mean([R[("rules", s)]["err"][n] - e[n] for n in ok]))
    print("share of scored nights laid up:", {k: S1.fmt(S1.ci(v), 3) for k, v in share.items()})
    for c in S1.CH:
        x = rows[c]
        print(f"{c:13s} focal error laid-up nights {S1.fmt(S1.ci(x['laid']))}  other nights {S1.fmt(S1.ci(x['ok']))}" +
              (f"   gap (rules - this) laid-up {S1.fmt(S1.ci(x['gap_laid']))}  other {S1.fmt(S1.ci(x['gap_ok']))}" if c != "rules" else ""))


if __name__ == "__main__":
    main()
