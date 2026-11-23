"""Fam6 gate (ii) and (iii) on CONTROLLED pasts (mindL5 stage A). Same exposure for every pair; only who lived with whom and who was seen
caring differs. lab/.venv/bin/python -m lab.mindL.gate_kin   (seconds; no world is run)."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.stats as st

from lab.mindL import household as H
from lab.mindL.concerns2 import CN2, KinConcerns, dials

SCR = Path("/home/nisargparikh/.cache/mark1-build/mindL5/A")
RES = Path(__file__).resolve().parents[1] / "results" / "mindL5" / "A"
N = 30


def ci(xs):
    x = np.asarray(xs, float)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def fm(c, d=3):
    return f"{c[0]:+.{d}f} [{c[1]:+.{d}f}, {c[2]:+.{d}f}]"


def household(S_join=-0.25):
    people = ["P1", "P2", "B", "A", "S", "L", "PL"]
    age = dict(P1=34, P2=32, B=10, A=10, S=10, L=30, PL=30)
    birth = {n: -float(a) for n, a in age.items()}
    spans = dict(P1=[(-14.0, 0.0)], P2=[(-14.0, 0.0)], B=[(-10.0, 0.0)], A=[(-8.0, 0.0)], S=[(S_join, 0.0)], L=[(-1.0, 0.0)], PL=[(-5.0, 0.0)])
    return people, age, birth, spans


def uniform(E):
    """The mean exposure of the pairs that engaged (the sample's), the same for every pair of the controlled past."""
    es = [e for row in E.values() for e in row.values() if e["eng"] > 0]
    U = {x: sum(e[x] for e in es) / len(es) for x in H.EXPO_KEYS}
    feat: dict = {}
    for e in es:
        for f, v in e["feat"].items():
            feat[f] = feat.get(f, 0.0) + v / len(es)
    U["feat"] = feat
    child_obs = [e["obs"] for i, row in E.items() for j, e in row.items() if H.cls(H.FAMILY_AGE[j]) == "child"]
    U["obs_child"] = sum(child_obs) / len(child_obs)
    care_ej = [e["ej"] for i, row in E.items() for j, e in row.items() if H.cls(H.FAMILY_AGE[i]) == "adult" and H.cls(H.FAMILY_AGE[j]) == "child"]
    U["ej_child"] = sum(care_ej) / len(care_ej)
    return U


def make_fn(U, spans, birth):
    def fn(i, j, t):
        e = {x: U[x] for x in H.EXPO_KEYS}
        e["feat"] = dict(U["feat"])
        e["res"] = 1.0
        e["ej"] = U["ej_child"] if (i in ("P1", "P2") and j in ("B", "A", "S")) else 0.0
        if "PL" in (i, j):                                  # present, never acted at or with
            e.update(eng=0.0, dbar=0.0, ej=0.0)
            e["feat"] = {"presence": U["bouts"]}
        car = [c for c in ("P1", "P2") if H.present(spans, c, t) and c not in (i, j)]
        e["obs_by"] = {c: U["obs_child"] / 2 for c in car} if j in ("B", "A", "S") and "PL" not in (i, j) else {}
        e["obs"] = sum(e["obs_by"].values())
        return e
    return fn


def run(E, seed, S_join=-0.25, scramble=False):
    people, age, birth, spans = household(S_join)
    U = uniform(E)
    fn = make_fn(U, spans, birth)
    sc = (lambda rows, s: H.scramble_cues(rows, s)) if scramble else None
    states, rows = H.prehistory(seed, None, spans=spans, birth=birth, t0=-14.0, t1=0.0, who=people, expo_fn=fn, noise=True, scramble=sc)
    return states


def w(states, i, j):
    return states[i][j]["w"] if j in states[i] else 0.0


def main():
    E = H.merge_expo([json.loads((SCR / f"sample_{s}.json").read_text())["tele"] for s in H.SAMPLE_SEEDS]) if hasattr(H, "SAMPLE_SEEDS") else None
    E = H.merge_expo([json.loads((SCR / f"sample_{s}.json").read_text())["tele"] for s in (101, 102, 103)])
    out = {}
    real = [run(E, s) for s in range(1, N + 1)]
    scr = [run(E, s, scramble=True) for s in range(1, N + 1)]
    # ---- ii-a: adopted against biological, and against the late lodger
    child_parent_B = [np.mean([w(r, "B", p) for p in ("P1", "P2")]) for r in real]
    child_parent_A = [np.mean([w(r, "A", p) for p in ("P1", "P2")]) for r in real]
    parent_child_B = [np.mean([w(r, p, "B") for p in ("P1", "P2")]) for r in real]
    parent_child_A = [np.mean([w(r, p, "A") for p in ("P1", "P2")]) for r in real]
    lodger_to_parent = [np.mean([w(r, "L", p) for p in ("P1", "P2")]) for r in real]
    parent_to_lodger = [np.mean([w(r, p, "L") for p in ("P1", "P2")]) for r in real]
    a = dict(
        child_to_parent_adopted=ci(child_parent_A), child_to_parent_bio=ci(child_parent_B),
        ratio_child_to_parent=float(np.mean(child_parent_A) / np.mean(child_parent_B)),
        parent_to_child_adopted=ci(parent_child_A), parent_to_child_bio=ci(parent_child_B),
        ratio_parent_to_child=float(np.mean(parent_child_A) / np.mean(parent_child_B)),
        adopted_minus_lodger_child_view=ci(np.array(child_parent_A) - np.array(lodger_to_parent)),
        adopted_minus_lodger_parent_view=ci(np.array(parent_child_A) - np.array(parent_to_lodger)),
        lodger_to_parent=ci(lodger_to_parent), parent_to_lodger=ci(parent_to_lodger))
    a["PASS"] = bool(abs(1 - a["ratio_child_to_parent"]) < 0.10 and abs(1 - a["ratio_parent_to_child"]) < 0.10
                     and a["adopted_minus_lodger_child_view"][1] > 0 and a["adopted_minus_lodger_parent_view"][1] > 0)
    out["ii_a"] = a
    # ---- ii-b: the sibling raised apart
    b_a = [w(r, "B", "A") for r in real]
    b_s = [w(r, "B", "S") for r in real]
    kh_s = [r["B"]["S"]["khat"] for r in real]
    kh_a = [r["B"]["A"]["khat"] for r in real]
    b = dict(B_to_adopted_sibling=ci(b_a), B_to_sibling_apart=ci(b_s), diff=ci(np.array(b_a) - np.array(b_s)), khat_apart=ci(kh_s), khat_together=ci(kh_a))
    b["PASS"] = bool(b["diff"][1] > 0 and np.mean(kh_s) < 0.35)
    # dose curve of joining late
    dose = {}
    for jy in (-0.25, -1.0, -2.0, -4.0, -8.0):
        rr = [run(E, s, S_join=jy) for s in range(1, 11)]
        dose[str(-jy)] = dict(years_with_family=-jy, B_to_S_w=ci([w(r, "B", "S") for r in rr]), khat=ci([r["B"]["S"]["khat"] for r in rr]),
                              P_to_S_w=ci([np.mean([w(r, p, "S") for p in ("P1", "P2")]) for r in rr]))
    b["dose_curve_join_years_before"] = dose
    out["ii_b"] = b
    # ---- ii-c: scrambled cues
    def gap(states_list):
        g = []
        for r in states_list:
            kin = [w(r, "B", "P1"), w(r, "B", "P2"), w(r, "B", "A"), w(r, "A", "P1"), w(r, "A", "P2"), w(r, "A", "B")]
            non = [w(r, "B", "L"), w(r, "A", "L")]
            g.append(np.mean(kin) - np.mean(non))
        return g
    g_real, g_scr = gap(real), gap(scr)
    kh_real = [np.mean([r["B"][p]["khat"] for p in ("P1", "P2", "A")]) - r["B"]["L"]["khat"] for r in real]
    kh_scr = [np.mean([r["B"][p]["khat"] for p in ("P1", "P2", "A")]) - r["B"]["L"]["khat"] for r in scr]
    c = dict(gap_real=ci(g_real), gap_scrambled=ci(g_scr), khat_gap_real=ci(kh_real), khat_gap_scrambled=ci(kh_scr))
    c["PASS"] = bool(c["gap_real"][1] > 0 and c["gap_scrambled"][1] <= 0 <= c["gap_scrambled"][2])
    # DIAGNOSTIC (after the pre-registered rule above was evaluated, not part of it): what is left in the scrambled gap is the lodger's shorter time
    # in the house (F), not a cue. The cue component of a weight is w minus the same weight with the kin estimate forced to 0.
    def nocue(s):
        return min(CN2["ceil"], (CN2["w_att"] * s["F"] + CN2["gain"] * max(s["a"], 0.0)) * (1.0 + s["EJ"]))

    def cuegap(states_list, key):
        g = []
        for r in states_list:
            kin = [r[i][j][key] if key != "cue" else r[i][j]["w"] - nocue(r[i][j]) for i, j in (("B", "P1"), ("B", "P2"), ("B", "A"), ("A", "P1"), ("A", "P2"), ("A", "B"))]
            non = [r[i]["L"][key] if key != "cue" else r[i]["L"]["w"] - nocue(r[i]["L"]) for i in ("B", "A")]
            g.append(np.mean(kin) - np.mean(non))
        return g
    c["diagnostic"] = dict(F_gap_real=ci(cuegap(real, "F")), F_gap_scrambled=ci(cuegap(scr, "F")), cue_part_gap_real=ci(cuegap(real, "cue")),
                           cue_part_gap_scrambled=ci(cuegap(scr, "cue")))
    d = c["diagnostic"]
    c["diagnostic"]["cue_part_PASS"] = bool(d["cue_part_gap_real"][1] > 0 and d["cue_part_gap_scrambled"][1] <= 0 <= d["cue_part_gap_scrambled"][2])
    out["ii_c"] = c
    # ---- iii: controls
    pl = max(max(w(r, i, "PL") for i in ("P1", "P2", "B", "A", "S", "L")) for r in real)
    pl2 = max(max(w(r, "PL", i) for i in ("P1", "P2", "B", "A", "S", "L")) for r in real)
    cn = KinConcerns(name="B", seed=1, dial=dials(1, "B"))
    from lab.mindL.household import rows_from_state
    cn.rows = rows_from_state(real[0]["B"])
    cn.fx.set_prior("serve", rho=0.8, eta=0.0, n0=50)
    t = cn.table()
    before = {k: round(cn.expected("serve", k, 0.6), 3) for k in ("P1", "L")}
    cn.set_table({**t, "P1": t["L"], "L": t["P1"]})
    after = {k: round(cn.expected("serve", k, 0.6), 3) for k in ("P1", "L")}
    out["iii"] = dict(placebo_max_w_to_placebo=float(pl), placebo_max_w_by_placebo=float(pl2), swap_before=before, swap_after=after,
                      PASS=bool(pl <= 0.02 and pl2 <= 0.02 and before["P1"] > before["L"] and after["P1"] < after["L"]))
    # ---- the dose-response of years co-resided without the observed-care cue (reported; Lieberman r = .41 in humans)
    ys, ws = [], []
    for k in range(200):
        yrs = float(np.random.default_rng(k).uniform(0, 18))
        people, age, birth, spans = household()
        spans = dict(spans, S=[(-yrs, 0.0)] if yrs > 0.05 else [])
        U = uniform(E)
        fn = make_fn(U, spans, birth)
        fn0 = lambda i, j, t, fn=fn: {**fn(i, j, t), "obs_by": {}, "ej": 0.0}                # no care seen, no costly acts
        s_, _ = H.prehistory(k, None, spans=spans, birth=birth, t0=-18.0, t1=0.0, who=["B", "S"], expo_fn=fn0, noise=True)
        ys.append(yrs)
        ws.append(w(s_, "B", "S"))
    out["dose_years"] = dict(r=float(np.corrcoef(ys, ws)[0, 1]), note="deterministic law plus count noise: no measurement noise, no other causes, so r is far above the human 0.41; the SHAPE is what is reported",
                             w_at=[(y, round(float(np.mean([x for yy, x in zip(ys, ws) if abs(yy - y) < 1.0])), 3)) for y in (1, 4, 8, 12, 16)])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_kin.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        print(k, json.dumps(v, default=float)[:1500])
    return all(out[k].get("PASS", True) for k in ("ii_a", "ii_b", "ii_c", "iii"))


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
