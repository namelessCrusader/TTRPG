"""The family-run inn for the learned-mind lab, its full simulation and its COARSE PRE-HISTORY (mindL5 stage A; docs/family-design.md Fam2-Fam6).

    full run     the family world (flag `family` over the realistic inn: field, lethal, perma), every person a learned mind with the cn2 concerns.
    exposure     a short full run with PASSIVE concerns (b_concern = 0, so the concern does not move behaviour) measures, per ordered pair, what
                 the law needs per night: engaged turns, bouts, mean bout contrast, features, costly acts for the other, care seen by whom.
    pre-history  the years before play in SEASON-SIZED steps (91 nights) with the same law (concerns2.coarse_advance), from a timeline of who
                 lived with whom (ages, arrival and departure). Nothing in the world says who is kin: only the timeline's co-residence and the
                 care seen enter, through the cues of concerns2.
    twin         the coarse run against a full run over 1-2 simulated years, on the facts (weights, closeness order, kin estimates).
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

from lab.mindL import reader
from lab.mindL.concerns2 import CN2, KinConcerns, _Row2, coarse_advance, dials
from lab.mindL.mind import FIXES, make_mind
from lab.world2.lifetime import hook, night_seed
from lab.world2.policy import Rules
from lab.world2.world import (ALL6, FAMILY_AGE, FAMILY_CAST, FAMILY_FLAGS, FIELD_FLAGS, HARD_FLAGS, LETHAL2_FLAGS, LETHAL_FLAGS, SIGNS_FLAGS, World)

NAMES = [n for n, _, _, _ in FAMILY_CAST]
KIND = {n: k for n, k, _, _ in FAMILY_CAST}
# The stage A world is the realistic inn WITHOUT lethal and perma: all-learner minds in the lethal inn die of thirst (3 of 5 within 16 nights in the old
# inn, 6 of 7 within 41 nights in the family inn, naive learners that have not yet learned to drink; stage A report), which would end the household
# before the concerns could form. FLAGS_L is the lethal one (for the survival documentation run and the next stage, after a warm start).
FLAGS = tuple(ALL6) + tuple(SIGNS_FLAGS) + tuple(HARD_FLAGS) + tuple(LETHAL2_FLAGS) + tuple(FIELD_FLAGS) + tuple(FAMILY_FLAGS)
FLAGS_L = FLAGS + tuple(LETHAL_FLAGS)
CFG = {**FIXES, "b_trust": 0.0, "effects": True, "fx_fix": True, "concerns": "cn2"}
SEASON = 365.0 / 4.0
CARE = frozenset({"give", "help up", "bandage", "defend", "serve", "tend"})

# THE TIMELINE (years relative to the start of play; ages at the start are world.FAMILY_AGE). Only the people of the inn are modelled: Mara grew up
# with her mother Gran (a child until 18), left, married Jon (they keep the inn), Pip was born in it, Gran came to live with them after being
# widowed, the lodgers came one after the other. Who is related to whom is NOT written anywhere: only who lived where, and when.
BIRTH = {n: -float(a) for n, a in FAMILY_AGE.items()}
SPANS = {"Mara": [(-36.0, -16.0), (-12.0, 0.0)], "Gran": [(-36.0, -16.0), (-5.0, 0.0)], "Jon": [(-12.0, 0.0)], "Pip": [(-7.0, 0.0)],
         "Tom": [(-3.0, 0.0)], "Sir Ada": [(-1.5, 0.0)], "Oren": [(-0.5, 0.0)]}
# the same for the strangers start: everyone arrives at the start of play
SPANS_NEW = {n: [(0.0, 0.0)] for n in NAMES}


def cls(age):
    return "child" if age < CN2["child_age"] else ("elder" if age >= 60 else "adult")


def present(spans, n, t):
    return any(a <= t < b for a, b in spans.get(n, ()))


# ------------------------------------------------------------------------------------------------ the full run
def make_policy(seed, passive=False, cn=None, **extra):
    reader.use_cast(FAMILY_CAST)
    cfg = {**CFG, **extra}
    if passive:
        cfg["b_concern"] = 0.0
    if cn:
        cfg["cn"] = cn
    return make_mind(seed, **cfg)


def run_full(seed, nights, *, arm="cn2", init=None, ages=None, snaps=(), tele=None, flows=True, extra=None, turns=300, lethal=False, res=False,
             wflags=(), drinks=False, pol_hook=None):
    """A lifetime of the family inn. arm: 'cn2' (concerns steer), 'passive' (concerns learn, do not steer), 'off' (no concerns), 'rules'.
    init: {name: state} loaded into each mind's concerns at the start (the coarse past), ages: {name: age at start}.
    snaps: nights after which the concern states are kept. tele: (a, b): the telemetry between night a and night b.
    Returns dict(snaps, tele, flows, deaths, seconds)."""
    reader.use_cast(FAMILY_CAST)
    if arm == "rules":
        pol = Rules()
    else:
        pol = make_policy(seed, passive=(arm == "passive"), **({"concerns": False} if arm == "off" else {}), **(extra or {}))
        if init is not None:
            make = pol.make

            def mk(n, k, make=make):
                a = make(n, k)
                if a.cn is not None and n in init:
                    a.cn.load_state(init[n])
                    a.cn.age = float((ages or {}).get(n, FAMILY_AGE[n]))
                    a.cn.nrs = 1
                return a
            pol.make = mk
    if (extra or {}).get("drives2") and "drives2" not in wflags:      # a drives2 mind needs the world to show thirst in the signs
        wflags = tuple(wflags) + ("drives2",)
    if pol_hook is not None:
        pol_hook(pol)
    carry, out = None, dict(snaps={}, tele=None, flows={}, deaths={}, care_by_night=[], res=[], drinks=[])
    t0 = time.perf_counter()
    for n in range(nights):
        w = World(seed=night_seed(seed, n), flags=(FLAGS_L if lethal else FLAGS) + tuple(wflags), carry=carry, night=n, turns=turns)
        hook(pol, "night_start", w, n)
        dead = set(w.gone)
        if res:   # stage B (read only): the lowest reserve fraction of the night, reserve burned, nights at the need cap
            from lab.world2.world import LETHAL_RESERVE
            rc = {m: dict(min=1.0, burn=0.0, caph=0, prev=dict(w.people[m].reserve)) for m in NAMES}
        for _ in range(turns):
            w.step(pol)
            if res:
                for m in NAMES:
                    q, c = w.people[m], rc[m]
                    for k, v in q.reserve.items():
                        if v < c["prev"][k]:
                            c["burn"] += c["prev"][k] - v
                        c["prev"][k] = v
                        c["min"] = min(c["min"], max(0.0, v) / LETHAL_RESERVE[k])
                    c["caph"] += int(q.hunger >= 0.999)
            for p in w.people.values():
                if p.dead and p.name not in dead:
                    dead.add(p.name)
                    out["deaths"][p.name] = dict(night=n, cause=p.cause)
                    blow = next((e for e in reversed(w.log) if e.get("result") == f"killed {p.name}"), None)
                    if p.name in NAMES:
                        p.memory.append({"night": n, "t": w.t, "what": "died", "by": blow["actor"] if blow else None, "event": blow and blow["id"]})
                    hook(pol, "on_death", w, p, blow)
            if all(p.dead or p.down for p in w.people.values()):
                break
        hook(pol, "night_end", w, n)
        if res:
            out["res"].append({m: dict(min=0.0 if w.people[m].dead else round(rc[m]["min"], 4), burn=round(rc[m]["burn"], 4), caph=rc[m]["caph"],
                                       dead=int(w.people[m].dead)) for m in NAMES})
        if drinks:
            dr = {m: 0 for m in NAMES}
            for e in w.log:
                if e["done"] and e["action"] == "drink" and e["actor"] in dr:
                    dr[e["actor"]] += 1
            out["drinks"].append(dr)
        if flows:
            for e in w.log:
                if e["done"] and e["action"] in CARE and e.get("target") in NAMES and e["actor"] in NAMES and e["actor"] != e["target"]:
                    k = f'{e["actor"]}>{e["target"]}'
                    out["flows"][k] = out["flows"].get(k, 0) + 1
        carry = w.carry_out(NAMES)
        agents = getattr(pol, "agents", {})
        if tele and n + 1 == tele[0]:
            for a in agents.values():
                if a.cn is not None:
                    a.cn.reset_telemetry()
        if tele and n + 1 == tele[1]:
            out["tele"] = {nm: a.cn.telemetry() for nm, a in agents.items() if a.cn is not None}
        if (n + 1) in snaps:
            out["snaps"][n + 1] = {nm: a.cn.state() for nm, a in agents.items() if a.cn is not None and nm in NAMES}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["nights"] = nights
    return out


# ------------------------------------------------------------------------------------------------ exposure
EXPO_KEYS = ("con", "eng", "bouts", "dbar", "cbar", "ej", "obs")


def classmeans(E):
    """Mean exposure by (class of i, class of j) over the measured pairs: what a pair of that kind did, for pasts in which the household differed."""
    acc: dict = {}
    for i, row in E.items():
        for j, e in row.items():
            k = (cls(FAMILY_AGE[i]), cls(FAMILY_AGE[j]))
            acc.setdefault(k, []).append(e)
    out = {}
    for k, es in acc.items():
        m = {x: sum(e[x] for e in es) / len(es) for x in EXPO_KEYS}
        feat: dict = {}
        for e in es:
            for f, v in e["feat"].items():
                feat[f] = feat.get(f, 0.0) + v / len(es)
        m["feat"] = feat
        out[k] = m
    return out


def merge_expo(Es):
    """The mean of several samples' telemetry ({name: {target: dict}})."""
    out: dict = {}
    for E in Es:
        for i, row in E.items():
            for j, e in row.items():
                if j not in NAMES or i not in NAMES:
                    continue
                o = out.setdefault(i, {}).setdefault(j, dict(n=0, feat={}, obs_by={}, **{x: 0.0 for x in EXPO_KEYS}))
                o["n"] += 1
                for x in EXPO_KEYS:
                    o[x] += e[x]
                for f, v in e["feat"].items():
                    o["feat"][f] = o["feat"].get(f, 0.0) + v
                for c, v in e["obs_by"].items():
                    o["obs_by"][c] = o["obs_by"].get(c, 0.0) + v
    for i, row in out.items():
        for j, o in row.items():
            n = o.pop("n")
            for x in EXPO_KEYS:
                o[x] /= n
            o["feat"] = {f: v / n for f, v in o["feat"].items()}
            o["obs_by"] = {c: v / n for c, v in o["obs_by"].items()}
            o["res"] = 1.0
    return out


def expo_at(i, j, t, E, CM, spans, birth=BIRTH):
    """The exposure of i to j at time t (years). The pair's own measured rates if both are now the kind of person they were when measured
    (child, adult, elder); else the mean of that kind of pair. Care seen is re-spread over the carers who are here at t."""
    ci, cj = cls(t - birth[i]), cls(t - birth[j])
    if (ci, cj) == (cls(FAMILY_AGE[i]), cls(FAMILY_AGE[j])) and j in E.get(i, {}):
        e = copy.deepcopy(E[i][j])
        e["obs_by"] = {c: v for c, v in e["obs_by"].items() if present(spans, c, t) and c not in (i, j)}
        if not e["obs_by"] and E[i][j]["obs_by"]:
            tot = sum(E[i][j]["obs_by"].values())
            here = [c for c in NAMES if present(spans, c, t) and c not in (i, j)]
            e["obs_by"] = {c: tot / len(here) for c in here} if here else {}
    else:
        m = CM.get((ci, cj)) or CM.get(("adult", "adult"))
        e = copy.deepcopy(m)
        carers = [c for c in NAMES if present(spans, c, t) and c not in (i, j) and cls(t - birth[c]) == "adult"]
        e["obs_by"] = {c: m["obs"] / len(carers) for c in carers} if carers else {}
    e["res"] = 1.0
    return e


# ------------------------------------------------------------------------------------------------ the coarse pre-history
def rows_from_state(st):
    rows = {}
    for T, s in st.items():
        r = rows[T] = _Row2()
        r.a, r.a_max, r.hrs, r.hn, r.ky, r.kc, r.ejn = (s[k] for k in ("a", "a_max", "hrs", "hn", "ky", "kc", "ejn"))
        r.kn = s.get("kn", 0.0)
        r.nights, r.first_age, r.hab = s["nights"], s["first_age"], dict(s["hab"])
    return rows


def state_from_rows(rows, dial, params=None):
    cn = KinConcerns(name="z", seed=0, dial=dial, params=params)
    cn.rows = rows
    return cn.state()


def prehistory(seed, E, spans=SPANS, birth=BIRTH, t0=None, t1=0.0, step=SEASON, who=None, expo_fn=None, noise=False, rows0=None, scramble=None, params=None):
    """The years before play, in steps of `step` nights, the same law at each step (concerns2.coarse_advance). Returns ({name: state}, {name: rows})."""
    who = who or NAMES
    CM = classmeans(E) if E else {}
    fn = expo_fn or (lambda i, j, t: expo_at(i, j, t, E, CM, spans, birth))
    dl = {n: dials(seed, n) for n in who}
    rng = {n: (random.Random(f"coarse/{seed}/{n}") if noise else None) for n in who}
    rows = {n: (rows_from_state(rows0[n]) if rows0 and n in rows0 else {}) for n in who}
    t = t0 if t0 is not None else min(a for n in who for a, _ in spans.get(n, [(0.0, 0.0)]))
    while t < t1 - 1e-9:
        nights = min(step, (t1 - t) * 365.0)
        mid = t + nights / 730.0
        here = [n for n in who if present(spans, n, mid)]
        for i in here:
            expo = {j: fn(i, j, mid) for j in here if j != i}
            coarse_advance(rows[i], expo, nights, mid - birth[i] - nights / 730.0, dl[i], c={**CN2, **(params or {})}, noise=rng[i])
        t += nights / 365.0
    if scramble is not None:
        scramble(rows, seed)
    return {n: state_from_rows(rows[n], dl[n], params) for n in who}, rows


def scramble_cues(rows, seed, who=None):
    """SCRAMBLED CUES: for each person, the cue records (years lived together as a child, the observed-care cue, the accumulated history) are dealt
    among the people they were recorded for, at random: the cues no longer say who is who."""
    for i, rw in rows.items():
        keys = [j for j in rw if j in NAMES or True]
        r = random.Random(f"scramble/{seed}/{i}")
        vals = [(rw[j].ky, rw[j].kc, rw[j].hn) for j in keys]
        r.shuffle(vals)
        for j, (ky, kc, hn) in zip(keys, vals):
            rw[j].ky, rw[j].kc, rw[j].hn = ky, kc, hn


# ------------------------------------------------------------------------------------------------ the twin check
def pairs(states):
    return {(i, j): s for i, st in states.items() for j, s in st.items() if j in NAMES and j != i}


def tau_b(x, y):
    n = len(x)
    c = d = tx = ty = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx, dy = x[a] - x[b], y[a] - y[b]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + tx) * (c + d + ty))
    return (c - d) / den if den > 0 else float("nan")


def compare(full, coarse, eps=0.01):
    """Per pair and per person, the coarse state against the full one: weights, kin estimates, closeness order. Returns summary and the table."""
    pf, pc = pairs(full), pairs(coarse)
    keys = sorted(set(pf) & set(pc))
    dw = [abs(pf[k]["w"] - pc[k]["w"]) for k in keys]
    dk = [abs(pf[k]["khat"] - pc[k]["khat"]) for k in keys]
    dF = [abs(pf[k]["F"] - pc[k]["F"]) for k in keys]
    dH = [abs(pf[k]["H"] - pc[k]["H"]) for k in keys]
    taus = []
    for i in NAMES:
        js = [j for j in NAMES if j != i and (i, j) in pf and (i, j) in pc]
        if len(js) >= 3:
            t = tau_b([pf[(i, j)]["w"] for j in js], [pc[(i, j)]["w"] for j in js])
            if not math.isnan(t):
                taus.append(t)
    return dict(n_pairs=len(keys), mean_dw=sum(dw) / len(dw), max_dw=max(dw), mean_dk=sum(dk) / len(dk), max_dk=max(dk), mean_dF=sum(dF) / len(dF),
                mean_dH=sum(dH) / len(dH), tau=sum(taus) / len(taus) if taus else float("nan"), n_tau=len(taus),
                mean_w_full=sum(pf[k]["w"] for k in keys) / len(keys), mean_w_coarse=sum(pc[k]["w"] for k in keys) / len(keys))


def dump(o, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(o))
