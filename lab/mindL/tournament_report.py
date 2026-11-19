"""Step 0a analysis: the lifetimes in lab/results/mindL3/0a_runs/ (made by lab/mindL/tournament.py) -> the numbers.

    lab/.venv/bin/python -m lab.mindL.tournament_report [--out DIR]

Writes 0a_tournament.json, 0a_oracle.json, 0a_bcqw.json, 0a_summary.json in lab/results/mindL3/ and prints the
tables as text. Every interval is a 95% t interval over the seeds (the seed is the unit: the nights of one life are
not independent). "Paired" = the same seed, the same focal cast member, the same world draws.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as st

from lab.mindL.tournament import A_CHOOSERS, HARSH_ALL, HELP_ACTS, OUT, REPLY_KIND, RUNS, SEEDS, WORLDS, Recorder
from lab.world2.world import DRIVES, HARMFUL, HARSH_ACTS, KIND_ACTS

FIELDS = Recorder.FIELDS
HELP = ("give", "bandage", "help up", "defend", "serve")          # kind acts that cost the giver something besides a word
KIND_ALL = tuple(sorted(KIND_ACTS))
HARSH_VERBS = tuple(sorted(HARMFUL | HARSH_ACTS))
PERSON_VERBS = {"serve", "buy", "work", "strike", "shove", "threaten", "insult", "praise", "search", "steal", "bandage", "help up", "defend",
                "accuse", "tell", "give", "gossip", "ask", "offer", "accept", "refuse"}


# ------------------------------------------------------------------ small statistics
def ci(xs):
    x = np.array([v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(x.mean())
    if n < 2:
        return {"mean": round(m, 4), "lo": None, "hi": None, "n": n}
    h = float(st.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return {"mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4), "n": n}


def above(c):
    return c["lo"] is not None and c["lo"] > 0


def below(c):
    return c["hi"] is not None and c["hi"] < 0


def fmt(c, d=3):
    if c is None or c["mean"] is None:
        return "-"
    if c["lo"] is None:
        return f"{c['mean']:+.{d}f}"
    return f"{c['mean']:+.{d}f} [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}]"


def fmt0(c, d=3):
    if c is None or c["mean"] is None:
        return "-"
    return f"{c['mean']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]" if c["lo"] is not None else f"{c['mean']:.{d}f}"


# ------------------------------------------------------------------ the runs
def load(runs=RUNS):
    out = {}
    for fp in sorted(Path(runs).glob("*.json")):
        r = json.loads(fp.read_text())
        out[(r["kind"], r["world"], r["chooser"], r["seed"], r.get("rate"))] = r
    return out


def outcome(r):
    w0, n = r["win0"], len(r["err"])
    e = r["err"][w0:]
    dead = sum(r["dead"][w0:])
    return {"err": float(np.mean(e)), "deaths": dead, "survived": (n - w0) - dead, "down": sum(r["down"][w0:]),
            "scored": n - w0, "hp": float(np.mean(r["hp"][w0:]))}


def table(R, kind, world, measure="err", rate=None):
    """{chooser: {seed: value}}"""
    t = {}
    for (k, w, c, s, rt), r in R.items():
        if k == kind and w == world and rt == rate:
            t.setdefault(c, {})[s] = outcome(r)[measure]
    return t


def paired(t, a, b):
    xs = [t[a][s] - t[b][s] for s in SEEDS if s in t.get(a, {}) and s in t.get(b, {})]
    return ci(xs)


def focal_of(R, kind, world, seed):
    for (k, w, c, s, rt), r in R.items():
        if k == kind and w == world and s == seed:
            return r["focal"]


# ------------------------------------------------------------------ tournament A
def tournament(R):
    res = {"measure": "focal drive error, mean over nights from win0 to the end; lower is better", "worlds": {}}
    for world in WORLDS:
        errs, deaths, surv = table(R, "A", world), table(R, "A", world, "deaths"), table(R, "A", world, "survived")
        downs = table(R, "A", world, "down")
        rows = {}
        for c in errs:
            rows[c] = {"err": ci(list(errs[c].values())), "deaths": ci(list(deaths[c].values())),
                       "survived": ci(list(surv[c].values())), "down": ci(list(downs[c].values())),
                       "seeds": sorted(errs[c])}
            if c != "rules" and "rules" in errs:
                rows[c]["minus_rules"] = paired(errs, c, "rules")
                rows[c]["deaths_minus_rules"] = paired(deaths, c, "rules")
                rows[c]["survived_minus_rules"] = paired(surv, c, "rules")
        extra = {}
        pairs = (("always-giving", "hermit"), ("tit-for-tat", "hermit"), ("always-harsh", "hermit"), ("extractor", "hermit"),
                 ("hoarder", "hermit"), ("protector", "hermit"), ("hermit_ask", "hermit"), ("hermit_ask", "rules"),
                 ("feeder", "rules"), ("feeder", "hermit"), ("feeder", "hermit_ask"), ("always-giving", "always-harsh"),
                 ("tit-for-tat", "always-harsh"), ("random", "rules"))
        extra["contrasts"] = {f"{a}_minus_{b}": paired(errs, a, b) for a, b in pairs if a in errs and b in errs}
        if "always-giving" in errs and "tit-for-tat" in errs:
            extra["giving_minus_tft"] = paired(errs, "always-giving", "tit-for-tat")
            extra["giving_minus_tft_deaths"] = paired(deaths, "always-giving", "tit-for-tat")
        # where the error comes from, by drive (mean over seeds, nights from win0)
        split = {}
        for (k, w, c, s, rt), r in R.items():
            if k == "A" and w == world and rt is None:
                split.setdefault(c, []).append([r["split"][d] for d in DRIVES])
        res["worlds"][world] = {"rows": rows, **extra,
                                "split": {c: dict(zip(DRIVES, np.round(np.mean(v, axis=0), 3).tolist()))
                                          for c, v in split.items()},
                                "per_seed_err": errs}
    return res


def flows(R):
    """Per night, what the focal does to others and what others do to it (acts finished, nights from win0)."""
    out = {}
    for (k, w, c, s, rt), r in R.items():
        if k != "A" or rt is not None:
            continue
        n = len(r["err"]) - r["win0"]
        row = [sum(v for a, v in r["made"].items() if a in REPLY_KIND) / n,
               sum(v for a, v in r["made"].items() if a in HARSH_ALL) / n,
               sum(v for a, v in r["recv"].items() if a in REPLY_KIND) / n,
               sum(v for a, v in r["recv"].items() if a in HELP_ACTS) / n,
               sum(v for a, v in r["recv"].items() if a in HARSH_ALL) / n,
               sum(v for a, v in r["recv"].items() if a == "gossip") / n,
               r["recv"].get("strike", 0) / n, r["made"].get("strike", 0) / n]
        out.setdefault(w, {}).setdefault(c, []).append(row)
    names = ("made_kind", "made_harsh", "recv_kind", "recv_help", "recv_harsh", "recv_gossip", "recv_strike", "made_strike")
    return {w: {c: dict(zip(names, np.round(np.mean(v, axis=0), 2).tolist())) for c, v in d.items()}
            for w, d in out.items()}


def by_type(T, R):
    """Exploratory: the same paired differences, split by what the focal's nature wants. Tom, Sir Ada and Marta have no
    wealth drive (seeds 1, 4, 5, 6, 9, 10); Oren and Fenn do (seeds 2, 3, 7, 8). Few seeds: wide intervals."""
    groups = {"no_wealth_drive": [s for s in SEEDS if (s % 5) in (1, 4, 0)], "wealth_drive": [s for s in SEEDS if (s % 5) in (2, 3)]}
    out = {}
    for world in WORLDS:
        t = T["worlds"][world]["per_seed_err"]
        for g, seeds in groups.items():
            for base in ("rules", "hermit"):
                for c in t:
                    if c == base or base not in t:
                        continue
                    xs = [t[c][s] - t[base][s] for s in seeds if s in t[c] and s in t[base]]
                    out.setdefault(world, {}).setdefault(g, {}).setdefault(f"minus_{base}", {})[c] = ci(xs)
    return out


def by_cast(T, R):
    """Exploratory: mean over the two seeds of each cast member of (chooser error minus rules error)."""
    out = {}
    for world in WORLDS:
        t = T["worlds"][world]["per_seed_err"]
        if "rules" not in t:
            continue
        seeds_of = {}
        for s in SEEDS:
            f = focal_of(R, "A", world, s)
            if f:
                seeds_of.setdefault(f, []).append(s)
        out[world] = {"rules_error": {f: round(float(np.mean([t["rules"][s] for s in ss])), 3) for f, ss in seeds_of.items()},
                      "minus_rules": {c: {f: round(float(np.mean([t[c][s] - t["rules"][s] for s in ss if s in t[c]])), 3)
                                          for f, ss in seeds_of.items() if any(s in t[c] for s in ss)}
                                      for c in t if c != "rules"}}
    return out


def pass_rules_a(T, R):
    out = {}
    w = T["worlds"]["all2"]["rows"]
    for c in ("hermit", "always-harsh"):
        d = w.get(c, {}).get("minus_rules")
        out[f"{c}_loses_to_rules"] = {"value": d, "pass": bool(d and above(d)),
                                      "other_worlds": {k: T["worlds"][k]["rows"].get(c, {}).get("minus_rules")
                                                       for k in WORLDS if k != "all2"}}
    out["hermit_and_always_harsh_lose_to_rules"] = {
        "value": None, "pass": bool(out["hermit_loses_to_rules"]["pass"] and out["always-harsh_loses_to_rules"]["pass"])}
    g = T["worlds"]["all2"].get("giving_minus_tft")
    out["always_giving_loses_to_tit_for_tat"] = {"value": g, "pass": bool(g and above(g)),
                                                 "other_worlds": {k: T["worlds"][k].get("giving_minus_tft")
                                                                  for k in WORLDS if k != "all2"}}
    ex = {k: T["worlds"][k]["rows"].get("extractor", {}).get("minus_rules") for k in WORLDS}
    wins = {k: bool(v and below(v)) for k, v in ex.items()}
    weak, strong = ("noR", "none"), ("all2", "noG")
    ok = any(wins.get(k) for k in weak) and not any(wins.get(k) for k in strong)
    out["extractor_wins_only_where_reprisal_is_weak"] = {
        "value": ex, "wins": wins, "pass": ok,
        "reading": ("wins somewhere with reprisal weak and nowhere with it strong" if ok else
                    "wins nowhere" if not any(wins.values()) else
                    "wins where reprisal is strong" if any(wins.get(k) for k in strong) else
                    "wins in no weak-reprisal world")}
    # by focal (exploratory: two seeds per cast member)
    byf = {}
    for world in WORLDS:
        t = T["worlds"][world]["per_seed_err"]
        if "extractor" in t and "rules" in t:
            for s in SEEDS:
                if s in t["extractor"] and s in t["rules"]:
                    f = focal_of(R, "A", world, s)
                    byf.setdefault(world, {}).setdefault(f, []).append(round(t["extractor"][s] - t["rules"][s], 3))
    out["extractor_by_focal"] = byf
    return out


# ------------------------------------------------------------------ oracle gap
def oracle(R):
    errs = {c: v for c, v in table(R, "O", "all2").items() if c == "rules" or c.startswith("oracle")}
    res = {"setting": "L2 life A, all2: Fenn P-kind(0.9), Oren P-hard(0.9), focal Tom / Sir Ada / Marta by seed % 3",
           "rows": {c: ci(list(errs[c].values())) for c in errs}, "per_seed": errs}
    if "rules" in errs and "oracle_both" in errs:
        for c in errs:
            if c != "rules":
                res.setdefault("gap", {})[c] = paired({"b": errs["rules"], "o": errs[c]}, "b", "o")
        g = res["gap"]["oracle_both"]
        res["pass"] = bool(g["mean"] is not None and g["mean"] >= 0.1 and above(g))
        # the part of the blind chooser's error that the partners could touch at all
        sp = [r["split"] for (k, w, c, s, rt), r in R.items() if k == "O" and c == "rules"]
        if sp:
            res["rules_error_by_drive"] = {d: round(float(np.mean([x[d] for x in sp])), 3) for d in DRIVES}
            ceil = {}
            for s in SEEDS:
                r = R.get(("O", "all2", "rules", s, None))
                if r:
                    ceil[s] = r["split"]["company"] + r["split"]["standing"] + r["split"]["safety"]
            res["ceiling"] = {"what": "blind chooser's company + standing + safety error: the most any policy could "
                                      "win back from these three drives", "ci": ci(list(ceil.values()))}
            res["share_recovered"] = round(g["mean"] / res["ceiling"]["ci"]["mean"], 3) \
                if res["ceiling"]["ci"]["mean"] else None
        # what the oracle does, per night, from the window
        doing = {}
        for (k, w, c, s, rt), r in R.items():
            if k == "O":
                n = r["win0"] and (len(r["err"]) - r["win0"])
                for key, v in r["made_to"].items():
                    verb, who = key.split(">")
                    if who in ("Fenn", "Oren"):
                        doing.setdefault(c, {}).setdefault(f"{verb}>{who}", []).append(v / n)
        res["acts_toward_partners_per_night"] = {c: {k: round(float(np.sum(v)) / 10, 2) for k, v in d.items()}
                                                 for c, d in doing.items()}
        rec = {}
        for (k, w, c, s, rt), r in R.items():
            if k == "O":
                for key, v in r["recv_from"].items():
                    verb, who = key.split("<")
                    rec.setdefault(c, {}).setdefault(f"{verb}<{who}", []).append(v)
        res["acts_received_from_partners_per_lifetime"] = {c: {k: round(float(np.sum(v)) / 10, 1) for k, v in d.items()
                                                               if k.split("<")[1] in ("Fenn", "Oren")}
                                                           for c, d in rec.items()}
    return res


def provocation_price(R):
    """The price of an occasional provocation in the L2 setting: the shared self-care plus about one provocation a
    night aimed at Oren (P-hard, hits back) or at Fenn (P-kind, never does). The two share one baseline, so the
    headline is hard minus kind (paired). Their difference from rules is NOT a price: rules has social acts of its
    own (accuse, tell, gossip, strike), which the provokers lack."""
    errs = table(R, "O", "all2")
    out = {}
    for c, victim in (("provoke_hard", "Oren"), ("provoke_kind", "Fenn")):
        if c not in errs or "rules" not in errs:
            continue
        d = paired(errs, c, "rules")
        n_prov, n_back = [], []
        for s in SEEDS:
            r = R.get(("O", "all2", c, s, None))
            if r:
                n = len(r["err"]) - r["win0"]
                n_prov.append(sum(v for k, v in r["made_to"].items() if k.endswith(">" + victim)
                                  and k.split(">")[0] in ("insult", "strike", "shove", "threaten")) / n)
                n_back.append(sum(v for k, v in r["recv_from"].items() if k.endswith("<" + victim)
                                  and k.split("<")[0] in ("insult", "strike", "shove", "threaten")) / n)
        out[c] = {"victim": victim, "error": ci(list(errs[c].values())), "minus_rules_not_a_price": d,
                  "provocations_per_night": round(float(np.mean(n_prov)), 2),
                  "harsh_acts_back_per_night": round(float(np.mean(n_back)), 2)}
    if "provoke_hard" in out and "provoke_kind" in out:
        hk = paired(errs, "provoke_hard", "provoke_kind")
        per = (out["provoke_hard"]["provocations_per_night"] + out["provoke_kind"]["provocations_per_night"]) / 2
        out["hard_minus_kind"] = {"value": hk, "provocations_per_night": per,
                                  "per_provocation": round(hk["mean"] / per, 3) if per else None}
    return out


def sensitivity(R):
    """The spontaneous rate of the stance changed (--rate): paired with the main rules lifetimes (rate None)."""
    base = table(R, "A", "all2")
    out = {}
    for rate in sorted({k[4] for k in R if k[4] is not None}):
        t = table(R, "A", "all2", rate=rate)
        for c in t:
            out.setdefault(c, {})[str(rate)] = {"err": ci(list(t[c].values())),
                                                "minus_rules": paired({**t, "rules": base["rules"]}, c, "rules")}
        for c in t:                                   # and the main rate for the same chooser
            if c in base:
                out[c]["main"] = {"err": ci(list(base[c].values())), "minus_rules": paired(base, c, "rules")}
    return out


# ------------------------------------------------------------------ b, c, q, w
def group_totals(runs, verbs, roles=("focal", "toF", "oo")):
    tot = dict.fromkeys(FIELDS, 0.0)
    for r in runs:
        for key, a in r["agg"].items():
            v, role = key.split("|")
            if v in verbs and role in roles:
                for k in FIELDS:
                    tot[k] += a.get(k, 0.0)
    return tot


def rates(t):
    n = t["n_done"]
    if n < 5:
        return None
    nk, nw = t["next_known"], t["n_w"]
    return {"n": int(n), "b": (t["b"] + t["bpot"]) / n, "b_imm": t["b"] / n, "b_rem": t["b_rem"] / n,
            "c": (t["c"] + t["cpot"] + t["ctime"]) / n, "c_imm": t["c"] / n, "c_rem": t["c_rem"] / n,
            "c_pot": t["cpot"] / n, "c_time": t["ctime"] / n,
            "q_seen": t["seen"] / n, "q_told": t["told"] / n, "q": t["third"] / n, "q_eff": t["eff"] / n,
            "present": t["present"] / n, "w_same": t["again"] / n, "w_next": t["again_next"] / nk if nk else None,
            "daff": t["daff"] / n, "reply_kind": t["rk_d"] / n, "reply_help": t["rp_d"] / n, "reply_harsh": t["rh_d"] / n,
            "witness_reply_kind": t["rk_w"] / nw if nw else None, "witness_reply_help": t["rp_w"] / nw if nw else None,
            "witness_reply_harsh": t["rh_w"] / nw if nw else None}


def bcqw(R):
    """b, c, q, w from the lifetimes of the eight choosers of the plan, pooled (every actor, every seed)."""
    res = {"definitions": {
        "b": "receiver's drive error before the act minus after it, plus the drop it could take by eating/drinking a "
             "food or drink it was given", "c": "giver's error after minus before, plus the use it loses by giving "
             "food/drink away, plus the error its needs grow by during the act's turns (upper bound)",
        "q": "a third party in the room saw the act, or was told of it that night",
        "q_eff": "a third party's liking of the giver moved in the deed's direction (on seeing it or being told)",
        "w_same": "the pair is in one room again later that night; w_next: at some time the next night"},
        "worlds": {}}
    groups = {"help": HELP, "kind_all": KIND_ALL, "harsh": HARSH_VERBS}
    for world in WORLDS:
        # the eight choosers of the plan (the same in every world); the extras (random, feeder, hermit_ask) are left out
        runs_by_seed = {s: [r for (k, w, c, sd, rt), r in R.items()
                            if k == "A" and w == world and sd == s and rt is None and c in A_CHOOSERS]
                        for s in SEEDS}
        wres = {"by_verb": {}, "groups": {}}
        for v in sorted(PERSON_VERBS):
            allr = [r for rs in runs_by_seed.values() for r in rs]
            ra = rates(group_totals(allr, (v,)))
            if ra:
                rb = rates(group_totals(allr, (v,), ("focal", "oo")))        # replies by rules characters only
                if rb:
                    for k in ("reply_kind", "reply_help", "reply_harsh", "witness_reply_kind", "witness_reply_help",
                              "witness_reply_harsh"):
                        ra[k] = rb[k]
                wres["by_verb"][v] = ra
        for gname, verbs in groups.items():
            per = {s: rates(group_totals(rs, verbs)) for s, rs in runs_by_seed.items()}
            per = {s: x for s, x in per.items() if x}
            if not per:
                continue
            g = {"pooled": rates(group_totals([r for rs in runs_by_seed.values() for r in rs], verbs))}
            for k in ("b", "c", "q", "q_eff", "q_seen", "q_told", "w_same", "w_next", "daff", "c_imm", "b_imm"):
                g[k] = ci([x[k] for x in per.values()])
            # replies by a rules character (the receiver or a witness): acts of the focal and acts among the others
            rep = {s: rates(group_totals(rs, verbs, ("focal", "oo"))) for s, rs in runs_by_seed.items()}
            rep = {s: x for s, x in rep.items() if x}
            for k in ("reply_kind", "reply_help", "reply_harsh", "witness_reply_kind", "witness_reply_help",
                      "witness_reply_harsh"):
                g[k] = ci([x[k] for x in rep.values()])
            g["qb_minus_c"] = ci([x["q"] * x["b"] - x["c"] for x in per.values()])
            g["qeffb_minus_c"] = ci([x["q_eff"] * x["b"] - x["c"] for x in per.values()])
            g["qb_over_c"] = round(g["q"]["mean"] * g["b"]["mean"] / g["c"]["mean"], 3) if g["c"]["mean"] else None
            g["seeds"] = len(per)
            wres["groups"][gname] = g
        res["worlds"][world] = wres
    # gossip: how far a told deed moves a hearer's liking
    ge = {w: {"good": [0, 0.0, 0], "bad": [0, 0.0, 0]} for w in WORLDS}
    for (k, w, c, s, rt), r in R.items():
        if k == "A" and rt is None and c in A_CHOOSERS:
            for kk in ("good", "bad"):
                for i in range(3):
                    ge[w][kk][i] += r["gossip_effect"][kk][i]
    res["gossip_effect"] = {w: {kk: {"tellings": v[0], "mean_liking_change": round(v[1] / v[0], 4) if v[0] else None,
                                     "share_that_moved_it": round(v[2] / v[0], 3) if v[0] else None}
                                for kk, v in d.items()} for w, d in ge.items()}
    return res


# ------------------------------------------------------------------ the archetypes' own acts
def stance_check(R):
    from lab.mindL.tournament import ARCHETYPES
    out = {}
    for c, cls in ARCHETYPES.items():
        made, tot, social, n = {}, 0, 0, 0
        for (k, w, ch, s, rt), r in R.items():
            if k == "A" and ch == c and rt is None and w == "all2":
                n += 1
                for v, x in r["made"].items():
                    made[v] = made.get(v, 0) + x
        tot = sum(made.values())
        if not tot:
            continue
        person = {v: x for v, x in made.items() if v in PERSON_VERBS}
        out[c] = {"lifetimes": n, "social_share": round(sum(person.values()) / tot, 3), "person_acts": person,
                  "outside_stance": {v: x for v, x in person.items() if v not in cls.allowed and v not in ("ask", "offer")}}
    for c in ("rules", "random"):
        made, n = {}, 0
        for (k, w, ch, s, rt), r in R.items():
            if k == "A" and ch == c and rt is None and w == "all2":
                n += 1
                for v, x in r["made"].items():
                    made[v] = made.get(v, 0) + x
        tot = sum(made.values())
        if tot:
            person = {v: x for v, x in made.items() if v in PERSON_VERBS}
            out[c] = {"lifetimes": n, "social_share": round(sum(person.values()) / tot, 3), "person_acts": person}
    return out


def cpu_hours(R):
    return round(sum(r["cpu_s"] for r in R.values()) / 3600, 3)


# ------------------------------------------------------------------ text
def text(T, P, O, B, S):
    L = []
    for world in WORLDS:
        w = T["worlds"][world]
        L.append(f"\nWORLD {world}: focal drive error, nights 20-99 (lower is better). minus rules = paired by seed.")
        L.append(f"{'chooser':14s} {'error [95% CI]':26s} {'minus rules':28s} {'deaths/80n':>10s} {'survived':>8s} {'seeds':>5s}")
        for c, r in sorted(w["rows"].items(), key=lambda kv: kv[1]["err"]["mean"]):
            L.append(f"{c:14s} {fmt0(r['err']):26s} {fmt(r.get('minus_rules')):28s} {r['deaths']['mean']:10.2f} "
                     f"{r['survived']['mean']:8.1f} {len(r['seeds']):5d}")
        if "giving_minus_tft" in w:
            L.append(f"always-giving minus tit-for-tat: {fmt(w['giving_minus_tft'])}")
        for k, v in w.get("contrasts", {}).items():
            L.append(f"  {k.replace('_minus_', ' minus '):34s} {fmt(v)}")
    L.append("\nORACLE (L2 setting, all2):")
    for c, v in O["rows"].items():
        L.append(f"  {c:14s} {fmt0(v)}   gap vs rules {fmt(O.get('gap', {}).get(c))}")
    if "ceiling" in O:
        L.append(f"  ceiling (blind company+standing+safety error) {fmt0(O['ceiling']['ci'])}; share recovered {O.get('share_recovered')}")
    L.append("\nPASS RULES:")
    L.append(f"  oracle gap >= 0.1 with CI above 0: {'PASS' if O.get('pass') else 'FAIL'}  ({fmt(O.get('gap', {}).get('oracle_both'))})")
    for k, v in P.items():
        if k == "extractor_by_focal":
            continue
        L.append(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  {fmt(v['value']) if isinstance(v['value'], dict) and 'mean' in v['value'] else v['value']}")
    for gname in ("help", "kind_all", "harsh"):
        g = B["worlds"]["all2"]["groups"].get(gname)
        if g:
            L.append(f"  b,c,q,w ({gname}, all2): n={g['pooled']['n']} b={g['b']['mean']:.3f} c={g['c']['mean']:.3f} "
                     f"q={g['q']['mean']:.2f} q_eff={g['q_eff']['mean']:.2f} w_same={g['w_same']['mean']:.2f} "
                     f"q*b-c {fmt(g['qb_minus_c'])}  q_eff*b-c {fmt(g['qeffb_minus_c'])}")
    return "\n".join(L)


def grid_text(T):
    """One table: every chooser in the four worlds (error, and the paired difference from rules)."""
    L = ["\nTHE FOUR WORLDS: focal drive error (mean over seeds) and, in brackets, minus rules [95% CI], paired by seed",
         f"{'chooser':14s} " + " ".join(f"{w:>34s}" for w in WORLDS)]
    names = sorted({c for w in T["worlds"].values() for c in w["rows"]},
                   key=lambda c: T["worlds"]["all2"]["rows"].get(c, {"err": {"mean": 9}})["err"]["mean"])
    for c in names:
        cells = []
        for world in WORLDS:
            r = T["worlds"][world]["rows"].get(c)
            if not r:
                cells.append(f"{'':>34s}")
            elif c == "rules":
                cells.append(f"{r['err']['mean']:>8.3f}{'':26s}")
            else:
                m = r.get("minus_rules")
                cells.append(f"{r['err']['mean']:>8.3f} ({m['mean']:+.3f} [{m['lo']:+.3f},{m['hi']:+.3f}])"
                             if m and m["lo"] is not None else f"{r['err']['mean']:>8.3f}")
        L.append(f"{c:14s} " + " ".join(f"{x:>34s}" for x in cells))
    L.append("\nDEATHS (nights the focal was dead at the end, of 80 scored), mean over seeds")
    L.append(f"{'chooser':14s} " + " ".join(f"{w:>8s}" for w in WORLDS))
    for c in names:
        L.append(f"{c:14s} " + " ".join(
            f"{T['worlds'][w]['rows'][c]['deaths']['mean']:8.2f}" if c in T["worlds"][w]["rows"] else f"{'':>8s}" for w in WORLDS))
    return "\n".join(L)


def text_more(T, B, S, F):
    """Further tables for the report: where the error comes from, the flows of acts, b/c/q/w by verb and by world."""
    L = []
    for world in ("all2", "noR"):
        w = T["worlds"][world]
        L.append(f"\nERROR BY DRIVE, world {world} (mean over seeds, nights 20-99); the energy column is hunger")
        L.append(f"{'chooser':14s} " + " ".join(f"{d:>9s}" for d in DRIVES) + f" {'total':>7s}")
        for c, v in sorted(w["split"].items(), key=lambda kv: sum(kv[1].values())):
            L.append(f"{c:14s} " + " ".join(f"{v[d]:9.3f}" for d in DRIVES) + f" {sum(v.values()):7.3f}")
    L.append("\nACTS PER NIGHT (world all2): the focal's kind and harsh acts made, and kind, help, harsh and gossip acts "
             "received from the others")
    L.append(f"{'chooser':14s} {'made_kind':>9s} {'made_harsh':>10s} {'recv_kind':>9s} {'recv_help':>9s} {'recv_harsh':>10s} "
             f"{'recv_gossip':>11s} {'strikes got':>11s} {'strikes made':>12s}")
    for c, v in sorted(F.get("all2", {}).items()):
        L.append(f"{c:14s} {v['made_kind']:9.2f} {v['made_harsh']:10.2f} {v['recv_kind']:9.2f} {v['recv_help']:9.2f} "
                 f"{v['recv_harsh']:10.2f} {v['recv_gossip']:11.2f} {v['recv_strike']:11.2f} {v['made_strike']:12.2f}")
    for world in WORLDS:
        L.append(f"\nb, c, q, w BY VERB, world {world} (pooled over all choosers and seeds; b and c in drive-error units)")
        L.append(f"{'verb':9s} {'n':>6s} {'b':>7s} {'c':>7s} {'q':>5s} {'q_eff':>5s} {'seen':>5s} {'told':>5s} {'w_same':>6s} "
                 f"{'w_next':>6s} {'daff':>7s} {'kind back':>9s} {'help back':>9s} {'harsh back':>10s}")
        for v, r in B["worlds"][world]["by_verb"].items():
            L.append(f"{v:9s} {r['n']:6d} {r['b']:7.3f} {r['c']:7.3f} {r['q']:5.2f} {r['q_eff']:5.2f} {r['q_seen']:5.2f} "
                     f"{r['q_told']:5.2f} {r['w_same']:6.2f} {(r['w_next'] if r['w_next'] is not None else float('nan')):6.2f} "
                     f"{r['daff']:7.3f} {r.get('reply_kind', float('nan')):9.2f} {r.get('reply_help', float('nan')):9.2f} "
                     f"{r.get('reply_harsh', float('nan')):10.2f}")
    L.append("\nTHE CONDITION q*b > c, by world and group of acts (seed is the unit; q = a third party saw or was told; "
             "q_eff = a third party's liking of the giver moved the way the deed points)")
    for world in WORLDS:
        for gname, g in B["worlds"][world]["groups"].items():
            L.append(f"  {world:5s} {gname:9s} n={g['pooled']['n']:6d} b={fmt0(g['b'])} c={fmt0(g['c'])} q={fmt0(g['q'], 2)} "
                     f"q_eff={fmt0(g['q_eff'], 2)} | q*b-c {fmt(g['qb_minus_c'])} | q_eff*b-c {fmt(g['qeffb_minus_c'])} "
                     f"| w_same {g['w_same']['mean']:.2f} w_next {g['w_next']['mean'] if g['w_next']['mean'] is not None else float('nan'):.2f}")
    L.append("\nGOSSIP: how far a told deed moves the hearer's liking of its doer")
    for w, d in B["gossip_effect"].items():
        L.append(f"  {w:5s} good deeds {d['good']}   bad deeds {d['bad']}")
    L.append("\nWHAT EACH STANCE DID (world all2, all lifetimes): share of finished acts that were person-directed, and "
             "person-directed acts outside the stance (ask and offer for food or drink are allowed for all but the strict hermit)")
    for c, v in S.items():
        L.append(f"  {c:14s} lifetimes {v['lifetimes']:3d} social share {v['social_share']:.3f} outside stance {v.get('outside_stance', '-')}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    R = load(a.runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = tournament(R)
    P = pass_rules_a(T, R) if "all2" in T["worlds"] and T["worlds"]["all2"]["rows"] else {}
    O = oracle(R)
    B = bcqw(R)
    S = stance_check(R)
    F = flows(R)
    BT = by_type(T, R) if "all2" in T["worlds"] and T["worlds"]["all2"]["rows"] else {}
    BC = by_cast(T, R) if BT else {}
    PP, SE = provocation_price(R), sensitivity(R)
    q_help = B["worlds"]["all2"]["groups"].get("help", {})
    summary = {"lifetimes": len(R), "cpu_hours": cpu_hours(R), "tournament_pass": {k: v for k, v in P.items()},
               "oracle_pass": O.get("pass"), "oracle_gap": O.get("gap"),
               "qb_vs_c": {"help_q": q_help.get("qb_minus_c"), "help_q_eff": q_help.get("qeffb_minus_c"),
                           "pass_q": bool(q_help and above(q_help["qb_minus_c"])),
                           "pass_q_eff": bool(q_help and above(q_help["qeffb_minus_c"]))}}
    for name, obj in (("0a_tournament", {**T, "pass_rules": P, "stance_check": S, "flows": F, "by_type": BT, "by_cast": BC}), ("0a_oracle", {**O, "provocation_price": PP}), ("0a_bcqw", B), ("0a_sensitivity", SE),
                      ("0a_summary", summary)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str))
    print(text(T, P, O, B, S) if P else "tournament runs missing")
    if P:
        for world in ("all2", "noR"):
            for g, d in BT.get(world, {}).items():
                print(f"\nBY NATURE (exploratory), world {world}, focal with {g.replace('_', ' ')}: paired difference, "
                      f"minus rules / minus hermit")
                for c in sorted(d["minus_rules"]):
                    print(f"  {c:14s} {fmt(d['minus_rules'][c]):28s} {fmt(d.get('minus_hermit', {}).get(c)) if c in d.get('minus_hermit', {}) else '':28s}")
        print(grid_text(T))
        for world in WORLDS:
            if world in BC:
                names = list(BC[world]["rules_error"])
                print(f"\nBY CAST MEMBER (exploratory, 2 seeds each), world {world}: chooser error minus rules error, "
                      f"mean of the 2 seeds (positive = worse than rules)")
                print(f"{'chooser':14s} " + " ".join(f"{f:>9s}" for f in names))
                for c, d in sorted(BC[world]["minus_rules"].items()):
                    print(f"{c:14s} " + " ".join(f"{d.get(f, float('nan')):+9.3f}" for f in names))
                print(f"{'rules error':14s} " + " ".join(f"{BC[world]['rules_error'][f]:9.3f}" for f in names))
        print(text_more(T, B, S, F))
    for c, v in PP.items():
        if c == "hard_minus_kind":
            print(f"provocation, hard partner minus kind partner (paired): {fmt(v['value'])}  "
                  f"per provocation {v['per_provocation']}")
        else:
            print(f"provocation {c}: error {fmt0(v['error'])}  provocations/night {v['provocations_per_night']}  "
                  f"harsh acts back/night {v['harsh_acts_back_per_night']}")
    for c, d in SE.items():
        print(f"sensitivity {c}: " + "  ".join(f"rate {k}: {fmt(v['minus_rules'])}" for k, v in d.items()))
    print("\nlifetimes", len(R), "cpu hours", cpu_hours(R))


if __name__ == "__main__":
    main()
