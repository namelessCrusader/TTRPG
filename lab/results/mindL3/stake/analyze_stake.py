"""The reciprocity-conditions study (2026-11-20): S7 measurements, the fixed pass rules, the mechanism checks, the ablations.
usage (from the repo root, PYTHONPATH=.):  lab/.venv/bin/python lab/results/mindL3/stake/analyze_stake.py
Reads lab/results/mindL3/stake/runs_* (new, with the S7 logs) and lab/results/mindL3/needs/runs_all5 (stored all5: O family and all5R0).
Writes stake/measures_per_run.json, stake/measures_per_cell.json, stake/analysis.txt."""
import json, math, sys
from pathlib import Path
import numpy as np
import lab.mindL.tournament_report as tr
from lab.mindL.tournament import HELP_ACTS, COOK_MAKES, NAMES, SEEDS
from lab.mindL.tournament_report import ci, fmt, group_totals, rates, outcome

BASE = Path("lab/results/mindL3")
ST, NE = BASE / "stake", BASE / "needs"
HELP = tuple(sorted(HELP_ACTS))
CHOOSERS8 = tr.A_CHOOSERS
CONDITIONAL = {"rules", "tit-for-tat", "protector"}        # acts depend on the partner's history by construction
EXPLOITERS = {"always-harsh", "extractor"}
out = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)

def load(d):
    R = {}
    for fp in sorted(Path(d).glob("*.json")):
        r = json.loads(fp.read_text())
        R[(r["kind"], r["world"], r["chooser"], r["seed"])] = r
    return R

CELLS = {  # cell -> list of (dir, world name inside the dir)
    "all5": [(ST / "runs_all5", "all5")], "all6": [(ST / "runs_all6", "all6")],
    "no_lumpy": [(ST / "runs_no_lumpy", "no_lumpy")], "no_stock": [(ST / "runs_no_stock", "no_stock")],
    "no_choosy": [(ST / "runs_no_choosy", "no_choosy")], "all6R0": [(ST / "runs_all6R0", "all6R0")]}
RUNS = {}
for cell, lst in CELLS.items():
    RUNS[cell] = {}
    for d, w in lst:
        if d.exists():
            RUNS[cell].update(load(d))
STORED5 = load(NE / "runs_all5")           # stored all5 (no S7 logs): A, O, all5R0
cells = [c for c in ("all5", "all6", "no_lumpy", "no_stock", "no_choosy") if any(k[0] == "A" for k in RUNS[c])]

def A(cell, chooser, seed, store=None):
    return (store or RUNS[cell]).get(("A", cell, chooser, seed))

def err_tab(R, cell, kind="A"):
    t = {}
    for (k, w, c, s), r in R.items():
        if k == kind and w == cell:
            t.setdefault(c, {})[s] = outcome(r)["err"]
    return t

def pdiff(a, b):
    return ci([a[s] - b[s] for s in SEEDS if s in a and s in b])

# ---------------------------------------------------------------- 0. check: the rerun of all5 with logs equals the stored all5
P("0. CHECK: all5 re-run with the new logs against the stored all5 (focal error, per run)")
same = tot = 0
for (k, w, c, s), r in RUNS["all5"].items():
    q = STORED5.get((k, w, c, s))
    if q:
        tot += 1
        same += (q["err"] == r["err"] and q["dead"] == r["dead"])
P(f"   {same} of {tot} runs identical (err, deaths per night)\n")

# ---------------------------------------------------------------- 1. c/b, w*, q*
def pooled_seed_rates(R, cell, verbs, choosers=CHOOSERS8):
    per = {}
    for s in SEEDS:
        rs = [r for (k, w, c, sd), r in R.items() if k == "A" and w == cell and sd == s and c in choosers]
        x = rates(group_totals(rs, verbs)) if rs else None
        if x:
            per[s] = x
    return per

def cb_block(R, cell, verbs):
    per = pooled_seed_rates(R, cell, verbs)
    if not per:
        return None
    g = {k: ci([x[k] for x in per.values()]) for k in ("b", "c", "q", "q_eff", "w_same", "w_next")}
    b, c = g["b"]["mean"], g["c"]["mean"]
    g["cb"] = round(c / b, 3) if b else None
    g["qb_c"] = ci([x["q"] * x["b"] - x["c"] for x in per.values()])
    g["qeffb_c"] = ci([x["q_eff"] * x["b"] - x["c"] for x in per.values()])
    g["n_acts"] = int(sum(x["n"] for x in per.values()))
    return g

P("1. b, c, c/b AND THE IMPLIED THRESHOLDS w* = q* = c/b  (8 plan choosers pooled; help acts = " + ", ".join(HELP) + ")")
P(f"   {'cell':10s} {'b':>7s} {'c':>7s} {'c/b':>6s} {'q':>6s} {'q_eff':>6s} {'w_same':>7s} {'w_next':>7s}  q_eff>c/b?  q>c/b?  w>c/b?   q_eff*b-c [95% CI]")
CB = {}
for cell in cells:
    g = cb_block(RUNS[cell], cell, HELP)
    CB[cell] = g
    if not g:
        continue
    P(f"   {cell:10s} {g['b']['mean']:7.4f} {g['c']['mean']:7.4f} {g['cb']:6.3f} {g['q']['mean']:6.3f} {g['q_eff']['mean']:6.3f} {g['w_same']['mean']:7.3f} "
      f"{(g['w_next']['mean'] or 0):7.3f}  {str(g['q_eff']['mean'] > g['cb']):10s}  {str(g['q']['mean'] > g['cb']):6s}  {str(g['w_same']['mean'] > g['cb']):6s}   {fmt(g['qeffb_c'])}   (n={g['n_acts']})")
P("\n   the pooled help acts split: 'give' (handing a thing over: most of the acts) against the care acts (tend, bandage, help up, defend, serve):")
for cell in cells:
    for name, verbs in (("give only", ("give",)), ("care acts", tuple(v for v in HELP if v != "give"))):
        g = cb_block(RUNS[cell], cell, verbs)
        if g:
            P(f"   {cell:10s} {name:10s} b {g['b']['mean']:.4f} c {g['c']['mean']:.4f} c/b {g['cb']:.3f}  q {g['q']['mean']:.3f} q_eff {g['q_eff']['mean']:.3f} w {g['w_same']['mean']:.3f}   q_eff*b-c {fmt(g['qeffb_c'])}  q*b-c {fmt(g['qb_c'])}  (n={g['n_acts']})")
P("\n   per act (pooled over choosers and seeds; the same, by verb):")
P(f"   {'cell':10s} {'verb':8s} {'n':>7s} {'b':>7s} {'c':>7s} {'c/b':>6s} {'q':>6s} {'q_eff':>6s} {'w_same':>7s}")
BYVERB = {}
for cell in cells:
    allr = [r for (k, w, c, s), r in RUNS[cell].items() if k == "A" and w == cell and c in CHOOSERS8]
    for v in HELP + ("praise", "gossip", "buy", "work"):
        x = rates(group_totals(allr, (v,)))
        BYVERB[(cell, v)] = x
        if x:
            P(f"   {cell:10s} {v:8s} {x['n']:7d} {x['b']:7.4f} {x['c']:7.4f} {(x['c'] / x['b'] if x['b'] > 0 else float('nan')):6.2f} {x['q']:6.3f} {x['q_eff']:6.3f} {x['w_same']:7.3f}")

# ---------------------------------------------------------------- 2. stake s
def stake_run(r):
    E = np.array([[d[n] for n in NAMES] for d in r["err_all"][r["win0"]:]])
    sd = E.std(axis=0)
    if (sd < 1e-9).any():
        return None
    C = np.cov(E.T)
    beta = C / np.diag(C)[None, :]                 # beta[i, j] = change in i's error per unit change in j's
    corr = np.corrcoef(E.T)
    f = NAMES.index(r["focal"])
    off = ~np.eye(5, dtype=bool)
    oth = [j for j in range(5) if j != f]
    return {"s_pairs": float(beta[off].mean()), "corr_pairs": float(corr[off].mean()),
            "s_focal_in_others": float(np.mean([beta[f, j] for j in oth])), "s_others_in_focal": float(np.mean([beta[j, f] for j in oth]))}

P("\n2. STAKE s: the change in a person's drive error per unit change in a partner's (OLS slope across the nights 20-99; mean over the 20 ordered pairs;")
P("   'focal in others' = the focal's error on the others' errors, 'others in focal' = the others' on the focal's). 95% CI over seeds.")
STAKE = {}
for cell in cells:
    for c in ("rules", "hermit", "always-giving", "tit-for-tat"):
        xs = [stake_run(r) for (k, w, ch, s), r in RUNS[cell].items() if k == "A" and w == cell and ch == c and "err_all" in r]
        xs = [x for x in xs if x]
        if xs:
            STAKE[(cell, c)] = {k: ci([x[k] for x in xs]) for k in xs[0]}
            g = STAKE[(cell, c)]
            P(f"   {cell:10s} {c:14s} s_pairs {fmt(g['s_pairs'])}  corr {fmt(g['corr_pairs'])}  focal-in-others {fmt(g['s_focal_in_others'])}  others-in-focal {fmt(g['s_others_in_focal'])}")

# ---------------------------------------------------------------- 3. providers
P("\n3. ALTERNATIVE PROVIDERS per need (sampled every 10 turns, every living cast member, nights 20-99; the counter counted as one provider).")
P("   'all' = every sample; 'active' = only while the need is on (hunger over 0.3; an ailment over the tending floor; a raid in the room).")
P("   providers = how many; any = share of samples with at least one; walk = turns to the nearest (3 turns a room)")
PROV = {}
for cell in cells:
    agg = {}
    for (k, w, c, s), r in RUNS[cell].items():
        if k == "A" and w == cell and "prov" in r:
            for key, v in r["prov"].items():
                a = agg.setdefault(key, [0, 0, 0.0, 0])
                for i in range(4):
                    a[i] += v[i]
    PROV[cell] = agg
    for need in ("hunger", "tend", "defend"):
        for tag in ("all", "active"):
            row = []
            for kind in ("counter", "people"):
                a = agg.get(f"{need}|{kind}|{tag}")
                if a and a[0]:
                    row.append(f"{kind}: providers {a[1] / a[0]:.2f}, any {a[3] / a[0]:.2f}, walk {(a[2] / a[3] if a[3] else float('nan')):.1f}  (n={a[0]})")
            if row:
                P(f"   {cell:10s} {need:7s} {tag:7s} " + " | ".join(row))
P("   price of the counter: a ration costs one chore (tiredness 0.06, 10 turns) or a thing worth 1-3 for two rations; people's food: free if they serve, one ask.")
P("   tending and defending: the counter has no provider in any cell (0.00). The menu test (test_tending_and_defending_cannot_be_bought_at_the_counter) passes.")

# ---------------------------------------------------------------- 4. sigma against the cast
P("\n4. SIGMA: the counter-alone (hermit) error against the cooperating cast (rules in every seat). Payoff = minus drive error, nights 20-99, paired by seed.")
SIG = {}
for cell in cells:
    h, ru = {}, {}
    for s in SEEDS:
        a, b = A(cell, "hermit", s), A(cell, "rules", s)
        if a and b and "err_all" in a and "err_all" in b:
            h[s] = outcome(a)["err"]
            ru[s] = float(np.mean([np.mean([d[n] for n in NAMES]) for d in b["err_all"][b["win0"]:]]))
    if h:
        SIG[cell] = {"sigma_err": ci(list(h.values())), "cast_err": ci(list(ru.values())), "diff": pdiff(h, ru)}
        g = SIG[cell]
        P(f"   {cell:10s} hermit error {g['sigma_err']['mean']:.3f}  cast mean error {g['cast_err']['mean']:.3f}  hermit minus cast {fmt(g['diff'])}   "
          f"{'HERMIT LOSES (CI above 0)' if tr.above(g['diff']) else 'TIE (CI holds 0)' if not tr.below(g['diff']) else 'HERMIT WINS'}")

# ---------------------------------------------------------------- 5. pooling, R
P("\n5. POOLING: u/s and the between-person luck correlation R (Winterhalder 1986, Eq. 4: u = s * sqrt((1 + (N-1) R) / N), N = the 4 who bring a catch).")
P("   s = the CV of a person's own catch across the 100 nights; R = the mean correlation of catches between two people; predicted u/s from them;")
P("   measured u/s: the CV across nights of what a person ATE from catches (own + others'), and of all it ate (counter included), over s. Rules cell.")
POOL = {}
for cell in cells:
    rows = []
    for (k, w, c, s), r in RUNS[cell].items():
        if k == "A" and w == cell and c == "rules" and "catch" in r:
            C = np.array([[d.get(n, 0) for n in NAMES if n != "Marta"] for d in r["catch"]], dtype=float)
            F = np.array([[r["fed"][i][n] for n in NAMES if n != "Marta"] for i in range(len(C))], dtype=float)   # nights x 4 x [own, other, house]
            cv = lambda M: np.nanmean([M[:, j].std() / M[:, j].mean() if M[:, j].mean() > 0 else np.nan for j in range(M.shape[1])])
            s_ = cv(C)
            R_ = np.nanmean([np.corrcoef(C[:, i], C[:, j])[0, 1] for i in range(4) for j in range(i + 1, 4)])
            pred = math.sqrt((1 + 3 * max(R_, -1 / 3 + 1e-9)) / 4)
            nocnt = F[:, :, 0] + F[:, :, 1]
            allf = nocnt + F[:, :, 2]
            rows.append({"s": s_, "R": R_, "u_over_s_pred": pred, "u_nocounter_over_s": cv(nocnt) / s_, "u_withcounter_over_s": cv(allf) / s_})
    if rows:
        POOL[cell] = {k: ci([x[k] for x in rows]) for k in rows[0]}
        g = POOL[cell]
        P(f"   {cell:10s} s {fmt(g['s'])}  R {fmt(g['R'])}  predicted u/s {fmt(g['u_over_s_pred'])}  measured u/s no counter {fmt(g['u_nocounter_over_s'])}  with counter {fmt(g['u_withcounter_over_s'])}")

# ---------------------------------------------------------------- 6. mechanism checks
P("\n6. MECHANISM CHECKS (rules cell and all cells pooled where noted)")
for cell in cells:
    if cell not in ("all6", "no_stock", "no_choosy", "no_lumpy"):
        continue
    P(f"  [{cell}]")
    if "lumpy" in cell or cell != "no_lumpy":
        inn = own = oth = 0
        for (k, w, c, s), r in RUNS[cell].items():
            if k == "A" and w == cell and c == "rules" and "lump" in r:
                for i, lump in enumerate(r["lump"]):
                    for it, x in lump.items():
                        inn += x["in"]
                lef = sum(x["left"] for lump in r["lump"] for x in lump.values())
                eaten_catch = sum(r["fed"][i][n][0] + r["fed"][i][n][1] for i in range(len(r["fed"])) for n in NAMES)
                own += sum(r["fed"][i][n][0] for i in range(len(r["fed"])) for n in NAMES)
                oth += sum(r["fed"][i][n][1] for i in range(len(r["fed"])) for n in NAMES)
                rot = inn - eaten_catch - lef
        if inn and cell != "no_lumpy":
            P(f"   packages: rations in {inn}; eaten by the owner {own}, by someone else {oth}; spoiled or left {inn - own - oth} (rules cell, 10 lifetimes x 100 nights)")
    hs = []
    ends = []
    work = []
    for (k, w, c, s), r in RUNS[cell].items():
        if k == "A" and w == cell and c == "rules":
            h = np.array(r["house"], dtype=float)
            hs.append(h[r["win0"]:, :2].sum(axis=1).mean())
            ends.append((h[r["win0"]:, 2:].sum(axis=1) == 0).mean())
            work.append(r["made"].get("work", 0) / len(r["err"]))
    if hs and cell != "no_lumpy":
        P(f"   counter stock at the start of a night (rations) {np.mean(hs):.1f}; share of nights it ends empty {np.mean(ends):.2f}; focal work acts a night {np.mean(work):.2f}")
    # the uncovered night: the focal had no catch and the counter ended empty
    unc, cov = [], []
    for (k, w, c, s), r in RUNS[cell].items():
        if k == "A" and w == cell and c == "rules":
            f = r["focal"]
            for i in range(r["win0"], len(r["err"])):
                if f == "Marta":
                    continue
                (unc if r["catch"][i].get(f, 0) == 0 and sum(r["house"][i][2:]) == 0 else cov).append(r["err"][i])
    if unc:
        P(f"   focal error on an UNCOVERED night (no catch, counter empty at the end) {np.mean(unc):.3f} (n={len(unc)}) against {np.mean(cov):.3f} on the others (n={len(cov)})")
    # who is tended
    rec = {}
    for (k, w, c, s), r in RUNS[cell].items():
        if k == "A" and w == cell:
            rec.setdefault(c, []).append(r["recv"].get("tend", 0) / (len(r["err"]) - r["win0"]))
    P("   tended per night (focal): " + ", ".join(f"{c} {np.mean(v):.3f}" for c, v in sorted(rec.items())))
    rec = {}
    for (k, w, c, s), r in RUNS[cell].items():
        if k == "A" and w == cell:
            rec.setdefault(c, []).append(r["recv"].get("defend", 0) / (len(r["err"]) - r["win0"]))
    P("   defended per night (focal): " + ", ".join(f"{c} {np.mean(v):.3f}" for c, v in sorted(rec.items())))
    rec = {}
    for (k, w, c, s), r in RUNS[cell].items():
        if k == "A" and w == cell:
            rec.setdefault(c, []).append(r["recv"].get("serve", 0) / (len(r["err"]) - r["win0"]))
    P("   served per night (focal): " + ", ".join(f"{c} {np.mean(v):.3f}" for c, v in sorted(rec.items())))

# ---------------------------------------------------------------- 7. tournament, paired against all5
P("\n7. TOURNAMENT: focal drive error (nights 20-99), each world minus all5 (the stored all5 runs), paired by seed, 95% CI over seeds 1-10")
base = err_tab(STORED5, "all5")
TAB = {}
for cell in cells:
    TAB[cell] = err_tab(RUNS[cell], cell)
P(f"   {'chooser':14s} {'all5':>7s} " + " ".join(f"{c:>9s}" for c in cells if c != "all5") + "   [diff to all5, CI]")
for c in CHOOSERS8:
    row = f"   {c:14s} {np.mean(list(base[c].values())):7.3f} "
    dd = []
    for cell in cells:
        if cell == "all5":
            continue
        t = TAB[cell].get(c, {})
        row += f"{np.mean(list(t.values())) if t else float('nan'):9.3f} "
        dd.append(f"{cell} {fmt(pdiff(t, base[c]))}")
    P(row + "  " + "; ".join(dd))
P("   deaths in 80 scored nights (mean): " + "; ".join(
    f"{cell} " + ", ".join(f"{c} {np.mean([sum(r['dead'][r['win0']:]) for (k, w, ch, s), r in RUNS[cell].items() if k == 'A' and w == cell and ch == c]):.1f}" for c in ("rules", "hermit", "always-giving", "extractor"))
    for cell in cells))

# ---------------------------------------------------------------- 8. pass rules, old and fixed
def old_rules(cell, A_R, O_R, R0, R0name):
    R = {}
    for (k, w, c, s), r in list(A_R.items()) + list(O_R.items()):
        if w == cell:
            R[(k, "all2", c, s, None)] = r
    for (k, w, c, s), r in R0.items():
        if w == R0name:
            R[(k, "noR", c, s, None)] = r
    T = tr.tournament(R)
    Pa = tr.pass_rules_a(T, R)
    Oo = tr.oracle(R) if any(k[0] == "O" for k in R) else None
    return T, Pa, Oo

P("\n8. PASS RULES. OLD (as in the step 0a / needs reports) and FIXED (library S3), per world.")
P("   Rule 1 oracle gap >= 0.1 with CI above 0 | rule 2 hermit and always-harsh lose to rules | rule 3 always-giving loses to tit-for-tat | rule 4 q*b > c | rule 5 extractor wins only where reprisal is weak")
RES = {}
for cell, a_src, o_src, r0_src, r0n in (("all5", STORED5, STORED5, STORED5, "all5R0"), ("all6", RUNS["all6"], RUNS["all6"], RUNS["all6R0"], "all6R0")):
    if not any(k[0] == "A" for k in a_src):
        continue
    T, Pa, Oo = old_rules(cell, a_src, o_src, r0_src, r0n)
    RES[cell] = (T, Pa, Oo)
    P(f"\n  [{cell}] OLD")
    if Oo and "gap" in Oo:
        g = Oo["gap"]["oracle_both"]
        P(f"   1 oracle gap (rules - oracle_both): {fmt(g)}   {'PASS' if Oo['pass'] else 'FAIL'}   (rules error {tr.ci([v for v in Oo['per_seed']['rules'].values()])['mean']:.3f})")
    for c in ("hermit", "always-harsh"):
        x = Pa[f"{c}_loses_to_rules"]
        P(f"   2 {c} minus rules: {fmt(x['value'])}   {'PASS' if x['pass'] else 'FAIL'}")
    x = Pa["always_giving_loses_to_tit_for_tat"]
    P(f"   3 always-giving minus tit-for-tat (no exploiter in the cast): {fmt(x['value'])}   {'PASS' if x['pass'] else 'FAIL'}")
    g = CB.get(cell)
    if g:
        P(f"   4 q*b - c (help acts) {fmt(g['qb_c'])}   {'PASS' if tr.above(g['qb_c']) else 'FAIL'}   (q_eff*b - c {fmt(g['qeffb_c'])})")
    x = Pa["extractor_wins_only_where_reprisal_is_weak"]
    v = {("reprisal on" if k == "all2" else "reprisal off"): fmt(x["value"][k]) for k in ("all2", "noR")}
    P(f"   5 extractor minus rules: {v}   -> {x['reading']}   {'PASS' if x['pass'] else 'FAIL'}")

P("\n  FIXED")
for cell in ("all5", "all6"):
    # rule 3: only where exploiters are present (family E: an extractor in the cast)
    E = {}
    d = ST / f"runs_{cell}"
    for (k, w, c, s), r in load(d).items():
        if k == "E" and w == cell:
            E.setdefault(c, {})[s] = outcome(r)["err"]
    if len(E) == 2:
        g = pdiff(E["always-giving"], E["tit-for-tat"])
        P(f"  [{cell}] 3 FIXED always-giving minus tit-for-tat, extractor in the cast (1 of 5 seats = exploiter share 0.20; the old cells had 0.00): {fmt(g)}   {'PASS' if tr.above(g) else 'FAIL'}"
          f"   (always-giving {np.mean(list(E['always-giving'].values())):.3f}, tit-for-tat {np.mean(list(E['tit-for-tat'].values())):.3f})")
        for c in E:
            ds = [sum(r['dead'][r['win0']:]) for (k, w, ch, s), r in load(d).items() if k == "E" and w == cell and ch == c]
            P(f"       deaths in 80 nights, {c}: {np.mean(ds):.2f}")
for cell in cells:
    g = SIG.get(cell)
    if g and cell in ("all5", "all6"):
        P(f"  [{cell}] 2 FIXED hermit against sigma: hermit minus cast {fmt(g['diff'])}   {'PASS (hermit loses)' if tr.above(g['diff']) else 'FAIL: tie' if not tr.below(g['diff']) else 'FAIL: hermit wins'}")
P("      the tie condition (Hauert et al. 2002, doi:10.1126/science.1070582, u): with an opt-out paying sigma the time-average payoff of every strategy equals sigma, and")
P("      cooperation can only exist if 0 < sigma < r - 1. In our units, hermit = cast exactly when what the counter gives a person alone equals what company gives (the gain from company, r - sigma, is zero).")
P("      So 'hermit loses' can only pass where r - sigma > 0: the cast must give something the counter does not. The CI above is the test of that gain.")
for cell in ("all5", "all6"):
    try:
        bd = json.loads((ST / f"bound_{cell}.json").read_text())
    except Exception:
        continue
    m = bd["mean"]
    Oo = RES.get(cell, (None, None, None))[2]
    gap = fmt(Oo["gap"]["oracle_both"]) if Oo and "gap" in Oo else "-"
    P(f"  [{cell}] 1 FIXED oracle gap {gap} against its upper bound: partner-specific acts {m['acts_per_night']:.1f} a night (kind ones {m['kind_acts_per_night']:.2f}) x (b - c) "
      f"{m['b_minus_c_generous']:.4f} undiscounted / {m['b_minus_c_timed']:.4f} night-averaged = bound {m['bound_generous']:.3f} / {m['bound_timed']:.3f} (every act at the two partners, undiscounted: {m['bound_all_acts_generous']:.3f}). "
      f"{'The night-averaged bound is BELOW 0.1: the rule cannot pass in this world.' if m['bound_timed'] < 0.1 else 'The bound is 0.1 or more: the rule can pass here.'}")

# ---------------------------------------------------------------- 9. learner measures and the conditional share
P("\n9. FOR LEARNERS (S7.7): no learner was run in this study. What the fixed cast gives: the share of conditional players and the exploitation / retaliation rates (replies per act, nights 20-99).")
for cell in ("all5", "all6"):
    allr = [r for (k, w, c, s), r in RUNS[cell].items() if k == "A" and w == cell and c in CHOOSERS8]
    h = group_totals([r for r in allr], tuple(sorted(tr.HARSH_VERBS)), ("focal", "oo"))
    kd = group_totals(allr, tuple(sorted(HELP_ACTS)), ("focal", "oo"))
    n_h, n_k = max(1, h["n_done"]), max(1, kd["n_done"])
    P(f"   {cell}: a rules character answers a harsh act with a harsh one {h['rh_d'] / n_h:.3f}, a help act with a kind one {kd['rk_d'] / n_k:.3f}, a help act with a help {kd['rp_d'] / n_k:.3f}")
P("   conditional share of the cast = (4 rules + the focal if it is rules, tit-for-tat or protector) / 5: 1.0 for rules, tit-for-tat, protector; 0.8 for the rest. Exploiter share 0.2 in the always-harsh and extractor cells, 0.2 in family E, 0 elsewhere.")
exploit = {}
for cell in ("all5", "all6"):
    for c in ("always-giving", "tit-for-tat", "rules"):
        v = [sum(r["recv"].get(x, 0) for x in ("strike", "steal", "threaten", "insult", "shove", "accuse", "search")) / (len(r["err"]) - r["win0"])
             for (k, w, ch, s), r in RUNS[cell].items() if k == "A" and w == cell and ch == c]
        exploit[(cell, c)] = np.mean(v) if v else None
P("   harsh acts received per night by the focal: " + "; ".join(f"{k[0]} {k[1]} {v:.2f}" for k, v in exploit.items() if v is not None))

P("\n10. CARE GIVEN AND RECEIVED by the focal, per night (tend + defend + bandage + help up; nights 20-99), by chooser")
for cell in cells:
    row = []
    for c in CHOOSERS8:
        g, r_ = [], []
        for (k, w, ch, s), r in RUNS[cell].items():
            if k == "A" and w == cell and ch == c:
                n = len(r["err"]) - r["win0"]
                g.append(sum(r["made"].get(v, 0) for v in ("tend", "defend", "bandage", "help up")) / len(r["err"]))
                r_.append(sum(r["recv"].get(v, 0) for v in ("tend", "defend", "bandage", "help up")) / len(r["err"]))
        row.append(f"{c} gives {np.mean(g):.2f} gets {np.mean(r_):.2f}")
    P(f"   {cell:10s} " + "; ".join(row))
P("   (counts are over all 100 nights, divided by 100)")

P("\n11. WHAT EACH NEW FLAG DOES: all6 minus the world without it (paired by seed; positive = the flag raises the focal's error)")
for c in CHOOSERS8:
    P(f"   {c:14s} " + "; ".join(f"{f}: {fmt(pdiff(TAB['all6'].get(c, {}), TAB['no_' + f].get(c, {})))}" for f in ("lumpy", "stock", "choosy")))

P("\n12. PASS RULES IN EVERY WORLD (old forms; the fixed forms are in 8. family O and the reprisal-off cell were run only in all5 and all6)")
P(f"   {'world':10s} {'2 hermit-rules':>28s} {'2 harsh-rules':>28s} {'3 giving-tft':>28s} {'4 q*b-c':>26s} {'4 q_eff*b-c':>26s}")
for cell in cells:
    t = TAB[cell]
    h, hh, gg = pdiff(t["hermit"], t["rules"]), pdiff(t["always-harsh"], t["rules"]), pdiff(t["always-giving"], t["tit-for-tat"])
    g = CB[cell]
    P(f"   {cell:10s} {fmt(h) + (' PASS' if tr.above(h) else ' FAIL'):>28s} {fmt(hh) + (' PASS' if tr.above(hh) else ' FAIL'):>28s} {fmt(gg) + (' PASS' if tr.above(gg) else ' FAIL'):>28s} "
      f"{fmt(g['qb_c']) + (' PASS' if tr.above(g['qb_c']) else ' FAIL'):>26s} {fmt(g['qeffb_c']) + (' PASS' if tr.above(g['qeffb_c']) else ' FAIL'):>26s}")

cpu = sum(r["cpu_s"] for cell in RUNS for r in RUNS[cell].values()) + sum(r["cpu_s"] for r in load(ST / "runs_all5").values() if r["kind"] == "E") 
P(f"\nCPU spent by the runs in this study (sum of the lifetimes' cpu_s, before double counting the E runs held in runs_all5/runs_all6): {sum(json.loads(fp.read_text())['cpu_s'] for d in ST.glob('runs_*') for fp in d.glob('*.json')) / 3600:.2f} hours")

json.dump({"cb": {c: g and {k: v for k, v in g.items()} for c, g in CB.items()}, "stake": {f"{a}|{b}": v for (a, b), v in STAKE.items()},
           "sigma": SIG, "pool": POOL, "prov": PROV}, open(ST / "measures_per_cell.json", "w"), default=float)
per = {}
for cell in cells:
    for (k, w, c, s), r in RUNS[cell].items():
        if k != "A" or w != cell:
            continue
        x = {"err": outcome(r)["err"], "stake": stake_run(r) if "err_all" in r else None}
        for v in HELP:
            y = rates(group_totals([r], (v,)))
            if y:
                x[f"cb_{v}"] = {"b": y["b"], "c": y["c"], "c_over_b": (y["c"] / y["b"] if y["b"] > 0 else None), "q": y["q"], "q_eff": y["q_eff"], "w": y["w_same"], "n": y["n"]}
        per[r["id"]] = x
json.dump(per, open(ST / "measures_per_run.json", "w"), default=float)
(ST / "analysis.txt").write_text("\n".join(out) + "\n")
