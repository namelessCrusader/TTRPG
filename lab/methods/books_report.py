"""The book pilot's numbers (wave 11b): which formulation, which model, and
whether book cases separate them more than the lab's templated cases do.

    python3 -m lab.methods.books_report          # lab/results/books/numbers.json and report.md
    python3 -m lab.methods.books_report --v2 MODEL [--root DIR]   # recipe v2: each form vs ctx_full

Reads the decisions of `lab.methods.screen run MODEL --set books` (in the
git-ignored books/derived/screen/d) and, for the comparison, the standing
screen's decisions on the labeled cases (lab/results/screen/d) and the
describe A/B's (lab/results/describe_ab/d). Writes NUMBERS AND CASE IDS ONLY:
no book-derived text leaves books/derived/.

THE LENGTH GATE. The pilot's quality check (2026-10-16) found the book's
option is often the longest, so a chooser that picks the longest option lands
in the acceptable set far above chance. The headline numbers are therefore
on the LENGTH-FAIR cases (the book's option is shorter than the longest
option), where the longest-option pick is printed next to chance and must not
beat it; the numbers on all cases are given beside them, each with the
longest-option baseline. The cases with an outcome leak (screen.BOOK_LEAKS)
are left out of everything.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from lab.methods import screen as S
from lab.methods.describe_ab import mcnemar, paired, wilson

HERE = Path(__file__).parent
OUT = HERE.parent / "results" / "books"
MODELS = ("gdecide", "gdecide1b", "laya", "gliclass", "von", "vegaml")
TEMPLATED = ("people", "personality", "personality2")


def hits(dec, meta, ids):
    return [int(dec[i]["pick"] in meta[i]["good"]) for i in ids]


def acc(xs):
    return {"n": len(xs), "acc": round(sum(xs) / len(xs), 4) if xs else None, "ci": wilson(sum(xs), len(xs))}


def holm(ps):
    """Holm-adjusted p values, in the order given."""
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (len(ps) - r)))
        adj[i] = run
    return adj


def pairs_table(names, hit, ids):
    """Every pair of `names` on the same `ids`: the paired difference (first -
    second), its 95% interval, the exact McNemar p and the Holm-adjusted p."""
    got = []
    for a, b in combinations(names, 2):
        p = paired([hit[a][i] for i in ids], [hit[b][i] for i in ids])
        got.append({"a": a, "b": b, **p})
    for g, h in zip(got, holm([g["p_mcnemar"] for g in got])):
        g["p_holm"] = h
    return got


def separation(names, hit, ids):
    """How far apart choosers are on `ids`: the range and SD of their acc@1,
    the mean |z| of the paired differences, and how many pairs differ at
    Holm p < 0.05."""
    accs = {n: sum(hit[n][i] for i in ids) / len(ids) for n in names}
    pt = pairs_table(names, hit, ids)
    zs = []
    for p in pt:
        se = (p["ci"][1] - p["ci"][0]) / (2 * 1.96)
        zs.append(abs(p["diff"]) / se if se > 0 else 0.0)
    return {"n": len(ids), "acc": {k: round(v, 4) for k, v in accs.items()},
            "range": round(max(accs.values()) - min(accs.values()), 4),
            "sd": round(statistics.pstdev(accs.values()), 4),
            "mean_abs_z": round(sum(zs) / len(zs), 2) if zs else None,
            "pairs_holm_05": sum(p["p_holm"] < 0.05 for p in pt), "pairs": len(pt)}


def nvo(dec, meta, ids):
    """On the nature-vs-obvious cases: how often the pick is the book's choice,
    the obvious act, and the book's share when it is one of the two."""
    ids = [i for i in ids if meta[i]["nvo"] and meta[i]["obvious"] is not None]
    b = sum(dec[i]["pick"] == meta[i]["choice"] for i in ids)
    o = sum(dec[i]["pick"] == meta[i]["obvious"] for i in ids)
    lb = sum(meta[i]["first_longest"] == meta[i]["choice"] for i in ids)
    lo = sum(meta[i]["first_longest"] == meta[i]["obvious"] for i in ids)
    return {"n": len(ids), "book": round(b / len(ids), 4) if ids else None, "book_ci": wilson(b, len(ids)),
            "obvious": round(o / len(ids), 4) if ids else None,
            "book_share_of_two": round(b / (b + o), 4) if b + o else None, "book_share_ci": wilson(b, b + o),
            "longest_is_book": round(lb / len(ids), 4) if ids else None,
            "longest_is_obvious": round(lo / len(ids), 4) if ids else None,
            "acc": acc(hits(dec, meta, ids))}


def gate(meta, ids):
    """The longest-option pick against a random pick, on `ids`."""
    lb = [int(meta[i]["first_longest"] in meta[i]["good"]) for i in ids]
    lbook = [int(meta[i]["first_longest"] == meta[i]["choice"]) for i in ids]
    ch = sum(len(meta[i]["good"]) / meta[i]["n"] for i in ids) / len(ids)
    chb = sum(1 / meta[i]["n"] for i in ids) / len(ids)
    return {"n": len(ids), "longest_in_good": acc(lb), "random_in_good": round(ch, 4),
            "longest_is_book": round(sum(lbook) / len(ids), 4), "random_is_book": round(chb, 4),
            "passes": sum(lb) / len(ids) <= ch}


def neutral_weight(meta, clean):
    """The weight w on the length-UNFAIR cases (the book's option is a longest
    one) such that, over w * unfair + (1 - w) * fair, the longest-option pick
    lands in the acceptable set exactly as often as a random pick: the mix in
    which option length carries no signal."""
    u = [i for i in clean if not meta[i]["fair"]]
    f = [i for i in clean if meta[i]["fair"]]
    g = gate(meta, u), gate(meta, f)
    lu, lf = g[0]["longest_in_good"]["acc"], g[1]["longest_in_good"]["acc"]
    cu, cf = g[0]["random_in_good"], g[1]["random_in_good"]
    return round((cf - lf) / ((lu - cu) + (cf - lf)), 4)


def neutral(h, meta, clean, w):
    """acc@1 over the length-neutral mix (neutral_weight), with a 95% interval."""
    u = [h[i] for i in clean if not meta[i]["fair"]]
    f = [h[i] for i in clean if meta[i]["fair"]]
    au, af = sum(u) / len(u), sum(f) / len(f)
    a = w * au + (1 - w) * af
    se = math.sqrt(w * w * au * (1 - au) / len(u) + (1 - w) ** 2 * af * (1 - af) / len(f))
    return {"acc": round(a, 4), "ci": [round(a - 1.96 * se, 4), round(a + 1.96 * se, 4)], "w_unfair": w}


def templated():
    """The standing screen's decisions on the templated people/personality
    cases (src text, wave 10), and the describe A/B's lab and new texts on them
    (gdecide)."""
    qr = S.rows(S.OUT / "q" / "labeled.jsonl")
    q = {r["id"]: set(r["good"]) for r in qr}
    nopt = {r["id"]: len(r["options"]) for r in qr}
    ids = sorted(i for i in q if i.split("/")[0] in TEMPLATED)
    hit = {}
    for m in MODELS:
        d = {r["request"]: r for r in S.rows(S.OUT / "d" / m / "labeled.jsonl")}
        if all(i in d for i in ids):
            hit[m] = {i: int(d[i]["pick"] in q[i]) for i in ids}
    man = S.rows(S.HERE.parent / "results" / "describe_ab" / "q" / "manifest.jsonl")[1:]
    good = {m["id"]: set(m["good"]) for m in man if m["set"] in TEMPLATED}
    fh = {}
    for t in ("lab", "new"):
        d = {r["request"].rsplit("#", 1)[0]: r for r in S.rows(S.HERE.parent / "results" / "describe_ab" / "d" / f"{t}.jsonl")}
        fh[t] = {i: int(d[i]["pick"] in g) for i, g in good.items() if i in d}
    fids = sorted(set(fh["lab"]) & set(fh["new"]))
    chance = round(sum(len(q[i]) / nopt[i] for i in ids) / len(ids), 4)
    return ids, hit, fids, fh, chance


def main():
    meta = S.book_cases()
    clean = [i for i in meta if not meta[i]["leak"]]
    fair = [i for i in clean if meta[i]["fair"]]
    res = {"cases": len(meta), "leaks_left_out": sorted(S.BOOK_LEAKS), "clean": len(clean), "fair": len(fair),
           "gate": {"all": gate(meta, clean), "fair": gate(meta, fair),
                    "unfair": gate(meta, [i for i in clean if not meta[i]["fair"]])}}
    w = res["neutral_w_unfair"] = neutral_weight(meta, clean)

    # ---- formulations on gdecide
    fdec = {f: S.book_decisions("gdecide", f) for f in S.BOOK_FORMS}
    forms = [f for f in S.BOOK_FORMS if all(i in fdec[f] for i in clean)]
    fhit = {f: {i: int(fdec[f][i]["pick"] in meta[i]["good"]) for i in clean} for f in forms}
    res["formulations"] = {}
    for f in forms:
        d = fdec[f]
        res["formulations"][f] = {
            "fair": acc([fhit[f][i] for i in fair]), "all": acc([fhit[f][i] for i in clean]),
            "neutral": neutral(fhit[f], meta, clean, w),
            "book_choice_fair": round(sum(d[i]["pick"] == meta[i]["choice"] for i in fair) / len(fair), 4),
            "picks_longest_all": round(sum(d[i]["pick"] in meta[i]["longest"] for i in clean) / len(clean), 4),
            "picks_longest_fair": round(sum(d[i]["pick"] in meta[i]["longest"] for i in fair) / len(fair), 4),
            "nvo_all": nvo(d, meta, clean), "nvo_fair": nvo(d, meta, fair)}
    res["picks_longest_chance"] = {k: round(sum(len(meta[i]["longest"]) / meta[i]["n"] for i in ids) / len(ids), 4)
                                   for k, ids in (("all", clean), ("fair", fair))}
    res["formulation_pairs"] = {"fair": pairs_table(forms, fhit, fair), "all": pairs_table(forms, fhit, clean)}
    for key, field in (("by_book", "book"), ("by_kind", "kind")):
        res[key] = {}
        groups = defaultdict(list)
        for i in clean:
            groups[meta[i][field]].append(i)
        for g, ids in sorted(groups.items()):
            fids = [i for i in ids if meta[i]["fair"]]
            res[key][g] = {"n_all": len(ids), "n_fair": len(fids), "gate_all": gate(meta, ids),
                           "gate_fair": gate(meta, fids) if fids else None,
                           "all": {f: acc([fhit[f][i] for i in ids]) for f in forms},
                           "fair": {f: acc([fhit[f][i] for i in fids]) for f in forms}}
    best = sorted(forms, key=lambda f: (-res["formulations"][f]["fair"]["acc"], -res["formulations"][f]["all"]["acc"]))
    res["best_two"] = best[:2]

    # ---- models on the best two
    board = [r for r in S.rows(S.OUT / "leaderboard_books.jsonl")]
    res["models"] = {}
    mhit_by_form = {}
    for f in best[:2]:
        mh = {}
        for m in MODELS:
            d = S.book_decisions(m, f)
            if d and all(i in d for i in clean):
                mh[m] = {i: int(d[i]["pick"] in meta[i]["good"]) for i in clean}
                row = next((r for r in board if r["model"] == m and r["formulation"] == f), None)
                res["models"].setdefault(f, {})[m] = {
                    "fair": acc([mh[m][i] for i in fair]), "all": acc([mh[m][i] for i in clean]),
                    "neutral": neutral(mh[m], meta, clean, w),
                    "picks_longest_all": round(sum(d[i]["pick"] in meta[i]["longest"] for i in clean) / len(clean), 4),
                    "nvo_fair": nvo(d, meta, fair), "nvo_all": nvo(d, meta, clean),
                    "refused": sum(d[i]["pick"] < 0 for i in clean),
                    "moves_rev": row["moves_rev"] if row else None, "ms": row["ms_median"] if row else None,
                    "by_book_fair": {b: acc([mh[m][i] for i in fair if meta[i]["book"] == b])
                                     for b in sorted({meta[i]["book"] for i in fair})}}
        mhit_by_form[f] = mh
        if len(mh) > 1:
            res.setdefault("model_pairs", {})[f] = pairs_table(list(mh), mh, fair)

    # ---- books against the templated people/personality cases
    tids, thit, fids, fh, tchance = templated()
    comp = {}
    names = [m for m in MODELS if m in thit and all(m in mhit_by_form.get(f, {}) for f in best[:1])]
    if best and len(names) > 1:
        f0 = best[0]
        comp["models"] = {"templated": separation(names, thit, tids),
                          f"books fair ({f0})": separation(names, mhit_by_form[f0], fair),
                          f"books all ({f0})": separation(names, mhit_by_form[f0], clean)}
        # does the ranking agree?
        ta = comp["models"]["templated"]["acc"]
        ba = comp["models"][f"books fair ({f0})"]["acc"]
        comp["models"]["rank_templated"] = sorted(names, key=lambda m: -ta[m])
        comp["models"]["rank_books_fair"] = sorted(names, key=lambda m: -ba[m])
    comp["formulations"] = {
        "templated lab vs src (gdecide, describe A/B text)": {
            "lab": acc([fh["lab"][i] for i in fids]), "src": acc([fh["new"][i] for i in fids]),
            "paired_src_minus_lab": paired([fh["new"][i] for i in fids], [fh["lab"][i] for i in fids])},
        "books fair a_lab vs b_src (gdecide)": {
            "lab": acc([fhit["a_lab"][i] for i in fair]), "src": acc([fhit["b_src"][i] for i in fair]),
            "paired_src_minus_lab": paired([fhit["b_src"][i] for i in fair], [fhit["a_lab"][i] for i in fair])},
        "books fair, all six formulations": separation(forms, fhit, fair),
        "books all, all six formulations": separation(forms, fhit, clean)}
    comp["templated_chance"] = tchance
    comp["books_fair_chance"] = res["gate"]["fair"]["random_in_good"]
    res["vs_templated"] = comp
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "tables.md").write_text(markdown(res))
    print(json.dumps({k: res[k] for k in ("gate", "best_two")}, indent=1))
    return res


def _p(x):
    return "-" if x is None else f"{100 * x:.1f}"


def _a(v):
    return f"{_p(v['acc'])} [{_p(v['ci'][0])}-{_p(v['ci'][1])}]" if v and v.get("n", 1) else "-"


def _pair_rows(pt):
    out = ["| a | b | a - b, points [95% CI] | only a / only b | McNemar p | Holm p |", "|---|---|---|---|---|---|"]
    for p in pt:
        out.append(f"| {p['a']} | {p['b']} | {_p(p['diff'])} [{_p(p['ci'][0])}, {_p(p['ci'][1])}] |"
                   f" {p['only_new']}/{p['only_lab']} | {p['p_mcnemar']:.2g} | {p['p_holm']:.2g} |")
    return out


def markdown(r):
    g = r["gate"]
    out = ["## The length gate", "",
           "| cases | n | longest option in acceptable set % [95% CI] | random pick % | longest is the book's choice % |"
           " random is the book's % | gate |", "|---|---|---|---|---|---|---|"]
    for k in ("all", "fair", "unfair"):
        x = g[k]
        out.append(f"| {k} | {x['n']} | {_a(x['longest_in_good'])} | {_p(x['random_in_good'])} |"
                   f" {_p(x['longest_is_book'])} | {_p(x['random_is_book'])} | {'passes' if x['passes'] else 'FAILS'} |")
    out += ["", f"Length-neutral mix: weight {r['neutral_w_unfair']} on the unfair cases, the rest on the fair ones"
            " (the longest-option pick scores exactly chance in it).", "",
            "## Formulations (gdecide, fp16)", "",
            "| formulation | acc@1 fair % [95% CI] | acc@1 neutral % [95% CI] | acc@1 all % [95% CI] |"
            " book's choice, fair % | picks a longest option, all % | fair % |"
            " nature-vs-obvious: book % / obvious % / book share of the two % (fair) | same, all cases |",
            "|---|---|---|---|---|---|---|---|---|"]
    for f, v in r["formulations"].items():
        nf, na = v["nvo_fair"], v["nvo_all"]
        out.append(f"| {f} | {_a(v['fair'])} | {_a(v['neutral'])} | {_a(v['all'])} | {_p(v['book_choice_fair'])} |"
                   f" {_p(v['picks_longest_all'])} | {_p(v['picks_longest_fair'])} |"
                   f" {_p(nf['book'])} / {_p(nf['obvious'])} / {_p(nf['book_share_of_two'])} (n {nf['n']}) |"
                   f" {_p(na['book'])} / {_p(na['obvious'])} / {_p(na['book_share_of_two'])} (n {na['n']}) |")
    pl = r["picks_longest_chance"]
    out += ["", f"Chance of picking a longest option: {_p(pl['all'])}% (all), {_p(pl['fair'])}% (fair).", "",
            "### Paired tests between formulations, fair cases", ""] + _pair_rows(r["formulation_pairs"]["fair"])
    out += ["", "### Paired tests between formulations, all clean cases (length bias inside)", ""] \
        + _pair_rows(r["formulation_pairs"]["all"])
    forms = list(r["formulations"])
    for key, name in (("by_book", "book"), ("by_kind", "kind")):
        out += ["", f"### Per {name}: acc@1 % fair (all)", "",
                f"| {name} | n fair (all) | longest pick fair / all % | chance % | " + " | ".join(forms) + " |",
                "|---|---|---|---|" + "---|" * len(forms)]
        for gname, v in r[key].items():
            gf = v["gate_fair"]
            out.append(f"| {gname} | {v['n_fair']} ({v['n_all']}) | "
                       f"{_p(gf['longest_in_good']['acc']) if gf else '-'} / {_p(v['gate_all']['longest_in_good']['acc'])} |"
                       f" {_p(v['gate_all']['random_in_good'])} | "
                       + " | ".join(f"{_p(v['fair'][f]['acc'])} ({_p(v['all'][f]['acc'])})" for f in forms) + " |")
    out += ["", f"## Models on the best two formulations ({', '.join(r['best_two'])})", ""]
    for f, ms in r.get("models", {}).items():
        out += [f"### {f}", "",
                "| model | acc@1 fair % [95% CI] | acc@1 neutral % [95% CI] | acc@1 all % [95% CI] |"
                " picks a longest option, all % | book over obvious, fair: book % / obvious % | moves when reversed % |"
                " refused | ms | per book, fair % |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m, v in sorted(ms.items(), key=lambda kv: -kv[1]["fair"]["acc"]):
            mv = v["moves_rev"]["moved"] if v.get("moves_rev") else None
            out.append(f"| {m} | {_a(v['fair'])} | {_a(v['neutral'])} | {_a(v['all'])} | {_p(v['picks_longest_all'])} |"
                       f" {_p(v['nvo_fair']['book'])} / {_p(v['nvo_fair']['obvious'])} | {_p(mv)} | {v['refused']} |"
                       f" {v['ms']} | " + ", ".join(f"{b} {_p(x['acc'])}" for b, x in v["by_book_fair"].items()) + " |")
        if r.get("model_pairs", {}).get(f):
            out += ["", f"Paired tests between models ({f}, fair cases):", ""] + _pair_rows(r["model_pairs"][f])
        out.append("")
    c = r["vs_templated"]
    out += ["## Books against the templated people/personality cases", ""]
    if c.get("models"):
        out += ["| cases | n | chance % | " + " | ".join(c["models"]["rank_templated"]) + " | range, points | SD, points |"
                " mean abs z of pairs | pairs apart at Holm p<0.05 |", "|---|---|---|" + "---|" * (len(c["models"]["rank_templated"]) + 4)]
        for k, v in c["models"].items():
            if not isinstance(v, dict):
                continue
            ch = c["templated_chance"] if k == "templated" else (c["books_fair_chance"] if "fair" in k else r["gate"]["all"]["random_in_good"])
            out.append(f"| {k} | {v['n']} | {_p(ch)} | " + " | ".join(_p(v["acc"][m]) for m in c["models"]["rank_templated"])
                       + f" | {_p(v['range'])} | {_p(v['sd'])} | {v['mean_abs_z']} | {v['pairs_holm_05']}/{v['pairs']} |")
        out += ["", f"Rank on templated: {', '.join(c['models']['rank_templated'])}; on books (fair):"
                f" {', '.join(c['models']['rank_books_fair'])}.", ""]
    out += ["| formulations | n | lab % | src % | src - lab, points [95% CI] | McNemar p |", "|---|---|---|---|---|---|"]
    for k, v in c["formulations"].items():
        if "lab" in v:
            p = v["paired_src_minus_lab"]
            out.append(f"| {k} | {v['lab']['n']} | {_a(v['lab'])} | {_a(v['src'])} | {_p(p['diff'])} [{_p(p['ci'][0])}, {_p(p['ci'][1])}]"
                       f" | {p['p_mcnemar']:.2g} |")
    for k, v in c["formulations"].items():
        if "range" in v:
            out.append(f"\n{k}: range {_p(v['range'])} points, SD {_p(v['sd'])}, mean abs z {v['mean_abs_z']},"
                       f" pairs apart at Holm p<0.05: {v['pairs_holm_05']}/{v['pairs']}.")
    return "\n".join(out) + "\n"


# ================================================================ recipe v2
# python3 -m lab.methods.books_report --v2 MODEL [--root DIR]
#     -> lab/results/books_v2/<model>.json and <model>.md (numbers and case ids only)
# Every books_v2 file the model has decided, against ctx_full (the full
# context) on the same cases: the paired difference in acc@1 (exact McNemar,
# Holm over the files) and in the mean grade of the pick (paired normal 95%
# interval). Grouped by family (pilot, person, question, menu, ablation,
# placement, format, amount, persona phrasing, ceiling).
V2_FAMILIES = (("pilot", ("a_lab", "b_src", "c_narrative", "d_traits", "e_bio", "f_past_acts")),
               ("person", ("person_1st", "person_2nd", "person_3rd")),
               ("question", ("q_likely", "q_character", "q_next")),
               ("menu", ("menu_two_level",)),
               ("off", ("rel_off", "dialogue_off")),
               ("ablation", ("no_persona", "no_beliefs", "no_percepts", "no_present", "no_needs", "no_holds",
                             "no_history")),
               ("placement", ("place_persona_first", "place_persona_last", "place_key_last", "place_goal_after")),
               ("format", ("fmt_prose", "fmt_labeled", "fmt_bullets")),
               ("amount", ("amt_minimal", "amt_full", "amt_distract")),
               ("persona phrasing", ("pp_habit", "pp_wants", "pp_never")),
               ("ceiling", ("diag_key_marked", "diag_option_reasons")))


def _paired_mean(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if not n:
        return {"n": 0}
    m = sum(d) / n
    sd = statistics.pstdev(d) if n > 1 else 0.0
    return {"n": n, "diff": round(m, 4), "ci": [round(m - 1.96 * sd / math.sqrt(n), 4),
                                                 round(m + 1.96 * sd / math.sqrt(n), 4)]}


def v2(model, root=None, ref="ctx_full"):
    _, _, q, dd, _ = S._v2paths(root)
    d = dd / model
    names = [n for n in S._v2names(q) if not n.endswith("_rev") and not n.startswith("pairs__")
             and (d / f"{n}.jsonl").exists()]
    if ref not in names:
        raise SystemExit(f"{model} has no decisions on {ref}")

    def load(n):
        reqs = {r["id"].rsplit("#", 1)[0]: r for r in S.rows(q / f"{n}.jsonl")}
        dec = {x["request"].rsplit("#", 1)[0]: x for x in S.rows(d / f"{n}.jsonl")}
        return {i: (int(dec[i]["pick"] in r["good"]),
                    r["grades"][dec[i]["pick"]] if dec[i]["pick"] >= 0 and r.get("grades") else 0.0)
                for i, r in reqs.items() if i in dec}
    base = load(ref)
    res = {"model": model, "ref": ref, "forms": {}}
    for n in names:
        x = load(n)
        ids = sorted(set(x) & set(base))
        if not ids:
            continue
        p = paired([x[i][0] for i in ids], [base[i][0] for i in ids])
        res["forms"][n] = {"n": len(ids), "acc": acc([x[i][0] for i in ids]), "ref_acc": acc([base[i][0] for i in ids]),
                           "grade": round(sum(x[i][1] for i in ids) / len(ids), 4),
                           "ref_grade": round(sum(base[i][1] for i in ids) / len(ids), 4),
                           "acc_vs_ref": p, "grade_vs_ref": _paired_mean([x[i][1] for i in ids], [base[i][1] for i in ids])}
    others = [n for n in res["forms"] if n != ref]
    for n, h in zip(others, holm([res["forms"][n]["acc_vs_ref"].get("p_mcnemar", 1.0) for n in others])):
        res["forms"][n]["acc_vs_ref"]["p_holm"] = h
    out = OUT.parent / "books_v2" if root is None else Path(root) / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model}.json").write_text(json.dumps(res, indent=1) + "\n")
    md = [f"## {model}: every formulation against {ref} (same cases)", "",
          "| family | formulation | n | acc@1 % [95% CI] | minus ref, points [95% CI] | Holm p |"
          " grade of pick | minus ref [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    fam = {f: k for k, fs in V2_FAMILIES for f in fs}
    for n in sorted(res["forms"], key=lambda n: ([k for k, _ in V2_FAMILIES] + ["other"]).index(fam.get(n, "other"))):
        v = res["forms"][n]
        a, g = v["acc_vs_ref"], v["grade_vs_ref"]
        md.append(f"| {fam.get(n, 'other')} | {n} | {v['n']} | {_a(v['acc'])} | {_p(a.get('diff'))} "
                  f"[{_p(a['ci'][0]) if a.get('ci') else '-'}, {_p(a['ci'][1]) if a.get('ci') else '-'}] |"
                  f" {format(a['p_holm'], '.2g') if n != ref and 'p_holm' in a else '-'} | {v['grade']:.2f} | {g['diff']:+.2f} "
                  f"[{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}] |")
    (out / f"{model}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


if __name__ == "__main__":
    import sys as _sys
    if "--v2" in _sys.argv:
        _a2 = [x for x in _sys.argv[1:] if x != "--v2"]
        _root = _a2[_a2.index("--root") + 1] if "--root" in _a2 else None
        v2(_a2[0], _root)
    else:
        main()
