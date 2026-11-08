"""The two-level menu on the MODEL: scope recall, the same act, "something else"
(wave 11, D11; src/mind/scope.py, step M12).

    python3 -m lab.methods.scope_eval next        # the questions the model must answer now: OUT/q/todo.jsonl
    (score them: lab.methods.score_requests gdecide:all OUT/q/todo.jsonl OUT/d/all.jsonl)
    python3 -m lab.methods.scope_eval next        # ... again, until it says 0 to do
    python3 -m lab.methods.scope_eval report      # OUT/numbers.json and OUT/report.md

THE INPUT is OUT/req/*.jsonl: every question that a two-level decision of a
sim run can ask, in one or more scope LAYOUTS (src/mind/record.py `tree`,
written by the recorder of wave 11; lab/ never imports src/). One decision
has one "full" row (the full menu, uncut, as one question) and, for each
layout (`variant`), its pages of scopes ("scope", page 1, 2, ...) and the
act question of each entry ("act", page k, entry j).

THE MODEL'S OWN PATH. The run's chooser was the rules; here the model goes
down the levels on its own: it picks an entry of page 1; when that entry is
"something else" (kind "else"), it picks on the next page; then it picks
an act among the options of the entry it took. A page or an act question
with one choice is not asked (`asked` False): its one choice is taken. So
the questions are scored in rounds: `next` lists the questions on the
model's paths that have no decision yet, the scorer answers them, and
`next` runs again, until nothing is left (at most the depth of the deepest
decision). The requests are keyed by their digest, so a question that two
layouts (or two decisions) share is scored once.

THE NUMBERS, for each layout and each backend:
    recall       the entry the model picks on page 1 holds the option it
                 picks from the full menu (its "something else" holds what
                 it folded). M9's flat pre-rank kept the model's full-menu
                 pick in 71.6% of the cut requests (a menu over 16 options).
    path recall  the entry it ends on (after the pages of "something else")
                 holds its full-menu pick
    same act     its two (or three) steps end on the act it picks from the
                 full menu
    else needed  its full-menu pick is behind "something else" on page 1
    else picked  it picks "something else" on page 1
    wait picked  it picks the entry "wait" on page 1 (wait full: its full-menu
                 pick is wait)
    questions    the mean number of questions it was asked
    chance       recall when page 1 is picked at random (1 / entries)
with Wilson 95% intervals, by menu size (16 options or fewer, more), and
each layout against the first one (paired, McNemar).

Like the scorer, this file imports nothing from src/ (lab/README.md).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "results" / "scope"
ELSE = "else"
CUT = 16            # M9's pre-rank cut: the menus it cut were bigger than this


def rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def write(path, rs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rs))


def load(out=OUT):
    """{decision: {"full": row, variant: {"scope": {page: row}, "act": {(page, entry): row}}}}"""
    got = defaultdict(lambda: defaultdict(lambda: {"scope": {}, "act": {}}))
    for f in sorted((Path(out) / "req").glob("*.jsonl")):
        for r in rows(f):
            d = got[r["decision"]]
            if r["level"] == "full":
                d["full"] = r
            elif r["level"] == "scope":
                d[r["variant"]]["scope"][r["page"]] = r
            else:
                d[r["variant"]]["act"][(r["page"], r["entry"])] = r
    return got


def decided(out=OUT):
    """{digest: decision row}, from OUT/d/all.jsonl."""
    return {d["digest"]: d for d in rows(Path(out) / "d" / "all.jsonl")}


def path(v, dec):
    """The model's path through the levels of one layout `v` of a decision:
    (the page-1 entry, the scope rows and picks on the way, the act row,
    the act text), or (..., None) where a question has no decision yet; the
    row that waits for one is the last of `need`."""
    steps, page, need = [], 1, []
    while True:
        r = v["scope"][page]
        if r["asked"]:
            d = dec.get(r["digest"])
            if d is None:
                return steps, None, [r]
            j = d["pick"]
        else:
            j = 0
        steps.append((r, j))
        if r["kinds"][j] != ELSE:
            break
        page += 1
    a = v["act"][(page, j)]
    if not a["asked"]:
        return steps, (a, a["options"][0]), need
    d = dec.get(a["digest"])
    if d is None:
        return steps, None, [a]
    return steps, (a, a["options"][d["pick"]]), need


def nxt(out=OUT):
    """Write OUT/q/todo.jsonl: the questions on the model's paths with no
    decision yet, each once (by digest; the request id is the digest)."""
    got, dec = load(out), decided(out)
    todo = {}
    for did, d in got.items():
        f = d["full"]
        if f["digest"] not in dec:
            todo[f["digest"]] = f
        for k, v in d.items():
            if k == "full":
                continue
            _, end, need = path(v, dec)
            for r in need:
                todo.setdefault(r["digest"], r)
    write(Path(out) / "q" / "todo.jsonl",
          [{"id": k, "situation": r["situation"], "question": r["question"], "options": r["options"],
            "digest": k} for k, r in todo.items()])
    (Path(out) / "d").mkdir(parents=True, exist_ok=True)       # the scorer appends to d/all.jsonl
    print(f"{len(todo)} to do (decisions {len(got)}, decided questions {len(dec)})")
    return len(todo)


# ---------------------------------------------------------------- the arithmetic
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(c - h, 0.0), 4), round(min(c + h, 1.0), 4)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def rate(xs):
    return {"n": len(xs), "k": sum(xs), "rate": round(sum(xs) / len(xs), 4) if xs else None,
            "ci": wilson(sum(xs), len(xs))}


def measure(d, v, dec):
    """The numbers of one decision in one layout, or None when a question on
    the model's path has no decision yet."""
    f = d["full"]
    fd = dec.get(f["digest"])
    steps, end, _ = path(v, dec)
    if fd is None or end is None:
        return None
    pick = f["options"][fd["pick"]]
    (r1, j1), (rl, jl) = steps[0], steps[-1]
    return {"recall": int(pick in r1["members"][j1]), "path_recall": int(pick in rl["members"][jl]),
            "same": int(end[1] == pick), "else_needed": int(any(k == ELSE and pick in m for k, m in
                                                                zip(r1["kinds"], r1["members"]))),
            "else_picked": int(r1["kinds"][j1] == ELSE), "wait_picked": int(r1["kinds"][j1] == "wait"),
            "wait_full": int(any(k == "wait" and pick in m for k, m in zip(r1["kinds"], r1["members"]))),
            "questions": sum(r["asked"] for r, _ in steps) + int(end[0]["asked"]),
            "chance": 1 / len(r1["options"]), "n": len(f["options"]), "entries": len(r1["options"]),
            "longest": max(len(t) for t in r1["options"])}


def report(out=OUT):
    out = Path(out)
    got, dec = load(out), decided(out)
    variants = sorted({k for d in got.values() for k in d if k != "full"},
                      key=lambda k: (k != "v1", k))
    m = {(did, k): measure(d, d[k], dec) for did, d in got.items() for k in variants if k in d}
    res = {"decisions": len(got), "variants": variants, "missing": sum(x is None for x in m.values()),
           "cut": CUT, "by": {}}
    for backend in sorted({d["full"]["backend"] for d in got.values()}) + ["all"]:
        ids = [i for i, d in got.items() if backend in ("all", d["full"]["backend"])]
        for size, keep in (("all", lambda n: True), (f"<= {CUT}", lambda n: n <= CUT),
                           (f"> {CUT}", lambda n: n > CUT)):
            for k in variants:
                xs = [m[(i, k)] for i in ids if m.get((i, k)) is not None and keep(m[(i, k)]["n"])]
                if not xs:
                    continue
                r = {f: rate([x[f] for x in xs]) for f in ("recall", "path_recall", "same", "else_needed",
                                                            "else_picked", "wait_picked", "wait_full")}
                r["questions"] = round(sum(x["questions"] for x in xs) / len(xs), 3)
                r["chance"] = round(sum(x["chance"] for x in xs) / len(xs), 4)
                r["longest_entry"] = max(x["longest"] for x in xs)
                r["menu_mean"] = round(sum(x["n"] for x in xs) / len(xs), 1)
                if k != variants[0]:
                    pairs = [(m[(i, k)], m[(i, variants[0])]) for i in ids
                             if m.get((i, k)) and m.get((i, variants[0])) and keep(m[(i, k)]["n"])]
                    for f in ("recall", "same"):
                        b = sum(1 for x, y in pairs if x[f] and not y[f])
                        c = sum(1 for x, y in pairs if y[f] and not x[f])
                        r[f"{f}_vs_{variants[0]}"] = {"n": len(pairs), "only_this": b, "only_first": c,
                                                      "diff": round((b - c) / len(pairs), 4),
                                                      "p_mcnemar": mcnemar(b, c)}
                res["by"][f"{backend} | {size} | {k}"] = r
    (out / "numbers.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "report.md").write_text(markdown(res))
    print(markdown(res))
    return res


def _p(x):
    return "-" if x is None else f"{100 * x:.1f}"


def markdown(r):
    out = ["# The two-level menu on the model (gdecide:all)\n",
           f"{r['decisions']} decisions; layouts {', '.join(r['variants'])}; {r['missing']} (decision,"
           " layout) pairs with a question not yet scored.\n",
           "| backend | menu | layout | n | recall % [95% CI] | path recall % | same act % [95% CI] |"
           " else needed % | else picked % | wait picked % (full) | questions | chance % | longest entry |"
           " recall vs first (only this / only first, p) | same vs first |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in r["by"].items():
        b, size, k = key.split(" | ")
        rc, sm = v["recall"], v["same"]
        vs = lambda f: ("-" if f"{f}_vs_{r['variants'][0]}" not in v else
                        "{diff:+.1%} ({only_this}/{only_first}, {p_mcnemar:.2g})".format(
                            **v[f"{f}_vs_{r['variants'][0]}"]).replace("%", " pts"))
        out.append(f"| {b} | {size} | {k} | {rc['n']} | {_p(rc['rate'])} [{_p(rc['ci'][0])}-{_p(rc['ci'][1])}]"
                   f" | {_p(v['path_recall']['rate'])} | {_p(sm['rate'])} [{_p(sm['ci'][0])}-{_p(sm['ci'][1])}]"
                   f" | {_p(v['else_needed']['rate'])} | {_p(v['else_picked']['rate'])}"
                   f" | {_p(v['wait_picked']['rate'])} ({_p(v['wait_full']['rate'])}) | {v['questions']}"
                   f" | {_p(v['chance'])} | {v['longest_entry']} | {vs('recall')} | {vs('same')} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m lab.methods.scope_eval")
    ap.add_argument("what", choices=("next", "report"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.what == "next":
        nxt(a.out)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
