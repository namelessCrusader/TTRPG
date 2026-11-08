"""The lab describe against the src describe, judged on LABELED cases (wave 10, LAB10;
again in wave 11, D11, with the v2 spelling).

    python3 tools/export_describe.py                       # the src text (system python, imports src)
    python3 -m lab.methods.describe_ab build               # the request files, in OUT/q
    (score them: lab.methods.score_requests gdecide:all OUT/q/lab.jsonl OUT/d/lab.jsonl --also ...)
    python3 -m lab.methods.describe_ab report              # OUT/numbers.json and OUT/report.md

The user decided (2026-10-10): a describe is judged by how often the model's
pick is in a case's ACCEPTABLE set (the `good` field of lab/methods/cases_*),
not by agreement with the rules' top pick (M9's stand-in).

THE TWO TEXTS of one case ask the same question with the same options in the
same order; only the situation differs:
    lab   the case's own `situation`, as the lab wrote it
    new   the same facts in src/mind/describe.py's words (mark1-mind-v2
          since wave 11; wave 10 judged v1), made by tools/export_describe.py
          (lab/ never imports src/)

WAVE 10 AGAINST WAVE 11. v1 began each text with "Tom holds nothing." and
v2 says what a character holds only when it holds something (the user,
2026-10-16). Wave 10's files are kept in OUT/w10 (its q, d, numbers.json
and report.md). The lab text is the same in both waves, so its decisions
are the same. The report also sets the new text of this wave against wave
10's new text, case by case (OUT/w10/d/new.jsonl).

WHICH CASES. Every case that the export could render AND whose lab text is the
lab's usual wording for its set. Left out are the lab's wording experiments,
whose lab text was changed on purpose (and whose src text would be the same as
the usual case's):
    ablate    verb, object, blank (the need cut out of the lab text)
    present   the ctx_* values (other ways to write the context)
    state2    missing_only, missing_first (other wordings of the state)
Kept: ablate full and slot, present default and opt_* (only the options or the
question change), state2 as_now. A question that two sets share (the same lab
text, question and options) is counted once overall, in the first set.

POSITION BIAS. A sample of the cases (every PERM_EVERY-th, so every set is in
it) is asked again in PERMS other orders of the options: reversed, then
shuffles seeded by the case id, with both texts. A pick "moves" when the
option it names changes.

THE NUMBERS. accuracy@1 for each text, per set, per (set, factor, value) cell
and overall, with Wilson 95% intervals; the paired difference (new - lab) with
a 95% interval and an exact McNemar test on the cases where the two texts
disagree; how often the pick moves under reordering, per text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE.parent / "results" / "describe_ab"
SETS = ("things", "places", "people", "personality", "personality2", "state", "state2", "creative",
        "lure", "pipe", "ablate", "present")
PERMS = 2
DIAG_SETS = ("people", "personality", "personality2")
W10 = "w10"     # wave 10's files, under OUT
PERM_EVERY = 4


def digest(situation, question, options):
    """src/mind/record.py `digest`, written out again (the isolation rule)."""
    data = json.dumps([situation, question, list(options)], ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def write(path, rs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rs))


def usual(kind, c):
    """True when the lab text of case `c` is the lab's usual wording for its set."""
    v = str(c.get("value"))
    if kind == "ablate":
        return v in ("full", "slot")
    if kind == "present":
        return v == "default" or v.startswith("opt_")
    if kind == "state2":
        return v == "as_now"
    return True


def cases(out=OUT):
    """The labeled cases of the A/B, each with its src text: [(case, set, new_text)],
    each question once, and the counts of what was left out."""
    src = rows(Path(out) / "src_describe.jsonl")
    if not src or not src[0].get("header"):
        raise SystemExit(f"no {out}/src_describe.jsonl: run python3 tools/export_describe.py first")
    new = {r["id"]: r["situation"] for r in src[1:]}
    got, seen, left = [], set(), defaultdict(int)
    for kind in SETS:
        for c in rows(HERE / f"cases_{kind}.jsonl"):
            if c["id"] not in new:
                left[f"{kind}: not rendered"] += 1
            elif not usual(kind, c):
                left[f"{kind}: lab wording experiment ({c.get('value')})"] += 1
            elif not c["good"]:
                left[f"{kind}: no acceptable set"] += 1
            else:
                k = digest(c["situation"], c["question"], c["options"])
                if k in seen:
                    left[f"{kind}: the same question as an earlier set"] += 1
                    continue
                seen.add(k)
                got.append((c, kind, new[c["id"]]))
    return got, dict(left), src[0]


def perms(c, k=PERMS):
    """Up to `k` other orders of the options of `c`: reversed, then shuffles."""
    n = len(c["options"])
    ident = list(range(n))
    got = [ident[::-1]] if n > 1 else []
    rng = random.Random(c["id"])
    tries = 0
    while len(got) < k and tries < 200 and n > 2:
        o = ident[:]
        rng.shuffle(o)
        tries += 1
        if o != ident and o not in got:
            got.append(o)
    return got


def request(c, situation, tag, order=None):
    opts = c["options"] if order is None else [c["options"][i] for i in order]
    r = {"id": c["id"] + tag, "situation": situation, "question": c["question"], "options": opts,
         "digest": digest(situation, c["question"], opts)}
    if order is not None:
        r["perm"] = order
    return r


def build(out=OUT):
    got, left, head = cases(out)
    q = Path(out) / "q"
    for name, pick in (("lab", lambda c, s: c["situation"]), ("new", lambda c, s: s)):
        write(q / f"{name}.jsonl", [request(c, pick(c, s), f"#{name}") for c, _, s in got])
        pr = []
        for i, (c, _, s) in enumerate(got):
            if i % PERM_EVERY == 0:
                pr += [request(c, pick(c, s), f"#{name}#p{k}", o) for k, o in enumerate(perms(c), 1)]
        write(q / f"{name}_perm.jsonl", pr)
        print(f"{name}: {len(got)} cases, {len(pr)} reorderings")
    # (wave 10 wrote a diagnostic here, the v1 text without "Tom holds
    # nothing."; v2 is that text, so it is gone. Its rows are in OUT/w10.)
    (q / "diag_noholds.jsonl").unlink(missing_ok=True)
    write(q / "manifest.jsonl", [{"cases": len(got), "left_out": left, "src": head}]
          + [{"id": c["id"], "set": k, "factor": c.get("factor"), "value": c.get("value"),
              "good": c["good"], "n": len(c["options"])} for c, k, _ in got])
    for k, v in sorted(left.items()):
        print(f"  left out, {k}: {v}")


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
    """The exact two-sided McNemar p: b cases only the new text got right, c only the lab's."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a_new, a_lab):
    """The paired difference new - lab of two lists of 0/1, its 95% interval
    (normal, paired) and the McNemar p."""
    n = len(a_new)
    b = sum(1 for x, y in zip(a_new, a_lab) if x and not y)
    c = sum(1 for x, y in zip(a_new, a_lab) if y and not x)
    if n == 0:
        return {"n": 0}
    d = (b - c) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0)) / n
    return {"n": n, "diff": round(d, 4), "ci": [round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)],
            "only_new": b, "only_lab": c, "p_mcnemar": mcnemar(b, c)}


def acc(xs):
    return {"n": len(xs), "acc": round(sum(xs) / len(xs), 4) if xs else None, "ci": wilson(sum(xs), len(xs))}


def decisions(path):
    return {d["request"]: d for d in rows(path)}


def report(out=OUT, d_dir=None, names=("lab", "new")):
    out = Path(out)
    d_dir = Path(d_dir) if d_dir else out / "d"
    man = rows(out / "q" / "manifest.jsonl")
    head, meta = man[0], {m["id"]: m for m in man[1:]}
    dec = {n: decisions(d_dir / f"{n}.jsonl") for n in names}
    hit = {n: {} for n in names}
    for i, m in meta.items():
        for n in names:
            d = dec[n].get(f"{i}#{n}")
            if d is not None:
                hit[n][i] = int(d["pick"] in m["good"])
    both = [i for i in meta if all(i in hit[n] for n in names)]
    res = {"cases": head["cases"], "scored": len(both), "left_out": head["left_out"], "src": head["src"]}

    def block(ids):
        r = {n: acc([hit[n][i] for i in ids]) for n in names}
        r["paired"] = paired([hit["new"][i] for i in ids], [hit["lab"][i] for i in ids])
        r["chance"] = round(sum(len(meta[i]["good"]) / meta[i]["n"] for i in ids) / len(ids), 4) if ids else None
        return r

    res["overall"] = block(both)
    # two kinds win for the new text by its rules, not by its words: a lure
    # sentence it never writes, and a state case's missing thing it says
    res["overall_without_lure"] = block([i for i in both if meta[i]["set"] != "lure"])
    res["overall_without_lure_or_state"] = block([i for i in both if meta[i]["set"] not in ("lure", "state")])
    by_set, by_cell = defaultdict(list), defaultdict(list)
    for i in both:
        m = meta[i]
        by_set[m["set"]].append(i)
        by_cell[f"{m['set']}/{m['factor']}={m['value']}"].append(i)
    res["sets"] = {k: block(v) for k, v in by_set.items()}
    res["cells"] = {k: block(v) for k, v in sorted(by_cell.items())}
    # who drives the difference: the net cases each cell adds to new - lab
    res["drivers"] = sorted(([k, v["paired"]["only_new"] - v["paired"]["only_lab"], v["paired"]["n"]]
                             for k, v in res["cells"].items()), key=lambda x: x[1])
    diag = {k[:-len("#diag")]: int(d["pick"] in meta[k[:-len("#diag")]]["good"])
            for k, d in decisions(d_dir / "diag_noholds.jsonl").items()}
    w10_dec = decisions(out / W10 / "d" / "new.jsonl")
    w10 = {k[:-len("#new")]: int(d["pick"] in meta[k[:-len("#new")]]["good"])
           for k, d in w10_dec.items() if k[:-len("#new")] in meta}
    if w10 and head["src"]["spelling"] != rows(out / W10 / "src_describe.jsonl")[0]["spelling"]:
        # this wave's new text against wave 10's, on the same cases and options
        ids_all = [i for i in both if i in w10]
        res["vs_w10"] = {"spelling": rows(out / W10 / "src_describe.jsonl")[0]["spelling"]}
        for k, ids in [("overall", ids_all)] + [(k, [i for i in ids_all if meta[i]["set"] == k])
                                                 for k in sorted(by_set)]:
            res["vs_w10"][k] = {"lab": acc([w10[i] for i in ids]), "new": acc([hit["new"][i] for i in ids]),
                                "paired": paired([hit["new"][i] for i in ids], [w10[i] for i in ids]),
                                "chance": round(sum(len(meta[i]["good"]) / meta[i]["n"] for i in ids)
                                                / len(ids), 4)}
        res["vs_w10_pick_same"] = round(sum(w10_dec[f"{i}#new"]["pick"] == dec["new"][f"{i}#new"]["pick"]
                                            for i in ids_all) / len(ids_all), 4)
    if diag:
        res["diag"] = {}
        for k in DIAG_SETS:
            ids = [i for i in both if meta[i]["set"] == k and i in diag]
            res["diag"][f"{k}: lab vs new without holds"] = {
                "lab": acc([hit["lab"][i] for i in ids]), "new": acc([diag[i] for i in ids]),
                "paired": paired([diag[i] for i in ids], [hit["lab"][i] for i in ids]),
                "chance": round(sum(len(meta[i]["good"]) / meta[i]["n"] for i in ids) / len(ids), 4)}
    res["position"] = {n: position(out, d_dir, n, meta) for n in names}
    out.joinpath("numbers.json").write_text(json.dumps(res, indent=1) + "\n")
    out.joinpath("report.md").write_text(markdown(res))
    print(markdown(res))
    return res


def position(out, d_dir, name, meta):
    """How often the pick moves when the options are reordered, and how often it
    is still acceptable. Picks are mapped back to the canonical order."""
    q = {r["id"]: r for r in rows(Path(out) / "q" / f"{name}_perm.jsonl")}
    base = decisions(Path(d_dir) / f"{name}.jsonl")
    perm = decisions(Path(d_dir) / f"{name}_perm.jsonl")
    moved, ok, ok_base, first, last, n_slots = 0, 0, 0, 0, 0, 0
    n = 0
    for rid, r in q.items():
        d = perm.get(rid)
        cid = rid.split(f"#{name}#p")[0]
        b = base.get(f"{cid}#{name}")
        if d is None or b is None:
            continue
        n += 1
        canon = r["perm"][d["pick"]]
        moved += canon != b["pick"]
        ok += canon in meta[cid]["good"]
        ok_base += b["pick"] in meta[cid]["good"]
        first += d["pick"] == 0
        last += d["pick"] == len(r["options"]) - 1
        n_slots += 1 / len(r["options"])
    if not n:
        return {"n": 0}
    return {"n": n, "moved": round(moved / n, 4), "moved_ci": wilson(moved, n),
            "acc_reordered": round(ok / n, 4), "acc_canonical_same_cases": round(ok_base / n, 4),
            "first_slot": round(first / n, 4), "last_slot": round(last / n, 4),
            "slot_chance": round(n_slots / n, 4)}


def _f(x):
    return "-" if x is None else f"{100 * x:.1f}"


def markdown(r):
    def line(k, v):
        p = v["paired"]
        return (f"| {k} | {v['lab']['n']} | {_f(v['lab']['acc'])} [{_f(v['lab']['ci'][0])}-{_f(v['lab']['ci'][1])}]"
                f" | {_f(v['new']['acc'])} [{_f(v['new']['ci'][0])}-{_f(v['new']['ci'][1])}]"
                f" | {_f(p.get('diff'))} [{_f(p['ci'][0])}, {_f(p['ci'][1])}] | {p['only_new']}/{p['only_lab']}"
                f" | {p['p_mcnemar']:.2g} | {_f(v['chance'])} |")
    hd = ("| set | n | lab acc@1 % [95% CI] | new acc@1 % [95% CI] | new - lab, points [95% CI] |"
          " only new / only lab | McNemar p | chance % |\n|---|---|---|---|---|---|---|---|")
    out = ["# Describe A/B on labeled cases (gdecide:all)\n",
           "Model: GLiNER2.5-Decide (gdecide:all), fp16 on the GPU with the direct load (the wave O default), one"
           " request at a time; the same model, dtype and options for both texts.\n",
           f"src spelling {r['src']['spelling']}, describe sha256 {r['src']['describe_sha256'][:12]}."
           f" {r['scored']} of {r['cases']} cases scored with both texts.\n", hd,
           line("**overall**", r["overall"])]
    out += [line("overall without lure", r["overall_without_lure"]),
            line("overall without lure or state", r["overall_without_lure_or_state"])]
    out += [line(k, v) for k, v in sorted(r["sets"].items())]
    if r.get("vs_w10"):
        v = r["vs_w10"]
        out += ["", f"## This wave's new text ({r['src']['spelling']}) against wave 10's ({v['spelling']})", "",
                "The same cases and options; only the src text changed. In the table, the \"lab\" columns"
                f" are wave 10's new text. The pick is the same on {_f(r['vs_w10_pick_same'])}% of the cases.",
                "", hd.replace("lab acc@1", "wave 10 acc@1").replace("new - lab", "new - wave 10")
                .replace("only lab", "only wave 10")]
        out += [line(k, x) for k, x in v.items() if k != "spelling"]
    if r.get("diag"):
        out += ["", "## Diagnostic: the new text without its \"holds nothing\" sentence (people, personality)", "",
                hd] + [line(k, v) for k, v in r["diag"].items()]
    out += ["", "## Cells (set/factor=value)", "", hd] + [line(k, v) for k, v in r["cells"].items()]
    out += ["", "## Position bias (the same cases in other orders)", "",
            "| text | reorderings | pick moves % [95% CI] | acc@1 reordered % | acc@1 canonical, same cases % |"
            " first slot % | last slot % | chance of a slot % |", "|---|---|---|---|---|---|---|---|"]
    for n, p in r["position"].items():
        if p.get("n"):
            out.append(f"| {n} | {p['n']} | {_f(p['moved'])} [{_f(p['moved_ci'][0])}-{_f(p['moved_ci'][1])}]"
                       f" | {_f(p['acc_reordered'])} | {_f(p['acc_canonical_same_cases'])}"
                       f" | {_f(p['first_slot'])} | {_f(p['last_slot'])} | {_f(p['slot_chance'])} |")
    out += ["", "## Cases left out", ""] + [f"- {k}: {v}" for k, v in sorted(r["left_out"].items())]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m lab.methods.describe_ab")
    ap.add_argument("what", choices=("build", "report"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.what == "build":
        build(a.out)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
