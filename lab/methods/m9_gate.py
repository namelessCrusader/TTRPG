"""The offline model gate, step M9 (docs/sim-mind-design.md 3.5 and 4.4).

    python3 -m lab.methods.m9_gate build REQ_DIR WORK_DIR     # the request files to score
    (score them: lab.methods.score_requests gdecide:all ... --also ...)
    python3 -m lab.methods.m9_gate report REQ_DIR WORK_DIR [--mem-gpu F --mem-cpu F --free-mib N]

REQ_DIR holds the case rows a sim run recorded (src/mind/record.py), one
file for the inn (rooms.jsonl) and one for the voxel scenes (voxel.jsonl),
with the fields M9's recorder adds: `lab_situation` (the lab's own scene
text of the same moment), the rule numbers of the full menu, and the rules'
draw (`rules_p`, `rules_argmax`, `gate_top`, `gated`).

BUILD writes the questions the model is asked, in WORK_DIR/q:
    rooms.jsonl, voxel.jsonl   the requests as they are: the new describe
                    (mark1-mind-v1), the cut menu in canonical order
    rooms_lab.jsonl the same menus with the lab's describe (lab/world2)
    rooms_full.jsonl  the FULL menu (every legal option before the pre-rank
                    cut), only for the requests the cut changed
    *_perm.jsonl    each menu of two or more options in up to 4 other orders:
                    reversed, then shuffles seeded by the request id, each
                    order once and never the canonical one. `perm[k]` is the
                    canonical index of the option at place k.
A derived row has an id of its own ("<id>#lab", "#full", "#p1") and its own
digest, so its decision is never taken for another question's.

REPORT reads the decisions (WORK_DIR/d/<the same names>) and gives the five
numbers of M9 and the verdict, as JSON (WORK_DIR/numbers.json) and text:
    (a) the new describe against the lab describe on the same inn menus: how
        often the model picks the rules' argmax, and layer-3-style signals of
        its pick (lab/judge/signals.py: unexpected, in character, harm/help)
    (b) recall: how often the model's pick over the FULL menu is still on the
        cut menu; and the surprise of the model's pick with and without the cut
    (c) position bias: how often the pick moves when the options are reordered
    (d) model-rules agreement on the gated decisions (Gate(0.7): the rules'
        top probability is 0.7 or more, so the model would not be asked)
    (e) memory, from the scorer's `memory:` lines, against free RAM (`fits`)
THE GATE (design, M9 Verify): M11 may go on only if recall >= 95%, the new
describe >= the lab describe, and the server fits in free RAM minus 1 GB.

Like the scorer, this file imports nothing from src/ (lab/README.md): the
digest and the Combined chooser's arithmetic are written out again here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

PERMS = 4
TAU, BETA, T_RULES, GUARD = 0.5, 1.0, 0.2, 0.5          # src/mind/choosers.py, core.py
IN_CHAR_P, IN_CHAR_FIT = 0.05, math.log(2)             # lab/judge/signals.py
# lab/judge/signals.py HARM and HELP, by the first words of an option's text
HARM = ("strike", "shoot", "throw", "shove", "grapple", "disarm", "threaten", "insult", "hold",
        "steal", "accuse")
HELP = ("give", "help", "bandage", "press", "carry", "praise", "offer", "wake", "warn", "defend")
RECALL_MIN = 0.95
SPARE_MIB = 1024


def digest(situation, question, options):
    """src/mind/record.py `digest`, written out again (the isolation rule)."""
    data = json.dumps([situation, question, list(options)], ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def write(path, rs):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rs))


def _q(base, suffix, situation, options, **more):
    return {"id": base["id"] + suffix, "situation": situation, "question": base["question"],
            "options": list(options), "digest": digest(situation, base["question"], options), **more}


def perms(c):
    """Up to PERMS orders of the menu of `c`, none of them canonical."""
    n = len(c["options"])
    if n < 2:
        return []
    ident = list(range(n))
    got = [ident[::-1]]
    rng = random.Random(c["id"])
    tries = 0
    while len(got) < PERMS and tries < 200:
        o = ident[:]
        rng.shuffle(o)
        tries += 1
        if o != ident and o not in got:
            got.append(o)
    return got


def build(req_dir, work):
    q = Path(work) / "q"
    q.mkdir(parents=True, exist_ok=True)
    for name in ("rooms", "voxel"):
        cs = rows(Path(req_dir) / f"{name}.jsonl")
        write(q / f"{name}.jsonl", [_q(c, "", c["situation"], c["options"]) | {"id": c["id"]} for c in cs])
        pr = []
        for c in cs:
            for k, o in enumerate(perms(c), 1):
                pr.append(_q(c, f"#p{k}", c["situation"], [c["options"][i] for i in o], perm=o))
        write(q / f"{name}_perm.jsonl", pr)
        if name == "rooms":
            write(q / "rooms_lab.jsonl", [_q(c, "#lab", c["lab_situation"], c["options"]) for c in cs])
            write(q / "rooms_full.jsonl", [_q(c, "#full", c["situation"], c["full_options"])
                                           for c in cs if c["cut"]])
        print(f"{name}: {len(cs)} requests, {len(pr)} reorderings"
              + (f", {sum(bool(c['cut']) for c in cs)} cut" if name == "rooms" else ""))


# ---------------------------------------------------------------- the arithmetic
def softmax(z, t=1.0):
    m = max(z)
    e = [math.exp((x - m) / t) for x in z]
    s = sum(e)
    return [x / s for x in e]


def zscore(scores):
    """src/mind/choosers.py `zscore` (the lab's policy._z)."""
    if all(0 <= x <= 1 for x in scores):
        scores = [math.log(max(x, 1e-9)) for x in scores]
    mu = sum(scores) / len(scores)
    sd = math.sqrt(sum((x - mu) ** 2 for x in scores) / len(scores)) or 1.0
    return [(x - mu) / sd for x in scores]


def combined(scores, prior, prior_avg, habit):
    """(p_char, p_avg) of the Combined chooser (choosers.Combined, guard 0.5):
    the draws a featured run would log for this decision."""
    z = zscore(scores)
    view = [x / TAU for x in z]
    floor = max(view) + math.log(GUARD)
    pc = softmax([v + (BETA * (a - b) / T_RULES + h / T_RULES if v >= floor else h / T_RULES)
                  for v, a, b, h in zip(view, prior, prior_avg, habit)])
    pa = softmax([v + h / T_RULES for v, h in zip(view, habit)])
    return pc, pa


def signals(pick, scores, prior, prior_avg, habit):
    """Layer-3-style signals of one pick (lab/judge/signals.py `surprise`):
    unexpected = -log p_avg of the pick, under the Combined draws and under
    the rules' own average draw; in character as the lab says it."""
    pc, pa = combined(scores, prior, prior_avg, habit)
    ra = softmax([a + h for a, h in zip(prior_avg, habit)], T_RULES)
    fit = math.log(max(pc[pick], 1e-9)) - math.log(max(pa[pick], 1e-9))
    return {"unexpected": -math.log(max(pa[pick], 1e-9)),
            "unexpected_rules": -math.log(max(ra[pick], 1e-9)),
            "in_char": pc[pick] >= IN_CHAR_P and fit >= IN_CHAR_FIT}


def _score(c):
    """The pre-rank's score of each option of the full menu (src/mind/core.py menu)."""
    return [max(a, b) + h for a, b, h in zip(c["full_prior"], c["full_prior_avg"], c["full_habit"])]


def kind(text):
    w = text.split(" ", 1)[0]
    return "harm" if w in HARM else "help" if w in HELP else "other"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def share(xs):
    xs = list(xs)
    return {"n": len(xs), "k": sum(map(bool, xs)), "rate": mean(map(bool, xs))}


def wilson(k, n, z=1.96):
    """The 95% Wilson interval of k in n."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def fits(m, room):
    """Does a scorer run whose `memory:` line is `m` fit in `room` MiB? Two
    ways to show it, either one is enough:
      by_peak  the worst peak seen (the process's RSS, or its cgroup's charge,
               page cache too) is at most `room`;
      by_cap   the run ended normally under a hard cap of at most `room`
               (cap.sh: no swap). The kernel drops file pages (the weights'
               file, the libraries) before it kills, so a peak above the cap
               was pages it could give back.
    Also, as a diagnostic: the peak of anonymous memory, which no cap can
    take back."""
    peak = max(m.get("rss_peak_mib") or 0, m.get("cgroup_peak_mib") or 0)
    cap = m.get("cgroup_max_mib")
    anon = max(m.get("rss_anon_peak_mib") or 0, m.get("cgroup_anon_peak_mib") or 0) or None
    out = {"room_mib": room, "peak_mib": peak, "by_peak": peak <= room, "cap_mib": cap,
           "by_cap": cap is not None and cap <= room, "anon_peak_mib": anon,
           "anon_fits": None if anon is None else anon <= room}
    out["fits"] = out["by_peak"] or out["by_cap"]
    return out


def _decisions(work, name):
    return {d["request"]: d for d in rows(Path(work) / "d" / f"{name}.jsonl")}


def report(req_dir, work, mem_gpu=None, mem_cpu=None, free_mib=None):
    out = {}
    room = rows(Path(req_dir) / "rooms.jsonl")
    vox = rows(Path(req_dir) / "voxel.jsonl")
    new = {**_decisions(work, "rooms"), **_decisions(work, "voxel")}
    lab = _decisions(work, "rooms_lab")
    full = _decisions(work, "rooms_full")
    perm = {**_decisions(work, "rooms_perm"), **_decisions(work, "voxel_perm")}
    qperm = {r["id"]: r for r in rows(Path(work) / "q" / "rooms_perm.jsonl")
             + rows(Path(work) / "q" / "voxel_perm.jsonl")}
    missing = {"new": sum(c["id"] not in new for c in room + vox),
               "lab": sum(c["id"] + "#lab" not in lab for c in room),
               "full": sum(c["id"] + "#full" not in full for c in room if c["cut"]),
               "perm": sum(i not in perm for i in qperm)}
    out["missing"] = missing

    # (a) the new describe against the lab describe, on the same inn menus
    both = [c for c in room if c["id"] in new and c["id"] + "#lab" in lab and len(c["options"]) > 1]
    a = {"n": len(both)}
    for tag, dec in (("new", lambda c: new[c["id"]]), ("lab", lambda c: lab[c["id"] + "#lab"])):
        sig = [signals(dec(c)["pick"], dec(c)["scores"], c["prior"], c["prior_avg"], c["habit"])
               for c in both]
        a[tag] = {"agree_argmax": share(dec(c)["pick"] == c["rules_argmax"] for c in both),
                  "mass_on_argmax": mean(dec(c)["scores"][c["rules_argmax"]] for c in both),
                  "rules_p_of_pick": mean(c["rules_p"][dec(c)["pick"]] for c in both),
                  "unexpected": mean(s["unexpected"] for s in sig),
                  "unexpected_rules": mean(s["unexpected_rules"] for s in sig),
                  "in_character": share(s["in_char"] for s in sig),
                  "harm": share(kind(c["options"][dec(c)["pick"]]) == "harm" for c in both),
                  "help": share(kind(c["options"][dec(c)["pick"]]) == "help" for c in both),
                  "take": share(c["options"][dec(c)["pick"]].startswith("take ") for c in both)}
        a[tag]["agree_argmax"]["ci95"] = wilson(a[tag]["agree_argmax"]["k"], len(both))
    a["rules_argmax"] = {"harm": share(kind(c["options"][c["rules_argmax"]]) == "harm" for c in both),
                         "help": share(kind(c["options"][c["rules_argmax"]]) == "help" for c in both),
                         "take": share(c["options"][c["rules_argmax"]].startswith("take ") for c in both)}
    nw = [new[c["id"]]["pick"] == c["rules_argmax"] for c in both]
    lb = [lab[c["id"] + "#lab"]["pick"] == c["rules_argmax"] for c in both]
    a["paired"] = {"new_only": sum(x and not y for x, y in zip(nw, lb)),
                   "lab_only": sum(y and not x for x, y in zip(nw, lb)),
                   "same_pick": share(new[c["id"]]["pick"] == lab[c["id"] + "#lab"]["pick"] for c in both)}
    # an exact sign test of the discordant pairs (McNemar)
    b_, c_ = a["paired"]["new_only"], a["paired"]["lab_only"]
    nd = b_ + c_
    tail = sum(math.comb(nd, i) for i in range(0, min(b_, c_) + 1)) / 2 ** nd if nd else 1.0
    a["paired"]["p_two_sided"] = min(1.0, 2 * tail)
    # null until both describes have decisions: 0 >= 0 is not a result
    a["new_ge_lab"] = (a["new"]["agree_argmax"]["k"] >= a["lab"]["agree_argmax"]["k"]) if both else None
    out["a_describe"] = a

    # (b) recall of the full-menu pick, and the surprise with and without the cut
    cut = [c for c in room if c["cut"] and c["id"] + "#full" in full and c["id"] in new]
    kept = []
    s_full, s_cut = [], []
    for c in cut:
        f = full[c["id"] + "#full"]
        text = c["full_options"][f["pick"]]
        kept.append(text in c["options"])
        s_full.append(signals(f["pick"], f["scores"], c["full_prior"], c["full_prior_avg"], c["full_habit"]))
        d = new[c["id"]]
        s_cut.append(signals(d["pick"], d["scores"], c["prior"], c["prior_avg"], c["habit"]))
    n_all = len([c for c in room + vox if c["id"] in new])
    k_all = sum(kept) + (n_all - len(cut))
    b = {"cut_requests": share(kept), "all_requests": {"n": n_all, "k": k_all,
                                                       "rate": k_all / n_all if n_all else None}}
    b["cut_requests"]["ci95"] = wilson(sum(kept), len(kept))
    b["full_menu"] = {"median_options": sorted(len(c["full_options"]) for c in cut)[len(cut) // 2]
                      if cut else None, "max_options": max((len(c["full_options"]) for c in cut), default=None)}
    # the rules' own mass that the cut took away (the pre-rank's softmax of max(prior, avg) + habit)
    b["rules_mass_cut"] = mean(sum(p for p, t in zip(softmax(_score(c), T_RULES), c["full_options"])
                                   if t not in c["options"]) for c in cut)
    # where the cut put the full-menu picks it dropped: their place in the rules' order
    lost = []
    for c, k in zip(cut, kept):
        if not k:
            f = full[c["id"] + "#full"]
            sc = _score(c)
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            lost.append({"id": c["id"], "pick": c["full_options"][f["pick"]],
                         "rules_rank": order.index(f["pick"]) + 1, "of": len(sc),
                         "p_model": round(f["scores"][f["pick"]], 4)})
    b["lost_examples"] = lost[:12]
    b["surprise"] = {
        "without_cut": {"unexpected": mean(s["unexpected"] for s in s_full),
                        "unexpected_rules": mean(s["unexpected_rules"] for s in s_full),
                        "in_character": share(s["in_char"] for s in s_full)},
        "with_cut": {"unexpected": mean(s["unexpected"] for s in s_cut),
                     "unexpected_rules": mean(s["unexpected_rules"] for s in s_cut),
                     "in_character": share(s["in_char"] for s in s_cut)}}
    b["recall"] = b["cut_requests"]["rate"]
    # while some full-menu decisions are missing: the recall if every one of
    # them were lost, and if every one were kept (the gate can be read early)
    n_cut = sum(bool(c["cut"]) for c in room)
    b["recall_bounds"] = [sum(kept) / n_cut, (sum(kept) + n_cut - len(cut)) / n_cut] if n_cut else None
    out["b_recall"] = b

    # (c) the canonical order against 4 reorderings
    c_ = {}
    for tag, cs in (("rooms", room), ("voxel", vox)):
        flips, anyflip, first, firstp, n = [], [], [], [], 0
        for c in cs:
            if c["id"] not in new or len(c["options"]) < 2:
                continue
            base = new[c["id"]]["pick"]
            mine = [(qperm[k], perm[k]) for k in (f"{c['id']}#p{i}" for i in range(1, PERMS + 1))
                    if k in perm]
            if not mine:
                continue
            n += 1
            moved = [q["perm"][d["pick"]] != base for q, d in mine]
            flips += moved
            anyflip.append(any(moved))
            first.append(base == 0)
            firstp += [d["pick"] == 0 for _, d in mine]
            # also: the share of the menus whose canonical pick is the rules' argmax
        c_[tag] = {"requests": n, "orders": len(flips), "pick_moves": share(flips),
                   "any_of_4_moves": share(anyflip), "picks_first_canonical": share(first),
                   "picks_first_reordered": share(firstp),
                   "first_by_chance": mean(1 / len(c["options"]) for c in cs
                                           if c["id"] in new and len(c["options"]) > 1)}
    out["c_order"] = c_

    # (d) model-rules agreement on gated decisions
    d_ = {}
    for tag, cs in (("rooms", room), ("voxel", vox), ("all", room + vox)):
        cs = [c for c in cs if c["id"] in new and len(c["options"]) > 1]
        g = [c for c in cs if c["gated"]]
        u = [c for c in cs if not c["gated"]]
        d_[tag] = {"gated": share(new[c["id"]]["pick"] == c["rules_argmax"] for c in g),
                   "not_gated": share(new[c["id"]]["pick"] == c["rules_argmax"] for c in u),
                   "gated_share": len(g) / len(cs) if cs else None}
        d_[tag]["gated"]["ci95"] = wilson(d_[tag]["gated"]["k"], len(g))
    out["d_gated"] = d_

    # (e) memory
    e = {"gpu": json.loads(Path(mem_gpu).read_text()) if mem_gpu else None,
         "cpu": json.loads(Path(mem_cpu).read_text()) if mem_cpu else None, "free_mib": free_mib}
    for mode in ("gpu", "cpu"):
        if e[mode] and free_mib is not None:
            e[mode + "_fits"] = fits(e[mode], free_mib - SPARE_MIB)
    out["e_memory"] = e

    fit = (e.get("gpu_fits") or {}).get("fits") or (e.get("cpu_fits") or {}).get("fits")
    out["gate"] = {"recall_ok": b["recall"] is not None and b["recall"] >= RECALL_MIN,
                   "describe_ok": bool(a["new_ge_lab"]), "memory_ok": bool(fit),
                   "complete": not any(missing.values())}
    out["gate"]["go_M11"] = all(out["gate"].values())
    Path(work, "numbers.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lab.methods.m9_gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("req_dir")
    b.add_argument("work")
    r = sub.add_parser("report")
    r.add_argument("req_dir")
    r.add_argument("work")
    r.add_argument("--mem-gpu")
    r.add_argument("--mem-cpu")
    r.add_argument("--free-mib", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.req_dir, a.work)
    else:
        print(json.dumps(report(a.req_dir, a.work, a.mem_gpu, a.mem_cpu, a.free_mib), indent=1))


if __name__ == "__main__":
    main()
