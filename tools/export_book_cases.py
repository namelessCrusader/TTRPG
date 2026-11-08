"""THE BOOK PILOT CASE SET, AND ITS FORMULATIONS (wave 11b).

    python3 tools/export_book_cases.py              # everything under books/derived/
    python3 tools/export_book_cases.py --seed 11    # the seed of the 30-case spot-check sample
    python3 tools/export_book_cases.py --v2         # recipe v2: books/derived/v2/formulations_v2/
                                                    # (see "RECIPE v2" below)

The user's idea (2026-10-10/16): labeled decision cases from the books the
user owns, to answer (1) which model chooses best and (2) which way of posing
the decision gets the right choice best. The writer made a case row and an
answer row per decision (books/derived/CASE_SPEC.md); a blind rater rated
every option of every case "acceptable", "unlikely", "decoy" or "impossible"
without seeing the answer. This tool joins them and writes the case set.

LOCAL ONLY. Everything this tool reads and writes is under books/derived/,
which is git-ignored. This file holds no book text.

THE KEEP RULE. A case is kept when the rater called it answerable AND rated
the book's choice acceptable. Its acceptable set (`good`) is every option the
rater rated acceptable, so the book's choice is always in it. The writer's
own `acceptable` is kept beside it as `writer_acceptable`.

WHAT IT WRITES (all under books/derived/):

    cases_books.jsonl          the kept cases in the lab case format
                               (situation, question, options, names, good, id,
                               kind="books", factor=<decision kind>, value=<book>)
                               plus the structured fields
    src_describe_books.jsonl   the src describe of each kept case (the format
                               of tools/export_describe.py: a header row, then
                               id, set, situation, spelling)
    formulations/<v>.jsonl     the same cases, one file per way of posing them,
                               each a lab case file (factor="formulation",
                               value=<v>); only the situation differs:
                                 a_lab        the lab-style text of the scene fields
                                 b_src        src/mind/describe.py
                                 c_narrative  the narrative paraphrase alone
                                 d_traits     persona traits + b_src
                                 e_bio        persona bio + b_src
                                 f_past_acts  persona past acts + b_src
                               The question is the case's own ("What does Carl
                               do?") in every file, so only the situation varies.
    stats.json                 counts, drops and their reasons, agreement
    sample30.md                30 random kept cases, for the user's spot-check

THE SCENE. The scene fields map onto a core.Scene as tools/export_describe.py
maps the lab's (CASE_SPEC.md, "Mapping onto core.Scene"): decider ->
SelfState.name; place, public, holds, feels -> SelfState; present[] ->
Present(Ref(person), holds, state=(relation_to_decider,)); each need ->
Need(says=(need,)); percepts -> Percept, "heard: " marks a heard one; the
first two beliefs -> Belief, "told: " marks a told one (core.Scene shows at
most two; the lab text and the narrative keep the third).

THE OWN-WORDS GUARD. Each kept case's written text (narrative, persona,
scene, options) is checked against the book's clean text for runs of more
than RUN consecutive words; the cases with one are listed in stats.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DERIVED = REPO / "books" / "derived"
BOOKS = ("dcc", "lattes", "locke", "treasure")
TITLES = {"dcc": "Dungeon Crawler Carl", "lattes": "Legends & Lattes",
          "locke": "The Lies of Locke Lamora", "treasure": "Treasure Island"}
VARIANTS = ("a_lab", "b_src", "c_narrative", "d_traits", "e_bio", "f_past_acts")
RUN = 10                        # the most consecutive book words a case may share
WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")


# ---------------------------------------------------------------- join
def _rows(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _ratings(r, options):
    """The rater's words, one per option in the case's order. A rater row
    gives a list of words, or a list of {option, rating} (any order)."""
    rs = r["ratings"]
    if rs and isinstance(rs[0], dict):
        by = {x["option"]: x["rating"] for x in rs}
        if set(by) != set(options):
            raise ValueError(f"{r['id']}: the rated options are not the case's options")
        return [by[o] for o in options]
    if len(rs) != len(options):
        raise ValueError(f"{r['id']}: {len(rs)} ratings for {len(options)} options")
    return list(rs)


def join():
    """Every case of every book with its answer and its rating, and the ids
    that are missing one of them."""
    out, missing = [], []
    for b in BOOKS:
        for f in sorted((DERIVED / b / "cases").glob("*.jsonl")):
            ans = {a["id"]: a for a in _rows(DERIVED / b / "answers" / f.name)}
            rat = {r["id"]: r for r in _rows(DERIVED / b / "ratings" / f.name)}
            for c in _rows(f):
                a, r = ans.get(c["id"]), rat.get(c["id"])
                if a is None or r is None:
                    missing.append((c["id"], "no answer row" if a is None else "no rating row"))
                    continue
                out.append((c, a, r))
    return out, missing


def verdict(c, a, r):
    """(kept, the reason it is dropped, the rater's words)."""
    words = _ratings(r, c["options"])
    if not r.get("answerable", True):
        return False, "rater: not answerable", words
    w = words[a["book_choice"]]
    if w != "acceptable":
        return False, f"book's choice rated {w}", words
    return True, "", words


# ---------------------------------------------------------------- scene texts
def _listed(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _marked(text, mark):
    """("the text", True) for "<mark>: the text", else (text, False)."""
    if text.startswith(mark + ":"):
        return text[len(mark) + 1:].strip(), True
    return text, False


def _and(words):
    words = list(words)
    return words[0] if len(words) < 2 else ", ".join(words[:-1]) + " and " + words[-1]


def _sentence(x):
    x = x.strip()
    if not x:
        return x
    x = x[0].upper() + x[1:]
    return x if x[-1] in ".!?" else x + "."


def lab_text(c):
    """The lab-style situation: plain sentences, the name first, every field
    of the scene (the lab's own texts: "Tom sees ... Tom is there.")."""
    s, name = c["scene"], c["decider"]
    out = [f"{name} is in {s['place']}"]
    if s.get("public"):
        out.append(f"{name} is {s['public']}")
    for p in s.get("present", []):
        rel = p.get("relation_to_decider")
        out.append(f"{p['name']} is there, {rel}" if rel else f"{p['name']} is there")
        if p.get("holds"):
            out.append(f"{p['name']} has {_and(p['holds'])}")
    if s.get("holds"):
        out.append(f"{name} has {_and(s['holds'])}")
    if _listed(s.get("feels")):
        out.append(f"{name} feels {_and(_listed(s['feels']))}")
    for n in s.get("needs", []):
        out.append(f"{name} needs {n}")
    for x in s.get("percepts", []):
        t, heard = _marked(x, "heard")
        out.append(f"{name} heard {t}" if heard else t)
    for x in s.get("beliefs", []):
        t, told = _marked(x, "told")
        if not told:
            t, told = _marked(t, "heard")
        out.append(f"{name} was told that {t}" if told else f"{name} thinks {t}")
    return " ".join(_sentence(x) for x in out)


def scene_of(c, api, core):
    """The core.Scene that holds the facts of book case `c`."""
    s, name = c["scene"], c["decider"]
    state = api.SelfState(name=name, public=s.get("public") or "", place=s["place"],
                          holds=tuple(s.get("holds") or ()), feels=tuple(_listed(s.get("feels"))))
    people = tuple(core.Present(api.Ref("person", p["name"], p["name"]),
                                holds=tuple(p.get("holds") or ()),
                                state=(p["relation_to_decider"],) if p.get("relation_to_decider") else ())
                   for p in s.get("present", []))
    needs = tuple(core.Need(f"need{i}", (n,)) for i, n in enumerate(s.get("needs") or ()))
    percepts = []
    for i, x in enumerate((s.get("percepts") or [])[-core.PERCEPTS:]):
        t, heard = _marked(x, "heard")
        percepts.append(api.Percept(t=i, channel="heard" if heard else "saw", text=t, source=f"case:{i}"))
    me = api.Ref("person", name, name)
    beliefs = []
    for i, x in enumerate((s.get("beliefs") or [])[:core.BELIEFS]):
        t, told = _marked(x, "told")
        if not told:                            # one writer row marks a belief "heard:"
            t, told = _marked(t, "heard")
        beliefs.append(api.Belief(kind="case", about=me, value=t, text=t, source=f"belief:{i}",
                                  how="told" if told else "saw"))
    return core.Scene(state, people, needs, tuple(percepts), tuple(beliefs))


def persona_text(c, how):
    """The persona sentence that goes before the src describe."""
    p, name = c["persona"], c["decider"]
    if how == "traits":
        return f"{name}'s traits: {_and(p['traits'])}."
    if how == "bio":
        return f"About {name}: {_sentence(p['bio'])}"
    return f"Earlier, {name} {_and(p['past_acts'])}."


# ---------------------------------------------------------------- own-words guard
def _grams(text, n):
    w = WORD.findall(text.lower().replace("’", "'"))
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _case_texts(c):
    p, s = c["persona"], c["scene"]
    yield c["narrative"]
    yield p["bio"]
    yield from p["past_acts"]
    yield from c["options"]
    for k in ("place", "public"):
        if s.get(k):
            yield s[k]
    for k in ("holds", "needs", "percepts", "beliefs"):
        yield from s.get(k) or ()
    for x in s.get("present", []):
        yield x.get("relation_to_decider", "")


# ---------------------------------------------------------------- main
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--v2" in argv or "--v3" in argv:
        return main_v2(argv)
    ap = argparse.ArgumentParser(prog="python3 tools/export_book_cases.py",
                                 description="Assemble the book pilot case set and its formulations.")
    ap.add_argument("--seed", type=int, default=11, help="the seed of the spot-check sample")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    from src.mind import api, core, describe

    joined, missing = join()
    kept, drops = [], collections.Counter()
    by_book = collections.defaultdict(collections.Counter)
    by_kind = collections.defaultdict(collections.Counter)
    by_book_kind = collections.defaultdict(collections.Counter)
    nvo = collections.Counter()
    for c, ans, r in joined:
        ok, why, words = verdict(c, ans, r)
        b, k = c["book"], c["kind"]
        for d in (by_book[b], by_kind[k], by_book_kind[f"{b}/{k}"]):
            d["total"] += 1
            d["kept"] += ok
        nvo[("kept" if ok else "dropped", bool(ans["nature_vs_obvious"]))] += 1
        if not ok:
            drops[why] += 1
            continue
        kept.append((c, ans, r, words))
    for i, why in missing:
        drops[why] += 1

    # the own-words guard: the book's word n-grams, n = RUN + 1
    grams = {b: _grams((DERIVED / b / "text.txt").read_text(errors="replace"), RUN + 1)
             for b in BOOKS if (DERIVED / b / "text.txt").exists()}
    copied = {}
    for c, *_ in kept:
        g = grams.get(c["book"])
        if g is None:
            continue
        hits = [t for t in _case_texts(c) if t and _grams(t, RUN + 1) & g]
        if hits:
            copied[c["id"]] = len(hits)

    # the case rows
    rows, src_rows, refused = [], [], []
    forms = {v: [] for v in VARIANTS}
    for c, ans, r, words in kept:
        good = [i for i, w in enumerate(words) if w == "acceptable"]
        base = {"question": c["question"], "options": c["options"], "names": c["options"],
                "good": good, "id": c["id"]}
        lab = lab_text(c)
        rows.append({"situation": lab, **base, "kind": "books", "factor": c["kind"], "value": c["book"],
                     "decider": c["decider"], "persona": c["persona"], "scene": c["scene"],
                     "narrative": c["narrative"], "book": c["book"], "chunk": c["chunk"],
                     "where_in_book": c["where_in_book"], "book_choice": ans["book_choice"],
                     "obvious": ans["obvious"], "decoys": ans["decoys"],
                     "nature_vs_obvious": ans["nature_vs_obvious"],
                     "writer_acceptable": ans["acceptable"], "writer_note": ans.get("note", ""),
                     "ratings": words, "rater_note": r.get("why", "")})
        try:
            src = describe.describe(scene_of(c, api, core))
        except ValueError as e:              # a digit: the describe refuses it
            refused.append((c["id"], str(e)))
            src = None
        texts = {"a_lab": lab, "c_narrative": c["narrative"]}
        if src is not None:
            src_rows.append({"id": c["id"], "set": "books", "situation": src, "spelling": describe.SPELLING,
                             "src_question": describe.question(api.SelfState(name=c["decider"]))})
            texts.update(b_src=src, d_traits=persona_text(c, "traits") + " " + src,
                         e_bio=persona_text(c, "bio") + " " + src,
                         f_past_acts=persona_text(c, "past_acts") + " " + src)
        for v, t in texts.items():
            forms[v].append({"situation": t, **base, "kind": "books", "factor": "formulation", "value": v,
                             "book": c["book"], "decision_kind": c["kind"]})

    def write(path, items):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))

    write(DERIVED / "cases_books.jsonl", rows)
    src_file = Path(describe.__file__).resolve()
    sha = hashlib.sha256(src_file.read_bytes()).hexdigest()
    head = {"header": True, "spelling": describe.SPELLING, "root": str(REPO), "describe_sha256": sha,
            "rendered": {"books": len(src_rows)}, "refused": len(refused)}
    write(DERIVED / "src_describe_books.jsonl", [head] + src_rows)
    for v in VARIANTS:
        write(DERIVED / "formulations" / f"{v}.jsonl", forms[v])

    # stats
    total = len(joined) + len(missing)
    nk = len(rows)
    goods = [len(x["good"]) for x in rows]
    opts = [len(x["options"]) for x in rows]

    def jac(x):
        g, w = set(x["good"]), set(x["writer_acceptable"])
        return len(g & w) / len(g | w)

    def share(n, d):
        return round(n / d, 4) if d else None

    stats = {
        "cases": total, "kept": nk, "dropped": total - nk, "keep_rate": share(nk, total),
        "drop_reasons": dict(drops.most_common()),
        "per_book": {b: {"title": TITLES[b], "total": by_book[b]["total"], "kept": by_book[b]["kept"],
                         "keep_rate": share(by_book[b]["kept"], by_book[b]["total"]),
                         "kept_by_kind": {k.split("/")[1]: v["kept"] for k, v in sorted(by_book_kind.items())
                                          if k.startswith(b + "/")}}
                     for b in BOOKS},
        "per_kind": {k: {"total": v["total"], "kept": v["kept"], "keep_rate": share(v["kept"], v["total"])}
                     for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]["kept"])},
        "nature_vs_obvious": {
            "share_of_kept": share(nvo[("kept", True)], nk),
            "share_of_all": share(nvo[("kept", True)] + nvo[("dropped", True)], total),
            "keep_rate_when_true": share(nvo[("kept", True)], nvo[("kept", True)] + nvo[("dropped", True)]),
            "keep_rate_when_false": share(nvo[("kept", False)], nvo[("kept", False)] + nvo[("dropped", False)]),
        },
        "good": {"mean_size": round(sum(goods) / nk, 3), "sizes": dict(sorted(collections.Counter(goods).items())),
                 "options": dict(sorted(collections.Counter(opts).items())),
                 "chance_hit_rate": round(sum(g / o for g, o in zip(goods, opts)) / nk, 4),
                 "single_answer_cases": goods.count(1)},
        "agreement": {
            "mean_jaccard_rater_vs_writer": round(sum(map(jac, rows)) / nk, 4),
            "obvious_in_good": share(sum(1 for x in rows if x["obvious"] is not None and x["obvious"] in x["good"]),
                                     sum(1 for x in rows if x["obvious"] is not None)),
            "writer_decoy_rated_acceptable": share(
                sum(1 for x in rows for d in x["decoys"] if x["ratings"][d] == "acceptable"),
                sum(len(x["decoys"]) for x in rows)),
        },
        "rater_notes": {"kept_with_note": sum(1 for x in rows if x["rater_note"]),
                        "leak_noted": [x["id"] for x in rows if "leak" in x["rater_note"].lower()]},
        "own_words_guard": {"run_limit_words": RUN, "cases_with_a_longer_run": len(copied),
                            "ids": sorted(copied)},
        "describe": {"spelling": describe.SPELLING, "describe_sha256": sha, "refused": refused},
        "formulations": {v: {"rows": len(forms[v]),
                             "mean_words": round(sum(len(x["situation"].split()) for x in forms[v])
                                                 / max(1, len(forms[v])), 1)}
                         for v in VARIANTS},
        "files": [str(p.relative_to(REPO)) for p in
                  [DERIVED / "cases_books.jsonl", DERIVED / "src_describe_books.jsonl"]
                  + [DERIVED / "formulations" / f"{v}.jsonl" for v in VARIANTS]
                  + [DERIVED / "stats.json", DERIVED / "sample30.md"]],
        "keep_rule": "rater answerable AND the book's choice rated acceptable; good = every option rated acceptable",
        "sample_seed": a.seed,
    }
    (DERIVED / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n")

    # the spot-check sample
    srcs = {x["id"]: x["situation"] for x in src_rows}
    pick = sorted(random.Random(a.seed).sample(rows, min(30, nk)), key=lambda x: x["id"])
    md = ["# Book pilot: 30 random kept cases (local only)", "",
          f"Seed {a.seed}, from {nk} kept cases. Marks: **B** the book's choice, "
          "**ok** rated acceptable (in `good`), **obv** the writer's obvious act, "
          "**decoy** the writer's decoy; the rater's word in brackets.", ""]
    for x in pick:
        p = x["persona"]
        md += [f"## {x['id']}: {x['decider']} ({x['factor']}, {TITLES[x['book']]}, {x['where_in_book']})", "",
               f"*Traits:* {', '.join(p['traits'])}. *Bio:* {p['bio']}", "",
               f"*Earlier:* {'; '.join(p['past_acts'])}", "",
               f"*Lab text:* {x['situation']}", "",
               f"*Src describe:* {srcs.get(x['id'], '(refused)')}", "",
               f"*Narrative:* {x['narrative']}", "",
               f"*{x['question']}*", ""]
        for i, o in enumerate(x["options"]):
            m = [t for t, on in (("B", i == x["book_choice"]), ("ok", i in x["good"]),
                                  ("obv", i == x["obvious"]), ("decoy", i in x["decoys"])) if on]
            md.append(f"{i}. {o}  [{x['ratings'][i]}]" + (f" **{' '.join(m)}**" if m else ""))
        md += ["", f"*Nature vs obvious:* {x['nature_vs_obvious']}. *Writer's note:* {x['writer_note']}"]
        if x["rater_note"]:
            md.append(f"*Rater's note:* {x['rater_note']}")
        md.append("")
    (DERIVED / "sample30.md").write_text("\n".join(md))

    print(json.dumps({k: stats[k] for k in ("cases", "kept", "keep_rate", "drop_reasons")}))
    print(f"refused by the describe: {len(refused)}; own-words runs: {len(copied)}")


# ================================================================ RECIPE v2 (wave 11b)
# python3 tools/export_book_cases.py --v2 [--limit N] [--out DIR] [--pairs DIR]
#
# Reads books/derived/v2/cases_v2.jsonl (the assembled v2 cases) and, when it
# exists, books/derived/v2/pairs/variants.jsonl (the minimal pairs: {pair_id,
# base_id, variant, the case fields that differ}). Writes one lab case file
# per way of posing the decision into books/derived/v2/formulations_v2/
# (pairs into formulations_v2/pairs/), and formulations_v2/manifest.json.
#
# Every file holds the same cases with the same options in the same order
# (only diag_option_reasons adds words to the options; `names` keeps them
# plain). `good` and `grades` come from the three blind raters. The context
# texts are built from the writer's fields only: the narrative, the key fact
# and the persona were written cut off at the decision, and nothing here reads
# an outcome. The FULL context (ctx_full) is: persona, history, place, who is
# present, holds, needs, percepts, beliefs, dialogue, in that order, in prose,
# third person.
#
# THE FILES (FORMS_V2 below says each one in a line). Files whose texts are
# the same (ctx_full = person_3rd = fmt_prose = amt_full = place_persona_first)
# are listed in the manifest as one group: a decision made for one is the
# decision for all (lab.methods.screen reuses it by digest, so it costs no
# GPU time).
#
# FIRST AND SECOND PERSON. The decider's name becomes I/me/my (you/your), and
# so do the decider's own pronouns (the decider's gender is read from the
# pronouns of the persona and history), but only in an item that names no
# other person: "Bea has been unfaithful to him" keeps "him". The manifest
# counts the items left in the third person.
#
# FALLBACKS. The persona phrasings and the reason-in-each-option ceiling read
# optional writer fields (persona.habit, persona.wants, persona.never,
# option_reasons). The v2 writers did not write them, so a template made from
# the existing fields stands in, and the manifest says how many rows used it:
#   pp_habit   "In moments like this, Carl acts as someone <traits> would,
#              as when Carl <past acts>."
#   pp_wants   "What Carl wants most now: <the needs>."
#   pp_never   "Carl never acts against being <traits>."
#   diag_option_reasons  each option + ", since <the scene fact that shares
#              the most words with it>" (needs, percepts, beliefs, key fact,
#              holds; no grade, answer or outcome is read).

V2DIR = DERIVED / "v2"
V3_FORMS = ("ctx_full", "place_key_last", "gate_options_only", "no_beliefs", "no_needs", "b_src", "no_persona")

FORMS_V2 = {
    # the pilot's six, on the v2 cases
    "a_lab": "the lab-style text of the scene fields (the pilot's a_lab)",
    "b_src": "src/mind/describe.py of the scene",
    "c_narrative": "the narrative paraphrase alone",
    "d_traits": "persona traits + b_src",
    "e_bio": "persona bio + b_src",
    "f_past_acts": "persona past acts + b_src",
    # the full context and its variants
    "ctx_full": "full context: persona, history, scene, dialogue; prose, third person",
    "person_1st": "ctx_full in the first person ('I am Carl.' ... 'What do I do?')",
    "person_2nd": "ctx_full in the second person ('You are Carl.' ... 'What do you do?')",
    "person_3rd": "ctx_full in the third person (= ctx_full)",
    "q_likely": "ctx_full; question 'Which of these is Carl most likely to do?'",
    "q_character": "ctx_full; question 'Being who Carl is, what would Carl do here?'",
    "q_next": "ctx_full; question 'What does Carl do next?'",
    "menu_two_level": "ctx_full; first the kind of act (row menu2), then the act within that kind",
    "rel_off": "ctx_full; people present named without their relation to the decider",
    "dialogue_off": "ctx_full without the dialogue",
    "no_persona": "ctx_full without the persona",
    "no_beliefs": "ctx_full without the beliefs",
    "no_percepts": "ctx_full without the percepts",
    "no_present": "ctx_full without the people present (names and relations)",
    "no_needs": "ctx_full without the needs",
    "no_holds": "ctx_full without what the decider holds",
    "no_history": "ctx_full without the history",
    "place_persona_first": "persona first (= ctx_full)",
    "place_persona_last": "persona last, after the dialogue",
    "place_key_last": "ctx_full + the key fact as the last sentence, next to the options",
    "place_goal_after": "ctx_full; the decider's needs restated in the question, after the scene"
                        " (row `after` holds them for an adapter that can put text after the options)",
    "fmt_prose": "prose (= ctx_full)",
    "fmt_labeled": "the same facts as labeled fields, one per line",
    "fmt_bullets": "the same sentences as a bulleted list",
    "amt_minimal": "place, needs and percepts only",
    "amt_full": "= ctx_full",
    "amt_distract": "ctx_full + 3 facts from another case of the same book, among the percepts",
    "pp_habit": "ctx_full; persona as the decider's habit in moments like this",
    "pp_wants": "ctx_full; persona as what the decider wants most now",
    "pp_never": "ctx_full; persona as what the decider never does",
    "diag_key_marked": "ceiling: ctx_full with the key fact first, marked 'The key fact:'",
    "diag_option_reasons": "ceiling: ctx_full; each option carries a short reason",
    "gate_options_only": "leak gate: no facts, only '<decider> is here.' (the recipe's GLiNER options-only gate;"
                         " near chance or the options give the answer away)",
}
FIELDS_V2 = ("persona", "history", "place", "present", "holds", "needs", "percepts", "beliefs", "dialogue")
SRC_FORMS = ("b_src", "d_traits", "e_bio", "f_past_acts")

# THE KINDS OF ACT for the two-level menu, by the option's first verb. A verb
# not listed is "do something else". Shown to the chooser as these labels.
SCOPES = (
    ("speak", "Say something to someone", set(
        "ask tell say offer order call refuse agree admit shout demand promise warn insist thank accept explain "
        "suggest claim threaten apologize apologise beg joke lie whisper propose remind argue scold tease praise "
        "answer accuse confess swear complain deny urge invite introduce blame plead report hint greet scream "
        "announce question reassure taunt talk comfort cheer wish repeat challenge excuse insult bargain confirm "
        "assure forbid haggle dismiss yell hiss recite request lecture interrupt chat message declare boast "
        "flatter rebuke concede vow denounce advise speak gloat reply remark compliment congratulate recount "
        "blurt mock name describe curse sing pray bless toast laugh grumble sneer negotiate persuade convince "
        "lie, admit, agree, mention reveal".split())),
    ("force", "Fight or use force", set(
        "throw fire kick punch shoot charge strike kill lunge swing shove attack bite stab smash hit slash ambush "
        "beat fight slap knock hurl blast firebomb cast tackle choke wrestle club bash ram cut break crush "
        "poke spit drag shake push pull".split())),
    ("move", "Go somewhere", set(
        "go leave run climb walk head jump step back slip follow leap ride hurry kneel chase dive retreat creep "
        "crawl row wade paddle swim sneak edge flee bolt duck hop dash wander storm explore sprint enter return "
        "approach move rush circle visit escape cross descend ascend head, lead".split())),
    ("things", "Take, give or use something", set(
        "take give hand grab pick put pour drink draw drop light open hold toss pay fetch buy pocket wrap wipe "
        "lift fill steal carry snatch dig bury hang paint pack lock tie sell feed loot pry load trade use eat "
        "close search check read count show point set raise press brush wash scoop store unlock untie bind gag "
        "plant rig clean tidy cook brew serve spill pass share sign post wear arm swap strap tuck lay haul "
        "scratch sniff examine inspect test study look watch listen glance scan peek write build".split())),
    ("hold", "Wait, hide or hold back", set(
        "keep stay wait let sit hide ignore stand pretend stop shrug nod rest sleep linger freeze sulk avoid "
        "pause feign bow smile wink sigh glare stare remain endure play act do".split())),
)
OTHER_SCOPE = ("other", "Do something else")


def _first_verb(o):
    w = o.split()
    return (w[0] if w else "").lower().strip(",.;:")


def scope_of(o):
    v = _first_verb(o)
    for key, label, verbs in SCOPES:
        if v in verbs:
            return key, label
    return OTHER_SCOPE


def menu2(options, decider):
    """The two-level menu of one case: the kinds of act its options fall in
    (in SCOPES order), and the option indices in each."""
    groups, labels = {}, {}
    for i, o in enumerate(options):
        k, lab = scope_of(o)
        groups.setdefault(k, []).append(i)
        labels[k] = lab
    order = [k for k, *_ in SCOPES + (OTHER_SCOPE,) if k in groups]
    return {"question": f"What kind of thing does {decider} do?", "scopes": [labels[k] for k in order],
            "keys": order, "groups": [groups[k] for k in order]}


# ---------------------------------------------------------------- the person of the text
_PRON = {"m": {"he": "S", "him": "O", "his": "P", "himself": "R"},
         "f": {"she": "S", "her": "OP", "hers": "PP", "herself": "R"}}
_SWAP = {1: {"S": "I", "O": "me", "P": "my", "PP": "mine", "R": "myself"},
         2: {"S": "you", "O": "you", "P": "your", "PP": "yours", "R": "yourself"}}
_FUNC = set("to and or the a an that in on at for with from out up down off away back again alone into onto "
            "over under about after before as if than but so is was through across along past toward towards "
            "around behind near by inside outside like beside between against until while when because then "
            "home here there now too very aside below above upon without within instead once".split())
_AGREE = {1: {"is": "am", "was": "was", "has": "have", "does": "do", "isn't": "am not", "wasn't": "wasn't",
              "hasn't": "haven't", "doesn't": "don't"},
          2: {"is": "are", "was": "were", "has": "have", "does": "do", "isn't": "aren't", "wasn't": "weren't",
              "hasn't": "haven't", "doesn't": "don't"}}
_NOT_VERBS = set("always perhaps sometimes nevertheless thus afterwards towards besides less unless was "
                 "this is has does as us its".split())


def _plain_verb(w):
    """The plain form of a third-person present verb ("finishes" -> "finish");
    any other word unchanged."""
    if w in _NOT_VERBS or len(w) < 4 or not w.endswith("s") or w.endswith("ss"):
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses", "oes", "zes")):
        return w[:-2]
    return w[:-1]


_STARTERS = set("the a an every some one two three all no then when after before while as at in on with his her "
                "their its it there this that these those once later now soon".split())
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’]*|[^A-Za-z]+")


GENDERS = {}                    # (book, name) -> "m" / "f", set by genders_v2()


def genders_v2(cases):
    """The gender of every decider of every book: each bio, past act or
    history item that STARTS with the decider's name votes with its first
    he/him/his/himself or she/her/herself (it nearly always means the
    subject); 'm' or 'f' when one side has at least three times the votes of
    the other, else left out (and its text stays in the third person)."""
    cnt = collections.defaultdict(lambda: [0, 0])
    for c in cases:
        name = c["decider"]
        for t in [c["persona"]["bio"], *c["persona"]["past_acts"], *c.get("history", [])]:
            if not t.startswith(name):
                continue
            for x in re.findall(r"[a-z']+", t[len(name):].lower()):
                if x in ("he", "him", "his", "himself"):
                    cnt[(c["book"], name)][0] += 1
                    break
                if x in ("she", "her", "hers", "herself"):
                    cnt[(c["book"], name)][1] += 1
                    break
    GENDERS.clear()
    for key, (m, f) in cnt.items():
        if m >= 3 * f and m:
            GENDERS[key] = "m"
        elif f >= 3 * m and f:
            GENDERS[key] = "f"
    return GENDERS


def gender_of(c):
    """'m', 'f' or None (see genders_v2)."""
    return GENDERS.get((c["book"], c["decider"]))


class Person:
    """How one text names the decider: 3 = by name, 1 = I, 2 = you."""
    def __init__(self, c, person=3):
        self.name, self.p, self.book = c["decider"], person, c["book"]
        self.gender = gender_of(c) if person != 3 else None
        self.left_third = 0                       # items that kept a third-person pronoun
        others = {p["name"] for p in c["scene"].get("present", [])}
        self._others = {w for o in others for w in o.split() if w[:1].isupper()} - set(self.name.split())

    # the subject and its verbs
    @property
    def subj(self):
        return {3: self.name, 1: "I", 2: "You"}[self.p]

    @property
    def poss(self):
        return {3: f"{self.name}'s", 1: "My", 2: "Your"}[self.p]

    def verb(self, v):
        """'has' / 'thinks' in the third person, 'have' / 'think' otherwise."""
        if self.p == 3:
            return {"have": "has", "be": "is", "was": "was"}.get(v, v + "s")
        return {"have": "have", "be": "am" if self.p == 1 else "are",
                "was": "was" if self.p == 1 else "were"}.get(v, v)

    def _names_other(self, text):
        """Does the item name someone who could own the decider's pronouns?
        A name known (GENDERS) to be of the other gender does not count."""
        caps = {re.sub(r"['’]s$", "", w.strip(".,;:!?'’\"")) for w in text.split() if w[:1].isupper()}
        caps = {w for w in caps if w.lower() not in _STARTERS} - set(self.name.split()) - {"I"}
        return any(GENDERS.get((self.book, w)) != {"m": "f", "f": "m"}[self.gender] for w in caps)

    def fix(self, text):
        """`text` with the decider named in this person (unchanged in the third)."""
        if self.p == 3 or not text:
            return text
        sw = _SWAP[self.p]
        names = self.name.split()
        swap_pron = self.gender is not None and not self._names_other(text)
        if self.gender is not None and not swap_pron and any(
                w.lower() in _PRON[self.gender] for w in re.findall(r"[A-Za-z]+", text)):
            self.left_third += 1
        toks = _TOKEN.findall(text)
        out, i, prev_subj = [], 0, False
        while i < len(toks):
            t = toks[i]
            low = t.lower()
            nxt = next((toks[j] for j in range(i + 1, len(toks)) if toks[j].strip()), "")
            at_start = not "".join(out).strip() or "".join(out).rstrip()[-1:] in ".;:!?"
            # the name, possibly several words ("Locke Lamora")
            if t == names[0] and toks[i:i + 2 * len(names) - 1:2] == names:
                end = i + 2 * len(names) - 1
                rest = toks[end] if end < len(toks) else ""
                if rest.startswith("'s") or rest.startswith("’s"):
                    out.append(sw["P"])
                    toks[end] = rest[2:]
                    i = end
                    prev_subj = False
                    continue
                prev_wd = next((x for x in reversed(out) if x.strip()), "").lower()
                subj = at_start or prev_wd in ("and", "that", "when", "while", "because", "but", "then", "so")
                out.append(sw["S"] if subj else sw["O"])
                i, prev_subj = end, subj
                continue
            if (t.endswith("'s") or t.endswith("’s")) and t[:-2] == self.name:
                out.append(sw["P"])
                i += 1
                prev_subj = False
                continue
            if swap_pron and low in _PRON[self.gender]:
                role = _PRON[self.gender][low]
                if role == "OP":                  # "her": possessive before a noun, else object
                    role = "O" if (not nxt or not nxt[:1].isalpha() or nxt.lower() in _FUNC) else "P"
                out.append(sw[role])
                prev_subj = role == "S"
                i += 1
                continue
            if prev_subj and low in _AGREE[self.p]:
                out.append(_AGREE[self.p][low])
                prev_subj = False
                i += 1
                continue
            if prev_subj and t.islower():          # "he finishes" -> "I finish"
                out.append(_plain_verb(t))
                prev_subj = False
                i += 1
                continue
            if t.strip():
                prev_subj = False
            out.append(t)
            i += 1
        s = "".join(out)
        # a sentence start inside the item is capitalised (the item's own start is the caller's)
        s = re.sub(r"([.!?]\s+)(you|my|me)\b", lambda m: m.group(1) + m.group(2)[0].upper() + m.group(2)[1:], s)
        return re.sub(r"\bi\b", "I", s)

    def act(self, text):
        """A past act: "<subject> <act>", whether or not the writer put the name first."""
        t = text.strip()
        if t.split(" ")[0] == self.name.split()[0] or t.startswith(self.name):
            return self.fix(t)
        return self.fix(f"{self.name} {t[0].lower() + t[1:]}") if t else t


_LOWER_FIRST = set("the a an his her their its he she they it there this that these those some one two".split())


def _lc(x):
    """`x` with its first letter small, unless it starts with a name."""
    w = x.split(" ")[0] if x else ""
    return x[0].lower() + x[1:] if w.lower() in _LOWER_FIRST else x


def _strip_dot(x):
    return x.strip().rstrip(".")


def _dialogue(c, P):
    out = []
    for line in c.get("dialogue") or []:
        who, _, what = line.partition(":")
        if not what:
            out.append(_sentence(P.fix(line)))
            continue
        who = who.strip()
        w = P.subj if who == c["decider"] and P.p != 3 else who
        out.append(_sentence(f"{w} just said: {P.fix(what.strip())}"))
    return out


def groups_v2(c, P, rel=True):
    """The context of case `c` as {field: [sentences]} in this person."""
    s, name = c["scene"], c["decider"]
    p = c["persona"]
    g = {}
    g["persona"] = [_sentence(f"{P.poss} traits: {_and(p['traits'])}"), _sentence(P.fix(p["bio"]))]
    if p.get("past_acts"):
        g["persona"].append(_sentence("Earlier, " + "; ".join(_strip_dot(P.act(a)) for a in p["past_acts"])))
    hist = c.get("history") or []
    g["history"] = ([_sentence("What has happened so far, oldest first: "
                               + "; ".join(_strip_dot(P.fix(e)) for e in hist))] if hist else [])
    g["place"] = [_sentence(f"{P.subj} {P.verb('be')} in {P.fix(s['place'])}")] if s.get("place") else []
    g["present"] = []
    for x in s.get("present", []):
        r = x.get("relation_to_decider")
        g["present"].append(_sentence(f"{x['name']} is there, {P.fix(r)}" if rel and r else f"{x['name']} is there"))
    g["holds"] = [_sentence(f"{P.subj} {P.verb('have')} {_and([P.fix(h) for h in s['holds']])}")] if s.get("holds") else []
    g["needs"] = [_sentence(f"{P.subj} {P.verb('need')} {P.fix(n)}") for n in s.get("needs") or []]
    g["percepts"] = []
    for x in s.get("percepts") or []:
        t, heard = _marked(x, "heard")
        g["percepts"].append(_sentence(f"{P.subj} heard {P.fix(t)}" if heard else P.fix(t)))
    g["beliefs"] = []
    for x in s.get("beliefs") or []:
        t, told = _marked(x, "told")
        if not told:
            t, told = _marked(t, "heard")
        g["beliefs"].append(_sentence(f"{P.subj} {P.verb('was')} told that {P.fix(t)}" if told
                                      else f"{P.subj} {P.verb('think')} {P.fix(t)}"))
    g["dialogue"] = _dialogue(c, P)
    return g


def labeled_v2(c):
    """The same facts as labeled fields, one per line (third person)."""
    s, p = c["scene"], c["persona"]
    out = [f"Who: {c['decider']}", f"Traits: {', '.join(p['traits'])}", f"About: {_sentence(p['bio'])}"]
    if p.get("past_acts"):
        out.append("Earlier: " + "; ".join(_strip_dot(a) for a in p["past_acts"]))
    if c.get("history"):
        out.append("History, oldest first: " + "; ".join(_strip_dot(e) for e in c["history"]))
    out.append(f"Place: {s['place']}")
    if s.get("present"):
        out.append("Present: " + "; ".join(f"{x['name']} ({x['relation_to_decider']})" if x.get("relation_to_decider")
                                           else x["name"] for x in s["present"]))
    if s.get("holds"):
        out.append("Holds: " + "; ".join(s["holds"]))
    if s.get("needs"):
        out.append("Needs: " + "; ".join(s["needs"]))
    if s.get("percepts"):
        out.append("Noticed: " + "; ".join(_strip_dot(x) for x in s["percepts"]))
    if s.get("beliefs"):
        out.append("Believes: " + "; ".join(_strip_dot(x) for x in s["beliefs"]))
    if c.get("dialogue"):
        out.append("Just said: " + "; ".join(_strip_dot(x) for x in c["dialogue"]))
    return "\n".join(out)


def bullets_v2(c):
    """The sentences of ctx_full as a bulleted list; the history one event a line."""
    g = groups_v2(c, Person(c, 3))
    out = []
    for f in FIELDS_V2:
        if f == "history" and c.get("history"):
            out.append("- What has happened so far, oldest first:")
            out += [f"  - {_strip_dot(e)}" for e in c["history"]]
            continue
        out += [f"- {x}" for x in g[f]]
    return "\n".join(out)


def _prose(g, order=FIELDS_V2, drop=()):
    return " ".join(x for f in order if f not in drop for x in g.get(f, []))


_CONTENT = re.compile(r"[a-z]+")
_STOPS = set("a an the and or but to of in on at by for with from his her their its him them he she they it is "
             "was be are were this that there here into onto up down out off over all some any one".split())


def _content(t):
    return {w for w in _CONTENT.findall(t.lower()) if w not in _STOPS and len(w) > 2}


def reasons_v2(c):
    """Each option with its writer reason (option_reasons), or else the scene
    fact that shares the most content words with it (no answer is read)."""
    if c.get("option_reasons") and len(c["option_reasons"]) == len(c["options"]):
        return [f"{o}, since {_lc(_strip_dot(r))}" for o, r in zip(c["options"], c["option_reasons"])], True
    s, name = c["scene"], c["decider"]

    def clean(x):
        return _strip_dot(_marked(_marked(x, "heard")[0], "told")[0])
    facts = ([(n, f"{name} needs {clean(n)}") for n in s.get("needs") or []]
             + [(x, _lc(clean(x))) for x in s.get("percepts") or []]
             + [(x, f"{name} thinks {clean(x)}") for x in s.get("beliefs") or []]
             + ([(c["key_fact"], _lc(clean(c["key_fact"])))] if c.get("key_fact") else [])
             + [(h, f"{name} has {clean(h)}") for h in s.get("holds") or []])
    out = []
    for o in c["options"]:
        ow = _content(o)
        best = max(range(len(facts)), key=lambda i: (len(ow & _content(facts[i][0])), -i), default=None)
        out.append(o if best is None else f"{o}, since {facts[best][1]}")
    return out, False


def persona_phrase(c, how, P):
    """The persona group said as a habit, a want or a never; (sentences, from the writer?)."""
    p, name = c["persona"], c["decider"]
    if how == "habit":
        if p.get("habit"):
            return [_sentence(P.fix(p["habit"]))], True
        acts = "; ".join(_strip_dot(P.act(a)) for a in p.get("past_acts") or [])
        t = f"In moments like this, {P.subj} {P.verb('act')} as someone {_and(p['traits'])} would"
        return [_sentence(t + (f", as when {acts}" if acts else ""))], False
    if how == "wants":
        if p.get("wants"):
            return [_sentence(f"What {P.subj if P.p != 3 else name} {'want' if P.p != 3 else 'wants'} most now: "
                              f"{P.fix(p['wants'])}")], True
        needs = c["scene"].get("needs") or []
        if not needs:
            return [_sentence(f"{P.poss} traits: {_and(p['traits'])}")], False
        w = "want" if P.p != 3 else "wants"
        return [_sentence(f"What {P.subj} {w} most now: {_and([P.fix(n) for n in needs])}")], False
    if p.get("never"):
        return [_sentence(P.fix(p["never"]))], True
    return [_sentence(f"{P.subj} never {P.verb('act')} against being {_and(p['traits'])}")], False


def distractors(c, pool, k=3):
    """`k` facts from another case of the same book (seeded by the case id):
    its percepts first, then its history events."""
    others = [o for o in pool.get(c["book"], []) if o["id"] != c["id"] and o.get("chunk") != c.get("chunk")] \
        or [o for o in pool.get(c["book"], []) if o["id"] != c["id"]]
    if not others:
        return [], None
    rng = random.Random(f"distract:{c['id']}")
    o = rng.choice(others)
    facts = [_marked(_marked(x, "heard")[0], "told")[0] for x in o["scene"].get("percepts") or []] + list(o.get("history") or [])
    facts = [f for f in facts if f]
    rng.shuffle(facts)
    return [_sentence(f) for f in facts[:k]], o["id"]


def render_v2(c, src_text, pool, notes):
    """{form: (situation, question, extra row fields)} of one case."""
    name, q = c["decider"], c["question"]
    P3 = Person(c, 3)
    g = groups_v2(c, P3)
    full = _prose(g)
    out = {"a_lab": (lab_text(c), q, {}), "c_narrative": (c["narrative"], q, {})}
    if src_text is not None:
        out.update(b_src=(src_text, q, {}),
                   d_traits=(persona_text(c, "traits") + " " + src_text, q, {}),
                   e_bio=(persona_text(c, "bio") + " " + src_text, q, {}),
                   f_past_acts=(persona_text(c, "past_acts") + " " + src_text, q, {}))
    out["ctx_full"] = (full, q, {})
    for n, P in ((1, Person(c, 1)), (2, Person(c, 2))):
        gp = groups_v2(c, P)
        intro = "I am" if n == 1 else "You are"
        out[f"person_{['', '1st', '2nd'][n]}"] = (f"{intro} {name}. " + _prose(gp),
                                                   "What do I do?" if n == 1 else "What do you do?", {})
        notes[f"person_{n}_items_left_third"] += P.left_third
        notes[f"person_{n}_no_gender"] += P.gender is None
    out["person_3rd"] = (full, q, {})
    out["q_likely"] = (full, f"Which of these is {name} most likely to do?", {})
    out["q_character"] = (full, f"Being who {name} is, what would {name} do here?", {})
    out["q_next"] = (full, f"What does {name} do next?", {})
    out["menu_two_level"] = (full, q, {"menu2": menu2(c["options"], name)})
    out["rel_off"] = (_prose(groups_v2(c, P3, rel=False)), q, {})
    out["dialogue_off"] = (_prose(g, drop=("dialogue",)), q, {})
    for f in ("persona", "beliefs", "percepts", "present", "needs", "holds", "history"):
        out[f"no_{f}"] = (_prose(g, drop=(f,)), q, {})
    out["place_persona_first"] = (full, q, {})
    out["place_persona_last"] = (_prose(g, order=FIELDS_V2[1:] + ("persona",)), q, {})
    kf = _sentence(c["key_fact"]) if c.get("key_fact") else ""
    out["place_key_last"] = ((full + " " + kf).strip(), q, {})
    needs = c["scene"].get("needs") or []
    goal = _sentence(f"{name} needs {_and(needs)}") if needs else ""
    out["place_goal_after"] = (full, f"{q[:-1] if q.endswith('?') else q}, given that {name} needs {_and(needs)}?"
                               if needs else q, {"after": goal})
    out["fmt_prose"] = (full, q, {})
    out["fmt_labeled"] = (labeled_v2(c), q, {})
    out["fmt_bullets"] = (bullets_v2(c), q, {})
    out["amt_minimal"] = (_prose(g, order=("place", "needs", "percepts")), q, {})
    out["amt_full"] = (full, q, {})
    dis, from_id = distractors(c, pool)
    gd = dict(g)
    pc = list(g["percepts"])
    rng = random.Random(f"place:{c['id']}")
    for x in dis:
        pc.insert(rng.randint(0, len(pc)), x)
    gd["percepts"] = pc
    out["amt_distract"] = (_prose(gd), q, {"distract_from": from_id})
    for how in ("habit", "wants", "never"):
        sents, own = persona_phrase(c, how, P3)
        notes[f"pp_{how}_writer"] += own
        notes[f"pp_{how}_template"] += not own
        out[f"pp_{how}"] = (_prose(dict(g, persona=sents)), q, {})
    out["diag_key_marked"] = (((f"The key fact: {_strip_dot(c['key_fact'])}. " if c.get("key_fact") else "") + full),
                              q, {})
    ro, own = reasons_v2(c)
    notes["reasons_writer" if own else "reasons_template"] += 1
    out["diag_option_reasons"] = (full, q, {"options": ro})
    out["gate_options_only"] = (f"{name} is here.", q, {})
    return out


def src_describe_v2(c):
    """The src describe of the scene, or None (it refuses digits)."""
    sys.path.insert(0, str(REPO))
    from src.mind import api, core, describe
    try:
        return describe.describe(scene_of(c, api, core))
    except ValueError:
        return None


def merged_variant(v, base):
    """A pair variant as a full case: the base case's fields, then the variant's."""
    c = {k: x for k, x in base.items() if k not in ("good", "grades", "ratings", "situation")}
    for k, x in v.items():
        if k in ("situation",):
            continue
        if k == "scene" and isinstance(x, dict):
            c["scene"] = {**base["scene"], **x}
        elif k == "persona" and isinstance(x, dict):
            c["persona"] = {**base["persona"], **x}
        else:
            c[k] = x
    c["id"] = v["pair_id"]
    if v.get("decider") and v["decider"] != base["decider"] and "question" not in v:
        c["question"] = f"What does {v['decider']} do?"
    if list(c["options"]) != list(base["options"]):
        raise ValueError(f"{v['pair_id']}: a variant must keep the base case's options")
    return c


def main_v2(argv):
    ap = argparse.ArgumentParser(prog="python3 tools/export_book_cases.py --v2",
                                 description="Write the v2 formulation files.")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--v3", action="store_true", help="recipe v2.1: books/derived/v3/cases_v3.jsonl and its pairs"
                    " into v3/formulations_v3/, only the forms of V3_FORMS (or --forms)")
    ap.add_argument("--forms", nargs="*", default=None, help="write only these formulation files")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--pairs", default=None, help="the directory of variants.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="the first N cases only (a dry run)")
    a = ap.parse_args(argv)
    v = DERIVED / ("v3" if a.v3 else "v2")
    a.cases = a.cases or str(v / ("cases_v3.jsonl" if a.v3 else "cases_v2.jsonl"))
    a.pairs = a.pairs or str(v / "pairs")
    a.out = a.out or str(v / ("formulations_v3" if a.v3 else "formulations_v2"))
    keep = a.forms if a.forms is not None else (V3_FORMS if a.v3 else None)
    cases = _rows(Path(a.cases))
    if not cases:
        raise SystemExit(f"no cases in {a.cases}")
    genders_v2(cases)
    pool = collections.defaultdict(list)
    for c in cases:
        pool[c["book"]].append(c)
    if a.limit:
        cases = cases[:a.limit]
    by_id = {c["id"]: c for c in _rows(Path(a.cases))}
    variants = _rows(Path(a.pairs) / "variants.jsonl")
    notes = collections.Counter()
    out = Path(a.out)

    def rows_of(c, extra):
        src = src_describe_v2(c)
        if src is None:
            notes["src_refused"] += 1
        got = {}
        for form, (sit, q, more) in render_v2(c, src, pool, notes).items():
            opts = more.pop("options", c["options"])
            row = {"situation": sit, "question": q, "options": opts, "names": list(c["options"]),
                   "good": c.get("good", []), "grades": c.get("grades"), "id": c["id"], "kind": "books_v2",
                   "factor": "formulation", "value": form, "book": c["book"], "decision_kind": c.get("factor"),
                   "decider": c["decider"], "book_choice": c.get("book_choice"), "obvious": c.get("obvious"),
                   "decoys": c.get("decoys"), **extra, **more}
            got[form] = row
        return got

    use = [f for f in FORMS_V2 if keep is None or f in keep]
    forms = {f: [] for f in use}
    for c in cases:
        for f, r in rows_of(c, {}).items():
            if f in forms:
                forms[f].append(r)
    pforms = {f: [] for f in use}
    skipped = []
    for v in variants:
        base = by_id.get(v.get("base_id"))
        if base is None or (a.limit and base["id"] not in {c["id"] for c in cases}):
            skipped.append(v.get("pair_id"))
            continue
        try:
            c = merged_variant(v, base)
        except (ValueError, KeyError) as e:
            skipped.append(f"{v.get('pair_id')}: {e}")
            continue
        for f, r in rows_of(c, {"pair_id": v["pair_id"], "base_id": base["id"], "variant": v.get("variant"),
                                "good": [], "grades": None}).items():
            if f in pforms:
                pforms[f].append(r)

    def write(path, items):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))

    for f, rs in forms.items():
        write(out / f"{f}.jsonl", rs)
    if any(pforms.values()):
        for f, rs in pforms.items():
            write(out / "pairs" / f"{f}.jsonl", rs)
    # files whose requests are the same, word for word
    sig = collections.defaultdict(list)
    for f, rs in forms.items():
        h = hashlib.sha256(json.dumps([(r["situation"], r["question"], r["options"], r.get("menu2"))
                                       for r in rs]).encode()).hexdigest()
        sig[h].append(f)
    from src.mind import describe
    man = {"cases": len(cases), "pairs": len(variants) - len(skipped), "pairs_skipped": skipped[:50],
           "forms": {f: {"what": FORMS_V2[f], "rows": len(rs),
                         "mean_words": round(sum(len(r["situation"].split()) for r in rs) / max(1, len(rs)), 1),
                         "max_words": max((len(r["situation"].split()) for r in rs), default=0)}
                     for f, rs in forms.items()},
           "same_text_groups": [g for g in sig.values() if len(g) > 1],
           "notes": dict(notes), "describe_sha256": hashlib.sha256(Path(describe.__file__).read_bytes()).hexdigest(),
           "genders": {f"{b}/{n}": g for (b, n), g in sorted(GENDERS.items())},
           "scopes": {k: lab for k, lab, _ in SCOPES} | {OTHER_SCOPE[0]: OTHER_SCOPE[1]},
           "scope_counts": dict(collections.Counter(scope_of(o)[0] for c in cases for o in c["options"]))}
    write(out / "manifest.json", [])
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: man[k] for k in ("cases", "pairs", "same_text_groups", "notes", "scope_counts")}))
    print(f"{len(forms)} files in {out}" + (f", {len(pforms)} pair files in {out / 'pairs'}" if any(pforms.values()) else ""))


if __name__ == "__main__":
    main()
