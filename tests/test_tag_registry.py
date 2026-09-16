"""D7 — the tag registry as an INVARIANT: a tag may be emitted only if something READS it.

'Load-bearing tags' stops being a cleanup pass and becomes CI: the scanner harvests every tag
the system can emit (yaml rule `add_tag`, pack verbs, literal `fx.set_tag` in code, seed
`tags={...}`) and every tag something consumes (yaml `has_tag`/`missing_tag`/`cell_has_tag`/
verb `tag:` targeting, code `"x" in ....tags` membership). Emitted-but-never-read tags are DEBT:
they live in the ledger below with a reason, and ADDING a new one fails CI. (The invented-status
failure — `blinded`, `blinded_turns` — becomes unreachable the moment the pen offers only
registry tags; until then this guards our own content.)"""
import os
import re

import yaml

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "core")
DATA = os.path.join(SRC, "data")

# mechanism namespaces — structured prefixes, not vocabulary tags
_PREFIX = ("inv:", "holds:", "soaked_")

# ── LANDMARK CELL TAGS ───────────────────────────────────────────────────────
# These HAVE a real mechanical consumer, just a generic one: any non-floor cell tag makes
# the cell NOTABLE in perceive() (it enters the belief slice and the LOOK description).
# For a cell, being-worth-seeing IS the read. Entity tags get no such pass — an entity
# renders regardless, so a decoration tag gates nothing.
LANDMARK = {"door", "foundation", "vault", "workbench"}

# ── DEBT LEDGER ──────────────────────────────────────────────────────────────
# Tags emitted today that nothing reads. Each needs a consumer or deletion. Shrink this
# list; never grow it silently (growth = a failing test = a decision).
DEBT = {
    "calm",          # set/cleared as a mood marker, but band() owns the calm/agitated
                     # mechanics — the TAG gates nothing. Either read it (menu: 'calm'
                     # animals approachable?) or stop emitting it.
    "crawler",       # DCC flavor tag on the player; the pack's concepts list is words, not
                     # a tag read. Candidate consumer: DCC rules keyed to crawlers only.
}


def _walk(node, found):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("add_tag", "remove_tag") and isinstance(v, str):
                found["emit"].add(v)
            elif k in ("has_tag", "missing_tag", "cell_has_tag", "has_material_tag") and isinstance(v, str):
                found["read"].add(v)
            elif k == "tag" and isinstance(v, str):          # pack-verb `tag:` targeting = a read
                found["read"].add(v)
            else:
                _walk(v, found)
    elif isinstance(node, list):
        for x in node:
            _walk(x, found)


def _harvest():
    found = {"emit": set(), "read": set()}
    # 1) yaml rules + packs (content: both emit and read)
    for root, _dirs, files in os.walk(DATA):
        for f in files:
            if f.endswith(".yaml"):
                _walk(yaml.safe_load(open(os.path.join(root, f))), found)
    code = ""
    for f in os.listdir(SRC):
        if f.endswith(".py"):
            code += open(os.path.join(SRC, f)).read()
    # 2) code emissions: fx.set_tag("id-expr", "tag") literals and tags={"a", "b"} seeds
    for m in re.finditer(r'set_tag\([^,)]+,\s*[f]?"([a-z_]+)"', code):
        found["emit"].add(m.group(1))
    for m in re.finditer(r'tags=\{([^}]*)\}', code):
        for t in re.findall(r'"([a-z_:]+)"', m.group(1)):
            found["emit"].add(t)
    for m in re.finditer(r'\.tags\.add\(\s*"([a-z_]+)"', code):
        found["emit"].add(m.group(1))
    # 3) code reads: "tag" in <...>tags / tags membership & set-intersection forms
    for m in re.finditer(r'"([a-z_]+)"\s+(?:not\s+)?in\s+[^\n]*?tags', code):   # non-greedy:
        found["read"].add(m.group(1))                  # nearest `tags`, not the line's last
    for m in re.finditer(r'tags\s*&\s*\{([^}]*)\}', code):
        for t in re.findall(r'"([a-z_]+)"', m.group(1)):
            found["read"].add(t)
    # material-template tags (state.MATERIALS) are read via has_material_tag paths
    from src.core.state import MATERIALS
    for spec in MATERIALS.values():
        found["read"] |= set(spec.get("tags", ()))
    return found


def test_every_emitted_tag_has_a_consumer_or_is_ledgered():
    found = _harvest()
    emitted = {t for t in found["emit"] if not t.startswith(_PREFIX)}
    dead = emitted - found["read"] - LANDMARK - DEBT
    assert not dead, (
        f"tags emitted but never read (and not in the DEBT ledger): {sorted(dead)}. "
        "Either give each a consumer (a rule/menu predicate that reads it), delete the "
        "emission, or add it to DEBT here — with a reason — as a conscious decision.")


def test_debt_only_shrinks():
    found = _harvest()
    emitted = {t for t in found["emit"] if not t.startswith(_PREFIX)}
    paid = DEBT - (emitted - found["read"] - LANDMARK)
    assert not paid, (
        f"DEBT entries that now have consumers (or are no longer emitted): {sorted(paid)}. "
        "Remove them from the ledger — debt is paid, keep the ledger honest.")
