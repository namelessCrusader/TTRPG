"""Item 3 — the coupling graph as DATA, with its first consumer: these tests. The graph and
the rules may never drift apart, in either direction: every declared `via` rule must exist,
and every reaction rule must be declared on some edge. Plus the finding-4 arc end-to-end:
fire → hp → death, all sim-owned (the channel the 10-point read said was missing — it is
OWNED now, and this test is the proof that stays)."""
import os

import yaml

from src.core import engine, reactions
from src.core.seed import vault

DATA = os.path.join(os.path.dirname(__file__), "..", "src", "core", "data")


def _graph():
    return yaml.safe_load(open(os.path.join(DATA, "couplings.yaml")))


def _rule_ids():
    return {r["id"] for r in yaml.safe_load(open(os.path.join(DATA, "reactions.yaml")))["reactions"]}


def _via(edge):
    return [v for v in edge["via"] if not str(v).startswith("code:")]


def test_every_declared_via_rule_exists():
    g, rules = _graph(), _rule_ids()
    ghosts = [(e["from"], e["to"], v) for e in g["edges"] for v in _via(e) if v not in rules]
    assert not ghosts, f"coupling edges citing rules that don't exist: {ghosts}"


def test_every_rule_is_declared_on_an_edge():
    g, rules = _graph(), _rule_ids()
    declared = {v for e in g["edges"] for v in _via(e)}
    undeclared = rules - declared
    assert not undeclared, (
        f"reaction rules not declared in the coupling graph: {sorted(undeclared)}. "
        "An interaction not in the table does not exist — add the edge (a reviewable decision).")


def test_channels_are_closed_vocabulary():
    g = _graph()
    chans = set(g["channels"])
    for section in ("edges", "missing"):
        for e in g[section]:
            assert e["from"] in chans and e["to"] in chans, f"unknown channel in {e}"


def test_finding_4_arc_fire_burns_and_kills_sim_owned():
    """The full arc the corpus never showed: an entity catches fire → the fire→hp edge
    drains it each tick → the hp→life edge derives death → death drops inventory. No
    author wrote any of it."""
    w, _ = vault(seed=12)
    thief = w.entities["thief"]
    thief.tags.add("inv:vial"); w.entities["vial"].tags.add("taken")   # carrying loot, for the drop
    thief.tags.add("on_fire")
    hp0 = thief.props["hp"]
    for _ in range(10):
        reactions.tick(w)
        if "dead" in thief.tags:
            break
    assert thief.props["hp"] < hp0, "fire→hp: burning drains life (living_flesh_burns)"
    assert "dead" in thief.tags and "alive" not in thief.tags, "hp→life: death is derived"
    assert "inv:vial" not in thief.tags, "death drops inventory — the vial is loose again"
    burn_events = [e for e in w.log if e.kind == "adjust_prop" and e.actor is None]
    assert burn_events, "every burn tick rode the bus as the world's own doing"
    assert not engine.invariants(w)
