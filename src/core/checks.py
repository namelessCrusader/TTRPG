"""Adjudication: graded bands, the consequence menu, the fact oracle, Clocks.

One organism (TTRPG_MECHANICS.md): every uncertain thing resolves as a seeded
roll → a BAND (crit/clean/partial/miss) → and any band below clean draws ONE
consequence from a closed menu, emitted as ordinary bus deltas. Unknowable facts
resolve once on a likelihood ladder and become permanent canon. Clocks are the
universal counter that complications feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import effects as fx
from .state import World

# optional LM hook: (scene, action, question, [(id, desc)]) -> index.
# None = seeded dice (deterministic; what fuzz and tests run under).
SELECTOR = None

# ── outcome bands (Blades): one roll, four grades ──
def roll_band(world: World, p: float) -> str:
    """p = chance of clean-or-better. Partial extends 0.3 past p: the action
    works but costs. Beyond that it's a miss — which also costs."""
    r = world.rng.random()
    if r < p * 0.30:
        return "crit"
    if r < p:
        return "clean"
    if r < min(0.97, p + 0.30):
        return "partial"
    return "miss"


# ── the closed consequence menu (5 types; content grows, the menu never does) ──
def consequence(world: World, actor, kind_hint: str, target_pos, victim_id=None) -> list:
    """Pick ONE consequence fitting the situation (seeded). Every consequence is
    ordinary deltas + a 'consequence' marker on the log — never silent."""
    menu = []
    if victim_id is not None:
        menu.append(("worse_position", [
            fx.set_tag(actor.id, "off_balance", "you overreach and stumble off balance")]))
        menu.append(("harm", [
            fx.adjust_prop(actor.id, "hp", -1.0, "you take a scrape in the exchange")]))
    menu.append(("noise", [
        fx.adjust_prop(actor.id, "hp", 0.0, "the commotion rings out — heads turn")]))
    if "alarm" in world.clocks:
        menu.append(("complication", [fx.clock_tick("alarm", 1)]))
    if SELECTOR is not None and len(menu) > 1:
        try:                                    # LM picks the FITTING cost; dice as fallback
            scene = "; ".join(e.cause for e in world.log[-6:] if e.cause)
            i = SELECTOR(scene, kind_hint, "Which cost best fits this moment?",
                         [(k, d[0].cause or k) for k, d in menu])
            kind, deltas = menu[i if 0 <= i < len(menu) else 0]
        except Exception:
            kind, deltas = world.rng.choice(menu)
    else:
        kind, deltas = world.rng.choice(menu)
    mark = fx.Effect("consequence", {"kind": kind, "of": kind_hint}, "")
    mark.actor = actor.id
    out = [mark] + deltas
    for e in out:
        e.actor = actor.id
    return out


# ── the fact oracle (Ironsworn): unknowables resolve ONCE, then they're canon ──
LADDER = {"certain": 0.93, "likely": 0.75, "even": 0.5, "unlikely": 0.25, "rare": 0.08}


def oracle(world: World, question: str, likelihood: str = "even") -> bool:
    key = " ".join(question.lower().split())
    if key in world.facts:
        return world.facts[key]
    truth = world.rng.random() < LADDER.get(likelihood, 0.5)
    eff = fx.Effect("fact", {"q": key, "truth": truth},
                    f"[canon] {question}? — {'yes' if truth else 'no'}")
    fx.apply(world, eff)
    return truth


# ── the universal Clock (Blades): one counter for alarms, quests, doom ──
@dataclass
class Clock:
    id: str
    size: int
    fill: int = 0
    on_full: str = ""
    fired: bool = False


def add_clock(world: World, cid: str, size: int, on_full: str = "") -> Clock:
    c = Clock(cid, size, on_full=on_full)
    world.clocks[cid] = c
    return c
