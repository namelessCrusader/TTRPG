"""reach.py — the REACHABLE-STATE COUNTER (design doc item 8): how many distinct salient
world-states can an actor reach in k beats? No author, no LM, no harvesting — pure dynamics.

This is the feedback loop for world design: score an edit (a new part, a coupling edge, a
property) by whether it makes the world MORE GENERATIVE — count before, count after, compare —
in milliseconds, before spending a single rollout. Generative worlds grow combinatorially with
the objects present; flat ones grow linearly (design doc §7, row 1).

Salience: two states are "the same" if they agree on what PLAY can see and use — entity
positions/tags/hp-integer/part-state, and cells' fire/fluid/material. Continuous residue
(heat drift, ml dribbles) is banded so physics noise doesn't inflate the count.

Deliberately NOT engine.step(): that path harvests traces and narrates. The sweep applies an
option's effects straight to the bus (steps via the composer), then one reactions.tick — the
same dynamics, none of the side channels. Social options are skipped (they need words)."""
import copy

from . import effects as fx
from . import engine, reactions


def salient_hash(world):
    ents = tuple(
        (eid, e.pos, tuple(sorted(e.tags)), round(e.props.get("hp", 0.0)),
         tuple(sorted((k, tuple(sorted((pk, round(pv) if isinstance(pv, float) else pv)
                                       for pk, pv in p.items())))
                      for k, p in e.parts.items())))
        for eid, e in sorted(world.entities.items()))
    cells = tuple(
        (pos, c.material, "on_fire" in c.tags,
         tuple(sorted(m for m, ml in c.fluids.items() if ml > 5.0)))
        for pos, c in sorted(world.cells.items())
        if c.fluids or "on_fire" in c.tags)
    return hash((ents, cells))


def _apply(world, actor_id, opt):
    """One branch: deep-copy, apply the option the sweep-safe way, tick physics once."""
    w = copy.deepcopy(world)
    a = w.entities[actor_id]
    if opt.steps:
        from . import composer
        effs, _ = composer.to_effects(w, a, opt.steps)
        fx.apply_all(w, effs)
    elif opt.effects:
        effs = copy.deepcopy(opt.effects)
        for ef in effs:
            ef.actor = actor_id
        fx.apply_all(w, effs)
    reactions.tick(w)
    return w


def count_reachable(world, actor_id, depth=2):
    """BFS over the actor's affordance menus → (#distinct states at the horizon,
    #distinct states seen anywhere). Graph search: revisited states are not re-expanded."""
    seen = {salient_hash(world)}
    frontier = [world]
    for _ in range(depth):
        nxt = []
        for w in frontier:
            a = w.entities.get(actor_id)
            if a is None or "dead" in a.tags:
                continue                                   # a dead actor reaches nothing further
            for opt in engine.affordance_menu(w, a):
                if opt.social or not (opt.effects or opt.steps):
                    continue                               # social needs words; look/wait branch nothing new
                w2 = _apply(w, actor_id, opt)
                h = salient_hash(w2)
                if h not in seen:
                    seen.add(h)
                    nxt.append(w2)
        frontier = nxt
    return len(frontier), len(seen)
