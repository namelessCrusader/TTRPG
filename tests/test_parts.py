"""D13 parts — the behavior layer: state that ACTS over ticks, which tags/props can't
express. Contract: a part's private counters mutate internally (sim physics), but every
world-visible consequence rides the effect bus (logged, replayable, actor=None)."""
from src.core import engine, reactions
from src.core.seed import vault
from src.core.state import Entity


def _lit_brazier(w, pos=(3, 0, 0), ticks=3):
    b = Entity("brazier", "iron brazier", pos, material="metal",
               tags={"item"}, parts={"fuel_burn": {"ticks": ticks, "heat": 60.0}})
    w.entities["brazier"] = b
    return b


def test_fuel_burn_warms_then_gutters():
    w, _ = vault(seed=12)
    b = _lit_brazier(w, ticks=3)
    cell = w.cell(b.pos)
    h0 = cell.heat
    for _ in range(3):
        reactions.tick(w)
    assert cell.heat > h0 + 30, "a fueled flame keeps its own cell hot"
    assert "spent" not in b.tags and b.parts["fuel_burn"]["ticks"] == 0

    reactions.tick(w)                                   # the fuel is gone: it gutters, once
    assert "spent" in b.tags
    gutter = [e for e in w.log if "gutters out" in e.cause]
    assert len(gutter) == 1, "guttering is a single bus event, not a repeated one"

    h_spent = cell.heat
    reactions.tick(w)
    assert cell.heat <= h_spent, "spent means spent — no further heat is added"
    assert not engine.invariants(w)


def test_world_visible_consequences_ride_the_bus():
    w, _ = vault(seed=12)
    _lit_brazier(w, ticks=1)
    reactions.tick(w)
    heats = [e for e in w.log if e.kind == "add_heat"]
    assert heats and all(e.actor is None for e in heats), \
        "part behavior is the world's own doing (actor=None), logged like all physics"


def test_parts_advertise_affordances_to_the_menu():
    """D2 in miniature: the option exists BECAUSE the component does — no verb table.
    Remove/spend the part and the option vanishes; consuming it writes part state
    through the bus (set_part), never by direct mutation."""
    w, p = vault(seed=12)
    _lit_brazier(w, pos=(1, 0, 0), ticks=5)            # adjacent to the player at (0,0)
    labels = [o.label for o in engine.affordance_menu(w, p)]
    assert "snuff the iron brazier" in labels

    snuff = next(o for o in engine.affordance_menu(w, p) if "snuff" in o.label)
    engine.step(w, snuff, actor=p)
    b = w.entities["brazier"]
    assert b.parts["fuel_burn"]["ticks"] == 0 and "spent" in b.tags
    assert any(e.kind == "set_part" and e.actor == "player" for e in w.log), \
        "consuming an affordance writes part state through the bus, attributed to the actor"

    labels = [o.label for o in engine.affordance_menu(w, p)]
    assert "snuff the iron brazier" not in labels, "a spent flame affords nothing"


def test_part_affordances_are_perception_gated():
    w, p = vault(seed=12)
    _lit_brazier(w, pos=(5, 3, 0), ticks=5)            # far corner: real but out of reach/sight
    labels = [o.label for o in engine.affordance_menu(w, p)]
    assert not any("snuff" in l for l in labels), \
        "a part's offer rides the same fresh-belief + reach gates as everything else"


def _rain_barrel(w, pos=(1, 0, 0), ml=600.0):
    b = Entity("barrel_rain", "rain barrel", pos, material="wood",
               tags={"item", "container"}, props={"hp": 4.0},   # hp REQUIRED: the burst rule
               parts={"liquid_volume": {"mat": "water", "ml": ml}})  # reads a missing prop as 0
    w.entities["barrel_rain"] = b                                    # = burst-on-arrival
    return b


def test_liquid_volume_tip_and_draw_sim_owned_amounts():
    """A resource part: no stepper, acts only when acted upon. The author picks the verb;
    the sim owns every number (a draught, a full tip) — D6 in its first concrete case."""
    w, p = vault(seed=12)
    b = _rain_barrel(w)                                 # adjacent to the player at (0,0)
    p.tags.discard("inv:water")                         # hands free to draw
    labels = [o.label for o in engine.affordance_menu(w, p)]
    assert "tip the rain barrel over" in labels and "draw water from the rain barrel" in labels

    draw = next(o for o in engine.affordance_menu(w, p) if "draw water from the rain barrel" in o.label)
    engine.step(w, draw, actor=p)
    assert "inv:water" in p.tags and b.parts["liquid_volume"]["ml"] == 400.0
    labels = [o.label for o in engine.affordance_menu(w, p)]
    assert not any("draw water from the rain barrel" in l for l in labels), "already holding water — no second draw"

    tip = next(o for o in engine.affordance_menu(w, p) if "tip the rain barrel" in o.label)
    engine.step(w, tip, actor=p)
    assert b.parts["liquid_volume"]["ml"] == 0.0
    total = sum(c.fluids.get("water", 0.0) for c in w.cells.values())
    assert w.cell(b.pos).fluids.get("water", 0.0) > 0, "water pooled where it was tipped"
    assert total > 300.0, "the tipped contents are in the world (spread by the fluid physics)"
    labels = [o.label for o in engine.affordance_menu(w, p)]
    assert not any("rain barrel" in l and (l.startswith("tip") or l.startswith("draw"))
                   for l in labels), "an empty vessel affords nothing"
    assert not engine.invariants(w)


def test_partless_entities_cost_nothing():
    w, _ = vault(seed=12)      # seed CONTAINERS carry liquid_volume now (the holds: migration);
    people = [w.entities[i] for i in ("player", "thief", "guard", "vial")]
    assert all(not e.parts for e in people), "people and plain items stay part-less"
    reactions.tick(w)          # the stepper no-ops harmlessly for them
    assert not engine.invariants(w)
