"""Item 8 — the reachable-state counter: the feedback loop for world design. Score an edit
by whether it makes the world more GENERATIVE, in milliseconds, no author in the loop.
First measurement (vault, depth 2): baseline 111 distinct states; +2 part-bearing objects
→ 312 (combinatorial — the design doc's central claim, measured); parts stripped → 90."""
from src.core import reach
from src.core.seed import vault
from src.core.state import Entity


def _rich(w):
    w.entities["brazier"] = Entity("brazier", "iron brazier", (1, 0, 0), material="metal",
                                   tags={"item"}, props={"hp": 4.0},
                                   parts={"fuel_burn": {"ticks": 5, "heat": 60.0}})
    w.entities["barrel_rain"] = Entity("barrel_rain", "rain barrel", (0, 1, 0), material="wood",
                                       tags={"item", "container"}, props={"hp": 4.0},
                                       parts={"liquid_volume": {"mat": "water", "ml": 600.0}})
    return w


def test_counter_is_deterministic():
    a = reach.count_reachable(vault(seed=12)[0], "player", depth=2)
    b = reach.count_reachable(vault(seed=12)[0], "player", depth=2)
    assert a == b, "same world, same seed, same counts — the sweep is deterministic"


def test_generativity_ordering():
    """The counter's whole job: rank world edits. More parts → combinatorially more
    reachable states; a world stripped of parts is flatter than the baseline."""
    _, base = reach.count_reachable(vault(seed=12)[0], "player", depth=2)
    _, rich = reach.count_reachable(_rich(vault(seed=12)[0]), "player", depth=2)
    flat_w, _ = vault(seed=12)
    for e in flat_w.entities.values():
        e.parts = {}
    _, flat = reach.count_reachable(flat_w, "player", depth=2)
    assert flat < base < rich, f"generativity ordering broken: {flat} < {base} < {rich} expected"
    assert rich > base * 2, "two part-bearing objects should grow the space combinatorially, not linearly"


def test_salience_bands_ignore_physics_noise():
    """Heat drift alone must not create 'new' states — salience is what play can see and use."""
    w, _ = vault(seed=12)
    h0 = reach.salient_hash(w)
    for c in w.cells.values():
        c.heat += 0.7                                   # ambient drift, sub-band
    assert reach.salient_hash(w) == h0, "banded salience: noise is not novelty"
    w.entities["thief"].tags.add("on_fire")             # a real, visible difference
    assert reach.salient_hash(w) != h0
