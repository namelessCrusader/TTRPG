"""Category-3 audit, slice 2: the melee to-hit curve stops faking "how likely is
this to land" with skill alone. `0.35 + 0.55·might` read exactly ONE tag (the
attacker's own off_balance) and NOTHING about the victim — though the sim tracks
the victim's condition precisely.

`_to_hit(world, attacker, victim, base)` now reads the REAL, sim-maintained
condition tags:
  - a victim who is `afraid` or `off_balance` is easier to hit,
  - a victim still `calm` (unaware of you) is a sitting duck; one already
    `hostile` (braced) defends better,
  - an attacker who is `afraid` or `on_fire` swings worse.

These are all facts other subsystems already set (mind.py flight/investigate,
the fire tag). We assert ORDERING and clamping, not exact floats, so the
magnitudes stay tunable without churning tests.
"""

from src.core import combat
from src.core.seed import vault


def _p(world, player, victim_tags, base=0.5):
    """To-hit against a fresh guard wearing exactly `victim_tags` (+person/alive).
    Computed eagerly so callers can compare several conditions without the shared
    entity's later mutations leaking backward."""
    g = world.entities["guard"]
    g.tags = {"person", "alive"} | set(victim_tags)
    return combat.to_hit(world, player, g, base)


# DESIGN: a frightened or staggered victim is easier to hit than a composed one.
def test_shaken_victim_is_easier_to_hit():
    world, player = vault(seed=0)
    p_steady = _p(world, player, {"hostile"})               # braced, steady
    p_scared = _p(world, player, {"hostile", "afraid"})
    p_stumble = _p(world, player, {"hostile", "off_balance"})
    assert p_scared > p_steady, f"fear opens a guard: {p_scared} vs {p_steady}"
    assert p_stumble > p_steady, f"a stagger opens a guard: {p_stumble} vs {p_steady}"


# DESIGN: an unaware target (still calm, hasn't reacted to you) is a sitting duck;
# one already braced for the fight (hostile) is the hardest to land on.
def test_unaware_target_beats_a_braced_one():
    world, player = vault(seed=0)
    p_unaware = _p(world, player, {"calm"})                 # never noticed you
    p_braced = _p(world, player, {"hostile"})               # squared up
    assert p_unaware > p_braced, f"the ambush lands easier: {p_unaware} vs {p_braced}"


# DESIGN: your OWN condition drags your swing — panic and pain make you miss.
def test_impaired_attacker_swings_worse():
    world, player = vault(seed=0)
    steady = _p(world, player, {"hostile"})
    player.tags.add("afraid")
    p_afraid = _p(world, player, {"hostile"})
    player.tags.discard("afraid")
    player.tags.add("on_fire")
    p_burning = _p(world, player, {"hostile"})
    assert p_afraid < steady, f"a shaking hand misses more: {p_afraid} vs {steady}"
    assert p_burning < steady, f"you can't aim while alight: {p_burning} vs {steady}"


# DESIGN: modifiers are GENTLE nudges — skill stays the spine. Even the worst
# stack of penalties can't drive a competent swing to a coin-flip-or-worse, and
# the best opening can't guarantee a hit. And the result is always a probability.
def test_modifiers_stay_bounded_and_gentle():
    world, player = vault(seed=0)
    # every victim-opening at once, attacker pristine → high but not certain
    p_open = _p(world, player, {"afraid", "off_balance", "calm"}, base=0.6)
    assert p_open <= 0.95, f"never a guaranteed hit: {p_open}"
    # every attacker penalty, victim braced → low but not hopeless if skilled
    player.tags |= {"afraid", "on_fire"}
    p_bad = _p(world, player, {"hostile"}, base=0.6)
    assert 0.05 <= p_bad, f"skill still buys you a chance: {p_bad}"
    assert 0.0 <= p_open <= 1.0 and 0.0 <= p_bad <= 1.0, "always a probability"
