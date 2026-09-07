"""Category-3 audit, slice 4: the THROW resolution stops faking two facts the sim
holds. Hit chance was pure finesse+distance — blind to whether the target could
see it coming. Thrown-weapon damage was a flat `damage` prop — blind to what it
struck. Both now READ state, reusing the melee helpers: a throw is a ranged
attack, so it obeys the same awareness and material rules.
"""

from src.core import combat
from src.core.seed import guildhall


# DESIGN: an unaware or shaken target is easier to HIT with a thrown thing, same
# as in melee — a person who never saw it coming can't twist away.
def test_throw_hit_reads_target_awareness():
    world, player = guildhall(seed=0)
    victim = world.entities["apprentice"]
    victim.tags = {"person", "alive", "hostile"}            # braced, watching the throw
    p_braced = combat.throw_hit(world, player, victim, dist=2)
    victim.tags = {"person", "alive", "calm"}               # never noticed you wind up
    p_unaware = combat.throw_hit(world, player, victim, dist=2)
    assert p_unaware > p_braced, f"the unseen throw lands easier: {p_unaware} vs {p_braced}"


# DESIGN: distance still matters — the same target is harder to hit farther off.
def test_throw_still_falls_off_with_distance():
    world, player = guildhall(seed=0)
    victim = world.entities["apprentice"]
    victim.tags = {"person", "alive", "hostile"}
    near = combat.throw_hit(world, player, victim, dist=1)
    far = combat.throw_hit(world, player, victim, dist=3)
    assert near > far, f"a long throw is chancier: near {near} vs far {far}"


# DESIGN: with no one at the target cell, it's just a finesse+distance throw —
# nothing to read, so the awareness term is simply absent.
def test_throw_at_empty_cell_is_pure_skill():
    world, player = guildhall(seed=0)
    p = combat.throw_hit(world, player, None, dist=1)
    assert 0.0 < p <= 0.95, f"a throw at empty air still has odds: {p}"


# DESIGN: a thrown weapon is a ranged attack — its damage reads the victim's
# material, so a hurled knife (cut) bites flesh harder than it dents a metal foe.
def test_thrown_damage_reads_target_material():
    world, player = guildhall(seed=0)
    knife = world.entities["knife"]
    knife.tags.add("edged")
    soft = combat.thrown_damage(knife, material="flesh", crit=False)
    hard = combat.thrown_damage(knife, material="metal", crit=False)
    assert soft > hard, f"a thrown blade bites flesh, skates off steel: {soft} vs {hard}"
    assert combat.thrown_damage(knife, "flesh", crit=True) > soft, "a crit hits harder"
