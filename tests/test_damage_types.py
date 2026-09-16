"""Category-3 audit, slice 3: attack damage stops being a flat -5/-3 that ignores
BOTH what you swing and what you hit. The GURPS lesson, computed not tabled:
damage TYPE (cut / crush / pierce) interacts with the target's MATERIAL, which
the sim already stores on every entity and cell.

  - cut bites flesh, skates off stone/metal,
  - crush shatters brittle things (glass, wood) and is what bare hands deal,
  - pierce is the even all-rounder.

The type is READ from the wielded weapon's tags (a metal belt-knife is edged →
cut); with empty hands you crush. Base magnitude is READ from the weapon's own
`damage` prop. None of this is authored per-scene — it falls out of material
tags every entity already carries.
"""

from src.core import combat


# DESIGN: the type×material rule is a small AUTHORED table (it's the RULE, not a
# faked fact) applied to state the sim holds. cut loves flesh, hates rigid metal.
def test_cut_bites_flesh_but_skates_off_metal():
    fl = combat.dmg_mult("cut", "flesh")
    st = combat.dmg_mult("cut", "metal")
    assert fl > 1.0, f"a blade opens flesh: {fl}"
    assert st < 1.0, f"a blade skates off metal: {st}"
    assert fl > st * 2, "the difference is real, not a rounding nudge"


# DESIGN: crush is the brittle-breaker — it shatters glass and wood far better
# than it dents metal or stone.
def test_crush_shatters_brittle_things():
    glass = combat.dmg_mult("crush", "glass")
    wood = combat.dmg_mult("crush", "wood")
    metal = combat.dmg_mult("crush", "metal")
    assert glass > 1.0 and wood >= 1.0, f"brittle breaks: glass {glass}, wood {wood}"
    assert glass > metal, f"a club shatters glass better than it dents steel: {glass} vs {metal}"


# DESIGN: what you SWING sets the damage type. A metal belt-knife is edged → cut.
# Empty hands → crush. This reads the wielded weapon's tags, not a per-scene flag.
def test_weapon_sets_the_damage_type():
    from src.core.seed import guildhall
    world, player = guildhall(seed=0)
    knife = world.entities["knife"]
    knife.tags.add("edged")                          # a blade cuts
    assert combat.attack_dtype(world, player) == "cut", "wielding a blade → you cut"
    player.tags.discard("inv:knife")                 # drop it
    assert combat.attack_dtype(world, player) == "crush", "bare hands crush"


# DESIGN: the whole thing composes — a knife (cut, base 5) vs flesh lands HARDER
# than the same knife vs a metal-bodied target, because material resists type.
def test_full_attack_damage_reads_weapon_and_target():
    from src.core.seed import guildhall
    world, player = guildhall(seed=0)
    world.entities["knife"].tags.add("edged")
    soft = combat.attack_damage(world, player, material="flesh", band="clean")
    hard = combat.attack_damage(world, player, material="metal", band="clean")
    assert soft > hard, f"steel resists the blade flesh can't: flesh {soft} vs metal {hard}"
    crit = combat.attack_damage(world, player, material="flesh", band="crit")
    assert crit > soft, "a crit still hits harder than a clean hit"
    assert soft > 0, "a clean hit always does something"
