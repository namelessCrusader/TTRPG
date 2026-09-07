"""Combat physics — the seam where blows become deltas.

Everything here READS the sim rather than faking it (the Category-3 rule): to-hit
reads both fighters' condition tags, damage reads the attacker's weapon type
against the target's material. Nothing is authored per-scene; it all falls out of
tags every entity already carries. `composer.to_effects` calls in — this module
holds no step-dispatch, only the numbers a contest resolves to.

The one authored thing is the type×material RULE (GURPS lesson): a small table
that is the *rule*, not a faked fact, applied to material tags the sim stores.
"""

from __future__ import annotations

from .state import mat_tags


def _skill(e, name):
    return e.props.get(name, 0.5)


def to_hit(world, attacker, victim, base):
    """To-hit READS the sim instead of faking it from skill alone. `base` is the
    skill curve; the condition tags both sides already wear nudge it — GENTLY, so
    might stays the spine. A scared/staggered/unaware victim is an opening; a
    braced (hostile) one closes it; a shaken or burning attacker swings worse.
    Every input is a fact another subsystem set — none of it is invented here."""
    p = base
    vt, at = victim.tags, attacker.tags
    if "afraid" in vt:       p += 0.10          # a panicking guard drops their guard
    if "off_balance" in vt:  p += 0.10          # a stagger is an opening
    if "hostile" not in vt:  p += 0.10          # hasn't squared up to you — caught flat
    if "afraid" in at:       p -= 0.10          # your own nerves throw the swing
    if "on_fire" in at:      p -= 0.10          # you can't aim while alight
    return max(0.05, min(0.95, p))


# The type×material RULE (GURPS lesson): a small authored table — it's the rule,
# not a per-scene fact — applied to the material the sim already stores. Rows are
# damage types, columns keyed by the material's own tags. Default 1.0 = "no
# special interaction", so unlisted materials still take normal damage.
_DMG_TABLE = {
    "cut":   {"biological": 1.5, "wooden": 1.0, "rigid": 0.4, "metal": 0.3},
    "crush": {"biological": 1.0, "wooden": 1.2, "rigid": 0.7, "metal": 0.5, "glass": 1.6},
    "pierce":{"biological": 1.2, "wooden": 0.9, "rigid": 0.6, "metal": 0.5},
}


def dmg_mult(dtype, material):
    """How well `dtype` damage bites `material` — read from the material's tags,
    so a new material with the right tags is covered without touching this."""
    tags = mat_tags(material) | ({"glass"} if material == "glass" else set())
    row = _DMG_TABLE.get(dtype, {})
    hits = [m for tag, m in row.items() if tag in tags]
    return min(hits) if hits else 1.0           # the most-resistant matching property wins


def _wielded(world, actor):
    """The weapon in hand, if any: an entity the actor carries (inv:<id>) tagged
    weapon. None = empty hands."""
    for tag in actor.tags:
        if tag.startswith("inv:"):
            w = world.entities.get(tag[4:])
            if w and "weapon" in w.tags:
                return w
    return None


def _dtype_of(item):
    """Damage type of a weapon or thrown thing, read from its tags: edged→cut,
    pointed→pierce, else crush (bare hands, a club, a hurled rock)."""
    if item is None:            return "crush"
    if "edged" in item.tags:    return "cut"
    if "pointed" in item.tags:  return "pierce"
    return "crush"


def attack_dtype(world, actor):
    """Damage type READ from what's in hand."""
    return _dtype_of(_wielded(world, actor))


def attack_damage(world, actor, material, band, dtype=None):
    """Base magnitude from the wielded weapon (bare hands are weaker), scaled by
    the band, then by how the damage type meets the target's material. `dtype`
    can be forced — a SMASH is always a crush blow, whatever's in hand."""
    w = _wielded(world, actor)
    base = (w.props.get("damage", 3.0) if w else 2.0)
    band_mult = {"crit": 1.6, "clean": 1.0, "partial": 0.6}.get(band, 0.0)
    return base * band_mult * dmg_mult(dtype or attack_dtype(world, actor), material)


def thrown_damage(item, material, crit):
    """A thrown weapon is a ranged attack: its base `damage` meets the victim's
    material through the type×material rule, +crit bonus."""
    base = item.props.get("damage", 1.0) + (2.0 if crit else 0.0)
    return base * dmg_mult(_dtype_of(item), material)


def throw_hit(world, thrower, victim, dist):
    """Ranged to-hit: finesse and distance set the base; if a person stands at the
    mark, their condition tilts it exactly as in melee (an unaware target can't
    dodge). No victim → pure skill throw."""
    base = 0.5 + 0.45 * _skill(thrower, "finesse") - 0.05 * (dist - 1)
    if victim is None:
        return max(0.05, min(0.95, base))
    return to_hit(world, thrower, victim, base)
