"""Category-3 audit, slice 5: witnessing a deed stops alarming a bystander by a
FLAT constant. The log carries the real magnitude — the hp lost, the heat dumped
— and a witness should be rattled in proportion. A killing blow is not a bruise;
a barrel-fire is not a struck match. Same move as the noise slice: read the delta
the sim already recorded instead of thresholding it flat.

Also retires the `add_heat delta >= 100 == arson` magic threshold: any real
ignition (a spark, or heat crossing the combustion line) is arson; the SIZE then
sets how loudly it registers, rather than a number deciding arson-or-not.
"""

from src.core import mind
from src.core.seed import guildhall
from src.core.state import LogEntry


def _witness_bump(world, deed_entry):
    """Salience a fresh, watching witness gains from one deed entry."""
    npc = world.entities["guard"]
    npc.mind["salience"] = 0.0
    npc.mind["memory"] = []
    npc.mind["seen_upto"] = len(world.log)
    world.log.append(deed_entry)
    # put the guard on top of the deed so line-of-sight is trivially satisfied
    pos = deed_entry.data.get("pos") or world.entities[deed_entry.data["t"]].pos
    npc.pos = pos
    mind._perceive_deeds(world, npc)
    return npc.mind["salience"]


# DESIGN: a heavier blow rattles a witness more than a glancing one — salience
# reads the real hp delta, it doesn't flatten every assault to one number.
def test_assault_alarm_scales_with_damage():
    world, _ = guildhall(seed=0)
    victim = world.entities["apprentice"]
    light = LogEntry(tick=1, kind="adjust_prop", actor="player",
                     data={"t": victim.id, "prop": "hp", "delta": -2.0}, cause="")
    heavy = LogEntry(tick=2, kind="adjust_prop", actor="player",
                     data={"t": victim.id, "prop": "hp", "delta": -12.0}, cause="")
    s_light = _witness_bump(world, light)
    s_heavy = _witness_bump(world, heavy)
    assert s_heavy > s_light, f"a savage blow alarms more: {s_heavy} vs {s_light}"


# DESIGN: a raging fire registers harder than a small flame. Both are arson; the
# heat delta sets how loudly it lands.
def test_arson_alarm_scales_with_heat():
    world, _ = guildhall(seed=0)
    small = LogEntry(tick=1, kind="add_heat", actor="player",
                     data={"pos": (5, 5, 0), "delta": 120.0}, cause="")
    big = LogEntry(tick=2, kind="add_heat", actor="player",
                   data={"pos": (5, 5, 0), "delta": 600.0}, cause="")
    assert _witness_bump(world, big) > _witness_bump(world, small), \
        "an inferno alarms more than a licking flame"


# DESIGN: arson is ANY real ignition, not a magic heat number. A spark counts,
# and heat crossing the combustion line counts — the size only sets loudness.
def test_a_spark_still_reads_as_arson():
    world, _ = guildhall(seed=0)
    spark = LogEntry(tick=1, kind="spark", actor="player",
                     data={"pos": (5, 5, 0)}, cause="")
    deed = mind._classify_deed(world, spark)
    assert deed and deed["deed"] == "arson", f"a spark is arson: {deed}"
    assert _witness_bump(world, spark) > 0, "and a witness notices it"


# DESIGN: even the biggest deed can't exceed the salience ceiling, and the base
# alarm still orders the KINDS (assault graver than a small theft-adjacent event).
def test_magnitude_scaling_stays_bounded():
    world, _ = guildhall(seed=0)
    victim = world.entities["apprentice"]
    huge = LogEntry(tick=1, kind="adjust_prop", actor="player",
                    data={"t": victim.id, "prop": "hp", "delta": -999.0}, cause="")
    assert _witness_bump(world, huge) <= 100.0, "salience never breaches its ceiling"
