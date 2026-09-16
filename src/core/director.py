"""The Director: chaos dial, lull detection, GM moves (TTRPG_MECHANICS #3/#5).

Mythic's chaos factor + PbtA's GM moves, compacted: one bounded float paced by
the drama pulse; when nothing notable happens for too long, a move fires from a
closed menu (seeded pick — an LM can be swapped in as the selector later). Soft
moves are WARNINGS with a fuse: ignored, they commit as the hard move; heeded,
they cancel. Every move is ordinary deltas plus a `gm_move` marker on the log.
"""

from __future__ import annotations

from . import effects as fx
from .spatial import l1
from .state import World

LULL_BASE = 16     # ticks of dead air tolerated at chaos 1; high chaos tolerates less


def _mark(world, move_id, cause=""):
    fx.apply(world, fx.Effect("gm_move", {"move": move_id}, cause))


# ── the move menu: (id, eligible?, fire) — content grows, the shape never does ──
def _mv_alarm_creeps(world, player):
    c = world.clocks.get("alarm")
    if not c or c.fired or c.fill >= c.size:
        return None
    return lambda: [fx.clock_tick("alarm", 1, "the household grows watchful...")]


def _mv_rumor_stirs(world, player):
    talkers = [e for e in world.entities.values()
               if "person" in e.tags and "dead" not in e.tags and e.mind.get("memory")]
    if not talkers:
        return None
    npc = world.rng.choice(sorted(talkers, key=lambda e: e.id))

    def fire():
        npc.mind["salience"] = min(100.0, npc.mind.get("salience", 0) + 25)
        return [fx.speech(npc.id, "...", f"{npc.name} broods on what they know")]
    return fire


def _mv_fresh_purpose(world, player):
    idlers = [e for e in world.entities.values()
              if "person" in e.tags and "dead" not in e.tags and not e.mind.get("goal")]
    if not idlers:
        return None
    npc = world.rng.choice(sorted(idlers, key=lambda e: e.id))

    def fire():
        npc.mind["salience"] = min(100.0, npc.mind.get("salience", 0) + 20)
        return [fx.speech(npc.id, "...", f"{npc.name}'s eyes narrow with fresh purpose")]
    return fire


def _mv_timber_groans(world, player):
    """The soft move: warn about a wooden hazard near the player; ignored → debris."""
    near = [p for p in [player.pos] + world.neighbors(player.pos)
            if world.in_bounds(p) and (world.cell(p).material == "wood"
                                       or "support" in world.cell(p).tags)]
    if not near:
        return None
    pos = near[0]
    return lambda: fire_soft(world, pos=pos, fuse=3,
                             warn="the old timber overhead groans ominously...",
                             commit_narr="the timber cracks — debris crashes down!",
                             damage=2.0, victim=player.id) or []


def _mv_ambient(world, player):
    """Always eligible — the DM is never out of material. Texture with teeth:
    a draft, a distant sound, a flicker; small but REAL (it's on the bus)."""
    lines = ["a draft makes the lamps gutter", "somewhere, a floorboard creaks",
             "a dog barks twice in the distance, then stops",
             "the smell of lamp-oil hangs heavier than it should"]
    line = lines[int(world.rng.random() * len(lines))]
    return lambda: [fx.add_heat(player.pos, 1.0, line)] if player else []


MOVES = [("alarm_creeps", _mv_alarm_creeps), ("rumor_stirs", _mv_rumor_stirs),
         ("fresh_purpose", _mv_fresh_purpose), ("timber_groans", _mv_timber_groans),
         ("ambient", _mv_ambient)]


def fire_soft(world: World, pos, fuse: int, warn: str, commit_narr: str,
              damage: float, victim: str) -> None:
    """A warning with a fuse. Cancels if the would-be victim moves clear."""
    _mark(world, "soft_warning", warn)
    world.pending.append({"fuse": fuse, "pos": tuple(pos), "victim": victim,
                          "narr": commit_narr, "damage": damage})


def _update_pending(world: World):
    keep = []
    for p in world.pending:
        v = world.entities.get(p["victim"])
        if v is None or "dead" in v.tags or l1(v.pos, p["pos"]) > 1:
            continue                          # heeded (or moot) — the moment passes
        p["fuse"] -= 1
        if p["fuse"] > 0:
            keep.append(p)
            continue
        eff = fx.adjust_prop(p["victim"], "hp", -p["damage"], p["narr"])
        fx.apply(world, eff)                  # ignored — the hard move commits
    world.pending = keep


def update(world: World) -> None:
    """Once per tick, after minds: pace chaos, resolve fuses, break lulls."""
    from . import engine
    _update_pending(world)

    pulse = engine.drama(world)
    world.chaos = max(1.0, min(9.0, world.chaos + (0.08 if pulse > 60 else -0.05)))

    if any(engine._notable(e) for e in world.log if e.tick == world.tick):
        world.last_notable = world.tick
        return
    if world.tick - world.last_notable < max(4, LULL_BASE - int(world.chaos)):
        return
    world.last_notable = world.tick           # one nudge per lull, then re-arm
    player = world.entities.get("player")
    eligible = [(mid, f(world, player)) for mid, f in MOVES]
    eligible = [(mid, fire) for mid, fire in eligible if fire]
    if not eligible:
        return
    mid, fire = eligible[int(world.rng.random() * len(eligible))]
    effects = fire()
    _mark(world, mid)
    fx.apply_all(world, effects or [])
