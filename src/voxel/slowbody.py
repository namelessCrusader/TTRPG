"""The body-heat twin and the coupled night jump (wave 13, T5 with T7, lane C10).

Once a body is a heat engine in the lattice (BodyWorld) it is never still: its
flesh sits near 34 C and its stores drain. A sleeper cannot be tier 0. The night
skip is TIER 1 (spec 6.7): the person's compact model (bodynodes) is advanced in
chunks of a few sim-minutes, and the room's air and walls are advanced between
the chunks by the thermal net (slownet), the way two programs that take turns
share a room.

ONE CHUNK (`coupled_jump`):
  1. every body that is carried by a twin takes `w.bh_skip(p, dt)`: its voxels go
     to the nodes, the compact model runs `dt` seconds against the air the body
     is in, and the voxels get the nodes back (energy exact). What the room is
     given comes out through the ONE sink the tick path also uses (`w.airsink`,
     the sealed-room RegionSink), which in this jump holds the heat back;
  2. that heat is a power over the chunk, put into the net as a source on the
     air cells that touch the body. The bodies' own cells are `skip` cells of
     the net: they keep their energy to the bit, and the air beside them sees an
     insulated wall. The room's air and walls advance `dt` seconds;
  3. the sink applies the chunk's oxygen, carbon dioxide and vapour to the room.
The next chunk re-reads the air the body is in (its temperature, the oxygen at
its mouth), so the two models see each other. The chunk is the coupling step:
the error is first order in `dt / (room time constant)` (u, measured in
tests/test_c10_tier1.py).

A jump stops early, at a chunk edge, when a body leaves the band the model is
good in (core under 30 C or over 40 C), when the breath twin's horizon (a
wake, a faint) comes inside two ticks, or when the net refuses; the driver then
ticks. It never ticks silently: the report says why.

APPROXIMATE IN WHERE, EXACT IN HOW MUCH. Not in golden or harvest: advance()
only reaches this with approx=True.
"""
import numpy as np

from . import slownet
from .airsink_world import RegionSink
from .materials import AIR, TICK_S
from .slowtwins import Twin, register_twin

CHUNK_S = 300.0                      # sim-seconds per coupling step (spec 7.3: 30 to 300 s)
CORE_BAND = (30.0, 40.0)             # C: the compact model's good range for a jump
O2_L_MIN_DEFAULT = 0.25              # L/min for a sleeper, when the body law has not measured it yet


def active(w, p):
    """True when this person's body is carried by the twin: a live heat record in a body-heat world."""
    h = p.get("heat")
    return bool(h is not None and getattr(w, "body_heat", False) and h.get("alive")
                and p["alive"] and not p["safe"] and hasattr(w, "bh_skip"))


def owned_cells(w):
    """(N, 3) world cells of every body the twin carries (the net's `skip`), or None."""
    cells = [np.asarray(p["heat"]["cells"], np.int64) for p in w.persons if active(w, p)]
    return np.concatenate(cells) if cells else None


def touching_air(w, cells):
    """Air cells that share a face with `cells` (the cells a body warms)."""
    cells = np.asarray(cells, np.int64).reshape(-1, 3)
    shape = np.array(w.shape)
    out = []
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        c = cells + np.array(d)
        c = c[((c >= 0) & (c < shape)).all(axis=1)]
        out.append(c)
    c = np.unique(np.concatenate(out), axis=0)
    return c[w.mat[tuple(c.T)] == AIR]


class BodyHeatTwin(Twin):
    """T7's body as a STEP twin. Inactive (no effect at all) for a person without a heat record or in a world
    without `body_heat`. `advance` runs `bh_skip`: it works on its own (the sink then lands the heat at once)
    and inside `coupled_jump` (the sink holds the heat for the net)."""
    name = "body_heat"
    keys = ("heat",)
    coupled = True

    def applicable(self, w, p):
        if not active(w, p):
            return True, ""
        h = p["heat"]
        if h["pid"] not in w._bh_cache():
            return False, "the body law has not met this person yet"
        Tp = float(h.get("T_pool", 36.9))
        if not (CORE_BAND[0] < Tp < CORE_BAND[1]):
            return False, f"core {Tp:.1f} C is outside the band {CORE_BAND} the compact model is good in"
        return True, ""

    def enter(self, w, p):
        return {"active": active(w, p)}

    def horizon(self, w, p, ctx):
        return []

    def power_w(self, w, p, ctx):
        return 0.0                   # the coupled jump hands the room's heat over chunk by chunk

    def o2_use_g(self, w, p, ctx, n):
        if not ctx.get("active"):
            return 0.0
        last = p["heat"].get("last") or {}
        dt = float(last.get("dt") or 0.0)
        rate = float(last.get("o2_L", 0.0)) / dt if dt > 0 else O2_L_MIN_DEFAULT / 60.0
        return rate * n * TICK_S * 1.429

    def advance(self, w, p, ctx, n):
        if n <= 0 or not ctx.get("active"):
            return {}
        w.bh_skip(p, n * TICK_S)
        return {}


register_twin("body_heat", BodyHeatTwin())


# ---------------------------------------------------------------- the jump --
def coupled_jump(w, n, q, sched):
    """The tier-1 jump of a world with carried bodies: chunks of CHUNK_S. Moves w.tick. Returns (ok, info)
    where info["n"] is the ticks actually jumped (a jump may end early at a chunk edge) and info["why"] says why
    it ended early ('' = it did not)."""
    from . import quiet as Q
    sink = getattr(w, "airsink", None)
    if not isinstance(sink, RegionSink):
        sink = RegionSink(w)
        w.airsink = sink
    mode0 = sink.mode
    chunk = max(1, int(CHUNK_S / TICK_S))
    t_end = w.tick + n
    done, why = 0, ""
    stats = {"chunks": 0, "source_J": 0.0, "boundary_J": 0.0}
    try:
        sink.mode = "collect"
        while w.tick < t_end:
            plan = []
            bl = Q.cheap_blockers(w, 1, plan)
            if bl:
                why = "%s: %s" % (bl[0].gate, bl[0].reason)
                break
            hz = Q.horizon(w, plan, sched)
            room = t_end - w.tick
            if hz and hz[0][0] - 2 < w.tick + 1:
                why = "an event is due: %s" % hz[0][1]
                break
            n_c = min(chunk, room, (hz[0][0] - 2 - w.tick) if hz else room)
            if n_c < 1:
                break
            skip = owned_cells(w)
            o2g = sum(tw.o2_use_g(w, p, ctx, n_c) for p, twins in plan for _n, tw, ctx in twins)
            if done == 0 and o2g > 0.0:
                ok, why0 = slownet.v2_o2_ok(w, slownet.region_for(w, (), n * TICK_S), o2g * n / n_c)
                if not ok:
                    return False, {"refusals": [("v2", None, why0)], "n": 0}
            for p, twins in plan:
                for _name, tw, ctx in twins:
                    sched._book(w, tw.advance(w, p, ctx, n_c))
            heat = sink.take_heat()
            dt_s = n_c * TICK_S
            sources = []
            if skip is not None and heat:
                sources = [(touching_air(w, skip), heat / dt_s)]
            ok, info = slownet.advance_region(w, n_c, sched, sources=sources, skip=skip)
            if not ok:
                sink._heat(heat)                 # the net refused: the room still gets the body's heat, at once
                sink.flush()
                w.tick += n_c
                done += n_c
                why = str(info["refusals"][0][2]) if info.get("refusals") else "the net refused"
                break
            stats["chunks"] += 1
            stats["source_J"] += info.get("source_J", 0.0)
            stats["boundary_J"] += info.get("boundary_J", 0.0)
            sink.flush()
            w.tick += n_c
            done += n_c
    finally:
        sink.mode = mode0
    stats.update({"n": done, "why": why})
    return True, stats


def sync_clocks(w):
    """After a jump, put each carried body's 'last body second' on the last tick of its own clock that the
    jump covered, so the first tick law after it sees at most one body second, not the whole jump."""
    per = w._bh_period()
    for p in w.persons:
        if active(w, p):
            w_ = p["heat"]
            w_["tick_last"] = int(w.tick) - ((int(w.tick) + p.get("pid", 0)) % per)


def sink_blockers(w, plan):
    """A carried body needs a room to give its heat and gases to. Without a sealed-room sink they would leave
    through `gone` and `shed` and the room would not be warmed: say so, and do not jump."""
    from .quiet import Blocker
    if isinstance(getattr(w, "airsink", None), RegionSink):
        return []
    for _p, twins in plan:
        for _nm, tw, ctx in twins:
            if getattr(tw, "coupled", False) and ctx.get("active"):
                return [Blocker("persons", "airsink",
                                "a carried body needs a sealed-room sink (w.airsink = RegionSink(w))")]
    return []
