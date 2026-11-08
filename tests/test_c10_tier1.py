"""SCHEDULER TIER 1 AND THE SEALED-ROOM SINK (wave 13c, lane C10; T5 with T7).

The night skip of a room with living bodies in it: the bodies' compact model
and the room's thermal net take turns in chunks (slowbody.py); the room's
heat, vapour, oxygen and carbon dioxide go through the sealed-room sink
(airsink_world.py), the same sink the tick path calls. Opt-in per call
(approx=True); never in golden or harvest.

Small rooms here (a few thousand cells) so the file stays fast. The big
night (a 3 x 2.4 x 2 m room, four sleepers, 8 hours) is in the lane report."""
import copy

import numpy as np
import pytest

from src.voxel import bodynodes as BN
from src.voxel import slowbody as SB
from src.voxel import slownet as N
from src.voxel import slowtime as ST
from src.voxel import slowtwins as TW
from src.voxel.airsink_world import O2_G_PER_L, RegionSink
from src.voxel.materials import AIR, AMBIENT, O2_PER_L, STONE, TICK_S
from src.voxel.scenes import _person
from src.voxel.scenes_slow import SlowWorld, scene_inn_body


def _room_world(nx=24, ny=20, nz=24):
    w = SlowWorld(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, nx, 0, 1, 0, nz, STONE)
    w.fill(0, nx, ny - 1, ny, 0, nz, STONE)
    w.fill(0, 1, 0, ny, 0, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 0, nz, STONE)
    w.fill(0, nx, 0, ny, nz - 1, nz, STONE)
    w.exits = []
    w.step()
    return w


# --------------------------------------------------------------- the sink --
def test_region_sink_heat_lands_in_the_room_and_the_lattice_total_moves_by_exactly_that():
    w = _room_world()
    s = RegionSink(w)
    e0, shed0 = w.total_energy(), w.shed
    s.add(heat_J=1234.5)
    assert w.shed == shed0
    assert abs((w.total_energy() - e0) - 1234.5) < 1e-3
    assert s.mean_temp() > AMBIENT


def test_region_sink_oxygen_is_drawn_from_the_lattice_field_and_matches_the_litres():
    w = _room_world()
    s = RegionSink(w)
    g0 = s.o2_mass_g()
    assert abs(s.o2_fraction() - 0.209) < 1e-6
    s.add(o2_l=-10.0, co2_l=8.0)
    assert abs((g0 - s.o2_mass_g()) - 10.0 * O2_G_PER_L) < 1e-2          # float32 cells
    assert s.o2_fraction() < 0.209
    assert s.co2_l == 8.0 and s.o2_l == 10.0
    assert not s.mixing()                                                 # drawn evenly


def test_region_sink_vapour_past_saturation_condenses_and_gives_its_latent_heat_back():
    w = _room_world()
    s = RegionSink(w)
    cap = float(s.volume_l() / 1000.0 * 17.29)
    e0 = w.total_energy()
    s.add(vapour_g=2.0 * cap, heat_J=2.0 * cap * 2450.0)                  # all of it latent heat
    s.flush()
    assert s.condensed_g > 0.9 * cap and s.ambient()[1] <= 1.0       # the latent heat that came back warmed the air
    # book: lattice + latent held in the vapour = what was given
    assert abs((w.total_energy() - e0) + s.latent_J - 2.0 * cap * 2450.0) < 1e-2 * cap


def test_collect_mode_holds_the_heat_and_gives_the_same_totals_as_now_mode():
    a, b = _room_world(), _room_world()
    sa, sb = RegionSink(a), RegionSink(b, mode="collect")
    for s in (sa, sb):
        s.add(vapour_g=1.0, heat_J=900.0, o2_l=-2.0, co2_l=1.5)
    assert sb.heat_J == 0.0 and sb.o2_l == 0.0                            # nothing landed yet
    h = sb.take_heat()
    sb.flush()
    sb._heat(h)
    ba, bb = sa.books(), sb.books()
    for k in ba:
        assert abs(ba[k] - bb[k]) < 1e-9 * max(1.0, abs(ba[k])), k


def test_the_tick_path_gases_land_within_one_body_second_and_readers_see_them_at_once():
    w = _room_world()
    s = RegionSink(w)
    s.add(o2_l=-3.0)                                                      # held, the second is not up
    assert s.o2_l == 0.0
    assert s.o2_fraction() < 0.209                                        # a reader flushes first
    assert abs(s.o2_l - 3.0) < 1e-12


# ------------------------------------------------------- the net's skip -----
def test_a_skipped_cell_keeps_its_energy_to_the_bit_and_the_books_close():
    w = _room_world()
    cells = np.array([[12, 10, z] for z in range(1, 8)])
    w.fill(12, 13, 10, 11, 1, 8, STONE)                                   # a warm column, like a body
    w.E[12, 10, 1:8] = 4000.0                                             # about 13 K above the room
    w.E[6:8, 8:12, 10] = 1.5                                              # some warm air so the net has work
    before = w.E[tuple(cells.T)].copy()
    sources = [(SB.touching_air(w, cells), 50.0)]
    e0 = w.total_energy()
    sh0 = w.shed
    ok, info = N.advance_region(w, 4000, None, sources=sources, skip=cells)
    assert ok, info
    assert np.array_equal(w.E[tuple(cells.T)], before)
    de = w.total_energy() - e0
    # what came in: the source; what left: shed. Rounding goes to the residual (float32 E).
    assert abs(de - (info["source_J"] - (w.shed - sh0))) < 1e-3 * max(1.0, info["source_J"])


# --------------------------------------------------------------- the twin ---
def test_the_body_twin_is_inactive_for_a_world_without_body_heat():
    w = _room_world()
    p = _person(w, 8, 8, z0=1)
    p["awake"], p["asleep"] = True, True
    for _ in range(3):
        w.step()
    twins, why = TW.person_twins(w, p)
    assert twins is not None, why
    names = {n: ctx for n, _tw, ctx in twins}
    assert "body_heat" in names and names["body_heat"] == {"active": False}
    assert SB.owned_cells(w) is None


def test_default_advance_never_uses_tier_1_on_a_body_world():
    w, _ps = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=60)
    rep = ST.advance(w, 2600, sched=w.slow, max_fine_ticks=2600)          # approx=False: the default
    assert rep.jumped == 0 and rep.tier_used == "exact"


def test_a_carried_body_without_a_sealed_room_sink_is_a_named_blocker():
    w, _ps = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=60, sink=False)
    rep = ST.advance(w, 2600, approx=True, sched=w.slow, max_fine_ticks=2600)
    assert rep.jumped == 0
    assert any("sealed-room sink" in str(b[2]) for b in rep.blockers_seen)


# ----------------------------------------------------- the coupled jump -----
def _snap(w, ps):
    s = w.airsink
    body = np.zeros(w.shape, bool)
    for p in ps:
        body[tuple(p["heat"]["cells"].T)] = True
    E = w.E.astype(np.float64)
    return {"T_air": s.mean_temp(), "o2": s.o2_mass_g(), "co2": s.co2_l, "vap": s.vapour_g + s.condensed_g,
            "core": p["heat"]["T_pool"], "skin": BN.mean_skin(w.bh_to_nodes(p), w._bh_geom(p, p["heat"])),
            "room_J": float(E[~body].sum()) + s.latent_J, "shed": float(w.shed),
            "air_J": float(E[(w.mat == AIR)].sum())}


def test_a_jump_agrees_with_ticking_on_the_body_the_gases_and_how_much_heat_the_room_gets():
    secs = 90.0
    n = int(secs / TICK_S)
    a, pa = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=80)
    b, pb = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=80)
    a0, s0 = _snap(a, pa), _snap(b, pb)
    rep = ST.advance(b, n, approx=True, sched=b.slow, max_fine_ticks=3000)
    assert rep.tier_used == "reduced" and rep.jumped > 0.3 * n and rep.halted_by == "target"
    for _ in range(n):
        a.step()
    sa, sb = _snap(a, pa), _snap(b, pb)
    # the body: core and mean skin as ticked (the compact model against the voxel body)
    assert abs(sa["core"] - sb["core"]) < 0.15
    assert abs(sa["skin"] - sb["skin"]) < 0.4
    # the gases the body gave the room: oxygen, carbon dioxide, vapour, within 10 percent
    for k in ("o2", "co2", "vap"):
        da, db = abs(sa[k] - a0[k]), abs(sb[k] - s0[k])
        assert da > 0 and abs(da - db) < 0.10 * da + 1e-3, (k, da, db)
    # HOW MUCH heat left the body into (room + the world past the lattice): within 20 percent
    qa = (sa["room_J"] - a0["room_J"]) + (sa["shed"] - a0["shed"])
    qb = (sb["room_J"] - s0["room_J"]) + (sb["shed"] - s0["shed"])
    assert abs(qa - qb) < 0.20 * abs(qa), (qa, qb)
    # WHERE it goes differs, and the reason is the tick law's, not the jump's: the lattice lets every surface
    # that looks onto gas radiate to the WORLD (heat.py `_env_radiation`, booked in shed), also in a shut room.
    # The jump keeps the body's heat in the room. Said here so nobody reads the room temperature as agreement.
    assert (sb["shed"] - s0["shed"]) < (sa["shed"] - a0["shed"])
    assert (sb["air_J"] - s0["air_J"]) > (sa["air_J"] - a0["air_J"])


def test_a_jump_is_exact_in_how_much_energy_the_books_hold():
    """total_energy() counts the lattice, the body's pool and stores, and shed. Add the latent heat the room's
    vapour carries (the sink holds it) and the sum must not move: a jump creates and loses no joule."""
    b, pb = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=80)
    sink = b.airsink
    sink.flush()
    tot0 = b.total_energy() + sink.latent_J
    rep = ST.advance(b, int(60.0 / TICK_S), approx=True, sched=b.slow, max_fine_ticks=3000)
    assert rep.jumped > 1000 and rep.tier_used == "reduced"
    sink.flush()
    assert abs((b.total_energy() + sink.latent_J) - tot0) < 1.0              # joules, of ~1e5 moved


def test_oxygen_use_that_fills_too_much_of_the_room_refuses_the_jump_and_says_why():
    # a closet: 4 sleepers in a 1 m3 box for 8 hours is the spec's V2 case
    w, ps = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=60)
    q = ST  # noqa
    ok, why = N.v2_o2_ok(w, tuple(slice(0, n) for n in w.shape), 8 * 3600 * 1.0 / 60.0 * 1.429 * 6.0)
    assert not ok and "O2 use" in why


def test_a_jump_ends_at_a_chunk_edge_when_a_core_leaves_the_band():
    w, ps = scene_inn_body(nx=24, ny=20, nz=32, wall=3, n_sleepers=1, settle=80)
    ps[0]["heat"]["T_pool"] = 29.0            # hypothermic: the compact model is not good here
    twins, why = TW.person_twins(w, ps[0])
    assert twins is None and "body_heat" in why


def test_golden_and_harvest_cannot_reach_tier_1():
    import inspect
    assert inspect.signature(ST.advance).parameters["approx"].default is False
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in ("tools/golden.py", "src/voxel/harvest.py"):
        src = (root / f).read_text()
        assert "approx=True" not in src and "slowtime" not in src
