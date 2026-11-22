"""Wave 14a, lane F: the humid boil, the vapour heat capacity in T_at, the ground rows, and
fall preparation from the body's state."""
import math

import numpy as np
import pytest

from src.voxel import anatomy as A
from src.voxel import impact as I
from src.voxel import materials as M
from src.voxel.materials import AIR, AMBIENT, STONE, WATER
from src.voxel.scenes_mixtures import POT_CELL, brine_pot
from src.voxel.sim import World


# ── the humid boil ───────────────────────────────────────────────────────────────────
def test_pure_water_in_a_humid_pot_boils_at_100_C_and_every_joule_is_booked():
    """Boiling is a property of the liquid, not of the air's humidity: the humid world pins at
    the boiling point (the dry law pins at 100.1 C: the heater runs a tick ahead) with no
    overshoot, and the books close."""
    w = brine_pot(None)
    assert w.wet is not None                       # a bare World is humid
    w0, e0, s0 = w.total_water(), w.total_energy(), w.shed
    put, temps = 0.0, []
    for _ in range(330):
        w.E[POT_CELL] += 120.0
        put += 120.0
        w.step()
        temps.append(float(w.T()[POT_CELL]))
    temps = np.array(temps)
    assert temps.max() < 100.3, temps.max()        # it was 103.4 C: steam crammed into a 75 ml headspace
    pinned = temps[temps > 99.0][20:]
    assert len(pinned) > 50 and (pinned > 99.6).all()
    assert abs(w.total_water() - w0) < 1e-3
    assert abs((w.total_energy() - e0) - put) < 1.0     # shed is inside total_energy, so dE is what was put in


def test_steam_a_cell_cannot_hold_goes_up_not_back_into_the_water():
    w = brine_pot(None)
    for _ in range(300):
        w.E[POT_CELL] += 120.0
        w.step()
    st = w.wet.store
    held = float(st.excess("H2O")[POT_CELL])
    # the headspace over 50 ml in a 1 L voxel at 100 C holds rho_sat x Vg = 0.6 g/L x 0.95 L at most
    from src.voxel import psychro
    from src.voxel.gasstore import gas_volume
    vg = float(gas_volume(w.vox_l, float(w.fvol[POT_CELL])))
    assert held <= psychro.rho_sat(100.0) * vg + 0.2
    assert w.wet.stats["boil_g"] > 1.0


# ── the vapour heat capacity ─────────────────────────────────────────────────────────
def test_T_at_reads_the_vapour_heat_capacity_a_wet_world_adds():
    w = brine_pot(None)
    for _ in range(300):
        w.E[POT_CELL] += 120.0
        w.step()
    st = w.wet.store
    ex = st.excess("H2O")
    cells = np.nonzero(ex > 1e-6)
    assert len(cells[0]) > 0, "the boil left vapour in the world"
    assert w.T_at(cells).tobytes() == w.T()[cells].tobytes()
    flat = np.ravel_multi_index(cells, w.shape)
    assert w.T_at(flat).tobytes() == w.T()[cells].tobytes()


def test_a_dry_world_adds_nothing_to_T_at():
    w = brine_pot(None)
    w.wet = None
    cells = (np.array([1]), np.array([2]), np.array([1]))
    assert w.T_at(cells).tobytes() == w.T()[cells].tobytes()


# ── the ground rows ──────────────────────────────────────────────────────────────────
def test_soil_sand_and_grass_are_rows_with_bearing_density_and_thermal_numbers():
    for name, ident in (("soil", M.SOIL), ("sand", M.SAND), ("grass", M.GRASS)):
        rec = M.MATREC.by_id_list[ident]
        assert rec.key == name
        assert 1.0e5 <= rec.sc <= 5.0e5, (name, rec.sc)            # R2's range for firm ground
        assert 1000.0 < rec.rho < 2000.0 and rec.k > 0.0 and rec.cp > 0.0
        assert not rec.structural()                                  # cannot be a beam
        assert M._DENS_ARR[ident] == pytest.approx(rec.rho)
        assert M._KSOLID_ARR[ident] == pytest.approx(rec.k)
        g = I.ground_of(ident, 1.0)
        assert g is not None and g[0] == rec.sc
    for hard in (M.STONE, M.WOOD, M.IRON, M.GLASS):
        assert I.ground_of(hard, 1.0) is None
    assert I.ground_of(M.ICE, 1.0) is None and I.ground_of(M.AIR, 1.0) is None   # no bearing figure: not soft ground


def test_a_fall_on_soil_breaks_less_than_the_same_fall_on_stone():
    sh = A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)
    for h in (6.0, 8.0):
        stone = I.landing(sh, "feet", h, ground=I.ground_of(M.STONE, 1.0))
        soil = I.landing(sh, "feet", h, ground=I.ground_of(M.SOIL, 1.0))
        assert soil["F_peak"] < stone["F_peak"]
        assert len(soil["failed"]) < len(stone["failed"])


# ── fall preparation from the body's state ───────────────────────────────────────────
def _adult():
    return A.make_sheet(mass_kg=70.0, height_m=1.75, draws=False)


def test_an_unforeseen_3_m_feet_first_fall_breaks_the_lumbar_spine_and_a_chosen_jump_does_not():
    sh = _adult()
    old = I.landing(sh, "feet", 3.0)                         # the legacy call: every accident prepared
    trip = I.landing(sh, "feet", 3.0, foreseen=False)
    jump = I.landing(sh, "feet", 3.0, foreseen=True)
    assert old["prep"] == 1.0 and not old["failed"]
    assert jump["prep"] == 1.0 and not jump["failed"]
    assert trip["prep"] == pytest.approx(I.BRACE_UNFORESEEN)
    assert [f["name"] for f in trip["failed"]] == ["lumbar"]
    assert trip["F_peak"] > jump["F_peak"]


def test_preparation_rises_with_the_time_to_impact_against_the_reaction_time():
    ps = [I.prep_from_body(t, True, False) for t in (0.0, 0.2, 0.25, 0.3, 0.35, 0.45, 0.8)]
    assert ps == sorted(ps)
    assert ps[0] == ps[1] == ps[2] == 0.0                    # no time to react: nothing is ready
    assert ps[-1] == pytest.approx(I.BRACE_UNFORESEEN)       # a surprised faller never reaches the planned brace
    slow = I.prep_from_body(0.5, True, False, react_s=0.45)
    quick = I.prep_from_body(0.5, True, False, react_s=0.20)
    assert quick > slow                                       # a slower person (the dial) is less ready


def test_foresight_and_consciousness_set_the_ceiling():
    assert I.prep_from_body(0.05, True, True) == 1.0         # a chosen jump is set at take-off
    assert I.prep_from_body(0.8, False, True) == 0.0         # the unconscious are never prepared
    assert I.prep_from_body(0.8, True, False) < 1.0


def test_the_fit_tool_and_old_callers_keep_the_old_law():
    sh = _adult()
    for h in (1.0, 2.0, 5.0):
        a = I.landing(sh, "feet", h)
        b = I.landing(sh, "feet", h, foreseen=None)
        assert a["F_peak"] == b["F_peak"] and a["prep"] == b["prep"]
        assert a["prep"] == I.prep_of(math.sqrt(2 * h / I.G), True)


def test_the_live_law_reads_the_leap_stamp_and_the_sheets_reaction_time():
    from src.voxel import injury as IJ
    p = {"pid": 1, "name": "f", "awake": True, "alive": True, "_eye": (0, 0, 29),
         "sheet": {"mass_kg": 70.0, "height_m": 1.75}}
    z = np.arange(0, 30)
    pose = np.stack([np.zeros(30), np.zeros(30), z], axis=1)
    m = IJ.sheet_of(p)["mass_kg"]
    got = IJ.landing_effects(p, m * 9.81 * 3.0, pose, None, 1, 0)
    assert got["res"]["prep"] == pytest.approx(I.BRACE_UNFORESEEN) and p.get("fractures") == ["lumbar"]
    q = {"pid": 2, "name": "g", "awake": True, "alive": True, "_eye": (0, 0, 29), "_chose": {"leap": 2},
         "sheet": {"mass_kg": 70.0, "height_m": 1.75}}
    got = IJ.landing_effects(q, m * 9.81 * 3.0, pose, None, 1, 0)
    assert got["res"]["prep"] == 1.0 and not q.get("fractures")
