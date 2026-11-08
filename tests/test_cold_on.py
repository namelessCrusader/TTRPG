"""Cold ON (wave 13b, lane C1). The seams of T8 are live: the 0 C floor is gone,
the matter rule and the born temperature of thrown things, the eddy reference,
the hot-surface radiation environment, the open-top sink and the books. Small
worlds at 5 cm voxels. The big proofs (W1 .. W13, S1) are in test_cold_world.py.
"""
import numpy as np
import pytest

from src.voxel import climate as K
from src.voxel import outside as O
from src.voxel.coldworld import ColdWorld
from src.voxel.materials import ACID, AIR, AMBIENT, IRON, STONE, WOOD
from src.voxel.sim import World


def _w(shape, climate=None, **kw):
    w = ColdWorld(*shape, voxel_cm=5, **kw)
    if climate is not None:
        w.set_climate(climate, soak=False)
    return w


# ---------------------------------------------------------------- no climate = World
def _arrays(w):
    out = {k: v for k, v in vars(w).items() if isinstance(v, np.ndarray)}
    out["shed"] = np.float64(w.shed)
    return out


def test_with_no_climate_a_cold_world_is_a_world_bit_for_bit_on_a_hot_scene():
    """A fire, a pour and a hot iron block: ColdWorld with no climate equals World tick by tick."""
    def build(cls):
        w = cls(30, 24, 20, voxel_cm=5)
        w.fill(0, 30, 0, 24, 0, 1, STONE)
        w.fill(8, 12, 8, 12, 1, 5, WOOD)
        w.fill(18, 21, 8, 11, 1, 4, IRON)
        w.pour(14, 12, 1, ACID, 80.0)
        w.E[9, 9, 2] += 9000.0
        w.E[19, 9, 2] += (500.0 - AMBIENT) * float(w.heat_capacity()[19, 9, 2])
        return w
    a, b = build(World), build(ColdWorld)
    for t in range(60):
        a.step()
        b.step()
        A, B = _arrays(a), _arrays(b)
        assert set(A) == set(B)
        for k in A:
            assert np.array_equal(A[k], B[k]), (t, k)


def test_a_cold_world_with_the_climate_taken_off_is_a_world_again():
    w = _w((16, 16, 10), K.Climate.reference(t_c=-5.0))
    w.set_climate(None)
    assert w.climate is None and w._rest_cell() == 0.0 and w._rest_ref((slice(None),) * 3) is None
    w.E[4, 4, 4] = 5.0
    assert w._unsettled(0)[1] == [] and w._heat_busy()


# ---------------------------------------------------------------- the floor is gone
def test_a_cell_at_minus_30_is_not_lifted_to_zero_even_with_no_climate():
    w = _w((12, 12, 8))
    w.fill(4, 8, 4, 8, 0, 3, STONE)
    C = w.heat_capacity()
    w.E[5, 5, 1] = (-30.0 - AMBIENT) * C[5, 5, 1]
    t0 = float(w.T()[5, 5, 1])
    for _ in range(5):
        w.step()
    assert float(w.T()[5, 5, 1]) < -20.0 and t0 == pytest.approx(-30.0, abs=1e-3)
    assert w.T().min() > -273.15


# ---------------------------------------------------------------- the matter rule
def test_acid_dissolving_a_cold_stone_cell_does_not_leave_it_at_minus_forty_thousand():
    c = K.Climate.reference(t_c=-5.0)
    w = _w((20, 20, 12), c)
    w.fill(5, 15, 5, 15, 0, 3, STONE)
    w.soak()
    e0 = w.total_energy()
    mask = np.zeros(w.shape, bool)
    mask[9, 9, 2] = True
    c_old = float(w.heat_capacity()[9, 9, 2])
    w._unmake(mask, "dissolves")
    c_new = float(w.heat_capacity()[9, 9, 2])
    assert c_new < 1.0 < c_old
    t = float(w.T()[9, 9, 2])
    assert -6.0 < t < -4.0, t                                   # the air that is left is at the rest
    assert abs(w.total_energy() - e0) < 1e-2 * abs(e0) * 1e-3 + 1e-3     # and the books close
    assert w.outside_booked["ref_shift"] != 0.0


def test_the_matter_rule_is_nothing_at_20_c():
    w = _w((12, 12, 8), K.Climate.reference(t_c=20.0))
    w.fill(4, 8, 4, 8, 0, 3, STONE)
    w.soak()
    mask = np.zeros(w.shape, bool)
    mask[5, 5, 1] = True
    w._unmake(mask, "melts")
    assert w.outside_booked["ref_shift"] == 0.0 and float(w.E[5, 5, 1]) == 0.0


def test_matter_change_on_the_kernel_closes_and_leaves_the_rest_with_the_capacity():
    e, ref, car = O.matter_change(np.float64(-8000.0), 260.0, 0.15, 0.0)
    assert e + ref + car == pytest.approx(-8000.0, abs=1e-9)    # nothing made or lost
    assert e / 0.15 + 20.0 == pytest.approx(-8000.0 / 260.0 + 20.0, abs=1e-6)   # the air that is left is as cold as the cell was (not -43,000 C)
    e, ref, car = O.matter_change(np.float64(0.0), 260.0, 0.15, 20.0)
    assert (e, ref, car) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- a thing let go is at the air's temperature
def test_a_bolt_let_go_in_cold_air_is_cold_and_in_20_c_air_has_no_heat():
    w = _w((20, 20, 12), K.Climate.reference(t_c=-12.0))
    w.soak()
    b = w.loose((8, 8, 6), (0.0, 30.0, 0.0), IRON, 60.0)
    th = float(b["Es"][0])
    from src.voxel.materials import _CSOLID_ARR
    assert th == pytest.approx((-12.0 - AMBIENT) * 60.0 * float(_CSOLID_ARR[IRON]), rel=1e-4)
    v = _w((20, 20, 12))
    assert float(v.loose((8, 8, 6), (0.0, 30.0, 0.0), IRON, 60.0)["Es"][0]) == 0.0


# ---------------------------------------------------------------- the eddy reference and T()
def test_the_eddy_term_measures_a_cold_worlds_gas_against_its_own_rest():
    w = _w((16, 16, 10), K.Climate.reference(t_c=-12.0))
    w.soak()
    ref = w._rest_ref((slice(None),) * 3)
    assert ref.shape == w.shape and np.allclose(ref, -32.0)
    assert float(w.T().max()) == pytest.approx(-12.0, abs=1e-3)


def test_T_fills_the_quiet_cells_at_their_own_rest_not_at_20():
    c = K.Climate.reference(t_c=-12.0).replace(rest_c=-5.0)
    w = _w((20, 20, 12), c)
    w.fill(6, 12, 6, 12, 0, 4, STONE)
    w.soak()
    T = w.T()
    assert float(T[8, 8, 1]) == pytest.approx(-5.0, abs=1e-3) and float(T[1, 1, 8]) == pytest.approx(-12.0, abs=1e-3)


# ---------------------------------------------------------------- the open top and the books
def test_a_warm_plume_at_the_open_top_leaves_and_is_booked_against_the_outside_air():
    w = _w((12, 12, 8), K.Climate.reference(t_c=-12.0))
    w.soak()
    C = w.heat_capacity()
    w.E[5, 5, 7] += 30.0 * C[5, 5, 7]                                # 30 K above the outside, at the top
    e0 = w.total_energy()
    w._top_sink()
    assert float(w.T()[5, 5, 7]) < -12.0 + 30.0 and w.outside_booked["top"] > 0.0
    assert abs(w.total_energy() - e0) < 1e-3                          # (float32 rounding of E near -5000 J is 5e-4)
    assert float(w.T()[3, 3, 7]) == pytest.approx(-12.0, abs=1e-3)   # air at T_out does not "leave"


def test_a_hot_surface_in_a_cold_closed_room_keeps_the_books():
    """Iron above RAD_HOT in a room: the hot route radiates to the room node, G7 lands what it can.
    The books: total energy is constant (shed carries what left)."""
    c = K.Climate.reference(t_c=-5.0)
    w = _w((30, 30, 16), c)
    w.fill(4, 26, 4, 26, 0, 1, STONE)
    w.fill(4, 26, 4, 26, 1, 10, STONE)
    w.fill(6, 24, 6, 24, 1, 9, AIR)
    w.fill(4, 26, 4, 26, 10, 12, STONE)
    w.fill(14, 17, 14, 17, 1, 3, IRON)
    w.soak()
    iron = w.mat == IRON
    w.E[iron] = (600.0 - AMBIENT) * w.heat_capacity()[iron]
    e0 = w.total_energy()
    for _ in range(200):
        w.step()
    assert abs(w.total_energy() - e0) < 1e-4 * abs(float(w.E[iron].sum()))
    assert w.T().min() > -273.15 and float(w.T()[iron].max()) < 600.0
    o = w.outside_totals()
    assert o["rad"] == o["rad"]                                          # booked, finite


def test_the_radiation_books_equal_the_part_of_shed_that_cold_wrote():
    c = K.Climate(day_s=600.0, cloud=0.2, ground_depth_m=1.0, rest_c=2.0, t_ann_c=2.0)
    w = _w((30, 30, 14), c)
    w.fill(6, 24, 6, 24, 0, 1, STONE)
    w.fill(6, 24, 6, 24, 1, 8, STONE)
    w.fill(9, 21, 9, 21, 1, 8, AIR)
    w.fill(6, 24, 6, 24, 8, 10, WOOD)
    w.soak()
    for _ in range(300):
        w.step()
    assert w.outside_totals()["total"] == pytest.approx(w.shed, rel=1e-9, abs=1e-6)


# ---------------------------------------------------------------- the stable seams cost nothing without a climate
def test_the_seams_exist_on_a_plain_world():
    w = World(8, 8, 8, voxel_cm=5)
    assert w._rest_ref((slice(None),) * 3) is None
    assert w._rest_pre(np.zeros(w.shape, bool)) is None
    assert w._hot_env_k4(None, None) == (AMBIENT + 273.15) ** 4
    assert w._warm_cells().shape == (w.shape[0] // w._CS, w.shape[1] // w._CS, -(-w.shape[2] // w._CS))


# ---------------------------------------------------------------- the plume rule of the outdoor relaxation
@pytest.mark.parametrize("fused", [True, False])
def test_outdoor_relaxation_leaves_a_plume_to_the_buoyancy_law_and_relaxes_mild_air(fused):
    """A cell more than hot_k above T_out is a plume or a flame: its heat flux is conserved as it
    widens, so relaxing it at 0.1/s would delete the heat a fire gives. Mild air still returns."""
    rng = np.random.default_rng(3)
    C = np.full((6, 6, 6), 0.15, np.float32)
    cls = np.full(C.shape, 1, np.uint8)
    cls[:, :3] |= 2
    th = np.float32(-32.0)
    base = (th * C).astype(np.float32)
    E = base + (rng.uniform(0.0, 300.0, C.shape) * C).astype(np.float32)     # 0 .. 300 K above T_out
    E0 = E.copy()
    a, b = O.outdoor_relax(E, C, cls, float(th), O.relax_factor(1.0, 0.025), O.relax_factor(0.1, 0.025),
                           O.SETTLE_K, fused=fused, hot_k=100.0)
    dev0, dev1 = (E0 - base) / C, (E - base) / C
    hot = dev0 > 100.0
    assert hot.any() and (~hot).any()
    assert np.array_equal(E[hot], E0[hot])                       # the plume is untouched
    assert (dev1[~hot] < dev0[~hot]).all()                       # mild air relaxed toward T_out
    assert a + b == pytest.approx(float((E0.astype(np.float64) - E).sum()), rel=1e-9)
    E2 = E0.copy()
    O.outdoor_relax(E2, C, cls, float(th), O.relax_factor(1.0, 0.025), O.relax_factor(0.1, 0.025),
                    O.SETTLE_K, fused=fused)                     # no limit = the old law
    assert (E2[hot] < E0[hot]).all()


def test_fused_and_plain_relaxation_with_a_plume_limit_are_the_same_bits():
    rng = np.random.default_rng(7)
    C = (rng.uniform(0.1, 3.0, (8, 8, 8))).astype(np.float32)
    cls = rng.integers(0, 4, C.shape).astype(np.uint8)
    E = (rng.uniform(-40, 400, C.shape) * C).astype(np.float32)
    Ea, Eb = E.copy(), E.copy()
    ra = O.outdoor_relax(Ea, C, cls, -12.0, 0.0247, 0.0025, O.SETTLE_K, fused=True, hot_k=80.0)
    rb = O.outdoor_relax(Eb, C, cls, -12.0, 0.0247, 0.0025, O.SETTLE_K, fused=False, hot_k=80.0)
    assert np.array_equal(Ea, Eb) and ra == pytest.approx(rb, rel=1e-12, abs=1e-9)
