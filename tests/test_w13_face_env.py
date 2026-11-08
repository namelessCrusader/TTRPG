"""THE FACE-ENVIRONMENT BUNDLE (wave 13, lane B1): src/voxel/faceenv.py and the one
new signature of `heat._face_heat` and its numpy twin in `_law_conduct`.

Four promises, each tested the way the repo tests a pair of laws:

  1. INERT. With no bundle (every world today) the law is the old one. Golden
     says so for 35 scenes; here, a bundle that is ACTIVE but made of defaults (an
     overlay of default rows, the plain boiling point) gives the same bits as no
     bundle, tick by tick, in the compiled law and in the numpy one. That is the
     "defaults reproduce the old constants bit for bit" line, tested where the new
     code actually runs.
  2. THE TWO LAWS AGREE. With the bundle doing real work (a different film law,
     a series resistance, an area factor, a boiling point per cell) the compiled
     face loop and the numpy twin give the same bits (numpy is the definition),
     with windows on and off and skip_quiet on and off, for 200 ticks.
  3. THE LAW MEANS WHAT THE SPEC SAYS. Each column of the overlay table does the
     one thing T7 (BH1 to BH3) and T3 say, checked against the formula.
  4. THE FLUX READOUT (FF) is the flux the law applied, face by face.
"""
import numpy as np
import pytest

from src.voxel import faceenv as FE
from src.voxel.heat import _hboil
from src.voxel.materials import H_CONV, IRON, STONE, TICK_S, WATER, WOOD
from src.voxel.sim import HAVE_NUMBA, World

f32 = np.float32


# ── worlds ───────────────────────────────────────────────────────────────────
def _random_world(seed, fused, windows=True, skip_quiet=True):
    """A small walled-open world with stone, wood and iron, water and oil, and
    heat: every kind of face (solid|solid, solid|gas, solid|liquid, liquid|gas)
    and a boiling curve that really acts."""
    rng = np.random.default_rng(seed)
    w = World(22, 18, 16, voxel_cm=5)
    w.fused, w.windows, w.skip_quiet = fused, windows, skip_quiet
    w.open_sky = bool(seed % 2)
    w.fill(0, 22, 0, 18, 0, 1, STONE)
    mats = (STONE, WOOD, IRON, IRON)
    for _ in range(int(rng.integers(3, 7))):
        x, y = int(rng.integers(1, 18)), int(rng.integers(1, 14))
        w.fill(x, x + int(rng.integers(1, 5)), y, y + int(rng.integers(1, 5)), 1,
               1 + int(rng.integers(1, 6)), mats[int(rng.integers(0, 4))],
               frac=float(rng.choice([0.4, 0.7, 1.0])))
    for _ in range(int(rng.integers(1, 4))):
        x, y = int(rng.integers(2, 20)), int(rng.integers(2, 16))
        w.pour(x, y, 1, WATER, 90.0)
    for _ in range(int(rng.integers(2, 6))):
        x, y, z = int(rng.integers(1, 20)), int(rng.integers(1, 16)), int(rng.integers(1, 6))
        w.E[x, y, z] += float(rng.choice([600.0, 3000.0, 20000.0, 90000.0]))
    return w


def _table():
    """Overlay rows: 1 a skin with the natural-convection law, 2 a clothed skin
    (series resistance, no linear radiation), 3 an area factor 1.5, 4 everything."""
    return FE.overlay_table([
        {"hnat": 2.38},
        {"rser": 0.08, "radw": 0.0},
        {"az": 1.5},
        {"hnat": 2.38, "rser": 0.05, "radw": 0.0, "az": 0.7},
    ])


def _bundle(w, seed, table=None, boil=None, ff=False):
    """Give the world an overlay on random solid cells (ids 0 to 4, and a few 9s:
    past the table, which reads row 0) and, if `boil`, a boiling point per cell."""
    rng = np.random.default_rng(1000 + seed)
    bz = w.ensure_bzone()
    ids = rng.integers(0, 5, size=w.shape).astype(np.uint16)
    ids[rng.random(w.shape) < 0.05] = 9
    bz[...] = np.where(w.mat != 0, ids, 0)
    bp = None
    if boil:
        bp = (100.0 + (rng.random(w.shape) * 25.0 * (boil == "varied"))).astype(np.float32)
    w.face_env = FE.FaceEnv(bz=bz, ov=_table() if table is None else table, bp=bp, ff=ff)


def _fields(w):
    """Every array of the world's state, except the overlay ids themselves
    (the world with the bundle holds them, the control does not)."""
    out = {k: v for k, v in vars(w).items() if isinstance(v, np.ndarray) and k != "bzone"}
    out["shed"] = np.float64(w.shed)
    return out


def _same(a, b, where, shed_rel=0.0):
    """Every array equal to the bit. `shed` too, unless `shed_rel` > 0: a world
    that sums its losses over a window and one that sums over the whole lattice
    add the same float32 numbers in float64 in another order (2.9e-11 J apart
    after ten ticks in base19 itself), so across windows the books are compared
    to a relative `shed_rel`, never the arrays."""
    A, B = _fields(a), _fields(b)
    assert set(A) == set(B), where
    for k in A:
        if k == "shed" and shed_rel:
            assert abs(float(A[k]) - float(B[k])) <= shed_rel * max(1.0, abs(float(A[k]))), \
                f"{where}: shed {float(A[k])!r} against {float(B[k])!r}"
            continue
        assert np.array_equal(A[k], B[k]), f"{where}: {k} differs"


needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="numba is optional; the plain path is the engine")


# ── 1. inert ─────────────────────────────────────────────────────────────────
def test_the_default_bundle_is_inert_and_a_window_of_it_is_itself():
    assert FE.NO_ENV.inert() and FE.NO_ENV.env == 0 and not FE.NO_ENV.ff
    assert FE.NO_ENV.window((slice(0, 3),) * 3) is FE.NO_ENV
    w = World(8, 8, 8, voxel_cm=5)
    assert w.face_env is None and w._face_env((slice(None),) * 3) is FE.NO_ENV
    assert not any(k in vars(w) for k in ("face_env", "bzone", "slow", "scomp"))


@pytest.mark.parametrize("fused", [True, False])
def test_an_active_bundle_of_default_rows_gives_the_bits_of_no_bundle(fused):
    if fused and not HAVE_NUMBA:
        pytest.skip("numba is optional")
    for seed in range(3):
        a, b = _random_world(seed, fused), _random_world(seed, fused)
        # every cell carries an overlay id, every id is a default row, the boiling
        # point is plain BOIL: the bundle is ACTIVE and says nothing new
        bz = b.ensure_bzone()
        bz[...] = np.where(b.mat != 0, 1, 0)
        b.face_env = FE.FaceEnv(bz=bz, ov=FE.overlay_table([{}, {}]),
                                bp=np.full(b.shape, 100.0, np.float32))
        assert b.face_env.env == FE.ENV_OVERLAY | FE.ENV_BOIL
        for t in range(40):
            a.step()
            b.step()
            _same(a, b, f"seed {seed} fused {fused} tick {t}")


# ── 2. the two laws agree ────────────────────────────────────────────────────
@needs_numba
@pytest.mark.parametrize("boil", [None, "plain", "varied"])
def test_the_compiled_face_law_and_the_numpy_one_agree_to_the_bit_with_the_bundle(boil):
    for seed in range(4):
        fast, plain = _random_world(seed, True), _random_world(seed, False)
        inert = _random_world(seed, False)
        for w in (fast, plain):
            _bundle(w, seed, boil=boil)
        for t in range(60):
            for w in (fast, plain, inert):
                w.step()
            _same(fast, plain, f"seed {seed} boil {boil} tick {t}")
        # ...and the bundle really acted (else this proves nothing)
        assert not np.array_equal(plain.E, inert.E), "the overlay changed nothing"


@needs_numba
def test_the_bundle_agrees_windowed_whole_lattice_and_with_skipping_off_for_200_ticks():
    """Windows on and off, skip_quiet on and off, compiled and numpy: one world
    each, the bundle doing real work, every tick compared with the plain
    whole-lattice run."""
    seed = 5
    ref = _random_world(seed, False, windows=False, skip_quiet=False)
    others = [_random_world(seed, True, True, True), _random_world(seed, False, True, True),
              _random_world(seed, True, False, True), _random_world(seed, True, True, False)]
    for w in [ref] + others:
        _bundle(w, seed, boil="varied")
    for t in range(200):
        ref.step()
        for i, w in enumerate(others):
            w.step()
            if t % 10 == 0 or t > 190:
                _same(ref, w, f"config {i} tick {t}", shed_rel=1e-9)


def test_a_flux_readout_does_not_change_the_world():
    a, b = _random_world(2, True), _random_world(2, True)
    _bundle(a, 2)
    _bundle(b, 2, ff=True)
    got = []

    class Reader(type(b)):
        def _face_flux(self, sl, fluxes):
            got.append([f.copy() for f in fluxes])
    b.__class__ = Reader
    for _ in range(25):
        a.step()
        b.step()
    _same(a, b, "readout")
    assert got and all(len(g) == 3 for g in got)


def test_an_id_past_the_end_of_the_table_reads_the_default_row():
    for fused in ([True, False] if HAVE_NUMBA else [False]):
        a, b = _random_world(1, fused), _random_world(1, fused)
        for w, ident in ((a, 0), (b, 7)):
            bz = w.ensure_bzone()
            bz[...] = np.where(w.mat != 0, ident, 0)
            w.face_env = FE.FaceEnv(bz=bz, ov=_table())          # 5 rows: 7 is past the end
        for _ in range(30):
            a.step()
            b.step()
        _same(a, b, f"fused {fused}")


# ── 3. what each column means ────────────────────────────────────────────────
def _conduct_once(w):
    """Call the face law once on the whole lattice, with nothing else running,
    and return the flux it applied across every face (the readout)."""
    seen = []

    class Reader(type(w)):
        def _face_flux(self, sl, fluxes):
            seen.append([f.copy() for f in fluxes])
    w.__class__ = Reader
    w._law_conduct()
    return seen[0]


def _hot_stone(row, hot=60.0):
    """One stone cell `hot` K above ambient alone in air, its overlay id 1 with
    `row`; returns the flux arrays of one pass of the face law."""
    w = World(7, 7, 7, voxel_cm=5)
    w.fill(3, 4, 3, 4, 3, 4, STONE)
    w.E[3, 3, 3] = hot * float(w.heat_capacity()[3, 3, 3])
    bz = w.ensure_bzone()
    bz[3, 3, 3] = 1
    w.face_env = FE.FaceEnv(bz=bz, ov=FE.overlay_table([row]), ff=True)
    return w, _conduct_once(w)


A_FACE = float(f32(0.05) * f32(0.05))


def _into_the_air(fx):
    """The joules the hot cell gave to the air through its +x face on the
    first tick (the flux is signed from the lower cell to the higher: the hot
    cell is the lower one)."""
    return float(fx[0][3, 3, 3])


def _plain(hot=60.0):
    return A_FACE * H_CONV * float(f32(TICK_S)) * hot


def test_the_default_row_is_the_constant_film():
    w, fx = _hot_stone({})
    assert _into_the_air(fx) == pytest.approx(_plain(), rel=1e-5)


def test_the_natural_convection_law_replaces_the_constant_film():
    w, fx = _hot_stone({"hnat": 2.38})
    hc = 2.38 * 60.0 ** 0.25
    assert _into_the_air(fx) == pytest.approx(A_FACE * hc * TICK_S * 60.0, rel=1e-5)
    assert hc != pytest.approx(H_CONV, rel=0.05)                  # (a different number, or this tests nothing)


def test_an_area_factor_multiplies_the_exchange_with_the_air():
    w, fx = _hot_stone({"az": 1.5})
    assert _into_the_air(fx) == pytest.approx(1.5 * _plain(), rel=1e-5)


def test_a_resistance_in_series_is_one_over_the_sum_of_the_resistances():
    r = 0.2
    w, fx = _hot_stone({"rser": r})
    g = H_CONV * A_FACE                                           # W/K
    g_series = 1.0 / (1.0 / g + r / A_FACE)
    assert _into_the_air(fx) == pytest.approx(g_series * TICK_S * 60.0, rel=1e-5)


def test_the_area_factor_comes_after_the_series_resistance():
    w, fx = _hot_stone({"rser": 0.2, "az": 2.0})
    g = H_CONV * A_FACE
    assert _into_the_air(fx) == pytest.approx(
        2.0 / (1.0 / g + 0.2 / A_FACE) * TICK_S * 60.0, rel=1e-5)


def test_the_weight_on_gas_radiation_turns_the_linear_radiation_of_hot_gas_off():
    """Hot gas beside a cold stone face gives it convection plus radiation; with
    OV_RADW = 0 (a garment: its outer skin radiates, not the skin under it) only
    the convection is left."""
    def run(row):
        w = World(7, 7, 7, voxel_cm=5)
        w.fill(3, 4, 3, 4, 3, 4, STONE)
        w.E[4, 3, 3] = 380.0 * float(w.heat_capacity()[4, 3, 3])    # the gas cell beside it
        bz = w.ensure_bzone()
        bz[3, 3, 3] = 1
        w.face_env = FE.FaceEnv(bz=bz, ov=FE.overlay_table([row]), ff=True)
        return float(_conduct_once(w)[0][3, 3, 3])                  # the face between the stone and that gas
    with_rad, without = run({}), run({"radw": 0.0})
    # the gas gave heat to the stone: the flux is negative (from the higher cell to the lower)
    assert with_rad < without < 0.0
    assert without == pytest.approx(-A_FACE * H_CONV * TICK_S * 380.0, rel=1e-4)


def test_the_core_to_skin_conductance_of_the_overlay_is_open_area_over_the_resistance():
    w = World(7, 7, 7, voxel_cm=5)
    w.fill(3, 4, 3, 4, 3, 4, STONE)
    sl = (slice(None),) * 3
    u = w._surfaces(sl)
    bz = w.ensure_bzone()
    bz[3, 3, 3] = 1
    env = FE.FaceEnv(bz=bz, ov=FE.overlay_table([{"rsc": 0.02}]))
    g = env.gsb(u, f32(0.05))
    assert u.has[3, 3, 3]
    assert float(g[3, 3, 3]) == pytest.approx(6 * A_FACE / 0.02, rel=1e-5)   # six open faces
    other = np.ones(w.shape, bool)
    other[3, 3, 3] = False
    assert np.array_equal(g[other], u.Gsb[other])                  # no overlay, no change
    assert FE.NO_ENV.gsb(u, f32(0.05)) is u.Gsb                    # and the inert bundle hands it back


def test_the_skin_step_of_the_face_law_uses_the_bundles_core_to_skin_conductance():
    """One pass on a hot stone cell: its skin node is left cooler than its core by what the
    air took (the default sqrt(a t) path), but with a core-to-skin resistance a millionth of
    that the skin and core stay one temperature, and the skin field is exactly zero."""
    def skin_after(row):
        w = World(7, 7, 7, voxel_cm=5)
        w.fill(3, 4, 3, 4, 3, 4, STONE)
        w.E[3, 3, 3] = 60.0 * float(w.heat_capacity()[3, 3, 3])
        bz = w.ensure_bzone()
        bz[3, 3, 3] = 1
        w.face_env = FE.FaceEnv(bz=bz, ov=FE.overlay_table([row]))
        w._law_conduct()
        return float(w.skin[3, 3, 3])
    assert skin_after({}) != 0.0
    assert skin_after({"rsc": 1e-9}) == 0.0


def test_a_boiling_point_per_cell_moves_where_the_boiling_curve_acts():
    """Hot iron beside water: at the plain boiling point the curve acts and the
    iron gives more than the plain wet film. Tell the water it boils at 400 C
    (a brine would at 104), above the wall, and the curve is off: the plain wet
    film is all there is."""
    def face(boil_c):
        w = World(8, 6, 6, voxel_cm=5)
        w.fill(2, 4, 2, 4, 2, 4, IRON)
        w.pour(4, 2, 2, WATER, 100.0)
        w.E[2:4, 2:4, 2:4] = 200.0 * float(w.heat_capacity()[2, 2, 2])
        bp = None if boil_c is None else np.full(w.shape, boil_c, np.float32)
        w.face_env = FE.FaceEnv(bp=bp, ff=True)
        return float(_conduct_once(w)[0][3, 2, 2])                  # iron (x=3) | water (x=4)
    plain, same = face(None), face(100.0)
    assert plain == same                                            # a plain boiling point IS the default
    cold = face(400.0)
    assert plain > cold > 0.0
    assert cold == pytest.approx(_hl_water() * A_FACE * TICK_S * 200.0, rel=1e-5)


def _hl_water():
    from src.voxel.materials import _FK_ARR
    return float(_FK_ARR[WATER])


def _iron_in_water(fused, boil, row):
    """Hot iron with a cell of water on EACH of its six sides, so that along every
    axis the liquid is the lower cell of one face and the higher cell of another (the
    compiled law has a branch for each), air beyond, a boiling point per cell and an
    overlay on the iron. One pass of the face law; returns (world, flux arrays)."""
    w = World(9, 9, 9, voxel_cm=5)
    w.fused = fused
    w.fill(3, 6, 3, 6, 3, 6, IRON)
    for cell in ((2, 4, 4), (6, 4, 4), (4, 2, 4), (4, 6, 4), (4, 4, 2), (4, 4, 6)):
        w.pour(*cell, WATER, 100.0)
    w.E[3:6, 3:6, 3:6] = 200.0 * float(w.heat_capacity()[3, 3, 3])
    bz = w.ensure_bzone()
    bz[3:6, 3:6, 3:6] = 1
    if boil == "varied":
        bp = (100.0 + 30.0 * np.random.default_rng(3).random(w.shape)).astype(np.float32)
    else:
        bp = None if boil is None else np.full(w.shape, boil, np.float32)
    w.face_env = FE.FaceEnv(bz=bz, ov=FE.overlay_table([row]), bp=bp, ff=True)
    return w, _conduct_once(w)


# (axis, index of the face in that axis's flux array) of the six iron | water faces
_WATER_FACES = ((0, (2, 4, 4)), (0, (5, 4, 4)), (1, (4, 2, 4)), (1, (4, 5, 4)),
                (2, (4, 4, 2)), (2, (4, 4, 5)))


@needs_numba
@pytest.mark.parametrize("boil", [None, 100.0, 125.0, 400.0, "varied"])
@pytest.mark.parametrize("row", [{}, {"az": 0.5}, {"rser": 0.1}, {"rser": 0.03, "az": 1.7, "hnat": 2.38}])
def test_hot_iron_in_water_and_air_is_the_same_bits_compiled_and_plain_with_the_bundle(boil, row):
    """The faces the random worlds may not reach: iron at 220 C with water on every
    side (the boiling curve, immersion, the liquid as the lower AND as the higher cell
    of a face), a boiling point per cell, an overlay on the iron. One pass of the face
    law, compiled and numpy, every array and every face flux compared."""
    (a, fa), (b, fb) = _iron_in_water(True, boil, row), _iron_in_water(False, boil, row)
    _same(a, b, f"boil {boil} row {row}")
    for x, y in zip(fa, fb):
        assert np.array_equal(x, y), "a face flux differs"
    for axis, idx in _WATER_FACES:
        assert float(fa[axis][idx]) != 0.0                          # every iron | water face carried heat


@pytest.mark.parametrize("fused", [True, False])
def test_the_boiling_point_acts_on_every_water_face_whichever_cell_is_the_lower(fused):
    """Water that boils at 400 C, above the iron, gets the plain wet film and nothing
    more, on all six faces; at the plain boiling point the curve acts on all six. A
    law that read the boiling point on one orientation only would differ here."""
    if fused and not HAVE_NUMBA:
        pytest.skip("numba is optional")
    _, hot = _iron_in_water(fused, 100.0, {})
    _, high = _iron_in_water(fused, 400.0, {})
    for axis, idx in _WATER_FACES:
        assert abs(float(hot[axis][idx])) > abs(float(high[axis][idx])) > 0.0, (axis, idx)


def test_the_boiling_curve_helper_keeps_its_old_name_and_its_old_answer():
    """`_hboil` is still the curve at the plain boiling point; `_hboil_t` takes the point."""
    from src.voxel.heat import _hboil_t
    args = (f32(1500.0), f32(130.0), f32(60.0), f32(14000.0), f32(1600.0), f32(0.8),
            f32(4 * 5.670374e-8), f32(273.15))
    assert _hboil(*args) == _hboil_t(*args, f32(100.0))
    assert _hboil_t(*args, f32(400.0)) == f32(1500.0)               # above the wall: only the wet film
    assert _hboil_t(*args, f32(100.0)) > f32(1500.0)                # bubbles at a 22 K interface


# ── 4. the flux readout ──────────────────────────────────────────────────────
def test_the_flux_readout_has_one_array_per_axis_shaped_like_the_faces_and_signed_low_to_high():
    w, fx = _hot_stone({})
    nx, ny, nz = w.shape
    assert [f.shape for f in fx] == [(nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1)]
    assert all(f.dtype == np.float32 for f in fx)
    hot = _plain()
    for axis, idx_hi, idx_lo in ((0, (3, 3, 3), (2, 3, 3)), (1, (3, 3, 3), (3, 2, 3)),
                                 (2, (3, 3, 3), (3, 3, 2))):
        assert float(fx[axis][idx_hi]) == pytest.approx(hot, rel=1e-5)     # hot cell is the lower one
        assert float(fx[axis][idx_lo]) == pytest.approx(-hot, rel=1e-5)    # hot cell is the higher one


def test_a_bundle_cut_for_other_cells_is_refused_before_the_compiled_law_reads_it():
    w = World(7, 7, 7, voxel_cm=5)
    w.fill(3, 4, 3, 4, 3, 4, STONE)
    small = FE.FaceEnv(bz=np.zeros((3, 3, 3), np.uint16), ov=FE.overlay_table([{}]))

    class Wrong(World):
        def _face_env(self, sl):
            return small                                   # not cut to `sl`
    w.__class__ = Wrong
    assert not small.fits(w.shape) and FE.NO_ENV.fits(w.shape)
    with pytest.raises(ValueError, match="face-environment bundle"):
        w._law_conduct()


def test_the_overlay_table_builder_validates_its_rows():
    t = FE.overlay_table([{"az": 2.0}, [1.0, 0.0, 0.0, 1.0, 0.0]])
    assert t.shape == (3, FE.NOV) and t.dtype == np.float32
    assert tuple(t[0]) == FE.DEFAULT_ROW and t[1, FE.OV_AZ] == 2.0
    with pytest.raises(ValueError):
        FE.overlay_table([[1.0, 2.0]])
    with pytest.raises(ValueError):
        FE.FaceEnv(bz=np.zeros((2, 2, 2), np.uint16))               # ids without a table
    with pytest.raises(TypeError):
        FE.FaceEnv(bz=np.zeros((2, 2, 2), np.int32), ov=t)         # ids are uint16
    with pytest.raises(TypeError):
        FE.FaceEnv(bp=np.zeros((2, 2, 2), np.float64))             # boiling points are float32
