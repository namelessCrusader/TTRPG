"""COMPOSITION IN THE LATTICE (wave 13a, lane B4; T1 spec, plan section 1).

A cell can hold a MIX (a row of the mix table) in its solid (`scomp`) or its
fluid (`fcomp`). `mat` and `fl` are then views. Nothing in 13a puts a mix in a
cell by itself, so the first group of tests proves that a pure world is the old
world, and the rest prove the machinery with scenes that do.

What "refused" means for acid on brine (T1 4.7): `_mix_ids` answers None, like
oil into water. Nothing moves, nothing is booked, no heat is made, both cells
stay as they were. The tests below pin each part of that sentence.
"""
import numpy as np
import pytest

from src.voxel import species as SP
from src.voxel.materials import ACID, AIR, GLASS, NOFLUID, OIL, STONE, WATER, WEAK_ACID, WOOD
from src.voxel.sim import World

BRINE = [("water", "dissolved", 0, 0.8), ("sodium_chloride", "dissolved", 0, 0.2)]
NACL = SP.BY_NAME["sodium_chloride"]
H2O = SP.BY_NAME["water"]


def _cup(layers, nx=5, ny=5, nz=8, **kw):
    """A stone cup 3 x 3 inside; `layers` is [(z, 'b' | 'w')] poured full."""
    w = World(nx, ny, nz, voxel_cm=10, **kw)
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    w.fill(0, 1, 0, ny, 1, nz, STONE)
    w.fill(nx - 1, nx, 0, ny, 1, nz, STONE)
    w.fill(1, nx - 1, 0, 1, 1, nz, STONE)
    w.fill(1, nx - 1, ny - 1, ny, 1, nz, STONE)
    row = w.ensure_mixes().parse(BRINE)
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z, k in layers:
                if k == "b":
                    w.pour(x, y, z, None, w.cap, mix=row)
                else:
                    w.pour(x, y, z, WATER, w.cap)
    return w, row


def _grams(w):
    """{species: grams} of the FLUID in the lattice: pure cells by the old
    densities (closed form), mixed cells by their rows."""
    out = {}
    T = w.mixes
    live = w.fcomp > 0
    pure = (~live) & (w.fvol > 0)
    for f in np.unique(w.fl[pure]):
        row = 64 + int(f)
        m = float(w.fvol[pure & (w.fl == f)].sum()) * float(T.prop("rho", np.array([row]))[0]) / 1000.0
        for k, v in T.totals(np.array([row]), np.array([m])).items():
            out[k] = out.get(k, 0.0) + v
    if live.any():
        ids = w.fcomp[live].astype(np.int64)
        m = w.fvol[live].astype(np.float64) * T.prop("rho", ids).astype(np.float64) / 1000.0
        ent = T.entries(ids)
        for sp in np.unique(ent.species):
            out[int(sp)] = out.get(int(sp), 0.0) + float((ent.frac * m[ent.cell])[ent.species == sp].sum())
    return out


# ── a pure world is the old world ─────────────────────────────────────────────
def test_a_pure_world_never_goes_live_and_the_writers_are_the_old_statements():
    w = World(8, 8, 8, voxel_cm=10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    w.fill(2, 4, 2, 4, 1, 3, WOOD, frac=0.6, edge=1e-4)
    w.pour(5, 5, 1, WATER, 400.0)
    w.pour(5, 5, 1, OIL, 400.0)                  # oil on water: refused, as ever
    w.set_mat((slice(6, 7),) * 3, GLASS)
    w.set_solid((slice(1, 2),) * 3, 0)           # pure row 0: nothing to do
    w.set_fluid((slice(1, 2),) * 3, 0)
    assert w.mixes_live is False and w.mixes is None and w.scomp is None and w.fcomp is None
    assert "_mix_key" not in vars(w)
    assert w.mat[6, 6, 6] == GLASS and w.fl[5, 5, 1] == WATER and w.fvol[5, 5, 1] == np.float32(400.0)
    assert w.legacy_ok() is None
    for _ in range(30):
        w.step()
    assert w.mixes_live is False and "_mix_key" not in vars(w)


def test_the_derived_fields_take_the_old_path_without_mixes():
    w = World(6, 6, 6, voxel_cm=10)
    w.fill(0, 6, 0, 6, 0, 1, STONE)
    w.pour(2, 2, 1, WATER, 300.0)
    assert w._mix_args() == (None, None)
    a = w._cap_of(w.mat, w.smass, w.fl, w.fvol)
    assert np.array_equal(a, w._cap_of(w.mat, w.smass, w.fl, w.fvol, None, None))
    assert np.array_equal(w.heat_capacity(), a)
    assert np.array_equal(w._fdens_grid(), w._cap_of.__globals__["_FDENS_ARR"][w.fl])


# ── fill and set_solid ────────────────────────────────────────────────────────
def test_fill_with_a_mix_writes_scomp_the_view_and_the_mixs_own_mass():
    w = World(8, 8, 8, voxel_cm=10)
    bronze = w.ensure_mixes().parse([("copper", "bound", 0, 0.88), ("tin", "bound", 0, 0.12)])
    w.fill(1, 3, 1, 3, 1, 3, None, mix=bronze)
    assert w.mixes_live and (w.scomp[1:3, 1:3, 1:3] == bronze).all()
    assert w.mat[1, 1, 1] != AIR                              # a view: the leader among the skeleton entries
    rho = float(w.mixes.prop("rho", np.array([bronze]))[0])
    assert w.smass[1, 1, 1] == pytest.approx(rho * w.vox_l, rel=1e-6)
    assert w.legacy_ok().sum() == w.legacy_ok().size - 8
    # a pure fill over it clears the row and books the change to `recast`
    g0 = w.mixes.ledger_total("recast")
    w.fill(1, 2, 1, 2, 1, 2, STONE)
    assert w.scomp[1, 1, 1] == 0 and w.mat[1, 1, 1] == STONE
    assert w.mixes.ledger("recast")                           # the old grams out, the new pure grams in
    assert abs(w.mixes.ledger_total("recast") - g0) < 1e-6 * rho


def test_set_solid_keeps_the_old_mat_while_it_is_within_five_percent_of_the_leader():
    w = World(6, 6, 6, voxel_cm=10)
    w.fill(1, 3, 1, 3, 1, 2, WOOD)
    row = w.ensure_mixes().parse([("wood", "bound", 0, 0.52), ("stone", "bound", 0, 0.48)])
    w.set_solid((slice(1, 3), slice(1, 3), slice(1, 2)), row)
    assert (w.mat[1:3, 1:3, 1] == WOOD).all()                 # hysteresis: wood stays
    w.set_solid((slice(1, 3), slice(1, 3), slice(1, 2)), 0)
    assert (w.scomp[1:3, 1:3, 1] == 0).all()


# ── the derived fields read the mix ───────────────────────────────────────────
def test_heat_capacity_of_a_mixed_cell_is_the_tables_and_a_convert_invalidates_the_cache():
    w, row = _cup([(1, "b")])
    T = w.mixes
    c = float(T.prop("c", np.array([row]))[0])
    rho = float(T.prop("rho", np.array([row]))[0]) / 1000.0
    want = np.float32(w.fvol[2, 2, 1]) * np.float32(rho) * np.float32(c) + w.c_airbase
    cap = w.heat_capacity()
    assert cap[2, 2, 1] == pytest.approx(float(want), rel=1e-6)
    # brine is not water: the table's c is lower than 4.2
    assert cap[2, 2, 1] < w.fvol[2, 2, 1] * 4.2 * 1.0
    # change the row and nothing else (mat, smass, fl, fvol stay): the cache must notice
    sugar = T.parse([("water", "dissolved", 0, 0.5), ("sucrose", "dissolved", 0, 0.5)])
    before = cap[2, 2, 1]
    w.fcomp[2, 2, 1] = sugar
    assert w.heat_capacity()[2, 2, 1] != before
    # T_at reads the same capacity as T()
    w.E[2, 2, 1] = 5000.0
    assert w.T_at((np.array([2]), np.array([2]), np.array([1])))[0] == pytest.approx(float(w.T()[2, 2, 1]), rel=1e-6)


def test_density_comes_from_the_mix_for_layers_and_for_packing():
    w, row = _cup([(1, "b")])
    d = w._fdens_grid()
    assert d[2, 2, 1] == pytest.approx(1.149, abs=0.01) and d[2, 2, 2] == 0.0
    assert d[2, 2, 1] > 1.0, "brine is denser than the water it views as"
    mix = w.mixes.parse([("copper", "bound", 0, 0.88), ("tin", "bound", 0, 0.12)])
    rho = w._solid_rho(np.array([STONE, STONE]), np.array([0, mix]))
    assert rho[0] == 2700.0 and rho[1] == pytest.approx(float(w.mixes.prop("rho", np.array([mix]))[0]))


# ── the fluid hand-over ───────────────────────────────────────────────────────
def test_pouring_water_on_brine_is_the_mass_exact_blend_and_species_are_conserved():
    w, row = _cup([])
    w.pour(2, 2, 1, None, 600.0, mix=row)
    g0 = _grams(w)
    w.pour(2, 2, 1, WATER, 300.0)
    rho_b = float(w.mixes.prop("rho", np.array([row]))[0]) / 1000.0
    m = 600.0 * rho_b + 300.0 * 1.0
    new = int(w.fcomp[2, 2, 1])
    rho_n = float(w.mixes.prop("rho", np.array([new]))[0]) / 1000.0
    assert w.fvol[2, 2, 1] == pytest.approx(m / rho_n, rel=1e-6)           # volume follows mass
    assert w.fvol[2, 2, 1] < 900.0                                          # (a small contraction)
    g1 = _grams(w)
    led = w.mixes.ledger("f32")
    for sp in (NACL, H2O):
        assert g1[sp] + led.get(sp, 0.0) == pytest.approx(g0[sp] + 300.0 * (sp == H2O), rel=1e-6)


def test_a_pour_past_the_cell_is_scaled_to_fit_and_the_rest_stays_with_the_caller():
    w, row = _cup([])
    w.pour(2, 2, 1, None, w.cap - 100.0, mix=row)
    w.pour(2, 2, 1, WATER, 500.0)
    assert w.fvol[2, 2, 1] <= np.float32(w.cap) * (1 + 1e-6)
    # (water into brine contracts a little, so the cell may be left a hair under full)
    assert w.fvol[2, 2, 1] == pytest.approx(w.cap, rel=2e-3)


def test_a_move_between_mixed_cells_keeps_every_species_gram():
    w, row = _cup([(1, "b")])
    g0 = _grams(w)
    for _ in range(40):
        w._move_fluid((2, 2, 1), (1, 1, 1), 50.0)
        w._move_fluid((1, 1, 1), (3, 3, 1), 70.0)
    g1 = _grams(w)
    led = w.mixes.ledger("f32")
    for sp in g0:
        assert g1[sp] + led.get(sp, 0.0) == pytest.approx(g0[sp], rel=1e-9)


# ── P2: the brine layers ──────────────────────────────────────────────────────
def test_brine_under_fresh_water_holds_for_ten_sim_minutes_and_loses_no_gram():
    w, row = _cup([(1, "b"), (2, "w")])
    g0 = _grams(w)
    for _ in range(int(600.0 / 0.025)):
        w.step()
    assert (w.fcomp[1:4, 1:4, 1] == row).all() and (w.fcomp[1:4, 1:4, 2] == 0).all()
    assert (w.fl[1:4, 1:4, 1:3] == WATER).all() and (w.fvol[1:4, 1:4, 1:3] == np.float32(w.cap)).all()
    assert w.mixes.n_rows < 200, "no row is made by a layer that does not mix"
    g1 = _grams(w)
    for sp in g0:
        assert g1[sp] == pytest.approx(g0[sp], rel=1e-12)


def test_brine_poured_on_water_sinks_and_water_on_brine_floats():
    w, row = _cup([(1, "w"), (2, "b")])
    for _ in range(300):
        w.step()
    assert (w.fcomp[1:4, 1:4, 1] == row).all() and (w.fcomp[1:4, 1:4, 2] == 0).all()
    w2, row2 = _cup([(1, "b"), (2, "w")])
    for _ in range(300):
        w2.step()
    assert (w2.fcomp[1:4, 1:4, 1] == row2).all() and (w2.fcomp[1:4, 1:4, 2] == 0).all()


def test_the_layers_give_the_same_cells_with_skipping_and_sleeping_off():
    out = []
    for quiet in (True, False):
        w, row = _cup([(1, "b"), (2, "w")], nz=6)
        w.skip_quiet = quiet
        w.sleep_regions = quiet
        for _ in range(300):
            w.step()
        out.append((w.fl.copy(), w.fvol.copy(), w.fcomp.copy(), w.E.copy()))
    for a, b in zip(*out):
        assert np.array_equal(a, b)


# ── acid on brine is refused ──────────────────────────────────────────────────
@pytest.mark.parametrize("acid", [ACID, WEAK_ACID])
def test_acid_poured_on_brine_is_refused_and_brine_poured_on_acid_too(acid):
    w, row = _cup([])
    w.pour(2, 2, 1, None, 400.0, mix=row)
    snap = (w.fl.copy(), w.fvol.copy(), w.fcomp.copy(), w.fpot.copy(), w.E.copy())
    ledgers = {k: dict(v) for k, v in w.mixes._ledgers.items()}
    w.pour(2, 2, 1, acid, 200.0)
    for a, b in zip(snap, (w.fl, w.fvol, w.fcomp, w.fpot, w.E)):
        assert np.array_equal(a, b), "refused: nothing changed"
    assert {k: dict(v) for k, v in w.mixes._ledgers.items()} == ledgers, "refused: nothing booked"
    w.pour(3, 3, 1, acid, 200.0)
    snap = (w.fl.copy(), w.fvol.copy(), w.fcomp.copy(), w.fpot.copy())
    w.pour(3, 3, 1, None, 100.0, mix=row)                   # brine on acid
    for a, b in zip(snap, (w.fl, w.fvol, w.fcomp, w.fpot)):
        assert np.array_equal(a, b)


def test_acid_beside_brine_never_crosses_into_it_through_the_flow_law():
    w, row = _cup([])
    w.pour(1, 2, 1, None, w.cap, mix=row)
    w.pour(2, 2, 1, ACID, w.cap)
    w.pour(2, 2, 1, WEAK_ACID, 1.0)                           # (a refused no-op: the cell is full)
    for _ in range(200):
        w.step()
    assert w.fcomp[2, 2, 1] == 0 and w.fl[2, 2, 1] in (ACID, WEAK_ACID)
    assert w.fcomp[1, 2, 1] == row
    assert not (w.fcomp[1:4, 1:4, 1:3][w.fl[1:4, 1:4, 1:3] == ACID] > 0).any()


def test_the_refusal_table():
    mi = World._mix_ids
    assert mi(WATER, True, WATER, False) == WATER             # brine into water: fine
    assert mi(WATER, False, NOFLUID, False) == WATER
    assert mi(WATER, True, ACID, False) is None               # brine on acid
    assert mi(ACID, False, WATER, True) is None               # acid on brine
    assert mi(WEAK_ACID, False, WATER, True) is None
    assert mi(WEAK_ACID, False, WATER, False) == WEAK_ACID    # pure acid and pure water: the old table
    assert mi(OIL, False, WATER, True) is None                # oil refuses water anyway
    assert mi(ACID, False, NOFLUID, False) == ACID            # an empty cell takes anything
    assert mi(ACID, False, NOFLUID, True) == ACID             # (and a stale row in it is not read)


# ── paths agree ───────────────────────────────────────────────────────────────
def test_a_world_with_one_mix_far_away_runs_a_pure_spill_as_a_pure_world_does():
    """The compiled sweeps are off in a world with mixes (`_compiled_fluids`);
    the Python path they are tested against must give the same bits."""
    def run(live):
        w = World(12, 12, 8, voxel_cm=10)
        w.fill(0, 12, 0, 12, 0, 1, STONE)
        w.fill(4, 8, 4, 8, 1, 2, STONE)
        for x in range(5, 7):
            w.pour(x, 5, 2, WATER, 800.0)
        w.pour(6, 6, 2, OIL, 500.0)
        if live:           # a block that is "stone" by the table's own pure row, so it is stone to every law
            w.fill(10, 11, 10, 11, 1, 2, STONE, mix=w.ensure_mixes().parse([("stone", "bound", 0, 1.0)]))
            assert w.mixes_live and not w._compiled_fluids() and w.scomp[10, 10, 1] > 0
        else:
            w.fill(10, 11, 10, 11, 1, 2, STONE)
        for _ in range(150):
            w.step()
        return w
    a, b = run(False), run(True)
    for n in ("fl", "fvol", "fpot", "E"):
        x, y = getattr(a, n), getattr(b, n)
        m = np.ones(x.shape, bool)
        assert np.array_equal(x[m], y[m]), n


def test_drops_carry_the_mix_in_a_tenth_entry_and_land_with_it():
    w = World(11, 11, 8, voxel_cm=10)
    w.fill(0, 11, 0, 11, 0, 1, STONE)
    row = w.ensure_mixes().parse(BRINE)
    w.pour(5, 5, 1, None, 800.0, mix=row)
    w.pour(4, 5, 1, WATER, 60.0)                  # a pure neighbour for the parcels to land in
    w._splash(5, 5, 1)
    assert w.drops, "a splash throws drops"
    assert all(len(d) in (9, 10) for d in w.drops)
    assert any(len(d) == 10 and d[9] == row for d in w.drops)
    g0 = _grams(w)
    air = lambda sp: sum(d[6] * (0.2 * 1.149 if len(d) == 10 else 0.0) for d in w.drops) if sp == NACL else 0.0
    n0 = air(NACL)
    for _ in range(60):
        w.step()
    assert not w.drops, "every parcel landed"
    assert n0 > 0
    g1 = _grams(w)
    led = w.mixes.ledger("f32")
    # the splash took its parcels from the cell without a ledger, so compare the
    # whole: grams on the floor + grams still in the cell = grams before - 0
    assert g1[NACL] + led.get(NACL, 0.0) == pytest.approx(g0[NACL] + n0, rel=2e-3)


# ── D6: legacy laws skip mixed cells ──────────────────────────────────────────
def test_acid_does_not_eat_a_mixed_solid_and_soaking_passes_over_it():
    w = World(8, 8, 8, voxel_cm=10)
    w.fill(0, 8, 0, 8, 0, 1, STONE)
    wood_mix = w.ensure_mixes().parse([("wood", "bound", 0, 0.9), ("water", "bound", 0, 0.1)])
    w.fill(3, 5, 3, 5, 1, 2, WOOD)                            # a pure block
    w.fill(3, 5, 3, 5, 2, 3, WOOD, mix=wood_mix)
    m0 = w.smass[3:5, 3:5, 2].copy()
    w.pour(2, 3, 1, WEAK_ACID, 600.0, pot=1.0)
    w.pour(2, 3, 2, WEAK_ACID, 600.0, pot=1.0)
    for _ in range(120):
        w.step()
    assert np.array_equal(w.smass[3:5, 3:5, 2], m0), "the mixed block was left alone"
    assert (w.scomp[3:5, 3:5, 2] == wood_mix).all()
    ok = w.legacy_ok()
    assert not ok[3:5, 3:5, 2].any() and ok[3:5, 3:5, 1].all()


def test_unmaking_a_mixed_cell_books_its_grams_to_gone():
    w = World(6, 6, 6, voxel_cm=10)
    row = w.ensure_mixes().parse([("copper", "bound", 0, 0.88), ("tin", "bound", 0, 0.12)])
    w.fill(2, 3, 2, 3, 1, 2, None, mix=row)
    m = float(w.smass[2, 2, 1])
    mask = np.zeros(w.shape, bool)
    mask[2, 2, 1] = True
    w._unmake(mask, "melts")
    assert w.mat[2, 2, 1] == AIR and w.scomp[2, 2, 1] == 0
    led = w.mixes.ledger("gone")
    assert sum(led.values()) == pytest.approx(m, rel=1e-6)
    assert led[SP.BY_NAME["tin"]] == pytest.approx(0.12 * m, rel=1e-5)


def test_a_shatter_carries_the_mix_to_every_shard_and_the_spilled_fluid_keeps_its_row():
    w = World(9, 9, 6, voxel_cm=10)
    w.fill(0, 9, 0, 9, 0, 1, STONE)
    row = w.ensure_mixes().parse([("glass", "bound", 0, 0.9), ("sodium_chloride", "crystal", 0, 0.1)])
    w.fill(4, 5, 4, 5, 1, 2, GLASS, mix=row)
    br = w.mixes.parse(BRINE)
    w.set_fluid((4, 4, 1), br)
    w.fl[4, 4, 1], w.fvol[4, 4, 1] = WATER, 300.0
    m = float(w.smass[4, 4, 1])
    w._shatter(4, 4, 1)
    shards = np.argwhere((w.scomp == row) & (w.mat == GLASS))
    assert len(shards) >= 2 and (w.mat[w.scomp == row] == GLASS).all()
    assert float(w.smass[w.scomp == row].sum()) == pytest.approx(m, rel=1e-5)
    assert w.scomp[4, 4, 1] == 0 and w.mat[4, 4, 1] == AIR
    assert w.fcomp[4, 4, 1] == br and w.fvol[4, 4, 1] == np.float32(300.0)
    assert not w.mixes.ledger("recast"), "a shatter is not a transmutation"
