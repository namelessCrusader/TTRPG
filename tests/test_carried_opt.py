"""CARRIED_OPT: the optional fields a cell's matter carries (wave 13, part 13a, lane B2).

CARRIED and AS_BODY are pinned (test_things.py, the base19 oracle) and a
`_lift` result is read by position, so the fields wave 13 adds (the mix of a
cell's solid and fluid, its damage, the body-zone overlay on it, the ice grams
in it) ride beside them in CARRIED_OPT, and only while their arrays exist.

What these tests hold:
  * nothing changed while no optional array exists: `_lift`, `_put`, `_as_body`
    and `_of_body` give the same values, dtypes, key order and bytes as the
    base19 functions (copied below as the REFERENCE), so the hash of a body
    cannot move;
  * when the arrays do exist, the matter takes them with it and puts them back,
    through a lift and a put, a flight and a landing, a release;
  * a body made before an array existed (or by hand) lands with 0 in it;
  * water standing in the cell a body lands in keeps its own mix.
Tags: the base19 comparisons are [I] arithmetic identity; the rest are
[I] conservation (what was lifted is what is put)."""
import hashlib
import json
import os

import numpy as np
import pytest

from src.voxel import bodies
from src.voxel.bodies import AS_BODY, CARRIED, CARRIED_OPT, OPT_BODY, Lifted
from src.voxel.materials import IRON, STONE, WATER
from src.voxel.sim import World

# The two lists as base19 has them (and the oracle test pins them).
BASE19_CARRIED = ("mat", "smass", "E", "fl", "fvol", "fpot", "edge", "fallh", "rubble", "eid")
BASE19_AS_BODY = ("mats", "masses", "Es", "fls", "fvols", "fpots", "edges", "fallhs",
                  "rubbles", "eids")
OPT_DTYPES = {"scomp": np.uint32, "fcomp": np.uint32, "dmg": np.float32,
              "bzone": np.uint16, "ice_g": np.float32, "wood_sp": np.uint8, "wood_grain": np.uint8,
              "wood_bound": np.float32}


# ── the base19 functions, verbatim: the REFERENCE ────────────────────────────
def ref_lift(w, idx):
    vals = tuple(getattr(w, f)[idx].copy() for f in BASE19_CARRIED)
    for f in BASE19_CARRIED:
        getattr(w, f)[idx] = 0
    w._things_moved(vals[-1])
    return vals


def ref_put(w, idx, vals):
    for f, v in zip(BASE19_CARRIED, vals):
        getattr(w, f)[idx] = v


def ref_as_body(vals):
    return dict(zip(BASE19_AS_BODY, vals))


def ref_of_body(w, b, i):
    out = []
    for f, k in zip(BASE19_CARRIED, BASE19_AS_BODY):
        v = b.get(k)
        out.append(np.zeros_like(b["masses"][i], dtype=getattr(w, f).dtype)
                   if v is None else v[i])
    return tuple(out)


def canon(x):
    """A byte-exact hash of arrays, tuples, lists and dicts (insertion order is
    state), as the golden recorder reads them."""
    h = hashlib.blake2b(digest_size=16)

    def go(o):
        if isinstance(o, np.ndarray):
            h.update(b"a" + o.dtype.str.encode() + repr(o.shape).encode())
            h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, np.generic):
            h.update(b"n" + o.dtype.str.encode() + o.tobytes())
        elif isinstance(o, dict):
            h.update(b"d%d" % len(o))
            for k, v in o.items():
                go(k)
                go(v)
        elif isinstance(o, (list, tuple)):
            h.update((b"l%d" if isinstance(o, list) else b"t%d") % len(o))
            for v in o:
                go(v)
        else:
            h.update(b"r" + repr(o).encode())
    go(x)
    return h.hexdigest()


def world_digest(w):
    return canon([getattr(w, f) for f in BASE19_CARRIED])


# ── worlds ───────────────────────────────────────────────────────────────────
def scattered(seed, shape=(14, 10, 12), n=70):
    """A world with n random cells holding random matter in every CARRIED
    field; returns it and the index tuple of those cells."""
    rng = np.random.default_rng(seed)
    w = World(*shape)
    pts = rng.choice(int(np.prod(shape)), size=n, replace=False)
    idx = np.unravel_index(pts, shape)
    w.mat[idx] = rng.integers(1, 12, n)
    w.smass[idx] = rng.uniform(1, 500, n)
    w.E[idx] = rng.uniform(-50, 900, n)
    w.fl[idx] = rng.integers(0, 9, n)
    w.fvol[idx] = rng.uniform(0, 30, n)
    w.fpot[idx] = rng.uniform(0, 1, n)
    w.edge[idx] = rng.uniform(0, 1e-4, n)
    w.fallh[idx] = rng.uniform(0, 3, n)
    w.rubble[idx] = rng.random(n) < 0.3
    w.eid[idx] = rng.integers(0, 5, n)
    return w, idx


def free_cells(w, idx, n=None):
    """n cells (the same number as idx) that hold nothing: distinct, in order."""
    n = len(idx[0]) if n is None else n
    taken = np.zeros(w.shape, bool)
    taken[idx] = True
    taken |= w.mat != 0
    free = np.argwhere(~taken)[:n]
    assert len(free) == n
    return tuple(free.T)


def with_optional(w, idx, seed, which=None):
    """Make the optional arrays named in `which` (default all) and put random
    values on the cells `idx`. Returns {field: the values put}."""
    rng = np.random.default_rng(seed + 991)
    put = {}
    for f in (which or [f for f, _ in CARRIED_OPT]):
        dt = OPT_DTYPES[f]
        a = np.zeros(w.shape, dt)
        n = len(idx[0])
        vals = (rng.uniform(0.0, 0.9, n) if np.dtype(dt).kind == "f"
                else rng.integers(1, 2 ** 15, n)).astype(dt)
        a[idx] = vals
        setattr(w, f, a)
        put[f] = vals
    return put


def floor_world(nx=20, ny=12, nz=16, cm=10):
    w = World(nx, ny, nz, voxel_cm=cm)
    w.open_sky = False
    w.fill(0, nx, 0, ny, 0, 1, STONE)
    return w


# ═════════════════════════════════════════════════════════════════════════════
# 1. the lists
# ═════════════════════════════════════════════════════════════════════════════
def test_CARRIED_and_AS_BODY_are_exactly_base19_and_the_optional_list_is_the_plans():
    assert CARRIED == BASE19_CARRIED and AS_BODY == BASE19_AS_BODY
    assert CARRIED_OPT == (("scomp", "scomps"), ("fcomp", "fcomps"), ("dmg", "dmgs"),
                           ("bzone", "bzones"), ("ice_g", "ice_gs"),
                           ("wood_sp", "wood_sps"), ("wood_grain", "wood_grains"),
                           ("wood_bound", "wood_bounds"))
    assert OPT_BODY == ("scomps", "fcomps", "dmgs", "bzones", "ice_gs", "wood_sps", "wood_grains", "wood_bounds")
    assert not set(f for f, _ in CARRIED_OPT) & set(CARRIED), "an optional field is not carried"
    assert not set(OPT_BODY) & set(AS_BODY)


def test_CARRIED_and_AS_BODY_equal_the_frozen_oracle_though_bodies_py_has_changed():
    """tests/test_oracle_format.py compares the live tables with the oracle only while the source
    files the oracle was frozen from (bodies.py among them) are byte-for-byte the same, and skips
    otherwise ("read the diff first"). bodies.py is edited now, so those live checks skip. The
    diff was read (tools/matrec_extract.py diff <oracle> <tree> --strict: 0 changed rows); this
    keeps the one thing that file pinned about bodies.py pinned."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "matrec_oracle_base19.json")) as fh:
        doc = json.load(fh)
    assert list(CARRIED) == doc["carried"] and list(AS_BODY) == doc["as_body"]


def test_a_fresh_world_has_no_optional_array():
    w = World(8, 8, 8)
    for f, _ in CARRIED_OPT:
        assert getattr(w, f, None) is None, f


# ═════════════════════════════════════════════════════════════════════════════
# 2. NOTHING MOVES while no optional array exists (arithmetic identity vs base19)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("seed", range(8))
def test_lift_and_put_equal_the_base19_functions_bit_for_bit(seed):
    wa, idx = scattered(seed)
    wb, _ = scattered(seed)
    sub = tuple(a[::2] for a in idx)                     # lift only some of them
    ref = ref_lift(wa, sub)
    new = wb._lift(sub)
    assert isinstance(new, tuple) and isinstance(new, Lifted)
    assert len(new) == len(CARRIED) == len(ref)
    assert [v.dtype for v in new] == [v.dtype for v in ref]
    assert canon(tuple(new)) == canon(ref)
    assert new.opt == {}, "no optional array exists: nothing optional is lifted"
    assert world_digest(wa) == world_digest(wb), "and the cells are left the same"
    # put it back somewhere else, the same way
    far = free_cells(wa, idx, len(sub[0]))               # somewhere else, distinct cells
    ref_put(wa, far, ref)
    wb._put(far, new)
    assert world_digest(wa) == world_digest(wb)
    wb._put(sub, list(new))                              # a list is a fine `vals`, too
    ref_put(wa, sub, list(ref))
    assert world_digest(wa) == world_digest(wb)


@pytest.mark.parametrize("seed", range(4))
def test_the_body_dict_of_a_lift_has_exactly_the_AS_BODY_keys_and_the_same_hash(seed):
    wa, idx = scattered(seed)
    wb, _ = scattered(seed)
    ref = ref_as_body(ref_lift(wa, idx))
    new = wb._as_body(wb._lift(idx))
    assert list(new) == list(AS_BODY) == list(ref), "same keys, same order"
    assert canon(new) == canon(ref)
    assert canon(wb._as_body(tuple(ref_lift(scattered(seed)[0], idx)))) == canon(ref), \
        "a plain tuple (no .opt) takes the same road"


@pytest.mark.parametrize("seed", range(4))
def test_of_body_equals_base19_for_a_body_with_and_without_the_newer_keys(seed):
    w, idx = scattered(seed)
    b = w._as_body(w._lift(idx))
    cells = np.stack(idx, 1).astype(np.float32)
    b["cells"] = cells
    for drop in (None, "fallhs", "rubbles"):             # a body made before they existed
        bb = dict(b)
        if drop:
            del bb[drop]
        for i in (0, 3, len(cells) - 1):
            assert canon(tuple(w._of_body(bb, i))) == canon(ref_of_body(w, bb, i))
        assert isinstance(w._of_body(bb, 0), Lifted) and w._of_body(bb, 0).opt == {}


def test_a_launched_body_has_the_same_keys_in_the_same_order_as_before():
    w = floor_world()
    w.fill(5, 8, 4, 5, 1, 2, IRON)
    b = w._launch(np.argwhere(w.mat == IRON), (2.0, 0.0, 3.0))
    assert list(b)[:1 + len(AS_BODY)] == ["cells", *AS_BODY]
    assert not set(OPT_BODY) & set(b), "no optional array, no optional key"
    assert all(len(b[k]) == 3 for k in AS_BODY)


def test_the_positional_readers_of_a_lift_still_work():
    """The eid is LAST and `fallh` is found by CARRIED.index: callers (_lift ->
    support.py's fall height, `_things_moved(vals[-1])`) read by position."""
    w, idx = scattered(1)
    with_optional(w, idx, 1)
    got = w._lift(idx)
    assert len(got) == len(CARRIED)
    assert np.array_equal(got[-1], got[CARRIED.index("eid")])
    assert got[CARRIED.index("fallh")].dtype == np.float32
    assert set(got.opt) == {f for f, _ in CARRIED_OPT}


# ═════════════════════════════════════════════════════════════════════════════
# 3. with the arrays present, matter takes them along
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("which", [None, ["dmg"], ["scomp", "fcomp"], ["bzone", "ice_g"]])
def test_a_lift_takes_the_optional_values_and_empties_the_cells(which):
    w, idx = scattered(2)
    put = with_optional(w, idx, 2, which)
    got = w._lift(idx)
    assert set(got.opt) == set(put)
    for f, v in put.items():
        assert np.array_equal(got.opt[f], v) and got.opt[f].dtype == v.dtype
        assert not getattr(w, f).any(), f"{f} is left empty where the matter was"
    for f in CARRIED:
        assert not getattr(w, f)[idx].any()


@pytest.mark.parametrize("seed", range(6))
def test_lift_then_put_elsewhere_moves_every_optional_value_with_its_cell(seed):
    w, idx = scattered(seed)
    put = with_optional(w, idx, seed)
    before = {f: getattr(w, f).copy() for f in put}
    dst = free_cells(w, idx)
    assert not any(getattr(w, f)[dst].any() for f in put)
    got = w._lift(idx)
    w._put(dst, got)
    for f in put:
        assert np.array_equal(getattr(w, f)[dst], before[f][idx]), f
        assert getattr(w, f).sum(dtype=np.float64) == pytest.approx(
            before[f].sum(dtype=np.float64)), f"{f}: nothing made, nothing lost"


def test_put_back_where_it_was_restores_the_world_exactly():
    w, idx = scattered(5)
    with_optional(w, idx, 5)
    snap = [getattr(w, f).copy() for f in BASE19_CARRIED + tuple(f for f, _ in CARRIED_OPT)]
    got = w._lift(idx)
    w._put(idx, got)
    for a, f in zip(snap, BASE19_CARRIED + tuple(f for f, _ in CARRIED_OPT)):
        assert np.array_equal(a, getattr(w, f)), f


def test_a_lift_in_a_world_with_one_array_leaves_the_others_alone():
    w, idx = scattered(6)
    put = with_optional(w, idx, 6, ["dmg"])
    got = w._lift(idx)
    assert set(got.opt) == {"dmg"}
    for f in ("scomp", "fcomp", "bzone", "ice_g"):
        assert getattr(w, f, None) is None
    b = w._as_body(got)
    assert "dmgs" in b and not {"scomps", "fcomps", "bzones", "ice_gs"} & set(b)
    assert np.array_equal(b["dmgs"], put["dmg"])


def test_a_put_of_plain_values_writes_zero_into_an_optional_array_that_exists():
    """Matter lifted before the array existed (or a tuple made by hand) has no
    value for it: the cells get 0, they never keep what they had."""
    w, idx = scattered(7)
    w_old, _ = scattered(7)
    got = w_old._lift(idx)                               # lifted when no array existed
    with_optional(w, idx, 7, ["dmg", "scomp"])           # ... and the arrays came later
    assert w.dmg[idx].all() and w.scomp[idx].all()
    w._put(idx, tuple(got))
    assert not w.dmg[idx].any() and not w.scomp[idx].any()
    # the explicit form: opt= names only some
    w._put(idx, tuple(got), {"dmg": np.full(len(idx[0]), 0.25, np.float32)})
    assert (w.dmg[idx] == np.float32(0.25)).all() and not w.scomp[idx].any()


def test_an_array_that_has_gone_since_the_lift_is_ignored_by_the_put():
    w, idx = scattered(8)
    with_optional(w, idx, 8, ["fcomp"])
    got = w._lift(idx)
    w.fcomp = None                                       # (a restore to an older snapshot)
    w._put(idx, got)                                     # no error, nothing to write into
    assert w.fcomp is None


# ═════════════════════════════════════════════════════════════════════════════
# 4. a free body: the keys, a flight, a landing
# ═════════════════════════════════════════════════════════════════════════════
def _block(w):
    """A block of iron in the air, wet (a fluid mix `fcomp` means something only
    where there is fluid, so the block holds a little water)."""
    w.fill(5, 8, 4, 6, 6, 8, IRON)
    cells = np.argwhere(w.mat == IRON)
    idx = tuple(cells.T)
    w.fl[idx], w.fvol[idx], w.fpot[idx] = WATER, 2.0, 1.0
    return cells, idx


def test_a_launched_body_carries_the_optional_keys_that_exist():
    w = floor_world()
    cells, idx = _block(w)
    put = with_optional(w, idx, 11, ["scomp", "dmg"])
    b = w._launch(cells, (0.0, 0.0, 0.0))
    assert "scomps" in b and "dmgs" in b and not {"fcomps", "bzones", "ice_gs"} & set(b)
    assert np.array_equal(b["scomps"], put["scomp"]) and np.array_equal(b["dmgs"], put["dmg"])
    assert len(b["scomps"]) == len(b["cells"]) == len(cells)
    assert not w.scomp.any() and not w.dmg.any(), "it left the lattice with its mix and damage"


@pytest.mark.parametrize("seed", range(3))
def test_a_flight_and_a_landing_put_every_optional_value_back_on_the_lattice(seed):
    w = floor_world(30, 14, 20)
    cells, idx = _block(w)
    put = with_optional(w, idx, 20 + seed)
    totals = {f: float(getattr(w, f).sum(dtype=np.float64)) for f in put}
    vel = [(0.0, 0.0, 0.0), (3.0, 0.0, 2.0), (1.0, 1.0, 4.0)][seed]
    b = w._launch(cells, vel)
    assert not any(getattr(w, f).any() for f in put)
    for _ in range(400):
        w._law_bodies()
        if not w.bodies:
            break
    assert not w.bodies, "it came down"
    landed = np.argwhere(w.mat == IRON)
    assert len(landed) == len(cells), "all of it landed on the lattice"
    for f in put:
        a = getattr(w, f)
        assert a.dtype == OPT_DTYPES[f]
        assert float(a.sum(dtype=np.float64)) == pytest.approx(totals[f], rel=1e-6), f
        assert sorted(a[tuple(landed.T)].tolist()) == sorted(put[f].tolist()), \
            f"{f}: the same values, in the cells the matter now fills"
        assert np.count_nonzero(a) == len(cells), f"{f}: and nowhere else"


def test_a_wedged_topple_puts_every_optional_row_back_where_it_stood():
    """`_topple` lifts a standing thing, finds it wedged against a wall and sets it back down:
    the lift and the put of that road are the same pair, so the optional rows must come back."""
    w = World(12, 6, 16)
    w.fill(0, 12, 0, 6, 0, 1, STONE)
    w.fill(6, 7, 0, 6, 1, 14, STONE)                   # the wall it would lean into
    w.fill(5, 6, 3, 4, 1, 13, IRON)
    cells = np.argwhere(w.mat == IRON)
    idx = tuple(cells.T)
    put = with_optional(w, idx, 31)
    before = {f: getattr(w, f).copy() for f in put}
    w._topple(cells, 0, 1, 5.5, 0.5, owner="nobody")
    assert not w.bodies, "it is wedged: nothing left the lattice"
    for f in put:
        assert np.array_equal(getattr(w, f), before[f]), f"{f} is back where it was"


def test_a_man_who_jumps_lands_whole_with_his_own_damage_and_overlay_on_his_own_flesh():
    """The person paths (a launch with an owner takes his limbs and what he holds; the landing
    sets him down by the skeleton's shears, not by rounding) carry the optional rows too."""
    from src.voxel.materials import FLESH
    from src.voxel.scenes import _person
    w = floor_world(40, 30, 60, cm=5)
    p = _person(w, 14, 12)
    w.step()
    comp, sl = w._person_cells(p)
    cells = w._claim_cells(p, comp, sl)
    idx = tuple(cells.T)
    flesh = w.mat[idx] == FLESH
    rng = np.random.default_rng(3)
    w.dmg = np.zeros(w.shape, np.float32)
    w.bzone = np.zeros(w.shape, np.uint16)
    w.dmg[idx] = rng.uniform(0.0, 0.9, len(cells)).astype(np.float32)
    w.bzone[idx] = rng.integers(1, 200, len(cells)).astype(np.uint16)
    want = {f: sorted(getattr(w, f)[idx].tolist()) for f in ("dmg", "bzone")}
    b = w._launch(cells, (0.0, 0.0, 2.0), owner=p["name"])
    assert len(b["dmgs"]) == len(b["cells"]) == len(b["bzones"])
    assert not w.dmg.any() and not w.bzone.any()
    for _ in range(400):
        w._law_bodies()
        if not w.bodies:
            break
    assert not w.bodies, "he came down"
    on = np.argwhere(w.mat != 0)
    on = on[on[:, 2] > 0]                                 # not the floor
    got = {f: sorted(getattr(w, f)[tuple(on.T)].tolist()) for f in ("dmg", "bzone")}
    assert got == want, "every row of damage and overlay is on a cell of him, none lost, none made"
    assert int(np.count_nonzero(w.dmg)) == int(np.count_nonzero(want["dmg"]))
    assert flesh.sum() == (w.mat == FLESH).sum(), "and he is all there"


def test_a_body_launched_before_the_arrays_existed_lands_with_zeros():
    w = floor_world(30, 14, 20)
    cells, idx = _block(w)
    b = w._launch(cells, (2.0, 0.0, 2.0))
    assert not set(OPT_BODY) & set(b)
    for f, dt in OPT_DTYPES.items():
        setattr(w, f, np.zeros(w.shape, dt))
        getattr(w, f)[2, 2, 2] = 9                       # something unrelated, elsewhere
    for _ in range(400):
        w._law_bodies()
        if not w.bodies:
            break
    assert not w.bodies
    landed = tuple(np.argwhere(w.mat == IRON).T)
    for f in OPT_DTYPES:
        a = getattr(w, f)
        assert not a[landed].any(), f"{f}: the body had no such key, so 0"
        assert a[2, 2, 2] == 9, f"{f}: an unrelated cell is not touched"


def test_a_hand_made_body_lands_with_zeros_in_the_optional_arrays():
    w = floor_world()
    for f, dt in OPT_DTYPES.items():
        setattr(w, f, np.zeros(w.shape, dt))
    b = w.loose((4, 4, 3), (0.0, 0.0, -1.0), IRON, 100.0)
    assert not set(OPT_BODY) & set(b)
    for _ in range(100):
        w._law_bodies()
        if not w.bodies:
            break
    assert not w.bodies and (w.mat == IRON).sum() == 1
    for f in OPT_DTYPES:
        assert not getattr(w, f).any()


def test_water_standing_where_a_body_lands_keeps_its_mix_and_a_wet_body_brings_its_own():
    """`_land_body` keeps the fluid standing in the cell when the arriving
    matter brings none (that is old behaviour); the fluid's mix `fcomp` is part
    of the fluid, so it stays too. A body that brings fluid brings its mix."""
    for brings in (False, True):
        w = floor_world()
        w.fcomp = np.zeros(w.shape, np.uint32)
        w.fl[4, 4, 2], w.fvol[4, 4, 2], w.fpot[4, 4, 2] = WATER, 5.0, 1.0
        w.fcomp[4, 4, 2] = 7
        b = w.loose((4, 4, 2), (0.0, 0.0, -1.0), IRON, 100.0)
        b["fcomps"] = np.array([9], np.uint32)
        if brings:
            b["fls"] = np.array([WATER], np.int16)
            b["fvols"] = np.array([2.0], np.float32)
            b["fpots"] = np.array([1.0], np.float32)
        w._land_body(b, None)
        assert w.mat[4, 4, 2] == IRON
        if brings:
            assert w.fvol[4, 4, 2] == np.float32(2.0) and w.fcomp[4, 4, 2] == 9
        else:
            assert w.fvol[4, 4, 2] == np.float32(5.0) and w.fcomp[4, 4, 2] == 7, \
                "the water that stood there is still that water, with its mix"
        assert int(w.fl[4, 4, 2]) == WATER


# ═════════════════════════════════════════════════════════════════════════════
# 5. letting go of part of a swinging body keeps every per-cell array in step
# ═════════════════════════════════════════════════════════════════════════════
def test_a_release_splits_the_optional_rows_with_the_rest():
    w = floor_world()
    n = 6
    cells = np.array([[8 + i, 6, 6] for i in range(n)], np.float32)
    b = {"cells": cells.copy(), "fly": False, "owner": None, "axis": 0, "s": 1,
         "pivot": 7.0, "zb": 6.0, "theta": 0.3, "omega": 0.05, "phi0": 0.02, "L": 2.0,
         "mats": np.full(n, IRON, np.int16), "masses": np.full(n, 100.0, np.float32),
         "Es": np.zeros(n, np.float32), "fls": np.zeros(n, np.int16),
         "fvols": np.zeros(n, np.float32), "fpots": np.zeros(n, np.float32),
         "edges": np.zeros(n, np.float32), "fallhs": np.zeros(n, np.float32),
         "rubbles": np.zeros(n, bool), "eids": np.zeros(n, np.int32),
         "scomps": np.arange(1, n + 1, dtype=np.uint32),
         "dmgs": np.linspace(0.0, 0.5, n).astype(np.float32),
         "segof": np.array([0, 0, 0, -1, -1, -1])}
    thrown = b["segof"] == -1
    flier = w._release(b, thrown, np.round(cells))
    assert flier["scomps"].tolist() == [4, 5, 6]
    assert np.array_equal(flier["dmgs"], np.linspace(0.0, 0.5, n).astype(np.float32)[3:])
    assert b["scomps"].tolist() == [1, 2, 3] and len(b["dmgs"]) == 3
    assert all(len(b[k]) == len(b["cells"]) for k in ("scomps", "dmgs", *AS_BODY))
    assert all(len(flier[k]) == len(flier["cells"]) for k in ("scomps", "dmgs", *AS_BODY))
    assert "fcomps" not in flier, "a key the body did not have is not invented"


def test_the_per_cell_list_names_the_optional_keys():
    assert set(OPT_BODY) <= set(bodies.BodiesMixin._PERCELL)
    assert bodies.BodiesMixin._PERCELL == ("cells",) + AS_BODY + OPT_BODY + ("segof",)
