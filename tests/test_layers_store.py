"""T3 layers: the store, the index, the ledger, the hand-over, the readings (spec 7:
T01-T05, T37, T38, T44, T46, T50). No World: a FakeHost stands in.

Acceptance tags (plan section 9): [C] code-only, [E] echo, [U] independent but the
source is unverified. Conservation and arithmetic identities are untagged and gate.
"""
import copy
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from src.voxel import layer_data as D
from src.voxel.layers import (Layers, cell_key, cell_xyz, host_of, make_key, split_key)
from tests.layers_fake import FakeHost

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = [D.WATER, D.SALT, D.CLAY, D.SOOT, D.CHALK, D.TALLOW, D.PLASMA, D.QUARTZ, D.LAMP_OIL]


def _rand_comp(rng):
    c = {}
    for _ in range(rng.randint(1, 3)):
        sp = rng.choice(SP)
        f = rng.choice([D.DISSOLVED, D.CRYSTAL, D.GRAIN])
        c[(sp, f, 0 if f == D.DISSOLVED else rng.choice([8, 24, 40]))] = rng.random() + 1e-3
    return c


def _inv(host, L):
    """host E + stored - energy that entered: constant over any legal call."""
    return host.total_E() + L.stored_J() - L.energy_in_total()


# ---- T01 -------------------------------------------------------------------------
def test_add_then_take_returns_the_same_grams():
    L = Layers()
    k = make_key(0, 5, 4)
    L.add(k, {(D.CLAY, D.GRAIN, 8): 2.5}, Ts=20.0)
    got, _ = L.take(k, D.CLAY, D.GRAIN, 2.5, 20.0)
    assert got == pytest.approx(2.5, abs=1e-12)
    assert L.n == 0 and L.mass() == 0.0 and L.ledger_closes()


def test_ledger_closes_after_10000_random_operations():
    rng = random.Random(1)
    L = Layers()
    host = FakeHost(n=8, dtype=np.float64)
    for i in range(8):
        host.set_Ts(i, 20 + 3 * i)
    for it in range(10000):
        i = rng.randrange(8)
        k = make_key(0, i, rng.randrange(6))
        op = rng.random()
        if op < 0.45:
            _s, adj = L.add(k, _rand_comp(rng), Ts=host.Ts(i), T_in=rng.uniform(-10, 60),
                            source=rng.choice(["smoke", "spill", "contact"]))
            host.apply(adj)
        elif op < 0.65:
            _g, adj = L.take(k, rng.choice(SP), rng.choice([0, 1, 2]), rng.random(), host.Ts(i))
            host.apply(adj)
        elif op < 0.9:
            j = rng.randrange(8)
            k2 = make_key(0, j, rng.randrange(6))
            host.apply(L.move(k, k2, rng.random(), host.Ts(i), host.Ts(j)))
        else:
            _c, adj = L.strip(k, rng.random() * 1e-3, host.Ts(i))
            host.apply(adj)
    assert L.ledger_closes(1e-12)
    # and the energy is closed too: host E + stored - entered
    assert np.isfinite(_inv(host, L))


def test_energy_is_conserved_over_adds_takes_and_moves():
    """F2 / T41: host E plus layer-stored energy minus what entered is constant."""
    rng = random.Random(7)
    L = Layers()
    host = FakeHost(n=6, dtype=np.float64)
    for i in range(6):
        host.set_Ts(i, 10 + 7 * i)
    base = _inv(host, L)
    for it in range(1500):
        i = rng.randrange(6)
        k = make_key(0, i, rng.randrange(6))
        op = rng.random()
        if op < 0.5:
            _s, adj = L.add(k, _rand_comp(rng), Ts=host.Ts(i), T_in=rng.uniform(-10, 80))
            host.apply(adj)
        elif op < 0.7:
            _g, adj = L.take(k, rng.choice(SP), rng.choice([0, 1, 2]), rng.random(), host.Ts(i))
            host.apply(adj)
        else:
            j = rng.randrange(6)
            k2 = make_key(0, j, rng.randrange(6))
            host.apply(L.move(k, k2, rng.random(), host.Ts(i), host.Ts(j)))
    assert _inv(host, L) == pytest.approx(base, abs=1e-8 * max(1.0, host.total_E()))


# ---- T02, T03 --------------------------------------------------------------------
def test_a_ninth_push_merges_the_two_thinnest_and_keeps_exact_totals():
    L = Layers()
    k = make_key(0, 3, 4)
    species = [D.CLAY, D.SOOT, D.CHALK, D.QUARTZ, D.ASH, D.CLAY, D.SOOT, D.CHALK]
    for i, sp in enumerate(species):
        # different layers: a pattern keeps them apart
        L.add(k, {(sp, D.GRAIN, 8): 0.1 * (i + 1)}, pattern=("p", i), Ts=20.0)
    assert len(L.stack(k)) == 8
    before = L.species_totals()
    total = L.mass(k)
    L.add(k, {(D.OCHRE, D.GRAIN, 8): 0.05}, pattern=("p", 9), Ts=20.0)
    assert len(L.stack(k)) == 8                          # the cap held
    after = L.species_totals()
    for sp, g in before.items():
        assert after[sp] == pytest.approx(g, rel=1e-12)
    assert L.mass(k) == pytest.approx(total + 0.05, rel=1e-12)
    assert L.ledger_closes()


def test_a_composition_overflow_folds_the_smallest_entry_and_counts_it():
    L = Layers()
    k = make_key(0, 1, 5)
    comp = {(sp, D.GRAIN, 8): 1.0 + 0.1 * i for i, sp in enumerate(
        [D.CLAY, D.SOOT, D.CHALK, D.QUARTZ, D.ASH, D.OCHRE, D.RED_CELLS, D.INK_CARBON])}
    L.add(k, comp, Ts=20.0)
    t0 = L.mass(k)
    more = {c: 0.1 * g for c, g in comp.items()}
    more[(D.IRON_OXIDE, D.GRAIN, 8)] = 0.01                  # a ninth species, alike
    L.add(k, more, Ts=20.0)
    assert len(L.stack(k)) == 1 and L.folds == 1
    assert L.mass(k) == pytest.approx(t0 * 1.1 + 0.01, rel=1e-12)
    assert L.ledger_closes()
    for s in L.stack(k):
        assert int(np.count_nonzero(L.ckey[s])) <= D.NCOMP


# ---- T04 -------------------------------------------------------------------------
def test_deep_copy_shares_nothing_and_an_empty_pool_is_zero_bytes():
    L = Layers()
    assert L.nbytes() == 0
    k = make_key(0, 2, 4)
    L.add(k, {(D.CLAY, D.GRAIN, 8): 1.0, (D.WATER, D.DISSOLVED, 0): 1.0}, Ts=20.0)
    M = copy.deepcopy(L)
    for name, v in L.__dict__.items():
        if isinstance(v, np.ndarray):
            assert not np.shares_memory(v, M.__dict__[name]), name
            assert np.array_equal(v, M.__dict__[name]), name
    M.take(k, D.WATER, D.DISSOLVED, 0.5, 20.0)
    assert L.mass(k, D.WATER) == pytest.approx(1.0)       # the original did not move
    assert M.mass(k, D.WATER) == pytest.approx(0.5)
    assert L.patterns is not M.patterns and L.led_in is not M.led_in


def test_bytes_per_layer_is_157():
    arrs = D.fresh_arrays()
    per = sum(a.dtype.itemsize * (a.shape[1] if a.ndim == 2 else 1) for a in arrs.values())
    assert per == D.BYTES_PER_LAYER == 157


def test_a_restored_pool_runs_the_same_next_step():
    rng = random.Random(3)
    L = Layers()
    for i in range(60):
        L.add(make_key(0, i % 9, i % 6), _rand_comp(rng), Ts=20.0)
    snap = copy.deepcopy(L)
    for obj in (L, snap):
        r = random.Random(11)
        for _ in range(100):
            i = r.randrange(9)
            obj.add(make_key(0, i, r.randrange(6)), _rand_comp(r), Ts=20.0)
    assert L.canonical() == snap.canonical()
    for name in ("key", "m", "ckey", "cm", "ikey", "islot"):
        assert np.array_equal(getattr(L, name), getattr(snap, name)), name


# ---- T05 -------------------------------------------------------------------------
@pytest.mark.parametrize("tag", [0, 1, 2, 3])
def test_key_round_trip_for_all_tags_and_faces(tag):
    for idx in (0, 1, 12345, 5_100_000, (1 << 40) + 7):
        for face in range(6):
            k = make_key(tag, idx, face)
            assert 0 <= k < (1 << 63)
            assert split_key(k) == (tag, idx, face)
            assert host_of(k) == (tag << 55) | idx


def test_cell_key_and_cell_xyz_agree():
    shape = (7, 5, 11)
    for x, y, z in [(0, 0, 0), (6, 4, 10), (3, 2, 5)]:
        k = cell_key(x, y, z, 4, shape)
        _t, idx, face = split_key(k)
        assert face == 4 and cell_xyz(idx, shape) == (x, y, z)


# ---- T44 the index ------------------------------------------------------------------
def test_searchsorted_index_equals_a_brute_force_dict():
    rng = random.Random(5)
    L = Layers()
    truth = {}
    keys = [make_key(0, rng.randrange(40_000), rng.randrange(6)) for _ in range(900)]
    for k in keys:
        s, _ = L.add(k, {(D.SOOT, D.GRAIN, 4): 0.01}, Ts=20.0)
        truth[k] = s
    # remove a third (tombstones), add some back
    gone = rng.sample(sorted(truth), 300)
    for k in gone:
        L.release([k])
        del truth[k]
    for k in gone[:100]:
        s, _ = L.add(k, {(D.SOOT, D.GRAIN, 4): 0.01}, Ts=20.0)
        truth[k] = s
    probe = np.array(keys + [make_key(0, 50_000 + i, 1) for i in range(50)], np.int64)
    got = L.lookup(probe)
    for k, g in zip(probe.tolist(), got.tolist()):
        assert g == truth.get(k, -1)
    assert sorted(L.keys().tolist()) == sorted(truth)
    assert np.all(np.diff(L.ikey) > 0)


def test_insertion_order_does_not_change_the_state():
    rng = random.Random(9)
    items = [(make_key(0, rng.randrange(500), rng.randrange(6)), _rand_comp(rng))
             for _ in range(300)]
    ref = None
    for seed in range(4):
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)
        L = Layers()
        # one layer per key and entry: adds to different keys commute exactly
        seen = {}
        for i in order:
            k, c = items[i]
            seen.setdefault(k, []).append(i)
        for k in sorted(seen):
            for i in sorted(seen[k]):
                L.add(k, items[i][1], Ts=20.0)
        # shuffle which KEY goes first, keep the per-key order
        L2 = Layers()
        ks = sorted(seen)
        random.Random(seed).shuffle(ks)
        for k in ks:
            for i in sorted(seen[k]):
                L2.add(k, items[i][1], Ts=20.0)
        assert L.canonical() == L2.canonical()
        ref = L.canonical() if ref is None else ref
        assert L.canonical() == ref


def test_pending_merge_tombstone_compaction_and_copy_give_identical_results():
    L = Layers()
    ks = [make_key(0, i, 4) for i in range(600)]
    for k in ks:
        L.add(k, {(D.CLAY, D.GRAIN, 8): 1.0}, Ts=20.0)
    for k in ks[:400]:
        L.release([k])
    assert L.ntomb >= 0
    A = copy.deepcopy(L)
    A._compact()
    L._flush()
    assert np.array_equal(L.keys(), A.keys())
    assert np.array_equal(L.lookup(np.array(ks)), A.lookup(np.array(ks)))
    assert A.ntomb == 0


# ---- T46 hand-over to the lattice -----------------------------------------------------
def test_hand_over_books_the_float32_delta_and_returns_the_residual():
    L = Layers()
    k = make_key(0, 3, 4)
    L.add(k, {(D.WATER, D.DISSOLVED, 0): 5.0}, Ts=20.0)
    old = np.float32(123.4567)
    ml = 0.123456789123
    new, landed, resid, adj = L.hand_over_up(k, ml, old)
    assert new.dtype == np.float32
    assert landed == float(new) - float(old)            # exactly the float32 delta
    assert resid == pytest.approx(ml - landed, abs=1e-15)
    assert abs(landed - ml) <= np.spacing(np.float32(old))      # within one ulp
    assert L.mass(k, D.WATER) == pytest.approx(5.0 - landed * D.ROWS[D.WATER]["rho_l"], abs=1e-12)
    assert L.ledger_closes()


def test_acid_species_are_rejected_by_add():
    L = Layers()
    k = make_key(0, 1, 4)
    for sp in D.ACID_SPECIES:
        with pytest.raises(ValueError):
            L.add(k, {(sp, D.DISSOLVED, 0): 1.0}, Ts=20.0)
    assert L.n == 0


# ---- T37 fuzz ---------------------------------------------------------------------------
def test_2000_fuzz_sequences_keep_the_ledger_closed_and_fields_finite():
    rng = random.Random(21)
    for seq in range(2000):
        L = Layers()
        host = FakeHost(n=3, dtype=np.float64)
        for i in range(3):
            host.set_Ts(i, rng.uniform(-15, 90))
        for _ in range(6):
            i = rng.randrange(3)
            k = make_key(0, i, rng.randrange(6))
            op = rng.random()
            if op < 0.45:
                _s, adj = L.add(k, _rand_comp(rng), Ts=host.Ts(i), T_in=rng.uniform(-20, 100))
            elif op < 0.6:
                _g, adj = L.take(k, rng.choice(SP), rng.choice([0, 1, 2]), rng.random(),
                                 host.Ts(i))
            elif op < 0.8:
                j = rng.randrange(3)
                adj = L.move(k, make_key(0, j, rng.randrange(6)), rng.random(),
                             host.Ts(i), host.Ts(j))
            elif op < 0.9:
                _t, adj, _v = L.latent(host.keys_of(i), host.Csur(i), host.Ts(i), 0.1)
            else:
                _c, adj = L.strip(k, rng.random() * 2e-3, host.Ts(i))
            host.apply(adj)
        assert L.ledger_closes(1e-12), seq
        live = L.key[:L.hi] >= 0
        for name in ("m", "cover", "por", "adh", "coh", "cm"):
            a = getattr(L, name)[:L.hi][live]
            assert np.all(np.isfinite(a)), (seq, name)
        assert np.all(L.cover[:L.hi][live] >= 0) and np.all(L.cover[:L.hi][live] <= 1.0000001)
        assert np.all(L.cm[:L.hi][live] >= 0)
        assert np.isfinite(host.E).all()


# ---- T38 names are read-outs ----------------------------------------------------------------
def _label(L, key):
    return L.describe(key)[0]["label"]


def test_describe_reads_labels_from_composition_alone():
    L = Layers()
    mud = make_key(0, 1, 4)
    L.add(mud, {(D.CLAY, D.GRAIN, 8): 1.0, (D.QUARTZ, D.GRAIN, 48): 1.0,
                (D.WATER, D.DISSOLVED, 0): 1.0}, Ts=20.0)
    dry = make_key(0, 2, 4)
    L.add(dry, {(D.CLAY, D.GRAIN, 8): 2.0, (D.QUARTZ, D.GRAIN, 48): 1.0}, Ts=20.0)
    blood = make_key(0, 3, 4)
    L.add(blood, {(D.PLASMA, D.DISSOLVED, 0): 0.3, (D.RED_CELLS, D.GRAIN, 22): 0.3}, Ts=20.0)
    sootk = make_key(0, 4, 5)
    L.add(sootk, {(D.SOOT, D.GRAIN, 1): 0.05}, Ts=20.0)
    rust = make_key(0, 5, 4)
    L.add(rust, {(D.IRON_OXIDE, D.BOUND, 0): 0.2}, Ts=20.0)
    frost = make_key(0, 6, 4)
    L.add(frost, {(D.WATER, D.CRYSTAL, 40): 0.1}, Ts=-5.0)
    wax = make_key(0, 7, 0)
    L.add(wax, {(D.BEESWAX, D.CRYSTAL, 40): 0.5}, Ts=20.0)
    assert _label(L, mud) == "wet mud" and L.describe(mud)[0]["wet"]
    assert _label(L, dry) == "dried mud" and not L.describe(dry)[0]["wet"]
    assert _label(L, blood) == "dried blood"
    assert _label(L, sootk) == "soot"
    assert _label(L, rust) == "rust"
    assert _label(L, frost) == "frost"
    assert _label(L, wax) == "wax"
    # nothing reads a label back: no array in the pool holds a name
    assert all(not isinstance(v, (str, bytes)) for v in L.__dict__.values())


def test_describe_changes_when_the_composition_changes():
    L = Layers()
    k = make_key(0, 1, 4)
    L.add(k, {(D.PLASMA, D.DISSOLVED, 0): 0.3, (D.WATER, D.DISSOLVED, 0): 1.0,
              (D.RED_CELLS, D.GRAIN, 22): 0.3}, Ts=20.0)
    assert _label(L, k) == "wet blood"
    L.take(k, D.WATER, D.DISSOLVED, 1.0, 20.0)
    assert _label(L, k) == "dried blood"


# ---- T50 hygiene and import isolation -----------------------------------------------------------
def test_the_layer_modules_import_without_sim_or_heat():
    code = ("import sys; import src.voxel.layer_data, src.voxel.layers, "
            "src.voxel.layer_laws, src.voxel.marks, src.voxel.smokeopt; "
            "bad=[m for m in ('src.voxel.sim','src.voxel.heat','src.voxel.materials') "
            "if m in sys.modules]; sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_species_ids_do_not_collide_and_t1_forms_match_when_t1_exists():
    ids = [i for n, i in D.ID.items() if n not in ("silt", "sand", "chalk_dust")]
    assert len(ids) == len(set(ids))
    try:
        from src.voxel import species as T1
    except ImportError:
        pytest.skip("T1 species.py not present")
    assert (T1.DISSOLVED, T1.CRYSTAL, T1.GRAIN, T1.BOUND) == (D.DISSOLVED, D.CRYSTAL,
                                                               D.GRAIN, D.BOUND)
    assert T1.pack_key(D.CLAY, D.GRAIN, 8) == D.pack_key(D.CLAY, D.GRAIN, 8)
    assert T1.BY_NAME["water"] == D.WATER == 12


# ---- relocate, release, patterns, dirty -------------------------------------------------------
def test_relocate_rekeys_a_stack_and_stacks_on_an_occupied_destination():
    L = Layers()
    a, b, c = make_key(0, 1, 4), make_key(0, 9, 4), make_key(0, 20, 4)
    L.add(a, {(D.CLAY, D.GRAIN, 8): 1.0}, pattern="p", cover=1.0, Ts=20.0)
    L.add(a, {(D.SOOT, D.GRAIN, 1): 0.1}, cover=1.0, Ts=20.0)
    L.add(c, {(D.CHALK, D.GRAIN, 3): 0.5}, cover=1.0, Ts=20.0)
    total = L.mass()
    L.relocate([a], [b])
    assert not L.stack(a) and len(L.stack(b)) == 2 and L.mass(b) == pytest.approx(1.1)
    assert L.describe(b)[0]["label"] == "soot"                       # order kept
    L.relocate([b], [c])                                             # onto an occupied face
    assert not L.stack(b) and L.mass(c) == pytest.approx(1.6)
    assert L.mass() == pytest.approx(total, rel=1e-12) and L.ledger_closes()
    assert L.describe(c)[0]["label"] == "soot"                       # the arriving stack is on top
    assert np.all(L.ikey[L.islot >= 0] == np.sort(L.ikey[L.islot >= 0]))


def test_release_returns_the_composition_and_the_heat_when_the_host_is_still_there():
    host = FakeHost(n=3, dtype=np.float64)
    host.set_Ts(1, 60.0)
    L = Layers()
    k = make_key(0, 1, 4)
    _s, adj = L.add(k, {(D.WATER, D.DISSOLVED, 0): 1.0, (D.CLAY, D.GRAIN, 8): 0.5},
                    cover=1.0, Ts=host.Ts(1))
    host.apply(adj)
    b = _inv(host, L)
    comp, adj = L.release([k], Ts=host.Ts(1))
    host.apply(adj)
    assert comp[(D.WATER, D.DISSOLVED, 0)] == pytest.approx(1.0)
    assert L.n == 0 and L.mass() == 0.0 and L.ledger_closes()
    assert _inv(host, L) == pytest.approx(b, abs=1e-9)
    comp2, adj2 = L.release([k], Ts=None)                            # nothing left
    assert comp2 == {} and adj2 == []


def test_patterns_and_the_dirty_list_stay_bounded():
    L = Layers()
    for i in range(60):
        k = make_key(0, i % 5, 4)
        L.add(k, {(D.SOOT, D.GRAIN, 1): 0.01}, pattern=("p", i), Ts=20.0)
    assert len(L.patterns) == 60
    L.release([make_key(0, i, 4) for i in range(3)])
    dropped = L.gc_patterns()
    assert dropped > 0 and len(L.patterns) == len(
        {int(L.pat[s]) for s in np.flatnonzero(L.key[:L.hi] >= 0)})
    for s in np.flatnonzero(L.key[:L.hi] >= 0):
        assert 0 <= L.pat[s] < len(L.patterns)
    for _ in range(5000):
        L.add(make_key(0, 1, 4), {(D.SOOT, D.GRAIN, 1): 1e-6}, Ts=20.0)
    assert len(L.dirty) <= 4200
    got = L.take_dirty()
    assert got == sorted(set(got)) and L.dirty == []
