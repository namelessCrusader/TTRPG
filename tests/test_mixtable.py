"""Wave 13, T1: the mix table (src/voxel/mixtable.py).

Tags: [E] echo, [C] code-only (test_codeonly_*), [U] independent but the number's
source is unverified (test_u_*), [I] independent exact / closed-form / arithmetic.
Only [I] lines gate a scene. Every test that builds a table ends by asserting
check_table() is empty (the table is sound).
"""
import copy

import numpy as np
import pytest

from src.voxel import mixrules as MR
from src.voxel import mixtable as MT
from src.voxel import species as SP
from src.voxel.mixtable import MixTable
from src.voxel.species import pack_key

D, C, G, B = SP.DISSOLVED, SP.CRYSTAL, SP.GRAIN, SP.BOUND
S = SP.BY_NAME
WATER, NACL, CU, SN, ETOH = S["water"], S["sodium_chloride"], S["copper"], S["tin"], S["ethanol"]
Q = MT.INV_QSUM

# component keys a random row may hold: every non-void species as CRYSTAL or BOUND, plus a few
# species that can be DISSOLVED (they all have a fluid view)
POOL = np.array([pack_key(s, f, 0) for s in range(1, 35) for f in (C, B)]
                + [pack_key(s, D, 0) for s in (WATER, CU, SN, ETOH, NACL, S["sucrose"], 3)], np.uint32)


def sound(t):
    assert t.check_table() == [], t.check_table()


def rand_rows(t, rng, n, K=4, alpha=0.7, pool=POOL):
    k = np.stack([rng.choice(pool, K, replace=False) for _ in range(n)])
    return t.intern(k, rng.dirichlet(np.ones(K) * alpha, n))


def tot_array(t, ids, mass, order):
    """Grams per species of the lumps as an array in `order`."""
    d = t.totals(ids, mass)
    return np.array([d.get(s, 0.0) for s in order])


ALL_SP = list(range(1, 35))


# ---- the pure rows (tests 6, 12) -------------------------------------------------------
def test_i_the_pure_rows_fill_0_to_127_and_the_table_starts_sound():
    """[I] 128 rows, nothing more, no row used twice (check_table includes the dedupe)."""
    t = MixTable()
    assert t.n_rows == 128
    sound(t)
    assert (t.refs == 0).all()


def test_i_pure_rows_hold_what_species_says_they_hold():
    """[I] every pure row's components, fractions (to one quantum) and form match
    species.pure_row; empty rows are empty."""
    t = MixTable()
    for r in range(128):
        want = SP.pure_row(r)
        ent = t.entries([r])
        assert len(ent.key) == len(want), r
        got = {(int(s), int(f), int(z)): fr for s, f, z, fr in zip(ent.species, ent.form, ent.size, ent.frac)}
        for sp, fo, sz, fr in want:
            assert got[(sp, fo, sz)] == pytest.approx(fr, abs=Q), r


def test_i_acid_and_weak_acid_pure_rows_are_the_stock_composition():
    """[I] row 64 + ACID is 0.983 sulfuric acid + 0.017 water; WEAK_ACID is 0.05 acetic + 0.95."""
    from src.voxel import materials as mat
    t = MixTable()
    for f, name in ((mat.ACID, "sulfuric_acid"), (mat.WEAK_ACID, "acetic_acid")):
        ent = t.entries([64 + f])
        w = dict(zip(ent.species, ent.frac))
        assert w[S[name]] == pytest.approx(mat.STOCK[f], abs=Q)
        assert w[WATER] == pytest.approx(1 - mat.STOCK[f], abs=Q)
        assert (ent.form == D).all()


def test_i_pure_row_views_are_the_mat_and_fluid_ids():
    """[I] row r (1..26, the listed mats) views as mat r; row 64 + f as fluid f; the pad
    rows are empty; ICE (mat 12) is water BOUND."""
    t = MixTable()
    for r in list(range(1, 27)):
        assert t.matv[r] == r and t.flv[r] == 0, r
    for f in list(range(1, 11)):
        assert t.flv[64 + f] == f and t.matv[64 + f] == 0, f
    for r in list(range(27, 64)) + list(range(75, 128)):
        assert t.row_len([r])[0] == 0
    assert t.entries([12]).species[0] == WATER and t.entries([12]).form[0] == B


def test_i_a_pure_composition_interns_to_its_low_row_number():
    """[I] content finds content: pure water dissolved is row 65, pure iron bound is 3,
    the stock acid is row 67."""
    t = MixTable()
    assert t.parse([("water", D, 0, 1.0)]) == 65
    assert t.parse([("iron", B, 0, 1.0)]) == 3
    assert t.parse([("sulfuric_acid", D, 0, 0.983), ("water", D, 0, 0.017)]) == 67
    assert t.n_rows == 128
    sound(t)


# ---- intern and the quantiser (tests 7, 8) --------------------------------------------------
def test_i_intern_rows_are_sorted_sum_to_2_24_and_have_no_zero_entries():
    """[I] check_table plus the explicit sums, for 5000 random rows of 1-6 components."""
    t = MixTable()
    rng = np.random.default_rng(1)
    for K in (1, 2, 3, 6):
        rand_rows(t, rng, 1000, K=K, alpha=0.4)
    sound(t)
    n = t.n_rows
    cell = np.repeat(np.arange(n), np.diff(t.off))
    sums = np.bincount(cell, weights=t.q.astype(float), minlength=n)
    nonempty = np.diff(t.off) > 0
    assert (sums[nonempty] == MT.QSUM).all()
    assert (t.q > 0).all()


def test_i_intern_dedupes_exactly_and_adds_duplicate_slots():
    """[I] the same content in two key orders is one row; a duplicate key adds; a zero
    fraction drops the slot; an all-zero mix is the empty row 0."""
    t = MixTable()
    ka, kb, kc = (pack_key(s, B, 0) for s in (CU, SN, S["lead"]))
    a = t.intern(np.array([[ka, kb]], np.uint32), np.array([[0.7, 0.3]]))[0]
    b = t.intern(np.array([[kb, ka]], np.uint32), np.array([[0.3, 0.7]]))[0]
    c = t.intern(np.array([[ka, kb, ka]], np.uint32), np.array([[0.4, 0.3, 0.3]]))[0]
    d = t.intern(np.array([[ka, kb, kc]], np.uint32), np.array([[0.7, 0.3, 0.0]]))[0]
    assert a == b == c == d
    assert t.intern(np.array([[ka]], np.uint32), np.array([[0.0]]))[0] == 0
    sound(t)


def test_i_intern_ids_are_deterministic_for_a_batch():
    """[I] the same batch on a fresh table gives the same ids; new rows are numbered in
    order of first occurrence."""
    rng = np.random.default_rng(2)
    k = np.stack([rng.choice(POOL, 3, replace=False) for _ in range(400)])
    f = rng.dirichlet(np.ones(3), 400)
    a, b = MixTable(), MixTable()
    ia, ib = a.intern(k, f), b.intern(k, f)
    assert (ia == ib).all()
    new = ia[ia >= 128]
    first = np.unique(new, return_index=True)[1]
    assert (np.diff(new[np.sort(first)]) > 0).all()
    sound(a)


def test_i_quantiser_error_is_below_one_quantum_and_the_sum_is_exact():
    """[I] test 8: |q 2^-24 - f| < 2^-24 for every kept entry; sum exactly 2^24 (1e5 rows)."""
    rng = np.random.default_rng(3)
    t = MixTable()
    n = 100_000
    keys = np.tile(POOL[:5], (n, 1))
    f = rng.dirichlet(np.ones(5) * 0.3, n)
    cq, kq, qq, _bk = MixTable._quantise(n, np.repeat(np.arange(n), 5), keys.ravel(), f.ravel(),
                                         np.ones(n))
    assert (np.bincount(cq, weights=qq.astype(float), minlength=n) == MT.QSUM).all()
    dense = np.zeros((n, 5))
    col = {int(k): i for i, k in enumerate(POOL[:5])}
    dense[cq, [col[int(k)] for k in kq]] = qq * Q
    assert np.abs(dense - f).max() < Q
    del t


def test_i_a_species_below_the_quantum_cannot_be_held():
    """[I] a fraction below 2^-25 rounds to nothing (spec 4.3: traces live in records)."""
    t = MixTable()
    ka, kb = pack_key(CU, B, 0), pack_key(SN, B, 0)
    r = t.intern(np.array([[ka, kb]], np.uint32), np.array([[1.0, 1e-9]]))[0]
    assert t.row_len([r])[0] == 1 and r == t.parse([("copper", B, 0, 1.0)])
    sound(t)


def test_i_the_fluid_view_refuses_a_dissolved_species_with_no_fluid_and_no_like():
    """[I] kaolinite DISSOLVED has neither fl nor fluid_like: intern raises and rolls the
    table back untouched; a dissolved species with a fluid_like (sucrose) is fine."""
    t = MixTable()
    n = t.n_rows
    with pytest.raises(ValueError, match="no fluid view"):
        t.parse([("kaolinite", D, 0, 1.0)])
    assert t.n_rows == n
    sound(t)
    t.parse([("sucrose", D, 0, 0.3), ("water", D, 0, 0.7)])
    sound(t)


# ---- identity: blend(a, a), take then put (test 8) -------------------------------------------
def test_i_self_blend_returns_the_same_row_for_1e5_random_cases():
    """[I] blend(a, a) is a no-op on the row."""
    rng = np.random.default_rng(4)
    t = MixTable()
    ids = rand_rows(t, rng, 2000, K=5, alpha=0.5)
    pick = rng.integers(0, 2000, 100_000)
    m = rng.uniform(0.1, 100, 100_000)
    i2, m2 = t.blend(ids[pick], m, ids[pick], m * 0.37)
    assert (i2 == ids[pick]).all()
    assert (m2 == m + m * 0.37).all()
    sound(t)


def test_i_blend_of_content_equal_rows_is_the_same_row_even_with_zero_mass_on_one_side():
    """[I] a lump blended with an empty lump keeps its row."""
    rng = np.random.default_rng(5)
    t = MixTable()
    ids = rand_rows(t, rng, 500)
    m = rng.uniform(1, 9, 500)
    i2, m2 = t.blend(ids, m, np.zeros(500, np.int32), np.zeros(500))
    assert (i2 == ids).all() and (m2 == m).all()
    i3, _ = t.blend(np.zeros(500, np.int32), np.zeros(500), ids, m)
    assert (i3 == ids).all()


@pytest.mark.parametrize("alpha", [0.3, 1.0, 6.0])
def test_i_take_then_put_of_the_same_grams_gives_back_the_same_row(alpha):
    """[I] test 8, 1e5 random cases per alpha: take g of a component, put g back. The
    quantiser is floor + largest remainder, so this is exact whenever the rounding
    error of the take, divided by the growth m / (m - g), is under half a unit; the
    measured failure count is asserted to be 0 on this seed (a regression guard), and
    the dense result is always within 2 units."""
    rng = np.random.default_rng(6)
    t = MixTable()
    bad = far = 0
    for _ in range(100):
        ids = rand_rows(t, rng, 1000, K=int(rng.integers(2, 7)), alpha=alpha)
        m = rng.uniform(0.5, 100, 1000)
        ent = t.entries(ids)
        first = np.r_[0, np.cumsum(t.row_len(ids))[:-1]]
        pick = first + rng.integers(0, t.row_len(ids))
        key = ent.key[pick]
        g = ent.frac[pick] * m * rng.uniform(0.0, 0.99, 1000)
        a, ma = t.take(ids, m, np.arange(1000), key, g)
        b, mb = t.put(a, ma, np.arange(1000), key, g)
        bad += int((b != ids).sum())
        if (b != ids).any():
            w = np.flatnonzero(b != ids)
            d = np.abs(t.to_dense(b[w], ALL_SP, 1.0) - t.to_dense(ids[w], ALL_SP, 1.0)).max()
            far += int(d > 2 * Q)
        assert np.allclose(mb, m, rtol=1e-12, atol=1e-12)
    assert far == 0
    assert bad == 0, bad
    sound(t)


# ---- conservation (tests 9, 10, 11) ---------------------------------------------------------------
def test_i_blend_conserves_grams_of_every_species_up_to_quant():
    """[I] test 9: totals(in) = totals(out) + the quant booked, to 1e-12 relative; and
    blend(a, b) equals blend(b, a) in content (same digest)."""
    rng = np.random.default_rng(7)
    t = MixTable()
    ia = rand_rows(t, rng, 3000)
    ib = rand_rows(t, rng, 3000)
    ma, mb = rng.uniform(0.1, 50, 3000), rng.uniform(0.1, 50, 3000)
    q0 = dict(t._ledgers["quant"])
    i1, m1 = t.blend(ia, ma, ib, mb)
    before = tot_array(t, np.concatenate([ia, ib]), np.concatenate([ma, mb]), ALL_SP)
    after = tot_array(t, i1, m1, ALL_SP)
    dq = np.array([t.ledger("quant").get(s, 0.0) - q0.get(s, 0.0) for s in ALL_SP])
    assert np.abs(before - after - dq).max() < 1e-12 * before.sum()
    j1, _ = t.blend(ib, mb, ia, ma)
    assert (t.content_digest(i1) == t.content_digest(j1)).all()
    sound(t)


def test_i_one_hundred_thousand_random_operations_conserve_species_and_quant_stays_small():
    """[I] test 10: take, put, blend, convert and split on 1000 lumps (and 1000 spare empty
    cells for the split pieces): per-species totals plus quant constant to 1e-12 relative;
    quant itself below 1e-6 of the total mass. 1000 rounds of 100 operations."""
    rng = np.random.default_rng(8)
    t = MixTable()
    N = 1000
    ids = np.zeros(2 * N, np.int32)
    mass = np.zeros(2 * N)
    ids[:N] = rand_rows(t, rng, N, K=4, alpha=0.6)
    mass[:N] = rng.uniform(1, 100, N)
    T0 = tot_array(t, ids, mass, ALL_SP)
    total = T0.sum()
    ops = ("move", "blend", "convert", "split")
    done = 0
    for rnd in range(1000):
        op = ops[rnd % 4]
        live = np.flatnonzero(ids != 0)
        a = rng.choice(live, 100, replace=False)
        if op == "move":
            b = rng.choice(live, 100, replace=False)
            keep = ~np.isin(b, a)
            a, b = a[keep], b[keep]
            ent = t.entries(ids[a])
            first = np.r_[0, np.cumsum(t.row_len(ids[a]))[:-1]]
            key = ent.key[first + rng.integers(0, t.row_len(ids[a]))]
            g = rng.uniform(0, 0.8, len(a)) * mass[a] * 0.3
            i1, m1, taken = t.take(ids[a], mass[a], np.arange(len(a)), key, g, return_taken=True)
            i2, m2 = t.put(ids[b], mass[b], np.arange(len(b)), key, taken)
            ids[a], mass[a], ids[b], mass[b] = i1, m1, i2, m2
        elif op == "blend":
            b = rng.choice(live, 100, replace=False)
            keep = ~np.isin(b, a)
            a, b = a[keep], b[keep]
            i1, m1 = t.blend(ids[a], mass[a], ids[b], mass[b])
            ids[a], mass[a] = i1, m1
            ids[b], mass[b] = 0, 0.0
        elif op == "convert":
            sp = np.array([WATER, CU, SN, ETOH, NACL])[rng.integers(0, 5, len(a))]
            ids_a = ids[a]
            ent = t.entries(ids_a)
            # convert whichever DISSOLVED/BOUND component the lump has: use its first line
            first = np.r_[0, np.cumsum(t.row_len(ids_a))[:-1]]
            sel = first + rng.integers(0, t.row_len(ids_a))
            sp, fo = ent.species[sel], ent.form[sel]
            to = np.where(fo == D, C, D)
            to = np.where(sp == 3, D, to)
            ok = ~((to == D) & ~np.isin(sp, [WATER, CU, SN, ETOH, NACL, S["sucrose"], 3]))
            idx = np.flatnonzero(ok)
            if len(idx):
                g = ent.frac[sel][idx] * mass[a][idx] * rng.uniform(0, 1, len(idx))
                fs = ent.size[sel][idx]
                i1, m1 = t.convert(ids_a[idx], mass[a][idx], np.arange(len(idx)), sp[idx], fo[idx],
                                   to[idx], g, from_size=fs)
                ids[a[idx]], mass[a[idx]] = i1, m1
        else:
            empty = np.flatnonzero(ids == 0)
            empty = empty[empty >= N][:len(a)]
            a = a[:len(empty)]
            ia, m1, ib, m2 = t.split(ids[a], mass[a], mass[a] * rng.uniform(0.1, 0.9, len(a)))
            ids[a], mass[a], ids[empty], mass[empty] = ia, m1, ib, m2
        done += len(a)
    assert done > 20_000
    T1 = tot_array(t, ids, mass, ALL_SP)
    qv = np.array([t.ledger("quant").get(s, 0.0) for s in ALL_SP])
    assert np.abs(T1 + qv - T0).max() < 1e-12 * total, np.abs(T1 + qv - T0).max() / total
    assert np.abs(qv).sum() < 1e-6 * total
    assert abs(qv.sum()) < 1e-9 * total                      # a signed ledger: it sums to ~0
    sound(t)


def test_i_convert_keeps_species_grams_changes_the_form_and_stops_at_what_is_there():
    """[I] test 11: 5 g of NaCl crystal dissolve into brine; asking for more than there is
    converts what there is; a missing source component converts nothing."""
    t = MixTable()
    sz = MT.SZ_NUCLEI
    r = t.parse([("water", D, 0, 0.9), ("sodium_chloride", C, sz, 0.1)])
    i1, m1, got = t.convert([r], [100.0], [0], [NACL], [C], [D], [5.0], from_size=sz, return_taken=True)
    assert got[0] == pytest.approx(5.0, abs=1e-12) and m1[0] == pytest.approx(100.0)
    d = t.totals(i1, m1, by_key=True)
    assert d[pack_key(NACL, C, sz)] == pytest.approx(5.0, abs=2e-5)
    assert d[pack_key(NACL, D, 0)] == pytest.approx(5.0, abs=2e-5)
    i2, m2, got = t.convert(i1, m1, [0], [NACL], [C], [D], [50.0], from_size=sz, return_taken=True)
    assert got[0] == pytest.approx(5.0, abs=2e-5)
    assert t.totals(i2, m2)[NACL] == pytest.approx(10.0, abs=2e-5)
    i3, m3, got = t.convert(i2, m2, [0], [NACL], [G], [D], [1.0], from_size=0, return_taken=True)
    assert got[0] == 0.0 and i3[0] == i2[0]
    sound(t)


def test_i_convert_rejects_two_requests_for_one_component_of_one_lump():
    """[I] the one-request-per-lump rule is enforced, not assumed."""
    t = MixTable()
    r = t.parse([("water", D, 0, 0.9), ("sodium_chloride", C, MT.SZ_NUCLEI, 0.1)])
    with pytest.raises(ValueError, match="one request"):
        t.convert([r, r], [10.0, 10.0], [0, 0], [NACL, NACL], [C, C], [D, D], [1.0, 1.0],
                  from_size=MT.SZ_NUCLEI)


def test_i_take_more_than_there_is_takes_what_there_is_and_taking_all_empties_the_lump():
    """[I] over-asking clamps; taking every component gives row 0 and mass 0."""
    t = MixTable()
    r = t.parse([("copper", B, 0, 0.7), ("tin", B, 0, 0.3)])
    kc, kt = pack_key(CU, B, 0), pack_key(SN, B, 0)
    i1, m1, got = t.take([r], [10.0], [0], [kc], [99.0], return_taken=True)
    assert got[0] == pytest.approx(7.0, abs=1e-5) and m1[0] == pytest.approx(3.0, abs=1e-5)
    assert i1[0] == t.parse([("tin", B, 0, 1.0)])
    i2, m2 = t.take([r, r], [10.0, 10.0], [0, 1], [kc, kt], [99.0, 99.0])
    assert i2[0] != 0 and i2[1] != 0
    i3, m3 = t.take([r], [10.0], [0, 0], [kc, kt], [99.0, 99.0])
    assert i3[0] == 0 and m3[0] == 0.0
    i4, m4, got = t.take([r], [10.0], [0], [pack_key(S["lead"], B, 0)], [1.0], return_taken=True)
    assert got[0] == 0.0 and i4[0] == r
    sound(t)


def test_i_split_keeps_the_row_in_both_pieces():
    """[I] breaking a lump does not change what it is made of."""
    t = MixTable()
    r = t.parse([("copper", B, 0, 0.88), ("tin", B, 0, 0.12)])
    a, ma, b, mb = t.split(np.array([r, r]), np.array([10.0, 3.0]), np.array([4.0, 1.0]))
    assert (a == r).all() and (b == r).all() and (ma + mb == [10.0, 3.0]).all()


def test_i_blend_many_mixes_a_records_nodes_by_group():
    """[I] 4 lumps into 2 groups: grams per species are the sums, rows by content."""
    t = MixTable()
    ids = np.array([t.parse([("water", D, 0, 1.0)]), t.parse([("sodium_chloride", D, 0, 1.0)]),
                    t.parse([("water", D, 0, 1.0)]), t.parse([("water", D, 0, 1.0)])])
    out, M = t.blend_many(ids, np.array([8.0, 2.0, 5.0, 5.0]), np.array([0, 0, 1, 1]))
    assert out[1] == t.parse([("water", D, 0, 1.0)]) and M[1] == 10.0
    assert out[0] == t.parse([("water", D, 0, 0.8), ("sodium_chloride", D, 0, 0.2)])
    sound(t)


# ---- the float32 lattice (test 14) ---------------------------------------------------------------------
def test_i_float32_lattice_handovers_conserve_species_with_the_f32_and_quant_ledgers():
    """[I] test 14: a float32 mass array and 1e5 store()-based hand-overs (a take and a put
    for each of 5e4 moves: 500 rounds of 100): species totals + quant + f32 constant to 1e-9 of the
    total (it is ~1e-13); the raw drift without the ledgers is reported by the bound
    asserted here (a placeholder, 1e-6 of the total, to be set from this measurement)."""
    rng = np.random.default_rng(9)
    t = MixTable()
    N = 1000
    ids = rand_rows(t, rng, N, K=4, alpha=0.6)
    m32 = rng.uniform(1, 100, N).astype(np.float32)
    T0 = tot_array(t, ids, m32.astype(np.float64), ALL_SP)
    total = T0.sum()
    for rnd in range(500):
        perm = rng.permutation(N)
        a, b = perm[:100], perm[100:200]
        ent = t.entries(ids[a])
        first = np.r_[0, np.cumsum(t.row_len(ids[a]))[:-1]]
        sel = first + rng.integers(0, t.row_len(ids[a]))
        key = ent.key[sel]
        g = ent.frac[sel] * m32[a].astype(np.float64) * rng.uniform(0, 0.5, 100)
        i1, m1, taken = t.take(ids[a], m32[a].astype(np.float64), np.arange(100), key, g,
                               return_taken=True)
        i2, m2 = t.put(ids[b], m32[b].astype(np.float64), np.arange(100), key, taken)
        t.store(m32, a, i1, m1)
        t.store(m32, b, i2, m2)
        ids[a], ids[b] = i1, i2
    T1 = tot_array(t, ids, m32.astype(np.float64), ALL_SP)
    led = np.array([t.ledger("quant").get(s, 0.0) + t.ledger("f32").get(s, 0.0) for s in ALL_SP])
    assert np.abs(T1 + led - T0).max() < 1e-9 * total
    assert np.abs(T1 - T0).max() < 1e-6 * total, np.abs(T1 - T0).max() / total
    assert abs(t.ledger_total("f32")) > 0                    # float32 really did round something
    sound(t)


# ---- reference counts, collection (test 12) -------------------------------------------------------------------
def _holder_world(rng, n_rows=6000, n_cells=20000, live=100):
    t = MixTable()
    ids = rand_rows(t, rng, n_rows, K=3)
    cells = np.zeros(n_cells, np.int32)
    use = rng.choice(np.unique(ids[ids >= 128]), live, replace=False)
    k = min(3000, n_cells // 2)
    sel = rng.choice(n_cells, k, replace=False)
    cells[sel] = rng.choice(use, k)
    return t, cells


def test_i_collect_remaps_holders_keeps_order_and_grams_and_refcounts_match_bincount():
    """[I] after collect every cell has the same content digest; ids keep their order;
    refs equal np.bincount of the holder; only referenced rows (and 0..127) remain."""
    rng = np.random.default_rng(10)
    t, scomp = _holder_world(rng)
    fcomp = np.zeros_like(scomp)
    fcomp[:50] = scomp[100:150]
    t.register_holder("scomp")
    t.register_holder("fcomp")
    before = t.content_digest(scomp).copy(), t.content_digest(fcomp).copy()
    mass = rng.uniform(1, 9, len(scomp))
    g0 = tot_array(t, scomp, mass, ALL_SP)
    refs = t.recount({"scomp": scomp, "fcomp": fcomp})
    assert (refs == np.bincount(np.concatenate([scomp, fcomp]), minlength=t.n_rows)[:t.n_rows]).all()
    alive = np.flatnonzero(refs > 0)
    lut = t.collect({"scomp": scomp, "fcomp": fcomp})
    live_user = [r for r in alive if r >= 128]
    assert t.n_rows == 128 + len(live_user)
    assert (np.diff(lut[live_user]) > 0).all() and (lut[:128] == np.arange(128)).all()
    assert (t.content_digest(scomp) == before[0]).all() and (t.content_digest(fcomp) == before[1]).all()
    assert np.allclose(tot_array(t, scomp, mass, ALL_SP), g0, rtol=0, atol=1e-12 * g0.sum())
    assert (t.refs == np.bincount(np.concatenate([scomp, fcomp]), minlength=t.n_rows)[:t.n_rows]).all()
    sound(t)
    # the table still works after a collection
    r = t.parse([("copper", B, 0, 0.5), ("tin", B, 0, 0.5)])
    assert r >= 128 and t.n_rows >= 128
    sound(t)


def test_i_collect_refuses_while_unsafe():
    """[I] an unregistered holder, matter in flight, or an open sweep each refuse."""
    rng = np.random.default_rng(11)
    t, scomp = _holder_world(rng, n_rows=300, n_cells=1000, live=10)
    t.register_holder("scomp")
    t.register_holder("drops")
    with pytest.raises(MT.CollectRefused, match="drops"):
        t.collect({"scomp": scomp})
    t._holders.discard("drops")
    with pytest.raises(MT.CollectRefused, match="in flight"):
        t.collect({"scomp": scomp}, in_flight=3)
    t.begin_sweep()
    with pytest.raises(MT.CollectRefused, match="sweep"):
        t.collect({"scomp": scomp})
    t.end_sweep()
    n = t.n_rows
    t.collect({"scomp": scomp})
    assert t.n_rows < n
    with pytest.raises(TypeError):
        t.collect([scomp])


def test_i_collect_keeps_ids_a_caller_pins_and_a_stale_id_is_loud():
    """[I] `keep` survives; a holder with an id past the table raises on recount."""
    t = MixTable()
    r = t.parse([("copper", B, 0, 0.3), ("tin", B, 0, 0.7)])
    cells = np.zeros(10, np.int32)
    lut = t.collect({"c": cells}, keep=[r])
    assert t.n_rows == 129 and lut[r] == 128
    with pytest.raises(ValueError, match="refers to row"):
        t.recount({"c": np.array([5000], np.int32)})


def test_i_auto_collect_triggers_above_max_of_twice_live_and_4096():
    """[I] 4100 rows with 10 live collects; 2 * live above the row count does not."""
    rng = np.random.default_rng(12)
    t = MixTable()
    rand_rows(t, rng, 4200, K=3)
    cells = np.zeros(100, np.int32)
    cells[:10] = np.arange(130, 140)
    assert t.n_rows > 4096
    assert t.maybe_collect({"c": cells}) is not None and t.n_rows == 138
    t2 = MixTable()
    rand_rows(t2, rng, 3000, K=3)
    cells2 = np.arange(128, 128 + 100).astype(np.int32)
    assert t2.maybe_collect({"c": cells2}) is None
    n = t2.n_rows
    assert n < 4096 and n > 2 * 100


def test_i_max_rows_raises_loudly():
    """[I] past MAX_ROWS the table refuses instead of growing."""
    rng = np.random.default_rng(13)
    t = MixTable(max_rows=300)
    with pytest.raises(MT.MixTableFull):
        rand_rows(t, rng, 400, K=3)


# ---- the compiled twin (test 13) ---------------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.3, 2.0, 10.0])
def test_i_compiled_blend_equals_the_plain_blend_bit_for_bit(alpha):
    """[I] ids, masses, rows (off, key, q, hash), views and cached columns are identical;
    with a forced tiny reserve (grow by one row) the result is the same as with a big one."""
    rng = np.random.default_rng(14)
    base = MixTable()
    ids = rand_rows(base, rng, 300, K=4, alpha=alpha)
    n = 4000
    ia, ib = rng.choice(ids, n), rng.choice(ids, n)
    ma, mb = rng.uniform(0, 50, n), rng.uniform(0, 50, n)
    ma[::17] = 0.0
    ia[::29] = ib[::29]
    plain, big, tiny = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    r0 = plain.blend(ia, ma, ib, mb)
    r1 = big.blend2_compiled(ia, ma, ib, mb)
    r2 = tiny.blend2_compiled(ia, ma, ib, mb, grow=1)
    for r in (r1, r2):
        assert (r[0] == r0[0]).all() and (r[1] == r0[1]).all()
    for t in (big, tiny):
        t.finish()
        assert t.n_rows == plain.n_rows
        for a in ("off", "key", "q", "hash", "matv", "flv", "dom", "kind"):
            assert np.array_equal(getattr(t, a), getattr(plain, a)), a
        for c in ("rho", "c", "k", "eps"):
            assert np.array_equal(t.prop(c, np.arange(t.n_rows)), plain.prop(c, np.arange(plain.n_rows)),
                                  equal_nan=True), c
        sound(t)
    sound(plain)
    # the grams booked in quant agree to rounding
    qa, qb = plain.ledger("quant"), big.ledger("quant")
    keys = set(qa) | set(qb)
    assert max(abs(qa.get(k, 0.0) - qb.get(k, 0.0)) for k in keys) < 1e-9


def test_i_stop_and_resume_does_not_drop_a_cell():
    """[I] a table at the edge of its room: every pair gets an answer and the rows are real."""
    rng = np.random.default_rng(15)
    t = MixTable(capacity=130)
    ids = rand_rows(t, rng, 60, K=3)
    n = 1500
    ia, ib = rng.choice(ids, n), rng.choice(ids, n)
    ma, mb = rng.uniform(1, 5, n), rng.uniform(1, 5, n)
    t.reserve(0)
    cap0 = len(t._hash)
    out_i, out_m = t.blend2_compiled(ia, ma, ib, mb, grow=3)
    assert len(t._hash) > cap0
    assert (out_m == ma + mb).all() and (out_i >= 0).all() and (out_i < t.n_rows).all()
    sound(t)


def test_i_different_batchings_give_different_row_numbers_but_equal_digests():
    """[I] spec 6.1: ids depend on history; content_digest does not."""
    rng = np.random.default_rng(16)
    base = MixTable()
    ids = rand_rows(base, rng, 200, K=3)
    n = 600
    ia, ib = rng.choice(ids, n), rng.choice(ids, n)
    ma, mb = rng.uniform(1, 50, n), rng.uniform(1, 50, n)
    t1, t2 = copy.deepcopy(base), copy.deepcopy(base)
    i1, _ = t1.blend(ia, ma, ib, mb)
    perm = rng.permutation(n)
    i2p, _ = t2.blend(ia[perm], ma[perm], ib[perm], mb[perm])
    i2 = np.empty(n, np.int32)
    i2[perm] = i2p
    assert (i1 != i2).any()
    assert (t1.content_digest(i1) == t2.content_digest(i2)).all()


def test_i_the_slow_collision_path_gives_the_same_table_as_the_fast_one():
    """[C] _resolve_slow (one lump at a time, the fallback for a 64-bit collision) and
    _resolve (batched) make the same rows in the same order."""
    rng = np.random.default_rng(17)
    t1, t2 = MixTable(), MixTable()
    k = np.stack([rng.choice(POOL, 3, replace=False) for _ in range(300)])
    f = rng.dirichlet(np.ones(3), 300)
    n = 300
    args = (n, np.repeat(np.arange(n), 3), k.ravel(), f.ravel(), np.ones(n))
    cq, kq, qq, _b = MixTable._quantise(*args)
    ids1 = t1._resolve(n, cq, kq, qq)
    cnt = np.bincount(cq, minlength=n)
    ids2 = t2._resolve_slow(n, cq, kq, qq, cnt, np.cumsum(cnt) - cnt, MT.row_hash(cq, kq, qq, n))
    assert (ids1 == ids2).all()
    assert np.array_equal(t1.key, t2.key) and np.array_equal(t1.q, t2.q)
    sound(t1)
    sound(t2)


def test_codeonly_numba_and_numpy_hash_agree():
    """[C] the numba row hash used by the compiled twin equals the numpy one."""
    rng = np.random.default_rng(18)
    for _ in range(50):
        K = int(rng.integers(1, 7))
        key = np.sort(rng.choice(POOL, K, replace=False)).astype(np.uint32)
        q = rng.integers(1, 1 << 20, K).astype(np.int64)
        assert MT._k_hash(key, q, K) == MT.row_hash(np.zeros(K, np.int64), key, q, 1)[0]


def test_i_the_table_survives_deepcopy_and_gives_the_same_digests():
    """[I] test 15: a deep copy is an independent table with the same content."""
    rng = np.random.default_rng(19)
    t = MixTable()
    ids = rand_rows(t, rng, 500)
    cp = copy.deepcopy(t)
    assert (cp.content_digest(ids) == t.content_digest(ids)).all()
    cp.parse([("copper", B, 0, 0.123), ("tin", B, 0, 0.877)])
    assert cp.n_rows == t.n_rows + 1
    sound(t)
    sound(cp)


def test_i_check_table_catches_a_corrupt_row():
    """[C] the checker is not a rubber stamp."""
    t = MixTable()
    r = t.parse([("copper", B, 0, 0.5), ("tin", B, 0, 0.5)])
    t._q[t._off[r]] += 1
    assert t.check_table()
    t._q[t._off[r]] -= 1
    assert t.check_table() == []
    t._hash[r] += np.uint64(1)
    assert t.check_table()
    t._hash[r] -= np.uint64(1)


# ---- the views (test 16) ------------------------------------------------------------------------------------
def test_i_an_ice_block_views_as_ice_and_wet_wood_stays_wood():
    """[I] pure water BOUND has no skeleton entry so the largest BOUND one leads (ICE,
    mat 12); wood with 60 percent bound water views as WOOD because water is not a
    skeleton species."""
    t = MixTable()
    ice = t.parse([("water", B, 0, 1.0)])
    assert t.mat_view([ice])[0] == 12
    wet = t.parse([("wood", B, 0, 0.4), ("water", B, 0, 0.6)])
    assert t.mat_view([wet])[0] == S["wood"] == 1
    sound(t)


def test_i_quartz_grain_and_quartz_rock_are_different_mats():
    """[I] the GRAIN mat 15 (sand) and the BOUND mat 14 (rock) of one species."""
    t = MixTable()
    sand = t.parse([("quartz", G, 64, 1.0)])
    rock = t.parse([("quartz", B, 0, 1.0)])
    assert t.mat_view([sand, rock]).tolist() == [15, 14]
    sound(t)


def test_i_bronze_views_as_copper_and_a_slurry_as_no_solid():
    """[I] the leader decides (copper mat 13); a salt slurry has no mat."""
    t = MixTable()
    bronze = t.parse([("copper", B, 0, 0.88), ("tin", B, 0, 0.12)])
    slurry = t.parse([("water", D, 0, 0.9), ("quartz", G, 48, 0.1)])
    assert t.mat_view([bronze, slurry]).tolist() == [13, 15]
    assert t.fl_view([slurry])[0] == 1
    sound(t)


@pytest.mark.parametrize("comps", [
    [("water", D, 0, 0.4), ("sucrose", D, 0, 0.6)],          # syrup
    [("copper", D, 0, 0.88), ("tin", D, 0, 0.12)],           # molten bronze
    [("water", D, 0, 0.7), ("ethanol", D, 0, 0.3)],          # 30 percent ethanol
    [("water", D, 0, 0.8), ("sodium_chloride", D, 0, 0.2)],  # brine
])
def test_i_syrup_molten_bronze_ethanol_and_brine_have_a_fluid_view(comps):
    """[I] I4 needs fl != 0 for a wet cell: the solvent rule always finds one here."""
    t = MixTable()
    r = t.parse(comps)
    assert t.fl_view([r])[0] != 0
    sound(t)


def test_i_the_solvent_rule_picks_the_largest_dissolved_species_that_has_a_fluid():
    """[I] brine views as WATER; molten bronze as molten copper (fl 10); ethanol-rich as ethanol (fl 9)."""
    t = MixTable()
    assert t.fl_view([t.parse([("water", D, 0, 0.8), ("sodium_chloride", D, 0, 0.2)])])[0] == 1
    assert t.fl_view([t.parse([("copper", D, 0, 0.88), ("tin", D, 0, 0.12)])])[0] == 10
    assert t.fl_view([t.parse([("water", D, 0, 0.3), ("ethanol", D, 0, 0.7)])])[0] == 9


def test_i_hysteresis_keeps_a_50_50_cell_from_flickering():
    """[I] a copper-tin cell at 50/50 with +-1e-3 noise: mat changes at most once in 500 steps."""
    rng = np.random.default_rng(20)
    t = MixTable()
    mat = 13
    changes = 0
    for _ in range(500):
        x = 0.5 + rng.uniform(-1e-3, 1e-3)
        r = t.parse([("copper", B, 0, x), ("tin", B, 0, 1 - x)])
        new = int(t.mat_view([r], prev_mat=mat)[0])
        changes += new != mat
        mat = new
    assert changes <= 1
    # a big shift does flip it, once the old species is more than the margin behind
    r = t.parse([("copper", B, 0, 0.2), ("tin", B, 0, 0.8)])
    assert t.mat_view([r], prev_mat=13)[0] == 9
    r = t.parse([("copper", B, 0, 0.48), ("tin", B, 0, 0.52)])
    assert t.mat_view([r], prev_mat=13)[0] == 13 and t.mat_view([r])[0] == 9


def test_i_set_solid_and_set_fluid_keep_the_view_invariants_and_a_stale_write_is_caught():
    """[I] I1-I4 hold after set_solid/set_fluid; writing mat behind the table's back breaks I2."""
    t = MixTable()
    mat = np.zeros((4, 4), np.uint8)
    scomp = np.zeros((4, 4), np.int32)
    fl = np.zeros((4, 4), np.uint8)
    fcomp = np.zeros((4, 4), np.int32)
    fvol = np.zeros((4, 4), np.float32)
    mat[0, 0] = 2                                   # pure stone: scomp 0, mat is the truth
    bronze = t.parse([("copper", B, 0, 0.88), ("tin", B, 0, 0.12)])
    brine = t.parse([("water", D, 0, 0.8), ("sodium_chloride", D, 0, 0.2)])
    idx = (np.array([1, 2]), np.array([1, 2]))
    t.set_solid(mat, scomp, idx, np.array([bronze, bronze]))
    t.set_fluid(fl, fcomp, (np.array([3]), np.array([3])), np.array([brine]))
    fvol[3, 3] = 5.0
    assert (mat[idx] == 13).all() and fl[3, 3] == 1 and mat[0, 0] == 2
    assert t.check_views(mat, scomp, fl, fcomp, fvol) == []
    mat[1, 1] = 4                                   # a stale mat write
    bad = dict(t.check_views(mat, scomp, fl, fcomp, fvol))
    assert list(bad["I2"]) == [1 * 4 + 1]
    mat[1, 1] = 13
    fl[3, 3] = 0
    bad = dict(t.check_views(mat, scomp, fl, fcomp, fvol))
    assert "I4" in bad and "I3" in bad
    t.set_solid(mat, scomp, (np.array([0]), np.array([0])), np.array([0]))
    assert mat[0, 0] == 2 and scomp[0, 0] == 0


def test_i_set_mat_books_the_old_mix_and_the_new_pure_grams_to_recast():
    """[I] a bronze cell written to WOOD: recast is -0.88 m copper, -0.12 m tin, +m wood; it
    sums to zero; a pure cell books nothing."""
    t = MixTable()
    bronze = t.parse([("copper", B, 0, 0.88), ("tin", B, 0, 0.12)])
    mat = np.array([13, 2], np.uint8)
    scomp = np.array([bronze, 0], np.int32)
    smass = np.array([50.0, 9.0], np.float32)
    t.set_mat(mat, scomp, smass, np.array([0, 1]), 1)
    rc = t.ledger("recast")
    assert rc[CU] == pytest.approx(-0.88 * 50.0, abs=1e-5) and rc[SN] == pytest.approx(-0.12 * 50.0, abs=1e-5)
    assert rc[1] == pytest.approx(50.0, abs=1e-9)
    assert sum(rc.values()) == pytest.approx(0.0, abs=1e-5)
    assert (mat == 1).all() and (scomp == 0).all()


# ---- adapters (test 17) ---------------------------------------------------------------------------------------------
def test_i_dense_adapter_round_trips_to_the_quantum():
    """[I] to_dense(from_dense(g)) equals g to M 2^-23 per entry; the mass is the sum."""
    rng = np.random.default_rng(21)
    t = MixTable()
    order = ["water", "sodium_chloride", "ethanol", "copper", "tin"]
    g = rng.dirichlet(np.ones(5) * 0.5, 500) * rng.uniform(1, 80, (500, 1))
    ids, M = t.from_dense(g, order, form=[D, D, D, D, D])
    back = t.to_dense(ids, order, mass=M)
    assert np.abs(back - g).max() <= 2 * Q * M.max()
    assert np.allclose(M, g.sum(axis=1), rtol=0, atol=1e-12)
    sound(t)


def test_i_dense_adapter_keeps_forms_apart_and_refuses_a_species_not_in_the_order():
    """[I] per-column forms; to_dense(form=) selects; a missing species raises KeyError."""
    t = MixTable()
    ids, M = t.from_dense(np.array([[6.0, 3.0, 1.0]]), ["water", "tannin", "tin"], form=[D, D, B])
    r = ids[0]
    assert t.to_dense([r], ["water", "tannin", "tin"], mass=M, form=B)[0].tolist() == pytest.approx([0, 0, 1.0], abs=1e-6)
    with pytest.raises(KeyError):
        t.to_dense([r], ["water", "tannin"], mass=M)


def test_i_parse_takes_names_and_ints_and_the_derived_column_registry_reaches_the_table():
    """[I] parse by name or id/int form; register_derived columns are computed lazily per row."""
    t = MixTable()
    a = t.parse([("copper", "bound", 0, 0.5), ("tin", "bound", 0, 0.5)])
    b = t.parse([(CU, 3, 0, 0.5), (SN, 3, 0, 0.5)])
    assert a == b
    MR.register_derived("test_n_components", lambda ent: np.bincount(ent.cell, minlength=ent.n).astype(float))
    try:
        assert t.prop("test_n_components", [a, 3, a]).tolist() == [2.0, 1.0, 2.0]
        assert "test_n_components" in t._derived
        with pytest.raises(KeyError):
            t.prop("no_such_column", [a])
    finally:
        MR.DERIVED.pop("test_n_components", None)
        MR._REV[0] += 1
    assert t.prop("rho", [a])[0] > 0                      # a revision bump recomputes, not breaks


def test_i_cached_columns_are_float32_and_equal_the_rule_outputs():
    """[I] the table's rho, c, k, eps are the mixrules numbers (engine slot) rounded to float32."""
    t = MixTable()
    r = t.parse([("water", D, 0, 0.8), ("sodium_chloride", D, 0, 0.2)])
    ent = t.entries([r])
    for name, fn in (("rho", MR.rho_mix), ("c", MR.c_mix), ("k", MR.k_mix), ("eps", MR.eps_mix)):
        got = t.prop(name, [r])
        assert got.dtype == np.float32 and got[0] == np.float32(fn(ent, "engine")[0])


# ---- energy (tests 18, 19) -------------------------------------------------------------------------------------------------
def _lump(t, comps, mass, T):
    r = t.parse(comps)
    C_ = t.cap64([r], [mass])[0]
    return r, mass, C_, C_ * (T - 20.0)               # row, grams, J/K, E above 20 C


@pytest.mark.parametrize("TA,TB", [(80.0, 20.0), (-5.0, 60.0), (300.0, 300.0)])
def test_i_take_and_put_move_heat_with_the_grams_and_conserve_total_E(TA, TB):
    """[I] test 18: matter leaves at the cell temperature, so the donor keeps its T (to the
    quantum) and the E that left arrives in the receiver; sum E is exact."""
    t = MixTable()
    ra, ma, Ca, Ea = _lump(t, [("water", D, 0, 0.8), ("sodium_chloride", D, 0, 0.2)], 100.0, TA)
    rb, mb, Cb, Eb = _lump(t, [("water", D, 0, 1.0)], 40.0, TB)
    kn = pack_key(NACL, D, 0)
    (ia, m_a, taken) = (lambda r: (r[0], r[1], r[2]))(t.take([ra], [ma], [0], [kn], [6.0], return_taken=True))
    c_k = t.key_c(np.array([kn], np.uint32))[0]
    dE = MT.heat_of(Ea, Ca, taken[0] * c_k)
    ib, m_b = t.put([rb], [mb], [0], [kn], taken)
    Ea2, Eb2 = Ea - dE, Eb + dE
    assert Ea2 + Eb2 == pytest.approx(Ea + Eb, rel=0, abs=1e-9)
    Ta2 = 20.0 + Ea2 / t.cap64(ia, m_a)[0]
    assert Ta2 == pytest.approx(TA, abs=1e-5 * max(1.0, abs(TA - 20.0)))
    # the receiver lands at the c-weighted temperature of what it had and what arrived
    Cb2 = t.cap64(ib, m_b)[0]
    want = 20.0 + (Eb2) / (Cb + taken[0] * c_k)
    assert 20.0 + Eb2 / Cb2 == pytest.approx(want, abs=1e-5 * max(1.0, abs(want - 20.0)))
    sound(t)


@pytest.mark.parametrize("Ta,Tb", [(10.0, 90.0), (250.0, 20.0), (-30.0, 5.0)])
def test_i_a_blend_lands_at_the_capacity_weighted_temperature(Ta, Tb):
    """[I] test 18: T = (Ca Ta + Cb Tb) / (Ca + Cb), with C summed per component in float64
    (the new row's own C matches to the quantum)."""
    t = MixTable()
    ra, ma, Ca, Ea = _lump(t, [("copper", B, 0, 0.88), ("tin", B, 0, 0.12)], 30.0, Ta)
    rb, mb, Cb, Eb = _lump(t, [("water", D, 0, 0.8), ("ethanol", D, 0, 0.2)], 12.0, Tb)
    ri, mi = t.blend([ra], [ma], [rb], [mb])
    Cn = t.cap64(ri, mi)[0]
    assert Cn == pytest.approx(Ca + Cb, rel=2e-7)
    Tn = 20.0 + (Ea + Eb) / Cn
    assert Tn == pytest.approx((Ca * Ta + Cb * Tb) / (Ca + Cb), abs=1e-5 * max(1.0, abs(Ta - Tb)))


@pytest.mark.parametrize("T", [20.0, 80.0])
def test_u_dissolving_nacl_gives_the_predicted_temperature_change(T):
    """[U] test 19: C' = C + m (c_B - c_A); C'(T' - 20) = C (T - 20) - m dH(20), so
    dT = -m dH(T) / C' with dH(T) = dH(20) + dc (T - 20) (NaCl: +66 J/g at 20 C; at 80 C
    66 + (0.28 - 0.864) 60 = 31 J/g). The numbers (c_app 0.28) are (u); the arithmetic and
    the agreement of the energy convention with the table's own capacity are [I]."""
    t = MixTable()
    sz = MT.SZ_NUCLEI
    r, m, C0, E0 = _lump(t, [("water", D, 0, 0.88), ("sodium_chloride", C, sz, 0.12)], 100.0, T)
    i1, m1, got = t.convert([r], [m], [0], [NACL], [C], [D], [5.0], from_size=sz, return_taken=True)
    g = got[0]
    kc, kd = np.array([pack_key(NACL, C, sz), pack_key(NACL, D, 0)], np.uint32)
    c_from, c_to = t.key_c(np.array([kc, kd], np.uint32))
    dH20 = MR.form_energy(NACL, C, D)
    assert dH20 == 66.0
    C1, E1 = MT.convert_energy(C0, E0, g, c_from, c_to, dH20)
    assert C1 == pytest.approx(t.cap64(i1, m1)[0], rel=2e-7)             # the table agrees on C'
    T1 = 20.0 + E1 / C1
    dHT = dH20 + (c_to - c_from) * (T - 20.0)
    assert T1 - T == pytest.approx(-g * dHT / C1, abs=1e-9 * max(1.0, abs(T)))
    if T == 80.0:
        assert dHT == pytest.approx(31.0, abs=0.1)
    assert T1 < T                                                      # dissolving salt cools
    # a round trip returns E exactly
    C2, E2 = MT.convert_energy(C1, E1, g, c_to, c_from, -dH20)
    assert E2 == pytest.approx(E0, rel=0, abs=1e-9) and C2 == pytest.approx(C0, rel=1e-12)
    sound(t)
