"""T3 layers: contact transfer, wipe, wash, wear, the Holmes boot walk (spec 7:
T19-T26, T18 re-wetted lift, T41 energy over random moves/freezes/dissolves).

Tags: test_codeonly_* tests the code against its own table; test_u_* independent but
unverified; the rest are conservation and arithmetic identities.
"""
import math
import random

import numpy as np
import pytest

from src.voxel import layer_data as D
from src.voxel import layer_laws as LL
from src.voxel.layers import Layers, make_key
from tests.layers_fake import FakeHost

A5 = 0.05 * 0.05


def _inv(host, L):
    return host.total_E() + L.stored_J() - L.energy_in_total()


def _mud(g=1.0, water=0.6):
    """A boot's mud: `water` share liquid so the film splits by adhesion."""
    return {(D.CLAY, D.GRAIN, 8): g * (1 - water), (D.WATER, D.DISSOLVED, 0): g * water}


# ================================================================ T19 liquid split
def test_codeonly_liquid_split_is_half_for_equal_angles_and_follows_one_plus_cos():
    assert LL.liquid_split(60.0, 60.0) == pytest.approx(0.5)
    for ts, td in ((60.0, 120.0), (30.0, 90.0), (110.0, 20.0)):
        a, b = 1 + math.cos(math.radians(ts)), 1 + math.cos(math.radians(td))
        assert LL.liquid_split(ts, td) == pytest.approx(b / (a + b))
        assert LL.liquid_split(ts, td) + LL.liquid_split(td, ts) == pytest.approx(1.0)


def test_a_liquid_contact_splits_both_sides_summing_to_the_input():
    L = Layers()
    S, D1 = make_key(0, 1, 4), make_key(0, 2, 4)
    L.add(S, {(D.WATER, D.DISSOLVED, 0): 1.0, (D.INK_GALL, D.DISSOLVED, 0): 0.01},
          cover=1.0, Ts=20.0)
    total = L.mass()
    moved, _adj = L.contact(S, [D1], c_c=1.0, p=0.0, s_slide=0.0, L_p=0.1)
    assert moved == pytest.approx(0.5 * total, rel=1e-9)          # equal angles: half
    assert L.mass(S) + L.mass(D1) == pytest.approx(total, rel=1e-12)
    assert L.mass(D1) == pytest.approx(moved, rel=1e-12)
    # a hydrophobic landing surface takes less
    sub_hi = dict(D.SUBSTRATE[2])
    sub_hi["theta"] = dict(water=120.0, oil=60.0, wax=60.0)
    M = Layers()
    M.add(S, {(D.WATER, D.DISSOLVED, 0): 1.0}, cover=1.0, Ts=20.0)
    m2, _ = M.contact(S, [D1], 1.0, 0.0, 0.0, 0.1, sub_D=sub_hi)
    assert m2 == pytest.approx(LL.liquid_split(60.0, 120.0), rel=1e-9)
    assert m2 < 0.5


# ================================================================ T20 dry transfer
def test_dry_transfer_rises_with_pressure_and_with_the_adhesion_ratio():
    a = [LL.dry_transfer(1.0, 3000.0, 3000.0, p) for p in (100.0, 1e3, 5e3, 3e4, 3e5)]
    assert all(y > x for x, y in zip(a, a[1:]))
    assert a[-1] < LL.TE_MAX + 1e-12
    b = [LL.dry_transfer(1.0, aD, 1000.0, 3e4) for aD in (100.0, 1000.0, 1e4, 1e6)]
    assert all(y > x for x, y in zip(b, b[1:]))
    assert LL.dry_transfer(0.5, 1000.0, 1000.0, 5e3) == pytest.approx(
        0.5 * 0.6 * 0.5 * (1 - math.exp(-1)))


def test_dry_contact_moves_exactly_what_leaves_the_source():
    L = Layers()
    S, D1 = make_key(0, 1, 4), make_key(0, 2, 4)
    L.add(S, {(D.CLAY, D.GRAIN, 8): 1.0}, cover=1.0, Ts=20.0)
    m0 = L.mass()
    moved, _adj = L.contact(S, [D1], c_c=1.0, p=3e4, s_slide=0.0, L_p=0.1)
    assert 0.0 < moved < m0
    assert L.mass(S) == pytest.approx(m0 - moved, rel=1e-12)
    assert L.mass(D1) == pytest.approx(moved, rel=1e-12)
    assert L.ledger_closes()


# ================================================================ T21 slide
def test_codeonly_slide_fraction_is_static_at_zero_and_tends_to_the_contacted_share():
    f_st, c_c = 0.2, 0.7
    assert LL.slide_fraction(f_st, c_c, 0.0, 0.1) == pytest.approx(f_st)
    vals = [LL.slide_fraction(f_st, c_c, s, 0.1) for s in (0.01, 0.1, 0.5, 2.0, 20.0)]
    assert all(y > x for x, y in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(c_c, rel=1e-6)
    assert LL.slide_fraction(f_st, c_c, 0.05, 0.001) == LL.slide_fraction(f_st, c_c, 0.05, 0.02)


def test_the_deposit_along_a_slide_sums_to_the_moved_mass():
    L = Layers()
    S = make_key(0, 1, 4)
    Ds = [make_key(0, 10 + i, 4) for i in range(3)]
    L.add(S, _mud(1.0, 0.6), cover=1.0, Ts=20.0)
    m0 = L.mass()
    moved, _adj = L.contact(S, Ds, c_c=0.8, p=1e4, s_slide=0.3, L_p=0.1, weights=[1, 2, 3])
    got = [L.mass(d) for d in Ds]
    assert sum(got) == pytest.approx(moved, rel=1e-12)
    assert got[0] < got[1] < got[2]
    assert got[2] / got[0] == pytest.approx(3.0, rel=1e-9)         # in proportion to path length
    assert L.mass(S) + sum(got) == pytest.approx(m0, rel=1e-12)
    assert moved / m0 == pytest.approx(LL.slide_fraction(0.5 * 0.8, 0.8, 0.3, 0.1), rel=1e-9)


# ================================================================ T22 the boot walk (Holmes)
def test_a_boot_walk_leaves_f_times_one_minus_f_to_the_n_minus_one_and_keeps_stack_order():
    L = Layers()
    boot = make_key(2, 5 * 64 + 3, 0)                                # a body region
    L.add(boot, {(D.CHALK, D.GRAIN, 3): 0.5}, cover=1.0, Ts=20.0)    # chalk first, under
    L.add(boot, _mud(1.0, 0.6), cover=1.0, Ts=20.0, pattern="sole") if False else \
        L.add(boot, _mud(1.0, 0.6), cover=1.0, Ts=20.0)
    assert len(L.stack(boot)) == 2                                   # chalk under mud
    f = 0.3                                                          # 0.5 split * c_c 0.6
    tracks = []
    for n in range(1, 14):
        floor = make_key(0, 1000 + n, 4)
        moved, _adj = L.contact(boot, [floor], c_c=0.6, p=3e4, s_slide=0.0, L_p=0.1)
        tracks.append((floor, moved))
    mud_total = 1.0
    for n, (floor, moved) in enumerate(tracks, start=1):
        assert moved == pytest.approx(mud_total * f * (1 - f) ** (n - 1), rel=1e-9), n
        assert L.mass(floor, D.CHALK) == 0.0                         # only mud while mud lasts
    left = L.mass(boot, D.CLAY) + L.mass(boot, D.WATER)
    assert left / mud_total < 0.01                                   # under 1 percent after 13
    assert left / mud_total == pytest.approx((1 - f) ** 13, rel=1e-9)
    assert L.mass(boot, D.CHALK) == pytest.approx(0.5)               # the chalk was never touched
    assert L.ledger_closes()


def test_chalk_under_the_mud_shows_only_after_the_mud_is_gone():
    L = Layers()
    boot = make_key(2, 7 * 64 + 1, 0)
    L.add(boot, {(D.CHALK, D.GRAIN, 3): 0.5}, cover=1.0, Ts=20.0)
    L.add(boot, _mud(1.0, 0.6), cover=1.0, Ts=20.0)
    first_chalk = None
    for n in range(1, 120):
        floor = make_key(0, 5000 + n, 4)
        L.contact(boot, [floor], c_c=0.6, p=3e4, s_slide=0.0, L_p=0.1)
        if L.mass(floor, D.CHALK) > 0.0 and first_chalk is None:
            first_chalk = n
    assert first_chalk is not None and first_chalk > 13
    mud_left_then = 1.0 * 0.7 ** first_chalk
    assert mud_left_then < 1e-5                                      # the mud was gone by then
    assert L.ledger_closes()


def test_a_clean_booted_man_leaves_no_tracks():
    L = Layers()
    boot = make_key(2, 9 * 64 + 1, 0)
    moved, adj = L.contact(boot, [make_key(0, 3, 4)], 0.6, 3e4, 0.0, 0.1)
    assert moved == 0.0 and adj == [] and L.n == 0


# ================================================================ T23 wipe
def test_wipe_below_cohesion_nothing_between_partial_above_adhesion_all():
    L = Layers()
    k, cloth = make_key(0, 1, 4), make_key(1, 99, 0)
    L.add(k, {(D.SOOT, D.GRAIN, 1): 0.02}, cover=1.0, Ts=20.0)
    s = L.stack(k)[0]
    adh, coh = float(L.adh[s]), float(L.coh[s])
    adh_s = LL.SHEAR_RATIO * adh
    assert coh < adh_s                                               # a loose deposit
    m0 = L.mass()
    got, _ = L.wipe(k, 0.5 * coh, cloth)
    assert got == 0.0 and L.mass(cloth) == 0.0
    tau = 0.5 * (coh + adh_s)
    got, _ = L.wipe(k, tau, cloth)
    assert got == pytest.approx(0.5 * m0, rel=1e-4)                  # (tau - coh)/(adh_s - coh)
    assert L.mass(cloth) == pytest.approx(got, rel=1e-12)            # the wiper gets what lifted
    got2, _ = L.wipe(k, 2 * adh_s, cloth)
    assert got2 == pytest.approx(0.5 * m0, rel=1e-3)
    assert L.mass(k) == pytest.approx(0.0, abs=1e-12) and L.mass(cloth) == pytest.approx(m0)
    assert L.ledger_closes()


def test_a_brittle_film_peels_whole_at_its_shear_adhesion_and_not_before():
    """adh_s <= coh (ice on stone): nothing under adh_s, the WHOLE layer at or over it."""
    L = Layers()
    k, cloth = make_key(0, 1, 4), make_key(1, 98, 0)
    L.add(k, {(D.WATER, D.CRYSTAL, 40): 0.4}, cover=1.0, Ts=-5.0)
    s = L.stack(k)[0]
    adh_s = LL.SHEAR_RATIO * float(L.adh[s])
    assert adh_s <= float(L.coh[s])
    assert LL.wipe_fraction(0.99 * adh_s, float(L.adh[s]), float(L.coh[s])) == 0.0
    got, _ = L.wipe(k, 0.99 * adh_s, cloth)
    assert got == 0.0
    got, _ = L.wipe(k, 1.01 * adh_s, cloth)
    assert got == pytest.approx(0.4, rel=1e-12) and L.mass(k) == 0.0


def test_u_the_wipe_cleans_a_sooty_patch_and_the_cloth_is_black():
    L = Layers()
    k, cloth = make_key(0, 5, 5), make_key(1, 77, 0)
    L.add(k, {(D.SOOT, D.GRAIN, 1): 0.05}, cover=1.0, Ts=20.0)
    assert L.look(k)["colour"][0] < 0.3                              # black wall
    L.wipe(k, 5000.0, cloth)                                         # cloth at 5 kPa
    assert L.mass(k) == 0.0
    assert L.look(cloth)["colour"][0] < 0.3                          # black cloth
    assert L.describe(cloth)[0]["label"] == "soot"


def test_a_set_stain_does_not_lift_and_a_rewetted_unset_part_lifts_by_the_formula():
    """T18: BOUND mass keeps its bound adhesion; the unset part lifts by
    (tau - coh)/(adh_s - coh) once re-wetted."""
    L = Layers()
    k, cloth = make_key(0, 1, 4), make_key(1, 55, 0)
    L.add(k, {(D.PLASMA, D.BOUND, 0): 0.02}, cover=1.0, Ts=20.0)
    got, _ = L.wipe(k, 1e4, cloth)                                   # 10 kPa: far under 0.7 * 500 kPa
    assert got == 0.0 and L.mass(k, D.PLASMA, D.BOUND) == pytest.approx(0.02)
    got, _ = L.wipe(k, 0.7 * D.ROWS[D.PLASMA]["a_bnd"] * 1.01, cloth)
    assert got == pytest.approx(0.02, rel=1e-9)                      # at its own adhesion it goes
    # half set, half not, re-wetted: only the unset part lifts at a middling shear
    M = Layers()
    M.add(k, {(D.PLASMA, D.BOUND, 0): 0.02, (D.PLASMA, D.DISSOLVED, 0): 0.02,
              (D.WATER, D.DISSOLVED, 0): 0.2}, cover=1.0, Ts=20.0)
    mob = [(g, D.ROWS[c >> 10], (c >> 8) & 3) for c, g in M._comp(M.stack(k)[0])
           if (c >> 8) & 3 != D.BOUND]
    tot = sum(g for _c, g in M._comp(M.stack(k)[0]))
    phi = M._liquid_mass(M._comp(M.stack(k)[0])) / tot
    adh = LL.mix_strength(mob, "a", phi)
    coh = LL.mix_strength(mob, "c", phi)
    tau = 0.5 * (coh + LL.SHEAR_RATIO * adh)
    want = LL.wipe_fraction(tau, adh, coh)
    assert 0.0 < want < 1.0
    got, _ = M.wipe(k, tau, cloth)
    mob_mass = 0.02 + 0.2
    assert got == pytest.approx(want * mob_mass, rel=1e-9)
    assert M.mass(k, D.PLASMA, D.BOUND) == pytest.approx(0.02)       # the set part stayed


# ================================================================ T24 wash
def test_wash_dissolves_the_dissolved_first_lifts_grain_by_shear_and_leaves_bound():
    L = Layers()
    k, down = make_key(0, 1, 0), make_key(0, 2, 0)
    L.add(k, {(D.PLASMA, D.BOUND, 0): 0.02}, cover=1.0, Ts=20.0)         # a set stain below
    L.add(k, {(D.INK_GALL, D.DISSOLVED, 0): 0.01, (D.SOOT, D.GRAIN, 1): 0.01},
          cover=1.0, Ts=20.0)
    assert len(L.stack(k)) == 2
    # a trickle (1 Pa) takes the dissolved dye but not the soot (coh 50 Pa)
    got, _ = L.wash(k, 1.0, down)
    assert got == pytest.approx(0.01, rel=1e-12)
    assert L.mass(down, D.INK_GALL) == pytest.approx(0.01)
    assert L.mass(k, D.SOOT) == pytest.approx(0.01)
    # a hose (100 Pa) takes the soot; the set protein does nothing
    got, _ = L.wash(k, 100.0, down)
    assert L.mass(down, D.SOOT) == pytest.approx(0.01)
    assert L.mass(k, D.PLASMA, D.BOUND) == pytest.approx(0.02)       # unchanged
    assert L.ledger_closes()


def test_wall_shear_of_a_falling_film_is_one_pascal_at_a_tenth_of_a_millimetre():
    assert LL.wall_shear_film(1e-4) == pytest.approx(0.98, rel=0.01)
    assert LL.wall_shear_nusselt(1e-3, LL.film_flow(1e-4, 1e-3, 1000.0), 1e-4) == pytest.approx(
        1000.0 * LL.G * 1e-4, rel=1e-9)                              # 3 eta Q/h^2 = rho g h


# ================================================================ T25 Archard
def test_archard_removed_thickness_is_linear_in_pressure_and_distance():
    base = LL.archard_dh(1e-3, 1e4, 1.0, 1e6)
    assert LL.archard_dh(1e-3, 2e4, 1.0, 1e6) == pytest.approx(2 * base)
    assert LL.archard_dh(1e-3, 1e4, 3.0, 1e6) == pytest.approx(3 * base)
    L = Layers()
    k, partner = make_key(0, 1, 4), make_key(0, 2, 4)
    L.add(k, {(D.CHALK, D.GRAIN, 3): 0.2}, cover=1.0, Ts=20.0)
    h0 = L.thickness(L.stack(k)[0])
    dh = LL.archard_dh(D.ROWS[D.CHALK]["K_wear"], 2e4, 1.0, D.ROWS[D.CHALK]["H"])
    g, _ = L.wear(k, D.ROWS[D.CHALK]["K_wear"], 2e4, 1.0, D.ROWS[D.CHALK]["H"], partner)
    assert g == pytest.approx(0.2 * dh / h0, rel=1e-9)
    assert L.mass(partner) == pytest.approx(g)                       # the mass went to the partner
    g2, _ = L.wear(k, 1e-3, 1e9, 1.0, 1e6)                           # wear through: all gone
    assert L.mass(k) == pytest.approx(0.0, abs=1e-12)
    assert L.ledger_closes()


# ================================================================ T41 energy over random sequences
def test_energy_is_conserved_over_10000_random_moves_adds_takes_freezes_and_dissolves():
    rng = random.Random(41)
    n = 6
    host = FakeHost(n=n, C=60.0, Cs=15.0, dtype=np.float64)
    for i in range(n):
        host.set_Ts(i, rng.uniform(-12, 40))
    L = Layers()
    sp_pool = [(D.WATER, D.DISSOLVED, 0), (D.WATER, D.CRYSTAL, 40), (D.SALT, D.CRYSTAL, 40),
               (D.SALT, D.DISSOLVED, 0), (D.CLAY, D.GRAIN, 8), (D.BEESWAX, D.CRYSTAL, 40),
               (D.PLASMA, D.DISSOLVED, 0)]
    base = _inv(host, L)
    for it in range(10000):
        i = rng.randrange(n)
        k = make_key(0, i, rng.randrange(6))
        op = rng.random()
        if op < 0.3:
            comp = {rng.choice(sp_pool): rng.random() * 0.3 for _ in range(rng.randint(1, 2))}
            _s, adj = L.add(k, comp, Ts=host.Ts(i), T_in=rng.uniform(-15, 50), source="x")
        elif op < 0.45:
            sp, f, _z = rng.choice(sp_pool)
            _g, adj = L.take(k, sp, f, rng.random() * 0.2, host.Ts(i))
        elif op < 0.8:
            j = rng.randrange(n)
            adj = L.move(k, make_key(0, j, rng.randrange(6)), rng.random(), host.Ts(i), host.Ts(j))
        elif op < 0.85:
            _t, adj, _v = L.latent(host.keys_of(i), host.Csur(i), host.Ts(i), 0.5)
        else:
            _m, adj = L.contact(k, [make_key(0, rng.randrange(n), rng.randrange(6))],
                                0.7, 2e4, rng.random() * 0.2, 0.05, Ts_S=host.Ts(i),
                                Ts_D=host.Ts(i))
        host.apply(adj)
        if it % 500 == 0:
            assert _inv(host, L) == pytest.approx(base, abs=1e-8 * max(1.0, abs(base) + 100.0)), it
    assert _inv(host, L) == pytest.approx(base, abs=1e-8 * max(1.0, abs(base) + 100.0))
    assert L.ledger_closes()
