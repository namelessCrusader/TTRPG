"""GAS MOMENTUM AND THE ONE PUSH LAW (wave 14a, lane D; src/voxel/gasmom.py, push.py).

The gas flow now has two more pieces, both behind a gate (`gas_momentum`; surfaces and wind
exist only where a scene puts them), and a world that does not use them is unchanged:

  * MOMENTUM: the face speeds are carried by the flow itself, so pushed air keeps going.
    The proof is the ceiling jet. Alpert's jet under a ceiling is what a plume that turns
    the corner at speed makes. Before: 2.3 times too hot, 3 times too slow.
  * ONE PUSH LAW: a moving surface pushes the gas it sweeps (displacement and drag, from
    four numbers a patch: place, area, normal, velocity). A bellows, a fan and the outdoor
    wind are scenes of it (src/voxel/scenes_push.py). There is no bellows or fan law.

Real numbers: Alpert (1972) ceiling jet; Heskestad's plume; a piston in a tube moves
exactly its swept volume out of the nozzle (continuity).
"""
import numpy as np
import pytest

from src.voxel import gasflow, gasmom, heat
from src.voxel import push as pushmod
from src.voxel.materials import AIR_DENS, AMBIENT, O2_PER_L, STONE, TICK_S
from src.voxel.scenes_push import DX, bellows, fan, wind_box
from src.voxel.sim import World

H = 4 * DX                                           # a cell


@pytest.fixture(autouse=True)
def _no_eddy(monkeypatch):
    """With the flow on, the gas-to-gas eddy conduction of heat.py is the very convection the
    flow carries (the lane report asks heat.py to drop it where gas_momentum is on)."""
    monkeypatch.setattr(heat, "EDDY", 1e12)


# ── the switch: off, nothing changes ─────────────────────────────────────────
def test_the_SWITCHES_are_OFF_by_default_and_a_world_that_does_not_use_them_is_untouched():
    w = World(12, 12, 12, voxel_cm=5)
    assert gasflow.GAS_MOMENTUM is False and w.gas_momentum is False
    assert w.wind is None and len(w.gas_surfaces) == 0
    assert "gas_surfaces" not in w.__dict__ and "wind" not in w.__dict__   # a snapshot is unchanged
    assert w._push_live() is False


def _plume_world(momentum=None, surfaces=False):
    w = World(20, 20, 30, voxel_cm=5)
    w.fill(0, 20, 0, 20, 0, 1, STONE)
    w.gas_flow = True
    if momentum is not None:
        w.gas_momentum = momentum
    if surfaces:
        w.add_surface(pushmod.Surface([[0.5, 0.5, 0.5]], 0.0025, (1, 0, 0), (0, 0, 0)))   # at rest
    return w


def test_with_momentum_OFF_the_flow_is_bit_for_bit_the_old_one_even_with_a_surface_at_rest():
    a, b, c = _plume_world(), _plume_world(False), _plume_world(False, surfaces=True)
    for t in range(60):
        for w in (a, b, c):
            w.E[8:12, 8:12, 1:5] += 2000.0 * TICK_S / 64
            w.step()
    assert np.array_equal(a.E, b.E) and np.array_equal(a.E, c.E)
    assert float(np.abs(a._gf["u"][2]).max()) > 0.1


# ── the momentum step on its own ─────────────────────────────────────────────
def _faces(n=(10, 8, 8)):
    return [np.zeros(tuple(m + (1 if a == c else 0) for a, m in enumerate(n)), np.float32)
            for c in range(3)]


def test_a_UNIFORM_flow_is_carried_unchanged_and_a_bump_travels_downstream_with_no_new_extreme():
    u = _faces()
    u[0][...] = 2.0
    gasmom.advect_faces(u, TICK_S, H)
    assert np.allclose(u[0], 2.0, atol=1e-5) and not u[1].any() and not u[2].any()
    # a bump in a uniform stream moves with it by u dt per tick, and stays within its own range
    u = _faces()
    u[0][...] = 4.0
    u[0][2:5, 3:6, 3:6] = 4.4
    lo, hi = float(u[0].min()), float(u[0].max())
    x0 = ((u[0][:, 4, 4] - 4.0) * np.arange(11)).sum() / (u[0][:, 4, 4] - 4.0).sum()
    for _ in range(8):
        gasmom.advect_faces(u, TICK_S, H)
    x1 = ((u[0][:, 4, 4] - 4.0) * np.arange(11)).sum() / (u[0][:, 4, 4] - 4.0).sum()
    assert float(u[0].min()) >= lo - 1e-5 and float(u[0].max()) <= hi + 1e-5
    moved = (x1 - x0) * H
    assert 0.8 * 4.2 * 8 * TICK_S < moved < 1.2 * 4.2 * 8 * TICK_S, moved


def test_the_MACCORMACK_correction_keeps_a_jet_sharper_than_plain_back_tracing():
    def run(correct):
        u = _faces((16, 8, 8))
        u[0][3:6, 3:5, 3:5] = 3.0
        for _ in range(30):
            gasmom.advect_faces(u, TICK_S, H, correct=correct)
        return float(u[0].max())
    assert run(True) > run(False) + 0.1


def test_the_MIXING_of_momentum_keeps_the_momentum_and_spreads_a_jet():
    u = _faces((10, 10, 10))
    u[0][4:7, 4:6, 4:6] = 3.0
    p0 = float(u[0].sum(dtype=np.float64))
    for _ in range(40):
        gasmom.mix_faces(u, TICK_S, H, 0.3)
    assert abs(float(u[0].sum(dtype=np.float64)) - p0) < 1e-4 * p0
    assert float(u[0].max()) < 3.0 and float((u[0] > 0.05).sum()) > 12


# ── Alpert: the check wave 12 asked for ──────────────────────────────────────
def _alpert(watts, r, height):
    kw = watts / 1000.0
    return 5.38 * (kw / r) ** (2 / 3) / height, 0.197 * kw ** (1 / 3) * height ** 0.5 / r ** (5 / 6)


def _ceiling_jet(momentum):
    nx, nz, q = 72, 56, 10000.0
    w = World(nx, nx, nz, voxel_cm=5)
    w.fill(0, nx, 0, nx, 0, 1, STONE)
    w.fill(4, nx - 4, 4, nx - 4, 44, 48, STONE)
    for x in (8, 24, 48, 62):
        for y in (8, 24, 48, 62):
            w.fill(x, x + 2, y, y + 2, 1, 44, STONE)
    w.gas_flow = True
    w.gas_momentum = momentum
    c = nx // 2
    for t in range(int(10 / TICK_S)):
        w.E[c - 2:c + 2, c - 2:c + 2, 1:5] += q * TICK_S / 64
        w.step()
    T = w.T()

    def ring(r):
        k = int(round(r / DX - 0.5))
        return float(np.mean([T[c + k, c - 1, 43], T[c - 1 - k, c - 1, 43],
                              T[c - 1, c + k, 43], T[c - 1, c - 1 - k, 43]])) - AMBIENT
    ux = w.vcell[0][:, c // 4 - 1:c // 4 + 1, 10].mean(axis=1)
    return ring, ux[c // 4 + 1:], 43 * DX, q


def test_the_CEILING_JET_is_Alpert_s_with_momentum_and_was_not_without():
    """A 10 kW fire 2.15 m under a stone plate, the open sky all round. With momentum the gas
    under the plate is within 30 percent of Alpert's temperature at 1 and 1.4 m from the axis
    (it was 2.3 times too hot) and runs at his speed to a factor of 1.6 (it was a third of it)."""
    ring_o, ux_o, H0, q = _ceiling_jet(False)
    ring, ux, H0, q = _ceiling_jet(True)
    for r in (1.0, 1.4):
        want, _ = _alpert(q, r, H0)
        assert 0.8 * want < ring(r) < 1.3 * want, (r, ring(r), want)
        assert ring_o(r) > 1.8 * want                      # the old flow: far too hot
    assert ring(0.6) > ring(1.0) > ring(1.4) > 0.0
    _, u1 = _alpert(q, 1.0, H0)
    assert u1 / 1.6 < ux[4] < u1 * 1.6, (ux[4], u1)        # the cell at 1.0 m from the axis
    assert float(ux[:6].mean()) > 2.5 * float(ux_o[:6].mean())


def test_the_PLUME_over_a_fire_is_still_Heskestad_s_with_momentum_on():
    """The retuned drag and the mixing of momentum keep the plume where the old drag had it."""
    n, q = 36, 3000.0
    w = World(n, n, 64, voxel_cm=5)
    w.fill(0, n, 0, n, 0, 1, STONE)
    w.gas_flow = True
    w.gas_momentum = True
    for t in range(int(10 / TICK_S)):
        w.E[16:20, 16:20, 1:5] += q * TICK_S / 64
        w.step()
    T, uz = w.T(), w._gf["u"][2]
    kw = q / 1000.0
    flow, rise = {}, {}
    for z in (1.0, 1.5, 2.0):
        k = int(z / DX) + 1
        flow[z] = float(uz[:, :, k // 4].sum()) * H * H
        rise[z] = float(T[16:20, 16:20, k].mean()) - AMBIENT
        if z < 1.5:
            continue
        speed = float(uz[4:6, 4:6, k // 4].max())
        dT = 25.0 * kw ** (2 / 3) * z ** (-5 / 3)
        u = 1.032 * kw ** (1 / 3) * z ** (-1 / 3)
        v = 0.071 * kw ** (1 / 3) * z ** (5 / 3) / 1.2
        assert u / 1.5 < speed < u * 1.5 and v / 1.5 < flow[z] < v * 1.5 and dT / 1.5 < rise[z] < dT * 1.5, \
            (z, speed, u, flow[z], v, rise[z], dT)
    assert flow[1.0] < flow[1.5] < flow[2.0] and rise[1.0] > rise[1.5] > rise[2.0]


def test_a_SEALED_ROOM_still_carries_heat_smoke_and_oxygen_EXACTLY_with_momentum_on():
    n = 24
    w = World(n + 2, n + 2, n + 2, voxel_cm=5)
    w.open_sky = False
    w.fill(0, n + 2, 0, n + 2, 0, n + 2, STONE)
    w.mat[1:n + 1, 1:n + 1, 1:n + 1] = 0
    w.smass[1:n + 1, 1:n + 1, 1:n + 1] = 0
    w.gas_flow = True
    w.gas_momentum = True
    w.o2 = np.full(w.shape, np.float32(O2_PER_L * w.vox_l), np.float32)
    rng = np.random.default_rng(3)
    w.smoke[6:18, 6:18, 2:6] = rng.random((12, 12, 4)).astype(np.float32)
    w.E[6:18, 6:18, 1:3] += rng.random((12, 12, 2)).astype(np.float32) * 30.0
    e0 = float(w.E.sum(dtype=np.float64))
    s0 = float(w.smoke.sum(dtype=np.float64))
    for t in range(60):
        w.tick = t
        w._law_gasflow()
    assert float(np.abs(w._gf["u"][2]).max()) > 0.2
    assert abs(float(w.E.sum(dtype=np.float64)) + w.shed - e0) < 1e-6 * e0
    assert abs(float(w.smoke.sum(dtype=np.float64)) - s0) < 1e-6 * s0


# ── the push law ─────────────────────────────────────────────────────────────
def test_a_PATCH_displaces_what_it_sweeps_gives_no_net_volume_and_feels_the_gas_push_back():
    w = World(16, 16, 16, voxel_cm=5)
    s = w.add_surface(pushmod.Surface([[0.4, 0.4, 0.4]], 0.0025, (1, 0, 0), (2.0, 0, 0)))
    nc = (4, 4, 4)
    vc = [np.zeros(nc) for _ in range(3)]
    Q, dU, work, box = w._push_terms(vc, nc, H, TICK_S)
    assert abs(Q.sum()) < 1e-15 and abs(Q).max() == pytest.approx(0.0025 * 2.0)   # a v_n out, a v_n in
    assert dU[0].sum() > 0 and not dU[1].any() and not dU[2].any()                # gas pushed along +n
    assert s.force[0, 0] < 0 and work > 0                                          # it pushes back
    # reaction = momentum given: rho V du / dt, equal and opposite
    assert s.force[0, 0] == pytest.approx(-AIR_DENS * H ** 3 * dU[0].sum() * 1.0 / TICK_S, rel=1e-9)
    # a patch moving WITH the gas (no relative speed) gives it nothing
    vc2 = [np.full(nc, 2.0), np.zeros(nc), np.zeros(nc)]
    _, dU2, work2, _ = w._push_terms(vc2, nc, H, TICK_S)
    assert not dU2[0].any() and work2 == 0.0


def _tube_flow(b, ticks):
    out = []
    for _ in range(ticks):
        b.step()
        out.append(b.nozzle_flow())
    return out


def test_a_BELLOWS_made_of_the_push_law_moves_exactly_the_air_its_board_sweeps_through_its_nozzle():
    """The board is 0.6 x 0.6 m and moves at 0.3 m/s: 0.108 m^3/s must leave by the 0.2 x 0.2 m
    nozzle (2.7 m/s), at every moment of the stroke. Pushed back, the same board draws air IN."""
    b = bellows(speed=0.3, stroke=0.3)
    b.start(1.0)
    q = _tube_flow(b, 30)
    assert all(abs(x - 0.108) < 0.108 * 0.06 for x in q[3:]), q
    b.stop()
    b.step(2)
    b.xstart = b.xm
    b.start(-1.0)
    q2 = _tube_flow(b, 20)
    assert all(x < -0.108 * 0.9 for x in q2[3:]), q2          # inhaling
    assert b.w.gas_flow_stats["push_work_J"] > 0


def test_the_JET_of_a_bellows_CARRIES_ON_downstream_only_with_momentum():
    far = {}
    for mom in (True, False):
        b = bellows(speed=0.3, stroke=0.45, momentum=mom)
        b.start(1.0)
        b.step(40)
        u = b.w._gf["u"][0]
        k = b.length // 4
        far[mom] = [float(u[k + j, b.cy // 4, b.cz // 4]) for j in (2, 4)]
    assert far[True][0] > 1.2 and far[True][1] > 0.8              # still a jet 0.8 m past the nozzle
    assert far[False][1] < 0.2 * far[True][1]                       # without momentum it stops dead


def test_a_FAN_is_blades_turning_and_it_blows_one_way_or_the_other_by_the_blade_pitch_alone():
    P = {}
    for pitch in (0.6, -0.6):
        f = fan(pitch=pitch, size=(40, 40, 40), r_out=0.4)
        f.step(20)                                                   # half a second
        ux = f.w._gf["u"][0]
        P[pitch] = (float(0.5 * (ux[:-1] + ux[1:]).sum()) * AIR_DENS * H ** 3, f.thrust())
    # the gas moves along one way, the blades are pushed the other way (Newton), and swapping the
    # pitch swaps both
    assert P[0.6][0] * P[-0.6][0] < 0 and P[0.6][1] * P[-0.6][1] < 0
    assert P[0.6][0] * P[0.6][1] < 0
    # momentum theorem: the gas gained about what the blades' force gave it in half a second
    assert 0.5 < abs(P[0.6][0]) / (abs(P[0.6][1]) * 0.5) < 1.5, P


# ── wind: the same law at the edge of the world ──────────────────────────────
def test_the_WIND_is_the_outdoor_case_it_fills_the_box_at_its_speed_and_carries_smoke_downwind():
    d = wind_box(wind=(3.0, 0.0))
    w = d.w
    for t in range(int(4 / TICK_S)):
        w.E[6:10, 14:18, 1:5] += 1500 * TICK_S / 64
        w.smoke[6:10, 14:18, 1:5] += 0.001
        d.step()
    u = w._gf["u"][0]
    row = u[:, 4, 2]
    assert 2.6 < float(row.min()) and float(row.max()) < 3.2, row     # (the old drag loses a few percent)
    assert float(w.smoke[10:].sum()) > 100 * float(w.smoke[:6].sum())  # it all went downwind
    plume_z = np.argmax(w.T()[24, 14:18, :].mean(axis=0))
    assert plume_z >= 1


def test_a_BLOCK_in_the_wind_has_a_lee_with_reversed_flow_and_the_flow_it_lets_by_is_unchanged():
    d = wind_box(wind=(3.0, 0.0), block=(20, 28, 12, 20, 24))
    d.step(int(4 / TICK_S))
    u = d.w._gf["u"][0]
    assert float(u[5:7, 4, 2].max()) < 0.2 and float(u[9:12, 4, 2].min()) < -0.2   # wake, recirculating
    assert float(u[3, 4, 2]) > 1.5                                                  # slowed ahead of it
    edge = u[0, :, :]
    assert float(edge.min()) == pytest.approx(3.0) and float(edge.max()) == pytest.approx(3.0)   # told, not solved


def test_NO_WIND_no_surface_no_cost_in_a_quiet_world():
    w = _plume_world(True)
    for _ in range(5):
        w.step()
    assert w._push_live() is False and w.gas_flow_stats.get("box", 0) == 0
