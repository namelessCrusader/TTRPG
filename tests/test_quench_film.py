"""Wave 15 lane 2f: a voxel quench in a cold bath cools at the quench handbooks' rate, the
vapour blanket collapses in a cold bath and holds in a hot one, the fused and plain face
laws agree, and every joule is booked."""
import numpy as np
import pytest

from src.voxel import heat as H
from src.voxel.materials import AMBIENT, IRON, WATER
from src.voxel.scenes_records import quench_blade
from src.voxel import surfaceh as S


def _h(tw, tl, es=None):
    es = S.effusivity(IRON) if es is None else es
    return float(H._hboil(np.float32(1500.0), np.float32(tw), np.float32(tl), np.float32(es),
                          np.float32(S.effusivity_water()), np.float32(0.8),
                          np.float32(4 * 5.670374e-8), np.float32(273.15)))


def test_the_blanket_collapses_only_in_a_cold_bath():
    assert _h(850.0, 95.0) < 400.0              # near boiling: the steam film (H27)
    assert _h(850.0, 60.0) < 400.0              # 40 K below: still the plain curve (H27's bath)
    assert _h(850.0, 20.0) == pytest.approx(H.H_COLLAPSE, rel=0.01)
    assert _h(850.0, 40.0) > _h(850.0, 60.0)


def test_the_fused_and_plain_face_laws_agree_on_a_cold_quench():
    out = []
    for fused in (True, False):
        w, _ = quench_blade(6.0, water=True, record=False)
        w.fused = fused
        for _ in range(30):
            w.step()
        out.append((w.E.copy(), w.total_energy()))
    assert out[0][1] == pytest.approx(out[1][1], rel=1e-5)
    assert np.allclose(out[0][0], out[1][0], rtol=1e-4, atol=1.0)


def test_a_6mm_blade_in_cold_water_quenches_in_seconds_and_the_joules_are_booked():
    """BEFORE: 850 -> 756 C in 5 s (the steam film, h 250). AFTER: below 400 C in 5 s, the
    lumped answer for h of 3000-4000 (tau = rho c d / 2h = 2.6-3.5 s, 850 -> 330 C in 4-5 s)."""
    w, _ = quench_blade(6.0, water=True, record=False)
    c = (slice(2, 10), slice(4, 5), slice(1, 2))
    e0 = w.total_energy()
    for _ in range(200):
        w.step()
    assert float(w.T()[c].mean()) < 400.0
    assert abs(w.total_energy() - e0) < 2e-3 * abs(e0)      # the stone floor and the tank hold it
