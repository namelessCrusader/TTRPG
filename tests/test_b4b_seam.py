"""Wave 13a, lane B4b: the optional mix, damage and ice fields on particles and
bonded members. They exist only when asked for; until then nothing changes."""
import numpy as np

from src.mpm.core import MPM, PARTICLE_OPT
from src.bonds.store import Bonds, MEMBER_OPT
from src.voxel.sim import World, STONE, WOOD


def _cells(x0, x1, y0, y1, z0, z1):
    return np.stack(np.meshgrid(np.arange(x0, x1), np.arange(y0, y1),
                                np.arange(z0, z1), indexing="ij"), -1).reshape(-1, 3)


def test_no_opt_array_exists_until_asked():
    w = MPM((12, 12, 12), dx=0.1)
    w.add_box((0.2, 0.2, 0.2), (0.5, 0.5, 0.4), "wood")
    for nm, *_ in PARTICLE_OPT:
        assert not hasattr(w, nm)
    assert w.opt_fields() == ()


def test_opt_fields_follow_append_keep_and_set_aside():
    w = MPM((12, 12, 12), dx=0.1)
    w.add_box((0.2, 0.2, 0.2), (0.5, 0.5, 0.4), "wood")
    n0 = len(w.x)
    w.ensure_opt("scomp")[:] = 7
    w.ensure_opt("dscal")
    w.add_box((0.6, 0.6, 0.6), (0.8, 0.8, 0.8), "stone")
    n = len(w.x)
    assert w.scomp.shape == (n,) and w.dscal.shape == (n,)
    assert (w.scomp[:n0] == 7).all() and (w.scomp[n0:] == 0).all()
    mask = np.arange(n) % 2 == 0
    held = w.set_aside(mask)
    assert len(w.scomp) == mask.sum()
    w.take_back(held)
    assert len(w.scomp) == n and (w.scomp[:n0] == 7).all()
    w.keep(mask)
    assert len(w.scomp) == len(w.x) == mask.sum()


def test_bonds_members_carry_opt_fields():
    b = Bonds((8, 8, 8), 0.1)
    b.ensure_opt_all()
    assert all(len(getattr(b, f)) == 0 for f, _ in MEMBER_OPT)


def test_a_mix_rides_the_seam_and_comes_back_to_the_lattice():
    w = World(12, 12, 14)
    w.fill(0, 12, 0, 12, 0, 1, STONE)
    w.fill(5, 7, 5, 7, 6, 8, WOOD)
    w.scomp = np.zeros(w.mat.shape, np.uint32)
    w.ice_g = np.zeros(w.mat.shape, np.float32)
    w.scomp[5:7, 5:7, 6:8] = 9
    w.ice_g[5:7, 5:7, 6:8] = 2.0
    assert w.matter.promote(_cells(5, 7, 5, 7, 6, 8)) == 8
    m = w.matter.mpm
    assert (m.scomp == 9).all() and m.ice_g.sum() > 0
    assert w.scomp[5:7, 5:7, 6:8].sum() == 0          # the lattice let go
    assert abs(m.ice_g.astype(float).sum() - 16.0) < 1e-3
    w.matter.rasterize()
    assert (w.scomp > 0).any() and abs(float(w.ice_g.sum()) - 16.0) < 1e-3
    w.matter.unrasterize()
    assert w.scomp.sum() == 0 and w.ice_g.sum() == 0
