"""Cold, body heat, mixtures and material laws in ONE world (wave 13b, FIX13b).

BodyWorld is a ColdWorld; mixtures and mat_laws live on World. A person holds a
cup of brine in a -10 C world: 200 ticks must run, and the energy book and the
mass books must close."""
import numpy as np

from src.voxel import climate as K
from src.voxel.bodyworld import BodyWorld
from src.voxel.materials import AIR, STONE, WATER, FLESH
from src.voxel.scenes import _person, _Wants
from src.voxel.scenes_mixtures import BRINE10


def test_cold_body_heat_mixtures_and_material_laws_share_one_world():
    n = 40
    w = BodyWorld(n, n, 40, voxel_cm=5)
    w.fill(0, n, 0, n, 0, 1, STONE)
    p = _person(w, 12, 12)
    w.policy = _Wants()
    cx, cy = 16, 12
    w.fill(cx, cx + 3, cy, cy + 3, 1, 4, STONE)
    w.fill(cx + 1, cx + 2, cy + 1, cy + 2, 1, 4, AIR)
    w.pour(cx + 1, cy + 1, 1, None, 50.0, mix=w.ensure_mixes().parse(BRINE10))
    w.set_climate(K.Climate.reference(t_c=-10.0), soak=True)
    p["held"] = {"cell": (cx, cy, 1), "mat": STONE, "label": "cup", "arm": "right arm"}
    assert w.climate is not None and w.mixes_live and w.body_heat and w.mat_laws
    mats = (WATER, STONE, FLESH)
    e0 = w.total_energy()
    m0 = {m: w.total_mass(m) for m in mats}
    f0 = w.total_fluid(WATER)
    for _ in range(200):
        w.step()
    scale = float(np.abs(w.E).sum())
    assert abs(w.total_energy() - e0) < 1e-9 * scale
    for m in mats:
        assert w.total_mass(m) == m0[m]
    assert abs(w.total_fluid(WATER) - f0) < 1e-6
    for nm in ("quant", "f32", "recast"):
        assert w.mixes.ledger_total(nm) == 0.0
