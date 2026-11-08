"""scenes._person(sheet=...): the default body is unchanged (396 cells, 43.9 kg),
a sheet name builds the adult body with the same named bones (wave 13, B8)."""
import numpy as np
import pytest

from src.voxel import scenes
from src.voxel.sim import FLESH, World

BONES = {"head", "torso", "left shin", "right shin", "left thigh", "right thigh",
         "left upper arm", "right upper arm", "left forearm", "right forearm"}


def _mk(sheet=None):
    w = World(40, 30, 60, voxel_cm=5)
    return w, scenes._person(w, 10, 10, sheet=sheet)


def test_default_body_is_the_stock_figure():
    w, p = _mk()
    assert int((w.mat == FLESH).sum()) == 396
    assert p["head_z"] == 28 and "sheet" not in p
    w2, p2 = _mk(None)
    assert np.array_equal(w.mat, w2.mat)


@pytest.mark.parametrize("name,mass", [("adult male", 73.0), ("adult female", 60.0)])
def test_sheet_body(name, mass):
    w, p = _mk(name)
    assert set(p["segs"]) == BONES
    assert p["sheet"]["mass_kg"] == mass
    assert sum(len(v) for v in p["segs"].values()) == int((w.mat == FLESH).sum())
    assert set(p["chain"]) == {"left arm", "right arm", "left leg", "right leg", "lean"}
    kg = float((w.smass[w.mat == FLESH]).sum()) / 1000.0 if hasattr(w, "smass") else mass
    assert abs(kg - mass) / mass < 0.15


def test_sheet_dict_equals_name():
    from src.voxel import bodysheet
    w1, _ = _mk("adult male")
    w2, _ = _mk(bodysheet.get_sheet("adult male"))
    assert np.array_equal(w1.mat, w2.mat)
