"""Wave 14a lane A: every scene has a measure of humidity, on by default, by setting."""
import numpy as np
import pytest

from src.voxel import climate as K
from src.voxel import psychro
from src.voxel import scenes
from src.voxel.materials import STONE, WATER
from src.voxel.wetworld import (DEFAULT_SETTING, EVERY, SETTING_RH, HumidMixin, HumidWorld,
                                in_setting)


def _puddle(setting):
    with in_setting(setting):
        w = HumidWorld(16, 16, 12, voxel_cm=5)
    w.fill(0, 16, 0, 16, 0, 1, STONE)
    for x in range(5, 11):
        for y in range(5, 11):
            w.pour(x, y, 1, WATER, 30.0)
    w.wet.edges = False
    return w


def test_a_humid_world_has_a_measure_of_humidity_with_no_call():
    w = HumidWorld(8, 8, 8, voxel_cm=10)
    assert w.wet is not None and w.air_rh == SETTING_RH[DEFAULT_SETTING]
    assert abs(w.rh_at((4, 4, 4)) - w.air_rh) < 1e-3


def test_settings_run_from_arid_to_humid():
    assert SETTING_RH["arid"] < 0.2 and SETTING_RH["humid"] > 0.8
    assert sorted(SETTING_RH.values()) == [SETTING_RH[k] for k in ("arid", "indoor", "temperate", "damp", "humid")]
    for s, rh in SETTING_RH.items():
        with in_setting(s):
            w = HumidWorld(8, 8, 8, voxel_cm=10)
        assert w.air_rh == rh and abs(w.rh_at((4, 4, 4)) - rh) < 1e-3
    with pytest.raises(KeyError):
        in_setting("moist")
    with in_setting("arid"):
        pass
    assert HumidWorld(8, 8, 8, voxel_cm=10).air_rh == SETTING_RH[DEFAULT_SETTING]


def test_arid_air_dries_a_puddle_faster_than_humid_air():
    dry, wet = _puddle("arid"), _puddle("humid")
    for w in (dry, wet):
        for _ in range(3 * EVERY):
            w.step()
    assert dry.wet.stats["evap_g"] > 5.0 * wet.wet.stats["evap_g"] > 0.0


def test_a_scene_world_keeps_the_water_books():
    w = _puddle("damp")
    m0 = w.total_water()
    for _ in range(2 * EVERY):
        w.step()
    assert abs(w.total_water() - m0) < 1e-3 + 2e-6 * m0


def test_every_scene_has_a_setting_and_a_humid_world():
    assert set(scenes.SCENE_SETTING) <= set(scenes.SCENARIOS)
    assert set(scenes.SCENE_SETTING.values()) <= set(SETTING_RH)
    assert issubclass(scenes.World, HumidMixin)


def test_a_climate_sets_the_humidity_from_the_weather():
    w = HumidWorld(12, 12, 12, voxel_cm=10)
    w.set_climate(K.Climate.reference(t_c=10.0), soak=True)
    assert 0.0 < w.air_rh <= 1.0
    w.step()        # R4: OutsideMixin.set_climate goes before the mixin, so the humidity follows at the tick end
    assert abs(float(w.wet.store.bg("H2O")) / float(psychro.rho_sat(w.t_out)) - w.air_rh) < 1e-6


def test_a_bare_world_is_humid_by_default():
    """R4: HumidMixin is in World's bases, so a plain World(...) has a humidity, no call."""
    from src.voxel.sim import World
    w = World(8, 8, 8, voxel_cm=10)
    assert issubclass(World, HumidMixin) and w.wet is not None
    assert w.air_rh == SETTING_RH[DEFAULT_SETTING] and abs(w.rh_at((4, 4, 4)) - w.air_rh) < 1e-3
    with in_setting("arid"):
        assert World(8, 8, 8, voxel_cm=10).air_rh == SETTING_RH["arid"]


def test_a_bare_world_with_no_water_runs_the_same_wet_or_dry():
    """The idle cost of default humidity: a world with no liquid water, no ice, no vapour
    excess is bit for bit the dry world (the idle gate)."""
    from src.voxel.sim import World
    a, b = World(12, 12, 10, voxel_cm=10), World(12, 12, 10, voxel_cm=10)
    for w in (a, b):
        w.fill(0, 12, 0, 12, 0, 1, STONE)
    b.wet = None
    for _ in range(3 * EVERY):
        a.step()
        b.step()
    for f in ("E", "fvol", "mat", "smoke"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
