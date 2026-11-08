"""Proof-scene builders for the cold topic (T8, spec 6.5). DRAFT: each is a
function that returns a World (5 cm voxels) with a Climate put on it by
`set_climate`, which exists only after phase B step 0 mixes OutsideMixin into
World. They are not registered in SCENARIOS (phase B does that, and records the
new golden cases cold_yard, clear_night_frost, hut_warm). Called on a base18
World they raise NotImplementedError with the reason.

    cold_yard         S1: a stone hut in a yard at -12 C, everything soaked,
                      reference sky (T_sky = T_out): nothing changes
    hut_night         S2: the hut at night with a hearth lit then left, glass
                      pane, wood door, floor coupled to the ground
    clear_night_slab  S4/W5: a wood plank under a clear night sky, air +3 C
    wall_flux         W4: a 1-D column, 20 C indoor air held by a thermostat |
                      3 cells of stone | outdoor air

No live person is placed in any of them: flesh in the cold waits for T7 (D6).
"""
from . import climate as K
from .materials import AIR, GLASS, STONE, WOOD
from .coldworld import ColdWorld as World

NEEDS = "needs phase B seams: World.set_climate (OutsideMixin) is not mixed in yet"


def _need(w):
    if not hasattr(w, "set_climate"):
        raise NotImplementedError(NEEDS)


def _hut(w, x0=30, y0=20, nx=60, ny=60, h=48, wall=3, door=True, pane=True):
    """3 m x 3 m x 2.4 m of 15 cm stone walls, a wood roof slab, a wood door
    cell column and one glass pane (a 5 cm cell, 25% low in U; a real sheet is
    T2's)."""
    w.fill(x0, x0 + nx, y0, y0 + ny, 0, 1, STONE)                          # floor
    w.fill(x0, x0 + nx, y0, y0 + ny, 1, h, STONE)                          # walls ...
    w.fill(x0 + wall, x0 + nx - wall, y0 + wall, y0 + ny - wall, 1, h, AIR)    # ... hollowed
    w.fill(x0, x0 + nx, y0, y0 + ny, h, h + 3, WOOD)                       # roof
    if door:
        w.fill(x0 + nx // 2 - 8, x0 + nx // 2 + 8, y0, y0 + wall, 1, 40, WOOD)
    if pane:
        w.fill(x0 + nx - wall, x0 + nx - wall + 1, y0 + ny // 2 - 6, y0 + ny // 2 + 6, 24, 36, GLASS)
    return (x0 + wall, x0 + nx - wall, y0 + wall, y0 + ny - wall, 1, h)


def cold_yard(shape=(120, 100, 60), t_c=-12.0):
    """S1: everything at -12 C, a sky equal to the air. Quiet: 200 ticks change
    nothing and cost what a quiet world costs."""
    w = World(*shape, voxel_cm=5)
    _need(w)
    _hut(w)
    w.set_climate(K.Climate.reference(t_c=t_c), soak=True)
    return w


def hut_night(shape=(120, 100, 60)):
    """S2: a winter night, a hearth fire lit in the hut and left. The ground
    under the floor is the lower boundary (ground_depth_m = 1 m)."""
    w = World(*shape, voxel_cm=5)
    _need(w)
    box = _hut(w)
    c = K.Climate(lat_deg=50.0, t_ann_c=-2.0, a_ann_k=0.0, a_day_k=3.0, cloud=0.5, z_cloud_km=0.5,
                  start_doy=21.0, start_hour=19.0, rest_c=-2.0, ground_depth_m=1.0, seed=2)
    for y0 in (46, 50, 54):                                                # a small wood crib
        w.fill(40, 56, y0, y0 + 2, 1, 3, WOOD, frac=0.6)
    w.set_climate(c, soak=True)                                           # soak LAST
    w.E[44, 50, 2] += 2500.0                                              # a torch held to one stick
    w.scene_info = {"interior": box, "heater": "hearth crib", "stage": "hut_warm"}
    return w


def clear_night_slab(shape=(60, 60, 30), margin=10):
    """S4/W5: a wood plank, air +3 C, clear sky, dew point +1 C, midnight.
    The top face drops toward q/(H_CONV + h_r) = 88/14.2 = 6.2 K with a time
    constant (e/h)^2 = 763 s (spec 6.5, computed for this spec); frost_forms()
    is True once it is below the dew point. Cloud stops it. `margin` is the
    bare ground around the plank; the top face is flat, so a smaller world
    gives the same drop at the middle for far fewer cells (a test uses that)."""
    w = World(*shape, voxel_cm=5)
    _need(w)
    w.fill(margin, shape[0] - margin, margin, shape[1] - margin, 0, 8, WOOD)
    c = K.Climate(lat_deg=45.0, a_day_k=0.0, a_ann_k=0.0, t_ann_c=3.0, cloud=0.0, dew_dep_k=2.0,
                  rest_c=3.0, start_hour=0.0, start_doy=15.0)
    w.set_climate(c, soak=True)
    return w


def wall_flux(length=16, t_in=20.0, t_out=-10.0):
    """W4: x runs from indoor air (held at t_in by a thermostat, under a stone
    roof so it is not outdoor) through 3 cells (0.15 m, k 2.0) of stone to
    outdoor air at t_out. The side and top edges are buried soil
    (ground_top_m = 10 m): adiabatic, so the column is 1-D. At steady state the
    flux is within 20% of dT/(1/10 + 0.15/2.0 + 1/(8.6 + h_r)), about 117 W/m2
    for 30 K."""
    w = World(length, 3, 3, voxel_cm=5)
    _need(w)
    w.fill(0, 6, 0, 3, 2, 3, STONE)                                      # a roof over the indoor air
    w.fill(0, 1, 0, 3, 0, 3, STONE)                                      # and an end wall: a roof with one
                                                                         # end free is a cantilever and falls
    w.fill(6, 9, 0, 3, 0, 3, STONE)                                      # the wall
    for y in range(3):
        for z in range(2):
            w.thermostats.append((2, y, z, t_in))
    c = K.Climate.reference(t_c=t_out).replace(rest_c=t_in, ground_top_m=10.0)
    w.set_climate(c, soak=True)
    return w
