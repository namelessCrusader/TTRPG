"""Scenes for the reaction runner and the Fick kernel (wave 15, S1 slices 1 and 2).

No craft word is in the engine: a scene lays out matter (mixes in cells) and names the
catalogue rows it wants. None is in SCENARIOS (no golden case moves). The scenes here:

    lime_kiln()      a cell of limestone held hot, CO2 going to the air beside it (rows: calcine)
    slake_cell()     a lump of quicklime wetted by pore water (rows: slake)
    putty_cell()     slaked lime in damp air (rows: carbonate): the surface layer only
    tan_hide()       a strip of hide, one end in tannin liquor (rows: tan_vegetable; Fick)
    wash_hide(w, res) the same hide after the liquor is swapped for clean water

`make_world` composes the two mixins on World until the merge agent adds them to World's
bases (then it returns World itself)."""
import numpy as np

from .fick_solids import FickSolidsMixin
from .materials import AIR, FLESH, STONE, WATER
from .reactions import ReactionMixin
from .sim import World

HIDE = [("collagen", "bound", 0, 1.0)]
LIQUOR = [("water", "dissolved", 0, 0.95), ("tannin", "dissolved", 0, 0.05)]   # 5 % w/w (u)
_CACHE = {}


def make_world(nx, ny, nz, voxel_cm=1, **kw):
    """A World with the runner and the Fick kernel available (both gates still off)."""
    base = World
    if not (issubclass(World, ReactionMixin) and issubclass(World, FickSolidsMixin)):
        base = _CACHE.get("w")
        if base is None:
            base = _CACHE["w"] = type("ChemWorld", (ReactionMixin, FickSolidsMixin, World), {})
    return base(nx, ny, nz, voxel_cm=voxel_cm, **kw)


def _lump(w, cell, mat_or_row, grams):
    """Put `grams` of a pure solid (mat id) or a mix row in a cell."""
    x, y, z = cell
    if mat_or_row < 64:
        w.mat[x, y, z] = mat_or_row
    else:
        w.set_solid((x, y, z), mat_or_row)
    w.smass[x, y, z] = grams


def _fluid(w, cell, row, ml):
    x, y, z = cell
    w.set_fluid((x, y, z), np.int64(row))
    w.fvol[x, y, z] = ml


# 5 cm world cells hold 125 ml; the lumps below are small so a test runs in seconds
def lime_kiln(grams=100.0, t_c=1000.0):
    """P: one cell of limestone (calcite) in open air. Returns (world, cell). The caller
    holds the cell at t_c (a kiln fire) by setting E between steps."""
    w = make_world(5, 5, 5, voxel_cm=5)
    w.fill(0, 5, 0, 5, 0, 1, STONE)                   # a stone floor: a lump rests on it
    cell = (2, 2, 1)
    _lump(w, cell, 16, grams)                        # mat 16 = calcite (limestone)
    w.set_reactions(["calcine"])
    return w, cell


def slake_cell(cao_g=50.0, water_ml=30.0):
    """P: quicklime with pore water (a lump of CaO in water), one open face above."""
    w = make_world(5, 5, 5, voxel_cm=5)
    w.fill(0, 5, 0, 5, 0, 1, STONE)                   # a stone floor: a lump rests on it
    cell = (2, 2, 1)
    _lump(w, cell, 24, cao_g)                        # mat 24 = quicklime
    _fluid(w, cell, 64 + WATER, water_ml)
    w.set_reactions(["slake"])
    return w, cell


def putty_cell(grams=100.0, rh=0.65):
    """P: a lump of slaked lime (plaster) in damp air, open on every side. Carbonation reaches
    the surface layer only (the CO2 beside the cell); through the plaster is slice 3."""
    w = make_world(5, 5, 5, voxel_cm=5)
    w.set_humidity(rh)
    w.fill(0, 5, 0, 5, 0, 1, STONE)                   # a stone floor: a lump rests on it
    cell = (2, 2, 1)
    _lump(w, cell, 25, grams)                        # mat 25 = slaked lime powder
    w.set_reactions(["carbonate"])
    return w, cell


def tan_hide(n=12, voxel_cm=0.2, liquor=LIQUOR, hide_ml_frac=0.6, hide_rho_g_ml=0.4):
    """P: a 1-D strip of n hide cells along x (collagen, pore water), with a liquor cell at
    x = 0 held as a reservoir by a large volume. Returns (world, hide cells, liquor cell).
    The cells are 2 mm (a 2 cm hide is 10 cells). The hide holds `hide_rho_g_ml` of collagen
    per ml of cell (hide is about 0.4 g/ml wet-salted skin, (u))."""
    w = make_world(n + 4, 3, 3, voxel_cm=voxel_cm)
    mx = w.ensure_mixes()
    hide = mx.parse(HIDE)
    liq = mx.parse(liquor)
    ml_cell = w.cap
    cells = [(2 + i, 1, 1) for i in range(n)]
    pore = mx.parse([("water", "dissolved", 0, 1.0)])
    for c in cells:
        _lump(w, c, hide, hide_rho_g_ml * ml_cell)
        w.mat[c] = FLESH                          # collagen has no mat of its own: it reads as flesh
        _fluid(w, c, pore, hide_ml_frac * ml_cell)
    res = (1, 1, 1)                                  # a reservoir: a free-liquor cell
    _fluid(w, res, liq, ml_cell)
    w.set_reactions(["tan_vegetable"])
    w.set_fick(["tannin"])
    w.fick_hold(res)                                 # the bath keeps its strength
    return w, cells, res


def wash_hide(w, res):
    """Swap the liquor of the bath cell `res` for clean water and hold it there: the hide
    now sits in a clean bath. The hide itself is untouched."""
    x, y, z = res
    w.fcomp[x, y, z] = 0
    w.fl[x, y, z] = WATER
    w.fick_hold(res)
