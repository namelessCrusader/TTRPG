"""THE FACE-ENVIRONMENT BUNDLE (wave 13, plan sections 1 and 3.3; lane B1).

ONE record that the heat face law (`heat._face_heat` and its numpy twin in
`HeatMixin._law_conduct`) reads, so that body heat (T7), cold (T8), mixtures
(T4) and injury (T10) all hang off the same plug and the signature of the
face law changes ONCE. At the defaults the bundle is inert and the law gives
the same bits it gave before it existed (`NO_ENV`). Nothing in the world
fills it in 13a.

WHAT IS IN IT, and the law each part belongs to:

  OVERLAY (env bit ENV_OVERLAY): `bz`, a uint16 id per cell (0 = no overlay;
      T7 puts person slot x 16 + zone there, T3 any face overlay), and `ov`,
      a small float32 table with one row per id (columns OV_*). Row 0 is the
      default row. It acts on the faces where a SURFACE cell meets gas or
      liquid, with the overlay id of the surface cell:

        OV_HNAT  > 0   the film coefficient is  OV_HNAT * |dT| ** 0.25  (the
                       natural-convection law, written sqrt(sqrt(.)) so the
                       compiled and the numpy law agree to the bit) instead of
                       the constant H_CONV.                       (T7 BH2)
        OV_RADW        the weight on the linearised radiation of hot gas onto
                       the surface (1 = as today, 0 = off: a garment's outer
                       skin radiates, not the skin under it).      (T7 BH3)
        OV_RSER  > 0   a resistance in SERIES, m2K/W (a garment, a layer of
                       dirt): the face conductance g becomes
                       1 / (1/g + OV_RSER / area).             (T7 BH3, T3)
        OV_AZ          the area factor: the true exposed area over the
                       lattice's. It multiplies the exchange with the
                       environment (gas and liquid faces), never the
                       conduction between two solids.              (T7 BH2)
        OV_RSC   > 0   core-to-skin resistance m2K/W of a skin node: the
                       conductance between a cell's skin and core is
                       open area / OV_RSC in place of the sqrt(a t) path.
                       This one acts in `_law_conduct`'s skin step, not
                       in the face kernel.                         (T7 BH1)

      An id past the end of the table reads row 0. A row whose columns are
      the defaults (DEFAULT_ROW) gives exactly the old bits.

  BOILING POINT (env bit ENV_BOIL): `bp`, a float32 per cell, the boiling
      point in C of the liquid in it (T4: brine boils hotter). It replaces
      the scalar BOIL in the boiling curve, in both laws.

  FLUX READOUT (`ff`, a bool): ask `_law_conduct` to hand the flux each face
      carried this tick (J, signed from the lower cell to the higher one, one
      array per axis) to `HeatMixin._face_flux(sl, fluxes)` (T10's skin burn
      reads the flux the law actually applied, and writes nothing back).

HOW A LANE FILLS IT (nothing does in 13a):

    w.ensure_bzone()[cells] = slot * 16 + zone            # the carried field
    w.face_env = FaceEnv(bz=w.bzone, ov=overlay_table([   # row 1, 2, ...
        {"hnat": 2.38, "rsc": 0.012, "az": 1.1},          # a bare skin zone
        {"hnat": 2.38, "rser": 0.1, "radw": 0.0}]),       # a clothed one
        bp=boiling_point_per_cell, ff=False)

The bundle is plain data (arrays, a table, two flags), so `snapshot` copies it
and the quiet probe digests it. Replacing `w.bzone` means building a new
FaceEnv. A lane that wants more than a table (a law of the bundle's own) can
override `HeatMixin._face_env(sl)` and return a FaceEnv built any way it likes.

WHAT IS NOT IN IT, and why. The radiation a surface throws out of the world
is written in numpy after the face loop in both paths, so cold's per-cell
environment temperature and room radiant node enter through the seam
`HeatMixin._env_radiation`, not through the kernel. T3's emissivity overlay
is written into the `eps` array the kernel already takes.

Plain numpy is the definition: the `*_np` helpers here are what the numpy twin
calls, and the compiled law is written to match them step for step in
float32 (tests/test_w13_face_env.py checks it to the bit).

NOTE FOR WHOEVER CHANGES THE NUMBERS BELOW. numba freezes ENV_OVERLAY, ENV_BOIL
and the OV_* column numbers into the compiled `heat._face_heat`, and its cache
is keyed on heat.py's file, not on this one: after changing one, touch heat.py
(or clear src/voxel/__pycache__) or the old numbers keep running.
"""
import numpy as np

# the bits of FaceEnv.env, which is also the `env` argument of heat._face_heat
ENV_OVERLAY = 1
ENV_BOIL = 2

# the columns of the overlay table
OV_AZ, OV_HNAT, OV_RSER, OV_RADW, OV_RSC = range(5)
NOV = 5
DEFAULT_ROW = (1.0, 0.0, 0.0, 1.0, 0.0)      # the old constants: nothing happens

# one-element stand-ins, so the compiled law has ONE signature whether or not
# a part of the bundle is in use (the same trick as air.py's _NO_MASK)
_NO_BZ = np.zeros((1, 1, 1), np.uint16)
_NO_OV = np.array([DEFAULT_ROW], np.float32)
_NO_BP = np.zeros((1, 1, 1), np.float32)


def overlay_table(rows=None):
    """A float32 overlay table of shape (n, NOV). Row 0 is always the default
    row; `rows` is a list of dicts {OV name: value} (or length-NOV sequences)
    for the rows after it, ids 1, 2, ..."""
    out = [DEFAULT_ROW]
    for r in rows or ():
        if isinstance(r, dict):
            row = list(DEFAULT_ROW)
            for name, v in r.items():
                row[{"az": OV_AZ, "hnat": OV_HNAT, "rser": OV_RSER,
                     "radw": OV_RADW, "rsc": OV_RSC}[name]] = v
        else:
            row = list(r)
            if len(row) != NOV:
                raise ValueError(f"an overlay row has {NOV} columns, not {len(row)}")
        out.append(row)
    return np.array(out, np.float32)


class FaceEnv:
    """The bundle (see the module docstring). `bz`/`bp` are arrays shaped like
    the lattice (or like the window `window` cut them to); `ov` is the table.
    The law reads the arrays and the table afresh every tick, so a lane changes
    a row (a garment getting wet, say) by writing into `ov` in place; a new
    bundle is only needed when an array is replaced or a part is added."""
    __slots__ = ("env", "bz", "ov", "bp", "ff")

    def __init__(self, bz=None, ov=None, bp=None, ff=False):
        env = 0
        if bz is not None or ov is not None:
            if bz is None or ov is None:
                raise ValueError("an overlay needs both its ids (bz) and its table (ov)")
            bz = np.asarray(bz)
            ov = np.ascontiguousarray(ov, np.float32)
            if bz.dtype != np.uint16:
                raise TypeError(f"overlay ids are uint16, not {bz.dtype}")
            if ov.ndim != 2 or ov.shape[1] != NOV or ov.shape[0] < 1:
                raise ValueError(f"the overlay table is (rows >= 1, {NOV}), not {ov.shape}")
            env |= ENV_OVERLAY
        else:
            bz, ov = _NO_BZ, _NO_OV
        if bp is not None:
            bp = np.asarray(bp)
            if bp.dtype != np.float32:
                raise TypeError(f"boiling points are float32, not {bp.dtype}")
            env |= ENV_BOIL
        else:
            bp = _NO_BP
        self.env, self.bz, self.ov, self.bp, self.ff = int(env), bz, ov, bp, bool(ff)

    @property
    def overlay(self):
        return bool(self.env & ENV_OVERLAY)

    @property
    def boils(self):
        return bool(self.env & ENV_BOIL)

    def inert(self):
        """True when the law will do exactly what it always did."""
        return not (self.env or self.ff)

    def window(self, sl):
        """The same bundle cut to the cells `sl` (a tuple of three slices),
        with contiguous arrays for the compiled law."""
        if self.inert() or sl is None:
            return self
        out = FaceEnv.__new__(FaceEnv)
        out.env, out.ov, out.ff = self.env, self.ov, self.ff
        out.bz = np.ascontiguousarray(self.bz[sl]) if self.overlay else self.bz
        out.bp = np.ascontiguousarray(self.bp[sl]) if self.boils else self.bp
        return out

    def fits(self, shape):
        """Do the arrays of the bundle have the shape of the cells the law is
        working on? The compiled law reads them without checking, so a bundle
        that does not fit is refused before it is used."""
        shape = tuple(shape)
        return ((not self.overlay or self.bz.shape == shape)
                and (not self.boils or self.bp.shape == shape))

    def gsb(self, u, dx):
        """BH1: the conductance between each cell's skin and core, W/K. `u`
        is the `_surfaces` block of the same cells. Where the overlay row of a
        cell has OV_RSC > 0 it is open area / OV_RSC; elsewhere it is `u.Gsb`,
        untouched. Returns `u.Gsb` itself when no overlay is in use."""
        if not self.overlay:
            return u.Gsb
        f32 = np.float32
        rsc = self.ov[ids_np(self.bz, self.ov.shape[0]), OV_RSC]
        use = u.has & (rsc > 0)
        if not use.any():
            return u.Gsb
        n_open = np.where(u.solid, u.n_out, np.where(u.wet, u.n_gas, 0))
        area = n_open.astype(f32) * (f32(dx) * f32(dx)) * u.af
        g = area / np.where(use, rsc, f32(1.0))
        return np.where(use, g, u.Gsb).astype(f32)


NO_ENV = FaceEnv()                           # the inert bundle: today's face law


# ── the numpy half of the law (the definition the compiled law matches) ─────
def ids_np(bz, nrows):
    """Overlay ids as table rows: an id past the end of the table is row 0."""
    return np.where(bz < nrows, bz, 0)


def film_np(ov, oid, d, h_const, rad):
    """The film coefficient plus the radiation weight of faces whose surface
    cell has overlay row `oid`: (hc, rad * radw), float32, where
    hc = OV_HNAT * sqrt(sqrt(|d|)) when OV_HNAT > 0 else `h_const`."""
    f32 = np.float32
    hn = ov[oid, OV_HNAT]
    rw = ov[oid, OV_RADW]
    hc = np.where(hn > 0, hn * np.sqrt(np.sqrt(np.abs(d))), f32(h_const))
    return hc, rad * rw


def exchange_np(g, ov, oid, ar):
    """What the overlay rows `oid` do to the conductance `g` of a face between
    a surface and its environment, over the area `ar`: first the resistance in
    series (OV_RSER), then the area factor (OV_AZ). Faces with the default row
    come out as they went in."""
    return series_np(g, ov[oid, OV_RSER], ar) * ov[oid, OV_AZ]


def series_np(g, rs, ar):
    """1 / (1/g + rs/ar) written g / (1 + g * rs / ar), for faces whose series
    resistance rs > 0 and area ar > 0; every other face keeps g."""
    one = np.float32(1.0)
    ok = (rs > 0) & (ar > 0)
    den = one + g * rs / np.where(ok, ar, one)
    return np.where(ok, g / den, g)
