"""MARKS (wave 13, T3): patterns on faces, the channel a mark must survive, and the
generic thin-film colour. Pure functions and two small classes; numpy only.

A mark is FORM plus CONTENT (F8). The form (a pattern: a bitmap or a list of
strokes) must survive the channel: light, distance, contrast, blur. The content is
a code (L19) that only a reader who knows the code gets; here a pattern only
carries a pointer to it (make_content). specs/T3-layers_v2.txt 4.11 and 4.12.

A pattern lives in the face's frame: u along the face's right, v up, seen from
outside. Pressing face S onto face D reverses the viewing side, so the lateral axis
is mirrored: u' = W - u about the contact centre line. Bitmaps are [v, u] arrays,
2 mm pitch, at most 32 x 32 per face window.
"""
import math

import numpy as np

PITCH = 2.0e-3                   # m, bitmap pitch (u)
MAX_WINDOW = 32                  # cells per side of a face window
TABOR = 3.0                      # impress needs p >= 3 tau_y (Tabor: H = 3 tau_y)


# ---- patterns ----------------------------------------------------------------------
class Bitmap:
    """A height or ink map. data[v, u]; pitch in metres. dtype may be float after
    blur."""

    def __init__(self, data, pitch=PITCH):
        self.data = np.array(data)
        self.pitch = float(pitch)

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return Bitmap(self.data.copy(), self.pitch)


class Strokes:
    """A list of strokes (x0, y0, x1, y1, width, amount) in metres plus glyph runs
    (symbol, x, y, size, amount) that name a code symbol. `width_m` and `height_m`
    are the pattern frame; `sigma` the blur it has taken (variance adds)."""

    def __init__(self, strokes=(), glyphs=(), width_m=0.05, height_m=0.05, sigma=0.0):
        self.strokes = [tuple(float(x) for x in s) for s in strokes]
        self.glyphs = [tuple(g) for g in glyphs]
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.sigma = float(sigma)

    def copy(self):
        return Strokes(self.strokes, self.glyphs, self.width_m, self.height_m, self.sigma)

    def rasterize(self, pitch=PITCH):
        """An antialiased Bitmap of the strokes (glyphs are not drawn)."""
        nu = max(1, int(round(self.width_m / pitch)))
        nv = max(1, int(round(self.height_m / pitch)))
        uu = (np.arange(nu) + 0.5) * pitch
        vv = (np.arange(nv) + 0.5) * pitch
        U, V = np.meshgrid(uu, vv)
        out = np.zeros((nv, nu))
        for x0, y0, x1, y1, w, amt in self.strokes:
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            t = np.clip(((U - x0) * dx + (V - y0) * dy) / L2, 0, 1) if L2 > 0 else 0.0
            d = np.hypot(U - (x0 + t * dx), V - (y0 + t * dy))
            cov = np.clip((w / 2.0 - d) / pitch + 0.5, 0.0, 1.0)
            out = np.maximum(out, cov * amt)
        return Bitmap(out, pitch)


def make_content(code, claimed_source=None, true_maker=None, payload=None):
    """The content pointer a pattern carries (L19 owns the meaning)."""
    return {"code": code, "claimed_source": claimed_source, "true_maker": true_maker,
            "payload": payload}


def window(bitmap, u0, v0, nu, nv):
    """A face window cut from a bigger stamp (a stamp that spans faces), at most
    MAX_WINDOW cells a side."""
    nu, nv = min(nu, MAX_WINDOW), min(nv, MAX_WINDOW)
    return Bitmap(bitmap.data[v0:v0 + nv, u0:u0 + nu].copy(), bitmap.pitch)


# ---- mirror, impress, print, deposit -----------------------------------------------
def mirror(pattern):
    """Mirror the lateral axis (u). Twice is the identity."""
    if isinstance(pattern, Bitmap):
        return Bitmap(pattern.data[:, ::-1].copy(), pattern.pitch)
    W = pattern.width_m
    return Strokes([(W - s[0], s[1], W - s[2], s[3], s[4], s[5]) for s in pattern.strokes],
                   [(g[0], W - g[1], g[2], g[3], g[4]) for g in pattern.glyphs],
                   pattern.width_m, pattern.height_m, pattern.sigma)


def wax_yield(T_C, tau_50=1.0e5, decades_per_k=0.15, cap=2.0e7):
    """Yield stress of beeswax, Pa: three orders of magnitude lower from 50 to 70 C
    (u curve in the row): tau_50 at 50 C, 0.15 decade per kelvin, capped when solid."""
    return min(cap, tau_50 * 10.0 ** (-decades_per_k * (T_C - 50.0)))


def can_impress(p, tau_y):
    """IMPRESS needs p >= 3 tau_y (Tabor) of the receiving layer."""
    return p >= TABOR * tau_y


def impress(die, p, tau_y, layer_thickness_m, relief_m=None):
    """A die or sole pressed into a soft layer. The new relief is the NEGATIVE of
    the die and MIRRORED: a die cut in reverse reads true in the wax. Returns a
    Bitmap of depression depth (same scale as the die's raised amount, 0..255 as
    given) or None when p < 3 tau_y. Depth is min(die relief, layer thickness); mass
    is unchanged (the caller compacts por in the patch)."""
    if not can_impress(p, tau_y):
        return None
    amt = mirror(die).data.astype(float)
    if relief_m is not None and relief_m > 0.0:
        amt = amt * min(1.0, layer_thickness_m / relief_m)       # cannot cut deeper than the layer
    return Bitmap(amt, die.pitch)


def mirrored_top_pattern(layers, key):
    """The mirror image of the pattern on the top layer of `key`, ready to pass to
    Layers.contact(pattern=...) (a footprint, a seal, a print), or None."""
    st = layers.stack(key)
    if not st or layers.pat[st[0]] < 0:
        return None
    return mirror(layers.patterns[int(layers.pat[st[0]])])


def ink_print(stamp, ink=1.0):
    """Relief printing (a stamp, a seal pad): the raised parts carry ink and D gets
    the mirrored positive."""
    m = mirror(stamp)
    return Bitmap(m.data * ink, m.pitch)


def deposit_patch(patch, share=1.0):
    """A foot in mud, a hand in blood: the contact patch's shape is the transfer
    area, mirrored."""
    m = mirror(patch)
    return Bitmap(m.data * share, m.pitch)


# ---- degrade ------------------------------------------------------------------------
def blur_sigma(D, t):
    """sigma = sqrt(2 D t), m: ink on paper feathers 0.1-0.3 mm (u)."""
    return math.sqrt(2.0 * D * t)


def _gauss_kernel(sigma_px):
    r = max(1, int(math.ceil(4.0 * sigma_px)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    return k / k.sum()


def blur(pattern, D, t):
    """Gaussian blur of variance 2 D t (m^2) while wet. A Bitmap is convolved
    (separably); a Strokes object just adds the variance (valid under one pitch)."""
    s2 = 2.0 * D * t
    if isinstance(pattern, Strokes):
        q = pattern.copy()
        q.sigma = math.sqrt(q.sigma ** 2 + s2)
        return q
    s_px = math.sqrt(s2) / pattern.pitch
    if s_px <= 1e-9:
        return pattern.copy()
    k = _gauss_kernel(s_px)
    a = pattern.data.astype(float)
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a)
    a = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, a)
    return Bitmap(a, pattern.pitch)


def smear(pattern, length_m):
    """Smear along u by a line kernel of the slide length: lowers contrast."""
    if isinstance(pattern, Strokes):
        q = pattern.copy()
        q.sigma = math.sqrt(q.sigma ** 2 + length_m ** 2 / 12.0)      # box variance
        return q
    n = max(1, int(round(length_m / pattern.pitch)))
    k = np.ones(n) / n
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"),
                            1, pattern.data.astype(float))
    return Bitmap(a, pattern.pitch)


def fade(pattern, k, dt):
    """amount *= exp(-k dt)."""
    f = math.exp(-k * dt)
    if isinstance(pattern, Strokes):
        q = pattern.copy()
        q.strokes = [s[:5] + (s[5] * f,) for s in q.strokes]
        return q
    return Bitmap(pattern.data.astype(float) * f, pattern.pitch)


def wear(pattern, dh_amount):
    """Archard wear on the amount map: the raised amounts lose dh_amount."""
    if isinstance(pattern, Strokes):
        q = pattern.copy()
        q.strokes = [s[:5] + (max(0.0, s[5] - dh_amount),) for s in q.strokes]
        return q
    return Bitmap(np.maximum(0.0, pattern.data.astype(float) - dh_amount), pattern.pitch)


def ncc(a, b):
    """Normalised cross-correlation of two patterns (Bitmap, Strokes or arrays):
    1.0 for a cast copy, lower for a hand copy. Shapes are cropped to the common
    window."""
    def arr(x):
        if isinstance(x, Strokes):
            x = x.rasterize()
        return x.data.astype(float) if isinstance(x, Bitmap) else np.asarray(x, float)
    A, B = arr(a), arr(b)
    nv, nu = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    A, B = A[:nv, :nu].ravel(), B[:nv, :nu].ravel()
    A, B = A - A.mean(), B - B.mean()
    d = math.sqrt(float(A @ A) * float(B @ B))
    return float(A @ B) / d if d > 0.0 else 0.0


# ---- relief seen by light ---------------------------------------------------------------
def relief_contrast(depth_m, elev_deg, pitch=PITCH):
    """Contrast of a relief under light at elevation elev: min(1, (depth/tan(elev))/pitch)
    (u). It shows best in raking light and not in flat light."""
    t = math.tan(math.radians(max(elev_deg, 1e-3)))
    return min(1.0, depth_m / t / pitch)


# ---- legibility: the form must survive the channel ------------------------------------------
def _finite(name, v):
    """A number the channel laws can use: a real, finite value (NaN and infinity are
    no light, no distance and no size: ValueError, never a silent NaN back)."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def luminance(rho_bg, E_lux=300.0):
    """cd/m2 of a diffuse background lit by E lux (300 lux until light exists). No
    light (E <= 0, or a black background) is 0.0 exactly: darkness is a legal input
    and every reader below answers it with "cannot read". A non-finite reflectance
    or illuminance is ValueError."""
    rho_bg, E_lux = _finite("rho_bg", rho_bg), _finite("E_lux", E_lux)
    return max(0.0, rho_bg) * max(0.0, E_lux) / math.pi


def lux_from_source(cd, d):
    """Illuminance of a point source of cd candela at distance d, lux. A source of
    0 cd (or less) gives 0.0 lux: darkness. d must be > 0 (ValueError), and cd and d
    must be finite numbers (ValueError: a NaN source was read as darkness before). A
    distance so small that d squared underflows (under 1e-162 m) gives math.inf, the
    source is on the eye; it was a ZeroDivisionError. A world sink must clamp that
    infinite illuminance before it is lit onto a background (luminance refuses it)."""
    cd, d = _finite("cd", cd), _finite("d", d)
    if not d > 0.0:
        raise ValueError(f"distance must be > 0, got {d!r}")
    return max(0.0, cd) / d / d


def contrast(rho_mark, rho_bg):
    """|rho_mark - rho_bg| / rho_bg. On a black background (rho_bg 0) the ratio is
    unbounded: math.inf for a mark that reflects anything, 0.0 for a mark that is
    black too."""
    if rho_bg <= 0.0:
        return math.inf if rho_mark > 0.0 else 0.0
    return abs(rho_mark - rho_bg) / rho_bg


def c_min(L):
    """Smallest contrast seen (Rose-de Vries; u): 0.02 for L >= 3 cd/m2, else
    0.02*sqrt(3/L). At L <= 0 (no light) nothing can be seen at any contrast: math.inf.
    A denormal L keeps the formula's value (3/L would overflow, so the root is split).
    A non-finite L is ValueError."""
    L = _finite("L", L)
    if L <= 0.0:
        return math.inf
    if L >= 3.0:
        return 0.02
    q = 3.0 / L
    return 0.02 * (math.sqrt(q) if math.isfinite(q) else math.sqrt(3.0) / math.sqrt(L))


def mar_arcmin(L):
    """Minimum angle of resolution, arcmin: 1 for L >= 10, else 1 + 2.5 log10(10/L) (u).
    At L <= 0 (no light) no angle resolves anything: math.inf. For a denormal L the
    ratio 10/L overflows, so the log is split: log10(10/L) = 1 - log10(L), the same
    number. A non-finite L is ValueError."""
    L = _finite("L", L)
    if L <= 0.0:
        return math.inf
    if L >= 10.0:
        return 1.0
    q = 10.0 / L
    return 1.0 + 2.5 * (math.log10(q) if math.isfinite(q) else 1.0 - math.log10(L))


def subtense_arcmin(size_m, d_m):
    """Angle a feature of size_m subtends at d_m, arcmin. At d_m <= 0 (the eye on the
    feature) the angle is the half-plane limit, 180 degrees = 10800 arcmin."""
    if d_m <= 0.0:
        return 180.0 * 60.0
    return math.degrees(2.0 * math.atan(size_m / (2.0 * d_m))) * 60.0


def letter_for_distance(d_m, arcmin=5.0):
    """Height of a letter that subtends `arcmin` at d (a 5 arcmin letter is 8.7 mm at 6 m)."""
    return 2.0 * d_m * math.tan(math.radians(arcmin / 60.0) / 2.0)


def read_distance(stroke_m, L):
    """The distance where a stroke just reaches the MAR, m: s/(2 tan(MAR/2)). A
    0.56 mm print stroke in moonlight (L 0.025) reads within 0.257 m. With no light
    (L <= 0) it reads at no distance: 0.0. A non-finite stroke or L is ValueError."""
    stroke_m, L = _finite("stroke_m", stroke_m), _finite("L", L)
    if L <= 0.0:
        return 0.0
    mar = math.radians(mar_arcmin(L) / 60.0)
    return stroke_m / (2.0 * math.tan(mar / 2.0))


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def legible(stroke_m, d_m, rho_bg, rho_mark, E_lux=300.0, sigma_m=0.0):
    """Probability 0..1 that a mark reads: a smooth step over both margins (feature
    angle against the MAR, contrast against Cmin; full at the threshold, none at 70
    percent of it), reduced by blur sigma against stroke width. In total darkness
    (no light on the page, E_lux <= 0) the answer is 0.0, "cannot read", for every
    distance and every mark: the function does not raise. On a black background lit
    from outside the mark is the only light in the field, so the eye adapts to the
    mark's own luminance."""
    L = luminance(rho_bg, E_lux)
    if L <= 0.0:
        L = luminance(rho_mark, E_lux)                  # black page: adapt to the mark
    if L <= 0.0:
        return 0.0
    ang = subtense_arcmin(stroke_m, d_m)
    ps = _smooth((ang / mar_arcmin(L) - 0.7) / 0.3)
    pc = _smooth((contrast(rho_mark, rho_bg) / c_min(L) - 0.7) / 0.3)
    pb = 1.0 / (1.0 + (sigma_m / stroke_m) ** 2) if stroke_m > 0 else 0.0
    return ps * pc * pb


# ---- thin-film colour (generic: no iron special case) ---------------------------------------
WAVES = (610e-9, 550e-9, 465e-9)           # R, G, B


def thin_film(n_film, n_sub, d, lam=None):
    """Reflectance of a thin film of index n_film and thickness d on a substrate of
    (complex) index n_sub, normal incidence in air:
    r = (r01 + r12 e^(-2 i delta))/(1 + r01 r12 e^(-2 i delta)), delta = 2 pi n d/lambda,
    R = |r|^2. lam=None gives (R at 610, 550, 465 nm); a number gives one value.
    Temper film, copper patina, anodised titanium and oil on water all use this
    function with other rows."""
    if lam is None:
        return tuple(thin_film(n_film, n_sub, d, w) for w in WAVES)
    r01 = (1.0 - n_film) / (1.0 + n_film)
    r12 = (n_film - n_sub) / (n_film + n_sub)
    e = np.exp(-2j * (2.0 * math.pi * n_film * d / lam))
    r = (r01 + r12 * e) / (1.0 + r01 * r12 * e)
    return float(abs(r) ** 2)
