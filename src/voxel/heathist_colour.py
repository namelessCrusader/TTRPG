"""TEMPER COLOUR (wave 15, lane 1e): a pure optics function of the oxide film thickness.

The film on hot steel is a thin transparent oxide on a reflecting metal. Two beams come back, one
off the top of the film and one off the metal under it, and they add with a phase set by the
film: delta = 4 pi n h / lambda (Cambridge MAP page, thin-film interference, (v); the quarter-wave
rule). Evaluated at three wavelengths it gives an RGB. NOTHING here is an input to the sim: it is
read on PERCEPTION only (a smith sees "straw" and quenches), so it costs nothing per tick.

The oxide law (layer_laws.oxide_step) is the cause; this is how it looks. Numbers:
  film index n 2.3 (iron oxides span 2.1-3.0, (u)); iron at 550 nm 2.9 - 3.0i (k).
The film thickness that the oxide law gives at 220-300 C after ten minutes is 70-134 nm. A
smith's temper chart puts straw at 200-230 C and blue near 290-300 C for an hour (k): the
law's growth is a little fast at the cold end (open problem, reported), the ORDER of the colours
is what is checked.
"""
import numpy as np

N_FILM = 2.3
N_METAL = complex(2.9, -3.0)
LAMBDA_NM = (610.0, 550.0, 465.0)       # red, green, blue (k)

def reflectance(h_m, lam_nm, n=N_FILM, nm=N_METAL):
    """Intensity reflectance of film (index n, thickness h_m) on metal (complex index nm)."""
    h = np.asarray(h_m, float) * 1e9
    r1 = (1.0 - n) / (1.0 + n)
    r2 = (n - nm) / (n + nm)
    d = 4.0 * np.pi * n * h / lam_nm
    r = (r1 + r2 * np.exp(-1j * d)) / (1.0 + r1 * r2 * np.exp(-1j * d))
    return np.abs(r) ** 2


def rgb(h_m):
    """RGB (0..1) of a film of thickness h_m, white light."""
    return np.stack([reflectance(h_m, l) for l in LAMBDA_NM], axis=-1)


def name(h_m):
    """What a smith calls it: bare, straw, brown, purple, blue or grey. The hue of the RGB
    normalised to its brightest channel (the eye adapts to a dim film), by named hue bands."""
    import colorsys
    c = np.clip(rgb(h_m), 0.0, 1.0)
    v = float(c.max())
    if v <= 1e-6:
        return "brown"
    r, g, b = (c / v)
    hh, ss, _ = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    deg = hh * 360.0
    if ss < 0.18:
        return "bare" if h_m < 5e-9 else "grey"
    if 35.0 <= deg < 80.0:
        return "straw"
    if deg >= 330.0 or deg < 35.0:
        return "brown"
    if 270.0 <= deg < 330.0:
        return "purple"
    if 170.0 <= deg < 270.0:
        return "blue"
    return "grey"
