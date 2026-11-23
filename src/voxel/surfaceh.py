"""ONE SURFACE COEFFICIENT (wave 15, lane 1b, spec S4 item 2).

`h_surface` is the heat-transfer coefficient of a hot or cold surface to what touches it:
single-phase convection, the boiling curve when the liquid is hot enough to boil at the
surface, and grey radiation to the air. It is written once. The record port
(records_world._port_env) calls it; the face kernel keeps its own vectorised copy of the
same curve (heat.py `_hboil_t`), and this function CALLS that same curve so the numbers
live in one place (heat.py DT_CHF, DT_LEID, Q_CHF, H_FILM, H_NUC).

THE LAWS (all inherited from heat.py and thinheat.py, nothing new is invented here):
  liquid side  plain h from thinheat.h_conv (forced flat plate + Churchill-Chu, Incropera
               7.30 and 9.26), raised by the boiling curve (Nukiyama 1934): nucleate boiling
               up to the critical heat flux (Zuber, about 1.1 MW/m2 at 1 atm), a falling
               transition, then a FILM of steam (Bromley 1950, h about 250 W/m2K plus the wall's
               radiation through the film). The interface temperature that boils the liquid is
               set by the effusivities of wall and liquid (contact temperature, Incropera 5.x).
  air side     convection from thinheat.h_conv plus the linearised grey radiation
               eps sigma (Ts^2 + Te^2)(Ts + Te) (Incropera 1.8, small body in large room).
Numbers: sigma 5.670374e-8 (CODATA). The film h and the transition points are textbook
values (k); no source value for hot steel in water was found, so those stay (u).
"""
import numpy as np

from . import thinheat as TH
from .materials import AIR, BOIL, SIGMA, WATER, _CSOLID_ARR, _DENS_ARR, _EMISS_ARR, _KSOLID_ARR

_K0 = 273.15
_F32 = np.float32


def effusivity(mat):
    """sqrt(k rho c) of a solid material, W s^0.5/(m2 K): the same figure heat.py uses."""
    return float(np.sqrt(float(_KSOLID_ARR[mat]) * float(_DENS_ARR[mat]) * float(_CSOLID_ARR[mat]) * 1000.0))


def effusivity_water():
    from .heat import _FEFF_ARR
    return float(_FEFF_ARR[WATER])


def radiation_h(Ts, Te, eps):
    """Linearised grey radiation coefficient of a small surface to a large room, W/m2K."""
    a, b = Ts + _K0, Te + _K0
    return float(eps) * SIGMA * (a * a + b * b) * (a + b)


def h_liquid(Ts, Te, Lc, eps, es, el, tbo=BOIL, fluid="water"):
    """Surface-to-liquid h: plain wet h, replaced by the boiling curve above the boiling point."""
    from .heat import _hboil_t
    h = float(TH.h_conv(0.0, Lc, Ts - Te, fluid))
    if Ts <= tbo or Ts <= Te:
        return h
    return float(_hboil_t(_F32(h), _F32(Ts), _F32(Te), _F32(es), _F32(el), _F32(eps),
                          _F32(4.0 * SIGMA), _F32(_K0), _F32(tbo)))


def h_air(Ts, Te, Lc, eps):
    """Surface-to-air h: convection plus grey radiation."""
    return float(TH.h_conv(0.0, Lc, Ts - Te, "air")) + radiation_h(Ts, Te, eps)


def port_of(Ts, faces, Lc, mat, tbo=BOIL):
    """ONE port (h, T) for a surface with `faces` = [(count, Te, wet), ...] (free faces by
    medium). Each medium gets its own h; the port is their face-weighted mix, with T the
    h-weighted mean, so that h_port * (Ts - T_port) equals the sum of the faces' fluxes at
    this surface temperature (exact for the current Ts). None if no faces."""
    eps = float(_EMISS_ARR[mat])
    es = effusivity(mat)
    el = None
    n = hs = hT = 0.0
    for cnt, Te, wet in faces:
        if cnt <= 0:
            continue
        if wet:
            if el is None:
                el = effusivity_water()
            h = h_liquid(Ts, Te, Lc, eps, es, el, tbo)
        else:
            h = h_air(Ts, Te, Lc, eps)
        n += cnt
        hs += cnt * h
        hT += cnt * h * Te
    if n <= 0:
        return None
    return hs / n, hT / hs
