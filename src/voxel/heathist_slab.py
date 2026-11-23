"""A QUENCHED BAR, THROUGH ITS THICKNESS (wave 15, lane 1e). The sampler that makes the skin and
the core differ.

A voxel is 5 cm: a blade 2 mm thick, or the skin of a 2 cm bar, is far below one cell, and a cell
mean cannot see that the surface of a quenched bar cools faster than its centre (spec 2.3, the
cell-mean estimate). This is the through-thickness profile: a half slab, surface to centre, in
finite volumes, every node with its own temperature PATH and so its own state from heathist.

  conduction   k(T) and cp(T) of steel (u: k 52 -> 28 W/m/K and cp 450 -> 650 J/kg/K from 20 to
               850 C, the handbook trend; rho 7850), backward Euler (stable at any step)
  surface      h_eff (data: still air, oil, water, brine; heathist_data.H_QUENCH, a curve of surface temperature, (u)) plus grey
               radiation sigma eps (T^4 - Tb^4), eps 0.8 (an oxidised surface)
  the books    every node holds a SPECIFIC ENTHALPY (J/kg). The law's latent heat is subtracted from
               it in the same step, so  sum m H + sum m h_state + heat shed  is constant (a test)

Cost note: this is the thin-record sampler and is only built by a scene or a record that asks for
it; a world with no quenched bar never makes one.
"""
import numpy as np

from . import heathist as H
from . import heathist_data as D

RHO = 7850.0
SIGMA = 5.670374e-8
EPS = 0.8


def k_steel(T):
    return 52.0 - 24.0 * np.clip((np.asarray(T, float) - 20.0) / 830.0, 0.0, 1.0)


def cp_steel(T):
    return 450.0 + 200.0 * np.clip((np.asarray(T, float) - 20.0) / 830.0, 0.0, 1.0)


def _h_of_T(T):
    """Sensible specific enthalpy above 20 C, J/kg: the integral of cp (linear in T up to 850)."""
    T = np.asarray(T, float)
    x = np.minimum(T, 850.0) - 20.0
    return 450.0 * x + 0.5 * (200.0 / 830.0) * x * x + 650.0 * np.maximum(T - 850.0, 0.0)


def _T_of_h(h):
    h = np.asarray(h, float)
    a, b = 0.5 * (200.0 / 830.0), 450.0
    x = (-b + np.sqrt(b * b + 4.0 * a * np.maximum(h, 0.0))) / (2.0 * a)
    T = x + 20.0
    hi = h > _h_of_T(850.0)
    T = np.where(hi, 850.0 + (h - _h_of_T(850.0)) / 650.0, T)
    return np.where(h < 0, 20.0 + h / 450.0, T)


class Slab:
    """Half a bar of thickness `thickness_m` (surface node 0, centre node n-1), per unit face
    area. `T0` its uniform start temperature (it also starts at the state of a bar that has been
    heated there for `soak_s`: see `heat_to`)."""

    def __init__(self, thickness_m, c=0.2, mn=0.5, n=17, T0=20.0):
        self.n = n
        self.dx = 0.5 * thickness_m / n
        self.m = np.full(n, RHO * self.dx)             # kg per m2 of face, per node
        self.Hs = np.full(n, float(_h_of_T(T0)))       # J/kg sensible
        self.T = np.full(n, float(T0))
        self.s = H.new_state(c, mn, T0=T0, n=n)
        self.t = 0.0
        self.shed_J = 0.0                              # J per m2 that left through the surface

    # -- conduction + surface for dt, then the law on the new path
    def step(self, dt, h_eff=0.0, T_bath=20.0):
        self._conduct(dt, h_eff, T_bath)
        self.t += dt
        dh = H.step(self.s, self.T, dt)                # J/g of the state
        self.Hs -= dh * 1000.0
        self.T = _T_of_h(self.Hs)

    def _conduct(self, dt, h_eff, T_bath):
        """One BACKWARD-EULER conduction step (stable for any dt, so a 0.2 mm blade is as cheap as a
        bar) with k and cp frozen at the start of the step and the surface radiation linearised;
        the energy that moved is then taken from the FLUXES the new temperatures give, so the books
        close exactly whatever the linearisation did to the temperatures."""
        T, n, dx = self.T, self.n, self.dx
        cp = cp_steel(T)
        kf = 0.5 * (k_steel(T[:-1]) + k_steel(T[1:]))
        c = kf / dx                                           # W/m2/K between nodes
        h = float(np.interp(T[0], *zip(*h_eff))) if not np.isscalar(h_eff) else float(h_eff)
        hs = h + EPS * SIGMA * 4.0 * (T[0] + 273.15) ** 3     # plus linearised radiation
        M = np.diag(self.m * cp / dt)
        i = np.arange(n - 1)
        M[i, i] += c
        M[i + 1, i + 1] += c
        M[i, i + 1] -= c
        M[i + 1, i] -= c
        M[0, 0] += hs
        rhs = self.m * cp / dt * T
        rhs[0] += hs * T_bath
        Tn = np.linalg.solve(M, rhs)
        q = c * (Tn[:-1] - Tn[1:])                            # W/m2, node i -> i+1
        qs = hs * (Tn[0] - T_bath)
        dE = np.zeros(n)
        dE[:-1] -= q * dt
        dE[1:] += q * dt
        dE[0] -= qs * dt
        self.shed_J += qs * dt
        self.Hs = self.Hs + dE / self.m
        self.T = _T_of_h(self.Hs)

    def heat_to(self, T_C, soak_s=300.0):
        """The furnace: a long, gentle, uniform heat to T_C. The nodes go up together (a slow
        enough rate that the bar is uniform), and then soak: the law sees the whole path."""
        for T in np.linspace(self.T[0], T_C, 40)[1:]:
            self.Hs[:] = _h_of_T(T)
            self.T = _T_of_h(self.Hs)
            self.s["Tp"][:] = self.s["Tp"]
            dh = H.step(self.s, self.T, 2.0)
            self.Hs -= dh * 1000.0           # the furnace pays for the latent heat
            self.shed_J -= float(np.sum(dh * 1000.0 * self.m))
            self.T = _T_of_h(self.Hs)
        dh = H.step(self.s, np.full(self.n, T_C), soak_s)
        self.Hs[:] = _h_of_T(T_C)
        self.shed_J -= float(np.sum(dh * 1000.0 * self.m))
        self.T = _T_of_h(self.Hs)

    def hv(self):
        return H.hardness(self.s)

    def books(self):
        """Sum of sensible enthalpy + state enthalpy + heat shed, J per m2 of face. Constant under
        `step` (the furnace's `heat_to` is booked as negative shed)."""
        return float(np.sum(self.m * self.Hs) + np.sum(self.m * H.enthalpy(self.s) * 1000.0) + self.shed_J)

    def quench(self, medium="water", t_end=20.0, dt=0.01):
        h = D.H_QUENCH[medium] if isinstance(medium, str) else medium
        t = 0.0
        while t < t_end:
            self.step(dt, h)
            t += dt
        return self
