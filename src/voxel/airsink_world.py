"""The sealed-room sink: where a body's breath, sweat and warmth go when the room has walls
(wave 13, T5 and T7, lane C10).

The open-world sink (airsink.py) lets the air be endless: vapour, oxygen and
carbon dioxide are only counted in `gone`, heat in `shed`. In a shut room that
is wrong. Four people asleep in a closed room breathe its oxygen down, fill it
with carbon dioxide and water vapour and warm it. This sink keeps all four in
the ROOM:

  * heat goes into the room's air cells (their energy E), by heat capacity, so
    the lattice total holds and `shed` does not move;
  * oxygen is drawn from the lattice oxygen field (`w.o2`, grams per voxel),
    evenly by what each air cell holds. Oxygen is mass the room had;
  * carbon dioxide and water vapour are kept in the sink as litres and grams in
    the room (the lattice has no field for either yet). The humidity reads from
    the vapour against what the room's air can hold (psychro.rho_sat); above
    saturation the rest is booked as `condensed_g` (it would wet the walls; no
    surface takes it yet, said in the report).

THE TICK PATH AND THE SKIP PATH CALL THE SAME SINK (plan 3.3). In mode "now" a
call lands at once (the tick law calls it every body second). In mode
"collect" the heat is held back (`take_heat`) so a tier-1 jump can hand it to
the room's thermal net as a source over the chunk, and the gases are applied
at the end of the chunk (`flush`). Same totals either way.

Nothing is switched on by this file: a world gets the sink by `w.airsink =
RegionSink(w)`. Marks: (d) design choice, (u) unverified.
"""
import numpy as np

from .airsink import DEFAULT_O2
from .materials import AIR, AMBIENT, O2_PER_L
from . import psychro

O2_G_PER_L = 1.429                      # grams of O2 in a litre of O2 at 0 C, 1 atm (matches slownet)
CO2_G_PER_L = 1.977
LATENT_J_G = float(psychro.h_fg(25.0))  # latent heat used to split a body's heat into 'with the vapour' and 'sensible'
FLUSH_TICKS = 40                        # in mode "now" the gases land at most this many ticks late (one body second)


class RegionSink:
    """An AirSink for the air of one closed room (the cells of `box` that are AIR)."""

    def __init__(self, w, box=None, mode="now"):
        self.w = w
        self.box = tuple(slice(0, n) for n in w.shape) if box is None else tuple(box)
        self.mode = mode
        self.vapour_g = 0.0             # water vapour in the room's air, grams
        self.co2_l = 0.0                # carbon dioxide put into the room, litres
        self.o2_l = 0.0                 # oxygen taken from the room, litres (positive = taken)
        self.condensed_g = 0.0          # vapour past saturation (would wet the walls), grams
        self.heat_J = 0.0               # heat put into the room's air, joules
        self.latent_J = 0.0             # latent heat the room's vapour carries (E + this closes the books)
        self.pend = {"heat": 0.0, "vap": 0.0, "o2": 0.0, "co2": 0.0, "lat": 0.0}
        self._air_gen = None
        self._air_mask = None
        self._air_C = None               # heat capacity of the air cells (cached with the mask)
        self._flushed = int(w.tick)      # tick of the last gas flush

    # --------------------------------------------------------- the room ----
    def air(self):
        """Boolean mask of the room's air cells (cached against the matter generation)."""
        gen = getattr(self.w, "_matter_gen", None)
        if self._air_mask is None or gen is None or gen != self._air_gen:
            self._air_mask = (self.w.mat[self.box] == AIR)
            self._air_gen = gen
            self._air_C = np.asarray(self.w.heat_capacity()[self.box], dtype=np.float64)[self._air_mask]
        return self._air_mask

    def volume_l(self):
        return float(self.air().sum()) * self.w.vox_l

    def mean_temp(self):
        m = self.air()
        if not m.any():
            return AMBIENT
        C = self._air_C
        return float(AMBIENT + np.asarray(self.w.E[self.box], dtype=np.float64)[m].sum() / C.sum())

    def o2_mass_g(self):
        self._sync()
        return float(np.asarray(self.w.o2[self.box], dtype=np.float64)[self.air()].sum())

    # -------------------------------------------------- the protocol -------
    def add(self, vapour_g=0.0, heat_J=0.0, o2_l=0.0, co2_l=0.0, where=None):
        lat = 0.0
        if vapour_g > 0.0 and heat_J > 0.0:
            # the part of the heat that left the body as the latent heat of its vapour stays WITH the vapour
            lat = min(heat_J, vapour_g * LATENT_J_G)
        heat_J -= lat
        p = self.pend
        p["lat"] += lat
        p["vap"] += vapour_g
        p["o2"] += o2_l
        p["co2"] += co2_l
        if self.mode == "collect":
            p["heat"] += heat_J                 # held for the net (take_heat)
            return
        self._heat(heat_J)                      # the joules land now, so total_energy() is exact
        if int(self.w.tick) - self._flushed >= FLUSH_TICKS:
            self.flush()                        # the gases land at most one body second late

    def ambient(self, where=None):
        self._sync()
        T = self.mean_temp()
        V_m3 = self.volume_l() / 1000.0
        try:
            cap_g = psychro.rho_sat(T) * V_m3 if V_m3 > 0 else 0.0
        except Exception:
            cap_g = 0.0
        rh = min(1.0, self.vapour_g / cap_g) if cap_g > 0 else 0.0
        return T, rh

    def o2_fraction(self, where=None):
        self._sync()
        m = self.air()
        n = int(m.sum())
        if n == 0:
            return DEFAULT_O2
        fresh = O2_PER_L * self.w.vox_l
        return DEFAULT_O2 * self.o2_mass_g() / (n * fresh)

    def mixing(self, tol=1e-3):
        """Is the room's own air uneven in oxygen (more than `tol` of the fresh amount from cell to cell)?
        A room drawn down as one body of gas is not mixing; a plume or a draught is."""
        self._sync()
        m = self.air()
        if not m.any():
            return False
        o2 = np.asarray(self.w.o2[self.box], dtype=np.float64)[m]
        return bool((o2.max() - o2.min()) > tol * O2_PER_L * self.w.vox_l)

    def co2_fraction(self):
        self._sync()
        v = self.volume_l()
        return self.co2_l / v if v > 0 else 0.0

    # ------------------------------------------------------ the chunks -----
    def take_heat(self):
        """The heat held back in collect mode (joules); clears it. A jump gives it to the net as a source."""
        h = self.pend["heat"]
        self.pend["heat"] = 0.0
        self.heat_J += h
        return h

    def _sync(self):
        """Land any held gases before a reader looks (mode "now" only; a jump flushes at its chunk edge)."""
        p = self.pend
        if self.mode == "now" and (p["vap"] or p["o2"] or p["co2"] or p["lat"]):
            self.flush()

    def flush(self):
        """Apply the held gases to the room. Returns (vapour g, o2 l, co2 l)."""
        p = self.pend
        out = (p["vap"], p["o2"], p["co2"])
        self.latent_J += p["lat"]
        self._gases(p["vap"], p["o2"], p["co2"])
        p["vap"] = p["o2"] = p["co2"] = p["lat"] = 0.0
        self._flushed = int(self.w.tick)
        return out

    # ------------------------------------------------------- the books -----
    def _heat(self, heat_J):
        """Heat into the room's air by capacity: E + J, the lattice total moves by exactly J."""
        if not heat_J:
            return
        m = self.air()
        if not m.any():
            self.w.shed += heat_J       # no air to warm: the heat left, as the open world books it
            return
        C = self._air_C
        E = self.w.E[self.box]
        E[m] = (np.asarray(E, dtype=np.float64)[m] + heat_J * C / C.sum()).astype(E.dtype)
        self.heat_J += heat_J

    def _gases(self, vapour_g, o2_l, co2_l):
        if o2_l:                        # signed relative to the AIR: negative = taken from it
            self.o2_l += -o2_l
            m = self.air()
            o2 = self.w.o2[self.box]
            have = np.asarray(o2, dtype=np.float64)[m]
            tot = float(have.sum())
            if tot > 0.0:
                grams = max(o2_l * O2_G_PER_L, -tot)         # signed; the room cannot give more than it has
                o2[m] = (have * (1.0 + grams / tot)).astype(o2.dtype)
        if co2_l:
            self.co2_l += co2_l
        if vapour_g:
            self.vapour_g += vapour_g
            T = self.mean_temp()
            V_m3 = self.volume_l() / 1000.0
            try:
                cap_g = psychro.rho_sat(T) * V_m3
            except Exception:
                cap_g = float("inf")
            if self.vapour_g > cap_g:
                over = self.vapour_g - cap_g
                self.condensed_g += over
                self.vapour_g = cap_g
                back = min(self.latent_J, over * float(psychro.h_fg(T)))
                self.latent_J -= back           # it condenses: the latent heat comes back to the air
                self._heat(back)

    def books(self):
        """Plain numbers for a ledger test."""
        self._sync()
        return {"heat_J": self.heat_J, "latent_J": self.latent_J, "vapour_g": self.vapour_g, "condensed_g": self.condensed_g,
                "o2_l": self.o2_l, "co2_l": self.co2_l}
