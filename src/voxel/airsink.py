"""Where a body's vapour, heat, oxygen and carbon dioxide go (wave 13, T7).

A person breathes, sweats and warms the air around them. The tick path and
the night-skip path must give the room the SAME heat, water and gas, so both
call one small protocol, `AirSink`. This file has the protocol and the
open-world default; a sealed room, the GAS vapour field and a region sink
(phase B) provide their own and keep the mass.

Sign convention (the one heat.py uses for igniters): an intake or injection
does `shed -= J`, so E + shed + pool + stores + work_out is constant except
for named injections. Here `heat_J` is the energy delivered to the sink
(latent heat leaving with vapour, sensible heat into the air): positive =
leaves the body. `o2_l` is signed relative to the AIR (-x = taken from it),
`co2_l` likewise (+y = added to it).
"""
from typing import Optional, Protocol, Tuple, runtime_checkable

DEFAULT_T_C = 20.0                       # the lattice's reference air temperature
DEFAULT_RH = 0.5                         # world scalar until a vapour field exists (u, from the weather inputs)
DEFAULT_O2 = 0.209


@runtime_checkable
class AirSink(Protocol):
    def add(self, vapour_g: float = 0.0, heat_J: float = 0.0, o2_l: float = 0.0,
            co2_l: float = 0.0, where=None) -> None:
        """Deliver vapour (g), heat (J), O2 (L, negative = consumed) and CO2 (L) at `where`."""

    def ambient(self, where=None) -> Tuple[float, float]:
        """(air temperature C, relative humidity 0..1) at `where`: what a breath draws in."""

    def o2_fraction(self, where=None) -> float:
        """Oxygen fraction of the air at `where`."""


class OpenWorldSink:
    """The open world: the air is endless. Vapour is booked to `gone["vapour"]`
    (grams) and heat to `shed` (joules, positive = left the lattice); O2 and
    CO2 are counted in `gone` (litres) so the totals show, but change no air.

    `gone` may be the world's own dict; `shed` is read and written as an
    attribute, and a caller that owns the real counter copies it back (phase B)."""

    def __init__(self, gone: Optional[dict] = None, t_c: float = DEFAULT_T_C,
                 rh: float = DEFAULT_RH, o2: float = DEFAULT_O2):
        self.gone = {} if gone is None else gone
        self.shed = 0.0
        self.t_c, self.rh, self.o2 = t_c, rh, o2

    def add(self, vapour_g=0.0, heat_J=0.0, o2_l=0.0, co2_l=0.0, where=None):
        if vapour_g:
            self.gone["vapour"] = self.gone.get("vapour", 0.0) + vapour_g
        if o2_l:
            self.gone["o2_l"] = self.gone.get("o2_l", 0.0) + o2_l
        if co2_l:
            self.gone["co2_l"] = self.gone.get("co2_l", 0.0) + co2_l
        self.shed += heat_J

    def ambient(self, where=None):
        return self.t_c, self.rh

    def o2_fraction(self, where=None):
        return self.o2


class SinkLog(OpenWorldSink):
    """An open-world sink that also keeps every call, for tests and parity checks."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def add(self, vapour_g=0.0, heat_J=0.0, o2_l=0.0, co2_l=0.0, where=None):
        self.calls.append((vapour_g, heat_J, o2_l, co2_l, where))
        super().add(vapour_g, heat_J, o2_l, co2_l, where)

    def totals(self):
        return {"vapour_g": sum(c[0] for c in self.calls), "heat_J": sum(c[1] for c in self.calls),
                "o2_l": sum(c[2] for c in self.calls), "co2_l": sum(c[3] for c in self.calls)}


def deliver(sink, vapour_g=0.0, latent_J=0.0, sensible_J=0.0, o2_l=0.0, co2_l=0.0, where=None):
    """One call for what a second of a body gives the air. Latent heat leaves
    with the vapour, sensible heat warms the air; the sink gets their sum."""
    sink.add(vapour_g=vapour_g, heat_J=latent_J + sensible_J, o2_l=o2_l, co2_l=co2_l, where=where)
