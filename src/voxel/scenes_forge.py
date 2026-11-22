"""The forge (wave 14a, lane C): a charcoal bed with an iron bar buried in it, and a bellows.

There is NO bellows code here. The bellows is lane D's board in a tube (scenes_push.Bellows):
a moving surface that pushes the gas it sweeps, by the one push law (push.py) and the gas
flow (gasflow.py). All this scene adds is the bed beyond the nozzle, and one reading: each
tick the volume of air that the flow solver carried out through the nozzle is handed to the
burning bed as its blast (`World.blast`), which the burn law turns into extra oxygen on the
faces that burn (heat.py `_blast_boost`). So the blast is the MEASURED flow of the board's
push, not a number typed in. A board that stops, a nozzle that is blocked or a bed with no
way for the air through it give no blast.

    forge(bellows=True)   a Forge; `step(n)` moves the board, ticks the world, feeds the blast
"""
import numpy as np

from .materials import AMBIENT, CHAR, IRON, STONE
from .scenes_push import Bellows


class Forge(Bellows):
    """A bellows tube (scenes_push.Bellows), and a bed of charcoal (80 percent full, at
    700 C, a bed already glowing) 8 x 8 x 6 voxels (40 x 40 x 30 cm) in front of the nozzle, with a
    thin rod of iron buried in it. The bed is kindled hot, as in the wave-12 forge test."""

    def __init__(self, bed_c=700.0, speed=0.04, stroke=0.8, bellows=True, **kw):
        super().__init__(speed=speed, stroke=stroke, **kw)
        w = self.w
        x0 = self.length + self.T
        y0, z0 = self.cy - 4, self.cz - 3
        w.fill(x0, x0 + 8, y0, y0 + 8, 1, z0, STONE)        # a hearth under the fire pot
        w.fill(x0, x0 + 8, y0, y0 + 8, z0, z0 + 6, CHAR, frac=0.8)   # 40 x 40 x 30 cm
        self.bed = w.mat == CHAR
        w.E[self.bed] = (bed_c - AMBIENT) * w.heat_capacity()[self.bed]
        bx, by, bz = x0 + 2, self.cy - 1, self.cz - 1
        w.fill(bx, bx + 4, by, by + 1, bz, bz + 1, IRON)      # a 20 cm rod, 5 cm thick
        self.bar = w.mat == IRON
        self.bar[:self.length + self.T] = False
        self.mouth = (x0, self.cy, self.cz)
        self.use_bellows = bellows
        self.blast_lps = 0.0
        self.peak = AMBIENT

    def step(self, n=1):
        for _ in range(n):
            self.move()
            if self.use_bellows:
                q = self.nozzle_flow() if "_gf" in self.w.__dict__ else 0.0                       # m^3/s, out through the nozzle
                self.blast_lps = max(q, 0.0) * 1000.0
                if self.blast_lps > 0.0:
                    self.w.blast(*self.mouth, self.blast_lps)
            self.w.step()
            self.t += 1
            if self.t % 10 == 0:
                char = self.w.mat == CHAR
                if char.any():
                    self.peak = max(self.peak, float(self.w.T()[char].max()))

    def bar_temp(self):
        T = self.w.T()
        return float(T[self.bar & (self.w.mat == IRON)].mean())


def forge(**kw):
    return Forge(**kw)
