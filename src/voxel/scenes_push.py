"""Proof scenes of the one push law (wave 14a, lane D). Each returns a World with the
gas flow and momentum on and a driver whose `step()` moves the surfaces and ticks the
world. There is NO bellows law and NO fan law: a driver only says where a surface is and
how fast it moves (kinematics); the gas does the rest by push.py and gasflow.py.

    bellows(...)   a board in a tube, one end of the tube a small nozzle
    fan(...)       pitched blades spinning in open air
    wind_box(...)  an open-sky box with the wind blowing through it, an optional block in it
"""
import numpy as np

from .materials import AIR, IRON, STONE, TICK_S
from .push import Surface, patch_grid
from .sim import World

DX = 0.05                                   # m, the voxel of these scenes (5 cm)


def _air(w, x0, x1, y0, y1, z0, z1):
    """Make a box of the lattice empty air (the mass of what stood there goes too)."""
    w.mat[x0:x1, y0:y1, z0:z1] = AIR
    w.smass[x0:x1, y0:y1, z0:z1] = 0


class _Driver:
    def __init__(self, w):
        self.w = w
        self.t = 0

    def move(self):
        """Move the surfaces for this tick (kinematics only)."""

    def step(self, n=1):
        for _ in range(n):
            self.move()
            self.w.step()
            self.t += 1


class Bellows(_Driver):
    """A tube along x, 12 x 12 voxels (3 x 3 cells of 20 cm) inside, closed by an iron board
    that slides along it, and a nozzle (a hole `nozzle` voxels square) in the wall at the far
    end. The back of the tube is open to the world's edge. The board is solid voxels (it shuts
    the gas) and carries one surface element per voxel face of its front (it pushes the gas).
    The walls are a whole cell thick: the flow sees a wall only where it fills a cell face."""

    def __init__(self, length=24, nozzle=4, speed=0.2, stroke=0.6, momentum=True, free=40):
        T, ny, nz = 4, 12, 12
        nx = length + T + free
        w = World(nx, ny + 2 * T, nz + 2 * T, voxel_cm=5)
        w.fill(0, length + T, 0, ny + 2 * T, 0, nz + 2 * T, STONE)               # the block...
        _air(w, 0, length, T, T + ny, T, T + nz)                                  # ...hollowed
        cy, cz = T + ny // 2, T + nz // 2
        _air(w, length, length + T, cy - nozzle // 2, cy - nozzle // 2 + nozzle,
             cz - nozzle // 2, cz - nozzle // 2 + nozzle)                        # the nozzle
        w.fill(0, nx, 0, ny + 2 * T, 0, 1, STONE)
        w.gas_flow = True
        w.gas_momentum = momentum
        super().__init__(w)
        self.T, self.ny, self.nz, self.length, self.cy, self.cz, self.nozzle = T, ny, nz, length, cy, cz, nozzle
        self.wall = length
        self.speed, self.stroke = speed, stroke
        self.vox = 4                                      # board's present voxel layer
        self.xm = (self.vox + 0.5) * DX                   # board's true position (m)
        self.xstart = self.xm
        self.dir = 1.0
        self.moving = False
        self._set_board(True)
        self.surf = patch_grid([self.xm, T * DX, T * DX],
                               [self.xm, (T + ny) * DX, (T + nz) * DX], 0, DX,
                               (0.0, 0.0, 0.0), sign=1.0)
        w.add_surface(self.surf)

    def _set_board(self, on):
        if on:
            self.w.fill(self.vox, self.vox + 1, self.T, self.T + self.ny, self.T, self.T + self.nz, IRON)
        else:
            _air(self.w, self.vox, self.vox + 1, self.T, self.T + self.ny, self.T, self.T + self.nz)

    def start(self, direction=1.0):
        self.dir = direction
        self.moving = True
        self.surf.vel[:] = (self.speed * direction, 0.0, 0.0)
        self.surf.normal[:] = (direction, 0.0, 0.0)

    def stop(self):
        self.moving = False
        self.surf.vel[:] = 0.0

    def move(self):
        if not self.moving:
            return
        d = self.speed * self.dir * TICK_S
        self.xm += d
        self.surf.pos[:, 0] += d
        new = int(np.floor(self.xm / DX))
        if new != self.vox:
            self._set_board(False)
            self.vox = new
            self._set_board(True)
        if abs(self.xm - self.xstart) >= self.stroke:
            self.stop()

    def nozzle_flow(self):
        """The volume flow (m^3/s) out through the end wall, from the solver's face speeds:
        the sum over the face between the last cell of the tube and the wall's cell."""
        H = 4 * DX
        k = self.length // 4
        return float((self.w._gf["u"][0][k] * self.w._gf_geometry()["F"][0][k]).sum()) * H * H


def bellows(**kw):
    return Bellows(**kw)


class Fan(_Driver):
    """`blades` flat blades about an axis along x, in open air, each `r_in` to `r_out` metres
    from the axis, tilted `pitch` radians from the rotation plane, turning at `omega` rad/s.
    A blade's patches carry the blade's normal and the speed omega x r. That is all."""

    def __init__(self, blades=4, r_in=0.1, r_out=0.5, chord=0.1, pitch=0.6, omega=40.0,
                 momentum=True, size=(60, 48, 48)):
        nx, ny, nz = size
        w = World(nx, ny, nz, voxel_cm=5)
        w.fill(0, nx, 0, ny, 0, 1, STONE)
        w.gas_flow = True
        w.gas_momentum = momentum
        super().__init__(w)
        self.c = np.array([0.8, ny * DX / 2, nz * DX / 2])
        self.omega, self.pitch = omega, pitch
        self.angle = 0.0
        rs = np.arange(r_in + 0.5 * DX, r_out, DX)
        nch = max(int(round(chord / DX)), 1)
        self.r = np.repeat(rs, nch)                      # radius of each patch
        self.base = np.tile(np.arange(blades) * 2 * np.pi / blades, 1)
        ang0 = np.repeat(self.base, len(self.r))
        self.rr = np.tile(self.r, blades)
        self.ang0 = ang0
        self.surf = Surface(np.zeros((len(self.rr), 3)), DX * DX, (1, 0, 0), np.zeros(3))
        w.add_surface(self.surf)
        self.thrust_hist = []
        self._place()

    def _place(self):
        a = self.ang0 + self.angle
        e_r = np.stack([np.zeros_like(a), np.cos(a), np.sin(a)], axis=1)       # radial
        e_t = np.stack([np.zeros_like(a), -np.sin(a), np.cos(a)], axis=1)      # along the motion
        self.surf.pos[:] = self.c + self.rr[:, None] * e_r
        self.surf.vel[:] = (self.omega * self.rr)[:, None] * e_t
        # the plate's normal: along the axis, tipped back by the pitch toward the motion
        n = np.cos(self.pitch) * np.array([1.0, 0, 0]) + np.sin(self.pitch) * -e_t
        self.surf.normal[:] = n / np.linalg.norm(n, axis=1, keepdims=True)

    def move(self):
        self.angle += self.omega * TICK_S
        self._place()

    def thrust(self):
        """Axial force of the gas on the blades, minus is toward +x (N): the reaction."""
        return float(self.surf.force[:, 0].sum())


def fan(**kw):
    return Fan(**kw)


def wind_box(wind=(3.0, 0.0), n=(48, 32, 32), block=None, alpha=0.0, momentum=True):
    """An open-sky box of n voxels with the wind blowing through it, and (optionally) a stone
    block `block` = (x0, x1, y0, y1, z1) standing on the floor."""
    w = World(*n, voxel_cm=5)
    w.fill(0, n[0], 0, n[1], 0, 1, STONE)
    if block:
        x0, x1, y0, y1, z1 = block
        w.fill(x0, x1, y0, y1, 1, z1, STONE)
    w.gas_flow = True
    w.gas_momentum = momentum
    w.wind = wind
    w.wind_alpha = alpha
    return _Driver(w)
