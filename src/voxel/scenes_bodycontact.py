"""Scenes for body-on-body contact and the Hof balance mechanisms (wave 14b, E).

`room` is a stone room, `stand` puts a person in it, `two_people` puts two face
to face along x. Everything else is the world's own laws: a shove or tackle is
`World.body_collide` and a lean is `push_person`."""
from .scenes import _person, _Wants
from .sim import AIR, STONE, World

FLOOR = 4          # the stone is 4 cells thick: a person stands on z = 4


def room(nx=40, ny=30, nz=40, gate=True):
    w = World(nx, ny, nz, voxel_cm=5)
    w.open_sky = False
    w.fill(0, nx, 0, ny, 0, nz, STONE)
    w.mat[1:nx - 1, 1:ny - 1, FLOOR:nz - 1] = AIR
    w.smass[1:nx - 1, 1:ny - 1, FLOOR:nz - 1] = 0.0
    w.exits = []
    w.body_contact = gate
    return w


def stand(w, px, py, name):
    p = _person(w, px, py, z0=FLOOR)
    p["name"] = name
    return p


def two_people(w, gap=4, x0=14, y0=15):
    """A at x0, B `gap` cells further along x; neither wants anything."""
    a = stand(w, x0, y0, "the pusher")
    b = stand(w, x0 + 7 + gap, y0, "the pushed")
    w.policy = _Wants(each={"the pusher": {"hands": "hands free", "legs": "stay"},
                            "the pushed": {"hands": "hands free", "legs": "stay"}})
    return a, b
