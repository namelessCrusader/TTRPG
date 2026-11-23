"""The memo state of the lazy laws (wave 15, S4-3), kept OUTSIDE the world.

It lives in a weak dictionary beside the world, like `fields._KEPT`, so it is
no part of the state a world is compared by (golden) or saved by (snapshot):
a restored world starts a tick with the memo off, and `_tick_start` says so.

`mm` is the matter memo, in three steps:
    0  off: a writer of matter is running, or no tick is open
    1  armed: nothing has written matter since the last writer ended, but
       nobody has compared the matter with the kept copy since then
    2  verified: the kept copy IS the matter now (a real comparison said so
       and no writer has run since); `_matter_moved` may answer "no" at once
`wdepth` counts the writer laws that are open (they nest through `super()`).
`hot` is the summary of the heat (lazylaws): kept until a law that can write E or
matter has ended, dropped by every other.
"""
import weakref

_LZ = weakref.WeakKeyDictionary()


class Memo:
    __slots__ = ("live", "mm", "wdepth", "cpor", "hot", "skipbump")

    def __init__(self):
        self.live = False          # a tick is open and past the point where the particles landed
        self.mm = 0
        self.wdepth = 0
        self.cpor = None           # (porosity array, its coarse mean), kept by identity
        self.hot = None            # what the cells not at rest say (lazylaws._Hot), until a law that can change them runs
        self.skipbump = False      # a law that has just proved it changed nothing says so, once


def get(world):
    """The memo of a world, or None if it never had one."""
    return _LZ.get(world)


def state(world):
    st = _LZ.get(world)
    if st is None:
        st = _LZ[world] = Memo()
    return st
