"""The rooms backend: the lab inn as a world that the mind interface can drive
(docs/sim-mind-design.md 3.5, step M3).

    world     the inn: a fork of lab/world2/world.py. step() takes no policy.
    rules     the inn's rule scores (a fork of lab/world2/policy.py), and the
              lab's Rules and Random policies as choosers
    backend   RoomsBackend (src/mind/api.py, Backend), and night(): a night in
              compat mode, which gives the lab's events byte for byte

src/rooms uses src/mind and the standard library. The lab and src/voxel are
never imported here, and src/mind never imports src/rooms.
"""
from .backend import RoomsBackend, jsonl, night
from .world import World

__all__ = ["RoomsBackend", "World", "jsonl", "night"]
