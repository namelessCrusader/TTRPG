"""NPC jobs — the DwarfCorp port, made pickle-safe.

Their Act trees are coroutines; ours are RESUMABLE STATE DICTS stepped once per
tick (`npc.mind["job"]`). Same properties, no generators to break saves:
- interruptible: the drive ladder in mind.think() preempts; state waits.
- resumable: each tick recomputes one step from state — nothing dangling.
- failure-aware: unreachable targets go on a TTL blacklist WITH a voiced grumble
  (DwarfCorp kept failure reasons; ours feed the social layer).

A job is data: {"kind": "patrol", "posts": [...], "i": 0, "linger": 0}.
Content grows by adding kinds; the stepper's shape never does.
"""

from __future__ import annotations

from . import effects as fx
from .spatial import bfs_path, l1
from .state import World

BLACKLIST_TTL = 60
LINGER = 4


def step_job(world: World, npc) -> list | None:
    """One tick of the npc's job. None = no job / nothing to do this tick."""
    job = npc.mind.get("job")
    if not job:
        return None
    if job.get("kind") == "patrol":
        return _patrol(world, npc, job)
    return None


def _patrol(world: World, npc, job) -> list:
    posts = [tuple(p) for p in job["posts"]]
    bl = npc.mind.setdefault("blacklist", {})
    for k in [k for k, until in bl.items() if world.tick >= until]:
        del bl[k]                                       # TTL expired — worth another try

    live = [p for p in posts if p not in bl]
    if not live:
        return []                                       # everything soured; wait out the TTL
    target = posts[job.get("i", 0) % len(posts)]
    if target in bl:
        job["i"] = (job.get("i", 0) + 1) % len(posts)
        return []

    if l1(npc.pos, target) <= 1:                        # at the post: linger, then move on
        job["linger"] = job.get("linger", 0) + 1
        if job["linger"] >= LINGER:
            job["linger"] = 0
            job["i"] = (job.get("i", 0) + 1) % len(posts)
        return []

    path = bfs_path(world, npc.pos, target)
    if path is None:                                    # unreachable: set aside, SAY so
        bl[target] = world.tick + BLACKLIST_TTL
        job["i"] = (job.get("i", 0) + 1) % len(posts)
        return [fx.speech(npc.id, "...",
                          f"{npc.name} grumbles about the blocked way to {target}")]
    if not path:
        return []
    eff = fx.move(npc.id, path[0])
    if npc.mind.get("narrated") != f"patrol:{target}":
        npc.mind["narrated"] = f"patrol:{target}"
        eff.cause = f"{npc.name} makes rounds toward {target}"
    return [eff]
