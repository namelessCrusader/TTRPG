"""NPC jobs (DwarfCorp port): resumable routines, interruption-by-drives,
failure blacklists with grumbling. Staff should LIVE in the guildhall, not idle.
"""

from src.core import acts, engine, mind, reactions
from src.core.seed import guildhall
from src.core.state import Cell


def _run(world, n):
    for _ in range(n):
        reactions.tick(world)
        mind.take_turns(world)


# DESIGN (DwarfCorp): a guard on duty PATROLS his posts, not a statue.
def test_guard_patrols_between_posts():
    world, _ = guildhall(seed=0)
    bran = world.entities["guard"]
    posts = bran.mind["job"]["posts"]
    visited = set()
    for _ in range(60):
        _run(world, 1)
        for i, p in enumerate(posts):
            if mind._l1(bran.pos, p) <= 1:
                visited.add(i)
    assert len(visited) >= 2, f"a patrol visits its posts: saw {visited} of {posts}"


# DESIGN: drives preempt the job; the job RESUMES when the drive passes.
# (The priority ladder already interrupts — resumability is the ported part.)
def test_job_resumes_after_interruption():
    world, _ = guildhall(seed=0)
    bran = world.entities["guard"]
    _run(world, 6)
    world.cells[bran.pos].tags.add("on_fire")           # danger at his feet — he flees
    _run(world, 3)
    world.cells = {p: c for p, c in world.cells.items()}
    for c in world.cells.values():
        c.tags.discard("on_fire")                       # danger passes
    posts = bran.mind["job"]["posts"]
    reached = False
    for _ in range(40):
        _run(world, 1)
        if any(mind._l1(bran.pos, p) <= 1 for p in posts):
            reached = True
            break
    assert reached, "the patrol resumes after the scare"


# DESIGN (DwarfCorp blacklist): an unreachable post gets set aside with a
# grumble — no retry-thrash — and retried only after the TTL.
def test_unreachable_post_is_blacklisted_with_a_grumble():
    world, _ = guildhall(seed=0)
    bran = world.entities["guard"]
    far_post = bran.mind["job"]["posts"][-1]
    x, y, z = far_post
    for q in [(x + dx, y + dy, z) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]:
        if world.in_bounds(q):
            world.cells[q] = Cell(material="stone")     # wall the post off completely
    _run(world, 30)
    assert any("grumbles" in e.cause for e in world.log if e.cause), \
        "failure is VOICED — it feeds the social layer"
    bl = bran.mind.get("blacklist", {})
    assert any(tuple(k) == tuple(far_post) for k in bl), f"set aside, not thrashed: {bl}"
