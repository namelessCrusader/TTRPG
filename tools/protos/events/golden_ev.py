"""The M0-M8 gate for the events topic (scratch tool): `golden.py`, the
whole-world per-tick hasher from the sim.py split, with the event topic's NEW
state taken out of the hash. Everything else -- every array, every person
dict (p["events"] text included), w.speech, w.sounds (tick, at, db, who),
traces (seam-spelled text included), bodies, MPM particles -- must hash the
same before and after each step. Record the base on TODAY's tree (the new
fields do not exist yet, so the ignore lists are inert there).

    flock <lock> python3 golden_ev.py record base_ev.json        # before M0
    flock <lock> python3 golden_ev.py check  base_ev.json        # after each step
    flock <lock> python3 golden_ev.py check  base_ev.json --all  # + SLOW set

The new state is named here and nowhere else; a step that adds a field not on
these lists fails the gate, which is the point: nothing new may ride inside
an old structure without being declared.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import golden                                         # noqa: E402  (re-execs
                                                      # under PYTHONHASHSEED=0)
# World attributes the topic adds (events.py, perceive.py, beliefs.py).
NEW_WORLD = {"events", "minds", "_cause", "_broke", "_fires", "_disturbed",
             "_epoch", "seed", "event_sink", "_burning_now", "_cover",
             "_said_ev"}
# Keys the topic adds to EXISTING dicts: person (pid), sound / body (ev_id,
# by_pid, pid), trace row (mind). Checked unused in src/voxel at 54fcc60.
NEW_KEYS = {"pid", "ev_id", "by_pid", "mind"}
# Attributes the MPM hook adds to Matter.
NEW_MATTER = {"cluster", "cause_of", "_dmg_prev", "_rest"}

golden.SKIP |= NEW_WORLD
_orig = golden._enc


def _enc(obj, h, world, stack):
    if isinstance(obj, dict) and not NEW_KEYS.isdisjoint(obj.keys()):
        obj = {k: v for k, v in obj.items() if k not in NEW_KEYS}
    elif type(obj).__name__ == "Matter" and hasattr(obj, "__dict__") \
            and id(obj) not in stack:
        # golden puts the OBJECT on the cycle stack before its vars; do the
        # same, or a restored world's ghost `matter.w` cycle is cut elsewhere
        h.update(b"o:Matter")
        stack.add(id(obj))
        try:
            return _orig({k: v for k, v in vars(obj).items()
                          if k not in NEW_MATTER}, h, world, stack)
        finally:
            stack.discard(id(obj))
    return _orig(obj, h, world, stack)


golden._enc = _enc                     # golden's own recursion looks the name
                                       # up in its module globals, so this
if __name__ == "__main__":             # wraps every level, not just the top
    golden.main(sys.argv[1:])
