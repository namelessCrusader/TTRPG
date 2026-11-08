"""Which of the 35 p["events"].append sites does a golden case set reach?
(scratch only). Every person's `events` list is swapped for a list that
classifies each appended line to its site by its text, so the count is of
TODAY's code: it says which emit sites an M1 golden would leave unexercised.

    flock <lock> python3 site_cover.py
"""
import collections
import contextlib
import io
import re
import sys
import tempfile

REPO = "/home/nisargparikh/Desktop/Fun Stuff/Mark_1"
sys.path.insert(0, REPO)
from src.voxel import sim as S, harvest, scenes          # noqa: E402

N = r"[^:]+?"
SITES = [
    (3338, rf"^{N} lets fly$"), (3665, rf"^{N} rolls with it$"),
    (3726, rf"^{N} lands hard$"), (5262, rf"^{N} is gone$"),
    (5310, rf"^{N} slumps — "), (5322, rf"^{N} comes round$"),
    (5328, rf"^{N} stops breathing$"), (5572, rf"^{N} wakes \(\d+ dB\)$"),
    (5636, rf'^{N} shouts: "'),
    (7092, rf"^{N} loses hold of the \w+ — an arm is only so long$"),
    (7483, rf"^{N} (throws the .+|swings an arm)$"),
    (7491, rf"^{N} takes hold of "), (7501, rf"^{N} takes up the "),
    (7506, rf"^{N} reaches out$"), (7510, rf"^{N} lowers the arm$"),
    (7525, rf"^{N} (catches (?!the ).+|grabs at .+ and cannot hold them)$"),
    (7542, rf"^{N} (catches the .+|cannot hold what is coming, and it goes past)$"),
    (7548, rf"^{N} lets go$"), (7553, rf"^{N} jumps$"),
    (7564, rf"^{N} startles \(.+\) and makes for "),
    (7568, rf"^{N} makes for "), (7573, rf"^{N} notices \(.+\) and stays put$"),
    (7581, rf"^{N} looks "), (7593, rf"^{N} leans out$"),
    (7598, rf"^{N} straightens up$"), (7819, rf"^{N} tries another way$"),
    (7831, rf"^{N} gives up on getting there$"),
    (7838, rf"^{N} reaches the doorway$"), (7851, r"^\.\.\.hauling .+ clear$"),
    (8019, rf"^{N} tucks to roll$"),
    ("8122|8171", rf"^{N} loses hold of (?!the ).+ — an arm is only so long$"),
    (8186, rf"^{N} is left holding .+ over the drop$"),
    (8190, r"^.+ goes over, and the grip is not enough to hold a hanging man$"),
    (8534, rf"^{N} (loses their footing|is pulled off their feet by what they "
           rf"are holding|overbalances)$"),
]
SITES = [(s, re.compile(p)) for s, p in SITES]
COV = collections.Counter()
UNMATCHED = collections.Counter()
BY_CASE = collections.defaultdict(set)
CASE = ["?"]


class Rec(list):
    def append(self, line):
        body = line.split(": ", 1)[1] if ": " in line else line
        for s, rx in SITES:
            if rx.search(body):
                COV[s] += 1
                BY_CASE[s].add(CASE[0])
                break
        else:
            UNMATCHED[body[:60]] += 1
        super().append(line)


_add = S.World.add_person


def add_person(self, *a, **k):
    p = _add(self, *a, **k)
    p["events"] = Rec(p["events"])
    return p


S.World.add_person = add_person


def run_harvest(seeds):
    for seed in seeds:
        CASE[0] = f"harvest_{seed}"
        w, people, _ = harvest.world(seed)
        w.policy = harvest.Roam(seed)
        for _ in range(90):
            w.step()
            for p in people:
                p["events"].clear()


def run_scene(name, ticks):
    CASE[0] = name
    saved = scenes._save
    scenes._save = lambda w, d, t: None
    try:
        with tempfile.TemporaryDirectory() as d, \
                contextlib.redirect_stdout(io.StringIO()):
            scenes.SCENARIOS[name](d, ticks=ticks, every=10 ** 9)
    finally:
        scenes._save = saved


def run_sound():
    from tests import test_sound as T
    CASE[0] = "test_sound"
    for name in dir(T):
        if name.startswith("test_"):
            getattr(T, name)()


if __name__ == "__main__":
    run_harvest(range(16))
    print("after harvest 0-15:", sorted(str(s) for s in COV), flush=True)
    run_harvest(range(16, 60))
    run_sound()
    for name, ticks in [("jumper", 80), ("swing", 60), ("arms", 100),
                        ("arrow_trap", 40), ("rescue", 100), ("ledge", 100),
                        ("leap", 60), ("parachute", 60), ("fire_alarm", 120),
                        ("duel", 200)]:
        run_scene(name, ticks)
        print(f"  {name} done", flush=True)
    allsites = [s for s, _ in SITES]
    print("\nsite         hits   first cases")
    for s in allsites:
        cases = sorted(BY_CASE[s])
        print(f"  {str(s):10s} {COV[s]:6d}   {', '.join(cases[:4])}"
              f"{' ...' if len(cases) > 4 else ''}")
    miss = [str(s) for s in allsites if COV[s] == 0]
    print(f"\nNEVER HIT ({len(miss)}): {miss}")
    print("unmatched lines:", UNMATCHED.most_common(8))
