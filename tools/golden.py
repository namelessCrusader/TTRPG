"""GOLDEN STATE: is the world after a change the SAME world, bit for bit, tick
for tick?

The test suite asks whether outcomes still come out right. A change that only
moves code (the sim.py split) or only adds structure (an entity id that nothing
reads yet) has a much stronger thing to promise: that NOTHING changed. So this
steps a set of small worlds and hashes the whole of each one -- every numpy
array, every person dict (events included), traces, speech, sounds, drops, free
bodies, the MPM particles behind `matter` -- after every tick, field by field.
Recorded on the base tree, checked on the changed one: the first
(case, tick, field) to differ says exactly which change broke what.

    python3 golden.py record  base.json --all       # on the base tree
    python3 golden.py check   base.json             # the quick cases
    python3 golden.py check   base.json --all       # quick + slow
    python3 golden.py check   base.json --only pond,duel150
    python3 golden.py check   base.json --all --except fire_alarm,duel
    python3 golden.py check   base.json --all --allow rewind   # may differ
    python3 golden.py fresh   base.json             # recorded from THIS tree?
    python3 golden.py coverage [--all | --only CASES]
    python3 golden.py list

SANDBOXES. The repo it imports is $MARK1_REPO (default: the repo this file is
in), and it refuses to run if `src.voxel.sim` came from anywhere else. So one
golden.py can record a base copy and check a step's copy (tools/stepcheck.sh).

NEW NAMES. A recording keeps every name it met, as a PATH from the World: a
World attribute is its own name (`eid`); a string dict key or an attribute of
an object adds `.key` to the path of what holds it (`persons[].pid`,
`matter.cluster`); a list, a tuple, an array, a set or a dict with keys that
are not strings adds `[]` (`traces[].mind`). A check leaves out a path the
base never had and lists it as `NEW:` -- so a step that only ADDS a field
(`w.eid`, `p["intent"]`) still checks SAME on everything else. A path and not
a bare name, so that the new World attribute `events` is not mistaken for the
person's old `persons[].events`. The check cannot tell a new field from a new
data key (a new person name used as a key), so every NEW line must be read
and explained.

It re-execs itself under PYTHONHASHSEED=0 so that any set-of-strings iteration
inside the sim is the same in the recording run and the checking run. Wrap
every run in the shared lock:

    flock "$LOCK" python3 tools/golden.py check base.json --all

The hasher never trusts object identity (CPython merges equal constants per
module, so a tuple shared between two functions may stop being shared once the
functions live in different files): containers are encoded BY VALUE, and the
World itself (reached again through `matter.w`) is a fixed token.
"""
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import time
import contextlib
import types

_SEED = os.environ.get("GOLDEN_HASHSEED", "0")    # override only to PROBE
if os.environ.get("PYTHONHASHSEED") != _SEED:      # the sim's own seed-
    os.environ["PYTHONHASHSEED"] = _SEED           # sensitivity
    os.execv(sys.executable, [sys.executable] + sys.argv)

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)

import numpy as np                                  # noqa: E402

from src.voxel import sim, scenes, harvest          # noqa: E402

# A sandbox that silently imported some other tree would check nothing.
if not os.path.realpath(sim.__file__).startswith(REPO + os.sep):
    sys.exit(f"golden.py: imported {sim.__file__}, not the repo {REPO} "
             f"(MARK1_REPO): refusing to hash the wrong tree")

# ── the canonical encoder ────────────────────────────────────────────────────

# Names, each a path from the World (see NEW NAMES above). NAMES collects
# every one met; KNOWN, when set, is the base recording's NAMES -- a path
# outside it is left out of the hash and collected in NEW. One namespace per
# path: a name that is old in one place is still new in another.
NAMES, NEW = set(), set()
KNOWN = None


# Names of attributes that hold a cache key built from id(): the address of an array
# differs from run to run, the world does not. wetworld.WetState._cap_key (wave 13c C3).
CACHE_TOKENS = {"_cap_key"}


def _keep(path):
    NAMES.add(path)
    if KNOWN is not None and path not in KNOWN:
        NEW.add(path)
        return False
    return True


def _slots(obj):
    """{slot: value} of a slotted object (a dataclass with slots=True, a
    Mind): repr() would hash its address."""
    out = {}
    for c in type(obj).__mro__:
        sl = c.__dict__.get("__slots__", ())
        for s in ((sl,) if isinstance(sl, str) else sl):
            if s not in ("__dict__", "__weakref__") and hasattr(obj, s):
                out.setdefault(s, getattr(obj, s))
    return out


def _enc(obj, h, world, stack, at):
    """Feed a canonical, identity-free encoding of `obj` into hash `h`. `at`
    is the path of `obj` (NEW NAMES); it names keys and never goes into `h`."""
    if obj is world:
        h.update(b"<world>")
        return
    if obj is None or isinstance(obj, (bool, int, str, bytes)):
        h.update(type(obj).__name__.encode() + b":" + repr(obj).encode() + b";")
        return
    if isinstance(obj, float):
        h.update(b"f:" + struct.pack("<d", obj))     # exact, -0.0 != 0.0
        return
    if isinstance(obj, np.ndarray):
        h.update(b"a:" + obj.dtype.str.encode() + repr(obj.shape).encode())
        if obj.dtype == object:
            for v in obj.ravel():
                _enc(v, h, world, stack, at + "[]")
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
        return
    if isinstance(obj, np.generic):
        h.update(b"n:" + obj.dtype.str.encode() + obj.tobytes())
        return
    if id(obj) in stack:                             # a real cycle
        h.update(b"<cycle>")
        return
    stack.add(id(obj))
    try:
        if isinstance(obj, dict):
            kv = []
            for k, v in obj.items():
                if isinstance(k, str):
                    if k in CACHE_TOKENS:            # an identity token, not state
                        continue
                    sub = at + "." + k
                    if not _keep(sub):
                        continue
                else:
                    sub = at + "[]"                  # a data key: (x, y), mat
                kv.append((k, v, sub))
            h.update(b"d%d:" % len(kv))
            for k, v, sub in kv:                     # insertion order IS state
                _enc(k, h, world, stack, sub)
                _enc(v, h, world, stack, sub)
        elif isinstance(obj, (list, tuple)):
            h.update((b"l%d:" if isinstance(obj, list) else b"t%d:") % len(obj))
            sub = at + "[]"
            for v in obj:
                _enc(v, h, world, stack, sub)
        elif isinstance(obj, (set, frozenset)):      # sorted: hash-seed proof
            parts = sorted(_digest(v, world, at + "[]") for v in obj)
            h.update(b"s%d:" % len(parts) + "".join(parts).encode())
        elif isinstance(obj, np.random.Generator):
            _enc(obj.bit_generator.state, h, world, stack, at)
        elif isinstance(obj, (types.FunctionType, types.MethodType,
                              types.BuiltinFunctionType, type)):
            h.update(b"c:" + getattr(obj, "__qualname__", "?").encode())
        elif hasattr(obj, "__dict__"):
            # class QUALNAME only: TablePolicy may change module in the split
            h.update(b"o:" + type(obj).__qualname__.encode())
            _enc(vars(obj), h, world, stack, at)     # its attributes: at.attr
        elif hasattr(type(obj), "__slots__"):
            h.update(b"o:" + type(obj).__qualname__.encode())
            _enc(_slots(obj), h, world, stack, at)
        else:
            h.update(b"r:" + repr(obj).encode())
    finally:
        stack.discard(id(obj))


def _digest(obj, world=None, at=""):
    h = hashlib.blake2b(digest_size=12)
    _enc(obj, h, world, set(), at)
    return h.hexdigest()


# Fields whose ORDER of keys in __dict__ is not state but whose value is.
# `policy` is the mind, which snapshot() leaves out too; its own state (a
# seeded RNG) shows up in the world the moment it matters.
SKIP = {"policy"}


def world_digest(w):
    """{field: digest} over everything the world is, plus the field list."""
    keys = [k for k in sorted(vars(w)) if k not in SKIP and _keep(k)]
    out = {k: _digest(getattr(w, k), w, k) for k in keys}
    out["__fields__"] = _digest(keys)
    return out


# ── a World that reports itself after every tick ─────────────────────────────

class _Recorder:
    def __init__(self, every=1):
        self.every = every
        self.rows = []                  # (world serial, tick, {field: digest})
        self.worlds = {}

    def note(self, w):
        n = self.worlds.setdefault(id(w), len(self.worlds))
        if w.tick % self.every == 0:
            self.rows.append((n, int(w.tick), world_digest(w)))

    def render(self, w, t):
        """What a scene's frame dump reads (bodies_array, burning, T): the
        render-export path, which nothing inside step() calls."""
        n = self.worlds.setdefault(id(w), len(self.worlds))
        self.rows.append((n, -1 - int(t), {"render": _digest(
            (w.bodies_array(), w.burning(), w.T()), None, "render")}))

    def tap(self, w, name, value):
        """A value that never lands in the world's state (a menu built only to
        be compared, a ledger total, a status line) -- digested where it is
        made, so a move that changes it cannot hide behind an unchanged world."""
        n = self.worlds.setdefault(id(w), len(self.worlds))
        self.rows.append((n, int(w.tick),
                          {"tap:" + name: _digest(value, w, "tap:" + name)}))


REC = None


class Probe(sim.World):
    """Exactly a World, plus one look at itself after each step. Scenes and the
    harvest build their worlds by the name `World`, so that name is pointed
    here for the duration of a case."""

    def step(self):
        super().step()
        if REC is not None:
            REC.note(self)

    # TAPS. `_full_menu` is the one result that is computed, compared and
    # thrown away: with recall at 100% a broken cap override leaves every
    # trace the same (would == pick), so the state hash alone cannot see it.
    def _full_menu(self, *a, **k):
        got = super()._full_menu(*a, **k)
        if REC is not None:
            REC.tap(self, "_full_menu", [o["key"] for o in got])
        return got


CONFIG = {}                             # per-case knobs applied at construction


class Configured(Probe):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for key, val in CONFIG.items():
            setattr(self, key, val)


@contextlib.contextmanager
def _patched(config):
    """Point every builder at the probe, silence prints, skip frame files."""
    global CONFIG
    saved = (scenes.World, harvest.World, scenes._save)
    CONFIG = dict(config)
    scenes.World = harvest.World = Configured
    scenes._save = lambda w, frames_dir, t: REC and REC.render(w, t)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        scenes.World, harvest.World, scenes._save = saved
        CONFIG = {}


# ── the cases: small, fast, and between them every law and every limb ───────

def _scene(name, ticks, **config):
    def run():
        with tempfile.TemporaryDirectory() as d, _patched(config):
            scenes.SCENARIOS[name](d, ticks=ticks, every=10 ** 9)
        return None
    return run


def _harvest(seed, ticks=90):
    def run():
        with _patched({}):
            rows = harvest.run(seed, ticks=ticks)    # seam.row text included
        return rows
    return run


def _pole():
    """test_a_hinged_tree_topples_toward_the_notch: the cheapest world that
    sends a cluster through the MPM seam (promote/step/rasterize)."""
    def run():
        with _patched({}):
            w = Configured(24, 8, 16)
            w.fill(0, 24, 0, 8, 0, 1, sim.STONE)
            w.fill(4, 7, 3, 5, 1, 12, sim.WOOD)
            w.mat[5:7, 3:5, 1] = sim.AIR
            w.smass[5:7, 3:5, 1] = 0.0
            for _ in range(80):
                w.step()
        return None
    return run


def _room(nx, ny, nz):
    """A sealed stone box with an air interior: the common start of the
    hand-built cases below (the same build as test_voxel.py's crowded hall)."""
    w = Configured(nx, ny, nz, voxel_cm=5)
    w.open_sky = False
    w.fill(0, nx, 0, ny, 0, nz, sim.STONE)
    w.mat[1:nx - 1, 1:ny - 1, 1:nz - 1] = sim.AIR
    w.smass[1:nx - 1, 1:ny - 1, 1:nz - 1] = 0.0
    w.exits = []
    return w


def _ledgers(w):
    """Every conservation total and every person's status line, as one tap."""
    REC.tap(w, "ledgers", (
        [w.total_mass(m) for m in (sim.WOOD, sim.STONE, sim.IRON, sim.FLESH)],
        w.total_wood(), w.total_energy(), w.total_lost(),
        [w.total_fluid(f) for f in (sim.WATER, sim.OIL, sim.BLOOD)],
        [w.person_status(p) for p in w.persons]))


def _flee():
    """SIGHT AND THE WAY OUT: a man faces a lit crib 1 m away with the door
    behind him. Reaches _in_cone/_sees (the main perception path, which no
    scene case under 100 ticks reaches), _plan_path, the walk, and _leave."""
    def run():
        from src.voxel.scenes import _person
        with _patched({}):
            w = _room(48, 24, 40)
            w.exits = [(2, 12)]
            w.fill(30, 36, 9, 15, 1, 3, sim.WOOD, frac=0.8)
            w.pour(31, 10, 3, sim.OIL, 100.0)
            p = _person(w, 14, 12)
            p["name"], p["facing"] = "Aldan", (1.0, 0.0)
            for t in range(140):
                if t < 20:
                    w.E[31, 10, 3] += 2500.0
                w.step()
                p["events"].clear()
                if t % 35 == 0:
                    _ledgers(w)
        return None
    return run


def _recall_hall(ticks=30):
    """test_the_ATTENTION_CAP_never_hides_the_answer's hall, short: the
    recall_check path (_full_menu and its cap override) with the cap biting."""
    def run():
        from src.voxel.scenes import _person
        with _patched({}):
            w = _room(90, 60, 40)
            w.exits = [(88, 12), (88, 30), (88, 48), (1, 12), (1, 48), (45, 58)]
            w.fill(10, 18, 20, 30, 1, 3, sim.WOOD, frac=0.8)
            w.E[10, 20, 1] = 9.0e5
            for n, (x, y) in enumerate(((30, 10), (60, 40), (20, 50), (70, 12))):
                _person(w, x, y)["name"] = f"P{n}"
            w.recall_check = True
            for _ in range(ticks):
                w.step()
        return None
    return run


def _wounded(ticks=120):
    """One axe cut (test_wounds.py's _cut), then the world runs: bleeding
    through _spill_from onto the floor, blood ledger, status line."""
    def run():
        from src.voxel.scenes import _person
        with _patched({}):
            w = Configured(30, 20, 40, voxel_cm=5)
            w.fill(0, 30, 0, 20, 0, 1, sim.STONE)
            p = _person(w, 12, 10)
            c = w._bone_parts(p, ["right forearm"])[0]
            cell = tuple(int(v) for v in c[np.argmax(c[:, 0])])
            REC.tap(w, "blow", w._blow(p, cell, 60.0, 2e-4))
            for t in range(ticks):
                w.step()
                p["events"].clear()
                if t % 40 == 0:
                    _ledgers(w)
        return None
    return run


def _throw(ticks=60):
    """test_a_WIND_UP_THROWS_HARDER's thrower, once: take, wind up, let go --
    _release and a thrown free body."""
    def run():
        from src.voxel.scenes import _person, _Wants
        with _patched({}):
            w = _room(60, 60, 46)
            p = _person(w, 20, 20)
            p["name"], p["facing"] = "thrower", (0.0, 1.0)
            p["strength_N"] = 3000.0
            w.fill(20, 21, 24, 25, 1, 3, sim.IRON)
            w.policy = _Wants(each={"thrower": {"hands": "take hold of the iron",
                                                "legs": "stay", "waist": "stand"}})
            for _ in range(30):
                w.step()
                p["events"].clear()
            w.policy = _Wants(each={"thrower": {"hands": "throw the iron",
                                                "legs": "stay", "waist": "stand"}})
            for _ in range(ticks):
                w.step()
                p["events"].clear()
        return None
    return run


def _rewind():
    """snapshot -> run on -> restore -> run again, on a harvest world with
    people in it: the ledgers mixin, and the world-as-a-value contract. Pins
    TODAY's behaviour, ghost World included (see the design, item 3)."""
    def run():
        with _patched({}):
            w, people, _lip = harvest.world(3)
            w.policy = harvest.Roam(3)
            for _ in range(20):
                w.step()
            snap = w.snapshot()
            _ledgers(w)
            for _ in range(10):
                w.step()
            w.restore(snap)
            _ledgers(w)
            for _ in range(15):
                w.step()
            _ledgers(w)
        return None
    return run


def _pendulum(calls=120):
    """test_a_SHOULDER_IS_A_JOINT_and_an_UNHELD_ARM_SWINGS, driven the way the
    test drives it: _law_joints called directly. Under step() an out-cold arm
    is put down before _law_joints sees it (the arms scene: 1.2 -> 0.0 within
    six ticks), so no stepped case reaches _limb_cells."""
    def run():
        from src.voxel.scenes import _person
        with _patched({}):
            w = Configured(40, 30, 50, voxel_cm=5)
            w.open_sky = False
            w.fill(0, 40, 0, 30, 0, 1, sim.STONE)
            _person(w, 12, 15)
            w.step()
            p = w.persons[0]
            p["facing"] = (0.0, 1.0)
            out = [0.35, 0.0, 0.0, 0.0]
            REC.tap(w, "repose", w._repose(p, "left arm", out))
            p["pose"]["left arm"] = list(out)
            p["awake"] = False
            for _ in range(calls):
                w._law_joints()
                REC.note(w)
        return None
    return run


class _Duel:
    """tests/test_wounds.py's _Duel, copied here so that an edit to the test
    cannot move the golden: the swinger takes what is at his feet and swings
    it; every other hand keeps what it has; every other limb stays."""
    name = "duel"

    def __init__(self, swinger, then="swing"):
        self.swinger, self.then = swinger, then

    def pick(self, sit, menu):
        want = "stay"
        if sit["who"] == self.swinger and sit["limb"] == "hands":
            want = self.then if sit["holding"] else "take"
        elif sit["limb"] == "hands":
            want = "keep"
        for i, o in enumerate(menu):
            if o["tag"] == want:
                return i
        return 0


def _duel150(ticks=150):
    """THE DECISION CASE: tests/test_wounds.py's `_duel(AXE)` as it stands --
    the attacker at x=17, an axe (edge 2e-4) at 22-24, the defender at 31 --
    for 150 ticks. Every limb menu, pick and trace of both bodies is in the
    hash, so a change to how a body decides shows here first."""
    def run():
        from src.voxel.scenes import _person
        with _patched({}):
            w = Configured(46, 20, 40, voxel_cm=5)
            w.fill(0, 46, 0, 20, 0, 1, sim.STONE)
            w.exits = [(44, 10)]
            _person(w, 17, 10)["name"] = "attacker"
            _person(w, 31, 10)["name"] = "defender"
            w.fill(22, 24, 10, 11, 1, 2, sim.IRON, edge=2e-4)
            w.policy = _Duel("attacker")
            for _ in range(ticks):
                w.step()
        return None
    return run


# ── the wave 13 proof cases (C11): the new scenes, small, one per capability ──
# Each builds the world of a 13a/13b/13c scene module and steps it, noting the
# whole world every `every` ticks. A case whose scene module is not in the tree
# being recorded (a base older than wave 13) raises Absent and is left out of
# the recording; it then shows as NOT RECORDED in a check, which a step must
# declare. Run `golden.py record` on the finished tree to take them in.

class Absent(RuntimeError):
    """The tree has no module this case needs."""


class _Noted:
    """Mixed in front of a scene's own world class: report after each tick."""

    def step(self):
        super().step()
        if REC is not None and self.tick % self._note_every == 0:
            REC.note(self)


def _need(*mods):
    import importlib.util
    for m in mods:
        try:
            ok = importlib.util.find_spec("src.voxel." + m) is not None
        except (ImportError, ValueError):
            ok = False
        if not ok:
            raise Absent(m)


def _proof(mods, build, ticks, every=10, drive=None):
    """build() -> a world, or a tuple whose first item is the world. The scene's
    own class keeps its identity; a subclass reports itself after each step."""
    def run():
        _need(*mods)
        with _patched({}):
            got = build()
            w = got[0] if isinstance(got, tuple) else got
            w.__class__ = type("Noted" + type(w).__name__, (_Noted, type(w)), {})
            w._note_every = every
            REC.note(w)
            for t in range(ticks):
                if drive is not None:
                    drive(w, got, t)
                w.step()
            _ledgers(w)
        return None
    return run


def _b_kettle():
    """A 108 C cup of water on the wet world: it boils, the steam is kept as vapour."""
    from src.voxel import wetworld
    w = wetworld.WetWorld(14, 14, 14, voxel_cm=5)
    w.fill(0, 14, 0, 14, 0, 1, sim.STONE)
    w.fill(4, 9, 4, 9, 1, 4, sim.STONE)
    w.fill(5, 8, 5, 8, 1, 4, sim.AIR)
    for x in range(5, 8):
        for y in range(5, 8):
            w.pour(x, y, 1, sim.WATER, w.cap)
    C = w.heat_capacity()
    w.E[5:8, 5:8, 1] = (108.0 - sim.AMBIENT) * C[5:8, 5:8, 1]
    w.enable_wet(rh=0.5)
    return w


def _b_pond():
    from src.voxel import scenes_wet
    return scenes_wet.pond()


def _b_sooty():
    from src.voxel import scenes_layers
    return scenes_layers.sooty_room(nx=20, ny=20, nz=10, crib=2, match_s=20.0)


def _b_quench():
    from src.voxel import scenes_records
    return scenes_records.quench_blade(6.0)


def _b_stairs():
    from src.voxel import scenes_stairs
    w, p = scenes_stairs.stair_flight()
    w._law_will = lambda: None                # the legs are driven below
    return w, p


def _d_stairs(w, got, t):
    p = got[1]
    comp, sl = w._person_cells(p)
    cells = w._claim_cells(p, comp, sl)
    ax, ay = float(cells[:, 0].mean()), float(cells[:, 1].mean())
    w._stride(p, cells, (w.shape[0] - 4, w.shape[1] // 2), ax, ay)


def _b_walk():
    from src.voxel import scenes_contact as S
    w, p = S.walkway(nx=70)
    w._law_will = lambda: None
    S.lay(w, 14, 26, S.mud(1.0))
    S.lay(w, 26, 38, S.chalk(0.3))
    return w, p


def _d_walk(w, got, t):
    p = got[1]
    comp, sl = w._person_cells(p)
    cells = w._claim_cells(p, comp, sl)
    ax, ay = float(cells[:, 0].mean()), float(cells[:, 1].mean())
    w._stride(p, cells, (64, 7), ax, ay)


def _b_cold():
    from src.voxel import scenes_cold
    return scenes_cold.cold_yard(shape=(80, 70, 40))


def _b_brine():
    from src.voxel import scenes_mixtures as M
    return M.brine_pot(M.BRINE20)


def _d_brine(w, got, t):
    w.E[1, 2, 1] += 120.0


def _b_bodyheat():
    from src.voxel import scenes_bodyheat
    return scenes_bodyheat.lattice_nude_20C(0.0)


def _b_fall():
    from src.voxel.scenes import _person
    w = sim.World(24, 24, 170, voxel_cm=5)
    w.fill(0, 24, 0, 24, 0, 1, sim.STONE)
    w.exits = []
    _person(w, 12, 12, z0=1 + 120)                      # 6 m onto stone: heels and lumbar break
    return w



CASES = {
    # matter
    "pond":          _scene("pond", 40),                  # fall, splash, settle
    "drop_test":     _scene("drop_test", 60),             # shatter, noise, spill
    "acid_bath":     _scene("acid_bath", 60),             # acid, fumes
    "torch_pillar":  _scene("torch_pillar", 80),          # heat, burn, O2, smoke, air
    "forge":         _scene("forge", 50),                 # melt / freeze / char
    "pole_mpm":      _pole(),                             # the particle seam
    # the plain numpy laws and the ungated scheduler, which the default skips
    "torch_plain":   _scene("torch_pillar", 40, fused=False, skip_quiet=False),
    # bodies
    "jumper":        _scene("jumper", 80),                # jump, flight, landing
    "swing":         _scene("swing", 60),                 # arm swing, fist on glass
    "arms":          _scene("arms", 80),                  # take, hold, joints, faint
    "crouch":        _scene("crouch", 0),                 # _crouch direct (fixed loop)
    "arrow_trap":    _scene("arrow_trap", 40),            # loose, wounds, plate
    "rescue":        _scene("rescue", 100),               # grip, catch, lean
    # minds: Roam picks off real menus; rows go through seam.row
    **{f"harvest_{s}": _harvest(s) for s in range(8)},
    # the paths the cases above never reach (golden.py coverage, 2026-10-01)
    "flee":          _flee(),                             # _sees _in_cone _plan_path _leave
    "recall_hall":   _recall_hall(),                      # _full_menu + its cap override
    "wounded":       _wounded(),                          # _spill_from, ledgers, status
    "throw":         _throw(),                            # _release
    "rewind":        _rewind(),                           # snapshot / restore
    "pendulum":      _pendulum(),                         # _law_joints -> _limb_cells
    # decisions: two bodies, every limb asked, a swing that lands
    "duel150":       _duel150(),                          # take, swing, wound, traces
    # wave 13 proof cases (C11): one small scene per switched-on capability
    "w13_wet_kettle":   _proof(("wetworld", "scenes_wet"), _b_kettle, 160, 20),
    "w13_wet_pond":     _proof(("wetworld", "scenes_wet"), _b_pond, 160, 20),
    "w13_sooty_room":   _proof(("scenes_layers",), _b_sooty, 400, 40),
    "w13_quench":       _proof(("scenes_records",), _b_quench, 200, 20),
    "w13_stairs":       _proof(("scenes_stairs", "pathlaw"), _b_stairs, 240, 20, _d_stairs),
    "w13_mud_chalk":    _proof(("scenes_contact", "contact"), _b_walk, 400, 40, _d_walk),
    "w13_cold_yard":    _proof(("coldworld", "scenes_cold"), _b_cold, 40, 10),
    "w13_brine_pot":    _proof(("scenes_mixtures", "heatmix"), _b_brine, 300, 30, _d_brine),
    "w13_body_heat":    _proof(("bodyworld", "scenes_bodyheat"), _b_bodyheat, 200, 40),
    "w13_fall_6m":      _proof(("injury",), _b_fall, 240, 20),
}
# Heavier, run before the full suite rather than after every batch.
SLOW = {
    "fire_alarm":    _scene("fire_alarm", 120),           # sight, hearing, speech
    "duel":          _scene("duel", 200),                 # axe, wound, bleed
    "ledge":         _scene("ledge", 100),                # pulled off a lip
    "leap":          _scene("leap", 60),
    "parachute":     _scene("parachute", 60),             # drag
    "vessels":       _scene("vessels", 60),
    # behind a flag nothing sets (sim.py:1000, 1399): pinned so it is moved too
    "seam_fall":     _scene("drop_test", 40, free_fall_to_seam=True),
}
ALL = {**CASES, **SLOW}


def run_case(name, known=None):
    """Run one case. `known` is the base recording's paths, or None to hash
    every name (a recording)."""
    global REC, KNOWN
    REC, KNOWN = _Recorder(), known
    NAMES.clear()
    NEW.clear()
    t0 = time.perf_counter()
    try:
        ret = ALL[name]()
        secs = time.perf_counter() - t0
        rec = REC
        returned = _digest(ret, None, "returned")
    finally:
        REC, KNOWN = None, None
    return {"seconds": round(secs, 2),
            "ticks": [[n, t] for n, t, _ in rec.rows],
            "digests": [d for _, _, d in rec.rows],
            "returned": returned,
            "paths": sorted(NAMES),
            "new": sorted(NEW)}


def compare(base, now):
    """First divergence, named: (tick index, world, tick, [fields])."""
    if len(base["digests"]) != len(now["digests"]):
        return f"tick count {len(base['digests'])} -> {len(now['digests'])}"
    for i, (a, b) in enumerate(zip(base["digests"], now["digests"])):
        if a != b:
            bad = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            n, t = base["ticks"][i]
            t = f"tick {t}" if t >= 0 else f"frame dump at t{-1 - t}"
            return f"row {i}, world {n} {t}: {', '.join(bad)}"
    if base["returned"] != now["returned"]:
        return "returned rows differ"
    return None


# ── what a recording was made from ───────────────────────────────────────────

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _sources():
    """{repo-relative path: sha} of every module this run loaded from REPO.
    Equal files load equal modules, so equal sources mean an equal world."""
    out = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if name != "__main__" and f and \
                os.path.realpath(f).startswith(REPO + os.sep):
            out[os.path.relpath(os.path.realpath(f), REPO)] = _sha(f)
    return dict(sorted(out.items()))


def _meta():
    return {"golden": _sha(os.path.realpath(__file__)),
            "numba": bool(sim.HAVE_NUMBA),
            "hashseed": os.environ["PYTHONHASHSEED"],
            "sources": _sources()}


def _stale(meta, sources=True):
    """Why a recording with this meta cannot stand for a run of this file on
    REPO, or None. Without `sources`, only the hasher itself is compared (a
    check runs on a CHANGED tree by design)."""
    if not meta:
        return "no __meta__: recorded by an older golden.py"
    now = _meta()
    for k in ("golden", "numba", "hashseed"):
        if meta.get(k) != now[k]:
            return f"{k} was {meta.get(k)!r}, is {now[k]!r}"
    if sources:
        for rel, sha in meta["sources"].items():
            p = os.path.join(REPO, rel)
            if not os.path.exists(p):
                return f"{rel} is gone"
            if _sha(p) != sha:
                return f"{rel} changed"
    return None


# ── coverage ─────────────────────────────────────────────────────────────────

def coverage(names):
    """Which World methods do these cases never call? A move that breaks one
    of THOSE is invisible to the golden check, so it is listed, and the
    suite (or a new case) has to cover it."""
    import ast
    seen = set()
    vox = os.path.join(REPO, "src", "voxel")

    def prof(frame, event, arg):
        if event == "call" and frame.f_code.co_filename.startswith(vox):
            seen.add(frame.f_code.co_name)
    global REC
    sys.setprofile(prof)
    try:
        for k in names:
            REC = _Recorder()           # so render taps and ledgers RUN
            ALL[k]()
    finally:
        sys.setprofile(None)
        REC = None
    world_methods = set()
    for f in os.listdir(vox):                       # works before AND after
        if f.endswith(".py"):                       # the split
            for n in ast.walk(ast.parse(open(os.path.join(vox, f)).read())):
                if isinstance(n, ast.ClassDef) and (
                        n.name == "World" or n.name.endswith("Mixin")):
                    world_methods |= {m.name for m in n.body
                                      if isinstance(m, ast.FunctionDef)}
    return sorted(world_methods - seen)


# ── the command line ─────────────────────────────────────────────────────────

def _cases(argv, flag):
    """The case list after `flag`, every name checked: a misspelled case
    would otherwise select, skip or allow nothing, silently."""
    if flag not in argv:
        return None
    got = [k for k in argv[argv.index(flag) + 1].split(",") if k]
    if flag == "--allow" and got in (["any"], ["none"]):
        return list(ALL) if got == ["any"] else []
    bad = [k for k in got if k not in ALL]
    if bad:
        sys.exit(f"golden.py: {flag}: no such case {bad}; see `golden.py list`")
    return got


def main(argv):
    if argv and argv[0] == "coverage":
        names = _cases(argv, "--only") or (list(ALL) if "--all" in argv
                                           else list(CASES))
        missed = coverage(names)
        print(f"{len(missed)} World methods never called by {len(names)} cases:")
        print("  " + "\n  ".join(missed))
        return 0
    if not argv or argv[0] == "list":
        for k in ALL:
            print(k, "(slow)" if k in SLOW else "")
        return 0
    mode, path = argv[0], argv[1]
    if mode == "fresh":
        why = _stale(json.load(open(path)).get("__meta__"))
        print(f"fresh: {path} stands for {REPO}" if not why
              else f"stale: {path}: {why}")
        return 1 if why else 0
    names = _cases(argv, "--only") or (list(ALL) if "--all" in argv
                                       else list(CASES))
    skip = set(_cases(argv, "--except") or [])
    names = [k for k in names if k not in skip]
    print(f"numba={sim.HAVE_NUMBA} hashseed={os.environ['PYTHONHASHSEED']} "
          f"repo={REPO}", flush=True)
    if mode == "record":
        out = {}
        if os.path.exists(path):
            out = json.load(open(path))
            why = _stale(out.get("__meta__"))
            if why:
                sys.exit(f"golden.py: {path} was recorded from another tree or "
                         f"hasher ({why}): record into a new file")
        for k in names:
            try:
                out[k] = run_case(k)
            except Absent as e:                     # a wave 13 case on an older tree
                print(f"  {k:14s} left out: this tree has no {e}", flush=True)
                continue
            del out[k]["new"]
            print(f"  {k:14s} {out[k]['seconds']:7.2f}s  "
                  f"{len(out[k]['digests'])} ticks", flush=True)
        meta = _meta()
        meta["sources"] = {**out.get("__meta__", {}).get("sources", {}),
                           **meta["sources"]}
        out["__meta__"] = meta
        tmp = path + ".part"
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, path)                       # never half a base
        return 0
    if mode == "check":
        base = json.load(open(path))
        why = _stale(base.get("__meta__"), sources=False)
        if why:
            print(f"  NOTE: the base was hashed differently ({why}): a DIFFERS "
                  f"may be the hasher, not the change", flush=True)
        allow = set(_cases(argv, "--allow") or [])
        if any("paths" not in v for k, v in base.items() if k != "__meta__"):
            print("  NOTE: the base has no name paths (an older golden.py): "
                  "every name is compared, a new one too", flush=True)
        fails = differs = 0
        for k in names:
            tag = " (allowed)" if k in allow else ""
            if k not in base:
                print(f"  {k:14s} NOT RECORDED{tag}")
                differs += 1
                fails += k not in allow
                continue
            try:
                now = run_case(k, set(base[k].get("paths", ())) or None)
                bad = compare(base[k], now)
                secs = now["seconds"]
            except Absent as e:                     # the tree has no such scene
                print(f"  {k:14s} NOT IN THIS TREE ({e}){tag}", flush=True)
                continue
            except Exception as e:                  # one broken case must not
                now = {"new": []}                   # hide the rest of the report
                bad, secs = f"RAISED {type(e).__name__}: {e}", 0.0
            differs += bool(bad)
            fails += bool(bad) and k not in allow
            print(f"  {k:14s} {secs:7.2f}s  "
                  f"{'SAME' if not bad else 'DIFFERS' + tag + ' at ' + bad}",
                  flush=True)
            if now["new"]:
                print(f"  {'':14s} NEW: {', '.join(now['new'])}", flush=True)
        if not differs:
            print("ALL SAME")
        else:
            print(f"{differs} CASE(S) DIFFER, {fails} not allowed")
        return 1 if fails else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
