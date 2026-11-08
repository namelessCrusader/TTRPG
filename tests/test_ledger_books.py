"""THE BOOKS and the ONE-FLOW rule (wave 13, part 13a, lane B2).

ledgers.py now holds BOOKS: the one list of every ledger term of wave 13, each
with its owner topic and the modules allowed to write it. tools/w13_manifest.py
keeps a copy for the audit tool; the first test here fails if the two differ.
The rest hold the two rules the plan states (plan section 1):

  ONE WRITER PER TERM   no function of any module of src/voxel (the 75 new
                        ones AND the old ones) writes a ledger term unless its
                        module belongs to the term's owner topic or the list
                        names it as `via`;
  ONE FLOW              a joule an act delivers is booked once: `mech_in` when
                        no actor delivers it, `work_out` when one does, never
                        both. `World.deposit` takes an actor and picks the
                        book; the static scan fails if any function but
                        `mechanisms.deposit` writes both.

Nothing in 13a switches a term on, so a fresh world reads every book as zero
and `total_energy` is what it was; the tests pin that, and pin that the two
books an act can move close against `total_energy`.
Tags: all [I] (identity, conservation, closed form)."""
import ast
import importlib.util
import os

import numpy as np
import pytest

from src.voxel.ledgers import BOOK_NAMES, BOOKS
from src.voxel.materials import AIR, IRON, STONE
from src.voxel.scenes import _person
from src.voxel.sim import World

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VOX = os.path.join(ROOT, "src", "voxel")


def _manifest():
    spec = importlib.util.spec_from_file_location(
        "w13_manifest_b2", os.path.join(ROOT, "tools", "w13_manifest.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MAN = _manifest()
TOPICS = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"}
# the two legacy terms every energy and mass law already writes
LEGACY = ("gone", "shed")
# words too common to read as a ledger write by name alone (a layers pool, a list of stores): left out of
# the by-name scan, as the phase A cross test leaves them out; their writers are held by the manifest test
GENERIC = ("pool", "stores")


# ── 1. the list ──────────────────────────────────────────────────────────────
def test_the_books_in_ledgers_are_the_manifests_books():
    keys = ("name", "topic", "writers", "via", "kind", "shared", "parts")
    mine = [{k: b[k] for k in keys} | {"note": b.get("note")} for b in BOOKS]
    theirs = [{k: b[k] for k in keys} | {"note": b.get("note")} for b in MAN.BOOKS]
    assert mine == theirs, "ledgers.BOOKS and tools/w13_manifest.BOOKS must be one list"
    assert BOOK_NAMES == MAN.BOOK_NAMES


def test_every_term_has_one_owner_a_kind_a_sign_and_only_owner_topic_writers():
    names = [b["name"] for b in BOOKS]
    assert len(names) == len(set(names)), "a term is declared once"
    for b in BOOKS:
        assert b["topic"] in TOPICS, b
        assert b["kind"] in ("mass", "energy", "volume", "mass+energy"), b
        assert isinstance(b["sign"], str) and len(b["sign"]) > 8, f"{b['name']} says what positive means"
        assert isinstance(b["parts"], tuple) and isinstance(b["writers"], tuple) \
            and isinstance(b["via"], tuple)
        if not b["shared"]:
            other = [w for w in b["writers"] if MAN.MODULES.get(w) != b["topic"]]
            assert not other, f"{b['name']}: owned by {b['topic']}, written by {other}"
    for need in ("recast", "quant", "f32", "squeeze", "records_owned", "outside_booked", "boiled_J",
                 "melted_J", "frozen_J", "dissolved_J", "made", "mech_in", "pool", "stores", "work_out",
                 "pooled_ml", "edema_ml", "layer_mass", "layer_enthalpy", "slow_residual"):
        assert need in names, f"the plan's term {need}"
    assert {b["name"] for b in BOOKS if b["shared"]} == set(LEGACY)


# ── 2. ONE WRITER PER TERM, over every module ───────────────────────────────
def _writes(src):
    """[(term, function, line, op)] for each `x.term op= ...`, `d["term"] op= ...`, and
    each class-level `term = ...` in `src`: op "decl" when that is the constant 0 or 0.0
    (a default at zero is a declaration, not a booking), "set" for any other value. Bare
    local names are not looked at: `made` as a local variable of some unrelated function is
    not a ledger write; and `pool` and `stores` (GENERIC) are not looked at by name at all."""
    terms = {t for t in BOOK_NAMES if t not in LEGACY + GENERIC} | set(MAN.SOURCE_ALIASES)
    out = []

    def term_of(node):
        while True:
            if isinstance(node, ast.Subscript):
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in terms:
                    return key.value
                node = node.value
            elif isinstance(node, ast.Attribute):
                if node.attr in terms:
                    return node.attr
                node = node.value
            else:
                return None

    def walk(node, fn, in_class):
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            c = in_class or isinstance(ch, ast.ClassDef)
            if isinstance(ch, ast.AugAssign):
                t = term_of(ch.target)
                if t:
                    out.append((MAN.SOURCE_ALIASES.get(t, t), fn, ch.lineno, "aug"))
            elif isinstance(ch, ast.Assign):
                for tg in ch.targets:
                    t = term_of(tg)
                    if t:
                        out.append((MAN.SOURCE_ALIASES.get(t, t), fn, ch.lineno, "set"))
                    elif isinstance(tg, ast.Name) and tg.id in terms and fn == "<module>" and in_class:
                        zero = isinstance(ch.value, ast.Constant) and ch.value.value in (0, 0.0)
                        out.append((tg.id, fn, ch.lineno, "decl" if zero else "set"))
            walk(ch, f, c)
    walk(ast.parse(src), "<module>", False)
    return out


def _modules():
    return {f[:-3]: open(os.path.join(VOX, f)).read()
            for f in sorted(os.listdir(VOX)) if f.endswith(".py")}


def test_no_module_writes_a_term_it_does_not_own_old_modules_and_new():
    """The phase A cross test scans the 75 new modules. This one scans EVERY file of
    src/voxel, so a lane that edits an old module (hands.py, bodies.py, heat.py...) cannot
    write a term of another topic either. A module outside the manifest owns nothing and
    writes nothing. A class-level default at zero (`mech_in = 0.0`, so every world can read
    the book) is a declaration, not a booking, and is allowed anywhere; one at any other
    value is a booking."""
    bad = []
    for name, src in _modules().items():
        for term, fn, line, op in _writes(src):
            if op == "decl":
                continue
            b = next(x for x in BOOKS if x["name"] == term)
            if MAN.MODULES.get(name) != b["topic"] and name not in b["via"]:
                bad.append(f"{name}.py:{line} ({fn}) {op} {term}: owner {b['topic']}, via {sorted(b['via'])}")
    assert not bad, "\n".join(bad)


def test_mech_in_has_exactly_one_writing_function_in_the_whole_tree():
    writers = sorted({(name, fn) for name, src in _modules().items()
                      for term, fn, _l, op in _writes(src) if term == "mech_in" and op != "decl"})
    assert writers == [("mechanisms", "deposit")], writers


def test_no_function_books_both_work_out_and_mech_in_except_deposit():
    """The one-flow rule, by source, over every module: a function that writes both
    books books two flows for one act. `mechanisms.deposit` names both because it picks one."""
    bad = []
    for name, src in _modules().items():
        per_fn = {}
        for term, fn, _l, op in _writes(src):
            if term in ("work_out", "mech_in") and op != "decl":
                per_fn.setdefault(fn, set()).add(term)
        bad += [f"{name}.{fn}" for fn, ts in per_fn.items() if len(ts) > 1 and (name, fn) != ("mechanisms", "deposit")]
    assert not bad, bad


def test_the_scan_catches_an_act_that_books_both_flows_and_a_foreign_writer():
    """A negative control: the detector must see what it claims to."""
    both = "def slam(w, J):\n    w.mech_in += J\n    w.work_out += J\n"
    assert {t for t, fn, _l, _o in _writes(both)} == {"mech_in", "work_out"}
    assert {fn for _t, fn, _l, _o in _writes(both)} == {"slam"}
    keyed = "def f(w):\n    w.outside_booked['rad'] += 1.0\n    w.books['mech_in'] = 3\n"
    assert {t for t, *_ in _writes(keyed)} == {"outside_booked", "mech_in"}
    assert _writes("class A:\n    mech_in = 0.0\n") == [("mech_in", "<module>", 2, "decl")]
    assert _writes("class A:\n    mech_in = 5.0\n") == [("mech_in", "<module>", 2, "set")], \
        "a default that is not zero is a booking"
    assert _writes("def f():\n    made = 3\n    pool = 4\n") == [], "a local variable is not a ledger write"
    assert _writes("def f(self):\n    self.pool += 1\n    self.stores = []\n") == [], "too common a word"


# ── 3. reading the books ─────────────────────────────────────────────────────
def test_a_fresh_world_reads_every_book_as_zero():
    w = World(8, 8, 8)
    b = w.books()
    assert tuple(b) == BOOK_NAMES
    for t in BOOKS:
        v = b[t["name"]]
        if t["name"] == "gone":
            assert v == {}
        elif t["parts"]:
            assert v == {}, t["name"]
        else:
            assert v == 0.0, t["name"]
    assert w.mech_in == 0.0 and w.work_out == 0.0
    assert "mech_in" not in vars(w) and "work_out" not in vars(w), \
        "declared at zero on the class: no new instance attribute until something is booked"


def test_books_reads_what_an_owner_has_written_and_calls_a_method_of_that_name():
    w = World(8, 8, 8)
    w.outside_booked = {"rad": 5.0, "edge": 1.5}
    w.boiled_J = 12.5
    w.layer_enthalpy = lambda: 7.0                       # served by a method
    b = w.books()
    assert b["outside_booked"] == {"rad": 5.0, "edge": 1.5} and b["boiled_J"] == 12.5
    assert b["layer_enthalpy"] == 7.0
    b["outside_booked"]["rad"] = 99.0
    assert w.outside_booked["rad"] == 5.0, "a dict comes back as a copy"


def test_a_stepped_world_closes_and_no_book_of_the_wave_moves():
    """The legacy `shed` is the one book that moves in an ordinary scene (heat leaving the lattice
    into the skin of the world); total_energy holds it, so total_energy is conserved to float32
    rounding, and every term wave 13 adds still reads zero: nothing writes them in 13a."""
    w = World(24, 24, 20, voxel_cm=5)
    w.open_sky = False
    w.fill(0, 24, 0, 24, 0, 20, STONE)
    w.mat[1:23, 1:23, 1:19] = AIR
    w.smass[1:23, 1:23, 1:19] = 0.0
    w.exits = []
    w.fill(10, 14, 10, 14, 1, 4, IRON)                   # a block of iron, hot
    w.E[10:14, 10:14, 1:4] = 4000.0
    e0 = w.total_energy()
    for _ in range(100):
        w.step()
    b = w.books()
    assert b["shed"] > 0.0, "heat left the lattice, and is booked where it always was"
    assert abs(w.total_energy() - e0) <= 1e-5 * e0
    new = {k: v for k, v in b.items() if k not in LEGACY}
    assert all(v == 0.0 or v == {} for v in new.values()), {k: v for k, v in new.items() if v}
    assert "mech_in" not in vars(w) and "work_out" not in vars(w)


def test_the_totals_do_not_take_a_book_in_13a():
    """Each lane that makes a term nonzero adds it to a total with its own closure proof;
    until then writing a book does not move total_mass, total_fluid or total_energy."""
    w = World(10, 8, 8)
    w.fill(0, 10, 0, 8, 0, 1, STONE)
    w.E[2:5, 2:5, 1] = 100.0
    ref = (w.total_energy(), w.total_mass(STONE), w.total_fluid(1), w.total_lost())
    w.mech_in, w.work_out = 11.0, 22.0
    w.outside_booked = {"rad": 3.0}
    w.records_owned = 5.0
    assert (w.total_energy(), w.total_mass(STONE), w.total_fluid(1), w.total_lost()) == ref


# ── 4. THE ONE-FLOW RULE on a real world ─────────────────────────────────────
def _room():
    w = World(30, 16, 36, voxel_cm=5)                    # tall enough for a person (30 voxels)
    w.open_sky = False
    w.fill(0, 30, 0, 16, 0, 1, STONE)
    return w


CELLS = np.array([[10, 8, 3], [11, 8, 3], [10, 9, 3]], np.int64)


def test_deposit_without_an_actor_books_mech_in_and_only_that():
    w = _room()
    e0, t0 = w.total_energy(), w.E.sum(dtype=np.float64)
    assert w.deposit(CELLS, 12.5) == "mech_in"
    assert (w.mech_in, w.work_out) == (12.5, 0.0)
    assert w.total_energy() - e0 == pytest.approx(12.5, rel=1e-6)
    assert w.E.sum(dtype=np.float64) - t0 == pytest.approx(12.5, rel=1e-6)
    assert w.books()["mech_in"] == 12.5 and w.books()["work_out"] == 0.0


def test_deposit_with_an_actor_debits_the_actor_credits_work_out_and_leaves_mech_in():
    w = _room()
    p = _person(w, 20, 6)
    e0 = w.total_energy()
    assert w.deposit(CELLS, 6.0, actor=p) == "work_out"
    assert (w.mech_in, w.work_out) == (0.0, 6.0)
    assert p["_work_J"] == 6.0, "the actor paid"
    assert w.total_energy() - e0 == pytest.approx(6.0, rel=1e-6)
    w.deposit(CELLS, 4.0, actor=p)
    assert w.work_out == 10.0 and p["_work_J"] == 10.0 and w.mech_in == 0.0


def test_an_act_never_moves_both_books_over_a_long_random_run():
    rng = np.random.default_rng(1307)
    w = _room()
    actors = [{}, {"name": "a"}, _person(w, 20, 6)]
    e0 = w.total_energy()
    for k in range(300):
        J = float(rng.uniform(0.01, 80.0))
        cells = CELLS[rng.integers(0, len(CELLS), size=int(rng.integers(1, 7)))]   # repeats on purpose
        who = actors[int(rng.integers(0, 3))] if rng.random() < 0.5 else None
        before = (w.mech_in, w.work_out)
        book = w.deposit(cells if k % 2 else tuple(cells.T), J, actor=who)
        moved = (w.mech_in - before[0], w.work_out - before[1])
        assert book == ("mech_in" if who is None else "work_out")
        assert (moved[0] == 0.0) != (moved[1] == 0.0), "exactly one book moved"
        assert max(moved) == pytest.approx(J, rel=1e-12) and min(moved) == 0.0
    # the joules are in the lattice once, and in exactly one book once
    assert w.total_energy() - e0 == pytest.approx(w.mech_in + w.work_out, rel=1e-5)


def test_a_cell_named_twice_takes_a_share_each_time_so_every_joule_is_stored():
    w = _room()
    cells = np.array([[10, 8, 3], [10, 8, 3], [11, 8, 3]], np.int64)
    w.deposit(cells, 9.0)
    assert float(w.E[10, 8, 3]) == pytest.approx(6.0, rel=1e-6)
    assert float(w.E[11, 8, 3]) == pytest.approx(3.0, rel=1e-6)


def test_the_heat_goes_into_the_dtype_of_the_lattice():
    w = _room()
    w.E = w.E.astype(np.float64)                         # (body heat will run float64)
    w.deposit(CELLS, 1.0)
    assert w.E[10, 8, 3] == 1.0 / 3.0, "not rounded to float32 on the way in"


def test_nothing_is_booked_for_no_joules_or_no_cells():
    w = _room()
    assert w.deposit(CELLS, 0.0) == "none"
    assert w.deposit(CELLS, -3.0) == "none", "a negative deposit is nothing, as the mechanism mixin has it"
    assert (w.mech_in, w.work_out, float(w.E.sum())) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        w.deposit(np.zeros((0, 3), np.int64), 5.0)
    assert (w.mech_in, w.work_out) == (0.0, 0.0), "refused before anything was booked"
    with pytest.raises(ValueError):
        w.deposit(CELLS, float("nan"))                   # a bug upstream is loud, not heat
    assert (w.mech_in, w.work_out, float(w.E.sum())) == (0.0, 0.0, 0.0)


def test_the_books_survive_a_snapshot_and_a_restore_that_goes_back_before_them():
    w = _room()
    s0 = w.snapshot()                                    # before anything was booked
    w.deposit(CELLS, 5.0)
    w.deposit(CELLS, 3.0, actor={})
    s1 = w.snapshot()
    w.deposit(CELLS, 100.0)
    assert w.mech_in == 105.0
    w.restore(s1)
    assert (w.mech_in, w.work_out) == (5.0, 3.0)
    w.restore(s0)
    assert (w.mech_in, w.work_out) == (0.0, 0.0), "back to the class default: nothing was booked yet"
    assert "mech_in" not in vars(w) and "work_out" not in vars(w)
    assert w.books()["mech_in"] == 0.0


def test_the_deposit_is_one_behaviour_the_mechanism_mixin_has_no_copy():
    """MechanismsMixin (T11) is in World since the M3 merge and its copies of
    deposit/_heat_cells/_debit are gone: one implementation, the ledger's."""
    from src.voxel.mechlaw import MechanismsMixin
    assert MechanismsMixin in World.__mro__
    for name in ("deposit", "_heat_cells", "_debit"):
        assert name not in vars(MechanismsMixin), name
        assert getattr(World, name).__qualname__.startswith("LedgersMixin"), name
