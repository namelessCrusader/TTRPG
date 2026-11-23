"""THE TWIN HARNESS: one place that runs a coarse version against the full one and reports the drift
(wave 15 batch 2, lane 2d; the owner's SC5 "generalise the twin protocol to every coarse version").

THE CONTRACT every coarse version keeps (slowtwins.Twin is the first writing of it; see its docstring):
  applicable  the test that this world, person or room may be coarsened at all (and why not);
  horizon     the first moment fine detail is needed again: a threshold about to flip, an OBSERVER, a rare event;
  advance     go there coarsely with ALL LEDGERS BOOKED (a conserved total moves by what the full law moves it by);
  materialise a coarse state is a state of the full world (the same keys, nothing hidden in the twin);
  grade       EXACT: bit for bit (to float rounding) against the full law; or APPROXIMATE: checked on FACTS within
              a stated error budget, never a hidden constant.

THE HARNESS (`compare`): build two identical worlds, run one by the full law (ticks) and one by the coarse
version, read the same FACTS from both, and report for each fact the full value, the coarse value, the drift and
the budget; also the conserved total of each world, and the time of each. An EXACT twin must also leave the
whole state equal (the caller's `state_diff`). The report is plain data with a `table()` text; a lane's test calls
`assert_ok()`, and the drift table is what the lane report prints. Nothing here knows a particular law.

Marks: (u) unverified; (d) design choice.
"""
import time

EXACT = "exact"
APPROX = "approximate"
GRADES = (EXACT, APPROX)


class Fact:
    """One thing both runs must agree on. `get(w)` reads a number (or anything comparable with ==) from a world.
    kind "level": compare the value at the end. kind "change": compare end minus start (a flow: heat that went
    to the walls). The budget is |coarse - full| <= abs + rel * |full|; an EXACT twin states its own (1e-9)."""
    __slots__ = ("name", "get", "kind", "rel", "abs")

    def __init__(self, name, get, kind="level", rel=0.0, abs=0.0):
        if kind not in ("level", "change"):
            raise ValueError("kind is 'level' or 'change'")
        self.name, self.get, self.kind, self.rel, self.abs = name, get, kind, float(rel), float(abs)

    def read(self, w0, w1):
        v = self.get(w1)
        if self.kind == "change":
            return v - self.get(w0)
        return v


class Row:
    __slots__ = ("name", "kind", "full", "coarse", "drift", "drift_rel", "budget", "ok")

    def __init__(self, name, kind, full, coarse, rel, abs_):
        self.name, self.kind, self.full, self.coarse = name, kind, full, coarse
        num = isinstance(full, (int, float)) and isinstance(coarse, (int, float))
        if num:
            self.drift = coarse - full
            self.drift_rel = abs(self.drift) / abs(full) if full else (0.0 if self.drift == 0 else float("inf"))
            self.budget = abs_ + rel * abs(full)
            self.ok = abs(self.drift) <= self.budget
        else:
            self.drift, self.drift_rel, self.budget = None, 0.0, 0.0
            self.ok = full == coarse


class Report:
    """What `compare` returns. `ok` is true when every fact is inside its budget, both worlds close their books
    to `books_tol` and (EXACT) the states are equal."""

    def __init__(self, name, grade, n_ticks):
        self.name, self.grade, self.n_ticks = name, grade, n_ticks
        self.rows = []
        self.books = None               # {"full": drift, "coarse": drift, "tol": tol} or None
        self.state_diff = []
        self.seconds = {"full": 0.0, "coarse": 0.0}
        self.notes = []

    @property
    def ok(self):
        if not all(r.ok for r in self.rows):
            return False
        if self.books is not None and not all(abs(self.books[k]) <= self.books["tol"] for k in ("full", "coarse")):
            return False
        return not self.state_diff

    @property
    def worst(self):
        """The row with the largest share of its budget used (None for no rows)."""
        best, share = None, -1.0
        for r in self.rows:
            if r.drift is None:
                s = 0.0 if r.ok else float("inf")
            else:
                s = abs(r.drift) / r.budget if r.budget > 0 else (0.0 if r.drift == 0 else float("inf"))
            if s > share:
                best, share = r, s
        return best

    @property
    def speedup(self):
        c = self.seconds["coarse"]
        return self.seconds["full"] / c if c > 0 else float("inf")

    def table(self):
        lines = ["TWIN %s  grade %s  %d ticks  full %.2f s  coarse %.2f s  (x%.1f)  -> %s"
                 % (self.name, self.grade, self.n_ticks, self.seconds["full"], self.seconds["coarse"],
                    self.speedup, "OK" if self.ok else "DRIFT OUT OF BUDGET")]
        lines.append("  %-28s %-6s %14s %14s %12s %8s %s" % ("fact", "kind", "full", "coarse", "drift", "rel", "budget"))
        for r in self.rows:
            if r.drift is None:
                lines.append("  %-28s %-6s %14s %14s %12s %8s %s" % (r.name, r.kind, r.full, r.coarse, "-", "-",
                                                                   "equal" if r.ok else "DIFFER"))
            else:
                lines.append("  %-28s %-6s %14.6g %14.6g %12.4g %7.2f%% %.4g %s"
                             % (r.name, r.kind, r.full, r.coarse, r.drift, 100.0 * r.drift_rel, r.budget,
                                "" if r.ok else "<-- OUT"))
        if self.books is not None:
            lines.append("  books drift (J): full %.4g  coarse %.4g  tolerance %.4g"
                         % (self.books["full"], self.books["coarse"], self.books["tol"]))
        for d in self.state_diff[:8]:
            lines.append("  state differs: %s" % (d,))
        lines += ["  note: %s" % n for n in self.notes]
        return "\n".join(lines)

    def as_dict(self):
        return {"name": self.name, "grade": self.grade, "ok": self.ok, "n_ticks": self.n_ticks,
                "rows": {r.name: {"full": r.full, "coarse": r.coarse, "drift": r.drift, "budget": r.budget,
                                  "ok": r.ok} for r in self.rows},
                "books": self.books, "state_diff": list(self.state_diff), "seconds": dict(self.seconds)}

    def assert_ok(self):
        assert self.ok, "\n" + self.table()


def step_full(w, n):
    """The full law: n ticks of the world's own step()."""
    for _ in range(n):
        w.step()


def compare(name, build, n, coarse, facts, *, grade=APPROX, full=None, books=None, books_tol=1.0,
            state_diff=None, settle=None):
    """Run the coarse version against the full law and report the drift.

    build()        -> a world (called twice: the two worlds must be identical);
    coarse(w, n)   -> advance a world n ticks coarsely (a twin's advance, a night jump, an LOD step);
    facts          a list of Fact (what the coarse version must agree on, each with its own budget);
    grade          EXACT or APPROX (documentation and the EXACT state check; budgets are the facts' own);
    full(w, n)     the full law, default step_full;
    books(w)       a conserved total (joules, or grams); each run's drift over the run is reported against
                   `books_tol` (the books of a coarse version close like the full law's);
    state_diff(a, b) -> [] for an EXACT twin: the two final worlds must have equal state;
    settle(w)      run on both worlds after build (warm up), before the starts are read.
    Returns a Report."""
    if grade not in GRADES:
        raise ValueError("grade is exact or approximate")
    rep = Report(name, grade, int(n))
    wf, wc = build(), build()
    if settle is not None:
        settle(wf)
        settle(wc)
    f_run = full or step_full
    # the starts (a "change" fact reads its own start) are read from a copy of the observable at tick 0
    starts_f = [(f.get(wf) if f.kind == "change" else None) for f in facts]
    starts_c = [(f.get(wc) if f.kind == "change" else None) for f in facts]
    b0f = books(wf) if books else None
    b0c = books(wc) if books else None
    t = time.perf_counter()
    f_run(wf, n)
    rep.seconds["full"] = time.perf_counter() - t
    t = time.perf_counter()
    coarse(wc, n)
    rep.seconds["coarse"] = time.perf_counter() - t
    for f, sf, sc in zip(facts, starts_f, starts_c):
        vf, vc = f.get(wf), f.get(wc)
        if f.kind == "change":
            vf, vc = vf - sf, vc - sc
        rep.rows.append(Row(f.name, f.kind, vf, vc, f.rel, f.abs))
    if books:
        rep.books = {"full": books(wf) - b0f, "coarse": books(wc) - b0c, "tol": float(books_tol)}
    if grade == EXACT and state_diff is not None:
        rep.state_diff = list(state_diff(wf, wc))
    rep.worlds = (wf, wc)
    return rep


def contract(twin):
    """[problem] a twin's description leaves out of the protocol: the five methods, a grade, a stated budget for an
    approximate twin (a twin with none has not said how wrong it may be)."""
    out = []
    for m in ("applicable", "enter", "horizon", "advance", "power_w", "o2_use_g"):
        if not callable(getattr(twin, m, None)):
            out.append("missing %s" % m)
    from .slowtwins import grade_of, budget_of
    g = grade_of(twin)
    if g not in GRADES:
        out.append("grade %r is not exact or approximate" % (g,))
    if g == APPROX and not budget_of(twin):
        out.append("an approximate twin must state its error budget")
    return out
