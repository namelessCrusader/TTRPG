"""AUDIT MODE for the lazy answers of the world (wave 15, S4-3).

A lazy answer is an exact speedup: a memo that says "matter has not moved", a
window that says "nothing can burn outside this box". It is exact only while
everything that could change the answer says so. The old comment in fields.py
feared the one failure of that bargain: "everyone who writes a voxel remembering
to invalidate it". This is the guard against it.

    MARK1_AUDIT_LAZY=1 python3 -m pytest ...

With the variable set, every lazy answer ALSO computes the full answer (the
definition, the code that was there before) and raises `LazyMismatch` when the
two differ. The golden check, the fuzzer and the whole suite are run once with
it on. Off (the default) it costs one dictionary lookup and nothing else.

The mode is read at call time (not at import), so a test can switch it with
`monkeypatch.setenv`. `COUNTS` says how many times each named answer was
audited, so a test can show that the audit really ran (a guard that never
fires proves nothing).

Marks: none needed; no number in here.
"""
import os

import numpy as np

COUNTS = {}


class LazyMismatch(AssertionError):
    """A lazy answer differed from the full one."""


def enabled():
    return os.environ.get("MARK1_AUDIT_LAZY") == "1"


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def check(name, lazy, full):
    """Return `lazy`. With the audit on, `full` (a function of no arguments that
    makes the full answer) is called too, and a difference raises."""
    if enabled():
        COUNTS[name] = COUNTS.get(name, 0) + 1
        want = full()
        if not _same(lazy, want):
            raise LazyMismatch("lazy answer '%s' differs from the full one: lazy=%r full=%r"
                               % (name, _short(lazy), _short(want)))
    return lazy


def proves(name, lazy, full):
    """Return `lazy`, a PROOF that a law has nothing to do. A proof may be stricter than the
    law's own test (say no when the law would do nothing anyway: that is only slower), so
    the audit objects only when it says yes and the full answer says the law acts."""
    if enabled():
        COUNTS[name] = COUNTS.get(name, 0) + 1
        if lazy and not full():
            raise LazyMismatch("proof '%s' says there is nothing to do, but the law acts" % name)
    return lazy


def _short(v):
    if isinstance(v, np.ndarray):
        return "array%s" % (v.shape,)
    return v
