"""PROTOTYPE (scratch only, never imported by src): a typed event log for the
voxel World, emitted alongside the legacy strings, plus a BATCHED perception
pass that is checked against today's `_law_hear` tick by tick.

Run:  flock <lock> python3 events_proto.py
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

REPO = "/home/nisargparikh/Desktop/Fun Stuff/Mark_1"
sys.path.insert(0, REPO)
from src.voxel import sim as S                                    # noqa: E402
from src.voxel.sim import (World, STONE, WOOD, GLASS, AIR, SOUND,  # noqa: E402
                           _DENS_ARR, _TRANSMIT_ARR, SMOKE_EXT, SEE_TAU, WILL)
from src.voxel.scenes import _person                               # noqa: E402


# ── the record ────────────────────────────────────────────────────────────
@dataclass(slots=True, frozen=True)
class Ev:
    id: int
    tick: int
    actor: int | None          # person id (pid), None = the world
    verb: str
    target: tuple | None       # ("person", pid) | ("thing", eid) | ("cell", xyz)
    instrument: tuple | None
    result: str | None
    why: str | None
    place: tuple               # (x, y, z) cell
    db: float | None           # loudness at 1 m; None = silent
    visible: bool
    cause: int | None
    text: str                  # legacy line WITHOUT the "t{tick}: " prefix
    data: tuple = ()           # extras as sorted (k, v) pairs (hashable, cheap)


class EventLog:
    """Bounded, id-addressed. Ids never repeat; old rows fall off the front."""
    __slots__ = ("rows", "first", "cap")

    def __init__(self, cap=4096):
        self.rows, self.first, self.cap = [], 0, cap

    @property
    def next_id(self):
        return self.first + len(self.rows)

    def append(self, **kw):
        ev = Ev(id=self.next_id, **kw)
        self.rows.append(ev)
        if len(self.rows) > self.cap:              # amortised trim
            cut = len(self.rows) - self.cap // 2
            del self.rows[:cut]
            self.first += cut
        return ev

    def get(self, eid):
        i = eid - self.first
        return self.rows[i] if 0 <= i < len(self.rows) else None

    def since(self, eid):
        return self.rows[max(eid - self.first, 0):]


# ── batched rays: the one primitive both senses need ───────────────────────
def rays(w, A, B, owner=None, skip=None):
    """For every ray A[i]->B[i]: optical depth (as `_sees`) and kg of solid on
    the path (as `_level`). `skip[i]` = (pid_a, pid_b): cells owned by either
    are not walls (the `own` rule of `_law_hear`)."""
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    n = np.ceil(np.abs(B - A).max(axis=1)).astype(np.int64) + 1
    k = np.maximum(n - 2, 0)
    tot = int(k.sum())
    ray = np.repeat(np.arange(len(A)), k)
    start = np.cumsum(k) - k
    j = np.arange(tot) - np.repeat(start, k) + 1
    ts = j * (1.0 / np.maximum(np.repeat(n, k) - 1, 1))   # == np.linspace
    a = A[ray]
    pts = np.round(a + ts[:, None] * (B[ray] - a)).astype(np.int64)
    for ax in range(3):
        np.clip(pts[:, ax], 0, w.shape[ax] - 1, out=pts[:, ax])
    idx = tuple(pts.T)
    keep = np.ones(tot, bool)
    if owner is not None and skip is not None:
        sk = np.asarray(skip, np.int64)[ray]
        o = owner[idx]
        keep = (o < 0) | ((o != sk[:, 0]) & (o != sk[:, 1]))
    sm = w.smass[idx]
    m = w.mat[idx]
    vox_m = 0.1 * w.scale
    packed = np.clip(sm / np.maximum(_DENS_ARR[m] * w.vox_l, 1e-9), 0.0, 0.999)
    through = (1.0 - packed) + packed * _TRANSMIT_ARR[m]
    d = -np.log(np.maximum(through, 1e-9)) \
        + SMOKE_EXT * (w.smoke[idx] / w.vox_l) * vox_m
    near = np.abs(pts - a).max(axis=1) > 1.0
    tau = np.bincount(ray, weights=np.where(near, d, 0.0), minlength=len(A))
    kg = np.bincount(ray, weights=np.where(keep, sm, 0.0),
                     minlength=len(A)) / 1000.0
    return tau, kg


def levels(w, A, B, db1m, kg):
    vox_m = 0.1 * w.scale
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    dist = np.maximum(np.array([float(np.linalg.norm(b - a))
                                for a, b in zip(A, B)]) * vox_m, 1.0)
    lv = np.asarray(db1m, np.float64) - 20.0 * np.log10(dist)
    sigma = kg / max(vox_m * vox_m, 1e-9)
    loss = np.where(sigma > 0,
                    np.maximum(0.0, 20.0 * np.log10(np.maximum(sigma, 1e-30)
                                                    * 500.0) - 47.0), 0.0)
    return lv - loss


# ── a World that keeps the log beside the legacy channels ─────────────────
class EvWorld(World):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = EventLog()
        self._heard_upto = 0
        self.shadow = {"pairs": 0, "agree": 0, "disagree": [], "us": [],
                       "legacy_us": []}

    def pid(self, name):
        for i, q in enumerate(self.persons):
            if q["name"] == name:
                return i
        return None

    def _noise(self, at, joules, mats, who=None):
        snd = super()._noise(at, joules, mats, who=who)
        if snd is not None:
            ev = self.log.append(
                tick=self.tick, actor=self.pid(who) if who else None,
                verb="noise", target=None, instrument=None, result=None,
                why=None, place=tuple(int(round(v)) for v in at),
                db=snd["db"], visible=False, cause=None, text="")
            snd["ev"] = ev.id
        return snd

    def _say(self, p, response):
        n = len(self.speech)
        super()._say(p, response)
        if len(self.speech) > n:
            ev = self.log.append(
                tick=self.tick, actor=self.pid(p["name"]), verb="says",
                target=None, instrument=None, result=None, why=None,
                place=tuple(int(round(v)) for v in (p.get("_eye") or
                                                    (*p["anchor"], 0))),
                db=S.SHOUT_DB, visible=True, cause=None,
                text=p["events"][-1].split(": ", 1)[1],
                data=(("line", response),))
            if self.sounds and self.sounds[-1].get("who") == p["name"]:
                self.sounds[-1]["ev"] = ev.id

    # SHADOW: compute what `_law_hear` will decide, batched, then let the real
    # one run and compare. Nothing the sim reads is touched.
    def _law_hear(self):
        t0 = time.perf_counter()
        sounds = [s for s in self.sounds if s["tick"] >= self.tick - 1]
        mine = {}
        if sounds and self.persons:
            owner = np.full(self.shape, -1, np.int16)
            for i, q in enumerate(self.persons):
                own = q.get("_own")
                if own:
                    c = np.array(list(own), np.int64)
                    owner[tuple(c.T)] = i
            A, B, SK, DB, who = [], [], [], [], []
            for i, p in enumerate(self.persons):
                if not p["alive"] or p["safe"]:
                    continue
                ear = p.get("_eye")
                if ear is None:
                    continue                  # (the legacy fallback floods;
                ear = (ear[0], ear[1],        #  skipped in the proto)
                       min(ear[2] + 1.0, self.shape[2] - 1.0))
                for s in sounds:
                    if s.get("who") == p["name"]:
                        continue
                    sp = self.pid(s["who"]) if s.get("who") else -1
                    A.append(s["at"]); B.append(ear)
                    SK.append((i, sp if sp is not None else -1))
                    DB.append(s["db"]); who.append((i, s))
            if A:
                _tau, kg = rays(self, A, B, owner=owner, skip=SK)
                lv = levels(self, A, B, DB, kg)
                for (i, s), l in zip(who, lv):
                    if i not in mine or l > mine[i][0]:
                        mine[i] = (float(l), s)
        self.shadow["us"].append(1e6 * (time.perf_counter() - t0))
        before = {i: dict(p.get("noise") or {}) for i, p in enumerate(self.persons)}
        t1 = time.perf_counter()
        super()._law_hear()
        self.shadow["legacy_us"].append(1e6 * (time.perf_counter() - t1))
        for i, p in enumerate(self.persons):
            if i not in mine:
                continue
            lvl, s = mine[i]
            want = None
            if lvl >= SOUND["notice_db"] and s["tick"] > before[i].get("tick", -1):
                want = (s["tick"], s["at"])
            got = p.get("noise")
            got = (got["tick"], got["at"]) if got and got != before[i] else None
            if want is not None or got is not None:
                self.shadow["pairs"] += 1
                if want == got:
                    self.shadow["agree"] += 1
                else:
                    self.shadow["disagree"].append((self.tick, i, want, got))


def scene():
    w = EvWorld(100, 80, 50, voxel_cm=5)
    w.fill(0, 100, 0, 80, 0, 1, STONE)
    w.fill(50, 51, 0, 80, 1, 44, STONE)                # wall, doorway 36..44
    w.mat[50, 36:44, 1:40] = AIR
    w.smass[50, 36:44, 1:40] = 0.0
    w.fill(30, 33, 60, 63, 20, 23, GLASS)              # a pane on a shelf...
    w.fill(30, 33, 60, 63, 19, 20, WOOD)
    ps = []
    for i, (x, y) in enumerate(((12, 40), (34, 30), (70, 40), (88, 52))):
        p = _person(w, x, y)
        p["name"] = f"p{i}"
        ps.append(p)
    ps[1]["goal"] = (40, 50)                           # someone walking
    return w, ps


def main():
    w, ps = scene()
    for t in range(120):
        if t == 20:
            w._say(ps[2], "fire")                      # a shout through the wall
        if t == 40:                                    # knock the shelf away
            w.mat[30:33, 60:63, 19:20] = AIR
            w.smass[30:33, 60:63, 19:20] = 0.0
        w.step()
        for p in ps:
            p["events"].clear()
    sh = w.shadow
    print(f"events logged: {w.log.next_id}  "
          f"verbs: {sorted({e.verb for e in w.log.rows})}")
    print(f"hearing decisions compared: {sh['pairs']}  agree: {sh['agree']}  "
          f"disagree: {len(sh['disagree'])}")
    for d in sh["disagree"][:5]:
        print("   ", d)
    us, lus = np.array(sh["us"]), np.array(sh["legacy_us"])
    busy = lus > 50
    print(f"_law_hear legacy: mean {lus.mean():.0f} us, on busy ticks "
          f"{lus[busy].mean() if busy.any() else 0:.0f} us; "
          f"batched (incl. owner grid): mean {us.mean():.0f} us, busy "
          f"{us[busy].mean() if busy.any() else 0:.0f} us")


if __name__ == "__main__":
    main()
