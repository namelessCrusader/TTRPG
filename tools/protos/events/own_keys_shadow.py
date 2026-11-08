"""PROTOTYPE v2 (scratch only): batched hearing with the own-body exclusion
built from `_own` AT HEARING TIME (sorted int64 keys + searchsorted), not
from `_claim_bodies`' label. Run in SHADOW: before the real `_law_hear` runs,
compute what it will decide, batched; let it run; compare.

Covers what the first prototype did not: harvest worlds with falls, holds,
drags and unconscious listeners (seeds 0-59), the sleepers and the stroll in
tests/test_sound.py, and a few people scenes cut short.

    flock <lock> python3 own_keys_shadow.py [harvest|sound|scenes|all]
"""
import contextlib
import io
import sys
import tempfile
import time

import numpy as np

REPO = "/home/nisargparikh/Desktop/Fun Stuff/Mark_1"
sys.path.insert(0, REPO)
from src.voxel import sim as S, harvest, scenes          # noqa: E402
from src.voxel.sim import World, SOUND                   # noqa: E402

STAT = {"calls": 0, "pairs": 0, "level_max_diff": 0.0, "dec": 0,
        "dec_diff": [], "us_batched": [], "us_legacy": [], "keys_us": [],
        "rays": 0, "busy": 0, "unconscious_listener_ticks": 0,
        "asleep_listener_ticks": 0, "aloft_ticks": 0, "drag_ticks": 0}


def own_keys(w, q, N):
    """Sorted flat keys of q's _own, cached by IDENTITY of the frozenset
    (`_own` is always replaced, never mutated). The cache lives outside the
    person dict so nothing the sim reads or snapshots changes."""
    own = q.get("_own") or ()
    cache = w.__dict__.setdefault("_proto_ok", {})
    hit = cache.get(id(q))
    if hit is not None and hit[0] is own:
        return hit[1]
    if own:
        c = np.fromiter((v for cell in own for v in cell), np.int64,
                        count=3 * len(own)).reshape(-1, 3)
        k = np.sort(np.ravel_multi_index(c.T, w.shape))
    else:
        k = np.zeros(0, np.int64)
    cache[id(q)] = (own, k)
    return k


def batched_decisions(w):
    """What `_law_hear` will do this tick: {listener index: (lvl, snd)}."""
    sounds = [s for s in w.sounds if s["tick"] >= w.tick - 1]
    if not sounds or not w.persons:
        return {}
    N = int(np.prod(w.shape))
    t0 = time.perf_counter()
    K = [own_keys(w, q, N) for q in w.persons]
    STAT["keys_us"].append(1e6 * (time.perf_counter() - t0))
    by_name = {}
    for i, q in enumerate(w.persons):           # legacy: FIRST person with
        by_name.setdefault(q["name"], i)        # that name (sim.py:5557)
    A, B, DB, LI, SJ, SN = [], [], [], [], [], []
    for i, p in enumerate(w.persons):
        if not p["alive"] or p["safe"]:
            continue
        if not p["awake"]:
            STAT["unconscious_listener_ticks"] += 1
        if p.get("asleep"):
            STAT["asleep_listener_ticks"] += 1
        ear = p.get("_eye")
        if ear is None:
            return None                          # legacy floods; not batched
        ear = (ear[0], ear[1], min(ear[2] + 1.0, w.shape[2] - 1.0))
        for si, s in enumerate(sounds):
            if s.get("who") == p["name"]:
                continue
            j = by_name.get(s.get("who"), -1) if s.get("who") else -1
            A.append(s["at"]); B.append(ear); DB.append(s["db"])
            LI.append(i); SJ.append(j); SN.append(si)
    if not A:
        return {}
    A = np.asarray(A, np.float32); B = np.asarray(B, np.float32)
    LI = np.asarray(LI); SJ = np.asarray(SJ)
    STAT["rays"] += len(A)
    # interior samples, exactly as _level: linspace(0,1,n)[1:-1], round, clip
    n = np.ceil(np.abs(B - A).max(axis=1)).astype(np.int64) + 1
    k = np.where(n > 2, n - 2, 0)
    tot = int(k.sum())
    ray = np.repeat(np.arange(len(A)), k)
    start = np.cumsum(k) - k
    j = np.arange(tot) - np.repeat(start, k) + 1
    ts = j * (1.0 / np.maximum(np.repeat(n, k) - 1, 1))
    a = A[ray]
    pts = np.round(a + ts[:, None] * (B[ray] - a)).astype(np.int64)
    for ax in range(3):
        np.clip(pts[:, ax], 0, w.shape[ax] - 1, out=pts[:, ax])
    flat = np.ravel_multi_index(pts.T, w.shape)
    # OWN-BODY EXCLUSION from _own, now: one sorted array of pid-offset keys
    allk = np.concatenate([K[i] + i * N for i in range(len(K))]) \
        if len(K) else np.zeros(0, np.int64)
    allk.sort()

    def member(pid_per_sample):
        q = pid_per_sample * N + flat
        pos = np.searchsorted(allk, q)
        pos = np.minimum(pos, max(len(allk) - 1, 0))
        return (len(allk) > 0) & (allk[pos] == q) & (pid_per_sample >= 0)

    li = LI[ray]; sj = SJ[ray]
    excl = member(li) | member(sj)
    kg = np.bincount(ray, weights=np.where(excl, 0.0, w.smass[tuple(pts.T)]),
                     minlength=len(A)) / 1000.0
    vox_m = 0.1 * w.scale
    dist = np.maximum(np.linalg.norm(B - A, axis=1).astype(np.float64) * vox_m, 1.0)
    lvl = np.asarray(DB, np.float64) - 20.0 * np.log10(dist)
    sigma = kg / max(vox_m * vox_m, 1e-9)
    loss = np.where(sigma > 0, np.maximum(
        0.0, 20.0 * np.log10(np.maximum(sigma, 1e-30) * 500.0) - 47.0), 0.0)
    lvl = lvl - loss
    out = {}
    for r in range(len(A)):                      # first max wins, as legacy
        i = int(LI[r])
        if i not in out or lvl[r] > out[i][0]:
            out[i] = (float(lvl[r]), sounds[SN[r]], r)
    return out, lvl, A, B, DB, LI, SJ


def shadow_hear(self):
    STAT["calls"] += 1
    if any(b.get("owner") and not b.get("part") for b in self.bodies):
        STAT["aloft_ticks"] += 1
    if any(q.get("dragging") for q in self.persons):
        STAT["drag_ticks"] += 1
    t0 = time.perf_counter()
    got = batched_decisions(self)
    tb = time.perf_counter() - t0
    before = [(dict(p.get("noise") or {}), p.get("asleep")) for p in self.persons]
    t1 = time.perf_counter()
    World._law_hear_legacy(self)
    tl = time.perf_counter() - t1
    if got is None:
        return
    if got == {}:
        STAT["us_batched"].append(1e6 * tb); STAT["us_legacy"].append(1e6 * tl)
        return
    STAT["busy"] += 1
    STAT["us_batched"].append(1e6 * tb); STAT["us_legacy"].append(1e6 * tl)
    out, lvl, A, B, DB, LI, SJ = got
    # exact level check against the scalar _level with the legacy own set
    for r in range(len(A)):
        p = self.persons[int(LI[r])]
        own = set(p.get("_own") or ())
        if SJ[r] >= 0:
            own |= set(self.persons[int(SJ[r])].get("_own") or ())
        ref = self._level(A[r], B[r], DB[r], own=own or None)
        STAT["pairs"] += 1
        STAT["level_max_diff"] = max(STAT["level_max_diff"], abs(ref - lvl[r]))
    for i, (l, snd, _r) in out.items():
        p = self.persons[i]
        was_noise, was_asleep = before[i]
        if was_asleep:
            want = ("wake", l >= SOUND["wake_db"])
            have = ("wake", not p.get("asleep"))
        else:
            fires = l >= SOUND["notice_db"] and snd["tick"] > was_noise.get("tick", -1)
            want = ("noise", (snd["tick"], tuple(snd["at"])) if fires else None)
            nz = p.get("noise")
            changed = nz is not None and nz != was_noise
            have = ("noise", (nz["tick"], tuple(nz["at"])) if changed else None)
        STAT["dec"] += 1
        if want != have:
            STAT["dec_diff"].append((self.tick, p["name"], round(l, 2), want, have))


@contextlib.contextmanager
def patched():
    World._law_hear_legacy = World._law_hear
    World._law_hear = shadow_hear
    try:
        yield
    finally:
        World._law_hear = World._law_hear_legacy


def run_harvest(seeds):
    for seed in seeds:
        w, people, _ = harvest.world(seed)
        w.policy = harvest.Roam(seed)
        for _ in range(90):
            w.step()
            for p in people:
                p["events"].clear()


def run_sound():
    from tests import test_sound as T
    for name in dir(T):
        if name.startswith("test_"):
            getattr(T, name)()


def run_scenes(spec):
    saved = scenes._save
    scenes._save = lambda w, d, t: None
    try:
        for name, ticks in spec:
            with tempfile.TemporaryDirectory() as d, \
                    contextlib.redirect_stdout(io.StringIO()):
                scenes.SCENARIOS[name](d, ticks=ticks, every=10 ** 9)
            print(f"  scene {name} {ticks} ticks done", flush=True)
    finally:
        scenes._save = saved


def report(tag):
    ub, ul = np.array(STAT["us_batched"] or [0]), np.array(STAT["us_legacy"] or [0])
    print(f"[{tag}] _law_hear calls {STAT['calls']}, busy ticks {STAT['busy']}, "
          f"rays {STAT['rays']}", flush=True)
    print(f"  levels compared {STAT['pairs']}, max |dB| diff {STAT['level_max_diff']:.2e}")
    print(f"  decisions compared {STAT['dec']}, differ {len(STAT['dec_diff'])}")
    for d in STAT["dec_diff"][:8]:
        print("    ", d)
    print(f"  coverage: unconscious-listener ticks {STAT['unconscious_listener_ticks']}, "
          f"asleep-listener ticks {STAT['asleep_listener_ticks']}, aloft ticks "
          f"{STAT['aloft_ticks']}, drag ticks {STAT['drag_ticks']}")
    print(f"  cost per call: legacy mean {ul.mean():.0f} us (busy {ul[ul > 30].mean() if (ul > 30).any() else 0:.0f}); "
          f"batched mean {ub.mean():.0f} us (busy {ub[ul > 30].mean() if (ul > 30).any() else 0:.0f}); "
          f"own-keys part mean {np.mean(STAT['keys_us'] or [0]):.0f} us", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    with patched():
        if what in ("harvest", "all"):
            run_harvest(range(0, 60))
            report("harvest 0-59")
        if what in ("sound", "all"):
            run_sound()
            report("+ test_sound")
        if what in ("scenes", "all"):
            run_scenes([("rescue", 100), ("ledge", 100), ("parachute", 60),
                        ("arms", 100), ("swing", 60), ("arrow_trap", 40),
                        ("fire_alarm", 120)])
            report("+ scenes")
