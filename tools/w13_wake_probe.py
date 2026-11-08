"""W13 M4: what does a warm body cost per tick, and how much of that can an
exact gate or a quiet band take back?

T7 puts warm flesh in the lattice. Measured on base18 (one person, 100x80x50):
7 ms a tick with flesh at 20 C, 25.7 ms at 34 C. The extra is not the body.
`E.any()` is true, so `_pushed()` runs the whole pneumatic law, `_busy(E)` runs
the whole oxygen law, and the single hot window is the room. This is a probe
subclass of World (nothing in the repo is edited) with the fixes the T7 spec
names, each switchable:

  base        World as it is.
  o2gate      `_law_o2` returns at once when it can change nothing: oxygen is
              uniform and there is no smoke (EXACT: diffusion and overturn of a
              flat field are zero whatever the temperature).
  gas         o2gate, plus buoyancy `warm` read from GAS cells only (a solid is
              not a buoyant gas), plus `_pushed` asking only about gas cells and
              the pressure and velocity fields (still EXACT: any warm gas, any
              non-zero field, wakes the law).
  band1/band3 gas, plus a quiet band: `_pushed` is false while every gas cell is
              within 1 K (3 K) of ambient and |pressure| and |velocity| are
              under PV_BAND. NOT exact: a stated threshold, the T7 spec's
              decision 2.

For each variant, a room with N people (null policy: they stand and breathe),
once with flesh at 34 C ("warm") and once at ambient ("cold"), at 0.4 M and
1.6 M voxels. Wake cost = median warm tick - median cold tick, in ms per tick,
and per person.

    python3 tools/w13_wake_probe.py [--tiny] [--out results.json] [--note TEXT]
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.realpath(os.environ.get("MARK1_REPO", os.path.dirname(HERE)))
sys.path.insert(0, REPO)
RESULTS = os.environ.get("W13_RESULTS", "/home/nisargparikh/.cache/mark1-build/w13/M/results.json")
KEY = "wake_probe"

import numpy as np                                                # noqa: E402

from src.voxel import scenes                                       # noqa: E402
from src.voxel.sim import World, FLESH, STONE, AIR, AMBIENT        # noqa: E402

PV_BAND = 1e-3                        # pressure / velocity quiet band (field units)
VARIANTS = {                          # name -> (o2 gate, gas-only buoyancy, quiet band K or None)
    "base": (False, False, None),
    "o2gate": (True, False, None),
    "gas": (True, True, None),
    "band1": (True, True, 1.0),
    "band3": (True, True, 3.0),
}
SIZES = {"0.4M": (100, 80, 50), "1.6M": (200, 160, 50)}


# ── results file (same few lines in every w13 probe) ──────────────────────────
def record(path, key, row):
    """Merge `row` into the json at `path` under `key` (read, update, write)."""
    try:
        import fcntl
        lk = open(path + ".lk", "a")
        fcntl.flock(lk, fcntl.LOCK_EX)
    except (ImportError, OSError):
        lk = None
    try:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data[key] = row
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(path + ".tmp", path)
    finally:
        if lk is not None:
            lk.close()


_T_START = time.time()                # when this process began (after the lock, if one was used)


def load_note(extra=""):
    """Machine load note for a result: cpus, load average, how long the shared
    test lock was waited for (the driver sets W13_WAIT_START, epoch seconds)."""
    l1, l5, l15 = os.getloadavg()
    t = os.environ.get("W13_WAIT_START")
    wait = f"waited {int(_T_START - float(t))} s for the shared test lock; " if t else ""
    return (f"{wait}{os.cpu_count()} cpus, load average {l1:.1f}/{l5:.1f}/{l15:.1f} at finish; "
            f"shared with wave 12 and other jobs, so timings are noisy"
            + (f"; {extra}" if extra else ""))



# ── the probe subclass ────────────────────────────────────────────────────────
def _old_warm_cells(self):
    """The wave 12 buoyancy line: every cell's own temperature over ambient, solids
    too. The base variant uses it; the gas variant uses World._warm_cells (C6 made
    that gas-only), so the probe overrides the seam and patches no source text."""
    return self._cells(np.clip(self.T() - AMBIENT, 0.0, 500.0).astype(np.float32))


class Probe(World):
    gate_o2, gas_buoy, band_K = False, False, None

    def _law_o2(self):
        if self.gate_o2 and not (self._mixing() or bool(self.smoke.any())):
            return None
        return World._law_o2(self)

    def _warm_cells(self):
        if self.gas_buoy:
            return World._warm_cells(self)
        return _old_warm_cells(self)

    def _pushed(self):
        if not self.gas_buoy:
            return World._pushed(self)
        if not self.skip_quiet or bool(self.p_add.any()):
            return True
        air = self.mat == AIR
        if self.band_K is None:
            if bool(self.E[air].any()):
                return True
            thr = 0.0
        else:
            if float(self.E[air].max(initial=0.0)) > self.band_K * float(self.c_airbase):
                return True
            thr = PV_BAND
        if self.pcell is None:
            return False
        return (float(np.abs(self.pcell).max()) > thr
                or any(float(np.abs(v).max()) > thr for v in self.vcell))


def make_world(variant, size, npeople, warm, tiny=False):
    gate, gas, band = VARIANTS[variant]
    if tiny:
        w = Probe(16, 16, 16, voxel_cm=5)
        w.fill(0, 16, 0, 16, 0, 1, STONE)
        w.fill(6, 9, 6, 9, 1, 5, FLESH)
    else:
        nx, ny, nz = size
        w = Probe(nx, ny, nz, voxel_cm=5)
        w.fill(0, nx, 0, ny, 0, 1, STONE)
        w.fill(10, nx - 10, 10, ny - 10, 0, 45, STONE)
        w.mat[11:nx - 11, 11:ny - 11, 1:44] = AIR
        w.smass[11:nx - 11, 11:ny - 11, 1:44] = 0.0
        for i in range(npeople):
            scenes._person(w, 18 + 14 * (i % 5), 18 + 14 * (i // 5))
        w.policy = scenes._Wants()            # they stand and breathe
    w.gate_o2, w.gas_buoy, w.band_K = gate, gas, band
    if warm:
        C = w.heat_capacity()
        f = (w.mat == FLESH)
        w.E[f] = (np.float32(34.0) - AMBIENT) * C[f]
    return w


def tick_ms(variant, size, npeople, warm, ticks, skip, tiny):
    w = make_world(variant, size, npeople, warm, tiny)
    ts = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        w.step()
        ts.append((time.perf_counter() - t0) * 1e3)
        for p in w.persons:
            p["events"].clear()
    return statistics.median(ts[skip:]), statistics.mean(ts[skip:])


def run(tiny=False, note=""):
    t_all = time.perf_counter()
    ticks, skip = (6, 2) if tiny else (40, 16)
    sizes = {"tiny": (16, 16, 16)} if tiny else SIZES
    counts = (1,) if tiny else (1, 4, 12)
    notrun = {}
    table = {}
    for vname in VARIANTS:
        table[vname] = {}
        for sname, size in sizes.items():
            table[vname][sname] = {}
            for n in counts:
                cold, _ = tick_ms(vname, size, n, False, ticks, skip, tiny)
                warm, warm_mean = tick_ms(vname, size, n, True, ticks, skip, tiny)
                table[vname][sname][str(n)] = {
                    "cold_ms": round(cold, 2), "warm_ms": round(warm, 2),
                    "warm_mean_ms": round(warm_mean, 2),
                    "wake_ms_per_tick": round(warm - cold, 2),
                    "wake_ms_per_tick_per_person": round((warm - cold) / n, 2)}
    first = next(iter(sizes))
    base = table["base"][first][str(counts[0])]
    best = min(((v, table[v][s][str(n)]["wake_ms_per_tick_per_person"])
                for v in VARIANTS for s in sizes for n in counts if (s, n) == (first, counts[0])),
               key=lambda x: x[1])
    return {"status": "ok", "value": base["wake_ms_per_tick_per_person"],
            "units": "ms per tick per person (warm minus cold flesh), base variant, "
                     f"{counts[0]} person, {first} voxels",
            "best_variant_at_that_point": {"variant": best[0], "ms_per_tick_per_person": best[1]},
            "pv_band": PV_BAND, "ticks": ticks, "ticks_skipped": skip,
            "variants": {k: {"o2gate": v[0], "gas_only_buoyancy": v[1], "quiet_band_K": v[2]}
                         for k, v in VARIANTS.items()},
            "table": table, "not_run": notrun,
            "seconds": round(time.perf_counter() - t_all, 1), "load_note": load_note(note)}


def main(argv):
    tiny = "--tiny" in argv
    out = argv[argv.index("--out") + 1] if "--out" in argv else RESULTS
    note = argv[argv.index("--note") + 1] if "--note" in argv else ""
    row = run(tiny, note)
    record(out, KEY + ("_tiny" if tiny else ""), row)
    print(json.dumps(row, indent=1)[:5000])
    return row


if __name__ == "__main__":
    main(sys.argv[1:])
