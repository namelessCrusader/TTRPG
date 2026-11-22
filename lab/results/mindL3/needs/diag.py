"""Did each mechanism do anything? (a check of the mechanism, not a pass rule). usage: diag.py
Reads runs_<world> dirs next to the cache; the tend and allies checks use the night's own schedule (same generator as the world)."""
import json, sys
from pathlib import Path
import numpy as np
from lab.world2.lifetime import night_seed
from lab.world2.world import World, ALL4
C = Path("/home/nisargparikh/.cache/mark1-build/mindL3/needs")
BASE = Path("/home/nisargparikh/Desktop/Fun Stuff/Mark_1/lab/results/mindL3/luck/runs_v03")
CH = ("hermit", "rules", "always-giving", "tit-for-tat", "protector", "hoarder")
def runs(d, world, ch):
    return [json.load(open(d / f"A_{world}_{ch}_seed{s}_n100.json")) for s in range(1, 11) if (d / f"A_{world}_{ch}_seed{s}_n100.json").exists()]
which = sys.argv[1] if len(sys.argv) > 1 else "all"
for w, flag in (("tend", "tend"), ("allies", "allies"), ("all5", "all5")):
    d = C / ("runs_" + w)
    if not d.exists(): continue
    print(f"\n== {w}: focal error on nights that had the event vs not (nights 20-99, 10 seeds); all4 error on the same nights in brackets")
    for ch in CH:
        rs, bs = runs(d, w, ch), runs(BASE, "luck", ch)
        if not rs: continue
        rows = {"ill": ([], []), "raid": ([], [])}
        for r, b in zip(rs, bs):
            f = r["focal"]
            for n in range(20, 100):
                wd = World(seed=night_seed(r["seed"], n), flags="all5", luck=0.3)
                ill = wd.people[f].ail > 0
                raid = wd._raid_at is not None
                rows["ill"][ill].append((r["err"][n], b["err"][n]))
                rows["raid"][raid].append((r["err"][n], b["err"][n]))
        fl = lambda x: f"{np.mean([a for a, _ in x]):.3f} ({np.mean([b for _, b in x]):.3f}) n={len(x)}"
        if w in ("tend", "all5"): print(f"  {ch:14s} ill night {fl(rows['ill'][1])}  well night {fl(rows['ill'][0])}")
        if w in ("allies", "all5"): print(f"  {ch:14s} raid night {fl(rows['raid'][1])}  quiet night {fl(rows['raid'][0])}")
        if w in ("tend", "all5"):
            n = 80 * len(rs)
            print(f"  {'':14s} focal tends {sum(r['made'].get('tend',0) for r in rs)/n:.3f}/night, is tended {sum(r['recv'].get('tend',0) for r in rs)/n:.3f}/night; fend off {sum(r['made'].get('fend off',0) for r in rs)/n:.3f}/night")
for w in ("labour", "all5"):
    d = C / ("runs_" + w)
    if not d.exists(): continue
    print(f"\n== {w}: focal drive-error split (mean over seeds), energy part; all4 in brackets")
    for ch in CH:
        rs, bs = runs(d, w, ch), runs(BASE, "luck", ch)
        if rs: print(f"  {ch:14s} energy {np.mean([r['split']['energy'] for r in rs]):.3f} ({np.mean([r['split']['energy'] for r in bs]):.3f})  safety {np.mean([r['split']['safety'] for r in rs]):.3f} ({np.mean([r['split']['safety'] for r in bs]):.3f})  eats/night {np.mean([r['made'].get('eat',0) for r in rs])/80:.2f} ({np.mean([r['made'].get('eat',0) for r in bs])/80:.2f})")
