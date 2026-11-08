"""Print a grid run as a play-by-play: each pick with its probability, and
what the world then did.  lab/.venv/bin/python -m lab.show rescue laya 0"""
import json
import sys
from pathlib import Path

scn, pol, seed = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 99
rows = [json.loads(l) for l in open(Path(__file__).parent / "results" / "grid" / f"{scn}__{pol}__{seed}.jsonl")]
first = True
for r in rows:
    if r["t"] >= limit:
        break
    if "frame" in r:
        continue
    if first:
        print(r["state"]); print("menus:", json.dumps(r["menus"], indent=0)[:900]); first = False
    pk = " | ".join(f"{k}: {v}" + (f" ({r['p'][k]:.2f})" if r["p"].get(k) is not None else "")
                    for k, v in r["pick"].items() if v not in ("stay", "keep", "quiet"))
    print(f"t{r['t']:<3} {r['who']:6} {pk or '-'}   => {'; '.join(r['events']) or '-'}")
