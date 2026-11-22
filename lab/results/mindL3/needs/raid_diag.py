"""A raid every night, all-rules cast, all5: how do raids end? seeds 1-3, 40 nights each (a raid on EVERY night: raid=1.0).
By the most people fighting at once: how often it is driven off, who is killed, the hp the fighters lose."""
import sys, collections
import numpy as np
from lab.world2.world import World
from lab.world2.lifetime import night_seed, NAMES
from lab.mindL.tournament import FastRules
pol = FastRules(); R = collections.defaultdict(list)
for seed in (1, 2, 3):
    carry = None
    for n in range(40):
        w = World(seed=night_seed(seed, n), flags="all5", carry=carry, night=n, raid=1.0, ill=0.0, luck=0.3)
        hp0 = {x: 10 for x in NAMES}
        mx = 0
        for _ in range(300):
            w.step(pol)
            mx = max(mx, sum(1 for p in w.people.values() if p.doing and p.doing["verb"] == "fend off"))
        beaten = any(e["action"] == "drives off" for e in w.log)
        fights = sum(1 for e in w.log if e["action"] == "fend off" and e["done"])
        hits = sum(1 for e in w.log if e["action"] == "raid hits")
        killed = sum(1 for e in w.log if (e.get("result") or "").startswith("killed") and e["actor"] in ("the wolf", "a raider"))
        R[min(mx, 3)].append((beaten, killed, fights, hits, sum(10 - w.people[x].hp for x in NAMES), (w._raid_at or 0)))
        carry = w.carry_out(NAMES)
print("most people fending at once | nights | driven off | people killed/night | fend acts | hits taken | total hp lost by the cast")
for k in sorted(R):
    a = np.array(R[k], float)
    print(f"  {k}{'+' if k == 3 else ' '}  {len(a):4d}  {a[:,0].mean():.2f}  {a[:,1].mean():.2f}  {a[:,2].mean():.1f}  {a[:,3].mean():.1f}  {a[:,4].mean():.1f}")
