"""A second pre-rank rule: score first until the rules mass is covered, then
one of each unseen verb, then by score."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rooms_proto as rp
from mind_api import softmax
from lab.world2.policy import MID, T_RULES, utility

def prerank2(opts, s, k=16, mass=0.95, first=12):
    n = len(opts)
    if n <= k:
        return list(range(n))
    order = sorted(range(n), key=lambda i: -s[i])
    p = softmax(s, T_RULES)
    keep, cum = [], 0.0
    for i in order:
        if cum >= mass or len(keep) >= first:
            break
        keep.append(i); cum += p[i]
    w = [i for i in order if opts[i].verb == "wait"]
    if w and w[0] not in keep:
        keep.append(w[0])
    seen = {opts[i].verb for i in keep}
    for i in order:
        if len(keep) >= k: break
        if opts[i].verb not in seen:
            keep.append(i); seen.add(opts[i].verb)
    for i in order:
        if len(keep) >= k: break
        if i not in keep: keep.append(i)
    return keep

cut, n = [], 0
orig = rp.Rooms.decide
def spy(self, p, reason):
    global n
    w = self.w
    raw = w.options(p)
    opts = [rp.to_option(w, p, o) for o in raw]
    u = [utility(w, p, o, p.traits) for o in raw]
    ua = [utility(w, p, o, MID) for o in raw]
    keep = prerank2(opts, [max(a, b) for a, b in zip(u, ua)])
    pc = softmax(u, T_RULES)
    cut.append(1 - sum(pc[i] for i in keep)); n += 1
    return orig(self, p, reason)
rp.Rooms.decide = spy
for seed in range(6):
    rp.port_night(seed, 300)
cut.sort()
print(f"prerank2: decisions {n}, rules mass cut mean {sum(cut)/n:.4f}, p99 {cut[int(.99*n)]:.4f}, max {cut[-1]:.4f}")
