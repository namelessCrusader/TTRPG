"""Two numbers for the design: how much of the rules policy's probability the
16-option pre-rank cuts away, and how many decisions a confidence gate would
answer without a model."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rooms_proto as rp
from mind_api import prerank, softmax
from lab.world2.policy import MID, T_RULES, utility

cut, gated, n, words_recent = [], {0.5: 0, 0.7: 0, 0.9: 0}, 0, 0
orig = rp.Rooms.decide
def spy(self, p, reason):
    global n
    w = self.w
    raw = w.options(p)
    opts = [rp.to_option(w, p, o) for o in raw]
    u = [utility(w, p, o, p.traits) for o in raw]
    ua = [utility(w, p, o, MID) for o in raw]
    keep = prerank(opts, [max(a, b) for a, b in zip(u, ua)], 16)
    pc = softmax(u, T_RULES)
    cut.append(1 - sum(pc[i] for i in keep))
    pk = softmax([u[i] for i in keep], T_RULES)
    for g in gated:
        gated[g] += max(pk) >= g
    n += 1
    return orig(self, p, reason)
rp.Rooms.decide = spy
for seed in range(6):
    rp.port_night(seed, 300)
cut.sort()
print(f"decisions {n}")
print(f"rules probability cut by the pre-rank: mean {sum(cut)/n:.4f}, p99 {cut[int(.99*n)]:.4f}, max {cut[-1]:.4f}")
for g, k in gated.items():
    print(f"gate: rules top option >= {g}: {k/n:.2f} of decisions need no model")
