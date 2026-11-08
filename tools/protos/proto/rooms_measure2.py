"""Second-round numbers for the revised design (rooms, 6 nights x 300 turns):
- the confidence gate after the v2 pre-rank (kept menu) and on the full menu
- how often a NEEDS sentence would fire (drive active, the serving verb blocked)
- how often the inn's blocked list is empty
- duplicate option texts inside one menu
- where the rules' favourite sits in a score-sorted vs canonical order"""
import sys, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rooms_proto as rp
from rooms_prerank2 import prerank2
from mind_api import softmax
from lab.world2.policy import MID, T_RULES, utility, _utility, habit

GROUP = {"walk to": "move", "flee to": "move", "take": "handle", "steal": "handle", "drop": "handle",
         "give": "handle", "hide": "handle", "eat": "body", "drink": "body", "open": "handle", "play": "social",
         "strike": "fight", "shove": "fight", "threaten": "social", "insult": "social", "praise": "social",
         "tell": "social", "accuse": "social", "search": "social", "shout for help": "social",
         "help up": "care", "bandage": "care", "defend": "fight", "wait": "wait"}
gate = {g: [0, 0] for g in (0.5, 0.7, 0.9)}
n = 0
need = collections.Counter()
dups = 0
pos_sorted, pos_canon = [], []
orig = rp.Rooms.decide

def spy(self, p, reason):
    global n, dups
    w = self.w
    raw = w.options(p)
    opts = [rp.to_option(w, p, o) for o in raw]
    u = [utility(w, p, o, p.traits) for o in raw]
    ua = [utility(w, p, o, MID) for o in raw]
    keep = prerank2(opts, [max(a, b) for a, b in zip(u, ua)])
    pf = softmax(u, T_RULES)
    pk = softmax([u[i] for i in keep], T_RULES)
    for g in gate:
        gate[g][0] += max(pk) >= g
        gate[g][1] += max(pf) >= g
    texts = [raw[i]["text"] for i in keep]
    dups += len(texts) != len(set(texts))
    best = max(keep, key=lambda i: u[i])
    srt = sorted(keep, key=lambda i: -u[i])
    can = sorted(keep, key=lambda i: (GROUP.get(raw[i]["verb"], "z"), raw[i]["text"]))
    pos_sorted.append(srt.index(best)); pos_canon.append(can.index(best))
    held = w.held(p)
    food = any("food" in w.things[x]["props"] for x in held)
    drink = any(w.things[x]["props"] & {"ale", "water"} for x in held) or any(
        w.things[x]["props"] & {"ale_source", "water_source"} for x in w.loose(p.room))
    armed = bool(w.weapon(p))
    if p.hunger > .6 and not food:
        need["hungry, no food held"] += 1
    if p.thirst > .6 and not drink:
        need["thirsty, nothing to drink here"] += 1
    if p.fear > .4 and not armed and w.here(p):
        need["afraid, unarmed, people here"] += 1
    n += 1
    return orig(self, p, reason)

rp.Rooms.decide = spy
for seed in range(6):
    rp.port_night(seed, 300)
print(f"decisions {n}")
for g, (k, f) in gate.items():
    print(f"gate p>={g}: kept-menu {k/n:.2f}, full-menu {f/n:.2f}")
print("needs that would fire:", {k: f"{v/n:.2f}" for k, v in need.items()},
      f"any: {sum(need.values())} rows (may double count)")
print(f"menus with a duplicate text after pre-rank: {dups}")
c1 = collections.Counter(pos_sorted); c2 = collections.Counter(pos_canon)
print("rules favourite at position 0: score-sorted", f"{c1[0]/n:.2f}", "canonical", f"{c2[0]/n:.2f}",
      "| canonical position spread", dict(sorted(c2.items())[:8]))
