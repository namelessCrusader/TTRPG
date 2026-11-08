"""PROTOTYPE (scratch): task-level jobs driving the voxel per-limb menus in the
test_wounds `_duel` geometry, through the hooks in patch_hooks.py.
- pick() is PURE: a function of (situation, menu, job state); all mutation is
  in prepare() (called once per _decide).
- only the limbs the job step needs are asked; legs are never asked mid-walk.
- rows are matched by PAYLOAD (cell / tag / goal), never by label.
- a decision is requested at t and lands at t + L via intent.wake_at."""
import sys, math, random, time, collections, hashlib
sys.path.insert(0, sys.argv[1])
from src.voxel.sim import World, STONE, IRON, REFLEXES, ORIENTING, TablePolicy, WILL
from src.voxel.scenes import _person

L = WILL["decide_every"]
NAMES = {(int(IRON), 2e-4): "the axe"}        # interim: names DECLARED by the scene
STRIKE_BAND_M = (0.55, 0.75)                   # unsquared stance, test_wounds._duel docstring

def seed_of(*xs):
    return int(hashlib.sha1("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)

class MiniMind:
    name = "mind-proto"

    def __init__(self, sheets):
        self.sheets, self.table = sheets, TablePolicy()
        self.log, self.asked = [], {}
        self.requests = collections.Counter()

    def on_restore(self, w):
        for p in w.persons:
            p.setdefault("intent", {}).pop("pending", None)

    # ---------------------------------------------------------------- tasks
    def options(self, w, p, cells):
        sheet = self.sheets.get(p["name"], {})
        harm = sheet.get("harm", {})
        out = [("wait", {}, 0.2)]
        if not p.get("held"):
            for ob in w._objects_within_reach(p, cells):
                nm = NAMES.get((ob["mat"], round(float(w.edge[ob["cell"]]), 6)))
                if nm:
                    out.append(("take", {"target": nm, "cell": ob["cell"]},
                                0.8 * max(harm.values(), default=0.0)))
        vox_m = 0.1 * w.scale
        for q in w.persons:
            if q is p or not q["alive"]:
                continue
            d = math.dist(p["anchor"], q["anchor"]) * vox_m
            if STRIKE_BAND_M[0] <= d <= STRIKE_BAND_M[1]:
                if p.get("held"):
                    out.append(("strike", {"target": q["name"], "with": "held"}, 1.5 * harm.get(q["name"], 0)))
                out.append(("strike", {"target": q["name"]}, 0.5 * harm.get(q["name"], 0)))
        if p.get("fear", 0) > 0.3 and w.exits:
            for e in w.exits:
                if p["known"][int(e[0]), int(e[1])]:
                    out.append(("flee_to", {"goal": (int(e[0]), int(e[1]))}, p["fear"]))
        return out

    def compile(self, w, p, verb, roles):
        if verb == "take":
            return [{"limb": "hands", "match": ("cell", roles["cell"]), "until": "_took"}]
        if verb == "strike":
            return [{"limb": "hands", "match": ("tag", "swing"), "until": "_swung", "aim": roles["target"]}]
        if verb == "flee_to":
            return [{"limb": "legs", "match": ("goal", roles["goal"]), "until": "_arrived"}]
        return [{"limb": None, "until": "wake", "ticks": 40}]

    # ---------------------------------------------------------------- hooks
    def prepare(self, w, p, percept):
        it = p.setdefault("intent", {})
        asked = set()
        job = p.get("job")
        news = percept is not None and percept not in ORIENTING and w.attention == "onset" \
            and bool(p.get("_onset"))
        if news and percept in REFLEXES:
            asked |= {l for l in REFLEXES[percept]}          # the reflex tier, now, 0 model calls
        if news and percept == "struck":
            p["fear"] = 0.8                                  # a sheet-free stand-in for feeling
        # step tests on the running job, from STRUCTURED flags only
        if job and job["status"] == "running":
            st = job["steps"][job["i"]]
            flag = p.get(st["until"]) if st["until"] != "wake" else None
            if st["until"] == "wake":
                done = w.tick >= job["started"] + st["ticks"]
            else:
                done = flag is not None and flag[0] >= st.get("asked_at", 10**9)
            if done:
                job["i"] += 1
                if job["i"] >= len(job["steps"]):
                    job["status"] = "done"
            elif st.get("asked_at") is not None and w.tick - st["asked_at"] > 3 * L:
                job["status"], job["fail"] = "failed", f"no {st['until']} after asking"
        if news and job and job["status"] == "running":
            job["status"], job["fail"] = "stopped", f"interrupt:{percept}"
        job = p.get("job")
        # a decision lands
        pend = it.get("pending")
        if pend and w.tick >= pend["apply_at"]:
            it.pop("pending")
            verb, roles = pend["verb"], pend["roles"]
            p["job"] = job = {"verb": verb, "roles": roles, "steps": self.compile(w, p, verb, roles),
                              "i": 0, "status": "running", "started": w.tick, "why": pend["why"]}
            self.log.append((w.tick, p["name"], "start", verb, roles.get("target") or roles.get("goal")))
        # ask for a decision when the job is over and none is pending
        if (job is None or job["status"] != "running") and "pending" not in it:
            comp, sl = w._person_cells(p)
            import numpy as np
            cells = np.argwhere(comp); cells[:, 0] += sl[0].start; cells[:, 1] += sl[1].start
            opts = self.options(w, p, cells)
            rng = random.Random(seed_of("w0", p["name"], w.tick))
            z = [u / 0.2 for _, _, u in opts]; m = max(z)
            wts = [math.exp(x - m) for x in z]
            i = rng.choices(range(len(opts)), weights=wts)[0]
            reason = "idle" if job is None else f"{job['status']}:{job.get('fail', '')}"
            it["pending"] = {"verb": opts[i][0], "roles": opts[i][1], "apply_at": w.tick + L,
                             "why": reason}
            it["wake_at"] = w.tick + L
            self.requests[w.tick] += 1
            self.log.append((w.tick, p["name"], "request", reason, [o[0] for o in opts], opts[i][0]))
            if job is not None:
                p["job"] = None
        elif "pending" not in it:
            it.pop("wake_at", None)
        # the job's current step: ask its limb once
        job = p.get("job")
        if job and job["status"] == "running":
            st = job["steps"][job["i"]]
            if st["limb"] and st.get("asked_at") is None:
                st["asked_at"] = w.tick
                asked.add(st["limb"])
                if st.get("aim"):
                    q = next(q for q in w.persons if q["name"] == st["aim"])
                    dx, dy = q["anchor"][0] - p["anchor"][0], q["anchor"][1] - p["anchor"][1]
                    n = math.hypot(dx, dy) or 1.0
                    it["aim"] = (dx / n, dy / n)
            elif st["until"] == "wake":
                it["wake_at"] = job["started"] + st["ticks"]
        self.asked[p["name"]] = asked

    def limbs(self, w, p, percept):
        return self.asked.get(p["name"], set())

    def pick(self, sit, menu):                 # PURE
        p_job = sit.get("_job")                # (not used: the job is looked up below)
        limb = sit["limb"]
        st = self._step.get(sit["who"]) if hasattr(self, "_step") else None
        return self._pick(sit, menu)

    def bind(self, w):
        self.w = w

    def _pick(self, sit, menu):
        p = next(q for q in self.w.persons if q["name"] == sit["who"])
        job = p.get("job")
        limb = sit["limb"]
        if job and job["status"] == "running":
            st = job["steps"][job["i"]]
            if st["limb"] == limb:
                key, val = st["match"]
                for i, o in enumerate(menu):
                    if key == "cell" and o.get("what") and tuple(o["what"]["cell"]) == tuple(val):
                        return i
                    if key == "tag" and o["tag"] == val:
                        return i
                    if key == "goal" and o.get("goal") is not None and tuple(o["goal"]) == tuple(val):
                        return i
                return 0
        return self.table.pick(sit, menu)

def duel(stand=31, ticks=150):
    w = World(46, 20, 40, voxel_cm=5)
    w.fill(0, 46, 0, 20, 0, 1, STONE)
    w.exits = [(44, 10)]
    a = _person(w, 17, 10); a["name"] = "attacker"
    b = _person(w, stand, 10); b["name"] = "defender"
    w.fill(22, 24, 10, 11, 1, 2, IRON, edge=2e-4)
    w.attention = "onset"
    m = MiniMind({"attacker": {"harm": {"defender": 0.8}}})
    m.bind(w)
    w.policy = m
    t0 = time.perf_counter()
    for _ in range(ticks):
        w.step()
        a["events"].clear(); b["events"].clear()
    return w, a, b, m, time.perf_counter() - t0

w, a, b, m, dt = duel()
print(f"wall {dt:.1f}s for 150 ticks; persons {len(w.persons)}")
print("defender wounds open:", [wd["open"] for wd in b.get("wounds") or []], "hurt", round(b["hurt"], 3))
for row in m.log:
    print("  ", row)
print("requests per tick:", dict(m.requests))
dec = collections.Counter((r["who"], str(r["percept"])) for r in w.traces)
print("decides (who, percept):", dict(dec))
print("limbs asked per decide:", collections.Counter(len(r["menus"]) for r in w.traces))
