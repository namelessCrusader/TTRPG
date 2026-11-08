"""A small 2D world with the sim's shape of choice, for watching a decision
model PLAY rather than answer a quiz.

    lab/.venv/bin/python -m lab.grid laya            # every scenario, 3 seeds
    lab/.venv/bin/python -m lab.grid random --seeds 5

What it keeps from the voxel sim is the part a model meets: every tick each
awake person is handed ONE STATE and a MENU PER LIMB — legs, hands, mouth —
whose options are built from what is physically there and carry their
argument ("go to the door (6 north)", "take hold of Pim", "say 'Fire!'").
Nothing the model says is an event until the world does it: fire spreads by
a rule, burns by a rule, a trade happens only if the other side accepts it,
a locked door opens only for the key. So "was that meaningful" can be read
off what HAPPENED, not off what was picked.

What it drops is everything a model does not see anyway: the lattice, heat,
mass. A cell is floor, wall, door or fire; a person is a position, hp, what
they hold, and a line of character.

Nothing here imports src/. It is a toy, and deliberately a different one from
the sim, so that "did the model do something sensible" is not tangled up with
"did the sim do something sensible".
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).parent
DIRS = {(0, -1): "north", (0, 1): "south", (1, 0): "east", (-1, 0): "west"}
HEAR = 8                                   # cells a voice carries


def _dir_words(dx, dy):
    ns = "north" if dy < 0 else "south" if dy > 0 else ""
    ew = "east" if dx > 0 else "west" if dx < 0 else ""
    return (ns + "-" + ew).strip("-") or "here"


class Grid:
    def __init__(self, rows, people, items, seed=0, fire_p=0.08):
        self.rng = random.Random(seed)
        self.fire_p = fire_p
        self.wall, self.fire, self.doors = set(), set(), {}
        self.h, self.w = len(rows), len(rows[0])
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.wall.add((x, y))
                elif ch == "F":
                    self.fire.add((x, y))
                elif ch in "DL":
                    self.doors[(x, y)] = {"locked": ch == "L"}
        self.people = {p["name"]: dict({"hp": 100, "holding": [], "awake": True,
                                        "out": False, "heard": [], "told": [],
                                        "carrying": None, "log": []}, **p)
                       for p in people}
        self.items = [dict(i) for i in items]    # {"name", "pos"} on the floor
        self.tick = 0
        self.said = []                           # (tick, who, text, pos)
        self.offers = []                         # pending trades
        self.memory = False                      # tell a body what it just did
        self.gives = 0                           # things handed over, ever
        self.as_prose = False                    # sentences instead of a form

    # ------------------------------------------------------------ geometry
    def free(self, c):
        x, y = c
        return (0 <= x < self.w and 0 <= y < self.h and c not in self.wall
                and not (c in self.doors and self.doors[c]["locked"]))

    def path_step(self, frm, to, avoid_fire=True):
        """One step along the shortest walkable route. Motor competence, like
        the sim's planner: knowing the way round the table is not a choice."""
        if frm == to:
            return frm
        occupied = {p["pos"] for p in self.people.values()
                    if not p["out"] and p["pos"] != frm}
        seen, q = {frm: None}, deque([frm])
        while q:
            c = q.popleft()
            if c == to:
                break
            for d in DIRS:
                n = (c[0] + d[0], c[1] + d[1])
                # the GOAL may be a locked door or a burning cell: you can walk
                # up to it, you just cannot walk into it (see `act`)
                if n in seen or (not self.free(n) and n != to):
                    continue
                if n != to and (n in occupied or (avoid_fire and n in self.fire)):
                    continue
                seen[n] = c
                q.append(n)
        if to not in seen:
            return None
        c = to
        while seen[c] != frm:
            c = seen[c]
        return c

    def dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def adjacent(self, a, b):
        return self.dist(a, b) <= 1

    # ------------------------------------------------------------ what a person sees
    def things(self, me):
        """Everything this person could name, with where it is from them."""
        out = []
        for c, d in self.doors.items():
            out.append(("door", "the door" + (" (locked)" if d["locked"] else ""), c))
        for it in self.items:
            out.append(("item", f"the {it['name']}", it["pos"]))
        for q in self.people.values():
            if q["name"] != me["name"] and not q["out"]:
                out.append(("person", q["name"], q["pos"]))
        return out

    def where(self, me, c):
        dx, dy = c[0] - me["pos"][0], c[1] - me["pos"][1]
        n = abs(dx) + abs(dy)
        return "right here" if n == 0 else f"{n} {_dir_words(dx, dy)}"

    def prose(self, me):
        """The same facts as `state`, written as sentences. Laya was trained on
        prose (emails, tickets) and reads a key: value form badly: in the form,
        a goal rewritten to "get the key" still sent Kael to the door (0.52);
        one plain sentence about the key sent him to it (0.74)."""
        n = me["name"]
        out = [f"{n} is {me['who']}. {n} wants to {me['goal'][0].lower() + me['goal'][1:]}."]
        if me["pos"] in self.fire:
            out.append(f"{n} is standing in the fire and burning.")
        elif me["hp"] < 100:
            out.append(f"{n} is hurt ({me['hp']} of 100).")
        hold = ", ".join(me["holding"])
        out.append(f"{n} is holding {hold}." if hold else f"{n}'s hands are empty.")
        if me["carrying"]:
            out.append(f"{n} is carrying {me['carrying']}.")
        if self.fire:
            near = min(self.fire, key=lambda c: self.dist(c, me["pos"]))
            out.append(f"A fire is burning {self.where(me, near)} of {n} and spreading.")
        for kind, name, c in self.things(me):
            w = self.where(me, c)
            if kind == "door":
                locked = self.doors[c]["locked"]
                out.append(f"The only way out is a door {w}"
                           + (", and it is locked; a key opens it." if locked else "."))
            elif kind == "item":
                out.append(f"{name[0].upper() + name[1:]} lies {w}.")
            else:
                q = self.people[name]
                bits = f"{name} ({q['who']}) is {w}"
                if not q["awake"]:
                    bits += ", lying unconscious and unable to move"
                if q["holding"]:
                    bits += ", holding " + ", ".join(q["holding"])
                out.append(bits + ".")
        for (tk, w, t, _) in me["heard"][-2:]:
            out.append(f"{w} shouted \"{t}\".")
        for o in self.offers:
            if o["to"] == me["name"]:
                out.append(f"{o['from']} offers {n} the {o['give']} in exchange for {n}'s {o['want']}.")
        out += [t + "." for t in me["told"][-2:]]
        if self.memory and me.get("last"):
            out.append(f"A moment ago {n} tried: {me['last']}.")
        return " ".join(out)

    def state(self, me):
        if self.as_prose:
            return self.prose(me)
        fire = sorted(self.fire, key=lambda c: self.dist(c, me["pos"]))
        lines = [f"you: {me['name']}, {me['who']}",
                 f"what you want: {me['goal']}",
                 f"health: {me['hp']}/100"
                 + (" — you are burning" if me["pos"] in self.fire else ""),
                 "holding: " + (", ".join(me["holding"]) or "nothing")
                 + (f"; carrying {me['carrying']}" if me["carrying"] else "")]
        if fire:
            lines.append(f"fire: nearest {self.where(me, fire[0])}, "
                         f"{len(self.fire)} cells burning")
        seen = []
        for kind, name, c in self.things(me):
            extra = ""
            if kind == "person":
                q = self.people[name]
                extra = (" lying still, unconscious" if not q["awake"] else "")
                extra += f" ({q['who']})"
                if q["holding"]:
                    extra += ", holding " + ", ".join(q["holding"])
            seen.append(f"{name}{extra}: {self.where(me, c)}")
        lines.append("you see: " + "; ".join(seen))
        heard = [f"{w} said \"{t}\"" for (tk, w, t, _) in me["heard"][-3:]]
        if heard:
            lines.append("you heard: " + "; ".join(heard))
        for o in self.offers:
            if o["to"] == me["name"]:
                lines.append(f"{o['from']} offers you {o['give']} for your {o['want']}")
        for t in me["told"][-2:]:
            lines.append(t)
        if self.memory and me.get("last"):
            lines.append("last moment you: " + me["last"])
        return "\n".join(lines)

    # ------------------------------------------------------------ the menus
    def menus(self, me):
        """Per limb, what this body could do right now. Built from what is
        there — the same rule as the sim: an option exists because the world
        makes it possible, not because a list of verbs says so."""
        legs = {"stay": "stay where you are"}
        hands = {"keep": "do nothing with your hands"}
        mouth = {"quiet": "say nothing"}
        for kind, name, c in self.things(me):
            if self.dist(c, me["pos"]) > 0 and self.path_step(me["pos"], c) is not None:
                legs[f"go to {name}"] = f"walk toward {name}, {self.where(me, c)}"
        if self.fire:
            legs["get away from the fire"] = "move to put distance between you and the flames"
            near = min(self.fire, key=lambda f: self.dist(f, me["pos"]))
            if self.dist(near, me["pos"]) > 1:
                legs["go toward the fire"] = f"walk up to the flames, {self.where(me, near)}"
        for it in self.items:
            if self.adjacent(it["pos"], me["pos"]):
                hands[f"pick up the {it['name']}"] = f"take the {it['name']} into your hands"
        for q in self.people.values():
            if q is me or q["out"] or not self.adjacent(q["pos"], me["pos"]):
                continue
            n = q["name"]
            if not q["awake"] and me["carrying"] != n:
                hands[f"take hold of {n}"] = f"lift {n} to carry them with you"
            if q["awake"]:
                hands[f"hit {n}"] = f"strike {n} with your fists"
            for h in me["holding"]:
                hands[f"give the {h} to {n}"] = f"hand {n} your {h}"
            for want in q["holding"]:
                for give in me["holding"]:
                    mouth[f"offer your {give} for {n}'s {want}"] = \
                        f"propose to trade your {give} for the {want} {n} holds"
                mouth[f"ask {n} for the {want}"] = f"ask {n} to hand over the {want}"
                mouth[f"threaten {n} for the {want}"] = f"warn {n} you will hurt them unless they give you the {want}"
            if q["awake"]:
                mouth[f"greet {n}"] = f"say hello to {n}"
        for o in self.offers:
            if o["to"] == me["name"]:
                hands[f"accept {o['from']}'s trade"] = f"give your {o['want']} and take the {o['give']}"
                mouth[f"refuse {o['from']}'s trade"] = f"tell {o['from']} no"
        for c, d in self.doors.items():
            if d["locked"] and self.adjacent(c, me["pos"]) and "key" in me["holding"]:
                hands["unlock the door"] = "use your key on the lock"
        if "bucket of water" in me["holding"] and any(
                self.dist(f, me["pos"]) <= 2 for f in self.fire):
            hands["throw the water on the fire"] = "douse the nearest flames"
        if me["carrying"]:
            hands[f"put down {me['carrying']}"] = f"set {me['carrying']} on the floor"
        if self.fire:
            mouth["shout 'Fire!'"] = "warn everyone about the fire"
        if any(not q["awake"] for q in self.people.values() if not q["out"]):
            mouth["shout for help"] = "call out for anyone to come help"
        return {"legs": legs, "hands": hands, "mouth": mouth}

    def questions(self, me):
        m = self.menus(me)
        return {
            "legs": {"type": "choice",
                     "instructions": f"You are {me['name']}. Where do your legs take you this moment?",
                     "criteria": m["legs"]},
            "hands": {"type": "choice",
                      "instructions": f"You are {me['name']}. What do your hands do this moment?",
                      "criteria": m["hands"]},
            "mouth": {"type": "choice",
                      "instructions": f"You are {me['name']}. What do you say this moment?",
                      "criteria": m["mouth"]},
        }

    # ------------------------------------------------------------ doing it
    def target(self, me, name):
        for kind, n, c in self.things(me):
            if n == name or n.startswith(name):
                return c
        return None

    def act(self, me, pick):
        ev = []
        legs, hands, mouth = pick.get("legs"), pick.get("hands"), pick.get("mouth")
        # hands first: grabbing happens where you stand
        if hands and hands.startswith("pick up the "):
            nm = hands[len("pick up the "):]
            it = next((i for i in self.items if i["name"] == nm), None)
            if it and self.adjacent(it["pos"], me["pos"]):
                self.items.remove(it)
                me["holding"].append(nm)
                ev.append(f"picks up the {nm}")
        elif hands and hands.startswith("take hold of "):
            n = hands[len("take hold of "):]
            me["carrying"] = n
            ev.append(f"lifts {n}")
        elif hands and hands.startswith("put down "):
            ev.append(f"puts {me['carrying']} down")
            me["carrying"] = None
        elif hands and hands.startswith("hit "):
            q = self.people[hands[4:]]
            q["hp"] -= 15
            q["told"].append(f"{me['name']} hit you")
            ev.append(f"hits {q['name']}")
        elif hands and hands.startswith("give the "):
            h, n = hands[len("give the "):].split(" to ")
            if h in me["holding"]:
                me["holding"].remove(h)
                self.people[n]["holding"].append(h)
                self.people[n]["told"].append(f"{me['name']} gave you the {h}")
                self.gives += 1
                ev.append(f"gives the {h} to {n}")
        elif hands and hands.startswith("accept "):
            frm = hands[len("accept "):].split("'s trade")[0]
            o = next((o for o in self.offers if o["from"] == frm and o["to"] == me["name"]), None)
            q = self.people[frm]
            if o and o["want"] in me["holding"] and o["give"] in q["holding"]:
                me["holding"].remove(o["want"]); q["holding"].append(o["want"])
                q["holding"].remove(o["give"]); me["holding"].append(o["give"])
                self.offers.remove(o)
                self.gives += 1
                q["told"].append(f"{me['name']} accepted your trade")
                ev.append(f"trades the {o['want']} to {frm} for the {o['give']}")
        elif hands == "unlock the door":
            for c, d in self.doors.items():
                if d["locked"] and self.adjacent(c, me["pos"]):
                    d["locked"] = False
                    ev.append("unlocks the door")
        elif hands == "throw the water on the fire":
            near = sorted(self.fire, key=lambda f: self.dist(f, me["pos"]))[:3]
            for f in near:
                self.fire.discard(f)
            me["holding"].remove("bucket of water")
            ev.append(f"douses {len(near)} burning cells")
        # mouth
        if mouth and mouth != "quiet":
            text = {"shout 'Fire!'": "Fire! Get out!",
                    "shout for help": "Help! Someone's down!"}.get(mouth, mouth)
            if mouth.startswith("offer your "):
                give, rest = mouth[len("offer your "):].split(" for ", 1)
                n, want = rest.split("'s ", 1)
                self.offers = [o for o in self.offers if o["from"] != me["name"]]
                self.offers.append({"from": me["name"], "to": n, "give": give, "want": want})
            if mouth.startswith("refuse "):
                frm = mouth[len("refuse "):].split("'s trade")[0]
                self.offers = [o for o in self.offers
                               if not (o["from"] == frm and o["to"] == me["name"])]
                self.people[frm]["told"].append(f"{me['name']} refused your trade")
            if mouth.startswith("threaten "):
                n = mouth.split(" ")[1]
                self.people[n]["told"].append(f"{me['name']} threatened you")
            if mouth.startswith("ask "):
                n = mouth.split(" ")[1]
                self.people[n]["told"].append(
                    f"{me['name']} asked you for the {mouth.split('the ', 1)[1]}")
            self.said.append((self.tick, me["name"], text, me["pos"]))
            ev.append(f"says \"{text}\"")
        # legs last
        if legs and legs != "stay":
            if legs == "go toward the fire" and not self.fire:
                nxt = None                   # the hands just put it out
            elif legs == "go toward the fire":
                near = min(self.fire, key=lambda f: self.dist(f, me["pos"]))
                nxt = self.path_step(me["pos"], near, avoid_fire=True)
            elif legs == "get away from the fire" and not self.fire:
                nxt = None
            elif legs == "get away from the fire":
                best = max(((self.path_step(me["pos"], (me["pos"][0] + d[0], me["pos"][1] + d[1]))
                             or me["pos"]) for d in DIRS),
                           key=lambda c: min((self.dist(c, f) for f in self.fire), default=0))
                nxt = best
            else:
                goal = self.target(me, legs[len("go to "):])
                nxt = self.path_step(me["pos"], goal) if goal else None
            if nxt and nxt != me["pos"] and (self.free(nxt) and nxt not in {
                    q["pos"] for q in self.people.values() if not q["out"] and q is not me
                    and q["name"] != me["carrying"]}):
                me["pos"] = nxt
                if me["carrying"]:
                    self.people[me["carrying"]]["pos"] = me["pos"]
                if nxt in self.doors and not self.doors[nxt]["locked"]:
                    me["out"] = True
                    ev.append("goes out the door")
                    if me["carrying"]:
                        self.people[me["carrying"]]["out"] = True
                        ev.append(f"carrying {me['carrying']} out")
        return ev

    def physics(self):
        """Fire spreads and burns. Nobody chose this."""
        new = set()
        for (x, y) in self.fire:
            for d in DIRS:
                n = (x + d[0], y + d[1])
                if self.free(n) and n not in self.doors and self.rng.random() < self.fire_p:
                    new.add(n)
        self.fire |= new
        for p in self.people.values():
            if p["out"]:
                continue
            if p["pos"] in self.fire:
                p["hp"] -= 25
            elif any(self.adjacent(p["pos"], f) for f in self.fire):
                p["hp"] -= 3
            if p["hp"] <= 0:
                p["hp"], p["awake"] = 0, False
        for (tk, w, t, pos) in self.said:
            if tk != self.tick:
                continue
            for p in self.people.values():
                if p["name"] != w and not p["out"] and self.dist(p["pos"], pos) <= HEAR:
                    p["heard"].append((tk, w, t, pos))

    def ascii(self):
        rows = []
        for y in range(self.h):
            r = ""
            for x in range(self.w):
                c = (x, y)
                who = [p for p in self.people.values() if p["pos"] == c and not p["out"]]
                if who:
                    r += who[0]["name"][0] if who[0]["awake"] else who[0]["name"][0].lower()
                elif c in self.wall:
                    r += "#"
                elif c in self.fire:
                    r += "^"
                elif c in self.doors:
                    r += "L" if self.doors[c]["locked"] else "D"
                elif any(i["pos"] == c for i in self.items):
                    r += "*"
                else:
                    r += "."
            rows.append(r)
        return "\n".join(rows)


# ---------------------------------------------------------------- scenarios
# Each says what a MEANINGFUL outcome would be, in world terms, so a run can
# be scored on what happened without reading a single pick.
SCENARIOS = {
    "fire_escape": dict(
        rows=["##########",
              "#........#",
              "#........D",
              "#..F.....#",
              "#........#",
              "##########"],
        people=[dict(name="Ana", who="a farmer", goal="stay alive", pos=(2, 2))],
        items=[],
        good=lambda g: g.people["Ana"]["out"],
        says="Ana gets out alive"),
    "rescue": dict(
        rows=["###########",
              "#.........#",
              "#.........D",
              "#.F.......#",
              "#.........#",
              "###########"],
        people=[dict(name="Ana", who="a farmer, brave, loves her brother",
                     goal="get herself and her brother Pim out alive", pos=(6, 2)),
                dict(name="Pim", who="a boy, Ana's brother", goal="-",
                     pos=(3, 2), awake=False)],
        items=[],
        good=lambda g: g.people["Ana"]["out"] and g.people["Pim"]["out"],
        says="Ana carries Pim out"),
    "locked_door": dict(
        rows=["##########",
              "#........#",
              "#........L",
              "#........#",
              "#.F......#",
              "##########"],
        people=[dict(name="Kael", who="a thief", goal="get out of the room", pos=(4, 2))],
        items=[dict(name="key", pos=(1, 1))],
        good=lambda g: g.people["Kael"]["out"],
        says="Kael fetches the key, unlocks the door, leaves"),
    "douse": dict(
        rows=["##########",
              "#........#",
              "#........#",
              "#.....F..#",
              "#........#",
              "##########"],
        people=[dict(name="Brenna", who="an innkeeper", goal="save her inn from burning",
                     pos=(2, 2))],
        items=[dict(name="bucket of water", pos=(1, 2)),
               dict(name="bucket of water", pos=(1, 3))],
        fire_p=0.03,
        good=lambda g: len(g.fire) == 0,
        says="Brenna fetches water and puts the fire out"),
    "market": dict(
        rows=["#########",
              "#.......#",
              "#.......#",
              "#########"],
        people=[dict(name="Dorn", who="a hungry hunter, honest", goal="get some bread",
                     pos=(2, 1), holding=["silver coin"]),
                dict(name="Mira", who="a baker, fair-minded, wants to sell",
                     goal="sell her bread for a fair price", pos=(6, 1),
                     holding=["loaf of bread"])],
        items=[],
        good=lambda g: ("loaf of bread" in g.people["Dorn"]["holding"]
                        and "silver coin" in g.people["Mira"]["holding"]
                        and g.gives <= 2),
        says="the bread and the coin change hands, once"),
    "guard": dict(
        rows=["#########",
              "#.......#",
              "#.......D",
              "#########"],
        people=[dict(name="Voss", who="a gate guard, stubborn, loyal to the city",
                     goal="let nobody through without the captain's pass", pos=(6, 2)),
                dict(name="Isra", who="a clever traveller with a silver coin",
                     goal="get through the door", pos=(2, 1), holding=["silver coin"])],
        items=[],
        good=lambda g: True,
        says="(no single right answer; read the log)"),
}


# ---------------------------------------------------------------- policies
class RandomPick:
    name = "random"

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def load(self):
        pass

    def decide(self, state, questions):
        return {n: {"answer": self.rng.choice(list(q["criteria"]))}
                for n, q in questions.items()}


def run(scn, policy, seed, ticks=40, log=None, memory=False, prose=False):
    s = SCENARIOS[scn]
    g = Grid(s["rows"], s["people"], s["items"], seed=seed,
             fire_p=s.get("fire_p", 0.08))
    g.memory = memory
    g.as_prose = prose
    for p in g.people.values():                 # a list per person, not shared
        p["holding"] = list(p.get("holding", []))
    for t in range(ticks):
        g.tick = t
        for me in g.people.values():
            if me["out"] or not me["awake"]:
                continue
            qs = g.questions(me)
            st = g.state(me)
            t0 = time.perf_counter()
            ans = policy.decide(st, qs)
            ms = (time.perf_counter() - t0) * 1000
            pick = {k: ans[k].get("answer") for k in qs}
            before = me["pos"]
            ev = g.act(me, pick)
            tried = [v for v in pick.values() if v not in ("stay", "keep", "quiet")]
            stuck = pick.get("legs") not in (None, "stay") and me["pos"] == before
            what = list(ev) + (["your legs could not take you any further"
                                if stuck else []][0:1] if stuck else [])
            me["last"] = ("; ".join(tried) or "did nothing") + " — " + (
                "; ".join(what) or "nothing happened")
            if log is not None:
                log.append({"t": t, "who": me["name"], "state": st,
                            "menus": {k: list(q["criteria"]) for k, q in qs.items()},
                            "pick": pick, "p": {k: ans[k].get("p") for k in qs},
                            "events": ev, "ms": ms})
        g.physics()
        if log is not None:
            log.append({"t": t, "frame": g.ascii()})
        if all(p["out"] or not p["awake"] for p in g.people.values()):
            break
    return g, s["good"](g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--memory", action="store_true",
                    help="tell each body what it did last tick and what came of it")
    ap.add_argument("--prose", action="store_true",
                    help="write the state as sentences instead of a key: value form")
    args = ap.parse_args()
    if args.policy == "random":
        pol = RandomPick(0)
    else:
        from lab.models import MODELS
        pol = MODELS[args.policy]()
    pol.load()
    out = HERE / "results" / "grid"
    out.mkdir(parents=True, exist_ok=True)
    for scn in SCENARIOS:
        if args.only and scn != args.only:
            continue
        wins = 0
        for seed in range(args.seeds):
            if isinstance(pol, RandomPick):
                pol.rng.seed(seed)
            log = []
            g, good = run(scn, pol, seed, log=log, memory=args.memory, prose=args.prose)
            wins += bool(good)
            tag = args.policy + ("+mem" if args.memory else "") + ("+prose" if args.prose else "")
            with (out / f"{scn}__{tag}__{seed}.jsonl").open("w") as f:
                for r in log:
                    f.write(json.dumps(r) + "\n")
        tag = args.policy + ("+mem" if args.memory else "") + ("+prose" if args.prose else "")
        print(f"{scn:12} {tag:15} {wins}/{args.seeds}  ({SCENARIOS[scn]['says']})")


if __name__ == "__main__":
    main()
