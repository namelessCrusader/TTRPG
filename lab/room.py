"""One room, primitive actions, and nothing on the menu that solves anything.

    lab/.venv/bin/python -m lab.room laya --seeds 3
    lab/.venv/bin/python -m lab.room random --seeds 5
    lab/.venv/bin/python -m lab.room hand          # the scripted solution

`grid.py` offered "go to the key" and "unlock the door": options that each
read as the answer, so all a model had to do was match the goal's words. That
measured the wording, not the model. Here every option is a PRIMITIVE, the way
the sim's menus are built from what is physically there:

  legs   step north / south / east / west (only if that cell is free), stay
  hands  take X · drop X · open X · close X · search X · read X
         use <what you hold> on <what is next to you>
  mouth  quiet · ask <person> about <thing> · greet <person>

Unlocking the door is "use the iron key on the door", offered beside "use the
candle on the door" and "use the kitchen knife on the chest" — every held thing
on every adjacent thing, whether or not it does anything. Getting out takes a
chain: find where the key is (read the letter, or ask the old woman), walk to
the chest, open it, take the key, walk to the door, use the key on it, walk
out. Each link is recorded as a MILESTONE, so a run that gets halfway says how
far, instead of just losing.

Nothing here imports src/.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
STEPS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

# The room. '#' wall, '.' floor, letters are furniture (solid, named below).
ROWS = [
    "##################",
    "#B.....T.........#",
    "#B.....T.........#",
    "#......T....W....#",
    "#C...............D",
    "#................#",
    "#.....r.....S..F.#",
    "#................#",
    "##################",
]
FURNITURE = {
    "B": ("the bed", "a narrow bed with a straw mattress"),
    "T": ("the table", "a long oak table"),
    "C": ("the chest", "a heavy wooden chest at the foot of the bed"),
    "W": ("the washstand", "a washstand with a basin"),
    "S": ("the shelf", "a low shelf of dusty jars"),
    "F": ("the fireplace", "a cold fireplace full of ash"),
    "D": ("the door", "the room's only door, iron-bound oak"),
}
DESCRIPTION = ("A cramped upstairs room in a roadside inn, late at night. Rain "
               "on the shutters. The only way out is the iron-bound door in the "
               "east wall, and it has been locked from outside.")


class Room:
    def __init__(self, seed=0, memory=True):
        self.rng = random.Random(seed)
        self.memory = memory
        self.h, self.w = len(ROWS), len(ROWS[0])
        self.wall, self.furn = set(), {}
        for y, row in enumerate(ROWS):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.wall.add((x, y))
                elif ch in FURNITURE:
                    name = FURNITURE[ch][0]
                    self.furn.setdefault(name, {"cells": [], "desc": FURNITURE[ch][1]})
                    self.furn[name]["cells"].append((x, y))
        # state of things
        self.furn["the chest"].update(closed=True, inside=["the iron key", "a wool blanket"])
        self.furn["the door"].update(closed=True, locked=True)
        self.furn["the shelf"].update(inside=["a jar of salt"])
        self.furn["the bed"].update(inside=["a copper coin"])       # under the mattress
        self.floor = {                                              # loose things
            "the letter": (8, 2), "the candle": (8, 1), "the kitchen knife": (8, 3),
            "the rope": (3, 7), "the loaf of bread": (14, 5),
        }
        self.letter = ("The innkeeper's note: 'The spare key to this room is in "
                       "the chest at the foot of the bed.'")
        self.me = {"name": "Kael", "who": "a traveller, locked in his room",
                   "goal": "get out of the room", "pos": (10, 5), "holding": [],
                   "known": [], "last": None, "out": False}
        self.npcs = {
            "the old woman": {"pos": (13, 2), "desc": "an old woman in a rocking chair, awake, knitting",
                              "knows": {"the key": "The spare key? In the chest by the bed, dear.",
                                        "the door": "They lock it from outside at night.",
                                        "the chest": "There's a spare key in there, and a blanket."}},
            "the dog": {"pos": (6, 6), "desc": "a big grey dog asleep on the rug", "knows": {}},
        }
        self.done = set()

    # ---------------------------------------------------------- geometry
    def solid(self, c):
        if c in self.wall:
            return True
        for f in self.furn.values():
            if c in f["cells"]:
                return not (f is self.furn["the door"] and not f["closed"])
        return any(n["pos"] == c for n in self.npcs.values())

    def near(self, cells):
        x, y = self.me["pos"]
        return any(abs(cx - x) + abs(cy - y) <= 1 for cx, cy in cells)

    def where(self, c):
        dx, dy = c[0] - self.me["pos"][0], c[1] - self.me["pos"][1]
        parts = []
        if dy:
            parts.append(f"{abs(dy)} {'north' if dy < 0 else 'south'}")
        if dx:
            parts.append(f"{abs(dx)} {'east' if dx > 0 else 'west'}")
        return " and ".join(parts) if parts else "right here"

    def closest(self, cells):
        x, y = self.me["pos"]
        return min(cells, key=lambda c: abs(c[0] - x) + abs(c[1] - y))

    # ---------------------------------------------------------- what he knows
    def state(self):
        me = self.me
        out = [DESCRIPTION,
               f"You are {me['name']}, {me['who']}. You want to {me['goal']}."]
        out.append("You are holding " + (", ".join(me["holding"]) or "nothing") + ".")
        seen = []
        for name, f in self.furn.items():
            bits = f"{name} ({f['desc']}"
            if "closed" in f:
                bits += ", closed" if f["closed"] else ", open"
            if f.get("locked"):
                bits += ", locked"
            if f.get("searched") or ("closed" in f and not f["closed"]):
                bits += "; inside: " + (", ".join(f.get("inside", [])) or "nothing")
            seen.append(bits + f") is {self.where(self.closest(f['cells']))}")
        for name, c in self.floor.items():
            seen.append(f"{name} is {self.where(c)}")
        for name, n in self.npcs.items():
            seen.append(f"{name} ({n['desc']}) is {self.where(n['pos'])}")
        out.append("Around you: " + "; ".join(seen) + ".")
        if me["known"]:
            out.append("You know: " + " ".join(me["known"][-3:]))
        if self.memory and me["last"]:
            out.append("A moment ago: " + me["last"])
        return "\n".join(out)

    # ---------------------------------------------------------- the menus
    def menus(self):
        me = self.me
        legs = {"stay": "stay where you are"}
        for d, (dx, dy) in STEPS.items():
            c = (me["pos"][0] + dx, me["pos"][1] + dy)
            if not self.solid(c):
                legs[f"step {d}"] = f"take one step {d}"
        hands = {"keep": "do nothing with your hands"}
        adjacent = []
        for name, f in self.furn.items():
            if self.near(f["cells"]):
                adjacent.append(name)
                if "closed" in f:
                    hands[f"{'open' if f['closed'] else 'close'} {name}"] = \
                        f"{'open' if f['closed'] else 'close'} {name}"
                if "inside" in f and name != "the door":
                    hands[f"search {name}"] = f"look through {name} with your hands"
                    if (not f.get("closed")) or f.get("searched"):
                        for it in f["inside"]:
                            hands[f"take {it} from {name}"] = f"pick up {it}"
        for name, c in self.floor.items():
            if self.near([c]):
                adjacent.append(name)
                hands[f"take {name}"] = f"pick up {name}"
        for n, npc in self.npcs.items():
            if self.near([npc["pos"]]):
                adjacent.append(n)
        for h in me["holding"]:
            hands[f"drop {h}"] = f"put {h} down"
            if h == "the letter":
                hands["read the letter"] = "read what the letter says"
            for a in adjacent:
                if a != h:
                    hands[f"use {h} on {a}"] = f"use {h} on {a}"
        mouth = {"quiet": "say nothing"}
        for n, npc in self.npcs.items():
            d = abs(npc["pos"][0] - me["pos"][0]) + abs(npc["pos"][1] - me["pos"][1])
            if d <= 6:
                mouth[f"greet {n}"] = f"say hello to {n}"
                for topic in ("the key", "the door", "the chest"):
                    mouth[f"ask {n} about {topic}"] = f"ask {n} what they know about {topic}"
        return {"legs": legs, "hands": hands, "mouth": mouth}

    def questions(self):
        m = self.menus()
        if getattr(self, "no_null", False):      # every tick a real move
            for limb, null in (("legs", "stay"), ("hands", "keep")):
                if len(m[limb]) > 1:
                    m[limb].pop(null, None)
        return {l: {"type": "choice",
                    "instructions": q,
                    "criteria": m[l]}
                for l, q in (("legs", "Where does Kael step this moment?"),
                             ("hands", "What does Kael do with his hands this moment?"),
                             ("mouth", "What does Kael say this moment?"))}

    # ---------------------------------------------------------- doing it
    def act(self, pick):
        me, ev = self.me, []
        hands, mouth, legs = pick.get("hands"), pick.get("mouth"), pick.get("legs")
        if hands and hands != "keep":
            ev.append(self._hands(hands))
        if mouth and mouth != "quiet":
            ev.append(self._mouth(mouth))
        if legs and legs.startswith("step "):
            dx, dy = STEPS[legs[5:]]
            c = (me["pos"][0] + dx, me["pos"][1] + dy)
            if not self.solid(c):
                me["pos"] = c
                if c in self.furn["the door"]["cells"]:
                    me["out"] = True
                    self.done.add("out")
                    ev.append("you step out through the open door")
            else:
                ev.append(f"you could not step {legs[5:]}, something is in the way")
        if self.near(self.furn["the chest"]["cells"]):
            self.done.add("reached the chest")
        if self.near(self.furn["the door"]["cells"]):
            self.done.add("reached the door")
        ev = [e for e in ev if e]
        me["last"] = ("you did: " + ", ".join(v for v in pick.values()
                                            if v not in ("stay", "keep", "quiet"))
                      if any(v not in ("stay", "keep", "quiet") for v in pick.values())
                      else "you did nothing") + (". " + " ".join(ev) if ev else ".")
        return ev

    def _hands(self, h):
        me = self.me
        if h.startswith("take ") and " from " in h:
            it, box = h[5:].split(" from ")
            self.furn[box]["inside"].remove(it)
            me["holding"].append(it)
            if it == "the iron key":
                self.done.add("took the key")
            return f"you take {it}"
        if h.startswith("take "):
            it = h[5:]
            del self.floor[it]
            me["holding"].append(it)
            return f"you pick up {it}"
        if h.startswith("drop "):
            it = h[5:]
            me["holding"].remove(it)
            self.floor[it] = me["pos"]
            return f"you put {it} down"
        if h.startswith("open ") or h.startswith("close "):
            verb, name = h.split(" ", 1)
            f = self.furn[name]
            if verb == "open" and f.get("locked"):
                return f"{name} will not open; it is locked"
            f["closed"] = verb == "close"
            if name == "the chest" and verb == "open":
                self.done.add("opened the chest")
                return "you lift the lid: inside are " + ", ".join(f["inside"])
            if name == "the door" and verb == "open":
                self.done.add("opened the door")
            return f"you {verb} {name}"
        if h.startswith("search "):
            name = h[7:]
            f = self.furn[name]
            if f.get("closed"):
                return f"{name} is closed; you cannot search inside it"
            f["searched"] = True
            return f"you search {name} and find: " + (", ".join(f["inside"]) or "nothing")
        if h == "read the letter":
            me["known"].append(self.letter)
            self.done.add("learned where the key is")
            return "it reads: " + self.letter
        if h.startswith("use "):
            it, target = h[4:].split(" on ")
            if it == "the iron key" and target == "the door":
                self.furn["the door"]["locked"] = False
                self.done.add("unlocked the door")
                return "the key turns; the door is unlocked"
            if target == "the dog":
                return "the dog stirs and growls, then settles"
            return f"nothing happens when you use {it} on {target}"
        return None

    def _mouth(self, m):
        if m.startswith("greet "):
            return f"{m[6:]} nods at you" if "woman" in m else f"{m[6:]} does not answer"
        if m.startswith("ask "):
            who, topic = m[4:].split(" about ")
            said = self.npcs[who]["knows"].get(topic)
            if said:
                self.me["known"].append(f"{who} said: \"{said}\"")
                if topic in ("the key", "the chest"):
                    self.done.add("learned where the key is")
                return f"{who} says: \"{said}\""
            return f"{who} does not answer"
        return None


MILESTONES = ["learned where the key is", "reached the chest", "opened the chest",
              "took the key", "reached the door", "unlocked the door",
              "opened the door", "out"]


class RandomPick:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def load(self):
        pass

    def decide(self, state, questions):
        return {n: {"answer": self.rng.choice(list(q["criteria"]))} for n, q in questions.items()}


class HandPlayed:
    """The chain, played by a script that knows the room. It only proves the
    room can be won through the same menus a model gets."""
    def __init__(self, room):
        self.room = room

    def load(self):
        pass

    def decide(self, state, questions):
        r = self.room
        m = {l: q["criteria"] for l, q in questions.items()}
        pick = {"legs": "stay", "hands": "keep", "mouth": "quiet"}
        def toward(cells):
            tx, ty = r.closest(cells)
            x, y = r.me["pos"]
            for d, (dx, dy) in STEPS.items():
                if f"step {d}" in m["legs"] and abs(tx - x - dx) + abs(ty - y - dy) < abs(tx - x) + abs(ty - y):
                    return f"step {d}"
            return next((k for k in m["legs"] if k.startswith("step")), "stay")
        if "the iron key" not in r.me["holding"]:
            if "take the iron key from the chest" in m["hands"]:
                pick["hands"] = "take the iron key from the chest"
            elif "open the chest" in m["hands"]:
                pick["hands"] = "open the chest"
            else:
                pick["legs"] = toward(r.furn["the chest"]["cells"])
        elif r.furn["the door"]["locked"]:
            if "use the iron key on the door" in m["hands"]:
                pick["hands"] = "use the iron key on the door"
            else:
                pick["legs"] = toward(r.furn["the door"]["cells"])
        elif r.furn["the door"]["closed"]:
            pick["hands"] = "open the door" if "open the door" in m["hands"] else "keep"
            if pick["hands"] == "keep":
                pick["legs"] = toward(r.furn["the door"]["cells"])
        else:
            pick["legs"] = toward(r.furn["the door"]["cells"])
        return {k: {"answer": v} for k, v in pick.items()}


class Focused:
    """Laya asked the way its typed questions are meant to chain: first WHAT to
    deal with next (a pointer over the room's things), then each limb asked
    about THAT. The answer to the first question rides in the next ones'
    instructions — the sim's `already chosen` conditioning. Nothing about the
    room is written here; the target list is whatever the room contains."""
    def __init__(self):
        from lab.models import Laya
        self.m = Laya()

    def load(self):
        self.m.load()

    def decide_all(self, room, st, qs):
        things = {n: f"{n}, {room.where(room.closest(f['cells']))}" for n, f in room.furn.items()}
        things.update({n: f"{n}, {room.where(c)}" for n, c in room.floor.items()})
        things.update({n: f"{n}, {room.where(v['pos'])}" for n, v in room.npcs.items()})
        focus_q = {"focus": {"type": "choice",
                             "instructions": "To get out of the room, what should Kael deal with next?",
                             "criteria": things}}
        f = self.m.decide(st, focus_q)["focus"]
        target = f.get("answer")
        out, toks = {"focus": f}, {"focus": self.m.tokens}
        ask = {"legs": f"Which way does Kael step to reach {target}?",
               "hands": f"Kael is dealing with {target}. What does he do with his hands?",
               "mouth": f"Kael is dealing with {target}. What does he say?"}
        for k, q in qs.items():
            out.update(self.m.decide(st, {k: dict(q, instructions=ask[k])}))
            toks[k] = self.m.tokens
        return out, toks


def run(policy, seed, ticks=80, log=None, memory=True, no_null=False):
    r = Room(seed, memory=memory)
    r.no_null = no_null
    if isinstance(policy, HandPlayed):
        policy.room = r
    for t in range(ticks):
        qs, st = r.questions(), r.state()
        # ONE LIMB A CALL. All three questions in one call came to 1,139
        # tokens against Laya's 512, so most of what it was asked would have
        # been cut off unseen. Asked one at a time each fits; `tokens` in the
        # log says by how much.
        if isinstance(policy, Focused):
            ans, toks = policy.decide_all(r, st, qs)
        else:
            ans, toks = {}, {}
            for k, q in qs.items():
                ans.update(policy.decide(st, {k: q}))
                toks[k] = getattr(policy, "tokens", None)
        pick = {k: ans[k].get("answer") for k in qs}
        probs = {k: ans[k].get("p") for k in qs}
        ev = r.act(pick)
        if log is not None:
            log.append({"t": t, "state": st, "menus": {k: list(q["criteria"]) for k, q in qs.items()},
                        "pick": pick, "p": probs, "events": ev, "pos": r.me["pos"],
                        "tokens": toks,
                        "focus": (ans.get("focus") or {}).get("answer"),
                        "done": sorted(r.done)})
        if r.me["out"]:
            break
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--no-null", action="store_true",
                    help="take 'stay' and 'keep' off the menu whenever there is anything else")
    args = ap.parse_args()
    if args.policy == "random":
        pol = RandomPick(0)
    elif args.policy == "hand":
        pol = HandPlayed(None)
    elif args.policy == "laya-focus":
        pol = Focused()
    else:
        from lab.models import MODELS
        pol = MODELS[args.policy]()
    pol.load()
    out = HERE / "results" / "room"
    out.mkdir(parents=True, exist_ok=True)
    counts = {m: 0 for m in MILESTONES}
    for seed in range(args.seeds):
        if isinstance(pol, RandomPick):
            pol.rng.seed(seed)
        log = []
        r = run(pol, seed, log=log, memory=not args.no_memory, no_null=args.no_null)
        for m in MILESTONES:
            counts[m] += m in r.done
        with (out / f"{args.policy}{'+nonull' if args.no_null else ''}__{seed}.jsonl").open("w") as f:
            for row in log:
                f.write(json.dumps(row) + "\n")
        worst = max((v for row in log for v in row["tokens"].values() if v), default=0)
        print(f"seed {seed}: {len(log)} ticks, most tokens {worst}, reached: {', '.join(m for m in MILESTONES if m in r.done) or 'nothing'}")
    print(f"{args.policy}: " + " | ".join(f"{m} {counts[m]}/{args.seeds}" for m in MILESTONES))


if __name__ == "__main__":
    main()
