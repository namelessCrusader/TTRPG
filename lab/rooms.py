"""Randomised locked rooms, and an expert that only uses what it can read.

The question this exists for is not "can Laya escape a room out of the box"
(`room.py` says no) but "is escaping LEARNABLE by Laya": fine-tune it on an
expert's decisions in some rooms, then put it in rooms it has never seen, built
from names it has never seen, and count how far it gets.

So every room is drawn from one of two name pools, TRAIN or TEST, and they
share no container, furniture, loose thing, key or person. A model that learned
"go to the chest" fails a TEST room, which hides the key in a wardrobe; a model
that learned "go to the container the note names, open it, take the key" does
not.

THE EXPERT SEES ONLY THE STATE. It does not know where the key is until the
room tells it — by the note (walk to it, take it, read it) or by a person (walk
close, ask). So every label it writes is a function of text Laya is also
given, which is what makes it a fair thing to learn.
"""
from __future__ import annotations

import random
from collections import deque

STEPS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}

POOLS = {
    "train": dict(
        containers=[("the chest", "a heavy wooden chest"), ("the cupboard", "a tall pine cupboard"),
                    ("the crate", "a nailed shipping crate"), ("the trunk", "a leather travelling trunk"),
                    ("the barrel", "an empty ale barrel with a lid")],
        furniture=[("the bed", "a narrow bed"), ("the table", "an oak table"), ("the bench", "a long bench"),
                   ("the anvil", "a blacksmith's anvil"), ("the loom", "a weaving loom"),
                   ("the stove", "a cold iron stove")],
        items=["the candle", "the rope", "the loaf of bread", "the kitchen knife", "the tin cup",
               "the prayer book", "the ball of wool"],
        keys=["the iron key", "the brass key"],
        people=[("the old woman", "an old woman knitting in a chair"),
                ("the innkeeper's boy", "a boy sweeping the floor")],
        rooms=["an upstairs room in a roadside inn", "a storeroom behind a smithy",
               "a weaver's workshop"],
    ),
    "test": dict(
        containers=[("the wardrobe", "a carved walnut wardrobe"), ("the strongbox", "a squat iron strongbox"),
                    ("the sea chest", "a battered sailor's sea chest"), ("the dresser", "a painted dresser")],
        furniture=[("the harp", "a dusty harp"), ("the spinning wheel", "a spinning wheel"),
                   ("the bookcase", "a bookcase of mouldy books"), ("the hammock", "a hammock slung between posts"),
                   ("the globe", "a great brass globe")],
        items=["the lantern", "the flute", "the apple", "the hammer", "the map", "the bottle of ink"],
        keys=["the rusty key", "the silver key"],
        people=[("the fisherman", "a fisherman mending a net"),
                ("the sleepy guard", "a guard dozing on a stool")],
        rooms=["a captain's cabin below deck", "a lighthouse keeper's study", "a music room in a manor"],
    ),
}


class Room:
    # FORMULATION. How the same room is put to a model, not what is in it.
    #   enrich: an option says what it would bring you to, computed from where
    #           things are ("step west: one step closer to the chest, the note")
    #   view:   "full" lists every thing with its bearing; "focus" gives detail
    #           only for what is within 3 steps or named in what you know, and
    #           the rest by name. Both are computed from what the agent can see.
    enrich = False
    view = "full"

    def __init__(self, seed, pool="train"):
        rng = random.Random(seed)
        P = POOLS[pool]
        self.w, self.h = rng.randint(12, 17), rng.randint(8, 10)
        self.kind = rng.choice(P["rooms"])
        self.wall = {(x, y) for x in range(self.w) for y in range(self.h)
                     if x in (0, self.w - 1) or y in (0, self.h - 1)}
        side = rng.choice(["north", "south", "east", "west"])
        if side in ("north", "south"):
            dx = rng.randint(2, self.w - 3)
            door = (dx, 0 if side == "north" else self.h - 1)
        else:
            dy = rng.randint(2, self.h - 3)
            door = (0 if side == "west" else self.w - 1, dy)
        self.wall.discard(door)
        self.door_side = side
        self.things = {}                    # name -> dict(pos, desc, kind, ...)
        taken = {door}
        inner = [(x, y) for x in range(2, self.w - 2) for y in range(2, self.h - 2)]

        def spot():
            for _ in range(200):
                c = rng.choice(inner)
                if c not in taken and all(abs(c[0] - t[0]) + abs(c[1] - t[1]) > 1 for t in taken):
                    taken.add(c)
                    return c
            c = next(c for c in inner if c not in taken)
            taken.add(c)
            return c

        self.things["the door"] = dict(pos=door, desc="the only door, iron-bound", kind="door",
                                       closed=True, locked=True)
        conts = rng.sample(P["containers"], rng.randint(2, 3))
        for name, desc in conts:
            self.things[name] = dict(pos=spot(), desc=desc, kind="container", closed=True, inside=[])
        for name, desc in rng.sample(P["furniture"], rng.randint(1, 3)):
            self.things[name] = dict(pos=spot(), desc=desc, kind="furniture")
        self.key = rng.choice(P["keys"])
        self.key_box = rng.choice([n for n, _ in conts])
        self.things[self.key_box]["inside"].append(self.key)
        for name, _ in conts:                              # decoys inside the others
            if name != self.key_box:
                self.things[name]["inside"].append(rng.choice(P["items"]))
        for it in rng.sample(P["items"], rng.randint(2, 3)):
            if it not in self.things:
                self.things[it] = dict(pos=spot(), desc=None, kind="item")
        # how the room tells you where the key is: a note, a person, or both
        told = rng.choice(["note", "person", "both"])
        self.note = None
        if told in ("note", "both"):
            self.note = f"A scrawled note: 'Spare key is in {self.key_box}.'"
            self.things["the note"] = dict(pos=spot(), desc=None, kind="item")
        self.person = None
        if told in ("person", "both"):
            name, desc = rng.choice(P["people"])
            self.person = name
            self.things[name] = dict(pos=spot(), desc=desc, kind="person")
        self.pos = spot()
        self.holding, self.known, self.last = [], [], None
        self.out, self.done = False, set()

    # ------------------------------------------------------------ geometry
    def solid(self, c):
        if c in self.wall:
            return True
        for n, t in self.things.items():
            if t["pos"] == c and t["kind"] != "item":
                return not (t["kind"] == "door" and not t["closed"])
        return not (0 <= c[0] < self.w and 0 <= c[1] < self.h)

    def dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def near(self, name):
        return self.dist(self.things[name]["pos"], self.pos) <= 1

    def where(self, c):
        dx, dy = c[0] - self.pos[0], c[1] - self.pos[1]
        p = []
        if dy:
            p.append(f"{abs(dy)} {'north' if dy < 0 else 'south'}")
        if dx:
            p.append(f"{abs(dx)} {'east' if dx > 0 else 'west'}")
        return " and ".join(p) or "right here"

    # ------------------------------------------------------------ what it reads
    def state(self):
        s = [f"You are locked in {self.kind}. You want to get out.",
             "You are holding " + (", ".join(self.holding) or "nothing") + "."]
        seen = []
        for n, t in self.things.items():
            b = n
            if t["kind"] == "door":
                b += " (" + ("locked" if t["locked"] else "closed" if t["closed"] else "open") + ")"
            elif t["kind"] == "container":
                b += " (" + ("closed" if t["closed"] else "open, holding " + (", ".join(t["inside"]) or "nothing")) + ")"
            elif t["kind"] == "person":
                b += f" ({t['desc']})"
            seen.append(f"{b}: {self.where(t['pos'])}")
        # what you know and what just happened go BEFORE the list of things:
        # past 512 tokens the input is cut from the end, and the list is the
        # part that can lose a line without losing the plot
        if self.known:
            s.append("You know: " + " ".join(self.known))
        if self.last:
            s.append("A moment ago: " + self.last)
        if self.view == "focus":
            near, far = [], []
            told = " ".join(self.known + self.holding)
            for (n, t), line in zip(self.things.items(), seen):
                if self.dist(t["pos"], self.pos) <= 3 or n in told or t["kind"] == "door":
                    near.append(line)
                else:
                    far.append(n)
            s.append("Near you: " + ("; ".join(near) or "nothing") + ".")
            if far:
                s.append("Also in the room: " + ", ".join(far) + ".")
            return "\n".join(s)
        s.append("Around you: " + "; ".join(seen) + ".")
        return "\n".join(s)

    def menus(self):
        legs = {"stay": "stay where you are"}
        for d, (dx, dy) in STEPS.items():
            c = (self.pos[0] + dx, self.pos[1] + dy)
            if not self.solid(c):
                legs[f"step {d}"] = f"take one step {d}"
                if self.enrich:
                    closer = sorted((n for n, t in self.things.items()
                                     if self.dist(c, t["pos"]) < self.dist(self.pos, t["pos"])),
                                    key=lambda n: self.dist(self.pos, self.things[n]["pos"]))
                    if closer:
                        legs[f"step {d}"] = "one step closer to " + ", ".join(closer[:4])
        hands = {"keep": "do nothing with your hands"}
        adj = [n for n in self.things if self.near(n)]
        for n in adj:
            t = self.things[n]
            if t["kind"] == "item":
                hands[f"take {n}"] = f"pick up {n}"
            if t["kind"] in ("container", "door") and t["closed"] and not t.get("locked"):
                hands[f"open {n}"] = f"open {n}"
            if t["kind"] == "container" and not t["closed"]:
                for it in t["inside"]:
                    hands[f"take {it} from {n}"] = f"take {it} out"
        for h in self.holding:
            if h == "the note":
                hands["read the note"] = "read the note"
            for n in adj:
                if self.things[n]["kind"] in ("door", "container"):
                    hands[f"use {h} on {n}"] = f"use {h} on {n}"
        mouth = {"quiet": "say nothing"}
        if self.person and self.dist(self.things[self.person]["pos"], self.pos) <= 4:
            mouth[f"ask {self.person} about the key"] = "ask where the key is"
            mouth[f"greet {self.person}"] = "say hello"
        return {"legs": legs, "hands": hands, "mouth": mouth}

    def target_question(self):
        """The pointer: of everything in the room, what are you dealing with
        next? The list is the room, so it grows with the room."""
        crit = {n: None for n in self.things}
        if self.enrich:
            told = " ".join(self.known)
            for n, t in self.things.items():
                bits = [self.where(t["pos"])]
                if t["kind"] == "door":
                    bits.insert(0, "locked" if t["locked"] else "closed" if t["closed"] else "open")
                elif t["kind"] == "container":
                    bits.insert(0, "closed" if t["closed"] else "open")
                if n in told:
                    bits.append("you know the key is in it")
                crit[n] = ", ".join(bits)
        return {"type": "choice",
                "instructions": "Of everything here, what do you deal with next?",
                "criteria": crit}

    QUESTIONS = {"legs": "Which way do you step now?",
                 "hands": "What do your hands do now?",
                 "mouth": "What do you say now?"}

    def questions(self):
        m = self.menus()
        return {l: {"type": "choice", "instructions": q, "criteria": m[l]}
                for l, q in self.QUESTIONS.items()}

    # ------------------------------------------------------------ doing it
    def act(self, pick):
        ev = []
        h, m, l = pick.get("hands"), pick.get("mouth"), pick.get("legs")
        if h and h != "keep":
            ev.append(self._hands(h))
        if m and m.startswith("ask "):
            ev.append(f"{self.person} says the spare key is in {self.key_box}")
            fact = f"The key is in {self.key_box}."
            if fact not in self.known:
                self.known.append(fact)
            self.done.add("learned where the key is")
        elif m and m.startswith("greet "):
            ev.append(f"{self.person} nods")
        if l and l.startswith("step "):
            dx, dy = STEPS[l[5:]]
            c = (self.pos[0] + dx, self.pos[1] + dy)
            if not self.solid(c):
                self.pos = c
                if c == self.things["the door"]["pos"]:
                    self.out = True
                    self.done.add("out")
                    ev.append("you step out through the door")
            else:
                ev.append("something is in the way")
        if self.near(self.key_box):
            self.done.add("reached the key's container")
        if self.near("the door") or self.out:
            self.done.add("reached the door")
        did = [v for v in pick.values() if v not in ("stay", "keep", "quiet")]
        ev = [e for e in ev if e]
        self.last = ("you " + ", ".join(did) if did else "you did nothing") + (
            "; " + "; ".join(ev) if ev else "") + "."
        return ev

    def _hands(self, h):
        if h.startswith("take ") and " from " in h:
            it, box = h[5:].split(" from ")
            self.things[box]["inside"].remove(it)
            self.holding.append(it)
            if it == self.key:
                self.done.add("took the key")
            return f"you take {it}"
        if h.startswith("take "):
            it = h[5:]
            del self.things[it]
            self.holding.append(it)
            return f"you pick up {it}"
        if h.startswith("open "):
            n = h[5:]
            self.things[n]["closed"] = False
            if n == self.key_box:
                self.done.add("opened the key's container")
            if n == "the door":
                self.done.add("opened the door")
            if self.things[n]["kind"] == "container":
                return f"inside {n}: " + (", ".join(self.things[n]["inside"]) or "nothing")
            return f"you open {n}"
        if h == "read the note":
            fact = f"The key is in {self.key_box}."
            if fact not in self.known:
                self.known.append(fact)
            self.done.add("learned where the key is")
            return "it reads: " + self.note
        if h.startswith("use "):
            it, n = h[4:].split(" on ")
            if it == self.key and n == "the door":
                self.things["the door"]["locked"] = False
                self.done.add("unlocked the door")
                return "the key turns in the lock"
            return "nothing happens"
        return None

    # ------------------------------------------------------------ the expert
    def _step_toward(self, goal):
        """First step of a shortest walk to any free cell next to `goal`."""
        targets = {(goal[0] + dx, goal[1] + dy) for dx, dy in STEPS.values()}
        if self.pos in targets:
            return None
        seen, q = {self.pos: None}, deque([self.pos])
        while q:
            c = q.popleft()
            if c in targets:
                while seen[c] != self.pos:
                    c = seen[c]
                return next(d for d, v in STEPS.items()
                            if (self.pos[0] + v[0], self.pos[1] + v[1]) == c)
            for v in STEPS.values():
                n = (c[0] + v[0], c[1] + v[1])
                if n not in seen and not self.solid(n):
                    seen[n] = c
                    q.append(n)
        return None

    def expert(self):
        """What to do, from what the state says. Returns {limb: option}."""
        pick = {"legs": "stay", "hands": "keep", "mouth": "quiet"}
        self.target = None
        m = self.menus()
        door = self.things["the door"]
        knows = any(self.key_box in k for k in self.known)

        def go(name):
            self.target = name
            d = self._step_toward(self.things[name]["pos"])
            if d and f"step {d}" in m["legs"]:
                pick["legs"] = f"step {d}"

        if not knows and self.key not in self.holding:
            if self.person and f"ask {self.person} about the key" in m["mouth"]:
                self.target = self.person
                pick["mouth"] = f"ask {self.person} about the key"
            elif "the note" in self.holding:
                self.target = "the note"
                pick["hands"] = "read the note"
            elif "the note" in self.things:
                self.target = "the note"
                if self.near("the note"):
                    pick["hands"] = "take the note"
                else:
                    go("the note")
            else:
                go(self.person)
        elif self.key not in self.holding:
            box = self.things[self.key_box]
            self.target = self.key_box
            if not self.near(self.key_box):
                go(self.key_box)
            elif box["closed"]:
                pick["hands"] = f"open {self.key_box}"
            else:
                pick["hands"] = f"take {self.key} from {self.key_box}"
        elif door["locked"]:
            self.target = "the door"
            if self.near("the door"):
                pick["hands"] = f"use {self.key} on the door"
            else:
                go("the door")
        elif door["closed"]:
            self.target = "the door"
            if self.near("the door"):
                pick["hands"] = "open the door"
            else:
                go("the door")
        else:
            self.target = "the door"
            dx = door["pos"][0] - self.pos[0]
            dy = door["pos"][1] - self.pos[1]
            d = next(k for k, v in STEPS.items() if v == (dx, dy)) if abs(dx) + abs(dy) == 1 else None
            if d:
                pick["legs"] = f"step {d}"
            else:
                go("the door")
        return pick


MILESTONES = ["learned where the key is", "reached the key's container",
              "opened the key's container", "took the key", "reached the door",
              "unlocked the door", "opened the door", "out"]


def play(room, choose, ticks=60, log=None):
    """`choose(room, state, questions) -> {limb: option}`."""
    for t in range(ticks):
        st, qs = room.state(), room.questions()
        pick = choose(room, st, qs)
        ev = room.act(pick)
        if log is not None:
            log.append({"t": t, "state": st, "menus": {k: list(q["criteria"]) for k, q in qs.items()},
                        "pick": pick, "events": ev, "done": sorted(room.done)})
        if room.out:
            break
    return room


if __name__ == "__main__":
    ok = {"train": 0, "test": 0}
    for pool in ok:
        for seed in range(200):
            r = play(Room(seed, pool), lambda rm, s, q: rm.expert())
            ok[pool] += r.out
    print("expert escapes:", ok, "of 200 each")
    r = Room(1, "test")
    print(r.state())
    print(r.questions())
