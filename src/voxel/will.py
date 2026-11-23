"""Menus, decisions, traces: the seam where a policy picks what each limb does."""
import math
import numpy as np
from . import mechanisms as mech
from . import seam
from .beliefs import HARM, view
from .events import _Pending
from .materials import (AIR, AMBIENT, BODY, FLESH, GRAVITY, LEAF, MATNAME, SOUND, TICK_S)
from src.mind.tourney import merge
from . import routing
from .policy import (ACTS, ANSWERS, CUES, FLINCH_ON, GAZE_SWEEP, LIMBS, LINES, MEANS, LINE_VOICE, MENU_BATCH, MENU_CAP, ORIENTING, PREMOVE_MIN_S, PREMOVE_ON, REFLEXES, VOICE, WANDER, WILL, rank, sheet)


# HOW LONG A BODY MUST BE WATCHED to be seen to run, in ticks (0.3 s): long
# enough that a walk and a run differ by more than the cell a body's middle
# can jump as it steps, at 10 cm as well as at 5.
WILL_TRAIL = 12


class WillMixin:

    # BATCHED THEN MERGED (wave 14b, lane F; ON at MENU_BATCH since 14bF). None is
    # the old law: a menu is cut to MENU_CAP by attention. A number B turns the cut
    # off for a policy that says it is `batchable` (a pure chooser; a mind or a
    # script keeps the old cut): a menu of B rows or fewer is picked as before, a
    # longer one is a tournament (`_merged_pick`).
    menu_batch = MENU_BATCH

    # DOORS IN THE MENU (wave 15, S3 item 1). `mech_menu`: a body next to a
    # door, lid or bar is offered rows for it (open or close, turn a key, work
    # the lock, seat or lift a bar, put a shoulder to it, kick it), and a row
    # to walk to one it has seen. `door_route`: a CLOSED door the body has not
    # found stuck is a passable edge of the route planner, with the cost of
    # opening it (the swing law), and the row that walks to the door says what
    # lies beyond it. Both are off in a world with no mechanism; `add_door`
    # turns them on (the ledger rows mech_menu and door_route).
    mech_menu = False
    door_route = False

    def _menu(self, p, cells, percept, limb, chosen, cap=None):
        """Everything ONE PART of this body could actually do, right now.

        Legality is CHECKED, not assumed. A place is only offered if a
        breadth-first route to it really exists, so an option that gets picked
        can never fail to happen. That is the property worth paying for:
        whatever is choosing cannot invent a door, because it never sees one
        that isn't there.

        One menu per limb, rather than one menu of every combination. Three
        parts with eight, three and two things to do is forty-eight
        combinations and thirteen rows, and the thirteen say everything the
        forty-eight do. That is not a saving trick — it is the shape a body
        actually has, and it is why nothing has to be written down twice when
        a new way to use a mouth arrives.

        Index 0 is always the NULL act: the part does nothing, which is always
        legal and never a failure. Hands doing nothing keep their grip."""
        menu = []
        danger = p.get("danger")
        # WHAT A MIND LINKED TO THIS BODY MEANS TO DO (3.5), if one is: rows it
        # wants are never cut, and a thing it means to take is offered even
        # when it is not one of the nearest. Nothing sets it otherwise.
        intent = p.get("intent") or {}
        if limb == "legs":
            menu.append({"key": "stay", "tag": "stay", "verb": "stay"})
            # a CARRIED thing is part of the walker, to the planner as much as
            # to the legs: its column is not an obstacle to its own carrier,
            # or "straight on" vanishes from the menu the moment the fist
            # closes on a stick hanging in front of the chest
            obj = self._held_cells(p) if p.get("held") else None
            nav = np.concatenate([cells, obj]) if obj is not None \
                and not self._underfoot(obj) else cells
            fit, _start = self._fit_grid(nav, p)
            ground = self._underfoot(cells)
            # (and the spots a linked mind means to strike from, 3.5 M10)
            places = self._places(p, cells, fit, intent.get("seek") or ())
            room = self._room(p, cells)
            places = self._wander_in_room(p, cells, fit, places, room)
            leaps = self._leap_targets(p, cells, fit, self._leap_speed(cells, p)) \
                if ground and self.tick - p.get("_landed", -999) \
                >= self._gather_ticks(p) else []
            # THE FEET OF THE CLOSED DOORS ARE ASKED IN THE SAME SWEEP as the
            # places (one more target each, so no other route moves)
            doorplan = self._door_plan(p, cells, fit) \
                if self.mech_menu and self.door_route and self.__dict__.get("mechs") \
                else None
            routes = self._routes(cells, [pl["xy"] for pl in places]
                                  + [lp["xy"] for lp in leaps]
                                  + ([] if doorplan is None else doorplan["targets"]),
                                  p, fit=fit)
            # you cannot push off thin air, and you cannot push off at all
            # until the legs have gathered from the last landing
            gathered = self.tick - p.get("_landed", -999) >= self._gather_ticks(p)
            if ground and gathered:
                menu.append({"key": "jump", "tag": "jump", "verb": "jump"})
            for lp in leaps:
                if routes.get(lp["xy"]):
                    continue                  # it could just WALK there. A leap
                                              # is worth weighing where the legs
                                              # cannot carry it — over a gap, not
                                              # across an open floor — and four
                                              # redundant leaps crowded a body's
                                              # whole attention out of the menu
                menu.append({"key": f"leap({lp['label']})", "tag": "leap",
                             "verb": "jump", "goal": lp["xy"],
                             "away": lp["away"]})
            for pl in places:
                route = routes.get(pl["xy"])
                if not route:
                    continue                  # no route it knows of: not legal
                # HOW BAD THE WAY THERE LOOKS, from what this body remembers
                # seeing. A place is as dangerous as the worst step on the way
                # to it — which is the right question, and a cheaper one than
                # asking about the destination, because the route is already
                # in hand. The sim states the fact; whether to mind it is the
                # policy's business, and a body with no taste at all still
                # gets the same menu.
                risk = 0.0
                if danger is not None:
                    risk = float(max(danger[int(x), int(y)] for x, y in route))
                row = {"key": f"go({pl['label']})", "tag": pl["tag"],
                       "verb": "go", "goal": pl["xy"], "route": route,
                       "away": pl["away"], "danger": round(risk, 3)}
                if percept is None and pl["tag"] in WANDER:
                    # whether the whole way stays in the room it stands in:
                    # a fact of the floor, for a body with nothing to do
                    row["inside"] = self._in_room(room, route)
                if "who" in pl:               # a spot to strike someone from:
                    row["who"] = pl["who"]    # whom it is for, and that it is
                    row["mid"] = True         # where the MIDDLE is to stand
                menu.append(row)
            if self.mech_menu and self.__dict__.get("mechs"):
                menu += self._door_rows(p, cells, fit, places, routes, ground,
                                        doorplan)
        elif limb == "hands":
            # WHAT IT IS CALLED, said to this body: a named thing by its name
            # ("the haft of the axe"), anything else by its material, as it
            # always was (`name` hands the old words back unchanged)
            it = self.name(self._held_eid(p), "the " + p["held"]["label"], p) \
                if p.get("held") else None
            held = p.get("dragging") or it
            menu.append({"key": "keep hold of " + held if held else "hands free",
                         "tag": "keep", "verb": "keep"})
            if held:
                menu.append({"key": "let go", "tag": "let_go", "verb": "let_go"})
            # WHICH ARM, named on the row. An arm already out is the one the
            # body would bring in; otherwise a FREE hand reaches, and which
            # free hand is `_hand`'s business.
            out_arm = next((a for a in sorted(self._limbs(p)) if "arm" in a
                            and np.abs(np.atleast_1d(
                                (p.get("drawn") or {}).get(a, 0.0))).max() > 1e-9),
                           None)
            arm = out_arm or self._hand(p, cells, free=True) \
                or self._hand(p, cells)
            if arm is not None:
                # REACHING IS AN ACT, not a way of drawing the body. An arm put
                # out is OUT: it occupies what it occupies, its hand is a good
                # deal further from the feet, and holding it there costs the
                # shoulder the limb's weight times the lever. So a body gets to
                # choose it, and gets to stop.
                # WHERE THE ARM IS, not what the body meant. These part
                # company whenever a reach is abandoned — the world said no and
                # the intention was dropped, but the flesh is still out there.
                # Read off the intention, the menu then offered a man with his
                # arm stuck half out the chance to "reach out" and never once
                # the chance to bring it down.
                out = out_arm is not None
                if out or not p.get("_no_reach") == (p["anchor"],
                                                     tuple(p.get("facing", (1.0, 0.0)))):
                    # AND A REACH IT HAS ALREADY FOUND IT CANNOT MAKE is not on
                    # the menu. A picked option must be one that can happen; a
                    # body pinned against a wall re-decided to reach every few
                    # ticks for ever, and every one of those picks went into the
                    # trace as a decision that changed nothing. It is forgotten
                    # the moment the body moves or turns, because then it is a
                    # different question.
                    menu.append({"key": "pull the arm back in" if out else "reach out",
                                 "tag": "pull_in" if out else "reach",
                                 "verb": "pull_in" if out else "reach",
                                 "arm": arm})
                # swinging at NOTHING is possible, just useless — the same way
                # shouting in an empty house is possible. What makes it legal
                # is having an arm, not having a target. A held thing swings
                # WITH the arm; that is most of what holding a tool is for
                # BUT NOT WHILE AN ARM IS ALREADY COMING ROUND. A swing is a
                # body in flight until it lands; offered again, a man who
                # re-decided mid-swing (a noise will do it) launched a second
                # copy of the same arm, and the two tore him in pieces.
                swinging = any(b.get("part") and b.get("owner") == p["name"]
                               for b in self.bodies)
                if not swinging:
                    menu.append({"key": "swing an arm" if not p.get("held") else
                                 "swing " + it, "tag": "swing",
                                 "verb": "swing",
                                 # what is held swings with the hand holding it
                                 "arm": (p.get("held") or {}).get("arm") or arm})
                if p.get("held"):
                    # THE SAME SWING, LET GO OF. Nothing new happens to the
                    # arm; the only difference is whether the hand opens.
                    menu.append({"key": "throw " + it,
                                 "tag": "throw", "verb": "throw",
                                 "arm": (p.get("held") or {}).get("arm") or arm})
                # AND A HAND THAT IS EMPTY CAN CLOSE ON SOMETHING PASSING.
                # A FREE hand, which is not the same as having no hands full: a
                # body has two, and one of them being busy is exactly the
                # situation a rescue is. Written as "only when holding nothing"
                # it meant a man with a hand on the rail could not catch the
                # friend going past him, which is the one moment it was for.
                #
                # Offered only for a thing genuinely within reach right now, so
                # a picked catch is a catch that can happen — whether the arm
                # can STOP it is the act's own question, and a hand that cannot
                # is a hand the thing goes past.
                spare = self._hand(p, cells, free=True)
                if spare is not None:
                    named = set()
                    # AND ONLY WHAT IT SAW COMING IN TIME. A catch is a reach
                    # at something seen, and it can start no sooner than this
                    # body answers what it sees (`_in_time`): a mug thrown
                    # from a metre at a man who was not looking is past him
                    # before his hand can move (rows P12, P22).
                    for fb in self._in_flight_near(p, cells):
                        if not self._in_time(p, fb):
                            continue
                        lab = MATNAME.get(int(fb["mats"][0]), "thing")
                        e = self._body_eid(fb)
                        if e and e in named:
                            continue          # the same thing, in two bits
                        named.add(e)
                        menu.append({"key": "catch " + self.name(e, f"the {lab}", p),
                                     "tag": "catch", "verb": "catch",
                                     "arm": spare, "body": fb, "away": 0.0})
                    for fb in self._falling_near(p, cells):
                        if not self._in_time(p, fb):
                            continue
                        menu.append({"key": f"catch {fb['owner']}",
                                     "tag": "catch_who", "verb": "catch_who",
                                     "arm": spare, "body": fb, "away": 0.0})
            for q in self._within_reach(p, cells):
                if q["name"] == held:
                    continue
                menu.append({"key": f"take hold of {q['name']}", "tag": "hold",
                             "verb": "hold", "who": q["name"]})
            # A THING AND A PERSON ARE NOT THE SAME ROW. They were both tagged
            # "hold", so a character sheet saying `hands: hold` meant "hold
            # whatever is nearest" — and a rescuer who wandered within reach of
            # the fire picked up a BURNING STICK instead of the man he had come
            # for, then stood there holding it until the smoke took him. What
            # the sheet meant was a person.
            if not p.get("held"):
                named, takes = set(), []
                for ob in self._objects_within_reach(p, cells):
                    # ONE ROW A THING. Reach joins cells on the key (3.3), so
                    # a named axe of iron and wood is one row already, and a
                    # thing that leaves the reach box and comes back is one
                    # too (`_objects_within_reach`); this is the last guard,
                    # by eid. HOW FAR it is goes on the row, so that the cap
                    # keeps the nearest things, as it keeps the nearest places
                    e = self.entity_at(*ob["cell"])
                    if e and e in named:
                        continue
                    if e and self._hinged(e):
                        continue          # a door on its hinges is not a thing to take up
                    named.add(e)
                    takes.append({"key": "take hold of "
                                  + self.name(e, f"the {ob['label']}", p),
                                  "tag": "take", "verb": "hold", "what": ob,
                                  "away": ob["away"]})
                # AND THE THING IT MEANT TO TAKE. Reach may be told to stop
                # after a few things (`reach_things_cap`), which is a model of
                # attention; but a body that set out to pick up the axe has
                # not forgotten the axe because a cup is nearer. Asked of that
                # thing alone, by the same tests: there, in reach, and light
                # enough to shift.
                ob = self._wanted_thing(p, cells, intent["take"]) \
                    if intent.get("take") is not None else None
                if ob is not None and not any(
                        o["what"]["cell"] in ob["cells"] for o in takes):
                    e = self.entity_at(*ob["cell"])
                    away = ob["away"]
                    ob = {k: ob[k] for k in ("cell", "mat", "label")}
                    takes.append({"key": "take hold of "
                                  + self.name(e, f"the {ob['label']}", p),
                                  "tag": "take", "verb": "hold", "what": ob,
                                  "away": away})
                menu += self._unique_takes(p, takes)
                # AND WHAT IT HOLDS DEAR (`dear` on its sheet, a thing's eid):
                # what a person picks up before leaving a fire, if anything
                # (the pre-movement, row P15). Its own row, by the same tests
                # as the thing a linked mind means to take; or the take row
                # that is already about it, marked.
                dear = p.get("dear")
                ob = self._wanted_thing(p, cells, dear) \
                    if dear is not None else None
                if ob is not None:
                    row = next((o for o in takes
                                if tuple(o["what"]["cell"]) in ob["cells"]), None)
                    if row is None:
                        e = self.entity_at(*ob["cell"])
                        row = {"key": "take hold of "
                               + self.name(e, f"the {ob['label']}", p),
                               "verb": "hold", "away": ob["away"],
                               "what": {k: ob[k] for k in ("cell", "mat", "label")}}
                        menu.append(row)
                    row["tag"] = "take:dear"
            if self.mech_menu and self.__dict__.get("mechs"):
                menu += [r for r in self.mech_affords(p, cells=cells)
                         if r["verb"] in ("operate", "shoulder")]
        elif limb == "eyes":
            # THE SWEEP IS THE NULL ACT. A body that decides nothing goes on
            # turning its head, which is what it did before eyes were a limb —
            # so nothing in the sim got worse the day they became one.
            held = (p.get("look") or (0, 0, -99))
            looking = self.tick - held[2] < WILL["look_for"]
            menu.append({"key": "go on looking about", "tag": "about",
                         "verb": "about"})
            nz_ = p.get("noise")
            ear = p.get("_eye")
            if nz_ is not None and ear is not None \
                    and self.tick - nz_["tick"] < WILL["look_for"]:
                dx_, dy_ = nz_["at"][0] - ear[0], nz_["at"][1] - ear[1]
                n_ = float(np.hypot(dx_, dy_))
                if n_ > 1e-6:
                    menu.append({"key": "look toward the noise",
                                 "tag": "look:noise", "verb": "look",
                                 "dir": (float(dx_ / n_), float(dy_ / n_))})
            # AND TOWARD WHAT ALARMED IT, for as long as it is alarmed: the
            # first thing a person does about a fire across the room, or a
            # shout, or a thing coming at them, is look at it (`_cue`)
            cue = p.get("_cue")
            if cue is not None and ear is not None \
                    and (self.tick - cue[2]) * TICK_S < WILL["alert_s"]:
                dx_, dy_ = cue[0] - ear[0], cue[1] - ear[1]
                n_ = float(np.hypot(dx_, dy_))
                if n_ > 1e-6:
                    menu.append({"key": "look toward " + CUES.get(cue[3], "it"),
                                 "tag": "look:cue", "verb": "look",
                                 "dir": (float(dx_ / n_), float(dy_ / n_))})
            fx, fy = p.get("facing", (1.0, 0.0))
            for name, (dx, dy) in (("behind", (-fx, -fy)),
                                   ("to the left", (-fy, fx)),
                                   ("to the right", (fy, -fx))):
                if looking and abs(held[0] - dx) < 1e-6 and abs(held[1] - dy) < 1e-6:
                    continue                  # already looking that way
                menu.append({"key": f"look {name}", "tag": f"look:{name}",
                             "verb": "look", "dir": (float(dx), float(dy))})
        elif limb == "waist":
            if not self._bones(p, "lean") or "lean" not in (p.get("chain") or {}):
                return []                     # nothing here bends
            # THE SAME SHAPE AS HANDS. The null act KEEPS what the body is
            # doing — a trunk takes many ticks to bend and the body must be
            # able to go on bending — and straightening up is an act of its
            # own. Made the null act "stand up straight", a body cancelled its
            # own lean on the very next decision and never bent at all.
            # DRAWN, not posed. `pose` is the motor command and it runs on
            # through every angle the lattice cannot draw; `drawn` is where the
            # flesh actually IS. They part company badly here, because a lean
            # has a long undrawable tail — measured, a body whose angle read
            # 45.8 degrees was bent 19.3. Ask the flesh.
            bent = abs(float(np.atleast_1d(
                (p.get("drawn") or {}).get("lean", 0.0))[0])) > 1e-9
            going = bool((p.get("reach") or {}).get("lean"))
            menu.append({"key": "keep leaning" if going else
                         ("stay bent" if bent else "stand as it is"),
                         "tag": "hold_pose", "verb": "hold_pose"})
            if going or bent:
                menu.append({"key": "straighten up", "tag": "upright",
                             "verb": "upright"})
            if not going:
                menu.append({"key": "lean out", "tag": "lean", "verb": "lean"})
        elif limb == "mouth":
            menu.append({"key": "say nothing", "tag": "quiet", "verb": "quiet"})
            usual_rows, other_rows = [], []
            for name, text in p.get("lines", LINES).items():
                if not text:
                    continue                  # this body has no such words
                if ANSWERS.get(name) not in (None, percept):
                    continue                  # nothing to answer
                # EVERY LINE AT EVERY LOUDNESS, whispered to shouted, each a row
                # of its own: who hears is the sound law's business, so a body
                # that wants only the man beside it to hear has a row for that.
                # The row the reflex table takes (`say:fire`) is the line's
                # own loudness (`LINE_VOICE`); the rest are `say:fire@whisper`
                # and so on.
                usual = LINE_VOICE.get(name, "say")
                for voice, db in VOICE:
                    row = {"key": f"say({name}, {voice})",
                           "tag": f"say:{name}" if voice == usual
                           else f"say:{name}@{voice}",
                           "verb": "say", "line": name, "voice": voice,
                           "db": db}
                    (usual_rows if voice == usual else other_rows).append(row)
            # every line at its own loudness comes first, so the cap on
            # attention, which keeps the first rows of a kind, cuts a line's
            # other loudnesses and never a whole line
            menu += usual_rows + other_rows
        cap = MENU_CAP if cap is None else cap
        self._offered = len(menu)             # what a full sweep found, before
        if len(menu) > cap:                   # attention is the scarce thing.
            want = (p.get("reflexes", REFLEXES).get(percept) or {}).get(limb) \
                if percept else None
            # and what a linked mind's job wants of this limb: a row that is
            # cut cannot be picked, so a step could never be carried out
            mine = (intent.get("want") or {}).get(limb)
            head, tail = menu[0], menu[1:]    # the null act never falls off
            keep = [o for o in tail if o["tag"] == want
                    or (mine is not None and self._wants(o, mine))]
            # BY IDENTITY. `not in` compares rows by value, and a row that
            # carries the thing it is about — a flier to catch, whose fields
            # are arrays — cannot be compared that way at all: four bolts
            # near four people raised instead of choosing.
            kept = {id(o) for o in keep}
            rest = sorted((o for o in tail if id(o) not in kept),
                          key=lambda o: o.get("away", 0.0))
            # ONE OF EACH KIND FIRST, then the nearest of what is left.
            #
            # Sorted by distance alone, the cap would drop every row of a whole
            # KIND — measured, it cut all four roam spots because each was
            # further off than a step, and the body then could not choose to
            # wander at all however much it wanted to. Recall was 81%: one
            # decision in five, the cap was hiding the answer.
            #
            # This is a model of attention rather than a trick for the policy's
            # benefit. What a body notices is categories before instances — that
            # there is a door, that there is somewhere unseen, that there is a
            # place it could go — and then the nearest of each. It keeps nothing
            # because the policy prefers it: salience is not preference, and a
            # fire is worth noticing whether you mean to run at it or from it.
            first, later, got = [], [], {o["tag"] for o in keep}
            for o in rest:
                (later if o["tag"] in got else first).append(o)
                got.add(o["tag"])
            menu = ([head] + keep + first + later)[:cap]
        return menu

    # ── doors ───────────────────────────────────────────────────────────────
    def _hinged(self, e):
        """Is thing `e` a leaf or a frame of a mechanism (a door on its
        hinges, in its jambs)? It is worked (`mech_affords`), not carried off."""
        tabs = self.__dict__.get("mechs")
        return bool(tabs) and any(e in (r["leaf"], r["frame"]) for r in tabs.values())

    def _door_plan(self, p, cells, fit):
        """The closed doors this body has seen and the spots to stand at to work
        each (routing.door_spots), before the routes are swept: {"doors":
        {mid: (rec, columns)}, "names", "spots": {mid: ((x, y), ...)},
        "targets": [(x, y), ...]} or None when there is no such door."""
        d = self._d()
        reach_c = max(1, int(round(0.5 / d)))
        known = p["known"]
        here = {m for _d, m, _r in self.mech_near(p, cells=cells)}
        doors = {}
        for mid, rec in self.mechs.items():
            if abs(rec["joints"][0].get("theta", 0.0)) > 1e-9:
                continue
            lc = self._cells_of(rec["leaf"])
            if lc is None or not known[lc[:, 0], lc[:, 1]].any():
                continue
            doors[mid] = (rec, np.unique(lc[:, :2], axis=0))
        if not doors:
            return None
        at = (float(cells[:, 0].mean()), float(cells[:, 1].mean()))
        names = self.mech_names(p, list(doors), at=at)
        spots = {}
        for mid, (rec, lc) in doors.items():
            if mid in here:
                continue                  # at it: the hands' rows are the way
            xy = routing.door_spots(fit, lc, reach_c, rec["swing_n"], rec["pivot2"])
            if xy:
                spots[mid] = xy
        return {"doors": doors, "names": names, "spots": spots, "here": here, "at": at,
                "targets": [s for mid in sorted(spots) for s in spots[mid]]}

    def _door_rows(self, p, cells, fit, places, routes, ground, plan):
        """The legs' rows for the mechanisms: a kick at one in reach (a body
        on its feet), and a walk to the foot of each CLOSED door it has seen and
        is not yet at. The walk carries what the planner knows: `cost_ticks`,
        the walk plus the opening (the swing law, mechlaw `_open_time_s`),
        and `beyond`, the places that have a route only through this door, or a
        cheaper one with the opening counted (routing.beyond_door). A door the
        body has found stuck (its own memory) is not priced as a way through:
        for this body it is a wall until it tries again. No row says locked."""
        rows = []
        if ground:
            near = self.mech_near(p, cells=cells)
            names = self.mech_names(p, [m for _d, m, _r in near],
                                    at=(float(cells[:, 0].mean()), float(cells[:, 1].mean())))
            for dd, mid, rec in near:
                if mid in names:
                    rows.append({"key": "kick %s" % names[mid], "tag": "kick",
                                 "verb": "kick", "mid": mid, "op": "kick",
                                 "away": round(dd, 4)})
        if plan is None:
            return rows
        d = self._d()
        doors, names, spots = plan["doors"], plan["names"], plan["spots"]
        to_spot = routes
        lim = self.push_limit(p, cells)       # this body's push, asked once
        # the planner with every closed door a way through (not the ones this
        # body has found stuck), against the plain one
        open_cols = {}
        for mid, (rec, lc) in doors.items():
            if mid in names and not self._door_memory(p["pid"], rec):
                for c in lc.tolist():
                    open_cols[(c[0], c[1])] = mid
        thru, start = {}, (int(round(float(cells[:, 0].mean()))),
                           int(round(float(cells[:, 1].mean()))))
        if open_cols and places:
            q, fit_o = self._open_view(p, cells, open_cols)
            thru = self._routes(cells, [pl["xy"] for pl in places], q, fit=fit_o)
        beyond = {}
        for pl in places:
            path = thru.get(pl["xy"])
            if not path:
                continue
            mid, k = routing.first_door(path, open_cols)
            if mid is None:
                continue
            t_open = self._open_time_s(p, doors[mid][0], lim)
            pen = routing.open_cost_units(t_open, WILL["walk_mps"], d)
            if routing.beyond_door(routes.get(pl["xy"]), path, start, pen):
                beyond.setdefault(mid, []).append((pl["label"], pl["tag"]))
        p["_door_beyond"] = {"t": self.tick, "by": {m: tuple(v) for m, v in beyond.items()}}
        for mid in sorted(spots):
            best = None
            for xy in spots[mid]:
                r = to_spot.get(xy)
                if r is None:
                    continue
                c = routing.path_units(r, start)
                if best is None or c < best[0]:
                    best = (c, xy, r)
            if best is None or not best[2]:
                continue
            c, xy, r = best
            t = c / 70.0 * d / WILL["walk_mps"] + self._open_time_s(p, doors[mid][0], lim)
            rows.append({"key": "go(to %s)" % names[mid], "tag": "go:door",
                         "verb": "go", "goal": xy, "route": r,
                         "door": mid, "away": float(len(r)),
                         "cost_ticks": int(np.ceil(t / TICK_S)) if np.isfinite(t) else 10 ** 9,
                         "beyond": tuple(b[0] for b in beyond.get(mid, ())),
                         "beyond_tags": tuple(b[1] for b in beyond.get(mid, ())),
                         "danger": 0.0})
        return rows

    def _open_view(self, p, cells, open_cols):
        """(q, fit): this body as the planner would see it if the given door
        columns (a {(x, y): mid}) were open: a copy of the person whose flat
        belief holds them standable, and, where it plans over levels, a level
        at its own sole height in those columns (a closed leaf shows only its
        top, which is no floor). The question is 'could this body get through
        if the door were open?'. The person itself is not touched."""
        free = p.get("free")
        free = (self._walkable(cells) if free is None else free).copy()
        for x, y in open_cols:
            free[x, y] = True
        q = dict(p)
        q["free"] = free
        if any(self._step_range(p)):
            q["fz"] = self._levels_for(cells, p)
        return q, self._fit_grid(cells, q)[0]

    def _open_time_s(self, p, rec, lim=None):
        """How long this body's push takes to swing a closed leaf a quarter
        turn (mechanisms.swing_time_closed, from its own push limit, the leaf's
        inertia and the hinge's friction)."""
        cells = self._cells_of(rec["leaf"])
        I, mass = self._leaf_I(rec, cells)
        arm = min(0.8, rec["W"] * self._d() - 0.1)
        lim = self.push_limit(p) if lim is None else lim
        tau = lim * arm - self._hinge_friction(rec, mass)
        if tau <= 0.0:
            return float("inf")
        return float(mech.swing_time_closed(I, tau, math.pi / 2.0))

    def _do_door(self, p, row):
        """Carry out a mechanism row: the body's hands (or shoulder) on it. The
        law answers; a refusal is a 'jams' event with the person as actor, and
        the person remembers it (mechlaw `_learn_door`)."""
        mid, op = row["mid"], row["op"]
        if mid not in self.mechs:
            return None
        if row["verb"] == "shoulder":
            return self.shoulder(p, mid)
        held = self.things.get(self._held_eid(p)) if p.get("held") else None
        kw = None
        if op == "key" and held and "key" in held:
            kw = {"key": held["key"]}
        elif op.startswith("pick:") and held and "tool" in held:
            kw = {"tool": held["tool"]}
        return self.operate(p, mid, op, lockkw=kw)

    def _room(self, p, cells):
        """THE ROOM A BODY STANDS IN: what an idle body wanders within.

        It is the airspace the body is in (`air_region`, the connected air
        `label.py` labels) and, where that airspace is a house with its front
        door open, the part of it under the same kind of cover: a body under a
        roof is in the house and not in the yard the door opens onto, a body in
        the yard is in the yard. Returns None where the body touches no air at
        all (it is embedded or underwater), and then nothing is held back."""
        if self.air_region is None:
            self._air_regions()
        reg = self.air_region
        lo = np.maximum(cells.min(axis=0) - 1, 0)
        hi = np.minimum(cells.max(axis=0) + 2, reg.shape)
        near = reg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].ravel()
        near = near[near >= 0]
        if not len(near):
            return None
        zb = int(cells[:, 2].max())
        cx = int(round(float(cells[:, 0].mean())))
        cy = int(round(float(cells[:, 1].mean())))
        return {"region": int(np.bincount(near).argmax()), "zb": zb,
                "covered": bool(self._covered(np.array([[cx, cy]]), zb)[0])}

    def _covered(self, xy, zb):
        """Is there anything solid above head height (`zb`) in each of these
        columns (an (n, 2) array)? A roof, a ceiling, a floor above: not a
        body, not leaves."""
        if zb + 1 >= self.shape[2]:
            return np.zeros(len(xy), bool)
        col = self.mat[xy[:, 0], xy[:, 1], zb + 1:]
        return ((col != AIR) & (col != FLESH) & (col != LEAF)).any(axis=1)

    def _in_room(self, room, xy):
        """Do all of these floor spots (a list of (x, y)) lie in `room`
        (`_room`)? A spot is in it when the air at head height there is the
        room's airspace (or, where something fills that cell, not another
        one) and it is under the same kind of cover."""
        if room is None:
            return True
        xy = np.asarray(xy, np.int64).reshape(-1, 2)
        lab = self.air_region[xy[:, 0], xy[:, 1], room["zb"]]
        if not ((lab == room["region"]) | (lab < 0)).all():
            return False
        return bool((self._covered(xy, room["zb"]) == room["covered"]).all())

    def _wander_in_room(self, p, cells, fit, places, room):
        """The places a body could name, with the one it WANDERS to picked
        inside its own room.

        `_places` picks the "somewhere it knows, a way off" spot from the
        whole of what the body has seen, and that is how a man in a house
        with its door open walked out into the yard at the first tick. A body
        with nothing to do wanders in the room it is in; it leaves by choice
        (a row of the menu it may take), a drive (a reflex's way out), or a
        job. So the roam spot is drawn from the spots of this room, by the
        same rule `_places` uses: deterministic, from the tick and the name."""
        keep = [pl for pl in places if pl["tag"] != "go:roam"]
        # (it goes where `_places` put it: before the spots to strike from)
        at = next((i for i, pl in enumerate(keep) if pl["tag"] == "go:seek"),
                  len(keep))
        spots = np.argwhere(fit & p["known"])
        if room is not None and len(spots):
            lab = self.air_region[spots[:, 0], spots[:, 1], room["zb"]]
            ok = (lab == room["region"]) | (lab < 0)
            ok &= self._covered(spots, room["zb"]) == room["covered"]
            spots = spots[ok]
        ax_ = float(cells[:, 0].mean())
        ay_ = float(cells[:, 1].mean())
        vox_m = 0.1 * self.scale
        arrive = WILL["arrive_m"] / max(vox_m, 1e-9)
        baulked = p.get("_baulked") or {}
        if len(spots) > 1:
            d = np.abs(spots[:, 0] - ax_) + np.abs(spots[:, 1] - ay_)
            far = spots[d > max(6.0, np.percentile(d, 70))]
            if len(far):
                i = (self.tick // max(WILL["decide_every"], 1)
                     + len(p["name"])) % len(far)
                xy = (int(far[i][0]), int(far[i][1]))
                here = float(np.abs(cells[:, :2] - np.array(xy)).max(axis=1).min())
                if here > arrive and self.tick - baulked.get(xy, -10 ** 9) \
                        >= WILL["baulk_for"]:
                    keep.insert(at, {"tag": "go:roam", "label": "the floor "
                                     + self._bearing(xy[0] - ax_, xy[1] - ay_),
                                     "xy": xy, "away": max(abs(xy[0] - ax_),
                                                           abs(xy[1] - ay_))})
        return keep

    def _unique_takes(self, p, takes):
        """The take rows, each saying something no other row says.

        Matter with no name is called by its material, so three unnamed
        lumps of iron in reach were three rows that each read "take hold of
        the iron", and a mind that picks a row by what it says could not
        pick one of them. Rows that read the same say where each thing is
        from the body, and then which of those there, nearest first: the
        words `name` gives two registered lookalikes (`_set_apart`).

        A NAMED THING KEEPS ITS NAME. Of rows that read the same, the
        nearest that is a registered thing keeps the words `name` gives it,
        the words every line about it says, and only the others move. And
        what moves never lands on what another row says: no row is dropped
        to make the rest unique. (A last guard did drop the further of two
        rows that still read the same, and two adopted irons, "the first
        iron at your feet" and "the second", read the same as two unnamed
        lumps set apart beside them: the irons a hand had put down left the
        menu, the one a linked mind meant to take with them.)

        A row that reads as no other row does is left exactly as it was."""
        seen = {}
        for o in takes:
            seen.setdefault(o["key"], []).append(o)
        redo = []
        for key, same in seen.items():
            if len(same) < 2:
                continue
            named = [o for o in same if self.name(
                self.entity_at(*o["what"]["cell"]), None) is not None]
            keep = min(named, key=lambda o: o["away"]) if named else None
            redo.append((key, [o for o in same if o is not keep]))
        if not redo:
            return takes
        moving = {id(o) for _, rows in redo for o in rows}
        taken = {o["key"] for o in takes if id(o) not in moving}
        for key, rows in sorted(redo, key=lambda r: min(o["away"]
                                                        for o in r[1])):
            words = self._set_apart(
                key[len("take hold of "):],
                [(self._object_at(*o["what"]["cell"]), o["away"])
                 for o in rows], p,
                {k[len("take hold of "):] for k in taken})
            for o, n in zip(rows, words):
                o["key"] = "take hold of " + n
                taken.add(o["key"])
        return takes

    def _wants(self, row, want):
        """Does a menu row match what a linked mind wants of the limb
        (`intent["want"][limb]`)? A string is a tag, as in REFLEXES. A dict
        matches on the row's PAYLOAD -- `goal`, `who`, `line`, `tag`, and
        `cell` or `eid` for the thing a take row is about -- never on its
        label, because the words for a place change as a body moves (3.5)."""
        if isinstance(want, str):
            return row["tag"] == want
        for k, v in want.items():
            if k in ("cell", "eid"):
                ob = row.get("what")
                if ob is None:
                    return False
                got = tuple(ob["cell"]) if k == "cell" \
                    else self.entity_at(*ob["cell"])
                if got != (tuple(v) if k == "cell" else v):
                    return False
            else:
                got = row.get(k)
                if isinstance(got, (tuple, list)) and isinstance(v, (tuple, list)):
                    got, v = tuple(got), tuple(v)
                if got != v:
                    return False
        return True

    def _wanted_thing(self, p, cells, want):
        """The thing a linked mind means to take (`intent["take"]`: a cell
        of it, or a named thing's eid), as a take row's `what` plus its
        `cells` -- or None when it is not there, not in reach or too heavy
        to shift. The tests `_objects_within_reach` puts every thing to,
        asked of one: `_object_at` for what the thing is, `_in_reach` at
        this body's reach, `_effort` against its strength."""
        if isinstance(want, (int, np.integer)):
            where = self.cells_of(int(want)) if want else np.zeros((0, 3))
            # NOT WHAT IS IN FLIGHT, as `_object_at` refuses it on the cell
            # path: a drawn particle cell is the solver's, not the thing's
            # (the kept index is only as fresh as the last move it was told)
            if len(where):
                where = where[~self.moving[tuple(np.asarray(where).T)]]
        else:
            x, y, z = (int(v) for v in want)
            nx, ny, nz = self.shape
            if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz) \
                    or int(self.mat[x, y, z]) in (AIR, FLESH):
                return None
            where = self._object_at(x, y, z)
        if not len(where) or len(where) >= 4000:
            return None                       # gone, or the world itself
        where = np.asarray(where)
        if (self.mat[tuple(where.T)] == FLESH).any():
            return None                       # a person is not a thing to take
        R = max(int(round(self._reach_of(p) / max(0.1 * self.scale, 1e-9))), 1)
        if not self._in_reach(cells, where, R):
            return None
        if self._effort(where)[1] > self._strength(p):
            return None
        cell = tuple(map(int, where[0]))
        m0 = int(self.mat[cell])
        # how far, in cells, as the reach rows say it (`_objects_within_reach`)
        own = np.asarray(cells)
        away = min(int(np.abs(where[i:i + 256, None, :] - own[None, :, :])
                       .max(axis=2).min()) for i in range(0, len(where), 256))
        return {"cell": cell, "mat": m0, "label": MATNAME.get(m0, "thing"),
                "cells": set(map(tuple, where.tolist())), "away": away}

    def _merged_pick(self, sit, limb, chosen, menu, batch):
        """The pick of a menu longer than `batch`, as a TOURNAMENT: rows in
        canonical order (their tag, then their key), batches of `batch` with
        the null act in each, the policy asked of each batch, the winners
        asked again. A batch that picks the null act has no winner. Returns
        an index into `menu`. The null act (index 0) is in every batch, so it
        counts one of the `batch` places."""
        head = menu[0]
        keys = [(o["tag"], o["key"]) for o in menu[1:]]

        def pick(members, _rnd):
            sub = [head] + [menu[m + 1] for m in members]
            j = self.policy.pick(dict(sit, limb=limb, chosen=dict(chosen)), sub)
            return members[j - 1] if isinstance(j, int) and 1 <= j < len(sub) else None

        w = merge(keys, batch - 1, pick)
        return 0 if w is None else w + 1

    def _full_menu(self, p, cells, percept, limb, chosen):
        """The menu WITHOUT the cap — what a body would have to choose from if
        attention were free. Only ever used to measure what the cap costs."""
        return self._menu(p, cells, percept, limb, chosen, cap=10 ** 6)

    def _decide(self, p, percept, cells, ax_, ay_, eye, vox_m, news=None,
                src=None, phase=None):
        """Offer one menu per part of the body, let the policy pick from each,
        carry the whole program out, log it.

        The result is a PROGRAM — go here AND hold them AND say this — built
        by composition rather than looked up. Every combination the old table
        spelled out still happens, and combinations nobody spelled out happen
        too: holding someone while walking deeper in, calling a warning while
        wandering, taking hold with no idea where the door is.

        `news` is what the attention law found new (`_attend`): only the
        "onset" law passes one, and only under it does the situation say
        whether this decision answers news (`sit["onset"]`).

        `src` is the event the percept came from (V3), maybe still a box.
        A decision is an event too (V5): `decides`, caused by `src`, and
        every act below runs inside `because(decides)`, so each act's event
        says which decision it carried out and `w.why` can walk from an act
        back through the decision to what the body perceived.

        `phase` is where this decision stands in answering the percept
        (`_react`), None for a plain one: "flinch" (a surprised body's eyes
        go to a thing coming at it, and only the eyes are asked), "premove"
        (making sense of an alarm before leaving: every limb is asked, and
        the body is not yet committed), "go" (the pre-movement is over: the
        legs and hands are asked again). It rides in the situation."""
        # who would actually HEAR a shout. Not a legality gate — shouting in
        # an empty house is possible, just useless — but a fact a thinking
        # policy should get to weigh, so it rides along in the situation.
        heard = sum(1 for q in self.persons
                    if q is not p and q["alive"] and q["awake"]
                    and self._hears(eye, q.get("_eye", (*q["anchor"], eye[2]))))
        sit = {"tick": self.tick, "who": p["name"], "percept": percept,
               "reflexes": p.get("reflexes", REFLEXES),
               "smoke": round(float(p["smoke"]), 3),
               "blood_o2": round(float(p["blood_o2"]), 3),
               "burn": round(float(p["burn"]), 3),
               "hurt": round(float(p["hurt"]), 3),
               "holding": p.get("dragging") or
                          (self.name(self._held_eid(p), "the " + p["held"]["label"])
                           if p.get("held") else None),
               "knows_a_way_out": any(p["known"][int(e[0]), int(e[1])]
                                      for e in self.exits),
               "seen_of_the_world": round(float(p["known"].mean()), 3),
               "ground_it_trusts": round(float(p["free"].mean()), 3)
                                   if p.get("free") is not None else 0.0,
               "others_in_earshot": heard}
        if self.attention == "onset":
            sit["onset"] = news is not None
        if phase is not None:
            sit["phase"] = phase
        # A LINKED MIND'S HOOKS (3.5), read with getattr so a policy that is
        # only `pick` is untouched. `observe` is shown what this mind holds
        # now (`beliefs.view`: its percepts and beliefs, in its own words),
        # built only for a policy that asks for it (V5), before anything
        # else is asked. `prepare` makes every change of state a
        # decision needs -- a job's step done, a job stopped by news, a
        # question asked, `p["intent"]` set -- so that `pick` can stay pure.
        # `limbs` names the parts of the body to ask: a body that is walking
        # to a door and is asked only for its hands goes on walking, where
        # its legs' null act would have stopped it.
        obs = getattr(self.policy, "observe", None)
        if obs is not None:
            obs(view(self, p))
        # A staged reaction says which stage this is (`phase`, `_react`), so
        # a mind that answers the percept once can tell the stages apart.
        prep = getattr(self.policy, "prepare", None)
        if prep is not None:
            if phase is None:
                prep(self, p, percept)
            else:
                prep(self, p, percept, phase=phase)
        ask = getattr(self.policy, "limbs", None)
        asked = ask(self, p, percept) if ask is not None else None
        menus, picks, chosen, seen, would = {}, {}, {}, {}, {}
        spelled = []
        # A NOISE ASKS THE EYES, and only the eyes. Asked of every limb, the
        # shards of a pane landing one after another re-decided a man's whole
        # body every tick, and a swing was relaunched before the last one had
        # come down until a cell of the arm came away. What the other limbs
        # were doing, they go on doing.
        for limb in (("eyes",) if percept in ORIENTING or phase == "flinch"
                     else ("legs", "hands") if phase == "go" else LIMBS):
            if asked is not None and limb not in asked:
                continue                      # not asked: it goes on as it was
            batch = self.menu_batch if getattr(self.policy, "batchable", False) else None
            menu = self._menu(p, cells, percept, limb, chosen,
                              cap=None if batch is None else 10 ** 6)
            if not menu:
                continue
            if batch is not None and len(menu) > batch:
                i = self._merged_pick(sit, limb, chosen, menu, batch)
            else:
                i = self.policy.pick(dict(sit, limb=limb, chosen=dict(chosen)), menu)
            i = i if isinstance(i, int) and 0 <= i < len(menu) else 0
            opt = menu[i]
            menus[limb] = [o["key"] for o in menu]
            seen[limb] = self._offered        # and how many there were to see
            if self.recall_check:
                # WHAT ATTENTION COST, measured rather than hoped. Build the
                # menu the body would have had with attention free, ask the
                # same policy, and record what it would have done. Off by
                # default: it doubles the work of deciding, and it exists to
                # answer one question — does the cap ever hide the answer?
                full = self._full_menu(p, cells, percept, limb, chosen)
                j = self.policy.pick(dict(sit, limb=limb, chosen=dict(chosen)),
                                     full) if len(full) > len(menu) else i
                j = j if isinstance(j, int) and 0 <= j < len(full) else 0
                would[limb] = full[j]["key"] if len(full) > len(menu) \
                    else opt["key"]
            # THE DECISION, SPELLED. One row per limb, written as the limb is
            # asked — `chosen` is a running state and there is no way to
            # reconstruct afterwards what the other limbs had settled on when
            # THIS one was put the question. A harvest that reads `menus` and
            # `pick` off the finished row gets the menu and the answer and
            # loses the half of the context that made the answer make sense.
            spelled.append(seam.row(sit, limb, dict(chosen), menu, i))
            picks[limb] = opt
            chosen[limb] = opt["tag"]
        if not picks:
            # A DECISION THAT ASKED NO LIMB IS STILL A DECISION: the mind was
            # consulted and wanted nothing of the body. Not counted, the body
            # was due again on the next tick, and the next, for ever.
            if percept not in ORIENTING:
                p["_decided"] = self.tick
            return
        prog = " & ".join(picks[l]["key"] for l in LIMBS
                          if l in picks and not ACTS[picks[l]["verb"]]["null"])
        dec = self._decides(p, percept, src, prog or "stay", "decide:decides")
        self.traces.append({"tick": self.tick, "who": p["name"],
                            "percept": percept, "situation": sit,
                            "spelled": spelled,   # one row a limb, see `seam`
                            "menus": menus,
                            "offered": seen,  # what a full sweep found, so the
                                              # cost of the cap is on the record
                            "would": would,   # and what it would have picked
                                              # with attention free, if asked
                            "pick": {l: o["key"] for l, o in picks.items()},
                            "tags": dict(chosen), "program": prog or "stay",
                            "by": getattr(self.policy, "name", "?"),
                            "mind": self._mind_row(p, dec)})
        # TURNING THE HEAD TO A NOISE IS NOT WEIGHING ONE'S AFTERNOON. A noise
        # asks the eyes and nothing else, so it does not start the wait before
        # the body deliberates again. It did: another person's footsteps come
        # about every 9 ticks, a deliberation every 10, so a body that heard
        # someone walk nearby looked round at every step and never got to
        # decide where to go (measured: fire_alarm's Berel stood in one cell
        # for 1,200 ticks while Aldan paced the room).
        if percept not in ORIENTING:
            p["_decided"] = self.tick
        with self.because(dec, "call"):
            self._carry_out(p, picks, percept, cells, phase)

    def _carry_out(self, p, picks, percept, cells, phase=None):
        """Do what `_decide` picked, limb by limb. Called inside
        `because(decides)`, so every event an act raises -- and every event
        its synchronous callees raise -- names the decision as its cause."""
        mind = self.minds[p["pid"]]      # where the acts below are kept (V3)
        # TURNING TOWARD A NOISE IS NOT AN EMERGENCY. Every percept used to
        # commit the body, and a committed body stops sensing — so a man who
        # heard a crackle and looked round was frozen there for good, and
        # burned without ever seeing the fire he had turned to look at.
        # NOR IS A FLINCH, OR MAKING SENSE OF AN ALARM: the body is not yet
        # committed to anything, and goes on sensing (`_react`).
        if percept is not None and percept not in ORIENTING \
                and phase not in ("flinch", "premove"):
            p["emergency"] = True        # committed: stop weighing the ordinary
        hands = picks.get("hands")
        if hands is not None:            # HANDS. Holding is not a rescue
            if hands["verb"] in ("swing", "throw"):
                p["_alert"] = self.tick  # a body that strikes is in a fight
                # AT WHAT A LINKED MIND AIMS AT, if it aims; else straight
                # ahead, as a swing always went
                b = self._swing(p, hands.get("arm") or self._hand(p, cells),
                                toward=(p.get("intent") or {}).get("aim")
                                or p.get("facing"),
                                back=(BODY["windup_rad"]
                                      if hands["verb"] == "throw" else 0.0))
                if b is not None and hands["verb"] == "throw":
                    b["throw"] = True
                thrown = hands["verb"] == "throw" and p.get("held")
                self.emit("swings", actor=p["pid"],
                          result="thrown" if thrown else "swung",
                          text=f"{p['name']} "
                               + ("throws " + self.name(self._held_eid(p),
                                                        "the " + p["held"]["label"])
                                  if thrown else "swings an arm"),
                          to=p, site="decide:swing")
            elif hands["verb"] == "hold":  # no subsystem: a grip is REACH, and
                if "who" in hands:             # what it can then do is force —
                    p["dragging"] = hands["who"]   # the same _effort a shove
                    if self.body_contact:          # closing on someone and gripping is
                        held = next((r for r in self.persons     # the contact law
                                     if r["name"] == hands["who"]), None)
                        if held is not None:                     # with a grip (grapple)
                            self.grapple(p, held)
                    mind.hold_ev = self.emit(
                        "holds", actor=p["pid"],
                        text=f"{p['name']} takes hold "
                             f"of {hands['who']}",
                        to=p, site="decide:hold_who").id   # uses, asked
                else:                              # again every tick
                    ob = hands["what"]
                    e = self.entity_at(*ob["cell"])   # before it is moved
                    p["held"] = {"cell": ob["cell"], "mat": ob["mat"],
                                 "label": ob["label"],
                                 # WHICH HAND TOOK IT, so the other stays free
                                 "arm": self._hand(p, cells, toward=ob["cell"],
                                                   free=True)}
                    self._take_up(p, cells)    # to the fist, if the arm can
                    mind.hold_ev = self.emit(
                        "takes", actor=p["pid"],
                        text=f"{p['name']} takes up "
                             + self.name(e, f"the {ob['label']}"),
                        to=p, site="decide:take").id
            elif hands["verb"] == "reach":
                p.setdefault("reach", {})[
                    hands.get("arm") or self._hand(p, cells)] = float(np.pi / 2)
                self.emit("reaches", actor=p["pid"],
                          text=f"{p['name']} reaches out",
                          to=p, site="decide:reach")
            elif hands["verb"] == "pull_in":
                (p.get("reach") or {}).pop(
                    hands.get("arm") or self._hand(p, cells), None)
                self.emit("pulls_in", actor=p["pid"],
                          text=f"{p['name']} lowers the arm",
                          to=p, site="decide:pull_in")
            elif hands["verb"] == "catch_who":
                fb = hands.get("body")
                # IS IT STILL IN THE AIR — asked by IDENTITY, not by
                # equality. `fb in self.bodies` compares the body DICT
                # against every other, and a body dict holds numpy
                # arrays, so the comparison comes back an array and
                # Python cannot say whether it is true. It only bites
                # when more than one flier is in the air at once, which
                # is why it took eight people in one room to find it
                # and why no scenario ever did.
                if fb is not None and any(fb is b for b in self.bodies):
                    who = fb.get("owner")
                    got = self._catch_person(p, fb, hands.get("arm")
                                             or self._hand(p, cells))
                    self.emit("catches", actor=p["pid"],
                              result="caught" if got else "missed",
                              text=f"{p['name']} "
                                   + (f"catches {who}" if got
                                      else f"grabs at {who} and cannot hold them"),
                              to=p, site="decide:catch_who")
            elif hands["verb"] == "catch":
                fb = hands.get("body")
                # IS IT STILL IN THE AIR — asked by IDENTITY, not by
                # equality. `fb in self.bodies` compares the body DICT
                # against every other, and a body dict holds numpy
                # arrays, so the comparison comes back an array and
                # Python cannot say whether it is true. It only bites
                # when more than one flier is in the air at once, which
                # is why it took eight people in one room to find it
                # and why no scenario ever did.
                if fb is not None and any(fb is b for b in self.bodies):
                    got = self._catch(p, fb, hands.get("arm")
                                      or self._hand(p, cells))
                    if got:
                        self._chose(p, "fall_grab", ticks=40)   # a hand mid-catch does not withdraw
                    self.emit("catches", actor=p["pid"],
                              result="caught" if got else "missed",
                              text=f"{p['name']} "
                                   + ("catches " + self.name(
                                       self._held_eid(p),
                                       "the " + (p.get("held") or {}).get("label", "thing"))
                                      if got else "cannot hold what is coming, and it goes past"),
                              to=p, site="decide:catch")
            elif hands["verb"] in ("operate", "shoulder"):
                self._do_door(p, hands)
            elif hands["verb"] == "let_go":
                p["dragging"], p["held"] = None, None
                self.emit("releases", actor=p["pid"],
                          text=f"{p['name']} lets go",
                          to=p, site="decide:let_go")
        legs = picks.get("legs")
        if legs is not None and legs["verb"] == "jump":
            self._leap(p, cells, legs.get("goal"))
            p["goal"], p["fleeing"] = None, False
            self.emit("jumps", actor=p["pid"],
                      text=f"{p['name']} jumps",
                      to=p, site="decide:jump")
        elif legs is not None and legs["verb"] == "kick":
            p["goal"] = None
            if legs["mid"] in self.mechs:
                self.kick(p, legs["mid"])
        elif legs is not None and legs["verb"] == "go":
            # FLEEING means driven, not merely walking. Every `go` used to set
            # it, which was harmless while the only reason to walk was an
            # alarm and wrong the moment idling became a real choice: a person
            # strolling to the window then read to everyone else as someone
            # BOLTING PAST, and set off alarms that had no fire behind them.
            # (or a linked mind says it is running from something)
            p["fleeing"] = percept is not None \
                or bool((p.get("intent") or {}).get("flee"))
            p["goal"] = legs["goal"]
            p["_path"], p["_planned"] = legs["route"], self.tick
            # A WALK WITH NOWHERE TO BE is held to the room it began in
            # (`_room`) when it is the table's own idle pick (a chooser that
            # picks a walk, a mind or a scene's script, has chosen it): the
            # way there is checked again whenever it is planned afresh, below
            p["_wander"] = (tuple(legs["goal"]), self._room(p, cells)) \
                if percept is None and legs["tag"] in WANDER \
                and getattr(self.policy, "idle_in_room", False) else None
            # A SPOT TO STAND AT is arrived at by the middle (`_seek_spot`)
            if legs.get("mid"):
                p["_goal_mid"] = True
            else:
                p.pop("_goal_mid", None)
            if percept:
                mind.go_ev = self.emit(
                    "goes", actor=p["pid"], why=percept,
                    text=f"{p['name']} startles "
                         f"({percept.replace('_', ' ')}) and makes "
                         f"for {legs['key'][3:-1]}",
                    to=p, site="decide:go_startled").id
            else:
                mind.go_ev = self.emit(
                    "goes", actor=p["pid"],
                    text=f"{p['name']} makes for "
                         f"{legs['key'][3:-1]}",
                    to=p, site="decide:go").id
        elif legs is not None:
            p["goal"] = None
            if percept:
                self.emit("stays", actor=p["pid"], why=percept,
                          text=f"{p['name']} notices "
                               f"({percept.replace('_', ' ')}) and stays put",
                          to=p, site="decide:stays")
        mouth = picks.get("mouth")
        if mouth is not None and mouth["verb"] == "say":
            self._say(p, mouth["line"], mouth.get("voice"))
        eyes = picks.get("eyes")
        if eyes is not None and eyes["verb"] == "look":
            p["look"] = (eyes["dir"][0], eyes["dir"][1], self.tick)
            self.emit("looks", actor=p["pid"],
                      text=f"{p['name']} looks {eyes['key'][5:]}",
                      to=p, site="decide:look")
        waist = picks.get("waist")
        if waist is not None and waist["verb"] == "lean":
            # AS FAR AS IT CAN, and the world decides how far. `_law_pose`
            # walks the angle up at the speed a trunk can manage and stops
            # where the back or the balance says stop — the body asks to lean
            # out, it does not ask for an angle it has no way of knowing.
            # AS FAR AS THIS SPINE GOES, which is whichever comes first:
            # anatomy, or the lattice. Asked once per body and kept.
            p.setdefault("reach", {})["lean"] = self._bend_max(
                p, "lean", BODY["lean_max"])
            mind.lean_ev = self.emit("leans", actor=p["pid"],
                                     text=f"{p['name']} leans out",
                                     to=p, site="decide:lean").id
        elif waist is not None and waist["verb"] == "upright":
            (p.get("reach") or {}).pop("lean", None)
            if abs(float(np.atleast_1d(
                    (p.get("drawn") or {}).get("lean", 0.0))[0])) > 1e-9:
                self.emit("stands", actor=p["pid"],
                          text=f"{p['name']} straightens up",
                          to=p, site="decide:stand")

    def trace_outcomes(self):
        """Stamp every logged decision with how that body ENDED, and hand the
        rows back. This is the blessing signal the harvest needs: a pick made
        by a person who walked out is worth learning from, one made by a
        person who did not is worth learning against.

        It is not yet menu RECALL — knowing whether a better option was on
        the menu and passed over needs the counterfactual run, which we
        cannot do here. What this does give is the pairing without which no
        recall number can ever be computed: the whole menu, the pick, the
        end."""
        end = {}
        for p in self.persons:
            end[p["name"]] = ("safe" if p["safe"] else "dead" if not p["alive"]
                              else "down" if not p["awake"] else "still inside")
        for row in self.traces:
            row["outcome"] = end.get(row["who"], "unknown")
        return self.traces

    def _attend(self, p, percept, src):
        """THE ATTENTION LAW: does this body decide this tick? Returns
        `(decide, news)`. It is a world law (`self.attention`), the same for
        every body; a flag on one body would be a body that thinks
        differently, and nobody wrote that body (3.5).

        "latch" is the law there has always been: any percept decides, and
        so does a body with nowhere to be once `decide_every` has passed.
        What stops a percept deciding every tick is not here -- the first
        one sets `emergency`, and `_law_will` stops sensing until it ends.
        `news` is always None.

        "onset" senses every tick and decides only on NEWS: a percept whose
        source -- the event it came from (V3), or the percept itself when
        nothing is known to have caused it -- this mind has not attended to
        within `attend_gap` ticks (`Mind.attn`, source -> last tick). A
        percept that goes on is attended to every tick it goes on, so it is
        news once, also when its source is found only while it goes on.
        Besides news: a body with no goal once `decide_every` has passed,
        as under latch, and a body whose linked mind asked to be woken
        (`intent["wake_at"]`; the alarm is spent once it rings).

        One place where a percept becomes a decision, for both laws, so a
        delay between the two (a reaction time per percept) has one place
        to go."""
        due = self.tick - p.get("_decided", -10 ** 9) >= WILL["decide_every"]
        if self.attention != "onset":
            return percept is not None or (due and p.get("goal") is None), None
        now, gap = self.tick, WILL["attend_gap"]
        mind = self.minds[p["pid"]]
        attn = mind.attn
        # FORGET WHAT WOULD BE NEWS AGAIN ANYWAY, and put real ids where an
        # earlier tick left a box: an event raised in the tick it was seen
        # has its id only once that tick ends (`_unbox`), and the same event
        # seen the tick after must find itself already attended to.
        if isinstance(mind.news, _Pending):
            mind.news = mind.news.id
        old = [k for k, t in attn.items()
               if now - t > gap or isinstance(k, _Pending)]
        for k in old:
            t = attn.pop(k)
            if isinstance(k, _Pending) and k.id is not None and now - t <= gap:
                attn[k.id] = max(attn.get(k.id, t), t)
        news = None
        if percept is not None:
            key = src if src is not None else percept
            if isinstance(key, _Pending) and key.id is not None:
                key = key.id
            if key not in attn:               # (what is older is gone above)
                news = key
            # A CAUSE FOUND LATER IS NOT NEWS. Smoke is choked on before any
            # fire is known to have made it, and the moment one is, the same
            # choking names a source this mind has never attended to: it
            # decided again, about nothing that had changed. A percept with
            # no known source is kept under its own name; if it went on to
            # last tick, the first source it is blamed on is not news.
            #
            # ONLY THAT FIRST. That it has been blamed on one is kept too, as
            # (percept, "blamed"), for as long as the percept goes on, with
            # a source or without one (and forgotten `attend_gap` after, like
            # any key). Without it a percept whose source flickered (none,
            # the fire, none, a second fire) hid the second fire as well:
            # every source after a sourceless tick looked like the first.
            #
            # AND THE PERCEPT ITSELF GOES ON while it has a source: its own
            # name is attended to every tick it goes on, blamed or not. Kept
            # only on the ticks it had no source, a long run blamed on one
            # fire outlived it; a short break and the percept came back with
            # no source as news, and the first source found for it the tick
            # after was news as well (the run had been blamed already): two
            # decisions in a row about one onset.
            mark = (percept, "blamed")
            if key != percept:
                if attn.get(percept, -gap) >= now - 1 and mark not in attn:
                    news = None
                attn[mark] = now
            elif mark in attn:
                attn[mark] = now              # the same run: its first is spent
            attn[key] = now
            attn[percept] = now
        if news is not None:
            mind.news = news
        intent = p.get("intent")
        wake = intent.get("wake_at") if intent else None
        woken = wake is not None and now >= wake
        if woken:
            intent.pop("wake_at")
        # AND A BLOW, ON THE TICK IT LANDS, for a linked mind that asked to
        # be told of one (`intent["felt"]`). Being hit is no percept of the
        # will layer -- the table has no reflex for it -- so it only came up
        # at the next decide period, up to `decide_every` ticks late.
        # And it comes first, as what happens to the body itself does
        # (`URGENT`): a wait on anything less is cut short (`_react`), and
        # anything less perceived with it -- the noise of the blow itself --
        # starts no wait of its own. It is answered with the blow.
        if intent and intent.get("felt") and self._struck_now(mind):
            woken = True
            pend = p.get("_react")
            if pend is not None and rank(pend["percept"]) < 2:
                del p["_react"]
            if percept is not None and rank(percept) < 2:
                news = None
        return (news is not None or (due and p.get("goal") is None)
                or woken), news

    def _struck_now(self, mind):
        """Did this mind FEEL a blow this tick (a `felt` row of a harmful
        event, written by `_law_perceive` before the will layer runs)?"""
        for r in reversed(mind.perceived):
            if r.tick < self.tick:
                return False
            if r.how == "felt" and r.verb in HARM:
                return True
        return False

    def _react(self, p, percept, src, news, go, since, cue_at):
        """THE REACTION TIME (rows P12-P16): between a percept and the
        decision it starts, a delay, counted from the moment of perception.
        Returns the decision to make now, `(percept, src, news, phase)`, or
        None. One place, for both attention laws: `_attend` says whether a
        percept would start a decision, and this says when it does.

        A percept that would start one (news, or under "latch" any percept)
        starts a wait instead (`_reaction`), kept in `p["_react"]`: when it
        is due, the decision is made with that percept, whatever the senses
        say by then. While a body waits it makes no other decision, idle or
        not, and a percept that goes on does not start the wait again; only
        a percept that outranks the one it waits on (`rank`) replaces it.
        A decision with no percept behind it (an idle body choosing what to
        do next, `decide_every`) is not a reaction and does not wait.

        The wait has up to three stages, each a decision when it is due:
        "flinch" (a surprised body's eyes go to a thing coming at it), then
        the reaction itself, which for a cue of a danger elsewhere
        (`PREMOVE_ON`) is a "premove" decision -- look, gather, warn,
        uncommitted -- and then "go" once the person's pre-movement time is
        over, when the legs are asked again. `since` is (the tick the thing
        was perceived, whether the body was ready then), or (now, None)."""
        now = self.tick
        pend = p.get("_react")
        fresh = go and percept is not None \
            and (self.attention != "onset" or news is not None)
        if fresh and (pend is None or rank(percept) > rank(pend["percept"])):
            pend = p["_react"] = self._reaction(p, percept, src, news, since)
            if cue_at is not None:          # where to look, if it is a where
                p["_cue"] = (float(cue_at[0]), float(cue_at[1]), now, percept)
        if pend is None:
            return (percept, src, news, None) if go else None
        if now < pend["due"]:
            return None                       # still answering
        stage = pend["stage"]
        out = (pend["percept"], pend["src"], pend["news"])
        if stage == "flinch":
            pend["stage"], pend["due"] = "react", pend["react"]
            return out + ("flinch",)
        if stage == "react" and pend["go"] > now:
            pend["stage"], pend["due"] = "go", pend["go"]
            return out + ("premove",)
        del p["_react"]
        return out + ("go" if stage == "go" else None,)

    def _reaction(self, p, percept, src, news, since):
        """The wait one percept starts (`_react`), from this person's own
        times (`sheet`, policy.REACT): a ready body answers in a simple
        reaction time, a surprised one in an organised act's time and
        flinches first at a thing coming at it, and a noise is answered at
        once (a head turning). A cue of a danger elsewhere also sets when
        the pre-movement ends. Perceiving anything but a noise leaves the
        body alert (`_primed`) for the next thing."""
        now, (t0, was) = self.tick, since
        primed = self._primed(p) if was is None else was
        tk = self._ticks
        react = t0 + tk(sheet(p, "ready_s" if primed or percept in ORIENTING
                              else "surprise_s"))
        stage, due = "react", react
        if not primed and percept in FLINCH_ON and now < react:
            stage, due = "flinch", max(t0 + tk(sheet(p, "flinch_s")), now)
        go = -1
        if percept in PREMOVE_ON:
            go = t0 + tk(max(sheet(p, "premove_s"), PREMOVE_MIN_S))
        if percept not in ORIENTING:
            p["_alert"] = now
        return {"percept": percept, "src": src, "news": news, "since": t0,
                "stage": stage, "due": due, "react": react, "go": go}

    @staticmethod
    def _ticks(s):
        """Seconds as whole ticks, at least one: nothing is answered on the
        tick it is perceived."""
        return max(1, int(round(s / TICK_S)))

    def _primed(self, p):
        """Is this body READY for what comes next: `ready` on its sheet,
        committed to something already, or alarmed (or in a fight) within
        `alert_s`? A ready body answers in a simple reaction time."""
        return bool(p.get("ready") or p.get("emergency")
                    or (self.tick - p.get("_alert", -10 ** 9)) * TICK_S
                    < WILL["alert_s"])

    def _seen_at(self, p, b):
        """(tick, ready then) when this body first saw flier `b`, or None."""
        return (b.get("_seen") or {}).get(p["pid"])

    def _in_time(self, p, b):
        """Has this body had time to answer flier `b` since it SAW it? The
        catch gate, counted from the sighting (`_watch_fliers`): a reach
        can start no sooner than a reaction, a ready body's or a surprised
        one's as it was when it saw the thing."""
        seen = self._seen_at(p, b)
        if seen is None:
            return False
        t, primed = seen
        return self.tick >= t + self._ticks(
            sheet(p, "ready_s" if primed else "surprise_s"))

    def _nearest_cell(self, b, eye):
        """The cell of flier `b` nearest the point `eye`, or None."""
        at = np.asarray(self._fly_pose(b), np.float64).reshape(-1, 3)
        if not len(at):
            return None
        return at[int(np.argmin(((at - np.asarray(eye, np.float64)[None]) ** 2)
                                .sum(axis=1)))]

    def _watch_fliers(self, p, eye, gaze):
        """Note the tick each thing in the air (and each person falling)
        first comes into this body's view: in the field of view
        (`_in_cone`), within `see_m`, with a clear line from the head
        (`rays`). It is kept on the flier (`_seen`, pid -> (tick, ready
        then)) and read by the catch rows and the percepts that answer it
        (row P22: they needed no sight before)."""
        if not self.bodies:
            return
        see = WILL["see_m"] / max(0.1 * self.scale, 1e-9)
        e = np.asarray(eye, np.float64)
        todo = []
        for b in self.bodies:
            if not b.get("fly") or b.get("part") \
                    or b.get("owner") == p["name"] \
                    or p["pid"] in (b.get("_seen") or {}):
                continue
            # ANY OF IT IS ENOUGH: its nearest cell and its highest (a
            # falling man's head over the post his legs are behind)
            at = np.asarray(self._fly_pose(b), np.float64).reshape(-1, 3)
            if not len(at):
                continue
            for c in (at[int(np.argmin(((at - e[None]) ** 2).sum(axis=1)))],
                      at[int(np.argmax(at[:, 2]))]):
                if float(np.linalg.norm(c - e)) <= see \
                        and self._in_cone(gaze, eye, c):
                    todo.append((b, c))
        if not todo:
            return
        ok = self.rays([eye] * len(todo), [c for _, c in todo],
                       skip=[(p["pid"], -1)] * len(todo))
        primed = self._primed(p)
        for (b, _c), seen in zip(todo, ok):
            if seen and p["pid"] not in (b.get("_seen") or {}):
                b.setdefault("_seen", {})[p["pid"]] = (self.tick, primed)

    def _runners(self, p, eye, gaze, vox_m):
        """Who this body SEES RUN, now: pid -> whether it can tell who (in
        the middle of the view, `_in_focus`).

        Seen, so in the field of view, within `run_see_m`, with a clear line
        head to head; and running, by the speed it is SEEN to move: watched
        without a break (`_watch`, pid -> the tick it came into view) for at
        least `WILL_TRAIL` ticks, and over those ticks its own `_trail` says
        it covered the ground at `runner_mps` or more. Not its `fleeing`
        flag, which is the other body's mind (row P20)."""
        far = WILL["run_see_m"] / max(vox_m, 1e-9)
        cand = []
        for q in self.persons:
            qe = q.get("_eye")
            if q is p or q["safe"] or not q["alive"] or qe is None:
                continue
            if float(np.hypot(qe[0] - eye[0], qe[1] - eye[1])) > far \
                    or not self._in_cone(gaze, eye, qe):
                continue
            cand.append(q)
        was = p.get("_watch") or {}
        now = {}
        if cand:
            ok = self.rays([eye] * len(cand), [q["_eye"] for q in cand],
                           skip=[(p["pid"], q["pid"]) for q in cand])
            for q, seen in zip(cand, ok):
                if seen:
                    now[q["pid"]] = was.get(q["pid"], self.tick)
        if now or was:
            p["_watch"] = now
        out = {}
        for q in cand:
            t0 = now.get(q["pid"])
            if t0 is None or self.tick - t0 < WILL_TRAIL:
                continue                      # not watched long enough to tell
            pts = [r for r in (q.get("_trail") or ())
                   if r[0] >= self.tick - WILL_TRAIL]
            if len(pts) < 2 or pts[-1][0] - pts[0][0] < WILL_TRAIL - 2:
                continue
            v = float(np.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])) \
                * vox_m / ((pts[-1][0] - pts[0][0]) * TICK_S)
            if v >= WILL["runner_mps"]:
                out[q["pid"]] = self._in_focus(gaze, eye, q["_eye"])
        return out

    def _notice_fire(self, p, eye, gaze, fire_at, vox_m):
        """The burning cell this body NOTICES, or None (row P4).

        Not a radius: a flame anywhere in the field of view, within `see_m`
        and in a clear line from the head, counts by how much of the view it
        fills -- its area over the distance squared (steradians), at full
        weight in the middle of the view and at `periphery` at its edge --
        and the fire is noticed when the sum passes `flame_sr`. Every lit
        cell in view counts, not the 25 nearest: rays go to the 8 nearest
        and 8 spread over the rest, and the rest are counted in the same
        proportion. The cell returned is the nearest one seen."""
        see = WILL["see_m"] / max(vox_m, 1e-9)
        e = np.asarray(eye, np.float64)
        rel = fire_at.astype(np.float64) - e[None]
        d = np.sqrt((rel ** 2).sum(axis=1))
        h = np.hypot(rel[:, 0], rel[:, 1])
        fx, fy = gaze
        fn = float(np.hypot(fx, fy)) or 1.0
        cos = (rel[:, 0] * fx + rel[:, 1] * fy) / (fn * np.maximum(h, 1e-9))
        close = h < 1.5                       # at arm's reach, as `_in_cone`
        view = (d <= see) & (close | (cos >= np.cos(np.radians(
            WILL["fov_deg"] / 2.0))))
        idx = np.flatnonzero(view)
        if not len(idx):
            return None
        idx = idx[np.argsort(d[idx], kind="stable")]
        if len(idx) > 16:
            rest = idx[8:]
            idx = np.concatenate([idx[:8], rest[np.linspace(
                0, len(rest) - 1, 8).astype(np.int64)]])
        n = int(view.sum())
        ok = self.rays([eye] * len(idx), fire_at[idx].astype(np.float32),
                       skip=[(p["pid"], -1)] * len(idx), lit=True)
        if not ok.any():
            return None
        w = np.where(close[idx] | (cos[idx] >= np.cos(np.radians(
            WILL["focus_deg"] / 2.0))), 1.0, WILL["periphery"])
        dm = np.maximum(d[idx] * vox_m, 0.25)
        sal = float((vox_m * vox_m * w / dm ** 2)[ok].sum()) * n / len(idx)
        if sal < WILL["flame_sr"]:
            return None
        return tuple(int(v) for v in fire_at[idx[ok][0]])

    def _law_will(self):
        """The WILL layer, in three separate parts — this is the MENU SEAM.

        1. NOTICE. The body's senses fire a percept: it sees flame in its
           cone, coughs on smoke, hears a shout through a wall, watches
           someone bolt past. Physics decides this; nothing chooses.
        2. OFFER. The sim builds the MENU: every response this body could
           actually carry out this instant, each one checked against the
           world (a "flee" option exists only if a route to a door really
           exists). Nothing chooses here either.
        3. PICK. A POLICY takes the situation and the menu and returns one
           index. Today that is the reflex table and the answer is the same
           one the table always gave. Tomorrow it can be a language model,
           and later a model distilled from these very decisions.

        The split is the whole point. Because the menu is built from what the
        sim just computed, a picked option cannot be illegal — the mind
        supplies taste, never facts, and can never invent a door. Every pick
        is written to `self.traces` with the full menu beside it, which is
        what makes a harvest possible: you cannot learn from a choice without
        knowing what else was on offer."""
        # A ROOM WITH NO DOOR STILL HAS PEOPLE IN IT. This used to return unless
        # the world registered an exit, which quietly made every mind in the sim
        # conditional on there being somewhere to escape to: a scene with no door
        # — two men on a clifftop, say — had bodies that could not decide
        # anything at all, and not because anything physical stopped them.
        # Escaping is one thing a body might want, not the reason it has a will.
        if not self.persons:
            return
        vox_m = 0.1 * self.scale
        # THE FIRES WERE ALREADY FOUND, by `_law_perceive`, this tick and under
        # this same gate (it keeps the fires' books with them). Asked again
        # only if perception did not run this tick, which `step` never does.
        now, self._burning_now = self._burning_now, None
        if now is not None and now[0] == self.tick:
            fire_at = now[2]
        else:
            burning = self.burning() if any(p["alive"] and p["awake"]
                                            for p in self.persons) else None
            fire_at = np.argwhere(burning) if burning is not None \
                and burning.any() else None
        for p in self.persons:
            if not (p["alive"] and p["awake"]) or p["safe"] or p.get("asleep"):
                p.pop("_react", None)        # out of it: nothing is answered
                continue
            comp, sl = self._person_cells(p)
            if comp is None:
                self._decide_falling(p)      # in the air, and not out of choices
                continue
            cells = self._claim_cells(p, comp, sl)
            ax_, ay_ = (float(cells[:, 0].mean()), float(cells[:, 1].mean()))
            p["anchor"] = (int(round(ax_)), int(round(ay_)))
            eye = (ax_, ay_, float(cells[:, 2].max()))
            p["_eye"] = eye
            if "facing" not in p:
                p["facing"] = (1.0, 0.0)
            # THE HEAD IS NOT THE FEET. A gaze sweeps around whichever way the
            # body is pointed, so a person crossing a room still looks about as
            # they go. It used to sweep by turning `facing` itself, and only
            # while the body had nowhere to be — survivable while standing
            # still was common, and a blindfold the moment bodies always had
            # somewhere to go: measured, a witness with a clear window onto a
            # burning bench walked about in front of it for 260 ticks and never
            # once perceived the fire, because one vector was doing both jobs
            # and every step reset it.
            if self.tick % WILL["scan_every"] == 0 and not p.get("emergency"):
                p["gaze_i"] = (p.get("gaze_i", 0) + 1) % len(GAZE_SWEEP)
            off = 0.0 if p.get("emergency") else GAZE_SWEEP[p.get("gaze_i", 0)]
            fx, fy = p["facing"]
            cg, sg = np.cos(off), np.sin(off)
            gaze = (fx * cg - fy * sg, fx * sg + fy * cg)
            # UNLESS IT HAS DECIDED TO LOOK SOMEWHERE. A chosen direction wins
            # over the sweep for as long as the body holds it — which is what
            # makes looking an ACT rather than a thing that happens to a head.
            want_look = p.get("look")
            if want_look is not None and not p.get("emergency") \
                    and self.tick - want_look[2] < WILL["look_for"]:
                gaze = (want_look[0], want_look[1])
            p["gaze"] = gaze                  # what the eyes are pointed at
            if self.tick % WILL["look_every"] == 0:   # look where the eyes
                self._look_levels(p, eye, cells)       # point, and REMEMBER it
            # WHERE THIS BODY IS, kept for a few ticks: what anyone watching
            # it sees it do (`_runners`). A fact about the body, not a mind.
            p["_trail"] = (tuple(p.get("_trail") or ())
                           + ((self.tick, ax_, ay_),))[-(WILL_TRAIL + 1):]
            # WHAT IS IN THE AIR, AND WHEN IT WAS SEEN. A catch can start no
            # sooner than the body answers what it SAW (`_in_time`), so the
            # moment a flier comes into view is kept, whatever else the body
            # is busy with: a committed body still has eyes.
            self._watch_fliers(p, eye, gaze)
            # SENSES RUN WHETHER OR NOT THE BODY IS BUSY. Having somewhere to
            # be is not the same as being in an emergency: a body strolling
            # across a room must still notice the room is alight. Gating the
            # percepts on "is moving" made a wandering person blind.
            # Under the "onset" law the senses run EVERY tick, committed or
            # not: whether a body decides is `_attend`'s question, and a
            # percept that goes on is not news, so it costs nothing.
            # (Under the default "latch" a committed body still LOOKS -- the
            # layout, the danger map and the fliers above -- but it does not
            # weigh percepts or decide again: the user's ruling, onset is
            # opt-in.)
            act = None
            if not p.get("emergency") or self.attention == "onset":
                # WHAT EACH PERCEPT CAME FROM (V3): the event the attention
                # law (`_attend`) calls news, and a decision's cause. It
                # changes nothing about which percept fires.
                percept, src, since, cue_at = None, None, (self.tick, None), None
                mind = self.minds[p["pid"]]
                running = self._runners(p, eye, gaze, vox_m)
                if fire_at is not None:
                    hit = self._notice_fire(p, eye, gaze, fire_at, vox_m)
                    if hit is not None:
                        percept = "sees_fire"        # a flame it can SEE, big
                        src = self._fire_of(hit)     # enough to notice ...of
                        cue_at = hit                 # THIS fire
                if percept is None and p.get("skin", AMBIENT) > BODY["feel_T"]:
                    # BEING COOKED IS NOTICING, and it does not go through the
                    # eyes. Sight needs a clear line from the head, in the
                    # direction the head is turned — so a body STANDING ON a
                    # fire did not see it, because the fire was under its own
                    # feet and behind its own legs. Measured the day stepping up
                    # was built: a man idly wandered onto a burning crib and
                    # suffocated there over two hundred ticks without once
                    # perceiving the thing he was stood in.
                    percept = "scorched"
                    src = self._fire_near(eye)
                    cue_at = None
                # SOMETHING IS COMING, OR SOMEBODY IS FALLING PAST: within a
                # hand's reach, which is the only range at which noticing is
                # any use to a hand, and SEEN, which it did not use to need
                # (row P22). It was perceived the moment it came into view
                # (`_watch_fliers`), so that is when the answer starts.
                if percept is None:
                    near = [b for b in self._falling_near(p, cells)
                            if self._seen_at(p, b) is not None]
                    if near:
                        percept = "sees_falling"     # somebody is going past you
                        src = near[0].get("falls_ev")
                        since = self._seen_at(p, near[0])
                        cue_at = self._nearest_cell(near[0], eye)
                if percept is None:
                    near = [b for b in self._in_flight_near(p, cells)
                            if self._seen_at(p, b) is not None]
                    if near:
                        percept = "sees_thrown"      # it came from whoever
                        by = near[0].get("by_pid")   # threw it
                        src = None if by is None \
                            else self._last_act(by, "swings", "thrown")
                        since = self._seen_at(p, near[0])
                        cue_at = self._nearest_cell(near[0], eye)
                if percept is None and p["smoke"] > 0.05:
                    percept = "chokes"                   # coughing IS noticing
                    src = mind.harm.get("air")
                    cue_at = None
                if percept is None:
                    for q in self.persons:
                        # EARS AND EYES ARE NOT THE SAME SENSE and had the same
                        # guard. Seeing someone RUN requires them to be running;
                        # hearing them shout does not — and a body that shouted
                        # "Fire!" without running was inaudible at any distance,
                        # measured. So sound is asked first and on its own.
                        if q is p or q["safe"]:
                            continue
                        qeye = q.get("_eye", (*q["anchor"], eye[2]))
                        said = q.get("_shouted")
                        if (said is not None
                                and self.tick - said[0] < 30
                                and self._hears(eye, qeye, said[2])):
                            # AND WHAT IT MEANT. A line nobody has given a
                            # meaning is a noise: heard, and no reason to run.
                            means = MEANS.get(said[1])
                            if means:
                                percept = means          # ears have no cone and
                                src = self.minds[q["pid"]].said_ev
                                cue_at = qeye
                                break                    # a wall only MUFFLES
                        # SEEN TO RUN: by its speed, in view (`_runners`),
                        # and not by its `fleeing` flag, which is the other
                        # body's mind (row P20). Who it is is known only from
                        # the middle of the view; from the edge, someone ran.
                        seen = running.get(q["pid"])
                        if seen is not None:
                            percept = "sees_runner"
                            src = self.minds[q["pid"]].go_ev if seen else None
                            cue_at = qeye
                            break
                # A BARE NOISE COMES LAST. A shout is a noise too, and asked
                # first it turned "Fire!" into "something, over there": the
                # hearer looked round instead of running. What a line MEANS
                # outranks that it was loud.
                nz_ = p.get("noise")
                if percept is None and nz_ is not None \
                        and self.tick - nz_["tick"] <= 1:
                    percept = "hears_noise"              # something, over there
                    src = nz_.get("ev_id")
                if percept is not None:
                    mind.noticed = (self.tick, percept, src)
                # a body with NOTHING happening to it still has a life. If no
                # percept fires it weighs its ordinary options — go and look at
                # what it has not seen, walk somewhere it has, or stand still —
                # through the same menu, picked by the same policy, written to
                # the same trace. Idling is a choice, not a gap between choices.
                # A PERCEPT DOES NOT WAIT ITS TURN. decide_every paces ordinary
                # deliberation — nobody re-weighs their afternoon every tick —
                # but an alarm must land the moment it arrives. Making percepts
                # queue behind the cooldown meant a shout could expire unheard
                # while the hearer was still inside its own 30-tick pause.
                # (It waits its REACTION TIME instead, which is a different
                # thing: `_react`.)
                # A LINKED MIND'S OWN WAKE (`intent["wake_at"]`, or a blow
                # it asked to be told of) IS NOT A PERCEPT, and it does not
                # queue behind the head turning to a noise: the noise is
                # answered when it is due, by the eyes, and the wake now.
                intent = p.get("intent")
                woke = bool(intent) and (
                    (intent.get("wake_at") is not None and self.tick >= intent["wake_at"])
                    or (intent.get("felt") and self._struck_now(mind)))
                go, news = self._attend(p, percept, src)
                act = self._react(p, percept, src, news, go, since, cue_at)
                if act is None and woke and (p.get("_react") is None
                                             or rank(p["_react"]["percept"]) == 0):
                    act = (None, None, None, None)
            elif p.get("_react") is not None:
                act = self._react(p, None, None, None, False, (self.tick, None),
                                  None)
            if act is not None:
                self._decide(p, act[0], cells, ax_, ay_, eye, vox_m,
                             news=act[2], src=act[1], phase=act[3])
            goal = p.get("goal")
            if goal is None:
                p["fleeing"] = False
                continue
            # arrival is measured from the BODY, not from its centre: you are
            # at the door when part of you is at the door. A doorway in an
            # outer wall has no room for a centre, so measuring from the middle
            # meant a body could stand IN the opening and not have arrived.
            reach = float(np.abs(cells[:, :2] - np.array(goal)).max(axis=1).min())
            # (but a spot to STAND AT is reached when the middle is on it:
            # within the cell `_stride` stops stepping toward its last one at)
            mid = p.get("_goal_mid")
            if mid:
                reach = max(abs(ax_ - goal[0]), abs(ay_ - goal[1]))
            there = reach < 1.0 if mid else reach * vox_m <= WILL["arrive_m"]
            # AND NOT ARRIVING IS ALSO AN OUTCOME. Arriving was the only thing
            # that ever cleared a goal, so a body that could not get where it
            # was going held that goal for the rest of the run: measured, a
            # witness spent 180 ticks walking into a wall two cells in front of
            # it, and never looked at the window it had been put there to see
            # through. The old roam target hid this by churning the goal every
            # ten ticks whatever happened, which is not the same as a body
            # deciding it has had enough.
            #
            # Getting no NEARER is the test, not time spent — a long walk is
            # not a failed one. Half a cell of progress counts, so a body
            # shuffling round a corner is not mistaken for a body stuck.
            if not there:
                best = p.get("_near")
                if best is None or best[2] != tuple(goal) or reach < best[0] - 0.5:
                    p["_near"] = (reach, self.tick, tuple(goal))
                elif self.tick - best[1] > WILL["give_up"]:
                    # TWO STRIKES, because giving up on the DESTINATION and
                    # giving up on the WAY THERE are different sizes of
                    # decision. Nobody in a burning room abandons the only door
                    # in the building because they were blocked for two
                    # seconds — they try another way to the same door. So the
                    # first failure throws away the ROUTE and keeps the goal,
                    # and only a second one says the place itself is no good.
                    tries = p.get("_tried", (None, 0))
                    tries = (tuple(goal), tries[1] + 1) \
                        if tries[0] == tuple(goal) else (tuple(goal), 1)
                    p["_tried"] = tries
                    p["_near"] = (reach, self.tick, tuple(goal))
                    if tries[1] < 2:
                        p["_path"] = None         # go round; same destination
                        self.emit("retries", actor=p["pid"],
                                  text=f"{p['name']} tries another way",
                                  to=p, site="will:retry")
                        continue
                    # AND IT REMEMBERS BAULKING, or the menu offers the same
                    # spot next decision and nothing has changed. This is the
                    # ordinary kind of knowledge — I tried that and could not
                    # get there — and it fades, because a doorway that was
                    # blocked an hour ago may not be now.
                    p.setdefault("_baulked", {})[tuple(goal)] = self.tick
                    p["goal"], p["_path"], p["_near"] = None, None, None
                    p["_tried"] = (None, 0)
                    p["fleeing"] = False
                    if self.attention == "onset":
                        p["emergency"] = False    # what it was for is over
                    self.emit("gives_up", actor=p["pid"],
                              text=f"{p['name']} gives up on getting there",
                              to=p, site="will:give_up")
                    continue
            if there:
                p["_near"] = None
                p.pop("_goal_mid", None)
                if tuple(goal) in {tuple(e) for e in self.exits}:
                    p["safe"] = True
                    self.emit("arrives", actor=p["pid"],
                              text=f"{p['name']} reaches the doorway",
                              to=p, site="will:arrive")
                    self._leave(p, cells)
                    who = p.get("dragging")
                    if who:                       # whoever came with them is out
                        for q in self.persons:    # too — that is all rescuing is
                            if q["name"] == who and q["alive"]:
                                q["safe"] = True
                                qc, qsl = self._person_cells(q)
                                if qc is not None and qc.any():
                                    qcell = self._claim_cells(q, qc, qsl)
                                    self._leave(q, qcell)
                                self.emit("rescues", actor=p["pid"],
                                          target=("person", q["pid"]),
                                          text=f"...hauling {who} clear",
                                          to=p, site="will:rescue")
                        p["dragging"] = None
                else:                                # arrived somewhere ordinary:
                    p["goal"] = None                 # pick again next time round
                    p["fleeing"] = False
                    p["_path"] = None
                    if self.attention == "onset":
                        p["emergency"] = False
                continue
            ex = goal
            room = p.get("_wander")
            room = room[1] if room is not None and room[0] == tuple(goal) \
                else None
            if room is not None:
                # AN IDLE WALK STAYS IN ITS ROOM. The route the menu showed
                # was in the room; the legs plan the way again every half
                # second or so (and after a block), from where the body is
                # now and with whatever stands in the way, and that way need
                # not be. Planned here, as `_stride` plans it, and given up
                # (and remembered as given up) if it leaves the room.
                if (not p.get("_path")) or self.tick - p.get("_planned", -99) > 30:
                    obj = self._held_cells(p) if p.get("held") else None
                    walk = np.concatenate([cells, obj]) \
                        if obj is not None and not self._underfoot(obj) else cells
                    p["_path"] = self._plan_path(walk, ex, p)
                    p["_planned"] = self.tick
                if p["_path"] and not self._in_room(room, p["_path"]):
                    p.setdefault("_baulked", {})[tuple(goal)] = self.tick
                    p["goal"], p["_path"], p["_near"] = None, None, None
                    p["_wander"] = None
                    continue
            # THE LEGS DO THE REST (`_stride`, motion.py): the pace in metres a
            # second for this gait, the route, the turn, each step and its
            # sound, and what is carried or dragged along.
            self._stride(p, cells, ex, ax_, ay_)

    def _decide_falling(self, p):
        """A body in the air is not a body with nothing to decide.

        It has no cells on the lattice while it flies, which is why the will
        layer used to skip it entirely — and that quietly made falling the one
        thing in the sim nobody could do anything about. It cannot walk, and
        there is nothing under it to push against, so the menu is SHORT. But
        the one choice still open is how to meet the ground, and that is the
        difference between getting up and not.

        The same seam as every other decision: a menu of what is really
        possible, a policy that picks an index, and the whole menu written to
        the trace beside the pick."""
        if not (p["alive"] and p["awake"]) or p["safe"]:
            return
        b = next((x for x in self.bodies if x.get("owner") == p["name"]
                  and not x.get("part")), None)
        if b is None:
            return
        if self.tick - p.get("_fell", -99) < WILL["decide_every"]:
            return
        p["_fell"] = self.tick
        menu = [{"key": "stay as you are", "tag": "stay", "verb": "stay"}]
        if not p.get("rolling"):
            menu.append({"key": "roll on landing", "tag": "roll", "verb": "roll"})
        sit = {"tick": self.tick, "who": p["name"], "percept": "falling",
               "reflexes": p.get("reflexes", REFLEXES),
               "smoke": round(float(p["smoke"]), 3),
               "blood_o2": round(float(p["blood_o2"]), 3),
               "burn": round(float(p["burn"]), 3),
               "hurt": round(float(p["hurt"]), 3),
               "holding": p.get("dragging"),
               "falling": True,
               "knows_a_way_out": False, "seen_of_the_world": 0.0,
               "others_in_earshot": 0}
        i = self.policy.pick(dict(sit, limb="legs", chosen={}), menu)
        i = i if isinstance(i, int) and 0 <= i < len(menu) else 0
        opt = menu[i]
        prog = opt["key"] if opt["tag"] != "stay" else "stay"
        # caused by the fall it is deciding about (`falls`, on the body)
        dec = self._decides(p, "falling", b.get("falls_ev"), prog,
                            "falling:decides")
        self.traces.append({"tick": self.tick, "who": p["name"],
                            "percept": "falling", "situation": sit,
                            "menus": {"legs": [o["key"] for o in menu]},
                            "pick": {"legs": opt["key"]},
                            "tags": {"legs": opt["tag"]},
                            "program": prog,
                            "by": getattr(self.policy, "name", "?"),
                            "mind": self._mind_row(p, dec)})
        if opt["verb"] == "roll":
            p["rolling"] = True
            with self.because(dec, "call"):
                self.emit("rolls", actor=p["pid"],
                          text=f"{p['name']} tucks to roll",
                          to=p, site="falling:roll")

    def _decides(self, p, percept, src, program, site):
        """Log one decision as a `decides` event (V5) and return its id: the
        cause every act of the decision is given (`because`).

        Its cause is the event the percept came from (`src`, a box while
        that event waits for the tick's end), and the link says how the
        news got here: "near" for a scorching blamed on the nearest fire
        (`_fire_near`, a guess by place), "carried" for the rest -- the id
        rode in on a percept row, a sound, a Mind's act slot. A decision
        with no percept behind it (an idle body choosing what to do next)
        has no cause. `why` is the percept, `result` the program picked.
        Not `visible`: a decision is in the head, and nobody sees or hears
        it (`_witness` passes it by)."""
        link = None if src is None else \
            "near" if percept == "scorched" else "carried"
        return self.emit("decides", actor=p["pid"], why=percept,
                         result=program, cause=src, link=link, visible=False,
                         site=site).id
