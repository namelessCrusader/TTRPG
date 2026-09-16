"""Turn engine: affordance menu · action-cost time · observation · invariants.

The menu is a QUERY over the unified substrate (cells in reach × their tags →
afforded macros), not an authored list. Time advances only on a costed action,
so `look`/`wait` are the "pause to study". invariants() runs every turn.

Rendering assumes 2D (len(dims)==2). The rest of the engine is dimension-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import effects as fx
from . import mind
from . import parts
from . import perceive
from . import reactions
from . import social
from . import spatial
from .state import FLUIDS, World, is_solid, mat_tags

DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0),
        "NE": (1, -1), "NW": (-1, -1), "SE": (1, 1), "SW": (-1, 1)}
DIRWORD = {"N": "north", "S": "south", "E": "east", "W": "west", "NE": "northeast",
           "NW": "northwest", "SE": "southeast", "SW": "southwest"}


def _at(name):
    """A spot as fiction, never a tuple: 'at your feet' / 'to the north'."""
    return "at your feet" if name == "here" else f"to the {DIRWORD.get(name, name)}"
EARSHOT = 5     # you can address anyone visible this close (walls block)
FLUID_ITEMS = {"oil": "pour oil on", "acid": "throw acid on", "water": "splash water on"}
BLOCK_GLYPH = {"wood": "#", "stone": "%", "metal": "+"}


@dataclass
class Option:
    label: str
    effects: list = field(default_factory=list)
    cost: int = 1
    social: tuple | None = None   # (npc_id, intent) — resolved via the LM brain at step time
    target: tuple | None = None   # the cell this action acts on (for the spatial UI)
    wait: bool = False            # fast-forward until something notable happens
    steps: list | None = None     # canonical composer steps — checked verbs roll at step time
    verb: str = ""                # the action's head word — the piece-table groups by this
    args: dict = field(default_factory=dict)   # named parts (what/on/to/dir…) — listed pieces
                                  # only: seen-target ids, ordinals, directions. Never numbers.


def _notable(entry) -> bool:
    """Events worth handing control back for — deeds, deaths, bursts, speech.
    Routine ticks (chars, heat, pathing) are dead air the player may skip."""
    if entry.kind == "speech":
        return True
    if entry.kind == "set_tag" and entry.data.get("tag") in (
            "taken", "dead", "broken", "collapsed", "on_fire", "burning", "charred"):
        return True
    if entry.kind == "set_edge":
        return True
    if entry.kind == "gm_move":
        return True                          # a Director beat is a story beat
    return False


def _advance(world: World, ticks: int = 1, wait_until: bool = False, max_wait: int = 0) -> None:
    """THE turn loop — the only place time moves. NPCs act every tick, so the
    world doesn't freeze while the player takes a long action. Waiting is paced
    by the drama pulse: tense scenes hand control back fast, calm ones skip far."""
    if wait_until and not max_wait:
        max_wait = 3 if drama(world) > 60 else 8
    from . import director
    for _ in range(max_wait if wait_until else max(1, ticks)):
        mark = len(world.log)
        reactions.tick(world)
        mind.take_turns(world)
        director.update(world)
        if wait_until and any(_notable(e) for e in world.log[mark:]):
            break


def _fingerprint(world: World):
    """Cheap structural hash of everything that IS the world (log excluded)."""
    ents = tuple(sorted((e.id, e.pos, tuple(sorted(e.tags)), e.material,
                         tuple(sorted((k, round(v, 3)) for k, v in e.props.items())))
                        for e in world.entities.values()))
    cells = tuple(sorted((p, c.material, c.floor, round(c.heat, 1),
                          tuple(sorted((m, round(ml)) for m, ml in c.fluids.items())),
                          tuple(sorted(c.tags))) for p, c in world.cells.items()))
    edges = tuple(sorted((k, tuple(sorted((kk, round(vv, 2)) for kk, vv in v.items())))
                         for k, v in world.edges.items()))
    clocks = tuple(sorted((cid, c.fill, c.fired) for cid, c in world.clocks.items()))
    return (ents, cells, edges, clocks)


def changes_world(world: World, effects: list) -> bool:
    """Jericho's validity trick: dry-run effects against a forked world and ask
    whether anything real moved. Options that change nothing are not options."""
    import copy
    if not effects:
        return False
    saved_log, saved_pending = world.log, world.pending
    world.log, world.pending = [], []
    try:
        fork = copy.deepcopy(world)
    finally:
        world.log, world.pending = saved_log, saved_pending
    before = _fingerprint(fork)
    try:
        fx.apply_all(fork, copy.deepcopy(effects))
    except Exception:
        return True                     # let the real apply surface the error
    return _fingerprint(fork) != before


def drama(world: World) -> float:
    """The scene's pulse (L4D Director, compacted): a quiet room reads ~0, a
    witnessed theft amid a blaze reads high. One scalar for pacing systems —
    entirely derived from meters we already keep (salience, fires, fresh deeds)."""
    pulse = sum(e.mind.get("salience", 0.0) for e in world.entities.values()
                if "person" in e.tags and "dead" not in e.tags)
    pulse += 8.0 * sum(1 for c in world.cells.values() if "on_fire" in c.tags)
    pulse += 5.0 * sum(1 for e in world.log[-40:] if _notable(e))
    return pulse


def _has(player, item):
    return f"inv:{item}" in player.tags


def _flammable(world, pos):
    c = world.cell(pos)
    if any(FLUIDS.get(m, {}).get("flammable") and ml > 0 for m, ml in c.fluids.items()):
        return True
    if "flammable" in mat_tags(c.material):
        return True
    return any("flammable" in mat_tags(e.material) and "on_fire" not in e.tags and "broken" not in e.tags
               for e in world.entities_at(pos) if "item" in e.tags)


def _fluid_worthwhile(world, pos):
    """Is splashing/pouring a fluid HERE worth a menu line? Only where it can act:
    a fire to douse, something flammable, a person or container, an existing pool,
    or your own feet (self-application). Empty air in 8 directions is not — that
    was the menu bloat the playtest flagged."""
    c = world.cell(pos)
    if "on_fire" in c.tags or _flammable(world, pos) or c.fluids:
        return True
    return any("person" in e.tags or "container" in e.tags for e in world.entities_at(pos))


def affordance_menu(world: World, actor, slice_=None) -> list[Option]:
    # The D8 legality predicate: ONE menu, any actor, PERCEPTION-FILTERED. Entity-targeted
    # options bind only to the actor's FRESH belief nodes (its perceive() slice), so the menu
    # structurally cannot offer a grab/strike/gift on something unperceived (findings 1 & 3);
    # stale memories may inform navigation, never a contact action. Pass the frozen `slice_`
    # for double-buffered play (everyone decides from slice-start belief; execution re-validates
    # against the live world) — omitting it perceives live. Cell facts (walls, pools, fire) are
    # read from ground truth only at arm's reach, where they are perceivable by definition.
    # NOTE: cause-strings still say "you …" — placeholder narration; voicing is D11's job.
    if slice_ is None:                                 # live path: refresh sight THEN render —
        perceive.observe(world, actor, world.tick)     # perceive() alone only renders the stored
        slice_ = perceive.perceive(world, actor, now=world.tick)   # belief and would go stale
    seen = [n for n in slice_["nodes"] if n["fresh"] and not n["self"]]
    at = {}
    for n in seen:
        at.setdefault(tuple(n["pos"]), []).append(n)

    px, py, pz = actor.pos
    opts = [Option("look around (study the scene)", [], cost=0, verb="look")]
    targets = [("here", actor.pos)] + [
        (n, (px + dx, py + dy, pz)) for n, (dx, dy) in DIRS.items() if world.in_bounds((px + dx, py + dy, pz))
    ]
    for name, tp in targets:
        if _has(actor, "torch") and _flammable(world, tp) and "on_fire" not in world.cell(tp).tags:
            opts.append(Option(f"ignite {name}", [fx.spark(tp, f"you set a spark {_at(name)}")], target=tp,
                               verb="ignite", args={"at": name}))
        for item, verb in FLUID_ITEMS.items():
            # only offer a fluid action where it would DO something — on a fire, a
            # flammable/container/person, or an existing pool. Splashing empty air
            # in all 8 directions was 24 dead menu lines (the playtest bloat).
            if _has(actor, item) and _fluid_worthwhile(world, tp):
                opts.append(Option(f"{verb} {name}",
                                   [fx.add_fluid(tp, item, 400.0,
                                                 f"you {verb} the ground {_at(name)}")], target=tp,
                                   verb="pour", args={"what": item, "at": name}))
        for n in at.get(tp, ()):                       # entities the actor SEES at this cell
            tags, nid, nname = set(n["tags"]), n["ref"], n["name"]
            live = world.entities.get(nid)             # parts ADVERTISE affordances (D2): options
            if live is not None:                       # that exist because a component does
                for lbl, effs, tgt, pverb, pargs in parts.affords(world, live, actor):
                    opts.append(Option(lbl, effs, target=tgt, verb=pverb, args=pargs))
            if "container" in tags and "broken" not in tags:
                dmg = -8.0 if "charred" in tags else -5.0     # charred = brittle: fire's aftermath
                opts.append(Option(f"smash the {nname}",      # is load-bearing, not decoration
                                   [fx.adjust_prop(nid, "hp", dmg, f"you smash the {nname}")], target=tp,
                                   verb="smash", args={"target": nid}))
            if "item" in tags and "taken" not in tags and "heavy" not in tags:
                opts.append(Option(f"pick up the {nname}",
                                   [fx.set_tag(nid, "taken", f"you pick up the {nname}"),
                                    fx.set_tag(actor.id, f"inv:{nid}")], target=tp,
                                   verb="take", args={"target": nid}))
            if "item" in tags and "heavy" in tags and tp != actor.pos:
                dest = (tp[0] + (tp[0] - px), tp[1] + (tp[1] - py), tp[2])
                if world.in_bounds(dest) and not is_solid(world.cell(dest).material):
                    opts.append(Option(f"push the {nname}",
                                       [fx.move(nid, dest, f"you heave the {nname} to {dest}")], target=tp,
                                       verb="push", args={"target": nid}))

    for lbl, effs, tgt, pverb, pargs in reactions.pack_affords(world, actor,
                                                               [n["ref"] for n in seen]):
        opts.append(Option(lbl, effs, target=tgt, verb=pverb, args=pargs))

    # social: anyone the actor SEES within earshot — conversation is not a contact sport
    carried = [world.entities[t[4:]] for t in actor.tags
               if t.startswith("inv:") and t[4:] in world.entities]
    for it in carried:
        opts.append(Option(f"drop the {it.name}",
                           [fx.clear_tag(actor.id, f"inv:{it.id}"), fx.clear_tag(it.id, "taken"),
                            fx.move(it.id, actor.pos, f"you set the {it.name} down")],
                           verb="drop", args={"what": it.id}))
    for n in seen:
        if "person" not in n["tags"] or "dead" in n["tags"] or n["d"] > EARSHOT:
            continue
        ep = tuple(n["pos"])
        eid, ename = n["ref"], n["name"]
        # the ACT of speaking is a sim event (earshot-gated, on the bus, logged); the WORDS are
        # a pick-time parameter — social=(eid, intent) resolves at step time, and the narration
        # around it stays the narrator's slot (D11). Thinking = the untargeted variant, later.
        opts.append(Option(f"speak to {ename}", social=(eid, None), target=ep,
                           verb="speak", args={"to": eid}))
        if max(abs(ep[0] - px), abs(ep[1] - py)) <= 1:            # arm's reach: violence & gifts
            opts.append(Option(f"attack {ename}", steps=[("attack", eid)], target=ep,
                               verb="attack", args={"target": eid}))
            cur = world.edges.get((eid, actor.id), {}).get("disposition", 0.0)
            for it in carried:
                opts.append(Option(
                    f"give the {it.name} to {ename}",
                    [fx.clear_tag(actor.id, f"inv:{it.id}"), fx.set_tag(eid, f"inv:{it.id}"),
                     fx.move(it.id, ep, f"you hand the {it.name} to {ename}"),
                     fx.set_edge(eid, actor.id, "disposition", cur + 0.4)], target=ep,
                    verb="give", args={"what": it.id, "to": eid}))

    for name, (dx, dy) in DIRS.items():
        np = (px + dx, py + dy, pz)
        if world.in_bounds(np) and not is_solid(world.cell(np).material):
            opts.append(Option(f"move {name}",
                               [fx.move(actor.id, np, f"you head {DIRWORD[name]}")], target=np,
                               verb="move", args={"dir": name}))
    for q in spatial.walk_neighbors(world, actor.pos):
        if q[2] > pz:
            opts.append(Option("climb up the stairs", [fx.move(actor.id, q, "you climb the stairs")], target=q,
                               verb="climb", args={"dir": "up"}))
        elif q[2] < pz:
            opts.append(Option("climb down the stairs", [fx.move(actor.id, q, "you climb down")], target=q,
                               verb="climb", args={"dir": "down"}))
    if "on_fire" in actor.tags:            # a burning body has an out — fight it (Haiku ask #1)
        opts.insert(1, Option("drop and roll to put out the flames!", steps=[("stop_drop_roll",)],
                              verb="drop_and_roll"))
    opts.append(Option("wait (until something happens)", [], wait=True, verb="wait"))
    # validity filter (Jericho): deterministic options must provably DO something.
    # Only idempotent-RISK kinds need the (deepcopy) dry-run; the rest always act.
    # (steps roll dice at execution; social resolves via the brain — both exempt.)
    RISK = {"spark", "set_tag", "clear_tag", "set_prop", "set_edge"}
    return [o for o in opts
            if o.social or o.wait or o.steps or not o.effects
            or not all(e.kind in RISK for e in o.effects)
            or changes_world(world, o.effects)]


def npcs_in_reach(world: World, actor) -> list:
    """[(id, name)] of living NPCs the actor can address: visible within earshot.
    Excludes the actor itself — required now that the actor may BE a person-tagged NPC
    (the player was never person-tagged, so this is a no-op for the player path)."""
    return [(e.id, e.name) for e in world.entities.values()
            if e.id != actor.id and "person" in e.tags and "dead" not in e.tags
            and spatial.visible(world, actor.pos, e.pos, EARSHOT)]


def free_text(world: World, player, text: str, picker=None) -> dict:
    """The player's 'do anything' box: utterance → composer → deltas → world.
    Inspect/refuse cost no time; physical steps cost their compiled ticks."""
    import re
    from . import composer, social, trace
    trace.begin(social.DATASTORE)          # open this turn's protoreasoning-trace capture
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(i |i'll |please |i want to |i try to )+", "", text, flags=re.I)
    text = text.rstrip("!.")
    # quoted input IS dialogue — no classification needed (one listener → speak)
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        inner = text[1:-1].strip()
        npcs = npcs_in_reach(world, player)
        if len(npcs) == 1:
            return say(world, npcs[0][0], inner)
        if npcs:
            return {"narrative": [f"Say that to whom? ({', '.join(n for _, n in npcs)})"],
                    "violations": []}
        return {"narrative": ["Nobody is close enough to hear you."], "violations": []}
    def _refused(msg):
        """NetPlay's stall rule: the third identical refusal becomes a nudge,
        typed on the log so it can be harvested as a golden-eval candidate."""
        trace.discard()          # a refused pick must never be learned as correct
        key = text.lower()
        world.stalls[key] = world.stalls.get(key, 0) + 1
        if world.stalls[key] >= 3:
            eff = fx.Effect("fallback", {"text": key}, "")
            eff.actor = "player"
            fx.apply(world, eff)
            return {"narrative": ["You've tried that — it isn't working. Perhaps something else?"],
                    "violations": []}
        return {"narrative": [msg], "violations": []}

    # compound sentences: ". . and/then <verb> . ." runs each clause in order —
    # both playtests caught the second clause being silently dropped
    for sep in (" and then ", ", then ", " then ", " and "):
        i = text.find(sep)
        while i > 0:
            head = text[i + len(sep):].split(" ", 1)[0].rstrip(",.!")
            if head in composer.VERB_LEX or any(head in v.get("match", [])
                                                for p in world.packs for v in p.get("verbs", [])):
                r1 = free_text(world, player, text[:i], picker)
                r2 = free_text(world, player, text[i + len(sep):], picker)
                return {"narrative": r1["narrative"] + r2["narrative"],
                        "violations": r1["violations"] + r2["violations"]}
            i = text.find(sep, i + 1)
    # ADDRESSING someone outranks every mechanical reading — "I ask Pip if
    # they'd like to help with the harvest" is a question, not a harvest, and
    # "I taunt/reason with the scuttler" is dialogue, not a swing (playtest:
    # reason/taunt fell through to attack).
    words0 = set(re.findall(r"[a-z']+", text.lower()))
    SPEAK = {"tell", "ask", "say", "speak", "greet", "chat", "talk", "beg", "warn",
             "thank", "reassure", "plead", "taunt", "reason", "mock", "bargain",
             "insult", "provoke", "persuade", "convince", "reassure", "apologize",
             "apologise", "beckon", "hail", "whisper", "shout"}

    def _named(name):
        return any(w in words0 for w in name.lower().split() if len(w) > 2)
    if words0 & SPEAK:
        for nid, name in npcs_in_reach(world, player):
            if _named(name):
                return say(world, nid, text)
        # named someone who exists but is out of earshot → say so, don't act
        far = [e for e in world.entities.values()
               if "person" in e.tags and "dead" not in e.tags and _named(e.name)]
        if far:
            return _refused(f"{far[0].name} is too far off to hear you.")
    # world-gated vocabulary: the ACTIVE packs' verbs get first claim on free text
    # ("put qi in the ground" is an action here, nonsense in a world with no qi)
    pv = reactions.pack_verb(world, player, text)
    if pv is not None and pv[0] == "fail":
        return _refused(pv[1])
    if pv is not None:
        _, effs, cost, again = pv
        for ef in effs:
            ef.actor = player.id
        if not changes_world(world, effs):
            # continuing a STATE (meditating on) isn't a refusal — time passes
            start = len(world.log)
            _advance(world, cost)
            return {"narrative": [again] + [e.story() for e in world.log[start:] if e.cause],
                    "violations": invariants(world)}
        start = len(world.log)
        fx.apply_all(world, effs)
        trace.bless(any(e.kind != "move" for e in effs))
        _advance(world, cost)
        return {"narrative": [e.story() for e in world.log[start:] if e.cause],
                "violations": invariants(world)}
    if re.search(r"\b(rest|nap|doze|breather)\b", text, re.I) or \
            re.search(r"\btake (a |another )?(deep |long |slow )?(breath|stock|moment)\b", text, re.I):
        start = len(world.log)                 # resting IS waiting — the world moves on
        _advance(world, wait_until=True)
        return {"narrative": ["You take a moment — unhurried, eyes half-open."]
                + [e.story() for e in world.log[start:] if e.cause],
                "violations": invariants(world)}
    if re.search(r"\b(stats?|status|health|hp|condition)\b", text, re.I):
        n = sum(1 for t in player.tags if t.startswith("inv:"))
        marks = sorted([t for t in player.tags if t.startswith("soaked_")]
                       + (["on fire!"] if "on_fire" in player.tags else []))
        nums = [f"{k} {v:g}" for k, v in sorted(player.props.items(),
                                                key=lambda kv: (kv[0] != "hp", kv[0]))
                if isinstance(v, (int, float))]
        return {"narrative": ["; ".join(nums) + f"; carrying {n} item(s)"
                              + (f"; {', '.join(marks)}" if marks else "") + "."],
                "violations": []}
    if re.search(r"\b(inventory|pockets?|belongings|carrying)\b", text, re.I):
        inv = sorted((world.entities[i].name if i in world.entities else i)
                     for i in (t[4:] for t in player.tags if t.startswith("inv:")))
        return {"narrative": [("You carry: " + ", ".join(inv) + ".") if inv else "You carry nothing."],
                "violations": []}
    # the concept gate: a word that is SOMEONE'S magic, in a world that lacks it
    alien = ((reactions.known_concepts() - reactions.world_concepts(world))
             & set(re.findall(r"[a-z']+", text.lower())))
    if alien:
        return _refused(f"You reach for {sorted(alien)[0]}, but this world holds no such thing. "
                        "You strain, visibly, at nothing; it is not a good look.")
    steps = composer.compose(world, player, text, picker)
    if steps and steps[0][0] in ("refuse", "clarify"):
        why = steps[0][1] if len(steps[0]) > 1 else "The world offers no way to do that."
        return _refused(why)
    if steps and steps[0][0] == "answer":      # a located thing, spoken as a bearing
        return {"narrative": [steps[0][1]], "violations": []}
    if steps and steps[0][0] == "oracle":
        from . import checks
        checks.oracle(world, steps[0][1], "even")
        return {"narrative": [world.log[-1].cause], "violations": []}
    if steps and steps[0][0] == "inspect":
        tgt = steps[0][1]
        if isinstance(tgt, tuple):
            return {"narrative": [describe(world, tgt)], "violations": []}
        e = world.entities[tgt]
        mood = next((t for t in ("afraid", "hostile", "compliant", "calm") if t in e.tags), "")
        return {"narrative": [f"{e.name} — {_hurt(e)}" + (f", {mood}" if mood else "") + "."],
                "violations": []}
    if steps and steps[0][0] == "social":
        return say(world, steps[0][1], text)
    start = len(world.log)
    effects, cost = composer.to_effects(world, player, steps)
    if not changes_world(world, effects):   # a no-op is a refusal, not a wasted turn
        return _refused("That would change nothing — the world is already as you'd leave it.")
    fx.apply_all(world, effects)
    # bless only if the world changed beyond position — a bare walk teaches nothing
    trace.bless(any(e.kind != "move" for e in effects))    # keystone: not "committed"
    _advance(world, cost)
    return {"narrative": [e.story() for e in world.log[start:] if e.cause],
            "violations": invariants(world)}


DEED_PHRASE = {"theft": "take the {item}", "arson": "start a fire", "assault": "attack {item}"}


def ask_about(world: World, npc_id: str, topic_id: str) -> dict:
    """DF-style topical conversation: ask what an NPC knows about someone.
    Firsthand = testimony; secondhand = hearsay; nothing = says so. Costs a tick."""
    start = len(world.log)
    npc = world.entities[npc_id]
    topic = world.entities.get(topic_id)
    tname = "you" if topic and "player" in topic.tags else (topic.name if topic else topic_id)
    lines = []
    for m in npc.mind.get("memory", []):
        if m["actor"] != topic_id and m.get("item") != topic_id:
            continue
        what = DEED_PHRASE.get(m["deed"], m["deed"]).format(
            item=world.entities[m["item"]].name if m.get("item") in world.entities else "someone")
        actor = "you" if m["actor"] == "player" else world.entities[m["actor"]].name \
            if m["actor"] in world.entities else m["actor"]
        if m["secondhand"]:
            lines.append(f"I heard {actor} dared to {what}. That's the word going round.")
        else:
            lines.append(f"I saw {actor} {what} with my own eyes.")
    if not lines:
        lines = [f"{tname}? Couldn't tell you a thing."]
    for ln in lines:
        eff = fx.speech(npc_id, ln, cause=f'{npc.name}: "{ln}"')
        eff.actor = npc_id
        fx.apply(world, eff)
    _advance(world, 1)
    return {"narrative": [e.story() for e in world.log[start:] if e.cause],
            "violations": invariants(world)}


def say(world: World, npc_id: str, text: str, tone: str | None = None) -> dict:
    """A free-text social turn: the player says `text` (in an optional declared
    tone — how you say it is yours to choose) to an npc; costs one tick."""
    from . import trace
    trace.begin(social.DATASTORE)          # (idempotent if free_text already opened the turn)
    start = len(world.log)
    social.address(world, "player", npc_id, utterance=text, tone=tone)
    trace.bless()            # an addressed NPC always replies — a committed social outcome
    _advance(world, 1)
    return {"narrative": [e.story() for e in world.log[start:] if e.cause],
            "violations": invariants(world)}


def is_over(world: World) -> bool:
    """The game ends when the player is dead. A dead player takes no more turns."""
    p = world.entities.get("player")
    return p is None or "dead" in p.tags


def step(world: World, opt: Option, actor=None) -> dict:
    # Any-actor execution (the other half of the D8 seam): the actor whose pick this is —
    # defaults to the player, so every existing caller is unchanged. Effects are attributed
    # to the actor, checked verbs roll against the actor, a social pick speaks AS the actor.
    actor = actor or world.entities["player"]
    from . import trace
    trace.begin(social.DATASTORE)          # menu picks harvest too (social options carry a scored reply)
    if not (opt.effects or opt.steps or opt.social or opt.wait) and opt.cost == 0:
        # the LOOK option narrates, same as free-text look
        return {"narrative": [describe(world, actor.pos)], "violations": invariants(world)}
    start = len(world.log)
    if opt.social:
        # opt.social = (npc_id, words). Words are the pick-time parameter — pass them as
        # the UTTERANCE (the 4th positional is a stance intent; an unknown one returns
        # silently, which ate the reply). No words = a plain approach; the npc still reacts.
        social.address(world, actor.id, opt.social[0], utterance=opt.social[1] or "")
        trace.bless()
    elif opt.steps:                      # checked verbs: the dice roll now, not at menu build
        from . import composer
        effects, _ = composer.to_effects(world, actor, opt.steps)
        fx.apply_all(world, effects)
    else:
        for ef in opt.effects:
            ef.actor = actor.id
        fx.apply_all(world, opt.effects)
    if opt.wait:
        _advance(world, wait_until=True)
    elif opt.cost > 0:
        _advance(world, opt.cost)
    return {"narrative": [e.story() for e in world.log[start:] if e.cause],
            "violations": invariants(world)}


def invariants(world: World) -> list[str]:
    bad = []
    for e in world.entities.values():
        if not world.in_bounds(e.pos):
            bad.append(f"{e.id} at out-of-bounds {e.pos}")
        elif is_solid(world.cell(e.pos).material):
            bad.append(f"{e.id} is inside a solid {world.cell(e.pos).material} block at {e.pos}")
        if "alive" in e.tags and "dead" in e.tags:
            bad.append(f"{e.id} is both alive and dead")
        if "dead" in e.tags and e.props.get("hp", 0.0) > 0:
            bad.append(f"{e.id} is dead but hp={e.props['hp']}")
        for kk, vv in e.props.items():
            if vv != vv or abs(vv) == float("inf"):
                bad.append(f"{e.id}.{kk} non-finite ({vv})")
    for p, c in world.cells.items():
        for m, ml in c.fluids.items():
            if ml < -1e-6:
                bad.append(f"cell {p} negative {m} ({ml})")
        if c.heat != c.heat or not (-100.0 <= c.heat <= 10000.0):
            bad.append(f"cell {p} heat out of range ({c.heat})")
    return bad


def _hurt(e):
    if "dead" in e.tags:
        return "dead"
    f = e.props.get("hp", 10.0) / 10.0
    return "unhurt" if f > 0.8 else "roughed up" if f > 0.4 else "badly hurt"


def describe(world, pos) -> str:
    """The LOOK renderer: the same facts cell_info dumps, spoken as a scene.
    cell_info stays the debugger panel; this is what the fiction shows (playtest:
    'I look for water' FOUND it, then narrated '(1,4) 60ml water; you')."""
    c, bits = world.cell(pos), []
    if "on_fire" in c.tags:
        bits.append("fire crawls over everything here")
    elif c.heat > 60:
        bits.append("the air shimmers with heat")
    for m, ml in sorted(c.fluids.items()):
        size = "a thin film" if ml < 120 else "a puddle" if ml < 400 else "a spreading pool"
        bits.append(f"{size} of {m} wets the floor")
    if c.material and is_solid(c.material):
        bits.append(f"rough {c.material} blocks the way")
    if "stairs" in c.tags:
        bits.append("a stairway leads down into the dark")
    if "ripe" in c.tags:
        bits.append("rice stands tall and golden, ready for the sickle")
    elif "sown" in c.tags:
        bits.append("seedlings stand in neat rows")
    elif "tilled" in c.tags:
        bits.append("the earth lies broken into dark, waiting rows")
    if "spirit" in c.tags:
        bits.append("the ground hums faintly, alive under your feet")
    for e in world.entities_at(pos):
        if "taken" in e.tags or "player" in e.tags:
            continue
        if "person" in e.tags:
            mood = next((t for t in ("afraid", "hostile", "compliant") if t in e.tags), "")
            bits.append(f"{e.name} is here — {_hurt(e)}" + (f" and {mood}" if mood else ""))
        elif "item" in e.tags:
            bits.append(f"a {e.name} lies within reach" if "container" not in e.tags
                        else f"a {e.name} sits here")
    if not bits:
        return "Bare floor, still air — nothing here worth a second glance."
    return ". ".join(b[0].upper() + b[1:] for b in bits) + "."


# --- observation: player inspect panel == the debugger ----------------------
def _glyph(world, p):
    ents = world.entities_at(p)
    people = [e for e in ents if e.tags & {"person", "player", "animal"}]
    if people:
        e = people[0]
        if "player" in e.tags: return "@"
        if "dead" in e.tags:   return "x"
        if "on_fire" in e.tags:return "E"
        return "e"          # any living creature — person or beast (hens were invisible)
    if any("prize" in e.tags and "taken" not in e.tags for e in ents):
        return "!"
    c = world.cell(p)
    if "stairs" in c.tags:  return ">"
    if "on_fire" in c.tags: return "*"
    if any("container" in e.tags and "broken" not in e.tags for e in ents):
        return "O"
    if c.fluids.get("acid"): return "a"
    if c.fluids.get("oil"):  return "o"
    if c.fluids.get("water"):return "~"
    if "burned" in c.tags:   return ","      # ash where timber burned away
    if c.material in BLOCK_GLYPH and c.props.get("hp", 1.0) <= 0.0:
        return ","                            # a smashed-through structure is rubble, not a wall
    return BLOCK_GLYPH.get(c.material, ".")


def render_grid(world: World, z: int = 0) -> list:
    """One z-slice as [{ch, dim}] rows; open air shows the level below, dimmed."""
    w, h, zmax = world.dims
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            p = (x, y, z)
            ch = _glyph(world, p)
            c = world.cell(p)
            if "stairs" in c.tags and ch == ".":
                ch = ">"
            dim = False
            if (z > 0 and ch == "." and not c.floor and not world.entities_at(p)
                    and not spatial.supported(world, p)):
                ch = _glyph(world, (x, y, z - 1))       # open air: see the storey below
                dim = True
            row.append({"ch": ch, "dim": dim})
        out.append(row)
    return out


def cell_info(world: World, pos) -> str:
    """One-line description of a cell — for the spatial UI's inspect-on-click."""
    c = world.cell(pos)
    bits = []
    if abs(c.heat - 20.0) > 1:
        bits.append(f"{c.heat:.0f}°C")
    if "on_fire" in c.tags:
        bits.append("on fire")
    bits += [f"{ml:.0f}ml {m}" for m, ml in c.fluids.items()]
    if c.material:
        bits.append(c.material + (f"(hp{c.props['hp']:.0f})" if "hp" in c.props else ""))
    bits += [t for t in sorted(c.tags) if t != "on_fire"]
    # taken items are inside a container or a pocket — not on view
    for e in (e for e in world.entities_at(pos) if "taken" not in e.tags or "player" in e.tags):
        marks = [f"soaked in {t[7:]}" for t in e.tags if t.startswith("soaked_")]
        if "player" in e.tags:
            bits.append("you" + (f" ({', '.join(marks)})" if marks else ""))
        else:
            emo = {t for t in e.tags if t in ("calm", "afraid", "hostile", "compliant")}
            status = sorted((e.tags & {"alive", "dead"}) | emo) + marks
            bits.append(f"{e.name} ({', '.join(status) or 'here'})")
    return f"({pos[0]},{pos[1]}) " + ("; ".join(bits) if bits else "empty floor")


def render(world: World, player) -> str:
    w, h = world.dims[0], world.dims[1]
    px, py, pz = player.pos
    L = [f"── tick {world.tick} ──  you: {player.pos}  hp={player.props.get('hp','?')}"]
    inv = sorted(t[4:] for t in player.tags if t.startswith("inv:"))
    L.append("carrying: " + (", ".join(inv) if inv else "nothing"))
    L.append("map (@you e·npc E·burning x·dead *·fire O·barrel !·prize >·stairs o·oil a·acid ~·water #·wood ,·rubble):")
    for y in range(h):                          # the PLAYER'S storey, not always z=0
        L.append("  " + " ".join(_glyph(world, (x, y, pz)) for x in range(w)))
    for name, tp in [("here", player.pos)] + [(n, (px + dx, py + dy, pz)) for n, (dx, dy) in DIRS.items()]:
        if not world.in_bounds(tp):
            continue
        c = world.cell(tp)
        bits = []
        if abs(c.heat - 20.0) > 1: bits.append(f"{c.heat:.0f}°C")
        bits += [f"{ml:.0f}ml {m}" for m, ml in c.fluids.items()]
        if c.material: bits.append(f"{c.material}" + (f"(hp{c.props['hp']:.0f})" if "hp" in c.props else ""))
        bits += sorted(c.tags)
        for e in world.entities_at(tp):
            if "player" not in e.tags and "taken" not in e.tags:
                rel = world.edges.get((e.id, "player"), {})
                rels = " ".join(f"{k}={v:+.1f}" for k, v in rel.items() if abs(v) > 0.05)
                shown = sorted(t for t in e.tags
                               if not t.startswith("inv:")            # inventory is internal
                               and not ("dead" in e.tags and t in social.EMOTIONS))  # corpses have no mood
                hp = e.props.get("hp")
                bits.append(f"[{e.name}: {','.join(shown)} hp={'' if hp is None else f'{hp:g}'}"
                            + (f"; toward you: {rels}" if rels else "") + "]")
        if bits:
            L.append(f"  {name} {tp}: " + "; ".join(bits))
    if world.log:
        L.append("recently:")
        L += ["    " + e.story() for e in world.log[-5:] if e.cause]
    return "\n".join(L)
