"""NPC minds (v1) — the two-speed brain.

FAST tier (every turn, deterministic, seeded — no LM):
  - a decaying SALIENCE meter fed by events → three bands (calm / agitated / engaged)
  - calm: pursue a GOAL via deterministic search over affordances (walk to its target)
  - agitated + threat: a trait-weighted, dice-resolved reaction (flee vs hold)

SLOW tier (occasional, budget-capped — the LM slot):
  - re-pick the GOAL on goal-break, and (future) propose novel affordances when search
    is stuck. Stubbed here to a deterministic trait scorer so v1 runs with no model and
    stays fuzz-reproducible. `social.address` remains the LM path for player-addressed NPCs.

Traits are static personality (bravery/warmth/self_interest/aggression); salience is
dynamic state. Every action returned is a bounded, logged delta through the one bus.
"""

from __future__ import annotations

import math as _math
import random as _rnd

from . import effects as fx
from .spatial import adjacent as _adjacent
from .spatial import l1 as _l1
from .spatial import visible as _visible
from .state import World, is_solid

CALM, ENGAGED = 20.0, 60.0          # salience band edges
DEFAULT_TRAITS = {"bravery": 0.5, "warmth": 0.5, "self_interest": 0.5, "aggression": 0.5}

# goal id -> (trait-scorer, target locator name). Goals fall out of traits, not authoring.
GOALS = {
    "acquire_prize": (lambda t: 1.2 * t["self_interest"] + 0.5 * (1 - t["warmth"]), "prize"),
    "guard_prize":   (lambda t: 1.0 * (1 - t["self_interest"]) + 0.6 * t["bravery"], "vault"),
    "work":          (lambda t: 0.6 * t["warmth"] + 0.3, "workbench"),
}


def _traits(npc):
    return {**DEFAULT_TRAITS, **npc.mind.get("traits", {})}


def _sal(npc):
    return npc.mind.get("salience", 0.0)


def band(npc):
    s = _sal(npc)
    return "calm" if s < CALM else "engaged" if s >= ENGAGED else "agitated"


def _tag_cell(world, tag):
    for pos, c in world.cells.items():
        if tag in c.tags:
            return pos
    return None


def _target(world, name):
    if name == "prize":
        for e in world.entities.values():
            if "prize" in e.tags and "taken" not in e.tags:
                return e.pos
        return None
    return _tag_cell(world, name)


def _passable(world, pos):
    """Walkable AND stand-able: an NPC won't voluntarily step onto an unsupported
    ledge and plunge off the gallery (the playtest saw NPCs walking off edges).
    Stairs are supported, so descending them is still fine."""
    from .spatial import supported
    return world.in_bounds(pos) and not is_solid(world.cell(pos).material) and supported(world, pos)


def _on_fire(world, pos):
    """The one honest fire query: the grid stores this fact as a tag. Everything
    that used to test `_danger >= 1000` asks this — no magic sentinel, no float
    standing in for a boolean the sim already knows exactly."""
    return "on_fire" in world.cell(pos).tags


def _danger(world, pos):
    """Real thermal load only — heat above ambient. Whether a cell is ALIGHT is a
    separate fact (`_on_fire`); danger no longer smuggles it in as a +1000 bump."""
    return max(0.0, world.cell(pos).heat - 20.0)


def _fire_near(world, npc):
    return _on_fire(world, npc.pos) or any(
        _on_fire(world, n) for n in world.neighbors(npc.pos))


# ── slow tier (LM slot) — deterministic stub: pick the goal traits favour ──
def choose_goal(world, npc):
    t, rng = _traits(npc), world.rng
    best, bs = None, -1e9
    for g, (score, tname) in GOALS.items():
        if _target(world, tname) is None:
            continue
        v = score(t) + rng.random() * 0.2      # seeded tie-break / personality noise
        if v > bs:
            best, bs = g, v
    npc.mind["goal"] = best


# ── fast tier: deterministic search toward the goal's target ──
def _step_toward(world, npc, tgt, why=""):
    """One pathing step — BFS over walkable space (stairs included), so upstairs
    NPCs can find their way down. `why` is the FACT the log states."""
    from .spatial import bfs_path, supported
    path = bfs_path(world, npc.pos, tgt)
    if path:
        best = path[0]
        if _on_fire(world, best):                         # not through fire
            return None
        if best[2] == npc.pos[2] and not supported(world, best):
            return None                                   # won't step off a ledge into a fall
    else:
        cands = [n for n in world.neighbors(npc.pos)
                 if n[2] == npc.pos[2] and _passable(world, n) and not _on_fire(world, n)]
        if not cands:
            return None
        best = min(cands, key=lambda n: _l1(n, tgt) + _danger(world, n) * 0.01)
    verb = "heads for" if _l1(best, tgt) < _l1(npc.pos, tgt) else "picks a way around toward"
    cause = f"{npc.name} {verb} {why}" if why else ""    # silent pathing between announcements
    return fx.move(npc.id, best, cause)


def _goal_why(world, goal):
    if goal == "acquire_prize":
        for e in world.entities.values():
            if "prize" in e.tags and "taken" not in e.tags:
                return f"the {e.name}"
    return {"guard_prize": "their post by the vault", "work": "the workbench"}.get(goal, "somewhere")


def _terminal(world, npc, goal):
    if goal == "acquire_prize":
        for e in world.entities.values():
            if "prize" in e.tags and "taken" not in e.tags and (e.pos == npc.pos or _adjacent(npc.pos, e.pos)):
                npc.mind["goal"] = None
                return [fx.set_tag(e.id, "taken", f"{npc.name} snatches the {e.name}!"),
                        fx.set_tag(npc.id, "has_prize")]
    return []      # guard_prize / work: hold station


# ── fast tier: dice-resolved threat reaction (flee vs hold, by traits) ──
def _dice_reaction(world, npc):
    t = _traits(npc)
    p_flee = max(0.0, min(1.0, 0.7 * (1 - t["bravery"]) + 0.3 * t["self_interest"] - 0.2 * t["aggression"]))
    if world.rng.random() < p_flee:
        opts = [n for n in world.neighbors(npc.pos) if _passable(world, n)]
        if opts:
            safe = min(opts, key=lambda n: _danger(world, n))
            if _danger(world, safe) < _danger(world, npc.pos):
                return [fx.move(npc.id, safe, f"{npc.name} bolts from the danger!")]
    # held ground — brave/aggressive stay put (NOT the default; it's earned by the roll)
    if "afraid" not in npc.tags and p_flee > 0.45:
        return [fx.clear_tag(npc.id, "calm"), fx.set_tag(npc.id, "afraid")]
    return []


SIGHT = 3          # line-of-sight radius; solid cells occlude (spatial.visible)
# base alarm per deed KIND, and the magnitude that EARNS that base (its reference
# size). A deed bigger than reference alarms more, smaller alarms less — the bump
# reads the real delta the log carries instead of flattening every deed to a flat
# constant. theft has no continuous magnitude, so its ref is 1 (the taken item).
DEEDS = {"theft": 40.0, "arson": 35.0, "assault": 45.0}
DEED_REF = {"theft": 1.0, "arson": 140.0, "assault": 4.0}
KINDLE_DELTA = 60.0        # smallest heat dump that can kindle anything (fluid ignition, per reactions.yaml)


def _classify_deed(world, entry):
    """A log entry that a bystander would recognise as a DEED, WITH the real
    magnitude it carries (heat dumped / hp lost). One mechanism: witnessing reads
    the delta log — new deed kinds are classifications, not systems. Arson is a
    spark, or a heat dump big enough to kindle SOMETHING (the lowest ignition
    point in the reaction rules — flammable fluid at 60° over ambient); the size
    only sets how loudly it registers, it no longer decides arson-or-not by a
    round magic number."""
    if entry.kind == "set_tag" and entry.data.get("tag") == "taken":
        actor = world.entities.get(entry.actor)
        return {"deed": "theft", "actor": entry.actor, "item": entry.data.get("t"),
                "pos": actor.pos if actor else None, "tick": entry.tick, "mag": 1.0}
    if entry.kind == "spark" or (entry.kind == "add_heat"
                                 and entry.data.get("delta", 0) >= KINDLE_DELTA):
        mag = 140.0 if entry.kind == "spark" else entry.data.get("delta", 0.0)
        return {"deed": "arson", "actor": entry.actor, "item": None,
                "pos": entry.data.get("pos"), "tick": entry.tick, "mag": mag}
    if (entry.kind == "adjust_prop" and entry.data.get("prop") == "hp"
            and entry.data.get("delta", 0) < 0):
        victim = world.entities.get(entry.data.get("t"))
        if victim is not None and "person" in victim.tags and entry.actor != entry.data.get("t"):
            return {"deed": "assault", "actor": entry.actor, "item": victim.id,
                    "pos": victim.pos, "tick": entry.tick, "mag": -entry.data.get("delta", 0.0)}
    return None


def _deed_alarm(deed):
    """Base alarm for the KIND, scaled by how the real magnitude compares to the
    deed's reference size — bounded so one big deed can't dwarf the ordering."""
    kind = deed["deed"]
    ratio = deed.get("mag", DEED_REF[kind]) / DEED_REF[kind]
    return DEEDS[kind] * max(0.5, min(1.8, ratio))


NOISE_FLOOR = 4.0          # hp-delta below which a hit makes no reportable sound


def _classify_noise(world, entry):
    """Loud log entries carry as SOUND: (description, radius). Sound crosses walls
    (muffled — half range without line of sight) and names no names.

    Loudness READS the real delta the log carries, it doesn't threshold-and-flatten:
    the FLOOR just decides audible-at-all; past it, the actual magnitude sets both
    how far the sound travels and how big it reads. A cracked stool and a shattered
    oak table are no longer the same canned crash — the sim knew they differed."""
    if entry.kind == "adjust_prop" and entry.data.get("prop") == "hp":
        mag = -entry.data.get("delta", 0.0)              # damage is negative hp
        if mag >= NOISE_FLOOR:
            t = entry.data.get("t")
            e = world.entities.get(t) if isinstance(t, str) else None
            if e is not None and "person" not in e.tags:
                radius = min(12, 5 + int(mag / 4))       # bigger break, longer reach
                sound = "a splintering crash" if mag < 10 else "a thunderous smash"
                return sound, radius, e.pos
    if entry.kind == "set_tag" and entry.data.get("tag") == "broken":
        e = world.entities.get(entry.data.get("t"))
        if e is not None:
            return "something bursting apart", 8, e.pos
    return None


def _hear(world, npc, entry):
    noise = _classify_noise(world, entry)
    if not noise or entry.actor == npc.id:
        return False
    sound, radius, pos = noise
    if not _visible(world, npc.pos, pos, radius):
        radius //= 2                                     # walls muffle, they don't silence
    if _l1(npc.pos, pos) > radius:
        return False
    mem = npc.mind.setdefault("memory", [])
    if any(m["deed"] == "noise" and m["tick"] == entry.tick and m["pos"] == pos for m in mem):
        return True                                      # one bang, one memory
    mem.append({"deed": "noise", "sound": sound, "actor": None, "item": None,
                "pos": pos, "tick": entry.tick, "secondhand": False})
    npc.mind["salience"] = min(100.0, _sal(npc) + 20.0)
    return True


def _perceive_deeds(world, npc):
    start = npc.mind.get("seen_upto", 0)
    for entry in world.log[start:]:
        deed = _classify_deed(world, entry) if entry.actor and entry.actor != npc.id else None
        if deed and deed["pos"] and _visible(world, npc.pos, deed["pos"], SIGHT):
            npc.mind.setdefault("memory", []).append({**deed, "secondhand": False})
            npc.mind["salience"] = min(100.0, _sal(npc) + _deed_alarm(deed))
        else:
            _hear(world, npc, entry)                     # not seen — maybe heard
    npc.mind["seen_upto"] = len(world.log)


def _perceive(world, npc):
    s = 0.0
    if _on_fire(world, npc.pos):
        s += 50
    elif _fire_near(world, npc):
        s += 30
    for n in [npc.pos] + world.neighbors(npc.pos):
        if any("player" in e.tags for e in world.entities_at(n)):
            s += 12
            break
    npc.mind["salience"] = min(100.0, _sal(npc) + s)
    _perceive_deeds(world, npc)
    _gossip(world, npc)


def _same_deed(a, b):
    return (a["deed"], a["actor"], a.get("item"), a["tick"]) == (b["deed"], b["actor"], b.get("item"), b["tick"])


def _gossip(world, npc):
    """Sharing is free: a witness tells anyone standing next to them. The copy is
    marked secondhand (hearsay < sight, for later systems) and sours the listener
    on the culprit — one witness becomes a reputation."""
    for m in npc.mind.get("memory", []):
        if m.get("actor") is None:
            continue                    # you can't gossip a name you never learned
        told = m.setdefault("told", set())
        for p in world.neighbors(npc.pos):
            for other in world.entities_at(p):
                if ("person" not in other.tags or "dead" in other.tags or other.id in told
                        or other.id == m["actor"]):     # you don't tattle to the culprit
                    continue
                told.add(other.id)
                if any(_same_deed(m, om) for om in other.mind.get("memory", [])):
                    continue
                other.mind.setdefault("memory", []).append(
                    {**{k: v for k, v in m.items() if k != "told"}, "secondhand": True, "told": set()})
                cur = world.edges.get((other.id, m["actor"]), {}).get("disposition", 0.0)
                eff = fx.set_edge(other.id, m["actor"], "disposition", cur - 0.3,
                                  f"{npc.name} murmurs to {other.name} about {world.entities[m['actor']].name}")
                eff.actor = npc.id
                fx.apply(world, eff)


def _aggress(world, npc, tgt, mem, kind, adj):
    """Go AT an adversary: chase if far; adjacent, either STRIKE (one who hit me / a
    hostile face) or CONFRONT + SEIZE (a culprit I witnessed). Returns
    (effects, memory-to-resolve-when-picked-or-None)."""
    if not adj:
        step = _step_toward(world, npc, tgt.pos)
        if step is None:
            return None, None
        tag = f"chase:{tgt.id}"
        step.cause = f"{npc.name} storms after {tgt.name}" if npc.mind.get("narrated") != tag else ""
        npc.mind["narrated"] = tag
        return [step], None                                    # keep chasing (don't resolve yet)
    who = "you" if "player" in tgt.tags else tgt.name
    if kind == "crime":                                        # police it: accuse, seize stolen goods
        v = world.entities.get(mem.get("item"))
        line = (f"Stop! I saw you take the {v.name}!" if mem["deed"] == "theft" and v else
                f"Stop! I saw you attack {'me' if v is npc else v.name}!" if mem["deed"] == "assault" and v else
                "Stop! I saw you set that fire!" if mem["deed"] == "arson" else "Stop right there!")
        out = [fx.set_edge(npc.id, tgt.id, "hostility", 0.8, f'{npc.name}: "{line}"'),
               fx.clear_tag(npc.id, "calm"), fx.set_tag(npc.id, "hostile")]
        item = mem.get("item")
        if item and (f"inv:{item}" in tgt.tags or "has_prize" in tgt.tags):
            nm = world.entities[item].name
            whose = "your" if "player" in tgt.tags else f"{tgt.name}'s"
            out += [fx.clear_tag(tgt.id, f"inv:{item}"), fx.clear_tag(tgt.id, "has_prize"),
                    fx.set_tag(npc.id, f"inv:{item}", f"{npc.name} wrenches the {nm} from {whose} hands!")]
            tgt.mind["salience"], tgt.mind["goal"] = 90.0, None
        return out, mem
    verb = "strikes back at" if kind == "hit" else "lashes out at"
    return [fx.set_edge(npc.id, tgt.id, "hostility", 1.0), fx.clear_tag(npc.id, "calm"),
            fx.set_tag(npc.id, "hostile"),
            fx.adjust_prop(tgt.id, "hp", -2.0, f"{npc.name} {verb} {who}!")], mem


def _investigate(world, npc):
    """A noise demands an answer: the brave walk toward it, the timid flinch."""
    noises = [m for m in npc.mind.get("memory", [])
              if m["deed"] == "noise" and not m.get("resolved")]
    if not noises:
        return None
    m = noises[-1]
    if world.tick - m["tick"] > 12:
        m["resolved"] = True
        return None
    t = _traits(npc)
    if t["bravery"] < 0.6:
        m["resolved"] = True
        if "afraid" not in npc.tags:
            return [fx.clear_tag(npc.id, "calm"),
                    fx.set_tag(npc.id, "afraid", f"{npc.name} flinches at the noise")]
        return []
    if _l1(npc.pos, m["pos"]) <= 1:
        m["resolved"] = True
        return [fx.speech(npc.id, "What happened here?",
                          cause=f"{npc.name} looks over the damage: \"What happened here?\"")]
    step = _step_toward(world, npc, m["pos"])
    if step:
        if npc.mind.get("narrated") != f"noise:{m['tick']}":
            step.cause = f"{npc.name} goes to see about the noise"
            npc.mind["narrated"] = f"noise:{m['tick']}"
        return [step]
    m["resolved"] = True
    return []


# ── ONE drive-utility selector for every reactive choice toward an adversary: fight
# back, police a crime (confront + seize), flee, yield, snatch, lash out. It replaced
# the hand-authored fight-or-flight / confront / social handlers — behaviour EMERGES
# from drives × this NPC's BELIEFS × the live affordances, softmax-picked (reproducible,
# not a hard argmax). Emotion tags a reply set feed the drives; they don't trigger
# actions. Adversaries: whoever HIT me (wrath/fear), whose crime I WITNESSED (duty),
# or the near PLAYER (fear/greed).
def _react(world, npc):
    t, memo = _traits(npc), npc.mind.get("memory", [])
    def _live(a): return a in world.entities and "dead" not in world.entities[a].tags
    concerns = []
    for m in memo:
        if m.get("resolved"):
            continue
        if m["deed"] == "assault" and m.get("item") == npc.id and _live(m["actor"]):
            concerns.append((world.entities[m["actor"]], m, "hit"))          # struck ME
        elif not m.get("secondhand") and m["deed"] in ("theft", "assault", "arson") \
                and m["actor"] != npc.id and _live(m["actor"]):
            concerns.append((world.entities[m["actor"]], m, "crime"))        # I witnessed it
    p = world.entities.get("player")
    if p is not None and "dead" not in p.tags and _l1(npc.pos, p.pos) <= 2:
        concerns.append((p, None, "near"))
    if not concerns:
        return None
    steady = 1.2 - t["bravery"]
    duty = (0.7 - t["self_interest"]) if (t["bravery"] > 0.6 and t["self_interest"] <= 0.4) else 0.0
    cands = [(0.25, None, None)]                                # HOLD → fall through to job/goal
    for tgt, m, kind in concerns:
        adj = _adjacent(npc.pos, tgt.pos)
        e = world.edges.get((npc.id, "player"), {}) if "player" in tgt.tags else {}
        fear = steady * (e.get("fear", 0.0) + 0.6 * ("afraid" in npc.tags)
                         + 0.6 * ("compliant" in npc.tags) + 0.5 * _fire_near(world, npc))
        # wrath needs a REASON — being hit, or already hostile. The calm don't strike
        # unprovoked (aggression is how hard you hit WHEN roused, not a standing menace).
        wrath = (t["aggression"] if "hostile" in npc.tags else 0.0) + (1.0 if kind == "hit" else 0.0)
        drive = duty if kind == "crime" else wrath
        if drive > 0.05:                                       # FIGHT / POLICE (chase → strike/seize)
            eff, res = _aggress(world, npc, tgt, m, kind, adj)
            if eff:
                cands.append((drive if adj else drive * 0.9, eff, res))
        opts = [q for q in world.neighbors(npc.pos) if _passable(world, q)]
        away = max(opts, key=lambda q: _l1(q, tgt.pos)) if opts else None
        if away and _l1(away, tgt.pos) > _l1(npc.pos, tgt.pos):    # FLEE
            nm = "you" if "player" in tgt.tags else tgt.name
            cands.append((fear * (1.0 if "afraid" in npc.tags else 0.3),
                          [fx.move(npc.id, away, f"{npc.name} shrinks back from {nm}")],
                          m if (m and _l1(away, tgt.pos) > 2) else None))
        if kind == "near":
            held = next((g[4:] for g in npc.tags if g.startswith("inv:") and g[4:] in world.entities), None)
            if held and adj:                                   # YIELD (appease)
                it = world.entities[held]
                cands.append((fear * (1.0 if "compliant" in npc.tags else 0.3)
                              + 1.0 * ("compliant" in npc.tags) - 0.5 * t["self_interest"],
                              [fx.clear_tag(npc.id, f"inv:{held}"), fx.clear_tag(npc.id, "has_prize"),
                               fx.clear_tag(npc.id, "compliant"), fx.set_tag("player", f"inv:{held}"),
                               fx.set_tag(held, "taken"),
                               fx.move(held, tgt.pos, f"{npc.name} hands you the {it.name}, hands shaking")], None))
            ptok = next((g[4:] for g in tgt.tags if g.startswith("inv:") and g[4:] in world.entities
                         and "prize" in world.entities[g[4:]].tags), None)
            if ptok and adj and ("hostile" in npc.tags or band(npc) != "calm"):   # SNATCH (greed over fear)
                it = world.entities[ptok]
                cands.append((t["self_interest"] - fear,
                              [fx.clear_tag("player", f"inv:{ptok}"), fx.clear_tag(npc.id, "compliant"),
                               fx.set_tag(npc.id, f"inv:{ptok}"),
                               fx.move(ptok, npc.pos, f"{npc.name} snatches the {it.name} from you!")], None))
    return _softmax_pick(world, npc, cands)


def _softmax_pick(world, npc, cands, temp=0.25):
    """Sample a candidate ∝ exp(utility/temp), seeded per (world,tick,npc) — reproducible
    yet not a hard argmax. Own RNG, so it doesn't perturb the sim's dice stream. Each
    candidate is (utility, effects, memory-to-mark-resolved-if-picked)."""
    s = (world.seed * 1000003 + world.tick * 131 + sum(ord(c) for c in npc.id)) & 0x7fffffff
    top = max(c[0] for c in cands)
    ws = [_math.exp((c[0] - top) / temp) for c in cands]
    r, acc, chosen = _rnd.Random(s).random() * sum(ws), 0.0, cands[-1]
    for cand, w in zip(cands, ws):
        acc += w
        if r <= acc:
            chosen = cand
            break
    if chosen[2] is not None:
        chosen[2]["resolved"] = True
    return chosen[1]


def think(world, npc):
    npc.mind["salience"] = max(0.0, _sal(npc) * 0.75)      # decay
    _perceive(world, npc)

    # agitated/engaged + a real threat → dice reaction (fight-or-flight)
    if _fire_near(world, npc) and band(npc) != "calm":
        return _dice_reaction(world, npc)

    # ONE drive-utility selector answers every adversary — fight back / flee /
    # confront+seize / yield / snatch / lash out, softmax-picked from drives × beliefs
    react = _react(world, npc)
    if react is not None:
        return react

    # an unexplained noise gets looked into — or flinched at
    inv = _investigate(world, npc)
    if inv is not None:
        return inv

    # a JOB outranks idle agendas: staff make rounds, tend, sweep (acts.py)
    from . import acts
    job = acts.step_job(world, npc)
    if job is not None:
        return job

    # otherwise pursue the agenda (goal-break → re-pick the goal: the slow tier)
    goal = npc.mind.get("goal")
    tname = GOALS[goal][1] if goal in GOALS else None
    if goal is None or _target(world, tname) is None:
        choose_goal(world, npc)
        goal = npc.mind.get("goal")
        tname = GOALS[goal][1] if goal in GOALS else None
    if not goal:
        return []
    tgt = _target(world, tname)
    if tgt is None:
        return []
    if npc.pos == tgt or _adjacent(npc.pos, tgt):
        return _terminal(world, npc, goal)
    why = _goal_why(world, goal)
    if npc.mind.get("narrated") == why:      # announce a goal once; path silently after
        why = ""
    else:
        npc.mind["narrated"] = _goal_why(world, goal)
    eff = _step_toward(world, npc, tgt, why=why)
    return [eff] if eff else []


def take_turns(world: World) -> None:
    for e in list(world.entities.values()):
        if "person" not in e.tags or "dead" in e.tags:   # only people think; items don't
            continue
        effects = think(world, e)
        for ef in effects:
            ef.actor = e.id                              # every NPC delta carries provenance
        fx.apply_all(world, effects)
