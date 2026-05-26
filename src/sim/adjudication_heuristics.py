"""
Mechanical adjudication heuristics — map creative intent to transitions.

Runs after template matchers and before generic WorldFact fallback.
Routes common player inventions into existing simulation hooks (tags,
structures, fluids, alertness) instead of narration-only facts.
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import (
    AdjudicationResult,
    AlertnessLevel,
    EmotionalState,
    EntityId,
    GroundingResult,
    ScheduledEffect,
    ScheduledEffectKind,
    SemanticAction,
    TransitionKind,
    TransitionProposal,
    ValidationResult,
    WorldFact,
    WorldFactScope,
    WorldState,
)

# Intent pattern → handler name
_WEDGE_RE = re.compile(r"\b(wedge|jam|barricade|block|prop|brace)\b.{0,30}\b(door|gate|window|opening)\b", re.I)
_UNLOCK_RE = re.compile(r"\b(unlock|pick|force open|pry open|break open)\b.{0,20}\b(door|lock|gate|chest)\b", re.I)
_IGNITE_RE = re.compile(r"\b(ignite|light|set fire|torch|burn|kindle|set alight)\b", re.I)
_EXTINGUISH_RE = re.compile(r"\b(extinguish|douse|put out|smother)\b.{0,20}\b(fire|flame|torch|hearth)\b", re.I)
_SPILL_RE = re.compile(r"\b(spill|pour|tip|empty|slosh)\b.{0,30}\b(oil|ale|water|drink|liquid|flask|wine)\b", re.I)
_HIDE_RE = re.compile(r"\b(hide|conceal|duck behind|take cover|crouch behind)\b", re.I)
_DISTRACT_RE = re.compile(r"\b(distract|divert attention|create a diversion|decoy)\b", re.I)
_TRAP_RE = re.compile(r"\b(trap|snare|tripwire|rig)\b", re.I)
_CLIMB_RE = re.compile(r"\b(climb|scale|scramble up|clamber)\b.{0,20}\b(wall|roof|ladder|window|ledge)\b", re.I)
_THROW_RE = re.compile(r"\b(throw|hurl|fling|toss)\b.{0,40}\b(at|toward|into)\b", re.I)
_SMEAR_RE = re.compile(r"\b(smear|spread|rub|coat|apply)\b.{0,30}\b(oil|grease|poison|substance)\b", re.I)
_LISTEN_RE = re.compile(r"\b(eavesdrop|listen in|overhear|spy on)\b", re.I)
_BAIT_RE = re.compile(r"\b(bait|lure|entice|tempt)\b", re.I)
_DISARM_RE = re.compile(r"\b(disarm|knock weapon|strip weapon|grab weapon)\b", re.I)
_SABOTAGE_RE = re.compile(r"\b(sabotage|tamper|disable|sabotage|rig to fail|loosen)\b", re.I)
_IMPROVISE_WEAPON_RE = re.compile(
    r"\b(improvise|fashion|make)\b.{0,20}\b(weapon|club|spear|molotov|bomb)\b", re.I,
)
_BARricade_FURNITURE_RE = re.compile(
    r"\b(push|drag|move|stack)\b.{0,30}\b(chair|table|barrel|crate|furniture)\b.{0,30}\b(block|against|in front)\b",
    re.I,
)
_SMASH_RE = re.compile(
    r"\b(smash|bash|break|break through|bust through|kick down|break down|"
    r"batter|demolish|destroy|punch through|tear down|rip off)\b"
    r".{0,35}\b(wall|door|window|gate|fence|barrier|plank|board|panel)\b",
    re.I,
)
_SHOVE_RE = re.compile(
    r"\b(shove|push|knock|slam|body-check)\b.{0,25}\b(into|against|through|off)\b",
    re.I,
)


def _actor(world: WorldState, action: SemanticAction):
    from .region_utils import entity_or_none
    return entity_or_none(world, action.actor)


def _target_id(action: SemanticAction, world: WorldState) -> Optional[str]:
    if isinstance(action.target, str):
        tid = str(action.target)
        if EntityId(tid) in world.spatial.entities:
            return tid
    return None


def _nearest_wall_tile(world: WorldState, x: int, y: int) -> Optional[tuple[int, int]]:
    """Return (wx, wy) of a wall or closed-door tile adjacent to or at (x,y)."""
    from .schemas import TerrainType

    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        cx, cy = x + dx, y + dy
        from .schemas import Coord

        tile = world.spatial.tile_at(Coord(x=cx, y=cy))
        if tile.terrain in (TerrainType.WALL, TerrainType.DOOR_CLOSED):
            return (cx, cy)
    return None


def _tile_material_hardness(world: WorldState, tx: int, ty: int) -> tuple[str, int]:
    """Return (material_name, hardness) for tile at (tx, ty)."""
    from .schemas import Coord

    tile = world.spatial.tile_at(Coord(x=tx, y=ty))
    mat_name = tile.env.get("material", "stone")
    phys = world.config.physics_config
    if phys is not None:
        mat = phys.materials.get(mat_name)
        if mat is not None:
            return (mat_name, mat.hardness)
    # Fallback hardness by terrain type
    from .schemas import TerrainType
    hardness = 8 if tile.terrain == TerrainType.WALL else 4
    return (mat_name, hardness)


def _nearby_scene_features(world: WorldState, x: int, y: int) -> list[str]:
    """Collect visible physics-relevant features near (x,y) for creative hints."""
    from .schemas import Coord

    features: list[str] = []
    grid = world.spatial
    phys = world.config.physics_config

    for obj in grid.objects.values():
        if obj.position is None:
            continue
        dist = abs(obj.position.x - x) + abs(obj.position.y - y)
        if dist > 5:
            continue
        obj_tags = set(obj.tags)
        mat_name = obj.meta.get("material", "")
        if phys and mat_name:
            mat = phys.materials.get(mat_name)
            if mat:
                obj_tags.update(mat.tags)
        if "flammable" in obj_tags or "on_fire" in obj_tags:
            features.append(f"the {obj.name} nearby is flammable")
        if "fire_suppressant" in obj_tags:
            features.append(f"the {obj.name} could douse fire")
        if "weapon" in obj_tags:
            features.append(f"the {obj.name} could serve as a weapon")
        if "liquid" in obj_tags:
            features.append(f"the {obj.name} contains liquid")

    for mark in grid.tile_at(Coord(x=x, y=y)).marks[-2:]:
        features.append(f"there is '{mark}' here")

    return features[:4]


def _witness_proposals(
    world: WorldState,
    action: SemanticAction,
    *,
    alertness: str = AlertnessLevel.MEDIUM.value,
    max_w: int = 3,
) -> list[TransitionProposal]:
    from .region_utils import witness_entity_ids

    actor_id = str(action.actor)
    out: list[TransitionProposal] = []
    for wid in witness_entity_ids(world, action.actor)[:max_w]:
        if str(wid) == actor_id:
            continue
        out.append(
            TransitionProposal(
                kind=TransitionKind.ENTITY_ALERTNESS_CHANGED.value,
                payload={
                    "entity_id": str(wid),
                    "to": alertness,
                    "cause": "witnessed_heuristic_action",
                },
            )
        )
    return out


def try_adjudication_heuristics(
    world: WorldState,
    action: SemanticAction,
    intent: str,
    grounding: GroundingResult,
    result: ValidationResult,
) -> Optional[AdjudicationResult]:
    """
    Match intent against mechanical patterns; return AdjudicationResult or None.
    """
    text = (intent or action.raw_input or "").strip()
    if not text:
        return None
    lower = text.lower()
    actor = _actor(world, action)
    if actor is None:
        return None

    actor_id = str(action.actor)
    tick = world.tick
    x, y = actor.position.x, actor.position.y
    adj = AdjudicationResult()

    def _finish(ruling: str, *, tags: list[str] | None = None) -> AdjudicationResult:
        adj.ruling_text = ruling
        fact = WorldFact(
            claim=text[:160],
            scope=WorldFactScope.TILE,
            subject_id=f"{x},{y}",
            established_tick=tick,
            established_by=actor_id,
            source_intent=text[:200],
            tags=["heuristic", *(tags or [])],
        )
        adj.facts.append(fact)
        if not adj.transition_proposals:
            adj.transition_proposals.append(
                TransitionProposal(
                    kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                    payload={
                        "entity_id": actor_id,
                        "stat": "stamina",
                        "delta": -2.0,
                        "cause": "heuristic_effort",
                    },
                )
            )
        adj.transition_proposals.extend(_witness_proposals(world, action))
        return adj

    # ── Barricade / wedge door ───────────────────────────────────────────
    if _WEDGE_RE.search(text) or _BARricade_FURNITURE_RE.search(text):
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.STRUCTURE_CREATED.value,
                payload={
                    "x": x,
                    "y": y,
                    "name": "improvised barricade",
                    "tags": ["barrier", "improvised"],
                    "passable": False,
                },
            ),
            TransitionProposal(
                kind=TransitionKind.TILE_MARKED.value,
                payload={"x": x, "y": y, "mark": "barricaded"},
            ),
            TransitionProposal(
                kind=TransitionKind.NOISE_EVENT.value,
                payload={
                    "x": x,
                    "y": y,
                    "radius": 6,
                    "level": 55,
                    "cause": "barricade",
                },
            ),
        ])
        return _finish("You wedge something into place — the way is blocked.", tags=["barrier"])

    # ── Unlock / force door ────────────────────────────────────────────────
    if _UNLOCK_RE.search(text):
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.STRUCTURE_MODIFIED.value,
                payload={"x": x, "y": y, "set_passable": True, "add_tags": ["forced"]},
            ),
            TransitionProposal(
                kind=TransitionKind.NOISE_EVENT.value,
                payload={"x": x, "y": y, "radius": 8, "level": 70, "cause": "forced_entry"},
            ),
        ])
        adj.transition_proposals.extend(
            _witness_proposals(world, action, alertness=AlertnessLevel.HIGH.value)
        )
        return _finish("Something gives way with a sharp crack.", tags=["structure"])

    # ── Ignite ─────────────────────────────────────────────────────────────
    if _IGNITE_RE.search(text):
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.TILE_MARKED.value,
                payload={"x": x, "y": y, "mark": "fire"},
            ),
            TransitionProposal(
                kind=TransitionKind.ENVIRONMENT_STATE_CHANGED.value,
                payload={"key": "lit", "value": True, "cause": "ignite"},
            ),
            TransitionProposal(
                kind=TransitionKind.NOISE_EVENT.value,
                payload={"x": x, "y": y, "radius": 10, "level": 40, "cause": "fire"},
            ),
        ])
        adj.transition_proposals.extend(
            _witness_proposals(world, action, alertness=AlertnessLevel.HIGH.value)
        )
        return _finish("Flames catch and spread heat through the room.", tags=["fire"])

    # ── Extinguish ─────────────────────────────────────────────────────────
    if _EXTINGUISH_RE.search(text):
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.TILE_MARKED.value,
                payload={"x": x, "y": y, "mark": "soot", "remove": False},
            ),
            TransitionProposal(
                kind=TransitionKind.ENVIRONMENT_STATE_CHANGED.value,
                payload={"key": "lit", "value": False, "cause": "extinguish"},
            ),
        ])
        return _finish("Smoke curls up as the fire dies.", tags=["fire"])

    # ── Spill liquid ───────────────────────────────────────────────────────
    if _SPILL_RE.search(text) or _SMEAR_RE.search(text):
        liquid = "oil" if "oil" in lower else "liquid"
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.TILE_MARKED.value,
                payload={"x": x, "y": y, "mark": f"{liquid}_spill"},
            ),
            TransitionProposal(
                kind=TransitionKind.FLUID_CHANGED.value,
                payload={
                    "x": x,
                    "y": y,
                    "z": actor.position.z,
                    "fluid": liquid,
                    "volume_ml": 200.0,
                    "cause": "spill",
                },
            ),
        ])
        return _finish(f"A pool of {liquid} spreads underfoot.", tags=["fluid"])

    # ── Hide / take cover ──────────────────────────────────────────────────
    if _HIDE_RE.search(text):
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.ENTITY_TAG_CHANGED.value,
                payload={"entity_id": actor_id, "tag": "hidden", "add": True},
            )
        )
        return _finish("You melt into cover, harder to spot.", tags=["stealth"])

    # ── Distract ───────────────────────────────────────────────────────────
    if _DISTRACT_RE.search(text) or _BAIT_RE.search(text):
        tid = _target_id(action, world)
        proposals = [
            TransitionProposal(
                kind=TransitionKind.NOISE_EVENT.value,
                payload={"x": x, "y": y, "radius": 12, "level": 65, "cause": "distraction"},
            ),
        ]
        if tid:
            proposals.append(
                TransitionProposal(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED.value,
                    payload={
                        "entity_id": tid,
                        "to": AlertnessLevel.LOW.value,
                        "cause": "distracted",
                    },
                )
            )
        adj.transition_proposals.extend(proposals)
        adj.scheduled_effects.append(
            ScheduledEffect(
                fire_tick=tick + 2,
                created_tick=tick,
                source_actor=actor_id,
                kind=ScheduledEffectKind.NARRATION,
                narration="The distraction buys a moment — attention wavers elsewhere.",
                rationale="distraction ripple",
            )
        )
        return _finish("You create a diversion; eyes turn away.", tags=["social"])

    # ── Trap / sabotage ────────────────────────────────────────────────────
    if _TRAP_RE.search(text) or _SABOTAGE_RE.search(text):
        adj.transition_proposals.extend([
            TransitionProposal(
                kind=TransitionKind.TILE_MARKED.value,
                payload={"x": x, "y": y, "mark": "trap_set"},
            ),
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": actor_id,
                    "stat": "stamina",
                    "delta": -5.0,
                    "cause": "trap_setup",
                },
            ),
        ])
        adj.scheduled_effects.append(
            ScheduledEffect(
                fire_tick=tick + 4,
                created_tick=tick,
                source_actor=actor_id,
                kind=ScheduledEffectKind.NARRATION,
                narration="Something you rigged waits to spring.",
                rationale="trap delayed",
            )
        )
        return _finish("You set something nasty in place.", tags=["trap"])

    # ── Climb ──────────────────────────────────────────────────────────────
    if _CLIMB_RE.search(text):
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": actor_id,
                    "stat": "stamina",
                    "delta": -8.0,
                    "cause": "climb",
                },
            )
        )
        adj.transition_proposals.extend(_witness_proposals(world, action, max_w=2))
        return _finish("You haul yourself up, muscles burning.", tags=["movement"])

    # ── Throw at target ────────────────────────────────────────────────────
    if _THROW_RE.search(text):
        tid = _target_id(action, world)
        proposals = [
            TransitionProposal(
                kind=TransitionKind.NOISE_EVENT.value,
                payload={"x": x, "y": y, "radius": 8, "level": 75, "cause": "thrown_object"},
            ),
        ]
        if tid:
            proposals.extend([
                TransitionProposal(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED.value,
                    payload={
                        "entity_id": tid,
                        "to": AlertnessLevel.HIGH.value,
                        "cause": "thrown_at",
                    },
                ),
                TransitionProposal(
                    kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                    payload={
                        "entity_id": tid,
                        "to": EmotionalState.SUSPICIOUS.value,
                        "cause": "thrown_at",
                    },
                ),
            ])
        adj.transition_proposals.extend(proposals)
        return _finish("Something flies through the air.", tags=["violence"])

    # ── Eavesdrop ──────────────────────────────────────────────────────────
    if _LISTEN_RE.search(text):
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={
                    "entity_id": actor_id,
                    "stat": "stamina",
                    "delta": -1.0,
                    "cause": "eavesdrop",
                },
            )
        )
        adj.scheduled_effects.append(
            ScheduledEffect(
                fire_tick=tick + 1,
                created_tick=tick,
                source_actor=actor_id,
                kind=ScheduledEffectKind.NARRATION,
                narration="You catch a fragment of conversation meant for other ears.",
                rationale="eavesdrop yield",
            )
        )
        return _finish("You strain to listen without being noticed.", tags=["social"])

    # ── Disarm ─────────────────────────────────────────────────────────────
    if _DISARM_RE.search(text):
        tid = _target_id(action, world)
        if tid:
            adj.transition_proposals.extend([
                TransitionProposal(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED.value,
                    payload={
                        "entity_id": tid,
                        "to": AlertnessLevel.COMBAT.value,
                        "cause": "disarm_attempt",
                    },
                ),
                TransitionProposal(
                    kind=TransitionKind.EDGE_UPDATED.value,
                    payload={
                        "source": tid,
                        "target": actor_id,
                        "edge_kind": "threatened",
                        "delta": 0.2,
                    },
                ),
            ])
            return _finish("Steel clashes as you go for their weapon.", tags=["combat"])

    # ── Improvise weapon ───────────────────────────────────────────────────
    if _IMPROVISE_WEAPON_RE.search(text):
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.ITEM_CREATED.value,
                payload={
                    "name": "improvised weapon",
                    "tags": ["weapon", "improvised"],
                    "owner": actor_id,
                    "x": x,
                    "y": y,
                    "passable": True,
                },
            )
        )
        return _finish("You cobble together something dangerous.", tags=["craft"])

    # ── Smash / break through structural tile ──────────────────────────
    if _SMASH_RE.search(text):
        wall = _nearest_wall_tile(world, x, y)
        if wall is not None:
            wx, wy = wall
            mat_name, hardness = _tile_material_hardness(world, wx, wy)
            strength = actor.attributes.get("strength", 50)
            import hashlib as _hlib
            _seed = f"{actor_id}_{tick}_smash_{wx}_{wy}"
            _roll = int.from_bytes(_hlib.sha256(_seed.encode()).digest()[:4], "big") / (2 ** 32)
            threshold = min(0.95, max(0.05, (hardness * 10 - strength * 0.4) / 100 + 0.3))
            if _roll > threshold:
                adj.transition_proposals.extend([
                    TransitionProposal(
                        kind=TransitionKind.STRUCTURE_MODIFIED.value,
                        payload={
                            "x": wx, "y": wy,
                            "set_passable": True,
                            "add_tags": ["breached", "damaged"],
                            "cause": "smashed",
                        },
                    ),
                    TransitionProposal(
                        kind=TransitionKind.ENVIRONMENT_STATE_CHANGED.value,
                        payload={"x": wx, "y": wy, "key": "hp", "value": 0, "cause": "smash_destroy"},
                    ),
                    TransitionProposal(
                        kind=TransitionKind.NOISE_EVENT.value,
                        payload={"x": x, "y": y, "radius": 12, "level": 90, "cause": "smash"},
                    ),
                    TransitionProposal(
                        kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                        payload={"entity_id": actor_id, "stat": "stamina", "delta": -15.0, "cause": "smash_effort"},
                    ),
                ])
                adj.transition_proposals.extend(
                    _witness_proposals(world, action, alertness=AlertnessLevel.HIGH.value)
                )
                return _finish(
                    f"You slam into the {mat_name} with everything you have. "
                    f"It gives way with a tremendous crash.",
                    tags=["structure", "violence"],
                )
            else:
                features = _nearby_scene_features(world, x, y)
                hint = (f" (Notice: {features[0]}.)" if features else "")
                adj.transition_proposals.extend([
                    TransitionProposal(
                        kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                        payload={"entity_id": actor_id, "stat": "stamina", "delta": -8.0, "cause": "smash_fail"},
                    ),
                    TransitionProposal(
                        kind=TransitionKind.TILE_MARKED.value,
                        payload={"x": wx, "y": wy, "mark": "dented"},
                    ),
                    TransitionProposal(
                        kind=TransitionKind.NOISE_EVENT.value,
                        payload={"x": x, "y": y, "radius": 8, "level": 60, "cause": "smash_fail"},
                    ),
                ])
                return _finish(
                    f"You throw yourself at the {mat_name} — it shudders but holds. "
                    f"Solid {mat_name} (hardness {hardness}/10). "
                    f"You'd need more strength, a tool, or a different approach.{hint}",
                    tags=["structure"],
                )
        features = _nearby_scene_features(world, x, y)
        hint_parts = [f"nearby: {f}" for f in features[:2]]
        hint = (" — " + "; ".join(hint_parts)) if hint_parts else ""
        adj.transition_proposals.append(
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={"entity_id": actor_id, "stat": "stamina", "delta": -4.0, "cause": "smash_nothing"},
            )
        )
        return _finish(f"Nothing solid enough is right there to break through{hint}.", tags=["structure"])

    # ── Shove entity into environment ──────────────────────────────────
    if _SHOVE_RE.search(text):
        tid = _target_id(action, world)
        proposals = [
            TransitionProposal(
                kind=TransitionKind.NOISE_EVENT.value,
                payload={"x": x, "y": y, "radius": 6, "level": 55, "cause": "shove"},
            ),
            TransitionProposal(
                kind=TransitionKind.ENTITY_STAT_CHANGED.value,
                payload={"entity_id": actor_id, "stat": "stamina", "delta": -4.0, "cause": "shove"},
            ),
        ]
        features = _nearby_scene_features(world, x, y)
        ruling = "You shove hard."
        if tid:
            tgt_ent = world.spatial.entities.get(EntityId(tid))
            tgt_name = tgt_ent.name if tgt_ent else "them"
            proposals.extend([
                TransitionProposal(
                    kind=TransitionKind.ENTITY_ALERTNESS_CHANGED.value,
                    payload={"entity_id": tid, "to": AlertnessLevel.HIGH.value, "cause": "shoved"},
                ),
                TransitionProposal(
                    kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED.value,
                    payload={"entity_id": tid, "to": EmotionalState.ANGRY.value, "cause": "shoved"},
                ),
            ])
            if features:
                ruling = (
                    f"You slam {tgt_name} — they stumble. "
                    f"The {features[0]} is right there if you wanted to use it."
                )
            else:
                ruling = f"You shove {tgt_name} hard. They stagger back."
        adj.transition_proposals.extend(proposals)
        adj.transition_proposals.extend(_witness_proposals(world, action))
        return _finish(ruling, tags=["violence"])

    return None
