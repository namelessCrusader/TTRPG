"""
M3: Semantic projection layer.

This is the architectural firewall between canonical world state and the LM.

project(world_state, focal_entity_id) → SemanticProjection

Guarantees:
  1. No WorldState object crosses this boundary.  The LM adapter receives
     only SemanticProjection.
  2. Projection is lossy but semantically dense — raw coordinates are
     converted to topological/relational descriptors.
  3. Projection is deterministic: same (WorldState, EntityId) → same bytes.
  4. Projection respects partial observability: entities outside LOS are
     excluded.
  5. Projection size is bounded by MAX_PROJECTION_TOKENS.
"""

from __future__ import annotations

import math
from typing import Optional

from .relational import (
    _get_edge,
    get_entity_faction_ids,
    get_faction_tension,
    get_relationship_summary,
)
from .schemas import (
    ActionType,
    Affordance,
    BeliefRecord,
    Coord,
    coord_key,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityState,
    EntitySummary,
    EnvironmentSummary,
    EventSummary,
    FactionId,
    NodeKind,
    ObjectState,
    RelationalGraph,
    SemanticProjection,
    SocialContext,
    SpatialGrid,
    TerrainType,
    TransitionKind,
    WorldState,
)
from .spatial import entities_visible_from, visible_from

# Default token budget for a projection.  Adjust at runtime via project().
MAX_PROJECTION_TOKENS: int = 1200

# How many recent events to include in projection
RECENT_EVENT_WINDOW: int = 10


# ---------------------------------------------------------------------------
# Token estimation (cheap approximation, no actual tokeniser dependency)
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return math.ceil(len(text) / 4)


def _projection_token_estimate(proj: SemanticProjection) -> int:
    # exclude_none=True omits Optional fields that are None (e.g.
    # interaction_history when there is no prior history), keeping the
    # token budget tight for entities with no interaction record.
    return _estimate_tokens(
        proj.model_dump_json(exclude_none=True, exclude_defaults=True)
    )


# ---------------------------------------------------------------------------
# Topological position descriptor
# ---------------------------------------------------------------------------

_DIRECTION_TABLE = [
    (0, 0, "same tile"),
    (1, 0, "immediately north"),
    (-1, 0, "immediately south"),
    (0, 1, "immediately east"),
    (0, -1, "immediately west"),
]


def _relative_position_label(origin: Coord, target: Coord) -> str:
    dx = target.x - origin.x
    dy = target.y - origin.y
    dz = target.z - origin.z
    dist = abs(dx) + abs(dy)
    if dist == 0 and dz == 0:
        return "same position"

    if abs(dx) >= abs(dy):
        direction = "east" if dx > 0 else "west"
    else:
        direction = "north" if dy > 0 else "south"

    if dist <= 2:
        proximity = "adjacent"
    elif dist <= 5:
        proximity = "nearby"
    elif dist <= 10:
        proximity = "across the room"
    else:
        proximity = "far"

    vertical = ""
    if dz > 0:
        vertical = f", {dz} level{'s' if dz > 1 else ''} above you"
    elif dz < 0:
        vertical = f", {-dz} level{'s' if -dz > 1 else ''} below you"

    return f"{proximity} to the {direction}{vertical}"


# ---------------------------------------------------------------------------
# Exit detection
# ---------------------------------------------------------------------------


def _detect_exits(
    grid: SpatialGrid, visible_coords: set[Coord]
) -> list[str]:
    exits: list[str] = []
    for coord in visible_coords:
        tile = grid.tile_at(coord)
        if tile.terrain in (TerrainType.DOOR_OPEN, TerrainType.DOOR_CLOSED):
            exits.append(f"door at {_tile_label(coord, grid)}")
        elif tile.terrain in (TerrainType.STAIRS_UP, TerrainType.STAIRS_DOWN):
            exits.append(f"stairs at {_tile_label(coord, grid)}")
        elif "exit" in tile.tags:
            exits.append(f"exit at {_tile_label(coord, grid)}")
    return exits


def _tile_label(coord: Coord, grid: SpatialGrid) -> str:
    """Return a tag-based label for a tile, falling back to terrain name."""
    tile = grid.tile_at(coord)
    if tile.tags:
        return tile.tags[0]
    return tile.terrain.value.replace("_", " ")


# ---------------------------------------------------------------------------
# Cover detection
# ---------------------------------------------------------------------------


def _detect_cover(
    grid: SpatialGrid, focal_pos: Coord, visible_coords: set[Coord]
) -> list[str]:
    cover: list[str] = []
    for coord in visible_coords:
        tile = grid.tile_at(coord)
        obj_here = [
            o
            for o in grid.objects.values()
            if o.position == coord and not o.passable
        ]
        if "cover" in tile.tags or obj_here:
            label = tile.tags[0] if tile.tags else (
                obj_here[0].name if obj_here else "obstacle"
            )
            if coord != focal_pos:
                cover.append(label)
    return list(dict.fromkeys(cover))  # deduplicate while preserving order


# ---------------------------------------------------------------------------
# Affordance detection
# ---------------------------------------------------------------------------


def _detect_affordances(
    focal: EntityState,
    visible_entities: list[EntityState],
    grid: SpatialGrid,
    visible_coords: set[Coord],
) -> list[Affordance]:
    affordances: list[Affordance] = []

    # Movement is always available if there are open neighbors
    open_neighbors = [
        n for n in focal.position.neighbors()
        if grid.is_passable(n)
    ]
    if open_neighbors:
        affordances.append(
            Affordance(
                verb=ActionType.MOVE,
                description="Move to an adjacent passable tile.",
                targets=[str(c) for c in open_neighbors[:4]],
            )
        )

    # Social and combat affordances based on visible entities
    for entity in visible_entities:
        dist = focal.position.manhattan(entity.position)
        if dist <= 6:
            affordances.append(
                Affordance(
                    verb=ActionType.SPEAK,
                    description=f"Speak with {entity.name}.",
                    targets=[entity.entity_id],
                )
            )
        if dist <= 8:
            if entity.alertness.value in ("low", "unaware"):
                affordances.append(
                    Affordance(
                        verb=ActionType.INTIMIDATE,
                        description=f"Attempt to intimidate {entity.name}.",
                        targets=[entity.entity_id],
                    )
                )
            affordances.append(
                Affordance(
                    verb=ActionType.OBSERVE,
                    description=f"Carefully observe {entity.name}.",
                    targets=[entity.entity_id],
                )
            )
        if dist <= 2:
            affordances.append(
                Affordance(
                    verb=ActionType.GIVE,
                    description=f"Give an item to {entity.name}.",
                    targets=[entity.entity_id],
                )
            )
            role = (entity.role or entity.occupation or "").lower()
            if any(k in role for k in ("merchant", "trader", "vendor", "bartender")):
                affordances.append(
                    Affordance(
                        verb=ActionType.TRADE,
                        description=f"Trade or haggle with {entity.name}.",
                        targets=[entity.entity_id],
                    )
                )

    # Objects on ground
    items_nearby = [
        o for o in grid.objects.values()
        if o.position is not None and o.position in visible_coords
    ]
    for obj in items_nearby:
        affordances.append(
            Affordance(
                verb=ActionType.TAKE,
                description=f"Pick up {obj.name}.",
                targets=[obj.object_id],
            )
        )

    # Doors and durable environmental marks
    for coord in visible_coords:
        tile = grid.tile_at(coord)
        if tile.marks and focal.position.manhattan(coord) <= focal.sight_range:
            for mark in tile.marks[-1:]:
                affordances.append(
                    Affordance(
                        verb=ActionType.EXAMINE,
                        description=f"Inspect mark: {mark[:60]}",
                        targets=[coord_key(coord)],
                    )
                )
        if tile.terrain == TerrainType.DOOR_CLOSED:
            if focal.position.manhattan(coord) <= 2:
                affordances.append(
                    Affordance(
                        verb=ActionType.OPEN,
                        description=f"Open the door at {coord}.",
                        targets=[coord_key(coord)],
                    )
                )

    return affordances


def _pack_open_verb_affordances(
    world: WorldState,
    focal: EntityState,
    visible_entities: list[EntityState],
    grid: SpatialGrid,
    visible_coords: set[Coord],
    *,
    max_rules: int = 8,
) -> list[Affordance]:
    """
    Surface pack-declared open verbs so the LM sees world-specific actions.

    Keeps the kernel affordance set small while exposing bribe/sabotage/etc.
    from ``open_verbs.yaml`` without hardcoding them here.
    """
    affordances: list[Affordance] = []
    seen_verbs: set[str] = set()

    for rule in world.config.open_verbs:
        for verb in rule.verbs:
            v = verb.lower().strip()
            if not v or v in seen_verbs:
                continue

            targets: list[str] = []
            if rule.requires == "entity":
                targets = [
                    str(e.entity_id)
                    for e in visible_entities
                    if focal.position.manhattan(e.position) <= 6
                ][:3]
                if not targets:
                    continue
            elif rule.requires == "object":
                targets = [
                    str(o.object_id)
                    for o in grid.objects.values()
                    if o.position is not None
                    and o.position in visible_coords
                    and focal.position.manhattan(o.position) <= 4
                ][:3]
                if not targets:
                    continue

            hint = rule.keywords[0] if rule.keywords else v.replace("_", " ")
            affordances.append(
                Affordance(
                    verb=v,
                    description=f"{v.replace('_', ' ').title()} — e.g. “{hint}”.",
                    targets=targets,
                )
            )
            seen_verbs.add(v)
            if len(affordances) >= max_rules:
                return affordances

    return affordances


def _visible_tile_mark_notes(
    grid: "SpatialGrid",
    focal: "EntityState",
    visible_coords: set,
    *,
    max_items: int = 6,
) -> list[str]:
    """Durable marks on visible tiles — graffiti, blood, chalk circles, etc."""
    from .schemas import coord_key

    notes: list[str] = []
    for coord in sorted(visible_coords, key=lambda c: focal.position.manhattan(c)):
        tile = grid.tile_at(coord)
        if not tile.marks:
            continue
        rel = _relative_position_label(focal.position, coord)
        for mark in tile.marks[-2:]:
            notes.append(f"{rel}: {mark}")
        if len(notes) >= max_items:
            break
    return notes[:max_items]


# ---------------------------------------------------------------------------
# Social context extraction
# ---------------------------------------------------------------------------


def _build_social_context(
    focal: EntityState,
    visible_entities: list[EntityState],
    graph: RelationalGraph,
    recent_events_window: list,
) -> SocialContext:
    active_conflicts: list[str] = []
    known_obligations: list[str] = []
    reputation_notes: list[str] = []
    recent_argument = False
    faction_tensions: list[str] = []

    for entity in visible_entities:
        tension = get_faction_tension(graph, focal.entity_id, entity.entity_id)
        if tension not in ("none", "low"):
            faction_tensions.append(tension)

        rels = get_relationship_summary(graph, focal.entity_id, entity.entity_id)
        for rel in rels:
            if any(
                kw in rel
                for kw in ("enemy", "fears", "distrusts", "owes")
            ):
                active_conflicts.append(rel)
            elif any(kw in rel for kw in ("employs", "employed", "kin", "debt")):
                known_obligations.append(rel)

    # Check recent events for arguments/violence
    for evt in recent_events_window:
        if evt.action.verb in (
            "attack", "intimidate", "threaten"
        ) and focal.entity_id in evt.witnesses:
            recent_argument = True
            break

    overall_tension = (
        "high" if "high" in faction_tensions
        else "medium" if "medium" in faction_tensions
        else "low" if faction_tensions
        else "none"
    )

    return SocialContext(
        recent_argument=recent_argument,
        faction_tension=overall_tension,
        active_conflicts=active_conflicts[:5],
        known_obligations=known_obligations[:5],
        reputation_notes=reputation_notes[:5],
    )


# ---------------------------------------------------------------------------
# Main projection function
# ---------------------------------------------------------------------------


def project(
    world: WorldState,
    focal_entity_id: EntityId,
    *,
    max_tokens: int = MAX_PROJECTION_TOKENS,
    recent_event_window: int = RECENT_EVENT_WINDOW,
    last_action_rejection: Optional[str] = None,
) -> SemanticProjection:
    """
    Build a SemanticProjection for the focal entity.

    This is the ONLY function allowed to read WorldState for the purpose of
    constructing what the LM will see.  It returns a SemanticProjection.
    The LM adapter never receives a WorldState argument.

    Raises KeyError if focal_entity_id is not in the spatial grid.
    """
    try:
        grid = world.grid_for_entity(focal_entity_id)
    except KeyError:
        grid = world.spatial
    graph = world.relational

    focal = grid.entities.get(focal_entity_id)
    if focal is None:
        raise KeyError(
            f"Focal entity '{focal_entity_id}' not found in spatial grid."
        )

    # --- 1. Spatial visibility (cone-filtered by focal entity's facing) ---
    visible_coords = visible_from(
        grid, focal.position, focal.sight_range,
        facing=focal.facing, fov_degrees=focal.fov_degrees,
    )
    visible_entity_states = entities_visible_from(grid, focal_entity_id)
    phys_cfg = world.config.physics_config

    # --- 2. Build entity summaries ---
    from .substance import describe_physical_state
    entity_summaries: list[EntitySummary] = []
    for entity in visible_entity_states:
        if not entity.alive:
            continue

        faction_ids = get_entity_faction_ids(graph, entity.entity_id)
        # Retrieve INTERACTED edge from focal entity → this entity.
        interacted_edge = _get_edge(
            graph,
            str(focal_entity_id),
            str(entity.entity_id),
            EdgeKind.INTERACTED,
        )
        _history = (
            interacted_edge.meta.get("recent_verbs") or None
            if interacted_edge is not None
            else None
        )
        interaction_history: Optional[list[str]] = _history if _history else None

        phys_state = describe_physical_state(entity, phys_cfg) or None
        from .consequences import traces_for_projection

        trace_lines = traces_for_projection(entity.meta, max_items=2)
        if trace_lines:
            extra = "; recently: " + "; ".join(trace_lines)
            phys_state = (phys_state + extra) if phys_state else extra.lstrip("; ")
        summary = EntitySummary(
            entity_id=entity.entity_id,
            name=entity.name,
            kind=entity.kind,
            relative_position=_relative_position_label(
                focal.position, entity.position
            ),
            distance=focal.position.manhattan(entity.position),
            emotional_state=entity.emotional_state,
            alertness=entity.alertness,
            armed=entity.armed,
            intoxication=entity.intoxication,
            faction_ids=faction_ids,
            visible_tags=entity.tags[:4],
            conditions=list(entity.conditions.keys()),
            interaction_history=interaction_history,
            facing=entity.facing.value,
            physical_state=phys_state,
        )
        entity_summaries.append(summary)

    # Sort by distance for stable deterministic ordering
    entity_summaries.sort(key=lambda s: (s.distance, s.entity_id))

    # --- 3. Environment summary ---
    from .substance import build_room_physical_notes, visible_object_physical_lines

    exits = _detect_exits(grid, visible_coords)
    cover = _detect_cover(grid, focal.position, visible_coords)

    entity_count = len(visible_entity_states)
    physical_notes = build_room_physical_notes(
        world, focal, visible_coords, phys_cfg
    )
    object_notes = visible_object_physical_lines(
        world, focal, visible_coords, phys_cfg
    )
    from .economy import format_market_for_projection

    market_notes = format_market_for_projection(world, focal)
    mark_notes = _visible_tile_mark_notes(grid, focal, visible_coords)
    hazards = list(world.meta.get("hazards", []))
    for ent in visible_entity_states:
        if "on_fire" in ent.tags:
            hazards.append(f"{ent.name} is on fire")

    from .fields import region_atmosphere_amount

    smoke_amt = region_atmosphere_amount(world, "smoke")
    if smoke_amt >= 35.0:
        hazards.append("thick smoke — hard to breathe")
    elif smoke_amt >= 12.0:
        hazards.append("smoke hangs in the air")

    # Burning tiles near the focal entity
    for coord in visible_coords:
        tile = grid.tile_at(coord)
        if "on_fire" in {t.lower() for t in tile.tags}:
            hazards.append(f"fire on the floor at {coord}")
            break

    environment = EnvironmentSummary(
        location_name=str(world.meta.get("location_name", "unknown")),
        crowded=entity_count >= 4,
        lighting=str(world.meta.get("lighting", "normal")),
        exits=exits[:6],
        nearby_cover=cover[:6],
        environmental_hazards=hazards[:8],
        noise_level=str(world.meta.get("noise_level", "quiet")),
        physical_notes=physical_notes + mark_notes,
        object_physical_notes=object_notes,
        market_notes=market_notes,
    )

    # --- 4. Recent events ---
    # Two categories are included:
    #   • events the focal entity witnessed (line of sight at event time);
    #   • ambient events (no witnesses — the world is perceived globally).
    # An event is also included if it targeted the focal entity directly,
    # which is how NPCs perceive open-verb interactions where the actor
    # was visible to them at the time (e.g., the player kissing the guard).
    def _focal_perceived(evt) -> bool:
        if focal_entity_id in evt.witnesses:
            return True
        if isinstance(evt.action.target, str) and evt.action.target == focal_entity_id:
            return True
        # Ambient transitions are universally perceived.
        for t in evt.transitions:
            if t.kind == TransitionKind.AMBIENT_EVENT:
                return True
        return False

    recent_raw = [
        evt
        for evt in world.event_log[-recent_event_window * 3:]
        if _focal_perceived(evt)
    ][-recent_event_window:]

    recent_event_summaries: list[EventSummary] = []
    for evt in recent_raw:
        participants = sorted(
            set(
                [evt.action.actor]
                + ([str(evt.action.target)] if evt.action.target else [])
            )
        )
        participant_names: list[str] = []
        for pid in participants:
            node = graph.nodes.get(pid)
            if node:
                participant_names.append(node.name)
            elif pid in grid.entities:
                participant_names.append(grid.entities[EntityId(pid)].name)
            else:
                participant_names.append(pid)

        description = (
            evt.narrative_hint
            or f"{evt.action.verb} by actor at tick {evt.tick}"
        )
        recent_event_summaries.append(
            EventSummary(
                tick=evt.tick,
                description=description,
                participants=participant_names,
            )
        )

    # --- 5. Social context ---
    social_context = _build_social_context(
        focal, visible_entity_states, graph, recent_raw
    )

    # --- 6. Affordances ---
    affordances = _detect_affordances(focal, visible_entity_states, grid, visible_coords)
    affordances.extend(
        _pack_open_verb_affordances(
            world, focal, visible_entity_states, grid, visible_coords,
        )
    )
    from .interaction_resolver import interaction_hints_for_actor

    grammar_affordances = interaction_hints_for_actor(world, focal_entity_id, max_hints=8)
    affordances.extend(grammar_affordances)

    interaction_hint_strings = [a.description for a in grammar_affordances[:8]]

    # --- 6b. Structured beliefs (BELIEVES_CLAIM edges) ---
    beliefs: list[BeliefRecord] = []
    for tgt, edges in graph.edges.get(str(focal_entity_id), {}).items():
        for edge in edges:
            if edge.kind == EdgeKind.BELIEVES_CLAIM:
                claim = (edge.meta or {}).get("claim", "")
                if claim:
                    beliefs.append(
                        BeliefRecord(
                            subject=tgt,
                            claim=claim[:120],
                            confidence=round(edge.weight, 2),
                        )
                    )
    beliefs = beliefs[:8]

    from .world_adjudicator import facts_for_projection
    from .memory_retrieval import memories_for_projection

    established_facts = facts_for_projection(world, focal_entity_id)
    retrieved_memories, chronicle_snippets = memories_for_projection(
        world, focal_entity_id,
    )

    # --- 7. Assemble projection ---
    # projection_id is derived deterministically so identical inputs
    # always produce the same projection bytes.
    import hashlib as _hashlib
    _pid_src = f"{world.world_id}:{focal_entity_id}:{world.tick}"
    _pid = "proj_" + _hashlib.sha256(_pid_src.encode()).hexdigest()[:8]

    # Current phase of the world clock — projection-stable so both
    # player and NPC LM prompts see the same time-of-day field.
    current_time_of_day = world.config.clock.time_of_day(world.tick)

    proj = SemanticProjection(
        projection_id=_pid,
        focal_entity=focal_entity_id,
        tick=world.tick,
        visible_entities=entity_summaries,
        environment=environment,
        social_context=social_context,
        available_affordances=affordances,
        recent_events=recent_event_summaries,
        time_of_day=current_time_of_day,
        last_action_rejection=last_action_rejection,
        current_region=focal.region_id,
        facing=focal.facing.value,
        beliefs=beliefs,
        world_facts=established_facts,
        retrieved_memories=retrieved_memories,
        chronicle_snippets=chronicle_snippets,
        interaction_hints=interaction_hint_strings,
    )

    # --- 8. Token budget enforcement ---
    token_count = _projection_token_estimate(proj)
    if token_count > max_tokens:
        proj = _trim_projection(proj, max_tokens)
        token_count = _projection_token_estimate(proj)

    proj.estimated_tokens = token_count
    return proj


def _trim_projection(
    proj: SemanticProjection, max_tokens: int
) -> SemanticProjection:
    """
    Reduce projection size by progressively dropping lower-priority fields
    until it fits within the token budget.

    Priority order (highest to lowest):
      visible_entities > environment > social_context >
      recent_events > available_affordances
    """
    # Drop affordances first
    proj_copy = proj.model_copy(deep=True)
    while _projection_token_estimate(proj_copy) > max_tokens and proj_copy.available_affordances:
        proj_copy.available_affordances = proj_copy.available_affordances[:-1]

    # Trim recent events
    while _projection_token_estimate(proj_copy) > max_tokens and proj_copy.recent_events:
        proj_copy.recent_events = proj_copy.recent_events[:-1]

    # Trim social context lists
    while _projection_token_estimate(proj_copy) > max_tokens and (
        proj_copy.social_context.active_conflicts
        or proj_copy.social_context.known_obligations
    ):
        if proj_copy.social_context.active_conflicts:
            proj_copy.social_context.active_conflicts = proj_copy.social_context.active_conflicts[:-1]
        elif proj_copy.social_context.known_obligations:
            proj_copy.social_context.known_obligations = proj_copy.social_context.known_obligations[:-1]

    # Trim physical notes (environmental colour, lower priority than entities).
    while _projection_token_estimate(proj_copy) > max_tokens and (
        proj_copy.environment.object_physical_notes
        or proj_copy.environment.physical_notes
    ):
        if proj_copy.environment.object_physical_notes:
            proj_copy.environment.object_physical_notes = (
                proj_copy.environment.object_physical_notes[:-1]
            )
        elif proj_copy.environment.physical_notes:
            proj_copy.environment.physical_notes = (
                proj_copy.environment.physical_notes[:-1]
            )

    if _projection_token_estimate(proj_copy) > max_tokens:
        for entity in proj_copy.visible_entities:
            if entity.physical_state:
                entity.physical_state = ""
                if _projection_token_estimate(proj_copy) <= max_tokens:
                    break

    while _projection_token_estimate(proj_copy) > max_tokens and (
        proj_copy.environment.environmental_hazards
    ):
        proj_copy.environment.environmental_hazards = (
            proj_copy.environment.environmental_hazards[:-1]
        )

    # Trim interaction_history from entity summaries (lowest priority within
    # the entity block — it is decorative context, not structural data).
    if _projection_token_estimate(proj_copy) > max_tokens:
        for entity in proj_copy.visible_entities:
            if entity.interaction_history is not None:
                entity.interaction_history = None
                if _projection_token_estimate(proj_copy) <= max_tokens:
                    break

    while _projection_token_estimate(proj_copy) > max_tokens and proj_copy.beliefs:
        proj_copy.beliefs = proj_copy.beliefs[:-1]

    while _projection_token_estimate(proj_copy) > max_tokens and proj_copy.world_facts:
        proj_copy.world_facts = proj_copy.world_facts[:-1]

    while _projection_token_estimate(proj_copy) > max_tokens and proj_copy.retrieved_memories:
        proj_copy.retrieved_memories = proj_copy.retrieved_memories[:-1]

    while _projection_token_estimate(proj_copy) > max_tokens and proj_copy.chronicle_snippets:
        proj_copy.chronicle_snippets = proj_copy.chronicle_snippets[:-1]

    # Trim visible entities (keep nearest first, already sorted)
    while _projection_token_estimate(proj_copy) > max_tokens and len(proj_copy.visible_entities) > 1:
        proj_copy.visible_entities = proj_copy.visible_entities[:-1]

    return proj_copy


# ---------------------------------------------------------------------------
# Projection validation (used in tests)
# ---------------------------------------------------------------------------


def assert_projection_invariants(
    proj: SemanticProjection,
    world: WorldState,
    focal_entity_id: EntityId,
    max_tokens: int = MAX_PROJECTION_TOKENS,
) -> None:
    """
    Raises AssertionError if any projection invariant is violated.

    Call from tests and from the integration loop as a sanity check.
    """
    try:
        grid = world.grid_for_entity(focal_entity_id)
    except KeyError:
        grid = world.spatial

    # 1. Token bound
    assert proj.estimated_tokens <= max_tokens, (
        f"Projection exceeds token budget: {proj.estimated_tokens} > {max_tokens}"
    )

    # 2. No raw coordinates in visible_entities
    for summary in proj.visible_entities:
        assert not hasattr(summary, "position"), (
            "EntitySummary must not expose raw grid position."
        )

    # 3. Partial observability: all summarised entities must actually be visible
    focal = grid.entities.get(focal_entity_id)
    assert focal is not None, "Focal entity missing from grid."
    from .spatial import visible_from as _vf
    visible_coords = _vf(
        grid, focal.position, focal.sight_range,
        facing=focal.facing, fov_degrees=focal.fov_degrees,
    )
    for summary in proj.visible_entities:
        entity = grid.entities.get(summary.entity_id)
        if entity is not None:
            assert entity.position in visible_coords, (
                f"Entity '{summary.entity_id}' is in projection but not visible "
                f"from focal entity position."
            )

    # 4. Focal entity is not in visible_entities (can't observe yourself)
    ids_in_proj = {s.entity_id for s in proj.visible_entities}
    assert focal_entity_id not in ids_in_proj, (
        "Focal entity must not appear in its own visible_entities list."
    )
