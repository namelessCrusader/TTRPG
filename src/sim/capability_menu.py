"""
Pack-driven capability candidates for NPC policy menus.

The LM (or reactive weights) only choose among actions the engine has
pre-validated. This module turns world-pack declarations into concrete
SemanticAction options:

  - verb_templates.yaml  — social/physical verbs with compiler effects
  - affordances.yaml     — role-tagged special verbs (pour, eavesdrop, …)
  - spell_vocab          — CAST options when entity has inscribed_spells

Kernel verbs (move, take, attack, …) are generated elsewhere in ReactivePolicy.
"""

from __future__ import annotations

from typing import Optional

from .fluids import (
    find_pour_source,
    fluid_volume_ml,
    is_receptacle,
    resolve_receptacle_for_pour,
)
from .projection import visible_from
from .schemas import (
    ActionType,
    EntityState,
    IntentBlock,
    ObjectAffordanceRule,
    ObjectId,
    ObjectState,
    SemanticAction,
    WorldState,
)

# Verbs with dedicated kernel compilers — templates are defaults only.
_KERNEL_VERBS: frozenset[str] = frozenset({
    "move", "take", "give", "attack", "throw", "turn", "speak", "ask",
    "observe", "wait", "rest", "eat", "flee", "contact", "cast",
    "edge_add", "edge_update", "symbolic",
    # Embodied interaction kernels (see interactions.py)
    "sit", "stand", "lean", "slouch", "kneel",
    "wipe", "clean", "polish", "relieve", "wash",
    "nibble", "munch", "sniff", "smell", "listen",
    "laugh", "chuckle", "sing", "hum", "sigh", "cough", "sneeze",
    "pat", "hug", "embrace", "kiss", "slap", "spit", "shove", "push",
    "nudge", "elbow", "bow",
})

# Pack verbs that need a visible entity target.
_ENTITY_TARGET_VERBS: frozenset[str] = frozenset({
    "console", "tease", "flirt", "compliment", "gossip", "haggle",
    "threaten", "pour_drink", "perform", "eavesdrop", "accuse", "wink",
    "shrug", "slam_fist", "toast", "greet", "warn", "challenge",
    "apologize", "thank", "forgive", "bow_to", "wave", "salute",
    "encourage", "mock", "scoff", "joke", "promise", "vow", "lie",
    "agree", "disagree", "smile", "frown", "glare", "nod",
})

# Embodied verbs that target other entities (need adjacency)
_CONTACT_VERBS: frozenset[str] = frozenset({
    "pat", "hug", "embrace", "kiss", "slap", "spit", "shove", "push",
    "nudge", "elbow",
})

# Embodied verbs that target the actor themselves (no external target)
_SELF_VERBS: frozenset[str] = frozenset({
    "sit", "stand", "lean", "slouch", "kneel", "bow",
    "sniff", "smell", "listen", "sigh", "laugh", "chuckle", "hum", "cough", "sneeze",
    "sing", "relieve", "wipe",
})


def _visible_objects(
    entity: EntityState,
    world: WorldState,
    visible_coords: set,
) -> list[ObjectState]:
    out: list[ObjectState] = []
    for obj in world.spatial.objects.values():
        if obj.position and obj.position in visible_coords:
            if entity.position.manhattan(obj.position) <= 10:
                out.append(obj)
    return out


def _object_matches_rule(obj: ObjectState, rule: ObjectAffordanceRule) -> bool:
    tags = {t.lower() for t in (obj.tags or [])}
    required = {t.lower() for t in rule.object_tags}
    absent = {t.lower() for t in rule.absent_tags}
    if required and not tags.intersection(required):
        return False
    if absent and tags.intersection(absent):
        return False
    return True


def pack_capability_candidates(
    entity: EntityState,
    world: WorldState,
    *,
    max_add: int = 24,
) -> list[tuple[SemanticAction, str, float]]:
    """
    Return pack-authored actions the entity can plausibly take this tick.

    Each tuple is (SemanticAction, menu_label, weight) ready for _lm_select or
    reactive weighting.
    """
    out: list[tuple[SemanticAction, str, float]] = []
    seen_verbs: set[tuple[str, str]] = set()

    def _push(
        verb: str,
        target: Optional[str],
        label: str,
        weight: float,
        *,
        raw: str = "[npc_auto:pack_capability]",
        manner: str = "",
    ) -> None:
        if len(out) >= max_add:
            return
        key = (verb, target or "")
        if key in seen_verbs:
            return
        seen_verbs.add(key)
        out.append(
            (
                SemanticAction(
                    verb=verb,
                    actor=entity.entity_id,
                    target=target,
                    intent=IntentBlock(
                        rationale="",
                        manner=manner or "purposeful",
                        desired_outcome=[verb],
                    ),
                    raw_input=raw,
                ),
                label,
                weight,
            )
        )

    grid = world.spatial
    eid = entity.entity_id
    visible_coords = visible_from(grid, entity.position, entity.sight_range)
    visible_entities: list[EntityState] = []
    for ent in grid.entities.values():
        if not ent.alive or ent.entity_id == eid:
            continue
        if ent.position in visible_coords:
            visible_entities.append(ent)

    nearest: Optional[EntityState] = None
    if visible_entities:
        nearest = min(
            visible_entities,
            key=lambda t: entity.position.manhattan(t.position),
        )
    entity_tags = {t.lower() for t in (entity.tags or [])}
    has_company = bool(visible_entities)

    # ── affordances.yaml first (role-tagged verbs beat generic templates) ─
    for rule in world.config.affordances or []:
        keyword = rule.keyword.lower().strip()
        if not keyword or keyword in _KERNEL_VERBS:
            continue
        # Actor has a matching role tag → offer toward nearest visible entity.
        if entity_tags.intersection(t.lower() for t in rule.tags) and nearest is not None:
            if entity.position.manhattan(nearest.position) <= 10:
                if keyword == "pour_drink":
                    if find_pour_source(entity, grid, max_dist=3) is None:
                        continue
                    cup, err = resolve_receptacle_for_pour(
                        grid, entity, str(nearest.entity_id),
                    )
                    if cup is None:
                        continue
                    label = f"pour drink into {cup.name} (for {nearest.name})"
                    _push(
                        keyword,
                        str(cup.object_id),
                        label,
                        0.68,
                        raw=f"[npc_auto:affordance:{keyword}]",
                    )
                else:
                    label = f"{keyword.replace('_', ' ')} toward {nearest.name}"
                    _push(
                        keyword,
                        str(nearest.entity_id),
                        label,
                        0.68,
                        raw=f"[npc_auto:affordance:{keyword}]",
                    )
        # Visible target has a tag the affordance cares about.
        elif nearest is not None:
            tgt_tags = {t.lower() for t in (nearest.tags or [])}
            if tgt_tags.intersection(t.lower() for t in rule.tags):
                if entity.position.manhattan(nearest.position) <= 10:
                    label = f"{keyword.replace('_', ' ')} involving {nearest.name}"
                    _push(
                        keyword,
                        str(nearest.entity_id),
                        label,
                        0.64,
                        raw=f"[npc_auto:affordance_target:{keyword}]",
                    )

    # ── object_affordances (fixtures: hearth, cask, mugs) ───────────────
    visible_objects = _visible_objects(entity, world, visible_coords)
    for rule in world.config.object_affordances or []:
        verb = rule.verb.lower().strip()
        if not verb:
            continue
        for obj in visible_objects:
            if not _object_matches_rule(obj, rule):
                continue
            if verb == "pour_drink":
                if find_pour_source(entity, grid, max_dist=3) is None:
                    continue
                if not is_receptacle(obj):
                    continue
            if verb == "drink" and fluid_volume_ml(obj) <= 0:
                continue
            label = f"{verb.replace('_', ' ')} {obj.name}"
            _push(
                verb,
                str(obj.object_id),
                label,
                0.66,
                raw=f"[npc_auto:object_affordance:{verb}]",
            )
            break  # one object per verb rule

    # Drink from any visible cup with liquid.
    for obj in visible_objects:
        if not is_receptacle(obj) or fluid_volume_ml(obj) <= 0:
            continue
        if entity.position.manhattan(obj.position) <= 2:
            _push("drink", str(obj.object_id), f"drink from {obj.name}", 0.6,
                  raw="[npc_auto:drink_visible]")
            break

    # ── Embodied self-care ─ posture / hygiene / perception ─
    # Gated so a truly idle NPC (no peers, no needs) still picks WAIT.
    posture = entity.meta.get("posture", "standing")
    bladder = float(entity.stats.get("bladder", 0))
    hunger = float(entity.stats.get("hunger", 50))
    fatigue = float(entity.stats.get("fatigue", 50))

    # Sit only when fatigue is real (low value = tired) and someone's
    # around or there's reason to settle. Stand back up when rested.
    if posture == "standing" and fatigue <= 40.0 and (has_company or visible_objects):
        _push("sit", None, "sit down to rest", 0.42 + (40 - fatigue) / 100,
              raw="[npc_auto:self:sit]")
    elif posture != "standing" and fatigue >= 75.0:
        _push("stand", None, f"stand up (was {posture})", 0.35,
              raw="[npc_auto:self:stand]")

    if bladder >= 80.0:
        _push("relieve", None, "relieve self", 0.85,
              raw="[npc_auto:self:relieve]")

    if "wet" in {t.lower() for t in entity.tags}:
        _push("wipe", None, "wipe self dry", 0.55,
              raw="[npc_auto:self:wipe]")

    # hunger semantics: high value = sated, low = hungry.
    if hunger <= 40.0:
        for obj in visible_objects:
            tags = {t.lower() for t in obj.tags}
            if ("food" in tags or "edible" in tags) and entity.position.manhattan(obj.position) <= 2:
                _push("eat", str(obj.object_id), f"eat {obj.name}", 0.8,
                      raw="[npc_auto:self:eat]")
                break

    # Perception verbs — only when there is *something* to perceive nearby.
    if has_company or visible_objects:
        _push("sniff", None, "sniff the air", 0.28, raw="[npc_auto:self:sniff]")
        _push("listen", None, "listen carefully", 0.26, raw="[npc_auto:self:listen]")

    # Expressive — only meaningful when peers can hear/see it.
    if has_company:
        if entity.emotional_state.value in ("happy", "friendly"):
            _push("laugh", None, "burst into laughter", 0.35,
                  raw="[npc_auto:self:laugh]")
            _push("hum", None, "hum a tune", 0.25, raw="[npc_auto:self:hum]")
        elif entity.emotional_state.value in ("grieving", "humiliated", "suspicious"):
            _push("sigh", None, "let out a sigh", 0.30, raw="[npc_auto:self:sigh]")
        inf = entity.meta.get("infection")
        if isinstance(inf, dict) and inf.get("stage") in ("incubating", "sick"):
            _push("cough", None, "cough", 0.55, raw="[npc_auto:self:cough]")

    # Contact verbs toward nearest adjacent entity, mood-conditioned
    if nearest is not None and entity.position.manhattan(nearest.position) <= 1:
        mood = entity.emotional_state.value
        if mood in ("happy", "friendly"):
            _push("pat", str(nearest.entity_id),
                  f"pat {nearest.name} on the shoulder",
                  0.45, raw="[npc_auto:contact:pat]")
        elif mood in ("angry", "hostile"):
            _push("shove", str(nearest.entity_id),
                  f"shove {nearest.name}",
                  0.55, raw="[npc_auto:contact:shove]")
            _push("spit", str(nearest.entity_id),
                  f"spit at {nearest.name}",
                  0.30, raw="[npc_auto:contact:spit]")

    # ── inscribed spells (before generic templates) ─────────────────────
    if not world.meta.get("magic_duel"):
        vocab = world.config.spell_vocab
        if vocab and entity.inscribed_spells:
            for spell_name in list(entity.inscribed_spells.keys())[:4]:
                _push(
                    str(ActionType.CAST),
                    spell_name,
                    f"cast {spell_name}",
                    0.55,
                    raw="[npc_auto:pack_spell]",
                    manner=spell_name,
                )

    # ── verb_templates.yaml (one nearest-target line per verb) ──────────
    # Sort so verbs with contests (richer outcomes) come first; gossip,
    # mock, lie, etc. then beat trivial smile/nod for menu slots.
    def _tmpl_priority(item):
        name, tmpl = item
        has_contest = tmpl.contest is not None
        n_effects = (
            len(getattr(tmpl, "effects_on_success", []) or [])
            + len(getattr(tmpl, "effects_on_failure", []) or [])
        )
        return (-int(has_contest), -n_effects, name)

    for verb, tmpl in sorted(
        (world.config.verb_templates or {}).items(), key=_tmpl_priority,
    ):
        v = verb.lower().strip()
        if v in _KERNEL_VERBS:
            continue
        if v in _ENTITY_TARGET_VERBS and nearest is not None:
            dist = entity.position.manhattan(nearest.position)
            if dist <= 10:
                w = 0.62
                if tmpl.contest is not None:
                    w += 0.06
                label = f"{v.replace('_', ' ')} with {nearest.name}"
                _push(
                    v,
                    str(nearest.entity_id),
                    label,
                    w,
                    raw=f"[npc_auto:verb_template:{v}]",
                )
        elif v not in _ENTITY_TARGET_VERBS:
            # Non-targeting social-style verbs (barter, perform-solo, etc.)
            # only make sense if peers are present. A truly alone NPC has
            # no one to barter with.
            if has_company or visible_objects:
                _push(v, None, v.replace("_", " "), 0.45,
                      raw=f"[npc_auto:verb_template:{v}]")

    return out[:max_add]
