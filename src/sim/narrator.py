"""
M8: Narrative renderer.

STRICTLY DOWNSTREAM.  This module:
  - Reads the event log and canonical entity names.
  - Produces natural language flavor text.
  - Never mutates WorldState, SpatialGrid, or RelationalGraph.
  - Never calls the LM (all rendering is template/rule-based).

The LM may optionally be used for *additional* flavor enrichment via
enrich_with_lm(), which is a separate, explicitly-optional call that
writes only to the returned string — never to canonical state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .schemas import (
    ActionType,
    ActionZone,
    AlertnessLevel,
    Coord,
    EmotionalState,
    EntityId,
    EntityState,
    Event,
    FacingDirection,
    GroundingResult,
    ObjectId,
    SemanticProjection,
    TransitionKind,
    WorldState,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from .speech_utils import is_planning_text as _is_planning_text
from .speech_utils import is_real_dialogue as _is_real_dialogue


def _best_spoken_text(event: "Event", dialogue_t: "Optional[object]") -> str:
    """
    Return the best available spoken text for a speech event.

    Priority:
      1. DIALOGUE_SPOKEN payload["text"] — set by _compile_freeform (loud speech path)
      2. intent.manner — enriched in-character line
      3. intent.rationale — only if it passes is_real_dialogue (not planning stubs)
    """
    # 1. Payload text (always verbatim)
    if dialogue_t is not None:
        t_text = dialogue_t.payload.get("text", "")  # type: ignore[union-attr]
        if t_text and not _is_planning_text(t_text):
            return t_text

    # 2. intent — prefer manner (enriched speech), then rationale
    intent = event.action.intent
    if intent:
        m = (intent.manner or "").strip()
        if m and _is_real_dialogue(m):
            return m
        r = (intent.rationale or "").strip()
        if r and _is_real_dialogue(r):
            return r

    # 3. Nothing real — return empty so narrator prints generic "speaks with"
    return ""


# ---------------------------------------------------------------------------
# Core rendering: rule-based, zero LM, zero state mutation
# ---------------------------------------------------------------------------


def render_event(event: Event, world: WorldState) -> str:
    """
    Produce a narrative sentence describing a single event.

    Uses entity names from the spatial grid.  If an entity has been removed,
    falls back to the entity id.

    Returns a non-empty string.  Never raises.
    """
    grid = world.spatial

    def name(eid: EntityId) -> str:
        e = grid.entities.get(eid)
        return e.name if e else str(eid)

    actor_name = name(event.action.actor)
    verb = event.action.verb

    # Find the most informative transition to base the narration on
    primary = _primary_transition(event)

    if verb == ActionType.MOVE:
        if primary and primary.kind == TransitionKind.ENTITY_MOVED:
            p = primary.payload
            base = f"{actor_name} moves to ({p['to']['x']}, {p['to']['y']})."
        else:
            base = f"{actor_name} moves."
        hints = ambient_affordance_hints(world, event.action.actor, max_hints=2)
        if hints:
            return base + " " + hints[0]
        return base

    if verb in (ActionType.TURN, ActionType.LOOK, ActionType.FACE):
        turned_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ENTITY_TURNED),
            None,
        )
        if turned_t:
            p = turned_t.payload
            base = f"{actor_name} turns to face {p['to']}."
        else:
            base = f"{actor_name} looks around."
        hints = ambient_affordance_hints(world, event.action.actor, max_hints=2)
        if hints:
            return base + " " + " ".join(hints[:2])
        return base

    if verb == ActionType.ATTACK:
        # Check if the attack killed the target.
        died_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ENTITY_DIED),
            None,
        )
        if died_t:
            dead_name = name(EntityId(died_t.payload["entity_id"]))
            return f"{actor_name} strikes {dead_name} a killing blow. {dead_name} collapses and dies."
        if primary and primary.kind == TransitionKind.ENTITY_HEALTH_CHANGED:
            p = primary.payload
            target_name = name(EntityId(p["entity_id"]))
            damage = abs(p["delta"])
            if p.get("backstab"):
                return (
                    f"{actor_name} lunges from behind — {target_name} never sees it coming! "
                    f"{damage} damage."
                )
            return (
                f"{actor_name} attacks {target_name} for {damage} damage."
            )
        return f"{actor_name} attacks."

    if verb in (ActionType.TURN, ActionType.LOOK, ActionType.FACE):
        # Handled above with ambient hints — this branch is no longer reached.
        pass

    if verb == ActionType.CAST:
        return _narrate_spell(event, world, actor_name)

    if verb == ActionType.DROP:
        transfer_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_TRANSFERRED), None
        )
        if transfer_t:
            oid = ObjectId(transfer_t.payload["object_id"])
            obj = grid.objects.get(oid)
            item_name = obj.name if obj else "item"
            return f"{actor_name} drops the {item_name} onto the floor."
        return f"{actor_name} drops something."

    if verb == ActionType.USE:
        item_name = (event.action.intent.rationale or "item") if event.action.intent else "item"
        heal_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
             and t.payload.get("delta", 0) > 0), None
        )
        mana_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
             and t.payload.get("prop_delta", {}).get("mana", 0) > 0), None
        )
        fire_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_CONDITION_CHANGED
             and t.payload.get("_tag_add") == "on_fire"), None
        )
        douse_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_CONDITION_CHANGED
             and t.payload.get("_tag_remove") == "on_fire"), None
        )
        if heal_t:
            hp = heal_t.payload.get("delta", 0)
            return f"{actor_name} drinks the {item_name}. The warmth of healing spreads through their body (+{hp} HP)."
        if mana_t:
            mp = mana_t.payload.get("prop_delta", {}).get("mana", 0)
            return f"{actor_name} drinks the {item_name}. Arcane energy rushes through them (+{mp} mana)."
        if fire_t:
            return f"{actor_name} touches the {item_name} to a flammable surface. It ignites."
        if douse_t:
            return f"{actor_name} douses the flames with the {item_name}. The fire sputters out."
        return f"{actor_name} uses the {item_name}."

    if verb in (ActionType.MIX, "combine"):
        a_name = (event.action.intent.rationale or "item") if event.action.intent else "item"
        b_name = (event.action.intent.manner or "item") if event.action.intent else "item"
        created_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_CREATED), None
        )
        tag_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_CONDITION_CHANGED
             and t.payload.get("_item_tag_add")), None
        )
        dmg_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
             and t.payload.get("cause") == "combination_accident"), None
        )
        result = f"{actor_name} mixes the {a_name} with the {b_name}."
        if created_t:
            result += f" They combine into {created_t.payload.get('name', 'something new')}."
        if tag_t:
            patch = tag_t.payload.get("_item_tag_add", {})
            result += f" The {a_name} gains the '{patch.get('tag', '?')}' property."
        if dmg_t:
            result += f" The mixture splatters — {actor_name} is burned by the accident!"
        return result

    if verb == ActionType.THROW:
        throw_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED), None
        )
        transfer_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_TRANSFERRED), None
        )
        item_name = "something"
        if throw_t:
            item_name = throw_t.payload.get("thrown_item_name", "something")
        target_name_str = _target_name(event, grid)
        base = f"{actor_name} hurls the {item_name} at {target_name_str}"
        if throw_t:
            dmg = abs(throw_t.payload.get("delta", 0))
            base += f" — striking for {dmg} damage"
        base += "."
        # Check physics side-effects (fire, acid, poison)
        phys_tags = [
            t.payload.get("_tag_add") for t in event.transitions
            if t.kind == TransitionKind.ENTITY_CONDITION_CHANGED and t.payload.get("_tag_add")
        ]
        if "on_fire" in phys_tags:
            base += f"  The burning {item_name} ignites {target_name_str}!"
        elif "poisoned" in phys_tags:
            base += f"  Venom splashes across {target_name_str}."
        elif "burned" in phys_tags:
            base += f"  Acid eats into {target_name_str}'s flesh."
        return base

    if verb == ActionType.EXAMINE or verb == "inspect":
        # Pull the examine_result dict from the DIALOGUE_SPOKEN payload
        dia_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.DIALOGUE_SPOKEN), None
        )
        if dia_t:
            r = dia_t.payload.get("examine_result", {})
            detailed = dia_t.payload.get("detailed", False)
            target_nm = r.get("name", _target_name(event, grid))
            if r.get("kind") == "tile":
                lines = [f"{actor_name} inspects the ground at {r.get('coord', 'here')}."]
            else:
                lines = [f"{actor_name} looks {target_nm} over carefully."]
            dist = r.get("distance")
            if dist is not None:
                lines.append(f"  Distance: {dist} paces.")
            em = r.get("emotional_state", "")
            al = r.get("alertness", "")
            if em:
                lines.append(f"  Emotional state: {em}; alertness: {al}.")
            facing_str = r.get("facing", "")
            if facing_str:
                lines.append(f"  Facing: {facing_str}.")
            if detailed:
                occ = r.get("occupation", "")
                faction = r.get("faction", "")
                if occ or faction:
                    lines.append(f"  Occupation: {occ or '—'}; Faction: {faction or '—'}.")
                hp = r.get("health_fraction", 1.0)
                sta = r.get("stamina_fraction", 1.0)
                lines.append(f"  Condition: {int(hp*100)}% health, {int(sta*100)}% stamina.")
                conds = r.get("conditions", [])
                if conds:
                    lines.append(f"  Conditions: {', '.join(conds)}.")
                tags = r.get("tags", [])
                if tags:
                    lines.append(f"  Notable traits: {', '.join(tags[:5])}.")
                weapon = r.get("equipped_weapon")
                armor = r.get("equipped_armor")
                if weapon:
                    lines.append(f"  Weapon: {weapon}.")
                if armor:
                    lines.append(f"  Armor: {armor}.")
                inv_visible = r.get("visible_inventory", [])
                if inv_visible:
                    lines.append(f"  Carrying (visible): {', '.join(inv_visible)}.")
                gold_visible = r.get("gold_visible")
                if gold_visible:
                    lines.append(f"  Appears to be carrying coin ({int(r.get('gold_amount', 0))} gold).")
            else:
                lines.append("  You can't make out many details from here.")
            facts = r.get("established_facts") or []
            if facts:
                lines.append("  The world remembers:")
                for f in facts[:3]:
                    lines.append(f"    • {f}")
            marks = r.get("marks") or []
            if marks:
                lines.append(f"  Marks here: {'; '.join(marks[:3])}.")
            return "\n".join(lines)
        return f"{actor_name} examines their surroundings."

    if verb == ActionType.OBSERVE:
        dia_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.DIALOGUE_SPOKEN), None
        )
        if dia_t and dia_t.payload.get("observe_result"):
            r = dia_t.payload["observe_result"]
            lines = [f"{actor_name} takes in the scene."]
            if r.get("kind") == "entity":
                lines.append(
                    f"  {r.get('name', 'Someone')} — {r.get('distance', '?')} away; "
                    f"mood {r.get('emotional_state', '?')}."
                )
                for f in (r.get("established_facts") or [])[:3]:
                    lines.append(f"    • remembered: {f}")
            elif r.get("kind") == "tile":
                if r.get("marks"):
                    lines.append(f"  Marks: {'; '.join(r['marks'][:3])}.")
                for f in (r.get("established_facts") or [])[:3]:
                    lines.append(f"    • {f}")
                env = r.get("environment") or {}
                if env:
                    lines.append(
                        "  Environment: "
                        + ", ".join(f"{k}={v}" for k, v in list(env.items())[:4])
                    )
            else:
                tiles = r.get("tiles_of_interest") or []
                for ts in tiles[:3]:
                    coord = ts.get("coord", "?")
                    mark_bit = "; ".join(ts.get("marks") or []) or "subtle traces"
                    lines.append(f"  At {coord}: {mark_bit}.")
                ents = r.get("visible_entities") or []
                if ents:
                    lines.append("  Present: " + ", ".join(ents[:4]) + ".")
            return "\n".join(lines)

    if verb == ActionType.TRADE or verb in ("buy", "sell"):
        intent = event.action.intent
        offer_str = intent.rationale if intent else "something"
        want_str = intent.manner if intent else "something"
        target_nm = _target_name(event, grid)

        transferred = [
            t for t in event.transitions if t.kind == TransitionKind.ITEM_TRANSFERRED
        ]
        gold_changes = [
            t for t in event.transitions
            if t.kind == TransitionKind.ENTITY_STAT_CHANGED
            and t.payload.get("stat") == "gold"
        ]

        lines = [f"{actor_name} makes a trade with {target_nm}."]
        if offer_str:
            lines.append(f"  Offered: {offer_str}.")
        if want_str:
            lines.append(f"  Requested: {want_str}.")
        if transferred:
            for tt in transferred:
                oid = tt.payload.get("object_id", "")
                obj = grid.objects.get(ObjectId(oid)) if oid else None
                item_nm = obj.name if obj else oid
                frm = tt.payload.get("from_entity", "?")
                frm_ent = grid.entities.get(EntityId(frm)) if frm else None
                frm_name = frm_ent.name if frm_ent else frm
                lines.append(f"  {item_nm} handed over by {frm_name}.")
        for gc in gold_changes:
            delta = gc.payload.get("delta", 0)
            eid = gc.payload.get("entity_id", "?")
            ent = grid.entities.get(EntityId(eid)) if eid else None
            nm = ent.name if ent else eid
            if delta < 0:
                lines.append(f"  {nm} pays {abs(delta):.0f} gold.")
            elif delta > 0:
                lines.append(f"  {nm} receives {delta:.0f} gold.")
        return "\n".join(lines)

    if verb == ActionType.STEAL or verb == "pickpocket":
        intent = event.action.intent
        item_hint = intent.rationale if intent else "something"
        target_nm = _target_name(event, grid)

        succeeded = event.validation and event.validation.valid and (
            event.validation.rejection_detail != "__steal_failed__"
        )
        transferred = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_TRANSFERRED), None
        )
        if succeeded and transferred:
            oid = ObjectId(transferred.payload["object_id"])
            obj = grid.objects.get(oid)
            item_nm = obj.name if obj else item_hint
            return (
                f"{actor_name} slips a hand into {target_nm}'s possessions "
                f"and quietly palms the {item_nm} — undetected."
            )
        else:
            return (
                f"{actor_name} reaches for {target_nm}'s {item_hint} — "
                f"but {target_nm} notices! {target_nm} recoils in alarm."
            )

    if verb == ActionType.REST or verb == "sleep":
        stamina_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_STAT_CHANGED
             and t.payload.get("stat") == "stamina"), None
        )
        health_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED
             and t.payload.get("cause") == "rest"), None
        )
        lines = [f"{actor_name} takes a moment to rest."]
        if stamina_t:
            delta = stamina_t.payload.get("delta", 0)
            lines.append(f"  Stamina restored (+{delta:.0f}).")
        if health_t:
            delta = health_t.payload.get("delta", 0)
            lines.append(f"  Minor wounds close (+{delta} HP).")
        if not stamina_t and not health_t:
            lines.append("  Fully rested — nothing to recover.")
        return "\n".join(lines)

    if verb in (ActionType.CRAFT, "make", "build", "forge", "whittle", "fabricate"):
        synth_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_SYNTHESIZED), None
        )
        consumed = [
            t for t in event.transitions if t.kind == TransitionKind.ITEM_DESTROYED
        ]
        intent = event.action.intent
        output_name = intent.rationale if intent else "something"

        if synth_t:
            item_nm = synth_t.payload.get("name", output_name)
            tags = synth_t.payload.get("tags", [])
            tag_str = ", ".join(t for t in tags if t not in ("crafted", "improvised"))[:60]
            lines = [
                f"{actor_name} carefully works the materials into a {item_nm}."
            ]
            if "improvised" in tags:
                lines.append(f"  It's crude but functional — [{tag_str}]." if tag_str else
                              "  It's crude but functional.")
            else:
                lines.append(f"  [{tag_str}]" if tag_str else "")
            if consumed:
                mats = [
                    (grid.objects.get(ObjectId(t.payload["object_id"])) or
                     type("", (), {"name": "material"})()).name
                    for t in consumed
                ]
                lines.append(f"  Used: {', '.join(mats)}.")
            attribs = synth_t.payload.get("attributes", {})
            if attribs:
                attr_str = ", ".join(f"{k}:{v}" for k, v in list(attribs.items())[:3])
                lines.append(f"  Properties: {attr_str}.")
            return "\n".join(l for l in lines if l)

        # craft validation failed — the action still fired (e.g. no materials)
        return f"{actor_name} tries to craft {output_name} but can't work out how with these materials."

    # ── mark ───────────────────────────────────────────────────────────────
    if verb in (ActionType.MARK, "mark", "carve", "scratch", "draw", "write", "inscribe"):
        marked_t = next((t for t in event.transitions
                         if t.kind == TransitionKind.TILE_MARKED), None)
        if marked_t:
            mark = marked_t.payload.get("mark", "")
            removing = marked_t.payload.get("remove", False)
            x, y = marked_t.payload.get("x", "?"), marked_t.payload.get("y", "?")
            if removing:
                return f"{actor_name} scrapes away the mark '{mark}' from the floor."
            return (
                f"{actor_name} {_mark_verb(verb)} \"{mark}\" onto the ground "
                f"at ({x},{y})."
            )
        return f"{actor_name} traces something onto the ground, though nothing permanent takes hold."

    # ── barricade ─────────────────────────────────────────────────────────
    if verb in (ActionType.BARRICADE, "barricade", "block", "blockade", "pile"):
        struct_t = next((t for t in event.transitions
                         if t.kind == TransitionKind.STRUCTURE_CREATED), None)
        if struct_t:
            name = struct_t.payload.get("name", "barricade")
            x, y = struct_t.payload.get("x", "?"), struct_t.payload.get("y", "?")
            return (
                f"{actor_name} heaves materials into place, forming a {name} "
                f"at ({x},{y}).  The path is now blocked."
            )
        return f"{actor_name} tries to barricade the way but lacks the materials."

    # ── unlock / lock ─────────────────────────────────────────────────────
    if verb in (ActionType.UNLOCK, "unlock", "lock", "open", "seal", "bolt"):
        env_t = next((t for t in event.transitions
                      if t.kind == TransitionKind.ENVIRONMENT_STATE_CHANGED
                      and t.payload.get("key") == "locked"), None)
        mod_t = next((t for t in event.transitions
                      if t.kind == TransitionKind.STRUCTURE_MODIFIED), None)
        if env_t or mod_t:
            coord = f"({env_t.payload.get('x', '?')},{env_t.payload.get('y', '?')})" if env_t else "the door"
            locked_val = env_t.payload.get("value", False) if env_t else None
            if locked_val:
                return f"{actor_name} locks {coord} with a decisive click."
            return f"{actor_name} fiddles with the mechanism — {coord} swings open."
        return f"{actor_name} attempts to work the lock but it holds fast."

    # ── ignite ────────────────────────────────────────────────────────────
    if verb in (ActionType.IGNITE, "ignite", "light", "kindle", "torch"):
        struct_t = next((t for t in event.transitions
                         if t.kind == TransitionKind.STRUCTURE_MODIFIED
                         and "on_fire" in t.payload.get("add_tags", [])), None)
        if struct_t:
            oid = struct_t.payload.get("object_id", "")
            obj = grid.objects.get(ObjectId(oid)) if oid else None
            obj_name = obj.name if obj else "it"
            return (
                f"{actor_name} touches flame to {obj_name}.  "
                f"It catches — orange light flickers across the room."
            )
        return f"{actor_name} tries to light it but can't find a flame."

    if verb in ("extinguish", "douse", "quench"):
        struct_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.STRUCTURE_MODIFIED
             and "on_fire" in (t.payload.get("remove_tags") or [])),
            None,
        )
        if struct_t:
            oid = struct_t.payload.get("object_id", "")
            obj = grid.objects.get(ObjectId(oid)) if oid else None
            obj_name = obj.name if obj else "the flames"
            return f"{actor_name} douses {obj_name}; smoke curls toward the rafters."
        return f"{actor_name} looks for a fire to put out but finds none."

    if verb == ActionType.INTIMIDATE:
        target_name = _target_name(event, grid)
        if primary and primary.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
            new_state = EmotionalState(primary.payload["to"])
            if new_state == EmotionalState.FEARFUL:
                return (
                    f"{actor_name} fixes {target_name} with a cold stare. "
                    f"{target_name} shrinks back, visibly unsettled."
                )
            return (
                f"{actor_name} attempts to intimidate {target_name}, "
                f"but {target_name} stands firm, jaw clenching."
            )
        return f"{actor_name} attempts to intimidate {target_name}."

    if verb == ActionType.PERSUADE:
        target_name = _target_name(event, grid)
        if primary and primary.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
            new_state = EmotionalState(primary.payload["to"])
            if new_state == EmotionalState.FRIENDLY:
                return (
                    f"{actor_name} speaks carefully to {target_name}. "
                    f"Something in the words lands — {target_name} nods slowly."
                )
            else:
                return (
                    f"{actor_name} tries to reason with {target_name}, "
                    f"but {target_name} remains unconvinced."
                )
        return f"{actor_name} attempts to persuade {target_name}."

    if verb == ActionType.DECEIVE:
        target_name = _target_name(event, grid)
        if primary and primary.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
            new_state = EmotionalState(primary.payload["to"])
            if new_state in (EmotionalState.NEUTRAL, EmotionalState.FRIENDLY):
                return (
                    f"{actor_name} weaves a plausible lie. "
                    f"{target_name} seems to accept it."
                )
            else:
                return (
                    f"{actor_name} tries to deceive {target_name}, "
                    f"but {target_name}'s eyes narrow with suspicion."
                )
        return f"{actor_name} attempts to deceive {target_name}."

    if verb == ActionType.THREATEN:
        target_name = _target_name(event, grid)
        return f"{actor_name} levels a threat at {target_name}."

    if verb == ActionType.SPEAK or verb == ActionType.ASK:
        target_name = _target_name(event, grid)
        # Look for a DIALOGUE_SPOKEN transition that carries an NPC reply or text.
        dialogue_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.DIALOGUE_SPOKEN),
            None,
        )
        reply_text = dialogue_t.payload.get("reply_text") if dialogue_t else None
        reply_speaker = (
            dialogue_t.payload.get("reply_speaker", target_name) if dialogue_t else target_name
        )
        spoken_text = _best_spoken_text(event, dialogue_t)
        if spoken_text:
            base = f"{actor_name} says to {target_name}: \"{spoken_text}\""
        else:
            base = f"{actor_name} speaks with {target_name}."
        if reply_text:
            return f"{base}\n  {reply_speaker}: \"{reply_text}\""
        return base

    if verb == ActionType.OBSERVE:
        import random as _rnd, hashlib as _hl
        target_name = _target_name(event, grid)
        _seed = int(_hl.md5(
            f"{event.action.actor}{event.action.target}{event.tick}".encode()
        ).hexdigest()[:8], 16)
        _rnd.seed(_seed)
        if target_name:
            _phrases = [
                f"{actor_name} glances at {target_name}.",
                f"{actor_name} studies {target_name} quietly.",
                f"{actor_name} watches {target_name} from the corner of an eye.",
                f"{actor_name} tracks {target_name}'s movements.",
                f"{actor_name} keeps an eye on {target_name}.",
                f"{actor_name} listens as {target_name} moves nearby.",
                f"{actor_name} takes note of {target_name}.",
                f"{actor_name} narrows their eyes at {target_name}.",
            ]
        else:
            _phrases = [
                f"{actor_name} surveys the room.",
                f"{actor_name} scans their surroundings.",
                f"{actor_name} takes stock of the area.",
                f"{actor_name} lets the silence settle.",
                f"{actor_name} sits with their thoughts for a moment.",
            ]
        return _rnd.choice(_phrases)

    if verb == ActionType.GIVE:
        if primary and primary.kind == TransitionKind.ITEM_TRANSFERRED:
            p = primary.payload
            target_name = name(EntityId(p["to_entity"]))
            obj = grid.objects.get(p["object_id"])
            obj_name = obj.name if obj else str(p["object_id"])
            return f"{actor_name} hands {obj_name} to {target_name}."
        return f"{actor_name} gives something away."

    if verb == ActionType.TAKE:
        if primary and primary.kind == TransitionKind.ITEM_RETRIEVED:
            p = primary.payload
            item = world.spatial.objects.get(p.get("object_id", ""))
            container = world.spatial.objects.get(p.get("container_id", ""))
            item_n = item.name if item else "item"
            container_n = container.name if container else "container"
            return f"{actor_name} retrieves {item_n} from {container_n}."
        return f"{actor_name} picks something up."

    if verb in (ActionType.EQUIP, ActionType.WEAR, ActionType.DRAW, ActionType.SHEATHE):
        equipped_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_EQUIPPED), None
        )
        if equipped_t:
            p = equipped_t.payload
            item = world.spatial.objects.get(p.get("object_id", ""))
            slot = p.get("slot", "slot")
            item_n = item.name if item else "item"
            replaced_n = ""
            if p.get("replaced_object_id"):
                r = world.spatial.objects.get(p["replaced_object_id"])
                if r:
                    replaced_n = f" (replacing {r.name})"
            return f"{actor_name} equips {item_n} to {slot.replace('_', ' ')}{replaced_n}."
        return f"{actor_name} equips an item."

    if verb in (ActionType.UNEQUIP, ActionType.REMOVE):
        unequipped_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_UNEQUIPPED), None
        )
        if unequipped_t:
            p = unequipped_t.payload
            item = world.spatial.objects.get(p.get("object_id", ""))
            slot = p.get("slot", "slot")
            item_n = item.name if item else "item"
            return f"{actor_name} removes {item_n} from {slot.replace('_', ' ')}."
        return f"{actor_name} unequips an item."

    if verb == ActionType.STORE:
        stored_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_STORED), None
        )
        if stored_t:
            p = stored_t.payload
            item = world.spatial.objects.get(p.get("object_id", ""))
            container = world.spatial.objects.get(p.get("container_id", ""))
            item_n = item.name if item else "item"
            container_n = container.name if container else "container"
            return f"{actor_name} tucks {item_n} into {container_n}."
        return f"{actor_name} stores an item."

    if verb == ActionType.RETRIEVE:
        retrieved_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.ITEM_RETRIEVED), None
        )
        if retrieved_t:
            p = retrieved_t.payload
            item = world.spatial.objects.get(p.get("object_id", ""))
            container = world.spatial.objects.get(p.get("container_id", ""))
            item_n = item.name if item else "item"
            container_n = container.name if container else "container"
            return f"{actor_name} pulls {item_n} out of {container_n}."
        return f"{actor_name} retrieves an item."

    if verb == ActionType.OPEN:
        return f"{actor_name} opens a door."

    if verb == ActionType.CLOSE:
        return f"{actor_name} closes a door."

    if verb == ActionType.FLEE:
        return f"{actor_name} flees."

    if verb == ActionType.HIDE:
        hidden_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_PROPERTY_CHANGED
             and t.payload.get("prop") == "hidden"),
            None,
        )
        moved = any(t.kind == TransitionKind.ENTITY_MOVED for t in event.transitions)
        if hidden_t and hidden_t.payload.get("new_value"):
            if moved:
                return f"{actor_name} ducks behind cover and vanishes from sight."
            return f"{actor_name} melts into the shadows, unseen."
        if moved:
            return f"{actor_name} dives for cover — but someone notices."
        return f"{actor_name} tries to hide, but fails to escape notice."

    if verb == ActionType.WAIT:
        rested = any(
            t.kind == TransitionKind.NEED_CHANGED
            and t.payload.get("need") == "fatigue"
            and float(t.payload.get("delta", 0)) > 0
            for t in event.transitions
        )
        if actor.meta.get("hidden") if (actor := grid.entities.get(event.action.actor)) else False:
            return f"{actor_name} holds still, staying out of sight."
        if rested:
            return f"{actor_name} pauses to catch their breath."
        return f"{actor_name} waits."

    if verb in (ActionType.EDGE_ADD, ActionType.EDGE_UPDATE, "edge_create", "relationship_add", "relationship_update"):
        edge_t = next(
            (t for t in event.transitions
             if t.kind in (TransitionKind.EDGE_CREATED, TransitionKind.EDGE_UPDATED)),
            None,
        )
        target_name = _target_name(event, grid) or "someone"
        if edge_t is not None:
            ek = (
                edge_t.payload.get("edge_kind")
                or edge_t.payload.get("kind")
                or "linked"
            )
            ek_label = str(ek).replace("_", " ")
            if edge_t.kind == TransitionKind.EDGE_CREATED:
                return (
                    f"{actor_name} marks their stance toward {target_name} "
                    f"— {ek_label}."
                )
            return (
                f"{actor_name}'s regard for {target_name} shifts "
                f"({ek_label})."
            )
        return f"{actor_name} adjusts how they see {target_name}."

    # Generic / freeform verb. The compiler stores a DIALOGUE_SPOKEN
    # transition with verb + manner + contest_outcome on every generic
    # interaction; use it for natural prose rather than a bare label.
    # Region transit (portal traversal)
    region_t = next(
        (t for t in event.transitions if t.kind == TransitionKind.REGION_TRANSIT),
        None,
    )
    if region_t is not None:
        to_rid = region_t.payload.get("to_region", "unknown")
        # Try to get a pretty region name from the world
        to_region = world.regions.get(to_rid)
        region_label = to_region.name if to_region else to_rid.replace("_", " ")
        arrival = region_t.payload.get("arrival", {})
        return (
            f"{actor_name} passes through the portal and arrives in "
            f"{region_label} ({arrival.get('x', '?')}, {arrival.get('y', '?')})."
        )

    # Generic death event (fired outside of an attack, e.g. poison)
    died_t = next(
        (t for t in event.transitions if t.kind == TransitionKind.ENTITY_DIED),
        None,
    )
    if died_t is not None:
        dead_name = name(EntityId(died_t.payload["entity_id"]))
        cause = died_t.payload.get("cause", "unknown causes")
        return f"{dead_name} perishes from {cause}."

    # Condition change
    cond_t = next(
        (t for t in event.transitions if t.kind == TransitionKind.ENTITY_CONDITION_CHANGED),
        None,
    )
    if cond_t is not None:
        affected = name(EntityId(cond_t.payload["entity_id"]))
        condition = cond_t.payload.get("condition", "unknown")
        ticks = int(cond_t.payload.get("ticks", 0))
        if ticks > 0:
            return f"{affected} is {condition} for {ticks} more tick(s)."
        return f"{affected}'s {condition} condition ends."

    fluid_ts = [t for t in event.transitions if t.kind == TransitionKind.FLUID_CHANGED]
    if fluid_ts and str(verb).lower() in ("pour_drink", "drink"):
        spill = next((t for t in fluid_ts if t.payload.get("cause") == "overflow"), None)
        pour = next((t for t in fluid_ts if t.payload.get("cause") == "pour"), None)
        accident = next((t for t in fluid_ts if t.payload.get("cause") == "bladder_accident"), None)
        if accident:
            return f"{actor_name} loses control — warmth spreads down their leg; the floor glistens."
        if spill:
            return (
                f"{actor_name} pours too fast — ale slops over the rim and "
                f"pools on the boards."
            )
        if pour and str(verb).lower() == "pour_drink":
            oid = pour.payload.get("object_id", "")
            obj = grid.objects.get(ObjectId(oid)) if oid else None
            oname = obj.name if obj else "the cup"
            return f"{actor_name} fills {oname} with a steady pour."
        if str(verb).lower() == "drink":
            return f"{actor_name} drinks deeply, throat working."

    v = str(verb).lower()

    # ── Embodied posture verbs ───────────────────────────────────────────
    if v in ("sit", "stand", "lean", "slouch", "kneel", "bow"):
        target_obj = None
        if event.action.target:
            target_obj = grid.objects.get(ObjectId(str(event.action.target)))
        sit_phrases = {
            "sit": "settles onto a seat",
            "stand": "rises to their feet",
            "lean": "leans back, weight off one leg",
            "slouch": "slouches with practised ease",
            "kneel": "lowers themselves to one knee",
            "bow": "bows their head",
        }
        if target_obj is not None:
            return f"{actor_name} {sit_phrases.get(v, v)} by the {target_obj.name}."
        return f"{actor_name} {sit_phrases.get(v, v)}."

    # ── Hygiene ──────────────────────────────────────────────────────────
    if v in ("wipe", "wash"):
        fluid = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.FLUID_CHANGED
             and t.payload.get("target_kind") == "tile_dry"),
            None,
        )
        if fluid:
            return f"{actor_name} wipes the wet patch off the boards."
        return f"{actor_name} wipes themselves down with a rag."

    if v in ("clean", "polish"):
        return f"{actor_name} sets to cleaning, focused and methodical."

    if v == "relieve":
        accident = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.FLUID_CHANGED
             and t.payload.get("cause") == "public_relief"
             and t.payload.get("target_kind") == "tile"),
            None,
        )
        if accident:
            return f"{actor_name} can hold it no longer — they relieve themselves where they stand, mortified."
        return f"{actor_name} steps aside to relieve themselves."

    # ── Sustenance ───────────────────────────────────────────────────────
    if v in ("eat", "nibble", "munch"):
        food_obj = None
        if event.action.target:
            food_obj = grid.objects.get(ObjectId(str(event.action.target)))
        if food_obj is None:
            for t in event.transitions:
                if t.kind == TransitionKind.ITEM_DESTROYED:
                    oid = t.payload.get("object_id")
                    if oid:
                        food_obj = grid.objects.get(ObjectId(str(oid)))
                        if food_obj is not None:
                            break
        food_n = food_obj.name if food_obj else "something"
        return f"{actor_name} {v}s the {food_n} hungrily."

    # ── Perception ───────────────────────────────────────────────────────
    if v in ("sniff", "smell"):
        smell_t = next(
            (t for t in event.transitions
             if t.kind == TransitionKind.ENTITY_PROPERTY_CHANGED
             and t.payload.get("prop") == "last_smell"),
            None,
        )
        if smell_t:
            note = str(smell_t.payload.get("new_value", "nothing notable"))
            return f"{actor_name} sniffs the air — {note}."
        return f"{actor_name} sniffs the air."

    if v == "listen":
        return f"{actor_name} tilts their head, listening."

    # ── Expressive ───────────────────────────────────────────────────────
    if v in ("laugh", "chuckle"):
        return f"{actor_name} bursts into open laughter; nearby moods lift."
    if v == "sing":
        return f"{actor_name} sings — clear, unbidden, the room shifts toward warmth."
    if v == "hum":
        return f"{actor_name} hums quietly under their breath."
    if v == "sigh":
        return f"{actor_name} lets out a long sigh."

    # ── Contact ──────────────────────────────────────────────────────────
    if v in ("pat", "hug", "embrace", "kiss", "slap", "spit", "shove", "push", "nudge", "elbow"):
        contact_t = next(
            (t for t in event.transitions if t.kind == TransitionKind.CONTACT_INITIATED),
            None,
        )
        target_name = ""
        if contact_t:
            target_name = name(EntityId(contact_t.payload.get("target", "")))
        elif event.action.target:
            target_name = name(EntityId(str(event.action.target)))
        target_name = target_name or "someone"
        phrases = {
            "pat":     f"{actor_name} pats {target_name} on the shoulder.",
            "hug":     f"{actor_name} pulls {target_name} into a brief hug.",
            "embrace": f"{actor_name} embraces {target_name} firmly.",
            "kiss":    f"{actor_name} kisses {target_name} — quick, deliberate.",
            "slap":    f"{actor_name} slaps {target_name} hard across the face.",
            "spit":    f"{actor_name} spits at {target_name}. The insult lands.",
            "shove":   f"{actor_name} shoves {target_name} backward.",
            "push":    f"{actor_name} pushes {target_name} aside.",
            "nudge":   f"{actor_name} nudges {target_name} with an elbow.",
            "elbow":   f"{actor_name} jabs {target_name} with their elbow.",
        }
        return phrases.get(v, f"{actor_name} makes contact with {target_name}.")

    record = next(
        (t for t in event.transitions
         if t.kind == TransitionKind.DIALOGUE_SPOKEN and "verb" in t.payload),
        None,
    )
    if record is not None:
        r_verb = record.payload.get("verb", verb).lower()
        target_name = _target_display_name(event, grid) or ""
        pack_line = _pack_verb_narrative(
            world, r_verb, record.payload.get("contest_outcome"), event, grid,
        )
        if pack_line:
            return pack_line
        loud = record.payload.get("loud", False)
        reply_text = record.payload.get("reply_text")
        reply_speaker = record.payload.get("reply_speaker", target_name)

        # Resolve the best spoken text — never show planning strings
        spoken = _best_spoken_text(event, record)

        # ── Loud speech (yell / shout / scream / etc.) ──────────────────
        _LOUD_V = {"yell", "shout", "scream", "bellow", "roar",
                   "holler", "cry", "announce", "declare", "exclaim", "call"}
        if loud or r_verb in _LOUD_V:
            # Conjugate: yell→yells, shout→shouts, scream→screams
            if r_verb.endswith("l") or r_verb.endswith("t") or r_verb.endswith("m"):
                conj = f"{r_verb}s"
            else:
                conj = f"{r_verb}s"
            prefix = f"{actor_name} {conj}"
            if target_name:
                prefix += f" at {target_name}"
            else:
                prefix += " into the air"
            output = prefix
            if spoken:
                output += f': "{spoken}"'
            if reply_text:
                output += f"\n  {reply_speaker}: \"{reply_text}\""
            return output

        # ── Normal speech ────────────────────────────────────────────────
        outcome = record.payload.get("contest_outcome")
        verb_past = r_verb.replace("_", " ")
        prefix = f"{actor_name} {verb_past}s"
        if target_name:
            prefix += f" to {target_name}"
        if spoken:
            prefix += f': "{spoken}"'
        if outcome == "failure":
            prefix += " (and falls flat)"
        base = prefix + "."
        if reply_text:
            return f"{base}\n  {reply_speaker}: \"{reply_text}\""
        return base
    # No DIALOGUE_SPOKEN but there may be a reply attached via narrative_hint
    # (set by game_loop for contact verbs that don't produce DIALOGUE_SPOKEN).
    return f"{actor_name} acts ({verb})."


def render_event_log(
    world: WorldState,
    *,
    start_tick: int = 0,
    end_tick: Optional[int] = None,
    witness: Optional[EntityId] = None,
) -> list[str]:
    """
    Render a slice of the event log into narrative sentences.

    Parameters
    ----------
    start_tick : Only include events at or after this tick.
    end_tick   : Only include events before this tick.  None = all.
    witness    : If provided, only render events witnessed by this entity.
    """
    lines: list[str] = []
    for event in world.event_log:
        if event.tick < start_tick:
            continue
        if end_tick is not None and event.tick >= end_tick:
            continue
        if witness is not None and witness not in event.witnesses:
            continue
        lines.append(f"[t{event.tick:04d}] {render_event(event, world)}")
    return lines


def render_entity_status(entity: EntityState) -> str:
    """Produce a compact status string for an entity."""
    parts = [
        f"{entity.name}",
        f"hp={entity.health}/{entity.max_health}",
        f"mood={entity.emotional_state.value}",
        f"alert={entity.alertness.value}",
    ]
    if entity.armed:
        parts.append("armed")
    if entity.intoxication > 20:
        parts.append(f"intox={entity.intoxication}")
    return "  ".join(parts)


def render_inventory(entity: EntityState, world: "WorldState") -> str:
    """
    Render a detailed inventory readout for an entity, including:
    • Carry capacity and current weight.
    • Equipped slots.
    • Loose inventory items (with weight and bulk).
    • Container contents.
    """
    from .compiler import current_carry_weight
    grid = world.spatial
    carried = current_carry_weight(entity, grid)
    enc_pct = int(carried / entity.carry_capacity * 100) if entity.carry_capacity else 0
    enc_bar = "■" * (enc_pct // 10) + "□" * (10 - enc_pct // 10)
    enc_label = "OVER-ENCUMBERED" if enc_pct > 100 else ("heavy" if enc_pct > 75 else "ok")

    lines: list[str] = [
        f"── Inventory: {entity.name} ──────────────────────────",
        f"  Carry: {carried:.1f} / {entity.carry_capacity:.1f} kg  "
        f"[{enc_bar}] {enc_pct}%  ({enc_label})",
    ]

    # Equipped slots
    if entity.equipped_slots:
        lines.append("  Equipped:")
        for slot, oid in sorted(entity.equipped_slots.items()):
            obj = grid.objects.get(oid)
            obj_n = obj.name if obj else str(oid)
            obj_w = f"{obj.weight:.1f} kg" if obj else "?"
            lines.append(f"    [{slot:<12}] {obj_n}  ({obj_w})")

    # Loose inventory (items not in a container)
    loose = [
        oid for oid in entity.inventory
        if oid not in entity.equipped_slots.values()
    ]
    if loose:
        lines.append("  Loose items:")
        for oid in loose:
            obj = grid.objects.get(oid)
            if obj is None:
                continue
            obj_w = f"{obj.weight:.1f} kg"
            obj_b = f"{obj.bulk:.1f} bulk"
            container_info = ""
            if obj.is_container:
                content_w = sum(
                    (grid.objects.get(c).weight if grid.objects.get(c) else 0.0)
                    for c in obj.contents
                )
                container_info = (
                    f"  [bag: {content_w:.1f}/{obj.carry_capacity:.1f} kg, "
                    f"{len(obj.contents)} items]"
                )
                # Show contents
                lines.append(
                    f"    {obj.name}  ({obj_w}, {obj_b}){container_info}"
                )
                for c_oid in obj.contents:
                    c_obj = grid.objects.get(c_oid)
                    if c_obj:
                        lines.append(
                            f"      └─ {c_obj.name}  ({c_obj.weight:.1f} kg, "
                            f"{c_obj.bulk:.1f} bulk)"
                        )
            else:
                lines.append(f"    {obj.name}  ({obj_w}, {obj_b})")

    if len(lines) == 2:
        lines.append("  (nothing)")
    lines.append("─" * 50)
    return "\n".join(lines)


def render_world_summary(world: WorldState) -> str:
    """One-paragraph snapshot of the world for dev/debug use."""
    lines = [
        f"World: {world.name}  tick={world.tick}  "
        f"entities={len(world.spatial.entities)}  "
        f"events={len(world.event_log)}"
    ]
    for eid, entity in sorted(world.spatial.entities.items()):
        lines.append(f"  {render_entity_status(entity)}  pos={entity.position}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NarratorActor — LM-powered downstream narration
# ---------------------------------------------------------------------------


class NarratorActor:
    """
    Downstream narrator that produces atmospheric prose after every action.

    Responsibilities:
      - Zone 3 (ORPHANED): primary response — the world's only meaningful
        reply to actions that have no simulation grounding.
      - Zone 1/2 (post-resolution): optional flavor text enriching the
        mechanical outcome.

    Invariants:
      - Never mutates WorldState, SpatialGrid, or RelationalGraph.
      - Only called AFTER the Compiler has already resolved the action.
      - If the adapter is a MockLMAdapter, falls back to template text.
      - All output is a single string; the caller decides how to display it.
    """

    # Template responses for Zone 3 when no real LM is available.
    # Keyed on orphan_reason prefix substrings.
    _ZONE3_TEMPLATES: list[tuple[str, str]] = [
        ("prayer", "Your lips move in silence. Nothing answers."),
        ("deity", "The gods, if they exist here, are not listening."),
        ("shrine", "There is no shrine. Whatever you were hoping for has no hook to catch on."),
        ("magic", "There is no magic here — or none you know how to reach."),
        ("ruler", "No crown rules this place, as far as you know."),
        ("temple", "No temple, no altar, no consecrated ground. Just stone and noise."),
    ]

    _ZONE3_DEFAULT = (
        "You attempt it. The world offers no response. "
        "The moment passes without consequence."
    )

    def __init__(self, adapter=None):
        """
        Parameters
        ----------
        adapter : Optional LMAdapter.  If None or MockLMAdapter, uses templates.
        """
        self._adapter = adapter

    def narrate_zone3(
        self,
        original_intent: str,
        orphan_reason: str,
        projection: "SemanticProjection",
        world: WorldState,
    ) -> str:
        """
        Produce a narrative response for a Zone 3 (ORPHANED) action.

        This is the sole meaningful output when the simulation has nothing
        to say — the world's aesthetic acknowledgment of an action it
        cannot resolve.
        """
        if self._is_real_adapter():
            return self._lm_narrate_zone3(
                original_intent, orphan_reason, projection, world
            )
        return self._template_zone3(orphan_reason, world, projection)

    def narrate_outcome(
        self,
        event: Event,
        grounding: "GroundingResult",
        projection: "SemanticProjection",
        world: WorldState,
    ) -> Optional[str]:
        """
        Produce optional flavor text for a Zone 1/2 outcome.

        Returns None for mundane actions (move, wait) to keep the output clean.
        Only enriches social/dramatic events worth narrating beyond the base
        render_event() sentence.
        """
        from .schemas import ActionType, TransitionKind
        skip_types = {ActionType.MOVE, ActionType.WAIT, ActionType.OBSERVE,
                      ActionType.INSPECT}
        if event.action.verb in skip_types:
            return None

        # Speech/yell verbs: the rule-based render_event already produces the
        # right output ("You yells: '...'").  Do NOT hand this to the LM for a
        # rewrite — it discards the actual text and produces generic atmosphere.
        _SPEECH_VERBS = {
            "say", "tell", "ask", "shout", "whisper", "yell", "call",
            "speak", "announce", "mutter", "cry", "declare", "exclaim",
            "reply", "respond", "answer", "greet", "hello", "hi",
            ActionType.SPEAK,
        }
        if event.action.verb in _SPEECH_VERBS:
            return None

        # For verbs that produce only a DIALOGUE_SPOKEN (freeform with text),
        # render_event handles it cleanly — skip LM unless Zone 3.
        has_spoken = any(
            t.kind == TransitionKind.DIALOGUE_SPOKEN and t.payload.get("text")
            for t in event.transitions
        )
        if has_spoken:
            return None

        if self._is_real_adapter():
            return self._lm_narrate_outcome(event, grounding, projection, world)
        return None  # Mock: base render_event() is sufficient

    # ── LM-powered paths ──────────────────────────────────────────────────────

    def _is_real_adapter(self) -> bool:
        """True if the adapter can produce free prose via narrate()."""
        if self._adapter is None:
            return False
        from .lm_adapter import MockLMAdapter
        return not isinstance(self._adapter, MockLMAdapter)

    def _lm_narrate_zone3(
        self,
        original_intent: str,
        orphan_reason: str,
        projection: "SemanticProjection",
        world: WorldState,
    ) -> str:
        prompt = _build_zone3_narrator_prompt(
            original_intent, orphan_reason, projection, world
        )
        # Use narrate() — unconstrained free-text, no SemanticAction schema
        result = self._adapter.narrate(prompt)
        if result and len(result.strip()) > 10:
            return result.strip()
        return self._template_zone3(orphan_reason, world, projection)

    def _lm_narrate_outcome(
        self,
        event: Event,
        grounding: "GroundingResult",
        projection: "SemanticProjection",
        world: WorldState,
    ) -> Optional[str]:
        base = render_event(event, world)
        prompt = _build_outcome_narrator_prompt(base, event, grounding, projection)
        result = self._adapter.narrate(prompt)
        if result and len(result.strip()) > len(base):
            return result.strip()
        return None

    # ── Template fallback for Zone 3 ─────────────────────────────────────────

    def _template_zone3(
        self,
        orphan_reason: str,
        world: WorldState,
        projection: "SemanticProjection",
    ) -> str:
        reason_lower = orphan_reason.lower()
        for keyword, template in self._ZONE3_TEMPLATES:
            if keyword in reason_lower:
                return template + _ambient_suffix(projection)
        return self._ZONE3_DEFAULT + _ambient_suffix(projection)


# ── Narrator prompt builders ─────────────────────────────────────────────────


def _build_zone3_narrator_prompt(
    original_intent: str,
    orphan_reason: str,
    projection: "SemanticProjection",
    world: WorldState,
) -> str:
    env = projection.environment
    entity_names = [e.name for e in projection.visible_entities[:3]]
    present = ", ".join(entity_names) if entity_names else "no one else"
    return (
        f"NARRATOR ROLE: You are writing 1-3 sentences of atmospheric prose "
        f"for a simulation game. The player attempted: '{original_intent}'. "
        f"The simulation cannot resolve this because: {orphan_reason}. "
        f"Setting: {env.location_name}. Present: {present}. "
        f"Noise: {env.noise_level}. "
        f"Write a brief, grounded response — what the player experiences "
        f"when the world offers no reaction to what they tried. "
        f"Do not invent simulation facts. Do not resolve the action. "
        f"Convey that the moment passed without mechanical consequence, "
        f"but with atmosphere. 1-3 sentences only."
    )


def _build_outcome_narrator_prompt(
    base_narration: str,
    event: Event,
    grounding: "GroundingResult",
    projection: "SemanticProjection",
) -> str:
    from .schemas import ActionZone
    zone_note = ""
    if grounding.zone == ActionZone.EXTENDING and grounding.asserted_fact:
        zone_note = (
            f"The player was asserting: '{grounding.asserted_fact}'. "
        )
    return (
        f"NARRATOR ROLE: Rewrite this game event with more atmospheric detail. "
        f"Keep it to 1-2 sentences. Do not invent new facts. "
        f"{zone_note}"
        f"Original: {base_narration}"
    )


def _ambient_suffix(projection: "SemanticProjection") -> str:
    """Append a brief environmental detail if the projection has one."""
    env = projection.environment
    if env.crowded:
        return " The crowd around you continues without pause."
    if env.noise_level == "loud":
        return " The noise swallows the moment completely."
    if projection.visible_entities:
        name = projection.visible_entities[0].name
        return f" {name} doesn't react."
    return ""


# ---------------------------------------------------------------------------
# Ambient affordance hints — physics-grounded cues woven into narration
# ---------------------------------------------------------------------------


def ambient_affordance_hints(
    world: WorldState,
    focal_entity_id: EntityId,
    *,
    max_hints: int = 3,
) -> list[str]:
    """
    Return short sensory cues that telegraph physics affordances to the player.

    These are NOT explicit instructions ("you can light this on fire").
    They are DM-style hints: *"The bar top is slick with spilled oil."*
    They call attention to features that have mechanical relevance without
    breaking immersion.

    Returned strings are already formatted as prose fragments suitable for
    appending to a room description.
    """
    grid = world.spatial
    focal = grid.entities.get(focal_entity_id)
    if focal is None:
        return []

    hints: list[str] = []
    phys = world.config.physics_config

    def _mat_tags(obj: "ObjectState") -> set[str]:
        tags = set(obj.tags)
        if phys:
            mat_name = obj.meta.get("material", "")
            mat = phys.materials.get(mat_name)
            if mat:
                tags.update(mat.tags)
        return tags

    px, py = focal.position.x, focal.position.y

    for obj in grid.objects.values():
        if obj.position is None:
            continue
        dist = abs(obj.position.x - px) + abs(obj.position.y - py)
        if dist > 6:
            continue
        mtags = _mat_tags(obj)

        if "oil" in mtags or "oil" in obj.tags:
            if "on_fire" not in obj.tags:
                hints.append(
                    f"The {obj.name} nearby reeks faintly of lamp oil — "
                    f"{'it has spilled across the floor' if obj.position else 'ready to spill'}."
                )
        elif "flammable" in mtags and "on_fire" not in obj.tags:
            hints.append(f"The {obj.name} is dry {'wood' if 'wooden' in mtags else 'tinder'} — it would catch easily.")
        if "on_fire" in obj.tags:
            hints.append(f"The {obj.name} is ablaze nearby, radiating fierce heat.")
        if "fire_suppressant" in mtags and "liquid" in mtags:
            hints.append(f"A pool of {obj.name.lower()} sits on the floor — enough to douse a small fire.")
        if "weapon" in obj.tags and obj.owner is None:
            hints.append(f"The {obj.name} lies within reach — unattended.")
        if "fragile" in mtags or "glass" in mtags:
            hints.append(f"The {obj.name} looks fragile — one good hit would shatter it.")

        if len(hints) >= max_hints:
            break

    # Tile marks near focal entity can also be hints
    from .schemas import Coord
    nearby_tiles_with_marks = []
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        tile = grid.tile_at(Coord(x=px + dx, y=py + dy))
        if tile.marks:
            nearby_tiles_with_marks.extend(tile.marks[-1:])
    for mark in nearby_tiles_with_marks[:1]:
        if mark not in ("dented", "barricaded"):
            hints.append(f"There are marks on the floor here: {mark}.")

    return hints[:max_hints]


# ---------------------------------------------------------------------------
# Optional LM enrichment (downstream, non-authoritative)
# ---------------------------------------------------------------------------


def enrich_with_lm(
    base_narration: str,
    event: Event,
    world: WorldState,
    adapter,  # LMAdapter — imported lazily to avoid circular import
) -> str:
    """
    Optionally enrich base_narration with LM-generated flavor text.

    This function is EXPLICITLY OPTIONAL and is never called by the game loop.
    It writes only to the returned string — it does NOT touch WorldState.

    The LM receives only the base narration and a minimal event description,
    not a projection or WorldState.
    """
    prompt = (
        f"Rewrite this game event description with more atmospheric detail. "
        f"Keep it to 1-2 sentences. Do not invent facts.\n\n"
        f"Original: {base_narration}"
    )
    try:
        from .schemas import SemanticProjection
        minimal_proj = SemanticProjection(
            focal_entity=event.action.actor,
            tick=event.tick,
        )
        # Use WAIT intent as a carrier for the narration prompt
        from .schemas import ActionType, SemanticAction
        dummy_action = SemanticAction(
            verb=ActionType.WAIT,
            actor=event.action.actor,
            raw_input=prompt,
        )
        result = adapter.infer(minimal_proj, prompt)
        # Only use the raw_input echo — we don't apply the action
        return result.raw_input or base_narration
    except Exception:
        return base_narration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ASCII map renderer
# ---------------------------------------------------------------------------

# Symbol table — drawn on the map
_TERRAIN_SYMBOLS = {
    "floor":        ".",
    "wall":         "#",
    "door_open":    "/",
    "door_closed":  "+",
    "window":       "=",
    "water":        "~",
    "stairs_up":    "<",
    "stairs_down":  ">",
}

_LEGEND = [
    ("@", "You (player)"),
    ("A-Z", "NPC  (first letter of name)"),
    ("C", "Creature / unknown"),
    ("%", "Corpse"),
    ("$", "Object on ground"),
    (".", "Floor"),
    ("#", "Wall"),
    ("+", "Door (closed)"),
    ("/", "Door (open)"),
    ("~", "Water"),
    ("<", "Stairs up"),
    (">", "Stairs down"),
    ("=", "Window"),
    ("·", "Out of sight"),
]


def render_map(
    world: WorldState,
    focal_entity_id: Optional[EntityId] = None,
    *,
    legend: bool = True,
) -> str:
    """
    Render the spatial grid as an ASCII map.

    The focal entity (player) is shown as @.  Other entities are shown by
    the first letter of their name.  Objects on the ground are shown as $.
    Cells outside the focal entity's sight range are shown as ·.

    Parameters
    ----------
    legend : Include the LEGEND sidebar to the right of the grid (default
             True).  Pass ``legend=False`` when the legend is rendered
             separately (e.g. in the TUI relations panel).

    Returns a multi-line string ready to print.
    """
    grid = world.spatial

    # Determine which cells are visible to the focal entity
    visible: Optional[set] = None
    focal_pos: Optional[Coord] = None
    focal_facing: Optional[str] = None
    if focal_entity_id is not None:
        entity = grid.entities.get(focal_entity_id)
        if entity is not None:
            focal_pos = entity.position
            focal_facing = entity.facing.value
            from .spatial import visible_from
            visible = visible_from(
                grid, focal_pos, entity.sight_range,
                facing=entity.facing, fov_degrees=entity.fov_degrees,
            )

    # Build entity position index — skip dead entities
    entity_at: dict[str, EntityState] = {}
    for e in grid.entities.values():
        if e.alive:
            entity_at[f"{e.position.x},{e.position.y}"] = e

    # Build object position index (ground only)
    object_at: set[str] = set()
    for o in grid.objects.values():
        if o.position is not None:
            object_at.add(f"{o.position.x},{o.position.y}")

    # Build corpse position index — dead entities shown as %
    corpse_at: set[str] = set()
    for e in grid.entities.values():
        if not e.alive:
            corpse_at.add(f"{e.position.x},{e.position.y}")

    lines: list[str] = []

    # Column ruler (every 5 columns)
    ruler_top = "    "
    for x in range(grid.width):
        ruler_top += str(x // 10) if x % 10 == 0 else (" " if x % 5 != 0 else str(x % 10))
    lines.append(ruler_top)
    ruler_bot = "    "
    for x in range(grid.width):
        ruler_bot += str(x % 10) if x % 5 == 0 else " "
    lines.append(ruler_bot)

    for y in range(grid.height):
        row = f"{y:>3} "
        for x in range(grid.width):
            coord = Coord(x=x, y=y)
            key = f"{x},{y}"

            # Fog of war — use a middle dot so the grid shape is always visible
            if visible is not None and coord not in visible:
                row += "·"
                continue

            # Corpse
            if key in corpse_at and key not in entity_at:
                row += "%"
                continue

            # Entity takes priority
            if key in entity_at:
                e = entity_at[key]
                if e.entity_id == focal_entity_id:
                    row += "@"
                elif e.kind.value == "player":
                    row += "@"
                elif e.kind.value in ("npc", "ambient"):
                    # Use the first letter of the NPC's name so multiple
                    # NPCs are distinguishable on the map.
                    row += (e.name[0].upper() if e.name else "N")
                else:
                    row += "C"
                continue


            # Object on ground
            if key in object_at:
                row += "$"
                continue

            # Terrain
            tile = grid.tile_at(coord)
            row += _TERRAIN_SYMBOLS.get(tile.terrain.value, "?")

        lines.append(row)

    # Compass / facing indicator below the grid
    if focal_facing:
        _compass_arrows = {
            "north": "↑", "northeast": "↗", "east": "→",
            "southeast": "↘", "south": "↓", "southwest": "↙",
            "west": "←", "northwest": "↖",
        }
        arrow = _compass_arrows.get(focal_facing, "?")
        lines.append(f"    facing {focal_facing} {arrow}  (fov {int(entity.fov_degrees)}°)")

    if not legend:
        return "\n".join(lines)

    # Interleave map and legend side by side
    legend_lines = ["", "  LEGEND"]
    for symbol, description in _LEGEND:
        legend_lines.append(f"   {symbol}  {description}")
    legend_lines.append("")

    max_map_width = max(len(l) for l in lines)
    padding = 4
    output_lines = []
    for i, map_line in enumerate(lines):
        pad = " " * (max_map_width - len(map_line) + padding)
        legend_part = legend_lines[i] if i < len(legend_lines) else ""
        output_lines.append(map_line + pad + legend_part)
    for j in range(len(lines), len(legend_lines)):
        output_lines.append(" " * (max_map_width + padding) + legend_lines[j])

    return "\n".join(output_lines)


def _narrate_spell(event: Event, world: WorldState, actor_name: str) -> str:
    """
    Produce a rich narrative for a CAST event.

    Reads the spell program source from the action, then describes each
    significant transition that resulted.
    """
    grid = world.spatial

    def ename(eid_str: str) -> str:
        e = grid.entities.get(EntityId(eid_str))
        return e.name if e else eid_str

    # Identify the program / spell name
    source = ""
    spell_name = ""
    if event.action.intent:
        source = event.action.intent.rationale or ""
        spell_name = event.action.intent.manner or ""

    # Collect effect summaries from the transitions
    effects: list[str] = []
    targets_affected: set[str] = set()

    for t in event.transitions:
        p = t.payload

        if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED:
            delta = p.get("delta", 0)
            cause = p.get("cause", "")
            if "spell" not in cause:
                continue
            eid = p.get("entity_id", "")
            tname = ename(eid)
            targets_affected.add(tname)
            # Check for associated property changes
            prop_delta = p.get("prop_delta", {})
            prop_set = p.get("prop_set", {})
            mana_delta = p.get("mana_delta")

            if mana_delta and delta == 0 and not prop_delta and not prop_set:
                continue  # skip pure mana-cost transition
            if delta != 0:
                if delta < 0:
                    effects.append(
                        f"{tname} takes {abs(delta)} magical damage"
                        + (f" (health: {grid.entities[EntityId(eid)].health}"
                           if EntityId(eid) in grid.entities else "")
                        + ")"
                    )
                else:
                    effects.append(f"{tname} is healed for {delta}")
            for prop, dv in prop_delta.items():
                entity = grid.entities.get(EntityId(eid))
                new_val = entity.meta.get(prop, "?") if entity else "?"
                targets_affected.add(tname)
                if prop == "temperature":
                    from .substance import get_temperature, resolve_material

                    cfg = world.config.physics_config
                    ent = grid.entities.get(EntityId(eid))
                    mat_name = ""
                    if ent and cfg:
                        mat = resolve_material(ent, cfg)
                        mat_name = f" ({mat.display})" if mat else ""
                        new_val = get_temperature(ent, cfg)
                    else:
                        new_val = p.get("prop_delta", {}).get("temperature", dv)
                    if dv > 0:
                        effects.append(
                            f"{tname}{mat_name} temperature +{dv:.0f}°C"
                            f" (now {new_val:.0f}°C)"
                            if isinstance(new_val, (int, float))
                            else f"{tname} heats up"
                        )
                    else:
                        effects.append(
                            f"{tname}{mat_name} temperature {dv:.0f}°C"
                            f" (now {new_val:.0f}°C)"
                            if isinstance(new_val, (int, float))
                            else f"{tname} chills"
                        )
                else:
                    effects.append(f"{tname}'s {prop} changes by {dv:+.0f}")

        elif t.kind == TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED:
            cause = p.get("cause", "")
            if "spell" not in cause:
                continue
            eid = p.get("entity_id", "")
            tname = ename(eid)
            targets_affected.add(tname)
            new_state = p.get("to", "?")
            _emotion_desc = {
                "fearful": "is overcome with dread",
                "friendly": "warms with sudden goodwill",
                "angry": "flares with sudden rage",
                "neutral": "settles into calm",
                "hostile": "eyes you with hostility",
            }
            desc = _emotion_desc.get(new_state, f"becomes {new_state}")
            effects.append(f"{tname} {desc}")

        elif t.kind == TransitionKind.ENTITY_CONDITION_CHANGED:
            eid = p.get("entity_id", "")
            tname = ename(eid)
            cond = p.get("condition", "")
            ticks = p.get("ticks", 0)
            tag_add = p.get("_tag_add")
            tag_rem = p.get("_tag_remove")
            if tag_add:
                targets_affected.add(tname)
                effects.append(f"{tname} gains the '{tag_add}' quality")
            elif tag_rem:
                targets_affected.add(tname)
                effects.append(f"{tname} loses the '{tag_rem}' quality")
            elif cond and not cond.startswith("spell_") and ticks > 0:
                targets_affected.add(tname)
                effects.append(f"{tname} is {cond} for {ticks} tick(s)")

        elif t.kind == TransitionKind.ENTITY_MOVED:
            eid = p.get("entity_id", "")
            tname = ename(eid)
            if eid != str(event.action.actor):  # don't narrate self-movement from push
                targets_affected.add(tname)
                to_c = p.get("to", {})
                effects.append(
                    f"{tname} is flung to ({to_c.get('x')}, {to_c.get('y')})"
                )

        elif t.kind == TransitionKind.ENTITY_DIED:
            eid = p.get("entity_id", "")
            tname = ename(eid)
            effects.append(f"{tname} collapses and dies")

    # Find mana cost
    mana_cost = 0
    for t in event.transitions:
        if t.kind == TransitionKind.DIALOGUE_SPOKEN:
            mc = t.payload.get("mana_cost")
            if mc:
                mana_cost = int(mc)
                break
        if t.kind == TransitionKind.ENTITY_HEALTH_CHANGED:
            md = t.payload.get("mana_delta")
            if md:
                mana_cost = abs(int(md))

    # Prefer the spell program (rationale) over the inscribed nickname (manner).
    program = (source or "").strip().replace("\n", ", ")
    display_name = program or spell_name or ""

    if not effects:
        return (
            f"{actor_name}: {display_name}"
            if display_name
            else f"{actor_name} casts a spell"
        ) + (
            " — no visible effect."
            + (f" (Mana: −{mana_cost})" if mana_cost else "")
        )

    effects_text = "; ".join(effects[:5])  # cap at 5 effects for readability
    if len(effects) > 5:
        effects_text += f" (…and {len(effects) - 5} more effects)"

    if display_name:
        return (
            f"{actor_name}: {display_name} → {effects_text}."
            + (f" (Mana: −{mana_cost})" if mana_cost else "")
        )
    return (
        f"{actor_name} casts. {effects_text}."
        + (f" (Mana: −{mana_cost})" if mana_cost else "")
    )


def _primary_transition(event: Event):
    """Return the first 'interesting' transition in the event, or None."""
    priority = [
        TransitionKind.ENTITY_DIED,
        TransitionKind.ENTITY_HEALTH_CHANGED,
        TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
        TransitionKind.ENTITY_MOVED,
        TransitionKind.ITEM_TRANSFERRED,
        TransitionKind.ITEM_EQUIPPED,
        TransitionKind.ITEM_UNEQUIPPED,
        TransitionKind.ITEM_STORED,
        TransitionKind.ITEM_RETRIEVED,
        TransitionKind.TILE_CHANGED,
        TransitionKind.DIALOGUE_SPOKEN,
        TransitionKind.REGION_TRANSIT,
        TransitionKind.ENTITY_TURNED,
    ]
    for kind in priority:
        for t in event.transitions:
            if t.kind == kind:
                return t
    return event.transitions[0] if event.transitions else None


def _mark_verb(verb: str) -> str:
    """Return a past-tense writing verb appropriate for the action word."""
    table = {
        "carve": "carves", "scratch": "scratches",
        "draw": "draws", "write": "writes", "inscribe": "inscribes",
        "mark": "marks",
    }
    return table.get(str(verb).lower(), "marks")


def _target_name(event: Event, grid) -> str:
    """Entity target name only (legacy callers)."""
    return _target_display_name(event, grid)


def _target_display_name(event: Event, grid) -> str:
    if event.action.target is None:
        return "the air"
    if isinstance(event.action.target, Coord):
        return f"({event.action.target.x}, {event.action.target.y})"
    raw = str(event.action.target)
    entity = grid.entities.get(EntityId(raw))
    if entity is not None:
        return entity.name
    obj = grid.objects.get(ObjectId(raw))
    if obj is not None:
        return obj.name
    return raw


def _format_pack_narrative(
    template: str,
    *,
    actor_name: str,
    target_name: str,
    object_name: str,
) -> str:
    return (
        template.replace("{actor}", actor_name)
        .replace("{target}", target_name or "someone")
        .replace("{object}", object_name or "it")
    )


def _pack_verb_narrative(
    world: WorldState,
    verb: str,
    contest_outcome: Optional[str],
    event: Event,
    grid,
) -> Optional[str]:
    tmpl = world.config.verb_templates.get(verb)
    if tmpl is None:
        return None
    actor = grid.entities.get(event.action.actor)
    actor_name = actor.name if actor else "Someone"
    target_name = _target_display_name(event, grid)
    object_name = target_name
    target_is_object = False
    if event.action.target is not None:
        obj = grid.objects.get(ObjectId(str(event.action.target)))
        if obj is not None:
            object_name = obj.name
            target_is_object = grid.entities.get(EntityId(str(event.action.target))) is None
            if target_is_object:
                target_name = ""

    if contest_outcome == "failure" and tmpl.narrative_failure:
        text = tmpl.narrative_failure
    elif target_is_object and tmpl.narrative_object_success:
        text = tmpl.narrative_object_success
    elif tmpl.narrative_success:
        text = tmpl.narrative_success
    else:
        return None
    return _format_pack_narrative(
        text,
        actor_name=actor_name,
        target_name=target_name,
        object_name=object_name,
    )
