"""
LM-driven NPC policy — one path: infer_npc → compile (with rejection retry).

No menu picks, no voice-line fallbacks, no secondary speech enrichment module.
Invalid utterances are rejected by the compiler; the LM sees the rejection
on the next infer attempt via projection.last_action_rejection.
"""

from __future__ import annotations

import difflib
import logging
from typing import TYPE_CHECKING, Optional

from .npc_policy import ReactivePolicy, build_npc_character_sheet
from .schemas import (
    ActionType,
    AlertnessLevel,
    EntityState,
    MalformedActionError,
    SemanticAction,
    WorldState,
    TransitionProposal,
    TransitionKind,
)
from .run_metrics import bump, record_mdp_choice
from .verb_targeting import action_missing_required_target

if TYPE_CHECKING:
    from .lm_adapter import LMAdapter
    from .npc_policy import NpcPolicy

logger = logging.getLogger(__name__)

GENERIC_META_VERBS: frozenset[str] = frozenset({
    "wait", "observe", "look", "interact", "continue",
    "find", "find_out", "investigate", "set_priority", "add_goals",
    "add_to_pending_tasks", "settle_room", "mark_tile", "poured",
})

_SOCIAL_SPEECH_VERBS: frozenset[str] = frozenset({
    "speak", "say", "tell", "ask", "shout", "whisper", "greet", "barter",
    "haggle", "gossip", "tease", "flirt", "compliment", "threaten", "warn",
    "challenge", "apologize", "thank", "console", "accuse", "mock", "joke",
    "promise", "lie", "agree", "disagree", "teach", "learn", "examine",
})


def _normalize_verb(verb: object) -> str:
    return str(verb).lower().split(".")[-1]


def _known_verbs(world: WorldState) -> list[str]:
    from .speech_utils import SPEECH_LIKE_VERBS

    verbs = {_normalize_verb(v) for v in world.config.verb_templates}
    for val in vars(ActionType).values():
        if isinstance(val, str) and not val.startswith("_"):
            verbs.add(_normalize_verb(val))
    verbs.update(SPEECH_LIKE_VERBS)
    verbs.update({
        "move", "attack", "flee", "observe", "wait", "rest", "turn",
        "examine", "inspect", "pickup", "drop", "use", "cast",
        "interact",  # keep meta verb stable so GENERIC_META_VERBS check works
    })
    return sorted(verbs)


def repair_npc_action(action: SemanticAction, world: WorldState) -> SemanticAction:
    """Map near-miss LM verbs to pack templates / core verbs (flexible decode)."""
    verb = _normalize_verb(action.verb)
    known = _known_verbs(world)
    if verb not in known:
        match = difflib.get_close_matches(verb, known, n=1, cutoff=0.72)
        if match:
            return action.model_copy(update={"verb": match[0]})
    return action


def _action_hint_text(action: SemanticAction) -> str:
    parts = [str(action.verb), str(action.raw_input or "")]
    if action.target is not None:
        parts.append(str(action.target))
    intent = action.intent
    if intent:
        for value in (intent.rationale, intent.manner):
            if value:
                parts.append(str(value))
        parts.extend(str(v) for v in (intent.desired_outcome or []))
    return " ".join(parts).lower()


def _overlap_score(a: str, b: str) -> float:
    toks_a = {
        t.strip(".,:;!?()[]{}\"'").lower()
        for t in a.split()
        if len(t.strip(".,:;!?()[]{}\"'")) >= 4
    }
    toks_b = {
        t.strip(".,:;!?()[]{}\"'").lower()
        for t in b.split()
        if len(t.strip(".,:;!?()[]{}\"'")) >= 4
    }
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(1, len(toks_a | toks_b))


def _format_candidate_for_lm(
    idx: int,
    entity: EntityState,
    world: WorldState,
    action: SemanticAction,
    label: str,
    score: float,
) -> str:
    target = action.target
    if isinstance(target, str) and target in world.spatial.entities:
        target_text = world.spatial.entities[target].name
    elif target is None:
        target_text = "none"
    else:
        target_text = str(target)
    hint = _action_hint_text(action)[:220]
    return (
        f"{idx}. {label}: verb={action.verb}, target={target_text}, "
        f"engine_score={score:.2f}, hint={hint}"
    )


class LMNpcPolicy:
    """
    Each tick: project → character sheet → ``infer_npc`` (JSON schema) → compile.

    Set ``cognition.mode`` to ``reactive_only`` for deterministic tests only.
    """

    def __init__(
        self,
        adapter: "LMAdapter",
        fallback: Optional["NpcPolicy"] = None,
        cognition_mode: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.adapter = adapter
        self.fallback: ReactivePolicy = fallback or ReactivePolicy()  # type: ignore[assignment]
        self.cognition_mode = cognition_mode
        self.model_name = model_name or getattr(adapter, "model", "") or ""

    @staticmethod
    def uses_lm_layer(world: WorldState) -> bool:
        return world.config.npc_policy.cognition.mode != "reactive_only"

    uses_infer_layer = uses_lm_layer

    def _effective_mode(self, world: WorldState) -> str:
        mode = self.cognition_mode or world.config.npc_policy.cognition.mode
        return {"menu_lm": "lm", "tiered": "lm", "infer_allowed": "lm"}.get(mode, mode)

    def decide(self, entity: EntityState, world: WorldState) -> SemanticAction:
        from .director import NarrativeDirector
        from .lm_adapter import MockLMAdapter
        from .npc_reflection import run_npc_reflection

        if world.meta.get("magic_duel"):
            action = self.fallback.decide_magic_duel(entity, world)
            entity.meta["last_policy_branch"] = "magic_duel_reactive"
            return action

        director = NarrativeDirector(world)
        brief = director.brief_for_npc(entity)
        entity.meta["director_hint"] = director.format_hint_block(brief)

        mode = self._effective_mode(world)
        if mode == "reactive_only" or isinstance(self.adapter, MockLMAdapter):
            action = self.fallback.decide(entity, world)
            entity.meta["last_policy_branch"] = (
                "reactive_only" if mode == "reactive_only" else "mock_reactive"
            )
            return action

        # Short-circuit when the adapter has explicitly signalled that it
        # is not ready (e.g. torch model failed to download / warmup).
        # Without this we'd call ``infer_npc`` once per NPC per tick, eat
        # an exception, log a warning, then fall back to reactive anyway —
        # 4 NPCs × N ticks of warning spam.  See
        # ``logs/autonomous_debug_20260522_055425.log`` for the failure
        # mode this guards against.
        if getattr(self.adapter, "is_ready", True) is False:
            action = self.fallback.decide(entity, world)
            entity.meta["last_policy_branch"] = "adapter_not_ready_reactive"
            return action

        run_npc_reflection(entity, world, self.adapter)
        action = self._decide_via_infer(entity, world)
        if action is None:
            reason = entity.meta.get("last_rejection") or "infer_npc failed"
            logger.warning(
                "NPC %s infer failed (%s) — falling back to reactive policy",
                entity.name,
                reason,
            )
            action = self.fallback.decide(entity, world)
            entity.meta["last_policy_branch"] = "infer_failed_reactive"
            return self._clamp_actor(entity, action)
        
        # If _decide_via_infer did not set a specific branch, set the default
        if not entity.meta.get("last_policy_branch") or entity.meta.get("last_policy_branch") in ("infer_npc", "mock_reactive"):
            entity.meta["last_policy_branch"] = "infer_npc"
        return self._clamp_actor(entity, action)

    @staticmethod
    def _clamp_actor(entity: EntityState, action: SemanticAction) -> SemanticAction:
        if action.actor != entity.entity_id:
            return action.model_copy(update={"actor": entity.entity_id})
        return action

    def _decide_via_infer(
        self, entity: EntityState, world: WorldState, *, max_attempts: int = 3
    ) -> Optional[SemanticAction]:
        from .compiler import compile_action
        from .npc_mind import interpret_mind
        from .projection import project

        interpret_mind(entity, world)
        
        try:
            proj = project(
                world,
                entity.entity_id,
                last_action_rejection=None,
            )
            sheet = build_npc_character_sheet(entity, world)

            # Generate up to 40 candidates from the fallback reactive policy
            try:
                candidates = self.fallback.generate_candidates(entity, world, max_candidates=40)
            except Exception as exc:
                logger.debug("Failed to generate fallback candidates: %s", exc)
                candidates = []

            # Compile and pre-validate options to form the Forest of Valid transitions
            valid_options: list[tuple[SemanticAction, str, float]] = []
            for action, label, weight in candidates:
                action = self._clamp_actor(entity, action)
                res = compile_action(action, world)
                if res.valid:
                    valid_options.append((action, label, weight))

            if not valid_options:
                logger.debug("No valid options available for %s", entity.name)
                return None

            if not callable(getattr(self.adapter, "infer_npc_mdp", None)):
                return None

            import random

            shuffled_options = list(valid_options)
            _seed = hash((world.tick, entity.entity_id, len(valid_options)))
            _rng = random.Random(_seed)
            _rng.shuffle(shuffled_options)

            options_text: list[str] = []
            option_indices: list[int] = []
            for idx, (action, label, _weight) in enumerate(shuffled_options, 1):
                target_str = f", target={action.target}" if action.target else ""
                options_text.append(
                    f"Index {idx}: {label} (verb={action.verb}{target_str})"
                )
                option_indices.append(idx)

            entity.meta["last_mdp_menu"] = " | ".join(options_text[:12])
            logger.debug(
                "[%s] MDP menu (tick=%s, %d options): %s",
                entity.name,
                world.tick,
                len(options_text),
                entity.meta["last_mdp_menu"],
            )

            selection: dict = {}
            selected_idx = 0
            rejected_indices: list[int] = []
            accepted = False

            for attempt in range(max_attempts):
                try:
                    modified_options = [
                        f"{opt} -- [REJECTED - DO NOT CHOOSE]"
                        if o_idx in rejected_indices
                        else opt
                        for o_idx, opt in enumerate(options_text, 1)
                    ]
                    rejection_hint = entity.meta.get("last_rejection")
                    if rejection_hint and attempt > 0:
                        modified_options.insert(
                            0,
                            f"NOTE: Previous choice was rejected: {rejection_hint}",
                        )
                    selection = self.adapter.infer_npc_mdp(
                        sheet, proj, modified_options, option_indices
                    )
                except NotImplementedError:
                    logger.debug("infer_npc_mdp not implemented for %s", entity.name)
                    return None
                except MalformedActionError:
                    return None
                except Exception as exc:
                    logger.debug(
                        "infer_npc_mdp failed (attempt %d) for %s: %s",
                        attempt + 1,
                        entity.name,
                        exc,
                    )
                    bump(world, "mdp_parse_failures")
                    selection = {}

                selected_idx = int(selection.get("selected_option_index", 0))

                if not (0 < selected_idx <= len(shuffled_options)):
                    bump(world, "mdp_out_of_range")
                    continue
                if selected_idx in rejected_indices:
                    continue

                proposed_action, label, weight = shuffled_options[selected_idx - 1]

                if proposed_action.target == entity.entity_id:
                    rejected_indices.append(selected_idx)
                    bump(world, "mdp_reject_self_target")
                    continue

                if action_missing_required_target(proposed_action, world):
                    logger.debug(
                        "Rejected targetless %s for %s (verb requires entity target)",
                        label,
                        entity.name,
                    )
                    rejected_indices.append(selected_idx)
                    bump(world, "mdp_reject_missing_target")
                    continue

                compile_res = compile_action(proposed_action, world)
                if not compile_res.valid:
                    reason = compile_res.rejection_reason
                    detail = compile_res.rejection_detail or ""
                    entity.meta["last_rejection"] = (
                        f"{reason}: {detail}" if reason else detail
                    )
                    rejected_indices.append(selected_idx)
                    bump(world, "mdp_reject_compile")
                    continue

                is_ser_aldric = entity.name.lower() == "ser aldric"
                hp = entity.stats.get("hp", 100.0) if entity.stats else 100.0
                if (
                    is_ser_aldric
                    and str(proposed_action.verb).lower() == "flee"
                    and hp >= 25.0
                ):
                    rejected_indices.append(selected_idx)
                    bump(world, "mdp_reject_persona")
                    continue

                is_mira = entity.name.lower() == "mira"
                if (
                    is_mira
                    and str(proposed_action.verb).lower() == "attack"
                    and entity.alertness != AlertnessLevel.COMBAT
                ):
                    rejected_indices.append(selected_idx)
                    bump(world, "mdp_reject_persona")
                    continue

                accepted = True
                break

            if accepted and 0 < selected_idx <= len(shuffled_options):
                action, label, weight = shuffled_options[selected_idx - 1]
            else:
                fallback_idx = _rng.randint(1, len(shuffled_options))
                action, label, weight = shuffled_options[fallback_idx - 1]
                selected_idx = fallback_idx
                bump(world, "mdp_fallback_random")
                logger.debug(
                    "MDP fallback random index %s for %s: %s",
                    selected_idx,
                    entity.name,
                    label,
                )

            record_mdp_choice(
                world,
                entity_name=entity.name,
                selected_index=selected_idx,
                menu_size=len(shuffled_options),
                verb=str(action.verb),
            )
            entity.meta["last_mdp_selected_index"] = selected_idx

            custom_speech = (selection.get("custom_speech_line") or "").strip()
            verb_norm = str(action.verb).lower().split(".")[-1]
            wants_speech = (
                verb_norm in _SOCIAL_SPEECH_VERBS
                or " with " in label.lower()
            )
            if not custom_speech and wants_speech and hasattr(
                self.adapter, "infer_npc_speech"
            ):
                try:
                    custom_speech = self.adapter.infer_npc_speech(
                        sheet, proj, label, verb_norm
                    )
                    if custom_speech:
                        bump(world, "open_ended_speech_calls")
                        entity.meta["last_policy_branch"] = (
                            f"infer_npc_mdp_option_{selected_idx}+speech"
                        )
                except Exception as exc:
                    logger.debug("infer_npc_speech failed for %s: %s", entity.name, exc)

            if custom_speech:
                action.proposed_effects = list(action.proposed_effects or [])
                action.proposed_effects.append(
                    TransitionProposal(
                        kind=TransitionKind.DIALOGUE_SPOKEN.value,
                        payload={
                            "actor": str(action.actor),
                            "target": str(action.target) if action.target else None,
                            "text": custom_speech,
                            "loud": True,
                        },
                    )
                )

            if "speech" not in (entity.meta.get("last_policy_branch") or ""):
                entity.meta["last_policy_branch"] = f"infer_npc_mdp_option_{selected_idx}"
            entity.meta.pop("last_rejection", None)
            return action

        except Exception as exc:
            logger.warning("Error in _decide_via_infer for %s: %s", entity.name, exc)
            return None

    def _choose_interesting_valid_action(
        self,
        entity: EntityState,
        world: WorldState,
        lm_action: SemanticAction,
        *,
        lm_weight: float,
    ) -> SemanticAction:
        """
        Let the LM propose, but let the engine prefer a more consequential
        valid action when the proposal is passive or repetitive.
        """
        from .compiler import compile_action
        from .npc_mind import PASSIVE_VERBS, _verb_of, pick_best_action, score_action

        lm_verb = _verb_of(lm_action)
        lm_hint = _action_hint_text(lm_action)
        generic_meta = lm_verb in GENERIC_META_VERBS
        candidates: list[tuple[SemanticAction, str, float]] = [
            (
                lm_action,
                "lm proposal",
                lm_weight - (0.35 if generic_meta else 0.0),
            )
        ]

        try:
            fallback_candidates = self.fallback.generate_candidates(
                entity, world, max_candidates=10
            )
        except Exception:
            fallback_candidates = []

        for action, label, weight in fallback_candidates:
            action = self._clamp_actor(entity, action)
            if compile_action(action, world).valid:
                label_text = f"{label} {_action_hint_text(action)}".lower()
                translated_bonus = 0.0
                if generic_meta:
                    translated_bonus += 0.18
                    translated_bonus += min(0.25, _overlap_score(lm_hint, label_text))
                    if lm_verb in ("interact", "settle_room") and str(action.verb) in (
                        "speak", "ask", "gossip", "toast", "pour_drink",
                    ):
                        translated_bonus += 0.12
                    if lm_verb in ("find", "find_out", "investigate") and str(action.verb) in (
                        "examine", "inspect", "eavesdrop", "speak",
                    ):
                        translated_bonus += 0.12
                    if lm_verb in ("set_priority", "add_goals", "add_to_pending_tasks"):
                        if str(action.verb) in ("speak", "move", "ask"):
                            translated_bonus += 0.16
                candidates.append((
                    action,
                    f"engine translation: {label}" if generic_meta else f"engine candidate: {label}",
                    weight + translated_bonus,
                ))

        if len(candidates) == 1:
            return lm_action

        lm_pick = None
        if generic_meta or lm_verb in PASSIVE_VERBS:
            scored_options: list[tuple[SemanticAction, str, float]] = []
            for cand_action, cand_label, cand_weight in candidates[:8]:
                scored_options.append((
                    cand_action,
                    cand_label,
                    score_action(entity, world, cand_action, cand_weight),
                ))
            options = "\n".join(
                _format_candidate_for_lm(
                    i,
                    entity,
                    world,
                    cand_action,
                    cand_label,
                    cand_score,
                )
                for i, (cand_action, cand_label, cand_score) in enumerate(scored_options, 1)
            )
            prompt = (
                f"You are scoring possible next actions for {entity.name} in a simulation.\n"
                "Pick the ONE option that best turns the character's intent into a concrete, "
                "valid, scene-changing action. Prefer specific social/physical actions over "
                "generic meta verbs, but keep the character's drive and target.\n\n"
                f"Original LM proposal: verb={lm_action.verb}, hint={lm_hint[:260]}\n\n"
                f"Options:\n{options}\n\n"
                "Reply with only the option number."
            )
            try:
                raw_pick = (self.adapter.choose_npc_option(prompt) or "").strip()
                digits = "".join(ch for ch in raw_pick if ch.isdigit())
                if digits:
                    pick_idx = int(digits) - 1
                    if 0 <= pick_idx < len(scored_options):
                        lm_pick = scored_options[pick_idx][0]
                        entity.meta["lm_engine_score_pick"] = {
                            "option": pick_idx + 1,
                            "from": str(lm_action.verb),
                            "to": str(lm_pick.verb),
                        }
            except Exception:
                lm_pick = None

        chosen = lm_pick or pick_best_action(entity, world, candidates)
        lm_score = score_action(entity, world, lm_action, lm_weight)
        chosen_score = score_action(entity, world, chosen, lm_weight)
        lm_passive = lm_verb in PASSIVE_VERBS
        threshold = 0.02 if generic_meta else 0.08
        if chosen is not lm_action and (lm_passive or generic_meta) and chosen_score >= lm_score + threshold:
            entity.meta["director_override"] = {
                "from": str(lm_action.verb),
                "to": str(chosen.verb),
                "lm_score": round(lm_score, 3),
                "chosen_score": round(chosen_score, 3),
                "translated_generic": generic_meta,
            }
            return chosen
        return lm_action
