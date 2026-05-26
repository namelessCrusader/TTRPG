"""
M5: LM Adapter interface and implementations.

The protocol boundary:
  Input:  SemanticProjection  +  player intent (natural language string)
  Output: SemanticAction

CRITICAL INVARIANTS
  1. Implementations NEVER receive a WorldState argument.
  2. All LM calls are logged (input projection, output action, latency, model).
  3. If the LM response cannot be parsed, MalformedActionError is raised —
     never silently corrupt state.
  4. The mock adapter is deterministic and requires no network access,
     enabling full test coverage of all layers above and below.
"""

from __future__ import annotations

import difflib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..schemas import (
    ActionType,
    AdjudicationResult,
    ConstraintBlock,
    EntityId,
    GroundingResult,
    IntentBlock,
    MalformedActionError,
    NpcCharacterSheet,
    ScheduledEffect,
    ScheduledEffectKind,
    SemanticAction,
    SemanticProjection,
    StyleBlock,
    TransitionProposal,
    ValidationResult,
    WorldFact,
    WorldFactScope,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol definition
# ---------------------------------------------------------------------------


class LMAdapter(ABC):
    """
    Abstract base for all LM adapter implementations.

    Three entry points:
      infer()      — schema-constrained.  Player intent → SemanticAction.
      infer_npc()  — schema-constrained.  NPC character sheet + their
                     projection → SemanticAction. Same compiler pipeline.
                     Subclasses may use a different system prompt than
                     infer(), but the output type and constraint level
                     are identical.
      narrate()    — unconstrained.  Returns a prose string for the
                     Narrator. No schema enforcement. None if unsupported.

    The reason infer/infer_npc are separate methods (not just a flag
    on infer) is so adapters with sharply different prompt-building or
    sampling settings for the two modes can specialize cleanly without
    branching inside the hot path.
    """

    @abstractmethod
    def infer(
        self,
        projection: SemanticProjection,
        player_intent: str,
    ) -> SemanticAction:
        """
        Translate player_intent into a SemanticAction grounded in projection.

        Returns
        -------
        SemanticAction — always a fully-typed struct, never free-form text.

        Raises
        ------
        MalformedActionError if the response cannot be parsed.
        """

    def infer_npc(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
    ) -> SemanticAction:
        """
        Produce an NPC's next SemanticAction.

        Default implementation raises NotImplementedError; concrete
        adapters override. The reason this lives on the base class is
        so the NPC policy layer can depend on the protocol shape
        without caring which adapter is wired in.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement infer_npc()."
        )

    def infer_npc_mdp(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
        options: list[str],
        option_indices: list[int],
    ) -> dict:
        """
        Produce a structured selection over a pre-compiled forest of options.

        Default implementation raises NotImplementedError; concrete adapters override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement infer_npc_mdp()."
        )

    def enrich_ambient_event(
        self,
        ambient_id: str,
        base_narrative: str,
        current_scene_info: dict,
    ) -> dict:
        """
        Produce a dynamic context-aware enrichment of an ambient world event.

        Default implementation raises NotImplementedError; concrete adapters override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement enrich_ambient_event()."
        )

    def narrate(self, prompt: str) -> Optional[str]:
        """
        Free-text generation for the NarratorActor.

        No schema enforcement.  Subclasses that support free generation
        override this.  Default returns None (fall back to templates).

        Parameters
        ----------
        prompt : Full narrator instruction prompt.

        Returns
        -------
        Prose string, or None if this adapter cannot produce free text.
        """
        return None

    def infer_semantic_pressure(self, slice_: "SemanticSlice") -> "list[SemanticDelta]":
        """
        Interpret a bounded event slice into semantic deltas.

        Used by the delayed semantic dynamics pass.  Default raises
        NotImplementedError; real adapters override.  Callers should fall
        back to ``MockSemanticPressureAdapter`` when unsupported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement infer_semantic_pressure()."
        )

    def choose_npc_option(self, prompt: str) -> Optional[str]:
        """
        Pick one numbered option for NPC candidate selection.

        Uses a decision-focused system prompt (not the atmospheric narrator).
        Default delegates to ``narrate`` for backward compatibility.
        """
        return self.narrate(prompt)

    def generate_npc_reply(
        self,
        sheet: NpcCharacterSheet,
        utterance: str,
        projection: SemanticProjection,
        *,
        speaker_name: str = "",
        intent_hint: str = "",
        seed_options: "list[str] | None" = None,
        anti_repeat: str = "",
    ) -> Optional[str]:
        """
        Generate an in-character verbal reply from an NPC to `utterance`.

        Unconstrained prose call — same modality as `narrate()`.  The
        default returns None (NPC stays silent); real adapters override.

        Parameters
        ----------
        sheet     : NPC self-knowledge (role, personality, goals, …)
        utterance : What was just said/done to this NPC.
        projection: NPC's view of visible entities (for context only).

        Returns
        -------
        A 1-2 sentence in-character reply string, or None for silence.
        """
        return None

    def propose_consequences(
        self,
        action: SemanticAction,
        projection: SemanticProjection,
        visible_ids: list[str],
    ) -> list[TransitionProposal]:
        """
        Propose 0-3 world-state consequences for a completed social action.

        Constrained call — output must be a JSON array of TransitionProposal
        objects validated against a tight schema.  The engine clips/filters
        proposals before applying them.  Default returns [] (no proposals).

        Parameters
        ----------
        action      : The SemanticAction that just succeeded.
        projection  : The focal entity's current world view.
        visible_ids : List of visible entity IDs for target validation.

        Returns
        -------
        List of TransitionProposal objects (may be empty).
        """
        return []

    def adjudicate(
        self,
        action: SemanticAction,
        projection: SemanticProjection,
        intent: str,
        grounding: GroundingResult,
        result: ValidationResult,
    ) -> AdjudicationResult:
        """
        DM adjudication for Zone 3 / rejected actions.

        Proposes durable facts, delayed effects, and validated transitions.
        Default returns empty — real adapters override; Mock uses rule fallback
        in ``world_adjudicator.invoke_adjudication``.
        """
        return AdjudicationResult()

    def enrich_goal(
        self,
        goal_template: str,
        character_name: str,
        *,
        context: str = "",
    ) -> Optional[str]:
        """
        Rewrite a rule-synthesized goal into vivid in-character prose.

        Optional enrichment — returns None when narrate is unavailable.
        """
        prompt = (
            f"Rewrite this NPC goal as ONE short in-character sentence "
            f"(max 22 words). Keep the same intent.\n"
            f"Character: {character_name}\n"
            f"Template goal: {goal_template}\n"
        )
        if context:
            prompt += f"Context: {context}\n"
        prompt += "Rewritten goal:"
        try:
            raw = self.narrate(prompt)
            if raw:
                return raw.strip().strip("\"'")[:240]
        except Exception as exc:
            logger.debug("enrich_goal failed: %s", exc)
        return None

    def _log_call(
        self,
        projection_id: str,
        player_intent: str,
        action: SemanticAction,
        latency_ms: float,
        model_id: str,
    ) -> None:
        logger.info(
            "LM call | model=%s | proj=%s | intent=%r | action=%s | latency=%.1fms",
            model_id,
            projection_id,
            player_intent,
            action.verb,
            latency_ms,
        )


# ---------------------------------------------------------------------------
# Mock adapter (deterministic, no network)
# ---------------------------------------------------------------------------

