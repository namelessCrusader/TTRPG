"""
Tiered LM adapter — route each LM responsibility to the model best suited
for it.

Originally a 2-way split between ``player_adapter`` (high-stakes player
intent parsing) and ``npc_adapter`` (cheaper, high-volume NPC cognition).
Now optionally a 4-way split, adding:

  - ``adjudicator_adapter`` — a small, schema-tight model whose only job
    is to emit ``TransitionProposal``-shaped JSON for rescued player
    actions.  Keeping this separate from infer/narrate keeps variance
    down because adjudication is the highest-impact LM mutation path:
    the same situation should resolve the same way more often.

  - ``narrator_adapter`` — a free-prose model with no schema constraint.
    Narration produces no state changes (the firewall enforces this), so
    a model tuned for evocative prose can serve here without leaking
    into mechanical paths.

All four slots fall back gracefully: if a role-specific slot is
``None``, the relevant call routes to the actor-based default
(``player_adapter`` for player paths, ``npc_adapter`` for NPC paths),
preserving the original 2-way contract for callers that don't opt in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..schemas import (
    AdjudicationResult,
    GroundingResult,
    NpcCharacterSheet,
    SemanticAction,
    SemanticProjection,
    TransitionProposal,
    ValidationResult,
)
from .base import LMAdapter

if TYPE_CHECKING:
    from ..semantic_slice import SemanticDelta, SemanticSlice


class TieredLMAdapter(LMAdapter):
    """
    Routes each LM responsibility to its best-fit adapter.

    Default routing (when role-specific adapters are ``None``):

      Player path : infer(), adjudicate(), propose_consequences()
                    → ``player_adapter``
      NPC path    : infer_npc(), generate_npc_reply(), enrich_goal()
                    → ``npc_adapter``
      Narration   : narrate()  → ``npc_adapter``

    With role-specific overrides:

      adjudicator_adapter → adjudicate() + propose_consequences()
      narrator_adapter    → narrate()

    Tests and existing callers that pass only ``player_adapter`` and
    ``npc_adapter`` get the original behaviour unchanged.
    """

    def __init__(
        self,
        player_adapter: LMAdapter,
        npc_adapter: LMAdapter,
        *,
        adjudicator_adapter: Optional[LMAdapter] = None,
        narrator_adapter: Optional[LMAdapter] = None,
    ):
        self.player_adapter = player_adapter
        self.npc_adapter = npc_adapter
        self.adjudicator_adapter = adjudicator_adapter
        self.narrator_adapter = narrator_adapter

    @property
    def player_model_id(self) -> str:
        return getattr(self.player_adapter, "model", type(self.player_adapter).__name__)

    @property
    def npc_model_id(self) -> str:
        return getattr(self.npc_adapter, "model", type(self.npc_adapter).__name__)

    @property
    def adjudicator_model_id(self) -> str:
        ad = self.adjudicator_adapter or self.player_adapter
        return getattr(ad, "model", type(ad).__name__)

    @property
    def narrator_model_id(self) -> str:
        ad = self.narrator_adapter or self.npc_adapter
        return getattr(ad, "model", type(ad).__name__)

    def infer(
        self,
        projection: SemanticProjection,
        player_intent: str,
    ) -> SemanticAction:
        return self.player_adapter.infer(projection, player_intent)

    def infer_npc(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
    ) -> SemanticAction:
        return self.npc_adapter.infer_npc(character_sheet, projection)

    def adjudicate(
        self,
        action: SemanticAction,
        projection: SemanticProjection,
        intent: str,
        grounding: GroundingResult,
        result: ValidationResult,
    ) -> AdjudicationResult:
        # Role-specific adjudicator wins; otherwise fall back to the
        # player-tier adapter (preserves original behaviour).
        ad = self.adjudicator_adapter or self.player_adapter
        fn = getattr(ad, "adjudicate", None)
        if fn:
            return ad.adjudicate(
                action, projection, intent, grounding, result,
            )
        return AdjudicationResult()

    def propose_consequences(
        self,
        action: SemanticAction,
        projection: SemanticProjection,
        visible_ids: list[str],
    ) -> list[TransitionProposal]:
        # Consequence proposal is the same "schema-tight rule output"
        # task as adjudication, so it shares the adjudicator slot.
        ad = self.adjudicator_adapter or self.player_adapter
        fn = getattr(ad, "propose_consequences", None)
        if fn:
            return ad.propose_consequences(
                action, projection, visible_ids,
            )
        return []

    def narrate(self, prompt: str) -> Optional[str]:
        ad = self.narrator_adapter or self.npc_adapter
        return ad.narrate(prompt)

    def generate_npc_reply(
        self,
        sheet: NpcCharacterSheet,
        utterance: str,
        projection: SemanticProjection,
        *,
        speaker_name: str = "",
        intent_hint: str = "",
        seed_options: list[str] | None = None,
        anti_repeat: str = "",
    ) -> Optional[str]:
        return self.npc_adapter.generate_npc_reply(
            sheet,
            utterance,
            projection,
            speaker_name=speaker_name,
            intent_hint=intent_hint,
            seed_options=seed_options,
            anti_repeat=anti_repeat,
        )

    def enrich_goal(
        self,
        goal_template: str,
        character_name: str,
        *,
        context: str = "",
    ) -> Optional[str]:
        return self.npc_adapter.enrich_goal(
            goal_template, character_name, context=context,
        )

    def infer_semantic_pressure(self, slice_: "SemanticSlice") -> "list[SemanticDelta]":
        fn = getattr(self.npc_adapter, "infer_semantic_pressure", None)
        if fn:
            return self.npc_adapter.infer_semantic_pressure(slice_)
        raise NotImplementedError

    def choose_npc_option(self, prompt: str) -> Optional[str]:
        return self.npc_adapter.choose_npc_option(prompt)

    def infer_npc_mdp(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
        options: list[str],
        option_indices: list[int],
    ) -> dict:
        return self.npc_adapter.infer_npc_mdp(
            character_sheet, projection, options, option_indices
        )

    def enrich_ambient_event(
        self,
        ambient_id: str,
        base_narrative: str,
        current_scene_info: dict,
    ) -> dict:
        ad = self.adjudicator_adapter or self.npc_adapter
        return ad.enrich_ambient_event(
            ambient_id, base_narrative, current_scene_info
        )

    def warm(self) -> None:
        for adapter in (self.player_adapter, self.npc_adapter):
            warm = getattr(adapter, "warm", None)
            if warm:
                warm()
