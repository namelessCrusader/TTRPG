from __future__ import annotations

import difflib
import json
import logging
import re
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
from .base import LMAdapter
from .helpers import (
    _build_adjudication_prompt,
    _build_adjudication_schema,
    _build_consequence_prompt,
    _build_consequence_schema,
    _build_dynamic_schema,
    _build_npc_reply_prompt,
    _build_npc_system_prompt,
    _build_npc_user_prompt,
    _build_strict_json_schema,
    _build_system_prompt,
    _build_user_prompt,
    _extract_json_array,
    _first_visible_entity,
    _parse_adjudication_response,
    _parse_lm_response,
    _parse_proposals,
    _strip_model_thinking,
    _strip_thinking_blocks,
    _validate_hf_model_dir,
    _build_mdp_selection_schema,
    _build_mdp_selection_prompt,
    _parse_mdp_selection_response,
)
logger = logging.getLogger(__name__)

class OpenAILMAdapter(LMAdapter):
    """
    Production adapter using OpenAI Structured Outputs (schema-constrained).

    Uses response_format=json_schema with strict=True, which enforces the
    SemanticAction schema server-side via constrained decoding.  This makes
    MalformedActionError structurally unreachable for schema violations
    (only network errors can still raise it).

    Requires OPENAI_API_KEY.  Supported on gpt-4o, gpt-4o-mini and later.
    """

    MODEL_ID = "gpt-4o-mini"

    def __init__(self, model: str = MODEL_ID, temperature: float = 0.3):
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILMAdapter. "
                "Run: pip install openai"
            ) from exc
        self.model = model
        self.temperature = temperature
        # Derive a strict JSON schema from SemanticAction.
        # OpenAI structured outputs requires additionalProperties: false
        # and all fields must be present (no $defs cycles).
        self._json_schema = _build_strict_json_schema()
        self.last_raw_response: Optional[str] = None
        self.last_latency_ms: Optional[float] = None

    def infer(
        self,
        projection: SemanticProjection,
        player_intent: str,
    ) -> SemanticAction:
        import openai

        t0 = time.monotonic()

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(projection, player_intent)

        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        call_schema = _build_dynamic_schema(self._json_schema, visible_ids)

        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SemanticAction",
                        "schema": call_schema,
                        "strict": True,
                    },
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.OpenAIError as exc:
            raise MalformedActionError(f"OpenAI API error: {exc}") from exc

        raw = response.choices[0].message.content or ""
        latency = (time.monotonic() - t0) * 1000
        self.last_raw_response = raw
        self.last_latency_ms = latency

        action = _parse_lm_response(raw, projection.focal_entity, player_intent, visible_ids)
        self._log_call(
            projection.projection_id, player_intent, action, latency, self.model
        )
        return action

    def infer_npc(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
    ) -> SemanticAction:
        import openai

        t0 = time.monotonic()
        system_prompt = _build_npc_system_prompt(character_sheet)
        user_prompt = _build_npc_user_prompt(character_sheet, projection)

        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        call_schema = _build_dynamic_schema(self._json_schema, visible_ids)

        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SemanticAction",
                        "schema": call_schema,
                        "strict": True,
                    },
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.OpenAIError as exc:
            raise MalformedActionError(f"OpenAI API error: {exc}") from exc

        raw = response.choices[0].message.content or ""
        latency = (time.monotonic() - t0) * 1000
        self.last_raw_response = raw
        self.last_latency_ms = latency

        action = _parse_lm_response(
            raw, character_sheet.npc_id, f"[npc:{character_sheet.name}]", visible_ids
        )
        self._log_call(
            projection.projection_id,
            f"[npc:{character_sheet.name}]",
            action,
            latency,
            self.model,
        )
        return action

    def infer_npc_mdp(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
        options: list[str],
        option_indices: list[int],
    ) -> dict:
        import openai

        t0 = time.monotonic()
        system_prompt, user_prompt = _build_mdp_selection_prompt(
            character_sheet, projection, options
        )
        call_schema = _build_mdp_selection_schema(option_indices)

        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "MdpSelection",
                        "schema": call_schema,
                        "strict": True,
                    },
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.OpenAIError as exc:
            raise MalformedActionError(f"OpenAI API error: {exc}") from exc

        raw = response.choices[0].message.content or ""
        latency = (time.monotonic() - t0) * 1000
        self.last_raw_response = raw
        self.last_latency_ms = latency

        selection = _parse_mdp_selection_response(raw)
        return selection

    def enrich_ambient_event(
        self,
        ambient_id: str,
        base_narrative: str,
        current_scene_info: dict,
    ) -> dict:
        import openai

        t0 = time.monotonic()
        system_prompt, user_prompt = _build_ambient_enrichment_prompt(
            ambient_id, base_narrative, current_scene_info
        )
        visible_ids = [ch["id"] for ch in current_scene_info.get("active_characters", [])]
        potential_coords = current_scene_info.get("potential_coords", [])
        call_schema = _build_ambient_enrichment_schema(visible_ids, potential_coords)

        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "AmbientEnrichment",
                        "schema": call_schema,
                        "strict": True,
                    },
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.OpenAIError as exc:
            raise MalformedActionError(f"OpenAI ambient API error: {exc}") from exc

        raw = response.choices[0].message.content or ""
        self.last_raw_response = raw
        self.last_latency_ms = (time.monotonic() - t0) * 1000

        enrichment = _parse_ambient_enrichment_response(raw)
        return enrichment

    def narrate(self, prompt: str) -> Optional[str]:
        """Unconstrained prose generation for the NarratorActor."""
        import openai
        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception:
            return None

    def choose_npc_option(self, prompt: str) -> Optional[str]:
        """NPC multiple-choice selection (decision role, not narrator)."""
        import openai
        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=8,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are simulating a character choosing their next action. "
                            "Reply with ONLY a single digit (the option number). "
                            "No explanation or other text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception:
            return None

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
        """Unconstrained in-character NPC reply (D1)."""
        import openai
        system, user = _build_npc_reply_prompt(
            sheet, utterance, projection,
            speaker_name=speaker_name,
            intent_hint=intent_hint,
            seed_options=seed_options,
            anti_repeat=anti_repeat,
        )
        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.75,
                max_tokens=80,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception as exc:
            logger.warning("generate_npc_reply failed: %s", exc)
            return None

    def propose_consequences(
        self,
        action: SemanticAction,
        projection: SemanticProjection,
        visible_ids: list[str],
    ) -> list[TransitionProposal]:
        """Constrained consequence proposals (D5)."""
        import openai
        if not visible_ids:
            return []
        schema = _build_consequence_schema(visible_ids)
        system, user = _build_consequence_prompt(action, projection)
        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.4,
                max_tokens=200,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ConsequenceProposals",
                        "schema": schema,
                        "strict": False,
                    },
                },
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = response.choices[0].message.content or ""
            return _parse_proposals(raw, visible_ids)
        except Exception as exc:
            logger.warning("propose_consequences failed: %s", exc)
            return []

    def adjudicate(
        self,
        action: SemanticAction,
        projection: SemanticProjection,
        intent: str,
        grounding: GroundingResult,
        result: ValidationResult,
    ) -> AdjudicationResult:
        """DM adjudication for unresolved actions."""
        import openai

        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        schema = _build_adjudication_schema(visible_ids)
        system, user = _build_adjudication_prompt(
            action, projection, intent, grounding, result,
        )
        client = openai.OpenAI()
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.5,
                max_tokens=350,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "AdjudicationResult",
                        "schema": schema,
                        "strict": False,
                    },
                },
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw = response.choices[0].message.content or ""
            return _parse_adjudication_response(raw, visible_ids, projection.tick)
        except Exception as exc:
            logger.warning("adjudicate failed: %s", exc)
            return AdjudicationResult()


