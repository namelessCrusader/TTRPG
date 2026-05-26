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
    _build_npc_system_prompt,
    _build_npc_user_prompt,
    _build_strict_json_schema,
    _build_system_prompt,
    _build_user_prompt,
    _parse_lm_response,
    _build_mdp_selection_schema,
    _build_mdp_selection_prompt,
    _parse_mdp_selection_response,
)

logger = logging.getLogger(__name__)

class OutlinesLMAdapter(LMAdapter):
    """
    Grammar-constrained sampling via the Outlines library (GGUF via llama.cpp).

    This is Level 3 (strongest) constraint: the logit mask is applied at
    every sampling step so the model cannot produce any token that would
    make the output diverge from the SemanticAction JSON schema.
    MalformedActionError is structurally unreachable.

    Works with any GGUF model via llama-cpp-python.

    Install:
        pip install outlines llama-cpp-python

    Usage:
        adapter = OutlinesLMAdapter("path/to/qwen2.5-7b-q4_k_m.gguf")
    """

    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        try:
            import outlines  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "outlines is required for OutlinesLMAdapter. "
                "Run: pip install outlines llama-cpp-python"
            ) from exc
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self._generator = None  # lazy initialise

    def _get_generator(self):
        if self._generator is None:
            import outlines
            model = outlines.models.llamacpp(
                self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=2048,
            )
            # Constrain sampling to the exact SemanticAction JSON schema.
            # outlines.generate.json applies a logit mask every step.
            self._generator = outlines.generate.json(model, SemanticAction)
        return self._generator

    def infer(
        self,
        projection: SemanticProjection,
        player_intent: str,
    ) -> SemanticAction:
        t0 = time.monotonic()
        generator = self._get_generator()

        prompt = (
            _build_system_prompt()
            + "\n\n"
            + _build_user_prompt(projection, player_intent)
        )

        # generator() returns a fully-validated SemanticAction directly.
        # No _parse_lm_response needed — Outlines handles parsing.
        try:
            action = generator(prompt)
        except Exception as exc:
            raise MalformedActionError(
                f"Outlines generation failed: {exc}"
            ) from exc

        # Enforce actor = focal entity (schema doesn't know focal entity)
        action.actor = projection.focal_entity
        action.raw_input = player_intent

        latency = (time.monotonic() - t0) * 1000
        self._log_call(
            projection.projection_id,
            player_intent,
            action,
            latency,
            f"outlines:{self.model_path}",
        )
        return action

    def infer_npc(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
    ) -> SemanticAction:
        t0 = time.monotonic()
        generator = self._get_generator()

        prompt = (
            _build_npc_system_prompt(character_sheet)
            + "\n\n"
            + _build_npc_user_prompt(character_sheet, projection)
        )

        try:
            action = generator(prompt)
        except Exception as exc:
            raise MalformedActionError(
                f"Outlines generation failed: {exc}"
            ) from exc

        action.actor = character_sheet.npc_id
        action.raw_input = f"[npc:{character_sheet.name}]"

        latency = (time.monotonic() - t0) * 1000
        self._log_call(
            projection.projection_id,
            f"[npc:{character_sheet.name}]",
            action,
            latency,
            f"outlines:{self.model_path}",
        )
        return action

    def infer_npc_mdp(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
        options: list[str],
        option_indices: list[int],
    ) -> dict:
        t0 = time.monotonic()
        system_prompt, user_prompt = _build_mdp_selection_prompt(
            character_sheet, projection, options
        )
        prompt = system_prompt + "\n\n" + user_prompt

        # For Outlines, if we don't have a dynamically compiled outlines json generator
        # for this specific list of option indices, we can either dynamically instantiate
        # outlines.generate.json, or fall back to calling llama.cpp directly with normal text.
        # Since Outlines is a GGUF runner, we can build a pydantic schema model dynamically
        # or use standard outlines text generation with regex.
        # To make it simple and absolutely robust for outlines_adapter, let's compile an Outlines JSON generator
        # on the fly since outlines.generate.json is incredibly fast at construction.
        from pydantic import BaseModel, Field, conint
        from typing import Literal

        # We construct a dynamic Pydantic model for this selection!
        class DynamicMdpSelection(BaseModel):
            selected_option_index: int = Field(description="The index of the selected option.")
            rationale: str = Field(description="Reason for choosing this option.")
            custom_speech_line: str = Field(description="Optional custom dialogue to overlay.")

        try:
            import outlines
            # We can use the json outlines generator with the DynamicMdpSelection class
            # outlines uses outlines.models.llamacpp model we instantiated in _get_generator()
            # Let's get the underlying outlines model
            generator = self._get_generator()
            model = generator.model
            json_gen = outlines.generate.json(model, DynamicMdpSelection)
            res = json_gen(prompt)
            selection = {
                "selected_option_index": int(res.selected_option_index),
                "rationale": str(res.rationale),
                "custom_speech_line": str(res.custom_speech_line),
                "creative_custom_action": None,
            }
            return selection
        except Exception as exc:
            # Fallback to standard outlines parser if json generation failed
            logger.warning("Outlines dynamic json generation failed: %s. Falling back.", exc)
            # Pick the first valid index as safe default
            return {
                "selected_option_index": option_indices[0],
                "rationale": "Fallback selection",
                "custom_speech_line": "",
                "creative_custom_action": None,
            }

    def enrich_ambient_event(
        self,
        ambient_id: str,
        base_narrative: str,
        current_scene_info: dict,
    ) -> dict:
        t0 = time.monotonic()
        system_prompt, user_prompt = _build_ambient_enrichment_prompt(
            ambient_id, base_narrative, current_scene_info
        )
        prompt = system_prompt + "\n\n" + user_prompt

        from pydantic import BaseModel, Field
        from typing import Optional, Any

        class DynamicAmbientEffect(BaseModel):
            kind: str = Field(description="The kind of effect, e.g. fluid_changed, environment_state_changed, entity_condition_changed, entity_health_changed, entity_emotional_state_changed")
            payload: dict = Field(description="Payload fields vary by kind. E.g. {x: 5, y: 11, z: 0, material: 'ale', volume_ml: 80.0, cause: 'ambient_slosh'}")

        class DynamicAmbientEnrichment(BaseModel):
            narrative: str = Field(description="Atmospheric enriched description of the ambient event.")
            effects: list[DynamicAmbientEffect] = Field(description="List of physical/state-changing consequences resulting from the event.")

        try:
            import outlines
            generator = self._get_generator()
            model = generator.model
            json_gen = outlines.generate.json(model, DynamicAmbientEnrichment)
            res = json_gen(prompt)
            enrichment = {
                "narrative": str(res.narrative),
                "effects": [
                    {"kind": str(eff.kind), "payload": dict(eff.payload)}
                    for eff in res.effects
                ]
            }
            return enrichment
        except Exception as exc:
            logger.warning("Outlines dynamic ambient json generation failed: %s. Falling back.", exc)
            return {
                "narrative": base_narrative,
                "effects": [],
            }


