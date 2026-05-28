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
from .helpers import _first_visible_entity

logger = logging.getLogger(__name__)

class MockLMAdapter(LMAdapter):
    """
    Deterministic stub adapter.  Used for all tests and as the default
    when no real model is configured.

    Parses a small set of natural language patterns into SemanticActions.
    Unrecognised input falls back to WAIT.
    """

    MODEL_ID = "mock-deterministic-v1"

    def infer(
        self,
        projection: SemanticProjection,
        player_intent: str,
    ) -> SemanticAction:
        t0 = time.monotonic()
        action = self._parse_intent(projection, player_intent)
        latency = (time.monotonic() - t0) * 1000
        self._log_call(
            projection.projection_id,
            player_intent,
            action,
            latency,
            self.MODEL_ID,
        )
        return action

    def infer_semantic_pressure(self, slice_) -> list:
        from ..semantic_pressure import MockSemanticPressureAdapter

        return MockSemanticPressureAdapter().infer_pressure(slice_)

    def infer_npc(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
    ) -> SemanticAction:
        """
        Deterministic NPC stub for tests.

        Returns OBSERVE on whoever is most recently visible (no LM call,
        no network). The point of MockLMAdapter is to keep the test
        suite hermetic — tests that want richer NPC behavior should
        inject a custom adapter or use ReactivePolicy directly.
        """
        target = _first_visible_entity(projection)
        action = SemanticAction(
            verb=ActionType.OBSERVE,
            actor=character_sheet.npc_id,
            target=target,
            raw_input="[npc:mock]",
        )
        self._log_call(
            projection.projection_id,
            f"[npc:{character_sheet.name}]",
            action,
            0.0,
            self.MODEL_ID,
        )
        return action

    def infer_npc_mdp(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
        options: list[str],
        option_indices: list[int],
    ) -> dict:
        """
        Deterministic mockup for option forest selection.
        """
        mid = len(option_indices) // 2
        idx = option_indices[mid] if option_indices else 1
        return {
            "selected_option_index": idx,
            "rationale": "mock rationale for selecting option",
            "custom_speech_line": "",
            "creative_custom_action": None,
        }

    def infer_npc_speech(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
        action_label: str,
        verb: str,
    ) -> str:
        return f"Well then — {action_label}."

    def enrich_ambient_event(
        self,
        ambient_id: str,
        base_narrative: str,
        current_scene_info: dict,
    ) -> dict:
        """
        Deterministic mockup for dynamic ambient event enrichment.
        """
        narrative = f"{base_narrative} (Enriched by Mock: time={current_scene_info.get('time_of_day', 'unknown')})"
        effects = []
        coords = current_scene_info.get("potential_coords", [])
        if coords:
            first_coord = coords[0]
            effects.append({
                "kind": "environment_state_changed",
                "payload": {
                    "x": first_coord["x"],
                    "y": first_coord["y"],
                    "z": first_coord["z"],
                    "key": "mock_ambient_key",
                    "value": "mock_ambient_value"
                }
            })
        return {
            "narrative": narrative,
            "effects": effects,
        }

    def _parse_intent(
        self,
        projection: SemanticProjection,
        intent: str,
    ) -> SemanticAction:
        """
        Deterministic stub for tests only.

        Handles a tiny set of exact patterns that the test suite exercises.
        Production inference belongs to a real LM adapter; this parser
        deliberately stays minimal so it does not start drifting into a
        secondary intent system.
        """
        lower = intent.lower().strip()
        focal = projection.focal_entity

        if lower.startswith("move ") or lower.startswith("go "):
            parts = lower.replace(",", " ").split()
            nums = [p for p in parts if p.lstrip("-").isdigit()]
            if len(nums) >= 2:
                from ..schemas import Coord
                return SemanticAction(
                    verb=ActionType.MOVE,
                    actor=focal,
                    target=Coord(x=int(nums[0]), y=int(nums[1])),
                    raw_input=intent,
                )
            # "move forward 5", "move north 3", etc. — game_loop pre-processor
            # handles these; return a bare MOVE so it reaches the pre-processor.
            direction_words = {
                "forward", "backward", "back", "left", "right", "ahead",
                "north", "south", "east", "west",
                "northeast", "northwest", "southeast", "southwest",
                "n", "s", "e", "w", "ne", "nw", "se", "sw",
            }
            non_verb_parts = parts[1:]  # strip "move"/"go"
            dir_part = next((p for p in non_verb_parts if p in direction_words), None)
            if dir_part:
                return SemanticAction(
                    verb=ActionType.MOVE,
                    actor=focal,
                    target=dir_part,
                    raw_input=intent,
                )

        if "throw" in lower:
            return SemanticAction(
                verb=ActionType.THROW,
                actor=focal,
                target=_first_visible_entity(projection),
                raw_input=intent,
            )

        if "attack" in lower:
            return SemanticAction(
                verb=ActionType.ATTACK,
                actor=focal,
                target=_first_visible_entity(projection),
                raw_input=intent,
            )

        if "intimidat" in lower:
            avoid_combat = "without" in lower and "combat" in lower
            avoid_witnesses = "quiet" in lower or "alone" in lower
            return SemanticAction(
                verb=ActionType.INTIMIDATE,
                actor=focal,
                target=_first_visible_entity(projection),
                intent=IntentBlock(desired_outcome=["compliance"]),
                style=StyleBlock(aggression=70, emotional_tone="menacing"),
                constraints=ConstraintBlock(
                    avoid_combat=avoid_combat, avoid_witnesses=avoid_witnesses
                ),
                raw_input=intent,
            )

        if "persuad" in lower or "convince" in lower:
            return SemanticAction(
                verb=ActionType.PERSUADE,
                actor=focal,
                target=_first_visible_entity(projection),
                intent=IntentBlock(desired_outcome=["agreement"]),
                style=StyleBlock(emotional_tone="calm", aggression=20),
                raw_input=intent,
            )

        if "deceiv" in lower or "bluff" in lower:
            return SemanticAction(
                verb=ActionType.DECEIVE,
                actor=focal,
                target=_first_visible_entity(projection),
                intent=IntentBlock(desired_outcome=["misdirection"]),
                style=StyleBlock(emotional_tone="neutral", visibility=20),
                raw_input=intent,
            )

        if "say" in lower or "speak" in lower or "talk" in lower:
            return SemanticAction(
                verb=ActionType.SPEAK,
                actor=focal,
                target=_first_visible_entity(projection),
                intent=IntentBlock(rationale=intent),
                raw_input=intent,
            )

        if lower.startswith(("turn ", "face ", "rotate ")):
            # "turn left", "turn north", "face east", etc.
            parts = lower.split(None, 1)
            target = parts[1].strip() if len(parts) > 1 else None
            return SemanticAction(
                verb=ActionType.TURN,
                actor=focal,
                target=target,
                raw_input=intent,
            )

        if "look" in lower or "observe" in lower or "watch" in lower:
            return SemanticAction(
                verb=ActionType.OBSERVE,
                actor=focal,
                target=_first_visible_entity(projection),
                raw_input=intent,
            )

        if "flee" in lower or "escape" in lower:
            return SemanticAction(
                verb=ActionType.FLEE, actor=focal, raw_input=intent
            )

        return SemanticAction(
            verb=ActionType.WAIT, actor=focal, raw_input=intent
        )

