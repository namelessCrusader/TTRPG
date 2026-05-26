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
    _parse_adjudication_response,
    _parse_lm_response,
    _parse_proposals,
    _strip_model_thinking,
    _strip_thinking_blocks,
    _build_mdp_selection_schema,
    _build_mdp_selection_prompt,
    _parse_mdp_selection_response,
)

logger = logging.getLogger(__name__)

class OllamaLMAdapter(LMAdapter):
    """
    Local Ollama adapter with JSON schema-constrained output.

    Ollama's `format` parameter accepts a JSON schema object (since v0.4).
    This enforces the SemanticAction schema at the sampling level — the
    model cannot produce tokens that violate the schema.

    Performance configuration:
      keep_alive: how long Ollama keeps the model resident in VRAM after
                  a call. Default "30m" prevents the model from being
                  unloaded between turns. Without this, every call after
                  ~5 minutes of idle time pays a full model-load cost
                  (often >60s on consumer GPUs) and the first request
                  after launch always times out unless warm()ed.
      infer_timeout / narrate_timeout: caps for the constrained and
                                        unconstrained calls respectively.

    Run:
        ollama serve
        ollama pull qwen2.5:7b
    """

    DEFAULT_MODEL = "qwen2.5:7b"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        keep_alive: str = "30m",
        infer_timeout: float = 120.0,
        narrate_timeout: float = 60.0,
        warmup: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.infer_timeout = infer_timeout
        self.narrate_timeout = narrate_timeout
        self._json_schema = _build_strict_json_schema()
        # Debug introspection — populated on every infer() call. Used by
        # the REPL's --debug-lm mode to surface the raw model output.
        self.last_raw_response: Optional[str] = None
        self.last_latency_ms: Optional[float] = None
        if warmup:
            self.warm()

    def warm(self) -> None:
        """
        Force Ollama to load the model into memory ahead of any real call.

        Without this, the very first inference pays the full model-load
        latency on top of generation, which routinely exceeds even a 60s
        timeout on consumer GPUs. POST /api/generate with empty prompt
        and the configured keep_alive causes Ollama to load the weights
        and pin them according to keep_alive.
        """
        import urllib.request as _req

        payload = json.dumps({
            "model": self.model,
            "prompt": "",
            "keep_alive": self.keep_alive,
            "stream": False,
        }).encode()
        request = _req.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with _req.urlopen(request, timeout=self.infer_timeout) as resp:
                resp.read()
            logger.info(
                "Ollama warmup complete | model=%s | %.1fs",
                self.model,
                time.monotonic() - t0,
            )
        except Exception as exc:
            logger.warning(
                "Ollama warmup failed for %s (will retry on first call): %s",
                self.model, exc,
            )

    def infer(
        self,
        projection: SemanticProjection,
        player_intent: str,
    ) -> SemanticAction:
        import urllib.request as _req

        t0 = time.monotonic()

        # Build a per-call schema with target constrained to the IDs of
        # entities that are actually visible in this projection.  The LM
        # cannot emit an arbitrary string for target — only a canonical
        # entity ID, a Coord object, or null.
        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        call_schema = _build_dynamic_schema(self._json_schema, visible_ids)

        payload = json.dumps({
            "model": self.model,
            "format": call_schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": _build_user_prompt(projection, player_intent)},
            ],
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.infer_timeout) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise MalformedActionError(f"Ollama request failed: {exc}") from exc

        raw = body.get("message", {}).get("content", "")
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
        import urllib.request as _req

        t0 = time.monotonic()
        system_prompt = _build_npc_system_prompt(character_sheet)
        user_prompt = _build_npc_user_prompt(character_sheet, projection)

        # Same per-call dynamic schema for NPC actions: target must be a
        # visible entity ID, a Coord, or null.
        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        call_schema = _build_dynamic_schema(self._json_schema, visible_ids)

        payload = json.dumps({
            "model": self.model,
            "format": call_schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.infer_timeout) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise MalformedActionError(f"Ollama request failed: {exc}") from exc

        raw = body.get("message", {}).get("content", "")
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
        import urllib.request as _req

        t0 = time.monotonic()
        system_prompt, user_prompt = _build_mdp_selection_prompt(
            character_sheet, projection, options
        )
        call_schema = _build_mdp_selection_schema(option_indices)

        payload = json.dumps({
            "model": self.model,
            "format": call_schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.infer_timeout) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise MalformedActionError(f"Ollama request failed: {exc}") from exc

        raw = body.get("message", {}).get("content", "")
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
        import urllib.request as _req

        t0 = time.monotonic()
        system_prompt, user_prompt = _build_ambient_enrichment_prompt(
            ambient_id, base_narrative, current_scene_info
        )
        visible_ids = [ch["id"] for ch in current_scene_info.get("active_characters", [])]
        potential_coords = current_scene_info.get("potential_coords", [])
        call_schema = _build_ambient_enrichment_schema(visible_ids, potential_coords)

        payload = json.dumps({
            "model": self.model,
            "format": call_schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.infer_timeout) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise MalformedActionError(f"Ollama ambient request failed: {exc}") from exc

        raw = body.get("message", {}).get("content", "")
        self.last_raw_response = raw
        self.last_latency_ms = (time.monotonic() - t0) * 1000

        enrichment = _parse_ambient_enrichment_response(raw)
        return enrichment

    def infer_semantic_pressure(self, slice_) -> list:
        from ..semantic_pressure import (
            SEMANTIC_DELTA_JSON_SCHEMA,
            build_semantic_pressure_prompts,
            parse_semantic_deltas,
        )
        import urllib.request as _req

        t0 = time.monotonic()
        system, user = build_semantic_pressure_prompts(slice_)
        payload = json.dumps({
            "model": self.model,
            "format": SEMANTIC_DELTA_JSON_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.35},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }).encode()
        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.infer_timeout) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise MalformedActionError(f"Ollama semantic request failed: {exc}") from exc
        raw = body.get("message", {}).get("content", "")
        self.last_latency_ms = (time.monotonic() - t0) * 1000
        self.last_raw_response = raw
        return parse_semantic_deltas(raw)

    def narrate(self, prompt: str) -> Optional[str]:
        """
        Unconstrained prose call — no `format` schema, plain chat completion.
        Used by NarratorActor for Zone 3 responses and outcome flavor text.
        """
        import urllib.request as _req

        payload = json.dumps({
            "model": self.model,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.7, "num_predict": 400},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a terse atmospheric narrator for a simulation game. "
                        "Write 1-3 sentences of grounded prose. "
                        "Do not invent simulation facts. No markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.narrate_timeout) as resp:
                body = json.loads(resp.read())
            import re as _re
            raw = (body.get("message", {}).get("content", "") or "").strip()
            text = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            return text or None
        except Exception:
            return None

    def choose_npc_option(self, prompt: str) -> Optional[str]:
        """NPC multiple-choice selection (decision role, not narrator)."""
        import urllib.request as _req

        payload = json.dumps({
            "model": self.model,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.3, "num_predict": 16},
            "messages": [
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
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.narrate_timeout) as resp:
                body = json.loads(resp.read())
            import re as _re
            raw = (body.get("message", {}).get("content", "") or "").strip()
            text = _re.sub(
                r"<think>.*?</think>", "", raw, flags=_re.DOTALL
            ).strip()
            return text or None
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
        import urllib.request as _req

        system, user = _build_npc_reply_prompt(
            sheet, utterance, projection,
            speaker_name=speaker_name,
            intent_hint=intent_hint,
            seed_options=seed_options,
            anti_repeat=anti_repeat,
        )
        payload = json.dumps({
            "model": self.model,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.75, "num_predict": 400},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.narrate_timeout) as resp:
                body = json.loads(resp.read())
            raw = (body.get("message", {}).get("content", "") or "").strip()
            # Strip any residual <think>…</think> blocks the model may emit
            import re as _re
            text = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            return text or None
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
        import urllib.request as _req

        if not visible_ids:
            return []

        schema = _build_consequence_schema(visible_ids)
        system, user = _build_consequence_prompt(action, projection)
        payload = json.dumps({
            "model": self.model,
            "format": schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.4, "num_predict": 500},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }).encode()

        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.narrate_timeout) as resp:
                body = json.loads(resp.read())
            msg = body.get("message", {}) or {}
            raw = msg.get("content", "") or ""
            if not raw.strip():
                raw = msg.get("thinking", "") or ""
            return _parse_proposals(raw, visible_ids)
        except Exception as exc:
            logger.debug("propose_consequences failed: %s", exc)
            return []

    def adjudicate(
        self,
        action: SemanticAction,
        projection: SemanticProjection,
        intent: str,
        grounding: GroundingResult,
        result: ValidationResult,
    ) -> AdjudicationResult:
        import urllib.request as _req

        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        schema = _build_adjudication_schema(visible_ids)
        system, user = _build_adjudication_prompt(
            action, projection, intent, grounding, result,
        )
        payload = json.dumps({
            "model": self.model,
            "format": schema,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.5, "num_predict": 600},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }).encode()
        request = _req.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _req.urlopen(request, timeout=self.narrate_timeout) as resp:
                body = json.loads(resp.read())
            msg = body.get("message", {}) or {}
            raw = msg.get("content", "") or msg.get("thinking", "") or ""
            return _parse_adjudication_response(raw, visible_ids, projection.tick)
        except Exception as exc:
            logger.debug("adjudicate failed: %s", exc)
            return AdjudicationResult()


