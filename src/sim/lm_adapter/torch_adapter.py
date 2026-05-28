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
    _strip_thinking_blocks,
    _build_mdp_selection_schema,
    _build_mdp_selection_prompt,
    _build_npc_speech_prompt,
    _parse_mdp_selection_response,
    _parse_npc_speech_response,
)

logger = logging.getLogger(__name__)

class TorchLMAdapter(LMAdapter):
    """
    Local HuggingFace ``transformers`` + PyTorch inference (no Ollama server).

    ``model`` is a HuggingFace hub id (e.g. ``Qwen/Qwen2.5-1.5B-Instruct``)
    or a local directory with ``config.json`` + weights.

    ``infer`` / ``infer_npc`` prompt for JSON and parse with ``_parse_lm_response``
    (same recovery path as unconstrained Ollama). For logit-masked JSON on local
    weights, use ``OutlinesLMAdapter`` with a GGUF file instead.

    Install:
        pip install torch transformers accelerate
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        temperature: float = 0.3,
        max_new_tokens_infer: int = 1024,
        max_new_tokens_narrate: int = 400,
        trust_remote_code: bool = True,
        warmup: bool = True,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TorchLMAdapter requires torch and transformers. "
                "Run: pip install torch transformers accelerate"
            ) from exc

        self.model_id = model
        self.device = device
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.max_new_tokens_infer = max_new_tokens_infer
        self.max_new_tokens_narrate = max_new_tokens_narrate
        self.trust_remote_code = trust_remote_code
        self.last_raw_response: Optional[str] = None
        self.last_latency_ms: Optional[float] = None

        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.RLock()
        self._load_error: Optional[Exception] = None
        self.load_failed: bool = False

        if warmup:
            self.warm()

    @property
    def is_ready(self) -> bool:
        """True when weights are loaded and inference can run."""
        return self._model is not None and self._load_error is None and not self.load_failed

    def warm(self) -> None:
        """Load weights into memory before the first player/NPC turn."""
        t0 = time.monotonic()
        try:
            self._chat(
                [
                    {"role": "system", "content": "Reply with the word ready."},
                    {"role": "user", "content": "ready"},
                ],
                max_new_tokens=8,
                temperature=0.0,
            )
            self.load_failed = False
            logger.info(
                "Torch warmup complete | model=%s | %.1fs",
                self.model_id,
                time.monotonic() - t0,
            )
        except Exception as exc:
            self.load_failed = True
            logger.warning(
                "Torch warmup failed for %s (LM calls will fail until the model loads): %s",
                self.model_id,
                exc,
            )

    @staticmethod
    def _from_pretrained_resilient(cls, model_id: str, *, label: str, **kwargs):
        """Call ``from_pretrained`` with retries for transient HF download errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                return cls.from_pretrained(model_id, **kwargs)
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = any(
                    token in msg
                    for token in ("brotli", "connection", "timeout", "429", "503")
                )
                if transient and attempt < 2:
                    logger.warning(
                        "Torch %s load transient error (attempt %d/3): %s",
                        label,
                        attempt + 1,
                        exc,
                    )
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
        assert last_exc is not None
        raise last_exc

    def _resolve_local_snapshot(self) -> Optional[str]:
        """Use huggingface_hub snapshot_download when hub fetch is flaky."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return None
        try:
            local_dir = snapshot_download(
                repo_id=self.model_id,
                local_files_only=False,
            )
            if Path(local_dir, "config.json").is_file():
                return str(local_dir)
        except Exception as exc:
            logger.debug("snapshot_download fallback failed for %s: %s", self.model_id, exc)
        return None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(
                f"Torch model load previously failed for {self.model_id}: {self._load_error}"
            ) from self._load_error
        with self._load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise RuntimeError(
                    f"Torch model load previously failed for {self.model_id}: {self._load_error}"
                ) from self._load_error
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_source = self.model_id
            dtype = self.torch_dtype
            if dtype is None:
                if torch.cuda.is_available():
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                else:
                    dtype = torch.float32
            load_kwargs: dict = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": dtype,
            }
            if self.device:
                load_kwargs["device_map"] = self.device
            elif torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = "cpu"

            try:
                logger.info("Loading torch model %s …", model_source)
                tokenizer = self._from_pretrained_resilient(
                    AutoTokenizer,
                    model_source,
                    label="tokenizer",
                    trust_remote_code=self.trust_remote_code,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = self._from_pretrained_resilient(
                    AutoModelForCausalLM,
                    model_source,
                    label="weights",
                    **load_kwargs,
                )
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
                self.load_failed = False
            except Exception as exc:
                snapshot = self._resolve_local_snapshot()
                if snapshot and snapshot != model_source:
                    logger.info("Retrying torch load from local snapshot %s", snapshot)
                    try:
                        tokenizer = self._from_pretrained_resilient(
                            AutoTokenizer,
                            snapshot,
                            label="tokenizer",
                            trust_remote_code=self.trust_remote_code,
                        )
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        model = self._from_pretrained_resilient(
                            AutoModelForCausalLM,
                            snapshot,
                            label="weights",
                            trust_remote_code=self.trust_remote_code,
                            torch_dtype=load_kwargs.get("torch_dtype", torch.float32),
                            device_map=load_kwargs.get("device_map", "cpu"),
                        )
                        model.eval()
                        self._tokenizer = tokenizer
                        self._model = model
                        self.load_failed = False
                        return
                    except Exception as retry_exc:
                        exc = retry_exc
                self._load_error = exc
                self.load_failed = True
                logger.error(
                    "Torch model load failed for %s. If this is a HuggingFace "
                    "download/tokenizer error, choose a known-loadable model "
                    "with TORCH_MODEL or pass a local checkpoint via TORCH_PATH. "
                    "GGUF-only downloads need Outlines (--outlines), not --torch. "
                    "Error: %s",
                    self.model_id,
                    exc,
                )
                raise

    def _resolve_torch_device(self):
        assert self._model is not None
        return next(self._model.parameters()).device

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        import torch

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        with self._gen_lock:
            if hasattr(self._tokenizer, "apply_chat_template"):
                try:
                    prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt = "\n\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in messages
                    )
            else:
                prompt = "\n\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                )

            inputs = self._tokenizer(prompt, return_tensors="pt")
            device = self._resolve_torch_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs: dict = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=temperature,
                )
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **gen_kwargs)

            new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _infer_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        actor: EntityId,
        raw_input: str,
        visible_ids: list[str],
        log_intent: str,
        projection_id: str,
    ) -> SemanticAction:
        t0 = time.monotonic()
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        user_prompt
                        + "\n\nRespond with ONE JSON object only (SemanticAction). "
                        "No markdown, no commentary."
                    ),
                },
            ],
            max_new_tokens=self.max_new_tokens_infer,
            temperature=self.temperature,
        )
        action = _parse_lm_response(raw, actor, raw_input, visible_ids)

        latency = (time.monotonic() - t0) * 1000
        self.last_raw_response = raw
        self.last_latency_ms = latency
        self._log_call(projection_id, log_intent, action, latency, self.model_id)
        return action

    def infer(
        self,
        projection: SemanticProjection,
        player_intent: str,
    ) -> SemanticAction:
        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        return self._infer_structured(
            system_prompt=_build_system_prompt(),
            user_prompt=_build_user_prompt(projection, player_intent),
            actor=projection.focal_entity,
            raw_input=player_intent,
            visible_ids=visible_ids,
            log_intent=player_intent,
            projection_id=projection.projection_id,
        )

    def infer_npc(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
    ) -> SemanticAction:
        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        return self._infer_structured(
            system_prompt=_build_npc_system_prompt(character_sheet),
            user_prompt=_build_npc_user_prompt(character_sheet, projection),
            actor=character_sheet.npc_id,
            raw_input=f"[npc:{character_sheet.name}]",
            visible_ids=visible_ids,
            log_intent=f"[npc:{character_sheet.name}]",
            projection_id=projection.projection_id,
        )

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
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=min(256, self.max_new_tokens_infer),
            temperature=min(0.85, self.temperature + 0.15),
        )
        latency = (time.monotonic() - t0) * 1000
        self.last_raw_response = raw
        self.last_latency_ms = latency

        selection = _parse_mdp_selection_response(raw)
        return selection

    def infer_npc_speech(
        self,
        character_sheet: NpcCharacterSheet,
        projection: SemanticProjection,
        action_label: str,
        verb: str,
    ) -> str:
        t0 = time.monotonic()
        system_prompt, user_prompt = _build_npc_speech_prompt(
            character_sheet, projection, action_label, verb
        )
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=min(128, self.max_new_tokens_infer),
            temperature=self.temperature,
        )
        self.last_latency_ms = (time.monotonic() - t0) * 1000
        return _parse_npc_speech_response(raw)

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
        raw = self._chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=self.max_new_tokens_infer,
            temperature=self.temperature,
        )
        self.last_raw_response = raw
        self.last_latency_ms = (time.monotonic() - t0) * 1000

        enrichment = _parse_ambient_enrichment_response(raw)
        return enrichment

    def infer_semantic_pressure(self, slice_) -> list:
        from ..semantic_pressure import (
            build_semantic_pressure_prompts,
            parse_semantic_deltas,
        )

        t0 = time.monotonic()
        system, user = build_semantic_pressure_prompts(slice_)
        raw = self._chat(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        user
                        + '\n\nRespond with ONE JSON object: {"deltas":[...]} only.'
                    ),
                },
            ],
            max_new_tokens=min(512, self.max_new_tokens_infer),
            temperature=0.35,
        )
        self.last_latency_ms = (time.monotonic() - t0) * 1000
        self.last_raw_response = raw
        return parse_semantic_deltas(_strip_thinking_blocks(raw))

    def narrate(self, prompt: str) -> Optional[str]:
        try:
            raw = self._chat(
                [
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
                max_new_tokens=self.max_new_tokens_narrate,
                temperature=0.7,
            )
            text = _strip_thinking_blocks(raw)
            return text or None
        except Exception:
            return None

    def choose_npc_option(self, prompt: str) -> Optional[str]:
        try:
            raw = self._chat(
                [
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
                max_new_tokens=16,
                temperature=0.3,
            )
            return _strip_thinking_blocks(raw) or None
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
        system, user = _build_npc_reply_prompt(
            sheet,
            utterance,
            projection,
            speaker_name=speaker_name,
            intent_hint=intent_hint,
            seed_options=seed_options,
            anti_repeat=anti_repeat,
        )
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_new_tokens=min(120, self.max_new_tokens_narrate),
                temperature=0.75,
            )
            import re as _re

            raw = _re.sub(r"^\s*[1-9][.)]\s+", "", raw.strip()).strip().strip("\"'")
            text = _strip_thinking_blocks(raw)
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
        if not visible_ids:
            return []
        system, user = _build_consequence_prompt(action, projection)
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            user
                            + "\n\nRespond with a JSON array of TransitionProposal objects only."
                        ),
                    },
                ],
                max_new_tokens=min(500, self.max_new_tokens_narrate),
                temperature=0.4,
            )
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
        visible_ids = [str(e.entity_id) for e in projection.visible_entities]
        system, user = _build_adjudication_prompt(
            action, projection, intent, grounding, result,
        )
        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            user
                            + "\n\nRespond with a single JSON AdjudicationResult object only."
                        ),
                    },
                ],
                max_new_tokens=min(600, self.max_new_tokens_narrate),
                temperature=0.5,
            )
            return _parse_adjudication_response(raw, visible_ids, projection.tick)
        except Exception as exc:
            logger.debug("adjudicate failed: %s", exc)
            return AdjudicationResult()


