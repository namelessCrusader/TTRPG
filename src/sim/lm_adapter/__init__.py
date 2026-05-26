"""
M5: LM Adapter interface and implementations.

Split into submodules for maintainability; this package re-exports the
public API so ``from src.sim.lm_adapter import MockLMAdapter`` unchanged.
"""

from __future__ import annotations

from .base import LMAdapter
from .helpers import (
    _build_adjudication_prompt,
    _build_adjudication_schema,
    _build_consequence_prompt,
    _build_consequence_schema,
    _build_dynamic_schema,
    _build_mutation_hints,
    _build_npc_drive_hint,
    _build_npc_reply_prompt,
    _build_npc_system_prompt,
    _build_npc_user_prompt,
    _build_strict_json_schema,
    _build_system_prompt,
    _build_user_prompt,
    _extract_json_array,
    _first_visible_entity,
    _fuzzy_resolve_entity_id,
    _parse_adjudication_response,
    _parse_lm_response,
    _parse_proposals,
    _strip_model_thinking,
    _strip_thinking_blocks,
    _validate_hf_model_dir,
)
from .helpers import get_adapter, resolve_torch_model_source
from .mock import MockLMAdapter
from .ollama_adapter import OllamaLMAdapter
from .openai_adapter import OpenAILMAdapter
from .outlines_adapter import OutlinesLMAdapter
from .torch_adapter import TorchLMAdapter
from .tiered import TieredLMAdapter

__all__ = [
    "LMAdapter",
    "MockLMAdapter",
    "OpenAILMAdapter",
    "OllamaLMAdapter",
    "TorchLMAdapter",
    "OutlinesLMAdapter",
    "TieredLMAdapter",
    "get_adapter",
    "resolve_torch_model_source",
    "_build_adjudication_prompt",
    "_build_adjudication_schema",
    "_build_consequence_prompt",
    "_build_consequence_schema",
    "_build_dynamic_schema",
    "_build_mutation_hints",
    "_build_npc_drive_hint",
    "_build_npc_reply_prompt",
    "_build_npc_system_prompt",
    "_build_npc_user_prompt",
    "_build_strict_json_schema",
    "_build_system_prompt",
    "_build_user_prompt",
    "_extract_json_array",
    "_first_visible_entity",
    "_fuzzy_resolve_entity_id",
    "_parse_adjudication_response",
    "_parse_lm_response",
    "_parse_proposals",
    "_strip_model_thinking",
    "_strip_thinking_blocks",
    "_validate_hf_model_dir",
]
