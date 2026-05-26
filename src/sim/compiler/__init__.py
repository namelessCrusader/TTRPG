"""
M4: Deterministic Compiler and Validator (package).

Public API is unchanged from the former monolithic ``compiler.py`` module.
"""

from __future__ import annotations

from .apply import apply_transitions
from .beliefs import build_belief_transitions
from .common import (
    _check_projection_firewall,
    _contest_roll,
    _entity_by_name,
    _resolve_entity_target,
    _resolve_relative_move,
    _seeded_float,
    current_carry_weight,
)
from .dispatch import compile_action
from .effects import compile_effects, substitution_table
from .proposals import _validate_proposal, _validate_proposals, _substitute_proposal

from .verbs import _interaction_record  # used by fluid_compiler

# Register kernel compilers (side effect on import).
from . import registry as _registry  # noqa: F401

__all__ = [
    "apply_transitions",
    "build_belief_transitions",
    "compile_action",
    "compile_effects",
    "current_carry_weight",
    "_check_projection_firewall",
    "_contest_roll",
    "_entity_by_name",
    "_resolve_entity_target",
    "_resolve_relative_move",
    "_seeded_float",
    "_validate_proposal",
    "_validate_proposals",
    "_interaction_record",
]
