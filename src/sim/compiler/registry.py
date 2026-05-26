"""Kernel and built-in compiler registries."""

from __future__ import annotations

from ..schemas import ActionType
from .verbs import (
    _BUILTIN_DEFAULTS,
    _KERNEL_COMPILERS,
    _compile_attack,
    _compile_barricade,
    _compile_contact,
    _compile_craft,
    _compile_drop,
    _compile_edge_add,
    _compile_edge_update,
    _compile_equip,
    _compile_examine,
    _compile_extinguish,
    _compile_flee,
    _compile_generic,
    _compile_give,
    _compile_hide,
    _compile_ignite,
    _compile_mark,
    _compile_mix,
    _compile_move,
    _compile_observe,
    _compile_open_close,
    _compile_rest,
    _compile_retrieve,
    _compile_social,
    _compile_steal,
    _compile_store_in,
    _compile_symbolic,
    _compile_take,
    _compile_throw,
    _compile_trade,
    _compile_turn,
    _compile_unequip,
    _compile_unlock,
    _compile_use,
    _compile_wait,
    _compile_cast,
)

from ..fluid_compiler import _compile_drink, _compile_pour_drink
from ..interactions import INTERACTION_COMPILERS

_KERNEL_COMPILERS.update({
    ActionType.MOVE: _compile_move,
    ActionType.ATTACK: _compile_attack,
    ActionType.TAKE: _compile_take,
    ActionType.GIVE: _compile_give,
    ActionType.THROW: _compile_throw,
    "equip": _compile_equip,
    "unequip": _compile_unequip,
    "store": _compile_store_in,
    "retrieve": _compile_retrieve,
    "wear": _compile_equip,
    "remove": _compile_unequip,
    "sheathe": _compile_equip,
    "draw": _compile_equip,
    "turn": _compile_turn,
    "look": _compile_turn,
    "face": _compile_turn,
    "cast": _compile_cast,
    "drop": _compile_drop,
    "use": _compile_use,
    "mix": _compile_mix,
    "combine": _compile_mix,
    "examine": _compile_examine,
    "inspect": _compile_examine,
    "trade": _compile_trade,
    "buy": _compile_trade,
    "sell": _compile_trade,
    "steal": _compile_steal,
    "pickpocket": _compile_steal,
    "rest": _compile_rest,
    "sleep": _compile_rest,
    "craft": _compile_craft,
    "make": _compile_craft,
    "build": _compile_craft,
    "construct": _compile_craft,
    "fabricate": _compile_craft,
    "whittle": _compile_craft,
    "forge": _compile_craft,
    "mark": _compile_mark,
    "scratch": _compile_mark,
    "carve": _compile_mark,
    "write": _compile_mark,
    "inscribe": _compile_mark,
    "barricade": _compile_barricade,
    "blockade": _compile_barricade,
    "block": _compile_barricade,
    "pile": _compile_barricade,
    "unlock": _compile_unlock,
    "lock": _compile_unlock,
    "open": _compile_unlock,
    "seal": _compile_unlock,
    "ignite": _compile_ignite,
    "light": _compile_ignite,
    "kindle": _compile_ignite,
    "torch": _compile_ignite,
    # extinguish/douse/quench are intentionally NOT kernel verbs.  Routing
    # them through the verb pipeline (templates → open_verbs →
    # property_interactions → builtin) lets pack rules like
    # ``worlds/voxel_tavern/property_interactions.yaml`` add staff/cask
    # narrative to the douse action without forking the kernel compiler.
    # The same `_compile_extinguish` body is wired up below as the
    # built-in default so behaviour is unchanged when no rule fires.
})

_KERNEL_COMPILERS["pour_drink"] = _compile_pour_drink
_KERNEL_COMPILERS["drink"] = _compile_drink
_KERNEL_COMPILERS.update(INTERACTION_COMPILERS)

_BUILTIN_DEFAULTS.update({
    ActionType.INTIMIDATE: _compile_social,
    ActionType.PERSUADE: _compile_social,
    ActionType.DECEIVE: _compile_social,
    ActionType.BRIBE: _compile_social,
    ActionType.THREATEN: _compile_social,
    ActionType.SPEAK: _compile_social,
    ActionType.ASK: _compile_social,
    ActionType.OBSERVE: _compile_observe,
    ActionType.INSPECT: _compile_observe,
    ActionType.OPEN: _compile_open_close,
    ActionType.CLOSE: _compile_open_close,
    ActionType.FLEE: _compile_flee,
    ActionType.WAIT: _compile_wait,
    ActionType.HIDE: _compile_hide,
    ActionType.CONTACT: _compile_contact,
    ActionType.SYMBOLIC: _compile_symbolic,
    ActionType.EDGE_ADD: _compile_edge_add,
    ActionType.EDGE_UPDATE: _compile_edge_update,
    "edge_create": _compile_edge_add,
    "edge_created": _compile_edge_add,
    "relationship_add": _compile_edge_add,
    "relationship_update": _compile_edge_update,
    "extinguish": _compile_extinguish,
    "douse": _compile_extinguish,
    "quench": _compile_extinguish,
})

# Silence unused import — generic path referenced by dispatch only.
_ = _compile_generic
