"""
Spell AST node definitions.

The spell programming language lets players write programs that manipulate
world state.  Programs are parsed into these dataclasses and then executed by
SpellExecutor in spell_runtime.py.

Grammar overview (braces delimit blocks, commas separate statements inline):
    program    := stmt (, stmt)* | stmt NEWLINE ...
    stmt       := cost_stmt | for_stmt | if_stmt | repeat_stmt
                | persist_stmt | trigger_stmt | effect_stmt
    cost_stmt  := COST resource NUMBER
    for_stmt   := FOR EACH selector { stmt* }
    if_stmt    := IF condition { stmt* } [ELSE { stmt* }]
    repeat_stmt:= REPEAT NUMBER { stmt* }
    persist_stmt:= PERSIST NUMBER { stmt* }
    effect_stmt:= OP_NAME arg* [selector]
    selector   := SELF | TARGET | NEAR NUMBER | LINE NUMBER
                | ALL tag | TAGGED tag
    condition  := HAS_TAG tag | PROP property op number | NOT condition
    arg        := NUMBER | IDENT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Selectors — choose which entities the body applies to
# ---------------------------------------------------------------------------

@dataclass
class SelfSelector:
    """The caster only."""


@dataclass
class TargetSelector:
    """The player's currently targeted entity."""


@dataclass
class NearSelector:
    """All entities within `radius` Manhattan tiles of the affected entity."""
    radius: int


@dataclass
class LineSelector:
    """All entities along a ray `length` tiles in the caster's facing direction."""
    length: int


@dataclass
class AllSelector:
    """All entities bearing `tag`."""
    tag: str


@dataclass
class TaggedSelector:
    """Alias for AllSelector (sugar: TAGGED flammable)."""
    tag: str


Selector = Union[
    SelfSelector, TargetSelector, NearSelector, LineSelector,
    AllSelector, TaggedSelector,
]


# ---------------------------------------------------------------------------
# Conditions — guards for IF blocks
# ---------------------------------------------------------------------------

@dataclass
class HasTagCondition:
    tag: str


@dataclass
class PropCondition:
    """PROP property_name op value — e.g. PROP health < 30"""
    prop: str
    op: str   # ">", "<", ">=", "<=", "==", "!="
    value: float


@dataclass
class NotCondition:
    inner: "Condition"


Condition = Union[HasTagCondition, PropCondition, NotCondition]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class CostStatement:
    """COST mana 25 — declares mana (or other resource) cost."""
    resource: str
    amount: float


@dataclass
class ForEachStatement:
    """FOR EACH selector { body }"""
    selector: Selector
    body: list["Statement"]


@dataclass
class IfStatement:
    """IF condition { then } [ELSE { else }]"""
    condition: Condition
    then_body: list["Statement"]
    else_body: list["Statement"] = field(default_factory=list)


@dataclass
class RepeatStatement:
    """REPEAT n { body } — execute body n times on same selection."""
    count: int
    body: list["Statement"]


@dataclass
class PersistStatement:
    """PERSIST n { body } — re-execute body every tick for n ticks.
    Adds an active trigger to the caster's entity state."""
    duration: int
    body: list["Statement"]


@dataclass
class EffectStatement:
    """
    A single operation call: OP_NAME arg1 arg2 ... [selector]

    Examples:
      HEAT 400
      HEAT 400 NEAR 3
      HARM 25 TARGET
      TAG burning
    """
    op_name: str
    args: list[Union[int, float, str]]
    selector: Optional[Selector] = None


Statement = Union[
    CostStatement, ForEachStatement, IfStatement,
    RepeatStatement, PersistStatement, EffectStatement,
]


# ---------------------------------------------------------------------------
# Top-level program
# ---------------------------------------------------------------------------

@dataclass
class SpellProgram:
    """A parsed, ready-to-execute spell program."""
    statements: list[Statement]
    # Source text preserved for error messages and grimoire display.
    source: str = ""


# ---------------------------------------------------------------------------
# Operation definition (loaded from magic.yaml)
# ---------------------------------------------------------------------------

@dataclass
class OpDefinition:
    """
    Describes one world-defined operation and how it maps to engine primitives.

    Fields
    ------
    name          Uppercase identifier players write in spells (e.g. "HEAT").
    primitive     Engine primitive to invoke.  One of:
                    ADJUST_PROP      — add `sign * amount` to a numeric property
                    SET_PROP         — set a property to a literal value
                    SET_EMOTIONAL    — set emotional_state to `state`
                    SET_CONDITION    — add/set a condition for `duration` ticks
                    ADD_TAG          — add `tag` to entity.tags
                    REMOVE_TAG       — remove `tag` from entity.tags
                    MOVE_AWAY        — push entity away from caster by `distance`
                    MOVE_TOWARD      — pull entity toward caster by `distance`
                    LINK_EDGE        — create relational edge (kind, weight)
    prop          Property name operated on (ADJUST_PROP / SET_PROP primitives).
    sign          +1 or -1 (ADJUST_PROP only, multiplied by the amount arg).
    state         EmotionalState value (SET_EMOTIONAL only).
    condition     Condition name string (SET_CONDITION only).
    tag           Tag string (ADD_TAG / REMOVE_TAG only).
    edge_kind     EdgeKind value (LINK_EDGE only).
    args          Ordered list of parameter names the operation accepts.
    requires_tag  If set, operation only fires on targets with this tag.
    mana_formula  Python expression string evaluated with arg values bound;
                  references the first arg as `n`, second as `m`, etc.
                  Examples: "n * 0.8"  "10"  "n * m * 0.5"
    description   Human-readable description shown in grimoires.
    """
    name: str
    primitive: str
    args: list[str] = field(default_factory=list)
    prop: Optional[str] = None
    sign: int = 1
    state: Optional[str] = None
    condition: Optional[str] = None
    tag: Optional[str] = None
    edge_kind: Optional[str] = None
    requires_tag: Optional[str] = None
    mana_formula: str = "0"
    description: str = ""


# ---------------------------------------------------------------------------
# Grimoire definition (loaded from worlds/*/grimoires/*.yaml)
# ---------------------------------------------------------------------------

@dataclass
class GrimoireDefinition:
    """A readable in-world document that teaches spell operations."""
    grimoire_id: str
    name: str
    text: str
    teaches_ops: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spell vocabulary (attached to WorldConfig.extra["spell_vocab"])
# ---------------------------------------------------------------------------

@dataclass
class SpellVocabulary:
    """
    The complete set of operations available in this world, plus any
    named spells that have been pre-inscribed into the world pack.
    """
    operations: dict[str, OpDefinition] = field(default_factory=dict)
    grimoires: dict[str, GrimoireDefinition] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class SpellError(Exception):
    """Raised when a spell program is malformed or cannot execute."""
