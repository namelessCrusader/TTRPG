"""
Spell Runtime: Lexer → Parser → Executor.

Takes a raw spell program string (everything after the [cast] tag) and
returns a list of validated Transitions ready for apply_transitions().

Architecture
────────────
  SpellLexer   — tokenises the source string into a flat Token list
  SpellParser  — recursive-descent parser → SpellProgram (AST)
  SpellExecutor— walks AST, resolves selectors against world state,
                 evaluates mana costs, emits Transitions

The executor never mutates world state directly — it only produces
Transitions.  Those are handed back to the game_loop which passes them
through apply_transitions() as usual.

Operation primitives map onto existing TransitionKind values where
possible, keeping the Transition machinery unchanged:

  ADJUST_PROP health    → ENTITY_HEALTH_CHANGED
  ADJUST_PROP mana      → ENTITY_MANA_CHANGED  (new kind handled below)
  ADJUST_PROP *         → ENTITY_PROPERTY_CHANGED (new, stored in meta)
  SET_PROP *            → ENTITY_PROPERTY_CHANGED
  SET_EMOTIONAL *       → ENTITY_EMOTIONAL_STATE_CHANGED
  SET_CONDITION *       → ENTITY_CONDITION_CHANGED
  ADD_TAG / REMOVE_TAG  → ENTITY_TAG_CHANGED   (new)
  MOVE_AWAY / TOWARD    → ENTITY_MOVED
  LINK_EDGE             → EDGE_CREATED / EDGE_UPDATED
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Union

from .schemas import (
    AlertnessLevel,
    Coord,
    EdgeKind,
    EmotionalState,
    EntityId,
    EntityState,
    FacingDirection,
    Transition,
    TransitionKind,
    WorldState,
)
from .spell_types import (
    AllSelector,
    Condition,
    CostStatement,
    EffectStatement,
    ForEachStatement,
    GrimoireDefinition,
    HasTagCondition,
    IfStatement,
    LineSelector,
    NearSelector,
    NotCondition,
    OpDefinition,
    PersistStatement,
    PropCondition,
    RepeatStatement,
    Selector,
    SelfSelector,
    SpellError,
    SpellProgram,
    SpellVocabulary,
    Statement,
    TaggedSelector,
    TargetSelector,
)

# ---------------------------------------------------------------------------
# New TransitionKind values needed by the spell runtime
# (these are injected into the existing Enum dynamically so existing code
#  needs no import change, but we define them as string constants here for
#  use inside this module).
# ---------------------------------------------------------------------------
_ENTITY_PROPERTY_CHANGED = "entity_property_changed"
_ENTITY_TAG_CHANGED      = "entity_tag_changed"
_ENTITY_MANA_CHANGED     = "entity_mana_changed"
_SPELL_CAST              = "spell_cast"

# Register them into the TransitionKind enum at import time.
# (Python Enums don't natively support dynamic extension, so we use the
#  functional API to rebuild — or more simply, we keep them as plain strings
#  and handle them explicitly in apply_transitions via the existing
#  "other transition kinds pass through" path in compiler.py.)


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class TType(Enum):
    NUMBER  = auto()
    IDENT   = auto()
    LBRACE  = auto()
    RBRACE  = auto()
    COMMA   = auto()
    COLON   = auto()
    OP_STR  = auto()   # ">", "<", ">=", "<=", "==", "!="
    EOF     = auto()


@dataclass
class Token:
    ttype: TType
    value: Any
    pos: int  # character offset in source (for error messages)


# Language keywords — recognised as IDENT tokens but treated specially
# by the parser.
_KEYWORDS = frozenset({
    "COST", "FOR", "EACH", "IF", "ELSE", "REPEAT", "PERSIST",
    "SELF", "TARGET", "NEAR", "LINE", "ALL", "TAGGED",
    "HAS_TAG", "PROP", "NOT",
})

_SELECTOR_KEYWORDS = frozenset({
    "SELF", "TARGET", "NEAR", "LINE", "ALL", "TAGGED",
})


def _lex(source: str) -> list[Token]:
    """Tokenise a spell program string."""
    tokens: list[Token] = []
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        # Whitespace (including newlines — we treat them as separators but
        # don't produce tokens; commas serve as explicit statement separators)
        if ch in " \t\n\r":
            i += 1
            continue

        # Comments: # to end of line
        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # Braces
        if ch == "{":
            tokens.append(Token(TType.LBRACE, "{", i))
            i += 1; continue
        if ch == "}":
            tokens.append(Token(TType.RBRACE, "}", i))
            i += 1; continue

        # Comma — statement separator for inline style
        if ch == ",":
            tokens.append(Token(TType.COMMA, ",", i))
            i += 1; continue

        # Colon
        if ch == ":":
            tokens.append(Token(TType.COLON, ":", i))
            i += 1; continue

        # Comparison operators (>=, <=, !=, >, <, ==)
        if ch in "><!=":
            if i + 1 < n and source[i + 1] == "=":
                tokens.append(Token(TType.OP_STR, source[i:i+2], i))
                i += 2; continue
            if ch != "!":
                tokens.append(Token(TType.OP_STR, ch, i))
                i += 1; continue
            # bare ! is not a token — fall through to error
            raise SpellError(f"Unexpected character '!' at position {i}")

        # Numbers (integers and floats, optional leading minus)
        if ch.isdigit() or (ch == "-" and i + 1 < n and source[i + 1].isdigit()):
            j = i + 1
            while j < n and (source[j].isdigit() or source[j] == "."):
                j += 1
            raw = source[i:j]
            val: Union[int, float] = float(raw) if "." in raw else int(raw)
            tokens.append(Token(TType.NUMBER, val, i))
            i = j; continue

        # Identifiers / keywords (letters, digits, underscores)
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            tokens.append(Token(TType.IDENT, word, i))
            i = j; continue

        # Anything else is silently skipped (defensive)
        i += 1

    tokens.append(Token(TType.EOF, None, n))
    return tokens


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

class SpellParser:
    """
    Parses a flat Token list into a SpellProgram AST.

    Statements may be delimited by commas (inline style) or simply appear
    in sequence separated by whitespace/newlines (both were stripped by
    the lexer — so we just parse until EOF or end of block).
    """

    def __init__(self, tokens: list[Token], source: str = ""):
        self._tokens = tokens
        self._pos = 0
        self._source = source

    # ── helpers ──────────────────────────────────────────────────────────

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.ttype != TType.EOF:
            self._pos += 1
        return tok

    def _expect(self, ttype: TType, value: Any = None) -> Token:
        tok = self._peek()
        if tok.ttype != ttype:
            raise SpellError(
                f"Expected {ttype.name} but got {tok.ttype.name}={tok.value!r} "
                f"at position {tok.pos}"
            )
        if value is not None and tok.value != value:
            raise SpellError(
                f"Expected {value!r} but got {tok.value!r} at position {tok.pos}"
            )
        return self._advance()

    def _match_ident(self, *names: str) -> bool:
        tok = self._peek()
        return tok.ttype == TType.IDENT and tok.value.upper() in names

    def _consume_ident(self, *names: str) -> Token:
        tok = self._peek()
        if tok.ttype != TType.IDENT or (names and tok.value.upper() not in names):
            raise SpellError(
                f"Expected one of {names} but got {tok.value!r} at {tok.pos}"
            )
        return self._advance()

    # ── entry ─────────────────────────────────────────────────────────────

    def parse(self) -> SpellProgram:
        stmts = self._parse_body()
        if self._peek().ttype != TType.EOF:
            tok = self._peek()
            raise SpellError(f"Unexpected token {tok.value!r} at position {tok.pos}")
        return SpellProgram(statements=stmts, source=self._source)

    # ── statement list (body of a block or top-level) ─────────────────────

    def _parse_body(self) -> list[Statement]:
        stmts: list[Statement] = []
        while self._peek().ttype not in (TType.EOF, TType.RBRACE):
            # Consume optional commas between statements
            while self._peek().ttype == TType.COMMA:
                self._advance()
            if self._peek().ttype in (TType.EOF, TType.RBRACE):
                break
            stmt = self._parse_stmt()
            if stmt is not None:
                stmts.append(stmt)
            # Consume trailing commas
            while self._peek().ttype == TType.COMMA:
                self._advance()
        return stmts

    def _parse_stmt(self) -> Optional[Statement]:
        tok = self._peek()
        if tok.ttype != TType.IDENT:
            raise SpellError(
                f"Expected statement keyword or operation at {tok.pos}, "
                f"got {tok.ttype.name}={tok.value!r}"
            )
        kw = tok.value.upper()

        if kw == "COST":
            return self._parse_cost()
        if kw == "FOR":
            return self._parse_for()
        if kw == "IF":
            return self._parse_if()
        if kw == "REPEAT":
            return self._parse_repeat()
        if kw == "PERSIST":
            return self._parse_persist()
        # Otherwise: operation effect statement
        return self._parse_effect()

    # ── individual statement parsers ──────────────────────────────────────

    def _parse_cost(self) -> CostStatement:
        self._consume_ident("COST")
        resource = self._expect(TType.IDENT).value
        amount_tok = self._expect(TType.NUMBER)
        return CostStatement(resource=resource, amount=float(amount_tok.value))

    def _parse_for(self) -> ForEachStatement:
        self._consume_ident("FOR")
        self._consume_ident("EACH")
        selector = self._parse_selector()
        self._expect(TType.LBRACE)
        body = self._parse_body()
        self._expect(TType.RBRACE)
        return ForEachStatement(selector=selector, body=body)

    def _parse_if(self) -> IfStatement:
        self._consume_ident("IF")
        condition = self._parse_condition()
        self._expect(TType.LBRACE)
        then_body = self._parse_body()
        self._expect(TType.RBRACE)
        else_body: list[Statement] = []
        if self._match_ident("ELSE"):
            self._advance()
            self._expect(TType.LBRACE)
            else_body = self._parse_body()
            self._expect(TType.RBRACE)
        return IfStatement(condition=condition, then_body=then_body, else_body=else_body)

    def _parse_repeat(self) -> RepeatStatement:
        self._consume_ident("REPEAT")
        count = int(self._expect(TType.NUMBER).value)
        self._expect(TType.LBRACE)
        body = self._parse_body()
        self._expect(TType.RBRACE)
        return RepeatStatement(count=max(1, count), body=body)

    def _parse_persist(self) -> PersistStatement:
        self._consume_ident("PERSIST")
        duration = int(self._expect(TType.NUMBER).value)
        self._expect(TType.LBRACE)
        body = self._parse_body()
        self._expect(TType.RBRACE)
        return PersistStatement(duration=max(1, duration), body=body)

    def _parse_effect(self) -> EffectStatement:
        op_tok = self._expect(TType.IDENT)
        op_name = op_tok.value.upper()

        # Collect arguments until we hit a selector keyword, brace,
        # comma, EOF, or another IDENT that looks like the next statement.
        args: list[Any] = []
        while True:
            tok = self._peek()
            if tok.ttype == TType.EOF:
                break
            if tok.ttype in (TType.LBRACE, TType.RBRACE, TType.COMMA):
                break
            if tok.ttype == TType.NUMBER:
                args.append(self._advance().value)
                continue
            if tok.ttype == TType.IDENT:
                kw = tok.value.upper()
                # Selector keyword → stop collecting args, parse selector
                if kw in _SELECTOR_KEYWORDS:
                    break
                # Language keyword that starts a new statement → stop
                if kw in _KEYWORDS and kw not in _SELECTOR_KEYWORDS:
                    break
                # Otherwise treat as string arg (e.g. tag name, property name)
                args.append(self._advance().value)
                continue
            break

        # Optional inline selector (HEAT 400 NEAR 3 ...)
        selector: Optional[Selector] = None
        if (self._peek().ttype == TType.IDENT
                and self._peek().value.upper() in _SELECTOR_KEYWORDS):
            selector = self._parse_selector()

        return EffectStatement(op_name=op_name, args=args, selector=selector)

    # ── selector ──────────────────────────────────────────────────────────

    def _parse_selector(self) -> Selector:
        tok = self._expect(TType.IDENT)
        kw = tok.value.upper()
        if kw == "SELF":
            return SelfSelector()
        if kw == "TARGET":
            return TargetSelector()
        if kw == "NEAR":
            radius = int(self._expect(TType.NUMBER).value)
            return NearSelector(radius=radius)
        if kw == "LINE":
            length = int(self._expect(TType.NUMBER).value)
            return LineSelector(length=length)
        if kw == "ALL":
            tag = self._expect(TType.IDENT).value
            return AllSelector(tag=tag)
        if kw == "TAGGED":
            tag = self._expect(TType.IDENT).value
            return TaggedSelector(tag=tag)
        raise SpellError(f"Unknown selector '{tok.value}' at position {tok.pos}")

    # ── condition ─────────────────────────────────────────────────────────

    def _parse_condition(self) -> Condition:
        tok = self._peek()
        if tok.ttype != TType.IDENT:
            raise SpellError(f"Expected condition keyword at {tok.pos}")
        kw = tok.value.upper()

        if kw == "NOT":
            self._advance()
            inner = self._parse_condition()
            return NotCondition(inner=inner)

        if kw == "HAS_TAG":
            self._advance()
            tag = self._expect(TType.IDENT).value
            return HasTagCondition(tag=tag)

        if kw == "PROP":
            self._advance()
            prop = self._expect(TType.IDENT).value
            op_tok = self._expect(TType.OP_STR)
            val = float(self._expect(TType.NUMBER).value)
            return PropCondition(prop=prop, op=op_tok.value, value=val)

        raise SpellError(
            f"Unknown condition keyword '{tok.value}' at position {tok.pos}"
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SpellExecutor:
    """
    Walks a SpellProgram AST, resolves selectors against the current world
    state, evaluates mana costs, and emits a list of Transitions.

    Parameters
    ----------
    actor_id    EntityId of the caster.
    target_id   EntityId of the currently targeted entity (may be None).
    world       The canonical WorldState (read-only).
    vocab       The world's SpellVocabulary (operations + grimoires).
    """

    def __init__(
        self,
        actor_id: EntityId,
        target_id: Optional[EntityId],
        world: WorldState,
        vocab: SpellVocabulary,
    ):
        self._actor_id = actor_id
        self._target_id = target_id
        self._world = world
        self._vocab = vocab
        self._transitions: list[Transition] = []
        self._mana_cost: float = 0.0

    # ── public API ───────────────────────────────────────────────────────

    def execute(self, program: SpellProgram) -> list[Transition]:
        """Execute *program* and return the resulting Transition list."""
        self._transitions = []
        self._mana_cost = 0.0

        actor = self._get_actor()
        if actor is None:
            raise SpellError("Caster not found in world.")

        # First pass: evaluate total mana cost
        self._mana_cost = self._compute_cost(program.statements)

        # Mana gate
        actor_mana = actor.meta.get("mana", actor.max_health)  # fallback
        if self._mana_cost > actor_mana:
            raise SpellError(
                f"Not enough mana. Spell costs {self._mana_cost:.0f} mana "
                f"but you only have {actor_mana:.0f}."
            )

        # Execute statements against the caster as initial selection
        initial_selection = [self._actor_id] if self._target_id is None else [self._target_id]
        self._exec_body(program.statements, initial_selection)

        # Deduct mana
        if self._mana_cost > 0:
            self._transitions.append(Transition(
                kind=TransitionKind.ENTITY_HEALTH_CHANGED,  # reuse existing; see apply below
                payload={
                    "entity_id": str(self._actor_id),
                    "delta": 0,                    # no health change
                    "mana_delta": -int(self._mana_cost),
                    "cause": "spell_cast",
                },
            ))

        # Spell cast record
        self._transitions.append(Transition(
            kind=TransitionKind.DIALOGUE_SPOKEN,  # repurposed as generic event record
            payload={
                "actor": str(self._actor_id),
                "target": str(self._target_id) if self._target_id else None,
                "verb": "cast",
                "spell_source": program.source[:120],
                "mana_cost": int(self._mana_cost),
                "effect_count": len(self._transitions),
            },
        ))

        return self._transitions

    # ── body execution ───────────────────────────────────────────────────

    def _exec_body(
        self,
        stmts: list[Statement],
        selection: list[EntityId],
    ) -> None:
        for stmt in stmts:
            self._exec_stmt(stmt, selection)

    def _exec_stmt(
        self,
        stmt: Statement,
        selection: list[EntityId],
    ) -> None:
        if isinstance(stmt, CostStatement):
            return  # already handled in _compute_cost

        if isinstance(stmt, ForEachStatement):
            targets = self._resolve_selector(stmt.selector, selection)
            for eid in targets:
                self._exec_body(stmt.body, [eid])
            return

        if isinstance(stmt, IfStatement):
            for eid in selection:
                entity = self._get_entity(eid)
                if entity is None:
                    continue
                if self._eval_condition(stmt.condition, entity):
                    self._exec_body(stmt.then_body, [eid])
                elif stmt.else_body:
                    self._exec_body(stmt.else_body, [eid])
            return

        if isinstance(stmt, RepeatStatement):
            for _ in range(stmt.count):
                self._exec_body(stmt.body, selection)
            return

        if isinstance(stmt, PersistStatement):
            # Record a condition on the caster that the WorldClock can trigger.
            # The full program source is stored in the condition name so it
            # can be re-executed — simplified to just applying the body once
            # now and relying on world_clock to re-apply.
            self._exec_body(stmt.body, selection)
            # Store the persist trigger on the caster
            self._transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": str(self._actor_id),
                    "condition": f"spell_persist",
                    "ticks": stmt.duration,
                },
            ))
            return

        if isinstance(stmt, EffectStatement):
            targets = (
                self._resolve_selector(stmt.selector, selection)
                if stmt.selector is not None
                else selection
            )
            for eid in targets:
                self._exec_effect(stmt, eid)
            return

    # ── effect execution ─────────────────────────────────────────────────

    def _exec_effect(self, stmt: EffectStatement, target_id: EntityId) -> None:
        op = self._vocab.operations.get(stmt.op_name)
        if op is None:
            raise SpellError(
                f"Unknown operation '{stmt.op_name}'. "
                f"Read a grimoire to learn this world's operations."
            )

        entity = self._get_entity(target_id)
        if entity is None:
            return

        # requires_tag guard (flammable OR material ignition path for IGNITE)
        if op.requires_tag and op.requires_tag not in entity.tags:
            from .substance import can_ignite as _can_ignite

            cfg = self._world.config.physics_config
            if not (
                op.requires_tag == "flammable"
                and _can_ignite(entity, cfg)
            ):
                return

        # Bind positional args to param names
        bound = self._bind_args(op, stmt.args)

        prim = op.primitive

        if prim == "ADJUST_PROP":
            prop = op.prop
            amount = bound.get("n", bound.get("amount", 0)) if bound else (
                stmt.args[0] if stmt.args else 0
            )
            actual_delta = op.sign * float(amount)
            cause = f"spell_{stmt.op_name.lower()}"

            if prop == "health":
                self._transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": str(target_id),
                        "delta": int(actual_delta),
                        "cause": cause,
                        "actor": str(self._actor_id),
                    },
                ))
            elif prop == "temperature":
                from .substance import apply_temperature_delta

                cfg = self._world.config.physics_config
                ent = self._get_entity(target_id)
                applied = actual_delta
                if ent is not None and cfg is not None:
                    applied = apply_temperature_delta(ent, actual_delta, cfg)
                self._transitions.append(Transition(
                    kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                    payload={
                        "entity_id": str(target_id),
                        "delta": 0,
                        "prop_delta": {prop: applied},
                        "cause": cause,
                        "actor": str(self._actor_id),
                    },
                ))
            else:
                    self._transitions.append(Transition(
                        kind=TransitionKind(TransitionKind.ENTITY_HEALTH_CHANGED.value),
                        payload={
                            "entity_id": str(target_id),
                            "delta": 0,
                            "prop_delta": {prop: actual_delta},
                            "cause": f"spell_{stmt.op_name.lower()}",
                            "actor": str(self._actor_id),
                        },
                    ))

        elif prim == "SET_PROP":
            prop = op.prop or (stmt.args[0] if stmt.args else "unknown")
            value = stmt.args[1] if len(stmt.args) > 1 else bound.get("value", 0)
            self._transitions.append(Transition(
                kind=TransitionKind.ENTITY_HEALTH_CHANGED,
                payload={
                    "entity_id": str(target_id),
                    "delta": 0,
                    "prop_set": {prop: value},
                    "cause": f"spell_{stmt.op_name.lower()}",
                    "actor": str(self._actor_id),
                },
            ))

        elif prim == "SET_EMOTIONAL":
            state_str = op.state or "neutral"
            try:
                new_state = EmotionalState(state_str)
            except ValueError:
                new_state = EmotionalState.NEUTRAL
            self._transitions.append(Transition(
                kind=TransitionKind.ENTITY_EMOTIONAL_STATE_CHANGED,
                payload={
                    "entity_id": str(target_id),
                    "from": entity.emotional_state.value,
                    "to": new_state.value,
                    "cause": f"spell_{stmt.op_name.lower()}",
                    "actor": str(self._actor_id),
                },
            ))

        elif prim == "SET_CONDITION":
            cond_name = op.condition or stmt.op_name.lower()
            duration = int(bound.get("n", bound.get("duration", 1)))
            self._transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": str(target_id),
                    "condition": cond_name,
                    "ticks": duration,
                },
            ))

        elif prim == "ADD_TAG":
            tag = op.tag or (str(stmt.args[0]) if stmt.args else "tagged")
            if tag == "on_fire":
                from .substance import (
                    can_ignite,
                    get_temperature,
                    queue_impulse,
                    resolve_material,
                )

                cfg = self._world.config.physics_config
                if not can_ignite(entity, cfg):
                    return
                mat = resolve_material(entity, cfg)
                if mat and mat.ignition_point is not None:
                    temp = get_temperature(entity, cfg)
                    if temp < mat.ignition_point:
                        queue_impulse(
                            self._world,
                            str(target_id),
                            "temperature",
                            mat.ignition_point - temp + 20.0,
                            source="spell_ignite",
                        )
                        return
            self._transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": str(target_id),
                    "condition": f"tag_{tag}",
                    "ticks": 9999,      # permanent until removed
                    "_tag_add": tag,    # handled by apply_spell_transitions
                },
            ))

        elif prim == "REMOVE_TAG":
            tag = op.tag or (str(stmt.args[0]) if stmt.args else "")
            self._transitions.append(Transition(
                kind=TransitionKind.ENTITY_CONDITION_CHANGED,
                payload={
                    "entity_id": str(target_id),
                    "condition": f"tag_{tag}",
                    "ticks": 0,          # ticks=0 = remove
                    "_tag_remove": tag,
                },
            ))

        elif prim == "MOVE_AWAY":
            distance = int(bound.get("n", bound.get("distance", 1)))
            dest = self._push_target(entity, distance, away=True)
            if dest:
                self._transitions.append(Transition(
                    kind=TransitionKind.ENTITY_MOVED,
                    payload={
                        "entity_id": str(target_id),
                        "from": {"x": entity.position.x, "y": entity.position.y},
                        "to":   {"x": dest.x, "y": dest.y},
                    },
                ))

        elif prim == "MOVE_TOWARD":
            distance = int(bound.get("n", bound.get("distance", 1)))
            dest = self._push_target(entity, distance, away=False)
            if dest:
                self._transitions.append(Transition(
                    kind=TransitionKind.ENTITY_MOVED,
                    payload={
                        "entity_id": str(target_id),
                        "from": {"x": entity.position.x, "y": entity.position.y},
                        "to":   {"x": dest.x, "y": dest.y},
                    },
                ))

        elif prim == "LINK_EDGE":
            edge_kind_str = op.edge_kind or "ally_of"
            weight = float(bound.get("n", 0.5))
            self._transitions.append(Transition(
                kind=TransitionKind.EDGE_CREATED,
                payload={
                    "source": str(self._actor_id),
                    "target": str(target_id),
                    "edge_kind": edge_kind_str,
                    "weight": max(-1.0, min(1.0, weight)),
                },
            ))

    # ── selector resolution ───────────────────────────────────────────────

    def _resolve_selector(
        self,
        selector: Selector,
        current_selection: list[EntityId],
    ) -> list[EntityId]:
        grid = self._world.spatial
        actor = self._get_actor()
        if actor is None:
            return []

        if isinstance(selector, SelfSelector):
            return [self._actor_id]

        if isinstance(selector, TargetSelector):
            return [self._target_id] if self._target_id else []

        if isinstance(selector, NearSelector):
            result = []
            for eid, ent in grid.entities.items():
                if ent.alive and actor.position.manhattan(ent.position) <= selector.radius:
                    result.append(eid)
            return result

        if isinstance(selector, LineSelector):
            # Cast a ray from actor in their facing direction
            dx, dy = actor.facing.vector
            result = []
            for step in range(1, selector.length + 1):
                cx = actor.position.x + dx * step
                cy = actor.position.y + dy * step
                coord = Coord(x=cx, y=cy)
                for eid, ent in grid.entities.items():
                    if ent.alive and ent.position == coord:
                        result.append(eid)
                # Stop at walls
                if not grid.is_transparent(coord):
                    break
            return result

        if isinstance(selector, (AllSelector, TaggedSelector)):
            tag = selector.tag
            return [
                eid for eid, ent in grid.entities.items()
                if ent.alive and tag in ent.tags
            ]

        return list(current_selection)

    # ── condition evaluation ──────────────────────────────────────────────

    def _eval_condition(self, cond: Condition, entity: EntityState) -> bool:
        if isinstance(cond, HasTagCondition):
            return cond.tag in entity.tags

        if isinstance(cond, PropCondition):
            # Check entity.meta first, then known scalar fields
            value = entity.meta.get(cond.prop)
            if value is None:
                value = getattr(entity, cond.prop, None)
            if value is None:
                return False
            try:
                fval = float(value)
            except (TypeError, ValueError):
                return False
            op = cond.op
            if op == ">":   return fval > cond.value
            if op == "<":   return fval < cond.value
            if op == ">=":  return fval >= cond.value
            if op == "<=":  return fval <= cond.value
            if op == "==":  return fval == cond.value
            if op == "!=":  return fval != cond.value
            return False

        if isinstance(cond, NotCondition):
            return not self._eval_condition(cond.inner, entity)

        return False

    # ── helpers ───────────────────────────────────────────────────────────

    def _get_actor(self) -> Optional[EntityState]:
        return self._world.spatial.entities.get(self._actor_id)

    def _get_entity(self, eid: EntityId) -> Optional[EntityState]:
        return self._world.spatial.entities.get(eid)

    def _push_target(
        self, entity: EntityState, distance: int, *, away: bool
    ) -> Optional[Coord]:
        actor = self._get_actor()
        if actor is None:
            return None
        dx = entity.position.x - actor.position.x
        dy = entity.position.y - actor.position.y
        # Normalise
        length = max(1, abs(dx) + abs(dy))
        ndx = (dx if away else -dx) // length
        ndy = (dy if away else -dy) // length
        grid = self._world.spatial
        best = entity.position
        for step in range(1, distance + 1):
            candidate = Coord(
                x=entity.position.x + ndx * step,
                y=entity.position.y + ndy * step,
            )
            if grid.is_in_bounds(candidate) and grid.tile_at(candidate).passable:
                best = candidate
            else:
                break
        return best if best != entity.position else None

    @staticmethod
    def _bind_args(op: OpDefinition, args: list) -> dict:
        """Map positional args to the operation's named parameter list."""
        bound: dict = {}
        for i, param in enumerate(op.args):
            if i < len(args):
                bound[param] = args[i]
        # Also expose as n, m for formula evaluation
        if args:
            bound["n"] = args[0]
        if len(args) > 1:
            bound["m"] = args[1]
        return bound

    def _compute_cost(self, stmts: list[Statement]) -> float:
        """Walk statements looking for COST declarations."""
        total = 0.0
        for stmt in stmts:
            if isinstance(stmt, CostStatement) and stmt.resource == "mana":
                total += stmt.amount
            elif isinstance(stmt, (ForEachStatement, IfStatement,
                                   RepeatStatement, PersistStatement)):
                body = (stmt.body if not isinstance(stmt, IfStatement)
                        else stmt.then_body + stmt.else_body)
                total += self._compute_cost(body)
        return total


# ---------------------------------------------------------------------------
# Helper: simple formula evaluator (safe subset of Python eval)
# ---------------------------------------------------------------------------

def _eval_formula(formula: str, bindings: dict) -> float:
    """
    Evaluate a mana-cost formula string with variable bindings.
    Only arithmetic is allowed — no function calls, no imports.
    Returns 0.0 on any error.
    """
    try:
        # Allow only numbers and the bound variable names
        safe_names = {k: float(v) for k, v in bindings.items()
                      if isinstance(v, (int, float))}
        return float(eval(formula, {"__builtins__": {}}, safe_names))  # noqa: S307
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# apply_spell_transitions — extend apply_transitions to handle spell payloads
# ---------------------------------------------------------------------------

def apply_spell_side_effects(
    world: WorldState,
    transition: Transition,
) -> None:
    """
    Called from compiler.apply_transitions for ENTITY_HEALTH_CHANGED
    transitions that carry extra spell payload keys (prop_delta, prop_set,
    mana_delta).  These are piggy-backed on ENTITY_HEALTH_CHANGED to reuse
    the existing infrastructure.

    This function is imported and called inside apply_transitions when the
    payload has spell-specific keys.
    """
    p = transition.payload
    grid = world.spatial

    eid = EntityId(p["entity_id"])
    entity = grid.entities.get(eid)
    if entity is None:
        return

    # Mana delta
    mana_delta = p.get("mana_delta")
    if mana_delta is not None:
        current = entity.meta.get("mana", entity.max_health)
        new_val = max(0, current + int(mana_delta))
        entity.meta["mana"] = new_val

    # Arbitrary property delta (e.g. temperature)
    prop_delta = p.get("prop_delta")
    if prop_delta:
        for prop, delta in prop_delta.items():
            current = entity.meta.get(prop, 0.0)
            entity.meta[prop] = current + float(delta)

    # Arbitrary property set
    prop_set = p.get("prop_set")
    if prop_set:
        for prop, value in prop_set.items():
            entity.meta[prop] = value

    # Tag add/remove (from ENTITY_CONDITION_CHANGED with _tag_add/_tag_remove)
    tag_add = p.get("_tag_add")
    if tag_add and tag_add not in entity.tags:
        entity.tags.append(tag_add)

    tag_remove = p.get("_tag_remove")
    if tag_remove and tag_remove in entity.tags:
        entity.tags.remove(tag_remove)


# ---------------------------------------------------------------------------
# Public convenience: compile_spell_text
# ---------------------------------------------------------------------------

def compile_spell_text(
    source: str,
    actor_id: EntityId,
    target_id: Optional[EntityId],
    world: WorldState,
    vocab: SpellVocabulary,
) -> list[Transition]:
    """
    Full pipeline: source text → Transitions.

    Raises SpellError on parse or execution failure.
    """
    tokens = _lex(source)
    parser = SpellParser(tokens, source=source)
    program = parser.parse()
    executor = SpellExecutor(
        actor_id=actor_id,
        target_id=target_id,
        world=world,
        vocab=vocab,
    )
    return executor.execute(program)
