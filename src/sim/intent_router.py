"""
Deterministic intent pre-parsers for player input.

Routes unambiguous REPL commands (movement, speech, items, spells) to
SemanticActions before the LM sees them.  Shared by GameLoop and play_client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import (
    ActionType,
    Coord,
    EntityId,
    EntityState,
    SemanticAction,
    SpatialGrid,
    WorldState,
)

# ---------------------------------------------------------------------------
# Orientation pre-processor
# ---------------------------------------------------------------------------

# Eight compass directions + their common abbreviations.
_COMPASS_WORDS: frozenset[str] = frozenset({
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "n", "s", "e", "w", "ne", "nw", "se", "sw",
    "up", "down",
})

# Relative-turn words that should become a TURN action with the word as target.
_RELATIVE_TURN_WORDS: frozenset[str] = frozenset({
    "left", "right", "around", "back", "backward", "reverse",
})


def try_parse_orientation(
    raw_input: str,
    actor_id: "EntityId",
) -> Optional["SemanticAction"]:
    """
    Intercept orientation intents before the LM sees them.

    Patterns matched (case-insensitive):
      turn [direction|left|right|around]
      face [direction]
      look [direction]          ← only when followed by compass word
      rotate [left|right]

    Returns a SemanticAction(verb="turn") or None if no match.
    """
    import re
    tokens = re.split(r"\s+", raw_input.lower().strip())
    if not tokens:
        return None

    first = tokens[0]
    rest = " ".join(tokens[1:]).strip() if len(tokens) > 1 else ""

    # Verbs that unambiguously mean "change facing"
    if first in ("turn", "rotate", "face"):
        if not rest:
            return None          # "turn" alone — ambiguous, let LM handle
        return SemanticAction(
            verb=ActionType.TURN,
            actor=actor_id,
            target=rest,
            raw_input=raw_input,
        )

    # "look north/east/…" → TURN (but NOT "look at guard" — that's OBSERVE)
    if first == "look" and rest in _COMPASS_WORDS:
        return SemanticAction(
            verb=ActionType.TURN,
            actor=actor_id,
            target=rest,
            raw_input=raw_input,
        )

    return None




_PROFANITY = frozenset({
    "fuck", "shit", "ass", "bitch", "bastard", "damn", "hell", "crap",
    "piss", "cock", "dick", "cunt", "motherfucker", "fucker", "fuckers",
    "bullshit", "asshole", "dumbass",
})


def try_parse_speech_command(
    raw: str,
    actor_id: "EntityId",
    world: "WorldState",
) -> "Optional[SemanticAction]":
    """
    Intercept explicit speech commands before the LM sees them, so the
    player's actual words are always preserved verbatim.

    Handled patterns (case-insensitive):
      say: <text>                     → speak to nearest entity
      say to <name>: <text>           → speak to named entity
      tell <name>: <text>             → speak to named entity
      ask <name>: <text> / about <X>  → ask named entity
      whisper [to] <name>: <text>     → whisper to named entity
      announce: <text>                → announce (no specific target)

    The player's literal <text> is stored in intent.manner so the
    narrator always shows exactly what was typed.
    """
    import re as _re
    from .schemas import IntentBlock as _IB

    stripped = raw.strip()

    # Pattern: verb [to|at] [<name>] [:|;|-] <text>
    #   group 1 = verb
    #   group 2 = optional target name (may be empty)
    #   group 3 = spoken text
    m = _re.match(
        r"^(say|tell|ask|whisper|announce|declare|shout|yell|call|mutter|state|claim|reply|answer)"
        r"(?:\s+(?:to|at)\s+)?"
        r"(?P<target>[A-Za-z][\w\s]{0,30}?)?"
        r"\s*[:;—–\-]\s*"
        r"(?P<text>.+)$",
        stripped,
        flags=_re.IGNORECASE,
    )
    if not m:
        return None

    verb_word = m.group(1).lower()
    target_raw = (m.group("target") or "").strip()
    spoken_text = m.group("text").strip().strip('"\'')

    if not spoken_text:
        return None

    # Map verb word to engine verb
    _VERB_MAP = {
        "say": "speak", "tell": "speak", "state": "speak", "claim": "speak",
        "reply": "speak", "answer": "speak", "announce": "speak",
        "declare": "speak", "call": "speak",
        "ask": "ask", "whisper": "whisper", "mutter": "mutter",
        "shout": "yell", "yell": "yell",
    }
    engine_verb = _VERB_MAP.get(verb_word, "speak")

    # Resolve target entity
    target_id = None
    if target_raw:
        target_id = _resolve_entity_name_by_name(target_raw, world.spatial)

    # Auto-target nearest visible entity if no explicit target
    if target_id is None and engine_verb != "yell":
        player_ent = world.spatial.entities.get(actor_id)
        if player_ent:
            best_id: "Optional[EntityId]" = None
            best_dist = 9999
            for eid, ent in world.spatial.entities.items():
                if eid == actor_id or not ent.alive:
                    continue
                d = player_ent.position.manhattan(ent.position)
                if d < best_dist:
                    best_dist = d
                    best_id = eid
            if best_id is not None and best_dist <= 12:
                target_id = best_id

    return SemanticAction(
        action_id=f"speech_{world.tick}",
        actor=actor_id,
        verb=engine_verb,
        target=target_id,
        intent=_IB(
            rationale=spoken_text,
            manner=spoken_text,         # verbatim text → used by narrator
            desired_outcome=[],
        ),
        raw_input=raw,
    )


def try_parse_shout(
    raw: str,
    actor_id: "EntityId",
    world: "WorldState",
) -> Optional[SemanticAction]:
    """
    Catch all-caps utterances and profanity bursts and route them as yell,
    regardless of what the LM would have labeled them.

    Conditions for a shout:
      • ≥ 3 consecutive capital-letter words, OR
      • Any profanity word present in the raw input.

    The entire raw text becomes the yell content so NPCs can hear what
    was actually said.
    """
    import re as _re
    from .schemas import IntentBlock as _IB
    stripped = raw.strip()
    # Strip leading verb prefix like "Yell: ", "Shout; ", "Say — " etc.
    cleaned = _re.sub(
        r"^(?:yell|shout|scream|say|shouts?|yells?|screams?)\s*[:\-;—–]\s*[\"']?",
        "",
        stripped,
        flags=_re.IGNORECASE,
    ).strip().strip('"\'')
    if not cleaned:
        cleaned = stripped
    words = stripped.split()
    if not words:
        return None

    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    has_profanity = any(w.lower().strip("!?.,'\"") in _PROFANITY for w in words)

    if caps_words >= 3 or has_profanity:
        return SemanticAction(
            action_id=f"shout_{world.tick}",
            actor=actor_id,
            verb="yell",
            intent=_IB(
                rationale=cleaned,
                manner=cleaned,
                desired_outcome=[],
            ),
            raw_input=raw,
        )
    return None


def try_parse_directional_move(
    raw_input: str,
    actor_id: "EntityId",
    actor: "EntityState",
) -> "Optional[SemanticAction]":
    """
    Intercept distance-qualified movement commands before the LM sees them.

    Matched patterns (case-insensitive):
      move/go/walk/run/step/march/stride/head [direction] [N] [tiles?]
      advance/retreat/rush [N] [tiles?]
      advance/retreat/rush [direction] [N] [tiles?]
      [direction] [N] [tiles?]   ← bare compass + number

    Direction may be a compass word (north, ne, …) or a relative word
    (forward, backward, left, right).

    Returns a SemanticAction(verb=MOVE, target=Coord) or None on no match.
    """
    import re

    # Pure motion verbs with no implied direction
    _MOVE_VERBS = {
        "move", "go", "walk", "run", "step", "march", "stride", "head",
        "hurry", "dash", "sprint", "travel",
    }
    # Motion verbs that also imply a direction relative to current facing
    _DIRECTIONAL_VERBS: dict[str, tuple[int, int]] = {
        "advance": (1, 0), "rush": (1, 0), "charge": (1, 0),
        "retreat": (-1, 0), "withdraw": (-1, 0), "back": (-1, 0),
    }
    _REL_DIR_MAP = {
        "forward": (1, 0), "ahead": (1, 0),
        "backward": (-1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    text = raw_input.lower().strip()
    # Strip trailing filler words to simplify the rest of the parsing
    text = re.sub(r"\b(tiles?|steps?|spaces?|squares?|units?)\s*$", "", text).strip()

    # Tokenize into words
    tokens = text.split()
    if not tokens:
        return None

    direction_str: Optional[str] = None
    distance: Optional[int] = None
    implied_rel: Optional[tuple[int, int]] = None

    # Try to pick off the first movement-verb or directional-verb
    i = 0
    if tokens[i] in _DIRECTIONAL_VERBS:
        # e.g. "retreat 3", "advance 2 tiles"
        implied_rel = _DIRECTIONAL_VERBS[tokens[i]]
        i += 1
    elif tokens[i] in _MOVE_VERBS:
        i += 1

    # After the optional verb, expect [direction] [N] or [N] [direction]
    # Peek at the remaining tokens to find direction + number in either order
    remaining = tokens[i:]
    if not remaining:
        return None

    _all_dirs = _COMPASS_WORDS | set(_REL_DIR_MAP)
    dir_tok: Optional[str] = None
    num_tok: Optional[int] = None
    for tok in remaining:
        if tok.isdigit() and num_tok is None:
            num_tok = int(tok)
        elif dir_tok is None and tok in _all_dirs:
            dir_tok = tok

    if num_tok is None or num_tok <= 0:
        return None          # no distance found — let the normal path handle it
    if dir_tok is None and implied_rel is None:
        # Distance but no direction and no directional verb → default to "forward"
        dir_tok = "forward"

    direction_str = dir_tok
    distance = num_tok

    # Resolve direction to absolute (dx, dy)
    from .schemas import FACING_VECTORS, FacingDirection
    if implied_rel is not None and direction_str is None:
        # Directional verb ("retreat", "advance") with no explicit direction word
        rel_fwd, rel_lat = implied_rel
        fx, fy = actor.facing.vector
        lft_x, lft_y = -fy, fx
        dx = rel_fwd * fx + rel_lat * lft_x
        dy = rel_fwd * fy + rel_lat * lft_y
    elif direction_str is not None and direction_str in _REL_DIR_MAP:
        # Relative direction: compute from actor facing
        rel_fwd, rel_lat = _REL_DIR_MAP[direction_str]
        fx, fy = actor.facing.vector
        lft_x, lft_y = -fy, fx
        dx = rel_fwd * fx + rel_lat * lft_x
        dy = rel_fwd * fy + rel_lat * lft_y
    else:
        # Compass abbreviation → look up canonical FacingDirection
        aliases = {
            "n": "north", "s": "south", "e": "east", "w": "west",
            "ne": "northeast", "nw": "northwest",
            "se": "southeast", "sw": "southwest",
        }
        canonical = aliases.get(direction_str, direction_str)
        try:
            fd = FacingDirection(canonical)
        except ValueError:
            return None
        dx, dy = fd.vector

    target = Coord(
        x=actor.position.x + dx * distance,
        y=actor.position.y + dy * distance,
    )
    return SemanticAction(
        verb=ActionType.MOVE,
        actor=actor_id,
        target=target,
        raw_input=raw_input,
    )


def try_parse_compass_step(
    raw_input: str,
    actor_id: "EntityId",
    actor: "EntityState",
    grid: Optional["SpatialGrid"] = None,
) -> Optional[SemanticAction]:
    """
    One-tile compass moves without a distance: ``move north``, ``go e``, ``n``.

    The distance parser requires an explicit number; HUD buttons and casual
    commands use this fast path so play stays instant (no LM round-trip).

    When ``grid.use_voxels`` (or ``depth > 1``), ``up`` / ``down`` move one
    vertical level.
    """
    import re

    text = raw_input.lower().strip()
    text = re.sub(r"\b(tiles?|steps?|spaces?|squares?)\s*$", "", text).strip()
    tokens = text.split()
    if not tokens:
        return None

    move_verbs = {
        "move", "go", "walk", "run", "step", "head", "travel", "stride",
        "climb", "ascend", "descend",
    }
    i = 0
    if tokens[i] in move_verbs:
        i += 1
    if i >= len(tokens):
        return None

    direction_str = tokens[i]
    if direction_str not in _COMPASS_WORDS:
        return None
    if i + 1 < len(tokens) and tokens[i + 1].isdigit():
        return None  # let try_parse_directional_move handle "move north 3"

    if direction_str in ("up", "down") and grid is not None:
        if grid.use_voxels or grid.depth > 1:
            dz = 1 if direction_str == "up" else -1
            target = Coord(
                x=actor.position.x,
                y=actor.position.y,
                z=actor.position.z + dz,
            )
            if grid.is_in_bounds(target) and grid.is_passable(target):
                return SemanticAction(
                    verb=ActionType.MOVE,
                    actor=actor_id,
                    target=target,
                    raw_input=raw_input,
                )
        return None

    aliases = {
        "n": "north", "s": "south", "e": "east", "w": "west",
        "ne": "northeast", "nw": "northwest",
        "se": "southeast", "sw": "southwest",
    }
    from .schemas import FacingDirection
    canonical = aliases.get(direction_str, direction_str)
    try:
        fd = FacingDirection(canonical)
    except ValueError:
        return None
    dx, dy = fd.vector
    target = Coord(
        x=actor.position.x + dx,
        y=actor.position.y + dy,
        z=actor.position.z,
    )
    return SemanticAction(
        verb=ActionType.MOVE,
        actor=actor_id,
        target=target,
        raw_input=raw_input,
    )


def try_parse_trivial_actions(
    raw_input: str,
    actor_id: "EntityId",
) -> Optional[SemanticAction]:
    """Instant wait / observe — never call the LM for these."""
    lower = raw_input.lower().strip()
    if lower in ("wait", "w", "pass", "skip", "do nothing", "rest here"):
        return SemanticAction(
            verb=ActionType.WAIT,
            actor=actor_id,
            raw_input=raw_input,
        )
    if lower in (
        "observe", "look around", "look", "survey", "scan",
        "look around.", "look.",
    ):
        return SemanticAction(
            verb=ActionType.OBSERVE,
            actor=actor_id,
            raw_input=raw_input,
        )
    return None


def try_parse_item_command(
    raw_input: str,
    actor_id: "EntityId",
    actor: "Optional[EntityState]",
    world: "WorldState",
) -> "Optional[SemanticAction]":
    """
    Intercept explicit item interaction commands before the LM sees them.

    Handled patterns:
      drop <item>               → DROP verb, item name in intent.rationale
      use <item>                → USE verb, item name in intent.rationale
      use <item> on <target>    → USE verb, item name in rationale, target in manner
      mix <A> with <B>          → MIX verb, A in rationale, B in manner
      combine <A> and <B>       → MIX verb, A in rationale, B in manner
      throw <item> at <target>  → THROW verb, item name in rationale, target resolved
      inventory / inv / items   → lists inventory (pure side-effect, returns a stub)

    Returns a SemanticAction if handled, None otherwise.
    """
    import re
    if actor is None:
        return None

    raw = raw_input.strip()
    lower = raw.lower()

    # ── inventory listing ────────────────────────────────────────────────
    if lower in ("inventory", "inv", "items", "i"):
        grid = world.spatial
        if not actor.inventory:
            _item_print("You are carrying nothing.")
            return None
        lines = ["You are carrying:"]
        for oid in actor.inventory:
            obj = grid.objects.get(oid)
            if obj:
                tags_str = ", ".join(obj.tags[:4]) if obj.tags else "no tags"
                lines.append(f"  • {obj.name}  [{tags_str}]  {obj.weight:.1f} kg")
        _item_print("\n".join(lines))
        return None

    # ── drop <item> ──────────────────────────────────────────────────────
    m = re.match(r"^drop\s+(.+)$", lower)
    if m:
        item_name = raw[5:].strip()
        from .schemas import IntentBlock as _IB
        return SemanticAction(
            action_id=f"item_drop_{world.tick}",
            actor=actor_id,
            verb=ActionType.DROP,
            intent=_IB(
                rationale=item_name,
                desired_outcome=[],
            ),
            raw_input=raw,
        )

    # ── use <item> [on <target>] ─────────────────────────────────────────
    m = re.match(r"^use\s+(.+?)\s+on\s+(.+)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        item_name = raw[4:raw.lower().find(" on ")].strip()
        target_name = raw[raw.lower().find(" on ") + 4:].strip()
        target_id = _resolve_entity_name_by_name(target_name, world.spatial)
        return SemanticAction(
            action_id=f"item_use_{world.tick}",
            actor=actor_id,
            verb=ActionType.USE,
            target=target_id,
            intent=_IB(
                rationale=item_name,
                manner=target_name,
                desired_outcome=[],
            ),
            raw_input=raw,
        )

    m_use = re.match(r"^use\s+(.+)$", lower)
    if m_use:
        from .schemas import IntentBlock as _IB
        item_name = raw[4:].strip()
        if item_name:
            return SemanticAction(
                action_id=f"item_use_{world.tick}",
                actor=actor_id,
                verb=ActionType.USE,
                intent=_IB(
                    rationale=item_name,
                    desired_outcome=[],
                ),
                raw_input=raw,
            )

    # ── mix <A> with <B>  /  combine <A> and <B> ────────────────────────
    m = re.match(r"^mix\s+(.+?)\s+with\s+(.+)$", lower)
    if not m:
        m = re.match(r"^combine\s+(.+?)\s+and\s+(.+)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        sep = " with " if " with " in lower else " and "
        a_name = raw[raw.lower().find(m.group(1)):
                     raw.lower().find(m.group(1)) + len(m.group(1))].strip()
        b_start = raw.lower().find(sep) + len(sep)
        b_name = raw[b_start:].strip()
        return SemanticAction(
            action_id=f"item_mix_{world.tick}",
            actor=actor_id,
            verb=ActionType.MIX,
            intent=_IB(
                rationale=a_name,
                manner=b_name,
                desired_outcome=[],
            ),
            raw_input=raw,
        )

    # ── throw <item> at <target> ─────────────────────────────────────────
    m = re.match(r"^throw\s+(.+?)\s+at\s+(.+)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        item_part = raw[6:raw.lower().find(" at ")].strip()
        target_part = raw[raw.lower().find(" at ") + 4:].strip()
        target_id = _resolve_entity_name_by_name(target_part, world.spatial)
        return SemanticAction(
            action_id=f"item_throw_{world.tick}",
            actor=actor_id,
            verb=ActionType.THROW,
            target=target_id,
            intent=_IB(
                rationale=item_part,
                manner="throw",
                desired_outcome=[],
            ),
            raw_input=raw,
        )

    # ── examine / inspect <target> ───────────────────────────────────────
    m = re.match(r"^(?:examine|inspect|look at)\s+(.+)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        target_name = raw[raw.lower().index(m.group(1)):].strip()
        target_id = _resolve_entity_name_by_name(m.group(1), world.spatial)
        return SemanticAction(
            action_id=f"examine_{world.tick}",
            actor=actor_id,
            verb=ActionType.EXAMINE,
            target=target_id or m.group(1),
            intent=_IB(rationale=target_name, desired_outcome=[]),
            raw_input=raw,
        )

    # ── trade / buy / sell ───────────────────────────────────────────────
    # Supported patterns:
    #   trade <item> for <N> gold [from/to <target>]
    #   trade <N> gold for <item> [from <target>]
    #   buy <item> from <target>
    #   sell <item> to <target>
    m_trade = re.match(
        r"^(?:trade|buy|sell)\s+(.+?)\s+(?:for|from|to)\s+(.+)$", lower
    )
    if m_trade:
        from .schemas import IntentBlock as _IB
        # Find the target entity from the tail of the sentence
        raw_tail = m_trade.group(2)
        # Check if tail ends with "from <entity>" or "to <entity>"
        target_match = re.search(r"\b(?:from|to)\s+(\w[\w\s]*)$", raw_tail)
        target_id = None
        target_name_part = ""
        if target_match:
            target_name_part = target_match.group(1).strip()
            target_id = _resolve_entity_name_by_name(target_name_part, world.spatial)

        offer_part = m_trade.group(1).strip()
        want_part = raw_tail[:target_match.start()].strip() if target_match else raw_tail.strip()

        # Detect gold amounts — "5 gold" in either slot
        gold_re = re.compile(r"(\d+)\s*gold")
        gm_offer = gold_re.search(offer_part)
        gm_want = gold_re.search(want_part)

        if gm_offer:
            offer_str = f"gold:{gm_offer.group(1)}"
            want_str = want_part if not gm_want else f"gold:{gm_want.group(1)}"
        else:
            offer_str = offer_part
            want_str = f"gold:{gm_want.group(1)}" if gm_want else want_part

        return SemanticAction(
            action_id=f"trade_{world.tick}",
            actor=actor_id,
            verb=ActionType.TRADE,
            target=target_id or (target_name_part or None),
            intent=_IB(
                rationale=offer_str,
                manner=want_str,
                desired_outcome=["trade"],
            ),
            raw_input=raw,
        )

    # ── steal / pickpocket <item> from <target> ──────────────────────────
    m = re.match(r"^(?:steal|pickpocket)\s+(.+?)\s+from\s+(.+)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        item_name = m.group(1).strip()
        target_name = m.group(2).strip()
        target_id = _resolve_entity_name_by_name(target_name, world.spatial)
        return SemanticAction(
            action_id=f"steal_{world.tick}",
            actor=actor_id,
            verb=ActionType.STEAL,
            target=target_id or target_name,
            intent=_IB(rationale=item_name, desired_outcome=[]),
            raw_input=raw,
        )

    # ── rest / sleep / wait here ─────────────────────────────────────────
    if lower in ("rest", "sleep", "wait here", "take a rest", "sit down"):
        return SemanticAction(
            action_id=f"rest_{world.tick}",
            actor=actor_id,
            verb=ActionType.REST,
            raw_input=raw,
        )

    # ── craft / make / forge / build <thing> [from <materials>] ─────────
    m = re.match(
        r"^(?:craft|make|build|forge|whittle|fabricate|construct)\s+(.+?)(?:\s+from\s+(.+))?$",
        lower,
    )
    if m:
        from .schemas import IntentBlock as _IB
        output_name = raw[len(m.group(0).split()[0]) + 1:].strip()
        # Split on " from " if present
        from_idx = output_name.lower().find(" from ")
        if from_idx >= 0:
            materials = output_name[from_idx + 6:].strip()
            output_name = output_name[:from_idx].strip()
        else:
            materials = m.group(2) or ""
        return SemanticAction(
            action_id=f"craft_{world.tick}",
            actor=actor_id,
            verb=ActionType.CRAFT,
            intent=_IB(
                rationale=output_name,
                manner=materials,
                desired_outcome=["synthesize"],
            ),
            raw_input=raw,
        )

    # ── mark / carve / scratch / draw / write <text> [on/at <dir>] ──────
    m = re.match(
        r"^(?:mark|carve|scratch|draw|write|inscribe)\s+(.+?)(?:\s+(?:on|at|to|in)\s+(.+))?$",
        lower,
    )
    if m:
        from .schemas import IntentBlock as _IB
        text_part = m.group(1).strip()
        direction_part = m.group(2) or ""
        return SemanticAction(
            action_id=f"mark_{world.tick}",
            actor=actor_id,
            verb=ActionType.MARK,
            intent=_IB(
                rationale=text_part,
                manner=direction_part,
                desired_outcome=[],
            ),
            raw_input=raw,
        )

    # ── erase / remove mark ───────────────────────────────────────────────
    m = re.match(r"^(?:erase|remove|clear)\s+mark(?:\s+(.+))?$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        text_part = m.group(1) or "mark"
        return SemanticAction(
            action_id=f"mark_erase_{world.tick}",
            actor=actor_id,
            verb=ActionType.MARK,
            intent=_IB(
                rationale=text_part,
                desired_outcome=["remove"],
            ),
            raw_input=raw,
        )

    # ── barricade [direction / target] ───────────────────────────────────
    m = re.match(r"^(?:barricade|block|blockade|pile)\s*(.*)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        direction_part = m.group(1).strip()
        return SemanticAction(
            action_id=f"barricade_{world.tick}",
            actor=actor_id,
            verb=ActionType.BARRICADE,
            intent=_IB(
                rationale="barricade",
                manner=direction_part,
                desired_outcome=[],
            ),
            raw_input=raw,
        )

    # ── unlock / lock / open [target] ────────────────────────────────────
    m = re.match(r"^(?:unlock|lock|open|seal|bolt)\s*(.*)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        target_raw = m.group(1).strip()
        target_id = _resolve_entity_name_by_name(target_raw, world.spatial) if target_raw else None
        return SemanticAction(
            action_id=f"unlock_{world.tick}",
            actor=actor_id,
            verb=lower.split()[0],  # "unlock" or "lock" etc.
            target=target_id or (target_raw or None),
            intent=_IB(rationale=target_raw, desired_outcome=[]),
            raw_input=raw,
        )

    # ── ignite / light / kindle <target> ─────────────────────────────────
    m = re.match(r"^(?:ignite|light|kindle)\s+(.+)$", lower)
    if m:
        from .schemas import IntentBlock as _IB
        target_raw = m.group(1).strip()
        target_id = _resolve_entity_name_by_name(target_raw, world.spatial)
        return SemanticAction(
            action_id=f"ignite_{world.tick}",
            actor=actor_id,
            verb=ActionType.IGNITE,
            target=target_id or target_raw,
            intent=_IB(rationale=target_raw, desired_outcome=[]),
            raw_input=raw,
        )

    # ── Body-action emote fallback ───────────────────────────────────────
    # Things like "I take my dick out", "I expose myself", "I flex",
    # "I scratch my head", "I pull my pants down" — body-part actions
    # that have no inventory object.  Route as freeform emote rather than
    # letting the LM parse them as TAKE (which then rejects TARGET_NOT_FOUND).
    _BODY_VERBS = {"flex", "scratch", "wave", "pull", "expose", "show", "jerk",
                   "rub", "touch", "stretch", "shrug", "bow", "curtsy", "kneel",
                   "crouch", "spit", "sneeze", "cough", "burp", "fart", "strip",
                   "undress", "disrobe", "gesture", "point", "wink", "grin",
                   "smirk", "laugh", "cry", "weep", "sigh", "gasp", "shriek"}
    _BODY_PARTS = {"dick", "cock", "ass", "pants", "shirt", "clothes", "hair",
                   "hand", "hands", "arm", "arms", "leg", "legs", "fist",
                   "head", "eyes", "finger", "fingers", "foot", "feet",
                   "chest", "stomach", "belly", "back", "face", "mouth"}

    first_word = lower.split()[0] if lower.split() else ""
    all_words = set(lower.split())
    if first_word in _BODY_VERBS or (all_words & _BODY_VERBS and all_words & _BODY_PARTS):
        from .schemas import IntentBlock as _IB
        return SemanticAction(
            action_id=f"emote_{world.tick}",
            actor=actor_id,
            verb="emote",
            intent=_IB(
                rationale=raw.strip(),
                manner=raw.strip(),
                desired_outcome=[],
            ),
            raw_input=raw,
        )

    return None


def _item_print(msg: str) -> None:
    """Side-channel output for item command listings (inventory, etc.)."""
    print(msg)


def _resolve_entity_name_by_name(name: str, grid: "SpatialGrid") -> "Optional[EntityId]":
    """Fuzzy match entity name → EntityId."""
    nl = name.lower().strip()
    best: Optional["EntityId"] = None
    best_score = -1
    for eid, ent in grid.entities.items():
        if not ent.alive:
            continue
        ol = ent.name.lower()
        score = 0
        if ol == nl:
            return eid
        if nl in ol or ol in nl:
            score = 10
        shared = len(set(nl.split()) & set(ol.split()))
        score += shared
        if score > best_score:
            best_score = score
            best = eid
    return best if best_score >= 0 else None


def try_parse_spell_command(
    raw_input: str,
    actor_id: "EntityId",
    actor: "Optional[EntityState]",
    world: "WorldState",
) -> "Optional[tuple[SemanticAction, Optional[str]]]":
    """
    Intercept spell-related commands before the LM sees them.

    Handled patterns:
      [cast] program...       → CAST action with program in intent.rationale
      cast program...         → same (without brackets)
      cast spell_name         → CAST action targeting inscribed spell
      inscribe name: program  → adds spell to actor.inscribed_spells, pure side-effect
      forget spell_name       → removes inscribed spell
      spells                  → lists known ops + inscribed spells (no action)
      read <grimoire_item>    → shows grimoire text + teaches ops (if player holds it)
      inspect <entity>        → already handled upstream, skip

    Returns (SemanticAction, optional_narrative) if handled, None otherwise.
    """
    import re as _re

    text = raw_input.strip()
    text_lower = text.lower()
    grid = world.spatial
    vocab = world.config.spell_vocab

    # ── "spells" command ─────────────────────────────────────────────────────
    if text_lower in ("spells", "spell list", "my spells", "list spells"):
        if actor is None:
            return None
        lines = ["═══ Your Spell Knowledge ═══"]
        if not actor.known_spell_ops:
            lines.append("  Operations known: (none — read a grimoire to learn some)")
        else:
            lines.append(f"  Operations known: {', '.join(sorted(actor.known_spell_ops))}")
        if actor.max_mana:
            lines.append(f"  Mana: {actor.meta.get('mana', actor.mana)}/{actor.max_mana}")
        if actor.inscribed_spells:
            lines.append("  Inscribed spells:")
            for sname, src in actor.inscribed_spells.items():
                lines.append(f"    {sname}: {src[:60]}{'…' if len(src) > 60 else ''}")
        else:
            lines.append("  Inscribed spells: (none — use 'inscribe name: program')")
        narrative = "\n".join(lines)
        return (
            SemanticAction(
                verb=ActionType.WAIT,
                actor=actor_id,
                raw_input=raw_input,
            ),
            narrative,
        )

    # ── "read <item>" command ─────────────────────────────────────────────────
    _read_match = _re.match(r"^read\s+(.+)$", text_lower)
    if _read_match and actor is not None:
        item_query = _read_match.group(1).strip()
        grimoire_obj = None
        for oid in actor.inventory:
            obj = grid.objects.get(oid)
            if obj is None:
                continue
            grimoire_id = obj.meta.get("grimoire_id")
            if not grimoire_id:
                continue
            if (item_query in obj.name.lower() or
                    item_query in (grimoire_id or "").lower()):
                grimoire_obj = obj
                grimoire_id_str = grimoire_id
                break

        if grimoire_obj is not None and vocab is not None:
            grimoire_def = vocab.grimoires.get(grimoire_id_str)
            if grimoire_def:
                # Teach the player any ops this grimoire unlocks
                newly_learned = []
                for op_name in grimoire_def.teaches_ops:
                    if op_name not in actor.known_spell_ops:
                        actor.known_spell_ops.append(op_name)
                        newly_learned.append(op_name)

                learn_line = ""
                if newly_learned:
                    learn_line = (
                        f"\n\n✦ You learn new spell operations: "
                        f"{', '.join(newly_learned)}"
                    )
                elif grimoire_def.teaches_ops:
                    learn_line = (
                        f"\n\n(You already know all operations from this grimoire: "
                        f"{', '.join(grimoire_def.teaches_ops)})"
                    )

                narrative = grimoire_def.text + learn_line
                return (
                    SemanticAction(
                        verb=ActionType.WAIT,
                        actor=actor_id,
                        raw_input=raw_input,
                    ),
                    narrative,
                )

    # ── "inscribe name: program" command ─────────────────────────────────────
    _inscribe_match = _re.match(r"^inscribe\s+(\w+)\s*:\s*(.+)$", text, _re.IGNORECASE | _re.DOTALL)
    if _inscribe_match and actor is not None:
        spell_name = _inscribe_match.group(1).strip()
        spell_program = _inscribe_match.group(2).strip()
        actor.inscribed_spells[spell_name] = spell_program
        return (
            SemanticAction(
                verb=ActionType.WAIT,
                actor=actor_id,
                raw_input=raw_input,
            ),
            f"✦ Spell '{spell_name}' inscribed in your grimoire.",
        )

    # ── "forget spell_name" command ───────────────────────────────────────────
    _forget_match = _re.match(r"^forget\s+(\w+)$", text_lower)
    if _forget_match and actor is not None:
        spell_name = _forget_match.group(1).strip()
        if spell_name in actor.inscribed_spells:
            del actor.inscribed_spells[spell_name]
            return (
                SemanticAction(
                    verb=ActionType.WAIT,
                    actor=actor_id,
                    raw_input=raw_input,
                ),
                f"✦ Spell '{spell_name}' removed from your grimoire.",
            )

    # ── "[cast] program" or "cast program" ───────────────────────────────────
    # Match: [cast] text, [cast]text, cast text
    _cast_match = (
        _re.match(r"^\[cast\]\s*(.*)", text, _re.DOTALL | _re.IGNORECASE) or
        _re.match(r"^cast\s+(.*)", text, _re.DOTALL | _re.IGNORECASE)
    )
    if _cast_match:
        program_or_name = _cast_match.group(1).strip()
        if not program_or_name:
            return None

        # Check if it matches an inscribed spell name (single word, no ops)
        import re as _re2
        is_single_word = bool(_re2.match(r"^\w+$", program_or_name))
        if (is_single_word and actor is not None
                and program_or_name in actor.inscribed_spells):
            # "cast fireball" → use inscribed program
            program_source = actor.inscribed_spells[program_or_name]
            from .schemas import IntentBlock as _IB
            action = SemanticAction(
                verb=ActionType.CAST,
                actor=actor_id,
                intent=_IB(
                    rationale=program_source,
                    manner=program_or_name,
                ),
                raw_input=raw_input,
            )
        else:
            # Raw spell program
            from .schemas import IntentBlock
            action = SemanticAction(
                verb=ActionType.CAST,
                actor=actor_id,
                intent=IntentBlock(rationale=program_or_name),
                raw_input=raw_input,
            )

        return (action, None)

    return None


@dataclass(frozen=True)
class PlayerFastPathResult:
    """Spell/item fast-path parse — optional narrative bypasses compiler render."""

    action: SemanticAction
    narrative_override: Optional[str] = None


def try_parse_player_fast_path(
    raw_input: str,
    actor_id: EntityId,
    world: WorldState,
    actor: Optional[EntityState] = None,
) -> Optional[PlayerFastPathResult]:
    """
    Spell and item commands that bypass the LM in interactive play.

    Consolidates the duplicate fast-path blocks that previously lived only
    in ``GameLoop.step()``.
    """
    ent = actor or world.spatial.entities.get(actor_id)

    spell_result = try_parse_spell_command(raw_input, actor_id, ent, world)
    if spell_result is not None:
        if isinstance(spell_result, tuple):
            action, narrative = spell_result
        else:
            action, narrative = spell_result, None
        return PlayerFastPathResult(action=action, narrative_override=narrative)

    item_action = try_parse_item_command(raw_input, actor_id, ent, world)
    if item_action is not None:
        return PlayerFastPathResult(action=item_action)

    return None


def parse_player_intent(
    intent: str,
    player_id: EntityId,
    world: WorldState,
    player_entity: Optional[EntityState] = None,
) -> Optional[SemanticAction]:
    """
    Run the standard pre-parser chain for REPL player input.

    Returns a SemanticAction when input is unambiguous; None to defer to LM.
    """
    ent = player_entity or world.spatial.entities.get(player_id)
    for parser, args in (
        (try_parse_trivial_actions, (intent, player_id)),
        (try_parse_orientation, (intent, player_id)),
        (try_parse_compass_step, (intent, player_id, ent, world.spatial)),
        (try_parse_directional_move, (intent, player_id, ent)),
        (try_parse_speech_command, (intent, player_id, world)),
        (try_parse_shout, (intent, player_id, world)),
    ):
        if args[-1] is None and parser not in (
            try_parse_trivial_actions, try_parse_orientation,
            try_parse_speech_command, try_parse_shout,
        ):
            continue
        result = parser(*args)
        if result is not None:
            return result

    from .interaction_resolver import resolve_interaction_action

    grammar_action = resolve_interaction_action(world, player_id, intent)
    if grammar_action is not None:
        return grammar_action

    return None


__all__ = [
    "PlayerFastPathResult",
    "parse_player_intent",
    "try_parse_compass_step",
    "try_parse_directional_move",
    "try_parse_item_command",
    "try_parse_orientation",
    "try_parse_player_fast_path",
    "try_parse_shout",
    "try_parse_speech_command",
    "try_parse_spell_command",
    "try_parse_trivial_actions",
]
