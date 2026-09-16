"""Protoreasoning-trace harvest — the corpus that makes a small model competent.

Every turn, the selectors (social.py, composer.py) push their scored decisions
into a per-turn BUFFER via capture(). At the end of the turn the ENGINE decides:
if the action ADVANCED THE WORLD, bless() flushes the buffer into the kNN datastore
as (situation → chosen) anchors; if it was refused / a no-op / a stall / or merely
shuffled position without changing anything, discard() throws the buffer away.

The blessing bar is "advanced the world", NOT "committed coherently". That upgrade
is load-bearing: a bare walk almost always COMMITS (moving is valid), so a
commit-only bar over-credits `walk` and training on it AMPLIFIES the walk-collapse
we measured. Requiring a real state change beyond position means an idle step earns
no positive example, while the step that reaches a goal / lands a blow / moves a
disposition does. See docs/SMALL_MODEL_DM.md (the keystone signal upgrade).

The engine owns "blessed" because it's the only layer that sees the whole
resolution. The decision sites stay ignorant of each other and of the outcome —
they just record what they chose. One process, one turn at a time, so a plain
module-level buffer is the right amount of machinery (same shape as social.BRAIN).
"""

from __future__ import annotations

_DS = None            # the datastore to flush into (None = learning disabled)
_BUF: list = []       # this turn's captured decisions, awaiting a verdict


def begin(datastore) -> None:
    """Start a turn's capture. Clears any stale buffer so an abandoned decision
    from a crashed turn never leaks into the next one."""
    global _DS, _BUF
    _DS, _BUF = datastore, []


def capture(intent: str, options: list, scores: list, chosen: str) -> None:
    """Buffer one scored decision. Cheap append; nothing is learned until bless()."""
    if _DS is not None:
        _BUF.append((intent, list(options), list(scores), chosen))


def pending() -> bool:
    return bool(_BUF)


def bless(advanced: bool = True) -> None:
    """The action resolved. Record its decisions as anchors ONLY if it advanced the
    world (`advanced` True). An action that committed but changed nothing beyond the
    actor's position is dropped like a refusal — it's not a positive example."""
    global _BUF
    if not advanced:
        _BUF = []
        return
    if _DS is not None:
        for intent, options, scores, chosen in _BUF:
            _DS.add(intent, options, scores, chosen)
    _BUF = []


def discard() -> None:
    """The action was refused / changed nothing: drop the buffer unlearned."""
    global _BUF
    _BUF = []
