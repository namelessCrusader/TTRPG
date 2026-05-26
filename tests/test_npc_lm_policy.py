"""Single-path LM NPC policy: verb repair and cognition mode."""

from src.sim.compiler import compile_action
from src.sim.game_loop import make_test_world
from src.sim.npc_lm_policy import LMNpcPolicy, repair_npc_action
from src.sim.schemas import ActionType, IntentBlock, RejectionReason, SemanticAction
from src.sim.speech_utils import validate_spoken_line


def test_repair_npc_action_fuzzy_verb():
    world = make_test_world()
    npc = next(e for e in world.spatial.entities.values() if e.name != "You")
    other = next(e for e in world.spatial.entities.values() if e.entity_id != npc.entity_id)
    action = SemanticAction(
        verb="gossipp",
        actor=npc.entity_id,
        target=other.entity_id,
        intent=IntentBlock(manner="Did you hear about the guard?"),
        raw_input="[test]",
    )
    fixed = repair_npc_action(action, world)
    assert str(fixed.verb).lower() == "gossip"


def test_uses_lm_layer_by_mode():
    world = make_test_world()
    world.config.npc_policy.cognition.mode = "lm"
    assert LMNpcPolicy.uses_lm_layer(world)
    world.config.npc_policy.cognition.mode = "reactive_only"
    assert not LMNpcPolicy.uses_lm_layer(world)


def test_compiler_rejects_personality_descriptor_as_speech():
    world = make_test_world()
    npc = next(e for e in world.spatial.entities.values() if e.name != "You")
    other = next(e for e in world.spatial.entities.values() if e.entity_id != npc.entity_id)
    action = SemanticAction(
        verb=ActionType.SPEAK,
        actor=npc.entity_id,
        target=other.entity_id,
        intent=IntentBlock(manner="calm, observant, slightly weary"),
        raw_input="[test]",
    )
    result = compile_action(action, world)
    assert not result.valid
    assert result.rejection_reason == RejectionReason.MALFORMED_ACTION


def test_validate_spoken_line_accepts_real_sentence():
    ok, _ = validate_spoken_line(
        "Step back, stranger.",
        personality="calm, observant, slightly weary",
    )
    assert ok


# ─────────────────────────────────────────────────────────────────────────
# Regression: when the LM adapter cannot infer (model failed to load,
# transient network error, etc.) the policy MUST fall back to the
# reactive policy instead of raising.  Repro for the 2026-05-22 crash
# where every NPC raised MalformedActionError and the sim stalled.
# ─────────────────────────────────────────────────────────────────────────


class _NotReadyAdapter:
    """Mimics a torch adapter whose warmup failed (load_failed=True)."""

    is_ready = False
    model = "fake/model"

    def infer_npc(self, sheet, projection):  # pragma: no cover - should not be called
        raise RuntimeError("infer_npc should not run when adapter is not ready")


class _AlwaysRaiseAdapter:
    """Mimics an adapter that does try to infer but blows up every call."""

    is_ready = True
    model = "fake/model"

    def infer_npc(self, sheet, projection):
        raise RuntimeError("simulated LM blow-up")


def test_decide_falls_back_to_reactive_when_adapter_not_ready():
    """Short-circuit: ``LMNpcPolicy.decide`` must NOT call ``infer_npc``
    when ``adapter.is_ready is False`` — it should go straight to the
    reactive fallback to avoid per-NPC warning spam.
    """
    world = make_test_world()
    world.config.npc_policy.cognition.mode = "lm"
    npc = next(e for e in world.spatial.entities.values() if e.name != "You")

    policy = LMNpcPolicy(adapter=_NotReadyAdapter())
    action = policy.decide(npc, world)

    assert action.actor == npc.entity_id
    assert npc.meta.get("last_policy_branch") == "adapter_not_ready_reactive"


def test_decide_falls_back_when_infer_always_raises():
    """Regression for ``logs/autonomous_debug_20260522_055425.log``:
    when the adapter is ready but every infer call throws (e.g. brotli
    decoder error), ``decide`` must catch the exception inside
    ``_decide_via_infer`` and fall back reactively rather than
    propagating ``MalformedActionError`` to ``_step_one_entity``.
    """
    world = make_test_world()
    world.config.npc_policy.cognition.mode = "lm"
    npc = next(e for e in world.spatial.entities.values() if e.name != "You")

    policy = LMNpcPolicy(adapter=_AlwaysRaiseAdapter())
    action = policy.decide(npc, world)

    assert action.actor == npc.entity_id
    assert npc.meta.get("last_policy_branch") == "infer_failed_reactive"


def test_mdp_forest_and_selection_success():
    """Verify that MDP Option Forest is built and the selected option is executed."""
    world = make_test_world()
    world.config.npc_policy.cognition.mode = "lm"
    npc = next(e for e in world.spatial.entities.values() if e.name != "You")

    # Mock adapter that returns a specific option selection with speech
    class _OptionSelectorAdapter:
        is_ready = True
        model = "fake/model"
        
        def infer_npc_mdp(self, sheet, projection, options, option_indices):
            # Select Option 1 (since Option 0 is creative wildcard, Option 1 is first candidate)
            return {
                "selected_option_index": 1,
                "rationale": "I want to perform the first pre-validated action.",
                "custom_speech_line": "Step aside, I am acting!",
            }

    policy = LMNpcPolicy(adapter=_OptionSelectorAdapter())
    action = policy.decide(npc, world)
    
    assert action is not None
    assert action.actor == npc.entity_id
    assert npc.meta.get("last_policy_branch") == "infer_npc_mdp_option_1"
    
    # Check that the speech has been appended as a simultaneous proposed effect
    assert action.proposed_effects is not None
    speech_effect = action.proposed_effects[-1]
    assert speech_effect.kind == "dialogue_spoken"
    assert speech_effect.payload.get("text") == "Step aside, I am acting!"


def test_mdp_forest_selection_fallback_to_unconstrained():
    """Verify that selecting Option 0 falls back to standard unconstrained infer_npc."""
    world = make_test_world()
    world.config.npc_policy.cognition.mode = "lm"
    npc = next(e for e in world.spatial.entities.values() if e.name != "You")

    class _OptionZeroAdapter:
        is_ready = True
        model = "fake/model"
        
        def infer_npc_mdp(self, sheet, projection, options, option_indices):
            return {
                "selected_option_index": 0,
                "rationale": "I want to do a custom creative action.",
                "custom_speech_line": "",
            }

        def infer_npc(self, sheet, projection):
            # Propose a custom examine action
            return SemanticAction(
                verb="examine",
                actor=sheet.npc_id,
                target=sheet.npc_id,
            )

    policy = LMNpcPolicy(adapter=_OptionZeroAdapter())
    action = policy.decide(npc, world)
    
    assert action is not None
    assert action.verb == "examine"
    assert npc.meta.get("last_policy_branch") == "infer_npc_custom"

