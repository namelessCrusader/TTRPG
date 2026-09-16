"""affordance_menu as the D8 legality predicate: ONE menu, any actor, perception-filtered.
Locks the contracts the prototype's two menu bugs (ungated reach, move-into-wall) taught us
to test: contact options bind only to FRESH belief, the frozen slice is honored
(double-buffer), and speech is an earshot-gated act whose words stay a pick-time parameter."""
from src.core.seed import vault
from src.core import engine
from src.core.perceive import observe, perceive


def _labels(opts):
    return [o.label for o in opts]


def test_any_actor_and_self_target_curation():
    """Self-directed actions are LEGAL and appear where a rule affords them (drop-and-roll
    when burning is a self-action; 'self' is a first-class ref on the bus — madness, grooming,
    self-harm are future rule-afforded self-verbs, possibly at body-part granularity via the
    D13 parts layer). What the menu curates away is only the DEGENERATE default spam:
    attack-yourself / speak-to-yourself offered to every person-tagged actor every beat."""
    w, _ = vault(seed=12)
    guard = w.entities["guard"]                      # a person-tagged NPC as the actor
    labels = _labels(engine.affordance_menu(w, guard))
    assert any(l.startswith("move") for l in labels)
    assert not any(guard.name in l and (l.startswith("attack") or l.startswith("speak"))
                   for l in labels), "degenerate self-spam is curated out of the default menu"
    guard.tags.add("on_fire")                        # ...but an afforded SELF-action appears
    labels = _labels(engine.affordance_menu(w, guard))
    assert any("drop and roll" in l for l in labels), "rule-afforded self-actions must appear"


def test_contact_options_bind_only_to_perceived_entities():
    w, _ = vault(seed=12)
    thief = w.entities["thief"]                      # far corner; the cask is across the vault
    labels = _labels(engine.affordance_menu(w, thief))
    assert not any("cask" in l for l in labels), "unseen entity must not appear in the menu"
    assert any("keg" in l for l in labels), "the adjacent, visible keg should appear"


def test_frozen_slice_is_honored_double_buffer():
    w, _ = vault(seed=12)
    thief = w.entities["thief"]                      # @(5,1); keg_oil @(4,1) adjacent
    observe(w, thief, tick=0)
    frozen = perceive(w, thief, now=0)

    w.entities["keg_oil"].pos = (0, 3, 0)            # hauled away, unseen by the thief
    w.tick = 1                                       # a beat passes
    live = _labels(engine.affordance_menu(w, thief))            # live path re-observes
    kept = _labels(engine.affordance_menu(w, thief, slice_=frozen))
    assert not any("keg" in l for l in live), "live menu must track current sight"
    assert any("keg" in l for l in kept), \
        "frozen slice must keep offering it — decide on slice-start belief; execution re-validates"


def test_stale_belief_never_yields_contact_options():
    w, _ = vault(seed=12)
    thief = w.entities["thief"]
    observe(w, thief, tick=0)
    stale = perceive(w, thief, now=5)                # same store, 5 ticks later: all nodes stale
    labels = _labels(engine.affordance_menu(w, thief, slice_=stale))
    assert not any("keg" in l or "attack" in l or "speak" in l for l in labels), \
        "stale memories may inform navigation, never a grab/strike/word"
    assert any(l.startswith("move") for l in labels), "movement stays available"


def test_any_actor_round_trip_pick_then_step():
    """The full D8 seam: an NPC gets a menu AND executes its pick through the official
    step() path — the deed attributed to the actor on the bus; derived consequences
    (the burst, the spill) stay actor=None, i.e. the world's own."""
    w, _ = vault(seed=12)
    guard = w.entities["guard"]
    pick = next(o for o in engine.affordance_menu(w, guard) if "smash" in o.label)
    out = engine.step(w, pick, actor=guard)
    assert not out["violations"]
    assert w.entities["cask_water"].props["hp"] <= 0, "the smash landed"
    deeds = [e for e in w.log if e.actor == "guard" and e.kind == "adjust_prop"]
    assert deeds, "the deed is attributed to the GUARD, not the player"
    consequences = [e for e in w.log if e.kind == "add_fluid"]
    assert consequences and all(e.actor is None for e in consequences), \
        "derived consequences belong to the world (actor=None), never the author"


def test_speech_is_an_earshot_act_with_pick_time_words():
    w, p = vault(seed=12)
    opts = engine.affordance_menu(w, p)
    speaks = [o for o in opts if o.label.startswith("speak to")]
    assert speaks, "visible people in earshot should be addressable"
    for o in speaks:
        npc_id, intent = o.social
        assert npc_id in w.entities
        assert intent is None, "the words are a pick-time parameter, not menu content"
