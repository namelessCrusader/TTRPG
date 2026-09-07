"""Decision 2 — the piece table. Three promises: every menu option has a verb (the table
covers the whole menu — no hidden actions), the index maps a composed answer back to one
exact option, and the table only lists pieces the actor can really use."""
from src.core import engine, pieces
from src.core.seed import vault
from src.core.state import Entity


def _scene():
    w, p = vault(seed=12)
    w.entities["brazier"] = Entity("brazier", "iron brazier", (1, 0, 0), material="metal",
                                   tags={"item"}, props={"hp": 4.0},
                                   parts={"fuel_burn": {"ticks": 5, "heat": 60.0}})
    return w, p


def test_every_option_has_a_verb():
    w, p = _scene()
    for actor in (p, w.entities["guard"], w.entities["thief"]):
        for o in engine.affordance_menu(w, actor):
            assert o.verb, f"option without a verb: {o.label!r} — the table would hide it"


def test_index_is_one_to_one():
    w, p = _scene()
    opts = engine.affordance_menu(w, p)
    idx = pieces.index(opts)
    assert len(idx) == len(opts), "two options collide on (verb, parts) — the answer would be ambiguous"
    for key, o in idx.items():
        assert (o.verb, tuple(sorted(o.args.items()))) == key


def test_table_lists_only_usable_pieces():
    w, p = _scene()
    text = pieces.table(engine.affordance_menu(w, p))
    assert "snuff" in text and "brazier" in text     # the part's offer is in the table
    assert "draw" not in text                        # already carrying water — not offered
    assert "vial" not in text                        # across the room, unseen — not a piece
    assert "(you write the words yourself)" in text  # speech stays the free-text field


def test_parser_flat_and_bare_values():
    w, p = _scene()
    opts = engine.affordance_menu(w, p)
    (c, o, _), = pieces.parse("smash target:cask_water", opts)
    assert c is None and o.verb == "smash"
    (c, o, _), = pieces.parse("pour water at:SE", opts)   # bare value finds its slot
    assert o.verb == "pour" and o.args == {"what": "water", "at": "SE"}


def test_parser_sequence_and_condition():
    w, p = _scene()
    opts = engine.affordance_menu(w, p)
    plan = pieces.parse("(then (take target:brazier) (move dir:E))", opts)
    assert [o.verb for _, o, _ in plan] == ["take", "move"]
    plan = pieces.parse("(if (near guard) (move dir:E) (smash target:cask_water))", opts)
    assert plan[0][0] == ("near", "guard") and plan[1][0] == ("not", "near", "guard")
    plan = pieces.parse('(then (speak to:guard "Sly is a thief!") (move dir:E))', opts)
    assert plan[0][2] == "Sly is a thief!" and plan[1][2] is None, \
        "quoted words stay inside their own node — a tree can speak then move"


def test_parser_names_the_bad_piece():
    w, p = _scene()
    opts = engine.affordance_menu(w, p)
    for text, bad in [("fly dir:N", "fly"), ("pour lava at:SE", "lava"),
                      ("smash", "smash"), ("contest my:hp", "contest")]:
        try:
            pieces.parse(text, opts)
            assert False, f"{text!r} should have been refused"
        except pieces.PieceError as e:
            assert e.piece == bad, f"refusal must name {bad!r}, named {e.piece!r}"


def test_condition_values_are_listed_pieces():
    """Ticket 1 from AI run #1: the AI guessed the tag 'vial' (real: has_prize); the test
    read quietly false and the wrong branch ran. With the slice, the table LISTS the legal
    condition values and the parser refuses a guessed one by name."""
    from src.core import perceive
    w, p = _scene()
    perceive.observe(w, p, w.tick)
    sl = perceive.perceive(w, p, now=w.tick)
    text = pieces.table(engine.affordance_menu(w, p), slice_=sl)
    assert "Condition pieces" in text and "near ID" in text and "carrying ITEM" in text
    opts = engine.affordance_menu(w, p)
    guard_tag = next(n["tags"][0] for n in sl["nodes"] if n["ref"] == "guard")
    plan = pieces.parse(f"(if (has guard {guard_tag}) (move dir:E))", opts, slice_=sl)
    assert plan[0][0] == ("has", "guard", guard_tag)     # a listed tag composes fine
    for bad_text, bad in [("(if (has guard vial) (move dir:E))", "vial"),
                          ("(if (near ghost) (move dir:E))", "ghost"),
                          ("(if (carrying sword) (move dir:E))", "sword")]:
        try:
            pieces.parse(bad_text, opts, slice_=sl)
            assert False, f"{bad_text!r} should have been refused"
        except pieces.PieceError as e:
            assert e.piece == bad, f"refusal must name {bad!r}, named {e.piece!r}"


def test_plan_runs_and_conditions_hold():
    """End to end: parse → run each step through the real engine; a false condition
    skips its step (a quiet no-op, like a guard in the dsl)."""
    w, p = _scene()
    opts = engine.affordance_menu(w, p)
    plan = pieces.parse("(if (near guard) (move dir:E) (take target:brazier))", opts)
    ran = []
    for cond, opt, _said in plan:
        if pieces.check(w, p, cond):
            engine.step(w, opt, actor=p)
            ran.append(opt.verb)
    assert ran == ["take"], "the guard is not near — only the else-branch runs"
    assert "inv:brazier" in p.tags, "the brazier was really picked up"
