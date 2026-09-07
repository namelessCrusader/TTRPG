"""The DSL: a total, loop-free program language the model emits over the effect
VM. The sim is the authoritative interpreter — it executes a program's exact
deterministic-or-seeded-stochastic edit, or refuses a malformed one.

Verbs/reactions/stances are all special cases of a program; this is the general
(state, intent, target) → edit operation the trained DM learns to compose.
"""
from src.core import dsl
from src.core.state import Cell, Entity, World


def _world():
    cells = {(x, y, 0): Cell() for x in range(3) for y in range(3)}
    player = Entity("player", "you", (1, 1, 0), tags={"player", "alive"},
                    props={"hp": 10.0, "qi": 20.0, "composure": 60.0})
    guard = Entity("guard", "Bran the guard", (2, 1, 0), tags={"person", "alive", "hostile"},
                   props={"hp": 10.0, "composure": 40.0})
    world = World(dims=(3, 3, 1), cells=cells,
                  entities={"player": player, "guard": guard}, seed=1)
    world.edges[("guard", "player")] = {"disposition": 0.0}
    return world, player


# DESIGN: a deterministic program — bind, no roll, a flat effect list — applies
# exactly, atomically, through the one bus.
def test_deterministic_program_applies():
    world, player = _world()
    prog = {"target": {"cell": "here"}, "guard": None, "roll": None,
            "do": [{"op": "transfer_prop", "from": "self", "to": "target", "prop": "qi",
                    "amount": {"prop": ["self", "qi"]}}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and r["changed"] and r["band"] == "do", r
    assert player.props["qi"] == 0.0 and world.cell((1, 1, 0)).props["qi"] == 20.0, "qi moved into the soil"


# DESIGN: a contest — seeded, replayable — picks clean/fail and applies that branch.
def test_stochastic_contest_picks_a_branch():
    world, player = _world()
    prog = {"target": {"entity": "guard"}, "guard": None,
            "roll": {"contest": [{"prop": ["self", "composure"]}, {"prop": ["target", "composure"]}]},
            "clean": [{"op": "set_tag", "t": "target", "tag": "afraid"},
                      {"op": "clear_tag", "t": "target", "tag": "hostile"}],
            "fail": [{"op": "adjust_prop", "t": "self", "prop": "composure", "by": -5}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and r["band"] in ("clean", "fail")
    if r["band"] == "clean":
        assert "afraid" in world.entities["guard"].tags and "hostile" not in world.entities["guard"].tags
    else:
        assert player.props["composure"] == 55.0


# DESIGN: compound guards + IF-args + a computed magnitude — the recursive depth
# the flat template couldn't express.
def test_compound_guard_and_computed_arg():
    world, player = _world()
    # only if the guard is BOTH alive AND hostile; damage scaled by my qi, minus his composure
    prog = {"target": {"entity": "guard"},
            "guard": {"and": [{"has_tag": ["target", "alive"]}, {"has_tag": ["target", "hostile"]}]},
            "roll": None,
            "do": [{"op": "adjust_prop", "t": "target", "prop": "hp",
                    "by": {"sub": [0, {"if": [{"cmp": [{"prop": ["self", "qi"]}, ">", 10]}, 5, 2]}]}}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and r["changed"]
    assert world.entities["guard"].props["hp"] == 5.0, "qi>10 → -5 damage branch"


# DESIGN: a guard that fails → the program is a legal no-op (not an error).
def test_guard_false_is_a_clean_noop():
    world, player = _world()
    prog = {"target": {"entity": "guard"}, "guard": {"has_tag": ["target", "asleep"]},
            "roll": None, "do": [{"op": "set_tag", "t": "target", "tag": "stabbed"}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and not r["changed"] and r["band"] == "guard-false"
    assert "stabbed" not in world.entities["guard"].tags


# DESIGN: spawn/despawn — the one lifecycle primitive a closed-state VM needs.
def test_spawn_and_despawn():
    world, player = _world()
    r = dsl.run(world, player, {"target": {"self": True}, "guard": None, "roll": None,
                                "do": [{"op": "spawn", "id": "spirit", "name": "a summoned spirit",
                                        "at": "N", "tags": ["animal", "alive"], "props": {"hp": 3.0}}]})
    assert r["ok"] and "spirit" in world.entities and world.entities["spirit"].pos == (1, 0, 0)
    r2 = dsl.run(world, player, {"target": {"entity": "spirit"}, "guard": None, "roll": None,
                                 "do": [{"op": "despawn", "t": "target"}]})
    assert r2["ok"] and "spirit" not in world.entities


# DESIGN: a spawn introduces its id into scope, so a LATER effect in the SAME program
# can target the just-created entity (summon a spirit AND pour a resource into it atomically).
def test_spawn_then_target_in_one_program():
    world, player = _world()
    prog = {"target": {"self": True}, "guard": None, "roll": None,
            "do": [{"op": "spawn", "id": "spirit", "name": "a summoned spirit", "at": "N",
                    "tags": ["spirit", "alive"], "props": {"qi": 0.0}},
                   {"op": "transfer_prop", "from": "self", "to": "spirit", "prop": "qi", "amount": 5},
                   {"op": "set_tag", "t": "spirit", "tag": "tending"}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and "spirit" in world.entities, r
    assert world.entities["spirit"].props["qi"] == 5.0 and "tending" in world.entities["spirit"].tags


# DESIGN: a CROSS-DOMAIN program in ONE tree — social + physics together — which
# per-primitive/per-domain models could never co-author. This is why ONE model.
def test_cross_domain_composition_in_one_program():
    world, player = _world()
    world.cell((2, 1, 0)).heat = 200.0                       # the guard stands in heat
    prog = {"target": {"entity": "guard"}, "guard": None, "roll": None,
            "do": [{"op": "adjust_edge", "a": "target", "b": "self", "kind": "disposition", "by": -0.4},
                   {"op": "set_tag", "t": "target", "tag": "insulted"},
                   {"op": "move", "t": "target", "dir": "W"}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and r["n_effects"] == 3
    assert world.edges[("guard", "player")]["disposition"] == -0.4
    assert "insulted" in world.entities["guard"].tags and world.entities["guard"].pos == (1, 1, 0)


# DESIGN: {"cell_of": ref} targets the CELL an entity stands in — so fire/fluids
# reach a creature's own tile (the DM flagged this: you couldn't ignite the
# scuttler, only a cell relative to yourself).
def test_cell_of_targets_an_entitys_own_cell():
    world, player = _world()
    prog = {"target": {"entity": "guard"}, "guard": None, "roll": None,
            "do": [{"op": "add_heat", "t": {"cell_of": "target"}, "by": 130},
                   {"op": "spark", "t": {"cell_of": "target"}}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and r["changed"]
    assert world.cell(world.entities["guard"].pos).heat >= 130, "the guard's own cell is now hot"


# DESIGN: min/max args express a CLAMP directly ("pour up to 8, or all I have left")
# without an if-hack or a loop — the loop-free VM still covers bounded exhaustion.
def test_min_max_args_clamp():
    world, player = _world()
    player.props["qi"] = 5.0
    prog = {"target": {"cell": "here"}, "guard": None, "roll": None,
            "do": [{"op": "transfer_prop", "from": "self", "to": {"cell": "here"}, "prop": "qi",
                    "amount": {"min": [8, {"prop": ["self", "qi"]}]}}]}
    assert dsl.run(world, player, prog)["ok"]
    assert world.cell((1, 1, 0)).props["qi"] == 5.0 and player.props["qi"] == 0.0, "clamped to the 5 available"


# DESIGN: {"cell":dir} is a first-class REF, not just a target — so you can pour a
# resource straight into the ground ("I put qi in the soil") and read what it drank.
def test_cell_is_a_first_class_ref():
    world, player = _world()
    prog = {"target": {"self": True}, "guard": None, "roll": None,
            "do": [{"op": "transfer_prop", "from": "self", "to": {"cell": "here"}, "prop": "qi",
                    "amount": 12}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and r["changed"]
    assert world.cell((1, 1, 0)).props["qi"] == 12.0 and player.props["qi"] == 8.0, "qi poured into the soil"
    # and the same cell-ref reads back inside a guard/arg
    ripe = {"target": {"self": True}, "guard": {"cmp": [{"prop": [{"cell": "here"}, "qi"]}, ">=", 10]},
            "roll": None, "do": [{"op": "set_tag", "t": "self", "tag": "vein_awake"}]}
    assert dsl.run(world, player, ripe)["changed"], "the guard read the soil's accumulated qi"


# DESIGN: pooled fluid is a READABLE cell field — spill oil, then GUARD ignition on
# "is there oil here to light?" (the arson chain: smash → spill → check → spark).
def test_cell_fluid_is_readable_as_a_prop():
    world, player = _world()
    spill = {"target": {"cell": "here"}, "guard": None, "roll": None,
             "do": [{"op": "add_fluid", "t": {"cell": "here"}, "mat": "oil", "ml": 400}]}
    assert dsl.run(world, player, spill)["ok"]
    ignite = {"target": {"cell": "here"},
              "guard": {"cmp": [{"prop": [{"cell": "here"}, "oil"]}, ">", 0]}, "roll": None,
              "do": [{"op": "add_heat", "t": {"cell": "here"}, "by": 150}, {"op": "spark", "t": {"cell": "here"}}]}
    r = dsl.run(world, player, ignite)
    assert r["ok"] and r["changed"], "the ignition guard read the pooled oil and fired"
    # a dry cell reads 0 → the guard is a clean no-op, not an error
    dry = {"target": {"cell": "N"},
           "guard": {"cmp": [{"prop": [{"cell": "N"}, "oil"]}, ">", 0]}, "roll": None,
           "do": [{"op": "spark", "t": {"cell": "N"}}]}
    assert not dsl.run(world, player, dry)["changed"], "no oil to the north → nothing lights"


# DESIGN: disposition is a normalized relationship in [-1,1] — repeated adjust_edge
# can't run it off the scale (like transfer_prop clamps a resource).
def test_disposition_edge_clamps():
    world, player = _world()
    for _ in range(20):
        dsl.run(world, player, {"target": {"entity": "guard"}, "roll": None,
                                "do": [{"op": "adjust_edge", "a": "guard", "b": "self",
                                        "kind": "disposition", "by": 0.5}]})
    assert world.edges[("guard", "player")]["disposition"] == 1.0, "clamped at the +1 ceiling"


# DESIGN: you cannot walk into a wall — move refuses an out-of-bounds or solid
# destination (the harness caught an authored move that put a guard inside a block).
def test_move_into_solid_is_refused():
    world, player = _world()
    world.cell((2, 1, 0)).material = "wood"          # a solid block east of the guard's neighbour
    prog = {"target": {"self": True}, "roll": None,
            "do": [{"op": "move", "t": "self", "dir": "E"}]}   # player (1,1,0) → (2,1,0), now solid
    r = dsl.run(world, player, prog)
    assert not r["ok"] and "solid" in r["error"] and player.pos == (1, 1, 0), "the move was refused, no partial state"


# DESIGN: a malformed program is REFUSED atomically — nothing partially applies.
def test_malformed_program_is_refused_atomically():
    world, player = _world()
    before = dict(world.entities["guard"].props)
    prog = {"target": {"entity": "guard"}, "guard": None, "roll": None,
            "do": [{"op": "adjust_prop", "t": "target", "prop": "hp", "by": -3},
                   {"op": "nonsense_op", "t": "target"}]}     # 2nd effect is invalid
    r = dsl.run(world, player, prog)
    assert not r["ok"] and "nonsense_op" in r["error"]
    assert world.entities["guard"].props == before, "atomic: the valid 1st effect must NOT apply"


# DESIGN: validate() dry-runs on a fork — the data engine's filter never mutates.
def test_validate_does_not_touch_the_world():
    world, player = _world()
    prog = {"target": {"entity": "guard"}, "guard": None, "roll": None,
            "do": [{"op": "set_tag", "t": "target", "tag": "marked"}]}
    r = dsl.validate(world, player, prog)
    assert r["ok"] and r["changed"] and "marked" not in world.entities["guard"].tags, "dry-run only"


# DESIGN: an arg can read an EDGE (a relationship) — so an outcome conditions on
# "does she already like me", and a program can SPEAK a reply (Meiling talks back).
def test_edge_read_and_speech():
    world, player = _world()
    world.edges[("guard", "player")] = {"disposition": 0.5}
    # warmth lands EASIER the more she likes you: contest = my composure + her disposition*40
    prog = {"target": {"entity": "guard"},
            "roll": {"contest": [{"add": [{"prop": ["self", "composure"]},
                                          {"mul": [{"edge": ["guard", "player", "disposition"]}, 40]}]},
                                 {"prop": ["target", "composure"]}]},
            "clean": [{"op": "speech", "who": "target", "text": "Aye, well met."},
                      {"op": "adjust_edge", "a": "target", "b": "self", "kind": "disposition", "by": 0.2}],
            "fail": [{"op": "speech", "who": "target", "text": "Hmph."}]}
    r = dsl.run(world, player, prog)
    assert r["ok"] and r["changed"]
    said = [e for e in world.log if e.kind == "speech" and e.data["who"] == "guard"]
    assert said, "the guard spoke a reply"


# DESIGN: character DRIFT — a `cruel` tag the actor carries taxes a later kindness
# contest (reading self's own tag via an IF-arg). Being cruel makes kindness harder.
def test_character_state_biases_a_later_roll():
    world, player = _world()
    player.tags.add("cruel")
    losses = 0
    for s in range(60):
        world.seed = s; world._rng = None
        prog = {"target": {"entity": "guard"},
                "roll": {"contest": [{"sub": [{"prop": ["self", "composure"]},
                                              {"if": [{"has_tag": ["self", "cruel"]}, 100, 0]}]},
                                     {"prop": ["target", "composure"]}]},
                "clean": [{"op": "set_tag", "t": "target", "tag": "charmed"}], "fail": []}
        if dsl.run(world, player, prog)["band"] == "fail":
            losses += 1
    assert losses == 60, "a heavy cruelty penalty makes the kindness contest always fail"


# DESIGN: canonicalize absorbs a free-form author's dialect (the spike: free-form
# invented {target, when, resolve:{type, success, failure}}) → runnable.
def test_canonicalize_absorbs_a_freeform_dialect():
    world, player = _world()
    freeform = {"target": {"entity": "guard"}, "when": {"has_tag": ["target", "hostile"]},
                "resolve": {"type": "contest",
                            "roll": [{"prop": ["self", "composure"]}, {"prop": ["target", "composure"]}],
                            "success": [{"op": "set_tag", "t": "target", "tag": "cowed"}],
                            "failure": [{"op": "adjust_prop", "t": "self", "prop": "hp", "by": -1}]}}
    canon = dsl.canonicalize(freeform)
    assert canon["guard"]["has_tag"] == ["target", "hostile"] and "resolve" not in canon
    assert canon["roll"]["contest"] and canon["clean"] and canon["fail"]
    assert dsl.run(world, player, canon)["ok"], "the translated program runs"
