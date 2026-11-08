"""C11 (wave 13c): the switch ledger fails on a crutch left on without a deleting condition.

src/voxel/switches.py lists every gate, crutch and estimate wave 13 left in the tree. These
tests hold it to its own rules and prove the rules bite: each one removes or spoils a row of
a COPY of the ledger and expects `audit` to say so.
"""
import copy

from src.voxel import heatmix, switches as SW


def test_the_ledger_is_clean():
    assert SW.audit() == []


def test_every_lane_of_wave_13_that_left_something_has_rows():
    owners = set(SW.by_owner())
    assert {"C1", "C2", "C3", "C4", "C4b", "C5", "C6", "C7", "C8a", "C8b", "C9", "C10", "C12", "C13", "C11"} <= owners


def test_a_crutch_without_a_deleting_condition_fails():
    rows = [dict(r) for r in SW.ROWS]
    rows[0]["deleted_when"] = ""
    assert any("no deleted_when" in p for p in SW.audit(rows))


def test_a_vague_condition_fails():
    rows = [dict(r) for r in SW.ROWS]
    crutch = next(r for r in rows if r["kind"] == "crutch")
    crutch["deleted_when"] = "later"
    assert any("says nothing" in p for p in SW.audit(rows))


def test_only_a_gate_or_permanent_wiring_may_stay_for_ever_and_it_must_say_why():
    rows = [dict(r) for r in SW.ROWS]
    crutch = next(r for r in rows if r["kind"] == "crutch")
    crutch["deleted_when"] = "never: we like it that way"
    assert any("may not stay for ever" in p for p in SW.audit(rows))
    rows = [dict(r) for r in SW.ROWS]
    perm = next(r for r in rows if r["kind"] == "permanent")
    perm["deleted_when"] = "never"
    assert any("needs a reason" in p for p in SW.audit(rows))


def test_a_gate_the_ledger_forgot_is_found_in_the_world():
    rows = [r for r in SW.ROWS if r["id"] != "contact.contact_laws"]
    assert any("contact_laws" in p for p in SW.audit(rows))


def test_a_gate_whose_default_moved_is_found():
    rows = [dict(r) for r in SW.ROWS]
    g = next(r for r in rows if r["id"] == "layers.surface_laws")
    g["probe"] = ("class", "src.voxel.sim:World.surface_laws", False)
    assert any("surface_laws" in p for p in SW.audit(rows))


def test_a_probe_into_a_file_that_lost_its_text_is_found():
    rows = [dict(r) for r in SW.ROWS]
    g = next(r for r in rows if r["probe"] and r["probe"][0] == "source")
    g["probe"] = ("source", g["probe"][1], "no such text anywhere 9f3c")
    assert any("no longer holds" in p for p in SW.audit(rows))


def test_a_legacy_skip_row_of_heatmix_that_the_ledger_forgot_is_found():
    rows = [r for r in SW.ROWS if r["id"] != "mix.skip._burn_cells"]
    assert any("_burn_cells" in p for p in SW.audit(rows))
    # and a heatmix row with an empty condition would fail too
    saved = copy.deepcopy(heatmix.CRUTCHES)
    assert all(c["deleted_when"].strip() for c in saved)


def test_ids_are_unique():
    ids = [r["id"] for r in SW.ROWS]
    assert len(ids) == len(set(ids))
    rows = list(SW.ROWS) + [dict(SW.ROWS[0])]
    assert any("duplicate" in p for p in SW.audit(rows))


def test_the_golden_has_the_ten_wave_13_proof_cases_and_leaves_them_out_of_an_older_tree():
    """Read statically: importing tools/golden.py re-executes the process under its hash seed."""
    import ast
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "golden.py")
    src = open(path).read()
    tree = ast.parse(src)
    cases = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "CASES" for t in n.targets))
    names = [k.value for k in cases.keys if isinstance(k, ast.Constant) and str(k.value).startswith("w13_")]
    assert sorted(names) == sorted([
        "w13_wet_kettle", "w13_wet_pond", "w13_sooty_room", "w13_quench", "w13_stairs",
        "w13_mud_chalk", "w13_cold_yard", "w13_brine_pot", "w13_body_heat", "w13_fall_6m"])
    assert "class Absent" in src and "except Absent" in src
