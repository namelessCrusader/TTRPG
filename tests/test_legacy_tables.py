"""Wave 13, T1: the padded legacy tables (src/voxel/legacy_tables.py).

The equality tests compare against the FROZEN ORACLE tests/data/matrec_oracle_base19.json
(written by lane A-O from the base18 tree, frozen again by B0 from base19: no table row differs), never against the live materials.py: a
test of the live file against itself passes by construction. If the oracle file is
absent the oracle tests are SKIPPED (the reason says so) and the lane reports it.

Tags: [I] independent of the code (the oracle is frozen data); [C] code-only (the
code against the live materials.py, test_codeonly_*), which never gates a scene.
"""
import ast
import json
import os

import numpy as np
import pytest

from src.voxel import legacy_tables as LT
from src.voxel import species as SP

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_PATH = os.path.join(HERE, "data", "matrec_oracle_base19.json")
NO_ORACLE = not os.path.exists(ORACLE_PATH)
needs_oracle = pytest.mark.skipif(
    NO_ORACLE, reason="tests/data/matrec_oracle_base19.json is absent (lane A-O, B0)")

# the old materials.py arrays this lane rebuilds (the oracle stores each as bytes)
ARRAYS = ["_DENS_ARR", "_CSOLID_ARR", "_KSOLID_ARR", "_IGN_ARR", "_EMISS_ARR",
          "_SKIN_DEPTH_ARR", "_SKIN_PATH_ARR", "_ALPHA",
          "_FDENS_ARR", "_FC_ARR", "_FK_ARR", "_FEMISS_ARR", "_FLK_ARR",
          "_FSKIN_DEPTH_ARR", "_FSKIN_PATH_ARR", "_FALPHA"]
FLUID_ARRAYS = {a for a in ARRAYS if a.startswith("_F")}


def _num(x):
    """An oracle number: a float that is not finite is {"$f": "inf"}."""
    if isinstance(x, dict) and "$f" in x:
        return float(x["$f"])
    return x


def _row(v):
    return tuple(_num(x) for x in v)


def _oracle():
    with open(ORACLE_PATH) as f:
        return json.load(f)


def _legacy_from(o):
    """The dicts build_tables takes, from the oracle."""
    t = o["tables"]
    intkeys = lambda d, f=lambda v: v: {int(k): f(v) for k, v in d.items()}
    return {"SOLID": intkeys(t["SOLID"], _row), "FLUID": intkeys(t["FLUID"], _row),
            "EMISS": intkeys(t["EMISS"], _num), "EMISS_FLUID": intkeys(t["EMISS_FLUID"], _num),
            "FLUID_K": intkeys(t["FLUID_K"], _num), "MELT": intkeys(t["MELT"], _row),
            "SKIN_T": _row(t["SKIN_T"]), "BOIL": o["scalars"]["BOIL"],
            "LATENT": o["scalars"]["LATENT"], "NMAT": o["nmat"], "FLIDS": tuple(o["flids"])}


def _bytes(o, name):
    return bytes.fromhex(o["arrays"][name]["hex"])


@pytest.fixture(scope="module")
def built():
    if NO_ORACLE:
        pytest.skip("oracle absent")
    o = _oracle()
    return o, LT.build_tables(_legacy_from(o), SP.seed_mat_rows(), SP.seed_fluid_rows(), 64, 64)


# ---- [I] the oracle tests ---------------------------------------------------
@needs_oracle
@pytest.mark.parametrize("name", ARRAYS)
def test_padded_then_sliced_array_equals_the_oracle_byte_for_byte(built, name):
    o, t = built
    n = o["nmat"] if name not in FLUID_ARRAYS else o["fluid_slots"]
    got = t[name][:n]
    want = o["arrays"][name]
    assert got.dtype == np.dtype(want["dtype"])
    assert list(got.shape) == want["shape"]
    assert got.tobytes() == _bytes(o, name), name


@needs_oracle
def test_the_pinned_slot_counts_come_from_the_oracle(built):
    o, t = built
    assert o["nmat"] == 12 and o["fluid_slots"] == 9
    assert len(t["_DENS_ARR"]) == 64 and len(t["_FDENS_ARR"]) == 64
    assert len(o["arrays"]["_FDENS_ARR"]["shape"]) == 1
    assert o["arrays"]["_FDENS_ARR"]["shape"][0] == o["fluid_slots"]


@needs_oracle
def test_BOIL_and_LATENT_equal_the_oracle_and_the_two_known_numbers(built):
    o, t = built
    assert t["BOIL"] == o["scalars"]["BOIL"] == 100.0
    assert t["LATENT"] == o["scalars"]["LATENT"] == 2260.0


@needs_oracle
def test_MELT_and_FREEZE_keep_every_oracle_row_and_the_dense_form_matches(built):
    o, t = built
    want_melt = {int(k): _row(v) for k, v in o["tables"]["MELT"].items()}
    want_freeze = {int(k): _row(v) for k, v in o["tables"]["FREEZE"].items()}
    for m, row in want_melt.items():
        assert t["MELT"][m] == row
        assert (t["MELT_T"][m], t["MELT_FL"][m], t["MELT_LAT"][m]) == (row[0], row[1], row[2])
    for f, row in want_freeze.items():
        assert t["FREEZE"][f] == row
        assert (t["FREEZE_T"][f], t["FREEZE_MAT"][f], t["FREEZE_LAT"][f]) == row
    # the legacy ids that do not melt have no dense entry
    for m in range(o["nmat"]):
        if m not in want_melt:
            assert np.isinf(t["MELT_T"][m]) and t["MELT_FL"][m] == 0
    # the only additions are the seed melts (copper), none on a legacy id
    assert set(t["MELT"]) - set(want_melt) == {m for m in t["MELT"] if m >= 12}


@needs_oracle
def test_a_changed_oracle_value_is_caught(built):
    """The comparison is not vacuous: change one number and the bytes differ."""
    o, _ = built
    leg = _legacy_from(o)
    leg["SOLID"][3] = (leg["SOLID"][3][0] + 1.0,) + leg["SOLID"][3][1:]
    t2 = LT.build_tables(leg, SP.seed_mat_rows(), SP.seed_fluid_rows(), 64, 64)
    assert t2["_DENS_ARR"][:12].tobytes() != _bytes(o, "_DENS_ARR")


@needs_oracle
def test_the_oracle_ids_are_the_species_registry_ids(built):
    """The oracle's mat and fluid names are the registry's legacy slots."""
    o, _ = built
    assert {int(k) for k in o["ids"]["mat"]} == set(range(12))
    assert {int(k) for k in o["ids"]["fluid"]} == set(range(9))
    for f, sid in {1: "water", 2: "lamp_oil", 3: "sulfuric_acid", 4: "acetic_acid",
                   8: "blood"}.items():
        assert SP.get(SP.FLUIDS[f].species).name == sid
    for m in range(1, 12):
        assert SP.get(SP.MATS[m].species).name == o["ids"]["mat"][str(m)].lower()


# ---- the padding rule (no oracle needed) -------------------------------------
def _built_live():
    return SP.padded_tables()


def test_unlisted_padded_entries_equal_their_like_row():
    t = _built_live()
    listed_m = {r["mat"] for r in SP.seed_mat_rows()}
    listed_f = {r["fl"] for r in SP.seed_fluid_rows()}
    mat_arrays = ["_DENS_ARR", "_CSOLID_ARR", "_KSOLID_ARR", "_IGN_ARR", "_EMISS_ARR",
                  "_SKIN_DEPTH_ARR", "_SKIN_PATH_ARR", "_ALPHA", "MELT_T", "MELT_FL", "MELT_LAT"]
    fl_arrays = ["_FDENS_ARR", "_FC_ARR", "_FK_ARR", "_FEMISS_ARR", "_FLK_ARR",
                 "_FSKIN_DEPTH_ARR", "_FSKIN_PATH_ARR", "_FALPHA", "FREEZE_T", "FREEZE_MAT",
                 "FREEZE_LAT"]
    n_pad = 0
    for i in range(12, 64):
        if i in listed_m:
            continue
        like = int(t["LIKE_MAT"][i])
        assert like < i
        for a in mat_arrays:
            assert t[a][i].tobytes() == t[a][like].tobytes(), (a, i)
        n_pad += 1
    for i in range(9, 64):
        if i in listed_f:
            continue
        like = int(t["LIKE_FLUID"][i])
        assert like < i
        for a in fl_arrays:
            assert t[a][i].tobytes() == t[a][like].tobytes(), (a, i)
        n_pad += 1
    assert n_pad > 80


def test_an_unlisted_id_copies_AIR_so_a_stray_id_acts_like_empty_space():
    t = _built_live()
    assert t["LIKE_MAT"][40] == 0 and t["_DENS_ARR"][40] == t["_DENS_ARR"][0]
    assert t["LIKE_FLUID"][40] == 0 and t["_FDENS_ARR"][40] == 0.0


def test_a_listed_seed_row_keeps_its_own_thermal_numbers_and_names_its_like():
    t = _built_live()
    cu = SP.get("copper")
    m = cu.solid_states[SP.BOUND]
    assert t["_DENS_ARR"][m] == np.float32(cu.get("rho_s"))
    assert t["_KSOLID_ARR"][m] == np.float32(cu.get("k_s"))
    assert t["LIKE_MAT"][m] == 3                         # IRON stands in mechanically
    assert t["MELT"][m] == (1085.0, SP.get("copper").liquid, 209.0)
    assert t["FREEZE"][SP.get("copper").liquid] == (1085.0, m, 209.0)
    e = SP.get("ethanol").liquid
    assert t["_FDENS_ARR"][e] == np.float32(0.789) and t["LIKE_FLUID"][e] == 1


def test_a_legacy_id_is_its_own_like():
    t = _built_live()
    assert list(t["LIKE_MAT"][:12]) == list(range(12))
    assert list(t["LIKE_FLUID"][:9]) == list(range(9))


def test_pad_like_refuses_a_forward_like_and_copies_backwards():
    assert LT.pad_like([1, 2, 3], [0, 1, 2, 1, 0], 5) == [1, 2, 3, 2, 1]
    with pytest.raises(ValueError):
        LT.pad_like([1, 2], [0, 1, 3, 0], 4)
    with pytest.raises(ValueError):
        LT.pad_like([1, 2], [0, 1, 2], 3)               # copies itself


def test_build_refuses_bad_seed_rows():
    leg = SP.legacy_inputs()
    bad = [{"mat": 5, "like": 0, "rho": 1, "c": 1, "k": 1, "ign": None, "emiss": 1}]
    with pytest.raises(ValueError):
        LT.build_tables(leg, bad, [], 64, 64)           # id inside the legacy range
    dup = SP.seed_mat_rows()[:1] * 2
    with pytest.raises(ValueError):
        LT.build_tables(leg, dup, [], 64, 64)
    with pytest.raises(ValueError):
        LT.build_tables(leg, [], [], 8, 64)


def test_with_no_seed_rows_the_padding_is_all_AIR_and_the_legacy_part_is_unchanged():
    leg = SP.legacy_inputs()
    a = LT.build_tables(leg, [], [], 12, 9)
    b = LT.build_tables(leg, [], [], 64, 64)
    for name in ARRAYS:
        n = 9 if name in FLUID_ARRAYS else 12
        assert a[name].tobytes() == b[name][:n].tobytes()
    assert (b["_DENS_ARR"][12:] == b["_DENS_ARR"][0]).all()


def test_the_build_is_deterministic():
    a, b = _built_live(), _built_live()
    for k, v in a.items():
        if isinstance(v, np.ndarray):
            assert v.tobytes() == b[k].tobytes()


# ---- [C] code-only: the code against the live materials.py ------------------
def test_codeonly_the_rebuild_equals_the_live_materials_arrays():
    """Passes by construction on the tree it was written for; NOT a gate. On the
    live tree it shows whether wave 12 moved a legacy row (then the oracle test
    above is the one that matters, and the diff is the review).

    The live arrays are 64 long since wave 13 (B3) and are built from the material RECORD
    (matrec: the twelve old materials and ICE), not from the species registry, so the two
    agree on the ids the record has a row for (0..12; liquids 0..8) and differ past that:
    the registry has the planned copper, quartz ... rows, the live arrays pad them with
    AIR until their records arrive (tests/test_mattables.py)."""
    from src.voxel import materials as M
    t = SP.padded_tables()
    for name in ARRAYS:
        live = getattr(M, name, None)
        if live is None:
            continue
        n = 9 if name in FLUID_ARRAYS else M.NMAT_ROWS
        assert len(live) == 64 and t[name][:n].tobytes() == live[:n].tobytes(), name
    assert t["BOIL"] == M.BOIL and t["LATENT"] == M.LATENT


# ---- module hygiene ----------------------------------------------------------
def test_legacy_tables_imports_nothing_at_the_top_and_no_world():
    path = os.path.join(HERE, "..", "src", "voxel", "legacy_tables.py")
    tree = ast.parse(open(path).read())
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert names == ["numpy"], names                  # numpy, inside the function only
    assert not any(x for x in names if "materials" in x or "sim" in x or "World" in x)


def test_the_cell_table_reads_bulk_density_for_a_packed_mat_and_particle_density_for_rock():
    """[I] sweep over every listed seed mat: its density in the padded table is the float32
    of species.bulk_density(species, form). Sand (mat 15, GRAIN) is the packed bed, 1590;
    quartz rock (mat 14, BOUND) is 2650: one species, two readers, two numbers."""
    import numpy as np
    t = SP.padded_tables()
    rows = {r["mat"]: r for r in SP.seed_mat_rows()}
    for m, r in rows.items():
        slot = SP.MATS[m]
        want = SP.bulk_density(slot.species, slot.form)
        assert r["rho"] == want
        assert t["_DENS_ARR"][m] == np.float32(want), m
    assert t["_DENS_ARR"][15] == np.float32(1590.0) and t["_DENS_ARR"][14] == np.float32(2650.0)
    assert t["_DENS_ARR"][15] < 0.7 * t["_DENS_ARR"][14]
