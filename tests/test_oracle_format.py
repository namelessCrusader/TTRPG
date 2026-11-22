"""The frozen oracles of the old material tables (tests/data/matrec_oracle_base19.json,
the current one, and matrec_oracle_base18.json, kept) and the tool that makes them
(tools/matrec_extract.py).

These tests check the FORMAT and the TOOL, not the physics: schema keys, that
every array decodes to the right dtype, shape and bytes, that the number of
fluid slots is pinned, that the extractor is repeatable, and that diff mode
prints exactly the rows that differ.

The checks that compare the oracle with a LIVE tree compare its LEGACY VIEW: wave 13 (lane
B3) pads every lookup array to 64 entries and adds constants and rows, and none of that is
a change of an old table. So an array is cut to the oracle's own length first
(tools/matrec_extract.py `legacy_view`, `diff --legacy`), and the tables, constants and rows
the oracle does not list are ignored. Everything the oracle lists must still be there,
bit for bit, and these checks always run (they used to skip once materials.py changed).
B0 (phase B, 13a) read the base18 -> base19 diff first (11 rows, all in "other": no table
row moved, see test_wave12_changed_no_table_row), THEN froze the base19 oracle.
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORACLE = os.path.join(HERE, "data", "matrec_oracle_base19.json")
ORACLE18 = os.path.join(HERE, "data", "matrec_oracle_base18.json")     # kept: the tree before wave 12
TOOL = os.path.join(ROOT, "tools", "matrec_extract.py")
BASE18 = os.environ.get("W13_BASE18", "/home/nisargparikh/.cache/mark1-build/base18")

_spec = importlib.util.spec_from_file_location("matrec_extract_tool", TOOL)
X = importlib.util.module_from_spec(_spec)
sys.modules["matrec_extract_tool"] = X
_spec.loader.exec_module(X)


@pytest.fixture(scope="module")
def doc():
    return X.load_oracle(ORACLE)


# ── format ──────────────────────────────────────────────────────────────────

TOP_KEYS = {"schema", "label", "source_sha256", "ids", "fluid_slots", "nmat", "flids",
            "scalars", "tables", "arrays", "other", "mpm_mat", "bond", "carried", "as_body"}
NEED_TABLES = ("SOLID", "EMISS", "SPAN", "TOUGH", "TSHOCK", "MELT", "FREEZE", "PYRO",
               "O2_RATIO", "SOOT", "STOCK", "FLUID", "EMISS_FLUID", "FLUID_K")
NEED_ARRAYS = ("_DENS_ARR", "_CSOLID_ARR", "_KSOLID_ARR", "_IGN_ARR", "_SPAN_ARR",
               "_TOUGH_ARR", "_TSHOCK_ARR", "_EMISS_ARR", "_SKIN_DEPTH_ARR",
               "_SKIN_PATH_ARR", "_FSKIN_DEPTH_ARR", "_FSKIN_PATH_ARR", "_FDENS_ARR",
               "_FC_ARR", "_FK_ARR", "_FEMISS_ARR", "_FLK_ARR", "_TRANSMIT_ARR")
FLUID_ARRAYS = ("_FDENS_ARR", "_FC_ARR", "_FK_ARR", "_FEMISS_ARR", "_FLK_ARR",
                "_FSKIN_DEPTH_ARR", "_FSKIN_PATH_ARR", "_FALPHA")


def test_schema_keys(doc):
    assert set(doc) == TOP_KEYS
    assert doc["schema"] == X.SCHEMA == 1
    for t in NEED_TABLES:
        assert t in doc["tables"], t
    for a in NEED_ARRAYS:
        assert a in doc["arrays"], a
    assert set(doc["bond"]) == {"ROW", "NO_BOND", "consts", "pairs"}
    for k in ("BOIL", "LATENT", "NMAT"):
        assert k in doc["scalars"]
    assert set(doc["source_sha256"]) == set(X.SOURCES)


def test_array_entries_decode_to_dtype_shape_and_bytes(doc):
    for name, e in doc["arrays"].items():
        assert set(e) == {"dtype", "shape", "hex", "values"}, name
        a = X.decode_array(e)
        assert a.dtype == np.dtype(e["dtype"]), name
        assert list(a.shape) == e["shape"], name
        assert a.tobytes().hex() == e["hex"], name
        # the readable values are the same numbers as the bytes
        assert np.array_equal(np.asarray(X.dec(e["values"]), dtype=a.dtype), a, equal_nan=True), name


def test_array_dtypes_are_the_ones_the_sim_uses(doc):
    for name, e in doc["arrays"].items():
        want = "<f8" if name in ("_ALPHA", "_FALPHA") else "<f4"
        assert e["dtype"] == want, name


def test_fluid_slot_count_is_pinned(doc):
    # one slot per fluid in _FLIDS plus the empty slot 0: eight fluids, nine slots
    assert doc["fluid_slots"] == 9
    assert doc["fluid_slots"] == len(doc["flids"]) + 1
    assert doc["flids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    for name in FLUID_ARRAYS:
        assert doc["arrays"][name]["shape"] == [9], name
    assert len(doc["tables"]["_FDENS"]) == 9
    assert len(doc["tables"]["FLUID"]) == 8
    assert doc["tables"]["_FDENS"][0] == 0.0 and doc["tables"]["_FK"][0] == 0.0


def test_material_slot_count_is_pinned(doc):
    assert doc["nmat"] == doc["scalars"]["NMAT"] == 12
    assert len(doc["tables"]["SOLID"]) == 12
    for name, e in doc["arrays"].items():
        if name not in FLUID_ARRAYS:
            assert e["shape"] == [12], name


def test_row_shapes(doc):
    t = doc["tables"]
    assert all(len(r) == 6 for r in t["SOLID"].values())
    assert all(len(r) == 7 for r in t["FLUID"].values())
    assert all(len(r) == 3 for r in t["MELT"].values())
    assert all(len(r) == 3 for r in t["PYRO"].values())
    assert set(t["MELT"]) == {"3", "9", "10"}                   # iron, tin, lead
    assert set(t["FREEZE"]) == {"5", "6", "7"}                  # by molten fluid id
    assert t["TSHOCK"] == {"6": 120.0}
    # a missing ignition is null, not inf
    assert doc["tables"]["SOLID"]["0"][3] is None
    # the infinite entries of the lookup arrays survive as real infinities
    ign = X.decode_array(doc["arrays"]["_IGN_ARR"])
    assert np.isinf(ign[0]) and ign[1] == np.float32(300.0)


def test_mpm_and_bond_rows(doc):
    assert set(doc["mpm_mat"]) >= {"wood", "stone", "iron", "glass", "flesh", "lead",
                                   "leaf", "ash", "char", "tin", "rope"}
    assert all(len(r) == 7 for r in doc["mpm_mat"].values())
    assert doc["mpm_mat"]["glass"][5] == 5.0e7 and doc["mpm_mat"]["iron"][3] == 0.002
    assert doc["bond"]["NO_BOND"] == [5]
    assert set(doc["bond"]["ROW"]) == {str(i) for i in (6, 2, 3, 9, 10, 4, 8, 7, 11)}
    assert all(len(r) == 5 for r in doc["bond"]["ROW"].values())
    # pair rows: 12 materials, ordered pairs a <= b, five grain angles
    assert len(doc["bond"]["pairs"]) == (12 * 13 // 2) * 5
    assert doc["bond"]["pairs"]["5,5,1.0"] is None              # ash holds nothing
    assert doc["bond"]["pairs"]["1,1,1.0"][0] == 1.1e10         # wood along the grain


def test_carried_and_as_body_are_recorded_as_data(doc):
    assert doc["carried"] == ["mat", "smass", "E", "fl", "fvol", "fpot", "edge",
                              "fallh", "rubble", "eid"]
    assert doc["as_body"] == ["mats", "masses", "Es", "fls", "fvols", "fpots",
                              "edges", "fallhs", "rubbles", "eids"]
    assert len(doc["carried"]) == len(doc["as_body"])


def test_ids_name_every_slot(doc):
    assert len(doc["ids"]["mat"]) == 12 and len(doc["ids"]["fluid"]) == 9
    assert doc["ids"]["mat"]["1"] == "WOOD" and doc["ids"]["fluid"]["5"] == "MTIN"


# ── encoding ────────────────────────────────────────────────────────────────

def test_non_finite_floats_round_trip():
    v = X.enc({"a": float("inf"), "b": [float("-inf"), float("nan"), 1.5], "c": None})
    text = json.dumps(v, allow_nan=False)
    back = X.dec(json.loads(text))
    assert back["a"] == float("inf") and back["b"][0] == float("-inf")
    assert back["b"][1] != back["b"][1] and back["b"][2] == 1.5 and back["c"] is None


def test_float_text_is_exact():
    for f in (0.1, 1e-300, 2.2250738585072014e-308, 1.7976931348623157e308, 20.0):
        assert json.loads(json.dumps(X.enc(f))) == f
    a = np.array([0.1, 3.4e38, -0.0, 1e-45], np.float32)
    b = X.decode_array(X.enc_array(a))
    assert a.tobytes() == b.tobytes()


def test_file_text_is_the_canonical_dump(doc):
    with open(ORACLE, "r", encoding="utf-8") as fh:
        text = fh.read()
    assert X.dumps(doc) == text
    assert text.endswith("\n") and "\r" not in text


# ── repeatability ───────────────────────────────────────────────────────────

def test_extractor_is_byte_identical_twice(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    X.write(ROOT, str(a))
    X.write(ROOT, str(b))
    assert a.read_bytes() == b.read_bytes()


def test_extractor_does_not_write_into_the_tree(tmp_path):
    before = {os.path.join(d, f) for d, _, fs in os.walk(os.path.join(ROOT, "src")) for f in fs}
    X.write(ROOT, str(tmp_path / "a.json"))
    after = {os.path.join(d, f) for d, _, fs in os.walk(os.path.join(ROOT, "src")) for f in fs}
    assert after == before


@pytest.mark.skipif(not os.path.isdir(BASE18), reason="base18 tree not on this machine")
def test_base18_extraction_equals_the_frozen_file(tmp_path):
    out = tmp_path / "o.json"
    X.write(BASE18, str(out))
    with open(ORACLE18, "rb") as fh:
        assert out.read_bytes() == fh.read()


# the review B0 did before freezing the base19 oracle: wave 12 changed these rows and no others
WAVE12_ROWS = {
    "ADDED    other/AIR_O2_MASS", "REMOVED  other/COMBUST_CEIL", "ADDED    other/C_GAS",
    "ADDED    other/GLOW_COLORS", "ADDED    other/GLOW_RED", "ADDED    other/IGNITER_W/candle",
    "ADDED    other/IGNITER_W/lamp", "ADDED    other/IGNITER_W/taper",
    "ADDED    other/IGNITER_W/torch", "ADDED    other/LAMBDA_FREE", "REMOVED  other/T_FLAME"}


def test_wave12_changed_no_table_row():
    """base18 -> base19, row by row: only constants of the flame law (FIRE) moved; no
    material, fluid, melt, burn or bond row, no lookup array, no CARRIED field."""
    a, b = X.load_oracle(ORACLE18), X.load_oracle(ORACLE)
    lines = X.diff_docs(a, b)
    heads = {l.split(":")[0] for l in lines}
    assert heads == WAVE12_ROWS, sorted(heads ^ WAVE12_ROWS)
    for k in ("tables", "arrays", "scalars", "mpm_mat", "bond", "carried", "as_body", "ids", "flids"):
        assert a[k] == b[k], k
    assert a["source_sha256"]["src/voxel/materials.py"] != b["source_sha256"]["src/voxel/materials.py"]


def test_live_module_arrays_equal_the_oracle_bit_for_bit(doc):
    """The legacy view of the live arrays: the first rows of each, as many as the oracle has
    (12 materials, 9 liquids); the arrays are 64 long since wave 13 (B3)."""
    from src.voxel import materials as M
    for name, e in doc["arrays"].items():
        a = getattr(M, name)
        n = e["shape"][0]
        assert a.dtype == np.dtype(e["dtype"]), name
        assert a.ndim == len(e["shape"]) and a.shape[0] == (M.NFLUID_PAD if name.startswith("_F")
                                                           else M.NMAT_PAD), name
        assert a[:n].tobytes().hex() == e["hex"], name
    assert len(M._FLIDS) + 1 == doc["fluid_slots"]
    assert M._FDENS_ARR.shape[0] >= doc["fluid_slots"]
    assert list(map(float, M._FDENS)) == doc["tables"]["_FDENS"]
    assert M.NMAT == doc["nmat"]
    for mid, row in doc["tables"]["SOLID"].items():
        assert list(M.SOLID[int(mid)]) == row, mid
    from src.voxel.bodies import CARRIED, AS_BODY
    assert list(CARRIED) == doc["carried"] and list(AS_BODY) == doc["as_body"]


# Declared physics changes since base19: the oracle stays frozen, and the new value is named here.
# SOOT_OIL 1.0 -> 1.6 (wave 14a refinement, lane R1): liquid hydrocarbons give 0.037-0.042 g/g
# of soot, the old 1.0 x 0.025 was about 1.5x too clear.
DECLARED_SCALAR_CHANGES = {"SOOT_OIL": 1.6}


def test_every_table_of_the_live_module_equals_the_oracle_row_for_row(doc):
    """All of them, not only SOLID: wave 13 made these names views of the material record, and
    the oracle (frozen from the literal tables of base19) must still read the same."""
    from src.voxel import materials as M
    for name, rows in doc["tables"].items():
        assert X.enc(getattr(M, name)) == rows, name
    for name, v in doc["scalars"].items():
        if name in DECLARED_SCALAR_CHANGES:
            assert X.enc(getattr(M, name)) == DECLARED_SCALAR_CHANGES[name], name
            continue
        assert X.enc(getattr(M, name)) == v, name
    for name, v in doc["other"].items():
        assert X.enc(getattr(M, name)) == v, name


def test_the_live_tree_diffed_against_the_oracle_is_empty_on_its_legacy_view(doc):
    """The whole record at once, by the tool: the live tree is extracted in a child process, as
    the tool always does, cut down to what the oracle lists, and diffed. Tables, arrays,
    constants, particle rows, bond rows and pair rows, CARRIED and AS_BODY."""
    live = X.legacy_view(X.extract(ROOT, "live"), doc)
    declared = {f"CHANGED  scalars/{k}: {doc['scalars'][k]} -> {v}" for k, v in DECLARED_SCALAR_CHANGES.items()}
    assert [l for l in X.diff_docs(doc, live) if l not in declared] == []


# ── diff mode ───────────────────────────────────────────────────────────────

def test_diff_of_a_doc_with_itself_is_empty(doc):
    assert X.diff_docs(doc, copy.deepcopy(doc)) == []


def test_diff_prints_each_changed_row(doc):
    b = copy.deepcopy(doc)
    b["tables"]["SOLID"]["1"][5] = 21.0                         # wood burn rate cap
    del b["tables"]["TOUGH"]["9"]                               # tin toughness row removed
    b["tables"]["FLUID"]["9"] = [1.0, 1.0, 0.5, None, 0.0, 0.0, 1.0]   # a new fluid row
    b["mpm_mat"]["wood"][1] = 1.2e10
    b["scalars"]["BOIL"] = 99.0
    e = b["arrays"]["_EMISS_ARR"]
    arr = X.decode_array(e)
    arr[3] = np.float32(0.85)                                   # iron emissivity
    b["arrays"]["_EMISS_ARR"] = X.enc_array(arr)
    lines = X.diff_docs(doc, b)
    text = "\n".join(lines)
    assert len(lines) == 6, text
    assert any(l.startswith("CHANGED  tables/SOLID/1 (WOOD):") for l in lines)
    assert any(l.startswith("REMOVED  tables/TOUGH/9 (TIN):") for l in lines)
    assert any(l.startswith("ADDED    tables/FLUID/9:") for l in lines)
    assert any(l.startswith("CHANGED  mpm_mat/wood:") for l in lines)
    assert any(l.startswith("CHANGED  scalars/BOIL:") for l in lines)
    assert any(l.startswith("CHANGED  arrays/_EMISS_ARR/3:") for l in lines)


def test_diff_sees_a_changed_bit_that_prints_the_same(doc):
    b = copy.deepcopy(doc)
    arr = X.decode_array(b["arrays"]["_DENS_ARR"])
    arr[0] = np.nextafter(arr[0], np.float32(1e9))
    entry = X.enc_array(arr)
    entry["values"] = b["arrays"]["_DENS_ARR"]["values"]        # same printed values
    b["arrays"]["_DENS_ARR"] = entry
    lines = X.diff_docs(doc, b)
    assert len(lines) == 1 and "arrays/_DENS_ARR/0" in lines[0]


def test_diff_sees_a_changed_array_shape(doc):
    b = copy.deepcopy(doc)
    arr = np.append(X.decode_array(b["arrays"]["_SPAN_ARR"]), np.float32(3.0))
    b["arrays"]["_SPAN_ARR"] = X.enc_array(arr)
    lines = X.diff_docs(doc, b)
    assert any("_SPAN_ARR/#layout" in l for l in lines)
    assert any(l.startswith("ADDED    arrays/_SPAN_ARR/12") for l in lines)


def test_cli_diff_between_a_tree_and_a_file(tmp_path):
    """The command line: a tree folder against an oracle file. With --legacy the tree is cut
    down to what the oracle lists (wave 13 padded its arrays); without it the padding shows."""
    r = subprocess.run([sys.executable, TOOL, "diff", ORACLE, ROOT, "--strict", "--legacy"],
                       capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines() if "changed row(s)" not in l]
    assert lines == ["CHANGED  scalars/SOOT_OIL: 1.0 -> 1.6"], r.stdout[-400:]   # the one declared change (R1)
    r = subprocess.run([sys.executable, TOOL, "diff", ORACLE, ROOT, "--strict"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "CHANGED  arrays/_DENS_ARR/#layout" in r.stdout


# ── the legacy view ─────────────────────────────────────────────────────────

def _grown(doc):
    """A copy of the oracle as a tree that has grown looks: arrays padded, a new array, a new
    table row, a new constant."""
    live = copy.deepcopy(doc)
    for name in ("_DENS_ARR", "_FDENS_ARR"):
        a = X.decode_array(live["arrays"][name])
        live["arrays"][name] = X.enc_array(np.concatenate([a, np.arange(52, dtype=a.dtype)]))
    live["arrays"]["NEW_ARR"] = X.enc_array(np.zeros(5, np.float32))
    live["tables"]["SOLID"]["12"] = [917.0, 2.09, 2.2, None, 0.0, 0.0]
    live["other"]["NMAT_PAD"] = 64
    live["scalars"]["NEW"] = 1
    live["bond"]["ROW"]["12"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    return live


def test_legacy_view_of_a_grown_tree_has_nothing_to_diff(doc):
    live, before = _grown(doc), copy.deepcopy(doc)
    raw = copy.deepcopy(live)
    assert X.diff_docs(doc, live) != []                                  # the growth is visible raw
    assert X.diff_docs(doc, X.legacy_view(live, doc)) == []
    assert doc == before and live == raw                                 # neither input is touched


def test_legacy_view_still_sees_a_changed_old_entry_a_lost_row_and_a_short_array(doc):
    live = _grown(doc)
    a = X.decode_array(live["arrays"]["_DENS_ARR"])
    a[3] = np.float32(7801.0)                                             # iron, an OLD entry
    live["arrays"]["_DENS_ARR"] = X.enc_array(a)
    del live["tables"]["TOUGH"]["9"]                                      # an old row lost
    live["arrays"]["_FC_ARR"] = X.enc_array(X.decode_array(live["arrays"]["_FC_ARR"])[:8])   # a short array
    live["scalars"]["BOIL"] = 99.0
    lines = X.diff_docs(doc, X.legacy_view(live, doc))
    text = "\n".join(lines)
    assert any(l.startswith("CHANGED  arrays/_DENS_ARR/3:") for l in lines), text
    assert any(l.startswith("REMOVED  tables/TOUGH/9") for l in lines), text
    assert any("_FC_ARR/#layout" in l for l in lines), text
    assert any(l.startswith("CHANGED  scalars/BOIL") for l in lines), text
    assert not any("_DENS_ARR/12" in l or "NMAT_PAD" in l or "NEW" in l for l in lines), text


def test_legacy_view_keeps_an_array_the_oracle_has_and_the_tree_lost(doc):
    live = _grown(doc)
    del live["arrays"]["_SPAN_ARR"]
    lines = X.diff_docs(doc, X.legacy_view(live, doc))
    assert lines and all(l.startswith("REMOVED  arrays/_SPAN_ARR/") for l in lines)


def test_cli_diff_strict_exit_code(tmp_path, doc):
    b = copy.deepcopy(doc)
    b["tables"]["EMISS"]["3"] = 0.81
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(X.dumps(doc))
    pb.write_text(X.dumps(b))
    r = subprocess.run([sys.executable, TOOL, "diff", str(pa), str(pb)], capture_output=True, text=True)
    assert r.returncode == 0 and "CHANGED  tables/EMISS/3 (IRON): 0.8 -> 0.81" in r.stdout
    assert r.stdout.strip().endswith("1 changed row(s)")
    r = subprocess.run([sys.executable, TOOL, "diff", str(pa), str(pb), "--strict"],
                       capture_output=True, text=True)
    assert r.returncode == 1
