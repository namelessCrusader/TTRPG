"""Wave 13a, lane B3: the material tables, padded to 64, and the record under them.

materials.py's per-id lookup arrays are NMAT_PAD = NFLUID_PAD = 64 long, its old
per-material dicts are VIEWS of the material record (src/voxel/matrec.py), and ICE is
row 12 of the record (icephase.ICE_ROW, ingested by matrec). What this file holds fixed:

  1. the old part of every array is the frozen oracle's, byte for byte, and the padding
     follows ONE copy rule (an id with no row of the record copies AIR / NOFLUID);
  2. every old table is the base19 table in value, number TYPE and key ORDER (the JSON
     oracle sorts its keys and cannot see order; `_law_melt` walks MELT and FREEZE in it);
  3. ICE is row 12: the numbers of icephase.ICE_ROW, in none of the old dicts or melt tables;
  4. a retired material keeps its id (tests/data/mat_ids.json pins the ids), and a new row
     is a ROW, not a code change;
  5. every array a cell's material id indexes is as long as the pad, and nothing here
     depends on which module was imported first;
  6. no reader sizes a per-id table by the old count, or loops over the padded ids and reads
     an old dict (a scan of src/, tools/ and tests/, tested on base19's own lines), and the one phase A
     reader that did, slownet.thermal_net, runs on a hearth beside an ICE block.

Tags (plan section 9): the oracle, repr, arithmetic and copy-rule tests are [I]
(independent of the code under test: frozen data or closed form); the ICE numbers are
(u) as icephase has them, and the tests only check they ARE icephase's.
"""
import ast
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from src.voxel import fluids as F
from src.voxel import icephase as IP
from src.voxel import materials as M
from src.voxel import matrec as MR
from src.voxel import mattables as MT
from src.voxel import species as SP
from src.voxel import species_seed as SEED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VOX = os.path.join(ROOT, "src", "voxel")
ORACLE = os.path.join(HERE, "data", "matrec_oracle_base19.json")
REPR_PIN = os.path.join(HERE, "data", "materials_legacy_repr_base19.json")
ID_PIN = os.path.join(HERE, "data", "mat_ids.json")

MAT_ARRAYS = ("_TOUGH_ARR", "_TSHOCK_ARR", "_DENS_ARR", "_CSOLID_ARR", "_KSOLID_ARR",
              "_SPAN_ARR", "_EMISS_ARR", "_IGN_ARR", "_ALPHA", "_SKIN_DEPTH_ARR",
              "_SKIN_PATH_ARR", "_TRANSMIT_ARR")
FLUID_ARRAYS = ("_FDENS_ARR", "_FC_ARR", "_FK_ARR", "_FEMISS_ARR", "_FLK_ARR", "_FALPHA",
                "_FSKIN_DEPTH_ARR", "_FSKIN_PATH_ARR")
THERMAL_MAT = ("_DENS_ARR", "_CSOLID_ARR", "_KSOLID_ARR", "_IGN_ARR", "_EMISS_ARR", "_ALPHA",
               "_SKIN_DEPTH_ARR", "_SKIN_PATH_ARR")
MOLTEN = {"mtin": M.MTIN, "mlead": M.MLEAD, "miron": M.MIRON}


def _json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _oracle_array(name):
    e = _json(ORACLE)["arrays"][name]
    return np.frombuffer(bytes.fromhex(e["hex"]), dtype=np.dtype(e["dtype"])).reshape(e["shape"])


# ── 1. padded arrays: the old part is the oracle ────────────────────────────────

@pytest.mark.parametrize("name", MAT_ARRAYS + FLUID_ARRAYS)
def test_every_array_is_64_long_and_its_old_part_is_the_oracle_byte_for_byte(name):
    want = _oracle_array(name)
    live = getattr(M, name)
    pad = M.NFLUID_PAD if name in FLUID_ARRAYS else M.NMAT_PAD
    assert live.dtype == want.dtype and live.shape == (pad,) == (64,), name
    assert live[:len(want)].tobytes() == want.tobytes(), name
    assert len(want) == (M.NFLUID if name in FLUID_ARRAYS else M.NMAT)


def test_the_sizes_are_pinned_and_NMAT_still_counts_the_ids_of_the_old_dict_tables():
    assert (M.NMAT, M.NFLUID, M.NMAT_ROWS, M.NMAT_PAD, M.NFLUID_PAD) == (12, 9, 29, 64, 64)    # 12 old, ICE, the registry's planned 13..26, soil and grass (lane F)
    assert M.NFLUID == len(M._FLIDS) + 1 == _json(ORACLE)["fluid_slots"]
    assert M.NMAT_PAD == SEED.NMAT_PAD and M.NFLUID_PAD == SEED.NFLUID_PAD
    assert M.ICE == 12 == MR.ICE == SEED.MAT_ICE
    assert (M.MTIN, M.MLEAD, M.MIRON) == (5, 6, 7)
    # NMAT is the domain of the old dicts: `SOLID[m] for m in range(NMAT)` must work
    for name in ("SOLID", "EMISS", "SPAN", "TOUGH"):
        assert sorted(getattr(M, name)) == list(range(M.NMAT)), name
    assert M.ICE not in M.SOLID and M.ICE not in M.MELT and M.ICE not in M.MATNAME
    # the ids that have a row of the record
    assert [r.id for r in M.MATREC.by_id_list] == list(range(M.NMAT_ROWS))
    assert M.MATREC.nmat == M.NMAT_ROWS and MR.BASE.nmat == 12     # the shipped BASE did not move


def test_a_padded_id_copies_AIR_and_ICE_has_a_row_of_its_own():
    """The one copy rule: an id with no row of the record reads row 0 (AIR; NOFLUID for a
    liquid), so a stray id acts like empty space, never like a material nobody described."""
    for name in MAT_ARRAYS:
        a = getattr(M, name)
        for i in range(M.NMAT_ROWS, M.NMAT_PAD):
            assert a[i].tobytes() == a[0].tobytes(), (name, i)
    for name in FLUID_ARRAYS:
        a = getattr(M, name)
        for i in range(M.NFLUID, M.NFLUID_PAD):
            assert a[i].tobytes() == a[0].tobytes(), (name, i)
    # a row with numbers of its own (the twelve, ICE) is its own `like`; an id with none copies AIR
    n = M.NMAT_ROWS
    assert list(M.LIKE_MAT[:n]) == list(range(n)) and list(M.LIKE_FLUID[:M.NFLUID]) == list(range(M.NFLUID))
    assert (M.LIKE_MAT[n:] == 0).all() and (M.LIKE_FLUID[M.NFLUID:] == 0).all()
    assert M._DENS_ARR[40] == M._DENS_ARR[0] == np.float32(1.2)
    assert M._TOUGH_ARR[40] == np.float32(1e9) and M._TRANSMIT_ARR[40] == 1.0   # empty space
    assert M._FDENS_ARR[40] == 0.0 and M._FC_ARR[40] == 1.0                    # no liquid
    # ICE is not a copy of anything
    assert M._DENS_ARR[12] != M._DENS_ARR[0] and M._SPAN_ARR[12] != M._SPAN_ARR[0]


def test_the_record_route_and_the_species_route_agree_on_every_id_both_list():
    """species.padded_tables (T1: seed rows from the species registry) and materials (the
    record) are two routes to the same thermal numbers on ids 0..12 and fluids 0..8. They
    differ past that on purpose: the record has no row yet for copper and the rest, so
    materials pads them with AIR until the row arrives (C2)."""
    t = SP.padded_tables()
    for name in THERMAL_MAT:
        assert getattr(M, name)[:M.NMAT_PLANNED_ROWS].tobytes() == t[name][:M.NMAT_PLANNED_ROWS].tobytes(), name   # soil and grass (27, 28) are not in the registry
    for name in ("_FDENS_ARR", "_FC_ARR", "_FK_ARR", "_FEMISS_ARR", "_FLK_ARR", "_FALPHA",
                 "_FSKIN_DEPTH_ARR", "_FSKIN_PATH_ARR"):
        assert getattr(M, name)[:M.NFLUID].tobytes() == t[name][:M.NFLUID].tobytes(), name


def test_the_dense_melt_and_freeze_forms_agree_with_the_dicts():
    for m in range(M.NMAT_PAD):
        want = M.MELT.get(m)
        got = (M.MELT_T[m], int(M.MELT_FL[m]), M.MELT_LAT[m])
        assert got == ((np.inf, 0, 0.0) if want is None else want), m
    for f in range(M.NFLUID_PAD):
        want = M.FREEZE.get(f)
        got = (M.FREEZE_T[f], int(M.FREEZE_MAT[f]), M.FREEZE_LAT[f])
        assert got == ((-np.inf, 0, 0.0) if want is None else want), f


# ── 2. the old tables: value, number type and key order ─────────────────────────

# tests/data/materials_legacy_repr_base19.json was made ONCE from the base19 tree's materials.py (its sha256
# is in the file): for each name, repr(the module attribute), and repr(SOUND["ring"]) under "SOUND.ring".
# To re-freeze from another tree, import its materials.py and dump {name: repr(value)} for the names in
# the file. A number of an old table is changed in the record (matrec.py) as a DECLARED diff, and this
# file is then re-frozen in the same commit, with the diff shown: it must never move silently.
_PIN = _json(REPR_PIN)


def _live(name):
    return M.SOUND["ring"] if name == "SOUND.ring" else getattr(M, name)


@pytest.mark.parametrize("name", sorted(_PIN["repr"]))
def test_every_old_table_is_the_base19_table_in_value_type_and_key_order(name):
    """repr() shows 0.9 against 0.90000001, 1 against 1.0 and the order the keys were listed
    in. base19's own text is pinned in tests/data/materials_legacy_repr_base19.json."""
    assert repr(_live(name)) == _PIN["repr"][name], name


def test_the_old_tables_are_views_of_the_record():
    v = MR.legacy_views((M.MTIN, M.MLEAD, M.MIRON), M.MATREC, upto=M.NMAT)
    for name in ("SOLID", "EMISS", "SPAN", "TOUGH", "TSHOCK", "MELT", "FREEZE", "PYRO", "O2_RATIO",
                 "SOOT", "GLOW", "BURN_AREA", "TRANSMIT", "MATNAME"):
        assert getattr(M, name) == v[name], name
    assert M.SOUND["ring"] == v["SOUND_RING"]
    # the record holds the twelve old materials, ICE, the registry's planned 13..26 and the ground rows
    assert [r.key for r in M.MATREC.by_id_list][:13] == [
        "air", "wood", "stone", "iron", "flesh", "ash", "glass", "leaf", "char", "tin", "lead",
        "rope", "ice"]
    assert [r.key for r in M.MATREC.by_id_list][13:] == [
        "copper", "quartz_rock", "sand", "calcite", "clay", "graphite", "magnetite", "iron_oxide", "salt",
        "tallow", "sugar", "quicklime", "slaked_lime", "humus", "soil", "grass"]


def test_a_row_the_old_order_does_not_name_is_refused():
    """A row added to or dropped from an old table is a change somebody must make on purpose."""
    with pytest.raises(ValueError):
        M._in_legacy_order({1: "a", 2: "b"}, (1,))
    with pytest.raises(ValueError):
        M._in_legacy_order({1: "a"}, (1, 2))
    with pytest.raises(ValueError):
        M._in_legacy_order({1: "a", 2: "b"}, (1, 1))
    assert list(M._in_legacy_order({1: "a", 2: "b"}, (2, 1))) == [2, 1]


# ── 3. ICE is row 12 ────────────────────────────────────────────────────────────

def test_ICE_is_row_12_and_its_numbers_are_icephase_ICE_ROW():
    row, f32 = IP.ICE_ROW, np.float32
    ice = M.MATREC.row("ice")
    assert ice is M.MATREC.row(12) and ice.id == 12 and ice.mu_class == "ice"
    assert (ice.rho, ice.cp / 1000.0, ice.k, ice.eps) == (row["rho"], row["c"], row["k"], row["emissivity"])
    assert M._DENS_ARR[12] == f32(row["rho"]) and M._CSOLID_ARR[12] == f32(row["c"])
    assert M._KSOLID_ARR[12] == f32(row["k"]) and M._EMISS_ARR[12] == f32(row["emissivity"])
    assert np.isinf(M._IGN_ARR[12]) and np.isinf(M._TSHOCK_ARR[12])
    assert M._SPAN_ARR[12] == f32(row["span"]) == 40.0
    assert M._TOUGH_ARR[12] == f32(row["tough"]) and M._TRANSMIT_ARR[12] == f32(row["transmit"])
    alpha = row["k"] / (row["rho"] * row["c"] * 1000.0)
    assert M._ALPHA[12] == pytest.approx(alpha, rel=1e-14)
    assert M._SKIN_DEPTH_ARR[12] == f32(np.sqrt(M._ALPHA[12] * M.SKIN_T[0]))
    assert M._SKIN_PATH_ARR[12] == f32(np.sqrt(M._ALPHA[12] * M.SKIN_T[1]))
    # ICE melts by T6's law, not by a row of the old tables
    assert 12 not in M.MELT and M.MELT_T[12] == np.inf and M.MELT_FL[12] == 0 and M.MELT_LAT[12] == 0.0
    assert not (M.FREEZE_MAT == 12).any()


def test_ICE_in_the_species_registry_carries_the_same_numbers_as_the_record():
    w = SP.get("water")
    row = IP.ICE_ROW
    assert (w.get("rho_s", "ref"), w.get("c_s", "ref"), w.get("k_s", "ref")) == (row["rho"], row["c"], row["k"])
    assert w.get("emiss_s", "ref") == row["emissivity"]
    assert SP.MATS[12].name == "ice" and SP.MATS[12].species == SP.BY_NAME["water"]


# ── 4. retired materials are never renumbered; a new material is a row ──────────

def _prefix_problem(pinned, current):
    return SP.prefix_problem(pinned, current)


def test_the_ids_are_pinned_so_a_retired_name_keeps_its_id():
    pin = _json(ID_PIN)
    mats = [[m.id, m.name] for m in SP.MATS if m.species is not None]
    fluids = [[f.id, f.name] for f in SP.FLUIDS if f.species is not None]
    assert _prefix_problem(pin["mat"], mats) is None
    assert _prefix_problem(pin["fluid"], fluids) is None
    # the record agrees with the pinned names on every id it has a row for
    names = dict((i, n) for i, n in pin["mat"])
    assert [(r.id, r.key) for r in M.MATREC.by_id_list][:M.NMAT_PLANNED_ROWS] == [(i, names[i]) for i in range(M.NMAT_PLANNED_ROWS)]
    assert [(r.id, r.key) for r in M.MATREC.by_id_list][M.NMAT_PLANNED_ROWS:] == [(27, 'soil'), (28, 'grass')]   # appended past the pin
    # the constants of materials.py are the pinned ids
    for key, mid in (("wood", M.WOOD), ("stone", M.STONE), ("iron", M.IRON), ("flesh", M.FLESH),
                     ("ash", M.ASH), ("glass", M.GLASS), ("leaf", M.LEAF), ("char", M.CHAR),
                     ("tin", M.TIN), ("lead", M.LEAD), ("rope", M.ROPE), ("ice", M.ICE)):
        assert names[mid] == key
    fnames = dict((i, n) for i, n in pin["fluid"])
    assert fnames[M.WATER] == "water" and fnames[M.MIRON] == "molten_iron" and fnames[M.BLOOD] == "blood"


def test_the_pin_catches_a_reorder_a_rename_and_a_removal_and_lets_an_append_pass():
    pin = _json(ID_PIN)["mat"]
    swapped = [list(p) for p in pin]
    swapped[5], swapped[6] = [5, "glass"], [6, "ash"]
    assert _prefix_problem(pin, swapped) is not None
    renamed = [list(p) for p in pin]
    renamed[12] = [12, "frost"]
    assert "id 12" in _prefix_problem(pin, renamed)
    assert "removed" in _prefix_problem(pin, pin[:12])
    assert _prefix_problem(pin, pin + [[27, "zinc"]]) is None


def test_retiring_a_row_keeps_its_id_and_its_numbers_and_a_new_row_never_reuses_the_id():
    t = M.MATREC.retire("ash", "char")
    assert t.row("ash").id == M.ASH == 5 and t.row("ash").successor == "char"
    assert t.resolve(5).key == "char" and 5 not in t.live_ids() and 12 in t.live_ids()
    assert [r.id for r in t.by_id_list] == list(range(M.NMAT_ROWS))      # nothing moved
    # a retired id is still a row of every array: cells that hold it keep their numbers
    v = MR.legacy_views((5, 6, 7), t, upto=M.NMAT_ROWS)
    legacy = _legacy_inputs()
    a = MT.build(legacy, [r for r in t.by_id_list if r.id >= M.NMAT], v, MOLTEN)
    b = MT.build(legacy, [r for r in M.MATREC.by_id_list if r.id >= M.NMAT],
                 MR.legacy_views((5, 6, 7), M.MATREC, upto=M.NMAT_ROWS), MOLTEN)
    for name in MAT_ARRAYS + FLUID_ARRAYS:
        assert a[name].tobytes() == b[name].tobytes(), name
    # and the next new row takes the next free id, not the retired one
    t2 = t.ingest(_TESTIUM)
    assert t2.row("testium").id == M.NMAT_ROWS and t2.row("ash").id == 5


_TESTIUM = {"key": "testium", "rho": 5000.0, "c": 0.5, "k": 20.0, "emissivity": 0.6,
            "E": 1.0e11, "nu": 0.3, "tensile": 2.0e8, "span": 7, "tough_kJm2": 12.0,
            "transmit": 0.0}


def _legacy_inputs():
    return {"SOLID": M.SOLID, "FLUID": M.FLUID, "EMISS": M.EMISS, "EMISS_FLUID": M.EMISS_FLUID,
            "FLUID_K": M.FLUID_K, "MELT": M.MELT, "BOIL": M.BOIL, "LATENT": M.LATENT,
            "SKIN_T": M.SKIN_T, "NMAT": M.NMAT, "FLIDS": M._FLIDS}


def test_a_new_material_is_a_row_not_a_code_change():
    """Ingest one row after the last and build the arrays from the table: the new id has the
    row's numbers in every array, the ids after it copy AIR, and the rows before it do not move."""
    n = M.NMAT_ROWS
    t = M.MATREC.ingest(_TESTIUM)
    assert t.nmat == n + 1 and t.row("testium").id == n
    v = MR.legacy_views((5, 6, 7), t, upto=n + 1)
    got = MT.build(_legacy_inputs(), [r for r in t.by_id_list if r.id >= M.NMAT], v, MOLTEN)
    f32 = np.float32
    assert got["_DENS_ARR"][n] == f32(5000.0) and got["_CSOLID_ARR"][n] == f32(0.5)
    assert got["_KSOLID_ARR"][n] == f32(20.0) and got["_EMISS_ARR"][n] == f32(0.6)
    assert np.isinf(got["_IGN_ARR"][n])
    assert got["_SPAN_ARR"][n] == 7.0 and got["_TOUGH_ARR"][n] == f32(12.0)
    assert got["_TRANSMIT_ARR"][n] == 0.0
    for name in MAT_ARRAYS:
        assert got[name][n + 1:].tobytes() == np.repeat(got[name][:1], 64 - n - 1).tobytes(), name
        assert got[name][:n].tobytes() == getattr(M, name)[:n].tobytes(), name
    # the build refuses rows that skip an id, and more materials than ids
    with pytest.raises(ValueError):
        MT.build(_legacy_inputs(), [t.row("testium")], v, MOLTEN)
    with pytest.raises(ValueError):
        MT.build(_legacy_inputs(), [r for r in t.by_id_list if r.id >= M.NMAT], v, MOLTEN, nmat=n)


# ── 5. nothing indexed by a cell's id is shorter than the pad ───────────────────

def test_the_fluid_kernel_tables_are_64_long_and_a_padded_id_reads_the_none_row():
    mixt, fd, fc, fv, sch, has, mhp, mhq, mhn, fdens = F._tables()
    nf = M.NFLUID
    assert mixt.shape == (64, 64) and sch.shape == (64,)
    for a in (fd, fc, fv, has, mhn, fdens, mhp, mhq):
        assert a.shape[0] == 64
    # the old 9 x 9 block is what the old double loop gave
    old = np.full((nf, nf), -1, np.int16)
    for a in range(nf):
        for b in range(nf):
            old[a, b] = a if b in (M.NOFLUID, a) else int(M.MISCIBLE.get(frozenset({a, b}), -1))
    assert (mixt[:nf, :nf] == old).all()
    # a padded fluid joins nothing but nothing and itself ("do not mix" is -1, never a zero row)
    for a in range(nf, 64):
        assert mixt[a, 0] == a and mixt[a, a] == a
        assert (np.delete(mixt[a], [0, a]) == -1).all() and (np.delete(mixt[:, a], [0, a]) == -1).all()
    assert (fd[nf:] == 0).all() and (fc[nf:] == 0).all() and (fv[nf:] == 0).all() and not has[nf:].any()
    # the specific heat of the solids: the old twelve exactly as the dict holds them
    assert list(sch[:12]) == [float(M.SOLID[m][1]) for m in range(12)]
    assert sch[12] == np.float32(IP.ICE_ROW["c"]) and sch[M.NMAT_ROWS] == sch[0] == 1.0
    film, ret, _tol = F._wet_tab(1)
    assert film.shape == ret.shape == (64,) and (film[nf:] == 0).all() and (ret[nf:] == 0).all()
    sw, wick = F._SORB
    assert sw.shape == (2, 64) and wick.shape == (64,) and (sw[:, M.NMAT:] == 0).all() and (wick[nf:] == 0).all()


# The READERS. A reader written for the 12-id world breaks in the 64-id one in three ways, and each is a real
# site of this repo (the cases below quote them; the scan finds base19's own lines, which the first version
# of this guard could not):
#   a. a table built by `zeros(NMAT)`, or by a loop over `range(NMAT)` / `range(len(<old dict>))` inside
#      np.array(...), is 12 long, and a cell of id 12 (ICE) or more indexes past its end (base19 materials.py,
#      things.py `_IGN_ARR`, fluids.py; heat.py's `_MELTABLE_ARR` is the one still open, P1);
#   b. a loop over the PADDED ids that reads an old dict by the id raises KeyError at id 12 (slownet.py read
#      `SOLID[m] for m in range(len(_DENS_ARR))`: thermal_net died on its first call, B3 fix round);
#   c. a loop `range(NMAT)` that reads no old dict at all builds something per id out of nothing.
# A loop `SOLID[m] for m in range(NMAT)` that makes a LIST or fills a SLICE is what NMAT counts, and is right.

OLD_DICTS = {"SOLID", "EMISS", "SPAN", "TOUGH", "TSHOCK", "MELT", "FREEZE", "PYRO", "O2_RATIO", "SOOT", "GLOW",
             "BURN_AREA", "TRANSMIT", "MATNAME", "FLUID", "FLUID_K", "EMISS_FLUID"}
PAD_COUNTS = {"NMAT_PAD", "NFLUID_PAD", "_NFL"}
PADDED_NAMES = set(MAT_ARRAYS) | set(FLUID_ARRAYS)
TABLE_BUILDERS = {"array", "asarray", "fromiter", "ascontiguousarray"}
TABLE_SIZERS = {"zeros", "ones", "empty", "full"}
LOOPS = (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
# A table another lane's file still builds the old way: (file from the repo root, the name it is bound to).
# Allowed, never required, so the edit that closes it (B3 report E1, needs_other_files) needs no edit here.
KNOWN_OPEN = {("src/voxel/heat.py", "_MELTABLE_ARR")}


def _tail(node):
    """The name an expression ends in: `X` for the Name X and for `a.b.X`."""
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None


def _mentions(node, names):
    return any(_tail(x) in names for x in ast.walk(node))


def _reads_an_old_dict_by_the_id(loop, ids):
    """Does the loop subscript an old dict with its own variable and no `id in dict` guard?"""
    body = ast.Module(body=loop.body + loop.orelse, type_ignores=[]) if isinstance(loop, ast.For) else loop
    guarded = {_tail(c.comparators[0]) for c in ast.walk(body)
               if isinstance(c, ast.Compare) and any(isinstance(o, ast.In) for o in c.ops)}
    return any(isinstance(x, ast.Subscript) and _tail(x.value) in OLD_DICTS - guarded and _mentions(x.slice, ids)
               for x in ast.walk(body))


def _unpadded_sites(source):
    """[(line, what, the name it is bound to)] of every place in `source` that sizes a per-id table by the
    old count, or loops over the padded ids and reads an old dict (the kinds a, b and c above)."""
    tree = ast.parse(source)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}

    def table_of(node):
        """The np.array(...) call whose argument holds this loop, and the name it is bound to."""
        while node in parents and not isinstance(node, ast.stmt):
            node = parents[node]
            if isinstance(node, ast.Call) and _tail(node.func) in TABLE_BUILDERS:
                p = parents.get(node)
                return node, (_tail(p.targets[0]) if isinstance(p, ast.Assign) else None)
        return None, None

    hits = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and n.args and _tail(n.func) in TABLE_SIZERS and _mentions(n.args[0], {"NMAT"}):
            hits.add((n.lineno, "a table sized by NMAT", None))
        loops = [(n, n.iter, n.target)] if isinstance(n, ast.For) else \
            [(n, g.iter, g.target) for g in n.generators] if isinstance(n, LOOPS) else []
        for loop, it, target in loops:
            if not (isinstance(it, ast.Call) and _tail(it.func) == "range" and it.args):
                continue
            ids = {x.id for x in ast.walk(target) if isinstance(x, ast.Name)}
            lens = [a.args[0] for a in it.args if isinstance(a, ast.Call) and _tail(a.func) == "len" and a.args]
            nmat = any(_mentions(a, {"NMAT"}) for a in it.args)
            old = nmat or any(_tail(x) in OLD_DICTS for x in lens)
            padded = any(_mentions(a, PAD_COUNTS) for a in it.args) or any(_tail(x) in PADDED_NAMES for x in lens)
            call, name = table_of(loop)
            if old and call is not None:
                hits.add((it.lineno, "a table built over the old count", name))
            elif nmat and not _mentions(loop, OLD_DICTS):
                hits.add((it.lineno, "range(NMAT) over no old table", None))
            if padded and _reads_an_old_dict_by_the_id(loop, ids):
                hits.add((it.lineno, "the padded ids read an old dict", name))
    return sorted(hits, key=lambda h: (h[0], h[1]))


def _py_files():
    """Every python file that can read the tables: src/, tools/ and tests/ (lab/ never imports src/)."""
    for top in ("src", "tools", "tests"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = sorted(d for d in dn if d not in ("__pycache__", "blender"))
            for f in sorted(fn):
                if f.endswith(".py"):
                    yield os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, "/"), os.path.join(dp, f)


def test_nothing_in_the_repo_sizes_a_per_id_table_by_the_old_count_or_reads_an_old_dict_over_the_padded_ids():
    """B3 changed every such site it found; this keeps a new one from appearing, in src/ (every package), in
    tools/ and in tests/ (a test helper that gathers a 12-long table by a cell's id breaks the day a world
    holds id 12)."""
    bad = {}
    for rel, path in _py_files():
        with open(path, encoding="utf-8") as fh:
            hits = [h for h in _unpadded_sites(fh.read()) if (rel, h[2]) not in KNOWN_OPEN]
        if hits:
            bad[rel] = hits
    assert not bad, bad


# base19's own lines (the table builds of materials.py, things.py, fluids.py, heat.py, slownet.py), each with
# what the scan must say. The first version of this guard saw only the `zeros(NMAT)` ones.
SCAN_HITS = [
    ("_TOUGH_ARR = np.array([TOUGH[m] for m in range(NMAT)], np.float32)",
     ["a table built over the old count"]),                                       # materials.py:750
    ("_TRANSMIT_ARR = np.array([1.0 if m == AIR else TRANSMIT.get(m, 0.0)\n                          "
     "for m in range(NMAT)], np.float32)", ["a table built over the old count"]),  # materials.py:780
    ("_IGN_ARR = np.array([np.inf if SOLID[m][3] is None else SOLID[m][3]\n                     "
     "for m in range(NMAT)], np.float32)", ["a table built over the old count"]),  # things.py:33 (mutation MV2)
    ("sch = np.array([float(SOLID[m][1]) for m in range(NMAT)])",
     ["a table built over the old count"]),                                       # fluids.py:113
    ("sw = np.zeros((2, NMAT))", ["a table sized by NMAT"]),                       # fluids.py:566
    ("on = np.zeros(NMAT, bool)", ["a table sized by NMAT"]),                     # fluids.py:1142
    ("heat = np.zeros(sim.NMAT, np.float32)", ["a table sized by NMAT"]),          # the same, through a module
    ("a_sol = np.array([SOLID[m][2] / (SOLID[m][0] * SOLID[m][1] * 1000.0) if SOLID[m][0] > 0 else 0.0\n"
     "                  for m in range(len(_DENS_ARR))])", ["the padded ids read an old dict"]),   # slownet.py:382
    ("_MELTABLE_ARR = np.array([m in MELT for m in range(len(SOLID))])",
     ["a table built over the old count"]),                                       # heat.py:53 (open, P1)
    ("dens = np.array([sim.SOLID[k][0] for k in range(sim.NMAT)], float)",
     ["a table built over the old count"]),                                       # tools/members_survey.py:98
    ("for m in range(NMAT_PAD):\n    print(SOLID[m])", ["the padded ids read an old dict"]),
    ("rows = [FLUID[f][0] for f in range(NFLUID_PAD)]", ["the padded ids read an old dict"]),
    ("c = [m for m in range(NMAT)]", ["range(NMAT) over no old table"]),
]
SCAN_CLEAN = [
    "d = [SOLID[m][0] for m in range(NMAT)]",                                      # a LIST over the dict's own ids
    "for m in range(NMAT):\n    print(SOLID[m])",
    "sch[:NMAT] = [float(SOLID[m][1]) for m in range(NMAT)]",                     # a slice of a padded table
    "a_sol = np.array(_ALPHA)\na_sol[:NMAT] = [SOLID[m][2] / SOLID[m][0] for m in range(NMAT)]",   # the slownet fix
    "e = np.zeros(NMAT_PAD)\nf = range(NMAT_PAD)\ng = np.zeros((2, NFLUID_PAD))",
    "_FREEZABLE_ARR = np.array([f in FREEZE for f in range(len(_FK_ARR))])",     # heat.py:54: padded, a membership test
    "for m in range(NMAT_PAD):\n    if m in SOLID:\n        print(SOLID[m])",       # guarded by `m in SOLID`
    "x = [MELT[m] for m in range(NMAT_PAD) if m in MELT]",
    "ids = np.array(_FLIDS, np.int64)",
]


@pytest.mark.parametrize("source,kinds", SCAN_HITS, ids=[str(i) for i in range(len(SCAN_HITS))])
def test_the_scan_finds_base19s_own_unpadded_sites(source, kinds):
    got = _unpadded_sites("import numpy as np\n" + source + "\n")
    assert sorted({h[1] for h in got}) == sorted(kinds), (source, got)


@pytest.mark.parametrize("source", SCAN_CLEAN, ids=[str(i) for i in range(len(SCAN_CLEAN))])
def test_the_scan_lets_the_right_shapes_through(source):
    assert _unpadded_sites("import numpy as np\n" + source + "\n") == []


def test_the_scan_reports_the_name_a_table_is_bound_to_and_the_line():
    got = _unpadded_sites("x = 1\n_MELTABLE_ARR = np.array([m in MELT for m in range(len(SOLID))])\n")
    assert got == [(2, "a table built over the old count", "_MELTABLE_ARR")]


def test_thermal_net_reads_the_padded_tables_on_a_hearth_beside_an_ICE_block():
    """The reader the first scan was blind to. slownet.thermal_net built its diffusivity table by looping
    over the PADDED ids and reading SOLID[m], and died on its first call with KeyError: 12 (ICE is in the
    arrays only). A stone hearth beside an ICE block (written into the arrays: World.fill refuses ICE in
    13a) must net, each node must read its own row, and the old twelve must be the dict's own floats."""
    from src.voxel import slownet as N
    from src.voxel.sim import World
    nx = ny = 14
    nz = 10
    w = World(nx, ny, nz, voxel_cm=5)
    w.fill(0, nx, 0, ny, 0, 1, M.STONE)
    w.fill(4, 7, 4, 7, 1, 4, M.STONE)
    ice = (slice(9, 11), slice(9, 11), slice(1, 3))
    w.mat[ice] = M.ICE
    w.smass[ice] = M._DENS_ARR[M.ICE] * w.vox_l
    w.E[4:7, 4:7, 1:4] = (60.0 - M.AMBIENT) * w.heat_capacity()[4:7, 4:7, 1:4]
    res = N.thermal_net(w, max_nodes=100000)
    assert res.B == 1 and res.refusals == []
    a_stone = M.SOLID[M.STONE][2] / (M.SOLID[M.STONE][0] * M.SOLID[M.STONE][1] * 1000.0)
    stone, icy = res.kind == M.STONE, res.kind == M.ICE
    assert stone.any() and icy.sum() == 8
    assert np.allclose(res.alpha[stone], a_stone, rtol=1e-12, atol=0)
    assert np.allclose(res.alpha[icy], M._ALPHA[M.ICE], rtol=1e-12, atol=0)
    assert M._ALPHA[M.ICE] == pytest.approx(IP.ICE_ROW["k"] / (IP.ICE_ROW["rho"] * IP.ICE_ROW["c"] * 1000.0), rel=1e-6)


def test_the_liquid_gather_tables_of_heat_cover_the_pad():
    """heat.py sizes the arrays it gathers by a cell's id from the lookup arrays; the freezable one follows
    the padded _FK_ARR by itself."""
    from src.voxel import heat
    assert len(heat._FREEZABLE_ARR) == M.NFLUID_PAD


def test_the_solid_gather_table_of_heat_covers_the_pad():
    from src.voxel import heat
    assert len(heat._MELTABLE_ARR) == M.NMAT_PAD


# ── 6. imports: no order, no cycle ──────────────────────────────────────────────

def _run(code):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=180)


@pytest.mark.parametrize("first", ["materials", "species", "mixrules", "matrec", "icephase",
                                   "mattables", "legacy_tables", "fluids"])
def test_materials_is_whole_whichever_module_is_imported_first(first):
    """species and mixrules read materials, and materials reads matrec: a cycle if any of them
    imported the others at import. A fresh interpreter, each module first in turn."""
    r = _run("import src.voxel.%s\n"
             "from src.voxel import materials as M, species, mixrules, matrec\n"
             "assert M._DENS_ARR.shape == (64,) and M._FDENS_ARR.shape == (64,) and M.MATREC.nmat == M.NMAT_ROWS\n"
             "n = M.NMAT_PLANNED_ROWS\n"
             "assert species.padded_tables()['_DENS_ARR'][:n].tobytes() == M._DENS_ARR[:n].tobytes()\n"
             "assert matrec._mixrules is mixrules and matrec._species is species\n" % first)
    assert r.returncode == 0, r.stderr[-1500:]


def test_importing_materials_does_not_import_species_or_mixrules():
    r = _run("import sys\nimport src.voxel.materials\n"
             "bad = [m for m in ('species', 'mixrules', 'mixtable', 'sim') if 'src.voxel.' + m in sys.modules]\n"
             "assert not bad, bad\n"
             "assert 'src.voxel.matrec' in sys.modules and 'src.voxel.icephase' in sys.modules\n")
    assert r.returncode == 0, r.stderr[-1500:]


def test_matrec_finds_its_siblings_on_first_use_and_a_missing_one_reads_as_none():
    r = _run("import sys\nfrom src.voxel import matrec\n"
             "assert not [m for m in ('species', 'mixrules', 'icephase') if 'src.voxel.' + m in sys.modules]\n"
             "assert matrec._sibling('no_such_module') is None and matrec._sibling('no_such_module') is None\n"
             "assert matrec._icephase.ICE_ROW['name'] == 'ice' and 'src.voxel.icephase' in sys.modules\n"
             "assert matrec._mixrules is not None and matrec._species is not None\n"
             "try:\n    matrec._nothing\nexcept AttributeError:\n    pass\nelse:\n    raise SystemExit('no AttributeError')\n")
    assert r.returncode == 0, r.stderr[-1500:]


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            out.add(("." * n.level) + (n.module or "") + "|" + ",".join(a.name for a in n.names))
        elif isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
    return out


def test_mattables_is_pure_and_materials_imports_only_the_record_and_the_pad():
    assert _imports(os.path.join(VOX, "mattables.py")) == {
        "numpy", ".legacy_tables|build_tables,pad_like", ".species_seed|NFLUID_PAD,NMAT_PAD"}
    mats = _imports(os.path.join(VOX, "materials.py"))
    assert mats == {"numpy", ".|matrec", ".|mattables", ".species_seed|NFLUID_PAD,NMAT_PAD"}, mats
    # matrec still imports no wave 13 module at import (its siblings are found on first use)
    top = ast.parse(open(os.path.join(VOX, "matrec.py"), encoding="utf-8").read()).body
    assert not [n for n in top if isinstance(n, ast.ImportFrom) and n.level], "matrec imports a sibling at import"
