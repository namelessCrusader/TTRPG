"""Freeze (or diff) the old material tables.

The material record (src/voxel/matrec.py, wave 13 topic T9) rebuilds every
old table from one set of rows. To know the rebuild is right, the old numbers
are frozen ONCE into tests/data/matrec_oracle_base18.json, by this script,
from the base18 tree. The oracle is data: no lane types it, and the T9 and T1
tests only read it.

Two jobs:

  python tools/matrec_extract.py write TREE -o OUT.json [--label NAME]
      Read the materials of the tree TREE (a folder that holds src/) and write
      the frozen JSON. The same tree gives byte-identical output every time.

  python tools/matrec_extract.py diff A B [--strict] [--legacy]
      A and B are each a tree folder or an oracle JSON file. Print every row
      that changed between them (rows added, removed, or changed). Run it at
      base19: the diff of the base18 oracle against a fresh extraction of
      base19 is the review of what wave 12 changed, BEFORE the oracle is
      regenerated (a "rerun and let the oracle decide" always passes, because
      both sides would come from the same file). --strict exits 1 on a diff.
      --legacy (wave 13): A is the frozen oracle and B a tree that has GROWN
      (lookup arrays padded to 64 entries, new constants, new rows). B is cut
      down to what A lists first (`legacy_view`): every array to the oracle's
      own length, and only the tables, constants and rows the oracle has. So
      "0 changed row(s)" says: the old tables did not move.

What is read, without ever importing a whole package (so nothing in TREE is
run except materials.py and bonds/materials.py, and nothing is written into
it: no .pyc, no numba cache):
  src/voxel/materials.py   every table, the derived lookup arrays, constants
  src/mpm/core.py          MAT, read as a literal out of the source text
  src/bonds/materials.py   the bond rows, and the pair rows they give
  src/voxel/bodies.py      CARRIED and AS_BODY, read as literals

An array is stored as dtype + shape + hex bytes (the authority) plus its
values (for people to read). A float that is not finite is stored as
{"$f": "inf" | "-inf" | "nan"}. A tuple comes back from JSON as a list.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import types

import numpy as np

SCHEMA = 1

# names of the ids, for people reading a diff (a missing one prints as its number)
MAT_NAMES = ("AIR", "WOOD", "STONE", "IRON", "FLESH", "ASH", "GLASS", "LEAF",
             "CHAR", "TIN", "LEAD", "ROPE", "ICE")
FLUID_NAMES = ("NOFLUID", "WATER", "OIL", "ACID", "WEAK_ACID", "MTIN", "MLEAD",
               "MIRON", "BLOOD")

# the tables the spec names, by their name in materials.py (kept even when empty)
TABLES = ("SOLID", "EMISS", "SPAN", "TOUGH", "TSHOCK", "MELT", "FREEZE", "PYRO",
          "O2_RATIO", "SOOT", "STOCK", "FLUID", "EMISS_FLUID", "FLUID_K",
          "PYRO_OIL", "BURN_AREA", "GLOW", "TRANSMIT", "SKIN_T", "MATNAME",
          "_FLIDS", "_FDENS", "_FC", "_FK")
# which tables are keyed by a fluid id, the rest by a material id (names in diffs)
FLUID_KEYED = ("FLUID", "EMISS_FLUID", "FLUID_K", "STOCK")
SCALARS = ("NMAT", "BOIL", "LATENT", "SOOT_OIL", "O2_RATIO_OIL", "AMBIENT", "O2_PER_L")
BOND_PAIR_COS2 = (0.0, 0.25, 0.5, 0.75, 1.0)

# files whose bytes are hashed into the oracle (provenance, ignored by diff)
SOURCES = ("src/voxel/materials.py", "src/mpm/core.py", "src/bonds/materials.py",
           "src/voxel/bodies.py")


# ── encoding ────────────────────────────────────────────────────────────────

class Skip(Exception):
    """A value that cannot be frozen (a function, an object)."""


def _key(k):
    """A dict key as JSON text."""
    if isinstance(k, (bool, np.bool_)):
        return str(bool(k))
    if isinstance(k, (int, np.integer)):
        return str(int(k))
    if isinstance(k, str):
        return k
    if isinstance(k, (tuple, frozenset, set)):
        parts = [_key(x) for x in k]
        return "+".join(sorted(parts) if not isinstance(k, tuple) else parts)
    if isinstance(k, (float, np.floating)):
        return repr(float(k))
    raise Skip(type(k).__name__)


def _sortkey(item):
    k = item[0]
    return (0, int(k), "") if isinstance(k, (int, np.integer)) and not isinstance(k, bool) else (1, 0, _key(k))


def enc(v):
    """A value as plain JSON data (see the module text)."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return {"$f": "nan"}
        if math.isinf(f):
            return {"$f": "inf" if f > 0 else "-inf"}
        return f
    if isinstance(v, np.ndarray):
        return [enc(x) for x in v.tolist()]
    if isinstance(v, (tuple, list)):
        return [enc(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((enc(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(v, dict):
        return {_key(k): enc(x) for k, x in sorted(v.items(), key=_sortkey)}
    raise Skip(type(v).__name__)


def dec(v):
    """The reverse of `enc` for numbers: {"$f": ...} back to a float. Lists stay lists."""
    if isinstance(v, dict):
        if set(v) == {"$f"}:
            return float(v["$f"])
        return {k: dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [dec(x) for x in v]
    return v


def enc_array(a):
    """An array as {dtype, shape, hex, values}; the hex bytes are the authority."""
    a = np.ascontiguousarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape), "hex": a.tobytes().hex(),
            "values": enc(a)}


def decode_array(entry):
    """The numpy array an `enc_array` entry stands for, bit for bit."""
    a = np.frombuffer(bytes.fromhex(entry["hex"]), dtype=np.dtype(entry["dtype"]))
    return a.reshape(entry["shape"]).copy()


def load_oracle(path):
    """The oracle JSON as a dict (arrays still as entries; see `decode_array`)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── reading a tree ──────────────────────────────────────────────────────────

def _literal(path, name):
    """The value of the module-level assignment `name = <literal>` in a source
    file, without running it (the last such assignment wins)."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    found = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name and node.value is not None:
            found = node.value
    if found is None:
        raise KeyError("%s not assigned in %s" % (name, path))
    return ast.literal_eval(found)


def _load_modules(tree):
    """Import src.voxel.materials and src.bonds.materials from `tree` through
    empty stand-ins for the packages, so no package __init__ is run."""
    sys.dont_write_bytecode = True
    for n in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[n]
    for name, sub in (("src", "src"), ("src.voxel", "src/voxel"), ("src.bonds", "src/bonds")):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(tree, sub)]
        sys.modules[name] = m
    mats = importlib.import_module("src.voxel.materials")
    bonds = importlib.import_module("src.bonds.materials")
    return mats, bonds


def _is_const(name):
    return name.isupper() or (name.startswith("_") and name[1:2].isupper() and name.endswith("_ARR"))


def _rows(table):
    """A table as {key: row} (a list or tuple becomes {"0": ..})."""
    return enc(table)


def _extract_here(tree, label):
    """The oracle document for `tree`, built in THIS process."""
    tree = os.path.abspath(tree)
    mats, bonds = _load_modules(tree)
    nmat = int(mats.NMAT)
    flids = [int(f) for f in mats._FLIDS]

    doc = {"schema": SCHEMA, "label": label}
    doc["source_sha256"] = {}
    for rel in SOURCES:
        with open(os.path.join(tree, rel), "rb") as fh:
            doc["source_sha256"][rel] = hashlib.sha256(fh.read()).hexdigest()

    doc["ids"] = {
        "mat": {str(getattr(mats, n)): n for n in MAT_NAMES if hasattr(mats, n)},
        "fluid": {str(getattr(mats, n)): n for n in FLUID_NAMES if hasattr(mats, n)},
    }
    # THE NUMBER OF FLUID SLOTS: one per fluid in _FLIDS plus the empty slot 0.
    # Every per-fluid array has exactly this length.
    doc["fluid_slots"] = len(flids) + 1
    doc["nmat"] = nmat
    doc["flids"] = flids

    doc["scalars"] = {n: enc(getattr(mats, n)) for n in SCALARS if hasattr(mats, n)}

    tables = {}
    for n in TABLES:
        if hasattr(mats, n):
            tables[n] = _rows(getattr(mats, n))
    doc["tables"] = tables

    arrays = {}
    for n in sorted(vars(mats)):
        v = getattr(mats, n)
        if isinstance(v, np.ndarray) and _is_const(n):
            arrays[n] = enc_array(v)
        elif isinstance(v, np.ndarray) and n.startswith("_") and n[1:2].isupper():
            arrays[n] = enc_array(v)       # _ALPHA, _FALPHA
    doc["arrays"] = arrays

    # every other public constant of materials.py (so a new fuel row, a new
    # constant, shows up in a diff even if no list above names it)
    skip = set(TABLES) | set(SCALARS) | set(arrays) | set(MAT_NAMES) | set(FLUID_NAMES)
    other = {}
    for n in sorted(vars(mats)):
        if n in skip or not (n.isupper() or n in ("_FLIDS",)):
            continue
        v = getattr(mats, n)
        if isinstance(v, (types.ModuleType, type)) or callable(v) or isinstance(v, np.ndarray):
            continue
        try:
            other[n] = enc(v)
        except Skip:
            pass
    doc["other"] = other

    doc["mpm_mat"] = enc(_literal(os.path.join(tree, "src/mpm/core.py"), "MAT"))

    brow = {n: enc(getattr(bonds, n)) for n in sorted(vars(bonds))
            if n.isupper() and isinstance(getattr(bonds, n), (tuple, float))}
    pairs = {}
    for ma in range(nmat):
        for mb in range(ma, nmat):
            for c2 in BOND_PAIR_COS2:
                r = bonds.pair_row(ma, mb, c2)
                pairs["%d,%d,%s" % (ma, mb, repr(c2))] = None if r is None else enc(tuple(r))
    doc["bond"] = {"ROW": enc(bonds.ROW), "NO_BOND": enc(sorted(bonds.NO_BOND)),
                   "consts": brow, "pairs": pairs}

    bodies = os.path.join(tree, "src/voxel/bodies.py")
    doc["carried"] = enc(_literal(bodies, "CARRIED"))
    doc["as_body"] = enc(_literal(bodies, "AS_BODY"))
    return doc


def extract(tree, label="base18"):
    """The oracle document for a tree folder. It runs in a child process, so two
    trees never meet in one process and `src` is never half loaded."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "_dump", tree, "--label", label],
                         check=True, capture_output=True, env=env)
    return json.loads(out.stdout.decode("utf-8"))


# ── writing ─────────────────────────────────────────────────────────────────

def dumps(doc, depth=0):
    """The JSON text, byte-stable: keys sorted, the first three levels one
    entry per line, deeper levels on one line."""
    def one(v):
        return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def rec(v, d, pad):
        if isinstance(v, dict) and d < 3 and v:
            items = sorted(v.items(), key=lambda kv: kv[0])
            if set(v) == {"$f"}:
                return one(v)
            inner = [pad + " " + json.dumps(k) + ": " + rec(x, d + 1, pad + " ")
                     for k, x in items]
            return "{\n" + ",\n".join(inner) + "\n" + pad + "}"
        return one(v)
    return rec(doc, depth, "") + "\n"


def write(tree, out, label="base18"):
    text = dumps(extract(tree, label))
    if out == "-":
        sys.stdout.write(text)
    else:
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return text


# ── diffing ─────────────────────────────────────────────────────────────────

def _doc(arg):
    """An oracle document from a tree folder or a JSON file."""
    if os.path.isdir(arg):
        return extract(arg, "tree")
    return load_oracle(arg)


def _rowname(doc, section, key):
    """A readable id name for a row, '' when there is none."""
    ids = doc.get("ids", {})
    if section.startswith("tables/") and section[7:] in FLUID_KEYED:
        return ids.get("fluid", {}).get(key, "")
    if section.startswith("tables/") or section.startswith("bond/ROW"):
        return ids.get("mat", {}).get(key, "")
    return ""


def flatten(doc):
    """The doc as {path: text}: one path per row. An array row is its hex chunk
    plus its value, so a changed bit shows even when the printed value is equal."""
    rows = {}

    def put(path, v):
        rows[path] = json.dumps(v, sort_keys=True, separators=(",", ":"))

    put("fluid_slots", doc.get("fluid_slots"))
    put("nmat", doc.get("nmat"))
    put("flids", doc.get("flids"))
    put("carried", doc.get("carried"))
    put("as_body", doc.get("as_body"))
    for k, v in sorted(doc.get("scalars", {}).items()):
        put("scalars/" + k, v)
    for t, rws in sorted(doc.get("tables", {}).items()):
        if isinstance(rws, dict):
            for k, v in rws.items():
                put("tables/%s/%s" % (t, k), v)
        else:
            put("tables/" + t, rws)
    for n, e in sorted(doc.get("arrays", {}).items()):
        put("arrays/%s/#layout" % n, [e["dtype"], e["shape"]])
        itemsize = np.dtype(e["dtype"]).itemsize
        count = int(np.prod(e["shape"])) if e["shape"] else 1
        row = max(1, count // e["shape"][0]) if e["shape"] else 1
        step = 2 * itemsize * row
        vals = e["values"]
        for i in range(e["shape"][0] if e["shape"] else 1):
            put("arrays/%s/%d" % (n, i),
                [e["hex"][i * step:(i + 1) * step], vals[i] if e["shape"] else vals])
    for n, v in sorted(doc.get("other", {}).items()):
        if isinstance(v, dict) and set(v) != {"$f"}:
            for k, x in v.items():
                put("other/%s/%s" % (n, k), x)
        else:
            put("other/" + n, v)
    for k, v in sorted(doc.get("mpm_mat", {}).items()):
        put("mpm_mat/" + k, v)
    b = doc.get("bond", {})
    for k, v in sorted(b.get("ROW", {}).items()):
        put("bond/ROW/" + k, v)
    put("bond/NO_BOND", b.get("NO_BOND"))
    for k, v in sorted(b.get("consts", {}).items()):
        put("bond/consts/" + k, v)
    for k, v in sorted(b.get("pairs", {}).items()):
        put("bond/pairs/" + k, v)
    return rows


def legacy_view(live, oracle):
    """`live` (the document of a tree that has grown since the oracle was frozen) cut
    down to what `oracle` lists, so that diffing the two says only whether the OLD
    tables moved. Wave 13 pads every lookup array to 64 entries (the first twelve,
    nine for the liquids, are the old ones) and adds constants and rows; none of that
    is a change of an old table.
      * an array: only those the oracle has, each cut to the oracle's own length (a
        live array that is SHORTER stays as it is, and shows as a layout change);
      * the tables, constants, scalars, particle rows and bond rows: only the keys
        the oracle has (a row the oracle has and the tree lost still shows).
    Anything the oracle lists that the tree changed or lost is left in, so the diff
    still prints it."""
    out = copy.deepcopy(live)
    arrays = {}
    for name, want in oracle.get("arrays", {}).items():
        got = live.get("arrays", {}).get(name)
        if got is None:
            continue
        a = decode_array(got)
        if want["shape"] and a.ndim and a.shape[0] > want["shape"][0]:
            a = a[:want["shape"][0]]
        arrays[name] = enc_array(a)
    out["arrays"] = arrays

    def keep(rows, want):
        if isinstance(rows, dict) and isinstance(want, dict):
            return {k: v for k, v in rows.items() if k in want}
        return rows
    for sec in ("tables", "other", "scalars", "mpm_mat", "ids"):
        out[sec] = keep(live.get(sec, {}), oracle.get(sec, {}))
    for t in list(out["tables"]):
        out["tables"][t] = keep(out["tables"][t], oracle.get("tables", {}).get(t))
    b, ob = live.get("bond", {}), oracle.get("bond", {})
    out["bond"] = {k: (keep(b[k], ob.get(k)) if k in b else b.get(k)) for k in b}
    for k in ("ROW", "consts", "pairs"):
        if k in out["bond"]:
            out["bond"][k] = keep(out["bond"][k], ob.get(k))
    for sec in ("ids",):
        for k in list(out.get(sec, {})):
            out[sec][k] = keep(out[sec][k], oracle.get(sec, {}).get(k))
    return out


def diff_docs(a, b):
    """The list of changed rows between two documents, as printable lines."""
    fa, fb = flatten(a), flatten(b)
    out = []
    for path in sorted(set(fa) | set(fb)):
        sec, _, key = path.rpartition("/")
        name_a = _rowname(a, sec, key)
        name_b = _rowname(b, sec, key)
        name = name_b or name_a
        tag = " (%s)" % name if name else ""
        if path not in fa:
            out.append("ADDED    %s%s: %s" % (path, tag, fb[path]))
        elif path not in fb:
            out.append("REMOVED  %s%s: %s" % (path, tag, fa[path]))
        elif fa[path] != fb[path]:
            out.append("CHANGED  %s%s: %s -> %s" % (path, tag, fa[path], fb[path]))
    return out


# ── command line ────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="freeze a tree's material tables to JSON")
    w.add_argument("tree")
    w.add_argument("-o", "--out", required=True, help="output file, or - for stdout")
    w.add_argument("--label", default="base18")
    d = sub.add_parser("diff", help="print the rows that differ between two trees or oracles")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--strict", action="store_true", help="exit 1 if anything differs")
    d.add_argument("--legacy", action="store_true",
                   help="A is the frozen oracle: cut B (a tree that has grown) down to what A lists")
    h = sub.add_parser("_dump")        # internal: one tree, in this process, to stdout
    h.add_argument("tree")
    h.add_argument("--label", default="base18")
    args = ap.parse_args(argv)
    if args.cmd == "_dump":
        sys.stdout.write(dumps(_extract_here(args.tree, args.label)))
        return 0
    if args.cmd == "write":
        write(args.tree, args.out, args.label)
        return 0
    doc_a, doc_b = _doc(args.a), _doc(args.b)
    if args.legacy:
        doc_b = legacy_view(doc_b, doc_a)
    lines = diff_docs(doc_a, doc_b)
    for ln in lines:
        print(ln)
    print("%d changed row(s)" % len(lines))
    return 1 if (lines and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
