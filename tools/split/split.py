"""MECHANICAL MOVER for the sim.py split (prototype; writes only to OUT).

Every byte of every method and table is CUT from the original sim.py by its
AST line range -- comments above a node travel with it -- and pasted into its
target module, so the bodies are identical by construction (astsame.py then
proves it). Imports for each new module are computed from what its methods
actually read (deps.json), never written by hand.

    python3 split.py <original sim.py> <out dir>   # out dir gets src/voxel/*.py

`--only fields,ledgers` limits which MIXINS are carved out (the batch order);
tables and kernels are always carved, since every later batch needs them.
"""
import ast
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SM = json.load(open(os.path.join(HERE, "splitmap.json")))
DEPS = json.load(open(os.path.join(HERE, "deps.json")))
WHERE = SM["where"]                    # method -> module
G = SM["G"]                            # module-level name -> module

MIXIN = {"fields": "FieldsMixin", "fluids": "FluidsMixin", "heat": "HeatMixin",
         "air": "AirMixin", "support": "SupportMixin", "bodies": "BodiesMixin",
         "skeleton": "SkeletonMixin", "pose": "PoseMixin",
         "persons": "PersonsMixin", "physiology": "PhysiologyMixin",
         "senses": "SensesMixin", "motion": "MotionMixin",
         "hands": "HandsMixin", "will": "WillMixin", "ledgers": "LedgersMixin"}
CLSATTR = {"_PERCELL": "bodies", "_CS": "air", "_SMOKE_REACH": "air",
           "_NOT_STATE": "ledgers"}
DOC = {
    "kernels": "Optional numba kernels: fused per-cell loops that must agree with the numpy laws.",
    "materials": "What matter IS: material and fluid ids, property tables, physical constants, body data.",
    "policy": "The decision contract: limbs, acts, reflexes, lines, attention cap, and the table policy.",
    "fields": "Derived fields (heat capacity, temperature, porosity), the authoring API, activity windows.",
    "fluids": "Fluid laws: fall, spread, head pressure, drops, stirring, splashes, and acid.",
    "heat": "Heat laws: conduction, buoyant rise, combustion, boiling, melting and freezing.",
    "air": "Air laws: coarse pressure/wind field, smoke transport, airspace regions, oxygen.",
    "support": "Lattice mechanics: support, falling columns, impacts, shattering, cracking, toppling.",
    "bodies": "Free rigid bodies in flight: launch, ballistics, drag, landing, and what a landing costs.",
    "skeleton": "Body geometry: bones, chains, joints, transforms, spans, fists, reach (reads, never moves).",
    "pose": "Limb motion on the lattice: swing, turn, repose, pose and joint laws, crouch, lean.",
    "persons": "Which flesh is whose: adding people, body ownership by flood fill, status.",
    "physiology": "Blows and wounds, bleeding, perfusion, breath, burns, faint and death.",
    "senses": "Sight, hearing, sound levels, speech, and noticing things in flight.",
    "motion": "Where legs can take a body: places, routes, walking, leaping, footing, balance.",
    "hands": "Grip, hold, carry, haul, shove, catch: what hands can reach and move.",
    "will": "Menus, decisions, traces: the seam where a policy picks what each limb does.",
    "ledgers": "The world as a value (snapshot/restore) and the conservation totals.",
}
EXT = {"np": "import numpy as np", "math": "import math", "copy": "import copy",
       "seam": "from . import seam", "Matter": "from ..mpm.matter import Matter"}

# Comment blocks that the "comments travel with the NEXT node" rule would put
# in the wrong home (they sit above an unrelated statement in the original):
# (first line, last line) -> the name they belong to. Line numbers are for
# the WORKING-TREE sim.py of 2026-10-01 (8,794 lines, uncommitted changes on
# top of 6120b6e); they are asserted to be comment lines below, and deps.py /
# splitmap.py must be re-run if sim.py changes before the split.
REHOME = {(267, 269): "BODY",           # "PHYSIOLOGY — what being alive means"
          (826, 842): "SMOKE_STILL"}     # "AIR THAT HOLDS A MILLIONTH OF A GRAM"
# ...and trailing comment lines that belong to the statement ABOVE them.
REHOME_AFTER = {(487, 489): "TablePolicy",       # the `return 0` explanation
                (741, 759): "FALL_SUBSTEPS",     # "It was FOUR, and the note..."
                (844, 846): "_SWEEP_RAD"}        # "how finely a limb's turn..."


def top_nodes(tree):
    """[(names, first_line, last_line)] for module-level statements, where a
    node's range starts right after the previous node ends."""
    out, prev_end = [], 0
    for n in tree.body:
        if isinstance(n, ast.Assign):
            names = []
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Tuple):
                    names += [e.id for e in t.elts]
                else:
                    names.append(ast.unparse(t))      # WILL["decide_every"]
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, ast.Try):
            names = ["HAVE_NUMBA", "njit"]
        else:
            names = [type(n).__name__]
        start = prev_end + 1
        if getattr(n, "decorator_list", None):
            start = min(start, n.decorator_list[0].lineno)
        out.append((names, start, n.end_lineno, n))
        prev_end = n.end_lineno
    return out


# The ONE input this map, REHOME and deps.json were derived from. Any other
# sim.py -- an edit landed after the commit the carve starts from -- is refused:
# astsame ignores comments, so a shifted REHOME range could misfile a comment
# block silently. Re-run deps.py and splitmap.py, re-derive REHOME, re-pin.
EXPECT_SHA256 = "4d0b87e91545fd2d32df2ad2f05985549b39d5f31c921029c64e3888ae1cf496"


def main(src_path, out, only=None):
    import hashlib
    got = hashlib.sha256(open(src_path, "rb").read()).hexdigest()
    if got != EXPECT_SHA256 and "--any-input" not in sys.argv:
        sys.exit(f"split.py: {src_path} is sha256 {got[:12]}..., not the "
                 f"{EXPECT_SHA256[:12]}... the map was made from. Re-derive.")
    lines = open(src_path).read().split("\n")
    tree = ast.parse("\n".join(lines))
    text = lambda a, b: "\n".join(lines[a - 1:b])           # 1-based inclusive

    nodes = top_nodes(tree)
    world = [n for _, _, _, n in nodes if isinstance(n, ast.ClassDef)
             and n.name == "World"][0]
    header_end = max(n.end_lineno for _, _, _, n in nodes
                     if isinstance(n, (ast.Import, ast.ImportFrom)))

    # ---- module-level statements -> table modules
    for (a, b), who in list(REHOME.items()) + list(REHOME_AFTER.items()):
        assert lines[a - 1].lstrip().startswith("#") and \
            lines[b - 1].lstrip().startswith("#"), (a, b)
    rehome_lines = {i for R in (REHOME, REHOME_AFTER)
                    for (a, b) in R for i in range(a, b + 1)}
    chunks = {"kernels": [], "materials": [], "policy": [], "pose": []}
    order_names = []                      # every old module-level name, in order
    for names, a, b, n in nodes:
        if b <= header_end or n is world:
            continue
        key = names[0]
        if key.startswith("WILL["):
            mod = "policy"
        elif key == "TablePolicy":
            mod = "policy"
        else:
            mod = G.get(key)
            assert mod, f"no home for module-level {names}"
        body = [i for i in range(a, b + 1) if i not in rehome_lines]
        pre, post = [], []
        for (ra, rb), who in REHOME.items():
            if who in names:
                pre = list(range(ra, rb + 1))
        for (ra, rb), who in REHOME_AFTER.items():
            if who in names:
                post = list(range(ra, rb + 1))
        chunks[mod].append("\n".join(lines[i - 1] for i in pre + body + post))
        order_names += [x for x in names if not x.startswith("WILL[")]

    # ---- World body -> mixins
    by_mod = {m: [] for m in MIXIN}
    keep = []                             # stays in World (sim.py)
    prev_end = world.body[0].lineno - 1
    # the class line itself: everything up to the first body node stays
    for n in world.body:
        a, b = prev_end + 1, n.end_lineno
        if getattr(n, "decorator_list", None):
            a = min(a, n.decorator_list[0].lineno)
        prev_end = b
        if isinstance(n, ast.FunctionDef):
            mod = WHERE[n.name]
        else:
            mod = CLSATTR[n.targets[0].id]
        if mod == "sim" or (only is not None and mod not in only):
            keep.append(text(a, b))
        else:
            by_mod[mod].append(text(a, b))

    vox = os.path.join(out, "src", "voxel")
    os.makedirs(vox, exist_ok=True)

    def imports_for(names_used, self_mod):
        by_src = {}
        ext = set()
        for g in sorted(names_used):
            if g in EXT:
                ext.add(EXT[g])
                continue
            src_mod = G[g]
            if src_mod == self_mod:
                continue
            by_src.setdefault(src_mod, set()).add(g)
        head = sorted(ext, key=lambda s: (not s.startswith("import"), s))
        for m in ("kernels", "materials", "policy", "pose"):
            if m in by_src:
                head.append(f"from .{m} import ({', '.join(sorted(by_src[m]))})")
        return head

    # ---- table modules
    table_uses = {"policy": {"np", "TICK_S"}, "materials": {"np"},
                  "kernels": set(), "pose": set()}
    for mod in ("kernels", "materials", "policy"):
        head = [f'"""{DOC[mod]}"""']
        if mod == "kernels":
            head += ["import numpy as np"]
        elif mod == "materials":
            head += ["import numpy as np"]
        else:
            head += ["import numpy as np", "", "from .materials import TICK_S"]
        open(os.path.join(vox, f"{mod}.py"), "w").write(
            "\n".join(head) + "\n\n" + "\n".join(chunks[mod]) + "\n")

    # ---- mixin modules
    bases = []
    for mod, cls in MIXIN.items():
        if not by_mod[mod]:
            continue
        used = set()
        for m, w in WHERE.items():
            if w == mod:
                used |= set(DEPS[m]["globals"])
        extra = chunks["pose"] if mod == "pose" else []
        head = [f'"""{DOC[mod]}"""'] + imports_for(used, mod)
        body = "\n".join(extra) + ("\n\n\n" if extra else "\n\n") + \
            f"class {cls}:\n" + "\n".join(by_mod[mod]) + "\n"
        open(os.path.join(vox, f"{mod}.py"), "w").write(
            "\n".join(head) + "\n" + body)
        bases.append((mod, cls))

    # ---- sim.py: docstring, re-exports, World
    doc_end = tree.body[0].end_lineno
    out_lines = [text(1, doc_end)]
    out_lines += ["import math", "import copy", "", "from . import seam",
                  "from ..mpm.matter import Matter", "import numpy as np", ""]
    homes = {}
    for nm in order_names:
        homes.setdefault(G.get(nm, "policy" if nm == "TablePolicy" else None),
                         []).append(nm)
    out_lines.append("# EVERY NAME sim.py ever defined is still importable from it")
    out_lines.append("# (tests, scenes, harvest, demo and room import from here).")
    for m in ("kernels", "materials", "policy", "pose"):
        if homes.get(m):
            if m == "pose" and "pose" not in dict(bases):
                continue
            out_lines.append(f"from .{m} import (  # noqa: F401")
            out_lines += [f"    {nm}," for nm in homes[m]]
            out_lines.append(")")
    for mod, cls in bases:
        out_lines.append(f"from .{mod} import {cls}")
    if "pose" not in dict(bases):         # _SWEEP_RAD stays here until pose moves
        out_lines.append("")
        out_lines.append(chunks["pose"][0])
    out_lines.append("")
    out_lines.append("")
    base_list = ", ".join(cls for _, cls in bases)
    out_lines.append(f"class World({base_list}):" if base_list else "class World:")
    out_lines.append("\n".join(keep))
    out_lines.append("")
    open(os.path.join(vox, "sim.py"), "w").write("\n".join(out_lines))

    # ---- the rest of the package, unchanged
    src_dir = os.path.dirname(src_path)
    for f in os.listdir(src_dir):
        if f.endswith(".py") and f != "sim.py":
            shutil.copy(os.path.join(src_dir, f), os.path.join(vox, f))
    root = os.path.dirname(os.path.dirname(src_dir))
    shutil.copy(os.path.join(root, "src", "__init__.py"),
                os.path.join(out, "src", "__init__.py"))
    shutil.copytree(os.path.join(root, "src", "mpm"), os.path.join(out, "src", "mpm"),
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    print("wrote", sorted(os.listdir(vox)))


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    main(sys.argv[1], sys.argv[2], only)
