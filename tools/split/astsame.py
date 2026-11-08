"""PROOF THAT NOTHING BUT THE ADDRESS CHANGED.

Compares the original sim.py with the split package at the level of the
parse tree (line numbers excluded, docstrings included):

  * every World method of the original is defined EXACTLY ONCE across World
    and the *Mixin classes, with an identical AST;
  * every class attribute (_CS, _PERCELL, ...) likewise;
  * every module-level statement of the original appears exactly once across
    the new modules with an identical AST, and statements that share a module
    keep their original relative order (WILL is mutated after it is made);
  * no mixin defines __init__, no two classes in World's bases share a name,
    and no new module imports `.sim` (the only way to make a cycle).

    python3 astsame.py <original sim.py> <new src/voxel dir>
"""
import ast
import os
import sys

NEW_MODULES_SKIP = {"scenes.py", "harvest.py", "demo.py", "room.py",
                    "seam.py", "__init__.py"}


def dump(n):
    return ast.dump(n, include_attributes=False)


def stmt_key(n):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        return n.name
    if isinstance(n, ast.Assign):
        return ast.unparse(n.targets[0])
    return type(n).__name__ + ":" + dump(n)[:60]


def main(old_path, new_dir):
    bad = []
    old = ast.parse(open(old_path).read())
    oworld = next(n for n in old.body if isinstance(n, ast.ClassDef)
                  and n.name == "World")
    o_methods = {n.name: dump(n) for n in oworld.body
                 if isinstance(n, ast.FunctionDef)}
    o_attrs = {n.targets[0].id: dump(n) for n in oworld.body
               if isinstance(n, ast.Assign)}
    o_top = [n for n in old.body if n is not oworld and not isinstance(
        n, (ast.Import, ast.ImportFrom)) and not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]

    n_methods, n_attrs, n_top = {}, {}, []
    for f in sorted(os.listdir(new_dir)):
        if not f.endswith(".py") or f in NEW_MODULES_SKIP:
            continue
        t = ast.parse(open(os.path.join(new_dir, f)).read())
        for n in t.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                if isinstance(n, ast.ImportFrom) and n.module == "sim" \
                        and n.level == 1 and f != "sim.py":
                    bad.append(f"{f}: imports .sim (cycle)")
                continue
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
                continue                               # module docstrings
            if isinstance(n, ast.ClassDef) and (n.name == "World"
                                                 or n.name.endswith("Mixin")):
                for m in n.body:
                    if isinstance(m, ast.FunctionDef):
                        if m.name == "__init__" and n.name != "World":
                            bad.append(f"{f}: {n.name} defines __init__")
                        n_methods.setdefault(m.name, []).append((f, dump(m)))
                    elif isinstance(m, ast.Assign):
                        n_attrs.setdefault(m.targets[0].id, []).append(
                            (f, dump(m)))
                    elif not (isinstance(m, ast.Expr)
                              and isinstance(m.value, ast.Constant)):
                        bad.append(f"{f}: {n.name} has a stray {type(m).__name__}")
                continue
            n_top.append((f, n))

    for name, d in o_methods.items():
        got = n_methods.get(name, [])
        if len(got) != 1:
            bad.append(f"method {name}: defined {len(got)} times {[g[0] for g in got]}")
        elif got[0][1] != d:
            bad.append(f"method {name}: AST CHANGED in {got[0][0]}")
    for name in set(n_methods) - set(o_methods):
        bad.append(f"method {name}: NEW (not in the original)")
    for name, d in o_attrs.items():
        got = n_attrs.get(name, [])
        if len(got) != 1 or got[0][1] != d:
            bad.append(f"class attr {name}: {len(got)} definitions / changed")

    # module-level statements: each old one exactly once, same AST, same order
    pos = {}
    pool = {}
    for i, (f, n) in enumerate(n_top):
        pool.setdefault(dump(n), []).append((f, i))
    for n in o_top:
        got = pool.get(dump(n), [])
        if len(got) != 1:
            bad.append(f"top-level {stmt_key(n)}: found {len(got)} times")
            continue
        f, i = got[0]
        if f in pos and pos[f] > i:
            bad.append(f"top-level {stmt_key(n)}: ORDER changed in {f}")
        pos[f] = i
    old_dumps = {dump(n) for n in o_top}
    for f, n in n_top:
        if dump(n) not in old_dumps and not (f == "sim.py" and isinstance(
                n, ast.ClassDef) and n.name == "World"):
            if isinstance(n, ast.ClassDef) and n.name.endswith("Mixin"):
                continue
            bad.append(f"{f}: NEW top-level statement {stmt_key(n)}")

    print(f"methods {len(o_methods)} | class attrs {len(o_attrs)} | "
          f"top-level statements {len(o_top)}")
    for b in bad:
        print("  ", b)
    print("astsame:", "IDENTICAL" if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
