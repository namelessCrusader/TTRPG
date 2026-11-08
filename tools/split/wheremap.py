"""OLD sim.py LINES -> NEW HOMES, for every citation written before the split.

docs/, the memory notes and other workstreams' plans cite `sim.py:N`. After
the carve those numbers point into a 375-line file. This prints a Markdown
table (old line range | name | new file:line) from the pristine sim.py and the
carved src/voxel, plus a `where` lookup for one old line number.

    python3 wheremap.py <pristine sim.py> <new src/voxel> > sim-split-map.md
    python3 wheremap.py <pristine sim.py> <new src/voxel> --line 5004
"""
import ast
import os
import sys


def defs(path, rel):
    """{(kind, name): (rel file, first line, last line)} for module-level
    statements and for methods/attributes of World and the *Mixin classes."""
    out = {}
    tree = ast.parse(open(path).read())
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and (n.name == "World" or n.name.endswith("Mixin")):
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    out[("method", m.name)] = (rel, m.lineno, m.end_lineno)
                elif isinstance(m, ast.Assign):
                    out[("attr", m.targets[0].id)] = (rel, m.lineno, m.end_lineno)
            continue
        names = []
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, ast.Tuple):
                    names += [e.id for e in t.elts if isinstance(e, ast.Name)]
                else:
                    names.append(ast.unparse(t))
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, ast.Try):
            names = ["HAVE_NUMBA"]
        for nm in names:
            out.setdefault(("global", nm), (rel, n.lineno, n.end_lineno))
    return out


def main(old_path, new_dir, line=None):
    old = defs(old_path, "sim.py")
    new = {}
    for f in sorted(os.listdir(new_dir)):
        if f.endswith(".py"):
            for k, v in defs(os.path.join(new_dir, f), f).items():
                if k[0] == "global" and f == "sim.py" and k in new:
                    continue                    # re-exports: keep the real home
                new.setdefault(k, v) if f == "sim.py" else new.__setitem__(k, v)
    rows = sorted((v[1], v[2], k) for k, v in old.items())
    if line is not None:
        for a, b, k in rows:
            if a <= line <= b:
                f, na, _ = new[k]
                print(f"sim.py:{line} was in {k[1]} ({k[0]}, sim.py:{a}-{b}) -> "
                      f"src/voxel/{f}:{na + (line - a)}")
                return 0
        print(f"sim.py:{line} is between definitions (a comment or blank line)")
        return 1
    print("| old sim.py lines | name | now at |")
    print("|---|---|---|")
    for a, b, k in rows:
        f, na, nb = new.get(k, ("MISSING", 0, 0))
        print(f"| {a}-{b} | `{k[1]}` | src/voxel/{f}:{na} |")
    return 0


if __name__ == "__main__":
    ln = int(sys.argv[sys.argv.index("--line") + 1]) if "--line" in sys.argv else None
    sys.exit(main(sys.argv[1], sys.argv[2], ln))
