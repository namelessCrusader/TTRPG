"""Static guards for the sim.py split, and for every change after it.

  1. UNDEFINED GLOBALS. Every global a function in src/voxel/*.py reads must be
     bound at that module's top level (or be a builtin). pyflakes would do this;
     it is not installed, and symtable is in the standard library.
  2. NO HIDDEN MODULE STATE. A `globals()` call or a `global` statement in
     src/voxel reaches only ITS OWN module, so it silently stops working the
     day the reader and the writer land in different files (sim.py:7380-7384,
     `_full_menu` rebinding MENU_CAP for `_menu`). Allowed only where listed.
  3. TESTS PATCH THE MODULE THAT READS. A test that rebinds `M.NAME = ...` on
     a src.voxel module must name a module whose functions READ NAME, or the
     rebinding reaches nothing (`S._SWEEP_RAD` in test_an_ARM_CANNOT_SWEEP_
     THROUGH_..., once `_repose` lives in pose.py). In-place mutation
     (`M.BODY[k] = v`) is fine: every module binds the same dict.

    python3 namecheck.py [--src DIR] [--tests DIR]     # exit 1 on any finding
"""
import ast
import builtins
import os
import sys
import symtable

REPO = os.environ.get("MARK1_REPO", os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))      # tools/split/ -> repo
VOX = os.path.join(REPO, "src", "voxel")
TESTS = os.path.join(REPO, "tests")

# (file, function) -> why. Rule 2's allow-list; empties when _menu takes `cap`.
GLOBALS_OK = {}
# Rule 1's known pre-existing hit, fixed by S4 (C5) after the carve.
UNDEFINED_OK = set()


def module_names(top):
    return {s.get_name() for s in top.get_symbols()
            if s.is_assigned() or s.is_imported() or s.is_namespace()}


def walk(t):
    yield t
    for c in t.get_children():
        yield from walk(c)


def undefined(path):
    src = open(path).read()
    top = symtable.symtable(src, path, "exec")
    bound = module_names(top) | set(dir(builtins)) | {"__file__", "__name__"}
    out = []
    for t in walk(top):
        if t.get_type() == "module":
            continue
        for s in t.get_symbols():
            if s.is_referenced() and (s.is_global() or s.is_declared_global()) \
                    and s.get_name() not in bound:
                out.append((t.get_name(), t.get_lineno(), s.get_name()))
    return out


def globals_reads(path):
    """{global name read inside any function of this module}"""
    top = symtable.symtable(open(path).read(), path, "exec")
    got = set()
    for t in walk(top):
        if t.get_type() != "module":
            got |= {s.get_name() for s in t.get_symbols()
                    if s.is_referenced() and (s.is_global() or s.is_declared_global())}
    return got


def hidden_state(path):
    """[(function, line, what)] for globals() calls and global statements."""
    tree = ast.parse(open(path).read())
    out = []

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(ch, ast.Global):
                out.append((fn, ch.lineno, "global " + ", ".join(ch.names)))
            if isinstance(ch, ast.Call) and isinstance(ch.func, ast.Name) \
                    and ch.func.id in ("globals", "vars") and not ch.args:
                out.append((fn, ch.lineno, ch.func.id + "()"))
            visit(ch, f)
    visit(tree, "<module>")
    return out


def test_rebinds(tests, vox):
    """[(test file, line, module, name)] for `alias.NAME = ...` on src.voxel
    modules where that module's functions never read NAME."""
    reads = {f[:-3]: globals_reads(os.path.join(vox, f))
             for f in os.listdir(vox) if f.endswith(".py")}
    bad = []
    for tf in sorted(os.listdir(tests)):
        if not (tf.startswith("test_") and tf.endswith(".py")):
            continue
        tree = ast.parse(open(os.path.join(tests, tf)).read())

        def aliases(scope):
            """Imports made directly in this scope (a function's own body, not
            the functions nested in it): `S` is a different module in two
            different tests (the sweep test and the wind-up test)."""
            got = {}
            for n in ast.walk(scope):
                if isinstance(n, ast.Import):
                    for a in n.names:
                        if a.name.startswith("src.voxel.") and a.asname:
                            got[a.asname] = a.name.split(".")[-1]
                elif isinstance(n, ast.ImportFrom) and n.module == "src.voxel":
                    for a in n.names:
                        got[a.asname or a.name] = a.name
            return got
        top = aliases(ast.Module(body=[n for n in tree.body if isinstance(
            n, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
            alias = {**top, **aliases(fn)}
            for n in ast.walk(fn):
                if isinstance(n, (ast.Assign, ast.AugAssign)):
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                                and t.value.id in alias:
                            mod = alias[t.value.id]
                            if t.attr not in reads.get(mod, set()):
                                bad.append((tf, n.lineno, mod, t.attr))
    return bad


def main(argv):
    vox = argv[argv.index("--src") + 1] if "--src" in argv else VOX
    tests = argv[argv.index("--tests") + 1] if "--tests" in argv else TESTS
    n = 0
    for f in sorted(os.listdir(vox)):
        if not f.endswith(".py"):
            continue
        p = os.path.join(vox, f)
        for scope, line, name in undefined(p):
            tag = "known" if (f, scope, name) in UNDEFINED_OK else "NEW"
            print(f"[1 {tag}] {f}:{line} {scope}: undefined global {name}")
            n += tag == "NEW"
        for fn, line, what in hidden_state(p):
            tag = "allowed" if (f, fn) in GLOBALS_OK else "NEW"
            print(f"[2 {tag}] {f}:{line} {fn}: {what}")
            n += tag == "NEW"
    for tf, line, mod, name in test_rebinds(tests, vox):
        print(f"[3 NEW] {tf}:{line} rebinds {mod}.{name}, which no function in "
              f"{mod}.py reads -- patch the module that reads it")
        n += 1
    print("namecheck:", "clean" if not n else f"{n} finding(s)")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
