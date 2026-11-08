"""MODULE HYGIENE for src/voxel: the three ways a move between files breaks
code without changing a line of it. Static, standard library only, < 1 s.

The voxel engine was one 8,800-line file; the split makes it about twenty
(docs/sim-mind-design.md, 3.1), and these rules hold before it. Inside one
file a name read in a method is always found and a module global rebound
anywhere is seen everywhere. Across files neither is true, and nothing at
import time says so -- the method fails the first time its branch runs.

  1. Every global a function reads is bound in ITS module (an import or a
     definition). `_crouch` read a `joints` nobody defined for as long as its
     branch never ran (sim.py:6564 before the split).
  2. No `globals()` and no `global`: they reach only their own module.
     `_full_menu` lifted the attention cap for `_menu` by rebinding MENU_CAP,
     which works only while both live in one file (sim.py:7377-7384).
  3. A test that rebinds `module.NAME` must name the module that READS NAME,
     or it patches nothing (test_an_ARM_CANNOT_SWEEP_THROUGH_..., _SWEEP_RAD).
     Every form counts: `S.NAME = v`, `S.NAME += v`, `setattr(S, "NAME", v)`,
     `monkeypatch.setattr(S, "NAME", v)` and
     `monkeypatch.setattr("src.voxel.sim.NAME", v)`, in a test function, a
     test class's method or the module body. Mutating a shared table in place
     (`sim.BODY[k] = v`) is fine.
"""
import ast
import builtins
import os
import symtable

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(os.path.dirname(HERE), "src", "voxel")

# Known and owned. Each names the file before the carve AND after it, so the
# carve needs no edit here -- and a move to any THIRD file still fails, which
# is the whole point of rule 2. Each entry names the change that removes it
# (step S4 of docs/sim-mind-design.md). Both emptied by S4: C5 made `_crouch`
# read `joints = p.get("joints")`, and C3 made `_menu` take `cap`.
UNDEFINED_OK = set()
GLOBALS_OK = set()


def _sources(d):
    return {f: open(os.path.join(d, f)).read()
            for f in sorted(os.listdir(d)) if f.endswith(".py")}


def _scopes(t):
    yield t
    for c in t.get_children():
        yield from _scopes(c)


def _globals_read(src, name):
    """[(scope, global name)] read inside functions of one module."""
    top = symtable.symtable(src, name, "exec")
    return [(t.get_name(), s.get_name()) for t in _scopes(top)
            if t.get_type() != "module" for s in t.get_symbols()
            if s.is_referenced() and (s.is_global() or s.is_declared_global())], top


def test_every_global_a_function_reads_is_BOUND_in_its_own_module():
    bad = []
    for f, src in _sources(VOX).items():
        reads, top = _globals_read(src, f)
        bound = {s.get_name() for s in top.get_symbols()
                 if s.is_assigned() or s.is_imported() or s.is_namespace()}
        bound |= set(dir(builtins)) | {"__file__", "__name__"}
        bad += [(f, sc, nm) for sc, nm in reads
                if nm not in bound and (f, sc, nm) not in UNDEFINED_OK]
    assert not bad, f"read but never bound in that module: {bad}"


def test_no_module_reaches_into_its_own_GLOBALS():
    bad = []
    for f, src in _sources(VOX).items():
        def visit(node, fn):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.Global) or (
                        isinstance(ch, ast.Call) and isinstance(ch.func, ast.Name)
                        and ch.func.id in ("globals", "vars") and not ch.args):
                    if (f, fn) not in GLOBALS_OK:
                        bad.append((f, fn, ch.lineno))
                visit(ch, ch.name if isinstance(ch, ast.FunctionDef) else fn)
        visit(ast.parse(src), "<module>")
    assert not bad, f"globals()/global statements: {bad} -- pass a parameter"


def _patches(scope, alias):
    """(line, module, NAME) for each rebinding of a voxel module's global in
    `scope`. `alias` maps a local name to the voxel module it is bound to."""
    for n in ast.walk(scope):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                        and t.value.id in alias:
                    yield n.lineno, alias[t.value.id], t.attr
        elif isinstance(n, ast.Call) and n.args and (
                isinstance(n.func, ast.Name) and n.func.id == "setattr"
                or isinstance(n.func, ast.Attribute) and n.func.attr == "setattr"):
            a = n.args
            if isinstance(a[0], ast.Name) and a[0].id in alias and len(a) > 1 \
                    and isinstance(a[1], ast.Constant) and isinstance(a[1].value, str):
                yield n.lineno, alias[a[0].id], a[1].value
            elif isinstance(a[0], ast.Constant) and isinstance(a[0].value, str) \
                    and a[0].value.startswith("src.voxel."):     # the dotted form
                mod, _, name = a[0].value[len("src.voxel."):].rpartition(".")
                if mod and "." not in mod:                   # not "sim.World.x"
                    yield n.lineno, mod, name


def test_a_test_PATCHES_the_module_that_READS_what_it_patches():
    reads = {f[:-3]: {nm for _, nm in _globals_read(src, f)[0]}
             for f, src in _sources(VOX).items()}
    bad = []
    for tf, src in _sources(HERE).items():
        if not tf.startswith("test_"):
            continue
        tree = ast.parse(src)

        def aliases(nodes):
            got = {}
            for n in nodes:
                for m in ast.walk(n):
                    if isinstance(m, ast.Import):
                        got.update({a.asname: a.name.split(".")[-1] for a in m.names
                                    if a.asname and a.name.startswith("src.voxel.")})
                    elif isinstance(m, ast.ImportFrom) and m.module == "src.voxel":
                        got.update({a.asname or a.name: a.name for a in m.names})
            return got
        top = aliases([n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))])
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        fns += [m for c in tree.body if isinstance(c, ast.ClassDef)
                for m in c.body if isinstance(m, ast.FunctionDef)]
        body = ast.Module(body=[n for n in tree.body if not isinstance(
            n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
        for scope in fns + [body]:
            alias = {**top, **aliases([scope])}
            bad += [(tf, line, f"{mod}.{name}") for line, mod, name
                    in _patches(scope, alias) if name not in reads.get(mod, set())]
    assert not bad, f"rebinding a global on a module that never reads it: {bad}"
