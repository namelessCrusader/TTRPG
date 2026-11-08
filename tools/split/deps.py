"""Per-method dependency analysis of a sim.py World: globals read, methods
called, class attributes used. ('reads'/'writes' cover self.<attr> only; writes
through person dicts or local aliases are NOT seen, so they are a hint.)

    python3 deps.py <sim.py to analyse> <out deps.json>

Point it at the PRISTINE copy: run on a half-carved repo sim.py it would see
only the methods still left in World and write a wrong map."""
import ast, json, sys, builtins
SRC, OUT = sys.argv[1], sys.argv[2]
src = open(SRC).read()
t = ast.parse(src)
modnames = set()
for n in t.body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        modnames.add(n.name)
    elif isinstance(n, ast.Assign):
        for x in n.targets:
            if isinstance(x, ast.Name): modnames.add(x.id)
            elif isinstance(x, ast.Tuple): modnames |= {e.id for e in x.elts if isinstance(e, ast.Name)}
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names: modnames.add((a.asname or a.name).split('.')[0])
    elif isinstance(n, ast.Try):
        modnames |= {"njit", "HAVE_NUMBA"}
W = [n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "World"][0]
methods = {n.name: n for n in W.body if isinstance(n, ast.FunctionDef)}
clsattrs = set()
for n in W.body:
    if isinstance(n, ast.Assign):
        for x in n.targets: clsattrs.add(x.id)
out = {}
for name, fn in methods.items():
    g = set(); calls = set(); reads = set(); writes = set(); clsref = set()
    local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in modnames: g.add(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("self", "World", "cls"):
            if node.value.id == "World": clsref.add(node.attr)
            if node.attr in methods:
                calls.add(node.attr)
            elif node.attr in clsattrs:
                clsref.add(node.attr)
            else:
                (writes if isinstance(node.ctx, ast.Store) else reads).add(node.attr)
    # subscript writes self.x[...] = count as writes
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            for x in tg:
                while isinstance(x, ast.Subscript): x = x.value
                if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == "self" and x.attr not in methods:
                    writes.add(x.attr)
    out[name] = dict(lines=[fn.lineno, fn.end_lineno], globals=sorted(g), calls=sorted(calls),
                     reads=sorted(reads), writes=sorted(writes), clsref=sorted(clsref),
                     static=any(ast.unparse(d) == "staticmethod" for d in fn.decorator_list))
assert len(methods) == 191 or "--any" in sys.argv, f"{len(methods)} methods: not the whole World"
json.dump(out, open(OUT, "w"), indent=0)
# callers
callers = {m: [] for m in methods}
for m, d in out.items():
    for c in d["calls"]: callers[c].append(m)
json.dump(callers, open(OUT.replace(".json", "_callers.json"), "w"), indent=0)
print(len(methods), "methods")
