"""Every name the original sim module had is still there, with the same value,
and the mutable tables are ONE object across modules (tests mutate BODY in
place; every module must see it).

    python3 reexport.py dump <repo> > names.json
    python3 reexport.py diff a.json b.json
    python3 reexport.py shared <repo>
"""
import json, os, sys, types, hashlib, pickle

def dump(repo):
    sys.path.insert(0, repo)
    import numpy as np
    from src.voxel import sim
    out = {}
    for k, v in vars(sim).items():
        if k.startswith("__"):
            continue
        if isinstance(v, types.ModuleType):
            out[k] = "module:" + v.__name__.split(".")[-1]
        elif isinstance(v, (types.FunctionType, type)) or callable(v):
            out[k] = "callable:" + getattr(v, "__qualname__", repr(type(v)))
        elif isinstance(v, np.ndarray):
            out[k] = "array:" + v.dtype.str + str(v.shape) + hashlib.sha1(v.tobytes()).hexdigest()
        else:
            out[k] = "value:" + repr(v)
    return out

if sys.argv[1] == "dump":
    json.dump(dump(sys.argv[2]), sys.stdout, indent=0, sort_keys=True)
elif sys.argv[1] == "diff":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    missing = sorted(set(a) - set(b)); changed = sorted(k for k in a if k in b and a[k] != b[k])
    print(f"{len(a)} names before, {len(b)} after; missing {missing}; changed {changed}")
    print("new names:", sorted(set(b) - set(a)))
    sys.exit(1 if missing or changed else 0)
elif sys.argv[1] == "shared":
    sys.path.insert(0, sys.argv[2])
    import importlib
    from src.voxel import sim
    bad = []
    for modname in ("materials", "policy", "kernels", "pose", "fields", "fluids", "heat", "air",
                    "support", "bodies", "skeleton", "persons", "physiology", "senses",
                    "motion", "hands", "will", "ledgers"):
        try:
            m = importlib.import_module(f"src.voxel.{modname}")
        except ModuleNotFoundError:
            continue
        for k, v in vars(m).items():
            if k in vars(sim) and not k.startswith("__") and vars(sim)[k] is not v \
                    and not isinstance(v, (int, float, str, bool, tuple, frozenset, type(None))):
                bad.append(f"{modname}.{k} is not sim.{k}")
    print("shared objects:", "OK" if not bad else bad)
    sys.exit(1 if bad else 0)
