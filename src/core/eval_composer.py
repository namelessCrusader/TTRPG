"""Golden-set benchmark for the composer: python -m src.core eval [--brain torch]

Loads data/composer_eval.yaml, builds each fixture world, runs compose(), and
matches canonical steps against the golden answer (or its alternates). Prints
score per band + every miss. This is the number we tune against — change a
knob, rerun, keep or revert.
"""

from __future__ import annotations

import os

import yaml

from . import composer
from .seed import vault

_DATA = os.path.join(os.path.dirname(__file__), "data", "composer_eval.yaml")


def _fixture(spec):
    world, player = vault(seed=0)
    player.pos = tuple(spec["player_pos"])
    # goldens assume mid-scene state: the containers have already spilled
    for eid in () if spec.get("intact") else ("barrel_oil", "keg_oil", "cask_water"):
        e = world.entities[eid]
        lv = e.parts.get("liquid_volume")
        if lv and lv.get("ml", 0) > 0:
            world.cell(e.pos).fluids[lv["mat"]] = lv["ml"]
            lv["ml"] = 0.0
        e.tags.add("broken")
    for p in spec.get("on_fire", []):
        world.cell(tuple(p)).tags.add("on_fire")
    for eid in spec.get("carry", []):
        e = world.entities[eid]
        e.tags.add("taken")
        e.pos = player.pos
        player.tags.add(f"inv:{eid}")
    return world, player


def _canon(world, player, step):
    """Golden step spec → canonical tuple (same space compose() emits).
    Golden coords are 2D-authored; lift to voxel space here, in one place."""
    from src.core.spatial import lift

    def tuple_(v):
        return lift(tuple(v))
    k = step[0]
    if k == "pour":
        tgt = player.pos if step[1] == "self" else tuple_(step[1])
        return ("pour", tgt, step[2])
    if k == "ignite":
        return ("ignite", tuple_(step[1]))
    if k == "smother":
        return ("smother", tuple_(step[1]))
    if k == "shove":
        eid, mode, ref = step[1], step[2], tuple_(step[3])
        dest = ref if mode == "into" else composer._away_dest(world, eid, ref)
        return ("shove", eid, dest)
    if k == "take":
        return ("take", step[1])
    if k == "smash":
        return ("smash", step[1])
    if k == "throw_obj":
        return ("throw", step[1], tuple_(step[2]))
    if k == "goto":
        return ("move", tuple_(step[1]))
    if k == "any_move":
        return ("move", "*")
    if k == "social":
        return ("social", step[1])
    if k == "inspect":
        return ("inspect", step[1] if isinstance(step[1], str) else tuple_(step[1]))
    if k == "refuse":
        return ("refuse",)
    raise ValueError(f"unknown golden step {k!r}")


def _match(got, want):
    if len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if w[0] == "refuse":
            if g[0] != "refuse":        # reasons vary; refusing at all is what's golden
                return False
        elif w[0] == "move" and w[1] == "*":
            if g[0] not in ("move", "goto_path"):
                return False
        elif g[0] == "goto_path" and w[0] == "ignite":   # auto-goto counts as its action
            if ("ignite", g[2]) != w:
                return False
        elif g != w:
            return False
    return True


def run(brain: str = "mock", band: str = "", verbose: bool = True, debug: bool = False) -> dict:
    with open(_DATA) as f:
        data = yaml.safe_load(f)
    picker = composer.TorchPicker() if brain == "torch" else composer.MockPicker()
    if debug:
        picker = composer.DebugPicker(picker)
    by_band, misses = {}, []
    for case in data["cases"]:
        cid = case["id"]
        if band and not cid.startswith(band):
            continue
        world, player = _fixture(data["fixtures"][case["fixture"]])
        got = composer.compose(world, player, case["say"], picker)
        golds = [case["steps"]] + case.get("alt", [])
        ok = any(_match(got, [_canon(world, player, s) for s in g]) for g in golds)
        b = cid[0]
        by_band.setdefault(b, [0, 0])
        by_band[b][0] += 1
        if ok:
            by_band[b][1] += 1
        else:
            misses.append((cid, case["say"], got,
                           [_canon(world, player, s) for s in case["steps"]]))
    total = sum(v[0] for v in by_band.values())
    passed = sum(v[1] for v in by_band.values())
    if verbose:
        names = {"d": "direct", "c": "compound", "v": "vague", "s": "stunts",
                 "o": "objects", "q": "questions", "t": "conditional", "i": "impossible"}
        print(f"composer eval — brain={brain}")
        for b, (n, p) in sorted(by_band.items()):
            print(f"  {names.get(b, b):12s} {p:2d}/{n}")
        print(f"  {'TOTAL':12s} {passed:2d}/{total}")
        for cid, say, got, want in misses:
            print(f"  MISS {cid}: \"{say}\"\n       got  {got}\n       want {want}")
    return {"passed": passed, "total": total, "misses": misses}
