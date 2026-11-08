"""On top of patch_onset.py: the policy hooks and structured flags the mind
link needs (prepare / limbs / on_restore, intent.aim, intent.wake_at, _swung,
_took, _gave_up, _arrived, the `struck` percept). Scratch copy only."""
P = "src/voxel/sim.py"
s = open(P).read()
def sub(old, new, count=1):
    global s
    k = s.count(old)
    assert k == count, (k, old[:90])
    s = s.replace(old, new)

# H1 prepare + limbs at the top of _decide (after `sit` is built)
sub("        menus, picks, chosen, seen, would = {}, {}, {}, {}, {}\n        spelled = []\n",
    "        prep = getattr(self.policy, \"prepare\", None)\n"
    "        if prep is not None:\n            prep(self, p, percept)\n"
    "        ask = getattr(self.policy, \"limbs\", None)\n"
    "        asked = ask(self, p, percept) if ask is not None else None\n"
    "        menus, picks, chosen, seen, would = {}, {}, {}, {}, {}\n        spelled = []\n")
sub("        for limb in ((\"eyes\",) if percept in ORIENTING else LIMBS):\n            menu = self._menu(p, cells, percept, limb, chosen)\n",
    "        for limb in ((\"eyes\",) if percept in ORIENTING else LIMBS):\n"
    "            if asked is not None and limb not in asked:\n                continue          # not asked: it goes on as it was\n"
    "            menu = self._menu(p, cells, percept, limb, chosen)\n")
# H2 a decide that asked nothing still counts as deciding (no-op today: picks is never empty)
sub("            chosen[limb] = opt[\"tag\"]\n        if not picks:\n            return\n",
    "            chosen[limb] = opt[\"tag\"]\n        if not picks:\n            p[\"_decided\"] = self.tick\n            return\n")
# H3 aim
sub("                                toward=p.get(\"facing\"),\n",
    "                                toward=(p.get(\"intent\") or {}).get(\"aim\")\n                                or p.get(\"facing\"),\n")
# H4 _swung, only when an arm really went
sub("                if b is not None and hands[\"verb\"] == \"throw\":\n                    b[\"throw\"] = True\n",
    "                if b is not None:\n                    p[\"_swung\"] = (self.tick, hands[\"verb\"])\n"
    "                if b is not None and hands[\"verb\"] == \"throw\":\n                    b[\"throw\"] = True\n")
# H5 _took
sub("                    self._take_up(p, cells)    # to the fist, if the arm can\n",
    "                    self._take_up(p, cells)    # to the fist, if the arm can\n                    p[\"_took\"] = (self.tick, tuple(ob[\"cell\"]))\n")
# H6 _gave_up / _arrived
sub("                    p.setdefault(\"_baulked\", {})[tuple(goal)] = self.tick\n",
    "                    p.setdefault(\"_baulked\", {})[tuple(goal)] = self.tick\n                    p[\"_gave_up\"] = (self.tick, tuple(goal))\n")
sub("            if reach * vox_m <= WILL[\"arrive_m\"]:\n                p[\"_near\"] = None\n",
    "            if reach * vox_m <= WILL[\"arrive_m\"]:\n                p[\"_near\"] = None\n                p[\"_arrived\"] = (self.tick, tuple(goal))\n")
# H7 wake: the mind's answer lands at a tick it chose (onset law only)
sub("                    if onset or (due and p.get(\"goal\") is None):\n",
    "                    wake_ = (p.get(\"intent\") or {}).get(\"wake_at\")\n"
    "                    if onset or (due and p.get(\"goal\") is None) or (wake_ is not None and self.tick >= wake_):\n")
# H8 struck: who hit whom, from _spend, and a felt percept under onset
sub("        for q, cell, e in blows.values():\n",
    "        for q, cell, e in blows.values():\n            if q is not None:\n                q[\"_struck\"] = (self.tick, b.get(\"owner\"))\n")
sub("                if percept is None and self._falling_near(p, cells):\n",
    "                if percept is None and self.attention == \"onset\" and p.get(\"_struck\") \\\n"
    "                        and self.tick - p[\"_struck\"][0] <= 1:\n"
    "                    percept = \"struck\"\n                    src = p[\"_struck\"]\n"
    "                if percept is None and self._falling_near(p, cells):\n")
# H9 restore tells the mind
sub("        self.__dict__.update(copy.deepcopy(snap))\n",
    "        self.__dict__.update(copy.deepcopy(snap))\n        cb = getattr(self.policy, \"on_restore\", None)\n        if cb is not None:\n            cb(self)\n")
open(P, "w").write(s)
print("hooks patched")
