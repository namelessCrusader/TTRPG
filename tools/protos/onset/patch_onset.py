"""Patch the SCRATCH copy of sim.py with the proposed attention law
(World.attention = "latch" | "onset"). Never touches the repo."""
import re, sys
P = "src/voxel/sim.py"
s = open(P).read()
n0 = s.count("\n")

def sub(old, new, count=1):
    global s
    k = s.count(old)
    assert k == count, (k, old[:80])
    s = s.replace(old, new)

# 0. the world-level law setting (default = today's latch: byte-equal)
sub("        self.recall_check = False\n",
    "        self.recall_check = False\n        self.attention = \"latch\"\n")
# 0b. a new WILL constant
sub('        "decide_every": 10}',
    '        "attend_gap": 40,\n        "decide_every": 10}')
# 1. senses run every tick under onset
sub("            if not p.get(\"emergency\"):\n                percept = None\n",
    "            if not p.get(\"emergency\") or self.attention == \"onset\":\n                percept = None\n                src = None\n")
# 2. percept sources (identity of what was noticed)
sub("                if percept is None and self._falling_near(p, cells):\n                    percept = \"sees_falling\"     # somebody is going past you\n",
    "                if percept is None and self._falling_near(p, cells):\n                    percept = \"sees_falling\"     # somebody is going past you\n                    src = tuple(sorted(str(b.get('owner')) for b in self._falling_near(p, cells)))\n")
sub("                            if means:\n                                percept = means          # ears have no cone and\n",
    "                            if means:\n                                percept = means          # ears have no cone and\n                                src = (q['name'], said[0])\n")
sub("                            percept = \"sees_runner\"\n                            break\n",
    "                            percept = \"sees_runner\"\n                            src = q['name']\n                            break\n")
sub("                    percept = \"hears_noise\"              # something, over there\n",
    "                    percept = \"hears_noise\"              # something, over there\n                    src = nz_['tick']\n")
# 3. the trigger
old = ("                if percept is not None or (due and p.get(\"goal\") is None):\n"
       "                    self._decide(p, percept, cells, ax_, ay_, eye, vox_m)\n")
new = ("                if self.attention == \"onset\":\n"
       "                    onset = False\n"
       "                    if percept is not None:\n"
       "                        seen_ = p.setdefault(\"_attn\", {})\n"
       "                        k_ = (percept, src)\n"
       "                        onset = k_ not in seen_ or self.tick - seen_[k_] > WILL[\"attend_gap\"]\n"
       "                        seen_[k_] = self.tick\n"
       "                    if onset or (due and p.get(\"goal\") is None):\n"
       "                        p[\"_onset\"] = onset\n"
       "                        self.attn_decides = getattr(self, 'attn_decides', 0) + 1\n"
       "                        self._decide(p, percept, cells, ax_, ay_, eye, vox_m)\n"
       "                elif percept is not None or (due and p.get(\"goal\") is None):\n"
       "                    self._decide(p, percept, cells, ax_, ay_, eye, vox_m)\n")
sub(old, new)
# 4. the situation says whether this is news (only under onset: latch rows stay byte-equal)
sub("               \"others_in_earshot\": heard}\n        menus, picks, chosen, seen, would = {}, {}, {}, {}, {}\n",
    "               \"others_in_earshot\": heard}\n        if self.attention == \"onset\":\n            sit[\"onset\"] = bool(p.get(\"_onset\"))\n        menus, picks, chosen, seen, would = {}, {}, {}, {}, {}\n")
# 5. emergency ends when the goal does (arrival or giving up), under onset
sub("                    p[\"goal\"], p[\"_path\"], p[\"_near\"] = None, None, None\n                    p[\"_tried\"] = (None, 0)\n                    p[\"fleeing\"] = False\n",
    "                    p[\"goal\"], p[\"_path\"], p[\"_near\"] = None, None, None\n                    p[\"_tried\"] = (None, 0)\n                    p[\"fleeing\"] = False\n                    if self.attention == \"onset\":\n                        p[\"emergency\"] = False\n")
sub("                else:                                # arrived somewhere ordinary:\n                    p[\"goal\"] = None                 # pick again next time round\n",
    "                else:                                # arrived somewhere ordinary:\n                    p[\"goal\"] = None                 # pick again next time round\n                    if self.attention == \"onset\":\n                        p[\"emergency\"] = False\n")
# 6. TablePolicy: a mouth reflex answers NEWS only (no-op when sit has no 'onset')
sub("        if situation[\"percept\"] is not None:\n            row = situation[\"reflexes\"].get(situation[\"percept\"]) or {}\n            if row.get(limb):\n",
    "        if situation[\"percept\"] is not None:\n            row = situation[\"reflexes\"].get(situation[\"percept\"]) or {}\n            if limb == \"mouth\" and not situation.get(\"onset\", True):\n                row = {}\n            if row.get(limb):\n")
open(P, "w").write(s)
print("patched, lines", n0, "->", s.count("\n"))
