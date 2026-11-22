"""Pass rules for one run dir. The fixed worlds are relabelled to the names the 0a report code expects:
fix -> all2, fixnoR -> noR (so the 'extractor wins only where reprisal is weak' rule compares fix vs fixnoR).
usage: analyze.py DIR [--fresh] [--old]   (--old: the 0a runs themselves; --fresh: seeds 11-13)"""
import sys, json
from pathlib import Path
import lab.mindL.tournament_report as tr
D = sys.argv[1]
R = tr.load(D)
if "--old" not in sys.argv:
    R = {(k, {"fix": "all2", "fixnoR": "noR", "luck": "all2", "luckR0": "noR"}.get(w, w), c, s, rt): r for (k, w, c, s, rt), r in R.items()}
if "--fresh" in sys.argv:
    tr.SEEDS = (11, 12, 13)
T = tr.tournament(R)
P = tr.pass_rules_a(T, R)
O = tr.oracle(R) if any(k[0] == "O" for k in R) else {"rows": {}}
B = tr.bcqw(R)
S = tr.stance_check(R)
F = tr.flows(R)
if O["rows"]:
    print(tr.text(T, P, O, B, S))
else:
    print(tr.text(T, P, {"rows": {}}, B, S))
print("\nFLOWS (all2/fix):", json.dumps(F.get("all2", {}), default=str)[:3000])
print("\nGOSSIP EFFECT", json.dumps(B["gossip_effect"]["all2"]))
print("\nBY VERB (focal+others) rates, selected")
for v in ("serve","give","praise","gossip","strike","insult","accuse","search","buy","work"):
    x = B["worlds"]["all2"]["by_verb"].get(v)
    if x: print(v, {k: (round(y,3) if isinstance(y,float) else y) for k,y in x.items()})
