"""The loader checks the design promises (all of them), run on the v2 draft."""
import re, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
caps, drives, verbs = c["capabilities"], c.get("drives", {}), c["verbs"]
LIMBS = {"legs", "hands", "mouth", "waist", "eyes", None}       # sim.py:348 + null
ROOMS_VERBS = {"walk to", "flee to", "take", "steal", "drop", "give", "eat", "drink", "strike", "shove",
               "threaten", "insult", "praise", "tell", "accuse", "shout for help", "help up", "bandage",
               "defend", "hide", "search", "open", "play", "wait"}          # lab world.py:87-90 DUR keys
bad = []
for v in verbs:
    vid, roles = v["id"], set(v["roles"])
    for key in ("text", "text_with", "told", "told_with", "told_start"):           # 1. roles declared
        if key in v:
            used = set(re.findall(r"{(\w+)}", v[key])) - {"actor"}
            if not used <= roles:
                bad.append((vid, key, sorted(used - roles)))
    for be in ("rooms", "voxel"):
        if v["backends"].get(be) and "told" not in v:
            bad.append((vid, "executable but no told template"))
        if v["backends"].get(be) and v.get("duration", {}).get(be) is None:
            bad.append((vid, be, "executable but no duration"))
    need = list(v.get("needs", []))
    for r in v["roles"].values():
        need += ([r["cap"]] if "cap" in r else []) + r.get("cap_any", [])
    if v.get("hint"):
        need.append(v["hint"])
    for cap in need:
        if cap not in caps:
            bad.append((vid, "unknown capability", cap))
            continue
        if "says" not in caps[cap]:                                                   # 2. says
            bad.append((cap, "no says"))
        for be in ("rooms", "voxel"):                                                 # 3. a test per backend
            if v["backends"].get(be) and caps[cap].get(be) is None and cap in v.get("needs", []):
                bad.append((vid, be, f"needs {cap} with no {be} test"))
    for d in v.get("serves", []):
        if d not in drives:
            bad.append((vid, "unknown drive", d))
    for f in v.get("flags", []):
        if f not in c["flags_known"]:
            bad.append((vid, "unknown flag", f))
    rb = v["backends"].get("rooms")
    if rb and rb["verb"] not in ROOMS_VERBS:
        bad.append((vid, "rooms verb not in the inn", rb["verb"]))
    for st in v["backends"].get("voxel") or []:
        if st.get("limb") not in LIMBS:
            bad.append((vid, "no such limb", st.get("limb")))
        if st["until"] not in c["step_tests"]:
            bad.append((vid, "unknown step test", st["until"]))
    if set(v) - {"id", "group", "null_act", "roles", "needs", "serves", "hint", "text", "text_with", "told",
                 "told_with", "told_start", "duration", "commit_after", "interrupt_on", "flags", "backends", "when"}:
        bad.append((vid, "unknown field", sorted(map(str, set(v)))))
for cap, spec in caps.items():
    if "says" not in spec:
        bad.append((cap, "no says"))
    if spec.get("voxel") and "edge_max_m2" in spec["voxel"] and spec["voxel"].get("edge_gt_m2") is None:
        bad.append((cap, "edge test admits edge=0 (blunt)"))
for be in ("rooms", "voxel"):                                                         # 4. verb sets
    print(be, "executes:", [v["id"] for v in verbs if v["backends"].get(be)])
print("problems:", bad or "none")
