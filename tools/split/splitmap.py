"""The proposed sim.py split, as data: method -> module; checks it is total.

    python3 deps.py <pristine sim.py> deps.json     # next to this file
    python3 splitmap.py                             # writes splitmap.json here

split.py reads deps.json and splitmap.json from this folder."""
import json, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, 'deps.json')))
callers = json.load(open(os.path.join(HERE, 'deps_callers.json')))
M = {
 "sim": "__init__ step".split(),
 "fields": "heat_capacity _matter_moved T porosity solid burning fill pour _mix _window _hot_window _busy".split(),
 "fluids": "_move_fluid _fly_drops _law_head _law_flow _law_stir _splash _law_acid".split(),
 "heat": "_law_conduct _law_rise _gas_access _law_burn _law_boil _law_melt".split(),
 "air": "_blur_cells _cells _upcell _law_air _law_smoke _air_regions _region_of_surface _law_o2 _mixing _pushed".split(),
 "support": "_law_support _support_once _fall_speed _relax_slack _free_fall_to_seam _settle _cash_impacts _flaw01 _shatter _law_crack _law_torque _topple _collapse".split(),
 "bodies": "_launch _fly_pose _drag_a _body_pose _release loose _law_bodies _left_world _spend _land_body bodies_array".split(),
 "skeleton": ("_limbs _bones _parents _kids _subtree _path_from _pivot_of _joint_xform _axis_xform _shear_steps _apply_shears "
              "_turn_cells _chain_shear _comp3 _apply3 _chain3 _chain_xform _apply_xform _limb_parts _bone_parts _bone_angle "
              "_limb_cells _still_joined _angles _inv_xform _comp_xform _bone_at _thickness _cut_section _body_span _reach_of "
              "_legs_of _torque_of _face_axis _crouch_of _fist_of _fists _hand _hand_strength _in_reach _span_xy").split(),
 "pose": "_swing _turn _square_up _repose _hanging_cells _hold_torque _swing_of _law_joints _law_pose _reach_around _bunch _crouch _bend_max".split(),
 "persons": "add_person _person_cells _claim_bodies _carry_own _own_claim _flood_person _person_at person_status".split(),
 "physiology": "_blow _blood0 _perfusion _delivered_o2 _law_bleed _spill_from _wound _law_life".split(),
 "senses": "_look_around _sees _in_cone _hears _law_hear _noise _level _say _in_flight_near _falling_near".split(),
 "motion": ("_walkable _bearing _reach_along _places _underfoot _contact _leap_speed _leap_targets _leap _gather_ticks "
            "_fit_grid _routes _plan_path _walk _leave _overbalanced _pulled_over _law_footing").split(),
 "hands": ("_resist _within_reach _objects_within_reach _held_cells _object_at _effort _shove _catch_person _catch "
           "_grip_holds _grip_cells _borne_cells _rub _haul_cost _held_clusters _take_up _haul_held _haul _law_grips").split(),
 "will": "_menu _full_menu _decide trace_outcomes _law_will _decide_falling".split(),
 "ledgers": "snapshot restore total_fluid total_mass total_wood total_energy total_lost".split(),
}
CLSATTR = {"_PERCELL": "bodies", "_CS": "air", "_SMOKE_REACH": "air", "_NOT_STATE": "ledgers"}
where = {}
for mod, ms in M.items():
    for m in ms:
        assert m not in where, m
        where[m] = mod
missing = set(d) - set(where); extra = set(where) - set(d)
print("missing", missing, "extra", extra, "assigned", len(where), "of", len(d))
# module-level names -> module
G = {}
def put(mod, names):
    for n in names.split(): G[n] = mod
put("kernels", "HAVE_NUMBA njit _conduct_pairs _o2_diffuse _gas_overturn")
put("materials", "NOFLUID WATER OIL ACID WEAK_ACID MTIN MLEAD MIRON BLOOD AMBIENT AIR WOOD STONE IRON FLESH ASH GLASS LEAF CHAR TIN LEAD NMAT "
    "SOLID FLUID _FLIDS _FDENS _FC _FK REACTIONS MISCIBLE CHEST0 BODY WOUND BLEED FRICTION GRAVITY TICK_S AIR_DENS DRAG_CD FALL_SUBSTEPS "
    "_REACTIVE PLUME_REACH_M BOIL LATENT COMBUST_CEIL RAD LEAK SPAN MELT FREEZE TOUGH TSHOCK MATNAME TRANSMIT SEE_TAU SMOKE_EXT "
    "_SIX SMOKE_STILL SHOUT_DB HEAR_DB SOUND PUSH_THROUGH _TOUGH_ARR _TSHOCK_ARR _DENS_ARR _CSOLID_ARR _KSOLID_ARR _SPAN_ARR "
    "_FDENS_ARR _FC_ARR _FK_ARR _TRANSMIT_ARR O2_PER_L O2_RATIO SOOT SOOT_OIL O2_RATIO_OIL")
put("pose", "_SWEEP_RAD")
put("policy", "GAZE_SWEEP WILL LIMBS _DIRS ACTS REFLEXES ORIENTING LINES ANSWERS MEANS IDLE MENU_CAP TablePolicy")
ext = {"np": "numpy", "math": "math", "copy": "copy", "seam": ". seam", "Matter": "..mpm.matter"}
report = {}
for mod, ms in M.items():
    gl = collections.defaultdict(set); calls_out = collections.defaultdict(set); cls = set()
    for m in ms:
        for g in d[m]["globals"]:
            gl[G.get(g, ext.get(g, "?" + g))].add(g)
        for c in d[m]["calls"]:
            if where[c] != mod: calls_out[where[c]].add(c)
        cls |= set(d[m]["clsref"])
    report[mod] = {"imports": {k: sorted(v) for k, v in sorted(gl.items())},
                   "calls_into": {k: sorted(v) for k, v in sorted(calls_out.items())},
                   "class_attrs_used": sorted(cls),
                   "lines": sum(d[m]["lines"][1]-d[m]["lines"][0]+1 for m in ms)}
json.dump({"where": where, "report": report, "G": G},
          open(os.path.join(HERE, "splitmap.json"), "w"), indent=1)
for mod, r in report.items():
    print(f"\n## {mod}  ({len(M[mod])} methods, {r['lines']} lines)")
    for k, v in r["imports"].items(): print(f"  import from {k}: {' '.join(v)}")
    for k, v in r["calls_into"].items(): print(f"  self-> {k}: {' '.join(v)}")
    if r["class_attrs_used"]: print("  class attrs:", r["class_attrs_used"])
# module dependency graph between table modules
