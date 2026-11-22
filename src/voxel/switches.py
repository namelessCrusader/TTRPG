"""THE SWITCH LEDGER OF WAVE 13 (C11): every gate and every crutch the wave left in the
tree, each with its owner and the condition that deletes it.

A GATE is a capability the world turns on (or off) as data. A CRUTCH is a stand-in that
keeps an old law working until a general one replaces it. An ESTIMATE is a cheaper law or
a constant marked (u) in a report. `audit()` returns the problems it finds, and
tests/test_switch_ledger.py fails on any of them:

  - a row with no owner, no `why`, or a `deleted_when` that says nothing (under 20 characters);
  - a crutch or an estimate whose condition is "never": only a gate or a permanent wiring row
    may keep itself for ever, and then it must say why after "never:";
  - a gate whose probe does not find the attribute, or finds another default than the ledger says;
  - a gate in World's class attributes (a name ending in _laws or _opacity) that has no row;
  - a legacy-skip row of heatmix.CRUTCHES that the ledger does not repeat.

Rows are plain dicts so a lane adds one by appending to ROWS. Kinds: gate, crutch, estimate,
permanent. A probe is ("class", "module:Class.attr", default) or ("source", "module", needle).
Plain words and real numbers; (u) marks what no source has confirmed.
"""
import importlib
import re

from . import heatmix

KINDS = ("gate", "crutch", "estimate", "permanent")

# Short names for the probes.
_G = "class"
_S = "source"


def _row(id, kind, owner, why, deleted_when, probe=None):
    return {"id": id, "kind": kind, "owner": owner, "why": why,
            "deleted_when": deleted_when, "probe": probe}


ROWS = (
    # ── 13b: the four switches ────────────────────────────────────────────────
    _row("cold.ColdWorld", "gate", "C1",
         "Cold (the outside climate, the 0 C floor removed, the matter rest rule) lives in ColdWorld, "
         "on only where w.climate is set; the quiet cost is +1.1 ms against a +0.3 ceiling.",
         "sim.py puts OutsideMixin in World's bases and the quiet scan is cheaper than 0.3 ms, "
         "then ColdWorld = World",
         (_G, "src.voxel.sim:World.climate", None)),
    _row("cold.quiet_memo", "crutch", "C1",
         "The memo of a quiet answer (_uns_quiet, list _E_WRITERS): a law that writes E and is not in the "
         "list leaves a stale quiet for the rest of one tick.",
         "the scan is incremental (SCALE) or every E write bumps a version counter",
         (_S, "outside_world", "_E_WRITERS")),
    _row("cold.plume_k", "crutch", "C1",
         "Climate.plume_k = 100 K and the smoke skip in the outdoor relaxation stand in for plume entrainment.",
         "a real plume entrainment law exists (wind or GAS topic) and k_far is calibrated against W6 and a campfire plume",
         (_S, "climate", "plume_k")),
    _row("cold.legacy_pressure_mean", "crutch", "C1",
         "The legacy pressure field keeps the old coarse mean over all voxels (the spec says gas cells).",
         "the legacy pressure field is retired because gas_flow becomes the default",
         None),
    _row("cold.matter_rule_partial", "crutch", "C1",
         "The matter rest rule runs only at things._unmake; burn, char, boil, melt, freeze, spawn and scenes.fill "
         "still change capacity at a fixed E.",
         "each of those laws calls outside.matter_change (C3 and C2 migrate them one by one)",
         None),
    _row("cold.pour_at_20C", "crutch", "C3",
         "fluids.pour makes the liquid at 20 C: right for a bottle at room temperature, wrong for outdoor water.",
         "pour takes a temperature argument and the scenes that spill outdoors pass it",
         None),
    _row("cold.soak_last", "crutch", "C1",
         "A scene must call soak() last or the lattice is out of step with the climate.",
         "set_climate(soak=True) is called by the scene builders themselves (scenes_cold does) and no scene calls it first",
         None),
    _row("mix.mix_laws", "gate", "C2",
         "Brine boils and mixes melt only while w.mix_laws is on (default on); off is the 13a state.",
         "C11 retires the 13a-state tests that set it False and no scene needs the old skip",
         (_S, "heat", "mix_laws")),
    _row("mix.mix_compiled", "gate", "C2",
         "w.mix_compiled False runs the Python fluid sweep that the compiled twin must equal bit for bit.",
         "with mix_laws: the twin test is the only user of the switch and the Python sweep is the plain path",
         (_S, "fluids", "mix_compiled")),
    _row("mix.steam_to_gone", "crutch", "C2",
         "Steam from a mixed boil goes to the gone book, not a vapour store.",
         "the wet world's vapour store takes the steam of a mixed liquid too (T4 6.3 item 4)",
         None),
    _row("mix.crystal_mat21", "crutch", "C2",
         "A crystal residue lands in mat 21, whose table row is air-like until ids 13-26 get rows.",
         "table rows for mat ids 13-26 exist in materials.py",
         None),
    _row("mix.skip._law_melt", "crutch", "C2",
         "_law_melt skips a mixed cell no metal system governs (aqueous, three species).",
         "the ice cell and the solution rows (brine freezing, M1 of C12) are migrated",
         (_S, "heatmix", "_law_melt")),
    _row("mix.skip._burn_cells", "crutch", "C2",
         "_burn_cells skips a mixed solid or oil pool: it would lose the other species.",
         "burning moves to the species fuel law (take and put); first user is the reaction row runner (M3 of C12)",
         (_S, "heatmix", "_burn_cells")),
    _row("mix.skip._law_skin_water", "crutch", "C2",
         "_law_skin_water skips a brine held in a porous skin.",
         "the skin water has a composition and boil_step runs on it",
         (_S, "heatmix", "_law_skin_water")),
    _row("mix.skip._law_boil", "crutch", "C2",
         "_law_boil skips a mixed liquid when the mix gate is off or the liquid holds no dissolved water.",
         "the gate is retired with mix.mix_laws",
         (_S, "heatmix", "_law_boil")),
    _row("body.BodyWorld", "gate", "C6",
         "Body heat (warm flesh, the thermoregulation reflexes, gas-only buoyancy) lives in BodyWorld, per world, "
         "because the warm-body wake costs +18 ms at 0.4 M and +67 ms at 1.6 M against a +3 ms ceiling.",
         "the air law is cheap enough (a measured wake under 3 ms per person at 1.6 M) and BodyHeatMixin joins World's bases",
         (_S, "bodyworld", "class BodyWorld")),
    _row("body.gas_busy_fallback", "crutch", "C6",
         "_gas_busy falls back to _heat_busy in a climate world.",
         "C1's rest-relative gas test exists",
         None),
    _row("body.reflex_out_when_asleep", "crutch", "C6",
         "Reflex rows with no stimulus scale keep the old 'out when asleep' (balance).",
         "balance gets a stimulus scale and the sleep threshold law applies to it",
         None),
    _row("body.sleep_constants", "estimate", "C6",
         "CORE_COLD_WEIGHT 3.0, sleep depth 0.6 and 0.9, gain 4, width 0.35, dial 0.6 to 1.6 are all (u).",
         "a sleep and thermoregulation data set fits them",
         None),
    _row("body.clothing_lattice_path", "crutch", "C6",
         "Clothing rows are built but the skin still emits at skin temperature on the lattice path (not validated, R10).",
         "a per-cell emission overlay carries the clothing resistance",
         None),
    _row("body.breath_to_shed", "crutch", "C6",
         "Breath heat and the forced-convection correction go to world.shed (about 2 W a person), not the room air.",
         "a lattice AirSink takes them (C10's RegionSink is the first)",
         None),
    _row("body.stores_injection", "permanent", "C6",
         "stores_init and body_warmup inject named joules at birth of a heat record.",
         "never: a body has to start with some heat; the injection is named and booked",
         None),
    _row("body.work_billing", "crutch", "C6",
         "Only drag is billed to the stores; lift, swing, throw and climb heat comes from nowhere.",
         "hands.py bills every act through energetics.release_J (C8a and C7 own the acts)",
         None),
    _row("body.p_follow", "crutch", "C6",
         "P_FOLLOW 0.03 is a stability fix for the pneumatic field.",
         "gas_flow replaces the pneumatic field",
         None),
    _row("mat.mat_laws", "gate", "C8a",
         "world.mat_laws is a frozenset of the law names that are on: friction, members, spall, edge, wound "
         "(damage is built and off).",
         "every scene runs all of them and nothing sets the set empty",
         (_G, "src.voxel.sim:World.mat_laws", frozenset({"friction", "members", "spall", "edge", "wound"}))),
    _row("mat.span_floor", "crutch", "C8a",
         "SPAN stays as a veto floor under the new load check (decision D5).",
         "W9: the user retires SPAN after arch detection in the lattice and a review of the overhangs",
         None),
    _row("mat.friction_voxel_only", "crutch", "C8a",
         "Friction is read only in the voxel _effort; the MPM and the bonds keep 0.40.",
         "W5b: a per-node wall-material field for the MPM contact",
         None),
    _row("mat.veto_changed_cells", "crutch", "C8a",
         "The members veto asks only about changed or hot cells: a world built standing is taken to stand.",
         "a full-structure audit runs at scene build (matlaws.analyse_world with every solid wanted)",
         None),
    _row("mat.veto_static", "crutch", "C8a",
         "The veto counts static loads only (no walker DAF, no landing).",
         "contact events feed extra_N (C13 gives the footfall, the landing event is still to come)",
         None),
    _row("mat.char_floor", "crutch", "C8a",
         "Wood above 350 C reads a char floor in matlaws._record.",
         "a char cell exists on the lattice and the derate reads the char row itself",
         None),
    _row("mat.veto_constants", "estimate", "C8a",
         "VETO_EVERY 10, _HOT 40, _SOON 2, HOT_STRUCT_C 100, CROP_MARGIN 24, TOUCH 4, MIN_FILL 0.15, CHAR_KS 0.05, "
         "CHAR_KE 0.09 are (u).",
         "each is measured against the full audit and replaced by the measured number",
         None),
    _row("mat.damage_off", "gate", "C8a",
         "The damage law is built and off by default.",
         "support.py has its two sites and test_carried_opt's fixture sets its laws",
         None),
    _row("mat.eta_impact", "estimate", "C8a",
         "ETA 0.3 (u) in matdamage.ETA_IMPACT.",
         "a sourced crack-energy number replaces it",
         None),
    _row("mat.spall_one_face", "crutch", "C8a",
         "Spall sees one exposed face and no pore pressure.",
         "moisture inside a solid (Fick) is wired (C3 open problem 2)",
         None),
    _row("mat.mpm_inversion", "crutch", "C8a",
         "MPM wood 10 MPa and stone 8 MPa (stone below wood) stay in the matrec legacy rows.",
         "D5: the numbers are unified one by one, wood and stone first",
         None),
    # ── 13c ───────────────────────────────────────────────────────────────────
    _row("wet.w.wet", "gate", "C3",
         "Humidity, kept steam, mist, ice and moisture run only where w.wet is a WetState (WetMixin, in World's "
         "bases since R4); `w.wet = None` is the old dry world, every method falls back to super().",
         "never: the gate is how a test or a scene asks for the old dry world",
         (_S, "wetmix", "self.wet")),
    _row("wet.humid_default", "gate", "R4",
         "wave 14a refinement R4: HumidMixin is in World's bases and World.__init__ ends with humid_default(), so a "
         "bare World(...) has a humidity (SETTING_RH, temperate 0.5 unless in_setting says other). Replaces the "
         "crutch wet.air_rh_none (World.air_rh = None and bodyworld's 'else 0.5') which is deleted. Idle cost is "
         "noise (R4 report).",
         "never: the setting is the scene's data; a climate sets air_rh from the weather",
         (_G, "src.voxel.sim:World.wet", None)),
    _row("wet.idle_flags", "crutch", "C3",
         "The idle gate reads flags (has_ice, store generation); a scene that puts ICE by hand after enable_wet must "
         "call w.wet_refresh().",
         "ICE cells are tracked by the matter stamp (SCALE)",
         None),
    _row("wet.heat_capacity_wrapper", "crutch", "C3",
         "Vapour, mist and ice grams add heat capacity in WetMixin.heat_capacity, not in fields._cap_of.",
         "fields._cap_of (SCALE, the list path) reads the store; the wrapper goes",
         None),
    _row("wet.seep_no_heat", "crutch", "C3",
         "Seep moves vapour grams but not their sensible heat.",
         "gas_flow is the default or seep carries E",
         None),
    _row("wet.edge_relax", "estimate", "C3",
         "Open edges relax vapour to the background at 0.5 per second (a cheap estimate of an open sky).",
         "C10's sealed-room sink or the airsink_gas adapter gives the room edge a real law",
         None),
    _row("wet.b0_fixed", "crutch", "C3",
         "The background vapour density is fixed at enable_wet from rh and t_c; the outdoor shift is not wired.",
         "T8's publish calls GasStore.shift_outdoor",
         None),
    _row("wet.mixed_cells_skip", "crutch", "C3",
         "Mixed cells (brine) skip the surface law and the ice law (D6).",
         "T4's mixed evaporation and freezing are migrated (M1 of C12 is the first)",
         None),
    _row("wet.gasflow_hooks", "permanent", "C3",
         "gasflow.py _gf_extra_* hooks carry nothing by default.",
         "never: a general hook that lets a store ride the flow",
         None),
    _row("wet.reaction_rows_unwired", "crutch", "C3",
         "rxn.py rows are pure; no burn-product line in heat.py and no adapter for the 8 REACTIONS (made-books unused).",
         "the reaction and sorption row runner exists (M3 of C12): it unlocks tanning, lime, germination and ferment",
         None),
    _row("layers.surface_laws", "gate", "C4",
         "Soot falls out of smoke onto faces and darkens them; default on for every world, set False for the old world.",
         "every lane that reads layers has landed and no scene needs the old world",
         (_G, "src.voxel.sim:World.surface_laws", True)),
    _row("layers.soot_no_thermal", "crutch", "C4",
         "A soot film has no resistance and no heat capacity in heat, and its sensible heat is not booked (3e-5 of the host).",
         "an ash, wax or water layer lane wires layers.thermal into the skin path",
         None),
    _row("layers.dep_every", "estimate", "C4",
         "DEP_EVERY 40 and EPS_EVERY 40: a pass every second (ratio 0.986 against every 8 ticks).",
         "a scene shows a dependence on sub-second soot",
         None),
    _row("layers.soot_true", "estimate", "C4",
         "SOOT_TRUE 0.025 g of soot per smoke unit (u); also Brownian 1e-5 m/s and a 0.5 um particle (u).",
         "gas species for true soot exist (factor 1)",
         None),
    _row("layers.temp_clip", "crutch", "C4",
         "The pass clips temperature to [-150, 5000] C (a stray negative E made a NaN).",
         "cell energy cannot go below zero heat capacity",
         None),
    _row("layers.thin_deposit", "crutch", "C4",
         "A deposit of 1e-8 g still makes a layer on a face (6879 faces for 1.2e-4 g in flee).",
         "a minimum mass per face or a slow-tier prune is added",
         None),
    _row("smoke.smoke_opacity", "gate", "C4b",
         "Smoke blinds sight 39 times more per gram (soot extinction); default on, set w.smoke_opacity False for clear smoke.",
         "no scene needs the clear-smoke world",
         (_G, "src.voxel.sim:World.smoke_opacity", True)),
    _row("smoke.sigma_e", "estimate", "C4b",
         "SIGMA_E 8.7 m2/g and SOOT_TRUE are (u) in smokeopt and layer_data.",
         "tests/data/sourced_numbers.json marks both as sourced",
         None),
    _row("smoke.cough_old", "crutch", "C7",
         "Cough, CO and Purser dose still use SMOKE_EXT 5.6 per g/L in physiology; the soot hooks are unwired.",
         "smokeopt.Hooks are registered with numbers retuned to the soot scalar",
         None),
    _row("smoke.see_tau", "estimate", "C4b",
         "Smoke is cut at SEE_SMOKE_TAU 3 (Jin reflecting C 2-4, middle); SEE_TAU 2 stays for solids. Flame uses 8 via SEE_LIT_SMOKE 3/8.",
         "the user picks another point in Jin's range, or a per-fuel/irritant law",
         None),
    _row("smoke.belief_ignores_smoke", "crutch", "C4b",
         "The belief sweep (_look_around) ignores smoke: a man in a smoke layer still knows the room by geometry.",
         "a later perception lane makes the belief sweep read the optical depth",
         None),
    _row("rec.records", "gate", "C5",
         "Records exist only where a scene or act declares one (w.records None otherwise); rec_laws turns the "
         "heat readout off for a world that wants the books only.",
         "every scene that needs thin-solid heat declares its records and the heat is delegated (slice 2b)",
         (_G, "src.voxel.sim:World.records", None)),
    _row("rec.rec_laws", "gate", "C5",
         "World.rec_laws (class attribute, True) runs the heat readout of a record; False keeps the books only.",
         "slice 2b delegation lands and the readout is the host's own heat",
         (_G, "src.voxel.sim:World.rec_laws", True)),
    _row("rec.lattice_cp", "crutch", "C5",
         "Records.lattice_cp: a record uses the lattice's constant c, not the record table's cp(T).",
         "T9's lattice cp(T) table lands and the record table is built from it",
         None),
    _row("rec.readout_only", "crutch", "C5",
         "Record heat is a readout; reconcile gives it back from the host every tick (the level is off by 4 on a 2 mm blade).",
         "slice 2b: the record's face flux is the host's, applied once (a heat.py kernel change, _face_heat and its twin)",
         None),
    _row("rec.port_bands", "estimate", "C5",
         "PORT_EVERY 4 ticks, PORT_H_BAND 5 percent, PORT_T_BAND 1 K.",
         "slice 2b: ports come from the host's faces",
         None),
    _row("rec.spmap", "crutch", "C5",
         "_SPMAP covers iron, wood and glass only; other host materials are not counted in a record.",
         "T1's pool replaces the 10-species stand-in table",
         None),
    _row("rec.cure_stand_in", "crutch", "C5",
         "Composite cure uses rh 1.0 and the first member's mean temperature; one JOIN kind per joint; loads come from the caller.",
         "T6 humidity feeds cure and hands.py passes the hand force to composite_check",
         None),
    _row("rec.books_stand_in", "crutch", "C5",
         "ledgers.total_mass does not add glue grams (stand-in species are not lattice materials).",
         "T1 maps record species to material ids",
         None),
    _row("injury.burn_shock", "crutch", "C7",
         "Burn shock reads the old p['burn'] integral as a plasma-loss equivalent (_SHOCK_X and _SHOCK_Y calibrated to 0.2 and 0.45).",
         "skinburn faces replace p['burn'] (needs the heat law's face flux) and %TBSA drives shock",
         None),
    _row("injury.fractures_unread", "crutch", "C7",
         "p['fractures'] is written by a landing but nothing reads it.",
         "the support law (bears, _footing) reads the bone state",
         None),
    _row("injury.landing_mode", "crutch", "C7",
         "The landing mode comes from the cell principal axis (injury.fall_mode), not per-bone contact.",
         "the landing code passes contact bones",
         None),
    _row("injury.head_blow_constants", "estimate", "C7",
         "A head blow uses a 1 kg effective weapon mass and a 500 kN/m contact spring (u); vagal median 0.6, sigma 0.6, "
         "flow drop 70 percent, fade 30 s, rise 4 s, fall 10 s, startle dial 0.5 to 2.0, withdraw dial 0.8 to 1.25 (u).",
         "_blow knows the weapon mass and the dials are drawn from a measured distribution",
         None),
    _row("injury.pain_seen_adopt", "crutch", "C7",
         "_pain_seen adopts a body's starting hurt and burn at tick <= 1 so harvest's initial bruises are not a new pain.",
         "scenes declare starting injuries as history",
         None),
    _row("injury.slowtwin_exclude", "crutch", "C10",
         "The breath twin is not applicable while an injury clock runs (ko_until, organ_dead, vagal, pain_acute, pool_left_ml).",
         "a twin of the injury clocks exists",
         None),
    _row("injury.bleed_grip_unwired", "crutch", "C7",
         "Bleeding by vessel flow and grip_N at the 11 strength sites are built but not wired (the wound azimuth and "
         "the limb fatigue are not recorded).",
         "the wound dict carries its azimuth and along-length and the effort law writes limb fatigue",
         None),
    _row("injury.hold_hot_tag", "crutch", "C7",
         "will.py does not set the hold_hot tag for a chosen hold of a hot thing, so the reflex drops it.",
         "the job layer sets the tag when a take or hold targets a believed-hot thing",
         None),
    _row("edge.edge_wound", "gate", "C8b",
         "'edge' (wear and roll) and 'wound' (the tip-area skin line) are names in mat_laws, default on.",
         "same condition as mat.mat_laws",
         (_S, "matlaws", "ALL_LAWS")),
    _row("edge.no_eid_state", "crutch", "C8b",
         "An edge no thing owns keeps its state on the flying body; after landing only the area survives and the next hit "
         "re-derives the geometry (theta 30 deg, r0 0.1 mm).",
         "tools carry an eid (scenes build them with thing()) or _put carries the edge state",
         None),
    _row("edge.temper_one", "crutch", "C8b",
         "H_edge = the record's hardness x temper, with temper 1 (no heat-history law).",
         "a heat-history law sets things[eid]['temper'] (M4 of C12, decision D1)",
         None),
    _row("edge.constants", "estimate", "C8b",
         "R_MAX 2 cm, MU_FLANK 0.4, SOFT_H 1e5 Pa, MAX_CHUNKS 4 per blow, the blow's whole joules for wear and roll (u); "
         "the skin constant is for theta 30 deg whatever the edge.",
         "sourced numbers replace them and the thing record carries theta",
         None),
    _row("edge.filings_wait", "crutch", "C8b",
         "Filings wait in the tool when no air cell is next to the contact.",
         "T1 gives a contaminant place on the target cell",
         None),
    _row("stairs.step_range", "gate", "C9",
         "The legs step 30 cm up and down (PathingMixin.step_up_m and step_down_m, were 0); a person's own p['step_up_m'] wins; "
         "set a world's to 0.0 for the old world.",
         "materials.BODY carries step_up_m and step_down_m and the class attributes go back to None",
         (_G, "src.voxel.sim:World.step_up_m", 0.3)),
    _row("stairs.fit_grid_union", "crutch", "C9",
         "_fit_grid returns the flat plane plus the near-level columns (never fewer than the flat rule).",
         "_places and _leap_targets read the level planes themselves (T11 7.2)",
         None),
    _row("stairs.look_fz", "crutch", "C9",
         "_look_levels writes fz only for new or flipped columns and never p['free'].",
         "senses._look_around exposes its reach plane and writes fz and free together",
         None),
    _row("stairs.levels_trust_flat", "crutch", "C9",
         "_levels_for trusts the flat belief at the soles' height and the body's own columns.",
         "p['free'] is retired and fz is the one belief",
         None),
    _row("stairs.lunge", "crutch", "C9",
         "A step with a rise reaches up to 4 cells (25 cm) along, balance-tested: a rigid body needs it not to be thrown at a kerb.",
         "the legs are modelled (a foot goes up and the weight over it); six-riser flights still tip a rigid person",
         None),
    _row("stairs.tumble_unwired", "crutch", "C9",
         "stairs.tumble is not wired to any fall.",
         "the footing or landing event carries a velocity on a flight (wave 15)",
         None),
    _row("stairs.mechanism_rows", "crutch", "C9",
         "No menu rows for doors and mechanisms (operate, kick, key, bar, pick); mechlaw.kick and the 215 J numbers are tested only.",
         "events.py verbs, policy ACTS, src/mind verbs.yaml and will.py _menu rows land (a lane of its own)",
         None),
    _row("contact.contact_laws", "gate", "C13",
         "Feet and hands move matter between faces and the slip law asks for friction; default on, False is the old world.",
         "C11 decides the default for every world and the False path is no longer wanted",
         (_G, "src.voxel.sim:World.contact_laws", True)),
    _row("contact.walking_rcof", "estimate", "C13",
         "Walking needs a friction of 0.20 x speed ratio (u).",
         "a measured gait table (heel-strike force trace) in the body-model lane",
         None),
    _row("contact.slip_speed", "crutch", "C13",
         "A slip is a speed over the feet handed to push_person (the rigid body has no foot).",
         "the legs are modelled (a foot that slides with a kinetic mu)",
         None),
    _row("contact.touches_verb", "crutch", "C13",
         "The 'touches' verb carries footfall, grasp, press and slip in its result.",
         "events.py gets real footfall and grasp verbs",
         None),
    _row("contact.pivot_dry_040", "crutch", "C13",
         "The toppling pivot keeps the dry value FRICTION 0.40; only a film lowers it (the ledge scene depends on it).",
         "the C8a friction numbers are unified for the toppling pivot and the ledge scene is recalibrated",
         None),
    _row("contact.film_no_ice", "crutch", "C13",
         "floor_mu calls film_of without a temperature, so ice is not seen.",
         "floor_mu reads the cell temperature (water at or below 0 C is ice)",
         None),
    _row("contact.adj_ignored", "crutch", "C13",
         "Contact ignores the heat adjustments the pool returns (touch passes ambient).",
         "a hot layer moved by a foot or a hand books its adj into the host heat",
         None),
    _row("contact.reflex_foothold", "crutch", "C13",
         "The balance reflex places its step without reading the floor friction, so a man on oil is never thrown.",
         "reflex.py reads contact.floor_mu for the foothold and slips again",
         None),
    _row("slow.airsink", "gate", "C10",
         "Tier-1 jumps (approx=True on advance) run only where w.airsink is a RegionSink; a carried body without one is a named blocker.",
         "the world owns a room-air store (the GAS vector for O2 and CO2 plus the vapour adapter) so a default sink exists",
         (_S, "quiet", "airsink")),
    _row("slow.exclude_rows", "crutch", "C10",
         "quiet.EXCLUDE rows for what the body twin carries (persons[*].heat, skin, airsink, _bh_inject, _bh_runtime).",
         "the body twin keys are declared by Twin.keys and the digest reads them (a quiet.py refactor, wave 14)",
         None),
    _row("slow.region_o2_owned", "crutch", "C10",
         "digest and gate_air treat o2 as owned inside the region only when a RegionSink is present.",
         "the RegionSink crutch above is deleted",
         None),
    _row("slow.constants", "estimate", "C10",
         "CHUNK_S 300 s, CORE_BAND (30, 40) C, FLUSH_TICKS 40, V2 5 percent are calibration constants.",
         "a measured accuracy-against-speed choice by the user",
         None),
    _row("slow.shut_room_radiation", "crutch", "C10",
         "The tick law radiates the surfaces of a shut room to the world (91 percent of a sleeper's heat leaves as shed); "
         "the jump keeps it in the room.",
         "heat._env_radiation lets the surfaces of a sink's box radiate onto each other",
         None),
    _row("slow.co2_vapour_numbers", "crutch", "C10",
         "CO2 and vapour have no lattice field; the sink holds them as numbers.",
         "gas species for CO2 and vapour exist in the lattice and a door vents them",
         None),
    _row("craft.missing_laws", "crutch", "C12",
         "Crafts lean on laws that do not exist yet: M1 brine freezes at its own point, M2 liquid expelled from a mushy solid, "
         "M3 reaction and sorption row runner, M4 heat history, M5 settling and separation, M6 farming. Three are strict xfails.",
         "each law lands: M3 first (it unlocks tanning, lime, germination and ferment), M4 is the gate for real smithing",
         None),
    # ── C11 ───────────────────────────────────────────────────────────────────
    _row("golden.new_proof_cases", "crutch", "C11",
         "The ten w13_* cases in tools/golden.py are recorded on the finished tree only; a base older than wave 13 leaves them out.",
         "base22 is taken with them recorded; then Absent in golden.py is dead code",
         (_S, "../../tools/golden", "class Absent")),
    _row("gas.momentum", "gate", "D14a", "World.gas_momentum carries the gas flow's face speeds by the flow itself (gasmom.py), "
         "with GAS_DRAG_MOM and momentum mixing; off by default, only inside gas_flow.",
         "gas_flow is on by default in every scene and all of them have passed with momentum on",
         (_S, "gasflow", "GAS_MOMENTUM = False")),
    _row("gas.drag_mom", "crutch", "D14a", "GAS_DRAG_MOM 0.1/s: a linear drag that stands for wall and canopy friction and the mixing no cell sees "
         "(it is 0.75/s without momentum).",
         "a wall-friction (shear stress at solids) law gives the flow its friction and GAS_DRAG_MOM is 0",
         (_S, "gasflow", "GAS_DRAG_MOM")),
    _row("gas.mix_mom", "estimate", "D14a", "Momentum mixes with alpha = GAS_ENTRAIN * GAS_MIX_MOM = 0.03 (the plume constant of the heat) (u).",
         "a turbulence closure (Smagorinsky or k-epsilon) replaces the entrainment constant",
         (_S, "gasflow", "GAS_MIX_MOM")),
    _row("push.cd_plate", "estimate", "D14a", "Every surface patch drags like a flat plate across the stream, Cd 1.2 (Hoerner); real shapes differ.",
         "a patch carries a Cd from its shape (sphere 0.47, cylinder 1.0) set by the thing that owns it",
         (_S, "push", "CD_PLATE")),
    _row("push.half_cell", "estimate", "D14a", "A patch's contacts are put half a cell (10 cm) either side of it: a plate thinner than a cell is a dipole across a cell face.",
         "a cut-cell treatment puts the displacement on the faces the patch really crosses",
         (_S, "push", "0.5 * H")),
    _row("push.work_unbooked", "crutch", "D14a", "The work the surfaces do on the gas is in gas_flow_stats (push_work_J), not in the books: the flow's kinetic energy is not on them either.",
         "ledgers.BOOKS has a gas-kinetic term, written by gasflow and push, that heat and drag dissipation close",
         (_S, "push", "push_work")),
    _row("gas.eddy_heat", "crutch", "D14a", "heat.py gas|gas conductivity grows with (T-20)/EDDY: a stand-in for the convection the flow carries; it is zero where gas_momentum is on (heat.py, slownet.py, slowforms.py).",
         "gas_momentum is on in every scene and the EDDY constant is deleted",
         (_S, "heat", "EDDY")),
    _row("wood.species", "gate", "B14a", "per-cell wood species and grain arrays (fluids.FluidsMixin.wood_sp, wood_grain), None until set_wood",
         "never: a lazy array is the permanent form of a per-cell property",
         (_G, "src.voxel.sim:World.wood_sp", None)),
    _row("wood.sorptivity_u", "estimate", "B14a", "S_along and S_across of beech, ash, balsa and oak are (u); pine is the Wood Handbook band middle (across u)",
         "a per-species capillary table (Aw by direction) is opened and the rows replaced from it",
         None),
    _row("wood.pools", "gate", "R3", "bound-water array wood_bound (woodwater.WoodWaterMixin), None until set_wood(pools=True)",
         "never: a lazy per-cell property",
         (_G, "src.voxel.sim:World.wood_bound", None)),
    _row("wood.bound_rates", "estimate", "R3", "GRAIN_RATIO, D_ACROSS, EA, HS0, HOT_TAU_S are (u); one grain-axis only",
         "a per-species bound-diffusion table by direction is read from sources",
         None),
    _row("smoke.smoke_light", "gate", "R1", "Smoke sight depends on ambient light from the climate's sun (Jin 2025); default on; no climate = reference light, old law. Set w.smoke_light False for the old law.",
         "an indoor light model (roofs and windows, with the summed light of flames and lamps) replaces the outdoor sun lux, so a room reads its own illuminance",
         (_G, "src.voxel.sim:World.smoke_light", True)),
    _row("smoke.see_smoke_tau", "estimate", "H14a", "materials.SEE_SMOKE_TAU 3.0: smoke is lost at optical depth 3, Jin's reflecting-object C (range 2-4, middle) (u).",
         "the user picks another point in Jin's range, or a per-fuel/irritant law",
         (_S, "materials", "SEE_SMOKE_TAU")),
    # ── 14b: bodies and choices ───────────────────────────────────────────────
    _row("mechlaw.exertion_from_body", "gate", "G14b",
         "ON for every world (14bF, W15-Q8 (a)): a kick, punch, throw, lift and haul are drawn from the body (muscle, segment mass, technique, training, fatigue, injury, a per person spread around 115 J) and paid from the body's books; False is the old 100.8 J flat kick and the flat 400 N strength.",
         "no test or scene wants the old flat numbers and the attribute and the False paths are deleted",
         (_G, "src.voxel.sim:World.exertion_from_body", True)),
    _row("exertion.numbers", "estimate", "G14b",
         "Stroke lengths of punch, shove, throw, mass fractions, technique curve, training gain 0.35, z sigma 0.22 are (u).",
         "a strike rig study or a measured population table replaces them",
         None),
    _row("mechlaw.KICK_M_KG_V", "crutch", "G14b",
         "The old kick constants 5.6 kg x 6 m/s (100.8 J) stay for the gate-off path.",
         "mechlaw.exertion_from_body is deleted and the constants go with the False path",
         None),
    _row("will.menu_batch", "gate", "F14b",
         "ON at MENU_BATCH = 4 for every world (14bF): a batchable policy (the argmax voxel table, exact against the whole menu in F's measure) sees a long menu in batches of B and the winners merged; a mind or a script policy keeps the MENU_CAP cut. None is the old cut.",
         "every policy of a world is batchable or a model policy sits in the seam, and the attribute is deleted",
         (_G, "src.voxel.will:WillMixin.menu_batch", 4)),
    _row("loop.MindConfig.batch", "gate", "F14b",
         "MindConfig.batch (src/mind/loop.py): a number B makes the flat NPC menu a tournament of batches of B instead of the pre-rank cut to MENU_MAX; None is the old cut.",
         "the tournament is the default flat menu and the pre-rank stays only for cap-only callers (it stays None in 14bF: a drawing chooser's tournament is sharper than one softmax, total variation 0.06 to 0.17, so flipping it changes the rules NPC)",
         None),
    _row("will.MENU_CAP", "crutch", "F14b",
         "MENU_CAP 7 and the categories-first cut in will._menu stay for the policies that are not batchable (scripted scenes, the mind link).",
         "every policy is batchable or a model policy sits in the seam, and recall_check is retired",
         None),
    _row("menu.BATCH_4", "estimate", "F14b",
         "BATCH = 4 is between the measured chooser sizes 4, 16 and 128 (u); tournament probabilities come from the realised path (u).",
         "a model policy is run at 3, 4, 5, 6, 8 through the gpu lock and an exact or bounded probability is wanted",
         None),
    _row("tourney.realised_path", "estimate", "F14b",
         "The probability of a tournament pick is read from the path that was realised (the batches it went through), not summed over every path (u).",
         "an exact enumeration or a cheap bound of the merge is wanted",
         None),
    _row("contact.step_friction", "crutch", "H14b",
         "A recovery step either grips or slides out (friction needed d / h against floor_mu): a binary test.",
         "the balance model has the ankle torque and counter-rotation mechanisms of lane E, which replace it with a force budget",
         (_S, "contact", "step_friction_needed")),
    _row("faintrates.fresh_band", "crutch", "H14b",
         "FRESH_BAND 0.95: air within 5 percent of fresh reads as before because the o2 field is float32.",
         "the o2 field is float64 (a fresh room is exactly 1.0), then the curve applies at every fraction",
         (_S, "faintrates", "FRESH_BAND")),
    # ── 14b: body-on-body contact (lane E) and the finish (14bF) ──────────────
    _row("bodycontact.body_contact", "gate", "E14b",
         "ON for every world (14bF, W15-Q8 (a)): body-on-body contact is one law (shove, tackle, grapple, a blunt blow, a thrown or falling body on a person) and the balance reflex has the three Hof mechanisms (ankle, counter-rotation, a hand on something that holds); False is the old hull test with only step and grab, and a blow is joules into flesh.",
         "no test or scene wants the old hull test and joules-only blow and the attribute and the False paths are deleted",
         (_G, "src.voxel.sim:World.body_contact", True)),
    _row("bodycontact.numbers", "estimate", "E14b",
         "Flesh restitution 0.20, body friction 0.5, contact stiffness 60 kN/m, contact area 0.03 m2, the flywheel inertias, accelerations and speeds, ankle 1.7 Nm/kg, arm push 0.4 BW, hold friction 0.5, hand closing speed 0.5 m/s are (u).",
         "a collision rig study or a measured recovery study replaces them",
         None),
    _row("motion._v_push", "crutch", "E14b",
         "motion._v is a push speed over the feet, not a rigid body in motion; the contact law hands it momentum.",
         "a person has a real velocity and the walk is a law of it",
         (_S, "motion", "def push_person")),
    _row("bodycontact._reaction_cap", "crutch", "E14b",
         "bodycontact._reaction_cap scans along one axis for the thing the hand reaches.",
         "the hand's reach is read from the arm's own cells (hands.py and skeleton)",
         (_S, "bodycontact", "def _reaction_cap")),
    _row("exertion.reserve_u", "estimate", "14bF",
         "A body with no heat record pays a strike from its limb's reserve above critical power: W' 250 J/kg of the whole body, recovery tau 300 s (fatigue.py, (u)), and the strike's metabolic cost J / 0.22.",
         "every person has a heat record (body heat is on everywhere) and the stores pay",
         (_S, "exertion", "def pay_reserve")),
    _row("exertion.swing_torque", "estimate", "14bF",
         "A swing or a throw's muscle torque is the stock arm's 60 Nm (materials.BODY arm_Nm) times the person's strike energy over the stock adult's (exertion.torque_scale); the arm's torque is not yet its own table of muscles.",
         "the arm torque is read from the arm's cross-section (skeleton torque_of) and the person's own muscle row",
         (_S, "exertion", "def torque_scale")),
)


def _gate_attrs_of_world():
    """The names in World's class attributes that look like gates."""
    from . import sim
    out = set()
    for cls in sim.World.__mro__:
        for name, val in vars(cls).items():
            if re.search(r"(_laws|_opacity)$", name) and isinstance(val, (bool, frozenset)):
                out.add(name)
    return out


def _probe(row):
    """None when the probe holds, else a short problem."""
    pr = row["probe"]
    if pr is None:
        return None
    if pr[0] == _G:
        mod, _, path = pr[1].partition(":")
        obj = importlib.import_module(mod)
        for part in path.split("."):
            if not hasattr(obj, part):
                return f"{row['id']}: probe {pr[1]} finds no {part}"
            obj = getattr(obj, part)
        if obj != pr[2] or type(obj) is not type(pr[2]):
            return f"{row['id']}: {pr[1]} is {obj!r}, the ledger says {pr[2]!r}"
        return None
    if pr[0] == _S:
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, pr[1] + ".py"))
        if not os.path.exists(path):
            return f"{row['id']}: source file {pr[1]} is missing"
        if pr[2] not in open(path).read():
            return f"{row['id']}: {pr[1]}.py no longer holds {pr[2]!r}"
        return None
    return f"{row['id']}: unknown probe kind {pr[0]!r}"


def audit(rows=None):
    """The list of problems in the ledger. Empty means it passes."""
    rows = ROWS if rows is None else rows
    bad, seen = [], set()
    for r in rows:
        i = r.get("id", "?")
        if i in seen:
            bad.append(f"{i}: duplicate id")
        seen.add(i)
        if r.get("kind") not in KINDS:
            bad.append(f"{i}: kind {r.get('kind')!r} is not one of {KINDS}")
        for k in ("owner", "why", "deleted_when"):
            if not str(r.get(k, "")).strip():
                bad.append(f"{i}: no {k}")
        dw = str(r.get("deleted_when", "")).strip()
        if dw and len(dw) < 20:
            bad.append(f"{i}: deleted_when says nothing ({dw!r})")
        if dw.lower().startswith("never"):
            if r.get("kind") not in ("gate", "permanent"):
                bad.append(f"{i}: a {r.get('kind')} may not stay for ever")
            elif not dw.lower().startswith("never:") or len(dw) < 15:
                bad.append(f"{i}: 'never' needs a reason after the colon")
        if r.get("probe") is not None:
            try:
                p = _probe(r)
            except Exception as e:          # an import that fails is a ledger failure, not a crash
                p = f"{i}: probe raised {type(e).__name__}: {e}"
            if p:
                bad.append(p)
    ids = {r.get("id") for r in rows}
    for c in heatmix.CRUTCHES:
        if f"mix.skip.{c['law']}" not in ids:
            bad.append(f"heatmix.CRUTCHES row {c['law']} is not in the ledger")
        for k in ("why", "deleted_when", "owner"):
            if not str(c.get(k, "")).strip():
                bad.append(f"heatmix.CRUTCHES {c['law']}: no {k}")
    for name in sorted(_gate_attrs_of_world()):
        if not any(name in str(r.get("probe")) or name in r["id"] for r in rows):
            bad.append(f"World gate {name} has no ledger row")
    return bad


def by_owner(rows=None):
    """{owner: [ids]} for the report."""
    out = {}
    for r in (ROWS if rows is None else rows):
        out.setdefault(r["owner"], []).append(r["id"])
    return out


def counts(rows=None):
    out = {}
    for r in (ROWS if rows is None else rows):
        out[r["kind"]] = out.get(r["kind"], 0) + 1
    return out
