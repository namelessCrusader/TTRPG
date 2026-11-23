"""WAVE 13 MANIFEST (lane A-X): every new file, who owns it, and the books.

One place that says (a) which phase A lane creates which file, (b) which topic
owns each new src/voxel module, and (c) the BOOKS: every ledger term of wave 13
with its ONE owner topic and the modules allowed to write it. The audit
(tools/w13_audit.py) and the cross tests (tests/test_w13_cross.py) read this;
nothing in src/ imports it.

Plain data and four small helpers. Standard library only.

Tags (plan section 9): a test is
  [I] independent: conservation, exactness, closed form, arithmetic identity,
      or a number A-S marked sourced
  [U] independent but its physical number is not yet sourced   (test_u_*)
  [E] echo: the constant was fitted to that number              (test_echo_*)
  [C] code-only: the code against its own table                 (test_codeonly_*)
Only [I] lines gate a scene.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- the lanes (from the phase A plan; new files and tests per lane) ----------
# (lane id, topic, batch, new files, tests)
_LANES = (
    ("A-O", "G0", 0,
     ["tools/matrec_extract.py", "tests/data/matrec_oracle_base18.json",
      "tests/data/matrec_oracle_base19.json", "tests/test_oracle_format.py"],
     ["tests/test_oracle_format.py"]),
    ("A-S", "G0", 0,
     ["tests/data/sourced_numbers.json", "tools/w13_sourced.py", "tests/test_sourced_numbers.py"],
     ["tests/test_sourced_numbers.py"]),
    ("A-M", "G0", 0,
     ["tools/w13_floor_probe.py", "tools/w13_route_bench.py", "tools/w13_scan_steps.py",
      "tools/w13_wake_probe.py", "tools/w13_walk_leak.py", "tools/w13_f64_conduct.py",
      "tests/test_w13_probes_smoke.py"],
     ["tests/test_w13_probes_smoke.py"]),
    ("A-G10", "G10", 0,
     ["src/voxel/bodysheet.py", "tests/test_bodysheet.py"], ["tests/test_bodysheet.py"]),
    ("A-T1a", "T1", 1,
     ["src/voxel/species_seed.py", "src/voxel/species.py", "src/voxel/legacy_tables.py",
      "tests/test_species.py", "tests/test_legacy_tables.py", "tests/data/species_ids.json"],
     ["tests/test_species.py", "tests/test_legacy_tables.py"]),
    ("A-T1b", "T1", 2,
     ["src/voxel/mixtable.py", "src/voxel/mixrules.py", "src/voxel/partition.py",
      "tests/test_mixtable.py", "tests/test_mixrules.py", "tests/test_partition.py"],
     ["tests/test_mixtable.py", "tests/test_mixrules.py", "tests/test_partition.py"]),
    ("A-T2a", "T2", 2,
     ["src/voxel/record.py", "src/voxel/record_data.py", "src/voxel/record_view.py",
      "src/voxel/thinheat.py", "src/voxel/handover.py", "tests/test_records_core.py",
      "tests/test_records_heat.py", "tests/test_records_handover.py",
      "tests/test_records_thinheat_solver.py"],
     ["tests/test_records_core.py", "tests/test_records_heat.py", "tests/test_records_handover.py",
      "tests/test_records_thinheat_solver.py"]),
    ("A-T2b", "T2", 4,
     ["src/voxel/breaking.py", "src/voxel/joinery.py", "src/voxel/lots.py", "src/voxel/vessel.py",
      "tests/test_records_breaking.py", "tests/test_records_joinery.py",
      "tests/test_records_lots_vessels.py"],
     ["tests/test_records_breaking.py", "tests/test_records_joinery.py",
      "tests/test_records_lots_vessels.py"]),
    ("A-T3", "T3", 2,
     ["src/voxel/layer_data.py", "src/voxel/layers.py", "src/voxel/layer_laws.py",
      "src/voxel/marks.py", "src/voxel/smokeopt.py", "tests/layers_fake.py",
      "tests/test_layers_store.py", "tests/test_layers_flow.py", "tests/test_layers_contact.py",
      "tests/test_layers_marks.py", "tests/test_layers_readouts.py", "tests/test_smokeopt.py",
      "tests/test_layers_guards.py", "tests/test_layers_inputs.py"],
     ["tests/test_layers_store.py", "tests/test_layers_flow.py", "tests/test_layers_contact.py",
      "tests/test_layers_marks.py", "tests/test_layers_readouts.py", "tests/test_smokeopt.py",
      "tests/test_layers_guards.py", "tests/test_layers_inputs.py"]),
    ("A-T4", "T4", 1,
     ["src/voxel/mixdata.py", "src/voxel/mixwater.py", "src/voxel/mixvle.py",
      "src/voxel/mixphase.py", "src/voxel/mixsol.py", "src/voxel/mixsolid.py",
      "src/voxel/mixtures.py", "tests/test_mixtures_data.py", "tests/test_mixtures_water.py",
      "tests/test_mixtures_phase.py", "tests/test_mixtures_solutions.py",
      "tests/test_mixtures_solid.py"],
     ["tests/test_mixtures_data.py", "tests/test_mixtures_water.py", "tests/test_mixtures_phase.py",
      "tests/test_mixtures_solutions.py", "tests/test_mixtures_solid.py"]),
    ("A-T5", "T5", 3,
     ["src/voxel/slowforms.py", "src/voxel/slowtime.py", "src/voxel/slowtwins.py",
      "src/voxel/slownet.py", "src/voxel/quiet.py", "tools/bench_slow.py",
      "tests/test_slowforms.py", "tests/test_slowtime.py", "tests/test_slowtwins.py",
      "tests/test_slownet.py", "tests/test_quiet.py"],
     ["tests/test_slowforms.py", "tests/test_slowtime.py", "tests/test_slowtwins.py",
      "tests/test_slownet.py", "tests/test_quiet.py"]),
    ("A-T6", "T6", 3,
     ["src/voxel/matterdata.py", "src/voxel/_standin.py", "src/voxel/psychro.py",
      "src/voxel/gasstore.py", "src/voxel/solution.py", "src/voxel/transport.py",
      "src/voxel/icephase.py", "src/voxel/rxn.py", "tests/test_t6_psychro.py",
      "tests/test_t6_transport.py", "tests/test_t6_gas_solutes.py", "tests/test_t6_icephase.py",
      "tests/test_t6_rxn.py"],
     ["tests/test_t6_psychro.py", "tests/test_t6_transport.py", "tests/test_t6_gas_solutes.py",
      "tests/test_t6_icephase.py", "tests/test_t6_rxn.py"]),
    ("A-T7a", "T7", 3,
     ["src/voxel/tissue.py", "src/voxel/bodyair.py", "src/voxel/thermoreg.py",
      "src/voxel/clothing.py", "src/voxel/airsink.py", "tests/test_tissue.py",
      "tests/test_bodyair_thermoreg.py", "tests/test_clothing.py"],
     ["tests/test_tissue.py", "tests/test_bodyair_thermoreg.py", "tests/test_clothing.py"]),
    ("A-T7b", "T7", 4,
     ["src/voxel/digestion.py", "src/voxel/energetics.py", "src/voxel/bodynodes.py",
      "src/voxel/bodyheat.py", "src/voxel/scenes_bodyheat.py", "tools/jos3_oracle.py",
      "tests/data/human_cases.json", "tests/data/jos3_cases.json", "tests/test_digestion_energetics.py",
      "tests/test_bodynodes.py", "tests/test_bodyheat_world.py", "tests/test_bodyheat_scenes.py"],
     ["tests/test_digestion_energetics.py", "tests/test_bodynodes.py",
      "tests/test_bodyheat_world.py", "tests/test_bodyheat_scenes.py"]),
    ("A-T8", "T8", 1,
     ["src/voxel/climate.py", "src/voxel/outside.py", "src/voxel/outside_world.py",
      "src/voxel/scenes_cold.py", "tests/ref_cold_lattice.py", "tests/test_climate.py",
      "tests/test_outside.py", "tests/test_cold_ref.py", "tests/test_cold_world.py"],
     ["tests/test_climate.py", "tests/test_outside.py", "tests/test_cold_ref.py",
      "tests/test_cold_world.py"]),
    ("A-T9a", "T9", 1,
     ["src/voxel/matrec.py", "src/voxel/matfric.py", "src/voxel/matdamage.py",
      "src/voxel/matedge.py", "tests/test_matrec.py", "tests/test_matfric.py",
      "tests/test_matdamage.py", "tests/test_matedge.py"],
     ["tests/test_matrec.py", "tests/test_matfric.py", "tests/test_matdamage.py",
      "tests/test_matedge.py"]),
    ("A-T9b", "T9", 4,
     ["src/voxel/matbeam.py", "src/voxel/matstress.py", "tools/members_survey.py",
      "tests/test_matbeam.py", "tests/test_matbeam_world.py", "tests/test_matstress.py"],
     ["tests/test_matbeam.py", "tests/test_matbeam_world.py", "tests/test_matstress.py"]),
    ("A-T10a", "T10", 3,
     ["src/voxel/skinburn.py", "src/voxel/fatigue.py", "src/voxel/signs.py",
      "tests/test_skinburn.py", "tests/test_fatigue.py", "tests/test_signs.py"],
     ["tests/test_skinburn.py", "tests/test_fatigue.py", "tests/test_signs.py"]),
    ("A-T10b", "T10", 4,
     ["src/voxel/anatomy.py", "src/voxel/impact.py", "src/voxel/bonestate.py",
      "src/voxel/vasculature.py", "tests/test_anatomy.py", "tests/test_impact.py",
      "tests/test_bonestate.py", "tests/test_vasculature.py"],
     ["tests/test_anatomy.py", "tests/test_impact.py", "tests/test_bonestate.py",
      "tests/test_vasculature.py"]),
    ("A-T11a", "T11", 2,
     ["src/voxel/fasteners.py", "src/voxel/mechanisms.py", "src/voxel/locks.py",
      "tests/test_fasteners.py", "tests/test_mechanisms.py", "tests/test_locks.py"],
     ["tests/test_fasteners.py", "tests/test_mechanisms.py", "tests/test_locks.py"]),
    ("A-T11b", "T11", 5,
     ["src/voxel/routing.py", "src/voxel/stairs.py", "src/voxel/mechlaw.py",
      "src/voxel/pathlaw.py", "tests/test_routing.py", "tests/test_stairs.py",
      "tests/test_mechlaw.py", "tests/test_pathlaw.py"],
     ["tests/test_routing.py", "tests/test_stairs.py", "tests/test_mechlaw.py",
      "tests/test_pathlaw.py"]),
    # wave 13c: a module the books name as a writer or via, built after phase A (C7, 2026-11-01)
    ("C7", "T10", 7, ["src/voxel/injury.py"], []),
    # wave 15 batch 1: new modules the books name as a writer or via (2026-11-22)
    ("W15-T9", "T9", 8, ["src/voxel/heathist_world.py"], []),
    ("W15-T6", "T6", 8, ["src/voxel/reactions.py"], []),
    ("A-X", "X", 6,
     ["tests/test_w13_cross.py", "tools/w13_audit.py", "tools/w13_manifest.py"],
     ["tests/test_w13_cross.py"]),
)

LANES = {lane: {"topic": topic, "batch": batch, "new_files": files, "tests": tests}
         for lane, topic, batch, files, tests in _LANES}

# every new file -> lane
FILE_LANE = {f: lane for lane, d in LANES.items() for f in d["new_files"]}


def new_files():
    """All files phase A may create, sorted."""
    return sorted(FILE_LANE)


# ---- the new src/voxel modules and their owner topic ---------------------------
MODULES = {os.path.basename(f)[:-3]: LANES[lane]["topic"]
           for f, lane in FILE_LANE.items()
           if f.startswith("src/voxel/") and f.endswith(".py")}


# Scene builders (scenes_*.py) make a World, so they may import it. Every other
# new module must import alone (plan section 9).
SCENE_PREFIX = "scenes_"


def modules(topic=None):
    """Sorted new module names, optionally for one topic."""
    return sorted(m for m, t in MODULES.items() if topic is None or t == topic)


def topic_of(module):
    """Owner topic of a new module, or None for a module that is not new."""
    return MODULES.get(module)


def tests_of(lane):
    return list(LANES[lane]["tests"])


# ---- the BOOKS: one list, each term has ONE owner topic ------------------------
# name      the ledger term (a dict key or an attribute of the world or a books object)
# topic     the one topic that owns the term
# writers   the new modules allowed to write it (a module is allowed only if its
#           topic is the owner or it is listed in `via`)
# via       modules of other topics that reach the term only through the owner's API
# kind      "mass" (g), "energy" (J), "volume" (ml)
# shared    True for the two legacy terms every energy law already writes
#           (shed, gone): they exist in base18 and keep their many writers;
#           they are listed so the manifest is the whole list.
# parts     sub-keys of a term that is a dict (outside_booked, made, gone[species])
# Plan section 1, THE BOOKS. The plan puts `shed` under T1; it is a base18 term
# (sim.py) with the sign "heat that LEFT", so it is listed as shared legacy here.
BOOKS = (
    dict(name="gone", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=True, parts=("species",), note="base18 term; T1 adds the per-species keys"),
    dict(name="shed", topic="T1", writers=(), via=("airsink", "bodyheat", "outside", "slownet"),
         kind="energy", shared=True, parts=(),
         note="base18 term; sign: heat that LEFT the lattice"),
    dict(name="recast", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",)),
    dict(name="quant", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",)),
    dict(name="f32", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",)),
    dict(name="squeeze", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",)),
    dict(name="records_owned", topic="T2", writers=("record", "lots", "vessel"), via=(),
         kind="mass", shared=False, parts=()),
    dict(name="outside_booked", topic="T8", writers=("outside", "outside_world"), via=(),
         kind="energy", shared=False,
         parts=("rad", "edge", "air_film", "air_far", "shift", "top", "ground",
                "ref_shift", "mrt_resid")),
    dict(name="boiled_J", topic="T4", writers=("mixvle", "mixtures"), via=(), kind="energy",
         shared=False, parts=()),
    dict(name="melted_J", topic="T4", writers=("mixphase", "mixtures"), via=(), kind="energy",
         shared=False, parts=()),
    dict(name="frozen_J", topic="T4", writers=("mixphase", "mixtures"), via=(), kind="energy",
         shared=False, parts=()),
    dict(name="dissolved_J", topic="T4", writers=("mixsol", "mixtures"), via=(), kind="energy",
         shared=False, parts=()),
    dict(name="made", topic="T6", writers=("rxn",), via=("reactions",), kind="mass+energy", shared=False,
         parts=("water", "latent")),
    dict(name="mech_in", topic="T11", writers=("mechanisms",), via=(), kind="energy",
         shared=False, parts=(), note="only for releases with NO actor (one-flow rule)"),
    dict(name="pool", topic="T7", writers=("bodyheat",), via=(), kind="energy",
         shared=False, parts=()),
    dict(name="stores", topic="T7", writers=("bodyheat", "energetics", "digestion"), via=(),
         kind="energy", shared=False, parts=()),
    dict(name="work_out", topic="T7", writers=("energetics",), via=("fatigue", "stairs", "mechanisms"),
         kind="energy", shared=False, parts=(),
         note="actor work -> work_out; an act never also books mech_in"),
    dict(name="bh_injected", topic="T7", writers=("bodyheat",), via=(), kind="energy",
         shared=False, parts=("body_warmup",), note="the warm-up injection"),
    dict(name="pooled_ml", topic="T10", writers=("vasculature",), via=("injury",), kind="volume",
         shared=False, parts=()),
    dict(name="edema_ml", topic="T10", writers=("skinburn", "vasculature"), via=(),
         kind="volume", shared=False, parts=()),
    dict(name="layer_mass", topic="T3", writers=("layers",), via=(), kind="mass",
         shared=False, parts=("led_in", "led_out")),
    dict(name="layer_enthalpy", topic="T3", writers=("layers", "layer_laws"), via=(),
         kind="energy", shared=False, parts=()),
    dict(name="slow_residual", topic="T5", writers=("slowtime", "slownet"), via=(),
         kind="energy", shared=False, parts=()),
    dict(name="phase_J", topic="T9", writers=("heathist_world",), via=(), kind="energy",
         shared=False, parts=()),
)

BOOK_NAMES = tuple(b["name"] for b in BOOKS)

# names a term goes by in source text (attribute or dict key) -> manifest name.
# `led_in`/`led_out` are layers' own two dicts; `_bh_inject` is bodyheat's. (T5's
# `Scheduler.residual` is not aliased: the name is too common to scan for.)
SOURCE_ALIASES = {"led_in": "layer_mass", "led_out": "layer_mass", "_bh_inject": "bh_injected"}


def book(name):
    for b in BOOKS:
        if b["name"] == name:
            return b
    raise KeyError(name)


def owner_of(term):
    return book(term)["topic"]


def writers_of(term):
    """Modules allowed to write a term: its own writers plus its `via` list."""
    b = book(term)
    return set(b["writers"]) | set(b["via"])


# ---- tags -----------------------------------------------------------------------
TAGS = ("I", "U", "E", "C")
_TAG_RE = (("E", re.compile(r"^test_echo_")), ("C", re.compile(r"^test_codeonly_")),
           ("U", re.compile(r"^test_u_")))


def tag_of(test_name):
    """[I] unless the name starts test_echo_ / test_codeonly_ / test_u_ (the base
    name, with any parametrise suffix and a class prefix removed)."""
    base = test_name.split("[")[0].split("::")[-1]
    for tag, rx in _TAG_RE:
        if rx.match(base):
            return tag
    return "I"


# ---- the existing tests that read the SOURCE TREE ----------------------------------------
# A new file can break an old test only by being seen by it: a test that globs src/ (a size
# budget, a hygiene scan, an "only these modules read X" scan). Those are the tests a new
# module can change, so the audit and the cross tests run them in the sandbox and in base18.
FS_CALLS = re.compile(r"\b(glob|rglob|listdir|iterdir|scandir)\s*\(|\bos\.walk\s*\(")

# Failures of those tests that are KNOWN and belong to phase B (an existing file must change).
# test id -> why, and the one-line fix. The audit reports them loudly and still fails on any
# OTHER failure; a phase B edit that fixes one makes the test pass and the entry stale (the audit
# says so).
# EMPTY since lane B0COPY (wave 13a): its one entry, tests/test_voxel_attention.py::
# test_INTENT_and_JOB_were_not_already_person_keys (quiet.py reads p.get('intent') in horizon(), a
# second legitimate reader of wake_at), was fixed by adding 'quiet.py' to the allowed set in that
# test, so the entry went stale and the cross test asked for it to be removed.
KNOWN_PHASE_B = {}


def source_globbing_tests(base_root):
    """Node ids ('tests/x.py::test_y') of the EXISTING tests (in base_root/tests) whose body, or a
    module-level helper they call, lists the file system."""
    import ast
    out = []
    tdir = os.path.join(base_root, "tests")
    for f in sorted(os.listdir(tdir)):
        if not (f.startswith("test_") and f.endswith(".py")):
            continue
        try:
            src = open(os.path.join(tdir, f)).read()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        helpers = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                   and not n.name.startswith("test") and FS_CALLS.search(ast.get_source_segment(src, n) or "")}
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test"):
                seg = ast.get_source_segment(src, n) or ""
                called = {c.func.id for c in ast.walk(n) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
                if FS_CALLS.search(seg) or called & helpers:
                    out.append(f"tests/{f}::{n.name}")
    return out
