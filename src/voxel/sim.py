"""The voxel physics core — an ACTUAL simulation (Ruling 1: model causes, never cases).

One voxel = a 10 cm cube = 1 liter. The world is dense numpy lattices over (x, y, z),
z up. Everything physical is a conserved amount:

  mat    what solid fills the voxel (air / wood / stone / iron)
  smass  grams of that solid remaining
  fl     which fluid sits here (none / water / oil) — one fluid per voxel for now
  fvol   milliliters of it
  E      thermal energy above ambient (J-ish); temperature is DERIVED: T = 20 + E/C

There are only four laws, and no special cases anywhere:
  1. fluids fall, then spread sideways by level difference (viscosity = how fast)
  2. heat conducts between neighbors by temperature difference
  3. combustion: where temperature >= a material's ignition point and fuel remains,
     mass converts at a temperature-driven rate and releases that material's heat.
     FIRE IS NOTHING BUT THIS HAPPENING — there is no on_fire flag.
  4. water above 100 °C boils away, and boiling COSTS energy (latent heat).

Consequences that fall out with zero case-code: wet wood will not light (its water
pins the voxel near 100 °C, below wood's 300 °C ignition); water quenches a wood
fire (cools below 300) but mostly fails against an oil fire (needs only 250 °C —
still hard — while the water boils off); a keg-sized oil pool is an inferno; fuel
runs out and the fire dies because the LAW has nothing left to convert.

Fluids are sparse (a puddle is ~dozens of voxels), so laws 1 and 4 walk an active
list in Python; laws 2 and 3 are dense numpy over the whole lattice. Deterministic:
no randomness anywhere.
"""
import math
import copy

from . import seam
from ..mpm.matter import Matter
import numpy as np

# EVERY NAME sim.py ever defined is still importable from it
# (tests, scenes, harvest, demo and room import from here).
from .kernels import (  # noqa: F401
    HAVE_NUMBA,
    njit,
    _o2_diffuse,
    _gas_overturn,
)
from .materials import (  # noqa: F401
    NOFLUID,
    WATER,
    OIL,
    ACID,
    WEAK_ACID,
    MTIN,
    MLEAD,
    MIRON,
    BLOOD,
    AMBIENT,
    AIR,
    WOOD,
    STONE,
    IRON,
    FLESH,
    ASH,
    GLASS,
    LEAF,
    CHAR,
    TIN,
    LEAD,
    NMAT,
    SOLID,
    FLUID,
    _FLIDS,
    _FDENS,
    _FC,
    _FK,
    REACTIONS,
    MISCIBLE,
    CHEST0,
    BODY,
    WOUND,
    BLEED,
    FRICTION,
    GRAVITY,
    TICK_S,
    AIR_DENS,
    DRAG_CD,
    FALL_SUBSTEPS,
    _REACTIVE,
    PLUME_REACH_M,
    BOIL,
    LATENT,
    SPAN,
    MELT,
    FREEZE,
    TOUGH,
    TSHOCK,
    MATNAME,
    TRANSMIT,
    SEE_TAU,
    SMOKE_EXT,
    _SIX,
    SMOKE_STILL,
    SHOUT_DB,
    HEAR_DB,
    SOUND,
    PUSH_THROUGH,
    _TOUGH_ARR,
    _TSHOCK_ARR,
    _DENS_ARR,
    _CSOLID_ARR,
    _KSOLID_ARR,
    _SPAN_ARR,
    _FDENS_ARR,
    _FC_ARR,
    _FK_ARR,
    _TRANSMIT_ARR,
    O2_PER_L,
    O2_RATIO,
    SOOT,
    SOOT_OIL,
    O2_RATIO_OIL,
)
from .policy import (  # noqa: F401
    GAZE_SWEEP,
    WILL,
    LIMBS,
    _DIRS,
    ACTS,
    REFLEXES,
    ORIENTING,
    LINES,
    ANSWERS,
    MEANS,
    IDLE,
    MENU_CAP,
    TablePolicy,
)
from .pose import (  # noqa: F401
    _SWEEP_RAD,
)
from .fields import FieldsMixin
from .fluids import FluidsMixin
from .heat import HeatMixin
from .air import AirMixin
from .support import SupportMixin
from .bodies import BodiesMixin
from .skeleton import SkeletonMixin
from .pose import PoseMixin
from .persons import PersonsMixin
from .physiology import PhysiologyMixin
from .senses import SensesMixin
from .motion import MotionMixin
from .hands import HandsMixin
from .will import WillMixin
from .ledgers import LedgersMixin
from .things import ThingsMixin
from .events import EventsMixin
from .perceive import PerceiveMixin
from .mechlaw import MechanismsMixin
from .layers_world import LayersMixin


class World(LayersMixin, MechanismsMixin, FieldsMixin, FluidsMixin, HeatMixin, AirMixin, SupportMixin,
            BodiesMixin, SkeletonMixin, PoseMixin, PersonsMixin,
            PhysiologyMixin, SensesMixin, MotionMixin, HandsMixin, WillMixin,
            LedgersMixin, ThingsMixin, EventsMixin, PerceiveMixin):
    # THE LAZY STATE OF WAVE 13 (docs/wave13-plan.md, section 1). Each is None
    # (or False) until a scene or a law makes it (`ensure_*` in fields.py), and
    # costs nothing before. They are CLASS attributes on purpose: an instance
    # attribute made by `__init__` would be deleted by `restore` from a
    # snapshot that never carried it (a world saved before they existed) and
    # then be missing; a class attribute is still there, and golden, which
    # hashes `vars(world)`, sees nothing until one is really made.
    slow = None            # T5: the scheduler (slowtime.Scheduler); step() calls its pre()
    climate = None         # T8: climate.Climate, None = no outside (every cold path is off)
    t_out = None           # T8: outside air temperature in force, C (None with no climate)
    t_sky = None           # T8: sky temperature in force, C
    air_rh = None          # T6/T8: relative humidity of fresh outside air (None = the old air)
    mixes = None           # T1: the MixTable (shared rows of what matter is made of)
    mixes_live = False     # T1: True once any cell holds a mix id; guards every gather
    scomp = None           # T1: solid mix id per cell, uint32 (0 = pure)
    fcomp = None           # T1: fluid mix id per cell, uint32 (0 = pure)
    dmg = None             # T9: damage per cell, float32 0 .. 0.999
    bzone = None           # T7: overlay id per cell (person slot x 16 + zone), uint16
    ice_g = None           # T6: grams of ice in a cell that is not an ICE cell, float32
    records = None         # T2: the record pools (record.Records)
    layers = None          # T3: the surface-layer pool (layers.Layers)
    face_env = None        # the face-environment bundle (faceenv.FaceEnv): heat's plug
    # THE TYPE OF E. float32 is the lattice's, and every golden hash. A world
    # that needs finer energy (a warm body: a flesh voxel holds 7400 J and a
    # float32 step there is 5e-4 J; D4) is built with E_dtype=np.float64.
    E_dtype = np.float32

    def __init__(self, nx, ny, nz, voxel_cm=10, E_dtype=None):
        if E_dtype is not None:
            self.E_dtype = np.dtype(E_dtype).type
        self.shape = (nx, ny, nz)
        self.scale = voxel_cm / 10.0              # length ratio to the 10 cm reference
        self.vox_l = self.scale ** 3              # liters per voxel (10 cm ref = 1 L)
        self.cap = 1000.0 * self.vox_l            # ml of fluid one voxel holds
        self.mat = np.zeros(self.shape, np.uint8)
        self.smass = np.zeros(self.shape, np.float32)
        self.fl = np.zeros(self.shape, np.uint8)
        self.fvol = np.zeros(self.shape, np.float32)
        self.fpot = np.zeros(self.shape, np.float32)    # POTENCY: fraction of the fluid
                                                        # that is still active reagent —
                                                        # spent/diluted acid pales toward
                                                        # inert sludge, never vanishes
        self.persons = []                               # living bodies: physiology state
        self.exits = []                                 # (x, y) places a fleeing
                                                        # body walks toward
        self.sounds = []                                # noises this tick and last
        self.speech = []                                # (tick, name, text): what
                                                        # was said aloud, for the
                                                        # TTS/subtitle pass
        self.policy = TablePolicy()                     # WHO CHOOSES. Swap this
                                                        # for a model-backed one;
                                                        # the sim never knows the
                                                        # difference
        self.traces = []                                # one row per decision:
                                                        # situation, the whole menu
                                                        # offered, and the pick.
                                                        # The harvest starts here
        self.thermostats = []                           # (x, y, z, T): DEV blocks
                                                        # pinned at a temperature —
                                                        # heat without fire, for
                                                        # testing what heat does
        self.vfall = np.zeros(self.shape[:2], np.float32)  # m/s DOWNWARD, per column
        self.fdrop = np.zeros(self.shape[:2], np.float32)  # voxels owed but not yet
                                                        # taken — a thing slower than
                                                        # one voxel a tick still falls
        self.fallh = np.zeros(self.shape, np.float32)   # voxels of ACCUMULATED free
                                                        # fall — cashed in as impact
                                                        # energy on landing
        self.rubble = np.zeros(self.shape, bool)        # cells that are BROKEN
                                                        # PIECES rather than whole
                                                        # material. A pane spans a
                                                        # frame; the shards of it
                                                        # span nothing. See
                                                        # `_shatter` and
                                                        # `_relax_slack`
        self._rub_gen = 0                               # bumped whenever it changes,
                                                        # so the slack cache knows
        self.edge = np.zeros(self.shape, np.float32)    # DECLARED contact area, m²:
                                                        # "this edge concentrates its
                                                        # blow into X mm²" is a fact
                                                        # about a manufactured thing
                                                        # (Ruling 2 q1), same class as
                                                        # density. 0 = blunt: a strike
                                                        # spreads over the voxel face.
                                                        # An edge is SUB-VOXEL shape,
                                                        # the one thing 5 cm cells
                                                        # cannot draw
        self.E = np.zeros(self.shape, self.E_dtype)
        self.smoke = np.zeros(self.shape, np.float32)   # grams of combustion gas per voxel
        self.o2 = None                                  # filled on first step, after the
        self.open_sky = True                            # scene is built (air gets fresh O2)
        self.air_region = None                          # compact id of each connected
        self._region_air = None                         # airspace; -1 in solids
        self.left_mass = 0.0                            # mass that walked OUT of
                                                        # the world through a door
        self._just_shattered = set()                    # cells emptied THIS pass:
                                                        # fragments must not plug
                                                        # the hole being made
        self._slack = None                              # support reach, cached
        self._slack_mat = None                          # ...against this layout
        self._slack_grip = frozenset()                  # ...and these held cells
        self._region_count = 0
        self._torque_solid = None                       # solids snapshot for the tip check
        self._torque_sleep = 0                          # cooldown after a wedged landing
        self.drops = []                                 # ballistic fluid parcels in flight:
        self.bodies = []                                # rigid bodies mid-TOPPLE (free
                                                        # of the grid until they land)
        # THE SEAM (roadmap 93). Matter that has to move leaves the lattice
        # through `self.matter` and is drawn back each tick; `moving` marks
        # the cells it is drawn into, so no law mistakes them for something
        # that stands still. This is the twelfth field, and it is "whose
        # matter is this": the lattice's, or the solver's.
        self.moving = np.zeros(self.shape, bool)
        self.matter = Matter(self)
        # OFF until the solver can do what the lattice fall already does. Sent
        # through the seam, a free fall lost air drag (a wide sheet fell as
        # fast as a brick), splashes, and the blow to whatever it landed on —
        # a dozen tests that were each about one of those. The switch stays so
        # the day the solver has them is a one-line change (roadmap 95).
        self.free_fall_to_seam = False
        # WHAT HAS LEFT THE WORLD ALTOGETHER, in grams by material. A lattice
        # has edges; a thing thrown past one is outside, and the books only
        # balance if the world can say how much went. See `_left_world`.
        self.gone = {}
        # Cells known to belong to something too big to pick up. Emptied
        # whenever matter moves; see `_objects_within_reach`.
        self._bulk, self._bulk_gen, self._matter_gen = None, -1, 0
        # AND WHAT IT HAS SHED AS HEAT, in joules — radiated into the colder
        # universe outside the lattice, less whatever the 0 °C floor had to
        # put back. See the end of `step`.
        self.shed = 0.0
        self.pcell = None                               # [x,y,z, vx,vy,vz, ml, fluid]
        self.vcell = None                               # coarse gas pressure + velocity
        self.p_add = np.zeros(self.shape, np.float32)   # pressure injected this tick
        self.tick = 0
        # DO NOT COMPUTE WHERE NOTHING IS HAPPENING. Most of a world is inert
        # most of the time, and a quiet room measured EXACTLY as expensive as a
        # burning one (98 ms/tick against 100). Every skip below has to be
        # provable, not merely plausible: a law is skipped only when the thing
        # it moves is not present at all, or cannot have changed since it last
        # ran. Set False to run every law over the whole world every tick —
        # which is what `test_ACTIVE_REGIONS_change_NOTHING` compares against.
        self.skip_quiet = True
        # AND WHERE IT IS HAPPENING, ONLY THERE. With this on (and skip_quiet),
        # a law that is busy somewhere does its work only where it can change
        # something: the columns a flame stands in, the cells that hold a
        # liquid, the cells whose matter moved. Every such skip gives the same
        # bits as the whole-world sweep, and tests/test_sleep_regions.py runs
        # worlds both ways, tick by tick, to say so.
        self.sleep_regions = True
        # Measure what the attention cap costs (see _decide). Off: it doubles
        # the work of deciding, and answers a question rather than doing one.
        self.recall_check = False
        # WHEN A BODY DECIDES AGAIN: the attention law. "latch" is the law
        # every body has always had; "onset" is the one a mind link needs. It
        # is ONE setting for the whole world, a law like the others, and never
        # a flag on one body. Nothing reads it yet (docs/sim-mind-design.md,
        # 3.5 and step M7).
        self.attention = "latch"
        # Use the compiled laws where there are any. Turned off, the world runs
        # on the numpy ones, which are what the laws MEAN — the compiled path
        # has to agree with them cell for cell, and a test says so.
        self.fused = HAVE_NUMBA
        self._cap_key = None                            # derived fields, cached on
        self._cap = self._por = None                    # what matter is where
        self._aircells = self._airblur = None           # and where the air is
        self._init_things()                             # which matter is one thing
        self._init_events()                             # what happened, and why

    def step(self):
        self._tick_start()                       # before any law runs
        if self.slow is not None:                # the scheduler's hook (T5): what is due
            self.slow.pre(self)                  # fires, what the tick wakes unfolds
        if self.o2 is None:                      # first step: fresh air fills every pore
            self.o2 = np.full(self.shape, O2_PER_L * self.vox_l, np.float32)
        # DO NOT COMPUTE WHERE NOTHING IS HAPPENING. Measured before any of
        # this: a room with nothing lit in it cost 98 ms a tick and a room on
        # fire cost 100 — the whole world was being swept for heat that was not
        # there, smoke that did not exist and puddles nobody had poured.
        #
        # Every gate below is PROVABLE rather than plausible, which is the only
        # kind worth having: a law is passed over exactly when the thing it
        # moves is absent, so running it could not have changed one cell.
        # `self.E` is energy ABOVE ambient, so all-zero means every voxel sits
        # at 20 °C — no gradient to conduct, no dT to rise, nothing hot enough
        # to light, boil or melt, and no face carrying a shock. Set
        # `skip_quiet = False` to run everything over everything, which is what
        # `test_ACTIVE_REGIONS_change_NOTHING` compares every scene against.
        # Particle matter comes OFF the lattice before the laws that would try
        # to move it, is carried, and goes back on before the laws that need to
        # see it. Support, torque and footing therefore never meet it and
        # cannot re-promote what is already promoted — which was roadmap 91's
        # whole loop, closed by ordering rather than by a memory.
        self.matter.unrasterize()
        self.matter.step(TICK_S)
        self._law_bodies()
        if self._busy(self.fvol) or self.drops:
            self._law_flow()                     # (drops in flight are fluid too)
            self._law_stir()
            self._law_head()
        self._law_grips()                        # a hold is re-earned every tick
        self._law_footing()                      # before support: a body with no
        self._law_support()                      # footing must leave as a BODY
        self._law_torque()
        self.matter.rasterize()                  # back on the lattice: it can
                                                 # burn, be seen, be walked round
        self._after_raster()                     # what the particles did, as events
        if getattr(self, "igniters", None):
            self._law_ignite()                   # a flame held to a cell is heat
        hot = self._hot_window()                 # ...and WHERE the heat is
        if hot is not None:
            self._law_conduct(hot)
            self._law_crack()                    # thermal shock reads what conduction wrote
            risen = self._hot_window()           # conduction just MOVED heat, so
            if risen is not None:                # ask again before moving it more
                self._law_rise(risen)
        self._air_regions()                      # (re)label airspaces if walls/gaps changed
        if self._heat_busy():
            self._law_burn()
            self._law_boil()
            self._law_melt()
        if self._busy(self.fvol):
            self._law_acid()
        sooty = self._window(self.smoke > SMOKE_STILL,   # asked AFTER burning,
                             self._SMOKE_REACH)          # which is where it
                                                         # comes from
        if sooty is not None:
            self._law_smoke(sooty)
        self._air_regions()                      # acid may have bored new airspace
        if self._heat_busy() or self._mixing():
            self._law_o2()
        if self._pushed():
            self._law_air()                      # the pneumatic field: pressure, wind
        self._law_identity()                     # what each thing now is
        self._law_bleed()                        # what is leaving them
        self._law_life()                         # the slow chemistry of anyone alive
        self._law_perceive()                     # what reached their senses
        self._law_will()                         # and their reflexes
        self._law_pose()                         # and where their limbs ended up
        self._law_joints()                       # and what gravity does to the rest
        if self.thermostats or self._heat_busy():
            if self.thermostats:                    # dev heater blocks hold their
                C = self.heat_capacity()            # set temperature against all
                for (tx, ty, tz, tT) in self.thermostats:   # losses — heat without fire
                    self.E[tx, ty, tz] = (tT - AMBIENT) * C[tx, ty, tz]
            # AND WHAT THE WORLD SHEDS, IT SAYS IT SHED. Heat leaves the world
            # only through exposed faces now (`_law_conduct`, `_law_rise`,
            # `_law_burn`), and each of those writes down what it sheds. What
            # is left here is the 0 °C floor, which can only put heat back:
            # it nets into `shed` too, so "energy is conserved" stays a
            # statement anyone can CHECK. (The old per-voxel RAD and LEAK
            # lines, zero since G4/G5, are gone: they cost about six
            # whole-world array passes a step and changed nothing.)
            self._floor()
        self._tick_end()                         # what a tick reports at its end
        self.tick += 1

    def _floor(self):
        """THE SEAM for the 0 °C floor: the one place the world puts heat back.
        No golden case ever goes below 0 °C (B0's floor probe, 35 of 35), so
        cold (T8) deletes it by overriding this; here it is as it always was."""
        was = float(self.E.sum())
        self.E = np.maximum(self.E, -AMBIENT * self.heat_capacity())   # nothing below 0 °C here
        self.shed += was - float(self.E.sum())  # net: the floor above is a
                                                # SOURCE, and nets in here
