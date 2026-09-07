"""Unified spatial substrate.

ONE kind of place: a Cell. A cell has spatial fields (heat, fluids — fluids may
be gases like oxygen/steam), plus tags/props, plus a material block that fills
it (wood/stone/metal) or None for air. Entities are mobile OCCUPANTS of cells.

Any property that "applies to all" applies to a *locus* — a cell or an entity —
through the same interface (see reactions._loci). Heat has exactly one home: the
cell. An entity's temperature is just the heat of the cell it stands in.

Positions are plain tuples, so 2D now and 3D later is a coordinate change, not a
rewrite. Gases are permitted (a cell fluid) but not yet simulated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

Pos = tuple[int, ...]
AMBIENT = 20.0  # °C

# Material → the tags every instance of that material carries. A wood block and
# a wooden entity are both `flammable` without anyone tagging them by hand.
# specific_heat (DF model): heat moves by Δtemp ÷ specific heat — LOW = answers
# fire fast (metal), HIGH = slow and steady (stone). One number per material.
MATERIALS: dict[str, dict] = {
    "air":   {"tags": set(), "solid": False, "specific_heat": 1.0},
    "wood":  {"tags": {"wooden", "flammable", "corrodible"}, "solid": True, "specific_heat": 2.5},
    "stone": {"tags": {"rigid"}, "solid": True, "specific_heat": 5.0},
    "metal": {"tags": {"metal", "rigid", "corrodible"}, "solid": True, "specific_heat": 0.6},
    "flesh": {"tags": {"biological", "flammable"}, "solid": False, "specific_heat": 3.0},
    "glass": {"tags": set(), "solid": False, "specific_heat": 1.5},
    "ash":   {"tags": set(), "solid": False, "specific_heat": 1.0},
}


def specific_heat(material) -> float:
    return MATERIALS.get(material or "air", {}).get("specific_heat", 1.0)

FLUIDS: dict[str, dict] = {
    # flow_rate/cling control the cellular-automaton spread: oil is viscous (clings,
    # barely flows, so it stays put to burn); water is thin and runs freely.
    "oil":   {"flammable": True, "ignition": 60.0, "fuel_per_tick": 60.0, "burn_heat": 45.0,
              "flow_rate": 0.06, "cling": 350.0},
    "acid":  {"corrosive": True, "flow_rate": 0.22, "cling": 140.0},
    # powder doesn't flow; it burns violently — an ignited keg is an explosion
    "blackpowder": {"flammable": True, "ignition": 80.0, "fuel_per_tick": 500.0,
                    "burn_heat": 320.0, "flow_rate": 0.0, "cling": 1e9},
    "water": {"suppressant": True, "flow_rate": 0.35, "cling": 60.0},
    "oxygen": {"gas": True},
    "steam":  {"gas": True},
}


def mat_tags(material) -> set[str]:
    return set(MATERIALS.get(material or "air", {}).get("tags", set()))


def is_solid(material) -> bool:
    return MATERIALS.get(material or "air", {}).get("solid", False)


@dataclass
class Cell:
    material: str | None = "air"    # what FILLS the cell (a wall) — None is open space
    floor: str | None = None        # the slab underfoot (a gallery's timber) — None = no floor
    heat: float = AMBIENT
    fluids: dict[str, float] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    props: dict[str, float] = field(default_factory=dict)


def _lift(p):
    return (p[0], p[1], 0) if len(p) == 2 else tuple(p)


class CellGrid(dict):
    """cells keyed by 3-tuple; 2D keys auto-lift so 2D-authored scenes/tests work."""

    def __getitem__(self, k):
        return super().__getitem__(_lift(k))

    def __setitem__(self, k, v):
        super().__setitem__(_lift(k), v)

    def __contains__(self, k):
        return super().__contains__(_lift(k))

    def get(self, k, default=None):
        return super().get(_lift(k), default)


@dataclass
class Entity:
    id: str
    name: str
    pos: Pos
    material: str | None = "flesh"
    tags: set[str] = field(default_factory=set)
    props: dict[str, float] = field(default_factory=dict)
    mind: dict = field(default_factory=dict)   # NPC cognition: traits, goal, salience, flaw/bond
    parts: dict = field(default_factory=dict)  # D13 behavior layer: {kind: state} — stateful/
                                               # resource-holding components (fuel_burn, liquid
                                               # volume, traps). Boundary rule: material property
                                               # → MATERIALS template; static boolean the menu
                                               # matches → tag; changes-over-ticks → PART.

    def __setattr__(self, k, v):               # positions are ALWAYS 3-tuples inside the engine
        if k == "pos" and v is not None:
            v = _lift(v)
        super().__setattr__(k, v)


@dataclass
class LogEntry:
    tick: int
    kind: str
    data: dict
    cause: str
    actor: str | None = None    # who caused it (None = the world itself)

    def story(self) -> str:
        return f"[t{self.tick}] {self.cause}"


@dataclass
class World:
    dims: Pos
    cells: dict[Pos, Cell]
    entities: dict[str, Entity]
    edges: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    tick: int = 0
    log: list[LogEntry] = field(default_factory=list)
    facts: dict = field(default_factory=dict)      # oracle canon: question -> truth
    clocks: dict = field(default_factory=dict)     # universal counters (checks.Clock)
    chaos: float = 3.0                             # the Director's pacing dial (1-9)
    stalls: dict = field(default_factory=dict)     # normalized free-text → refusal count
    pending: list = field(default_factory=list)    # soft moves: warnings with fuses
    last_notable: int = 0                          # for lull detection
    packs: list = field(default_factory=list)      # active rule-packs: a LitRPG "system"
                                                   # (cultivation, a dungeon System, an alien
                                                   # magic) is DATA — extra reactions layered
                                                   # on the base physics, not engine code.
    seed: int = 0
    _rng: random.Random | None = None

    def __post_init__(self):
        # normalise everything to voxel space: 2D-authored scenes become z=0 slabs
        self.dims = (self.dims[0], self.dims[1], 1) if len(self.dims) == 2 else tuple(self.dims)
        self.cells = CellGrid({_lift(p): c for p, c in self.cells.items()})
        for e in self.entities.values():
            e.pos = e.pos               # setter lifts

    @property
    def rng(self) -> random.Random:
        if self._rng is None:
            self._rng = random.Random(self.seed)
        return self._rng

    def in_bounds(self, p: Pos) -> bool:
        p = _lift(p)
        return all(0 <= c < d for c, d in zip(p, self.dims))

    def cell(self, p: Pos) -> Cell:
        return self.cells[_lift(p)]

    def neighbors(self, p: Pos) -> list[Pos]:
        p = _lift(p)
        out = []
        for axis in range(len(p)):
            for step in (1, -1):
                q = list(p)
                q[axis] += step
                q = tuple(q)
                if self.in_bounds(q):
                    out.append(q)
        return out

    def entities_at(self, p: Pos) -> list[Entity]:
        p = _lift(p)
        return [e for e in self.entities.values() if e.pos == p]
