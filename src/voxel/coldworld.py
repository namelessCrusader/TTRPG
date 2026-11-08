"""ColdWorld: a World with the outside (T8, wave 13b lane C1).

`class ColdWorld(OutsideMixin, World)`. The mixin goes in FRONT, so its seams
(`_floor`, `_top_sink`, `_unsettled`, `_heat_busy`, `_env_radiation`, `_T_in`,
`_rest_cell`, `_tick_start`, `_law_rise`) win. With no climate on it, it
behaves as World does, bit for bit (tests/test_cold_on.py says so on golden
scenes). `set_climate(...)` is the capability gate: no climate, no cold.

Why a subclass and not World's bases: C1 does not own sim.py. The merge step
puts OutsideMixin into World's bases (see the C1 report, needs_other_files);
from then on ColdWorld is World and this file is a one-line alias.
"""
from .outside_world import OutsideMixin
from .sim import World


class ColdWorld(OutsideMixin, World):
    pass
