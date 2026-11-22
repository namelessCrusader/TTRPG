"""WetWorld / HumidWorld: the plain names (wave 13c lane C3, 14a lane A, 14a refinement R4).

The code lives in wetmix.py (WetState, WetMixin, HumidMixin, SETTING_RH, in_setting) so that
sim.py can put HumidMixin into World's bases: every World is humid by default. Everything
is re-exported here; WetWorld and HumidWorld are kept as names for ColdWorld + the mixin.
"""
from .coldworld import ColdWorld
from .wetmix import *  # noqa: F401,F403
from .wetmix import (DEFAULT_SETTING, EVERY, HumidMixin, SETTING_RH, WetMixin, WetState,  # noqa: F401
                     _NOW, in_setting)
from .wetmix import (D_WATER, EDGE_RELAX, HEADSPACE, R_INT_SOLID, SEEP_K, TINY_G)  # noqa: F401


class WetWorld(ColdWorld):
    pass


class HumidWorld(ColdWorld):
    pass
