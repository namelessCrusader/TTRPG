"""The mind interface: how a character chooses a task, the same way in every
world (docs/sim-mind-design.md 3.5).

A backend writes its world into the types of `api`: percepts, beliefs, the
character's own state, and the options that the verbs of the catalog bind to.
The mind makes a menu and a scene text from them. A chooser (the rules, a
model, a player) picks one option. The backend runs it as a job.

`src/mind` imports nothing from `src/voxel`, `src/rooms` or `lab`. It uses the
standard library only; reading the catalog file also needs PyYAML.

    api       the types, and the protocols Chooser and Backend
    catalog   the verb catalog (data/verbs.yaml): load() reads it and checks it
    scope     the two-level menu: a scope first, then an act in it (step M12)
"""
from .api import (Backend, Belief, Chooser, Decision, DecisionRequest, Job, JobEvent, Menu,
                  Option, Percept, Ref, SelfState, Why)
from .catalog import Catalog, CatalogError, load

__all__ = ["Backend", "Belief", "Catalog", "CatalogError", "Chooser", "Decision",
           "DecisionRequest", "Job", "JobEvent", "Menu", "Option", "Percept", "Ref",
           "SelfState", "Why", "load"]
