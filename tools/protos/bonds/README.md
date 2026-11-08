# Bonds against the continuum (lane BD, wave 9)

A prototype for roadmap decision 2 (explicit bonds in place of the continuum),
built beside `src/mpm` and changing nothing in `src/`. The user chose "measure
first"; this folder is the measurement. It is not wired into the voxel world.

| file | what it is |
|---|---|
| `bonds.py` | the model: rigid pieces, beam bonds that stretch, bend, twist and break, walls, contacts, air drag; one numba kernel; a momentum ledger |
| `scenes.py` | the four scenes as bonded objects (tree, rope, glass pane, plank) |
| `continuum.py` | the same four in `src/mpm`, built as the solver builds them today |
| `common.py` | what both are measured with (side views as text, long axis, clock, MPM momentum) |
| `run.py` | one experiment per process: `python3 run.py <b_*/c_* name> <out_dir>` |
| `test_bonds.py` | the prototype's own checks: conservation, free fall, beam sag, root break, pendulum period, and a shaken tree that holds and a felled one that goes over on its hinge |

Run everything under the shared lock and memory cap. The tests run by path:
`python3 -m pytest tools/protos/bonds/test_bonds.py` (tools/protos/conftest.py
keeps pytest from collecting this folder on its own).
