# Design prototypes

These files are copies of the scratch prototypes that `docs/sim-mind-design.md` cites. They are here so that they are not lost when `/tmp` is cleaned. The copies are exact, and nobody maintains them.

- They are evidence and starting points. They are not tools and not tests.
- Most of them take the repo (or a copy of it) as their first argument and import `src.voxel` from there. Some of them contain scratch paths from the session that wrote them.
- Some of them patch `World` while they run (`onset/patch_*.py`) or need a tree that is already carved. Run them only on a sandbox copy, never on the live tree.
- `conftest.py` stops pytest from collecting this folder. `ghost_test.py` matches pytest's `*_test.py` pattern, and it calls `sys.exit` when it is imported.

| folder | what the design uses it for |
|---|---|
| `ghost_test.py` | S5: the snapshot "ghost World" test |
| `label_golden.py`, `eid_proto2.py`, `index_cost.py` | the entity layer (E1, E2) |
| `review/` | E0: the edge probes and `drawn_shove.py` |
| `mpm/exact_check.py`, `mpm2/` | the particle lane (X1 to X7): exactness, budget, sparse rasterize, write-back |
| `events/` | the events lane (V0 to V5) |
| `onset/` | M7: the attention law and the policy hooks (`sim_mind.diff`, `mind_duel.py`, `duel_baseline.py`) |
| `proto/` | the mind lane: the mind API and the rooms backend |
| `mind_iface/` | the verb catalog (`verbs_v2.yaml`) and its checker |
