"""perceive() — the single perception source behind the perception-filtered affordance menu
(design doc D8). Locks the two load-bearing properties: an id-addressed slice, and partial
observability with stale memory (unseen beliefs persist and carry an age)."""
from src.core.seed import vault
from src.core.perceive import observe, perceive, render


def _refs(sl, *, fresh=None, self_=False):
    return sorted(n["ref"] for n in sl["nodes"]
                  if (self_ or not n["self"]) and (fresh is None or n["fresh"] == fresh))


def test_slice_is_id_addressed_and_self_first():
    w, player = vault(seed=12)
    observe(w, player, tick=0)
    sl = perceive(w, player, now=0)
    assert sl["viewer"] == player.id
    # ids are dense 0..n-1 in traversal order; the self node is present and marked
    assert [n["id"] for n in sl["nodes"]] == list(range(len(sl["nodes"])))
    self_nodes = [n for n in sl["nodes"] if n["self"]]
    assert len(self_nodes) == 1 and self_nodes[0]["ref"] == player.id
    # nodes are in nondecreasing distance order (proximity traversal)
    dists = [n["d"] for n in sl["nodes"]]
    assert dists == sorted(dists)
    assert render(sl).startswith("### PERCEIVED by")


def test_partial_observability_stale_memory():
    w, player = vault(seed=12)
    observe(w, player, tick=0)
    seen0 = set(_refs(perceive(w, player, now=0)))
    assert seen0, "should perceive something at the start"

    player.pos = (5, 3, 0)                 # walk to the far corner: the start-room entities leave sight
    observe(w, player, tick=6)
    sl = perceive(w, player, now=6)

    fresh = set(_refs(sl, fresh=True))
    stale = {n["ref"]: n["age"] for n in sl["nodes"] if not n["fresh"] and not n["self"]}
    # things no longer visible PERSIST in belief, marked stale with the elapsed age
    assert stale, "unseen-but-remembered entities must persist as stale nodes"
    assert seen0 & set(stale), "the start-room entities should now be stale, not dropped"
    assert all(age == 6 for age in stale.values()), "stale age = ticks since last seen"
    # a fresh sighting refreshes (age 0), and fresh/stale are disjoint
    assert fresh.isdisjoint(stale)
    for n in sl["nodes"]:
        assert n["fresh"] == (n["age"] == 0)


def test_no_belief_store_falls_back_to_live_observe():
    w, player = vault(seed=12)
    assert "belief" not in player.mind
    sl = perceive(w, player, now=0)         # perceive without a prior observe() still works
    assert any(n["self"] for n in sl["nodes"])
    assert all(n["fresh"] for n in sl["nodes"])   # a live fallback sees everything as fresh
