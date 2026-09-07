"""Protoreasoning-trace harvest (SMALL_MODEL_COMPETENCE): every decision the
selector makes is CAPTURED, and only the ones the sim later BLESSES (a clean,
committed outcome — not a refusal, no-op, or stall) are recorded into the kNN
datastore as (situation → what-was-right) anchors.

The paper's lesson: a trace teaches only when its CONTENT is correct. So a pick
that got refused must never be learned as the right answer — that would poison
the corpus. The engine owns "blessed" because it's the only layer that sees the
whole resolution; social.py and composer.py just push their scored picks into a
shared per-turn buffer, ignorant of each other.
"""

from src.core import knn, trace


def _capture(intent="threaten", chosen="cower"):
    trace.capture(intent, ["cower", "defy", "comply"], [-1.0, -3.0, -2.0], chosen)


# DESIGN: capture is a cheap append to a per-turn buffer — nothing is learned yet.
# The datastore stays empty until the turn is blessed.
def test_capture_buffers_without_learning():
    ds = knn.Datastore()
    trace.begin(ds)
    _capture()
    assert ds.anchors("threaten", ("cower", "defy", "comply")) == [], "capture alone learns nothing"
    assert trace.pending(), "but the decision is buffered"


# DESIGN: a BLESSED turn flushes every buffered decision into the datastore.
def test_bless_records_the_trace():
    ds = knn.Datastore()
    trace.begin(ds)
    _capture()
    trace.bless()
    anchors = ds.anchors("threaten", ("cower", "defy", "comply"))
    assert len(anchors) == 1, "the blessed decision becomes an anchor"
    assert anchors[0][1] == "cower", "and it remembers WHAT was chosen"
    assert not trace.pending(), "the buffer is drained"


# DESIGN (keystone signal upgrade): bless(advanced=False) drops the buffer like a
# refusal — an action that committed but did NOT advance the world (a bare walk) is
# not a positive example. This is what stops training amplifying the walk-collapse.
def test_bless_requires_advancing_the_world():
    ds = knn.Datastore()
    trace.begin(ds)
    trace.capture("verb", ["walk", "pour"], [-0.5, -2.0], "walk")
    trace.bless(advanced=False)                 # committed, but only shuffled position
    assert sum(len(v) for v in ds.buckets.values()) == 0, "an idle walk earns no anchor"
    assert not trace.pending(), "and the buffer is drained"
    trace.begin(ds)
    trace.capture("verb", ["walk", "pour"], [-2.0, -0.5], "pour")
    trace.bless(advanced=True)                  # changed the world
    assert sum(len(v) for v in ds.buckets.values()) == 1, "a consequential act is kept"


# DESIGN (keystone, end to end): through the real engine, a turn whose only effect
# is a move must NOT bless the verb pick that led to it — even though the composer
# scored and captured that pick. Before, the walk fed the corpus and the collapse.
def test_engine_does_not_bless_a_bare_walk():
    from src.core import engine, social, composer
    from src.core.seed import guildhall
    social.use_brain("mock")
    social.DATASTORE = knn.Datastore()

    class WalkPicker(composer.MockPicker):
        def pick_scored(self, scene, utterance, question, options):
            ids = [o[0] for o in options]
            if "walk" in ids:                   # force the walk branch so a pick is captured
                i = ids.index("walk")
                sc = [0.0] * len(options); sc[i] = 5.0
                return i, sc
            return super().pick_scored(scene, utterance, question, options)

    world, player = guildhall(seed=3)
    res = engine.free_text(world, player, "head over there", WalkPicker())
    assert any("move" in ln for ln in res["narrative"]), "sanity: the turn was a walk"
    assert sum(len(v) for v in social.DATASTORE.buckets.values()) == 0, \
        "a bare walk must not be harvested as a positive example"
    social.DATASTORE = None                     # reset global so other tests aren't affected


# DESIGN: a REFUSED/discarded turn throws the buffer away — a pick the sim
# rejected must NOT be learned as correct (that would poison the corpus).
def test_discard_drops_the_trace():
    ds = knn.Datastore()
    trace.begin(ds)
    _capture()
    trace.discard()
    assert ds.anchors("threaten", ("cower", "defy", "comply")) == [], "rejected picks are not learned"
    assert not trace.pending(), "the buffer is drained on discard too"


# DESIGN: with no datastore (fuzz/tests), capture is a silent no-op — the harness
# never crashes a turn just because learning is disabled.
def test_no_datastore_is_inert():
    trace.begin(None)
    _capture()               # must not raise
    trace.bless()            # must not raise
    assert not trace.pending()


# DESIGN: a new turn's begin() clears any stale buffer — an un-blessed decision
# from a crashed/abandoned turn never leaks into the next.
def test_begin_clears_stale_buffer():
    ds = knn.Datastore()
    trace.begin(ds)
    _capture()               # buffered but never resolved
    trace.begin(ds)          # next turn starts
    assert not trace.pending(), "a fresh turn starts clean"
    trace.bless()
    assert ds.anchors("threaten", ("cower", "defy", "comply")) == [], "stale pick was dropped"


# DESIGN: several picks in one turn (composer picks a verb, then a slot) all ride
# the same blessing — one coherent action, one trace of decisions.
def test_multiple_captures_share_one_blessing():
    ds = knn.Datastore()
    trace.begin(ds)
    trace.capture("verb", ["pour", "ignite"], [-0.5, -2.0], "pour")
    trace.capture("fluid", ["oil", "water"], [-0.4, -1.5], "oil")
    trace.bless()
    assert ds.anchors("verb", ("pour", "ignite")), "the verb pick was learned"
    assert ds.anchors("fluid", ("oil", "water")), "and the slot pick too"
