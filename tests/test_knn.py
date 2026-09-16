"""kNN-Prompting (SMALL_MODEL_COMPETENCE #1): the frozen model produces a score
distribution; the DECISION is a non-parametric vote over past (distribution →
correct-choice) anchors. The only technique with published evidence at 0.8B, and
it needs no training — it feeds on the log we already keep.
"""

import math

from src.core import knn


# DESIGN: a datastore remembers (score-vector over options → the choice that was
# right). It's append-only and keyed by the option set so votes stay comparable.
def test_datastore_remembers_and_recalls():
    ds = knn.Datastore()
    ds.add("threaten", ["cower", "defy", "comply"], [-1.0, -3.0, -2.0], "cower")
    ds.add("threaten", ["cower", "defy", "comply"], [-3.0, -1.0, -2.0], "defy")
    assert len(ds.anchors("threaten", ("cower", "defy", "comply"))) == 2
    assert ds.anchors("talk", ("wary",)) == [], "different menu, different bucket"


# DESIGN: KL-nearest anchors vote. A query that looks like the cower-anchors
# should be pulled toward "cower" even when the raw argmax was something else.
def test_knn_vote_pulls_toward_similar_past_decisions():
    ds = knn.Datastore()
    opts = ["cower", "defy", "comply"]
    for _ in range(5):
        ds.add("threaten", opts, [-0.5, -4.0, -3.0], "cower")   # scared-shaped → cower
        ds.add("threaten", opts, [-4.0, -0.5, -3.0], "defy")    # bold-shaped  → defy
    vote = knn.vote(ds, "threaten", opts, [-0.7, -3.8, -3.1], k=4)
    assert vote["cower"] > vote["defy"], f"scared-shaped query votes cower: {vote}"


# DESIGN: blending is a knob — α=1 is the pure model, α=0 is pure memory. The
# blend must be able to OVERRIDE a wrong model pick when memory is confident.
def test_blend_can_override_a_wrong_model_pick():
    ds = knn.Datastore()
    opts = ["cower", "defy", "comply"]
    for _ in range(8):
        ds.add("threaten", opts, [-0.6, -3.9, -3.0], "cower")
    model_scores = {"cower": -2.0, "defy": -1.0, "comply": -3.0}   # model wrongly likes defy
    q = [-0.7, -3.8, -3.1]                                          # but query is scared-shaped
    picked = knn.blend_pick(ds, "threaten", opts, q, model_scores, alpha=0.3, k=8)
    assert picked == "cower", "confident memory corrects the model"
    solo = knn.blend_pick(ds, "threaten", opts, q, model_scores, alpha=1.0, k=8)
    assert solo == "defy", "at alpha=1 the model stands alone"


# DESIGN: with no relevant anchors, blend falls back to the model — kNN never
# HURTS a cold start, it only helps once the log has seen similar situations.
def test_cold_start_defers_to_the_model():
    ds = knn.Datastore()
    opts = ["wary", "warm", "dismiss"]
    model_scores = {"wary": -1.0, "warm": -2.0, "dismiss": -3.0}
    picked = knn.blend_pick(ds, "talk", opts, [-1.0, -2.0, -3.0], model_scores, alpha=0.3, k=8)
    assert picked == "wary", "empty memory = trust the model"


# DESIGN: the datastore round-trips to disk (it IS the accumulated play log).
def test_datastore_persists(tmp_path):
    ds = knn.Datastore()
    ds.add("threaten", ["cower", "defy"], [-1.0, -2.0], "cower")
    p = tmp_path / "store.json"
    ds.save(p)
    ds2 = knn.Datastore.load(p)
    assert len(ds2.anchors("threaten", ("cower", "defy"))) == 1
