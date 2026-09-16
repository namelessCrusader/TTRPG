"""The play-test loop on the real engine: prompt → composed answer → parse → run.
Promises: the prompt holds the view AND the table; a good answer runs and the other
characters live; a bad answer names its piece and costs NO time."""
from src.core import playtest


def test_prompt_holds_view_and_table():
    w, p = playtest.new("vault", seed=12)
    text = playtest.prompt(w, p)
    assert "PERCEIVED by you" in text and "Build ONE action" in text
    assert "YOUR GOAL" in text and "speak" in text


def test_good_answer_runs_and_world_lives():
    w, p = playtest.new("vault", seed=12)
    out = playtest.act(w, p, "(then (move dir:S) (move dir:S))")
    assert out["ok"] and out["ran"] == ["move", "move"]
    assert p.pos[:2] == (0, 2) and w.tick >= 2
    assert not out["violations"]


def test_bad_answer_names_piece_and_costs_no_time():
    w, p = playtest.new("vault", seed=12)
    t0, pos0 = w.tick, p.pos
    out = playtest.act(w, p, "grab target:vial")
    assert not out["ok"] and out["piece"] == "grab"
    assert w.tick == t0 and p.pos == pos0, "a refusal must not move the world"


def test_speak_carries_words_and_gets_a_reply():
    w, p = playtest.new("vault", seed=12)
    out = playtest.act(w, p, "speak to:guard | Evening, Bran.")
    story = " ".join(out["narrative"])
    assert 'You: "Evening, Bran."' in story, "the words are said"
    assert "Bran the doorman:" in story, "the npc replies in character"
