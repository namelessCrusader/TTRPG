"""The balance reflex and the footing law behind it (wave 12, lane BODY).

A body never puts full weight on a foot over a gap. Weight goes only on
contacts that bear it, the base is the hull of those, and the weight that
counts is the extrapolated centre of mass (Hof 2005). Past the base, a reflex
answers after the person's own delay (step, grab, else fall); an unconscious
body has no reflex and crumples; a body that goes over turns about the edge of
its base under real g. The reflex contract itself is in src/voxel/reflex.py.

Every scene is a small stone box with one or two people in it, and a test is
one cause."""
import numpy as np

from src.voxel import reflex
from src.voxel.materials import GRAVITY, TICK_S
from src.voxel.scenes import _person, _Wants
from src.voxel.sim import AIR, FLESH, IRON, STONE, WATER, World

FLOOR = 4          # the stone is 4 cells thick: a person stands on z = 4


def _room(nx=40, ny=30, nz=40):
    w = World(nx, ny, nz, voxel_cm=5)
    w.open_sky = False
    w.fill(0, nx, 0, ny, 0, nz, STONE)
    w.mat[1:nx - 1, 1:ny - 1, FLOOR:nz - 1] = AIR
    w.smass[1:nx - 1, 1:ny - 1, FLOOR:nz - 1] = 0.0
    w.exits = []
    return w


def _stands(w, px=15, py=15, name="the person"):
    """A person standing still on the floor."""
    p = _person(w, px, py, z0=FLOOR)
    p["name"] = name
    w.policy = _Wants(each={name: {"hands": "hands free", "legs": "stay"}})
    return p


def _hole(w, x0, x1, y0=1, y1=29):
    """A gap in the floor: air from z = 1 up, so there is a pit 3 deep."""
    w.mat[x0:x1, y0:y1, 1:FLOOR] = AIR
    w.smass[x0:x1, y0:y1, 1:FLOOR] = 0.0
    w._torque_solid = None


def _cells(w, p):
    return np.argwhere((w.mat == FLESH) & (w.eid == p["eid"]))


def _events(w, verb):
    return [e for e in w.events.since(-1) if e.verb == verb]


def _reflex_events(w, kind="balance"):
    return [e for e in w.events.since(-1) if e.why == f"reflex:{kind}"]


# ── the contract ────────────────────────────────────────────────────────────
def test_the_reflex_DELAY_is_the_persons_own_and_drink_fatigue_and_sleep_slow_it():
    w = World(10, 10, 10, voxel_cm=5)
    a = {"pid": 0, "name": "a", "alive": True, "awake": True}
    b = {"pid": 1, "name": "b", "alive": True, "awake": True}
    lo, hi = reflex.REFLEX["balance"]["band"]
    da, db = w.reflex_delay_s(a, "balance"), w.reflex_delay_s(b, "balance")
    assert lo <= da <= hi and lo <= db <= hi, "from the band on the sheet"
    assert da != db, "two people have two delays"
    assert da == w.reflex_delay_s(dict(a), "balance"), "and the same every run"
    a["balance_s"] = 0.2
    assert w.reflex_delay_s(a, "balance") == 0.2, "the sheet's own value wins"
    for key, share in (("drunk", 0.5), ("fatigue", 0.3), ("sleep_debt", 0.4)):
        c = dict(a, **{key: 1.0})
        assert abs(w.reflex_delay_s(c, "balance") - 0.2 * (1 + share)) < 1e-9, key
    tired = dict(a, drunk=1.0, fatigue=1.0, sleep_debt=1.0)
    assert w.reflex_delay_s(tired, "balance") > 1.5 * 0.2, "they add up"
    assert w.reflex_reach_m(tired, "balance") < w.reflex_reach_m(a, "balance"), \
        "and the same tiredness shortens the step"


def test_NO_reflex_when_out_and_the_CHOSEN_act_wins():
    w = World(10, 10, 10, voxel_cm=5)
    p = {"pid": 0, "name": "a", "alive": True, "awake": True}
    assert w._reflex_off(p, "balance") is None
    for why, state in (("awake", False), ("alive", False)):
        q = dict(p, **{why: state})
        assert w._reflex_off(q, "balance") == "out", why
    assert w._reflex_off(dict(p, asleep=True), "balance") == "out", "asleep"
    w._chose(p, "leap", 2)
    assert w._reflex_off(p, "balance") == "chosen", "a leap was chosen"
    w.tick += 3
    assert w._reflex_off(p, "balance") is None, "and only while it lasts"
    w._chose(p, "step_off", 100)
    assert w._reflex_gate(p, "balance") == "off"


def test_the_delay_is_counted_in_ticks_and_never_zero():
    w = World(10, 10, 10, voxel_cm=5)
    p = {"pid": 0, "name": "a", "alive": True, "awake": True,
         "balance_s": 0.1}
    assert w._reflex_gate(p, "balance") == "wait", "not on the tick it is seen"
    n = 0
    while w._reflex_gate(p, "balance") == "wait":
        w.tick += 1
        n += 1
    assert n == round(0.1 / TICK_S), n
    w._reflex_done(p, "balance")
    assert not p.get("_reflex")


# ── the footing law ─────────────────────────────────────────────────────────
def test_a_foot_over_nothing_is_not_in_the_base():
    """One foot on the floor and one over a gap: the base is the one foot, so
    the weight (between the feet) is past it."""
    w = _room()
    p = _stands(w)
    own = _cells(w, p)
    assert w._overbalanced(p, own) is None, "on a flat floor: balanced"
    _hole(w, 17, 19)                      # under the right foot
    own = _cells(w, p)
    foot, _g = w._footing(p, own)
    assert len(foot) and foot[:, 0].max() < 17, "only the firm foot bears him"
    over = w._overbalanced(p, own)
    assert over is not None and over[0] > 0 and abs(over[1]) < 1e-9, over


def test_the_base_SPANS_a_gap_between_two_firm_feet():
    """Both feet on firm ground with a hole between them: he stands."""
    w = _room()
    p = _stands(w)
    _hole(w, 16, 17)                      # between the legs (x 15 and 17)
    assert w._overbalanced(p, _cells(w, p)) is None


def test_a_held_thing_is_not_what_holds_him_up():
    """What you carry is not what carries you: a hand over a held block does
    not make the block a floor."""
    w = _room()
    p = _stands(w)
    own = _cells(w, p)
    top = own[own[:, 2] == own[:, 2].max()][0]
    w.fill(int(top[0]), int(top[0]) + 1, int(top[1]), int(top[1]) + 1,
           int(top[2]) + 1, int(top[2]) + 2, IRON)
    block = np.array([[top[0], top[1], top[2] + 1]])
    w.stamp(w.register("block"), block)
    p["held"] = {"cell": tuple(map(int, block[0])), "mat": IRON,
                 "label": "iron"}
    foot, _g = w._footing(p, own)
    assert not (foot[:, 2] > FLOOR + 1).any(), "nothing above the floor bears him"


def test_a_man_with_only_what_he_HOLDS_under_him_FALLS_free_and_does_not_tip():
    """Fuzzer seed 119: a hand over a block it held counted as the floor, so
    he was turned about the block and set down on it again, every third tick.
    What you carry is not what carries you: nothing bears him, and he falls."""
    w = _room()
    p = _person(w, 15, 15, z0=10)                 # over nothing at all
    p["name"] = "high"
    w.fill(18, 20, 14, 17, 26, 27, IRON)          # under his right forearm
    p["held"] = {"cell": (18, 15, 26), "mat": IRON, "label": "iron"}
    w._law_footing()
    mine = [b for b in w.bodies if b.get("owner") == "high"]
    assert len(mine) == 1 and mine[0].get("fly"), "he falls; he does not tip"
    assert not np.any(mine[0]["vel"]), "straight down: nothing to turn about"


# ── the balance reflex ──────────────────────────────────────────────────────
def test_a_man_with_ONE_FOOT_over_a_gap_STEPS_BACK_without_falling():
    w = _room()
    p = _stands(w)
    x0 = int(_cells(w, p)[:, 0].min())
    _hole(w, 17, 19)
    delay = round(w.reflex_delay_s(p, "balance") / TICK_S)
    for t in range(40):
        w.step()
        assert not w.bodies, f"he never went over (t{t})"
        if t < delay - 1:
            assert int(_cells(w, p)[:, 0].min()) == x0, "he waits out his delay"
    c = _cells(w, p)
    assert len(c) == 396 and int(c[:, 0].min()) < x0, "and then he stepped back"
    assert w._overbalanced(p, c) is None, "to where he stands firm"
    ev = _reflex_events(w)
    assert len(ev) == 1 and ev[0].verb == "stands" and ev[0].result == "steadied"
    assert abs(ev[0].tick - delay) <= 1, (ev[0].tick, delay)
    assert not _events(w, "falls")


def test_the_reflex_is_an_EVENT_the_mind_PERCEIVES():
    w = _room()
    p = _stands(w)
    _hole(w, 17, 19)
    for _ in range(12):
        w.step()
    rows = [r for r in w.minds[p["pid"]].perceived
            if r.how == "did" and r.verb == "stands"]
    assert rows, "the person's own mind has the step among what it did"
    assert any("catches their balance" in e for e in p["events"])


def test_an_UNCONSCIOUS_man_over_a_gap_has_NO_reflex(monkeypatch):
    from src.voxel import physiology
    monkeypatch.setitem(physiology.BODY, "wake_o2", 9e9)   # he stays out
    w = _room()
    p = _stands(w)
    p["awake"] = False
    _hole(w, 17, 19)
    for _ in range(80):
        w.step()
    assert not _reflex_events(w), "nothing in him answered"
    assert _events(w, "falls"), "he went over"


def test_the_man_who_CHOSE_to_step_off_is_not_stopped():
    w = _room()
    p = _stands(w)
    _hole(w, 17, 19)
    w._chose(p, "step_off", 60)
    for _ in range(30):
        w.step()
    assert not _reflex_events(w)
    assert _events(w, "falls")


def test_a_man_who_LEAPS_from_the_edge_is_not_stopped_by_the_reflex():
    """The one chosen act that covers the balance reflex today is a leap, set
    by the leap act itself (not by hand). Nothing in him answers while the
    leap is his choice (the first ticks of it, with his foot over the gap);
    when he lands with a foot over the gap again it is not a choice any more,
    and the reflex steps him back like anyone."""
    w = _room(nz=60)                      # (head room for the leap)
    p = _stands(w)
    _hole(w, 17, 19)
    w.step()
    w._leap(p, _cells(w, p))
    assert p["_chose"]["leap"] >= w.tick, "the act booked itself as chosen"
    assert any(b.get("fly") for b in w.bodies), "and he left the ground"
    for _ in range(40):
        w.step()
        if not any(b.get("fly") for b in w.bodies):
            break
    assert not _reflex_events(w), "so nothing in him answered it in the air"
    landed = w.tick
    for _ in range(40):
        w.step()
    ev = _reflex_events(w)
    assert ev and not _events(w, "falls"), \
        "and once down, with a foot over the gap, the reflex is his again"
    delay = round(w.reflex_delay_s(p, "balance") / TICK_S)
    assert ev[0].tick - landed >= delay - 1, \
        "with a fresh delay of its own: the count that began before the " \
        "leap did not run on through the air and answer the moment he landed"


def test_a_PUSHED_body_past_its_margin_STEPS_and_one_inside_it_does_not():
    for v, steps in ((0.15, False), (0.8, True)):
        w = _room()
        p = _stands(w)
        w.step()
        x0 = int(_cells(w, p)[:, 0].min())
        w.push_person(p, v, 0.0)
        for _ in range(30):
            w.step()
        c = _cells(w, p)
        assert not w.bodies and len(c) == 396, "he did not go over"
        if steps:
            assert int(c[:, 0].min()) > x0, "pushed past his margin, he stepped"
            assert _reflex_events(w)
        else:
            assert int(c[:, 0].min()) == x0 and not _reflex_events(w), \
                "pushed inside it, he leaned and stood"
        assert sum(abs(u) for u in (p.get("_v") or (0.0, 0.0))) < 0.02, \
            "and the speed is spent"


def _ledge_and_pillar(w):
    """Ground two cells wide under his left foot (x 15, 16) and a pillar with
    an iron post on it an arm's length out on the other side (x 21, 22): no
    step puts both feet on stone (the post is in the way of the only place
    that would), and the post is something to take hold of."""
    w.mat[1:39, 1:29, 1:FLOOR] = AIR
    w.smass[1:39, 1:29, 1:FLOOR] = 0.0
    rock = w.smass[0, 0, 0]
    for x0, x1, y0, y1 in ((15, 17, 1, 29), (20, 25, 12, 18)):
        w.mat[x0:x1, y0:y1, 1:FLOOR] = STONE
        w.smass[x0:x1, y0:y1, 1:FLOOR] = rock
    w.fill(21, 23, 14, 16, FLOOR, FLOOR + 19, IRON)    # 75 kg: it will not come
    w._torque_solid = None


def test_with_NO_STEP_to_take_he_GRABS_a_hold_within_reach():
    w = _room()
    p = _stands(w)
    _ledge_and_pillar(w)
    x0 = int(_cells(w, p)[:, 0].min())
    for _ in range(30):
        w.step()
    assert not w.bodies and len(_cells(w, p)) == 396, "he did not go over"
    assert int(_cells(w, p)[:, 0].min()) == x0, "he did not step: nowhere to"
    assert p.get("held"), "he took hold"
    ev = _reflex_events(w)
    assert len(ev) == 1 and ev[0].verb == "holds"
    assert w._overbalanced(p, _cells(w, p)) is None


def test_with_nothing_to_step_to_and_nothing_to_hold_he_goes_over_at_once():
    """Waiting out a delay that buys nothing would be a lie: he falls on the
    tick he is found over."""
    w = _room()
    p = _stands(w)
    _ledge_and_pillar(w)
    w.fill(21, 23, 14, 16, FLOOR, FLOOR + 19, AIR)     # no post
    w.step()
    assert w.bodies and _events(w, "falls")
    assert not _reflex_events(w)


def test_being_PROPPED_holds_a_man_only_the_way_that_stopped_his_fall():
    """A fall that a wall cut short leaves him stood up against it, and his
    weight past his feet on that side is held by the wall. The same weight
    past his feet on the OTHER side (he was stood back up on the edge of
    another man's shoulder) is held by nothing, and he goes."""
    outcome = {}
    for name, flip in (("same way", 1), ("other way", -1)):
        w = _room()
        p = _stands(w)
        _ledge_and_pillar(w)
        w.fill(21, 23, 14, 16, FLOOR, FLOOR + 19, AIR)     # no post, no step
        cells = _cells(w, p)
        over = w._overbalanced(p, cells)
        axis = 0 if abs(over[0]) >= abs(over[1]) else 1
        sign = 1 if over[axis] > 0 else -1
        p["_propped"] = tuple(int(v) for v in cells.min(axis=0))
        p["_propped_way"] = (axis, sign * flip)
        for _ in range(8):
            w.step()
        outcome[name] = bool(_events(w, "falls"))
    assert outcome == {"same way": False, "other way": True}


# ── going over ──────────────────────────────────────────────────────────────
def _pendulum_time(m, inertia, dxc, dzc, theta_end, w0=0.0):
    """Seconds for I a = m g (dx cos t + dz sin t) from t = 0 to `theta_end`
    (RK4, independent of the sim)."""
    th, om, t, dt = 0.0, w0, 0.0, 1e-4

    def acc(x):
        return m * GRAVITY * (dxc * np.cos(x) + dzc * np.sin(x)) / inertia
    while th < theta_end:
        k1t, k1o = om, acc(th)
        k2t, k2o = om + 0.5 * dt * k1o, acc(th + 0.5 * dt * k1t)
        k3t, k3o = om + 0.5 * dt * k2o, acc(th + 0.5 * dt * k2t)
        k4t, k4o = om + dt * k3o, acc(th + dt * k3t)
        th += dt * (k1t + 2 * k2t + 2 * k3t + k4t) / 6
        om += dt * (k1o + 2 * k2o + 2 * k3o + k4o) / 6
        t += dt
    return t


def test_a_body_TOPPLES_as_a_pendulum_under_real_g():
    w = _room(60, 60, 40)
    p = _stands(w, 30, 30)
    w.step()
    p["awake"] = False                    # a slack body: no feet to slide
    c = _cells(w, p)
    foot = c[c[:, 2] <= c[:, 2].min() + 1]
    b = w._tip(p, c.astype(np.float32), 0, 1, float(foot[:, 0].max()),
               float(c[:, 2].min()))
    m, inertia, dxc, dzc = b["pend"]
    assert 40.0 < m < 50.0, "a person's mass"
    # a 1.5 m body is 30 cells tall: its centre is a little over half way up
    assert 0.7 < dzc < 1.0, dzc
    thetas = []
    for _ in range(120):
        w.tick += 1
        w._law_bodies()
        if not w.bodies:
            break
        thetas.append(float(w.bodies[0]["theta"]))
    # to 1.0 rad: the sim agrees with the law written out on its own numbers
    k = next(i for i, th in enumerate(thetas) if th >= 1.0)
    want = _pendulum_time(m, inertia, dxc, dzc, 1.0) / TICK_S
    assert abs((k + 1) - want) <= 2, (k + 1, want)
    # and the whole way down takes about what a rod 1.5 m long does, from the
    # same small lean (3 g / 2 L, here 9.8 per s squared)
    t_down = (len(thetas) + 1) * TICK_S
    lean0 = np.arctan2(dxc, dzc)
    rod = _pendulum_time(1.0, 1.0 * 1.5 ** 2 / 3.0, 0.75 * np.sin(lean0),
                         0.75 * np.cos(lean0), np.pi / 2)
    assert 0.7 * rod <= t_down <= 1.3 * rod, (t_down, rod)
    assert 0.6 < t_down < 2.2, "a second or so, not a blink"


def test_a_man_who_is_AWAKE_LEAVES_the_edge_when_his_feet_slip_and_one_who_is_not_never_does():
    """Fuzzer seed 124: an awake man came down flat on a skeleton laid out
    standing, threw what he held, and was in two pieces. An awake man is not a
    plank on a hinge: when the push his feet must give the turn is more than
    friction can supply, they slide and he leaves the edge as a free body with
    the speed his centre of mass has, and lands on his feet. An unconscious
    man has no feet to slide, and goes down to the floor on the hinge."""
    from src.voxel.materials import FRICTION
    for awake in (True, False):
        w = _room(60, 60, 40)
        p = _stands(w, 30, 30)
        w.step()
        p["awake"] = awake
        c = _cells(w, p)
        foot = c[c[:, 2] <= c[:, 2].min() + 1]
        b = w._tip(p, c.astype(np.float32), 0, 1, float(foot[:, 0].max()),
                   float(c[:, 2].min()))
        m, inertia, dxc, dzc = b["pend"]
        flew = None
        for t in range(120):
            w.tick += 1
            w._law_bodies()
            if flew is None and w.bodies and w.bodies[0].get("fly"):
                flew = (t + 1, w.bodies[0]["theta"], w.bodies[0]["vel"].copy())
            if not w.bodies:
                break
        assert not w.bodies, "he came down"
        got = _cells(w, p)
        assert len(got) == 396 and p["_own"] == {tuple(x) for x in got.tolist()}
        tall = int(np.ptp(got[:, 2])) + 1
        if awake:
            assert flew is not None, "his feet slipped"
            t_slip, theta, vel = flew
            # where the law says: the horizontal push outruns friction. Written
            # out here on the numbers `_tip` measured, with its own turn.
            th, om = 0.0, 0.0
            for k in range(1, 200):
                for _ in range(5):
                    om += m * GRAVITY * (dxc * np.cos(th) + dzc * np.sin(th)) \
                        / inertia * (TICK_S / 5)
                    th += om * (TICK_S / 5)
                x_ = dxc * np.cos(th) + dzc * np.sin(th)
                z_ = dzc * np.cos(th) - dxc * np.sin(th)
                al = m * GRAVITY * x_ / inertia
                ax = z_ * al - x_ * om * om
                az = -x_ * al - z_ * om * om
                if abs(ax) > FRICTION * (GRAVITY + az):
                    break
            assert abs(k - t_slip) <= 1, (k, t_slip)
            assert 0.3 < theta < 0.9, f"a few tens of degrees ({theta:.2f} rad)"
            # (the floor under his soles takes the downward part of the speed
            # his weight has: he slides off it, he does not sink into it)
            assert 0.8 < vel[0] < 3.0 and vel[2] == 0.0, vel
            assert tall >= 28, "and he lands on his feet"
        else:
            assert flew is None and tall < 10, "an unconscious man lies where he fell"


def test_a_man_HAULED_over_an_edge_goes_over_with_the_speed_he_was_hauled_at():
    """Nobody pushes him: a hauler walks and he is dragged. His weight arrives
    at the edge with the speed it was moving at, so he goes over faster than
    one whose weight simply crossed the edge from rest."""
    starts = {}
    for name, speed in (("rest", 0.0), ("hauled", 1.0)):
        w = _room(60, 60, 40)
        p = _stands(w, 30, 30)
        w.step()
        c = _cells(w, p)
        if speed:
            vox = 0.1 * w.scale
            # ten ticks of a body moving toward +x at `speed` m/s
            p["_xy_hist"] = [(w.tick - 9 + k, 30.0 + speed * TICK_S * k / vox, 30.0)
                             for k in range(10)]
        # (the edge of his base is his front foot)
        w._reflex_done(p, "balance")
        b = w._tip_over(p, c, (1.0, 0.0))
        assert b, name
        starts[name] = float(b["w"])
    assert starts["rest"] == 0.0
    h = 0.75
    assert 0.6 * (1.0 / h) < starts["hauled"] < 1.4 * (1.0 / h), starts


def test_an_UNCONSCIOUS_body_CRUMPLES_in_about_a_second_instead_of_tipping_like_a_plank():
    w = _room(60, 60, 40)
    p = _stands(w, 30, 30)
    w.step()
    h0 = int(np.ptp(_cells(w, p)[:, 2])) + 1
    p["ko_until"] = 10 ** 9               # a head blow: out cold (a bruise no longer faints anyone)
    heights, plank, down = [], False, None
    for t in range(60):
        w.step()
        c = _cells(w, p)
        heights.append(int(np.ptp(c[:, 2])) + 1 if len(c) else None)
        plank |= any(b.get("pend") and b["theta"] > 0.05 and t < 8
                     for b in w.bodies)
        if down is None and t > 3 and not w.bodies:
            down = t + 1
    assert not p["awake"], "he fainted"
    # a sixth of a second in, the body has already sunk: the knees went
    assert heights[6] is not None and heights[6] <= h0 - 3, (heights[:12], h0)
    assert not plank, "it did not start to tip as a rigid stick"
    # and he is down within about a second
    assert down is not None and down * TICK_S <= 1.25, (down, heights)
    assert int(np.ptp(_cells(w, p)[:, 2])) + 1 < 0.4 * h0, "lying down"
    assert len(_cells(w, p)) == 396


# ── physiology and the water strip ──────────────────────────────────────────
def test_the_body_in_the_AIR_still_BREATHES():
    """A man in the air used to have his clock frozen. With no air to breathe
    (the oxygen is gone from the cells he falls through), his blood oxygen
    falls while he falls."""
    w = _room()
    p = _person(w, 15, 15, z0=30)         # over nothing: he falls on step one
    w.step()
    assert w._person_cells(p)[0] is None, "he is off the lattice"
    o0 = p["blood_o2"]
    for _ in range(10):
        w.o2[:] = 0.0
        w._law_life()
        w.tick += 1
        w._law_bodies()
    assert w.bodies, "and still falling"
    assert p["blood_o2"] < o0 - 1e-4, (o0, p["blood_o2"])


def test_a_walker_CROSSES_a_2_mm_strip_of_water_and_a_pool_still_stops_him():
    w = _room()
    p = _person(w, 10, 15, z0=FLOOR)
    ml = 5.0                              # 2 mm over a 5 cm cell
    for y in range(1, 29):
        w.fvol[13, y, FLOOR] = ml
        w.fl[13, y, FLOOR] = WATER
    water = float(w.fvol.sum())
    for _ in range(8):
        assert w._walk(_cells(w, p), 1, 0), "the strip is not a wall"
    assert int(_cells(w, p)[:, 0].min()) > 13, "he is across"
    assert abs(float(w.fvol.sum()) - water) < 1e-3, "and none of it was lost"
    # a cell more than half full is a pool
    w2 = _room()
    q = _person(w2, 10, 15, z0=FLOOR)
    for y in range(1, 29):
        w2.fvol[13, y, FLOOR] = 0.9 * w2.cap
        w2.fl[13, y, FLOOR] = WATER
    ok = [w2._walk(_cells(w2, q), 1, 0) for _ in range(6)]
    assert not all(ok), "a pool is still in the way"
