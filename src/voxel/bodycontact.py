"""BODY-ON-BODY CONTACT AND THE HOF BALANCE MECHANISMS (wave 14b, lane E).

ONE CONTACT LAW. A shove, a tackle, a grapple and a collision are the same
thing: two bodies with masses, velocities and a normal, meeting with a flesh
restitution and a friction between them. They differ only in the INPUTS
(speed, the mass that is moving, whether the hands hold). No act has its own
code.

THE LAW (`collide`, pure numpy-free arithmetic on 3-vectors)
  n        unit normal from body 1 to body 2.
  v_n      closing speed (v1 - v2) . n. Not closing: nothing happens.
  J_n      (1 + e) mu_r v_n,   mu_r = m1 m2 / (m1 + m2), e the restitution.
           The normal impulse. Momentum is conserved exactly.
  J_t      the tangential impulse that would stop the sliding, capped by
           Coulomb: min(mu_r |v_t|, mu_b J_n). A held contact (grapple: the
           hands grip) has e = 0 and never slides (J_t is the full sticking
           impulse).
  Energy:  KE_before = KE_after + deform_J + friction_J, every joule.
           deform_J = (1 - e^2) mu_r v_n^2 / 2 is what the flesh absorbs in the
           normal direction; friction_J is what sliding wastes. Both leave as
           heat in the contact cells (the caller deposits them).
  The two bodies are springs IN SERIES, so the same force presses on both and
  the energy goes to each in the ratio of its compliance, 1/k
  (`partition`). The peak compression energy (the whole closing energy
  mu_r v_n^2 / 2, before the flesh gives some back) is what a bone feels
  (`peak_force`, impact.blow_on_bone's criterion); the absorbed share is what
  a bruise reads (physiology._blow).

THE HOF MECHANISMS (Hof 2005 doi:10.1016/j.jbiomech.2004.03.025, Hof 2008
doi:10.1016/j.gaitpost.2006.05.016). A body is an inverted pendulum with the
extrapolated centre of mass xi = x + v / w0 (w0 = sqrt(g / l)) and a base
(the convex hull of the weight-bearing contacts). While xi is inside the base
the body is stable. When it is past the base by b, three things beside the
step can pull it back, and each moves the centre of pressure by an equivalent
amount (the torque it makes divided by m g):
  1. THE ANKLE (moving the centre of pressure inside the foot): free up to the
     edge of the foot, but the torque of the ankle is limited (`ankle_cop_m`).
     A tired or injured leg cannot use the whole foot: the base SHRINKS.
  2. COUNTER-ROTATING SEGMENTS (trunk and arms swung against the fall): an
     angular acceleration alpha of a flywheel of inertia I makes a torque
     I alpha, an equivalent centre-of-pressure shift I alpha / (m g), for as
     long as the segment can keep turning (until its speed limit w_max). The
     pulse that recovers b from the pendulum's exponential growth lasts
     T = w_max / alpha, and the shift it buys is
         b_max = (I alpha / (m g)) (1 - exp(-w0 w_max / alpha)).
     The body's own delay costs it the growth of b meanwhile, exp(w0 t_d).
  3. AN EXTERNAL FORCE (a hand on a wall, on a heavy thing, on another
     person): a horizontal force F at hand height h shifts the equivalent
     centre of pressure by F h / (m g). The reaction can be no bigger than the
     arm can push, and no bigger than what the thing pushed against can hold
     (its weight times the floor friction: a wall holds, a light person
     gives way).
The numbers marked (u) are unverified.

SOURCES
  Hof, Gazendam, Sinke 2005 (the extrapolated centre of mass); Hof 2007, Hof
  2008 (the three mechanisms: foot placement, ankle moment, counter-rotation;
  the equivalent shift I alpha / (m g) is Hof's own way of reading the
  flywheel). Pijnappels, Kingma, Wezenberg, Reurink, van Dieen 2010, Gait
  Posture 32:? (arm elevation and trunk flexion in recovery) for the size.
  Winter 2009 (segment masses and radii of gyration): trunk about 0.50 of
  body mass, each arm 0.05, radius of gyration of the trunk about the hip
  0.30 to 0.55 of its length. Collision: flesh against flesh has a
  restitution of 0.1 to 0.4 in contact-sport and dummy tests (u: Kendall,
  Hrysomallis for a chest against padding); 0.2 is the figure the foot on wood
  already uses (mechlaw.E_C_FOOT)."""
import math

G = 9.81

E_FLESH = 0.20             # restitution of flesh on flesh (u)
MU_BODY = 0.50             # clothed body on clothed body, Coulomb (u)
K_TORSO = 60e3             # N/m, the chest's contact stiffness at 70 kg (impact.BLOW_K fist/torso) (u)
AREA_70 = 0.03             # m2, a shoulder or torso contact at 70 kg (u)
REF_KG = 70.0
REF_H = 1.75

# the counter-rotating flywheels (u), at 70 kg and 1.75 m; I scales as m H^2
FLY = {
    "trunk": {"I": 2.6, "alpha": 30.0, "w_max": 4.0},     # trunk and head about the hips
    "arms": {"I": 1.1, "alpha": 80.0, "w_max": 12.0},     # both arms swung about the shoulders
}
ANKLE_NM_PER_KG = 1.7      # plantarflexor torque, 120 Nm at 70 kg (u)
ARM_PUSH_BW = 0.40         # bilateral horizontal push against a wall, share of weight (u)
HAND_H_FRAC = 0.62         # the hand's height as a share of the body's height
ARM_REACH_FRAC = 0.40      # arm's reach beyond the body's edge, share of height
HAND_CLOSE_MS = 0.5        # how fast a reaching hand closes on a body, m/s (u)
HOLD_MU = 0.50             # what a free thing resists a push with: this x its weight (u)


# ── the contact law ──────────────────────────────────────────────────────────
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def collide(m1, v1, m2, v2, n, e=E_FLESH, mu=MU_BODY, grip=False):
    """Two bodies meet along the unit normal `n` (from 1 to 2). Returns a dict:
    v1, v2 (after), J_n, J_t, closing, deform_J, friction_J, KE_before,
    KE_after, hit (False when they were not closing). A held contact
    (`grip`) has e = 0 and no slip."""
    v1 = tuple(float(c) for c in v1)
    v2 = tuple(float(c) for c in v2)
    rel = (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])
    vn = _dot(rel, n)
    ke0 = 0.5 * m1 * _dot(v1, v1) + 0.5 * m2 * _dot(v2, v2)
    if vn <= 0.0:
        return dict(v1=v1, v2=v2, J_n=0.0, J_t=0.0, closing=0.0, deform_J=0.0,
                    friction_J=0.0, KE_before=ke0, KE_after=ke0, hit=False, mu_r=0.0)
    mu_r = m1 * m2 / (m1 + m2)
    e = 0.0 if grip else float(e)
    Jn = (1.0 + e) * mu_r * vn
    vt = (rel[0] - vn * n[0], rel[1] - vn * n[1], rel[2] - vn * n[2])
    vtm = math.sqrt(_dot(vt, vt))
    if vtm > 1e-12:
        Jt = mu_r * vtm if grip else min(mu_r * vtm, mu * Jn)
        t = (vt[0] / vtm, vt[1] / vtm, vt[2] / vtm)
    else:
        Jt, t = 0.0, (0.0, 0.0, 0.0)
    imp = tuple(Jn * n[i] + Jt * t[i] for i in range(3))
    w1 = tuple(v1[i] - imp[i] / m1 for i in range(3))
    w2 = tuple(v2[i] + imp[i] / m2 for i in range(3))
    ke1 = 0.5 * m1 * _dot(w1, w1) + 0.5 * m2 * _dot(w2, w2)
    deform = 0.5 * (1.0 - e * e) * mu_r * vn * vn
    fric = max(ke0 - ke1 - deform, 0.0)
    return dict(v1=w1, v2=w2, J_n=Jn, J_t=Jt, closing=vn, deform_J=deform,
                friction_J=fric, KE_before=ke0, KE_after=ke1, hit=True, mu_r=mu_r)


def stiffness(m_kg):
    """The contact stiffness of a body of this mass: k goes as its length,
    m^(1/3) (u)."""
    return K_TORSO * (m_kg / REF_KG) ** (1.0 / 3.0)


def contact_area(m_kg):
    """The contact patch of a torso or shoulder, m2: m^(2/3) (u)."""
    return AREA_70 * (m_kg / REF_KG) ** (2.0 / 3.0)


def partition(k1, k2):
    """Springs in series carry the same force; the energy goes in the ratio of
    the compliances: body 1 takes (1/k1) / (1/k1 + 1/k2). Returns (s1, s2)."""
    c1, c2 = 1.0 / k1, 1.0 / k2
    return c1 / (c1 + c2), c2 / (c1 + c2)


def peak_force(mu_r, closing, k1, k2):
    """The peak contact force of the two springs in series with the closing
    energy mu_r v_n^2 / 2 all in them: F = v_n sqrt(k_eff mu_r)."""
    k = k1 * k2 / (k1 + k2)
    return closing * math.sqrt(k * mu_r)


# ── the Hof mechanisms, pure ─────────────────────────────────────────────────
def w0_of(h_m):
    """The pendulum's rate sqrt(g / l) for a centre of mass h_m up."""
    return math.sqrt(G / max(h_m, 0.25))


def ankle_cop_m(m_kg, state=1.0):
    """The furthest the ankle can move the centre of pressure off the middle of
    the foot: its torque over the weight, m (u). 120 Nm at 70 kg is 0.17 m, more
    than half a foot, so a healthy body uses the whole foot; a body with a
    quarter of the strength uses a quarter."""
    return ANKLE_NM_PER_KG * m_kg * max(state, 0.0) / (m_kg * G)


def flywheel_b_max(m_kg, h_body, w0, t_delay_s, state=1.0, arms_free=True):
    """The most an extrapolated-centre-of-mass excess b (m, past the base) the
    trunk and the arms can bring back, with the delay the reflex has. The sum
    of the flywheels' shifts (module doc), divided by exp(w0 t_delay)."""
    scale = (m_kg / REF_KG) * (h_body / REF_H) ** 2
    total = 0.0
    for name, f in FLY.items():
        if name == "arms" and not arms_free:
            continue
        I = f["I"] * scale
        a, wm = f["alpha"], f["w_max"]
        total += I * a / (m_kg * G) * (1.0 - math.exp(-w0 * wm / a))
    return total * max(state, 0.0) * math.exp(-w0 * max(t_delay_s, 0.0))


def push_force_N(m_kg, state=1.0, reaction_cap_N=math.inf):
    """The horizontal force a hand can put on a wall or a body: the arm's
    (a share of weight) or what the thing pushed on can hold, the smaller."""
    return min(ARM_PUSH_BW * m_kg * G * max(state, 0.0), reaction_cap_N)


def external_b_max(m_kg, h_hand_m, force_N):
    """The equivalent centre-of-pressure shift of a horizontal force at hand
    height: F h / (m g)."""
    return force_N * h_hand_m / (m_kg * G)


def holds_N(thing_kg, mu=HOLD_MU):
    """What a thing resists a push with, N: its weight times the friction."""
    return mu * thing_kg * G


# ── the world side ───────────────────────────────────────────────────────────
class BodyContactMixin:
    """Body-on-body contact for a World, behind the gate `body_contact` (ON for every
    world since 14bF, W15-Q8 (a)).

    Gate on: the three Hof mechanisms in `_overbalanced`, and `body_collide` as the
    one contact law (a shove, a tackle, a grapple, a blunt blow, a thrown or falling
    body on a person). Gate off: `body_collide` does nothing, the balance reflex is
    exactly as before (the base is the hull, the centre of pressure is free and
    instant, and only a step or a grab recovers) and a blow is joules into flesh.
    A class attribute (not an instance one) so a world restored from an older
    snapshot still has it and golden hashes nothing."""
    body_contact = True
    contact_book = None            # the joules of the contact law (see `_book`)

    def _cbook(self):
        """The contact ledger: every joule the contact law moved.
        KE_in = deform_J + friction_J + KE_kept at the collision; what is kept
        is later heat at the feet (foot_J) or goes into a fall (to_fall_J)."""
        if self.contact_book is None:
            self.contact_book = {"KE_in_J": 0.0, "deform_J": 0.0, "friction_J": 0.0,
                                 "KE_kept_J": 0.0, "foot_J": 0.0, "to_fall_J": 0.0,
                                 "paid_J": 0.0, "arm_J": 0.0, "ground_J": 0.0, "n": 0}
        return self.contact_book

    # the person's floor-plane velocity is the sim's own (`_v`, push_person)
    @staticmethod
    def _v3(p):
        v = p.get("_v") or (0.0, 0.0)
        return (float(v[0]), float(v[1]), 0.0)

    def _mass_cells(self, p):
        comp, sl = self._person_cells(p)
        if comp is None or not comp.any():
            return None, None, 0.0
        cells = self._claim_cells(p, comp, sl)
        com, kg = self._com_of(cells)
        return cells, com, kg

    def body_collide(self, a, b, v_a=None, normal=None, grip=False, payer=None,
                     m_a_eff=None, e=E_FLESH, mu=MU_BODY, m_b_eff=None,
                     area_m2=None, deposit_heat=True, striker_pid=None, hit_kw=None):
        """ONE contact: `a` (moving at `v_a`, m/s in the floor plane, default its
        own `_v`) meets person `b` (at its own `_v`).

        `a` is a person, or None for a FREE striker that is not on the lattice as
        a person (a thrown thing, a falling body, an arm in a swing): it then needs
        `v_a` (3-vector), `normal` and `m_a_eff`, and its reaction is not
        modelled (it is spent where it lands, as before).
        `m_a_eff` is the mass of `a` that is moving into the contact (a shove:
        the arm and the lean, exertion.KINDS; a tackle: the whole body, the
        default). When it is given, `a`'s feet take the reaction and `a` is not
        pushed back; the effective mass's leftover speed is spent in `a`.
        `m_b_eff` is the mass of `b` that answers (default: the whole body; a body
        held up by the ground along the normal answers with the ground, a big mass).
        `area_m2` is the contact patch (default: a trunk's); a fist is a voxel face.
        `payer` is the person whose muscle made the closing energy (the
        pusher, the tackler): the books are theirs (mechlaw._pay_strike), and
        the joules the muscle gave leave as heat in the contact cells, credited
        once. With no payer the energy came from motion nobody paid for
        (a fall, a slip) and is booked as mech_in by `deposit`.
        `striker_pid` and `hit_kw` name the actor and the cause of the hit event of
        a free striker (who threw it, the fall that caused it).
        `deposit_heat` False: the caller books the joules itself (a swing was
        billed tick by tick and is deposited when it ends); the ledger still
        records them.

        Returns the collide() dict (plus the injuries), or None with the gate
        off or when the bodies were not closing."""
        if not self.body_contact:
            return None
        free = a is None
        if free:
            if v_a is None or normal is None or m_a_eff is None:
                return None
            ca, com_a, ma = None, None, float(m_a_eff)
        else:
            ca, com_a, ma = self._mass_cells(a)
        cb, com_b, mb = self._mass_cells(b)
        if ma <= 0.0 or mb <= 0.0:
            return None
        if normal is None:
            d = (com_b[0] - com_a[0], com_b[1] - com_a[1])
            nrm = math.hypot(*d) or 1.0
            normal = (d[0] / nrm, d[1] / nrm, 0.0)
        va = tuple(v_a) + (0.0,) * (3 - len(v_a)) if v_a is not None else self._v3(a)
        vb = self._v3(b)
        m1 = float(m_a_eff) if m_a_eff is not None else ma
        m2 = float(m_b_eff) if m_b_eff is not None else mb
        r = collide(m1, va, m2, vb, normal, e=e, mu=mu, grip=grip)
        if not r["hit"]:
            return None
        bk = self._cbook()
        # the energy the muscle puts in is billed from its own books
        if payer is not None:
            put = 0.5 * m1 * _dot(va, va)
            bk["paid_J"] += put * self._pay_strike(payer, put, "contact")
        # who moves now: the struck body always; the mover too unless its feet
        # took the reaction (an arm's shove)
        w2 = r["v2"]
        src = -1 if payer is None else payer["pid"]
        if payer is None and not deposit_heat:
            src = -2                  # its joules are booked by whoever billed the strike
        if math.hypot(w2[0], w2[1]) > 1e-6:      # a body the ground answers for does not move
            self.push_person(b, w2[0], w2[1])
            b["_v_src"] = src
        kept = 0.5 * m2 * (w2[0] ** 2 + w2[1] ** 2)
        if m_a_eff is None:
            w1 = r["v1"]
            self.push_person(a, w1[0], w1[1])
            a["_v_src"] = src
            kept += 0.5 * ma * (w1[0] ** 2 + w1[1] ** 2)
            lost_in_a = 0.0
        else:
            w1 = r["v1"]
            lost_in_a = 0.5 * m1 * _dot(w1, w1)       # the arm stops: heat in the arm
        # the books at the collision
        # the part of the speed the bodies leave with that is VERTICAL is answered by the
        # floor (a person is held up by the ground; the push law moves them on the floor
        # plane only): its energy is heat now, booked as ground_J
        vert = 0.5 * m2 * w2[2] ** 2 + (0.5 * ma * r["v1"][2] ** 2 if m_a_eff is None else 0.0)
        heat = r["deform_J"] + r["friction_J"] + lost_in_a + vert
        bk["ground_J"] += vert
        bk["KE_in_J"] += r["KE_before"]
        bk["deform_J"] += r["deform_J"]
        bk["friction_J"] += r["friction_J"]
        bk["arm_J"] += lost_in_a
        bk["KE_kept_J"] += kept
        bk["n"] += 1
        # where the contact is
        cell_b = self._bc_nearest(self._trunk_cells(b, cb), com_a if not free else normal_point(com_b, normal))
        k2 = stiffness(mb)
        if free:
            cell_a = cell_b
            k1 = math.inf                    # a free striker is a rigid impactor here (u)
            s1, s2 = 0.0, 1.0
        else:
            cell_a = self._bc_nearest(self._trunk_cells(a, ca), com_b)
            k1 = stiffness(ma)
            s1, s2 = partition(k1, k2)
        area = float(area_m2) if area_m2 is not None else contact_area(min(ma, mb))
        F = peak_force(r["mu_r"], r["closing"], k1, k2) if not free else \
            r["closing"] * math.sqrt(k2 * r["mu_r"])
        r["F_peak_N"] = F
        r["injury"] = {}
        # heat: the joules lost go into the cells of the contact
        if deposit_heat:
            self._contact_heat(cell_a, cell_b, heat, s1, s2, payer)
        # the injury law: each body reads its share (absorbed = bruise; peak = bone)
        peak_E = 0.5 * r["mu_r"] * r["closing"] ** 2
        for who, other, cell, share, mass in ((a, b, cell_a, s1, ma), (b, a, cell_b, s2, mb)):
            if who is None or (m_a_eff is not None and who is a):
                continue                         # the pusher's own arm is not struck
            r["injury"][who["name"]] = self._contact_injury(
                who, other, cell, share * r["deform_J"], share * peak_E, area,
                k1 if who is a else k2, F, striker_pid, hit_kw)
        return r

    # ── the acts: which bodies meet, and with what ───────────────────────────
    def _toward(self, a, b):
        """Unit floor-plane normal from person a to person b, and the two centres."""
        _, com_a, ma = self._mass_cells(a)
        _, com_b, mb = self._mass_cells(b)
        if ma <= 0.0 or mb <= 0.0:
            return None
        d = (com_b[0] - com_a[0], com_b[1] - com_a[1])
        n = math.hypot(*d) or 1.0
        return (d[0] / n, d[1] / n, 0.0)

    def shove(self, a, b):
        """Person a shoves person b: the arm and the lean drive into the chest at
        the speed a's own muscle gives them (exertion.strike "shove"). One call of
        the contact law; the pusher's feet take the reaction."""
        from . import exertion as EX
        n = self._toward(a, b)
        if n is None:
            return None
        k = EX.strike(a, "shove")
        v = k["v_ms"]
        return self.body_collide(a, b, v_a=(n[0] * v, n[1] * v, 0.0), normal=n,
                                 payer=a, m_a_eff=k["m_kg"])

    def tackle(self, a, b, grip=False):
        """Person a runs into person b with the whole body at a's own running
        speed (the speed the legs' reserve allows). The same call; `grip` is a tackle
        that seizes (both end at one speed)."""
        from . import exertion as EX
        from .policy import WILL
        n = self._toward(a, b)
        if n is None:
            return None
        v = WILL["run_mps"] * EX.state_factor(a, "kick")
        return self.body_collide(a, b, v_a=(n[0] * v, n[1] * v, 0.0), normal=n,
                                 payer=a, grip=grip)

    def grapple(self, a, b):
        """Person a closes a hand on person b and holds: the same call with a grip (no
        bounce, no slide). A standing person grips with the arm and hand (the mass of
        an arm, 4 percent of the body, exertion.KINDS) at the speed a hand closes
        (HAND_CLOSE_MS), so a rescuer taking hold of a friend costs both about a
        joule; a person who RUNS at someone to seize them is a tackle with a grip
        (`tackle` with grip=True is the same call)."""
        from . import exertion as EX
        n = self._toward(a, b)
        if n is None:
            return None
        m = EX.KINDS["punch"]["mass_frac"] * EX.mass_of(a)
        v = HAND_CLOSE_MS
        return self.body_collide(a, b, v_a=(n[0] * v, n[1] * v, 0.0), normal=n,
                                 payer=a, grip=True, m_a_eff=m)

    def _contact_blow(self, b, q, cell, e, joules, a_hit, owner_pid, instrument, cause):
        """A moving body `b` (a swung arm or tool, a thrown or loosed thing, a body
        falling or toppling) strikes person `q` with `e` of its `joules`: the one
        contact law with the striker as a free mass at the speed it arrives with,
        not a joules-only blow. True when the contact law took it (the caller then
        skips the old blow), False when it did not apply (the striker is the
        struck body's own arm, or nothing was closing).

        What differs between the strikers is only the INPUTS: a swing is the
        inertia over the tip radius squared at the tip's speed (so 1/2 m v^2 is
        the swing's 1/2 I w^2), a flier its mass at its velocity, a falling body
        its mass at the speed its drop gave it. The part of the striker's mass
        that hit this person is the share of the cells that did (e / joules)."""
        import numpy as np
        from .materials import TICK_S
        if not self.body_contact or joules <= 0.0 or e <= 0.0:
            return False
        vox_m = 0.1 * self.scale
        owner = next((r for r in self.persons if r["name"] == b.get("owner")), None)
        if owner is q:
            return False
        c = np.asarray(cell, dtype=np.float64)
        paid_elsewhere = False
        if b.get("part") and b.get("Nm"):
            r = max((abs(c[b["axis"]] - b["pivot"]) + abs(c[2] - b["zb"])) * vox_m, vox_m)
            w = b["omega"] / max(TICK_S, 1e-9)
            m_eff, v = b["I"] / (r * r), w * r
            if owner is not None:
                _, com, _m = self._mass_cells(owner)
                src = np.asarray(com, dtype=np.float64) if com is not None else c
            else:
                src = c
            n = c - src
            paid_elsewhere = True            # billed tick by tick, deposited when the swing ends
        elif b.get("fly") and (b.get("vel") is None or b.get("masses") is None):
            return False                     # a body that carries no speed or mass has no contact to make
        elif b.get("fly"):
            vel = np.asarray(b["vel"], dtype=np.float64)
            v = float(np.linalg.norm(vel))
            m_eff = float(b["masses"].sum()) / 1000.0
            n = vel
            paid_elsewhere = b.get("by_pid") is not None     # a throw: the muscle was billed
        elif b.get("masses") is None or b.get("cells") is None:
            return False
        else:
            m_eff = float(b["masses"].sum()) / 1000.0
            v = math.sqrt(2.0 * joules / max(m_eff, 1e-9))
            n = c - np.asarray(b["cells"], dtype=np.float64).mean(axis=0)
        ln = float(np.linalg.norm(n))
        if ln < 1e-9 or v <= 0.0 or m_eff <= 0.0:
            return False
        n = n / ln
        m1 = m_eff * min(e / joules, 1.0)
        held_up = abs(float(n[2])) > 0.7           # the ground answers along a vertical blow
        r = self.body_collide(None, q, v_a=tuple(float(x) for x in v * n),
                              normal=tuple(float(x) for x in n), m_a_eff=m1,
                              area_m2=a_hit, deposit_heat=not paid_elsewhere,
                              m_b_eff=1e6 if held_up else None,
                              striker_pid=owner_pid,
                              hit_kw={"instrument": instrument, "cause": cause})
        return r is not None

    def _trunk_cells(self, p, cells):
        """The cells of the trunk (shoulder, chest, belly are what bodies meet
        with, not the reaching hand), else all of them."""
        parts = self._bone_parts(p, ["torso"]) if p.get("segs") else None
        if parts and len(parts[0]):
            return np_array(parts[0])
        return cells

    @staticmethod
    def _bc_nearest(cells, point):
        d = ((cells - np_array(point)) ** 2).sum(axis=1)
        return tuple(int(v) for v in cells[int(d.argmin())])

    def _contact_heat(self, cell_a, cell_b, J, s1, s2, payer):
        """Put the heat of the contact in the two bodies' contact cells (in the
        series ratio), booked once by `deposit`."""
        if J <= 0.0:
            return
        for cell, sh in ((cell_a, s1), (cell_b, s2)):
            if sh * J > 0.0:
                self.deposit(np_array([cell]), sh * J, actor=payer)

    def _contact_injury(self, q, other, cell, absorbed_J, peak_J, area, k_c, F,
                        striker_pid=None, hit_kw=None):
        """What this body's share does: the bruise and the head through
        physiology._blow (energy per area, armour, wound), and the bone
        through impact.blow_on_bone (force against the bone's own bending
        strength). Returns {"wound": ..., "bone": ..., "broke": bool}."""
        from . import injury as IJ
        from . import impact as IM
        out = {"E_abs_J": absorbed_J, "E_peak_J": peak_J, "broke": False, "bone": None}
        who = other["pid"] if other is not None else striker_pid
        box = self._defer("hits", actor=who, target=("person", q["pid"]),
                          place=cell, db=float(absorbed_J), text="", **(hit_kw or {}))
        with self.because(box):
            wound = self._blow(q, cell, absorbed_J, area)
        out["wound"] = wound
        bone = self._bone_at(q, cell)
        out["bone"] = bone
        if bone is not None and q["alive"]:
            try:
                bb = IM.blow_on_bone(IJ.sheet_of(q), bone, peak_J, area, 0.0, k_c, 0.0)
            except KeyError:
                bb = None                          # a part the anatomy has no member for
            if bb is not None:
                out["F_c"], out["F_bend"] = bb["F_c"], bb["F_bend"]
                if bb["broke"]:
                    out["broke"] = True
                    q["fractures"] = sorted(set(q.get("fractures", [])) | {bb["member"]})
        return out

    # ── the Hof mechanisms, asked of a body in `_overbalanced` ───────────────
    def _state_of(self, p, kind):
        from . import exertion as EX
        return EX.state_factor(p, kind)

    def _dynamic_recovery(self, p, own, com, kg, hull_dist_cells, direction, z_foot):
        """This body is moving (it was pushed) and its extrapolated centre of
        mass is `hull_dist_cells` past its base. Can the counter-rotating
        trunk and arms, or a hand on something that holds, bring it back
        (module doc)? Returns (recovered, how), `how` the mechanisms used."""
        vox_m = 0.1 * self.scale
        b = hull_dist_cells * vox_m
        h_com = max((float(com[2]) - z_foot + 0.5) * vox_m, 0.25)
        w0 = w0_of(h_com)
        zlo, zhi = int(own[:, 2].min()), int(own[:, 2].max())
        h_body = (zhi - zlo + 1) * vox_m
        t_d = self.reflex_delay_s(p, "balance")
        arms_free = not (p.get("held") or p.get("dragging"))
        st_arm = self._state_of(p, "shove")
        how = []
        m_fly = flywheel_b_max(kg, h_body, w0, t_d, st_arm, arms_free)
        room = m_fly
        if b <= room:
            return True, ("counter-rotation",)
        how.append("counter-rotation")
        # a hand on something that holds
        if arms_free:
            cap = self._reaction_cap(p, own, direction, zhi, zlo, vox_m, h_body)
            if cap is not None:
                F = push_force_N(kg, st_arm, cap[0])
                room += external_b_max(kg, cap[1], F)
                if b <= room:
                    return True, ("counter-rotation", "hand on " + cap[2])
        return False, tuple(how)

    def _reaction_cap(self, p, own, direction, zhi, zlo, vox_m, h_body):
        """What is within the arm's reach in the way the body leans, at hand
        height: (reaction cap N, hand height m, what). A wall, a heavy thing
        or another person: the cap is its weight times the friction it holds
        with (a fixed thing, one that the world holds up, caps at the arm's
        own strength)."""
        ax, s = direction
        zh = zlo + int(round(HAND_H_FRAC * (zhi - zlo)))
        sel = own[own[:, 2] == zh]
        if not len(sel):
            return None
        edge = int(sel[:, ax].max()) if s > 0 else int(sel[:, ax].min())
        reach = max(int(ARM_REACH_FRAC * h_body / vox_m), 1)
        mine = set(map(tuple, own.tolist()))
        nx, ny, nz = self.shape
        base = sel[sel[:, ax] == edge][0].tolist()
        for k in range(1, reach + 1):
            c = list(base)
            c[ax] += s * k
            if not (0 <= c[0] < nx and 0 <= c[1] < ny):
                return None
            t = tuple(c)
            if t in mine:
                continue
            if self.mat[t] == 0:
                continue
            cl = self._object_at(*t)
            if not len(cl):
                return None
            kgc = float(self.smass[tuple(cl.T)].sum()) / 1000.0
            if len(cl) >= 4000:
                cap = math.inf                      # the world: it holds
            else:
                cap = holds_N(kgc)
            other = self._person_at(t)
            what = "another person" if other is not None else "something solid"
            return cap, (zh - zlo + 0.5) * vox_m, what
        return None

    def _book_foot_loss(self, p, own, v_old, v_new):
        """The speed a pushed body had is spent at its feet (a step, or the
        slow settling): the kinetic energy it loses is heat in the soles,
        booked once. Only for a push the contact law made (`_v_src`)."""
        src = p.get("_v_src")
        if src is None or not self.body_contact:
            return
        com, kg = self._com_of(own)
        if com is None:
            return
        dE = 0.5 * kg * ((v_old[0] ** 2 + v_old[1] ** 2) - (v_new[0] ** 2 + v_new[1] ** 2))
        if dE <= 0.0:
            return
        zlo = int(own[:, 2].min())
        sole = own[own[:, 2] == zlo]
        nx, ny, nz = self.shape
        sole = sole[((sole >= 0) & (sole < (nx, ny, nz))).all(axis=1)]     # a body at the edge
        if not len(sole):
            sole = own[((own >= 0) & (own < (nx, ny, nz))).all(axis=1)][:1]
        if not len(sole):
            return
        if src == -2:                          # the swing that pushed it deposited every joule
            self._cbook()["foot_J"] += dE
            return
        payer = None if src < 0 else next((q for q in self.persons if q["pid"] == src), None)
        self.deposit(np_array(sole), dE, actor=payer)
        self._cbook()["foot_J"] += dE


def normal_point(com, normal):
    """A point one body-length behind the struck body's centre, against the way the
    striker comes (so the nearest trunk cell is the one facing the striker)."""
    return (com[0] - 2.0 * normal[0], com[1] - 2.0 * normal[1], com[2] - 2.0 * normal[2])


def np_array(x):
    import numpy as np
    return np.asarray(x)
