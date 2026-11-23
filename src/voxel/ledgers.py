"""The world as a value (snapshot/restore), the conservation totals and, since
wave 13, THE BOOKS: the one list of every ledger term (BOOKS below) and the one
road a joule takes into the world from an act (`deposit`)."""
import copy
import numpy as np
from .materials import (AIR, BLOOD, WOOD)

# THE BOOKS (wave 13, plan section 1). Every ledger term the wave adds to the
# two that always existed (`gone`, `shed`), each with
#   topic    the ONE topic that owns it, and so the only one allowed to write it
#   writers  the modules of that topic that write it
#   via      modules of other topics that reach it only through the owner's API
#   kind     mass (g), energy (J), volume (ml) or mass+energy
#   shared   True for the two legacy terms that many laws already write
#   parts    the sub-keys of a term that is a dict
#   sign     what a positive value means (from the topic specs; the owner lane
#            confirms it when its writer lands)
# This is the list the code stands on. tools/w13_manifest.py keeps a copy for
# the audit tool, and tests/test_ledger_books.py fails if the two differ, if a
# module that is not a writer writes a term, or if one act books both flows.
#
# A term is stored on the world under its own name, as a number or a dict (the
# owner writes it, on first use: an unwritten term does not exist yet), or
# served by a method of that name; `books()` reads them all, and an unwritten
# term reads 0.0 (a dict term: {}). NOTHING here adds a term to `total_mass`,
# `total_fluid` or `total_energy`: each lane that makes its term nonzero adds
# it there, with a closure proof, because the sign in which a term enters a
# total is that lane's to prove (the specs disagree about `work_out`, for one).
BOOKS = (
    dict(name="gone", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=True, parts=("species",), note="base18 term; T1 adds the per-species keys",
         sign="grams that LEFT the lattice (past the edge, or dropped by a law)"),
    dict(name="shed", topic="T1", writers=(), via=("airsink", "bodyheat", "outside", "slownet"),
         kind="energy", shared=True, parts=(),
         note="base18 term; sign: heat that LEFT the lattice",
         sign="joules that LEFT the lattice as heat"),
    dict(name="recast", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",),
         sign="signed grams: minus the old mix, plus the new pure grams, when a legacy law changes a material"),
    dict(name="quant", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",), sign="signed grams: rounding of a mix to its integer weights"),
    dict(name="f32", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",), sign="signed grams: what float32 storage took of an exact mass"),
    dict(name="squeeze", topic="T1", writers=("mixtable",), via=(), kind="mass",
         shared=False, parts=("species",), sign="grams clamped away when a cell was over-filled"),
    dict(name="records_owned", topic="T2", writers=("record", "lots", "vessel"), via=(),
         kind="mass", shared=False, parts=(),
         sign="grams, ml and joules the records own (held in records, not in cells)"),
    dict(name="outside_booked", topic="T8", writers=("outside", "outside_world"), via=(),
         kind="energy", shared=False,
         parts=("rad", "edge", "air_film", "air_far", "shift", "top", "ground",
                "ref_shift", "mrt_resid"),
         sign="joules, shed's sign (heat that LEFT), split by cause"),
    dict(name="boiled_J", topic="T4", writers=("mixvle", "mixtures"), via=(), kind="energy",
         shared=False, parts=(), sign="joules the steam took (latent heat of boiling)"),
    dict(name="melted_J", topic="T4", writers=("mixphase", "mixtures"), via=(), kind="energy",
         shared=False, parts=(), sign="joules melting took in"),
    dict(name="frozen_J", topic="T4", writers=("mixphase", "mixtures"), via=(), kind="energy",
         shared=False, parts=(), sign="joules freezing gave out"),
    dict(name="dissolved_J", topic="T4", writers=("mixsol", "mixtures"), via=(), kind="energy",
         shared=False, parts=(), sign="signed joules of the heat of solution"),
    dict(name="made", topic="T6", writers=("rxn",), via=("reactions",), kind="mass+energy", shared=False,
         parts=("water", "latent"),
         sign="signed grams of water, and joules of latent heat, a reaction row made or ate"),
    dict(name="mech_in", topic="T11", writers=("mechanisms",), via=(), kind="energy",
         shared=False, parts=(), note="only for releases with NO actor (one-flow rule)",
         sign="joules a release with no actor put into the world"),
    dict(name="pool", topic="T7", writers=("bodyheat",), via=(), kind="energy",
         shared=False, parts=(), sign="joules in the body heat pool, above the 20 C reference"),
    dict(name="stores", topic="T7", writers=("bodyheat", "energetics", "digestion"), via=(),
         kind="energy", shared=False, parts=(), sign="joules in the chemical stores of persons"),
    dict(name="work_out", topic="T7", writers=("energetics",), via=("fatigue", "stairs", "mechanisms"),
         kind="energy", shared=False, parts=(),
         note="actor work -> work_out; an act never also books mech_in",
         sign="joules of mechanical work actors delivered to the world"),
    dict(name="bh_injected", topic="T7", writers=("bodyheat",), via=(), kind="energy",
         shared=False, parts=("body_warmup",), note="the warm-up injection",
         sign="joules a named injection put into a body"),
    dict(name="pooled_ml", topic="T10", writers=("vasculature",), via=("injury",), kind="volume",
         shared=False, parts=(), sign="ml of blood pooled out of circulation"),
    dict(name="edema_ml", topic="T10", writers=("skinburn", "vasculature"), via=(),
         kind="volume", shared=False, parts=(), sign="ml of fluid moved into oedema"),
    dict(name="layer_mass", topic="T3", writers=("layers",), via=(), kind="mass",
         shared=False, parts=("led_in", "led_out"),
         sign="grams in the surface layers: what came in, what went out"),
    dict(name="layer_enthalpy", topic="T3", writers=("layers", "layer_laws"), via=(),
         kind="energy", shared=False, parts=(), sign="joules in the surface layers"),
    dict(name="slow_residual", topic="T5", writers=("slowtime", "slownet"), via=(),
         kind="energy", shared=False, parts=(),
         sign="signed joules and grams a skip could not place"),
    dict(name="phase_J", topic="T9", writers=("heathist_world",), via=(), kind="energy",
         shared=False, parts=(),
         sign="joules held in the steel's state (positive = stored as austenite or martensite enthalpy); total_energy() + phase_J is conserved"),
)
BOOK_NAMES = tuple(b["name"] for b in BOOKS)

# THE GHOST WORLD (S5). `self.matter` holds `matter.w`, a reference straight
# back to this World — the seam needs to read fields on the world it moves
# matter for. A plain `deepcopy(self.__dict__)` does not know that: it walks
# into `matter.w`, finds an object `deepcopy` has not seen before (the live
# World is not itself a key of the dict being copied), and copies THE WHOLE
# WORLD A SECOND TIME. The snapshot comes back carrying a second, unreachable
# World hanging off `matter.w` — a ghost, never stepped, never read, just
# sitting there doubling the memory a snapshot costs and waiting to be
# accidentally read through `snap["matter"].w`.
#
# The fix is `deepcopy`'s own memo: told `id(self) -> _THIS`, it substitutes
# the sentinel wherever it meets `self` again instead of copying it. `_THIS`
# is a module-level object so its id is stable and it carries no state of its
# own. `restore` runs the memo the other way, `id(_THIS) -> self`, so the
# world the snapshot is restored INTO is the one `matter.w` ends up pointing
# at — which is also how a snapshot taken on one World restores cleanly onto
# a second one (`w2.restore(snap)` gives `w2.matter.w is w2`).
_THIS = object()


class LedgersMixin:

    # ── a world as a value ───────────────────────────────────────────────────
    # `event_sink` is an external hook (never a plain value), and `_epoch`
    # exists precisely so that restoring an old snapshot does not: it is
    # bumped on every restore, never carried BY one, so an event id that
    # repeats after a rewind still comes from a different epoch (3.4).
    _NOT_STATE = ("policy", "event_sink", "_epoch")

    def snapshot(self):
        """Everything this world IS, as a value you can put back later.

        The state is a handful of arrays and a few plain lists, so a copy is
        cheap and — the point — EXACT: restore one and the next tick is the
        tick that would have followed. Three different things want it.

        A guard that compares two runs only at the END can say a divergence
        happened; one that can start both from the SAME state says which change
        caused it, with the build of the world removed as a variable. That is
        what makes it safe to rewrite how the laws are applied at all.

        A table wants to rewind and try the other thing. And a harvest of
        decisions is only honest if the run behind it can be played again.

        The policy is left out on purpose: it is a mind, not a world, and it may
        one day be a model whose weights dwarf the lattice. `matter.w`, which
        points back at this very World, is stood in for by `_THIS` rather than
        copied (see the note above `_THIS`)."""
        state = {k: v for k, v in self.__dict__.items()
                 if k not in self._NOT_STATE}
        return copy.deepcopy(state, {id(self): _THIS})

    def restore(self, snap):
        """Put a snapshot back, and keep whatever mind is in place now.

        Copied on the way in as well as the way out, so one snapshot can be
        restored as many times as you like — which is the whole use of it.
        `_THIS` is swapped for the World being restored INTO (`self`), so
        `matter.w` ends up pointing at the live world, not at a ghost.

        AND THE MIND IS TOLD. It was kept, so it still holds whatever it was
        waiting on in the world that has just been put away: a question asked
        at a tick that now has not happened. A policy with an `on_restore`
        hook (3.5) is called once the world is back, to drop those; one
        without is left alone, as it always was."""
        state = copy.deepcopy(snap, {id(_THIS): self})
        for k in [k for k in self.__dict__ if k not in self._NOT_STATE]:
            del self.__dict__[k]
        self.__dict__.update(state)
        self._epoch = getattr(self, "_epoch", 0) + 1
        told = getattr(getattr(self, "policy", None), "on_restore", None)
        if told is not None:
            told(self)

    # ── totals (the conservation the tests watch) ────────────────────────────
    def total_fluid(self, f):
        """Every millilitre of that fluid there is: in cells, in parcels still
        in flight, in bodies mid-topple, and past the edge of the world."""
        # NOT `~moving`: that is the rule for MASS, where every cell the seam
        # draws into holds the seam's own matter. Fluid ownership is finer.
        # A vat's hollow holds liquid the lattice still owns while the glass
        # around it has become particles, and the falling vat draws solid
        # over that hollow — so `moving` there means "a solid of ours passed
        # through", not "the liquid is ours". Excluding it hid up to all
        # 500 ml of a vessel's contents for whichever ticks a particle
        # happened to cover it. The seam owns exactly the fluid it wrote.
        return float(self.fvol[(self.fl == f)
                               & ~self.matter.drawn_fluid].sum()) \
            + sum(d[6] for d in self.drops if d[7] == f) \
            + sum(float(b["fvols"][b["fls"] == f].sum()) for b in self.bodies) \
            + self.matter.fluid(int(f)) \
            + (sum(q.get("blood_ml", 0.0) for q in self.persons)
               if f == BLOOD else 0.0) \
            + self.gone.get(("fluid", int(f)), 0.0)

    def total_mass(self, m):
        """Every gram of that material there is.

        ONE ANSWER, COUNTING EVERYWHERE IT CAN BE. Matter in this sim lives in
        three places — cells, bodies that have left the grid to topple or fly,
        and the tally of what went past the edge — and a total that reads only
        the first is not a total. It quietly said a swung axe had ceased to
        exist for as long as it was in the air.

        This had gone wrong in the small: `total_wood` counted the tally and
        nothing else did, which is worse than none of them counting it, because
        a conservation check that is right for one material and wrong for the
        rest fails at whichever moment is least convenient."""
        # `moving` cells are particle matter DRAWN onto the lattice for the
        # laws that read it; the grams are the solver's and are counted there.
        # Summing both doubled every promoted body — measured, flesh at
        # exactly 2x — which is the one mistake a seam exists to not make.
        return (float(self.smass[(self.mat == m) & ~self.moving].sum())
                + sum(float(b["masses"][b["mats"] == m].sum())
                      for b in self.bodies)
                + self.matter.mass().get(int(m), 0.0)
                + self.gone.get(int(m), 0.0))

    def total_wood(self):
        return self.total_mass(WOOD)

    def total_energy(self):
        """Every joule this world has had: what is in it, what is riding on
        bodies in flight, what went past the edge, and what it has radiated
        away. The only form in which "energy is conserved" is a question you
        can put to the sim and get a straight answer to."""
        return (float(self.E[~self.moving].sum())
                + sum(float(b["Es"].sum()) for b in self.bodies)
                + self.matter.energy()
                + self.gone.get("E", 0.0) + self.shed
                + (self.records_owned_E() if self.records is not None else 0.0))

    def total_lost(self):
        """Every gram that has gone past the edge of the world. Zero in any
        scene nobody is throwing things out of, and the only honest way to
        write a conservation check in one that is."""
        return sum(v for k, v in self.gone.items()
                   if isinstance(k, int) and k != AIR)

    # ── THE BOOKS (wave 13) ──────────────────────────────────────────────────
    # The two books an act can move. They are declared here, at zero, so that
    # every world can read them; the first booking makes an instance value
    # (and a `restore` takes it away again). Only `mechanisms.deposit` writes
    # them (see BOOKS).
    mech_in = 0.0
    work_out = 0.0

    def books(self):
        """Every term of BOOKS as it stands now: {name: value}. A term nobody
        has written reads 0.0, and a term that has parts reads {}; a dict is
        returned as a copy. A term is stored on the world under its own name
        (a number or a dict) or served by a method of that name."""
        out = {}
        for t in BOOKS:
            v = getattr(self, t["name"], None)
            if callable(v):
                v = v()
            if v is None:
                v = {} if t["parts"] else 0.0
            out[t["name"]] = dict(v) if isinstance(v, dict) else v
        return out

    def _heat_cells(self, cells, J):
        """Put J joules of heat into `cells` (an (n, 3) array or an index tuple of
        three arrays), shared equally. A cell named twice (a door one cell tall
        has three hinges at two heights) takes a share each time: numpy's `+=`
        on repeated indices would keep only one, and the joules would be booked
        and never stored."""
        if isinstance(cells, np.ndarray):
            idx = tuple(cells.reshape(-1, 3).T.astype(np.int64))
        else:
            idx = tuple(np.asarray(a, np.int64) for a in cells)
        n = len(idx[0])
        if n:
            np.add.at(self.E, idx, self.E.dtype.type(J / n))

    @staticmethod
    def _debit(actor, J):
        """Take J joules from the actor's stores. The stopgap until the body
        heat topic (T7) has stores: the work is written on the person."""
        actor["_work_J"] = float(actor.get("_work_J", 0.0)) + J

    def deposit(self, cells, J, actor=None):
        """THE ONE-FLOW RULE. Turn J joules into heat in `cells` and book them
        ONCE: with an `actor` (a person's dict) they were delivered by that
        person, who pays for them, and they are credited to `work_out`; with
        none they came from nowhere that is booked (a spring let go, a door
        slamming from its stored swing, a burnt jamb) and go to `mech_in`. It
        returns the name of the book, or "none" for no joules (a negative
        amount, which is the rounding of a subtraction, counts as none; NaN is
        a bug upstream and raises). An act books one flow or the other, never
        both: `mechanisms.deposit` is the only writer of both books, and
        tests/test_ledger_books.py fails if any other function writes both."""
        J = float(J)
        if J != J:
            raise ValueError("a deposit of NaN joules")
        J = max(J, 0.0)
        if J == 0.0:
            return "none"
        from . import mechanisms                  # the one writer of the books
        return mechanisms.deposit(self, cells, J, actor=actor, heat=self._heat_cells,
                                  debit=self._debit)
