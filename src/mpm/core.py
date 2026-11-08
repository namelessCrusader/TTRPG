"""MATTER THAT IS NOT ON THE GRID.

The voxel sim can say how MUCH is in a cell (`smass`) and how concentrated a
blow is (`edge`), and it cannot say WHERE INSIDE the cell anything sits. So a
plank that fell on its side is identical to one that did not, a melting bar
holds its full span until the instant it vanishes, and a shattered pane breaks
into pieces exactly as big as the pane's own cells. Those are not three bugs.
They are one missing thing, and no number of extra per-cell fields supplies it:
a cell with a damage field, an offset field and a facing field is a bad
particle assembled from three worse ones, each needing its own law to talk to
the other eleven.

So: particles carry the matter, and the lattice becomes scratch paper for the
momentum solve. That is the Material Point Method, and the important half of
the sentence is the second one — MPM KEEPS THE GRID. Fire, conduction, smoke,
oxygen, fluid flow, acid, sight and every mind in the world go on reading the
same lattice they read today. Nothing that works is thrown away.

The method here is MLS-MPM (Hu et al., "A Moving Least Squares Material Point
Method with Displacement Discontinuity and Two-Way Rigid Body Coupling", ACM
TOG / SIGGRAPH 2018), ported to 3-D numpy from the author's 88-line reference
(github.com/yuanming-hu/taichi_mpm, MIT) and the Taichi examples it grew into
(github.com/taichi-dev/taichi, Apache-2.0). MLS rather than plain MPM because
it folds the force integral into the same affine matrix the velocity transfer
already carries, so one scatter does both and there is no gradient term to get
subtly wrong.

Units are SI throughout — metres, kilograms, seconds — because the material
data is SI and every conversion is somewhere for a factor to hide. The voxel
sim's `vox_m` is the grid spacing `dx`, so a particle at x=0.34 m sits a third
of the way into the fourth 10 cm cell and both representations agree about
where it is.
"""

import numpy as np

from . import kernels as _k

# Grid axes match the voxel sim: z is UP. Gravity is the only body force here;
# buoyancy, wind and the pneumatic field stay on the lattice where they live.
GRAVITY = np.array([0.0, 0.0, -9.81], np.float64)

# How far a CRACKED piece may be strained before the strain stops counting.
# Small, because a shard is hard; non-zero, because without it nothing bounds
# the deformation gradient of matter that carries no stress.
DMG_YIELD = 0.02

# How many cells past a particle's stencil a wall can matter to it, in cells.
# The distance field is only ever read near a surface the particle is touching,
# so a wall further off than this cannot change how anything moves. The seam
# watches the lattice this far around the particles and no further.
SDF_MARGIN = 4

# How many cells further the distance field's window reaches so that a
# particle up to SDF_MARGIN deep inside a wall reads exact distances around
# it, not only at it (see `MPM._near_walls`).
DEEP_READ = 3

# How far past what the particles need the distance field's window reaches, in
# cells, so that a moving body does not rebuild it every substep. Any value
# gives the same answer; this one only decides how often it is paid for.
SDF_SLACK = 6

# How many grid nodes a separate ISLAND must save before it is worth having
# (see `MPM._islands`). Any value gives the same answer: two groups that share
# no node never meet in a sum, so solving them on one grid or on two adds the
# same numbers in the same order. It only decides what is cheaper. Measured on
# two 512-particle blocks (build/X6): a second island costs about half a
# millisecond a substep (finding it, its own grid, its own distance field, its
# own calls), and an empty node in a shared window about 25 ns, so a split
# starts to pay at twenty to thirty thousand nodes left out. A pane that
# shattered into shards a cell apart stays one window; two things at the ends
# of a hall do not.
ISLAND_NODES = 32768

# density kg/m^3, Young's modulus Pa, Poisson ratio, yield strain (None = no
# plastic flow, the thing just breaks or does not), internal damping as a
# fraction of critical, tensile strength Pa (None = not brittle, it yields).  The densities are the voxel sim's own SOLID table so
# the two agree about what a kilogram of wood is; E and nu are ordinary
# handbook values.
#
# `damp` is not a fudge for the solver, it is a property matter has: drop a
# wooden block and it thuds, because the impact goes into heat and sound and
# does not come back. Pure elasticity has NO way to lose energy, and measured
# with none, the block bounced higher the stiffer it got and was still in the
# air a second and a half later. Timber is heavily damped, glass and stone
# ring, lead barely bounces at all.
# THE LAST COLUMN IS HOW MUCH OF ITS SHAPE A THING KEEPS, and it is the
# answer to a measurement, not a taste. Softening every modulus by `stiff`
# is what let any of this run at all, and it is also why a felled trunk
# drooped like a hose: wood at a ten-thousandth of its modulus IS rubber.
# Measured on a notched tree, raising the softening a hundredfold halved
# the trunk's bow and cost TWENTY TIMES the time — because the substep
# count goes as the wave speed, which goes as the square root of the
# modulus. Stiffness bought that way is unaffordable and, at 2x for 20x,
# barely bought.
#
# Projecting each bound cluster's motion onto RIGID motion buys it for
# nothing instead (see `_hold_shape`; the idea is shape matching, Müller et
# al. SIGGRAPH 2005, applied to velocities rather than positions). It is a
# projection, not a force, so there is no wave and no step limit — a rigid
# trunk costs what a rubber one did. 1.0 is a thing that does not bend;
# 0.0 is a thing that does not care.
MAT = {
    "wood":  (600.0,  1.1e10, 0.35, None,   1.00, 1.0e7,  1.00),
    "stone": (2700.0, 5.0e10, 0.25, None,   0.50, 8.0e6,  1.00),
    "iron":  (7800.0, 2.0e11, 0.29, 2.0e-3, 0.40, None,   0.90),
    "glass": (2500.0, 7.0e10, 0.22, None,   0.30, 5.0e7,  1.00),
    "flesh": (1000.0, 1.0e6,  0.45, 5.0e-2, 4.00, None,   0.05),
    "lead":  (11340.0, 1.6e10, 0.44, 4.0e-4, 3.00, None,  0.30),
    # FOLIAGE AS A SPRINGY MASS. A canopy is twigs and leaves, and as a whole
    # it holds its shape and springs back; it does not flow. The first row
    # here was "soft plastic", 1e8 with a 5% yield, and after softening that
    # was ten kilopascals with permanent flow: a 2.75 m tree that was four
    # fifths canopy by cell count DEFLATED to thirteen voxels tall instead of
    # falling over, its trunk leaning five cells inside the slump, and the
    # numbers (top at z=11, centre moved +5) read as a topple until the
    # frame was looked at. Stiff enough to hold under its own height and no
    # yield. Damped like wood, not like lead: at 3.0 the canopy's VISCOUS
    # step limit (1.3e-4 s) was the tightest in the scene and set the pace
    # for everything — 215 substeps a tick, 2.5x the wood's own, and a
    # fifteen-minute scene became one that timed out at twenty-eight. The
    # softest, most-damped material in a scene is the bill for all of it.
    # A CANOPY IS TETHERED, NOT RIGID. Leaves have no tensile strength here,
    # which used to mean they could not break — and in MPM a thing that
    # cannot break still comes apart, because two particles that drift more
    # than a cell apart simply stop sharing grid nodes and there is nothing
    # left to pull them back. Bound into the tree's cluster at 0.15 they
    # hang on and still flutter, and having no strength now means the bond
    # never parts: leaves do not fall off a falling tree.
    "leaf":  (200.0,  2.0e9,  0.30, None,   1.00, None,   0.15),
    # THESE THREE ROWS HAD SIX FIELDS, and every reader unpacks seven, so a
    # charred stump or a tin ingot that toppled raised ValueError inside
    # `promote` (design Q4(d); docs/realism-references.md G6). Ash is a heap
    # that keeps no shape, char is still a stiff stick of carbon, and tin
    # holds its shape about as iron does.
    "ash":   (150.0,  1.0e8,  0.30, 5.0e-2, 3.00, None,   0.00),
    "char":  (300.0,  1.0e9,  0.30, 2.0e-2, 2.00, None,   1.00),
    "tin":   (7300.0, 5.0e10, 0.33, 2.0e-3, 0.40, None,   0.90),
    # ROPE goes to bonds (src/bonds), which part it link by link; a cell of
    # it alone, or any rope with bonds turned off, comes here as a limp,
    # damped mass that keeps hardly any shape (its stretch modulus is the
    # bonds' 12 mm manila; it has no strength to part at).
    "rope":  (885.0,  9.4e8,  0.40, None,   1.00, None,   0.05),
}

# The last field is TENSILE STRENGTH in pascals, straight out of a handbook —
# and getting to a number that could come out of a handbook took three wrong
# ones first, so the reasoning is written down.
#
# The first criterion was a critical STRAIN, and strain is exactly what
# softening destroys: strain = stress / E, so dividing E by ten thousand
# multiplies every strain by ten thousand and no threshold means anything any
# more. It was mis-scaled twice in one session — once as 1/E when the static
# case was in mind, once again when the impact case needed 1/sqrt(E) — and the
# value that finally worked, 3e-2, was fitted to a sweep and stood for nothing.
#
# Stress is the better variable because stress is LOAD-DETERMINED. A beam
# holding up a weight carries the same internal stress whatever its modulus.
# So the threshold can be a real material property again.
#
# What softening then costs, and the scaling that pays it: an impact delivers
# sigma = rho c v = sqrt(E rho) v, which goes as sqrt(E). Soften E by `stiff`
# and scale strength by sqrt(stiff) alongside it, and
#
#     sigma / strength  =  sqrt(E rho) v / strength
#
# with `stiff` cancelling exactly. THE SPEED AT WHICH A THING BREAKS IS
# PRESERVED. That is the whole justification, and it is derived rather than
# fitted, which is the difference that matters.
#
# The bill arrives on the static side, where stress does not scale: the margin
# between sitting still and breaking shrinks as sqrt(stiff). For glass at
# stiff=1e-4 that is 14 kPa of self-weight against 500 kPa of scaled strength,
# a factor of 36, which holds. At stiff=1e-6 it would not.


# EVERY ARRAY THAT HOLDS ONE ENTRY PER PARTICLE, in one table: its name, its
# type, the shape of one particle's entry, and what a new particle starts with
# ("eye" is the 3x3 identity). `MPM.__init__` makes them in this order and
# `_append` grows every one of them from here, so a field added to the table
# cannot be forgotten by the code that adds particles. `keep`, which drops
# particles, walks the same list.
PARTICLE_FIELDS = (
    ("x",        np.float64, (3,),   0.0),    # position, metres
    ("v",        np.float64, (3,),   0.0),    # velocity, m/s
    ("C",        np.float64, (3, 3), 0.0),    # affine velocity: MLS-MPM's B/D term
    ("F",        np.float64, (3, 3), "eye"),  # deformation gradient: the FACING and
    ("Jp",       np.float64, (),     1.0),    #   the stretch, in one matrix
    ("m",        np.float64, (),     0.0),    # kg
    ("vol0",     np.float64, (),     0.0),    # m^3 at rest
    ("mu",       np.float64, (),     0.0),    # Lame parameters, already softened
    ("lam",      np.float64, (),     0.0),
    ("yld",      np.float64, (),     0.0),    # yield strain; 0 = perfectly elastic
    ("eta",      np.float64, (),     0.0),    # viscosity, Pa.s: where energy GOES
    ("strength", np.float64, (),     0.0),    # tensile stress it parts at, Pa;
                                              #   0 = not brittle, it yields instead
    ("dmg",      np.bool_,   (),     False),  # has it? ONCE TRUE, ALWAYS TRUE: a
                                              #   crack does not heal
    ("kind",     np.int32,   (),     0),      # index into MPM.kinds
    # WHAT A VOXEL CARRIED RIDES WITH THE PARTICLE. Energy is what the heat
    # seam exchanges through the lattice; fluid is held moisture (a fresh leaf)
    # or a vessel's contents; edge is a manufactured working edge. All in the
    # voxel sim's own units so the ledgers add up across the seam without a
    # conversion anyone can get wrong.
    ("E",        np.float64, (),     0.0),    # joules above ambient
    ("fl",       np.int16,   (),     0),      # fluid id, 0 = none
    ("fvol",     np.float64, (),     0.0),    # its volume
    ("fpot",     np.float64, (),     0.0),    # its potency
    ("edge",     np.float64, (),     0.0),    # m^2 a blow concentrates into
    ("vmat",     np.int16,   (),     0),      # the VOXEL material id, for going back
    ("cid",      np.int32,   (),     -1),     # which bound cluster, -1 = loose
    ("x0",       np.float64, (3,),   0.0),    # ...and where in its REST shape
    ("rig",      np.float64, (),     0.0),    # how hard it is held to that shape
    # WHICH THING IT IS, and how it got here. `eid` is the thing the cell it
    # came from belonged to (the lattice's own `eid`, 0 = no thing); `batch`
    # is the serial of the `promote` call that brought it in (-1 = it was put
    # here directly), so a cause stored against a promote can find its matter.
    ("eid",      np.int32,   (),     0),
    ("batch",    np.int32,   (),     -1),
    # ASLEEP OR NOT. -1 is awake; any other value is the dormant record that
    # holds it (`Matter._naps`). A sleeping particle is still drawn and still
    # handed what the lattice did to it, but the solver does not carry it
    # (`set_aside`). The solver itself never reads this: the seam decides.
    ("nap",      np.int32,   (),     -1),
)

# OPTIONAL PARTICLE FIELDS (wave 13). Matter that wave 13 adds to a voxel rides
# with the particle, like `E` and `fvol` above: the mix id of its solid
# (`scomp`) and of its fluid (`fcomp`) (T1), its damage as a float (`dscal`,
# T9; `dmg` is already the bool "has it cracked"), the grams of ice in it
# (`ice_g`, T6). They are NOT in PARTICLE_FIELDS: an array that does not exist
# costs nothing and cannot change a bit. `MPM.ensure_opt(name)` makes one (all
# zeros), and from then on `_append`, `keep`, `set_aside` and `take_back` carry
# it, because they walk `MPM.opt_fields()`, which lists the ones that exist.
PARTICLE_OPT = (
    ("scomp",  np.uint32,  (), 0),    # MixTable row of the solid, 0 = pure
    ("fcomp",  np.uint32,  (), 0),    # MixTable row of the fluid, 0 = pure
    ("dscal",  np.float32, (), 0.0),  # damage, 0 .. 0.999
    ("ice_g",  np.float32, (), 0.0),  # grams of ice held in it
)


def _box(lo, hi):
    """The slices of the cells from `lo` up to (not including) `hi`."""
    return tuple(slice(int(lo[a]), int(hi[a])) for a in range(3))


# ----------------------------------------------- 3x3 algebra in a stated order
#
# THE SOLVER'S SMALL LINEAR ALGEBRA IS WRITTEN OUT HERE, one multiply and one
# add at a time, so that the compiled substep (kernels.py) can do the very
# same operations and get the very same bits. LAPACK's `svd`, `eigvalsh` and
# `solve` iterate until they are satisfied and are free to order their sums
# as they like; `@` on a stack of 3x3 matrices goes to BLAS, which may fuse a
# multiply and an add into one rounding; a numpy sum of three numbers adds
# them in an order of its own choosing. None of that can be matched by a loop.
# So each one is replaced by a fixed recipe: a FIXED number of Jacobi sweeps
# (no "until it converges"), and every sum of three as (a + b) + c. The
# answers agree with LAPACK to about 1e-15 (see SVD_SWEEPS), which is not the
# same bits, so this was a change of definition, made once.

# How many sweeps of three rotations each. Measured on every scene in
# tests/test_mpm.py (39 million matrices, against the same algorithm run to
# 12 sweeps): after 3 sweeps the turn R was still 1.5e-7 out and the top
# eigenvalue 1e-9; after 4 both were at rounding (2e-15 and 0), and against
# LAPACK the singular values, R and the eigenvalue agreed to 6e-15, with no
# crack decided differently. Jacobi converges quadratically, so one sweep
# more than enough is the margin: 5.
SVD_SWEEPS = 5
EIG_SWEEPS = 5


def _dot3(a, b):
    """Row by row dot products of two (..., 3) arrays: (a0 b0 + a1 b1) + a2 b2."""
    t = a * b
    return (t[..., 0] + t[..., 1]) + t[..., 2]


def _mm(A, B):
    """A @ B for stacks of 3x3 matrices, each entry summed as (k0 + k1) + k2."""
    return (A[:, :, 0, None] * B[:, None, 0, :] + A[:, :, 1, None] * B[:, None, 1, :]) \
        + A[:, :, 2, None] * B[:, None, 2, :]


def _mm_t(A, B):
    """A @ B^T for stacks of 3x3 matrices, in the same order as `_mm`."""
    return (A[:, :, None, 0] * B[:, None, :, 0] + A[:, :, None, 1] * B[:, None, :, 1]) \
        + A[:, :, None, 2] * B[:, None, :, 2]


def _cross(a, b):
    """Row by row cross products of two (n, 3) arrays."""
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                     a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], -1)


def _rot(app, aqq, apq):
    """The Jacobi rotation (c, s) that zeroes apq between two diagonal
    entries app and aqq, taking the smaller angle; none (c = 1, s = 0) where
    apq is already zero. An overflowing ratio rounds the angle to zero,
    which is what it is to sixteen digits."""
    nz = apq != 0.0
    th = (aqq - app) / np.where(nz, 2.0 * apq, 1.0)
    t = np.where(th >= 0.0, 1.0, -1.0) / (np.abs(th) + np.sqrt(1.0 + th * th))
    t = np.where(nz, t, 0.0)
    c = 1.0 / np.sqrt(1.0 + t * t)
    return c, c * t, t


def _svd3(F):
    """U, S, V with F = U diag(S) V^T, for a stack of 3x3 matrices: S >= 0
    and largest first, U and V orthonormal, as LAPACK gives them.

    ONE-SIDED JACOBI (Hestenes): turn the columns of F, two at a time, until
    every pair is at right angles, and keep the turns in V. Then the columns
    of F V are U's columns scaled by S. U's first two columns are the first
    two of those, made unit (the second made square to the first), and the
    third is their cross product, flipped to point along the third column.
    That last step gives a U even where F is flat, as LAPACK does: a column
    with nothing left in it has no direction of its own to give."""
    B = F.copy()
    V = np.zeros_like(F)
    V[:, 0, 0] = V[:, 1, 1] = V[:, 2, 2] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(SVD_SWEEPS):
            for p, q in ((0, 1), (0, 2), (1, 2)):
                bp, bq = B[:, :, p], B[:, :, q]
                c, s, _t = _rot(_dot3(bp, bp), _dot3(bq, bq), _dot3(bp, bq))
                c, s = c[:, None], s[:, None]
                B[:, :, p], B[:, :, q] = c * bp - s * bq, s * bp + c * bq
                vp, vq = V[:, :, p], V[:, :, q]
                V[:, :, p], V[:, :, q] = c * vp - s * vq, s * vp + c * vq
    S = np.sqrt(np.stack([_dot3(B[:, :, k], B[:, :, k]) for k in range(3)], -1))
    # largest first: three compare-and-swaps, a tie left in place
    for p, q in ((0, 1), (1, 2), (0, 1)):
        sw = S[:, q] > S[:, p]
        if sw.any():
            for A in (B, V):
                ap, aq = A[sw, :, p].copy(), A[sw, :, q].copy()
                A[sw, :, p], A[sw, :, q] = aq, ap
            sp, sq = S[sw, p].copy(), S[sw, q].copy()
            S[sw, p], S[sw, q] = sq, sp
    s0 = S[:, 0]
    u0 = np.where((s0 > 0.0)[:, None], B[:, :, 0] / np.where(s0 > 0.0, s0, 1.0)[:, None],
                  np.array([1.0, 0.0, 0.0]))
    y = B[:, :, 1] - _dot3(u0, B[:, :, 1])[:, None] * u0
    n1 = np.sqrt(_dot3(y, y))
    # nothing left in the second column: any direction square to the first,
    # taken from the axis the first leans along least
    au = np.abs(u0)
    k = np.where(au[:, 1] < au[:, 0], 1, 0)
    k = np.where(au[:, 2] < au[np.arange(len(k)), k], 2, k)
    f = np.eye(3)[k] - u0[np.arange(len(k)), k][:, None] * u0
    f = f / np.sqrt(_dot3(f, f))[:, None]
    u1 = np.where((n1 > 0.0)[:, None], y / np.where(n1 > 0.0, n1, 1.0)[:, None], f)
    u2 = _cross(u0, u1)
    u2 = np.where((_dot3(u2, B[:, :, 2]) < 0.0)[:, None], -u2, u2)
    return np.stack([u0, u1, u2], -1), S, V


def _top_eig3(A):
    """The largest eigenvalue of each symmetric 3x3 matrix in a stack, by
    EIG_SWEEPS sweeps of cyclic Jacobi on its six distinct entries."""
    d = [A[:, 0, 0].copy(), A[:, 1, 1].copy(), A[:, 2, 2].copy()]
    o = {(0, 1): A[:, 0, 1].copy(), (0, 2): A[:, 0, 2].copy(), (1, 2): A[:, 1, 2].copy()}
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(EIG_SWEEPS):
            for p, q, r in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                apq = o[(p, q)]
                c, s, t = _rot(d[p], d[q], apq)
                d[p], d[q] = d[p] - t * apq, d[q] + t * apq
                o[(p, q)] = np.zeros_like(apq)
                rp, rq = (min(r, p), max(r, p)), (min(r, q), max(r, q))
                arp, arq = o[rp], o[rq]
                o[rp], o[rq] = c * arp - s * arq, s * arp + c * arq
    return np.maximum(np.maximum(d[0], d[1]), d[2])


def _solve3(A, b):
    """x with A x = b for a stack of symmetric positive definite 3x3
    matrices: Gaussian elimination in a fixed order, without pivoting, which
    such a matrix never needs."""
    l10 = A[:, 1, 0] / A[:, 0, 0]
    l20 = A[:, 2, 0] / A[:, 0, 0]
    a11 = A[:, 1, 1] - l10 * A[:, 0, 1]
    a12 = A[:, 1, 2] - l10 * A[:, 0, 2]
    a21 = A[:, 2, 1] - l20 * A[:, 0, 1]
    a22 = A[:, 2, 2] - l20 * A[:, 0, 2]
    b1 = b[:, 1] - l10 * b[:, 0]
    b2 = b[:, 2] - l20 * b[:, 0]
    l21 = a21 / a11
    a22 = a22 - l21 * a12
    b2 = b2 - l21 * b1
    x2 = b2 / a22
    x1 = (b1 - a12 * x2) / a11
    x0 = ((b[:, 0] - A[:, 0, 1] * x1) - A[:, 0, 2] * x2) / A[:, 0, 0]
    return np.stack([x0, x1, x2], -1)


_CONSTS = {}


def _consts(shape, dx):
    """Two arrays the fused substep reads and never writes, made once per
    grid: the highest index `_sample` clips to, and the highest position
    the domain edge allows (see the end of `MPM.substep`)."""
    key = (shape, dx)
    if key not in _CONSTS:
        shp = np.array(shape)
        _CONSTS[key] = (np.array(shape, float) - 1.0 - 1e-6, (shp - 1.5) * dx)
    return _CONSTS[key]


class MPM:
    """A world of particles over a scratch lattice.

    `stiff` is the one honest compromise in the file. Real wood carries sound
    at 4.3 km/s, and the CFL condition on a 10 cm grid then wants a time step
    of 23 microseconds — a thousand substeps per 25 ms sim tick, which numpy
    will not do for a hundred thousand particles. Graphics MPM always softens;
    what it costs is stated rather than hidden: things RING slower than real
    matter does. A falling plank looks identical because its deformation is
    invisible at either stiffness, so the cost is paid where it does not show.
    Set `stiff=1.0` for real moduli and watch the substep count.
    """

    def __init__(self, shape, dx=0.1, stiff=1e-4, friction=0.4):
        self.shape = tuple(int(s) for s in shape)      # GRID NODES, not voxels
        self.dx = float(dx)
        self.inv_dx = 1.0 / self.dx
        self.stiff = float(stiff)
        self.friction = float(friction)
        self.grid_contact = True        # diagnostics: each contact component
        self.particle_contact = True    # can be switched off to isolate it
        # THE ACTIVE WINDOW. A substep only touches the nodes that some
        # particle's stencil reaches, so it is solved over their bounding box
        # and not over the whole world: a 512-particle block in a 400k-node
        # house cost 156 ms a substep, nearly all of it spent adding zeros.
        # Same nodes, same sums in the same order, so the same answer bit for
        # bit. False solves the whole grid, which is what
        # `test_ACTIVE_WINDOW_changes_NOTHING` compares it against.
        self.window = True
        # ISLANDS. One window over two things far apart is mostly the empty
        # space between them, so a block falling at each end of a hall cost
        # what a block filling the hall would. Groups of particles that share
        # no node, far enough apart to be worth it, are solved on separate
        # windows (see `_islands`), each with its own distance field, inside
        # the same substep and with the same dt. A node belongs to one island
        # at most, so its sums are the same numbers in the same order: the
        # same bits as the single window, which `test_ISLANDS_change_NOTHING`
        # holds it to. False solves one window over everything.
        self.islands = True
        # THE COMPILED PATH (src/mpm/kernels.py): the same P2G and G2P sums in
        # the same order, so the same bits, at a third of the time. False runs
        # the numpy loops below, which stay the definition.
        self.compiled = _k.HAVE_NUMBA
        # THE FUSED SUBSTEP (`kernels.substep`): the whole substep of one
        # island in one compiled call, with the same operations in the same
        # order as the numpy path, so the same bits. It needs `compiled`;
        # False runs the substep stage by stage as above. Inside a World,
        # `Matter.step` also turns it off when the world's own `fused` is.
        self.fused = _k.HAVE_NUMBA
        # the particles: none yet, one empty array per field (PARTICLE_FIELDS)
        for name, dtype, tail, _ in PARTICLE_FIELDS:
            setattr(self, name, np.zeros((0,) + tail, dtype))
        self.n_cl = 0                      # cluster ids handed out so far
        self._opt = ()                     # the PARTICLE_OPT fields that exist
        self.kinds = []
        # STATIC MATTER the particles cannot enter. This is the seam: today it
        # is a box drawn by hand, and when the two halves are married it will
        # be the voxel lattice's own `mat != AIR`, read straight across. A
        # table is a wall, a floor is a wall, and so is a stone that has not
        # been asked to move.
        self.wall = np.zeros(self.shape, bool)
        # The distance to it over the WHOLE world, built only when asked for
        # (`_build_sdf`): the definition the solver's window is tested
        # against. The solver itself reads `_near_walls`. These four arrays
        # are 64 bytes a cell that nothing in the solver reads any more; they
        # stay only because tools/golden.py hashes every attribute and counts
        # a REMOVED one as a change in every world. Drop them when the golden
        # base is next recorded.
        self.phi = np.full(self.shape, 1e9)        # signed distance to it
        self.pgrad = np.zeros(self.shape + (3,))   # ...and its gradient
        self.phi_node = np.full(self.shape, 1e9)   # the same two, read at the
        self.pgrad_node = np.zeros(self.shape + (3,))   # momentum grid's nodes
        # ...and the same field over the window the particles can reach:
        # [lo, hi) in cells, the walls it was built from, distance, direction
        # out, distance at the nodes, and whether it holds any wall at all
        self._near = None
        self._nears = []                   # ...one per island, when there are
                                           #   several (`_island_walls`)
        self.deep_hits = 0                 # particle-substeps read deeper than
                                           # SDF_MARGIN inside a wall
        self._flaw, self._flaw_seed = None, None
        self.t = 0.0

    @staticmethod
    def _dt1d(f):
        """Felzenszwalb & Huttenlocher's 1-D squared distance transform.

        D(p) = min_q (p - q)^2 + f(q), in one pass, by tracking the lower
        envelope of the parabolas. Separable, so running it down each axis in
        turn gives the EXACT Euclidean transform rather than a chamfer
        approximation — which matters here because the field's GRADIENT is
        what gets used, and an approximate distance has a lumpy one.
        """
        n = len(f)
        d = np.empty(n)
        v = np.zeros(n, np.int64)
        z = np.empty(n + 1)
        k = 0
        z[0], z[1] = -np.inf, np.inf
        for q in range(1, n):
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
            while s <= z[k]:
                k -= 1
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
            k += 1
            v[k], z[k], z[k + 1] = q, s, np.inf
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            d[q] = (q - v[k]) ** 2 + f[v[k]]
        return d

    def _edt2(self, seed):
        """Squared distance, in CELLS, from every cell to the nearest `seed`."""
        big = 1e12
        f = np.where(seed, 0.0, big)
        for ax in range(3):
            f = np.apply_along_axis(self._dt1d, ax, f)
        return f

    def _build_sdf(self):
        """Signed distance to the wall surface, and which way is out, over
        the whole world. The solver does not build this any more (see
        `_near_walls`); it is the definition the window is tested against.

        THE THIRD CONTACT BUG PAID FOR THIS ONE. A per-node normal hands the
        same direction to every particle near that node, whatever side of the
        surface it is on — so the single node at a table's corner, which
        honestly carries a 45-degree normal, told particles resting on the
        FLAT TOP beside it that they were on a ramp. Measured: a glass slab
        with four fifths of itself supported picked up 1.12 m/s sideways in
        one tick from a purely vertical 6.3 m/s landing, and every geometry
        and every centre of mass slid off the table.

        With a distance field the normal is asked for AT THE PARTICLE: one
        sitting on the top sees `up`, one past the edge sees the corner, and
        neither is told about the other's surface.
        """
        out = np.sqrt(self._edt2(self.wall))        # cells to nearest solid
        ins = np.sqrt(self._edt2(~self.wall))       # cells to nearest free
        self.phi = (out - ins) * self.dx            # <0 inside, 0 at the face
        g = np.stack(np.gradient(self.phi, self.dx), -1)
        ln = np.linalg.norm(g, axis=-1, keepdims=True)
        self.pgrad = np.where(ln > 1e-12, g / np.maximum(ln, 1e-12), 0.0)
        # THE SAME FIELD, READ AT THE NODES. `phi[i]` is cell i, centred at
        # (i + 1/2) dx; the momentum grid's node i sits at i dx. Reading cell
        # values for nodes puts the surface half a cell off, and a grid
        # constraint built on that floated a resting block ten centimetres
        # high while a contact layer beside a table edge shoved a falling slab
        # sideways at 1.4 m/s. Sampled at node positions the face node reads
        # exactly zero, the node above it is free, and the diagonal normal
        # is confined to the one node that IS the corner.
        nodes = np.stack(np.meshgrid(*[np.arange(n) * self.dx for n in self.shape],
                                     indexing="ij"), -1).reshape(-1, 3)
        self.phi_node = self._sample(self.phi, nodes).reshape(self.shape)
        gn = self._sample(self.pgrad, nodes).reshape(self.shape + (3,))
        ln = np.linalg.norm(gn, axis=-1, keepdims=True)
        self.pgrad_node = np.where(ln > 1e-12, gn / np.maximum(ln, 1e-12), 0.0)

    def _near_walls(self):
        """The distance field where the particles can read it, and nowhere
        else: (lo, hi, walls, phi, pgrad, phi at the nodes, any wall).

        THE WHOLE WORLD'S FIELD COST MORE THAN EVERYTHING IT WAS FOR. Over a
        400k-cell house it took 6 s to build, the seam built it again every
        time the lattice changed, and a substep only ever reads it within two
        cells of a particle. So it is built over the particles' box, made
        SDF_MARGIN cells wider than anything a substep reads and SDF_SLACK
        wider again so that a moving body does not rebuild it every substep.
        It is built again when the particles leave that box, or when a wall
        inside it changes.

        EXACT, NOT CLOSE. A distance in the window is the world's distance
        whenever the nearest wall (or, from inside a wall, the nearest free
        cell) is in the window too, and near a surface it always is.
        Positions are not shifted into the window: `_sample` takes whole cells
        off the index, so the weights are the world field's own. The one
        exception is a particle more than SDF_MARGIN cells deep inside a wall.
        It still reads "inside" and is still pushed out, but by a distance
        measured to the window's edge; `deep_hits` counts those.

        AND A PARTICLE UP TO SDF_MARGIN DEEP READS FURTHER THAN ITS OWN DEPTH.
        Its direction out comes from cells two away on every side, and the
        nearest free cell of one of those can be SDF_MARGIN + 2 cells off, on
        a diagonal. With the window only SDF_MARGIN + 3 past the particles,
        that free cell could fall outside it: a direction measured to the
        window's edge, for a particle `deep_hits` does not count. `DEEP_READ`
        more cells on every side close it."""
        self._near = self._sdf_window(self.x, (self._near,))
        return self._near

    def _island_walls(self, groups):
        """`_near_walls` for each island on its own: one window per group of
        particle indices, in the order of `groups`.

        Each is built around its own island exactly as the single window is
        built around all of them, with the same margins, so what an island
        reads is exact for the same reason (see `_near_walls`). A window is
        kept from the last substep while it still holds an island and its
        walls have not changed; two islands that just parted may share the
        window that held them both."""
        self._nears = [self._sdf_window(self.x[g], self._nears) for g in groups]
        return self._nears

    def _sdf_window(self, x, kept, fill=True, cells=None):
        """The distance field window the particles at `x` need: the first of
        `kept` that holds it and whose walls are unchanged, or a new one.
        With `fill` False only its box and walls are settled (`_fill` builds
        the field), which is all `substep` needs to keep the one window's box
        what it would have been while islands solve the substeps. `cells`,
        when given, is the lowest and highest cell of the particles at `x`,
        already found (`kernels.bounds`)."""
        if cells is None:
            c = np.floor(x * self.inv_dx).astype(np.int64)
            cells = c.min(axis=0), c.max(axis=0)
        pad_lo, pad_hi = 3 + SDF_MARGIN + DEEP_READ, 4 + SDF_MARGIN + DEEP_READ
        for near in kept:
            if near is not None and self._holds(near, cells, pad_lo, pad_hi):
                return self._fill(near) if fill else near
        shp = np.array(self.shape)
        need_lo = np.maximum(cells[0] - pad_lo, 0)
        need_hi = np.minimum(cells[1] + pad_hi, shp)
        lo = np.maximum(need_lo - SDF_SLACK, 0)
        hi = np.minimum(need_hi + SDF_SLACK, shp)
        sub = self.wall[_box(lo, hi)].copy()
        near = (lo, hi, sub, None, None, None, bool(sub.any()))
        return self._fill(near) if fill else near

    def _holds(self, near, cells, pad_lo, pad_hi):
        """Does the kept window `near` still serve particles whose cells
        span `cells`: does it reach `pad_lo` cells below them and `pad_hi`
        above (clipped to the world), with its walls unchanged?"""
        if self.compiled:                        # the same test, in one call
            return _k.window_holds(self.wall, cells[0], cells[1], pad_lo, pad_hi,
                                   near[0], near[1], near[2])
        need_lo = np.maximum(cells[0] - pad_lo, 0)
        need_hi = np.minimum(cells[1] + pad_hi, np.array(self.shape))
        return bool((need_lo >= near[0]).all() and (need_hi <= near[1]).all()
                    and np.array_equal(self.wall[_box(near[0], near[1])], near[2]))

    def _fill(self, near):
        """The window `near` with its distance field built, from the walls it
        was cut with; as it is when it has one already."""
        if near[3] is not None:
            return near
        lo, hi, sub = near[0], near[1], near[2]
        edt = _k.edt2 if self.compiled else self._edt2
        out = np.sqrt(edt(sub))                     # as `_build_sdf`, cell for cell
        ins = np.sqrt(edt(~sub))
        phi = (out - ins) * self.dx
        g = np.stack(np.gradient(phi, self.dx), -1)
        ln = np.linalg.norm(g, axis=-1, keepdims=True)
        pgrad = np.where(ln > 1e-12, g / np.maximum(ln, 1e-12), 0.0)
        nodes = np.stack(np.meshgrid(*[np.arange(lo[a], hi[a]) * self.dx
                                       for a in range(3)],
                                     indexing="ij"), -1).reshape(-1, 3)
        node = self._sample(phi, nodes, lo).reshape(tuple(int(n) for n in hi - lo))
        return (lo, hi, sub, phi, pgrad, node, near[6])

    def _sample(self, field, pts, lo=None):
        """Trilinear lookup of a grid field at world positions. With `lo`,
        `field` is a window of the world's field whose first cell is `lo`."""
        # CELL i IS CENTRED AT (i + 1/2) dx. The distance field is between
        # cell centres, so its zero must land on a cell FACE — and mapping
        # position to index as x/dx puts node i at the cell's low corner and
        # the surface half a cell too low. Measured: every particle sank 5 cm
        # into every floor. The half is the whole correction.
        if self.compiled:
            # the same lookup compiled (`kernels.sample`), into one column
            # per component of the field
            f4 = field if field.ndim == 4 else field[..., None]
            low = np.zeros(3, np.int64) if lo is None else lo
            out = np.empty((len(pts), f4.shape[3]))
            _k.sample(f4, pts, self.inv_dx, np.array(self.shape, float) - 1.0 - 1e-6,
                      np.array(self.shape) - 1, low, np.array(f4.shape[:3]) - 1, out)
            return out if field.ndim == 4 else out[:, 0]
        g = np.clip(pts * self.inv_dx - 0.5, 0.0,
                    np.array(self.shape, float) - 1.0 - 1e-6)
        i0 = g.astype(np.int64)
        fr = g - i0
        i1 = np.minimum(i0 + 1, np.array(self.shape) - 1)
        if lo is not None:
            # the same cells and the same weights as the world's field, read
            # from the window; only a node on the window's own low face would
            # need the cell below it, and nothing reads the window's faces
            top = np.array(field.shape[:3]) - 1
            i0 = np.clip(i0 - lo, 0, top)
            i1 = np.clip(i1 - lo, 0, top)
        out = 0.0
        for a in (0, 1):
            wa = fr[:, 0] if a else 1.0 - fr[:, 0]
            ia = i1[:, 0] if a else i0[:, 0]
            for b in (0, 1):
                wb = fr[:, 1] if b else 1.0 - fr[:, 1]
                ib = i1[:, 1] if b else i0[:, 1]
                for c in (0, 1):
                    wc = fr[:, 2] if c else 1.0 - fr[:, 2]
                    ic = i1[:, 2] if c else i0[:, 2]
                    w = (wa * wb * wc)
                    v = field[ia, ib, ic]
                    out = out + (w[:, None] * v if v.ndim > 1 else w * v)
        return out

    def _flaw_field(self, seed=0):
        """Where the matter of this world is WEAK, as a smooth field in [0,1].

        Glass does not fail at its theoretical strength, it fails at its worst
        flaw — which is why a fibre is stronger than a block of the same stuff.
        Without it every particle in a struck slab passes the same threshold in
        the same instant and the whole body turns to powder at once: measured,
        472 of 480 pieces cracked and the result POURED off the table like
        sand instead of breaking.
        
        Smoothed, not per-particle noise. Independent randomness per particle
        gives salt-and-pepper damage; a correlated field gives WEAK REGIONS,
        and a crack is what you get when one of those lets go and hands its
        load to its neighbours.
        """
        if self._flaw is None or self._flaw_seed != seed:
            rng = np.random.default_rng(seed)
            f = rng.random(self.shape)
            for _ in range(2):
                for a in range(3):
                    pad = np.pad(f, [(1, 1) if b == a else (0, 0)
                                     for b in range(3)], mode="edge")
                    lo_ = [slice(None)] * 3; lo_[a] = slice(0, -2)
                    hi_ = [slice(None)] * 3; hi_[a] = slice(2, None)
                    f = (f + pad[tuple(lo_)] + pad[tuple(hi_)]) / 3.0
            f -= f.min()
            self._flaw = f / max(float(f.max()), 1e-12)
            self._flaw_seed = seed
        return self._flaw

    def add_wall(self, lo, hi):
        """Mark a box (metres) as immovable. Grid nodes inside it hold still,
        so anything arriving there stops — the floor, a table, an anvil."""
        # CELL i SPANS [i dx, (i+1) dx), so a wall to `hi` ends at cell
        # ceil(hi/dx) EXCLUSIVE. This had a +1, which made every wall one cell
        # taller than asked — a floor "to 0.2" ran to 0.3 — and the old
        # half-cell-low distance field hid it by putting the surface at 0.25.
        # The 1e-9 keeps 1.2 * 10 from ceiling to 13 through float error.
        sl = tuple(slice(max(int(np.floor(lo[a] * self.inv_dx + 1e-9)), 0),
                         min(int(np.ceil(hi[a] * self.inv_dx - 1e-9)),
                             self.shape[a])) for a in range(3))
        self.wall[sl] = True
        return int(self.wall.sum())

    # ---------------------------------------------------------------- filling

    def add_box(self, lo, hi, material, per_cell=2, jitter=True, seed=0):
        """Fill a box (metres, corner to corner) with particles.

        `per_cell` particles PER AXIS per grid cell — 2 means 8 per cell, which
        is the standard choice: fewer and the grid sees gaps between a solid's
        own particles, more and it costs without looking different.
        """
        dens, E, nu, yld, damp, strength, rigid = MAT[material]
        step = self.dx / per_cell
        grids = [np.arange(lo[a] + step * 0.5, hi[a], step) for a in range(3)]
        pts = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
        if jitter:
            # A PERFECT LATTICE OF PARTICLES IS A STANDING WAVE. Placed exactly
            # on grid symmetry, every particle's neighbours push back equally
            # and the block sits ringing in place instead of settling. Half a
            # particle spacing of noise breaks the symmetry and costs nothing.
            rng = np.random.default_rng(seed)
            pts = pts + rng.uniform(-0.25 * step, 0.25 * step, pts.shape)
        k = len(self.kinds)
        self.kinds.append(material)
        vol = step ** 3
        # the threshold each piece actually fails at: its material's, scaled by
        # how sound the matter happens to be just there
        # softened alongside the modulus so the BREAKING SPEED is preserved,
        # then varied by how sound the matter happens to be just there
        c0 = 0.0 if strength is None else strength * np.sqrt(self.stiff)
        if c0 > 0.0:
            # SAMPLED IN THE OBJECT'S OWN COORDINATES, not the world's.
            # Indexed by world position instead, a thing's strength depends on
            # WHERE IT IS: the same slab dropped from a greater height samples
            # a different part of the field and is made of different glass.
            # Measured, and it inverted an experiment — a slab dropped from
            # 3.2 m broke LESS than one dropped from 2.0 m, which is not
            # possible, and was not a crumple zone or a shielding effect or
            # any of the physical stories available. It was 15% stronger glass.
            fl = self._flaw_field(seed)
            ij = np.clip(((pts - np.asarray(lo, float)) * self.inv_dx
                          ).astype(np.int64), 0, np.array(self.shape) - 1)
            crit_p = c0 * (0.45 + 1.55 * fl[ij[:, 0], ij[:, 1], ij[:, 2]])
        else:
            crit_p = np.zeros(len(pts))
        return self._append(pts, dens * vol, vol, E * self.stiff, nu,
                            0.0 if yld is None else yld, damp, dens,
                            crit_p, k, rigid)

    def _append(self, pts, mass, vol, E, nu, yld, damp, dens, crit, kind,
                rigid=0.0):
        """`n = len(pts)` new particles at `pts`. `mass`, `vol` and `crit`
        are one value for all of them or one each; every field this call is
        not told about starts at its PARTICLE_FIELDS default."""
        n = len(pts)
        given = {"x": pts, "m": mass, "vol0": vol,
                 **self._material(E, nu, yld, damp, dens, crit, kind, rigid)}
        for name, dtype, tail, start in PARTICLE_FIELDS:
            add = np.empty((n,) + tail, dtype)
            add[...] = given.get(name, np.eye(3) if start == "eye" else start)
            setattr(self, name, np.concatenate([getattr(self, name), add]))
        for name, dtype, tail, start in self.opt_fields():
            add = np.full((n,) + tail, start, dtype)
            setattr(self, name, np.concatenate([getattr(self, name), add]))
        return n

    def opt_fields(self):
        """The PARTICLE_OPT rows whose arrays exist (none, in a world that
        never asked for one)."""
        return tuple(f for f in PARTICLE_OPT if f[0] in self._opt)

    def ensure_opt(self, name):
        """Make the optional per-particle array `name` (all zeros) if it does
        not exist yet, and return it. Particles already here start at the
        default, as `_append` gives new ones."""
        if name not in self._opt:
            for nm, dtype, tail, start in PARTICLE_OPT:
                if nm == name:
                    setattr(self, nm, np.full((len(self.x),) + tail, start, dtype))
                    self._opt = self._opt + (nm,)
                    break
            else:
                raise KeyError(name)
        return getattr(self, name)

    def _material(self, E, nu, yld, damp, dens, crit, kind, rigid):
        """The particle fields a material's row sets, by name."""
        return {
            "mu": E / (2.0 * (1.0 + nu)),
            "lam": E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
            "yld": yld,
            # critical damping for this material on THIS grid: the viscous
            # stress that stops a wave one voxel wide in about one crossing
            # of it. Written against the SOFTENED modulus, so dialling
            # `stiff` does not silently change how bouncy the world is.
            "eta": damp * np.sqrt(E * dens) * self.dx,
            "strength": crit, "kind": kind, "rig": float(rigid)}

    def become(self, idx, E, nu, yld, damp, dens, crit, kind, rigid=0.0):
        """The particles `idx` are made of another material now: wood that
        burned to char, char that burned to ash. They keep their mass, heat,
        place, motion and deformation; what the material sets changes, and
        the volume at rest becomes their mass at the new density (as in
        `promote`: less char is a smaller piece, not a lighter char). A
        particle with no mass keeps its volume, as `promote` gives it one."""
        for name, value in self._material(E, nu, yld, damp, dens, crit,
                                          kind, rigid).items():
            getattr(self, name)[idx] = value
        kg = self.m[idx]
        self.vol0[idx] = np.where(kg > 0.0, kg / dens, self.vol0[idx])

    def keep(self, mask):
        """Keep only the particles where `mask` is true. Every field in
        PARTICLE_FIELDS loses the others, and the ones kept stay in order."""
        for name, _, _, _ in PARTICLE_FIELDS + self.opt_fields():
            setattr(self, name, getattr(self, name)[mask])

    def set_aside(self, mask):
        """Solve only the particles where `mask` is true until `take_back`.
        The others leave every field in PARTICLE_FIELDS for the time being,
        so a step costs what the ones left cost and cannot touch the ones
        put aside. The answer is what `take_back` needs."""
        aside = {}
        for name, _, _, _ in PARTICLE_FIELDS + self.opt_fields():
            a = getattr(self, name)
            aside[name] = a[~mask]
            setattr(self, name, a[mask])
        return mask, aside

    def take_back(self, held):
        """Put back what `set_aside` put aside, every particle in its old
        place. Nothing may add or drop particles in between."""
        mask, aside = held
        for name, dtype, tail, _ in PARTICLE_FIELDS + self.opt_fields():
            out = np.empty((len(mask),) + tail, dtype)
            out[mask] = getattr(self, name)
            out[~mask] = aside[name]
            setattr(self, name, out)

    # ------------------------------------------------------- what holds together

    def bind(self, first, last=None):
        """Bind a run of particles into ONE cluster that keeps its shape.

        Everything `promote()` hands over in one call is one connected lump
        of lattice — a trunk with its canopy, a vat, a shelf — so the caller
        already knows the grouping and nothing here has to rediscover it.
        A cluster remembers where each of its particles sat in its own frame;
        that rest shape is what it is pulled back toward, and what tells it
        it has been pulled apart.
        """
        last = len(self.x) if last is None else last
        if last - first < 2:
            return -1
        sl = slice(first, last)
        mm = self.m[sl]
        com = (mm[:, None] * self.x[sl]).sum(axis=0) / max(mm.sum(), 1e-12)
        self.x0[sl] = self.x[sl] - com
        self.cid[sl] = self.n_cl
        self.n_cl += 1
        return self.n_cl - 1

    def _hold_shape(self, dt):
        """Pull every bound cluster's MOTION toward a rigid one, and part the
        bonds that were carrying more than their material can.

        VELOCITIES, NOT POSITIONS. The first version moved particles toward
        the best-fit rigid copy of their rest shape (Müller 2005) and set
        their velocity from the move. That fought the solver's own elastic
        stress, which kept pushing back against a strain the projection had
        quietly erased, and every round of the fight added speed: a toppled
        pole landed, all 62 pieces broke at once, and it was flung back
        across the world at 7.7 m/s. Energy from nowhere.

        So instead each cluster's velocity field is projected onto rigid
        motion — one velocity for the centre of mass, one spin about it —
        and each particle moves a fraction `rig` of the way there. The fit is
        weighted by m * rig, which makes the total change in momentum and in
        angular momentum exactly zero even when a stiff trunk and a floppy
        canopy share one cluster, and makes the kinetic energy after the
        projection never larger than before it: sum m*rig*(rig - 2)|e|^2 <= 0.
        It can only remove energy. Contact, gravity and stress still arrive
        through the grid; this only refuses the parts of the motion that
        would bend the thing.

        PARTING is judged the way the fracture law judges an impact: the
        stress of a sudden change of speed is rho * c * dv (the elastic
        wave the jolt sends into the material), set against its strength.
        Both are softened by the same square root, so the speed at which a
        bond parts is the real material's, not the rubber's.
        """
        live = self.cid >= 0
        if not live.any():
            return
        ncl = self.n_cl
        idx = np.flatnonzero(live)
        cid = self.cid[idx]
        w = self.m[idx] * self.rig[idx]
        W = np.bincount(cid, w, ncl)
        ok = W > 1e-12
        x, v = self.x[idx], self.v[idx]
        xc = np.zeros((ncl, 3)); vc = np.zeros((ncl, 3))
        for a in range(3):
            xc[:, a] = np.bincount(cid, w * x[:, a], ncl)
            vc[:, a] = np.bincount(cid, w * v[:, a], ncl)
        xc[ok] /= W[ok, None]; vc[ok] /= W[ok, None]
        r = x - xc[cid]
        L = np.zeros((ncl, 3))
        rv = _cross(r, v - vc[cid])
        for a in range(3):
            L[:, a] = np.bincount(cid, w * rv[:, a], ncl)
        I = np.zeros((ncl, 3, 3))
        rr = _dot3(r, r)
        for a in range(3):
            for b in range(3):
                term = (rr if a == b else 0.0) - r[:, a] * r[:, b]
                I[:, a, b] = np.bincount(cid, w * term, ncl)
        # a cluster that is a line or a point has no inertia about some axis;
        # spin about it is meaningless, not infinite
        reg = 1e-9 + 1e-6 * ((I[:, 0, 0] + I[:, 1, 1]) + I[:, 2, 2])
        for a in range(3):
            I[:, a, a] += reg
        om = _solve3(I, L)
        vr = vc[cid] + _cross(om[cid], r)
        dv = self.rig[idx][:, None] * (vr - v)
        self.v[idx] = v + dv
        rho = self.m[idx] / np.maximum(self.vol0[idx], 1e-12)
        mod = self.lam[idx] + 2.0 * self.mu[idx]
        sig = np.sqrt(np.maximum(mod * rho, 0.0)) * np.sqrt(_dot3(dv, dv))
        part = (self.strength[idx] > 0.0) & (sig > self.strength[idx])
        if part.any():
            gone = idx[part]
            self.cid[gone] = -1         # it is loose now, and it is rubble
            self.dmg[gone] = True

    # ------------------------------------------------------------ the solver

    def cfl_dt(self):
        """The largest step that cannot outrun the material's own sound — or
        its own VISCOSITY, which is the one this first forgot.

        Damping is a diffusion, and an explicit diffusion has a step limit of
        its own that has nothing to do with the wave speed. Measured: with the
        elastic condition alone, raising wood's damping past about 0.5 sent the
        deformation gradient to infinity inside a dozen substeps and the SVD
        stopped converging — a blow-up that looked like a bad material and was
        really a bad time step.

        AND NOT RIGHT AT THAT LIMIT: 0.8 of it. Stable is not the same as
        accurate. The felled tree, run exactly at wood's viscous limit (86
        substeps a tick), cracked 113 trunk particles on a 0.5 m/s landing,
        an impact stress near 16 kPa against wood's weakest scaled strength
        of 45 kPa. The same landing at 215, 172, 130 or 108 substeps cracks
        none. A fracture the step size makes is not one the matter makes, so
        the step keeps a margin below the limit (108 substeps for the tree).
        """
        if not len(self.x):
            return 1e-3
        rho = np.maximum(self.m / self.vol0, 1e-9)
        c = np.sqrt((self.lam + 2.0 * self.mu) / rho)
        wave = 0.4 * self.dx / max(float(c.max()), 1e-9)
        visc = np.max(self.eta / rho)
        if visc <= 0.0:
            return wave
        return min(wave, 0.8 * 0.25 * self.dx ** 2 / visc)

    def _weights(self):
        """Quadratic B-spline over the 3x3x3 stencil around each particle.

        `base` is the lowest of the three nodes on each axis; `fx` is where the
        particle sits between them, always in [0.5, 1.5) — which is the point
        of the -0.5, and why the three weights below are the ones they are.
        """
        px = self.x * self.inv_dx
        base = np.floor(px - 0.5).astype(np.int64)
        fx = px - base
        w = np.stack([0.5 * (1.5 - fx) ** 2,
                      0.75 - (fx - 1.0) ** 2,
                      0.5 * (fx - 0.5) ** 2])       # (3, N, 3)
        return base, fx, w

    def _islands(self, base, top):
        """The particles in groups that share no grid node: a list of index
        arrays, each in increasing order, the groups in order of their first
        particle. One group when nothing is worth splitting.

        A group is cut in two by a plane that no stencil crosses, with the
        particles whose stencils lie below it on one side and the rest on
        the other, so no node hears from both halves. Of all such planes on
        all three axes the cut is the one whose two windows hold the fewest
        nodes, and it is kept when they hold at least ISLAND_NODES fewer than
        the one window they replace. Then each half is tried again, because
        two things side by side in x may be one above the other inside the
        half the x cut left.

        BY SORTING, NOT BY A GRID. The cost is the particle count (a sort per
        axis), never the distance between the groups: an occupancy grid over
        their span would bring back exactly the cost that islands remove."""
        # a window smaller than ISLAND_NODES cannot save that many, and most
        # worlds are one small window: asked first, and cheaply
        span = (np.minimum(base.max(axis=0) + 2, top)
                - np.maximum(base.min(axis=0), 0) + 1)
        if int(np.prod(span)) < ISLAND_NODES:
            return [np.arange(len(base))]
        # the nodes each stencil reaches on each axis, [first, last], clipped
        # to the world as P2G clips them; one row per axis, because numpy
        # reduces a contiguous row several times faster than a column
        first = np.ascontiguousarray(np.minimum(np.maximum(base, 0), top).T)
        last = np.ascontiguousarray(np.minimum(np.maximum(base + 2, 0), top).T)

        def boxes(f, l):                 # the window over each prefix of a run
            out = 1
            for a in range(3):
                out = out * (np.maximum.accumulate(l[a])
                             - np.minimum.accumulate(f[a]) + 1)
            return out

        todo, done = [np.arange(len(base))], []
        while todo:
            g = todo.pop()
            f, l = first[:, g], last[:, g]
            whole = 1
            for a in range(3):
                whole *= int(l[a].max()) - int(f[a].min()) + 1
            best, halves = ISLAND_NODES - 1, None
            for a in range(3 if whole >= ISLAND_NODES and len(g) > 1 else 0):
                order = np.argsort(f[a], kind="stable")
                # a cut only where the next stencil starts past every one
                # before it: no shared node
                apart = f[a, order[1:]] > np.maximum.accumulate(l[a, order[:-1]])
                if not apart.any():
                    continue
                fo, lo_ = f[:, order], l[:, order]
                # the window over the first k+1 particles in this order, and
                # over the rest, for every k at once
                below = boxes(fo, lo_)
                above = boxes(fo[:, ::-1], lo_[:, ::-1])[::-1]
                saved = np.where(apart, whole - below[:-1] - above[1:], -1)
                k = int(np.argmax(saved))
                if saved[k] > best:
                    best = int(saved[k])
                    halves = (np.sort(g[order[:k + 1]]), np.sort(g[order[k + 1:]]))
            if halves is None:
                done.append(g)
            else:
                todo += halves
        done.sort(key=lambda g: int(g[0]))
        return done

    def _stress(self, dt):
        """Fixed-corotated elasticity, in the form MLS-MPM wants it.

        The rotation R is thrown away by the (F - R) difference, which is the
        whole trick: a plank that has merely TURNED is unstressed, a plank that
        has been BENT is not, and the deformation gradient tells them apart.
        That is exactly the fact a voxel cannot hold.
        """
        U, S, V = _svd3(self.F)
        # CRACKED MATTER FLOWS LIKE GRAVEL, and that is not a flourish, it is
        # what stops the solver exploding. A shard carries almost no stress, so
        # nothing opposes its deformation gradient, and `F` is integrated every
        # substep whether anything is restraining it or not: measured, the
        # glass broke correctly, spread correctly, and then ran to infinity
        # about a second later and the SVD gave up.
        #
        # The physical reading is the better one anyway. A piece that has lost
        # cohesion has no rest shape left to be measured against, so strain
        # past a small yield simply does not accumulate. It keeps its ROTATION,
        # which is why a shard stays a hard turned chip and not a droplet.
        yld = np.where(self.dmg, DMG_YIELD, self.yld)
        plastic = yld > 0.0
        if plastic.any():
            # Plastic flow: strain past yield does not spring back, it STAYS.
            # Iron dents. Lead dents easily. Intact glass and stone are given
            # no yield, so they hold their shape until something breaks them.
            lo, hi = 1.0 - yld[plastic, None], 1.0 + yld[plastic, None]
            S[plastic] = np.clip(S[plastic], lo, hi)
        R = _mm_t(U, V)
        J = (S[:, 0] * S[:, 1]) * S[:, 2]
        if plastic.any():
            # re-form F from the clamped singular values: the part that flowed
            # is forgotten, and only the elastic part goes on carrying stress
            self.F[plastic] = _mm_t(U[plastic] * S[plastic, None, :], V[plastic])
        PF = _mm_t(2.0 * self.mu[:, None, None] * (self.F - R), self.F)
        vol = self.lam * (J - 1.0) * J
        for a in range(3):
            PF[:, a, a] += vol
        # KELVIN-VOIGT: a real solid resists being deformed FAST, not just
        # being deformed. `C` is the velocity gradient the affine transfer
        # already carries, so its symmetric part is the strain RATE and costs
        # nothing extra to read. Rigid motion has none of it — a block in free
        # fall is undamped and falls at exactly g, which is the test.
        PF += self.eta[:, None, None] * (self.C + np.swapaxes(self.C, 1, 2))
        # BRITTLE FAILURE, read off the STRESS rather than the stretch.
        #
        # In a particle world a break is not an event that removes a cell, it
        # is a bond that stops holding: the piece keeps its mass, its heat and
        # its place, and loses its grip on its neighbours. Fragments are then
        # whatever the cracks happened to enclose, at whatever size that is —
        # which is the thing a lattice cannot do, since there the smallest
        # possible shard is one whole cell of the pane it came from.
        #
        # `PF` is the Kirchhoff stress, so dividing by J gives Cauchy, and its
        # largest eigenvalue is the most TENSILE direction — a thing is pulled
        # apart along the axis it is pulled hardest along. Compression has
        # negative eigenvalues and cannot break anything here, which is the
        # discrimination a stretch threshold could never make: a shock passing
        # through a slab strains every part of it and separates none of them.
        brittle = self.strength > 0.0
        if brittle.any():
            sym = 0.5 * (PF + np.swapaxes(PF, 1, 2))
            princ = _top_eig3(sym / np.maximum(J, 1e-9)[:, None, None])
            self.dmg |= brittle & (princ > self.strength)
        if self.dmg.any():
            # A SHARD IS STILL HARD. Cracked matter keeps its resistance to
            # being squashed and loses its ability to PULL — so pieces press
            # on each other and on the floor, and come apart without being
            # dragged back by the pane they used to be part of. Zeroing the
            # stress outright instead turns broken glass into a gas.
            d = self.dmg
            PF[d] *= 0.05
            free = d & (J > 1.0)                  # being pulled apart: let go
            PF[free] = 0.0
        return -(dt * self.vol0)[:, None, None] * (4.0 * self.inv_dx ** 2) * PF

    def _contact_faces(self, top, blk):
        """Into `blk`: +1 or -1 on an axis where a wall lies within a tenth
        of a cell of the particle on that side (see `substep`). The
        definition: `kernels.contact_faces` must give these bits exactly."""
        cell = np.clip((self.x * self.inv_dx).astype(int), 0, top)
        free = ~self.wall[cell[:, 0], cell[:, 1], cell[:, 2]]
        for a in range(3):
            for sgn in (-1.0, 1.0):
                q = self.x.copy()
                q[:, a] += sgn * 0.1 * self.dx     # a centimetre: half a cell held matter 3 cm in the air
                c = np.clip((q * self.inv_dx).astype(int), 0, top)
                touch = free & self.wall[c[:, 0], c[:, 1], c[:, 2]]
                blk[touch, a] = sgn           # the wall lies on this side

    def _p2g(self, base, fx, w, m, v, affine, blk, lo, top, wy, wz, gm, gp, gb):
        """Every particle (masses `m`, velocities `v`) hands its mass and
        momentum to the 27 nodes around it, added into `gm`, `gp` and `gb`
        over the window that starts at `lo`. The definition: `kernels.p2g`
        must give these bits exactly."""
        ncell = len(gm)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    off = np.array([i, j, k])
                    node = base + off
                    np.clip(node, 0, top, out=node)
                    node -= lo
                    flat = (node[:, 0] * wy + node[:, 1]) * wz + node[:, 2]
                    wt = w[i, :, 0] * w[j, :, 1] * w[k, :, 2]
                    dpos = (off - fx) * self.dx
                    # MLS-MPM: the force shows up as the affine matrix acting
                    # on the offset, so momentum and internal stress ride the
                    # SAME scatter. No separate force loop, nothing to skew.
                    mv_ = wt[:, None] * (m[:, None] * v)
                    # THE PRODUCT IS SUMMED IN A STATED ORDER. This was
                    # np.einsum("nij,nj->ni"), and the order in which einsum
                    # adds is its own business: measured on numpy 2.5 it is
                    # (a0 d0 + a2 d2) + a1 d1, not left to right, and a kernel
                    # that summed left to right drifted 1e-15 m off this path
                    # within a substep. Written out in einsum's own order the
                    # result is the same bits as before, and the compiled
                    # kernel can match it.
                    af_ = wt[:, None] * ((affine[:, :, 0] * dpos[:, None, 0]
                                          + affine[:, :, 2] * dpos[:, None, 2])
                                         + affine[:, :, 1] * dpos[:, None, 1])
                    mom = mv_ + af_
                    gm += np.bincount(flat, wt * m, ncell)
                    for a in range(3):
                        gp[:, a] += np.bincount(flat, mom[:, a], ncell)
                        gb[:, a] += np.bincount(flat, wt * m * blk[:, a], ncell)

    def _g2p(self, base, fx, w, gv, lo, top, wy, wz, newv, newC):
        """Every particle takes back a velocity and its local gradient from
        its 27 nodes. The definition `kernels.g2p` must match."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    off = np.array([i, j, k])
                    node = base + off
                    np.clip(node, 0, top, out=node)
                    node -= lo
                    flat = (node[:, 0] * wy + node[:, 1]) * wz + node[:, 2]
                    wt = w[i, :, 0] * w[j, :, 1] * w[k, :, 2]
                    dpos = (off - fx) * self.dx
                    gvn = gv[flat]
                    newv += wt[:, None] * gvn
                    # an outer product: one multiplication per entry, so
                    # written as a broadcast it is einsum's answer exactly
                    newC += (4.0 * self.inv_dx ** 2) * wt[:, None, None] * \
                        (gvn[:, :, None] * dpos[:, None, :])

    def _solve(self, g, near, base, fx, w, affine, blk, top, dt, newv, newC):
        """P2G, the grid and G2P for one island: the particles `g` (None:
        all of them), the walls near them `near`. Their new velocities and
        velocity gradients go into their rows of `newv` and `newC`."""
        shp = top + 1
        lo_s, hi_s, phi_node = near[0], near[1], near[5]
        m, v = self.m, self.v
        if g is not None:
            base, fx, w, affine, blk = base[g], fx[g], w[:, g], affine[g], blk[g]
            m, v = m[g], v[g]
        # ---- the window: every node a stencil reaches, [lo, hi) on each axis
        if self.window:
            lo = np.maximum(base.min(axis=0), 0)
            hi = np.minimum(base.max(axis=0) + 3, shp)
        else:
            lo, hi = np.zeros(3, np.int64), shp
        wx, wy, wz = (int(s) for s in hi - lo)
        ncell = wx * wy * wz
        gm = np.zeros(ncell)
        gp = np.zeros((ncell, 3))                    # momentum, not velocity
        gb = np.zeros((ncell, 3))                    # mass that arrived STOPPED, signed by axis
        # ---- P2G: every particle hands its mass and momentum to 27 nodes
        if self.compiled:
            _k.p2g(base, fx, w, m, v, affine, blk, lo, top, wy, wz,
                   self.dx, gm, gp, gb)
        else:
            self._p2g(base, fx, w, m, v, affine, blk, lo, top, wy, wz, gm, gp, gb)
        # ---- grid: momentum becomes velocity, gravity applies, walls refuse
        live = gm > 1e-12
        gv = np.zeros((ncell, 3))
        gv[live] = gp[live] / gm[live, None]
        gv[live] += GRAVITY * dt
        gv = gv.reshape(wx, wy, wz, 3)
        # the outermost node layer of the WORLD, on each face the window reaches
        for a, n in enumerate(self.shape):
            # A WALL ONLY STOPS WHAT IS GOING INTO IT. Zeroing the whole
            # component instead glues anything that touches a wall to it.
            if lo[a] == 0:
                face = [slice(None)] * 3; face[a] = slice(0, 1)
                s = gv[tuple(face)][..., a]
                gv[tuple(face)][..., a] = np.where(s < 0.0, 0.0, s)
            if hi[a] == n:
                face = [slice(None)] * 3; face[a] = slice(gv.shape[a] - 1, gv.shape[a])
                s = gv[tuple(face)][..., a]
                gv[tuple(face)][..., a] = np.where(s > 0.0, 0.0, s)
        # A TABLE HOLDS THINGS UP; IT DOES NOT HOLD ON TO THEM. Zeroing the
        # velocity at every wall node instead GLUES whatever touches it, and
        # the measured symptom was exact: a plank with five eighths of itself
        # hanging off a table never fell. It bobbed between 4 and 19 degrees
        # for three seconds, because a clamped cantilever oscillates and a
        # resting one tips. Only motion INTO the surface is refused; motion
        # away is free, and motion along it pays Coulomb friction.
        # MOMENTUM LIVES ON THE GRID, SO THE WALL MUST TOO. Per-particle
        # contact alone only corrects a particle once it has crossed the
        # surface; the layer above keeps arriving through the grid, and an
        # undamaged block with its bottom pinned on the floor face compacted
        # to sixty percent of its height. The grid constraint goes on nodes
        # strictly INSIDE the wall — the face node itself holds a resting body
        # a full cell high (measured 0.498 for a rest at 0.400; inside-only
        # gives 0.386 and keeps the bounce), which is the stencil's reach.
        #
        # AXIS BY AXIS, NOT ALONG A NORMAL. The collider is voxel geometry
        # and every face of it is axis-aligned; the distance field's gradient
        # at an interior node one cell under a table's top and one cell in
        # from its side is diagonal, and projecting onto that turns a slab's
        # vertical arrival into 1.4 m/s of sideways motion — three times what
        # per-particle contact alone produced, and not from the corner point,
        # which reads as free. So each component is blocked on its own, by
        # the sign of its own gradient component: at that wedge node vz < 0
        # and vx < 0 are refused separately, and a body arriving straight
        # down with vx = 0 is not touched in x at all.
        #
        # THE GRID HOLDS UP; IT DOES NOT HOLD ON. No friction here, only the
        # normal. A node's stencil reaches a cell and a half, so inside nodes
        # go on constraining a body that has bounced 8 cm clear of the
        # surface — and grid friction, with a normal impulse from a 6 m/s
        # arrival, deleted every tangential component at every such node
        # every substep. That ate the -x half of the slab's own bending pair
        # while the +x half stayed on free nodes: a table pushing sideways on
        # something it was no longer touching (per-substep budget: the
        # constraint's +1 to +4 kg m/s matched the slab's gain exactly). The
        # particle knows whether it is actually touching; friction lives
        # there, and it is what stopped the felled pole.
        # THE NODE REFUSES THE DIRECTIONS ITS MATTER IS TOUCHING IN. blk is
        # +1 where the wall lies on the particle's +a side, so a node whose
        # mass says "+a" refuses va > 0. Not a majority: a face node holds
        # the touching layer and the free layers above it.
        # AND FRICTION LIVES HERE TOO, sized by the normal impulse the node
        # actually removed (Coulomb: the tangential loss is at most mu times
        # it). It was moved to the particle when inside nodes were holding a
        # body that had bounced clear; a node now holds only what touches
        # it. With no friction at the node nothing penetrated far enough to
        # feel the particle's, and the felled pole slid off the world.
        if self.grid_contact:
            frac = gb.reshape(wx, wy, wz, 3) / np.maximum(gm.reshape(wx, wy, wz), 1e-12)[..., None]
            removed = np.zeros(gv.shape[:3])
            for a in range(3):
                va = gv[..., a]
                stop = ((frac[..., a] > 0.05) & (va > 0.0)) | ((frac[..., a] < -0.05) & (va < 0.0))
                removed[stop] += np.abs(va[stop])
                va[stop] = 0.0
            rubbing = removed > 0.0
            if rubbing.any():
                vt = gv[rubbing]
                vtn = np.sqrt(_dot3(vt, vt))
                keep = np.maximum(0.0, 1.0 - self.friction * removed[rubbing] / np.maximum(vtn, 1e-12))
                gv[rubbing] = vt * keep[:, None]
        # (over the part of the window the distance field covers: a node past
        # it is out of every particle's reach and carries nothing to stop)
        a, b = np.maximum(lo, lo_s), np.minimum(hi, hi_s)
        gv[_box(a - lo, b - lo)][phi_node[_box(a - lo_s, b - lo_s)] < -1.5 * self.dx] = 0.0
        gv = gv.reshape(ncell, 3)
        # ---- G2P: particles take back a velocity AND its local gradient
        if g is None:
            nv, nC = newv, newC
        else:
            nv, nC = np.zeros((len(g), 3)), np.zeros((len(g), 3, 3))
        if self.compiled:
            _k.g2p(base, fx, w, gv, lo, top, wy, wz, self.dx,
                   4.0 * self.inv_dx ** 2, nv, nC)
        else:
            self._g2p(base, fx, w, gv, lo, top, wy, wz, nv, nC)
        if g is not None:
            newv[g], newC[g] = nv, nC

    def _contact(self, g, near):
        """Contact at the particle for one island: the particles `g` (None:
        all of them) against the walls near them, `near`."""
        # CONTACT, ASKED AT THE PARTICLE rather than at the node it happens to
        # be near. A node carries one normal for everything around it, so the
        # single node at a table's corner told particles resting on the flat
        # top beside it that they were on a 45-degree ramp, and a slab landing
        # straight down picked up 1.12 m/s sideways and slid off. The distance
        # field answers per position instead: on the top the normal is up, a
        # few centimetres past the edge there is no surface at all.
        lo_s, phi, pgrad = near[0], near[3], near[4]
        ph = self._sample(phi, self.x if g is None else self.x[g], lo_s)
        inside = ph < 0.0
        hit = inside if g is None else g[inside]
        if inside.any():
            ph = ph[inside]
            deep = ph < -SDF_MARGIN * self.dx
            self.deep_hits += int(deep.sum())
            nrm = self._sample(pgrad, self.x[hit], lo_s)
            # BURIED DEEPER THAN SDF_MARGIN, a particle reads a distance to
            # its window's edge (`_near_walls`), and an island's window is
            # smaller than the one window: a block 16 cells into a slab was
            # pushed out by a different distance with islands than without
            # (build/X6fix). It is read from the one window instead, the box
            # the single-window solve would have read it from. Which
            # particles are that deep does not depend on the window: a window
            # can only see fewer free cells, so a deep one reads deeper still,
            # and one less deep reads its exact distance in either.
            if g is not None and deep.any():
                self._near = whole = self._fill(self._near)
                ph[deep] = self._sample(whole[3], self.x[hit[deep]], whole[0])
                nrm[deep] = self._sample(whole[4], self.x[hit[deep]], whole[0])
            ln = np.sqrt(_dot3(nrm, nrm))[:, None]
            nrm = np.where(ln > 1e-12, nrm / np.maximum(ln, 1e-12), 0.0)
            # AXIS BY AXIS HERE TOO. The grid got this rule first and the
            # particle side kept projecting onto the unit normal — and a
            # particle resting on a tabletop within one cell of the edge
            # samples a normal that blends the corner cell's diagonal.
            # Arriving at -6 m/s straight down against (0.7, 0, 0.7) it
            # left at +1.8 m/s sideways; twenty of them put +5 kg m/s of
            # x-momentum into the slab in ONE substep, before any grid
            # constraint acted (measured with the grid probe: px(m v) =
            # +5.30 at first contact, px(affine) = 0). Blocking each
            # component by the sign of its own gradient component, a
            # vertical arrival with vx = 0 is not touched in x at all.
            # AND ONE FACE PER PARTICLE. A particle 1 cm into the top,
            # 4 cm from the edge, samples a blended (0.57, 0, 0.80) and
            # under a per-component test was blocked in x too — every
            # gram of the slab's sideways shove came from there once the
            # grid stopped doing the same thing (particle probe: +40
            # kg m/s in one tick, all at x=1.46 z=1.19). The face it is
            # against is the one its normal mostly points along.
            v = self.v[hit].copy()
            dom = np.argmax(np.abs(nrm), axis=1)
            for a in range(3):
                ga, va = nrm[:, a], v[:, a]
                into = (dom == a) & (((ga > 0.0) & (va < 0.0)) | ((ga < 0.0) & (va > 0.0)))
                if not into.any():
                    continue
                vn = np.abs(va[into])
                others = [k for k in range(3) if k != a]
                vt = v[into][:, others]
                vtn = np.sqrt(vt[:, 0] * vt[:, 0] + vt[:, 1] * vt[:, 1])[:, None]
                keep = np.maximum(0.0, 1.0 - self.friction * vn[:, None]
                                  / np.maximum(vtn, 1e-12))
                sub = v[into]
                sub[:, a] = 0.0
                sub[:, others] = vt * keep
                v[into] = sub
            self.v[hit] = v
            # and it is put back out along the axis it came in on — the
            # dominant gradient component — not along the blended diagonal
            push = np.zeros_like(nrm)
            push[np.arange(len(dom)), dom] = np.sign(nrm[np.arange(len(dom)), dom])
            self.x[hit] = self.x[hit] - ph[:, None] * push

    def _fused_one(self, dt, shp, top):
        """The whole substep in one compiled call (`kernels.substep`) when
        every particle is one island. False, having changed nothing, when
        there are several: `substep` then goes stage by stage."""
        _, _, cmin, cmax, nodes = _k.bounds(self.x, self.inv_dx, top)
        if self.window and self.islands:
            # `_islands`' own first question, asked of the bounds alone
            if nodes >= ISLAND_NODES \
                    and len(self._islands(self._weights()[0], top)) > 1:
                return False
        near = self._near = self._sdf_window(None, (self._near,), cells=(cmin, cmax))
        gtop, edge = _consts(self.shape, self.dx)
        x, v, C, F, deep = _k.substep(
            self.x, self.v, self.C, self.F, self.m, self.vol0, self.mu, self.lam,
            self.yld, self.eta, self.strength, self.dmg, self.cid, self.rig,
            self.n_cl, self.wall, near[6], near[0], near[1], near[3], near[4],
            near[5], shp, gtop, self.dx, self.inv_dx, 4.0 * self.inv_dx ** 2, dt,
            GRAVITY, self.friction, self.grid_contact, self.particle_contact,
            self.window, DMG_YIELD, SVD_SWEEPS, EIG_SWEEPS, -1.5 * self.dx,
            -SDF_MARGIN * self.dx, 0.5 * self.dx, edge)
        self.x, self.v, self.C, self.F = x, v, C, F
        self.deep_hits += int(deep)
        self.t += dt
        return True

    def substep(self, dt):
        if not len(self.x):                          # nothing to carry
            self.t += dt
            return
        shp = np.array(self.shape)
        top = shp - 1
        fused = self.fused and self.compiled
        if fused and self._fused_one(dt, shp, top):
            return
        base, fx, w = self._weights()
        # ---- ISLANDS: one window per group of particles that share no node
        # with any other group (`_islands`), each with the walls near it as a
        # distance field (`_near_walls`). One group is the single window.
        groups = self._islands(base, top) if self.window and self.islands else [None]
        if len(groups) > 1:
            nears = self._island_walls(groups)
            # and the one window's box and walls, kept what they would be
            # without islands, for a particle buried too deep for its own
            # island's window (`_contact`). Its field is built only then.
            self._near = self._sdf_window(self.x, (self._near,), fill=False)
        else:
            groups, nears = [None], [self._near_walls()]
        # ---- WHICH FACE EACH PARTICLE TOUCHES, told to the grid. A node is
        # one cell across and a table's corner node has two faces. Hold
        # both and a slab sliding across the top is shoved off the edge
        # (+1.2 m/s); hold only flat faces and the strip within a cell of
        # the edge is supported by nothing, so a slab with any overhang
        # tips about x=1.4 and every drop height ends on the floor. The node
        # cannot know which face its matter is against; the particle can,
        # and the geometry is axis-aligned, so it is a lookup: a particle
        # within a tenth of a cell of a wall cell along an axis, not itself in
        # a wall cell, touches that face. The signal rides P2G (gb) and the
        # node refuses only the directions its own matter is touching in.
        # Momentum is still deleted at the NODE, after the stress force has
        # been added, which is what makes the support elastic — deleting it
        # at the particle first gave e=0.04 and a block that never rebounds.
        # (A particle whose own island holds no wall has none within reach,
        # so the lookup would find none for it either.)
        blk = np.zeros((len(self.x), 3))
        if any(near[6] for near in nears):
            if self.compiled:
                _k.contact_faces(self.x, self.wall, self.inv_dx, self.dx, top, blk)
            else:
                self._contact_faces(top, blk)
        if fused:                                # the same, compiled
            affine = np.empty_like(self.F)
            _k.stress(self.F, self.C, self.m, self.mu, self.lam, self.yld, self.eta,
                      self.strength, self.dmg, self.vol0, dt, 4.0 * self.inv_dx ** 2,
                      DMG_YIELD, SVD_SWEEPS, EIG_SWEEPS, affine)
        else:
            affine = self._stress(dt) + self.m[:, None, None] * self.C
        newv = np.zeros((len(self.x), 3))
        newC = np.zeros((len(self.x), 3, 3))
        for g, near in zip(groups, nears):
            self._solve(g, near, base, fx, w, affine, blk, top, dt, newv, newC)
        self.v, self.C = newv, newC
        if fused:
            self.x, self.F = _k.move(self.x, self.v, self.C, self.F, dt)
            if self.n_cl:
                _k.hold(self.x, self.v, self.m, self.rig, self.cid, self.n_cl,
                        self.vol0, self.lam, self.mu, self.strength, self.dmg)
        else:
            self.x = self.x + dt * self.v
            # the deformation gradient is carried, never recomputed: this is
            # where a turn becomes a FACING and a squeeze becomes stress
            self.F = _mm(np.eye(3) + dt * self.C, self.F)
            if self.n_cl:
                self._hold_shape(dt)
        if self.particle_contact:
            for g, near in zip(groups, nears):
                if near[6]:
                    self._contact(g, near)
        # THE DOMAIN EDGE, NOT A FLOOR. The stencil needs half a cell below
        # and one and a half above; this used to be two cells each way, and
        # in a world whose floor is one cell thick that was an invisible,
        # frictionless floor a cell ABOVE the real one. A felled pole rested
        # on it and slid for ever at 0.689 m/s, touching nothing.
        np.clip(self.x, 0.5 * self.dx, (np.array(self.shape) - 1.5) * self.dx,
                out=self.x)
        self.t += dt

    def step(self, dt, max_sub=4000, each=None, after=None):
        """Advance by `dt` of world time in as many substeps as CFL demands.

        `max_sub` is a seatbelt, not a budget. Clipping it does not make the
        sim cheaper, it makes it WRONG — the step goes past the stability
        limit and the world quietly detonates — so hitting it says so out
        loud rather than carrying on.

        `each`, when given, is called with no arguments after every substep.
        It may read the particles and must not write them: the seam's event
        tap (`Matter.step`) watches the motion through it. `after`, called
        first, MAY write them: the seam's walls of bonded matter
        (`Matter._cross`) act there, once a substep.
        """
        want = max(1, int(np.ceil(dt / self.cfl_dt())))
        if want > max_sub:
            raise RuntimeError(
                f"step {dt} needs {want} substeps, over the {max_sub} cap: "
                f"the material is too stiff for this grid, not too slow")
        sub = want
        h = dt / sub
        for _ in range(sub):
            self.substep(h)
            if after is not None:
                after(h)
            if each is not None:
                each()
        return sub

    # ------------------------------------------------------------- reporting

    def broken(self):
        """How much of the world has cracked — the number a shatter test reads."""
        return int(self.dmg.sum())

    def total_mass(self):
        return float(self.m.sum())

    def bbox(self):
        return self.x.min(axis=0), self.x.max(axis=0)
