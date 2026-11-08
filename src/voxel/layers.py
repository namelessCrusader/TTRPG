"""THE SURFACE-LAYER POOL (wave 13, T3): a sparse struct-of-arrays store of the thin
layers on faces, with an ARRAY index, exact mass, exact energy, and the passes that
move, soak, freeze, boil, cure, wipe and wash them.

Read specs/T3-layers_v2.txt. The shape of it:

 * A face has a STACK of at most 8 layers, top first (the latest on top, F7). A
   layer is a small mix of species in T1's four forms with a mass, a cover share, a
   porosity, an adhesion and a cohesion. Thickness, wet, tacky, "mud" are READ from
   those numbers (describe), never stored (F4).
 * Keys are int64: (tag << 58) | (id << 3) | face. tag 0 cell face (id = flat cell
   index, face 0..5 = +x -x +y -y +z -z), 1 record face, 2 body region, 3 outdoor.
 * The index is two sorted arrays (ikey, islot) plus a small pending block;
   np.searchsorted resolves keys. Nothing hides in a cache: Layers holds arrays,
   small lists and a ledger, so a deepcopy of the instance is the snapshot.
 * Mass is exact (F1): every gram has a named way in and a named way out
   (led_in / led_out by (species, tag)); ledger_closes() proves it.
 * Energy is exact (F2): every mutating call returns host adjustments
   [(key, dE_J, dC_J_per_K)]. dE is the energy the host skin node of that key's
   cell must gain; dC the change of its heat capacity. total = host E + stored_J()
   - sum(energy_in) is constant.

Layers imports only numpy, layer_data and layer_laws: no World, no heat, no sim.
"""
import copy
import math

import numpy as np

from . import layer_laws as LL
from .layer_data import (ACID_SPECIES, AMBIENT, BOUND, COLOUR_WET, CRYSTAL, DEPTH_CAP,
                         DISSOLVED, FILM, GRAIN, HOLD_MIN, IRON_OXIDE, LABELS, NCOMP,
                         ROWS, SHEAR_RATIO, SOOT, SOOT_TRUE, STEFAN_MIN, SUBSTRATE, WATER,
                         WET_MEMORY, fluid_id_for, fresh_arrays, pack_key, substrate)

TAG_CELL, TAG_RECORD, TAG_BODY, TAG_COLUMN = 0, 1, 2, 3
D_WATER = WATER                 # (alias used by the alike rule)
SZ_CRYSTAL = 40                  # default size class of new crystals (32 um)
PENDING_MAX = 256
MIN_LAYER_G = 1.0e-6             # a layer thinner than this merges into the one under
KAPPA_MIN = 1.0                  # m2/g, hiding area of a deposit whose row has none
_ARR = tuple(fresh_arrays().keys())


# ---- keys -------------------------------------------------------------------------
def make_key(tag, idx, face=0):
    """int64 key (tag << 58) | (id << 3) | face."""
    return (int(tag) << 58) | (int(idx) << 3) | int(face)


def split_key(key):
    key = int(key)
    return key >> 58, (key >> 3) & ((1 << 55) - 1), key & 7


def host_of(key):
    """The host a key's adjustments land on: tag and id, every face of a cell
    shares one skin node."""
    return int(key) >> 3


def cell_index(x, y, z, shape):
    return (x * shape[1] + y) * shape[2] + z


def cell_key(x, y, z, face, shape):
    return make_key(TAG_CELL, cell_index(x, y, z, shape), face)


def cell_xyz(idx, shape):
    nz, ny = shape[2], shape[1]
    return idx // (ny * nz), (idx // nz) % ny, idx % nz


def orientation(key):
    """'up', 'down' or 'wall' of a key's face."""
    tag, _i, face = split_key(key)
    if tag == TAG_CELL:
        return "up" if face == 4 else "down" if face == 5 else "wall"
    if tag == TAG_RECORD:
        return "up" if face < 2 else "wall"
    if tag == TAG_COLUMN:
        return "up"
    return "wall"


def _entries(comp):
    """{(species, form, sz): g} or {ckey: g} or list of (ckey, g) -> sorted list of
    (ckey, g) with duplicates added and non-positive dropped."""
    out = {}
    items = comp.items() if isinstance(comp, dict) else comp
    for k, g in items:
        g = float(g)
        if g <= 0.0:
            continue
        ck = pack_key(*k) if isinstance(k, tuple) else int(k)
        out[ck] = out.get(ck, 0.0) + g
    return sorted(out.items())


def _check_face_key(key):
    """A face key must be a non-negative int64 made by make_key: a real integer (not a
    bool, a float or a string), a known tag, and a face that exists for that tag.
    ValueError otherwise (a bad key would silently land matter on some other face)."""
    if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
        raise ValueError(f"face key must be an integer made by make_key, got {key!r}")
    k = int(key)
    if k < 0 or k >= (1 << 63):
        raise ValueError(f"face key {k} is outside the int64 range of make_key")
    tag, _i, face = split_key(k)
    if tag not in (TAG_CELL, TAG_RECORD, TAG_BODY, TAG_COLUMN):
        raise ValueError(f"face key {k} has unknown tag {tag}")
    if tag == TAG_CELL and face > 5:
        raise ValueError(f"a cell has faces 0..5, key {k} names face {face}")
    return k


def _materialize(comp):
    """The one reading of comp. A dict stays a dict; anything else that holds pairs (a
    list, a tuple, a generator, an iterator) is read ONCE into a list of (key, grams)
    pairs, so the check and the add see the same pairs (a generator read by the check
    alone would be empty for the add, and the matter would vanish). ValueError when
    comp is not a mapping or a sequence of pairs."""
    if comp is None or isinstance(comp, (str, bytes)):
        raise ValueError("comp must be a mapping of component key to grams")
    if isinstance(comp, dict):
        return comp
    if hasattr(comp, "items") and callable(comp.items):
        comp = comp.items()
    try:
        pairs = [tuple(p) for p in comp]
    except TypeError:
        raise ValueError(f"comp must be a mapping or a sequence of (key, grams) pairs, "
                         f"got {type(comp).__name__}") from None
    if any(len(p) != 2 for p in pairs):
        raise ValueError("every item of comp must be a (component key, grams) pair")
    return pairs


def _check_comp(comp):
    """Check what Layers.add is given before anything is touched: a mapping (or list
    of pairs) of component key -> grams. Each key is (species, form, sz) or a packed
    int; the species must have a row, form is 0..3, sz is 0..255; each mass is a
    finite number that is not negative (zero is allowed and adds nothing); the total
    is finite. ValueError for a bad value or key (unknown species is KeyError, as
    _check_species raises it; acid stays ValueError)."""
    comp = _materialize(comp)
    items = comp.items() if isinstance(comp, dict) else comp
    total = 0.0
    for k, g in items:
        if isinstance(g, (bool, str, bytes)):
            raise ValueError(f"mass of {k!r} must be a number, got {g!r}")
        try:
            g = float(g)
        except (TypeError, ValueError):
            raise ValueError(f"mass of {k!r} must be a number, got {g!r}") from None
        if not math.isfinite(g):
            raise ValueError(f"mass of {k!r} must be finite, got {g!r}")
        if g < 0.0:
            raise ValueError(f"mass of {k!r} must not be negative, got {g!r}")
        if isinstance(k, tuple):
            if len(k) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer))
                                  for x in k):
                raise ValueError(f"component key must be (species, form, sz) of ints, got {k!r}")
            sp, form, sz = (int(x) for x in k)
            if not 0 <= form <= 3:
                raise ValueError(f"form of {k!r} must be 0..3 (dissolved, crystal, grain, bound)")
            if not 0 <= sz <= 255:
                raise ValueError(f"size class of {k!r} must be 0..255")
        elif isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 0:
            raise ValueError(f"component key must be (species, form, sz) or a packed "
                             f"non-negative int, got {k!r}")
        else:
            sp = int(k) >> 10
        if sp not in ROWS:
            raise KeyError(f"no layer row for species {sp}")
        total += g
        if not math.isfinite(total):
            raise ValueError("the total mass of comp is not finite")
    return comp


def _check_number(name, v, lo=None, hi=None, lo_open=False, none_ok=True):
    """None passes (the caller's default) unless none_ok is False; else a finite
    real in the range. ValueError for everything else (also a string, a bool)."""
    if v is None:
        if none_ok:
            return
        raise ValueError(f"{name} must be a finite number, got None")
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    v = float(v)
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise ValueError(f"{name} must be {'>' if lo_open else '>='} {lo}, got {v!r}")
    if hi is not None and v > hi:
        raise ValueError(f"{name} must be <= {hi}, got {v!r}")


def _check_int(name, v, lo, hi):
    """A real integer (not a bool, a float or a string) in lo..hi, as int."""
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {v!r}")
    if not lo <= int(v) <= hi:
        raise ValueError(f"{name} must be {lo}..{hi}, got {v!r}")
    return int(v)


def _check_keys(name, keys):
    """A non-empty list of face keys (a generator is read once); returns the list."""
    try:
        out = [_check_face_key(k) for k in keys]
    except TypeError:
        raise ValueError(f"{name} must be a sequence of face keys, got {keys!r}") from None
    if not out:
        raise ValueError(f"{name} must hold at least one face key")
    return out


def _check_sub(name, sub):
    """A substrate argument must be what `substrate` reads: None (the default), a row dict, the
    NAME of a substrate row or a material id that has one. ValueError for anything else (a bool,
    a float, a list: a float used to be cut to an int and pick another substrate); KeyError for a
    name or an id with no row, as an unknown species is. Read before anything is touched."""
    if sub is None or isinstance(sub, dict):
        return
    if isinstance(sub, str):
        if all(r["name"] != sub for r in SUBSTRATE.values()):
            raise KeyError(f"{name}: no substrate named {sub!r}")
        return
    if isinstance(sub, bool) or not isinstance(sub, (int, np.integer)):
        raise ValueError(f"{name} must be None, a substrate row, a substrate name or a material id, got {sub!r}")
    if int(sub) not in SUBSTRATE:
        raise KeyError(f"{name}: no substrate for material id {int(sub)}")


def _check_species(ck):
    sp = ck >> 10
    r = ROWS.get(sp)
    if r is None:
        raise KeyError(f"no layer row for species {sp}")
    if r["acid"] or sp in ACID_SPECIES:
        raise ValueError("acid is not a layer species in wave 13 (4.14): it stays lattice")
    return r


def _merge_adj(adj):
    acc = {}
    for k, e, c in adj:
        a = acc.get(k)
        acc[k] = (e, c) if a is None else (a[0] + e, a[1] + c)
    return [(k, e, c) for k, (e, c) in acc.items()]


def _default_sz(form):
    return SZ_CRYSTAL if form in (CRYSTAL, GRAIN) else 0


class Layers:
    """The pool. `dx` is the cell edge in metres (a face is dx*dx)."""

    def __init__(self, dx=0.05):
        self.dx = float(dx)
        for name, arr in fresh_arrays().items():
            setattr(self, name, arr)
        self.hi = 0                                   # slots ever used
        self.n = 0                                    # live layers
        self.free = np.empty(0, np.int32)
        self.ikey = np.empty(0, np.int64)
        self.islot = np.empty(0, np.int32)
        self.pkey = np.empty(0, np.int64)
        self.pslot = np.empty(0, np.int32)
        self.ntomb = 0
        self.patterns = []
        self.contents = []
        self.led_in = {}
        self.led_out = {}
        self.energy_in = {}
        self.folds = 0
        self.now = 0.0
        self.gen = 0                                  # bumped by every mutation
        self.dirty = []                               # hosts whose thermal read-out changed

    # ---- copies ---------------------------------------------------------------------
    def __deepcopy__(self, memo):
        new = Layers.__new__(Layers)
        for k, v in self.__dict__.items():
            new.__dict__[k] = v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v, memo)
        return new

    def nbytes(self):
        """Bytes of every array this pool holds (0 when empty)."""
        return sum(v.nbytes for v in self.__dict__.values() if isinstance(v, np.ndarray))

    # ---- the dirty list and the pattern store ------------------------------------------
    def _touch(self, *hosts):
        """Remember which hosts' thermal read-outs changed (the heat geometry ORs these
        into its `moved` set). Kept bounded: duplicates collapse past 4096 entries."""
        self.dirty.extend(hosts)
        if len(self.dirty) > 4096:
            self.dirty = sorted(set(self.dirty))

    def take_dirty(self):
        """Sorted unique hosts touched since the last call; clears the list."""
        out = sorted(set(self.dirty))
        self.dirty = []
        return out

    def gc_patterns(self):
        """Drop patterns and contents no live layer points at and renumber the
        pointers. Returns how many were dropped."""
        live = np.flatnonzero(self.key[:self.hi] >= 0)
        dropped = 0
        for store, arr in ((self.patterns, self.pat), (self.contents, self.content)):
            used = sorted({int(arr[s]) for s in live if arr[s] >= 0})
            remap = {old: new for new, old in enumerate(used)}
            dropped += len(store) - len(used)
            store[:] = [store[i] for i in used]
            for s in live:
                if arr[s] >= 0:
                    arr[s] = remap[int(arr[s])]
        return dropped

    # ---- storage --------------------------------------------------------------------
    def _grow(self):
        cap = self.key.size
        cap2 = max(8, 2 * cap)
        for name in _ARR:
            old = getattr(self, name)
            new = np.zeros((cap2,) + old.shape[1:], old.dtype)
            if name in ("key", "below", "pat", "content"):
                new[:] = -1
            new[:cap] = old
            setattr(self, name, new)

    def _alloc(self):
        if self.free.size:
            s = int(self.free[-1])
            self.free = self.free[:-1].copy()
        else:
            if self.hi >= self.key.size:
                self._grow()
            s = self.hi
            self.hi += 1
        self.n += 1
        return s

    def _release_slot(self, s):
        self.key[s] = -1
        self.below[s] = -1
        self.m[s] = 0.0
        self.pat[s] = -1
        self.content[s] = -1
        self.ckey[s] = 0
        self.cm[s] = 0.0
        self.free = np.append(self.free, np.int32(s))
        self.n -= 1

    # ---- the index --------------------------------------------------------------------
    def _flush(self):
        if self.pkey.size:
            k = np.concatenate([self.ikey, self.pkey])
            sl = np.concatenate([self.islot, self.pslot])
            o = np.argsort(k, kind="stable")
            self.ikey, self.islot = k[o], sl[o]
            self.pkey = np.empty(0, np.int64)
            self.pslot = np.empty(0, np.int32)

    def _compact(self):
        self._flush()
        live = self.islot >= 0
        self.ikey, self.islot = self.ikey[live], self.islot[live]
        self.ntomb = 0

    def _find(self, key):
        """Top slot of a key's stack, or -1."""
        if self.pkey.size:
            w = np.flatnonzero(self.pkey == key)
            if w.size:
                return int(self.pslot[w[0]])
        if self.ikey.size:
            i = int(np.searchsorted(self.ikey, key))
            if i < self.ikey.size and self.ikey[i] == key:
                return int(self.islot[i])
        return -1

    def _set_top(self, key, slot):
        if self.pkey.size:
            w = np.flatnonzero(self.pkey == key)
            if w.size:
                j = int(w[0])
                if slot >= 0:
                    self.pslot[j] = slot
                else:
                    keep = np.arange(self.pkey.size) != j
                    self.pkey, self.pslot = self.pkey[keep], self.pslot[keep]
                return
        if self.ikey.size:
            i = int(np.searchsorted(self.ikey, key))
            if i < self.ikey.size and self.ikey[i] == key:
                old = int(self.islot[i])
                self.islot[i] = slot
                if slot < 0 <= old:
                    self.ntomb += 1
                    if self.ntomb > 16 and self.ntomb * 4 > self.ikey.size:
                        self._compact()
                elif old < 0 <= slot:
                    self.ntomb -= 1
                return
        if slot < 0:
            return
        self.pkey = np.append(self.pkey, np.int64(key))
        self.pslot = np.append(self.pslot, np.int32(slot))
        if self.pkey.size >= PENDING_MAX:
            self._flush()

    def lookup(self, keys):
        """Batched: top slot of each key (-1 for none). 10,000 keys take ~0.2 ms."""
        self._flush()
        keys = np.asarray(keys, np.int64)
        if not self.ikey.size:
            return np.full(keys.shape, -1, np.int32)
        i = np.searchsorted(self.ikey, keys)
        i = np.minimum(i, self.ikey.size - 1)
        hit = self.ikey[i] == keys
        return np.where(hit, self.islot[i], -1).astype(np.int32)

    def keys(self):
        """Every key that has layers, sorted (the one deterministic order)."""
        self._flush()
        return self.ikey[self.islot >= 0].copy()

    def stack(self, key, inside=False):
        """Slots of a key, top first. Surface layers only unless inside=True (the
        INSIDE layer, what soaked into the pores, is last)."""
        s = self._find(key)
        out = []
        guard = 0
        while s >= 0 and guard < 64:
            if not self.inside[s]:
                out.append(s)
            elif inside:
                out.append(s)
            s = int(self.below[s])
            guard += 1
        return out

    def _chain(self, key):
        s = self._find(key)
        out = []
        while s >= 0 and len(out) < 64:
            out.append(s)
            s = int(self.below[s])
        return out

    # ---- composition access ----------------------------------------------------------
    def _comp(self, s):
        return [(a, b) for a, b in zip(self.ckey[s].tolist(), self.cm[s].tolist())
                if a and b > 0.0]

    @staticmethod
    def _volumes(comp):
        return LL.comp_volumes(comp)

    def _write_comp(self, s, entries):
        """Store a composition (dict or list of (ckey, g)). More than 8 entries fold
        the smallest into the biggest other entry of its species (else into the
        dominant entry), count a 'fold', and return the (dE, dC) the fold costs."""
        d = {}
        items = entries.items() if isinstance(entries, dict) else entries
        for ck, g in items:
            if g > 0.0:
                d[ck] = d.get(ck, 0.0) + g
        dE = dC = 0.0
        while len(d) > NCOMP:
            small = min(d, key=lambda c: (d[c], c))
            g = d.pop(small)
            same = [c for c in d if (c >> 10) == (small >> 10)]
            tgt = max(same or list(d), key=lambda c: (d[c], -c))
            ra, fa = ROWS[small >> 10], (small >> 8) & 3
            rb, fb = ROWS[tgt >> 10], (tgt >> 8) & 3
            dC += g * (LL.cp_of(rb, fb) - LL.cp_of(ra, fa))
            dE -= g * (LL.form_offset(rb, fb) - LL.form_offset(ra, fa))
            self.led_out[(small >> 10, "fold")] = self.led_out.get((small >> 10, "fold"), 0.0) + g
            self.led_in[(tgt >> 10, "fold")] = self.led_in.get((tgt >> 10, "fold"), 0.0) + g
            d[tgt] += g
            self.folds += 1
        ks = sorted(d)
        self.ckey[s] = 0
        self.cm[s] = 0.0
        for i, ck in enumerate(ks):
            self.ckey[s, i] = ck
            self.cm[s, i] = d[ck]
        self.m[s] = float(self.cm[s].sum())
        return dE, dC

    def _bulk_cm3(self, s, comp=None):
        comp = self._comp(s) if comp is None else comp
        vl, vc, vg, _gm = LL.comp_volumes(comp)
        return LL.bulk_volume(vl, vc, vg, float(self.por[s]))

    @property
    def area(self):
        return self.dx * self.dx

    def thickness(self, s):
        """m: bulk volume over the area the layer covers."""
        return self._bulk_cm3(s) * 1e-6 / (self.area * max(float(self.cover[s]), 1e-6))

    def cover_of(self, key):
        """Combined cover share of a face's surface layers (random overlap)."""
        keep = 1.0
        for s in self.stack(key):
            keep *= 1.0 - float(self.cover[s])
        return 1.0 - keep

    # ---- derived properties of a layer -----------------------------------------------
    @staticmethod
    def _liquid_mass(comp):
        t = 0.0
        for ck, g in comp:
            if (ck >> 8) & 3 == DISSOLVED:
                r = ROWS[ck >> 10]
                if "solute" not in r["tags"]:
                    t += g
        return t

    @staticmethod
    def _grain_params(comp):
        """(mass weighted por_dep, por_wet) over GRAIN entries."""
        gm = pd = pw = 0.0
        for ck, g in comp:
            if (ck >> 8) & 3 == GRAIN:
                r = ROWS[ck >> 10]
                gm += g
                pd += g * r["por_dep"]
                pw += g * r["por_wet"]
        if gm <= 0.0:
            return 0.0, 0.0
        return pd / gm, pw / gm

    @staticmethod
    def _mud(comp, T_C=AMBIENT):
        """(phi, tau_y, rho_kgm3 of the bulk) of a layer's grain suspension."""
        vl, _vc, vg, gm = LL.comp_volumes(comp)
        if vg <= 0.0:
            return 0.0, 0.0, 1000.0
        phi = vg / (vg + vl) if (vg + vl) > 0.0 else 1.0
        num = den = pr = 0.0
        for ck, g in comp:
            if (ck >> 8) & 3 == GRAIN:
                r = ROWS[ck >> 10]
                v = g / r["rho_c"]
                num += v * r["tau_ref"]
                pr += v * r["phi_ref"]
                den += v
        tau_ref, phi_ref = num / den, pr / den
        tau = LL.yield_stress(phi, tau_ref, phi_ref)
        mt = sum(g for _c, g in comp)
        vt = vl + vg + sum(g / ROWS[ck >> 10]["rho_c"] for ck, g in comp
                           if (ck >> 8) & 3 in (CRYSTAL, BOUND))
        return phi, tau, 1000.0 * mt / vt if vt > 0 else 1000.0

    def _strengths(self, comp):
        tot = sum(g for _c, g in comp)
        if tot <= 0.0:
            return 0.0, 0.0
        phi_l = self._liquid_mass(comp) / tot
        rows = [(g, ROWS[ck >> 10], (ck >> 8) & 3) for ck, g in comp]
        a = LL.mix_strength(rows, "a", phi_l)
        c = LL.mix_strength(rows, "c", phi_l)
        _phi, tau, _rho = self._mud(comp)
        return a, max(c, tau)

    def _fluid_row(self, comp):
        """The dominant liquid or crystal species row of a layer (hold limits)."""
        best, bg = None, 0.0
        for ck, g in comp:
            r = ROWS[ck >> 10]
            if (ck >> 8) & 3 in (DISSOLVED, CRYSTAL) and r["liq_class"] and \
                    "solute" not in r["tags"] and g > bg:
                best, bg = r, g
        return best

    def _hold(self, comp, key, sub, film_m=None):
        """Hold thickness of a face for this layer's liquid (4.4)."""
        r = self._fluid_row(comp)
        sb = substrate(sub)
        _phi, tau, rho = self._mud(comp)
        if r is None:
            return HOLD_MIN
        film = FILM * self.dx if film_m is None else film_m
        theta = sb["theta"].get(r["liq_class"], 60.0)
        return LL.hold_thickness(orientation(key), sb["rough_um"] * 1e-6, tau,
                                 1000.0 * r["rho_l"] if rho <= 0 else rho, r["sigma"],
                                 theta, film)

    def _rule_cover(self, comp, key, sub):
        """The cover rule (4.1a). Liquid-like layers (liquid or ice at least as much
        as the solid framework): c = min(1, V/(A h_eq)). Particulate layers:
        c = 1 - exp(-kappa_eff m_a)."""
        vl, vc, vg, _gm = LL.comp_volumes(comp)
        A = self.area
        vf = vl
        vo = vg
        for ck, g in comp:
            f = (ck >> 8) & 3
            if f in (CRYSTAL, BOUND):
                r = ROWS[ck >> 10]
                if f == CRYSTAL and r["liq_class"]:
                    vf += g / r["rho_c"]
                else:
                    vo += g / r["rho_c"]
        if vf > 0.0 and vf >= vo:
            V = LL.bulk_volume(vl, vc, vg, 0.0) * 1e-6
            return LL.cover_liquid(V, A, self._hold(comp, key, sub))
        mp = kp = 0.0
        for ck, g in comp:
            if (ck >> 8) & 3 != DISSOLVED:
                k = ROWS[ck >> 10]["kappa"] or KAPPA_MIN
                mp += g
                kp += g * k
        if mp <= 0.0:
            return 0.0
        return LL.cover_particulate(kp / mp, mp / A)

    def _derive(self, s, key, sub=None, shrink=True):
        """Refresh m, porosity, adhesion, cohesion and (optionally) cover after a
        composition change."""
        comp = self._comp(s)
        self.m[s] = float(self.cm[s].sum())
        if not comp:
            return
        vl, _vc, vg, _gm = LL.comp_volumes(comp)
        if vg > 0.0:
            pd, pw = self._grain_params(comp)
            self.por[s] = LL.porosity_collapse(float(self.por[s]), pd, pw, vl, vg)
        a, c = self._strengths(comp)
        self.adh[s], self.coh[s] = a, c
        if shrink:
            self.cover[s] = min(float(self.cover[s]), self._rule_cover(comp, key, sub))

    # ---- energy bookkeeping -----------------------------------------------------------
    @staticmethod
    def _cp_mass(entries):
        return sum(g * LL.cp_of(ROWS[ck >> 10], (ck >> 8) & 3) for ck, g in entries)

    @staticmethod
    def _offset_sum(entries):
        return sum(g * LL.form_offset(ROWS[ck >> 10], (ck >> 8) & 3) for ck, g in entries)

    def stored_J(self):
        """J the layers STORE as phase and solution offsets (the part of F2 that is
        not sensible heat): sum of mass * offset over every entry."""
        t = 0.0
        for s in np.flatnonzero(self.key[:self.hi] >= 0):
            t += self._offset_sum(self._comp(int(s)))
        return t

    def energy_in_total(self):
        return float(sum(self.energy_in.values()))

    def _book(self, d, sp, tag, g):
        d[(sp, tag)] = d.get((sp, tag), 0.0) + g

    # ---- add ----------------------------------------------------------------------------
    def add(self, key, comp, cover=None, T_in=None, Ts=None, source="", pattern=None,
            content=None, inside=False, sub=None, latent_in=0.0):
        """Put matter on a face. comp = {(species, form, sz): grams}. Returns
        (slot, adj). The mass enters at T_in (default the host's Ts) and brings
        `latent_in` J/g on top (condensation passes Lv). It merges into the top
        layer when alike (same liquid class and species cosine >= 0.9), else it
        pushes a new top layer; a ninth layer merges the two thinnest. Acid
        species raise ValueError (they stay lattice, 4.14). Bad input is refused
        before anything is touched: a key that make_key could not have made, a
        negative or non-finite mass, a form or size class out of range, a cover
        outside (0, 1], a non-finite temperature, a missing or non-finite latent:
        ValueError; a species with no row: KeyError. comp may be a dict or any
        iterable of (key, grams) pairs (a generator is read once)."""
        key = _check_face_key(key)
        comp = _check_comp(comp)                     # read once: a generator survives
        _check_number("cover", cover, 0.0, 1.0, lo_open=True)
        _check_number("T_in", T_in)
        _check_number("Ts", Ts)
        _check_number("latent_in", latent_in, none_ok=False)
        _check_sub("sub", sub)
        ents = _entries(comp)
        if not ents:
            return -1, []
        for ck, _g in ents:
            _check_species(ck)
        if T_in is None:
            T_in = Ts if Ts is not None else AMBIENT
        if Ts is None:
            Ts = T_in
        dC = self._cp_mass(ents)
        dE = dC * (T_in - AMBIENT) + sum(g for _c, g in ents) * latent_in
        off = self._offset_sum(ents)
        tag = source or "add"
        for ck, g in ents:
            self._book(self.led_in, ck >> 10, tag, g)
        self.energy_in[tag] = self.energy_in.get(tag, 0.0) + dE + off
        pat = -1
        if pattern is not None:
            self.patterns.append(pattern)
            pat = len(self.patterns) - 1
        con = -1
        if content is not None:
            self.contents.append(content)
            con = len(self.contents) - 1
        slot, extra = self._put(key, ents, cover, pat, con, inside, sub)
        adj = [(key, dE, dC)] + extra
        if slot >= 0:
            adj += self._equilibrate(slot, key, Ts)
            for _ in range(4):
                more = self._cross_dissolve(key, Ts)
                if not more:
                    break
                adj += more
        self.gen += 1
        self._touch(host_of(key))
        return slot, _merge_adj(adj)

    def _alike(self, s, ents):
        old = dict(self._comp(s))
        cls_a = {ROWS[c >> 10]["liq_class"] for c, _g in old.items()
                 if (c >> 8) & 3 == DISSOLVED and ROWS[c >> 10]["liq_class"]}
        cls_b = {ROWS[c >> 10]["liq_class"] for c, _g in ents
                 if (c >> 8) & 3 == DISSOLVED and ROWS[c >> 10]["liq_class"]}
        if cls_a and cls_b and cls_a != cls_b:
            return False
        sa = {c >> 10 for c in old}
        sb_ = {c >> 10 for c, _g in ents}
        if (D_WATER in sa and all(ROWS[x]["sol0"] is not None for x in sb_)) or \
                (D_WATER in sb_ and all(ROWS[x]["sol0"] is not None for x in sa)):
            return True                     # soluble matter on or in water is one solution
        va, vb = {}, {}
        for c, g in old.items():
            va[c >> 10] = va.get(c >> 10, 0.0) + g
        for c, g in ents:
            vb[c >> 10] = vb.get(c >> 10, 0.0) + g
        dot = sum(va.get(k, 0.0) * vb.get(k, 0.0) for k in set(va) | set(vb))
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        return na > 0 and nb > 0 and dot / (na * nb) >= 0.9

    def _new_por(self, ents):
        pd, pw = self._grain_params(ents)
        if pd == 0.0:                                     # a crystal deposit (frost)
            for ck, _g in ents:
                if (ck >> 8) & 3 == CRYSTAL:
                    pd = max(pd, ROWS[ck >> 10]["por_dep"])
            return pd
        vl, _vc, vg, _gm = LL.comp_volumes(ents)
        return LL.porosity_collapse(pd, pd, pw, vl, vg)

    def _put(self, key, ents, cover, pat, content, inside, sub, por=None, adh_coh=None):
        """Merge matter into the top layer of the right kind or push it. No ledger,
        no energy: callers book. Returns (slot, extra adj)."""
        chain = self._chain(key)
        extra = []
        c_new = cover if cover is not None else self._rule_cover(ents, key, sub)
        p_new = por if por is not None else self._new_por(ents)
        if inside:
            tgt = next((s for s in chain if self.inside[s]), -1)
            if tgt >= 0:
                e = self._merge_into(tgt, key, ents, c_new, p_new, sub)
                return tgt, [(key, e[0], e[1])]
            s = self._alloc()
            self._init_slot(s, key, ents, c_new, p_new, pat, content, True)
            if chain:
                self.below[chain[-1]] = s
            else:
                self._set_top(key, s)
            e = self._write_comp(s, ents)
            extra.append((key, e[0], e[1]))
            self._derive(s, key, sub, shrink=False)
            return s, extra
        surf = [s for s in chain if not self.inside[s]]
        top = surf[0] if surf else -1
        if top >= 0 and pat < 0 and self.pat[top] < 0 and self._alike(top, ents):
            e = self._merge_into(top, key, ents, c_new, p_new, sub)
            return top, [(key, e[0], e[1])]
        s = self._alloc()
        self._init_slot(s, key, ents, c_new, p_new, pat, content, False)
        self.below[s] = chain[0] if chain else -1
        self._set_top(key, s)
        e = self._write_comp(s, ents)
        extra.append((key, e[0], e[1]))
        self._derive(s, key, sub, shrink=False)
        if len(surf) + 1 > DEPTH_CAP:
            s, e1 = self._merge_thinnest(key, s, sub)
            extra.append((key, e1[0], e1[1]))
        extra += self._thin_top(key)
        return s, extra

    def _init_slot(self, s, key, ents, cover, por, pat, content, inside):
        self.key[s] = key
        self.cover[s] = cover
        self.por[s] = por
        self.t_wet[s] = 0.0
        self.t_last[s] = self.now
        self.inside[s] = 1 if inside else 0
        self.depth[s] = 0.0
        self.pat[s] = pat
        self.content[s] = content

    def _merge_into(self, s, key, ents, c_new, p_new, sub):
        old = self._comp(s)
        _vl, _vc, g1, _m = LL.comp_volumes(old)
        _vl2, _vc2, g2, _m2 = LL.comp_volumes(ents)
        por = LL.merge_porosity(g1, float(self.por[s]), g2, p_new) if g1 + g2 > 0 else \
            max(float(self.por[s]), p_new)
        self.cover[s] = LL.cover_overlap(float(self.cover[s]), c_new)
        e = self._write_comp(s, old + list(ents))
        self.por[s] = por
        self._derive(s, key, sub, shrink=False)
        return e

    def _merge_thinnest(self, key, newest, sub):
        """A ninth push merges the two thinnest neighbours (4.9): masses add exactly;
        por, adh, coh mix; cover by random overlap. Returns the surviving slot of the
        merge that holds `newest`'s matter (or newest itself)."""
        surf = self.stack(key)
        best, bi = None, 0
        for i in range(len(surf) - 1):
            t = self.thickness(surf[i]) + self.thickness(surf[i + 1])
            if best is None or t < best:
                best, bi = t, i
        up, dn = surf[bi], surf[bi + 1]
        comp_u, comp_d = self._comp(up), self._comp(dn)
        _a, _b, gu, _c = LL.comp_volumes(comp_u)
        _a, _b, gd, _c = LL.comp_volumes(comp_d)
        por = LL.merge_porosity(gu, float(self.por[up]), gd, float(self.por[dn])) \
            if gu + gd > 0 else max(float(self.por[up]), float(self.por[dn]))
        cov = LL.cover_overlap(float(self.cover[up]), float(self.cover[dn]))
        e = self._write_comp(up, comp_u + comp_d)
        self.por[up] = por
        self.cover[up] = cov
        if self.pat[up] < 0:
            self.pat[up] = self.pat[dn]
        self._derive(up, key, sub, shrink=False)
        self.below[up] = self.below[dn]
        self._release_slot(dn)
        return up, e

    def _thin_top(self, key):
        """A top layer thinner than MIN_LAYER_G merges into the one under it, so
        the stack under a vanishing layer shows (mud gone, chalk under it)."""
        surf = self.stack(key)
        if len(surf) < 2:
            return []
        t = surf[0]
        if self.m[t] < MIN_LAYER_G and self.pat[t] < 0:
            dn = surf[1]
            e = self._write_comp(dn, self._comp(dn) + self._comp(t))
            self._derive(dn, key, None, shrink=False)
            self.below[t], nxt = -1, int(self.below[t])
            self._set_top(key, nxt)
            self._release_slot(t)
            return [(key, e[0], e[1])]
        return []

    # ---- solutes crystallise and redissolve (4.7) -------------------------------------
    def _equilibrate(self, s, key, Ts):
        """Dissolved solutes beyond solubility(T) move to CRYSTAL and back, with the
        heat of solution booked. Returns adj."""
        comp = self._comp(s)
        wl = sum(g for ck, g in comp if (ck >> 8) & 3 == DISSOLVED and
                 "solvent" in ROWS[ck >> 10]["tags"])
        by = {}
        for ck, g in comp:
            sp, f = ck >> 10, (ck >> 8) & 3
            if f in (DISSOLVED, CRYSTAL) and ROWS[sp]["sol0"] is not None:
                a = by.setdefault(sp, [0.0, 0.0])
                a[0 if f == DISSOLVED else 1] += g
        deltas = {}
        for sp, (gd, gc) in by.items():
            sol = LL.solubility(ROWS[sp], Ts)
            tot = gd + gc
            tgt = min(tot, sol * wl) if wl > 0.0 else 0.0
            if abs(tgt - gd) > 1e-15 * max(tot, 1.0):
                deltas[(sp, DISSOLVED)] = tgt - gd
                deltas[(sp, CRYSTAL)] = gd - tgt
        if not deltas:
            return []
        dC, dst, fE, fC = self._apply_sf(s, key, deltas)
        return [(key, -dst + fE, dC + fC)]

    def _cross_dissolve(self, key, Ts):
        """Soluble CRYSTAL in a layer next to a water layer dissolves into it up to
        saturation (liquid returning to a salt crust, salt thrown on a wet film), with
        the heat of solution booked. Returns adj."""
        st = self.stack(key)
        adj = []
        for i in range(len(st) - 1):
            for liq, cry in ((st[i], st[i + 1]), (st[i + 1], st[i])):
                if self.key[liq] < 0 or self.key[cry] < 0:
                    continue
                lc = self._comp(liq)
                wl = sum(g for c, g in lc if (c >> 8) & 3 == DISSOLVED and
                         "solvent" in ROWS[c >> 10]["tags"])
                if wl <= 0.0:
                    continue
                cc = self._comp(cry)
                move = {}
                for sp in sorted({c >> 10 for c, _g in cc}):
                    r = ROWS[sp]
                    if r["sol0"] is None:
                        continue
                    crys = sum(g for c, g in cc if c >> 10 == sp and (c >> 8) & 3 == CRYSTAL)
                    if crys <= 0.0:
                        continue
                    dis = sum(g for c, g in lc if c >> 10 == sp and (c >> 8) & 3 == DISSOLVED)
                    room = LL.solubility(r, Ts) * wl - dis
                    if room > 0.0:
                        move[sp] = min(crys, room)
                if not move:
                    continue
                dC = dst = 0.0
                d_c, d_l = dict(cc), dict(lc)
                for sp, g in move.items():
                    r = ROWS[sp]
                    cks = [c for c in d_c if c >> 10 == sp and (c >> 8) & 3 == CRYSTAL]
                    tot = sum(d_c[c] for c in cks)
                    for c in cks:
                        d_c[c] -= d_c[c] * (g / tot)
                    ck = pack_key(sp, DISSOLVED, 0)
                    d_l[ck] = d_l.get(ck, 0.0) + g
                    dC += g * (LL.cp_of(r, DISSOLVED) - LL.cp_of(r, CRYSTAL))
                    dst += g * (LL.form_offset(r, DISSOLVED) - LL.form_offset(r, CRYSTAL))
                e1 = self._write_comp(cry, d_c)
                e2 = self._write_comp(liq, d_l)
                self._derive(liq, key, None, shrink=False)
                if self.m[cry] <= 0.0 or not self._comp(cry):
                    self._unlink(key, cry)
                else:
                    self._derive(cry, key, None, shrink=True)
                adj.append((key, -dst + e1[0] + e2[0], dC + e1[1] + e2[1]))
                return adj
        return adj

    def _apply_sf(self, s, key, deltas, sub=None):
        """Internal conversion: {(species, form): dg}. Positive adds to the largest
        entry of that species/form (or a new one), negative removes in proportion.
        Returns (dC, dStored, fold dE, fold dC). The caller's host energy change
        is dE = -dStored (nothing crossed the boundary)."""
        d = dict(self._comp(s))
        dC = dst = 0.0
        for (sp, f), g in deltas.items():
            if g == 0.0:
                continue
            r = ROWS[sp]
            cks = [c for c in d if (c >> 10) == sp and (c >> 8) & 3 == f]
            if g > 0.0:
                ck = max(cks, key=lambda c: (d[c], -c)) if cks else pack_key(sp, f, _default_sz(f))
                d[ck] = d.get(ck, 0.0) + g
            else:
                tot = sum(d[c] for c in cks)
                if tot <= 0.0:
                    continue
                take = min(-g, tot)
                fr = take / tot
                for c in cks:
                    d[c] -= d[c] * fr
                g = -take
            dC += g * LL.cp_of(r, f)
            dst += g * LL.form_offset(r, f)
        m0 = float(self.m[s])
        fE, fC = self._write_comp(s, d)
        self._derive(s, key, sub, shrink=float(self.m[s]) < m0 * (1.0 - 1e-12))
        self._touch(host_of(key))
        return dC, dst, fE, fC

    # ---- take / move / strip / relocate / release ---------------------------------------
    def take(self, key, species, form, grams, Ts=AMBIENT, latent=0.0, tag="evap", sub=None):
        """Remove up to `grams` of species/form from the top down (evaporation).
        Returns (grams taken, adj). `latent` J/g is the heat the caller's phase
        change takes out of the skin (Lv for evaporation); the mass leaves with its
        enthalpy plus that latent, booked under `tag`. Bad input is refused before
        anything is touched, as in add: a key that make_key could not have made, a
        form not 0..3, a negative or non-finite mass, a non-finite temperature or
        latent, a tag that is not text: ValueError; a species with no row: KeyError."""
        key = _check_face_key(key)
        species = _check_int("species", species, 0, (1 << 53))
        if species not in ROWS:
            raise KeyError(f"no layer row for species {species}")
        form = _check_int("form", form, 0, 3)
        _check_number("grams", grams, 0.0, none_ok=False)
        _check_number("Ts", Ts, none_ok=False)
        _check_number("latent", latent, none_ok=False)
        if not isinstance(tag, str):
            raise ValueError(f"tag must be text, got {tag!r}")
        _check_sub("sub", sub)
        want = float(grams)
        got = 0.0
        adj = []
        for s in self._chain(key):
            if want - got <= 0.0:
                break
            d = {}
            comp = self._comp(s)
            cks = [(c, g) for c, g in comp if c >> 10 == species and (c >> 8) & 3 == form]
            tot = sum(g for _c, g in cks)
            if tot <= 0.0:
                continue
            t = min(tot, want - got)
            fr = t / tot
            new = dict(comp)
            for c, g in cks:
                new[c] = g - g * fr
            r = ROWS[species]
            dC = -t * LL.cp_of(r, form)
            fE, fC = self._write_comp(s, new)
            self._book(self.led_out, species, tag, t)
            H_out = t * LL.cp_of(r, form) * (Ts - AMBIENT) + t * LL.form_offset(r, form)
            self.energy_in[tag] = self.energy_in.get(tag, 0.0) - H_out - t * latent
            adj.append((key, dC * (Ts - AMBIENT) - t * latent + fE, dC + fC))
            got += t
            adj += self._after_loss(key, s, sub)
            adj += self._equilibrate(s, key, Ts) if self.key[s] >= 0 else []
        self.gen += 1
        self._touch(host_of(key))
        return got, _merge_adj(adj)

    def _after_loss(self, key, s, sub):
        """Drop a layer that ran empty, else re-derive it (cover may shrink); a top
        layer left thinner than MIN_LAYER_G merges into the one under it. Returns adj
        (the fold cost of that merge, normally empty)."""
        if self.m[s] <= 0.0 or not self._comp(s):
            self._unlink(key, s)
            return []
        self._derive(s, key, sub, shrink=True)
        return self._thin_top(key)

    def _unlink(self, key, s):
        chain = self._chain(key)
        i = chain.index(s)
        nxt = int(self.below[s])
        if i == 0:
            self._set_top(key, nxt)
        else:
            self.below[chain[i - 1]] = nxt
        self._release_slot(s)

    def _layer_frac(self, s, frac_entries):
        """{ck: g} of the share frac_entries (a float or {ck: fraction}) of layer s."""
        comp = self._comp(s)
        if isinstance(frac_entries, dict):
            return {c: g * frac_entries.get(c, 0.0) for c, g in comp if frac_entries.get(c, 0.0) > 0}
        return {c: g * frac_entries for c, g in comp}

    def _transfer(self, s, src_key, dst_key, part, Ts_src, Ts_dst, cover=None, sub_dst=None,
                  sub_src=None, tag="move", mirror_pat=None, to_inside=None):
        """Move `part` ({ck: g}) of layer s to the top of dst. Ledger neutral. Returns
        adj for both hosts: the moved heat capacity at the donor's temperature."""
        ents = sorted((c, g) for c, g in part.items() if g > 0.0)
        if not ents:
            return []
        comp = self._comp(s)
        tot = sum(g for _c, g in comp)
        moved = sum(g for _c, g in ents)
        new = dict(comp)
        for c, g in ents:
            new[c] = max(0.0, new.get(c, 0.0) - g)
        fE, fC = self._write_comp(s, new)
        dC = self._cp_mass(ents)
        cov_src = float(self.cover[s])
        pat, con = int(self.pat[s]), int(self.content[s])
        por_src = float(self.por[s])
        was_inside = bool(self.inside[s])
        loss_adj = self._after_loss(src_key, s, sub_src)
        c_dst = cover if cover is not None else min(1.0, cov_src)
        pat_dst = mirror_pat if mirror_pat is not None else pat
        slot, extra = self._put(dst_key, ents, c_dst, pat_dst, con,
                                was_inside if to_inside is None else to_inside, sub_dst,
                                por=por_src)
        adj = [(src_key, -dC * (Ts_src - AMBIENT) + fE, -dC + fC),
               (dst_key, dC * (Ts_src - AMBIENT), dC)] + extra + loss_adj
        self.gen += 1
        self._touch(host_of(src_key), host_of(dst_key))
        return adj

    def move(self, src_key, dst_key, fractions, Ts_src=AMBIENT, Ts_dst=AMBIENT, cover=None,
             sub_src=None, sub_dst=None):
        """Move shares of the surface layers of src onto dst. `fractions` is a float
        (the same share of every layer) or a sequence aligned with stack(src), top
        first (one share per layer, read once: a generator works). Layers land bottom
        first so the order is kept (F7). Returns adj. Bad input is refused before
        anything is touched: a key that make_key could not have made, a share outside
        0..1 or not finite (a share over 1 would book matter the layer never held), a
        list of shares that is not one per layer, a non-finite temperature, a cover
        outside (0, 1], a substrate argument that substrate() could not read:
        ValueError (KeyError for a substrate name or id with no row)."""
        src_key, dst_key = _check_face_key(src_key), _check_face_key(dst_key)
        if hasattr(fractions, "__next__"):               # a one-shot iterator or generator: read once
            fractions = list(fractions)
        if np.ndim(fractions) == 0:
            _check_number("fractions", fractions, 0.0, 1.0, none_ok=False)
        else:
            fractions = list(fractions)
            for f in fractions:
                _check_number("fractions", f, 0.0, 1.0, none_ok=False)
        _check_number("Ts_src", Ts_src, none_ok=False)
        _check_number("Ts_dst", Ts_dst, none_ok=False)
        _check_number("cover", cover, 0.0, 1.0, lo_open=True)
        _check_sub("sub_src", sub_src)
        _check_sub("sub_dst", sub_dst)
        st = self.stack(src_key)
        if np.ndim(fractions) != 0 and st and len(fractions) != len(st):
            raise ValueError(f"fractions has {len(fractions)} shares for the {len(st)} layers "
                             "on the source face (one per layer, top first)")
        if not st or src_key == dst_key:
            return []
        fr = [float(fractions)] * len(st) if np.ndim(fractions) == 0 else list(fractions)
        parts = [(s, self._layer_frac(s, f)) for s, f in zip(st, fr) if f > 0.0]
        adj = []
        for s, part in reversed(parts):
            if self.key[s] != src_key:
                continue
            adj += self._transfer(s, src_key, dst_key, part, Ts_src, Ts_dst, cover,
                                  sub_dst, sub_src)
        return _merge_adj(adj)

    def strip(self, key, depth_m, Ts=AMBIENT, sub=None):
        """What a cut or plane to depth_m removes: surface layers from the top down
        by thickness, then (if the cut goes past them by at least the INSIDE depth)
        the INSIDE layer. Scrubbing never calls this. Returns (comp, adj). A key that
        make_key could not have made, a negative or non-finite depth or a non-finite
        temperature is ValueError."""
        key = _check_face_key(key)
        _check_number("depth_m", depth_m, 0.0, none_ok=False)
        _check_number("Ts", Ts, none_ok=False)
        _check_sub("sub", sub)
        removed = {}
        adj = []
        left = float(depth_m)
        for s in self.stack(key):
            h = self.thickness(s)
            if left <= 0.0:
                break
            fr = 1.0 if left >= h else left / h
            part = self._layer_frac(s, fr)
            for c, g in part.items():
                removed[c] = removed.get(c, 0.0) + g
            adj += self._take_out(key, s, part, Ts, "strip", sub)
            left -= h
        ins = [s for s in self._chain(key) if self.inside[s]]
        if ins and left >= float(self.depth[ins[0]]):
            s = ins[0]
            part = self._layer_frac(s, 1.0)
            for c, g in part.items():
                removed[c] = removed.get(c, 0.0) + g
            adj += self._take_out(key, s, part, Ts, "strip", sub)
        self.gen += 1
        return {(c >> 10, (c >> 8) & 3, c & 255): g for c, g in removed.items()}, _merge_adj(adj)

    def _take_out(self, key, s, part, Ts, tag, sub=None):
        """Remove `part` of layer s to outside the pool, with ledger and energy."""
        ents = sorted(part.items())
        comp = self._comp(s)
        new = dict(comp)
        for c, g in ents:
            new[c] = max(0.0, new.get(c, 0.0) - g)
        fE, fC = self._write_comp(s, new)
        dC = -self._cp_mass(ents)
        H = self._cp_mass(ents) * (Ts - AMBIENT) + self._offset_sum(ents)
        for c, g in ents:
            self._book(self.led_out, c >> 10, tag, g)
        self.energy_in[tag] = self.energy_in.get(tag, 0.0) - H
        more = self._after_loss(key, s, sub)
        self._touch(host_of(key))
        return [(key, dC * (Ts - AMBIENT) + fE, dC + fC)] + more

    def release(self, keys, Ts=None):
        """The host is gone: every layer of these keys leaves the pool (tag
        'release'). Returns (composition dict, adj); adj is empty when Ts is None
        (the host cell no longer exists to take it)."""
        out = {}
        adj = []
        for key in keys:
            for s in list(self._chain(key)):
                part = self._layer_frac(s, 1.0)
                for c, g in part.items():
                    out[c] = out.get(c, 0.0) + g
                adj += self._take_out(key, s, part, AMBIENT if Ts is None else Ts, "release")
        self.gen += 1
        return ({(c >> 10, (c >> 8) & 3, c & 255): g for c, g in out.items()},
                _merge_adj(adj) if Ts is not None else [])

    def relocate(self, src_keys, dst_keys):
        """A cell moved (fell, carried, thrown): re-key its faces. A destination that
        already has layers gets the moved stack on top (through the normal put).
        Mass and heat do not change (the host moves its own E)."""
        for a, b in zip(src_keys, dst_keys):
            if a == b:
                continue
            ch = self._chain(a)
            if not ch:
                continue
            self._set_top(a, -1)
            if self._find(b) < 0:
                for s in ch:
                    self.key[s] = b
                self._set_top(b, ch[0])
            else:
                for s in reversed(ch):
                    ents = self._comp(s)
                    self._put(b, ents, float(self.cover[s]), int(self.pat[s]),
                              int(self.content[s]), bool(self.inside[s]), None,
                              por=float(self.por[s]))
                    self.pat[s] = -1
                    self.below[s] = -1
                    self._release_slot(s)
            self._touch(host_of(a), host_of(b))
        self.gen += 1

    def _chain_from(self, s):
        out = []
        while s >= 0 and len(out) < 64:
            out.append(s)
            s = int(self.below[s])
        return out

    # ---- totals, ledger, canonical form --------------------------------------------------
    def mass(self, key=None, species=None, form=None):
        """Grams, optionally of one key, species and form."""
        if key is None:
            slots = [int(s) for s in np.flatnonzero(self.key[:self.hi] >= 0)]
        else:
            slots = self._chain(key)
        t = 0.0
        for s in slots:
            for ck, g in self._comp(s):
                if species is not None and ck >> 10 != species:
                    continue
                if form is not None and (ck >> 8) & 3 != form:
                    continue
                t += g
        return t

    def mass_on(self, keys):
        """Grams on each of these keys (support loads, thrown-object momentum, body
        mass and carrying, 4.13): mud on boots, ice on a rope and 500 ml of blood
        have weight. Returns a float64 array aligned with keys."""
        return np.array([self.mass(int(k)) for k in keys], np.float64)

    def total(self, species, form=None):
        return self.mass(None, species, form)

    def species_totals(self):
        t = {}
        for s in np.flatnonzero(self.key[:self.hi] >= 0):
            for ck, g in self._comp(int(s)):
                t[ck >> 10] = t.get(ck >> 10, 0.0) + g
        return t

    def ledger_closes(self, rtol=1e-12):
        """Per species, grams in - grams out equals what the pool holds."""
        tot = self.species_totals()
        sps = set(tot) | {k[0] for k in self.led_in} | {k[0] for k in self.led_out}
        for sp in sps:
            i = sum(v for k, v in self.led_in.items() if k[0] == sp)
            o = sum(v for k, v in self.led_out.items() if k[0] == sp)
            held = tot.get(sp, 0.0)
            if abs(i - o - held) > rtol * max(i, o, held, 1e-12) + 1e-15:
                return False
        return True

    def canonical(self):
        """Order-free picture of the state: {key: [(ckey, grams, cover, por, inside)]}
        top first. Equal for equal states however they were built."""
        out = {}
        for k in self.keys().tolist():
            out[k] = [(tuple(self._comp(s)), round(float(self.cover[s]), 6),
                       round(float(self.por[s]), 6), int(self.inside[s]))
                      for s in self._chain(k)]
        return out

    # ---- read-outs ----------------------------------------------------------------------
    def describe(self, key):
        """Top first: [{label, mass_g, thick_m, wet, tacky, colour}], labels read from
        composition alone (F4) through the LABELS data table."""
        out = []
        for s in self._chain(key):
            comp = self._comp(s)
            tot = sum(g for _c, g in comp)
            if tot <= 0.0:
                continue
            rd = {"liq": 0.0, "por": float(self.por[s])}
            for ck, g in comp:
                r = ROWS[ck >> 10]
                f = (ck >> 8) & 3
                x = g / tot
                if f == DISSOLVED and "solute" not in r["tags"]:
                    rd["liq"] += x
                pre = {DISSOLVED: "dis", CRYSTAL: "cry", GRAIN: "grain", BOUND: "bnd"}[f]
                for t in r["tags"]:
                    rd[f"{pre}:{t}"] = rd.get(f"{pre}:{t}", 0.0) + x
                    rd[f"any:{t}"] = rd.get(f"any:{t}", 0.0) + x
            label = "layer"
            for lab, conds in LABELS:
                if all(_cmp(rd.get(q, 0.0), op, v) for q, op, v in conds):
                    label = lab
                    break
            wet = rd["liq"] >= 0.10
            tacky = 0.02 <= rd["liq"] < 0.6 and float(self.adh[s]) >= 1000.0
            out.append({"label": label, "mass_g": tot, "thick_m": self.thickness(s),
                        "wet": wet, "tacky": bool(tacky), "colour": self._colour(comp, tot, wet)})
        return out

    @staticmethod
    def _colour(comp, tot, wet):
        rgb = np.zeros(3)
        w = 0.0
        for ck, g in comp:
            r = ROWS[ck >> 10]
            if (ck >> 8) & 3 == DISSOLVED and "solvent" in r["tags"]:
                continue
            rgb += g * np.array(r["albedo"])
            w += g
        rgb = rgb / w if w > 0 else np.array([0.5, 0.5, 0.5])
        return tuple(float(x * (COLOUR_WET if wet else 1.0)) for x in rgb)

    def thermal(self, key, sub=None):
        """(R_series m2K/W, eps_eff, alpha_solar_eff, C_layers J/m2K) of a face
        (4.13): R = sum V_i/(A k_i), emissivity and absorptivity composited from the
        top down with each layer's cover, C = sum m cp /A (inside layers count for C
        only)."""
        sb = substrate(sub)
        A = self.area
        R = C = 0.0
        eps = alpha = 0.0
        rem = 1.0
        for s in self._chain(key):
            comp = self._comp(s)
            if not comp:
                continue
            C += self._cp_mass(comp) / A
            if self.inside[s]:
                continue
            V = self._bulk_cm3(s, comp) * 1e-6
            mt = sum(g for _c, g in comp)
            k = sum(g * (ROWS[c >> 10]["k_l"] if (c >> 8) & 3 == DISSOLVED
                         else ROWS[c >> 10]["k_c"]) for c, g in comp) / mt
            R += V / (A * k)
            c = float(self.cover[s])
            e = sum(g * (ROWS[c_ >> 10]["eps_l"] if (c_ >> 8) & 3 == DISSOLVED
                         else ROWS[c_ >> 10]["eps_c"]) for c_, g in comp) / mt
            a = sum(g * (ROWS[c_ >> 10]["alpha_l"] if (c_ >> 8) & 3 == DISSOLVED
                         else ROWS[c_ >> 10]["alpha_c"]) for c_, g in comp) / mt
            eps += rem * c * e
            alpha += rem * c * a
            rem *= 1.0 - c
        return R, eps + rem * sb["eps"], alpha + rem * sb["alpha"], C

    def wet(self, key, fvol_above=0.0, cap_ml=None):
        """(liquid grams, wetted cover share, {species: liquid grams}). `fvol_above`
        is the lattice film (ml) on the up face, so wetness reads both stores."""
        liq = {}
        keep = 1.0
        for s in self.stack(key):
            comp = self._comp(s)
            lm = 0.0
            for ck, g in comp:
                if (ck >> 8) & 3 == DISSOLVED and "solute" not in ROWS[ck >> 10]["tags"]:
                    liq[ck >> 10] = liq.get(ck >> 10, 0.0) + g
                    lm += g
            if lm > 0.0:
                keep *= 1.0 - float(self.cover[s])
        share = 1.0 - keep
        if fvol_above > 0.0:
            cap = (self.dx ** 3 * 1e6) if cap_ml is None else cap_ml
            lat = min(1.0, fvol_above / (FILM * cap))
            share = 1.0 - (1.0 - share) * (1.0 - lat)
            liq[WATER] = liq.get(WATER, 0.0) + fvol_above * ROWS[WATER]["rho_l"]
        return sum(liq.values()), share, liq

    def fuel(self, key):
        """(g/m2 burnable, dHc J/g mass-weighted, T_ign C lowest, rate cap g/m2/s) of
        the fuel species in a face's surface layers (4.13)."""
        A = self.area
        g = h = rc = 0.0
        ti = None
        for s in self.stack(key):
            for ck, gm in self._comp(s):
                r = ROWS[ck >> 10]
                if r["dHc"] > 0.0 and (ck >> 8) & 3 != BOUND:
                    g += gm
                    h += gm * r["dHc"]
                    rc += gm * r["rate_cap"]
                    if r["T_ign"] is not None:
                        ti = r["T_ign"] if ti is None else min(ti, r["T_ign"])
        if g <= 0.0:
            return 0.0, 0.0, None, 0.0
        return g / A, h / g, ti, rc / g

    def grip(self, key, mu_dry, h_extra=0.0, lube=None, sub=None):
        """Effective friction (4.13): mu = mu_dry*(1-f_w) + mu_lube*f_w, f_w =
        1 - exp(-h/Ra), h the thickness of the layers (plus a lattice film
        h_extra), Ra the substrate roughness, mu_lube the mass-weighted lubricity of
        what lies there unless `lube` is given."""
        sb = substrate(sub)
        Ra = sb["rough_um"] * 1e-6
        h = float(h_extra)
        lm = lw = 0.0
        for s in self.stack(key):
            comp = self._comp(s)
            h += self.thickness(s) * float(self.cover[s])
            for ck, g in comp:
                r = ROWS[ck >> 10]
                lm += g * (r["mu_lube"] if (ck >> 8) & 3 != CRYSTAL else r["mu_lube_c"])
                lw += g
        mu_l = lube if lube is not None else (lm / lw if lw > 0.0 else mu_dry)
        return LL.friction_mu(mu_dry, mu_l, h, Ra)

    def look(self, key, sub=None):
        """Appearance for the sight code: {colour (r,g,b), gloss, patterns}. Colour is
        composited from the top down; a wet surface is darker (x0.7)."""
        sb = substrate(sub)
        A = self.area
        rem = 1.0
        rgb = np.zeros(3)
        gloss = 0.0
        pats = []
        for s in self.stack(key):
            comp = self._comp(s)
            if not comp:
                continue
            kap = mp = 0.0
            col = np.zeros(3)
            cw = 0.0
            liq = 0.0
            for ck, g in comp:
                r = ROWS[ck >> 10]
                if (ck >> 8) & 3 == DISSOLVED and "solvent" in r["tags"]:
                    liq += g
                    continue
                kap += g * r["kappa"]
                mp += g
                col += g * np.array(r["albedo"])
                cw += g
            if cw > 0.0:
                kap /= mp
                c_opt = min(float(self.cover[s]), LL.cover_particulate(kap, mp / A)) if kap > 0 \
                    else 0.0
                col /= cw
                rgb += rem * c_opt * col
                rem *= 1.0 - c_opt
            if liq > 0.0:
                gloss = max(gloss, float(self.cover[s]))
            if self.pat[s] >= 0:
                pats.append(int(self.pat[s]))
        base = (1.0 - sb["alpha"]) * np.ones(3)
        rgb = rgb + rem * base
        if gloss > 0.0:
            rgb = rgb * (1.0 - gloss * (1.0 - COLOUR_WET))
        return {"colour": tuple(float(x) for x in rgb), "gloss": gloss, "patterns": pats}

    # ---- hand-over to the lattice (4.14) --------------------------------------------------
    def excess_up(self, key, sub=None, film_m=None):
        """Liquid on an up face beyond the hold limit h_up: [(slot, {ck: g})] of the
        excess (a share of the DISSOLVED entries), not yet removed."""
        out = []
        for s in self.stack(key):
            comp = self._comp(s)
            if not any((c >> 8) & 3 == DISSOLVED and "solute" not in ROWS[c >> 10]["tags"]
                       for c, _g in comp):
                continue
            hold = self._hold(comp, key, sub, film_m)
            h = self.thickness(s)
            if h <= hold:
                continue
            V = self._bulk_cm3(s, comp) * 1e-6
            V_ex = (h - hold) * self.area * float(self.cover[s])
            fr = min(1.0, V_ex / V) if V > 0 else 0.0
            out.append((s, {c: g * fr for c, g in comp if (c >> 8) & 3 == DISSOLVED}))
        return out

    def hand_over_up(self, key, ml, fvol_old, species=WATER, Ts=AMBIENT, sub=None):
        """Hand `ml` of the liquid layer on an up face to the lattice cell above. The
        host's fvol is float32, so the amount that LANDS is
        float32(old + ml) - float32(old); that is booked and removed from the layer,
        the residual stays. Returns (new_fvol float32, landed_ml, residual_ml, adj)."""
        old32 = np.float32(fvol_old)
        new32 = np.float32(float(old32) + ml)
        landed = float(new32) - float(old32)
        rho = ROWS[species]["rho_l"]
        grams, adj = self.take(key, species, DISSOLVED, landed * rho, Ts, tag="handover",
                               sub=sub)
        got_ml = grams / rho
        if got_ml < landed:                                   # the layer held less
            new32 = np.float32(float(old32) + got_ml)
            landed = float(new32) - float(old32)
        return new32, landed, ml - landed, adj

    # ---- the passes --------------------------------------------------------------------------
    def visit_time(self, s, t_now):
        dt = t_now - float(self.t_last[s])
        self.t_last[s] = t_now
        return max(dt, 0.0)

    def kinetics(self, key, Ts, t_now, sub=None):
        """Irreversible set, cure and dye binding (4.9): DISSOLVED -> BOUND by
        X' = 1 - (1-X) exp(-k dt) at the host Ts, every time the layer is visited.
        X is the BOUND share of the species, kept in the composition. Cure gains O2
        from the air ledger (tag 'o2'); a coat thicker than h_ref cures slower.
        Returns adj."""
        adj = []
        for s in self._chain(key):
            dt = self.visit_time(s, t_now)
            if dt <= 0.0:
                continue
            comp = self._comp(s)
            sps = {ck >> 10 for ck, _g in comp if ROWS[ck >> 10]["kin"] is not None}
            if not sps:
                continue
            h = self.thickness(s)
            deltas = {}
            o2_total = 0.0
            for sp in sorted(sps):
                kin = ROWS[sp]["kin"]
                src = sum(g for c, g in comp if c >> 10 == sp and (c >> 8) & 3 == kin["src"])
                bnd = sum(g for c, g in comp if c >> 10 == sp and (c >> 8) & 3 == BOUND)
                if src <= 0.0:
                    continue
                k = LL.arrhenius(kin["k_ref"], kin["T_ref"], kin["Ea"], Ts)
                k = LL.cure_k_eff(k, h, kin["h_ref"])
                X = bnd / (src + bnd)
                X2 = LL.first_order_step(X, k, dt)
                move = min(src, X2 * (src + bnd) - bnd)
                if move <= 0.0:
                    continue
                deltas[(sp, kin["src"])] = -move
                deltas[(sp, BOUND)] = move
                o2_total += kin["o2"] * move
            if not deltas:
                continue
            dC, dst, fE, fC = self._apply_sf(s, key, deltas, sub)
            dE = -dst + fE
            dC += fC
            for sp in sorted(sps):
                kin = ROWS[sp]["kin"]
                mv = deltas.get((sp, BOUND), 0.0)
                if kin["o2"] and mv > 0.0:
                    g = kin["o2"] * mv
                    self._o2_in(s, key, sp, g, Ts)
                    dC += g * ROWS[sp]["cp_c"]
                    dE += g * ROWS[sp]["cp_c"] * (Ts - AMBIENT)
                    self.energy_in["o2"] = (self.energy_in.get("o2", 0.0)
                                            + g * ROWS[sp]["cp_c"] * (Ts - AMBIENT)
                                            + g * LL.form_offset(ROWS[sp], BOUND))
            adj.append((key, dE, dC))
            self._derive(s, key, sub, shrink=False)
        self.gen += 1
        return _merge_adj(adj)

    def _o2_in(self, s, key, sp, g, Ts):
        """Oxygen taken from the air joins the BOUND entry of its species."""
        d = dict(self._comp(s))
        cks = [c for c in d if c >> 10 == sp and (c >> 8) & 3 == BOUND]
        ck = max(cks, key=lambda c: d[c]) if cks else pack_key(sp, BOUND, 0)
        d[ck] = d.get(ck, 0.0) + g
        self._write_comp(s, d)
        self._book(self.led_in, sp, "o2", g)

    def grow_oxide(self, key, T_C, dt, species=None, sub=None):
        """Film growth from the substrate (4.9): two-regime law on the BOUND oxide
        mass of this key. Iron comes from the substrate (ledger 'substrate'), the
        oxygen from the air ('o2'). Returns (adj, grams of iron the substrate lost)."""
        sp = species if species is not None else self._oxide_sp()
        r = ROWS[sp]
        A = self.area
        have = self.mass(key, sp, BOUND)
        x0 = have / (A * r["rho_c"] * 1e6)
        x1 = LL.oxide_step(x0, T_C, dt)
        dm = (x1 - x0) * A * r["rho_c"] * 1e6
        if dm <= 0.0:
            return [], 0.0
        fe = dm * 0.6994                       # 2 Fe / Fe2O3 by mass
        o2 = dm - fe
        _, adj = self.add(key, {(sp, BOUND, 0): fe}, T_in=T_C, Ts=T_C, source="substrate",
                          cover=1.0, sub=sub)
        _, adj2 = self.add(key, {(sp, BOUND, 0): o2}, T_in=T_C, Ts=T_C, source="o2",
                           cover=1.0, sub=sub)
        return _merge_adj(adj + adj2), fe

    @staticmethod
    def _oxide_sp():
        return IRON_OXIDE

    def soak(self, key, dt, sub, t_now=None, s_host=0.0, back_key=None):
        """Soak into a porous SUBSTRATE (4.6a): dI = S_eff (sqrt(t_w + dt) - sqrt(t_w)).
        All DISSOLVED goes in (the liquid with its solutes); GRAIN only if
        d_grain < 0.2 d_pore. What goes in becomes the INSIDE layer on the same key,
        water included, with depth V_liq/(porosity A). A page passes the rest to its
        back face. `t_now` is the sim clock (default self.now): a face dry for more than
        60 s forgets how long it was wet. Returns (grams soaked, adj)."""
        sb = substrate(sub)
        if sb["porosity"] <= 0.0:
            return 0.0, []
        st = self.stack(key)
        liquid = None
        for s in st:
            if self._liquid_mass(self._comp(s)) > 0.0:
                liquid = s
                break
        if liquid is None:
            return 0.0, []
        comp = self._comp(liquid)
        lm = self._liquid_mass(comp)
        # t_w restarts only when the face has been dry longer than the wetting memory
        now = self.now if t_now is None else float(t_now)
        tw = float(self.t_wet[liquid])
        if now - float(self.t_last[liquid]) > dt + WET_MEMORY:
            tw = 0.0
        self.t_last[liquid] = now
        S = sb["S_along"] * (1.0 - s_host)
        dI = LL.soak_increment(S, tw, dt)                 # g/m2
        self.t_wet[liquid] = tw + dt
        want = min(dI * self.area * float(self.cover[liquid]), lm)
        ins = [s for s in self._chain(key) if self.inside[s]]
        v_in = sum(LL.comp_volumes(self._comp(s))[0] for s in ins)          # cm3 inside
        thick = sb["thickness"] if sb["thickness"] is not None else 5e-3
        cap_cm3 = sb["porosity"] * thick * (1.0 - s_host) * self.area * 1e6
        v_per_g = LL.comp_volumes(comp)[0] / lm                              # solutes add volume
        room = max(0.0, cap_cm3 - v_in) / v_per_g
        to_here = min(want, room)
        over = want - to_here if (sb["thickness"] is not None and back_key is not None) else 0.0
        if to_here + over <= 0.0:
            return 0.0, []
        d_pore = sb["d_pore_um"] * 1e-6
        Ts = AMBIENT
        adj = []

        def share(amount):
            f = amount / lm
            part = {}
            for ck, g in comp:
                fm = (ck >> 8) & 3
                if fm == DISSOLVED:
                    part[ck] = g * f
                elif fm == GRAIN and d_pore > 0.0 and \
                        LL.sieve_passes(LL.grain_diameter(ck >> 10, ck & 255), d_pore):
                    part[ck] = g * f
            return part
        if to_here > 0.0:
            adj += self._transfer(liquid, key, key, share(to_here), Ts, Ts, 1.0, sub, sub,
                                  "soak", to_inside=True)
        if over > 0.0 and self.key[liquid] == key:
            comp = self._comp(liquid)
            lm = self._liquid_mass(comp)
            if lm > 0.0:
                adj += self._transfer(liquid, key, back_key, share(min(over, lm)), Ts, Ts,
                                      1.0, sub, sub, "soak", to_inside=True)
        for k in (key, back_key):
            if k is None:
                continue
            for s in self._chain(k):
                if self.inside[s]:
                    vl, _vc, _vg, _gm = LL.comp_volumes(self._comp(s))
                    self.depth[s] = LL.inside_depth(vl * 1e-6, sb["porosity"], self.area)
        return to_here + over, _merge_adj(adj)

    def imbibe(self, key, dt, T_C=AMBIENT, sub=None):
        """A liquid layer on a GRAIN layer wicks into it as a porous medium (4.6b)
        and makes a paste (wet mud from dust and water). cos(theta) <= 0 (soot)
        wicks nothing. Returns (grams imbibed, adj)."""
        st = self.stack(key)
        for i in range(len(st) - 1):
            lq, gr = st[i], st[i + 1]
            lc, gc = self._comp(lq), self._comp(gr)
            lm = self._liquid_mass(lc)
            vg = sum(g / ROWS[c >> 10]["rho_c"] for c, g in gc if (c >> 8) & 3 == GRAIN)
            if lm <= 0.0 or vg <= 0.0:
                continue
            frow = self._fluid_row(lc)
            theta = sum(g * ROWS[c >> 10]["theta_wet"] for c, g in gc
                        if (c >> 8) & 3 == GRAIN) / sum(g for c, g in gc if (c >> 8) & 3 == GRAIN)
            dg = sum(g * LL.grain_diameter(c >> 10, c & 255) for c, g in gc
                     if (c >> 8) & 3 == GRAIN) / sum(g for c, g in gc if (c >> 8) & 3 == GRAIN)
            eta = LL.liquid_viscosity(frow, T_C)
            S = LL.imbibition_S(1000.0 * frow["rho_l"], float(self.por[gr]), dg,
                                frow["sigma"], theta, eta)
            if S <= 0.0:
                continue
            tw = float(self.t_wet[lq])
            dI = LL.soak_increment(S, tw, dt)
            self.t_wet[lq] = tw + dt
            vl, _vc, vg2, _gm = LL.comp_volumes(gc)
            _pd, pw = self._grain_params(gc)
            room = max(0.0, LL.void_capacity(vg2, min(float(self.por[gr]), pw)) - vl)
            want = min(dI * self.area * float(self.cover[lq]), lm, room * frow["rho_l"])
            if want <= 0.0:
                continue
            f = want / lm
            part = {c: g * f for c, g in lc if (c >> 8) & 3 == DISSOLVED}
            adj = self._transfer_layer_to_layer(lq, gr, key, part, T_C, sub)
            return want, adj
        return 0.0, []

    def _transfer_layer_to_layer(self, s_from, s_to, key, part, T_C, sub):
        """Liquid moves between two layers of ONE face: no host change."""
        ents = sorted((c, g) for c, g in part.items() if g > 0.0)
        old = dict(self._comp(s_from))
        for c, g in ents:
            old[c] = max(0.0, old[c] - g)
        e1 = self._write_comp(s_from, old)
        new = self._comp(s_to) + ents
        e2 = self._write_comp(s_to, new)
        self._derive(s_to, key, sub, shrink=False)
        if self.m[s_from] <= MIN_LAYER_G or not self._comp(s_from):
            self._unlink(key, s_from)
        else:
            self._derive(s_from, key, sub, shrink=True)
        self.gen += 1
        return [(key, e1[0] + e2[0], e1[1] + e2[1])]

    # ---- freeze, melt, boil: the joint latent pin (4.8) -----------------------------------
    def latent(self, keys, Csur, Ts, dt, P=101325.0, q_cap=None, boil=True, area=None):
        """Freeze, melt, dissolve and boil all the layers on a cell's faces (`keys`)
        TOGETHER against its ONE skin node (Csur J/K at Ts C). Solves
        Csur (T' - Ts) + sum dH(T') = 0 by bisection (lever rule with salt, melting
        range for waxes, Stefan cap for thick films, boiling elevation).
        Returns (T_new, adj, vapour_g)."""
        A = self.area if area is None else area
        faces = []
        slots = []
        for key in keys:
            for s in self.stack(key):
                f = self._phase_face(s, Ts, dt, P, q_cap, boil, A)
                if f is not None:
                    faces.append(f)
                    slots.append((key, s))
        if not faces:
            return Ts, [], 0.0
        T_new, dms, _pl = LL.solve_pin(Csur, Ts, faces)
        adj = []
        vap = 0.0
        for (key, s), f, dm in zip(slots, faces, dms):
            deltas = {}
            x_vap = 0.0
            for ck, d in zip(f.keys, dm):
                if ck == f.VAPOUR:
                    x_vap = float(d)
                    continue
                if d != 0.0:
                    deltas[(ck >> 10, (ck >> 8) & 3)] = deltas.get((ck >> 10, (ck >> 8) & 3), 0.0) + float(d)
            dC, dst, fE, fC = self._apply_sf(s, key, deltas)
            dE = -dst + fE
            dC += fC
            if x_vap > 0.0:
                r = ROWS[WATER]
                h_v = r["cp_l"] * (T_new - AMBIENT) + LL.lv_water(T_new)
                dE -= x_vap * h_v
                self._book(self.led_out, WATER, "vapour", x_vap)
                self.energy_in["vapour"] = self.energy_in.get("vapour", 0.0) - x_vap * h_v
                vap += x_vap
            if self.key[s] >= 0 and not self._comp(s):
                self._unlink(key, s)
            adj.append((key, dE, dC))
            self._touch(host_of(key))
        self.gen += 1
        return T_new, _merge_adj(adj), vap

    def _phase_face(self, s, Ts, dt, P, q_cap, boil, A):
        comp = self._comp(s)
        wl = sum(g for c, g in comp if c >> 10 == WATER and (c >> 8) & 3 == DISSOLVED)
        wcs = [(c, g) for c, g in comp if c >> 10 == WATER and (c >> 8) & 3 == CRYSTAL]
        wc = sum(g for _c, g in wcs)
        if wl + wc > 0.0:
            ckl = pack_key(WATER, DISSOLVED, 0)
            ckc = max(wcs, key=lambda t: t[1])[0] if wcs else pack_key(WATER, CRYSTAL, SZ_CRYSTAL)
            sol = []
            for sp in sorted({c >> 10 for c, _g in comp}):
                r = ROWS[sp]
                if r["sol0"] is None:
                    continue
                gd = sum(g for c, g in comp if c >> 10 == sp and (c >> 8) & 3 == DISSOLVED)
                gcs = [(c, g) for c, g in comp if c >> 10 == sp and (c >> 8) & 3 == CRYSTAL]
                gc = sum(g for _c, g in gcs)
                ckd = pack_key(sp, DISSOLVED, 0)
                ckx = max(gcs, key=lambda t: t[1])[0] if gcs else pack_key(sp, CRYSTAL, SZ_CRYSTAL)
                sol.append((ckd, ckx, gd, gc, sp))
            cap = None
            h = self.thickness(s)
            if h >= STEFAN_MIN:
                rw = ROWS[WATER]
                solutes = {x[4]: x[2] + x[3] for x in sol}
                t_liq = LL.freeze_point(wl + wc, solutes)
                h_cry = wc / (rw["rho_c"] * 1e6 * A * max(float(self.cover[s]), 1e-6))
                cap = LL.stefan_cap_grams(A * float(self.cover[s]), rw["k_c"], t_liq, Ts,
                                          LL.lf_kirchhoff(rw, min(Ts, 0.0)), h_cry, dt)
            qg = None
            if q_cap is not None:
                qg = q_cap * A * float(self.cover[s]) * dt / LL.lv_water(100.0)
            return LL.WaterFace((ckl, ckc, wl, wc), sol, Ts, P, cap, qg, boil)
        for sp in sorted({c >> 10 for c, _g in comp}):
            r = ROWS[sp]
            if r["Tsol"] is None or r["Lf0"] <= 0.0 or r["liq_class"] is None:
                continue
            gl = sum(g for c, g in comp if c >> 10 == sp and (c >> 8) & 3 == DISSOLVED)
            gcs = [(c, g) for c, g in comp if c >> 10 == sp and (c >> 8) & 3 == CRYSTAL]
            gc = sum(g for _c, g in gcs)
            if gl + gc <= 0.0:
                continue
            ckc = max(gcs, key=lambda t: t[1])[0] if gcs else pack_key(sp, CRYSTAL, SZ_CRYSTAL)
            return LL.MeltFace(sp, pack_key(sp, DISSOLVED, 0), ckc, gl, gc)
        return None

    # ---- contact, wipe, wash, wear (4.10) ------------------------------------------------
    def contact(self, S, Ds, c_c, p, s_slide, L_p, sub_S=None, sub_D=None, Ts_S=AMBIENT,
                Ts_D=AMBIENT, weights=None, pattern=None):
        """A contact from the top layer of face S to faces Ds. Returns (moved grams,
        adj). Whatever leaves S arrives at the Ds, split by `weights` (the path
        length in each); mass is exact. Liquid-rich layers (phi_l >= 0.5) split by the
        work of adhesion, dry ones by pressure and adhesion ratio; both slide to the
        whole contacted share with distance (f(s) = f_st + (c_c - f_st)(1 - exp(-s/L_s))).
        Bad input is refused before anything is touched: keys that make_key could not
        have made (Ds is read once, so a generator works), no destination, a contact
        share c_c outside 0..1, a negative or non-finite pressure, slide or length, a
        non-finite temperature, weights that are not one non-negative finite number
        per destination with a positive sum, a substrate argument that substrate()
        could not read: ValueError (KeyError for a substrate name or id with no
        row). The source face may be one of the destinations: the matter that goes
        there is booked and the energy closes (the 10000-operation conservation
        test draws such a contact), so it is not refused."""
        S = _check_face_key(S)
        Ds = _check_keys("Ds", Ds)
        _check_sub("sub_S", sub_S)
        _check_sub("sub_D", sub_D)
        _check_number("c_c", c_c, 0.0, 1.0, none_ok=False)
        _check_number("p", p, 0.0, none_ok=False)
        _check_number("s_slide", s_slide, 0.0, none_ok=False)
        _check_number("L_p", L_p, 0.0, none_ok=False)
        _check_number("Ts_S", Ts_S, none_ok=False)
        _check_number("Ts_D", Ts_D, none_ok=False)
        if weights is not None:
            try:
                weights = list(weights)                  # read once (a generator)
                if any(isinstance(x, (bool, str, bytes)) for x in weights):
                    raise TypeError
                weights = [float(x) for x in weights]
            except (TypeError, ValueError):
                raise ValueError("weights must be a sequence of numbers") from None
            if len(weights) != len(Ds):
                raise ValueError(f"weights has {len(weights)} items for {len(Ds)} destinations")
            for x in weights:
                _check_number("weight", x, 0.0, none_ok=False)
            if not sum(weights) > 0.0 or not math.isfinite(sum(weights)):
                raise ValueError("weights must have a positive finite sum")
        st = self.stack(S)
        if not st:
            return 0.0, []
        top = st[0]
        comp = self._comp(top)
        tot = sum(g for _c, g in comp)
        if tot <= 0.0:
            return 0.0, []
        phi_l = self._liquid_mass(comp) / tot
        sbS, sbD = substrate(sub_S), substrate(sub_D)
        w = np.ones(len(Ds)) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        if phi_l >= 0.5:
            fr = self._fluid_row(comp)
            cls = fr["liq_class"] if fr else "water"
            f_st = c_c * LL.liquid_split(sbS["theta"].get(cls, 60.0), sbD["theta"].get(cls, 60.0))
        else:
            a_S = float(self.adh[top])
            dst_top = self.stack(Ds[0])
            a_D = float(self.coh[dst_top[0]]) if dst_top else sbD["a_pull"]["grain"]
            f_st = LL.dry_transfer(c_c, a_D, a_S, p)
        f = min(1.0, LL.slide_fraction(f_st, c_c, s_slide, L_p))
        adj = []
        moved = 0.0
        cover_src = float(self.cover[top])
        for D, wi in zip(Ds, w):
            part = {c: g * f * wi for c, g in comp}
            pat = None
            if pattern is not None:
                pat = len(self.patterns)
                self.patterns.append(pattern)
            adj += self._transfer_keep(top, S, D, part, Ts_S, Ts_D, c_c * wi if len(Ds) > 1 else c_c,
                                       sub_D, sub_S, pat)
            moved += sum(part.values())
            if self.key[top] < 0:
                break
        return moved, _merge_adj(adj)

    def _transfer_keep(self, s, src_key, dst_key, part, Ts_src, Ts_dst, cover, sub_dst, sub_src, pat):
        """_transfer, but a vanished source layer is ok (several destinations)."""
        if self.key[s] < 0:
            return []
        return self._transfer(s, src_key, dst_key, part, Ts_src, Ts_dst, cover, sub_dst,
                              sub_src, "contact", pat)

    def wipe(self, key, tau, wiper_key, Ts=AMBIENT, Ts_w=AMBIENT, sub=None, sub_w=None):
        """Wipe or scrub with shear tau (Pa) from the top layer down (4.10d). Per
        layer the mobile part lifts by wipe_fraction(tau, adh, coh) (a brittle film
        with adh_s <= coh peels whole); BOUND mass lifts only at tau >= 0.7 a_bnd. What
        lifts goes onto the wiper's stack. Stops under the first layer not wiped
        clean. Returns (grams lifted, adj)."""
        adj = []
        lifted = 0.0
        for s in self.stack(key):
            comp = self._comp(s)
            mob = [(c, g) for c, g in comp if (c >> 8) & 3 != BOUND]
            bnd = [(c, g) for c, g in comp if (c >> 8) & 3 == BOUND]
            tot = sum(g for _c, g in comp)
            phi_l = self._liquid_mass(comp) / tot if tot else 0.0
            part = {}
            clean = True
            if mob:
                rows = [(g, ROWS[c >> 10], (c >> 8) & 3) for c, g in mob]
                a = LL.mix_strength(rows, "a", phi_l)
                co = LL.mix_strength(rows, "c", phi_l)
                _p, ty, _r = self._mud(mob)
                co = max(co, ty)
                r = LL.wipe_fraction(tau, a, co)
                if r > 0.0:
                    part.update({c: g * r for c, g in mob})
                clean = r >= 1.0
            if bnd:
                rows = [(g, ROWS[c >> 10], BOUND) for c, g in bnd]
                ab = LL.mix_strength(rows, "a", 0.0)
                if tau >= SHEAR_RATIO * ab:
                    part.update({c: g for c, g in bnd})
                else:
                    clean = False
            if part:
                lifted += sum(part.values())
                adj += self._transfer(s, key, wiper_key, part, Ts, Ts_w, None, sub_w, sub, "wipe")
            if not clean:
                break
        return lifted, _merge_adj(adj)

    def wash(self, key, tau_w, dst_key=None, Ts=AMBIENT, Ts_d=AMBIENT, sub=None, sub_d=None):
        """Flowing water takes DISSOLVED matter first (it dissolves into the water),
        and GRAIN when the wall shear tau_w reaches the layer's cohesion; BOUND does
        nothing (4.10e). It goes to dst_key (the water layer downstream) or leaves
        the pool (tag 'wash'). Returns (grams carried, adj)."""
        adj = []
        carried = 0.0
        for s in self.stack(key):
            comp = self._comp(s)
            part = {}
            for c, g in comp:
                f = (c >> 8) & 3
                if f == DISSOLVED and "solvent" not in ROWS[c >> 10]["tags"]:
                    part[c] = g
                elif f == GRAIN and tau_w >= float(self.coh[s]):
                    part[c] = g
            if not part:
                continue
            carried += sum(part.values())
            if dst_key is None:
                adj += self._take_out(key, s, part, Ts, "wash", sub)
            else:
                adj += self._transfer(s, key, dst_key, part, Ts, Ts_d, None, sub_d, sub, "wash")
        return carried, _merge_adj(adj)

    def wear(self, key, K, p, s_slide, H, partner=None, Ts=AMBIENT, Ts_p=AMBIENT, sub=None):
        """Archard wear of the top layer: removed thickness dh = K p s/H; the mass goes
        to the partner face or leaves (tag 'wear'). Returns (grams, adj)."""
        st = self.stack(key)
        if not st:
            return 0.0, []
        s = st[0]
        h = self.thickness(s)
        dh = LL.archard_dh(K, p, s_slide, H)
        fr = min(1.0, dh / h) if h > 0.0 else 0.0
        part = self._layer_frac(s, fr)
        g = sum(part.values())
        if partner is None:
            return g, self._take_out(key, s, part, Ts, "wear", sub)
        return g, self._transfer(s, key, partner, part, Ts, Ts_p, None, None, sub, "wear")

    # ---- deposition from air and condensation (4.2, 4.3) ----------------------------------
    def deposit_soot(self, key, smoke_units, Ts, T_gas=None, sub=None, sz=1):
        """Soot settling: `smoke_units` is the sim's smoke removed from the air; the
        face gets SOOT_TRUE of each unit as GRAIN soot (the rest the caller books in
        gone['smoke']). Returns (slot, adj)."""
        return self.add(key, {(SOOT, GRAIN, sz): smoke_units * SOOT_TRUE},
                        T_in=T_gas if T_gas is not None else Ts, Ts=Ts, source="smoke",
                        sub=sub)

    def deposit_soot_batch(self, keys, grams, sz=1):
        """MANY faces at once: `grams` of soot (already SOOT_TRUE of the smoke units)
        on each of `keys`, which must be unique. The arithmetic of `deposit_soot`
        with T_in = AMBIENT, done as array steps for every face whose top layer is
        already pure soot (the merge case, which is nearly all of them after the
        first pass); a face with no layer, or a different top, goes through `add`.
        The pure-soot merge is exact: mass adds, cover by random overlap (the same
        exponential), porosity and strengths stay (one species, one form). Returns
        the grams put down. A lockstep test holds it equal to `deposit_soot`."""
        keys = np.asarray(keys, np.int64)
        grams = np.asarray(grams, np.float64)
        pos = grams > 0.0                      # a ceiling in cool air takes nothing
        if not pos.all():
            keys, grams = keys[pos], grams[pos]
        if keys.size == 0:
            return 0.0
        ck = pack_key(SOOT, GRAIN, sz)
        top = self.lookup(keys)
        has = top >= 0
        if self.hi == 0:                       # an empty pool has no slot 0 to read
            pure = has
        else:
            s_ = np.where(has, top, 0)
            pure = (has & (self.ckey[s_, 0] == ck) & (self.cm[s_, 1:].sum(axis=1) == 0.0)
                    & (self.inside[s_] == 0) & (self.pat[s_] < 0))
        ok = np.nonzero(pure)[0]
        if ok.size:
            sl = top[ok]
            g = grams[ok]
            kappa = ROWS[SOOT]["kappa"] or KAPPA_MIN
            c_new = 1.0 - np.exp(-kappa * g / self.area)
            self.cm[sl, 0] += g
            self.m[sl] = self.cm[sl].sum(axis=1)
            c_old = self.cover[sl].astype(np.float64)
            self.cover[sl] = (1.0 - (1.0 - c_old) * (1.0 - c_new)).astype(self.cover.dtype)
            self.t_last[sl] = self.now
            self._touch(*(keys[ok] >> 3).tolist())
        for i in np.nonzero(~pure)[0]:
            self.add(int(keys[i]), {(SOOT, GRAIN, sz): float(grams[i])}, T_in=AMBIENT,
                     Ts=AMBIENT, source="smoke")
        if ok.size:
            self._book(self.led_in, SOOT, "smoke", float(grams[ok].sum()))
            self.energy_in["smoke"] = self.energy_in.get("smoke", 0.0)
            self.gen += 1
        return float(grams.sum())

    def condense(self, key, grams, Ts, freeze_c=0.0, sub=None):
        """Water vapour condenses on a face at Ts: DISSOLVED above the film's freezing
        point, CRYSTAL (frost) below, the latent heat (Lv, or Lv + Lf) booked into
        the skin through the adj. Returns (slot, adj). A bad key, a negative or
        non-finite mass, or a non-finite temperature or freezing point is ValueError
        (nothing is touched)."""
        key = _check_face_key(key)
        _check_number("grams", grams, 0.0, none_ok=False)
        _check_number("Ts", Ts, none_ok=False)
        _check_number("freeze_c", freeze_c, none_ok=False)
        form = DISSOLVED if Ts > freeze_c else CRYSTAL
        return self.add(key, {(WATER, form, SZ_CRYSTAL if form == CRYSTAL else 0): grams},
                        T_in=Ts, Ts=Ts, source="vapour", sub=sub,
                        latent_in=LL.condense_latent(Ts))

    # ---- drain and drip (4.5) --------------------------------------------------------------
    def _flow_info(self, key, s, sub, film_m):
        """What a wall layer would flow with: thickness, hold, viscosity, density,
        yield stress, kind ('paste' or 'film'), or None if nothing here can run."""
        comp = self._comp(s)
        vl, vc, vg, _gm = LL.comp_volumes(comp)
        if vl <= 0.0:
            return None
        A = self.area
        c = max(float(self.cover[s]), 1e-6)
        por = float(self.por[s])
        phi, tau, rho = self._mud(comp)
        paste = vg > 0.0 and vl >= LL.void_capacity(vg, por) and tau > 0.0
        frow = self._fluid_row(comp)
        if frow is None:
            return None
        hold = self._hold(comp, key, sub, film_m)
        if paste:
            V = LL.bulk_volume(vl, vc, vg, por) * 1e-6
            return dict(comp=comp, h=V / (A * c), hold=hold, c=c, kind="paste", V=V,
                        eta=LL.liquid_viscosity(frow, AMBIENT) * LL.krieger_dougherty(phi),
                        rho=rho, tau=tau, vl=vl)
        free = vl - (LL.void_capacity(vg, por) if vg > 0.0 else 0.0)
        if free <= 0.0:
            return None
        return dict(comp=comp, h=free * 1e-6 / (A * c), hold=hold, c=c, kind="film",
                    V=free * 1e-6, eta=LL.liquid_viscosity(frow, AMBIENT),
                    rho=1000.0 * frow["rho_l"], tau=0.0, vl=vl)

    def _flow_part(self, s, info, h_face, dt):
        """({ck: g} the layer sheds, cfl) with the flux taken at the face thickness
        h_face (a second-order reconstruction), capped by what lies over the hold."""
        h = info["h"]
        if h <= info["hold"]:
            return None
        A = self.area
        c = info["c"]
        q = LL.film_flow(max(h_face, h * 1e-3), info["eta"], info["rho"], info["tau"])
        cfl = q * self.dx * dt / (h * A * c)
        dV = min(q * self.dx * dt, (h - info["hold"]) * A * c)
        if dV <= 0.0:
            return None
        comp = info["comp"]
        if info["kind"] == "paste":
            fr = min(1.0, dV / info["V"])
            return {ck: g * fr for ck, g in comp if (ck >> 8) & 3 != BOUND}, cfl
        f_liq = min(1.0, dV * 1e6 / info["vl"])
        tau_w = LL.wall_shear_film(h, info["rho"])
        out = {}
        for ck, g in comp:
            fm = (ck >> 8) & 3
            if fm == DISSOLVED:
                out[ck] = g * f_liq
            elif fm == GRAIN and tau_w >= float(self.coh[s]):
                out[ck] = g * f_liq
        return out, cfl

    def flow_pass(self, solid, dt, Ts=AMBIENT, sub=None, keys=None, film_m=None):
        """Drain and drip over wall, ceiling and up-face layers, with an injected solid
        mask (3-D bool). A wall film is moved in sub-steps so that no layer moves more
        than ~0.35 of its volume per sub-step (u dt/dx <= 0.35), which keeps the
        explicit scheme stable and on the Jeffreys curve at 5 cm cells. Moves are
        planned from the state at the start of a sub-step and applied after, so a film
        does not cascade several cells in one step. Returns dict(adj, drops, handover):
        drops are entries of World.drops in the real 9-field format; handover is a list
        of dicts (key, comp, ml, fluid) the host adds to the lattice cell above an up
        face (the ledger books 'handover')."""
        res = self._flow_once(solid, dt, Ts, sub, keys, film_m, True)
        n = min(32, max(1, int(math.ceil(res["cfl"] / 0.35))))
        if n <= 1:
            return res
        adj, handover = [], []
        drops = []
        for i in range(n):
            r = self._flow_once(solid, dt / n, Ts, sub, keys, film_m, i == n - 1,
                                dry=False)
            adj += r["adj"]
            handover += r["handover"]
            drops += r["drops"]
        return {"adj": _merge_adj(adj), "drops": drops, "handover": handover,
                "cfl": res["cfl"]}

    def _flow_once(self, solid, dt, Ts, sub, keys, film_m, drip, dry=True):
        """One planned-then-applied drain step; `dry` only measures the CFL."""
        shape = solid.shape
        sub_of = sub if callable(sub) else (lambda k: sub)
        Ts_of = Ts if callable(Ts) else (lambda k: Ts)
        ks = [int(k) for k in (self.keys() if keys is None else keys)]
        cfl_max = 0.0
        infos = []
        hmap = {}
        for key in sorted(ks):
            tag, idx, face = split_key(key)
            if tag != TAG_CELL or face >= 4:
                continue
            x, y, z = cell_xyz(idx, shape)
            if not solid[x, y, z]:
                continue
            for s in self.stack(key):
                info = self._flow_info(key, s, sub_of(key), film_m)
                if info:
                    kind, tgt = LL.drain_target(solid, x, y, z, face)
                    tkey = cell_key(tgt[0], tgt[1], tgt[2], tgt[3], shape)
                    infos.append((key, s, info, tgt, tkey, kind))
                    hmap[key] = max(hmap.get(key, 0.0), info["h"])
        feeder = {tk: k for k, _s, _i, _t, tk, kd in infos if kd == "wall"}
        plans = []
        for key, s, info, tgt, tkey, kind in infos:
            h = info["h"]
            h_up = hmap.get(feeder.get(key), 0.0)
            h_dn = hmap.get(tkey, 0.0) if kind == "wall" else 0.0
            h_face = h + 0.5 * _minmod(h - h_up, h_dn - h)
            pl = self._flow_part(s, info, h_face, dt)
            if pl:
                part, cfl = pl
                cfl_max = max(cfl_max, cfl)
                plans.append((key, s, part, tgt))
        if dry:
            n_est = int(math.ceil(cfl_max / 0.35))
            if n_est > 1:
                return {"adj": [], "drops": [], "handover": [], "cfl": cfl_max}
        adj = []
        touched_up = []
        for key, s, part, tgt in plans:
            if self.key[s] != key:
                continue
            tkey = cell_key(tgt[0], tgt[1], tgt[2], tgt[3], shape)
            adj += self._transfer(s, key, tkey, part, Ts_of(key), Ts_of(tkey), None,
                                  sub_of(tkey), sub_of(key), "drain")
            if tgt[3] == LL.UP_FACE:
                touched_up.append(tkey)
        handover = []
        for tkey in sorted(set(touched_up)):
            g = {}
            for st_, part in self.excess_up(tkey, sub_of(tkey), film_m):
                for c, v in part.items():
                    g[c] = g.get(c, 0.0) + v
                adj += self._take_out(tkey, st_, part, Ts_of(tkey), "handover", sub_of(tkey))
            if g:
                sp = {}
                for c, v in g.items():
                    sp[c >> 10] = sp.get(c >> 10, 0.0) + v
                ml = sum(v / ROWS[c >> 10]["rho_l"] for c, v in g.items()
                         if "solute" not in ROWS[c >> 10]["tags"])
                handover.append({"key": tkey, "comp": g, "ml": ml, "fluid": fluid_id_for(sp)})
        drops, dadj = self._drip(shape, dt, Ts_of, sub_of, film_m) if drip else ([], [])
        return {"adj": _merge_adj(adj + dadj), "drops": drops, "handover": handover,
                "cfl": cfl_max}

    def _drip(self, shape, dt, Ts_of, sub_of, film_m):
        """Ceiling layers above h_c drip: n Tate drops per tick merged into one parcel
        of at most 5 ml, as a real 9-field World.drops entry."""
        drops, adj = [], []
        for key in sorted(int(k) for k in self.keys()):
            tag, idx, face = split_key(key)
            if tag != TAG_CELL or face != 5:
                continue
            for s in self.stack(key):
                comp = self._comp(s)
                vl, vc, vg, _gm = LL.comp_volumes(comp)
                frow = self._fluid_row(comp)
                if vl <= 0.0 or frow is None:
                    continue
                phi, tau, rho = self._mud(comp)
                hold = self._hold(comp, key, sub_of(key), film_m)
                A = self.area
                c = max(float(self.cover[s]), 1e-6)
                h = self._bulk_cm3(s, comp) * 1e-6 / (A * c)
                if h <= hold:
                    continue
                V_ex = (h - hold) * A * c
                v_d = LL.tate_drop_volume(frow["sigma"], 1000.0 * frow["rho_l"])
                n, ml, _left = LL.pendant_drops(V_ex, v_d)
                if n <= 0:
                    continue
                f = min(1.0, ml / self._bulk_cm3(s, comp))
                part = {ck: g * f for ck, g in comp if (ck >> 8) & 3 == DISSOLVED}
                sp = {}
                for ck, g in part.items():
                    sp[ck >> 10] = sp.get(ck >> 10, 0.0) + g
                x, y, z = cell_xyz(idx, shape)
                drops.append(LL.make_drop(x + 0.5, y + 0.5, z, ml, fluid_id_for(sp)))
                adj += self._take_out(key, s, part, Ts_of(key), "drip", sub_of(key))
        return drops, adj


def _minmod(a, b):
    """The smaller of two slopes when they agree in sign, else 0 (a limiter)."""
    if a * b <= 0.0:
        return 0.0
    return a if abs(a) < abs(b) else b


def _cmp(a, op, b):
    return a >= b if op == ">=" else a > b if op == ">" else a < b if op == "<" else a <= b
