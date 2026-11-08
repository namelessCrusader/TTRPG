"""THE MIX TABLE (wave 13, T1): what a cell's matter is MADE OF.

A MIX is a short list of components. A component is a species in a FORM
(dissolved, crystal, grain, bound) and, for grains and crystals, a size class
(species.py owns the key: key = species << 10 | form << 8 | sz). A cell points at
its mixes by two integers (scomp, fcomp, wiring phase B); 0 means "pure: read mat
or fl the old way". Rows are immutable and shared: a bronze bar of 10,000 cells
costs one row.

THE TABLE. Rows are stored in CSR form: off[n+1], key[nnz] (sorted ascending in
a row), q[nnz] (integer weights, each > 0, summing to EXACTLY QSUM = 2^24 per
row; fraction = q * 2^-24, exact in float32 and float64). Per row: refs, dom,
matv, flv, kind, hash and the cached float32 columns rho, c, k, eps (computed by
mixrules.py). A hash table (open addressing, linear probing) finds a row by its
content. The 64-bit row hash IS the content digest: it depends on the sorted
(key, weight) pairs only, never on the row number.

ROW NUMBERS. 0..63 pure solids (row r = the pure solid of mat r; 0 is empty);
64..127 pure fluids (row 64 + f = the pure fluid f; 64 is empty); 128 up user
mixes. A pure row is found by content like any other, so interning a pure
composition gives back its low number. The rows of ACID and WEAK_ACID are the
STOCK composition at potency 1. A pad slot that copies an empty `like` is empty.

LUMPS. An operation works on lumps: (row ids, float64 grams). take, put, blend,
convert, split and blend_many return new lumps. Every one does its arithmetic in
float64 and quantises its result; the rounding (at most 6e-8 of the lump per
species) is booked in the "quant" ledger, so grams of every species plus quant
stay constant to 1e-12. A species below about 6e-8 of a lump cannot be held in a
mix (it lives in a record or a store, spec 4.3).

IDENTITY. blend(a, a) and "take then put the same grams" give the SAME row id.
Row ids are assigned in creation order (the order of the law's cells, first
occurrence), but two different batchings can give different integers: equality
tests compare content_digest, never ids.

THE COMPILED TWIN. blend2_compiled runs the blend of many lump pairs in one
numba loop. It cannot grow arrays, so when it runs out of room it returns the
index it stopped at and Python grows the table and resumes (never drops a cell).
It writes the row STRUCTURE (off, key, q, hash); the cached columns of rows it
made are filled by finish(), which every read of a column calls first. The result
is bit-identical to blend() (structure, hash, ids, masses). A sweep that needs the
density of a row it has just made must call finish() between chunks (note for
phase B).

FLOAT32 CELLS. The lattice stores float32. store() writes the new float32 mass
and books the exact float64 mass minus what was stored, per species of the new
row, in the "f32" ledger.

ENERGY. E is sensible heat above the 20 C reference (T = 20 + E / C). Matter
leaves a cell at the cell's temperature (heat_of); a convert() of m grams from
form A to form B changes the cell's heat capacity and absorbs dH(20) (convert_energy).

Imports: species.py (keys, pure rows, the mat and fluid slots), mixrules.py
(columns), kernels.py (njit). It imports no World.
"""
from collections import defaultdict

import numpy as np

from . import mixrules as MR
from . import species as SP
from .kernels import HAVE_NUMBA, njit
from .species_seed import (BOUND, CRYSTAL, DISSOLVED, FLUID_ROWS, GRAIN, MIX_BASE,
                           SOLID_ROWS)

__all__ = ["MixTable", "MixTableFull", "CollectRefused", "QBITS", "QSUM", "INV_QSUM",
           "MAX_ROWS", "SZ_NUCLEI", "heat_of", "convert_energy", "row_hash"]

QBITS = 24
QSUM = 1 << QBITS
INV_QSUM = 2.0 ** -QBITS
MAX_ROWS = 4_000_000            # a hard budget: more raises MixTableFull (spec 3.3)
SZ_NUCLEI = 16                  # size class of a new crystal (4 um) (u)
AUTO_MIN_ROWS = 4096            # auto-collect when rows > max(2 * live, this)

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_MUL2 = np.uint64(0x94D049BB133111EB)
_GOLD2 = np.uint64(0xD6E8FEB86659FD93)
_S30, _S27, _S31, _S32 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(32)


class MixTableFull(RuntimeError):
    """The table would pass MAX_ROWS."""


class CollectRefused(RuntimeError):
    """collect() will not run: an unregistered holder, matter in flight, or a sweep."""


# ---- the 64-bit row hash (numpy and numba must agree) -----------------------
def _mix64(x):
    x = x + _GOLD
    x = (x ^ (x >> _S30)) * _MUL1
    x = (x ^ (x >> _S27)) * _MUL2
    return x ^ (x >> _S31)


def row_hash(cell, key, q, n):
    """Content hash of each of n rows given flat sorted (cell, key, q) lines:
    the wrap-around sum of splitmix64(key << 32 | q) over the row's lines, then
    mixed with the line count. Empty rows all hash alike."""
    cell = np.asarray(cell, np.int64)
    if len(cell):
        e = (np.asarray(key, np.uint64) << _S32) | np.asarray(q, np.uint64)
        m = _mix64(e)
        cs = np.concatenate([np.zeros(1, np.uint64), np.cumsum(m, dtype=np.uint64)])
        cnt = np.bincount(cell, minlength=n)
        end = np.cumsum(cnt)
        s = cs[end] - cs[end - cnt]
    else:
        s = np.zeros(n, np.uint64)
        cnt = np.zeros(n, np.int64)
    return _mix64(s + cnt.astype(np.uint64) * _GOLD2)


@njit(cache=True)
def _k_mix64(x):
    x = x + _GOLD
    x = (x ^ (x >> _S30)) * _MUL1
    x = (x ^ (x >> _S27)) * _MUL2
    return x ^ (x >> _S31)


@njit(cache=True)
def _k_hash(tk, tq, cnt):
    s = np.uint64(0)
    for j in range(cnt):
        s += _k_mix64((np.uint64(tk[j]) << _S32) | np.uint64(tq[j]))
    return _k_mix64(s + np.uint64(cnt) * _GOLD2)


@njit(cache=True)
def _ht_insert_many(ht, hashes, rows):
    """Insert rows (in the order given) at linear-probe slots of their hashes."""
    mask = np.uint64(ht.shape[0] - 1)
    for i in range(rows.shape[0]):
        s = np.int64(hashes[i] & mask)
        while ht[s] >= 0:
            s = (s + 1) & np.int64(mask)
        ht[s] = rows[i]


@njit(cache=True)
def _blend_batch(off, key, q, hsh, ht, st, ia, ma, ib, mb, out_i, out_m, start,
                 tk, tg, tq, tr):
    """Blend lump pairs i = start.. . Returns the index it stopped at (n when it
    finished). st = [n_rows, nnz, n_ready]. Same arithmetic, in the same order, as
    MixTable.blend (see there)."""
    n = ia.shape[0]
    mask = np.int64(ht.shape[0] - 1)
    qs = float(QSUM)
    i = start
    while i < n:
        a = ia[i]
        b = ib[i]
        if a == b:
            out_i[i] = a
            out_m[i] = ma[i] + mb[i]
            i += 1
            continue
        M = ma[i] + mb[i]
        out_m[i] = M
        if not (M > 0.0):
            out_i[i] = 0
            i += 1
            continue
        pa, ea = off[a], off[a + 1]
        pb, eb = off[b], off[b + 1]
        cnt = 0
        while pa < ea or pb < eb:
            if pb >= eb or (pa < ea and key[pa] < key[pb]):
                k = key[pa]
                g = (float(q[pa]) * INV_QSUM) * ma[i]
                pa += 1
            elif pa >= ea or key[pb] < key[pa]:
                k = key[pb]
                g = (float(q[pb]) * INV_QSUM) * mb[i]
                pb += 1
            else:
                k = key[pa]
                g = (float(q[pa]) * INV_QSUM) * ma[i] + (float(q[pb]) * INV_QSUM) * mb[i]
                pa += 1
                pb += 1
            if g > 0.0:
                tk[cnt] = k
                tg[cnt] = g
                cnt += 1
        sfl = 0
        for j in range(cnt):
            x = (tg[j] / M) * qs
            f = np.floor(x)
            tq[j] = np.int64(f)
            tr[j] = x - f
            sfl += np.int64(f)
        deficit = QSUM - sfl
        if deficit < 0:
            deficit = 0
        if deficit > cnt:
            deficit = cnt
        for _d in range(deficit):
            best = 0
            bv = -1.0
            for j in range(cnt):
                if tr[j] > bv:
                    bv = tr[j]
                    best = j
            tq[best] += 1
            tr[best] = -2.0
        c2 = 0
        for j in range(cnt):
            if tq[j] > 0:
                tk[c2] = tk[j]
                tq[c2] = tq[j]
                c2 += 1
        if c2 == 0:
            out_i[i] = 0
            i += 1
            continue
        h = _k_hash(tk, tq, c2)
        s = np.int64(h & np.uint64(mask))
        found = -1
        while ht[s] >= 0:
            r = ht[s]
            if hsh[r] == h and (off[r + 1] - off[r]) == c2:
                same = True
                base = off[r]
                for j in range(c2):
                    if key[base + j] != tk[j] or q[base + j] != tq[j]:
                        same = False
                        break
                if same:
                    found = r
                    break
            s = (s + 1) & mask
        if found >= 0:
            out_i[i] = found
            i += 1
            continue
        nr = st[0]
        nz = st[1]
        if nr + 2 > off.shape[0] or nz + c2 > key.shape[0] or (nr + 1) * 2 > ht.shape[0]:
            return i
        for j in range(c2):
            key[nz + j] = tk[j]
            q[nz + j] = tq[j]
        off[nr + 1] = nz + c2
        hsh[nr] = h
        ht[s] = nr
        st[0] = nr + 1
        st[1] = nz + c2
        out_i[i] = nr
        i += 1
    return n


def _pow2_at_least(x):
    p = 1
    while p < x:
        p <<= 1
    return p


# ---- energy helpers (float64; spec 3.5 and 4.5) ----------------------------
def heat_of(E, C, taken_cap):
    """The share of a cell's sensible heat E (J, above 20 C) that leaves with the
    taken grams: E * (sum g_i c_i) / C. Matter leaves at the cell temperature."""
    return np.asarray(E, np.float64) * np.asarray(taken_cap, np.float64) / np.asarray(C, np.float64)


def convert_energy(C, E, m, c_from, c_to, dH20):
    """A convert() of m grams from a form with heat capacity c_from to one with
    c_to, absorbing dH20 J/g (at 20 C). Returns (C', E'):
        C' = C + m (c_to - c_from)       E' = E - m dH20
    so T' = 20 + E' / C' and dH(T) = dH20 + (c_to - c_from)(T - 20) (Kirchhoff).
    The reverse convert returns E exactly."""
    C, E, m = (np.asarray(v, np.float64) for v in (C, E, m))
    return C + m * (np.asarray(c_to, np.float64) - np.asarray(c_from, np.float64)), E - m * dH20


class MixTable:
    """The mix table. See the module docstring."""

    def __init__(self, max_rows=MAX_ROWS, slot="engine", capacity=1024):
        self.max_rows = int(max_rows)
        self.slot = slot
        self._ledgers = defaultdict(lambda: defaultdict(float))
        self._holders = set()
        self._sweep = 0
        self._derived = {}
        self._rev = MR.revision()
        self.collections = 0
        cap = max(int(capacity), MIX_BASE + 64)
        self._st = np.zeros(3, np.int64)                  # n_rows, nnz, n_ready
        self._alloc(cap, cap * 4)
        self._ht = np.full(_pow2_at_least(4 * cap), -1, np.int32)
        self._build_pure_rows()

    # ---- storage ---------------------------------------------------------
    def _alloc(self, rows, nnz):
        self._off = np.zeros(rows + 1, np.int32)
        self._key = np.zeros(nnz, np.uint32)
        self._q = np.zeros(nnz, np.int32)
        self._hash = np.zeros(rows, np.uint64)
        self._refs = np.zeros(rows, np.int32)
        self._dom = np.zeros(rows, np.uint16)
        self._matv = np.zeros(rows, np.uint8)
        self._flv = np.zeros(rows, np.uint8)
        self._kind = np.zeros(rows, np.uint8)
        self._nohash = np.zeros(rows, np.uint8)           # 1: a pure duplicate, not in the hash table
        self._cols = {k: np.zeros(rows, np.float32) for k in MR.CORE}

    @property
    def n_rows(self):
        return int(self._st[0])

    @property
    def nnz(self):
        return int(self._st[1])

    @property
    def off(self):
        return self._off[:self.n_rows + 1]

    @property
    def key(self):
        return self._key[:self.nnz]

    @property
    def q(self):
        return self._q[:self.nnz]

    @property
    def hash(self):
        return self._hash[:self.n_rows]

    @property
    def refs(self):
        return self._refs[:self.n_rows]

    @property
    def dom(self):
        return self._dom[:self.n_rows]

    @property
    def matv(self):
        return self._matv[:self.n_rows]

    @property
    def flv(self):
        return self._flv[:self.n_rows]

    @property
    def kind(self):
        return self._kind[:self.n_rows]

    def row_len(self, ids):
        ids = np.asarray(ids, np.int64)
        return (self._off[ids + 1] - self._off[ids]).astype(np.int64)

    def reserve(self, rows=0, nnz=None):
        """Make room for `rows` more rows (and nnz more lines, default 6 per row)
        and keep the hash table at most half full. Python calls this before a
        compiled sweep (spec 3.3). Raises MixTableFull past max_rows."""
        n = self.n_rows
        if n + rows > self.max_rows:
            raise MixTableFull(f"mix table would hold {n + rows} rows (MAX_ROWS {self.max_rows})")
        nnz = rows * 6 if nnz is None else nnz
        cap_r, cap_z = len(self._hash), len(self._key)
        need_r, need_z = n + rows + 2, self.nnz + nnz + 2
        if need_r > cap_r or need_z > cap_z:
            nr = max(need_r, int(cap_r * 1.5)) if need_r > cap_r else cap_r
            nz = max(need_z, int(cap_z * 1.5)) if need_z > cap_z else cap_z
            self._regrow(nr, nz)
        if (n + rows + 1) * 2 > len(self._ht):
            self._ht = np.full(_pow2_at_least(2 * (n + rows + 1) * 2), -1, np.int32)
            self._ht_rebuild()

    def _regrow(self, rows, nnz):
        n, z = self.n_rows, self.nnz
        old = (self._off, self._key, self._q, self._hash, self._refs, self._dom, self._matv,
               self._flv, self._kind, self._nohash, self._cols)
        self._alloc(rows, nnz)
        self._off[:n + 1] = old[0][:n + 1]
        self._key[:z] = old[1][:z]
        self._q[:z] = old[2][:z]
        for new, o in zip((self._hash, self._refs, self._dom, self._matv, self._flv,
                           self._kind, self._nohash), old[3:10]):
            new[:n] = o[:n]
        for k in self._cols:
            self._cols[k][:n] = old[10][k][:n]

    def _ht_rebuild(self):
        self._ht[:] = -1
        rows = np.flatnonzero(self._nohash[:self.n_rows] == 0).astype(np.int32)
        rows = rows[self.row_len(rows) > 0]
        _ht_insert_many(self._ht, self._hash[rows], rows)

    # ---- the pure rows 0..127 ---------------------------------------------
    def _build_pure_rows(self):
        cells, keys, fr = [], [], []
        for r in range(MIX_BASE):
            for (sp, fo, sz, f) in SP.pure_row(r):
                cells.append(r)
                keys.append(SP.pack_key(sp, fo, sz))
                fr.append(f)
        keys = np.array(keys, np.uint32)
        order = np.lexsort((keys, np.array(cells)))
        cell, key, grams = np.array(cells, np.int64)[order], keys[order], np.array(fr)[order]
        cell_q, key_q, q_q, _book = self._quantise(MIX_BASE, cell, key, grams, np.ones(MIX_BASE))
        h = row_hash(cell_q, key_q, q_q, MIX_BASE)
        cnt = np.bincount(cell_q, minlength=MIX_BASE)
        self.reserve(MIX_BASE + 8, len(key_q) + 8)
        n = MIX_BASE
        self._key[:len(key_q)] = key_q
        self._q[:len(key_q)] = q_q
        self._off[1:n + 1] = np.cumsum(cnt)
        self._hash[:n] = h
        self._st[0], self._st[1] = n, len(key_q)
        # a pure row whose content an earlier pure row already holds is not hashed
        seen = {}
        for r in range(n):
            if cnt[r] == 0:
                self._nohash[r] = 1
                continue
            sl = slice(int(self._off[r]), int(self._off[r + 1]))
            sig = (self._key[sl].tobytes(), self._q[sl].tobytes())
            if sig in seen:
                self._nohash[r] = 1
            else:
                seen[sig] = r
        self._ht_rebuild()
        self._finalise(0, n)
        # the views of the pure rows are the mat and fluid ids themselves
        for r in range(n):
            if cnt[r] == 0:
                continue
            if r < SOLID_ROWS[1]:
                self._matv[r], self._flv[r] = r, 0
            else:
                self._matv[r], self._flv[r] = 0, r - FLUID_ROWS[0]
        self._st[2] = n

    # ---- quantiser ----------------------------------------------------------
    @staticmethod
    def _quantise(n, cell, key, grams, M):
        """Quantise lumps given as flat lines (cell, key, grams) and each cell's
        total M. Drops lines with no grams, adds duplicates, then rounds f * 2^24
        down and gives the missing units to the largest remainders (ties to the
        lowest key). Returns sorted (cell, key, q) with q > 0 and the signed
        grams (exact minus realised) of every merged line, for the quant ledger:
        (species, diff)."""
        cell = np.asarray(cell, np.int64)
        key = np.asarray(key, np.uint32)
        grams = np.asarray(grams, np.float64)
        M = np.asarray(M, np.float64)
        keep = (grams > 0) & (M[cell] > 0)
        cell, key, grams = cell[keep], key[keep], grams[keep]
        if len(cell) == 0:
            e = np.zeros(0, np.int64)
            return e, np.zeros(0, np.uint32), np.zeros(0, np.int32), (e, np.zeros(0))
        ck = (cell << 32) | key.astype(np.int64)
        uk, inv = np.unique(ck, return_inverse=True)
        g = np.bincount(inv, weights=grams, minlength=len(uk))
        cu = uk >> 32
        ku = (uk & 0xFFFFFFFF).astype(np.uint32)
        x = (g / M[cu]) * float(QSUM)
        fl = np.floor(x)
        rem = x - fl
        cnt = np.bincount(cu, minlength=n)
        sfl = np.bincount(cu, weights=fl, minlength=n)
        if (sfl > QSUM).any():
            raise ValueError("fractions of a lump sum to more than 1")
        deficit = np.clip(QSUM - sfl, 0, cnt).astype(np.int64)
        m = len(uk)
        order = np.lexsort((np.arange(m), -rem, cu))
        start = np.cumsum(cnt) - cnt
        rank = np.empty(m, np.int64)
        rank[order] = np.arange(m) - start[cu[order]]
        qq = fl.astype(np.int64) + (rank < deficit[cu])
        realised = qq * INV_QSUM * M[cu]
        book = (ku >> 10).astype(np.int64), g - realised
        ok = qq > 0
        return cu[ok], ku[ok], qq[ok].astype(np.int32), book

    # ---- finding and adding rows ---------------------------------------------
    def _rows_equal(self, rows, bcell, bstart, bcnt, bkey, bq):
        """Does table row rows[i] hold the same lines as batch line-group bcell[i]?"""
        ln = self._off[rows + 1].astype(np.int64) - self._off[rows]
        same = ln == bcnt[bcell]
        idx = np.flatnonzero(same)
        if len(idx):
            L = ln[idx]
            tot = int(L.sum())
            grp = np.repeat(np.arange(len(idx)), L)
            within = np.arange(tot) - np.repeat(np.cumsum(L) - L, L)
            tpos = self._off[rows[idx]].astype(np.int64)[grp] + within
            bpos = bstart[bcell[idx]][grp] + within
            bad = (self._key[tpos] != bkey[bpos]) | (self._q[tpos] != bq[bpos])
            badn = np.bincount(grp, weights=bad.astype(float), minlength=len(idx))
            same[idx[badn > 0]] = False
        return same

    def _lookup(self, h, cells, bstart, bcnt, bkey, bq):
        """Row id of each batch line-group by content (-1 if absent). h: hash per
        group; cells: the group numbers."""
        m = len(h)
        res = np.full(m, -1, np.int64)
        mask = np.uint64(len(self._ht) - 1)
        slot = (h & mask).astype(np.int64)
        active = np.arange(m)
        while len(active):
            r = self._ht[slot[active]].astype(np.int64)
            occupied = r >= 0
            active, r = active[occupied], r[occupied]
            if not len(active):
                break
            ok = self._hash[r] == h[active]
            eq = np.zeros(len(active), bool)
            if ok.any():
                eq[ok] = self._rows_equal(r[ok], cells[active[ok]], bstart, bcnt, bkey, bq)
            res[active[eq]] = r[eq]
            rest = active[~eq]
            slot[rest] = (slot[rest] + 1) & int(mask)
            active = rest
        return res

    def _resolve(self, n, cell, key, q):
        """Row ids for n lumps whose quantised lines are (cell, key, q), sorted
        by cell. Existing rows are found by content; new content is appended in
        order of first occurrence (spec 3.3). Empty lumps get row 0."""
        ids = np.zeros(n, np.int32)
        if len(cell) == 0:
            return ids
        self.finish()
        cnt = np.bincount(cell, minlength=n)
        bstart = np.cumsum(cnt) - cnt
        h = row_hash(cell, key, q, n)
        live = np.flatnonzero(cnt > 0)
        hl = h[live]
        uh, first, inv = np.unique(hl, return_index=True, return_inverse=True)
        rep = live[first]                                      # one lump per distinct hash
        # a 64-bit collision inside the batch: verify every lump equals its representative
        grp_rep = rep[inv]
        if (live != grp_rep).any():
            moved = np.flatnonzero(live != grp_rep)
            same = self._lumps_equal(live[moved], grp_rep[moved], cnt, bstart, key, q)
            if not same.all():
                return self._resolve_slow(n, cell, key, q, cnt, bstart, h)
        found = self._lookup(h[rep], rep, bstart, cnt, key, q)
        out = found.copy()
        new = np.flatnonzero(found < 0)
        if len(new):
            new = new[np.argsort(rep[new], kind="stable")]     # first occurrence order
            self._append(rep[new], cnt, bstart, key, q, h, out_rows=out, out_index=new)
        ids[live] = out[inv]
        return ids

    def _lumps_equal(self, a, b, cnt, bstart, key, q):
        ok = cnt[a] == cnt[b]
        idx = np.flatnonzero(ok)
        if len(idx):
            L = cnt[a[idx]]
            tot = int(L.sum())
            grp = np.repeat(np.arange(len(idx)), L)
            within = np.arange(tot) - np.repeat(np.cumsum(L) - L, L)
            pa, pb = bstart[a[idx]][grp] + within, bstart[b[idx]][grp] + within
            bad = (key[pa] != key[pb]) | (q[pa] != q[pb])
            badn = np.bincount(grp, weights=bad.astype(float), minlength=len(idx))
            ok[idx[badn > 0]] = False
        return ok

    def _resolve_slow(self, n, cell, key, q, cnt, bstart, h):
        """One lump at a time, in order: the fallback for a hash collision."""
        ids = np.zeros(n, np.int32)
        for c in np.flatnonzero(cnt > 0):
            sl = slice(int(bstart[c]), int(bstart[c] + cnt[c]))
            kk, qq = key[sl], q[sl]
            hh = h[c:c + 1]
            r = self._lookup(hh, np.zeros(1, np.int64), np.zeros(1, np.int64),
                             np.array([len(kk)]), kk, qq)[0]
            if r < 0:
                out = np.full(1, -1, np.int64)
                self._append(np.zeros(1, np.int64), np.array([len(kk)]), np.zeros(1, np.int64),
                             kk, qq, hh, out_rows=out, out_index=np.zeros(1, np.int64))
                r = out[0]
            ids[c] = r
        return ids

    def _append(self, groups, cnt, bstart, key, q, h, out_rows, out_index):
        """Append batch groups (in this order) as new rows and write their row
        numbers into out_rows[out_index]. Views and columns are made here."""
        k = len(groups)
        L = cnt[groups]
        z = int(L.sum())
        self.reserve(k, z)
        n0, z0 = self.n_rows, self.nnz
        grp = np.repeat(np.arange(k), L)
        within = np.arange(z) - np.repeat(np.cumsum(L) - L, L)
        src = bstart[groups][grp] + within
        self._key[z0:z0 + z] = key[src]
        self._q[z0:z0 + z] = q[src]
        self._off[n0 + 1:n0 + k + 1] = z0 + np.cumsum(L)
        self._hash[n0:n0 + k] = h[groups]
        rows = np.arange(n0, n0 + k, dtype=np.int32)
        self._st[0], self._st[1] = n0 + k, z0 + z
        # the views come first: a row the fluid rule refuses is rolled back
        try:
            self._finalise(n0, n0 + k, strict=True)
        except ValueError:
            self._st[0], self._st[1] = n0, z0
            raise
        _ht_insert_many(self._ht, self._hash[n0:n0 + k], rows)
        self._st[2] = n0 + k
        out_rows[out_index] = rows

    # ---- columns of new rows -----------------------------------------------
    def entries(self, ids):
        """The Entries table (flat aligned arrays) of rows `ids`."""
        ids = np.asarray(ids, np.int64).ravel()
        L = self.row_len(ids)
        tot = int(L.sum())
        cell = np.repeat(np.arange(len(ids)), L)
        within = np.arange(tot) - np.repeat(np.cumsum(L) - L, L)
        pos = self._off[ids].astype(np.int64)[cell] + within
        sp, fo, sz = SP.unpack_key(self._key[pos])
        return MR.Entries(len(ids), cell, within, self._key[pos], sp.astype(np.int64),
                          fo.astype(np.int64), sz.astype(np.int64), self._q[pos] * INV_QSUM)

    def _tables(self):
        """Per (species, form) lookup tables for the views."""
        t = getattr(self, "_viewtabs", None)
        if t is None:
            n = len(SP.SPECIES)
            matof = np.full((n, 4), -1, np.int64)
            fl = np.full(n, -1, np.int64)
            fll = np.full(n, -1, np.int64)
            for s in SP.SPECIES:
                if s is None:
                    continue
                st = s.solid_states
                b, g = st.get(BOUND, -1), st.get(GRAIN, -1)
                matof[s.id, BOUND] = b if b >= 0 else g
                matof[s.id, CRYSTAL] = b if b >= 0 else g
                matof[s.id, GRAIN] = g if g >= 0 else b
                fl[s.id] = -1 if s.liquid is None else s.liquid
                fll[s.id] = -1 if s.fluid_like is None else s.fluid_like
            t = self._viewtabs = (matof, fl, fll, MR.sp_arrays(self.slot)["skel"])
        return t

    def _mat_parts(self, ent):
        """(emat per entry, eligible per entry, lead entry per row): spec 3.4."""
        matof, _fl, _fll, skel = self._tables()
        emat = matof[ent.species, ent.form]
        if len(ent.cell) == 0:
            return emat, np.zeros(0, bool), np.full(ent.n, -1, np.int64)
        sk = skel[ent.species, ent.form] & (emat >= 0)
        has_sk = ent.sum(sk.astype(float)) > 0
        bound = (ent.form == BOUND) & (emat >= 0)
        elig = np.where(has_sk[ent.cell], sk, bound)
        lead = MR._first_per_cell(ent, np.where(elig, -ent.frac, 1.0))
        lead = np.where((lead >= 0) & elig[np.maximum(lead, 0)], lead, -1)
        return emat, elig, lead

    def _fl_parts(self, ent):
        """(efl per entry or -1, lead entry per row or -1, refused per row)."""
        _m, fl, fll, _s = self._tables()
        if len(ent.cell) == 0:
            z, no = np.full(ent.n, -1, np.int64), np.zeros(ent.n, bool)
            return np.zeros(0, np.int64), fll, z, z, no, no, no
        dis = ent.form == DISSOLVED
        efl = np.where(dis, fl[ent.species], -1)
        has_fl = ent.sum((efl >= 0).astype(float)) > 0
        lead_fl = MR._first_per_cell(ent, np.where(efl >= 0, -ent.frac, 1.0))
        lead_any = MR._first_per_cell(ent, np.where(dis, -ent.frac, 1.0))
        any_dis = ent.sum(dis.astype(float)) > 0
        la = np.maximum(lead_any, 0)
        like_ok = any_dis & (fll[ent.species[la]] >= 0) & (lead_any >= 0)
        refused = any_dis & ~has_fl & ~like_ok
        return efl, fll, lead_fl, lead_any, has_fl, any_dis, refused

    def _views_of(self, ent):
        """(dom, matv, flv or -1 if refused, kind) for the rows of an Entries."""
        n = ent.n
        emat, elig, lead = self._mat_parts(ent)
        matv = np.zeros(n, np.int64)
        okm = lead >= 0
        matv[okm] = emat[lead[okm]]
        efl, fll, lead_fl, lead_any, has_fl, any_dis, refused = self._fl_parts(ent)
        flv = np.zeros(n, np.int64)
        a = any_dis & has_fl
        flv[a] = efl[lead_fl[a]]
        b = any_dis & ~has_fl & ~refused
        flv[b] = fll[ent.species[lead_any[b]]]
        flv[refused] = -1
        # dominant species: the largest total fraction over forms (ties to the lowest id)
        dom = np.zeros(n, np.int64)
        if len(ent.cell):
            pair = ent.cell * 65536 + ent.species
            up, inv = np.unique(pair, return_inverse=True)
            tot = np.bincount(inv, weights=ent.frac, minlength=len(up))
            uc, us = up >> 16, up & 0xFFFF
            order = np.lexsort((us, -tot, uc))
            first = np.r_[True, uc[order][1:] != uc[order][:-1]]
            dom[uc[order][first]] = us[order][first]
        isb = ent.sum((ent.form == BOUND).astype(float)) > 0
        isd = ent.sum((ent.form == DISSOLVED).astype(float)) > 0
        kind = np.where(isb, 0, np.where(isd, 1, 0))
        return dom, matv, flv, kind

    def _finalise(self, r0, r1, strict=False):
        """Fill dom, matv, flv, kind and the cached columns for rows r0..r1-1."""
        ids = np.arange(r0, r1)
        ent = self.entries(ids)
        dom, matv, flv, kind = self._views_of(ent)
        if (flv < 0).any():
            bad = int(ids[np.flatnonzero(flv < 0)[0]])
            if strict:
                names = [SP.SPECIES[s].name for s in ent.species[ent.cell == bad - r0]]
                raise ValueError(f"no fluid view for a mix of {names}: the largest dissolved "
                                 "species has no fluid id and no fluid_like")
            flv = np.maximum(flv, 0)
        self._dom[r0:r1], self._matv[r0:r1] = dom, matv
        self._flv[r0:r1], self._kind[r0:r1] = flv, kind
        for k, v in MR.core_columns(ent, self.slot).items():
            self._cols[k][r0:r1] = v

    def finish(self, strict=False):
        """Fill the cached columns of rows the compiled twin appended."""
        r0, r1 = int(self._st[2]), self.n_rows
        if r1 > r0:
            self._finalise(r0, r1, strict=strict)
            self._st[2] = r1

    def _check_rev(self):
        if self._rev != MR.revision():
            self._rev = MR.revision()
            MR.refresh()
            self._viewtabs = None
            self._derived.clear()
            ids = np.arange(self.n_rows)
            ent = self.entries(ids)
            for k, v in MR.core_columns(ent, self.slot).items():
                self._cols[k][:self.n_rows] = v

    # ---- reading --------------------------------------------------------------
    def prop(self, name, ids):
        """A column for rows `ids`: 'rho', 'c', 'k', 'eps' are cached float32;
        any other registered derived column is computed on first use and cached.
        Row 0 and 64 (empty) give NaN for the cached columns."""
        self.finish()
        self._check_rev()
        ids = np.asarray(ids, np.int64)
        if name in self._cols:
            return self._cols[name][ids]
        if name in MR.DERIVED:
            cache = self._derived.get(name)
            if cache is None or len(cache[0]) < self.n_rows:
                vals = np.full(max(len(self._hash), self.n_rows), np.nan)
                valid = np.zeros(len(vals), bool)
                if cache is not None:
                    vals[:len(cache[0])], valid[:len(cache[1])] = cache
                cache = self._derived[name] = (vals, valid)
            vals, valid = cache
            need = np.unique(ids[~valid[ids]])
            if len(need):
                vals[need] = MR.evaluate(name, self.entries(need), self.slot)
                valid[need] = True
            return vals[ids]
        raise KeyError(f"no column {name!r}")

    def dominant(self, ids):
        """Species id with the largest mass fraction (over forms) of each row."""
        self.finish()
        return self._dom[np.asarray(ids, np.int64)].astype(np.int64)

    def mat_view(self, ids, prev_mat=None, margin=0.05):
        """The mat of each row (spec 3.4): the leader among skeleton entries
        (the largest BOUND entry if none is skeleton). With prev_mat, the old mat
        is kept while its entry is within `margin` of the leader's fraction."""
        self.finish()
        ids = np.asarray(ids, np.int64)
        out = self._matv[ids].astype(np.int64)
        if prev_mat is None:
            return out
        prev = np.broadcast_to(np.asarray(prev_mat, np.int64), ids.shape)
        ent = self.entries(ids)
        emat, elig, lead = self._mat_parts(ent)
        lf = np.where(lead >= 0, ent.frac[np.maximum(lead, 0)], 0.0) if len(ent.cell) else np.zeros(len(ids))
        cand = elig & (emat == prev[ent.cell])
        pf = np.zeros(len(ids))
        np.maximum.at(pf, ent.cell[cand], ent.frac[cand])
        keep = (pf > 0) & (pf >= lf - margin)
        return np.where(keep, prev, out)

    def fl_view(self, ids, prev_fl=None, margin=0.05):
        """The fl of each row by the solvent rule (spec 3.4). With prev_fl, the old
        fluid is kept while its dissolved mass is within `margin` of the leader."""
        self.finish()
        ids = np.asarray(ids, np.int64)
        out = self._flv[ids].astype(np.int64)
        if prev_fl is None:
            return out
        prev = np.broadcast_to(np.asarray(prev_fl, np.int64), ids.shape)
        ent = self.entries(ids)
        efl, _fll, lead_fl, _la, has_fl, _ad, _rf = self._fl_parts(ent)
        lf = np.where(has_fl, ent.frac[np.maximum(lead_fl, 0)], 0.0) if len(ent.cell) else np.zeros(len(ids))
        cand = (efl >= 0) & (efl == prev[ent.cell])
        pf = np.zeros(len(ids))
        np.maximum.at(pf, ent.cell[cand], ent.frac[cand])
        keep = (pf > 0) & (pf >= lf - margin)
        return np.where(keep, prev, out)

    def content_digest(self, ids):
        """uint64 per row: a hash of the sorted (key, weight) pairs, independent of
        row numbers (spec 6.5). Compare these, never ids."""
        return self._hash[np.asarray(ids, np.int64)]

    def totals(self, ids, mass, by_key=False):
        """Grams per species (or per component key) over lumps: {id: grams}."""
        ent = self.entries(ids)
        g = ent.frac * np.asarray(mass, np.float64)[ent.cell]
        k = ent.key.astype(np.int64) if by_key else ent.species
        u, inv = np.unique(k, return_inverse=True)
        s = np.bincount(inv, weights=g, minlength=len(u))
        return {int(a): float(b) for a, b in zip(u, s)}

    def cap64(self, ids, mass):
        """Heat capacity of lumps in J/K, summed over the components in float64."""
        ent = self.entries(ids)
        c = MR.comp_c(ent.species, ent.form, self.slot)
        return np.asarray(mass, np.float64) * ent.sum(ent.frac * c)

    def key_c(self, keys):
        """Specific heat (J/g/K) of each component key."""
        return MR.comp_c_keys(keys, self.slot)

    # ---- ledgers --------------------------------------------------------------
    def book(self, name, species, grams):
        """Add signed grams per species to the named ledger (quant, f32, recast,
        squeeze, gone ...)."""
        species = np.asarray(species, np.int64)
        grams = np.asarray(grams, np.float64)
        if len(species):
            u, inv = np.unique(species, return_inverse=True)
            s = np.bincount(inv, weights=grams, minlength=len(u))
            led = self._ledgers[name]
            for a, b in zip(u, s):
                led[int(a)] += float(b)

    def ledger(self, name):
        """{species: grams} of one ledger."""
        return dict(self._ledgers[name]) if name in self._ledgers else {}

    def ledger_total(self, name):
        return float(sum(self._ledgers[name].values())) if name in self._ledgers else 0.0

    # ---- making rows -----------------------------------------------------------
    def _make(self, n, cell, key, grams, M, book=True):
        """Quantise, book the rounding in `quant`, resolve to ids."""
        cq, kq, qq, (bsp, bdiff) = self._quantise(n, cell, key, grams, M)
        if book:
            self.book("quant", bsp, bdiff)
        return self._resolve(n, cq, kq, qq)

    def intern(self, keys, fracs, mass=None):
        """Rows for n mixes given as component keys [n, K] and fractions [n, K]
        (a zero fraction drops a slot; duplicates are added; each row is
        normalised to sum 1). With `mass` the rounding is booked in `quant`.
        Returns int32 ids; an empty mix gets row 0."""
        keys = np.atleast_2d(np.asarray(keys, np.uint32))
        fracs = np.atleast_2d(np.asarray(fracs, np.float64))
        n, K = fracs.shape
        s = fracs.sum(axis=1)
        cell = np.repeat(np.arange(n), K)
        grams = (fracs / np.where(s > 0, s, 1.0)[:, None]).ravel()
        M = np.ones(n) if mass is None else np.asarray(mass, np.float64)
        ok = s > 0
        return self._make(n, cell, keys.ravel(), np.where(ok[cell], grams, 0.0),
                          np.where(ok, M, 0.0), book=mass is not None)

    def parse(self, comps):
        """One row from [(species name or id, form name or int, size, fraction), ...]."""
        keys, fr = [], []
        for sp, fo, sz, f in comps:
            sid = SP.BY_NAME[sp] if isinstance(sp, str) else int(sp)
            fid = ["dissolved", "crystal", "grain", "bound"].index(fo) if isinstance(fo, str) else int(fo)
            keys.append(SP.pack_key(sid, fid, sz))
            fr.append(f)
        return int(self.intern(np.array([keys], np.uint32), np.array([fr]))[0])

    # ---- the verbs ----------------------------------------------------------------
    def _edit(self, ids, mass, cell_i, key, dg):
        """Apply signed gram changes dg of components `key` to lumps cell_i.
        A change that would take a component below 0 stops at 0. Returns
        (ids', mass', applied dg)."""
        ids = np.asarray(ids)
        out_i = ids.astype(np.int32).copy()
        out_m = np.asarray(mass, np.float64).copy()
        cell_i = np.asarray(cell_i, np.int64)
        key = np.asarray(key, np.uint32)
        dg = np.asarray(dg, np.float64)
        applied = np.zeros(len(dg))
        if len(cell_i) == 0:
            return out_i, out_m, applied
        U, uinv = np.unique(cell_i, return_inverse=True)
        ent = self.entries(out_i[U])
        old_g = ent.frac * out_m[U][ent.cell]
        # union of the lumps' own lines and the requested components
        allc = np.concatenate([ent.cell, uinv])
        allk = np.concatenate([ent.key, key])
        ck = (allc.astype(np.int64) << 32) | allk.astype(np.int64)
        uk, inv = np.unique(ck, return_inverse=True)
        nold = len(ent.cell)
        og = np.bincount(inv[:nold], weights=old_g, minlength=len(uk))
        dl = np.bincount(inv[nold:], weights=dg, minlength=len(uk))
        new = np.maximum(og + dl, 0.0)
        new = np.where((dl < 0) & (og + dl <= 0), 0.0, new)
        # applied change per request: share the clamped change in proportion
        delta = new - og
        rq = inv[nold:]
        tot_req = dl[rq]
        with np.errstate(divide="ignore", invalid="ignore"):
            applied = np.where(tot_req != 0, dg * delta[rq] / tot_req, 0.0)
        cu = uk >> 32
        ku = (uk & 0xFFFFFFFF).astype(np.uint32)
        M2 = np.bincount(cu, weights=new, minlength=len(U))
        ids2 = self._make(len(U), cu, ku, new, M2)
        out_i[U] = ids2
        out_m[U] = M2
        return out_i, out_m, applied

    def take(self, ids, mass, cell_i, key, grams, return_taken=False):
        """Remove `grams` of component `key` from lump `cell_i` (arrays, one entry
        per request; several for one lump add up). A request for more than the lump
        holds takes what there is. Returns (ids', mass') and, with return_taken,
        the grams actually taken per request."""
        i2, m2, ap = self._edit(ids, mass, cell_i, key, -np.asarray(grams, np.float64))
        return (i2, m2, -ap) if return_taken else (i2, m2)

    def put(self, ids, mass, cell_i, key, grams):
        """Add `grams` of component `key` to lump `cell_i`."""
        i2, m2, _ap = self._edit(ids, mass, cell_i, key, np.asarray(grams, np.float64))
        return i2, m2

    def convert(self, ids, mass, cell_i, species, from_form, to_form, grams, size=None,
                from_size=0, return_taken=False):
        """Change `grams` of `species` from from_form to to_form (to size class
        `size`: default SZ_NUCLEI for CRYSTAL, the species' grain size for GRAIN, 0
        otherwise). Take then put the same grams; the most that can change is what
        the lump holds of the source component. At most one request per
        (lump, species, from_form, to_form)."""
        cell_i = np.asarray(cell_i, np.int64)
        sp = np.broadcast_to(np.asarray(species, np.int64), cell_i.shape)
        ff = np.broadcast_to(np.asarray(from_form, np.int64), cell_i.shape)
        tf = np.broadcast_to(np.asarray(to_form, np.int64), cell_i.shape)
        fs = np.broadcast_to(np.asarray(from_size, np.int64), cell_i.shape)
        if size is None:
            gs = np.array([0 if s is None else s.grain_sz for s in SP.SPECIES])
            size_ = np.where(tf == CRYSTAL, SZ_NUCLEI,
                             np.where(tf == GRAIN, np.where(gs[sp] > 0, gs[sp], SZ_NUCLEI), 0))
        else:
            size_ = np.broadcast_to(np.asarray(size, np.int64), cell_i.shape)
        kf = SP.pack_key(sp, ff, fs)
        kt = SP.pack_key(sp, tf, size_)
        dup = np.unique(np.stack([cell_i, kf, kt], axis=1).astype(np.int64), axis=0)
        if len(dup) != len(cell_i):
            raise ValueError("convert takes one request per (lump, source component, target)")
        ids = np.asarray(ids)
        ent = self.entries(ids[cell_i])
        have = np.zeros(len(cell_i))
        m = (ent.key == kf[ent.cell])
        have[ent.cell[m]] = ent.frac[m] * np.asarray(mass, np.float64)[cell_i][ent.cell[m]]
        g = np.minimum(np.asarray(grams, np.float64), have)
        # one edit: -g from the source, +g to the target, so a lump changes in one step
        cc = np.concatenate([cell_i, cell_i])
        kk = np.concatenate([kf, kt]).astype(np.uint32)
        dd = np.concatenate([-g, g])
        i2, m2, _ap = self._edit(ids, mass, cc, kk, dd)
        return (i2, m2, g) if return_taken else (i2, m2)

    def blend(self, ids_a, m_a, ids_b, m_b):
        """The mix of two lumps (arrays, pairwise). If the two rows are the same
        the row is unchanged. The result mass is m_a + m_b."""
        ia = np.asarray(ids_a).astype(np.int32)
        ib = np.asarray(ids_b).astype(np.int32)
        ma, mb = np.asarray(m_a, np.float64), np.asarray(m_b, np.float64)
        n = len(ia)
        out_i = np.zeros(n, np.int32)
        out_m = ma + mb
        same = ia == ib
        out_i[same] = ia[same]
        w = np.flatnonzero(~same)
        if len(w):
            ea, eb = self.entries(ia[w]), self.entries(ib[w])
            ga = ea.frac * ma[w][ea.cell]
            gb = eb.frac * mb[w][eb.cell]
            cell = np.concatenate([ea.cell, eb.cell])
            M = out_m[w]
            out_i[w] = self._make(len(w), cell, np.concatenate([ea.key, eb.key]),
                                  np.concatenate([ga, gb]), np.where(M > 0, M, 0.0))
        return out_i, out_m

    def blend_many(self, ids, mass, group, n_groups=None):
        """Blend every lump of a group into one (a record's nodes). Returns
        (ids[n_groups], mass[n_groups])."""
        ids = np.asarray(ids)
        mass = np.asarray(mass, np.float64)
        group = np.asarray(group, np.int64)
        G = int(group.max()) + 1 if n_groups is None else int(n_groups)
        ent = self.entries(ids)
        g = ent.frac * mass[ent.cell]
        M = np.bincount(group, weights=mass, minlength=G)
        return self._make(G, group[ent.cell], ent.key, g, M), M

    def split(self, ids, m, m1):
        """Break a lump: both pieces keep the row. Returns (ids, m1, ids, m - m1)."""
        ids = np.asarray(ids).astype(np.int32)
        m = np.asarray(m, np.float64)
        m1 = np.asarray(m1, np.float64)
        return ids.copy(), m1, ids.copy(), m - m1

    # ---- the compiled twin -------------------------------------------------------
    def blend2_compiled(self, ids_a, m_a, ids_b, m_b, grow=None):
        """blend() in one numba loop with stop-and-resume (spec 3.3). `grow` is how
        many rows to add each time it runs out of room (default: half again).
        Rows it makes get their cached columns at the next finish()."""
        ia = np.ascontiguousarray(ids_a, np.int32)
        ib = np.ascontiguousarray(ids_b, np.int32)
        ma = np.ascontiguousarray(m_a, np.float64)
        mb = np.ascontiguousarray(m_b, np.float64)
        n = len(ia)
        out_i = np.zeros(n, np.int32)
        out_m = np.zeros(n)
        self.finish()
        before = self.entries(np.concatenate([ia, ib]))
        maxlen = int(max(self.row_len(ia).max(initial=0), self.row_len(ib).max(initial=0)))
        tk = np.zeros(2 * maxlen + 2, np.uint32)
        tg = np.zeros(2 * maxlen + 2)
        tq = np.zeros(2 * maxlen + 2, np.int64)
        tr = np.zeros(2 * maxlen + 2)
        start = 0
        while True:
            start = _blend_batch(self._off, self._key, self._q, self._hash, self._ht, self._st,
                                 ia, ma, ib, mb, out_i, out_m, start, tk, tg, tq, tr)
            if start >= n:
                break
            k = max(1, int(self.n_rows // 2)) if grow is None else int(grow)
            self.reserve(k, k * (2 * maxlen + 2))
        # quant: grams in minus grams out, per species
        after = self.entries(out_i)
        gin = np.concatenate([before.frac * np.concatenate([ma, mb])[before.cell]])
        gout = after.frac * out_m[after.cell]
        self.book("quant", before.species, gin)
        self.book("quant", after.species, -gout)
        return out_i, out_m

    # ---- the float32 lattice --------------------------------------------------------
    def store(self, arr, idx, ids, mass):
        """Write lumps to a float32 mass array and book the difference between the
        exact float64 mass and what float32 kept, per species, in `f32`. `idx` is
        anything arr[idx] accepts; `ids` must be the new rows of those cells."""
        mass = np.asarray(mass, np.float64)
        m32 = mass.astype(np.float32)
        arr[idx] = m32
        diff = mass - m32.astype(np.float64)
        ent = self.entries(ids)
        self.book("f32", ent.species, ent.frac * diff[ent.cell])
        return m32

    # ---- adapters for dense records ----------------------------------------------
    def to_dense(self, ids, order, mass=None, form=None):
        """Grams (or fractions, with no `mass`) per species in `order` (a list of
        species ids or names), summed over forms (or only `form`). A species in a
        row but not in `order` raises KeyError."""
        sid = [SP.BY_NAME[s] if isinstance(s, str) else int(s) for s in order]
        col = {s: i for i, s in enumerate(sid)}
        ent = self.entries(ids)
        sel = np.ones(len(ent.cell), bool) if form is None else ent.form == int(form)
        out = np.zeros((ent.n, len(sid)))
        miss = [int(s) for s in np.unique(ent.species[sel]) if int(s) not in col]
        if miss:
            raise KeyError(f"species {miss} are in the rows but not in `order`")
        cols = np.array([col[int(s)] for s in ent.species[sel]], np.int64)
        m = np.ones(ent.n) if mass is None else np.asarray(mass, np.float64)
        np.add.at(out, (ent.cell[sel], cols), ent.frac[sel] * m[ent.cell[sel]])
        return out

    def from_dense(self, grams, order, form=DISSOLVED, size=0):
        """Rows for dense grams [n, S] of the species in `order`, all in `form`
        (a scalar, or one per column). Quantised like intern; the rounding is booked
        in `quant`. Returns (ids, mass)."""
        grams = np.atleast_2d(np.asarray(grams, np.float64))
        n, S = grams.shape
        sid = np.array([SP.BY_NAME[s] if isinstance(s, str) else int(s) for s in order], np.int64)
        forms = np.broadcast_to(np.asarray(form, np.int64), (S,))
        sizes = np.broadcast_to(np.asarray(size, np.int64), (S,))
        keys = SP.pack_key(sid, forms, sizes)
        M = grams.sum(axis=1)
        cell = np.repeat(np.arange(n), S)
        ids = self._make(n, cell, np.tile(keys, n), grams.ravel(), M)
        return ids, M

    # ---- views on lattice arrays (wiring helpers) --------------------------------
    def set_solid(self, mat, scomp, idx, rows):
        """Write scomp and the matching mat view together (hysteresis against the
        old mat). Row 0 means "pure": scomp becomes 0 and mat is left alone."""
        rows = np.asarray(rows, np.int64)
        scomp[idx] = rows
        view = self.mat_view(rows, prev_mat=mat[idx])
        mat[idx] = np.where(rows == 0, mat[idx], view)

    def set_fluid(self, fl, fcomp, idx, rows):
        """Write fcomp and the matching fl view together."""
        rows = np.asarray(rows, np.int64)
        fcomp[idx] = rows
        view = self.fl_view(rows, prev_fl=fl[idx])
        fl[idx] = np.where(rows == 0, fl[idx], view)

    def set_mat(self, mat, scomp, smass, idx, new_mat):
        """The legacy writer of mat: write mat, zero scomp, and where the cell held a
        user mix book the old grams (negative) and the new pure grams (positive) to
        `recast`, so species totals stay exact up to recast."""
        idx_ = np.atleast_1d(idx) if not isinstance(idx, tuple) else idx
        old = np.asarray(scomp[idx_], np.int64)
        m = np.asarray(smass[idx_], np.float64)
        new_mat = np.broadcast_to(np.asarray(new_mat, np.int64), old.shape)
        mixed = old > 0
        if mixed.any():
            eo, en = self.entries(old[mixed]), self.entries(new_mat[mixed])
            self.book("recast", eo.species, -(eo.frac * m[mixed][eo.cell]))
            self.book("recast", en.species, en.frac * m[mixed][en.cell])
        mat[idx_] = new_mat
        scomp[idx_] = 0

    def check_views(self, mat, scomp, fl=None, fcomp=None, fvol=None, margin=0.05):
        """The view invariants (spec 3.4) on whole arrays. Returns a list of
        (invariant, flat cell indices) that break, empty if all hold:
        I2 scomp > 0 and mat is not the view (within hysteresis);
        I3 fcomp > 0 and fl is not the view;
        I4 fvol > 0 and fl == 0."""
        bad = []
        s = np.flatnonzero(np.asarray(scomp).ravel() > 0)
        if len(s):
            mv = np.asarray(mat).ravel()[s]
            ok = self.mat_view(np.asarray(scomp).ravel()[s], prev_mat=mv, margin=margin) == mv
            if not ok.all():
                bad.append(("I2", s[~ok]))
        if fl is not None and fcomp is not None:
            f = np.flatnonzero(np.asarray(fcomp).ravel() > 0)
            if len(f):
                fv = np.asarray(fl).ravel()[f]
                ok = self.fl_view(np.asarray(fcomp).ravel()[f], prev_fl=fv, margin=margin) == fv
                if not ok.all():
                    bad.append(("I3", f[~ok]))
        if fl is not None and fvol is not None:
            w = np.flatnonzero((np.asarray(fvol).ravel() > 0) & (np.asarray(fl).ravel() == 0))
            if len(w):
                bad.append(("I4", w))
        return bad

    # ---- reference counts and garbage collection -------------------------------------
    def register_holder(self, name):
        """Name an array that holds row ids. collect() refuses to run unless every
        registered holder is passed."""
        self._holders.add(str(name))

    def begin_sweep(self):
        self._sweep += 1

    def end_sweep(self):
        self._sweep = max(0, self._sweep - 1)

    def recount(self, holders):
        """Set refs[] from the arrays in `holders` (a dict name -> int array, or a
        list of arrays) and return it. An id past the table raises (a stale id)."""
        arrs = list(holders.values()) if isinstance(holders, dict) else list(holders)
        n = self.n_rows
        refs = np.zeros(n, np.int64)
        for a in arrs:
            a = np.asarray(a).ravel()
            a = a[a >= 0]
            if len(a) and int(a.max()) >= n:
                raise ValueError(f"a holder refers to row {int(a.max())} but the table has {n}")
            refs += np.bincount(a, minlength=n)
        self._refs[:n] = refs
        return refs

    def live_rows(self):
        """User rows with refs > 0 (after recount)."""
        return int((self.refs[MIX_BASE:] > 0).sum())

    def should_collect(self):
        return self.n_rows > max(2 * self.live_rows(), AUTO_MIN_ROWS)

    def collect(self, holders, in_flight=0, keep=None):
        """Free the user rows nothing refers to and remap every holder in place
        (new id = rank among the kept rows, so the order is kept and the result is
        deterministic). Refuses (CollectRefused) while a registered holder is not
        passed, matter is in flight (in_flight > 0), or a sweep is open. `keep` is
        extra row ids that must survive. Returns the old-to-new array (-1 freed)."""
        if not isinstance(holders, dict):
            raise TypeError("collect takes a dict name -> array so it can check the holders")
        miss = self._holders - set(holders)
        if miss:
            raise CollectRefused(f"holders not passed: {sorted(miss)}")
        if in_flight:
            raise CollectRefused("promoted matter is in flight")
        if self._sweep:
            raise CollectRefused("inside a sweep")
        self.finish()
        n = self.n_rows
        refs = self.recount(holders)
        alive = refs > 0
        alive[:MIX_BASE] = True
        if keep is not None:
            alive[np.asarray(keep, np.int64)] = True
        kept = np.flatnonzero(alive)
        lut = np.full(n, -1, np.int64)
        lut[kept] = np.arange(len(kept))
        L = self.row_len(kept)
        z = int(L.sum())
        grp = np.repeat(np.arange(len(kept)), L)
        within = np.arange(z) - np.repeat(np.cumsum(L) - L, L)
        src = self._off[kept].astype(np.int64)[grp] + within
        keyn, qn = self._key[src].copy(), self._q[src].copy()
        cols = {c: self._cols[c][kept].copy() for c in self._cols}
        sm = [a[kept].copy() for a in (self._hash, self._dom, self._matv, self._flv, self._kind,
                                       self._nohash)]
        k = len(kept)
        self._alloc(max(len(self._hash), k + 2), max(len(self._key), z + 2))
        self._key[:z], self._q[:z] = keyn, qn
        self._off[1:k + 1] = np.cumsum(L)
        for dst, v in zip((self._hash, self._dom, self._matv, self._flv, self._kind, self._nohash), sm):
            dst[:k] = v
        for c in cols:
            self._cols[c][:k] = cols[c]
        self._st[0], self._st[1], self._st[2] = k, z, k
        self._derived.clear()
        self._ht_rebuild()
        for a in holders.values():
            a = np.asarray(a)
            m = a >= MIX_BASE
            if m.any():
                a[m] = lut[a[m]]
        self.recount(holders)
        self.collections += 1
        return lut

    def maybe_collect(self, holders, in_flight=0, keep=None):
        """The automatic hook for a quiet slow-clock tick: recount, and collect when
        rows > max(2 * live rows, 4096). Returns the old-to-new array or None."""
        self.recount(holders)
        if not self.should_collect():
            return None
        return self.collect(holders, in_flight=in_flight, keep=keep)

    # ---- checks -------------------------------------------------------------------------
    def check_table(self):
        """Invariants of the table; a list of problems (empty if sound)."""
        bad = []
        n = self.n_rows
        off, key, q = self.off, self.key, self.q
        if off[0] != 0 or (np.diff(off) < 0).any():
            bad.append("off is not monotone from 0")
        cell = np.repeat(np.arange(n), np.diff(off).astype(np.int64))
        if len(key) != int(off[-1]):
            bad.append("nnz does not match off")
            return bad
        if (q <= 0).any():
            bad.append("a weight is not positive")
        same = cell[1:] == cell[:-1]
        if (key[1:][same] <= key[:-1][same]).any():
            bad.append("keys are not strictly ascending inside a row")
        sums = np.bincount(cell, weights=q.astype(np.float64), minlength=n)
        ln = np.diff(off)
        if (sums[ln > 0] != QSUM).any():
            bad.append("a non-empty row does not sum to 2^24")
        if (self.hash != row_hash(cell, key, q, n)).any():
            bad.append("a stored hash is wrong")
        ht = self._ht[self._ht >= 0]
        want = np.flatnonzero((self._nohash[:n] == 0) & (ln > 0))
        if len(ht) != len(want) or (np.sort(ht) != want).any():
            bad.append("the hash table does not hold exactly the hashed rows")
        h = self.hash[want]
        # exact dedupe: no two hashed rows share content
        if len(np.unique(h)) != len(want):
            bad.append("two rows share a hash (duplicate content or a collision)")
        if self.n_rows > self.max_rows:
            bad.append("more rows than max_rows")
        return bad
