"""BodyWorld: a World whose people are heat engines (T7, wave 13b, lane C6).

`class BodyWorld(BodyHeatMixin, ColdWorld)`. This is the CAPABILITY GATE of
body heat (decision N5): the class, and the world setting `body_heat`
(True here). A plain `World` has no body heat and is bit for bit what it was;
C6 does not own sim.py, so the mixin is not in World's bases (the merge step
decides whether it ever is; see the C6 report, N5 outcome).

WHAT THIS CLASS ADDS TO THE MIXIN (`bodyheat.BodyHeatMixin`, which was tested on
a slab fixture and on a throwaway subclass in 13a):

  * `_law_life` runs the body law after the slow chemistry of the person, once
    a body second (the mixin staggers it by pid);
  * a person gets a heat record the first time the law meets them (lazy; nothing
    in `add_person` calls it), warm, with the flesh at its resting state. The
    warm-up is a NAMED injection (`bh_injected()["body_warmup"]`), not a leak;
  * the face-law hooks BH1-BH3 (`bh_set_hooks`): a uint16 overlay id per flesh
    voxel (`bzone` = slot x 16 + zone + 1) and one row per (person, zone) in
    the face-environment table (faceenv.py): natural-convection law, area
    factor, core-to-skin resistance, series garment resistance;
  * emissivity of skin 0.98 (the film column of the material row);
  * lattice energy as float64 (D4), the cure of the walking leak of float32
    (0.83 J per 100 ticks for a walking 34 C body);
  * the person's own cells follow the body when it walks (the overlay id is a
    CARRIED_OPT field, so `_lift` and `_put` move it);
  * the books: `total_energy` adds the pool, the stores, the pending acts and
    the mechanical work given to the world.

Marks: (u) unverified; (d) design choice.
"""
import numpy as np

from . import bodynodes as BN
from . import digestion as DG
from . import faceenv as FE
from . import tissue as TI
from .bodyheat import BodyHeatMixin
from .coldworld import ColdWorld
from .materials import AIR

SLOT_SPAN = 16                       # overlay ids per person: slot x 16 + zone + 1 (zones are 0..13)


class BodyWorld(BodyHeatMixin, ColdWorld):
    E_dtype = np.float64             # D4: the energy array is float64 where bodies are warm
    body_heat = True                 # the capability: off, and a person gets no record (d)

    # ---------------------------------------------------------- the host ---
    def _emiss_table(self):
        """Skin emits at 0.98 (ISO 7730 / Incropera) and not the old 0.95: the
        film column of the material row, read where body heat is on."""
        from .materials import _EMISS_FILM_ARR
        return _EMISS_FILM_ARR

    def bh_env(self, p):
        """The air a body stands in, read from the lattice: the mean of the gas
        voxels that touch it (what its faces see), the relative humidity of the
        outside air if the world has one (else 0.5, (d)), no wind (the lattice
        carries its own, and the forced term of the mixin is for a measured
        draught). The mean radiant temperature of the world past the lattice
        edge is the air's."""
        h = None if p is None else p.get("heat")
        T_air = 20.0
        if h is not None:
            T_air = self._bh_touching_air(h)
        rh = float(self.air_rh)
        return {"T_air": T_air, "rh": rh, "v": 0.0, "T_rad": T_air}

    def _bh_touching_air(self, h):
        """Mean temperature, C, of the gas voxels next to the body's cells."""
        cells = h["cells"]
        nx, ny, nz = self.shape
        acc = []
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            c = cells + np.array(d)
            ok = ((c >= 0) & (c < np.array(self.shape))).all(axis=1)
            c = c[ok]
            if len(c):
                acc.append(c)
        if not acc:
            return 20.0
        c = np.unique(np.concatenate(acc), axis=0)
        idx = tuple(c.T)
        gas = self.mat[idx] == AIR
        if not gas.any():
            return 20.0
        idx = tuple(a[gas] for a in idx)
        C = np.asarray(self.heat_capacity()[idx], dtype=np.float64)
        return float(BN.AMBIENT + (np.asarray(self.E[idx], dtype=np.float64) / C).mean())

    def bh_set_hooks(self, pid, cells, zone_idx, tables):
        """BH1-BH3: hand the face law its per-zone rows. One overlay id per
        (person slot, zone); the table row is rewritten in place on every call
        (a garment getting wet changes its row)."""
        p = next((q for q in self.persons if q.get("pid") == pid), None)
        h = None if p is None else p.get("heat")
        slot = int(h["slot"]) if h is not None and "slot" in h else int(p.get("_bh_slot", 0)) if p else 0
        bz = self.ensure_bzone()
        zone = np.asarray(zone_idx, dtype=np.int64)
        idx = tuple(np.asarray(cells, dtype=np.int64).T)
        bz[idx] = (slot * SLOT_SPAN + zone + 1).astype(np.uint16)
        n_rows = (slot + 1) * SLOT_SPAN + 1
        fe = self.face_env
        old = None if fe is None else fe.ov
        ov = FE.overlay_table() if old is None else np.array(old, np.float32)
        if len(ov) < n_rows:                                   # grow with the DEFAULT row
            pad = np.tile(np.array(FE.DEFAULT_ROW, np.float32), (n_rows - len(ov), 1))
            ov = np.concatenate([ov, pad])
        for z in range(BN.NZ):
            R_cl, f_cl = float(tables["R_cl"][z]), float(tables["f_cl"][z])
            az = float(tables["az"][z])
            gsb = float(tables["gsb"][z])
            row = list(FE.DEFAULT_ROW)
            row[FE.OV_HNAT] = float(tables["conv_nat"][z])
            row[FE.OV_AZ] = az * (f_cl if R_cl > 0.0 else 1.0)
            row[FE.OV_RSC] = 1.0 / gsb if gsb > 0.0 else 0.0   # m2K/W of lattice area: 1/gsb
            if R_cl > 0.0:                                      # a garment in series, outer skin radiates
                row[FE.OV_RSER] = R_cl / max(az, 1e-9)
                row[FE.OV_RADW] = 0.0
            ov[slot * SLOT_SPAN + z + 1] = row
        if old is None or fe.bz is not bz or ov.shape != old.shape:
            self.face_env = FE.FaceEnv(bz=bz, ov=ov, bp=None if fe is None or not fe.boils else fe.bp,
                                       ff=False if fe is None else fe.ff)
        else:
            old[:] = ov                                         # in place: the law re-reads the table each tick

    # ------------------------------------------------------ the person ----
    def body_heat_person(self, p):
        """Give one person a heat record now (warm), from their own sheet if
        they have one. Returns the record."""
        h = p.get("heat")
        if h is not None:
            return h
        sheet = None
        if p.get("sheet"):
            sheet = TI.Sheet.from_dict(_sheet_numbers(p["sheet"]))
        slot = sum(1 for q in self.persons if q.get("heat") is not None)
        p["_bh_slot"] = slot
        h = self.body_heat_on(p, sheet=sheet)
        h["slot"] = slot
        # the chemical stores a body starts with (fat, glycogen, gut) come from nowhere
        # the lattice knows: a NAMED injection, like the warm-up, so the books close
        J = DG.stores_J(h["stores"])
        inj = self.bh_injected()
        inj["stores_init"] = inj.get("stores_init", 0.0) + J
        self.shed -= J
        self._bh_hooks(p, h, self._bh_geom(p, h))              # again, now the slot is on the record
        return h

    def _bh_follow(self, p, h):
        """The body has moved (it walked, was carried, turned): read where its
        voxels are now from the overlay ids, which travelled with the flesh. A
        pure shift keeps the order of the cells; any other change (a limb in a
        new pose, a voxel lost) rebuilds the geometry on the next second."""
        mask, sl = self._person_cells(p)
        if mask is None or not mask.any():
            return
        bz = self.bzone
        sub = bz[sl]
        base = h["slot"] * SLOT_SPAN
        sel = mask & (sub > base) & (sub <= base + BN.NZ)
        off = np.array([s.start or 0 for s in sl])
        cells = np.argwhere(sel) + off
        if len(cells) == len(h["cells"]) and np.array_equal(cells, h["cells"]):
            return
        zone = (sub[sel].astype(np.int64) - base - 1)
        if len(cells) == 0:
            return
        h["cells"], h["zone"] = cells.astype(np.int64), zone
        ent = self._bh_cache().get(h["pid"])
        if ent is not None:                                    # where the rays end is not where it was
            ent["rayc"] = None
            ent["zgrid"] = None

    def _bh_sync_in(self, p, h):
        """What the rest of the person knows that the body law needs: alive,
        asleep or out (an unconscious body rests like a sleeper), and the MET
        the physiology law has smoothed."""
        h["met"] = float(p.get("met", 1.0))
        h["asleep"] = bool(p.get("asleep")) or not p["awake"]

    def _law_life(self):
        super()._law_life()
        if not self.body_heat:
            return
        period = self._bh_period()
        for p in self.persons:
            if (self.tick + p.get("pid", 0)) % period:
                continue
            h = p.get("heat")
            if h is None:
                if not p["alive"] or p.get("safe"):
                    continue
                mask, _sl = self._person_cells(p)
                if mask is None or not mask.any():             # not on the lattice (in the air, gone): later
                    continue
                h = self.body_heat_person(p)
            if h["alive"] and not p["alive"]:
                self.body_heat_death(p)                        # the pool's heat goes into the flesh
                continue
            if not h["alive"]:
                continue
            self._bh_follow(p, h)
            self._bh_sync_in(p, h)
            self.bh_second(p, h)

    def bh_emit(self, p, verb, edge):
        """Every firing of a thermoregulation row is a typed event the person's
        own mind perceives (reflex contract, item 5). Only the onset is told; the
        offset is in the record's event list."""
        if edge != "on":
            return
        kind = "shiver" if verb == "shivers" else "sweat"
        word = "starts to shiver" if kind == "shiver" else "starts to sweat"
        self._reflex_fire(p, kind, verb, "%s %s" % (p["name"], word))

    def _law_body_heat(self):
        """The mixin's own loop is run by `_law_life` above (it has to create and
        follow the records first)."""
        return None

    # --------------------------------------------------------- the books ---
    def total_energy(self):
        """The base total plus the body terms (T7 3.10): the pool, the stores,
        the acts billed and not yet deposited, and the mechanical work given
        to the world. The named injections (warm-up, food) are in `shed`."""
        t = self.bh_terms()
        return (super().total_energy() + t["pool_J"] + t["stores_J"] + t["pending_J"]
                + t["work_out"])


def _sheet_numbers(sheet):
    """The Sheet fields out of a body sheet dict (bodysheet.get_sheet has more
    keys than tissue.Sheet takes)."""
    return {"sex": sheet.get("sex", "male"), "mass_kg": sheet["mass_kg"],
            "height_m": sheet["height_m"], "fat_frac": sheet.get("fat_frac", 0.19),
            "age_y": sheet.get("age_y", 30.0), "bmr_scale": sheet.get("bmr_scale", 1.0)}
