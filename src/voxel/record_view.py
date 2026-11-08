"""READ-ONLY HEADER VIEWS over the stores that are deliberately NOT T2 pools (critic point 21).

Wave 13 keeps four other stores for speed and shape: the T3 layers pool, a person's body
(bones and wounds), a person's clothes, and mechanism dicts. Each is the right shape for
its job; none is forced into a T2 pool. But the user asked for ONE record schema, so each
exposes the same HEADER through a view: store, host kind, host id, rid, class, totals
(mass_g, E_J, ml), and an audit hook. `records_audit(recs, views=[...])` runs every hook,
so one audit covers every store, and `headers_of(recs)` gives the pools' own rows in the
same shape, so a reader can walk all of them as one list.

The views take plain dict or list FIXTURES (the stores' own layouts are built in other
lanes; at wiring each builder is pointed at the real object). A view never writes. The
audit hook recomputes the totals from the rows and compares them with the store's
DECLARED totals field: a corrupted foreign totals field is a finding.

Fixture layouts (all values plain Python numbers):
  layers pool : {"rows": [{"rid", "tag", "host", "mass_g", "E_J"}], "totals": {...}}
                tag 0 cell face, 1 record, 2 body region, 3 outdoor column (T3 key tags)
  person body : {"person": id, "bones": [{"rid", "zone", "mass_g"}],
                 "wounds": [{"rid", "zone", "blood_ml"}], "totals": {"mass_g", "ml"}}
  clothes     : [{"rid", "name", "mass_g", "E_J"}] (+ host id and optional totals)
  mechanism   : {"rid", "host", "parts": [{"mass_g"}], "stored_J", "totals": {...}}"""
import math
from collections import namedtuple

from .record import (CLASS_NAMES, HOST_CELL, HOST_PLOT, HOST_REGION, HOST_THING)

Header = namedtuple("Header", "store host_kind host_id rid cls totals")

_TAG_KIND = {0: HOST_CELL, 1: HOST_THING, 2: HOST_REGION, 3: HOST_PLOT}
REL_TOL = 1e-12


def headers_of(recs):
    """The pools' own rows as Headers (store 'records'), in class then row order."""
    out = []
    for cls in sorted(recs.pools):
        p = recs.pools[cls]
        r, g, E = recs.totals(cls)
        for i, row in enumerate(r):
            ml = float(p.a["ml"][row].sum()) if "ml" in p.schema else 0.0
            out.append(Header("records", int(p.a["host_kind"][row]), int(p.a["host_id"][row]),
                              int(p.a["rid"][row]), CLASS_NAMES[cls],
                              {"mass_g": float(g[i].sum()), "E_J": float(E[i]), "ml": ml}))
    return out


class ForeignView:
    """Headers over a foreign store plus an audit hook. `declared` is the totals field the
    store carries (or None): the hook compares it with the sum of the rows."""

    def __init__(self, store, headers, declared=None):
        self.store = store
        self._headers = list(headers)
        self.declared = declared

    def headers(self):
        return list(self._headers)

    def totals(self):
        t = {}
        for h in self._headers:
            for k, v in h.totals.items():
                t[k] = t.get(k, 0.0) + v
        return t

    def audit(self):
        out = []
        for h in self._headers:
            for k, v in h.totals.items():
                if not math.isfinite(v):
                    out.append(dict(kind="corrupt", store=self.store, rid=h.rid, field=k,
                                    detail="non-finite total in a foreign row"))
                elif k in ("mass_g", "ml") and v < 0.0:
                    out.append(dict(kind="corrupt", store=self.store, rid=h.rid, field=k,
                                    detail="negative total in a foreign row"))
        if self.declared is not None:
            have = self.totals()
            for k, want in self.declared.items():
                got = have.get(k, 0.0)
                scale = max(abs(got), abs(want), 1e-300)
                if not math.isfinite(want) or abs(got - want) > REL_TOL * scale:
                    out.append(dict(kind="foreign_totals", store=self.store, field=k,
                                    rows=got, declared=want,
                                    detail="the store's totals field does not match its rows"))
        return out


def view_layers(pool):
    """The T3 layers pool (face stacks). Layer key tag picks the host kind: 0 cell face,
    1 record (host = rid), 2 body region, 3 outdoor column."""
    hs = [Header("layers", _TAG_KIND[int(r["tag"])], int(r["host"]), int(r["rid"]), "LAYER",
                 {"mass_g": float(r["mass_g"]), "E_J": float(r.get("E_J", 0.0))})
          for r in pool["rows"]]
    return ForeignView("layers", hs, pool.get("totals"))


def view_body(person):
    """A person's bones and wounds (p['body']): host kind REGION, host id person*16 + zone."""
    pid = int(person["person"])
    hs = [Header("body", HOST_REGION, pid * 16 + int(b["zone"]), int(b["rid"]), "BONE",
                 {"mass_g": float(b["mass_g"])}) for b in person.get("bones", ())]
    hs += [Header("body", HOST_REGION, pid * 16 + int(w["zone"]), int(w["rid"]), "WOUND",
                  {"ml": float(w["blood_ml"])}) for w in person.get("wounds", ())]
    return ForeignView("body", hs, person.get("totals"))


def view_clothes(items, host_id, totals=None):
    """A person's clothes list (p['heat']['clothes']): host kind THING, host id the person."""
    hs = [Header("clothes", HOST_THING, int(host_id), int(c["rid"]), "CLOTH",
                 {"mass_g": float(c["mass_g"]), "E_J": float(c.get("E_J", 0.0))}) for c in items]
    return ForeignView("clothes", hs, totals)


def view_mechanism(mech):
    """A mechanism dict (T11): host kind THING; totals = parts' mass and the stored energy."""
    mass = float(sum(p["mass_g"] for p in mech.get("parts", ())))
    hs = [Header("mechanism", HOST_THING, int(mech["host"]), int(mech["rid"]), "MECHANISM",
                 {"mass_g": mass, "E_J": float(mech.get("stored_J", 0.0))})]
    return ForeignView("mechanism", hs, mech.get("totals"))
