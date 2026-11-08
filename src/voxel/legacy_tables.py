"""LEGACY TABLES, PADDED (wave 13, T1): one pure function.

materials.py builds its per-mat and per-fluid lookup arrays (_DENS_ARR, _FC_ARR,
...) from the SOLID and FLUID dicts, 12 and 9 entries long. Wave 13 pads them to
64 and 64 so a new material is a ROW, not a code change. This file is that
rebuild, written once, with the SAME expressions materials.py uses, so the first
12 and 9 entries come out equal byte for byte (tests/test_legacy_tables.py checks
that against the frozen base18 oracle, never against the live materials.py).

Imports nothing at the top (no materials, no World): the legacy dicts and the
seed rows are PASSED IN. numpy is imported inside the function.

Padding rule: an id with no row of its own copies its `like` row. A new
skeleton mat that has its own thermal numbers (copper, quartz...) keeps them;
the MECHANICAL tables (SPAN, TOUGH, TSHOCK, TRANSMIT...) are not built here, so
`pad_like` hands the same copy rule to whoever builds them, and LIKE_MAT says
which row to copy. An id nobody listed copies AIR (row 0), which fails safe:
a stray id acts like empty space.
"""

DEFAULT_PAD_LIKE = 0


def pad_like(rows, like, n):
    """Pad `rows` (a list holding the first len(rows) entries) to length n: an
    entry past the end copies the entry its `like` index names. `like[i]` must
    be below i (a row may only copy an earlier row), or the entry is refused."""
    out = list(rows)
    for i in range(len(out), n):
        j = like[i] if i < len(like) else DEFAULT_PAD_LIKE
        if not 0 <= j < i:
            raise ValueError(f"row {i} cannot copy row {j}: like must be earlier")
        out.append(out[j])
    return out


def _like_vector(n, n_legacy, rows, key):
    """Per-id like index: legacy ids copy themselves, a listed id its row's
    `like`, an unlisted id DEFAULT_PAD_LIKE."""
    like = list(range(n_legacy)) + [DEFAULT_PAD_LIKE] * (n - n_legacy)
    for r in rows:
        like[r[key]] = r["like"]
    return like


def build_tables(legacy, seed_mats=(), seed_fluids=(), nmat=64, nfluid=64):
    """Padded per-mat and per-fluid arrays, MELT, FREEZE, BOIL, LATENT.

    legacy: dict with
      SOLID        {mat: (rho g/L, c, k, ign or None, burn, rate)}
      FLUID        {fl: (rho g/ml, c, spread, ign, burn, rate, h)}
      EMISS        {mat: emissivity}
      EMISS_FLUID  {fl: emissivity}
      FLUID_K      {fl: conductivity W/(m K)}
      MELT         {mat: (Tm, fl, Lf)}
      BOIL, LATENT  scalars
      SKIN_T       (depth s, path s)
      NMAT         legacy mat count (12)
      FLIDS        legacy fluid ids in array order (1..8)
    seed_mats:   rows {"mat","like","rho","c","k","ign","emiss", optional "melt"}
    seed_fluids: rows {"fl","like","rho","c","h","emiss","k"}  (rho in g/ml)
    Returns a dict. The array names are the old materials.py names, so a test
    compares one to one: _DENS_ARR _CSOLID_ARR _KSOLID_ARR _IGN_ARR _EMISS_ARR
    _SKIN_DEPTH_ARR _SKIN_PATH_ARR (length nmat) and _FDENS_ARR _FC_ARR _FK_ARR
    _FEMISS_ARR _FLK_ARR _FSKIN_DEPTH_ARR _FSKIN_PATH_ARR (length nfluid; float64
    _ALPHA and _FALPHA too, the diffusivities the skin depths come from), plus
    MELT / FREEZE dicts, dense MELT_T MELT_FL MELT_LAT FREEZE_T FREEZE_MAT
    FREEZE_LAT, BOIL, LATENT, LIKE_MAT, LIKE_FLUID.
    """
    import numpy as np

    solid, fluid = legacy["SOLID"], legacy["FLUID"]
    emiss, emiss_f, flk = legacy["EMISS"], legacy["EMISS_FLUID"], legacy["FLUID_K"]
    skin = legacy["SKIN_T"]
    n_leg, flids = legacy["NMAT"], tuple(legacy["FLIDS"])
    n_legf = 1 + len(flids)
    if flids != tuple(range(1, n_legf)):
        raise ValueError("legacy fluid ids must be 1..n in array order")
    if nmat < n_leg or nfluid < n_legf:
        raise ValueError("padded size is smaller than the legacy size")
    seed_mats, seed_fluids = list(seed_mats), list(seed_fluids)
    for r in seed_mats:
        if not n_leg <= r["mat"] < nmat:
            raise ValueError(f"seed mat id {r['mat']} outside {n_leg}..{nmat - 1}")
    for r in seed_fluids:
        if not n_legf <= r["fl"] < nfluid:
            raise ValueError(f"seed fluid id {r['fl']} outside {n_legf}..{nfluid - 1}")
    if len({r["mat"] for r in seed_mats}) != len(seed_mats) or \
            len({r["fl"] for r in seed_fluids}) != len(seed_fluids):
        raise ValueError("a seed id is listed twice")

    # one row of python floats per id: (rho, c, k, ign, emiss)
    mrow = [(solid[m][0], solid[m][1], solid[m][2],
             np.inf if solid[m][3] is None else solid[m][3], emiss[m])
            for m in range(n_leg)]
    like_m = _like_vector(nmat, n_leg, seed_mats, "mat")
    by_mat = {r["mat"]: r for r in seed_mats}
    for i in range(n_leg, nmat):
        r = by_mat.get(i)
        if r is not None:
            mrow.append((r["rho"], r["c"], r["k"],
                         np.inf if r["ign"] is None else r["ign"], r["emiss"]))
        else:
            mrow = pad_like(mrow, like_m, i + 1)
    # one row per fluid id: (rho, c, h, emiss, k_liquid, alpha)
    frow = [(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)]
    for f in flids:
        frow.append((fluid[f][0], fluid[f][1], fluid[f][6], emiss_f[f], flk[f],
                     flk[f] / (fluid[f][0] * fluid[f][1] * 1e6)))
    like_f = _like_vector(nfluid, n_legf, seed_fluids, "fl")
    by_fl = {r["fl"]: r for r in seed_fluids}
    for i in range(n_legf, nfluid):
        r = by_fl.get(i)
        if r is not None:
            frow.append((r["rho"], r["c"], r["h"], r["emiss"], r["k"],
                         r["k"] / (r["rho"] * r["c"] * 1e6)))
        else:
            frow = pad_like(frow, like_f, i + 1)

    f32 = np.float32
    out = {}
    out["_DENS_ARR"] = np.array([r[0] for r in mrow], f32)
    out["_CSOLID_ARR"] = np.array([r[1] for r in mrow], f32)
    out["_KSOLID_ARR"] = np.array([r[2] for r in mrow], f32)
    out["_IGN_ARR"] = np.array([r[3] for r in mrow], f32)
    out["_EMISS_ARR"] = np.array([r[4] for r in mrow], f32)
    alpha = np.array([r[2] / (r[0] * r[1] * 1000.0) for r in mrow])
    out["_ALPHA"] = alpha
    out["_SKIN_DEPTH_ARR"] = np.sqrt(alpha * skin[0]).astype(f32)
    out["_SKIN_PATH_ARR"] = np.sqrt(alpha * skin[1]).astype(f32)
    out["_FDENS_ARR"] = np.array([r[0] for r in frow], f32)
    out["_FC_ARR"] = np.array([r[1] for r in frow], f32)
    out["_FK_ARR"] = np.array([r[2] for r in frow], f32)
    out["_FEMISS_ARR"] = np.array([r[3] for r in frow], f32)
    out["_FLK_ARR"] = np.array([r[4] for r in frow], f32)
    falpha = np.array([r[5] for r in frow])
    out["_FALPHA"] = falpha
    out["_FSKIN_DEPTH_ARR"] = np.sqrt(falpha * skin[0]).astype(f32)
    out["_FSKIN_PATH_ARR"] = np.sqrt(falpha * skin[1]).astype(f32)

    # MELT / FREEZE: own data only for a listed id; an unlisted id copies like.
    melt = {int(m): tuple(v) for m, v in legacy["MELT"].items()}
    for r in seed_mats:
        if r.get("melt") is not None:
            melt[r["mat"]] = tuple(r["melt"])
    mrows = [melt.get(m) for m in range(nmat)]
    for i in range(n_leg, nmat):
        if i not in by_mat:
            mrows[i] = mrows[like_m[i]]
    # a copied row is a pad of a pad: no melt entry is invented for a padded id
    # whose like row has none, and a copied entry never enters the MELT dict
    out["MELT"] = melt
    out["FREEZE"] = {v[1]: (v[0], m, v[2]) for m, v in melt.items()}
    inf = float("inf")
    out["MELT_T"] = np.array([inf if r is None else r[0] for r in mrows], np.float64)
    out["MELT_FL"] = np.array([0 if r is None else r[1] for r in mrows], np.int16)
    out["MELT_LAT"] = np.array([0.0 if r is None else r[2] for r in mrows], np.float64)
    freeze = out["FREEZE"]
    frows = [freeze.get(f) for f in range(nfluid)]
    for i in range(n_legf, nfluid):
        if i not in by_fl:
            frows[i] = frows[like_f[i]]
    out["FREEZE_T"] = np.array([-inf if r is None else r[0] for r in frows], np.float64)
    out["FREEZE_MAT"] = np.array([0 if r is None else r[1] for r in frows], np.int16)
    out["FREEZE_LAT"] = np.array([0.0 if r is None else r[2] for r in frows], np.float64)
    out["BOIL"] = legacy["BOIL"]
    out["LATENT"] = legacy["LATENT"]
    out["LIKE_MAT"] = np.array(like_m, np.int16)
    out["LIKE_FLUID"] = np.array(like_f, np.int16)
    out["NMAT"], out["NFLUID"] = nmat, nfluid
    return out
