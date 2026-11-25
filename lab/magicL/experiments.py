"""R2.1 stages. usage: python -m lab.magicL.experiments STAGE   (writes JSON to scratch + results)"""
import sys, os, json, time
import numpy as np
from lab.magicL.lgt import Sim, block_err, observables, plaq_angles, energy_parts, TWO_PI
from lab.magicL import euclid4d as e4

SCR = "/home/nisargparikh/.cache/mark1-build/magicL/R2_1"
RES = "/home/nisargparikh/Desktop/Fun Stuff/Mark_1/lab/results/magicL/R2_1"
SEEDS = (11, 22, 33)
PAUSE = "/home/nisargparikh/.cache/mark1-build/lab.pause"


def wait_pause():
    while os.path.exists(PAUSE):
        time.sleep(10)


def save(name, obj):
    for d in (SCR, RES):
        with open(os.path.join(d, name + ".json"), "w") as f:
            json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def done(name):
    return os.path.exists(os.path.join(SCR, name + ".json"))


def sd_across(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


# ------------------------------------------------------------------ checks
def stage_checks():
    out = {}
    for tag, kw in (("pure", dict(K=2.0, use_m=False)), ("matter", dict(K=2.0, J=2.0, chi=0.25, use_m=True))):
        for dt in (0.04, 0.02, 0.01):
            wait_pause()
            s = Sim(8, seed=5, dt=dt, **kw)
            s.set_charges(s.site(1, 1, 1), s.site(5, 3, 2))
            s.hot_start(0.0)
            s.thermalise(30, int(2 / dt))
            s.G0 = s.gauss().copy()
            T_total = 600.0
            nchunk = 300
            per = int(T_total / dt / nchunk)
            H = []; gm = 0.0
            for i in range(nchunk):
                s.step(per)
                H.append(sum(s.energies()))
                gm = max(gm, float(np.abs(s.gauss() - s.G0).max()))
            H = np.array(H); t = np.arange(nchunk) * per * dt
            slope = np.polyfit(t, H, 1)[0]
            out[f"{tag}_dt{dt}"] = dict(H0=float(H[0]), rel_fluct=float(H.std() / abs(H.mean())),
                                        max_dev=float(np.abs(H - H[0]).max() / abs(H[0])),
                                        drift_per_1e5_steps_rel=float(slope * dt * 1e5 / abs(H[0])),
                                        max_gauss_dev=gm, steps=nchunk * per)
            print(tag, dt, out[f"{tag}_dt{dt}"], flush=True)
    save("checks", out)


# ------------------------------------------------------------------ waves
def spectrum_run(s, nsamp=8192, stride=5, ns=(0, 1, 2, 3), obs_every=64):
    """microcanonical run; record transverse E_y(k_x=2 pi n/L) averaged over y,z. returns peaks (omega) per n and observables."""
    L = s.L
    ser = {n: np.empty(nsamp, complex) for n in ns}
    obs = []
    for i in range(nsamp):
        s.step(stride)
        Ey = s.E[1].reshape(L, L, L).mean(axis=(1, 2))
        F = np.fft.fft(Ey) / L
        for n in ns:
            ser[n][i] = F[n]
        if i % obs_every == 0:
            obs.append(s.obs())
    dts = stride * s.dt
    res = {}
    w = np.hanning(nsamp)
    om = 2 * np.pi * np.fft.fftfreq(nsamp, dts)
    for n in ns:
        x = ser[n] - (ser[n].mean() if n == 0 else 0)
        F = np.fft.fft(x * w)
        if n == 0:
            P = np.abs(F) ** 2
            P = P[:nsamp // 2] + np.concatenate([[0], P[:nsamp // 2:-1][:nsamp // 2 - 1]])
            oo = om[:nsamp // 2]
        else:
            P = np.abs(F) ** 2
            Pp = P[:nsamp // 2]
            Pm = np.concatenate([[0], P[::-1][:nsamp // 2 - 1]])  # P(-omega)
            P = Pp + Pm[:len(Pp)]
            oo = om[:nsamp // 2]
        lo = 3  # skip DC bins
        j = lo + int(np.argmax(P[lo:]))
        if 1 <= j < len(P) - 1:
            a, b, c = np.log(P[j - 1] + 1e-300), np.log(P[j] + 1e-300), np.log(P[j + 1] + 1e-300)
            den = a - 2 * b + c
            dlt = 0.5 * (a - c) / den if den != 0 else 0.0
        else:
            dlt = 0.0
        res[n] = float(oo[j] + dlt * (oo[1] - oo[0]))
        res[f"dom{n}"] = float(oo[1] - oo[0])
        # half max width of the line (crude)
    ob = np.array(obs)
    return res, ob


def stage_waves():
    out = {}
    L = 16
    cases = []
    for K in (0.5, 1, 2, 4, 8, 16):
        cases.append((f"pure_K{K}_T1", dict(K=K, T=1.0, use_m=False)))
    for T in (0.25, 4.0):
        cases.append((f"pure_K4_T{T}", dict(K=4.0, T=T, use_m=False)))
    for J in (0.5, 2.0, 8.0):
        cases.append((f"matter_K2_J{J}", dict(K=2.0, J=J, chi=0.25, T=1.0, use_m=True)))
    for name, kw in cases:
        if name in out: continue
        per_seed = []
        for sd in SEEDS:
            wait_pause()
            s = Sim(L, seed=sd, **kw)
            s.hot_start(0.0)
            s.thermalise(30, 100)
            res, ob = spectrum_run(s)
            res["plaq"] = float(ob[:, 0].mean())
            res["higgs"] = float(ob[:, 1].mean())
            per_seed.append(res)
        agg = {}
        for key in ("0", "1", "2", "3"):
            vals = [r[int(key)] for r in per_seed]
            agg[key] = sd_across(vals)
        agg["plaq"] = sd_across([r["plaq"] for r in per_seed])
        agg["higgs"] = sd_across([r["higgs"] for r in per_seed])
        agg["res_dom"] = per_seed[0]["dom1"]
        out[name] = agg
        print(name, {k: (round(v[0], 4), round(v[1], 4)) if isinstance(v, tuple) else v for k, v in agg.items()}, flush=True)
        save("waves", out)


# ------------------------------------------------------------------ force law and screening
# FINDING while building: A = <E,Ec>/<Ec,Ec> is identically 1 because Gauss law fixes it (it is a check of the code, not a
# physical test). The physical observables are the LOCAL mean field (axial field between the charges, induced matter charge)
# and, for matter, how fast the screening cloud forms (mode P2: charges inserted bare, matter must move to screen them).
def bare_field(s):
    um = s.use_m
    s.use_m = False
    try:
        E0 = np.zeros_like(s.E); p0 = np.zeros(s.N)
        s.project(E0, p0)
    finally:
        s.use_m = um
    return E0


def force_run(L, cond, r, seed, mode="P1", t_end=800.0, every=10, ctrl=False):
    """P1: insert charges into the equilibrium sector (E and pi shifted to the Gauss minimiser), microcanonical afterwards.
    P2: insert bare (only E shifted), watch the screening form. ctrl: no charges (same seed) for the ensemble comparison."""
    s = Sim(L, seed=seed, **cond)
    s.hot_start(0.0)
    s.thermalise(40, 100)
    c = L // 2
    a = s.site(c - r // 2, c, c); b = s.site(c - r // 2 + r, c, c)
    axis = [s.site(c - r // 2 + i, c, c) for i in range(r)]
    s.set_charges(a, b)
    Ec, pc = s.mean_field()
    Eb = bare_field(s)
    if ctrl:
        s.rho[:] = 0
    else:
        if mode == "P1":
            s.E += Ec; s.pi += pc
        else:
            s.E += Eb
    s.G0 = s.gauss().copy()
    nE = float((Ec ** 2).sum())
    nP = float((pc ** 2).sum()) if s.use_m else 1.0
    nsteps = int(t_end / s.dt)
    ns = nsteps // every
    ax = np.empty(ns); aP = np.empty(ns); aE = np.empty(ns); pl = np.empty(ns); hg = np.empty(ns); mo = np.empty(ns)
    for i in range(ns):
        s.step(every)
        ax[i] = s.E[0, axis].mean()
        aE[i] = float((s.E * Ec).sum()) / nE
        aP[i] = float((s.pi * pc).sum()) / nP if s.use_m else 0.0
        if i % 20 == 0:
            o = s.obs(); pl[i] = o[0]; hg[i] = o[1]; mo[i] = o[2] / s.N
        else:
            pl[i] = hg[i] = mo[i] = np.nan
    gm = float(np.abs(s.gauss() - s.G0).max())
    return dict(ax=ax, aE=aE, aP=aP, pl=pl, hg=hg, mo=mo, gauss_dev=gm, ax_eq=float(Ec[0, axis].mean()),
                ax_bare=float(Eb[0, axis].mean()), U_eq=0.5 * (float((Ec ** 2).sum()) + (float((pc ** 2).sum()) / s.chi if s.use_m else 0)),
                U_bare=0.5 * float((Eb ** 2).sum()), dt=s.dt * every)


def win(x, dt, t0, t1):
    return x[int(t0 / dt):int(t1 / dt)]


def stage_force():
    """P1 consistency (does the Coulomb field leave the gauge/matter ensemble alone?) and the axial field vs r."""
    L = 16
    conds = {
        "pure_K4": dict(K=4.0, use_m=False), "pure_K1": dict(K=1.0, use_m=False), "pure_K0.25": dict(K=0.25, use_m=False),
        "chi.25_K2_J2": dict(K=2.0, J=2.0, chi=0.25, use_m=True), "chi.25_K0.5_J4": dict(K=0.5, J=4.0, chi=0.25, use_m=True),
        "chi.25_K4_J8": dict(K=4.0, J=8.0, chi=0.25, use_m=True),
    }
    path = os.path.join(SCR, "force.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for cn, cond in conds.items():
        for r in (2, 4, 6, 8):
            key = f"{cn}|r{r}"
            if key in out: continue
            R = []; C = []
            for sd in SEEDS:
                wait_pause()
                R.append(force_run(L, cond, r, sd, "P1", t_end=300.0))
                C.append(force_run(L, cond, r, sd, "P1", t_end=300.0, ctrl=True))
            axm = [x["ax"].mean() for x in R]
            if cond["use_m"] and r == 4:
                pass
            d = dict(ax=sd_across(axm), ax_eq=R[0]["ax_eq"], ax_bare=R[0]["ax_bare"], U_eq=R[0]["U_eq"], U_bare=R[0]["U_bare"],
                     aP=sd_across([x["aP"].mean() for x in R]) if cond["use_m"] else None,
                     gauss_dev=max(x["gauss_dev"] for x in R),
                     plaq=sd_across([np.nanmean(x["pl"]) for x in R]), higgs=sd_across([np.nanmean(x["hg"]) for x in R]),
                     mono=sd_across([np.nanmean(x["mo"]) for x in R]))
            d["ax_paired"] = sd_across([x["ax"].mean() - c["ax"].mean() for x, c in zip(R, C)])
            d["aP_paired"] = sd_across([x["aP"].mean() - c["aP"].mean() for x, c in zip(R, C)]) if cond["use_m"] else None
            d["dplaq"] = sd_across([np.nanmean(x["pl"]) - np.nanmean(c["pl"]) for x, c in zip(R, C)])
            d["dhiggs"] = sd_across([np.nanmean(x["hg"]) - np.nanmean(c["hg"]) for x, c in zip(R, C)]) if cond["use_m"] else None
            if C:
                d["ctrl_plaq"] = sd_across([np.nanmean(x["pl"]) for x in C]); d["ctrl_higgs"] = sd_across([np.nanmean(x["hg"]) for x in C])
                d["ctrl_mono"] = sd_across([np.nanmean(x["mo"]) for x in C])
            out[key] = d
            print(key, {k: v for k, v in d.items()}, flush=True)
            save("force", out)


def stage_screen():
    """P2: bare insertion; the screened fraction aP(t) (induced matter charge on the equilibrium cloud) vs time and J, K."""
    L = 16
    conds = {
        "K2_J0": dict(K=2.0, J=0.0, chi=0.25, use_m=True), "K2_J0.5": dict(K=2.0, J=0.5, chi=0.25, use_m=True),
        "K2_J2": dict(K=2.0, J=2.0, chi=0.25, use_m=True), "K2_J8": dict(K=2.0, J=8.0, chi=0.25, use_m=True),
        "K0.5_J2": dict(K=0.5, J=2.0, chi=0.25, use_m=True), "K8_J2": dict(K=8.0, J=2.0, chi=0.25, use_m=True),
        "K2_J2_chi.04": dict(K=2.0, J=2.0, chi=0.04, use_m=True),
    }
    path = os.path.join(SCR, "screen.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    windows = [(0, 1), (1, 3), (3, 10), (10, 40), (40, 150), (150, 800)]
    for cn, cond in conds.items():
        for r in ((4, 8) if cn in ("K2_J2",) else (4,)):
            key = f"{cn}|r{r}"
            if key in out: continue
            R = []; C = []
            for sd in SEEDS:
                wait_pause()
                R.append(force_run(L, cond, r, sd, "P2", t_end=800.0, every=5))
                C.append(force_run(L, cond, r, sd, "P2", t_end=800.0, every=5, ctrl=True))
            d = dict(ax_bare=R[0]["ax_bare"], ax_eq=R[0]["ax_eq"], U_bare=R[0]["U_bare"], U_eq=R[0]["U_eq"],
                     gauss_dev=max(x["gauss_dev"] for x in R), win={})
            for (t0, t1) in windows:
                d["win"][f"{t0}-{t1}"] = dict(aP=sd_across([win(x["aP"], x["dt"], t0, t1).mean() - win(c["aP"], c["dt"], t0, t1).mean() for x, c in zip(R, C)]),
                                              aE=sd_across([win(x["aE"], x["dt"], t0, t1).mean() - win(c["aE"], c["dt"], t0, t1).mean() for x, c in zip(R, C)]),
                                              ax=sd_across([win(x["ax"], x["dt"], t0, t1).mean() - win(c["ax"], c["dt"], t0, t1).mean() for x, c in zip(R, C)]),
                                              hg=sd_across([np.nanmean(win(x["hg"], x["dt"], t0, t1)) for x in R]))
            out[key] = d
            print(key, {k: (v if k != "win" else {w: round(z["aP"][0], 3) for w, z in v.items()}) for k, v in d.items()}, flush=True)
            save("screen", out)


# ------------------------------------------------------------------ monopoles (3D classical) and 4D control
def stage_mono():
    path = os.path.join(SCR, "mono.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    def one(L, K, J, use_m, seed, nmeas=1500):
        s = Sim(L, K=K, J=J, chi=0.25, use_m=use_m, seed=seed)
        s.hot_start(0.0)
        s.thermalise(40, 100)
        P = []; M = []; H = []; V = []
        for i in range(nmeas):
            s.step(100); s.refresh()
            pc, hc, mon, nv, vxy = s.obs()
            P.append(pc); M.append(mon / s.N); H.append(hc); V.append(nv / (3 * s.N))
        return np.array(P), np.array(M), np.array(H), np.array(V)
    grid = [("pure", 0.0, False, (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)),
            ("J0.5", 0.5, True, (0.25, 0.5, 1.0, 2.0, 4.0)),
            ("J2", 2.0, True, (0.25, 0.5, 1.0, 2.0, 4.0))]
    for tag, J, um, Ks in grid:
        for K in Ks:
            for L in ((8, 12, 16) if (tag == "pure" and K in (0.5, 1.0, 1.5)) else (12,)):
                key = f"{tag}|K{K}|L{L}"
                if key in out: continue
                wait_pause()
                res = [one(L, K, J, um, sd, 1500 if L < 16 else 800) for sd in SEEDS]
                dens = [r[1].mean() for r in res]; plq = [r[0].mean() for r in res]
                hig = [r[2].mean() for r in res]; vor = [r[3].mean() for r in res]
                nsite = L ** 3 * len(res[0][1]) * len(SEEDS)
                dm, de = sd_across(dens)
                pe = [block_err(r[1])[1] for r in res]
                out[key] = dict(dens=dm, dens_err=max(de, float(np.sqrt(np.mean(np.square(pe)) / 3))), plaq=sd_across(plq),
                                higgs=sd_across(hig), vort=sd_across(vor), upper3=3.0 / nsite if dm == 0 else None,
                                susc_plaq=float(np.mean([r[0].var() * 3 * L ** 3 for r in res])))
                print(key, out[key], flush=True)
                save("mono", out)


def stage_e4():
    path = os.path.join(SCR, "e4.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for L in (8,):
        for beta in (0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0):
            key = f"L{L}|b{beta}"
            if key in out: continue
            wait_pause()
            runs = [e4.run(L, beta, ntherm=500, nmeas=2000, every=5, seed=sd) for sd in SEEDS]
            d = {}
            def W(r, k): return r["W"][k]
            def creutz(r, R):
                a = W(r, f"{R}x{R}"); b = W(r, "1x1")
                # chi(R,R) = -ln[ W(R,R) W(R-1,R-1) / (W(R,R-1) W(R-1,R)) ]; R=2 uses W(1,1); R=3 uses 2x2
                if R == 2:
                    num = W(r, "2x2") * W(r, "1x1"); den = W(r, "1x2") * W(r, "2x1")
                else:
                    num = W(r, "3x3") * W(r, "2x2"); den = W(r, "2x3") * W(r, "3x2")
                return num.mean(), den.mean()
            for R in (2, 3):
                vals = []
                for r in runs:
                    n_, d_ = creutz(r, R)
                    vals.append(-np.log(n_ / d_) if n_ > 0 and d_ > 0 else float("nan"))
                d[f"creutz{R}"] = sd_across(vals)
            d["plaq"] = sd_across([r["W"]["1x1"].mean() for r in runs])
            d["mono"] = sd_across([r["mono"].mean() for r in runs])
            out[key] = d
            print(key, d, flush=True)
            save("e4", out)


# ------------------------------------------------------------------ flux tubes
def twist_flux(s, nf):
    L = s.L
    th = np.zeros((3, L, L, L))
    x = np.arange(L)[:, None, None] * np.ones((1, L, L))
    y = np.arange(L)[None, :, None] * np.ones((L, 1, L))
    th[1] = 2 * np.pi * nf * x / L ** 2
    th[0, L - 1, :, :] = -2 * np.pi * nf * y[L - 1] / L
    s.th[:] = ((th + np.pi) % TWO_PI - np.pi).reshape(3, -1)


def flux_stats(s, nf):
    L = s.L
    B = plaq_angles(s.th, s.fwd)[2].reshape(L, L, L)  # xy plane plaquettes
    # vorticity in xy plane at z=0
    th = s.th.reshape(3, L, L, L); phi = s.phi.reshape(L, L, L)
    def wrap(a): return a - TWO_PI * np.floor((a + np.pi) / TWO_PI)
    d0 = wrap(phi - np.roll(phi, -1, 0) - th[0])
    d1 = wrap(phi - np.roll(phi, -1, 1) - th[1])
    w = d0 + np.roll(d1, -1, 0) - np.roll(d0, -1, 1) - d1
    v = np.rint((w + B) / TWO_PI).astype(int)
    flux_plane = B[:, :, 0].sum() / TWO_PI
    nvp = int((v[:, :, 0] > 0).sum()); nvm = int((v[:, :, 0] < 0).sum())
    Bz = B[:, :, 0]
    ipr = float(Bz.size * (Bz ** 2).sum() / (Bz.sum() ** 2))
    # flux around vortices (3x3) in plane 0
    tube = []
    pos = np.argwhere(v[:, :, 0] > 0)
    tot_in = 0.0
    mask = np.zeros_like(Bz, bool)
    for (i, j) in pos:
        blk = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                blk += Bz[(i + di) % L, (j + dj) % L]
                mask[(i + di) % L, (j + dj) % L] = True
        tube.append(blk / TWO_PI)
    frac_in = float(Bz[mask].sum() / Bz.sum()) if Bz.sum() != 0 and mask.any() else float("nan")
    return dict(flux_quanta=float(flux_plane), n_vort_pos=nvp, n_vort_neg=nvm, vort_net=int(v[:, :, 0].sum()),
                ipr=ipr, tube_flux_mean=float(np.mean(tube)) if tube else float("nan"), frac_in_tubes=frac_in,
                npix_in=int(mask.sum()))


def stage_flux():
    path = os.path.join(SCR, "flux.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    L = 16; nf = 4
    for J in (0.0, 0.1, 0.25, 0.5, 1.0, 4.0):
        for sd in SEEDS:
            key = f"J{J}|s{sd}"
            if key in out: continue
            wait_pause()
            s = Sim(L, K=1.0, J=J, chi=0.25, T=0.3, use_m=True, seed=sd)
            twist_flux(s, nf)
            s.phi[:] = s.rng.normal(0, 0.3, s.N)
            s.refresh(); s.G0 = s.gauss().copy()
            f0 = flux_stats(s, nf)
            for T in np.geomspace(0.3, 0.01, 12):
                s.T = T
                for _ in range(40):
                    s.step(50); s.refresh()
            for _ in range(600):   # damped relaxation, same factor on E and pi keeps G = 0
                s.step(10); s.E *= 0.93; s.pi *= 0.93
            st = flux_stats(s, nf)
            st["gauss_dev"] = float(np.abs(s.gauss() - s.G0).max()); st["init_ipr"] = f0["ipr"]
            st["plaq"], st["higgs"] = s.obs()[0], s.obs()[1]
            st["energy"] = [float(x) for x in s.energies()]
            out[key] = st
            print(key, st, flush=True)
            save("flux", out)


# ------------------------------------------------------------------ phase map
def stage_map():
    path = os.path.join(SCR, "map.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    L = 12
    for K in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for J in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            key = f"K{K}|J{J}"
            if key in out: continue
            wait_pause()
            rs = []
            for sd in SEEDS[:2]:
                s = Sim(L, K=K, J=J, chi=0.25, T=1.0, use_m=True, seed=sd)
                s.hot_start(0.0)
                s.thermalise(30, 100)
                P = []; H = []; M = []; V = []
                for i in range(200):
                    s.step(100); s.refresh()
                    pc, hc, mon, nv, vxy = s.obs()
                    P.append(pc); H.append(hc); M.append(mon / s.N); V.append(nv / (3 * s.N))
                res, ob = spectrum_run(s, nsamp=2048, stride=5, ns=(0, 1))
                rs.append(dict(plaq=np.mean(P), higgs=np.mean(H), mono=np.mean(M), vort=np.mean(V), w0=res[0], w1=res[1],
                               dom=res["dom0"], susc_h=np.var(H) * 3 * L ** 3, susc_p=np.var(P) * 3 * L ** 3))
            out[key] = {k: sd_across([r[k] for r in rs]) for k in rs[0]}
            print(key, {k: round(v[0], 4) for k, v in out[key].items()}, flush=True)
            save("map", out)


def stage_sharp():
    """finite-size scaling along K=4: J scan with L=8,12,16; susceptibility of the Higgs link and of the plaquette."""
    path = os.path.join(SCR, "sharp.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for L in (8, 12, 16):
        for J in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
            key = f"L{L}|J{J}"
            if key in out: continue
            wait_pause()
            vals = []
            for sd in SEEDS:
                s = Sim(L, K=4.0, J=J, chi=0.25, T=1.0, use_m=True, seed=sd)
                s.hot_start(0.0); s.thermalise(30, 100)
                H = []; P = []
                for i in range(600):
                    s.step(50); s.refresh()
                    o = s.obs(); H.append(o[1]); P.append(o[0])
                vals.append(dict(h=np.mean(H), sh=np.var(H) * 3 * L ** 3, p=np.mean(P), sp=np.var(P) * 3 * L ** 3))
            out[key] = {k: sd_across([v[k] for v in vals]) for k in vals[0]}
            print(key, {k: round(v[0], 4) for k, v in out[key].items()}, flush=True)
            save("sharp", out)


if __name__ == "__main__":
    globals()["stage_" + sys.argv[1]]()
