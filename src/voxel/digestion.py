"""DIGESTION AND THE BODY'S STORES (wave 13, T7 section 3.9).

Pure functions on plain dicts: no World, no numpy. A person's `stores` is a
dict of floats and lists (so it deep-copies for a snapshot), all float64,
in GRAMS and JOULES. This file knows food, the gut, the fuel tanks, the
kidneys; the heat law (bodyheat.py) calls `step` once a body second and
deposits the heat it returns.

THE ENERGY BOOK (the one rule the user cares about: nothing appears)
  Every food has a GROSS energy (GE_KJ per gram of carbohydrate, protein,
  fat, alcohol, fibre). Digestibility d is the fraction of the GROSS energy
  (and of the mass of that macro) that crosses the gut wall; the rest is
  faeces and carries its chemical energy with it. Absorbed matter goes to a
  store at its gross energy. When protein is burned some chemical energy
  leaves as urine (urea and the rest). So, for a whole meal once it has
  been absorbed and burned:

        GE_in  =  ME  +  faeces_J  +  urine_J         (ME = heat yield)

  THERMIC EFFECT OF FOOD is taken ONCE, at absorption: a share TEF of every
  absorbed gram is burned at once (heat to the caller, O2 used, CO2 made)
  and the rest goes to the store. So stores receive (1 - TEF) x ME and the
  TEF share is heat. It is never also an extra demand on the stores.

  Mass closes the same way: grams burned + O2 breathed in = CO2 breathed
  out + metabolic water + urea. CO2 is found BY THAT CLOSURE, and the
  respiratory quotient falls out of it (carbohydrate 1.00, fat 0.70,
  protein 0.85, alcohol 0.67).

Marks: (u) number from memory of the literature, to be sourced.
"""

KJ_PER_KCAL = 4.184
# O2 and CO2 grams per litre (0 C, 1 atm; the unit physiology.py thinks in).
O2_G_L, CO2_G_L = 1.429, 1.977

# --- gross energy of each macro, kJ per gram (Atwater general factors) ----
GE_KJ = {"carb": 4.1 * KJ_PER_KCAL, "protein": 5.65 * KJ_PER_KCAL,
         "fat": 9.4 * KJ_PER_KCAL, "alcohol": 7.1 * KJ_PER_KCAL,
         "fibre": 4.1 * KJ_PER_KCAL}
MACROS = ("carb", "protein", "fat", "alcohol", "fibre")

# Urinary chemical energy per gram of PROTEIN burned (urea, ammonia, ...).
# Atwater: 5.65 x 0.92 - 1.25 = 3.95 kcal/g metabolisable. (u)
URINE_KJ_PER_G_PROTEIN = 5.7
# Urea mass per gram of protein burned (N 16%, urea 2.14 g per g N).
UREA_G_PER_G_PROTEIN = 0.16 * 2.14

# --- fuels: what each gram of a store becomes when burned ------------------
# kJ of HEAT per gram, kJ per litre of O2 (Zuntz/Lusk), metabolic water g/g.
# CO2 is found by mass closure, not tabled. (u: 0.556 is starch units; the
# spec's 0.60 is glucose; 1.07 fat; 0.41 protein.)
FUEL = {
    "carb": {"kj_g": GE_KJ["carb"], "kj_l": 21.1, "h2o": 0.556, "urea": 0.0},
    "fat": {"kj_g": GE_KJ["fat"], "kj_l": 19.6, "h2o": 1.07, "urea": 0.0},
    "protein": {"kj_g": GE_KJ["protein"] - URINE_KJ_PER_G_PROTEIN,
                "kj_l": 19.3, "h2o": 0.41, "urea": UREA_G_PER_G_PROTEIN},
    # ethanol: C2H5OH + 3 O2 -> 2 CO2 + 3 H2O, so O2 is exact (96/46 g/g)
    "alcohol": {"kj_g": GE_KJ["alcohol"], "kj_l": GE_KJ["alcohol"] * O2_G_L / (96.0 / 46.0),
                "h2o": 54.0 / 46.0, "urea": 0.0},
}
FUEL["fibre"] = FUEL["carb"]

# Thermic effect, per macro as a fraction of the absorbed energy (Westerterp
# 2004: carb 5-10%, protein 20-30%, fat 0-3%, alcohol 10-30%); a mixed meal
# comes to 8-10%. (u)
TEF = {"carb": 0.075, "protein": 0.25, "fat": 0.03, "alcohol": 0.20, "fibre": 0.075}
# Turning surplus carbohydrate into fat (lipogenesis) costs 23% (u).
DNL_COST = 0.23

# --- digestibility (fraction of GROSS energy and mass absorbed) ------------
# (raw, cooked); fibre is fermented to short-chain acids, about 2 kcal/g. (u)
DIGEST = {"carb": (0.60, 0.95), "protein": (0.80, 0.92), "fat": (0.95, 0.95),
          "alcohol": (1.0, 1.0), "fibre": (0.49, 0.49)}
# --- absorption half times from the small gut, minutes (u) -----------------
ABS_T_HALF_MIN = {"carb": 45.0, "protein": 90.0, "fat": 150.0, "alcohol": 20.0,
                  "fibre": 600.0}
WATER_T_HALF_MIN = 12.0

# --- the stomach -----------------------------------------------------------
# Solids: a lag, then a power exponential, remaining = 2^-(((t-lag)/th)^k).
SOLID_LAG_MIN, SOLID_TH_MIN, SOLID_K = 20.0, 70.0, 1.6
FAT_SLOWS = 0.6          # th x (1 + FAT_SLOWS x fat share of the energy) (u)
# Liquids: first order, th 10-20 min, slower the more calories they carry.
LIQ_TH_MIN, LIQ_KCAL_ML_SLOWS = 15.0, 0.4
# Hunt's delivery cap into the duodenum (u). The spec's 2.5 kcal/min would
# put a 700 kcal meal at 140 min and a glass of milk (190 kcal) at 40 min, both
# outside the spec's own windows (80-130 and 10-20 min), so the cap is set
# where a normal meal or glass never meets it (7 kcal/min) and a 3000 kcal feast does.
HUNT_KCAL_MIN = 7.0

# --- the tanks ---------------------------------------------------------------
LIPID_FRAC_ADIPOSE = 0.80     # "15 kg fat" is adipose TISSUE, 80% lipid (u 0.7-0.9)
GLYC_G_PER_KG = 6.5           # whole-body glycogen capacity (u 6-7)
GLUC_TARGET_G = 5.0
LIVER_OUT_G_MIN = 1.0         # glycogen -> glucose (u)
FAT_OX_MAX_G_MIN = 0.5        # fat oxidation cap, about 300 W (u)
ETOH_OX_MAX_G_H = 8.0         # alcohol clearance (u)
PROT_OX_MAX_G_MIN = 0.25      # the most protein the body will burn (gluconeogenesis), about 75 W (u)
PMAX_W_PER_KG = 14.0          # about 11 MET, the crossover reference (u)
PROTEIN_SHARE_FED, PROTEIN_SHARE_FAST = 0.10, 0.06
TBW_FRAC = {"male": 0.60, "female": 0.50}
BLADDER_COMFORT_ML, BLADDER_MAX_ML = 600.0, 700.0
OBLIGATORY_URINE_ML_DAY = 500.0
FREE_URINE_ML_DAY = 1000.0    # on top, when hydrated

# Foods: grams per 100 g of macro, fibre and water. Edible names are data.
FOOD_ROWS = {
    "water": dict(carb=0, protein=0, fat=0, alcohol=0, fibre=0, water=100, liquid=True, cook=1.0),
    "bread": dict(carb=49, protein=9, fat=3, alcohol=0, fibre=3, water=36, liquid=False, cook=1.0),
    "rice": dict(carb=28, protein=2.7, fat=0.3, alcohol=0, fibre=0.4, water=68, liquid=False, cook=1.0),
    "raw_flour": dict(carb=72.5, protein=11, fat=1.5, alcohol=0, fibre=3, water=12, liquid=False, cook=0.0),
    "meat": dict(carb=0, protein=26, fat=15, alcohol=0, fibre=0, water=58, liquid=False, cook=1.0),
    "raw_meat": dict(carb=0, protein=26, fat=15, alcohol=0, fibre=0, water=58, liquid=False, cook=0.0),
    "apple": dict(carb=11.4, protein=0.3, fat=0.2, alcohol=0, fibre=2.4, water=85.6, liquid=False, cook=1.0),
    "stew": dict(carb=8, protein=7, fat=5, alcohol=0, fibre=1.5, water=78, liquid=False, cook=1.0),
    "milk": dict(carb=4.8, protein=3.4, fat=3.3, alcohol=0, fibre=0, water=88, liquid=True, cook=1.0),
    "juice": dict(carb=10.5, protein=0.5, fat=0.1, alcohol=0, fibre=0.2, water=88.5, liquid=True, cook=1.0),
    "ale": dict(carb=3.5, protein=0.5, fat=0, alcohol=3.9, fibre=0, water=92, liquid=True, cook=1.0),
    "oil": dict(carb=0, protein=0, fat=100, alcohol=0, fibre=0, water=0, liquid=False, cook=1.0),
}


def food(name, mass_g, cook=None):
    """A food record of `mass_g` grams: grams of each macro and of water."""
    r = FOOD_ROWS[name]
    f = mass_g / 100.0
    rec = {"name": name, "mass_g": float(mass_g), "liquid": bool(r["liquid"]),
           "cook": float(r["cook"] if cook is None else cook),
           "water_g": r["water"] * f}
    for m in MACROS:
        rec[m] = r[m] * f
    return rec


def food_ge_kj(rec):
    """Gross energy of a food record, kJ."""
    return sum(rec.get(m, 0.0) * GE_KJ[m] for m in MACROS)


def digestibility(macro, cook):
    raw, cooked = DIGEST[macro]
    return raw + (cooked - raw) * min(max(cook, 0.0), 1.0)


def new_stores(mass_kg, fat_frac, sex="male", full=1.0):
    """The tanks of a fed adult. `full` scales glycogen (0 = empty)."""
    lean = mass_kg * (1.0 - fat_frac)
    return {
        "glyc_g": GLYC_G_PER_KG * mass_kg * 0.9 * full,
        "gluc_g": GLUC_TARGET_G,
        "fat_g": fat_frac * mass_kg * 1000.0 * LIPID_FRAC_ADIPOSE,
        "lean_g": 0.19 * lean * 1000.0,           # protein of the lean tissue
        "etoh_g": 0.0,
        "water_ml": TBW_FRAC[sex] * mass_kg * 1000.0,
        "water0_ml": TBW_FRAC[sex] * mass_kg * 1000.0,
        "mass0_kg": float(mass_kg),
        "glyc_cap_g": GLYC_G_PER_KG * mass_kg,
        "bladder_ml": 100.0, "urea_g": 0.0, "urine_J": 0.0,
        "colon_g": 0.0, "colon_J": 0.0,
        "gut": [], "si": {"water": 0.0}, "nutr": {},
        "t_empty_s": 0.0, "leak_ml": 0.0,
        "books": {"GE_in": 0.0, "faeces_J": 0.0, "urine_J": 0.0, "tef_J": 0.0,
                  "heat_J": 0.0, "dnl_J": 0.0},
    }


# ---------------------------------------------------------------- the gut --
def ingest(st, rec):
    """Put a food record into the stomach. Returns the mass eaten (g) so the
    food thing can leave the world through its own ledger."""
    ge = food_ge_kj(rec)
    fat_share = rec.get("fat", 0.0) * GE_KJ["fat"] / ge if ge > 0 else 0.0
    liquid = bool(rec.get("liquid", False))
    if liquid:
        kcal_ml = ge / KJ_PER_KCAL / max(rec["mass_g"], 1e-9)
        lag, th, k = 0.0, LIQ_TH_MIN * (1.0 + LIQ_KCAL_ML_SLOWS * kcal_ml) * 60.0, 1.0
    else:
        lag, th, k = (SOLID_LAG_MIN * 60.0,
                      SOLID_TH_MIN * 60.0 * (1.0 + FAT_SLOWS * fat_share), SOLID_K)
    cook = rec.get("cook", 1.0)
    item = {"name": rec.get("name", "food"), "liquid": liquid,
            "m0": {m: float(rec.get(m, 0.0)) for m in MACROS},
            "water0": float(rec.get("water_g", 0.0)),
            "d": {m: digestibility(m, cook) for m in MACROS},
            "lag": lag, "th": th, "k": k, "tau": 0.0, "rem": 1.0}
    st["gut"].append(item)
    # whatever is not macro or water is ash: no energy, and it goes straight
    # to the colon, so the mass we are handed is the mass we hold
    st["colon_g"] += max(rec["mass_g"] - sum(item["m0"].values()) - item["water0"], 0.0)
    st["books"]["GE_in"] += ge * 1000.0
    st["t_empty_s"] = 0.0
    return float(rec["mass_g"])


def _remaining(item, tau):
    if tau <= item["lag"]:
        return 1.0
    return 2.0 ** (-(((tau - item["lag"]) / item["th"]) ** item["k"]))


def stomach_kcal(st):
    """Gross kcal still in the stomach."""
    return sum(sum(it["m0"][m] * GE_KJ[m] for m in MACROS) * it["rem"]
               for it in st["gut"]) / KJ_PER_KCAL


def _energy_of(item):
    return sum(item["m0"][m] * GE_KJ[m] for m in MACROS) * 1000.0     # J


def _empty(st, dt):
    """Advance the stomach by dt. Emptied matter goes to the small gut
    (absorbable share) and the colon (the rest)."""
    items = st["gut"]
    if not items:
        return
    want = sum(_energy_of(it) * (it["rem"] - _remaining(it, it["tau"] + dt))
               for it in items)
    cap = HUNT_KCAL_MIN * KJ_PER_KCAL * 1000.0 / 60.0 * dt            # J this step
    speed = 1.0
    if want > cap > 0.0:                          # the pylorus is the limit
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            got = sum(_energy_of(it) * (it["rem"] - _remaining(it, it["tau"] + mid * dt))
                      for it in items)
            lo, hi = (mid, hi) if got <= cap else (lo, mid)
        speed = lo
    si = st["si"]
    for it in items:
        new = _remaining(it, it["tau"] + speed * dt)
        frac = it["rem"] - new
        it["tau"] += speed * dt
        it["rem"] = new
        for m in MACROS:
            g = it["m0"][m] * frac
            d = it["d"][m]
            si[m] = si.get(m, 0.0) + g * d
            st["colon_g"] += g * (1.0 - d)
            st["colon_J"] += g * (1.0 - d) * GE_KJ[m] * 1000.0
            st["books"]["faeces_J"] += g * (1.0 - d) * GE_KJ[m] * 1000.0
        si["water"] += it["water0"] * frac
    st["gut"] = [it for it in items if it["rem"] > 1e-12]
    for it in items:                                # dust: hand over the last crumb
        if it["rem"] <= 1e-12 and it["rem"] > 0.0:
            for m in MACROS:
                g = it["m0"][m] * it["rem"]
                si[m] = si.get(m, 0.0) + g * it["d"][m]
                st["colon_g"] += g * (1.0 - it["d"][m])
                st["colon_J"] += g * (1.0 - it["d"][m]) * GE_KJ[m] * 1000.0
                st["books"]["faeces_J"] += g * (1.0 - it["d"][m]) * GE_KJ[m] * 1000.0
            si["water"] += it["water0"] * it["rem"]


def _oxidise(st, fuel, grams, out):
    """Burn `grams` of a fuel that has ALREADY left its tank. Adds to `out`
    the heat, O2, CO2 and water, and books the urea and the urine energy."""
    if grams <= 0.0:
        return 0.0
    f = FUEL[fuel]
    heat = grams * f["kj_g"] * 1000.0
    o2_l = heat / 1000.0 / f["kj_l"]
    o2_g = o2_l * O2_G_L
    h2o = grams * f["h2o"]
    urea = grams * f["urea"]
    co2_g = grams + o2_g - h2o - urea
    out["heat_J"] += heat
    out["o2_L"] += o2_l
    out["co2_L"] += co2_g / CO2_G_L
    out["co2_g"] += co2_g
    out["o2_g"] += o2_g
    st["water_ml"] += h2o
    out["h2o_ml"] += h2o
    if urea > 0.0:
        st["urea_g"] += urea
        uj = grams * URINE_KJ_PER_G_PROTEIN * 1000.0
        st["urine_J"] += uj
        st["books"]["urine_J"] += uj
    return heat


def _absorb(st, dt, out):
    """First-order absorption from the small gut; the TEF share burns at once."""
    si = st["si"]
    for m in MACROS:
        g = si.get(m, 0.0)
        if g <= 0.0:
            continue
        a = g * (1.0 - 2.0 ** (-dt / (ABS_T_HALF_MIN[m] * 60.0)))
        si[m] = g - a
        burn = a * TEF[m]
        fuel = {"carb": "carb", "fibre": "carb", "protein": "protein",
                "fat": "fat", "alcohol": "alcohol"}[m]
        h = _oxidise(st, fuel, burn, out)
        st["books"]["tef_J"] += h
        out["tef_J"] += h
        keep = a - burn
        if m in ("carb", "fibre"):
            _store_carb(st, keep, out)
        elif m == "protein":
            st["lean_g"] += keep
        elif m == "fat":
            st["fat_g"] += keep
        else:
            st["etoh_g"] += keep
    w = si["water"]
    if w > 0.0:
        a = w * (1.0 - 2.0 ** (-dt / (WATER_T_HALF_MIN * 60.0)))
        si["water"] = w - a
        st["water_ml"] += a


def _store_carb(st, g, out):
    """Carbohydrate refills the glucose pool, then glycogen; only the surplus
    becomes fat (at a cost that is heat and a little CO2)."""
    room = max(GLUC_TARGET_G - st["gluc_g"], 0.0)
    take = min(g, room)
    st["gluc_g"] += take
    g -= take
    room = max(st["glyc_cap_g"] - st["glyc_g"], 0.0)
    take = min(g, room)
    st["glyc_g"] += take
    g -= take
    if g > 0.0:
        e = g * GE_KJ["carb"] * 1000.0
        fat = e * (1.0 - DNL_COST) / (GE_KJ["fat"] * 1000.0)
        heat = e * DNL_COST
        st["fat_g"] += fat
        out["heat_J"] += heat
        out["tef_J"] += heat
        st["books"]["tef_J"] += heat
        st["books"]["dnl_J"] += heat
        # mass that leaves: carbohydrate in, lipid kept; the difference is
        # breathed out as CO2 (and no O2 comes in) -- mass closes.
        out["co2_g"] += g - fat
        out["co2_L"] += (g - fat) / CO2_G_L


def fuel_split(P_w, mass_kg, glyc_g, gluc_g, fed=True, pmax_w=None):
    """Joule shares of one second of demand: (carb, fat, protein) fractions
    BEFORE the tanks and the fat ceiling are looked at. Carbohydrate rises
    with relative intensity (crossover, Brooks and Mercier 1994, u)."""
    pmax = pmax_w if pmax_w is not None else PMAX_W_PER_KG * mass_kg
    inten = min(max(P_w / pmax, 0.0), 1.5)
    f_c = min(max(0.35 + 0.65 * inten, 0.0), 1.0)
    prot = PROTEIN_SHARE_FED if fed else PROTEIN_SHARE_FAST
    return f_c * (1.0 - prot), (1.0 - f_c) * (1.0 - prot), prot


def _burn(st, demand_J, dt, mass_kg, out, pmax_w=None):
    """Pay `demand_J` out of the tanks. Fills out['delivered_J']."""
    if demand_J <= 0.0:
        return
    # the liver tops up circulating glucose from glycogen
    need = max(GLUC_TARGET_G - st["gluc_g"], 0.0)
    mv = min(need, st["glyc_g"], LIVER_OUT_G_MIN / 60.0 * dt)
    st["gluc_g"] += mv
    st["glyc_g"] -= mv
    J = demand_J
    # alcohol is cleared first, at its own limit
    e_g = min(st["etoh_g"], ETOH_OX_MAX_G_H / 3600.0 * dt, J / (FUEL["alcohol"]["kj_g"] * 1000.0))
    st["etoh_g"] -= e_g
    J -= _oxidise(st, "alcohol", e_g, out)
    if J < 0.0:
        J = 0.0
    fed = st["glyc_g"] > 0.1 * st["glyc_cap_g"]
    fc, ff, fp = fuel_split(demand_J / dt, mass_kg, st["glyc_g"], st["gluc_g"], fed, pmax_w)
    cj = FUEL["carb"]["kj_g"] * 1000.0
    fj = FUEL["fat"]["kj_g"] * 1000.0
    pj = FUEL["protein"]["kj_g"] * 1000.0
    fat_cap = FAT_OX_MAX_G_MIN / 60.0 * dt * fj
    carb_have = (st["glyc_g"] + st["gluc_g"]) * cj
    prot_have = st["lean_g"] * pj
    prot_cap = PROT_OX_MAX_G_MIN / 60.0 * dt * pj
    j_p = min(J * fp, prot_have, prot_cap)           # protein first: it has the lowest ceiling
    rest = J - j_p
    j_c = rest * fc / max(fc + ff, 1e-12)
    j_f = rest - j_c
    # fat above its ceiling goes to carbohydrate; carbohydrate that is not
    # there goes to fat, up to the ceiling; what neither can pay is unmet
    j_f2 = min(j_f, fat_cap, st["fat_g"] * fj)
    j_c += j_f - j_f2
    j_c2 = min(j_c, carb_have)
    short = j_c - j_c2
    room_f = min(fat_cap, st["fat_g"] * fj) - j_f2
    take_f = min(short, max(room_f, 0.0))
    j_f2 += take_f
    short -= take_f
    # protein covers what is left (gluconeogenesis and starvation)
    extra_p = min(short, max(min(prot_have, prot_cap) - j_p, 0.0))
    j_p += extra_p
    short -= extra_p
    delivered = 0.0
    g = j_c2 / cj
    use = min(g, st["gluc_g"])
    st["gluc_g"] -= use
    st["glyc_g"] -= g - use
    st["glyc_g"] = max(st["glyc_g"], 0.0)
    delivered += _oxidise(st, "carb", g, out)
    gf = j_f2 / fj
    st["fat_g"] -= gf
    delivered += _oxidise(st, "fat", gf, out)
    gp = j_p / pj
    st["lean_g"] -= gp
    delivered += _oxidise(st, "protein", gp, out)
    out["delivered_J"] += delivered + (demand_J - J)
    out["unmet_J"] = max(short, 0.0)


def _kidneys(st, dt):
    """Water out of the body into the bladder at a rate that follows hydration."""
    w0 = st["water0_ml"]
    s = min(max(1.0 + (st["water_ml"] - w0) / (0.02 * w0), 0.0), 4.0)
    ml = (OBLIGATORY_URINE_ML_DAY + FREE_URINE_ML_DAY * s) / 86400.0 * dt
    ml = min(ml, max(st["water_ml"], 0.0))
    st["water_ml"] -= ml
    st["bladder_ml"] += ml
    if st["bladder_ml"] > BLADDER_MAX_ML:           # never grows without bound
        extra = st["bladder_ml"] - BLADDER_MAX_ML
        st["bladder_ml"] = BLADDER_MAX_ML
        st["leak_ml"] += extra


def step(st, dt, demand_W, mass_kg=None, pmax_w=None, water_loss_ml=0.0):
    """One call = `dt` seconds of the whole digestive and metabolic life.

    `demand_W` is P_total of energetics.py. `water_loss_ml` is what the
    heat law's skin and breath took (it is taken here, from the tank).
    Returns a dict: delivered_J (paid to the demand), tef_J (extra heat from
    digestion, to the torso), heat_J (all heat made here), o2_L, co2_L,
    co2_g, o2_g, h2o_ml, unmet_J, capacity (0..1 share of the demand the
    tanks could pay), signals.
    """
    mass_kg = st["mass0_kg"] if mass_kg is None else mass_kg
    out = {"delivered_J": 0.0, "tef_J": 0.0, "heat_J": 0.0, "o2_L": 0.0, "co2_L": 0.0,
           "co2_g": 0.0, "o2_g": 0.0, "h2o_ml": 0.0, "unmet_J": 0.0}
    _empty(st, dt)
    _absorb(st, dt, out)
    _burn(st, max(demand_W, 0.0) * dt, dt, mass_kg, out, pmax_w)
    st["water_ml"] -= water_loss_ml
    _kidneys(st, dt)
    if not st["gut"] and not any(st["si"].get(m, 0.0) > 1e-9 for m in MACROS):
        st["t_empty_s"] += dt
    books = st["books"]
    books["heat_J"] += out["heat_J"]
    dem = max(demand_W, 0.0) * dt
    out["capacity"] = 1.0 if dem <= 0 else max(0.0, 1.0 - out["unmet_J"] / dem)
    out["signals"] = signals(st, mass_kg)
    return out


# -------------------------------------------------------------- the books --
def stores_J(st):
    """Chemical energy (J) held in the body's tanks, gut, colon and urine:
    the 'stores' term of total_energy."""
    e = (st["glyc_g"] + st["gluc_g"]) * GE_KJ["carb"] + st["fat_g"] * GE_KJ["fat"] \
        + st["lean_g"] * GE_KJ["protein"] + st["etoh_g"] * GE_KJ["alcohol"]
    e *= 1000.0
    for it in st["gut"]:
        e += _energy_of(it) * it["rem"]
    for m in MACROS:
        e += st["si"].get(m, 0.0) * GE_KJ[m] * 1000.0
    return e + st["colon_J"] + st["urine_J"]


def stores_g(st):
    """Mass (g) held in the tanks, gut, water, bladder and colon."""
    g = st["glyc_g"] + st["gluc_g"] + st["fat_g"] + st["lean_g"] + st["etoh_g"] \
        + st["water_ml"] + st["bladder_ml"] + st["urea_g"] + st["colon_g"]
    for it in st["gut"]:
        g += (sum(it["m0"].values()) + it["water0"]) * it["rem"]
    g += sum(st["si"].get(m, 0.0) for m in MACROS) + st["si"]["water"]
    return g


def meal_split(rec):
    """Closed-form energy split of one food (J): GE, faeces, urine (if the
    absorbed protein is all burned), ME, TEF heat. GE = ME + faeces + urine."""
    cook = rec.get("cook", 1.0)
    ge = faeces = urine = tef = 0.0
    for m in MACROS:
        g = rec.get(m, 0.0)
        d = digestibility(m, cook)
        ge += g * GE_KJ[m] * 1000.0
        faeces += g * (1.0 - d) * GE_KJ[m] * 1000.0
        if m == "protein":
            fuel_e = g * d * URINE_KJ_PER_G_PROTEIN * 1000.0
            urine += fuel_e
        tef += g * d * TEF[m] * FUEL[{"fibre": "carb"}.get(m, m)]["kj_g"] * 1000.0
    return {"GE": ge, "faeces": faeces, "urine": urine, "ME": ge - faeces - urine,
            "TEF": tef}


def weir_kj_per_l(rq):
    """Weir (1949) without the protein term, kJ per litre of O2."""
    return 16.318 + 4.602 * rq


# ---------------------------------------------------------------- signals --
def deficit_pct(st, mass_kg=None):
    """Body water lost, percent of body mass."""
    mass_kg = st["mass0_kg"] if mass_kg is None else mass_kg
    return max(st["water0_ml"] - st["water_ml"], 0.0) / (mass_kg * 1000.0) * 100.0


def thirst(st, mass_kg=None):
    return min(max((deficit_pct(st, mass_kg) - 0.5) / 2.0, 0.0), 1.0)


def hunger(st):
    """0..1 from the glycogen share, the circulating glucose and the time
    since the gut was last busy (all numbers (u))."""
    share = st["glyc_g"] / max(st["glyc_cap_g"], 1e-9)
    low = min(max((0.8 - share) / 0.7, 0.0), 1.0)
    glu = min(max((GLUC_TARGET_G - st["gluc_g"]) / GLUC_TARGET_G, 0.0), 1.0)
    idle = min(st["t_empty_s"] / (5.0 * 3600.0), 1.0)
    return min(max(0.55 * low + 0.25 * glu + 0.2 * idle * low + 0.1 * idle, 0.0), 1.0)


def signals(st, mass_kg=None):
    return {"thirst": thirst(st, mass_kg), "hunger": hunger(st),
            "bladder": min(st["bladder_ml"] / BLADDER_COMFORT_ML, 1.5),
            "glyc_share": st["glyc_g"] / max(st["glyc_cap_g"], 1e-9)}


def void_bladder(st):
    """The act of passing water: returns (ml, urea_g, chemical_J) that leave
    the body and become matter on the floor (the pour topic places it)."""
    ml, urea, j = st["bladder_ml"], st["urea_g"], st["urine_J"]
    st["bladder_ml"] = st["urea_g"] = st["urine_J"] = 0.0
    return ml, urea, j


def void_colon(st):
    g, j = st["colon_g"], st["colon_J"]
    st["colon_g"] = st["colon_J"] = 0.0
    return g, j


def rq_of(out):
    return out["co2_L"] / out["o2_L"] if out["o2_L"] > 0 else 0.0
