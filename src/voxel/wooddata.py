"""WOOD SPECIES FOR SORPTION (wave 14a, lane B): pure data, no imports.

A wood cell is the one material WOOD. When a scene NAMES a species (fluids.FluidsMixin.set_wood),
the cell also carries a species id and a grain axis, and the soak law reads this table in place of
the single generic row. A cell with no species reads fluids.SORPTIVITY[WOOD] exactly as before
(S 1.0 g/m2/s^0.5 on every face, 0.6 g/g held), so nothing changes until a scene asks.

WHAT EACH NUMBER IS
  G     specific gravity, oven-dry mass over volume at 12 % moisture. Wood Handbook FPL-GTR-282,
        ch. 5 (Table 5-3a/b); the repo already holds the oak and white-pine pairs (realism-references-2
        K18: white oak 0.60 green / 0.68 at 12 %, red oak 0.56 / 0.63, eastern white pine 0.34 / 0.35).
        Beech 0.64, white ash 0.60 are the same table (from memory of it, marked in `unverified`);
        balsa 0.16 likewise.
  wmax  the most water a gram of dry wood holds, g/g. DERIVED, not typed: Wood Handbook ch. 4
        (Glass and Zelinka), the maximum moisture content when every void is full,
        Mmax = (1.5 - G) / (1.5 G), 1.5 being the density of cell-wall substance. It reproduces the
        handbook's own ends: 267 % at G 0.30 and 44 % at G 0.90 (repo L19).
  S_along, S_across  the sorptivity of liquid water (i = S sqrt(t), g per m2 of face, per sqrt s)
        into an END face (water travels along the grain) and into a SIDE face (across the grain).
        ANCHORS: Wood Handbook ch. 4 as the repo records it (L19, source 109): softwoods 10-16 along
        and 1-7 across. Pine takes the anchors' middle. The other species are placed inside or
        beside that band by the anatomy the literature names; each number below is `unverified`
        unless it is an anchor. I could not open a per-species table (the Lund review, Wadso 1995,
        says there are few published and that sapwood and heartwood of one pine differ widely).
        Direction of the ordering, which is the sourced part:
          - along the grain is much faster than across it in every species (Wadso 1995; Wood Handbook);
          - heartwood is slower than sapwood (deposits block the pits; Wadso 1995, conclusion 7);
          - white oak heartwood is filled with tyloses, which is why it is the wood of tight casks
            (Wood Handbook, cooperage): the lowest across-grain value of the five;
          - beech has the highest absorption coefficients of beech, robinia and spruce in a published
            comparison (search result, not opened): the highest of the dense woods here;
          - balsa is mostly empty cell space, G 0.16: it takes water fastest (scaled by Mmax, u).
"pine" MEANS SAPWOOD (wave 14a refinement R3). The Wood Handbook softwood bands are for the permeable
wood, heartwood of most pines is far less permeable (Wadso 1995; the multi-phase pine study,
doi:10.1007/s00226-025-01635-9, gives 500x radial-to-axial for one sample whose zone is not stated (u)),
so the row "pine_heart" holds the heartwood: a quarter of the sapwood end-face rate and a quarter of
its side rate (both u). Neither is the source's 500x: that is a permeability, and sorptivity goes as
its square root (about 22x), which the single across-grain number here does not split into radial and
tangential (the grain axis is the only direction a cell carries).
A species is a row. Add a wood by adding a row; nothing else in the engine names one.
"""

# name -> dict(G, S_along, S_across, unverified, note). Index 0 is "no species": the old generic row.
_ROWS = {
    "oak":   dict(G=0.68, S_along=6.0,  S_across=0.3,
                  unverified=("S_along", "S_across"),
                  note="white oak heartwood, the cask wood: tyloses plug the vessels (u numbers)"),
    "pine":  dict(G=0.35, S_along=13.0, S_across=4.0, unverified=("S_across",),
                  note="eastern white pine SAPWOOD (the permeable part of the log, and what a plank of "
                       "young pine is made of): the Wood Handbook softwood bands 10-16 / 1-7, mid values"),
    "beech": dict(G=0.64, S_along=20.0, S_across=5.0,
                  unverified=("G", "S_along", "S_across"),
                  note="diffuse-porous, no tyloses, readily wetted"),
    "ash":   dict(G=0.60, S_along=16.0, S_across=3.0,
                  unverified=("G", "S_along", "S_across"),
                  note="white ash, ring-porous: open earlywood vessels give a fast end face"),
    "balsa": dict(G=0.16, S_along=30.0, S_across=8.0,
                  unverified=("G", "S_along", "S_across"),
                  note="almost all void: G 0.16 (u), so the fastest and the thirstiest"),
    "pine_heart": dict(G=0.35, S_along=3.0, S_across=1.0,
                  unverified=("S_along", "S_across"),
                  note="pine HEARTWOOD: the deposits and aspirated pits make it much slower (Wadso 1995)"),
}
NAMES = ("",) + tuple(_ROWS)                # id 0 = generic wood; ids are append-only
CELL_WALL_G = 1.5                           # g/cm3, Wood Handbook ch. 4


def mmax(G):
    """g of water per g of dry wood when every void is full (Wood Handbook ch. 4)."""
    return (CELL_WALL_G - G) / (CELL_WALL_G * G)


def rho(name):
    """kg/m3 (= g/L) at 12 % moisture: the specific gravity times 1000, oven-dry basis."""
    return 1000.0 * _ROWS[name]["G"]


def row(name):
    """(S_along, S_across, wmax g/g) of a named species."""
    r = _ROWS[name]
    return r["S_along"], r["S_across"], mmax(r["G"])


def species_id(name):
    try:
        return NAMES.index(name)
    except ValueError:
        raise KeyError(f"unknown wood {name!r}; known: {NAMES[1:]}") from None
