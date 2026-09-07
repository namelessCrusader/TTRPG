# two_looters_race
_A RACE for the starfire vial (id=vial): you AND Sly the cutpurse both want it; Bran the doorman guards. Approach and CLAIM it with a guarded take — but Sly acts on his own perception and may beat you (precedence + the possession-guard decide the winner)._

### slice t0
```
@·a···
·····T
G·····
·····$
```
_@=you  T=Sly the cutpurse  G=Bran the doorman  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(0,2) — «I step toward the vault door to keep watch.»
    grammar: aims at **the cell to the S**; **certain** → Bran the doorman moves south
    → ✓ **done** → Bran the doorman moves south
    · sees: cask_water d2, player d2
- **Sly the cutpurse** @(5,1) — «I slip south along the wall toward the glint of starfire.»
    grammar: aims at **the cell to the S**; **certain** → Sly the cutpurse moves south
    → ✓ **done** → Sly the cutpurse moves south
    · sees: keg_oil d1, barrel_oil d2, vial d2
- **you** @(0,0) — «I head south into the vault to find the vial.»
    grammar: aims at **the cell to the S**; **certain** → you moves south
    → ✓ **done** → you moves south
    · sees: apprentice d2, cask_water d2, guard d2

### slice t1
```
··a···
@·····
·····T
G····$
```
_@=you  T=Sly the cutpurse  G=Bran the doorman  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(0,3) — «I move east along the vault to head off the cutpurse.»
    grammar: aims at **the cell to the E**; **certain** → Bran the doorman moves east
    → ✓ **done** → Bran the doorman moves east
    · sees: player d2, cask_water d3
- **Sly the cutpurse** @(5,2) — «I lunge for the vial and snatch it up before anyone stops me.»
    grammar: aims at **starfire vial** — only if not (vial is taken); **certain** → (malformed None); (malformed None)
    → ✗ **REFUSED** — DSLError: bad effect {'set_tag': ['vial', 'taken']}
    · sees: vial d1, keg_oil d2, barrel_oil d3
- **you** @(0,1) — «I keep pushing south into the vault after the vial.»
    grammar: aims at **the cell to the S**; **certain** → you moves south
    → ✓ **done** → you moves south
    · sees: cask_water d1, guard d2, apprentice d3, barrel_oil d3

### slice t2
```
··a···
······
@····T
·G···$
```
_@=you  T=Sly the cutpurse  G=Bran the doorman  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(1,3) — «I move east along the vault to head off the cutpurse.»
    grammar: aims at **the cell to the E**; **certain** → Bran the doorman moves east
    → ✓ **done** → Bran the doorman moves east
    · sees: cask_water d2, player d2
- **Sly the cutpurse** @(5,2) — «I lunge for the vial and snatch it up before anyone stops me.»
    grammar: aims at **starfire vial** — only if not (the target is taken); **certain** → the target becomes taken; Sly the cutpurse becomes has_prize
    → ✓ **done** → the target becomes taken; Sly the cutpurse becomes has_prize
    · sees: vial d1, keg_oil d2, barrel_oil d3
- **you** @(0,2) — «I keep pushing south into the vault after the vial.»
    grammar: aims at **the cell to the S**; **certain** → you moves south
    → ✓ **done** → you moves south
    · sees: cask_water d2, guard d2, apprentice d4~1t, barrel_oil d4~1t

### slice t3
```
··a···
······
·····T
@·G··$
```
_@=you  T=Sly the cutpurse  G=Bran the doorman  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(2,3) — «I continue east along the vault, keeping watch.»
    grammar: aims at **the cell to the E**; **certain** → Bran the doorman moves east
    → ✓ **done** → Bran the doorman moves east
    · sees: player d2, apprentice d3, barrel_oil d3, cask_water d3
- **Sly the cutpurse** @(5,2) — «I've got it — I slip north away from the door before anyone notices.»
    grammar: aims at **the cell to the N**; **certain** → Sly the cutpurse moves north
    → ✓ **done** → Sly the cutpurse moves north
    · sees: vial d1, keg_oil d2, barrel_oil d3
- **you** @(0,3) — «I press east through the vault, still hunting for the vial.»
    grammar: aims at **the cell to the E**; **certain** → you moves east
    → ✓ **done** → you moves east
    · sees: guard d2, cask_water d3, apprentice d5~2t, barrel_oil d5~2t
- **!! INVARIANTS BROKEN:** ['guard is inside a solid wood block at (3, 3, 0)']

### slice t4
```
··a···
·····T
······
·@·G·$
```
_@=you  T=Sly the cutpurse  G=Bran the doorman  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(3,3) — «The vial's already been taken — I push east to find who did it.»
    grammar: aims at **the cell to the E**; **certain** → Bran the doorman moves east
    → ✓ **done** → Bran the doorman moves east
    · sees: barrel_oil d2, player d2, vial d2, keg_oil d3, apprentice d4~1t, cask_water d4~1t
- **Sly the cutpurse** @(5,1) — «I put more distance between me and the vault, vial in hand.»
    grammar: aims at **the cell to the N**; **certain** → Sly the cutpurse moves north
    → ✓ **done** → Sly the cutpurse moves north
    · sees: keg_oil d1, barrel_oil d2, vial d2
- **you** @(1,3) — «I hurry east to catch up with Bran and see what's going on.»
    grammar: aims at **the cell to the E**; **certain** → you moves east
    → ✓ **done** → you moves east
    · sees: cask_water d2, guard d2, apprentice d4~3t, barrel_oil d4~3t
