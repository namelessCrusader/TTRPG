# dungeon_press
_Loot the floor while TWO hostiles close in. Grab boxes, then fight or flee as the scuttler and broodmother arrive — they hunt on their own perception each beat, and they start close enough to reach you._

### slice t0
```
@········
··SB·····
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(3,1) — «I ooze after the scuttler's lead, sensing prey nearby.»
    grammar: aims at **a chittering scuttler**; **certain** → the pale broodmother moves west
    → ✓ **done** → the pale broodmother moves west
    · sees: scuttler d1
- **a chittering scuttler** @(2,1) — «I scuttle toward the intruder, claws clicking on stone.»
    grammar: aims at **you**; **certain** → a chittering scuttler moves west
    → ✓ **done** → a chittering scuttler moves west
    · sees: broodmother d1, box1 d3, coin d3, player d3
- **you** @(0,0) — «I push south into the room, hunting for loot before those two reach me.»
    grammar: aims at **the cell to the S**; **certain** → you moves south
    → ✓ **done** → you moves south
    · sees: scuttler d3

### slice t1
```
·········
@SB······
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(2,1) — «I close the last stretch toward the intruder, hungry.»
    grammar: aims at **you**; **certain** → the pale broodmother moves west
    → ✓ **done** → the pale broodmother moves west
    · sees: scuttler d1, player d2, box1 d3, coin d3
- **a chittering scuttler** @(1,1) — «I snap my claws at the intruder before it can flee.»
    grammar: aims at **you**; **contest** a chittering scuttler's composure vs (55 + 0)  [win → the target's hp changes by -3 · lose → a chittering scuttler's composure changes by -5]
    → ✗ **lose** → a chittering scuttler's composure changes by -5
    · sees: broodmother d1, player d1, box1 d2, coin d2
- **you** @(0,1) — «I lash out at the scuttler as it closes into strike range.»
    grammar: aims at **a chittering scuttler**; **contest** (55 + 0) vs the target's composure  [win → the target's hp changes by (-6 if the target is afraid else -3); the target becomes wounded · lose → you becomes off_balance]
    → ✓ **win** → the target's hp changes by (-6 if the target is afraid else -3); the target becomes wounded
    · sees: scuttler d1, broodmother d2, box1 d3, coin d3

### slice t2
```
·········
@B·······
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(1,1) — «I strike at the intruder now that I've closed the gap.»
    grammar: aims at **you**; **contest** the pale broodmother's composure vs (55 + 0)  [win → the target's hp changes by (-6 if the target is afraid else -4) · lose → the pale broodmother's composure changes by -5]
    → ✓ **win** → the target's hp changes by (-6 if the target is afraid else -4)
    · sees: scuttler d0, player d1, box1 d2, coin d2
- **a chittering scuttler** @(1,1) — «Wounded, I snap at the intruder anyway, cornered.»
    grammar: aims at **you**; **contest** a chittering scuttler's composure vs (55 + 0)  [win → the target's hp changes by -3 · lose → a chittering scuttler's composure changes by -5]
    → ✗ **lose** → a chittering scuttler's composure changes by -5
    · sees: broodmother d0, player d1, box1 d2, coin d2
- **you** @(0,1) — «I press the attack on the wounded scuttler, trying to put it down before the broodmother lands a hit.»
    grammar: aims at **a chittering scuttler**; **contest** (55 + 0) vs the target's composure  [win → the target's hp changes by (-6 if the target is afraid else -3); the target becomes wounded · lose → you becomes off_balance]
    → ✗ **lose** → you becomes off_balance
    · sees: broodmother d1, scuttler d1, box1 d3, coin d3

### slice t3
```
·········
@B·······
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(1,1) — «I press my advantage on the wounded, retreating prey.»
    grammar: aims at **you**; **contest** the pale broodmother's composure vs (55 + 0)  [win → the target's hp changes by (-6 if the target is afraid else -4) · lose → the pale broodmother's composure changes by -5]
    → ✓ **win** → the target's hp changes by (-6 if the target is afraid else -4)
    · sees: scuttler d0, player d1, box1 d2, coin d2
- **a chittering scuttler** @(1,1) — «Wounded and desperate, I keep clawing at the retreating intruder.»
    grammar: aims at **you**; **contest** a chittering scuttler's composure vs (55 + 0)  [win → the target's hp changes by -3 · lose → a chittering scuttler's composure changes by -5]
    → ✓ **win** → the target's hp changes by -3
    · sees: broodmother d0, player d1, box1 d2, coin d2
- **you** @(0,1) — «Bleeding and off-balance, I break off and stagger south, putting distance between me and their claws while angling toward that box.»
    grammar: aims at **the cell to the S**; **certain** → you moves south
    → ✓ **done** → you moves south
    · sees: broodmother d1, scuttler d1, box1 d3, coin d3

### slice t4
```
·········
·B·······
@········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(1,1) — «I settle over the kill, done hunting.»
    grammar: aims at **yourself**; **certain** → nothing happens
    → ✓ **done** → nothing happens  _(no state change)_
    · sees: scuttler d0, box1 d2, coin d2, player d2
- **a chittering scuttler** @(1,1) — «The intruder stopped moving. I back off, wary, composure spent.»
    grammar: aims at **yourself**; **certain** → nothing happens
    → ✓ **done** → nothing happens  _(no state change)_
    · sees: broodmother d0, box1 d2, coin d2, player d2
- **you** @(0,2) — «I lie still, gone.»
    grammar: aims at **yourself**; **certain** → nothing happens
    → ✓ **done** → nothing happens  _(no state change)_
    · sees: box1 d2, broodmother d2, coin d2, scuttler d2
