# vault_ablaze
_FIRE PRESSURE: smash oil and set the vault alight, then grab a prize and ESCAPE while the fire SPREADS on its own and Bran & Pip react and flee. Physics is the antagonist._

### slice t0
```
@·A···
·····t
G·····
·····$
```
_@=you  G=Bran the doorman  A=Pip the apprentice  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(0,2) — «I stand watch at the vault door, senses sharp for anything out of place.»
    grammar: aims at **yourself**; **certain** → Bran the doorman becomes watchful
    → ✓ **done** → Bran the doorman becomes watchful
    · sees: cask_water d2, player d2
- **Pip the apprentice** @(2,0) — «I keep sorting the oil ledgers at the workbench, unaware anything is wrong.»
    grammar: aims at **yourself**; **certain** → Pip the apprentice becomes busy
    → ✓ **done** → Pip the apprentice becomes busy
    · sees: barrel_oil d2, cask_water d2, player d2, keg_oil d3
- **you** @(0,0) — «I smash the oil barrel with my torch haft, staving in its side so oil gushes across the stone.»
    grammar: aims at **barrel of oil**; **certain** → the target's hp changes by -10; the target becomes smashed; 800ml of oil pools at the ground under the target
    → ✓ **done** → the target's hp changes by -10; the target becomes smashed; 800ml of oil pools at the ground under the target
    · sees: apprentice d2, cask_water d2, guard d2

### slice t1
```
@·A~··
··~~~t
G··~··
·····$
```
_@=you  G=Bran the doorman  A=Pip the apprentice  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(0,2) — «I smell oil where there shouldn't be any and shout the alarm.»
    grammar: aims at **yourself**; **certain** → the words "Oil on the floor — that's no accident! Sound the alarm!"
    → ✓ **done** → the words "Oil on the floor — that's no accident! Sound the alarm!"  _(no state change)_
    · sees: cask_water d2, player d2
- **Pip the apprentice** @(2,0) — «I see the barrel smashed and oil flooding the tiles and cry out in alarm.»
    grammar: aims at **yourself**; **certain** → the words "The oil barrel's stove in — it's everywhere! Something's very wrong!"
    → ✓ **done** → the words "The oil barrel's stove in — it's everywhere! Something's very wrong!"  _(no state change)_
    · sees: barrel_oil d2, cask_water d2, player d2, keg_oil d3
- **you** @(0,0) — «I touch my torch to the spreading oil, and the barrel's tile catches with a whump.»
    grammar: aims at **barrel of oil** — only if the ground under the target's oil > 0; **certain** → the ground under the target heats by 150; the ground under the target bursts to flame
    → ✓ **done** → the ground under the target heats by 150; the ground under the target bursts to flame
    · sees: apprentice d2, cask_water d2, guard d2
- _⏳ meanwhile:_ [t1] Bran the doorman: "Oil on the floor — that's no accident! Sound the alarm!"; [t1] Pip the apprentice: "The oil barrel's stove in — it's everywhere! Something's very wrong!"; [t2] a lick of flame runs across the pooled fuel at (3, 1, 0) and it goes up with a whump!

### slice t2
```
@·A~··
··~*~t
G··~··
·····$
```
_@=you  G=Bran the doorman  A=Pip the apprentice  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(0,2) — «The whump of flame sends me a step back before I catch myself.»
    grammar: aims at **the cell to the S**; **certain** → Bran the doorman moves south; the words "Fire in the vault! Everyone out, NOW!"
    → ✓ **done** → Bran the doorman moves south; the words "Fire in the vault! Everyone out, NOW!"
    · sees: cask_water d2, player d2
- **Pip the apprentice** @(2,0) — «I bolt from the workbench, putting the wall between me and the flame.»
    grammar: aims at **the cell to the W**; **certain** → Pip the apprentice moves west; the words "It's spreading— I'm not staying here!"
    → ✓ **done** → Pip the apprentice moves west; the words "It's spreading— I'm not staying here!"
    · sees: barrel_oil d2, cask_water d2, player d2, keg_oil d3, fire@(3,1)
- **you** @(0,0) — «I push deeper into the vault, torch raised, chasing the glint of the prize past the spreading flame.»
    grammar: aims at **the cell to the E**; **certain** → you moves east
    → ✓ **done** → you moves east
    · sees: apprentice d2, cask_water d2, guard d2
- _⏳ meanwhile:_ [t2] Bran the doorman: "Fire in the vault! Everyone out, NOW!"; [t2] Pip the apprentice: "It's spreading— I'm not staying here!"; [t3] barrel of oil catches — fire climbing fast, hungry and bright!

### slice t3
```
·A·~··
··~*~t
···~··
G····$
```
_@=you  G=Bran the doorman  A=Pip the apprentice  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(0,3) — «Too proud to run further, I plant myself at the doorway and hold it open for the others.»
    grammar: aims at **yourself**; **certain** → the words "Get clear! I'll hold the door — go, go!"
    → ✓ **done** → the words "Get clear! I'll hold the door — go, go!"  _(no state change)_
    · sees: cask_water d3, player d3~1t
- **Pip the apprentice** @(1,0) — «I scramble further from the heat, making for the open corner by the door.»
    grammar: aims at **the cell to the W**; **certain** → Pip the apprentice moves west; the words "Bran, come on, it's not worth it!"
    → ✓ **done** → Pip the apprentice moves west; the words "Bran, come on, it's not worth it!"
    · sees: player d0, cask_water d1, barrel_oil d3, keg_oil d4~1t, fire@(3,1)
- **you** @(1,0) — «I press south past Pip, watching the fire chew through the stores, still hunting for the vault's prize.»
    grammar: aims at **the cell to the S**; **certain** → you moves south
    → ✓ **done** → you moves south
    · sees: apprentice d0, cask_water d1, barrel_oil d3, guard d3~1t, fire@(3,1)
- _⏳ meanwhile:_ [t3] Bran the doorman: "Get clear! I'll hold the door — go, go!"; [t3] Pip the apprentice: "Bran, come on, it's not worth it!"

### slice t4
```
A··~··
·@~*~t
···~··
G····$
```
_@=you  G=Bran the doorman  A=Pip the apprentice  ($=prize ~=fluid *=fire)_
- **Bran the doorman** @(0,3) — «With Pip clear and the flames past holding, I abandon the post myself and get out.»
    grammar: aims at **yourself**; **certain** → Bran the doorman becomes escaped; the words "Everyone's clear — out, out, the roof won't hold!"
    → ✓ **done** → Bran the doorman becomes escaped; the words "Everyone's clear — out, out, the roof won't hold!"
    · sees: apprentice d3, cask_water d3, player d3
- **Pip the apprentice** @(0,0) — «Safe by the door at last, I slip out into the cool air, shaking.»
    grammar: aims at **yourself**; **certain** → Pip the apprentice becomes escaped; the words "Oh thank goodness— out, I'm out!"
    → ✓ **done** → Pip the apprentice becomes escaped; the words "Oh thank goodness— out, I'm out!"
    · sees: cask_water d2, player d2, guard d3, barrel_oil d4~1t, keg_oil d5~2t
- **you** @(1,1) — «I dash through the rising smoke, snatch the prize from its case, and burst out through the door into open air.»
    grammar: aims at **starfire vial** — only if not (the target is taken); **certain** → the target becomes taken; you becomes has_prize; you becomes escaped
    → ✓ **done** → the target becomes taken; you becomes has_prize; you becomes escaped
    · sees: cask_water d0, apprentice d2, barrel_oil d2, guard d3, keg_oil d3, fire@(3,1)
- _⏳ meanwhile:_ [t4] Bran the doorman: "Everyone's clear — out, out, the roof won't hold!"; [t4] Pip the apprentice: "Oh thank goodness— out, I'm out!"; [t5] a lick of flame runs across the pooled fuel at (2, 1, 0) and it goes up with a whump!; [t5] a lick of flame runs across the pooled fuel at (3, 0, 0) and it goes up with a whump!; [t5] a lick of flame runs across the pooled fuel at (3, 2, 0) and it goes up with a whump!; [t5] a lick of flame runs across the pooled fuel at (4, 1, 0) and it goes up with a whump!
