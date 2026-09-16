# broodmother_ambush
_AMBUSH: the broodmother and scuttler start CLOSE and rouse the moment they perceive you, then HUNT and STRIKE. Survive — fight back, flee, or use the terrain; let HP actually drop and the danger mount._

### slice t0
```
@·B······
·S·······
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(2,0) — «I rouse to violence and lurch toward the crawler.»
    grammar: aims at **you**; **certain** → the pale broodmother is no longer calm; the pale broodmother becomes enraged; the pale broodmother moves west
    → ✓ **done** → the pale broodmother is no longer calm; the pale broodmother becomes enraged; the pale broodmother moves west
    · sees: player d2, scuttler d2, box0 d3, crowbar d3
- **a chittering scuttler** @(1,1) — «I skitter awake and dart to cut off the crawler.»
    grammar: aims at **you**; **certain** → a chittering scuttler is no longer calm; a chittering scuttler becomes enraged; a chittering scuttler moves north
    → ✓ **done** → a chittering scuttler is no longer calm; a chittering scuttler becomes enraged; a chittering scuttler moves north
    · sees: broodmother d2, player d2, box0 d3, crowbar d3
- **you** @(0,0) — «I raise my torch and back off, watching them stir.»
    grammar: aims at **the cell to the E**; **certain** → you moves east
    → ✓ **done** → you moves east
    · sees: broodmother d2, scuttler d2

### slice t1
```
·B·······
·········
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(1,0) — «I sink my mandibles into the crawler!»
    grammar: aims at **you**; **contest** the pale broodmother's composure vs (55 + 0)  [win → nothing happens · lose → the words "It snaps at the crawler and misses!"]
    → ✓ **win** → nothing happens  _(no state change)_
    · sees: player d0, scuttler d0, box0 d4~1t, crowbar d4~1t
- **a chittering scuttler** @(1,0) — «I dart in and claw at the crawler's flank!»
    grammar: aims at **you**; **contest** a chittering scuttler's composure vs (55 + 0)  [win → nothing happens · lose → the words "Its claws rake empty air!"]
    → ✓ **win** → nothing happens  _(no state change)_
    · sees: broodmother d0, player d0, box0 d4~1t, crowbar d4~1t
- **you** @(1,0) — «They're on me both at once! I drive my torch into the broodmother's eyes!»
    grammar: aims at **the pale broodmother**; **contest** (55 + 0) vs the target's composure  [win → nothing happens · lose → the words "The torch glances off its carapace!"]
    → ✗ **lose** → the words "The torch glances off its carapace!"  _(no state change)_
    · sees: broodmother d0, scuttler d0
- _⏳ meanwhile:_ [t1] you: "The torch glances off its carapace!"

### slice t2
```
·B·······
·········
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(1,0) — «I sink my mandibles into the crawler!»
    grammar: aims at **you**; **contest** the pale broodmother's composure vs (55 + 0)  [win → the target's hp changes by (-5 if the target is afraid else -3) · lose → the words "It snaps at the crawler and misses!"]
    → ✓ **win** → the target's hp changes by (-5 if the target is afraid else -3)
    · sees: player d0, scuttler d0, box0 d4~2t, crowbar d4~2t
- **a chittering scuttler** @(1,0) — «I claw at the crawler's flank!»
    grammar: aims at **you**; **contest** a chittering scuttler's composure vs (55 + 0)  [win → the target's hp changes by (-4 if the target is afraid else -2) · lose → the words "Its claws rake empty air!"]
    → ✓ **win** → the target's hp changes by (-4 if the target is afraid else -2)
    · sees: broodmother d0, player d0, box0 d4~2t, crowbar d4~2t
- **you** @(1,0) — «The broodmother's too strong to bite through -- I pivot and jam the torch into the scuttler instead!»
    grammar: aims at **a chittering scuttler**; **contest** (55 + 0) vs the target's composure  [win → the target's hp changes by (-6 if the target is afraid else -4) · lose → the words "The torch glances off its shell!"]
    → ✗ **lose** → the words "The torch glances off its shell!"  _(no state change)_
    · sees: broodmother d0, scuttler d0
- _⏳ meanwhile:_ [t2] you: "The torch glances off its shell!"

### slice t3
```
·B·······
·········
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(1,0) — «I rear up and shriek, mandibles flared, to break the crawler's nerve!»
    grammar: aims at **you**; **contest** the pale broodmother's composure vs (55 + 0)  [win → the target becomes afraid · lose → the words "The shriek echoes off the walls, but the crawler holds steady!"]
    → ✓ **win** → the target becomes afraid
    · sees: player d0, scuttler d0, box0 d4~3t, crowbar d4~3t
- **a chittering scuttler** @(1,0) — «I lunge and claw at the fleeing crawler before it can escape!»
    grammar: aims at **you**; **contest** a chittering scuttler's composure vs (55 + 0)  [win → the target's hp changes by (-4 if the target is afraid else -2) · lose → the words "Its claws rake empty air as the crawler breaks away!"]
    → ✗ **lose** → the words "Its claws rake empty air as the crawler breaks away!"  _(no state change)_
    · sees: broodmother d0, player d0, box0 d4~3t, crowbar d4~3t
- **you** @(1,0) — «Five hits from dead -- I break away, shoving past the scuttler and bolting east!»
    grammar: aims at **the cell to the E**; **certain** → you moves east
    → ✓ **done** → you moves east
    · sees: broodmother d0, scuttler d0
- _⏳ meanwhile:_ [t3] a chittering scuttler: "Its claws rake empty air as the crawler breaks away!"

### slice t4
```
·B@······
·········
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(1,0) — «I stalk wide, closing the last of the gap to cut off any escape.»
    grammar: aims at **you**; **certain** → the pale broodmother moves east
    → ✓ **done** → the pale broodmother moves east
    · sees: scuttler d0, player d1, box0 d4~4t, crowbar d4~4t
- **a chittering scuttler** @(1,0) — «I press the attack on the trembling crawler!»
    grammar: aims at **you**; **contest** a chittering scuttler's composure vs (55 + 0)  [win → the target's hp changes by (-4 if the target is afraid else -2) · lose → the words "Its claws rake empty air!"]
    → ✗ **lose** → the words "Its claws rake empty air!"  _(no state change)_
    · sees: broodmother d0, player d1, box0 d4~4t, crowbar d4~4t
- **you** @(2,0) — «Heart hammering, afraid but cornered -- I stab the torch at the scuttler's eyes!»
    grammar: aims at **a chittering scuttler**; **contest** (55 + 0) vs the target's composure  [win → the target's hp changes by (-6 if the target is afraid else -4) · lose → the words "The torch glances off its shell again!"]
    → ✗ **lose** → the words "The torch glances off its shell again!"  _(no state change)_
    · sees: broodmother d1, scuttler d1, box0 d3, crowbar d3
- _⏳ meanwhile:_ [t4] a chittering scuttler: "Its claws rake empty air!"; [t4] you: "The torch glances off its shell again!"

### slice t5
```
·SB······
·········
·········
·········
·········
·········
```
_@=you  S=a chittering scuttler  B=the pale broodmother  ($=prize ~=fluid *=fire)_
- **the pale broodmother** @(2,0) — «I lunge for the killing bite as it twists away!»
    grammar: aims at **you**; **contest** the pale broodmother's composure vs (55 + 0)  [win → the target's hp changes by (-3 if the target is afraid else -1) · lose → the words "Mandibles snap shut on nothing but air as the crawler dives clear!"]
    → ✓ **win** → the target's hp changes by (-3 if the target is afraid else -1)
    · sees: player d0, scuttler d1, box0 d3, crowbar d3
- **a chittering scuttler** @(1,0) — «I scuttle to cut off the passage south before it can escape!»
    grammar: aims at **you**; **certain** → a chittering scuttler moves south
    → ✓ **done** → a chittering scuttler moves south
    · sees: broodmother d1, player d1, box0 d4~5t, crowbar d4~5t
- **you** @(2,0) — «I hurl my waterskin into the broodmother's eyes and dive south, breaking for the passage!»
    grammar: aims at **the pale broodmother**; **contest** (55 + 0) vs the target's composure  [win → the target becomes blinded; you moves south · lose → the words "The water splashes uselessly -- I bolt south anyway!"; you moves south]
    → ✗ **lose** → the words "The water splashes uselessly -- I bolt south anyway!"; you moves south
    · sees: broodmother d0, scuttler d1, box0 d3, crowbar d3
- _⏳ meanwhile:_ [t5] you: "The water splashes uselessly -- I bolt south anyway!"
