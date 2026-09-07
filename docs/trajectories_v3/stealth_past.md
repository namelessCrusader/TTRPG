# stealth_past
_SNEAK to the stairs without waking the scuttler. Partial observability is the game — stay out of its sight; it acts only on what IT perceives._

### slice t0
```
@········
·········
····S····
·········
·········
·········
```
_@=you  S=a chittering scuttler  ($=prize ~=fluid *=fire)_
- **a chittering scuttler** @(4,2) — «I paw at the sealed boxes, sensing nothing amiss.»
    grammar: aims at **yourself**; **certain** → a chittering scuttler becomes watchful
    → ✓ **done** → a chittering scuttler becomes watchful
    · sees: box1 d1, coin d1, box0 d2, crowbar d2
- **you** @(0,0) — «I creep south along the wall, hugging the shadows.»
    grammar: aims at **yourself**; **certain** → you moves south; you becomes hidden
    → ✓ **done** → you moves south; you becomes hidden
    · sees: nothing

### slice t1
```
·········
@········
····S····
·········
·········
·········
```
_@=you  S=a chittering scuttler  ($=prize ~=fluid *=fire)_
- **a chittering scuttler** @(4,2) — «I rattle the crowbar idly, still noticing nothing.»
    grammar: aims at **yourself**; **certain** → a chittering scuttler becomes watchful
    → ✓ **done** → a chittering scuttler becomes watchful  _(no state change)_
    · sees: box1 d1, coin d1, box0 d2, crowbar d2
- **you** @(0,1) — «I keep to the wall, sliding further south, still hidden.»
    grammar: aims at **yourself**; **certain** → you moves south; you becomes hidden
    → ✓ **done** → you moves south; you becomes hidden
    · sees: box0 d3, crowbar d3

### slice t2
```
·········
·········
@···S····
·········
·········
·········
```
_@=you  S=a chittering scuttler  ($=prize ~=fluid *=fire)_
- **a chittering scuttler** @(4,2) — «I shuffle a step, patrolling near the boxes, unaware.»
    grammar: aims at **yourself**; **certain** → a chittering scuttler moves north; a chittering scuttler becomes watchful
    → ✓ **done** → a chittering scuttler moves north; a chittering scuttler becomes watchful
    · sees: box1 d1, coin d1, box0 d2, crowbar d2
- **you** @(0,2) — «I press on south, keeping four cells of distance between us.»
    grammar: aims at **yourself**; **certain** → you moves south; you becomes hidden
    → ✓ **done** → you moves south; you becomes hidden
    · sees: box1 d3, coin d3, box0 d4~1t, crowbar d4~1t

### slice t3
```
·········
····S····
·········
@········
·········
·········
```
_@=you  S=a chittering scuttler  ($=prize ~=fluid *=fire)_
- **a chittering scuttler** @(4,1) — «I drift back south toward the boxes, patrol unbroken.»
    grammar: aims at **yourself**; **certain** → a chittering scuttler moves south; a chittering scuttler becomes watchful
    → ✓ **done** → a chittering scuttler moves south; a chittering scuttler becomes watchful
    · sees: box0 d1, crowbar d1, box1 d2, coin d2
- **you** @(0,3) — «I edge east now that it's turned its back, still four cells clear.»
    grammar: aims at **yourself**; **certain** → you moves east; you becomes hidden
    → ✓ **done** → you moves east; you becomes hidden
    · sees: box1 d4~1t, coin d4~1t, box0 d5~2t, crowbar d5~2t

### slice t4
```
·········
·········
····S····
·@·······
·········
·········
```
_@=you  S=a chittering scuttler  ($=prize ~=fluid *=fire)_
- **a chittering scuttler** @(4,2) — «I keep pawing through the loot, none the wiser.»
    grammar: aims at **yourself**; **certain** → a chittering scuttler becomes watchful
    → ✓ **done** → a chittering scuttler becomes watchful  _(no state change)_
    · sees: box1 d1, coin d1, box0 d2, crowbar d2
- **you** @(1,3) — «I duck south past its blind spot, closing on the stairwell.»
    grammar: aims at **yourself**; **certain** → you moves south; you becomes hidden
    → ✓ **done** → you moves south; you becomes hidden
    · sees: box1 d3, coin d3, box0 d4~3t, crowbar d4~3t
