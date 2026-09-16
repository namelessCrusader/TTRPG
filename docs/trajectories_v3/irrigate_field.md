# irrigate_field
_Farm work with WATER PHYSICS: till, sow, and flood the plot so water flows across the soil beat by beat; growth ticks up on watered, sown ground. Meiling works nearby, helps, and warms to you (keep disposition changes SMALL, ~0.1-0.2 — it lives in [-1,1])._

### slice t0
```
········
··@···N·
~·······
~·······
···hhr··
········
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «Looks like good work, friend! Let me come lend a hand.»
    grammar: aims at **you**; **certain** → the words "Looks like good work, friend! Let me come lend a hand."; neighbour's disposition toward player changes by 0.1; Meiling the herbalist moves None
    → ✗ **REFUSED** — KeyError: 'dir'
    · sees: nothing
- **you** @(2,1) — «I till the soil here to break it up for sowing.»
    grammar: aims at **here**; **certain** → the ground here becomes tilled
    → ✓ **done** → the ground here becomes tilled
    · sees: nothing

### slice t1
```
········
~·@···N·
~~······
~~······
~··hhr··
········
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «I'll come fetch water with you, dear - many hands make light work.»
    grammar: aims at **you**; **certain** → the words "I'll come fetch water with you, dear - many hands make light work."; neighbour's disposition toward player changes by 0.1; Meiling the herbalist moves west
    → ✓ **done** → the words "I'll come fetch water with you, dear - many hands make light work."; neighbour's disposition toward player changes by 0.1; Meiling the herbalist moves west
    · sees: nothing
- **you** @(2,1) — «I sow seeds into the freshly tilled soil.»
    grammar: aims at **here** — only if the ground here is tilled; **certain** → the ground here becomes sown; the ground here's growth is set to 0
    → ✓ **done** → the ground here becomes sown; the ground here's growth is set to 0
    · sees: nothing
- _⏳ meanwhile:_ [t1] Meiling the herbalist: "I'll come fetch water with you, dear - many hands make light work."

### slice t2
```
~·······
~~@··N··
~~~·····
~~~·····
~~·hhr··
~·······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(5,1) — «Let it soak in nicely - I'll bring more water over in a moment.»
    grammar: aims at **you**; **certain** → the words "Let it soak in nicely - I'll bring more water over in a moment."; neighbour's disposition toward player changes by 0.1; Meiling the herbalist moves west
    → ✓ **done** → the words "Let it soak in nicely - I'll bring more water over in a moment."; neighbour's disposition toward player changes by 0.1; Meiling the herbalist moves west
    · sees: player d3, rooster d3
- **you** @(2,1) — «I pour my jug of water over the sown seedbed.»
    grammar: aims at **here** — only if the ground here is sown; **certain** → 400ml of water pools at the ground here
    → ✓ **done** → 400ml of water pools at the ground here
    · sees: neighbour d3
- _⏳ meanwhile:_ [t2] Meiling the herbalist: "Let it soak in nicely - I'll bring more water over in a moment."

### slice t3
```
~~~·····
~~@~N···
~~~~····
~~~~····
~~~hhr··
~~······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(4,1) — «I'll fetch more water from the pool here to help it along.»
    grammar: aims at **the cell to the W**; **certain** → 200ml of water pools at the cell to the west; the words "Almost there - just a little more water should do it."; neighbour's disposition toward player changes by 0.1
    → ✓ **done** → 200ml of water pools at the cell to the west; the words "Almost there - just a little more water should do it."; neighbour's disposition toward player changes by 0.1
    · sees: player d2, hen1 d3, rooster d4~1t
- **you** @(2,1) — «I top up the water and tend the seedbed as it drinks it in.»
    grammar: aims at **here** — only if the ground here's water > 0; **certain** → 300ml of water pools at the ground here; the ground here's growth changes by 1
    → ✓ **done** → 300ml of water pools at the ground here; the ground here's growth changes by 1
    · sees: neighbour d2
- _⏳ meanwhile:_ [t3] Meiling the herbalist: "Almost there - just a little more water should do it."

### slice t4
```
~~~~····
~~@~N···
~~~~····
~~~~····
~~~hhr··
~~······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(4,1) — «The seedbed's really coming along - you've done fine work today.»
    grammar: aims at **you**; **certain** → the words "The seedbed's really coming along - you've done fine work today."; neighbour's disposition toward player changes by 0.1
    → ✓ **done** → the words "The seedbed's really coming along - you've done fine work today."; neighbour's disposition toward player changes by 0.1
    · sees: player d2, hen1 d3, rooster d4~2t
- **you** @(2,1) — «I tend the watered seedbed once more, watching the seedlings take root.»
    grammar: aims at **here** — only if the ground here's water > 0; **certain** → 150ml of water pools at the ground here; the ground here's growth changes by 1
    → ✓ **done** → 150ml of water pools at the ground here; the ground here's growth changes by 1
    · sees: neighbour d2
- _⏳ meanwhile:_ [t4] Meiling the herbalist: "The seedbed's really coming along - you've done fine work today."
