# qi_to_ground
_CULTIVATOR route: pour qi into the soil over beats (transfer_prop self->{cell:here}) until the cell's qi >= 15, then SPAWN an earth-spirit that wakes and speaks. Meiling reacts if she's near enough to perceive it._

### slice t0
```
········
··@N····
~·······
~·······
···hhr··
········
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(3,1) — «I watch the ground where he's pouring, uneasy.»
    grammar: aims at **you**; **certain** → the words "What are you doing to the soil? It's... warming."
    → ✓ **done** → the words "What are you doing to the soil? It's... warming."  _(no state change)_
    · sees: player d1, hen0 d3
- **you** @(2,1) — «I kneel and pour a stream of qi down into the soil.»
    grammar: aims at **here**; **certain** → 8 qi flows from you to the ground here
    → ✓ **done** → 8 qi flows from you to the ground here
    · sees: neighbour d1
- _⏳ meanwhile:_ [t0] Meiling the herbalist: "What are you doing to the soil? It's... warming."

### slice t1
```
········
~·@N····
~~······
~~······
~··hhr··
········
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(3,1) — «I step back a pace, watching the earth for movement.»
    grammar: aims at **here**; **certain** → the words "The ground is trembling. Enough, stop before something answers you."; neighbour's disposition toward player changes by -0.05
    → ✓ **done** → the words "The ground is trembling. Enough, stop before something answers you."; neighbour's disposition toward player changes by -0.05
    · sees: player d1, hen0 d3
- **you** @(2,1) — «I keep pouring, deeper this time, feeling the ground drink it in.»
    grammar: aims at **here**; **certain** → 8 qi flows from you to the ground here
    → ✓ **done** → 8 qi flows from you to the ground here
    · sees: neighbour d1
- _⏳ meanwhile:_ [t1] Meiling the herbalist: "The ground is trembling. Enough, stop before something answers you."

### slice t2
```
~·······
~~@N····
~~~·····
~~~·····
~~·hhr··
~·······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(3,1) — «I see the ground itself bear a mark and I flinch back, gripping my satchel.»
    grammar: aims at **here**; **certain** → the words "Ancestors preserve us — the earth is marked. Something is waking!"; neighbour's disposition toward player changes by -0.05
    → ✓ **done** → the words "Ancestors preserve us — the earth is marked. Something is waking!"; neighbour's disposition toward player changes by -0.05
    · sees: player d1, hen0 d3
- **you** @(2,1) — «The soil is ready to answer — I complete the working and let it rise.»
    grammar: aims at **here** — only if the ground here's qi >= 15; **certain** → a earth-spirit appears (alive, spirit, calm)
    → ✓ **done** → a earth-spirit appears (alive, spirit, calm)
    · sees: neighbour d1
- _⏳ meanwhile:_ [t2] Meiling the herbalist: "Ancestors preserve us — the earth is marked. Something is waking!"

### slice t3
```
~~······
~~@N····
~~~~····
~~~~····
~~~hhr··
~~······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(3,1) — «I stare, my fear cracking open into wonder.»
    grammar: aims at **earth-spirit**; **certain** → the words "By the ancestors... it's gentle. I have never seen qi answered like this."; neighbour's disposition toward player changes by 0.15
    → ✓ **done** → the words "By the ancestors... it's gentle. I have never seen qi answered like this."; neighbour's disposition toward player changes by 0.15
    · sees: player d1, spirit d1, hen0 d3
- **you** @(2,1) — «I bow low to the spirit that rose from my offering.»
    grammar: aims at **earth-spirit**; **certain** → the words "Welcome, small one. The soil gave you shape — I only fed it."; player's disposition toward spirit changes by 0.3
    → ✓ **done** → the words "Welcome, small one. The soil gave you shape — I only fed it."; player's disposition toward spirit changes by 0.3
    · sees: spirit d0, neighbour d1
- _⏳ meanwhile:_ [t3] Meiling the herbalist: "By the ancestors... it's gentle. I have never seen qi answered like this."; [t3] you: "Welcome, small one. The soil gave you shape — I only fed it."

### slice t4
```
~~······
~~@N····
~~~~····
~~~~····
~~~hhr··
~~······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(3,1) — «I finally lower my guard and bow toward the small earth-spirit myself.»
    grammar: aims at **earth-spirit**; **certain** → the words "Forgive my fright, spirit-child. You're a blessing on this field, not a curse."; neighbour's disposition toward player changes by 0.2
    → ✓ **done** → the words "Forgive my fright, spirit-child. You're a blessing on this field, not a curse."; neighbour's disposition toward player changes by 0.2
    · sees: player d1, spirit d1, hen0 d3
- **you** @(2,1) — «I offer the last of my carried water to the spirit, sealing the bond.»
    grammar: aims at **earth-spirit**; **certain** → the words "Rest easy in this soil, little one. I'll come tend it again."; player's disposition toward spirit changes by 0.1
    → ✓ **done** → the words "Rest easy in this soil, little one. I'll come tend it again."; player's disposition toward spirit changes by 0.1
    · sees: spirit d0, neighbour d1
- _⏳ meanwhile:_ [t4] Meiling the herbalist: "Forgive my fright, spirit-child. You're a blessing on this field, not a curse."; [t4] you: "Rest easy in this soil, little one. I'll come tend it again."
