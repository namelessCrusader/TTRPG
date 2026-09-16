# teach_meiling
_SOCIAL over several beats: earn Meiling's trust (small gifts/warmth, disposition in small steps) and get her to teach you; she replies and her stance shifts with how she now feels — a real arc._

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
- **Meiling the herbalist** @(6,1) — «I keep sorting herbs, not yet noticing anyone nearby.»
    grammar: aims at **yourself**; **certain** → the target's focus changes by 1
    → ✓ **done** → the target's focus changes by 1
    · sees: nothing
- **you** @(2,1) — «I walk east toward Meiling, wanting to talk to her.»
    grammar: aims at **yourself**; **certain** → the target moves east
    → ✓ **done** → the target moves east
    · sees: nothing

### slice t1
```
········
~··@··N·
~~······
~~······
~··hhr··
········
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «I set down my basket and take a few steps toward the stranger approaching.»
    grammar: aims at **yourself**; **certain** → the target moves west
    → ✓ **done** → the target moves west
    · sees: player d3
- **you** @(3,1) — «I close the distance, walking toward Meiling.»
    grammar: aims at **yourself**; **certain** → the target moves east
    → ✓ **done** → the target moves east
    · sees: hen0 d3, neighbour d3

### slice t2
```
~·······
~~··@N··
~~~·····
~~~·····
~~·hhr··
~·······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(5,1) — «I blink, surprised by his warmth, and accept the gift with a careful little smile.»
    grammar: aims at **yourself**; **certain** → the words "Oh! ...That's kind of you. Thank you."; Meiling the herbalist becomes touched
    → ✓ **done** → the words "Oh! ...That's kind of you. Thank you."; Meiling the herbalist becomes touched
    · sees: player d1, rooster d3
- **you** @(4,1) — «I smile and greet her warmly, then press a small share of my own qi into her hands as a gift, freely given.»
    grammar: aims at **Meiling the herbalist**; **certain** → the words "Meiling -- it's good to see you. Please, take this; a small gift, freely given."; 4 qi flows from you to the target; the target's disposition toward you changes by 0.15
    → ✓ **done** → the words "Meiling -- it's good to see you. Please, take this; a small gift, freely given."; 4 qi flows from you to the target; the target's disposition toward you changes by 0.15
    · sees: neighbour d1, hen1 d3, hen0 d4~1t
- _⏳ meanwhile:_ [t2] Meiling the herbalist: "Oh! ...That's kind of you. Thank you."; [t2] you: "Meiling -- it's good to see you. Please, take this; a small gift, freely given."

### slice t3
```
~~······
~~~·@N··
~~~~····
~~~~····
~~~hhr··
~~······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(5,1) — «I laugh softly, my reserve breaking, and tell him plainly that I like him.»
    grammar: aims at **yourself**; **certain** → the words "Hah -- you daft, generous thing. Alright. I like you."; Meiling the herbalist becomes fond
    → ✓ **done** → the words "Hah -- you daft, generous thing. Alright. I like you."; Meiling the herbalist becomes fond
    · sees: player d1, rooster d3
- **you** @(4,1) — «I offer a second share of my qi, gently, telling her I'd like to know her better.»
    grammar: aims at **Meiling the herbalist**; **certain** → the words "There's more where that came from -- I'd like to know you better, Meiling."; 4 qi flows from you to the target; the target's disposition toward you changes by 0.15
    → ✓ **done** → the words "There's more where that came from -- I'd like to know you better, Meiling."; 4 qi flows from you to the target; the target's disposition toward you changes by 0.15
    · sees: neighbour d1, hen1 d3, hen0 d4~2t
- _⏳ meanwhile:_ [t3] Meiling the herbalist: "Hah -- you daft, generous thing. Alright. I like you."; [t3] you: "There's more where that came from -- I'd like to know you better, Meiling."

### slice t4
```
~~······
~~~·@N··
~~~~····
~~~~····
~~~hhr··
~~······
```
_@=you  N=Meiling the herbalist  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(5,1) — «I set my basket aside and look at him properly, warmth plain on my face now.»
    grammar: aims at **yourself**; **certain** → the words "You've been nothing but kind to me. Go on then -- ask what's on your mind."
    → ✓ **done** → the words "You've been nothing but kind to me. Go on then -- ask what's on your mind."  _(no state change)_
    · sees: player d1, rooster d3
- **you** @(4,1) — «I ask her plainly: Meiling, will you teach me? I want to learn from you.»
    grammar: aims at **Meiling the herbalist**; **contest** (50 + (neighbour's disposition toward player × 60)) vs the target's composure  [win → you becomes apprentice; the target becomes mentor; the words "Thank you -- I won't waste your teaching." · lose → the words "...I understand. Maybe another time."]
    → ✓ **win** → you becomes apprentice; the target becomes mentor; the words "Thank you -- I won't waste your teaching."
    · sees: neighbour d1, hen1 d3, hen0 d4~3t
- _⏳ meanwhile:_ [t4] Meiling the herbalist: "You've been nothing but kind to me. Go on then -- ask what's on your mind."; [t4] you: "Thank you -- I won't waste your teaching."
