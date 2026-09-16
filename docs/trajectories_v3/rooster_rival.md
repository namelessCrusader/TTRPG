# rooster_rival
_The proud ROOSTER (a Beware-of-Chicken beast) is territorial and acts on its OWN each beat — puff up, peck, charge, or flee if cowed. Shove, cow, feed, or befriend it; Meiling watches and her disposition shifts by how kindly you handle the bird (small steps)._

### slice t0
```
········
··@R··N·
~·······
~·······
···hh···
········
```
_@=you  N=Meiling the herbalist  R=proud young rooster  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «I glance up from my herbs, watching the rooster puff up near the newcomer»
    grammar: aims at **yourself**; **certain** → the words "That one's been strutting for days -- mind he doesn't test you."
    → ✓ **done** → the words "That one's been strutting for days -- mind he doesn't test you."  _(no state change)_
    · sees: rooster d3
- **you** @(2,1) — «I plant my feet, spread my arms wide, and shout to stare the rooster down before it decides to charge»
    grammar: aims at **proud young rooster**; **contest** you's qi vs (the target's hp × 3)  [win → the target becomes cowed; neighbour's disposition toward you changes by 0.05 · lose → the words "It's not backing down!"]
    → ✓ **win** → the target becomes cowed; neighbour's disposition toward you changes by 0.05
    · sees: rooster d1
- **proud young rooster** @(3,1) — «I fluff up my hackles and strut toward the stranger, sizing them up for a charge»
    grammar: aims at **you** — only if not (proud young rooster is cowed); **certain** → proud young rooster becomes aggressive; proud young rooster moves west
    → · **guard false** → no-op (a precondition wasn't met — e.g. someone beat them to it)
    · sees: player d1, hen0 d3, neighbour d3
- _⏳ meanwhile:_ [t0] Meiling the herbalist: "That one's been strutting for days -- mind he doesn't test you."

### slice t1
```
········
~·@R··N·
~~······
~~······
~··hh···
········
```
_@=you  N=Meiling the herbalist  R=proud young rooster  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «I watch, surprised, as the rooster lets itself be fed by a stranger»
    grammar: aims at **yourself**; **certain** → the words "Feeding him already? Careful -- he'll follow you home like a fool for scraps."
    → ✓ **done** → the words "Feeding him already? Careful -- he'll follow you home like a fool for scraps."  _(no state change)_
    · sees: rooster d3
- **you** @(2,1) — «I crouch low and offer a trickle of my own qi to the still-wary rooster, trying to calm it»
    grammar: aims at **proud young rooster** — only if the target is cowed; **certain** → 3 qi flows from you to the target; the target becomes fed; the target is no longer cowed; neighbour's disposition toward you changes by 0.1
    → ✓ **done** → 3 qi flows from you to the target; the target becomes fed; the target is no longer cowed; neighbour's disposition toward you changes by 0.1
    · sees: rooster d1
- **proud young rooster** @(3,1) — «I settle, feathers smoothing, and peck curiously at the offered qi instead of bolting»
    grammar: aims at **yourself** — only if proud young rooster is fed; **certain** → proud young rooster becomes calm; proud young rooster's disposition toward player changes by 0.2
    → ✓ **done** → proud young rooster becomes calm; proud young rooster's disposition toward player changes by 0.2
    · sees: player d1, hen0 d3, neighbour d3
- _⏳ meanwhile:_ [t1] Meiling the herbalist: "Feeding him already? Careful -- he'll follow you home like a fool for scraps."

### slice t2
```
~·······
~~@R··N·
~~~·····
~~~·····
~~·hh···
~·······
```
_@=you  N=Meiling the herbalist  R=proud young rooster  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «I smile, watching you win the rooster's trust, though I keep an eye on the hen he's now strutting toward»
    grammar: aims at **yourself**; **certain** → the words "You've a gentle hand -- that's more than most manage with him. Mind the hen now, he's got ideas."
    → ✓ **done** → the words "You've a gentle hand -- that's more than most manage with him. Mind the hen now, he's got ideas."  _(no state change)_
    · sees: rooster d3
- **you** @(2,1) — «I reach out and gently scratch the calmed rooster's neck, sealing something like a truce»
    grammar: aims at **proud young rooster** — only if the target is fed; **certain** → you's disposition toward the target changes by 0.1; the target's disposition toward you changes by 0.1; neighbour's disposition toward you changes by 0.05; the words "Good bird. We're square, yeah?"
    → ✓ **done** → you's disposition toward the target changes by 0.1; the target's disposition toward you changes by 0.1; neighbour's disposition toward you changes by 0.05; the words "Good bird. We're square, yeah?"
    · sees: rooster d1
- **proud young rooster** @(3,1) — «Calmed toward the human, my pride turns elsewhere -- I puff up and strut off toward the hen»
    grammar: aims at **yourself** — only if proud young rooster is calm; **certain** → proud young rooster becomes territorial; proud young rooster moves south; the words "Buk-GAWK!"
    → ✓ **done** → proud young rooster becomes territorial; proud young rooster moves south; the words "Buk-GAWK!"
    · sees: player d1, hen0 d3, neighbour d3
- _⏳ meanwhile:_ [t2] Meiling the herbalist: "You've a gentle hand -- that's more than most manage with him. Mind the hen now, he's got ideas."; [t2] you: "Good bird. We're square, yeah?"; [t2] proud young rooster: "Buk-GAWK!"

### slice t3
```
~~······
~~@···N·
~~~R····
~~~~····
~~~hh···
~~······
```
_@=you  N=Meiling the herbalist  R=proud young rooster  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «I straighten up, half-worried -- that's my best layer he's eyeing»
    grammar: aims at **yourself**; **certain** → the words "Careful now -- that's my best layer he's squaring up to."
    → ✓ **done** → the words "Careful now -- that's my best layer he's squaring up to."  _(no state change)_
    · sees: rooster d3~1t
- **you** @(2,1) — «I whistle sharply and call the rooster back before he starts trouble with the hen»
    grammar: aims at **proud young rooster** — only if the target is territorial; **contest** you's qi vs (the target's hp × 4)  [win → the target becomes recalled; the target is no longer territorial; the target's disposition toward you changes by 0.05; the words "Hey! Leave her be." · lose → the words "He's not listening this time..."]
    → ✓ **win** → the target becomes recalled; the target is no longer territorial; the target's disposition toward you changes by 0.05; the words "Hey! Leave her be."
    · sees: rooster d2
- **proud young rooster** @(3,2) — «I stalk toward the hen, hackles up, ready to assert dominance over the flock»
    grammar: aims at **black-feathered hen** — only if proud young rooster is territorial; **contest** (proud young rooster's hp + proud young rooster's qi) vs (the target's hp × 2)  [win → the target's hp changes by -1; proud young rooster's disposition toward the target changes by -0.1 · lose → proud young rooster's qi changes by -1; the words "Bwaak?"]
    → · **guard false** → no-op (a precondition wasn't met — e.g. someone beat them to it)
    · sees: hen0 d2, player d2, hen1 d3, neighbour d4~1t
- _⏳ meanwhile:_ [t3] Meiling the herbalist: "Careful now -- that's my best layer he's squaring up to."; [t3] you: "Hey! Leave her be."

### slice t4
```
~~······
~~@···N·
~~~R····
~~~~····
~~~hh···
~~······
```
_@=you  N=Meiling the herbalist  R=proud young rooster  ($=prize ~=fluid *=fire)_
- **Meiling the herbalist** @(6,1) — «I look up as the rooster trots back, tamer than I've ever seen him, and grin»
    grammar: aims at **yourself** — only if rooster is settled; **certain** → the words "Well now -- fed, calmed, and minding you like an old hen. You've a way with him I didn't expect."; Meiling the herbalist's disposition toward player changes by 0.05
    → · **guard false** → no-op (a precondition wasn't met — e.g. someone beat them to it)
    · sees: rooster d3~2t
- **you** @(2,1) — «I relax my stance now that he's heeded me, and let him go about his business»
    grammar: aims at **proud young rooster** — only if the target is recalled; **certain** → the target is no longer recalled; the words "There we go. Go on, you're alright."
    → ✓ **done** → the target is no longer recalled; the words "There we go. Go on, you're alright."
    · sees: rooster d2
- **proud young rooster** @(3,2) — «Calm and fond of the human now, I trot back that way instead of staying with the hens»
    grammar: aims at **yourself** — only if proud young rooster is calm; **certain** → proud young rooster moves north; proud young rooster becomes settled
    → ✓ **done** → proud young rooster moves north; proud young rooster becomes settled
    · sees: hen0 d2, player d2, hen1 d3, neighbour d4~2t
- _⏳ meanwhile:_ [t4] you: "There we go. Go on, you're alright."
