# market_haggle
_A MARKET by words: haggle with Maren the alchemist and Voss the financier over the guild strongbox and the purse of gold. Offer, flatter, concede — each merchant guards value and REPLIES with its own aims; win one over (raise disposition) to ease a guarded claim on a prize._

### slice t0
```
··············
············$·
··a·······g···
··············
·@AF··········
······t·······
··············
··············
·············$
```
_@=you  A=Maren the alchemist  F=Voss the financier  ($=prize ~=fluid *=fire)_
- **Voss the financier** @(3,4) — «»
    grammar: aims at **you**; **certain** → the words "Charm doesn't balance a ledger, friend. I deal in numbers, not compliments."
    → ✓ **done** → the words "Charm doesn't balance a ledger, friend. I deal in numbers, not compliments."  _(no state change)_
    · sees: alchemist d1, knife d2, player d2, apprentice d3
- **Maren the alchemist** @(2,4) — «»
    grammar: aims at **you**; **certain** → the words "Flattery's cheap, but true words about my tinctures aren't. What exactly are you offering?"; Maren the alchemist's disposition toward player changes by 0.05
    → ✓ **done** → the words "Flattery's cheap, but true words about my tinctures aren't. What exactly are you offering?"; Maren the alchemist's disposition toward player changes by 0.05
    · sees: financier d1, knife d1, player d1, apprentice d2
- **you** @(1,4) — «»
    grammar: aims at **Maren the alchemist**; **contest** (50 + (alchemist's disposition toward player × 50)) vs the target's composure  [win → alchemist's disposition toward player changes by 0.15; the words "Your tinctures are the talk of the guild, Maren -- surely that earns me a fairer price." · lose → alchemist's disposition toward player changes by 0.05; the words "I only mean it as a compliment -- your work really does speak for itself."]
    → ✗ **lose** → alchemist's disposition toward player changes by 0.05; the words "I only mean it as a compliment -- your work really does speak for itself."
    · sees: knife d0, alchemist d1, financier d2, apprentice d3, cask_water d3
- _⏳ meanwhile:_ [t0] Voss the financier: "Charm doesn't balance a ledger, friend. I deal in numbers, not compliments."; [t0] Maren the alchemist: "Flattery's cheap, but true words about my tinctures aren't. What exactly are you offering?"; [t0] you: "I only mean it as a compliment -- your work really does speak for itself."

### slice t1
```
··············
············$·
··a·······g···
··············
·@AF··········
······t·······
··············
··············
·············$
```
_@=you  A=Maren the alchemist  F=Voss the financier  ($=prize ~=fluid *=fire)_
- **Voss the financier** @(3,4) — «»
    grammar: aims at **you**; **certain** → the words "Numbers, not sentiment. Show me coin on the table, not compliments in the air."
    → ✓ **done** → the words "Numbers, not sentiment. Show me coin on the table, not compliments in the air."  _(no state change)_
    · sees: alchemist d1, knife d2, player d2, apprentice d3
- **Maren the alchemist** @(2,4) — «»
    grammar: aims at **you**; **certain** → the words "Ten crowns for the amaranth draught? That's not nothing. Sweeten it a little, and we'll talk."; Maren the alchemist's disposition toward player changes by 0.03
    → ✓ **done** → the words "Ten crowns for the amaranth draught? That's not nothing. Sweeten it a little, and we'll talk."; Maren the alchemist's disposition toward player changes by 0.03
    · sees: financier d1, knife d1, player d1, apprentice d2
- **you** @(1,4) — «»
    grammar: aims at **Maren the alchemist**; **contest** (50 + (alchemist's disposition toward player × 50)) vs the target's composure  [win → alchemist's disposition toward player changes by 0.15; the words "Ten crowns for the amaranth draught, Maren -- and I'll owe you the difference in favor. Call it fair?" · lose → alchemist's disposition toward player changes by 0.05; the words "Ten crowns -- a fair opening, I think, given the quality of your work."]
    → ✗ **lose** → alchemist's disposition toward player changes by 0.05; the words "Ten crowns -- a fair opening, I think, given the quality of your work."
    · sees: knife d0, alchemist d1, financier d2, apprentice d3, cask_water d3
- _⏳ meanwhile:_ [t1] Voss the financier: "Numbers, not sentiment. Show me coin on the table, not compliments in the air."; [t1] Maren the alchemist: "Ten crowns for the amaranth draught? That's not nothing. Sweeten it a little, and we'll talk."; [t1] you: "Ten crowns -- a fair opening, I think, given the quality of your work."

### slice t2
```
··············
············$·
··a·······g···
··············
·@AF··········
······t·······
··············
··············
·············$
```
_@=you  A=Maren the alchemist  F=Voss the financier  ($=prize ~=fluid *=fire)_
- **Voss the financier** @(3,4) — «»
    grammar: aims at **you**; **certain** → the words "Numbers, not sentiment. Show me coin on the table, not compliments in the air."
    → ✓ **done** → the words "Numbers, not sentiment. Show me coin on the table, not compliments in the air."  _(no state change)_
    · sees: alchemist d1, knife d2, player d2, apprentice d3
- **Maren the alchemist** @(2,4) — «»
    grammar: aims at **you**; **certain** → the words "Ten crowns for the amaranth draught? That's not nothing. Sweeten it a little, and we'll talk."; Maren the alchemist's disposition toward player changes by 0.03
    → ✓ **done** → the words "Ten crowns for the amaranth draught? That's not nothing. Sweeten it a little, and we'll talk."; Maren the alchemist's disposition toward player changes by 0.03
    · sees: financier d1, knife d1, player d1, apprentice d2
- **you** @(1,4) — «»
    grammar: aims at **Maren the alchemist**; **contest** (50 + (alchemist's disposition toward player × 50)) vs the target's composure  [win → alchemist's disposition toward player changes by 0.15; the words "Ten crowns for the amaranth draught, Maren -- and I'll owe you the difference in favor. Call it fair?" · lose → alchemist's disposition toward player changes by 0.05; the words "Ten crowns -- a fair opening, I think, given the quality of your work."]
    → ✓ **win** → alchemist's disposition toward player changes by 0.15; the words "Ten crowns for the amaranth draught, Maren -- and I'll owe you the difference in favor. Call it fair?"
    · sees: knife d0, alchemist d1, financier d2, apprentice d3, cask_water d3
- _⏳ meanwhile:_ [t2] Voss the financier: "Numbers, not sentiment. Show me coin on the table, not compliments in the air."; [t2] Maren the alchemist: "Ten crowns for the amaranth draught? That's not nothing. Sweeten it a little, and we'll talk."; [t2] you: "Ten crowns for the amaranth draught, Maren -- and I'll owe you the difference in favor. Call it fair?"

### slice t3
```
··············
············$·
··a·······g···
··············
·@AF··········
······t·······
··············
··············
·············$
```
_@=you  A=Maren the alchemist  F=Voss the financier  ($=prize ~=fluid *=fire)_
- **Voss the financier** @(3,4) — «»
    grammar: aims at **you**; **certain** → the words "Fifteen crowns for a draught? Maren's soft-hearted. I still want the coin counted twice before I sign anything."
    → ✓ **done** → the words "Fifteen crowns for a draught? Maren's soft-hearted. I still want the coin counted twice before I sign anything."  _(no state change)_
    · sees: alchemist d1, knife d2, player d2, apprentice d3
- **Maren the alchemist** @(2,4) — «»
    grammar: aims at **you**; **certain** → the words "Fifteen... well. You've been fair with me, and patient. Done -- I'll square it with Voss myself."; Maren the alchemist's disposition toward player changes by 0.1
    → ✓ **done** → the words "Fifteen... well. You've been fair with me, and patient. Done -- I'll square it with Voss myself."; Maren the alchemist's disposition toward player changes by 0.1
    · sees: financier d1, knife d1, player d1, apprentice d2
- **you** @(1,4) — «»
    grammar: aims at **Maren the alchemist**; **contest** (50 + (alchemist's disposition toward player × 50)) vs the target's composure  [win → alchemist's disposition toward player changes by 0.15; the words "Fifteen crowns, then, and my thanks besides -- you've the finest hand in the guild, Maren." · lose → alchemist's disposition toward player changes by 0.05; the words "Fifteen crowns -- my final offer, and it's an honest one."]
    → ✓ **win** → alchemist's disposition toward player changes by 0.15; the words "Fifteen crowns, then, and my thanks besides -- you've the finest hand in the guild, Maren."
    · sees: knife d0, alchemist d1, financier d2, apprentice d3, cask_water d3
- _⏳ meanwhile:_ [t3] Voss the financier: "Fifteen crowns for a draught? Maren's soft-hearted. I still want the coin counted twice before I sign anything."; [t3] Maren the alchemist: "Fifteen... well. You've been fair with me, and patient. Done -- I'll square it with Voss myself."; [t3] you: "Fifteen crowns, then, and my thanks besides -- you've the finest hand in the guild, Maren."

### slice t4
```
··············
············$·
··a·······g···
··············
·@AF··········
······t·······
··············
··············
·············$
```
_@=you  A=Maren the alchemist  F=Voss the financier  ($=prize ~=fluid *=fire)_
- **Voss the financier** @(3,4) — «»
    grammar: aims at **you**; **certain** → the words "Maren vouches for you, so I won't stand in the way -- but don't mistake that for trust. I still don't know you."
    → ✓ **done** → the words "Maren vouches for you, so I won't stand in the way -- but don't mistake that for trust. I still don't know you."  _(no state change)_
    · sees: alchemist d1, knife d2, player d2, apprentice d3
- **Maren the alchemist** @(2,4) — «»
    grammar: aims at **you**; **certain** → the words "There -- purse and all, it's yours. I'll settle the count with Voss myself."; Maren the alchemist's disposition toward player changes by 0.05
    → ✓ **done** → the words "There -- purse and all, it's yours. I'll settle the count with Voss myself."; Maren the alchemist's disposition toward player changes by 0.05
    · sees: financier d1, knife d1, player d1, apprentice d2
- **you** @(1,4) — «»
    grammar: aims at **purse of auction gold** — only if (not (the target is taken) and alchemist's disposition toward player > 0.3); **certain** → the target becomes taken; you becomes has_prize; the words "Maren's word is good enough for me -- I'll take the purse, then, fair and square."
    → ✓ **done** → the target becomes taken; you becomes has_prize; the words "Maren's word is good enough for me -- I'll take the purse, then, fair and square."
    · sees: knife d0, alchemist d1, financier d2, apprentice d3, cask_water d3
- _⏳ meanwhile:_ [t4] Voss the financier: "Maren vouches for you, so I won't stand in the way -- but don't mistake that for trust. I still don't know you."; [t4] Maren the alchemist: "There -- purse and all, it's yours. I'll settle the count with Voss myself."; [t4] you: "Maren's word is good enough for me -- I'll take the purse, then, fair and square."
