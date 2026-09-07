# Rulings — the case-law file

This file is the growing specification. It is not written up front; it grows one ruling at a
time. Each ruling is a plain sentence. New engine work is judged against these sentences.
Cases no ruling covers go back to the user. (Started 2026-08-31 from the first ten cases,
all taken from play-test runs 1 and 2.)

## Ruling 1 — model the causes, never the cases

An interaction must FALL OUT of explicitly modeled state. It must never be written directly
as "if verb X meets thing Y, set flag Z." Every hardcoded case is a thousand neighboring
cases silently wrong.

**The stopping rule (keeps this computable):** model a quantity explicitly at the moment two
cases the user wants to be different would otherwise be the same. Dig exactly that deep and
no deeper.

- Water must end a wood fire but not an oil fire → fire needs a FUEL field, not more tags.
- Acid on a boot must differ from acid on skin → the body needs covered PARTS.
- A drunk hearing a shout must differ from a sober one → people need ATTENTION.
- We do NOT model the boot buckle's alloy until some wanted case needs it.

**The standard already met once:** heat. Heat is a number on every cell moved by one
conduction formula; "metal answers fire faster than stone" falls out. Nobody wrote a case
for it. Everything else should reach this standard.

## The ten founding cases (2026-08-31)

1. **Bumping a barrel** — tipping-by-walking wants SPEED, which the grid does not carry.
   Parked: not expressible today; do not fake it with a dice roll.
2. **Standing in acid** — damage depends on depth, on the body part touched, and on what
   covers it (boot melts over time; skin burns; metal depends on the metal). Needs the body
   parts + coverings model. Until then, acid-hurts-people stays unmodeled rather than faked.
3. **Witnessing a crime** — the reaction depends on relationships (love/hate the victim),
   personality, and whether the witness was awake/attentive. "Witness → hostile" as a flat
   rule is a case; the causes are relations + drives + attention.
4. **Reporting a crime** — a told guard may act, ignore, or ask for a bribe. Same causes as
   3: relationship, personality, incentives. Speech CONTENT must be able to reach the brain
   (speech-intent pieces), but the response is the brain's, never scripted.
5. **NPCs handing things over** — NPCs must be able to give, throw, and take things, as
   decided by their own brains. (The only clean YES-add-the-capability ruling of the ten.)
6. **Water vs fire** — which wins depends on what burns (oil fire vs dry-material fire).
   Fire must become fuel + heat + burning material, not an on_fire tag.
7. **Torch over oil** — presence alone may ignite via modeled transfer (an ember, radiated
   heat), never via an "if torch near oil" case. The fire model owns this.
8. **Money** — a coin does nothing physically; people ACCEPT it in exchange. Value is a
   belief held by minds — a knife can be legal tender too. Model exchange/acceptance in the
   brain, not a special money mechanic.
9. **Monsters and loot** — whether the scuttler guards gold depends on its drives (eat
   humans? protect the brood? flee?). Model the creature as an entity with drives; guarding
   or cowering falls out.
10. **Shouting and addressing** — hearing rides on a modeled ATTENTION quantity (loudness,
    distance, drunkenness, distraction). Addressing someone is just facing them and speaking;
    everyone in earshot may attend.

## Agreed order of first-principles rebuilds

Fire (fuel+heat+material; heat half exists) → body parts + coverings (case 2) →
attention (case 10) → drives (cases 3, 4, 9). Money/exchange (case 8) waits on drives.
