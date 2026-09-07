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

## Ruling 2 — a property table is a hiding place for cases (2026-09-07)

Ruling 1 catches `if X meets Y`. It does not catch the same case written as a row in a
material table, and that is the easier mistake to make, because a table looks like physics.

**The case that produced this ruling.** A shut door was a perfect gas seal. The fix shipped
as `PORO = {WOOD: 0.01, STONE: 0.0002, ...}` — "wood passes a hundredth of a doorway."
Wood does not. A *door* does, because a door has a gap under it. "A door has gaps" had been
written into the WOOD row, where it would have been silently wrong for every wooden crate,
hull, chest and floorboard in the world. A case wearing a material's clothes.

**Two questions before any new table:**

1. **Is this a property of the MATERIAL, or of this OBJECT in this scene?** Density, toughness
   and ignition point are the material's. Gaps, fit, wear and how something was built are the
   object's, and they belong in the scene's geometry — a scene is *allowed* to describe shape.
2. **Could the sim DERIVE this from something it already tracks?** If yes, deriving beats
   declaring, always. A derived quantity cannot drift out of agreement with the thing it came
   from, and it keeps working in situations nobody enumerated.

**How it was actually fixed.** `smass` had carried the answer the whole time — `fill()`'s own
docstring says "frac < 1 is a PARTIAL voxel: a stick is mostly air inside its cube." Every gas
law was discarding it by asking `mat == AIR`, a binary question about a continuous field.
Porosity is now void fraction: solid mass over what that material weighs packed. The table is
gone. The door leaks because the *scene* says it fills 97% of its cells — true of that door,
claimed of nothing else.

**The tell that it was right:** things nobody wrote started working. Rubble and thatch breathe.
A wall half-eaten by acid breathes through the loss. And wood grows more permeable as it burns
away — measured, a packed block opening from void 0.000 to 0.163 at its charred face — so a
fire thins the very barrier that is starving it. A table would never have produced that.

**The general shape:** when a lie needs patching, first look for information the sim is already
throwing away. Binary reads of continuous fields are where it hides.

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
