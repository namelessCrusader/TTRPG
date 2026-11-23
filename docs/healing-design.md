# Healing, infection, and what injuries leave behind

Scoped with the owner on 2026-11-22 (decisions HL1-HL5), after wave 15 decision Q3: burns can heal, but burn healing is complex and porting it to voxels is "a separate thing in itself".

What exists:
- BONES HEAL through a callus law (src/voxel/bonestate.py: callus grows while strain stays inside a window, stalls above it as non-union, and a splint helps);
- FIRST AID (src/voxel/firstaid.py: press, bind and tourniquet all write one outside pressure on the vessel flow law);
- BURNS per skin face (src/voxel/skinburn.py).
- Wounds and burns do not heal yet.

## HL1. One tissue-repair law

- It GENERALISES the bone callus law to every tissue: skin, cuts and wounds, muscle, vessels, nerves, burns.
- It ACTS ON THE BODY'S DAMAGE RECORDS (per bone, per skin face, per vessel), not on voxels. The voxels and skin layers only SHOW the result, such as a scar's colour and texture. This answers the porting worry.
- There are real phases (haemostasis, inflammation, proliferation, remodelling). Their rates are set by:
  - perfusion (the vessel law);
  - nutrition and energy, paid from the body's books;
  - age (a dial);
  - REST AGAINST STRAIN (a strain window, as for bone: a wound kept moving will not close);
  - infection (HL2);
  - treatment (GAP CLOSURE: a stitched cut heals faster, as bone's gap does).
- The outcome is a RESTORED share plus SCAR: weaker, stiffer and less sensitive. Skin scars regain about 80 percent of their original strength (u; check at build time).

## HL2. Infection: a population in the wound

- CONTAMINATION comes from the world: dirt, dung, dirty hands, a soiled blade, river water (the substances the sim already tracks on surfaces and in layers).
- Microbes GROW faster in warmth, dead tissue and poor perfusion. The IMMUNE RESPONSE clears them and is paid from the body's energy, so a starving or exhausted body fights worse.
- FEVER, swelling and pus are the body's responses, seen by others through signs (SV2). Untreated, infection spreads to blood poisoning and death.
- TREATMENT acts on the same law: cleaning, debridement, honey, wine, vinegar, later real antiseptics, with effects from sources.
- FALSE REMEDIES spread through the knowledge layer (bloodletting, dung poultices) and make things worse. Medical knowledge can be right or wrong, and it spreads or is lost (docs/knowledge-design.md).
- Care matters in exactly the way the lab needs: the tended and cleaned live, and the neglected die. That is the dependence that makes kindness pay.

## HL3. What an injury leaves

Lasting effects come from HOW THE HEALING WENT, through the repair law's own outcome:
- scars weaker, stiffer and less sensitive, and big burn scars that tighten (contractures);
- MALUNION: a break healed without a splint keeps its angle (the bone record stores it), so the leg is crooked and the gait changes;
- chronic pain from nerve damage or sensitisation, flaring with cold or strain;
- joint stiffness from immobility;
- REHABILITATION restores part of the function through use-dependent plasticity (magic MC3).

Bodies carry their histories: the veteran's limp, the burned smith's tight hand, the old break that aches before rain.

## HL4. Magical healing: two lawful routes

1. FEEDING THE BODY'S OWN HEALING: magic supplies energy, warmth, perfusion and an immune boost. It is faster, but by the SAME LAW, so a misaligned bone still sets crooked and scars still form unless the wound is closed well.
2. BOUNDED REBUILD (coverage CV1): restoring the tissue to its PATTERN.
   - It is mass-limited and costly.
   - It is only as good as the healer's knowledge of anatomy (information cost, cost kind 3).
   - A perfect pattern can heal without a scar.

There is no raising the dead unless a pack adds it.

## HL5. When

- WAVE 15 BATCH 4, after batch 3:
  - the tissue-repair law;
  - infection with the immune fight;
  - lasting effects;
  - healing fast-forward by TWINS (docs/daily-life-design.md SC5), handing back at a fever spike or a reopened wound.
- Magical healing with magic: route 1 with the first slices, route 2 with bounded rebuild.
