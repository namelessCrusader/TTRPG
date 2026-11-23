# Self-care, background life, foresight, and one approximation framework

Scoped with the owner on 2026-11-22 (decisions SC1-SC7), after the owner's call for "one general self-care rule for everyone, scoped later".

## SC1. What drives self-care

- THE SAME BODY DRIVES FOR EVERYONE: hunger, thirst, fatigue, cold, pain, fear, company. They are fed to whatever chooser a character has: a learned mind, a rules mind, or a script using the general self-care rule.
- HABITS form from repeated relief (the learned mind's habit term).
- ROUTINES are chunked sequences cued by time and place, like the techniques of PA5 (docs/physical-action-design.md).
- A BODY CLOCK (a circadian law) shapes sleepiness and hunger over the day.
- There is no separate autopilot layer and no authored routines. Odd habits, broken habits, and habits kept after they stop serving all follow from learning.

## SC2. Idle and background life

It emerges from:
- the drives that are never fully met: company, curiosity, boredom, standing, and acquired concerns (tending what one cares about);
- FORESIGHT (SC3);
- CULTURE: shared schedules held as beliefs (market day, the evening meal, rest days), so a village's rhythm emerges;
- PERSONALITY DIALS (novelty-seeking, need for company), so who wanders, who talks and who sits alone reveals who they are.

## SC3. Foresight

Foresight comes from five things:
- the world model's lookahead (near term);
- learned routines (seasonal habits: last winter hurt, so stack wood now);
- CULTURAL KNOWLEDGE passed down as beliefs ("store grain before the snows"), which carries lessons nobody alive lived through, and can be forgotten;
- explicit plans through the plan cache;
- a DISCOUNTING DIAL per person (present bias, real and uneven).

The owner's addition: some or many people can PLAN LONGER AT A HIGHER COST TO BODILY ENERGY.
- Deliberation depth is a choice that costs time, MENTAL FATIGUE (a fatigue drive) and a little energy.
- The capacity is a per-person dial, raised by training (use-dependent plasticity, magic MC3).
- Tired or stressed people plan shorter.
- Check the mental-effort literature at build time: the cost is mostly subjective fatigue, and brain energy use changes little.

## SC4. Affording background life: level of detail

- People near the player or salient (in the scene, involved in events, or the focus of someone's attention) run in FULL.
- Everyone else runs at the ROUTINE LEVEL: habits and schedules advance coarsely, with drives and needs booked in summary.
- On meeting someone, their details are MATERIALISED CONSISTENT WITH WHAT WAS ALREADY SEEN (alibi generation: if you watched the guard leave at dusk, he was not home all evening).
- The coarse version is checked against full runs, as for offscreen knowledge (K9) and the family's past (Fam2).

## SC5. One approximation framework: the twin contract, generalised

The owner: "we need a way to approximate full simulations closely". The night fast-forward already has a TWIN PROTOCOL (src/voxel/slowtwins.py; the breath twin is exact to float rounding over 432,000 ticks). It is generalised to every coarse version (night jumps, the coarse past, offscreen knowledge, background life, speed-ups):
- `applicable`: when the twin may run;
- `horizon`: when fine detail is needed again (a threshold about to flip, an OBSERVER, a rare event). The coarse version always knows when to hand back to the fine laws, so the moments that matter are simulated in full;
- `advance`: coarse progress with EVERY LEDGER BOOKED;
- for people and things, materialisation consistent with what was seen;
- TWO GRADES: EXACT twins (bit for bit) and APPROXIMATE twins (checked on facts: distributions, invariants and interventions, within a stated error budget, per the optimise-first rule);
- ONE SHARED VALIDATION HARNESS that runs coarse against full and reports drift.

## SC6. The player character's self-care

- The character has their OWN HABITS AND ROUTINES, which handle routine needs by default.
- The player can ALWAYS OVERRIDE ("skip dinner, keep watch").
- The player is TOLD when a need becomes pressing or a habit cannot be met.
- A DIAL runs from mostly automatic to fully manual, for survival packs.
- One interface for everyone (PA4).

## SC7. When

- WAVE 15, batch 2: the twin framework starts with tick against jump (the shut-room star), with the shared harness.
- WAVE 16a: the body clock; habits and routines as chunked primitive sequences; the player self-care dial.
- WAVE 16b: cultural schedules as beliefs; foresight from cultural knowledge.
- THE FAMILY STEP (lab): the coarse past as the first approximate twin, checked against a full run.
- LATER, with several settlements: level of detail for background life, alongside the offscreen knowledge summary (K9).
