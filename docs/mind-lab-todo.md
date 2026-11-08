# Mind lab: to-do list

In order of priority. Updated 2026-09-30. The chooser we are settling on is
**GLiNER2.5-Decide** (340M, about 30 ms a choice); the big models are
references only. History and results are in `docs/voxel-roadmap.md` (entries
100-104).

## Plan v4: done (queue run 2026-09-29, 22:34-23:28). Results in roadmap 104.

Reports: `lab/results/methods/stages.md`, `prior.md`, `learnable.md`,
`state.md`, `examples.md`.

## Next (from the v4 results)

Each item says what it runs on:
- **CPU:** on data we already saved. No GPU. It can start now.
- **GPU:** a short job. I add it to a queue; you start the queue.
- **Yours:** off this machine.

"Done when" is the number that says the item worked.

**Done 2026-09-30, 21:57-23:34 (roadmap 106):** B6, B2, B4, B1, A7, B5. The
main finding: GLiNER-Decide matches what the scene NAMES (a named useless
thing is picked 100%), and with a trait sentence it ignores the scene
("state-blind"). Writing the input as data works (what is missing: 39% to
82%; the verb's capability: 62% to 75%); asking the model to infer or
appraise does not.

**Next, in order:**
1. [ ] **D1. Fine-tune data against the mention lure** (CPU, for C1): add
   named-but-useless things as hard negatives, and yes/no questions whose
   answer is "no". The fine-tune itself is yours (C1).
2. [ ] **D2. Input rules for the engine** (CPU, a short doc + code in the
   inn's `describe`): state what is missing, never name what should not be
   picked, the capability from the verb, the relevant sentences only. Then
   test on the inn snapshots.
3. [ ] **D3. The inn with the trait-bias policy + the D2 input** (GPU,
   about 15 minutes for 10 nights). Drop the appraisal policy, or retry it
   with a soft guard and a smaller g.
4. [ ] **D4. The capability table for every verb in the catalog**, written
   from the catalog, then tested on new needs (the B1 table was written
   knowing the test needs).
5. [ ] **D5. Multi-hop questions from sim worlds** (CPU to generate; the
   training is yours). From PhantomEnvironments (Kabra et al. 2026,
   arxiv.org/abs/2609.40221): agents trained in rule-made fictional worlds
   transfer to real tasks, and the number of reasoning hops drives the
   transfer more than anything else they varied. Our weak kinds are
   multi-hop: danger 39%, fault 0%, false belief 10% (roadmap 106).
   - Needs the cause log (`why()` chains, sim-mind wave 7). Then generate,
     from harvest and inn worlds, questions at 1, 2 and 3 hops whose answers
     the sim guarantees ("whose fault is it that Tom fell?" = follow the
     cause links), with the D1 hard negatives mixed in.
   - Done when: a model trained on 2-3 hop items beats one trained on the
     same number of 1-hop items on the danger, fault and false-belief
     battery (`lab/methods/primitives.py`), with 95% ranges that do not
     overlap.
- B3, B7, B8 are on hold: few question kinds pass (place, goal, norms), so a
  chain of model answers would mostly carry errors. Facts come from the sim.

### A. CPU, on saved data: done 2026-09-30 (roadmap 105)

- [x] **A1. A list-wise head** on the frozen option vectors: 67% on new
  needs (GLiNER 61%, per-option head 66%); near-miss lists 57% (51%). It did
  not reach 70%: a new head alone does not unlock the choice. The fine-tune
  (C1) is the next lever. (`learnable.md`)
- [x] **A2. The personality guard:** a trade-off, not both. Appraisal with a
  hard guard (g 2, d 0.5): shift +0.16 at about 70% valid; (g 2, d 0.3):
  +0.38 at 51-60%. New: with no persona, the DRAW puts only 43% on valid
  options (the pick: 76%); a hard guard raises it to 67%. Only 15 scenes.
  (`combine.md`)
- [x] **A3. "Calls for" as a weight:** works where the calls differ. Head's
  call: near-miss lists 28% to 50%, unrelated 50% to 60%, 8 options 60% to
  67%, 16 options 38% to 40%. Right call: 62%, 85%, 75%, 67%. It cannot pick
  the PERSON (wrong-person lists 40% to 25%). (`combine.md`)
- [x] **A4. The prior correction** is a chooser flag:
  `LAB_PRIOR=lab/results/methods/prior_offsets.json` (7,918 items). Checked
  against `prior.md`.
- [x] **A5. Relevance:** the model ranks the sentences perfectly (AUC 1.00);
  the threshold was wrong. Keep at p(yes) 0.3 or more: every sentence that
  matters is kept, 67% of the filler dropped. B4 is now worth running.
- [x] **A6. Stage heads with needs held out:** capability AUC 0.92, top pick
  55% (GLiNER 58%); "calls for" AUC 0.86, top pick 75% (GLiNER 61%).
- Also found: the per-person role question (22%) asked about people who are
  not in the scene text (125 of 175). Named people: 56%; unnamed: 9%.

### A. Next CPU jobs

- [x] **A7. More personality scenes** (done: temper +0.24 appraisal, +0.14 text; kindness +0.20 both; `stages personality2`, 140 lists, temper 15 scenes + kindness 20 scenes). A2 rests on 45 lists from 15 scenes.
  Build lists from all 35 people scenes and the inn snapshots, with confront
  and avoid options, and with the draw's validity reported. CPU (building).
- [ ] **A8. The personality guard on the draw.** The inn SAMPLES from the
  distribution. Tune the guard for the draw's validity (now 43%), not only
  the top pick. CPU, seconds, after A7's GPU run.

### B. Short GPU jobs (queued; you start them)

- [x] **B1. The capability from the verb data** (done: 62% to 75%, 47% to 60%, equal to the oracle; `stages capability2`; the verb table covers 33 of 38 needs and holds a right phrase for 31).
  - What: a table "verb → capability phrase" for every verb in
    `docs/action-catalog.md` ("cut: something with a sharp edge"). The need's
    hint is looked up from its verb. A verb with two senses ("open the
    chest": pry it or pick the lock) gets two phrases.
    - Write the table from the catalog, not from the 38 test needs, so the
      test stays fair.
    - First, a CPU check: how many test needs the table covers, and how
      often its phrase is the right one.
  - Why: the right hint gives 62% to 75%. The model's own hint gives 58%
    (right 60% of the time; a wrong hint drops the choice to 27%). It must
    be right about 73% of the time to break even.
  - Runs on: GPU. The capability stage with a new "lookup" arm, 120 cases,
    about 5 minutes.
  - Done when: the lookup arm reaches 72% or more on 16 options.
  - Then: the hint becomes part of the things question, and of the inn's
    choice question.
- [x] **B2. The mention lure** (done: missing only 39% to 82% and 31% to 55%; a named useless thing is picked 100%; `cases_state2.jsonl`, `cases_lure.jsonl`; report `lure.md`).
  - The failure: GLiNER picks what the scene names. "Dry wood is laid in the
    hearth, but it is not lit" gives the firewood 100%. "Kael has a quill
    and ink, but nothing to write on" gives the quill and ink 100%.
  - What: the same state cases in three wordings:
    - as now (what is there, and what is missing);
    - only what is missing ("The hearth has no flame yet.");
    - what is missing, first ("He still needs a flame. Dry wood is laid
      ...").

    6 needs × 2 steps × 30 lists × 3 wordings, about 1,080 cases, about 8
    minutes.
  - Also a general lure test: add one sentence that names a useless thing
    from the list ("A broom leans by the door.") to the plain things cases,
    and count how often the model picks it. About 240 cases, 2 minutes.
  - Why it matters for the game: the sim knows every precondition. It can
    write what is missing and leave out what is done. If "only what is
    missing" fixes the choice, the model never has to work out the next step.
  - Done when: the right step is 70% or more in both steps (now 39% and 31%).
  - Then: the engine writes state as what is missing. The mentioned-but-done
    things go into the fine-tune data as hard negatives (C1).
- [ ] **B3. People: WHOM, then what.** A3 showed the call weight helps
  where the options differ in kind, but it cannot pick the person. Add a
  person weight from a "who" question asked about people NAMED in the scene
  ("Who is in danger or hurt?", "Who is doing wrong?"; B6 measures these),
  times the call weight. 300 cases, about 5 minutes. After B6.
- [x] **B4. Relevance re-run with the fix** (done: 42% to 48%, the oracle 47%): keep sentences at p(yes) 0.35 or
  more, and always keep the need sentence (A5 found the threshold). Expected:
  near the oracle filter (47% against 42%). 120 cases, about 5 minutes.
- [x] **B5. The inn, with personality from appraisal** (done, it FAILED: an accuse loop, 2,675 in 10 nights; the take/drop fix cut the bias policy's loop 76%; policy `appraise:M` in `lab/world2/policy.py`; the take/drop fix is in; results in `lab/results/world2b/`).
  - What: a new policy. At each choice, GLiNER answers "How does Tom feel?"
    and "What does Tom want to do?" for the scene as it is now (2 more
    calls, about 60 ms). The choice uses the answer, with the guard from A2
    and the prior correction from A4.
  - First, the inn fixes (CPU, code; see "Next: the inn", item 6): "drop"
    right after "take" counts as a repeat (the take/drop loop), and the scene
    is filtered to what bears on the drives.
  - Compare on the same seeds with text:M (0.3% surprising and in character)
    and bias:M (10.4%; odd 11.6%). Also count the confront share for each
    trait in the log.
  - Runs on: GPU, about as long as one earlier inn model run.
  - Done when: surprising-and-in-character is at least bias:M's, odd is 12%
    or less, and the traits show in the log.

- [x] **B6. The question battery** (done; `primitives.md`) (`lab/methods/primitives.py`; the cases
  are built: 2,307 questions, 12,172 cases). Which kinds of question
  GLiNER-Decide answers reliably, so that chains are built only from those.
  - 17 groups with known answers (presence, possession, place, who did it,
    to whom, inner state, danger, property, use, goal, event kind, roles,
    fault, norms, calls for, false belief, next step) and 4 with unknown
    answers (feeling, urge, coping, approach).
  - Each question in 3 wordings, reversed order, with filler, and (names,
    things, places) with a lure sentence. Pick-one against yes/no; person
    against role.
  - Unknown answers: direction (a trait sentence must raise the expected
    answer), stability, and a trait x scene grid with an empty scene
    ("state-blind" check).
  - Runs on: GPU, one GLiNER call per case, about 10 minutes.
  - Done when: each group has a verdict. Proposal: trust it at 85% or more
    right (yes/no; 70% for pick-one) and 80% or more the same across
    wordings; a soft weight between that and 60%; do not use it below.
    `sure: right` says whether a confidence gate rescues a weak group.
  - Then: B7.
- [ ] **B7. A line of questioning from trusted kinds only.** Chain the
  groups that pass B6, and pass each answer on as a soft weight, behind a
  confidence gate, never as a hard filter. Where the answer is unknown,
  keep the top few answers and weight the final choice by their
  probability (marginalise), in place of taking one answer. Test the whole
  chain with four arms: no answers, the model's answers, the right answers,
  shuffled answers. GPU, after B6.
- [ ] **B8. Disposition before the choice** (your idea: "How does Kael
  feel?", "What would Kael try to do?"). The approach answer ("force it",
  "look for the right tool", "ask someone to help") as a soft weight on the
  things and people choices, with the guard. It needs B6's direction and
  state-blind results first.

### C. Yours

- [ ] **C1. The LoRA fine-tune** (`lab/methods/finetune_gliner.py`). Not
  started here; you may run it on another machine.
  - The data is built: train 1,131, eval 125, and 485 held-out lists (8 needs
    and 15 things never seen in training).
  - Before: rate `nearmiss_audit.md` (60 pairs). If B2 has run, add its hard
    negatives.
  - Bring back the adapter folder. `LAB_GLINER_ADAPTER=<folder>` loads it in
    our runner; then run the held-out lists.
  - A1 did not reach 70% (67%), so the fine-tune is the next lever for
    choosing.

### Closed (no gain; not pursued)

- Kind first as a soft weight (λ = 0.5: 63% against 62%).
- The "Will that work?" re-check of the top 5 (63% against 62%).
- The per-person role question as a sentence (22% right), unless A3 trains
  it.
- The speech demo. It cites scene fragments ("Help! runs off through the
  crowd."). It waits for Workstream G (the line bank).

## Paused older queues (big-model references, optional)

- [ ] `lab/methods/resume_all.sh` restarts them where they stopped:
      - the presentation battery: onli2b, qwen17:two, bev:hier, laya_en:hier;
      - people: qwen17:two, laya_en:two, onli2b;
      - places: every model except GLiNER-Decide.

## Next: GLiNER-Decide (sense)

1. [ ] **The winning formats together.** Key-value need + capability hint +
       short option names. Change one thing at a time from the default, then
       all together. Things cases, 16 and 64 options. (`lab/methods/present.py`)
2. [ ] The capability line for each verb: now B1 above.
3. [ ] The same hint for people ("calls for"): now A3 and B3 above.
4. [ ] **Places:** check that the capability hint helps there too.

## Next: the inn (sense, then personality)

5. [ ] **A bank of about 150 decision snapshots** frozen from the inn logs,
       labelled offline with acceptable sets. The scene formats and the
       personality measures both need it.
6. [ ] **Fixes to the inn:**
       - [ ] filter the scene description to what bears on the drives
       - [ ] recent events as short lines
       - [ ] "drop" right after "take" counts as a repeat
       - [ ] the capability hint in the choice question
7. [ ] **Scene formats on the snapshots:** prose, entity cards, events as
       lines, filtered; options grouped by target; whom first, then what.
       (`docs/presentation-plan.md`, step 2)
8. [ ] **Personality** (`docs/personality-design.md`, in its order):
       - [ ] baselines and the sign of the persona effect on the snapshots
       - [ ] trait features from cached NLI entailment ("This is a brave act")
             in place of hand tags
       - [ ] learned weights, γ drawn per scene, the plausibility guard
       - [ ] guidance on option scores, then a steering vector per trait (Qwen
             only)
       - [ ] trained conditioning, off-laptop

## Diagnostics (why the models fail)

9. [ ] An **affordance quiz and probes** ("does a sword cut?"), to learn
       whether GLiNER-Decide does not know, or knows and does not use it.
       (`docs/mech-interp-review.md`)
10. [ ] **Activation patching** on the failure pairs (the rope for "cut the
        rope").
11. [ ] **Laya:** block attention between options, to test why it collapses on
        long lists. Lower priority now that GLiNER-Decide is the chooser.
12. [ ] **A per-option prior table** for the one-at-a-time models (openjev,
        Qwen: +8 to +10 points). It does not help GLiNER-Decide.

## Waiting for you

- [ ] Label audits:
      - [ ] `lab/results/methods/nearmiss_audit.md`: are the hand-picked near
            misses really wrong? (60 to rate; do this before the fine-tune)
      - [ ] `lab/results/methods/creative_labels.md` (the unusual tools)
      - [ ] `lab/results/methods/label_check_people.md`
      - [ ] `lab/results/methods/label_check_places.md`
      - [ ] `lab/results/methods/label_check.md` (things)
- [ ] Notes on `docs/action-catalog.md` (its section 6 questions).
- [ ] Plan v3, section 14 questions: the speed rule, the rating tool, the
      traits, speech.
- [ ] The pass thresholds in `docs/personality-design.md`.

## Later (plan v3)

- [ ] Workstream C: the judge, taught in rating rounds. It adds "makes sense"
      to layer 3.
- [ ] Workstream G: speech (start with the line bank).
- [ ] Workstream H: undefined effects and things made when needed, beyond the
      chest.
- [ ] Non-text presentation: entity vectors and a slot head
      (`docs/presentation-plan.md`, step 3). Factorized scoring is the zero-shot
      step.

## The voxel simulation (paused for the mind lab)

- [ ] Meshes into the world: quick mesh generation, then CuACD (convex
      pieces on the GPU, https://github.com/eliphatfs/cuacd), then MPM
      material.
- [ ] Taichi (https://github.com/taichi-dev/taichi) to run our MPM on the GPU
      and render it. Our MPM is NumPy on the CPU (`src/mpm/core.py`).

- [ ] Dragging a turned unconscious man leaves 2 foot cells behind.
- [ ] A man wedged at the world edge tips again about every 45 ticks.
- [ ] Roadmap 95, step 5: light.
