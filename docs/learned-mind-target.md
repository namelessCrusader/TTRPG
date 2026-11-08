# What the learned mind should learn toward, and how we know

**Provenance.** This synthesis rests on six design write-ups and two judge reports. I did not re-read the repo or re-check citations. Compute numbers are the designs' own estimates, plus my arithmetic from them. Items marked (u) were not verified by their authors.

**Revised 2026-10-30.** Section 9 replaces the fixed `F_body` as what selection optimises (it becomes a report lens). It adds acquired values, emergent dominance, observer lenses, modelled senses and priming, and it supersedes question 1 of section 8.

---

## 1. The answer in one paragraph

There is no ground truth of "appropriate behaviour" to look up, and we should stop looking for one. You are right on both counts. A list of human facts used as a target is circular. As a fixed list it is also an if-else ladder in disguise. So the target is not a behaviour. The target has three parts.

1. **A fixed bodily fitness.** It covers staying alive, fed, safe, rested and solvent. The mind's genome cannot change it.
2. **A world whose consequences are checked.** Gossip, reprisal and reputation must really cost or pay.
3. **An optimization loop.** A slow outer loop searches the mind's inherited "nature". The fast inner loop is the lifetime learning the mind already has. Human-like conduct is a prediction we test on probes that selection never saw. It is never a thing we fit.

The arbitrary part does not disappear. It shrinks to two short, readable places: the fitness definition and the consequence table. We can argue about those, ablate them and publish how sensitive results are to them. That is much better than a list of 40 behaviours.

## 2. Recommended architecture

**Backbone: "two loops, as nature did it" (design 1).** Four other designs contribute only the parts that fit your principles.

| Part | Source | What it does |
|---|---|---|
| Outer loop | Design 1 | Evolution strategy over a small genome. It uses common random numbers (every genome meets the same world seeds) and risk-averse fitness (mean minus SD). |
| Inner loop | Existing mind | Unchanged. It learns online for the whole life from the genome's starting state. Learned content is not inherited. |
| Choice step fix, only if needed | Design 4 | One currency for the choice instead of seven terms mixed by hand-tuned weights. This is conditional on step 1 below. |
| Gates and canaries | Designs 1, 4, 5 | Oracle gap, archetype tournament, belief-swap test, planted loophole. |
| Hashed pattern suite, report-only | Design 6 (validation only) | Human patterns are never fitted. |
| Social learning, much later | Design 3 | Founders are earlier generations of evolved minds, not authored elders. |

**What we write and what emerges**

| We write (below the capability) | Emerges |
|---|---|
| World, menus, reflexes, drives, learning rules | Who trusts whom, reciprocity, wariness, retaliation |
| Fixed bodily fitness F_body, with the same weights for every genome | Drive weights and set points, if we evolve them in a later step |
| Consequence table (gossip, reprisal, trade dependency) | Link strengths from partner models to choice, and learning rates |
| Bounds on the genome | Individual differences (nature plus a lived life) |

**Two measures kept apart.**
- Reward is the drop in the genome-weighted drive error. The genome shapes it.
- Fitness is a fixed canonical bodily error. The genome cannot touch it. If fitness used the genome's own weights, the best weights would be zero.
- Company and standing stay out of F_body. They go into a report-only F_wellbeing. If a mind comes to care about them, it is because they paid through trade, gossip or protection.
- Death counts as maximum error for every remaining night.
- F_body weights are the mean of today's NATURE table. We also test two alternative weightings.

**Switched off in the first pass**
- The genome has no acts and no rules in it.
- Group selection is off. It smuggles in pro-sociality.
- Default-trust and initial-valence priors (Baldwin genes) are off. They sit close to naming the behaviour. They come back later as a reported arm.
- Nothing pretrained chooses anything.

## 3. What the world must provide

| Need | Why | Check |
|---|---|---|
| Cooperation can pay: benefit b, cost c, gossip reach q, meeting-again chance w | Indirect reciprocity works only if q > c/b (Nowak and Sigmund 1998). If this fails, no search can help. | Measure from event logs before evolving. |
| Others hold what you need: food, protection, trade, information | Without it, the best genome is a loner. | Archetype tournament. |
| Consequences that travel: gossip, reputation, reprisal, refusal to trade | Restraint and kindness must be the economical choice. | The reputation-off ablation (step 5) must change evolved weights. |
| No free actions | The interrupt loophole paid in drive error without a world cost. Evolution finds such holes better than PPO (Lehman et al. 2020). | Planted honeypot, ledger audit, held-out world variants. |
| Invariants the engine enforces | Items conserved, harm acts take at least HARM_MIN turns, one act per person per turn, no write path from mind to fitness. | Engine tests. |
| Shocks from the world's own laws (theft, bandit, scarcity, a partner who changes) | Fitness must cover routine and crisis nights. | Report both separately. Rate 10 to 20 percent, swept 5 to 50 percent. |
| Time scales | The inner loop (100 to 1,000 nights) is long enough to learn a partner. The outer loop is 15 to 25 generations. | Genes needing more than about 100 generations are out of reach. |
| A per-person nature override | Reward weights and fitness weights must be able to differ. This is a small change to `world.py` `drive_error` and `drives.py`. It is a proposal, not made. | Unit test. |

**Round 2 is a precondition.** In round 1 the partner models learned (hit-back +0.247, payback +0.118) but valence stayed 0 and behaviour did not follow. Evolution can tune link strength. It cannot create a missing pathway. Nothing below is interpretable until the core passes L2.

## 4. Stability: criteria and measurement

Calibrate every threshold once on seed 0 and on the `rules` noise floor. Fix them in writing. Then evaluate on fresh seeds. The numbers below are starting values.

| # | Criterion | How measured | Pass |
|---|---|---|---|
| S1 | Seed divergence | CV of F_body and death rate across life seeds. Compare with the round-2 mind2 baseline. | No worse than mind2. Round 1: 3 deaths in 10 lives for the mind, 43 to 110 for the others. |
| S2 | A self from a life, not dice | Jensen-Shannon divergence of act shares per 50-night block. D_rng is same genome, same partners, different mind RNG. D_life is swapped partner roles (L1). | D_life / D_rng at least 2. D_rng no more than 1.5 times the `rules` noise floor. Deterministic copies also fail. |
| S3 | Drift | JS between consecutive 100-night windows after burn-in. Share of likings pinned at ±1. | Drift no more than 1.25 times the noise floor. Pinned likings under 10 percent. |
| S4 | Perturbation response | Twin runs, one forced betrayal at night 100. Track treat(X). | Moves at once, half-life 3 to 60 nights, under a quarter of peak by +100 nights. Rules out both "no memory" and "chaos". |
| S5 | Behavioural bounds | Act logs | Harmful acts per night no more than 3 times `rules`. No verb above 40 percent of acts for more than 10 nights in a row. |
| S6 | Hyperparameter robustness | Grid of key settings at 0.5x, 1x, 2x | The sign of each main contrast holds in all cells. |
| S7 | Outer-loop reproducibility | At least 3 independent evolution runs. Distance between runs measured on behaviour (act profile and treat profile), not on parameters. | Median between-run distance at most half the median between random genomes. If it fails, the fitness does not pin down a nature. That answers "the target is arbitrary" with data. |
| S8 | Exploit audit | Honeypot (reopen the interrupt hole) and ledger audit (share of drive-error drop with no matching resource flow). A tabular bandit with unlimited exploration hunts for verbs. | Honeypot gain no more than `rules`. Unexplained share under 5 percent. |

Stability is also pushed into the loop itself. Genes are bounded. Fitness is risk-averse. Each evaluation draws a world variant (reprisal ±30 percent, scarcity ±30 percent, cast, shock rate). A hall of fame of past opponents guards against cycling.

## 5. Validation without circularity

**The rule.** Nothing human is ever in the loss. Selection sees only F_body. Everything below is held out by construction. "Held out" means the loop never touches it. Fresh seeds alone do not count.

**Held-out layers, from most to least author-independent**

| Layer | What | Notes |
|---|---|---|
| People | Blind side-by-side ratings (L6) of evolved-cast nights against `rules` and mind2. A "can you tell?" test mixing your own play with the mind's. | The only judge that is not us. Use it on a shortlist, not as a search. Raters' acceptable-option sets are pass bands. |
| Books | Genre contrast only. Seed or compare per genre. The DCC-lineage cast must score better on DCC scenes than on cultivation scenes, and the reverse. | Contrasts cancel the sanitising bias. Levels do not. It needs a mapping from inn acts to book options, so it is optional and late. Books also audit the genre pack's consequence table. |
| Pattern suite | At least 20 hashed, pre-registered, mostly ordinal patterns, all report-only. | See below. |
| Causal world ablations | Evolve with one consequence removed (no gossip and reputation, no reprisal, no trade dependency). | Evolved weights on company and standing must shrink. No human number needed. |
| Our own probes | L1 to L5, shuffled-answer control, P-deaf, free-threshold ladder control. | Fresh seeds. Never used in selection. These test mechanism. They cannot show appropriateness. |

**The pattern suite.** It borrows only the discipline from design 6.
- Commit a registry before any run, and quote its hash in every result.
- Each entry has a source, a log-to-statistic mapping, a direction, a tolerance and a falsifier.
- Prefer shapes, ratios, signs, rank orders and genre contrasts over levels.
- Skip patterns that cannot be mapped to the inn, and list them. Never force a mapping.
- Keep a look ledger. Count every look and apply Holm correction. Burned patterns are retired.
- Run an alien-pattern control: a sign-flipped registry. A pass means something only if the alien registry does clearly worse. Below about 20 entries this is not a test.
- Pass means "consistent with these patterns". It does not mean "appropriate". Shared bias in lab data (mostly WEIRD) is named and not removed.

**Day-to-day sources (unsanitised)**
- The time-use shares and their ratios. ATUS 2023 gives sleep 9.07 h and eating 1.20 h per day. Use ratios and bout shapes. The inn night is not a human day.
- A solved single-character optimum. Value iteration on the sim's own drive dynamics gives the best eating and resting schedule. Measure the mind's regret against it. This needs no paper.
- Conversation content: the social share of talk is contested, 60 to 85 percent (u).
- The L4 need-dependence index.

**Surprise-point sources**
- Book category profiles and genre contrasts.
- The sim's own betrayal, theft and loophole probes (S4, L3).
- Cross-regime transfer: whatever is fitted or tuned on one regime must not wreck the other. We do not fit patterns, so this applies to any tuning we do.
- Report ordinary nights and shock nights separately. A mind that is jumpy all day must show up.

**Answer to the if-else worry.** Run a control. A `rules` chooser with free thresholds gets the same evaluation budget as the evolved mind. Compare both on held-out probes. If the ladder matches the mind, the mind adds nothing, and we say so.

## 6. ROUND 3 plan for lab/mindL

Stop at the first failure. Each failure names its own fix. Pick two or three primary criteria per step. Report the rest as diagnostics. Score paired by seed. Round 1 intervals were about ±0.25 against effects of 0.1, so power is low.

Cost basis: a `rules` night is about 0.2 s. A focal mind plus four `rules` is about 0.35 to 0.45 s a night (u). Multi-mind casts are 1.0 to 1.5 s a night (u). Measure one lifetime of each cast type first. Run under the existing lock and cooling gate, nice, at most 4 workers, while the two engineering workflows run.

| Step | What | Pass rule (primary) | CPU cost | Wall time (shared laptop) |
|---|---|---|---|---|
| **0a. Does the world pay?** | Oracle gap in L2: partner-blind chooser against an oracle that knows partner rules. Archetype tournament (hermit, always-harsh, always-giving, tit-for-tat, rules). Measure b, c, q, w from logs. | Oracle gap at least 0.1 with CI above 0. Hermit and always-harsh lose to `rules`. Always-giving loses to tit-for-tat. q·b > c. | Minutes to about 1 hour | Same day |
| **0b. Write the yardsticks** | Pattern registry v1 (at least 20 ordinal entries) plus the alien registry. Hash it. Draft the S1 to S8 scripts. | You approve it. No CPU. | None | 1 to 2 days of authoring |
| **1. Read round 2, then belief-swap test** | Read the round-2 report. Swap the mind's posterior about Fenn with Oren at decision time and measure the change in the action distribution. Set the noise floors from round-2 baselines. | TV at least 0.1 for kind-vs-hard, and under 0.03 for P-deaf. TV high means the choice rule is fine, so skip step 2. TV low means the beliefs are unused, so do step 2. | About 1 to 2 hours (u) | Same day |
| **2. Fix the choice rule (conditional)** | Arms G1 and G2 against mind2 on fresh seeds 21 to 30: single currency, leaky Beta partner models, volatility-gated forgetting. Tune only on seed 0. | Belief-swap TV as above. L2 contrast CI above 0, paired, and better than mind2. L0 error within 0.3 of `rules`. If both fail, the problem is the world's reprisal timing and witnesses, not the rule. | About 6 to 12 CPU hours | 3 to 5 hours at 2 to 3 workers |
| **Core gate** | The core passes L2. | Nothing below is interpretable without this. | | |
| **3. Freeze the yardsticks** | Run S1 to S8 on `rules` and mind2. Set thresholds in writing. Build the free-threshold ladder control. Plant the honeypot. | Thresholds written down before step 4. Honeypot flags the known interrupt hole. | About 3 to 4 CPU hours | About 1 hour |
| **4. Small evolution** | Evolve about 8 link and plasticity genes: b_partner, b_trust, tau (or lambda and eta), lr_actor, lr_partner, pm_forget, habit_fade, replay strength. Drive weights stay hand-set. Population 10, about 15 generations, 5 life seeds of 100 nights, 1 focal plus 4 `rules`. 3 independent outer runs. Ladder control gets the same budget. | On held-out seeds at 300 to 400 nights: evolved F_body beats hand-set mind2, paired CI above 0. L2 contrast CI above 0 and above the matching null. S7 passes. | About 8 CPU hours per run, about 25 for three. Ladder is cheaper. (My arithmetic from the designs' figures.) | About 6 to 7 hours. One overnight job. |
| **5. Does the world select social drives?** | Only if step 4 passes. Add the 14 reward genes. Evolve in W_rep (all2) and W_norep (gossip, watch and check off). 3 outer seeds each. | Evolved company-plus-standing weight is higher in W_rep than in W_norep in all 3 pairs, and the paired CI excludes 0. If weights are near 0 in both, the world does not pay for sociality. That is a world finding. | About 40 to 60 CPU hours (u, extrapolated) | Two overnight runs |
| **6. Held-out report** | Frozen genomes on the registry, S-battery, honeypot, L3 and L5. Register the look count. | Report only, with the alien control. | About 4 to 6 CPU hours | About 1.5 hours |
| **7. People and books** | Blind L6 over 4 to 5 versions. "Can you tell?" night set. Book genre contrast if the act mapping exists. | Raters cannot reliably separate evolved nights from your play, or prefer them to `rules`. | Small | One sitting of your time |

**Later, each with its own gate:** one genre pack for the inn (cultivation), then the home-advantage cross-transplant. Mind-versus-mind casts, then generations watching earlier generations (design 3). That needs a multi-learner inn that is cheap enough.

**Failure readings for step 5**
- P0 holds, sociality does not evolve: the world does not reward it. Fix the consequences.
- Drives evolve, but L2 still fails: architecture, not evolution.
- Stability fails: the fitness lets exploits or instability pay.

## 7. What not to do, and the risks

**Do not**
- Fit anything to human numbers, or put a pattern in a reward, rule or menu filter.
- Use scripted caregivers or elders as targets. They author the early behaviour (designs 3 and 5).
- Build the tiered thinking cascade (design 2). It is a hand-built router close to a special-case subsystem. Its P3 checks are true by construction.
- Use simulation-based inference fitting (design 6). It is the circular approach you rejected.
- Tune and grade on the same quantity. Design 5 did this with S1 and S2.
- Let selection see L1 to L5, the honeypot or the registry.
- Evolve before the core gate passes.
- Read results from a world without a recent exploit audit.
- Move thresholds after seeing results.
- Claim "human-like". The defensible claim is: **evolved conditional reciprocity and reputation concern, stable under the tests, in a world whose consequences were checked.** Unconditional altruism, fairness and moral emotion are not predicted.
- Run heavy jobs while the other workflows and round 2 hold the machine. Only step 0 is light enough.

**Risks**
1. **F_body is the new arbitrary choice.** If it is wrong, evolution optimises the wrong thing very efficiently. Mitigation: it is short and readable, and we run two alternative weightings. This is also the "wealth is a value judgement" problem.
2. **Exploits.** Evolution is a better hole-finder than PPO. Mitigation: invariants, domain randomisation, honeypot, ledger audit. Someone should look at replays of every top genome.
3. **Noisy selection.** 100-night lives select for early learning, and 300 to 400 nights are needed for drift. Mitigation: 5 seeds, common random numbers, held-out at 300 to 400 nights, S7.
4. **Architecture is the bottleneck.** Handled by the core gate and the belief-swap test.
5. **Low joint power.** Conjunctive pass rules over 10 seeds are weak. Mitigation: 2 to 3 primary criteria and paired scoring.
6. **Only conditional cooperation is expected.** An Austen heroine needs more than one pack and probably group structure. That is future work.
7. **Genre packs are authored.** The books only audit their tables.
8. **Search cost is hidden cost.** Report generations and CPU hours with every result.
9. **Co-evolution can cycle or collapse into a monoculture.** Individuality is a goal, so collapse counts as failure.
10. **Unverified numbers.** Fehr and Gächter figures, Johnson and Mislin proportions, Roberts and DelVecchio coefficients and Dunbar splits are all (u). None goes into the registry until checked at the source.
11. **Compute figures differ by about 3x across the designs.** Measure first.

## 8. Open questions for you

1. **What is a good life (F_body)?** Recommendation: alive, fed, safe, rested, solvent, with the same weights for every genome. Company and standing are excluded, so that caring about them has to pay. Confirm that "solvent" belongs in. It is a value judgement.
2. **Do you accept the narrow claim?** Recommendation: yes. "Conditionally reciprocal, reputation-aware, retaliating, individually different, stable" is the target. Fairness and moral emotion are out of scope for now.
3. **How much of your time for human judging?** Recommendation: one sitting. Rate 4 to 5 blind versions, plus one "can you tell?" set that includes nights you play. Agree the protocol before any run.
4. **Who picks the pattern registry?** Recommendation: I draft 20 to 30 ordinal and contrast patterns. You veto before the hash. None is ever fitted. Entries that cannot be mapped to the inn are listed and skipped.
5. **When do we build the genre pack?** Recommendation: after step 5 passes, and just one (cultivation, where insulting a senior is lethal). The books then audit its consequence table, and the home-advantage test becomes possible.

---

## 9. Values, dominance and whose good

**Provenance.** Values study, 2026-10-30. Its four design drafts were lost to an output error, so this is the synthesis agent's own reading of this doc, `docs/learned-mind-plan.md`, the mindL brief and `lab/world2/world.py`. The main session checked its claims about the code and corrected two (the contest row and the wear rule). Sources are cited from memory and not re-checked, so all are (u). Compute figures are arithmetic from the earlier sections' numbers (u). Sections 9.8 and 9.9 add your points from the same day.

This section replaces the fixed bodily fitness `F_body` of sections 1 to 3 as the thing selection optimises. The rest of the earlier plan stays, except where section 9.6 says otherwise.

---

### 9.1 The answer in one paragraph

You are right, and the earlier section was wrong to call `F_body` "the good life". It made one answer to "what is good" (alive, fed, safe, rested, solvent) and applied it to every mind. That cannot cover someone who starves for a cause, or someone who grows rich by trampling others. So we split three questions that were mixed together.

| Question | Who answers it | What it is |
|---|---|---|
| What stays in the world? | Selection (the outer loop) | A count of persistence. It does not weigh needs. |
| What does this mind count as good? | The mind itself | Inborn drives, plus values it acquired in its own life. |
| Is this life or this world good? | An observer, from outside | A lens over the logs. It is never a reward and never a target. |

"Good for the whole" and "good by a creed" are lenses. A character can also hold a creed. Then the creed is that character's acquired value, and it is not ours. No lens ever feeds back into reward or selection.

How this meets your principles:

| Principle | How |
|---|---|
| Author below the capability | We write drives, learning rules and the world. Values, and any dominance, come from lives. |
| Never name a behaviour in a reward | Reward stays "the drop in weighted drive error". Acquired values are extra drives with learned weights. Their targets are things in the world (a person, a coin), never acts. |
| No special-case subsystem | One acquired-value rule serves the martyr, the hoarder and the dominator. There is no dominance module. |
| Nothing pretrained chooses | Unchanged. |

---

### 9.2 How worldly hunger arises and can override the body

**What is authored today.** `NATURE` in `world.py` gives each type a wealth weight: 1.2 to 1.4 for merchant, thief and bandit, and 0 for peasant, knight and cook. So worldly hunger is written in by type. The mind never learns to want a coin. We keep that table as one control arm and add a learned arm beside it.

**The mechanism: acquired concerns.** Each mind gets a small table of concerns, empty at birth. A concern has a target and a learned weight.

| Part | Rule |
|---|---|
| Target | Any thing the mind perceives and the log names: a person, an object, a room. Not an act. |
| Its need | How far the target is from a good state. For a person, their hp and fear as perceived. For an object, whether I still hold it. For a room, whether I can get to it. |
| Reward | The drop in that need, times the weight, added to the existing drop in drive error. Reward is still only a drop in error. |
| Weight grows with | (a) Drive-error drops that followed contact with the target. This is the existing valence rule V, and it makes the target a conditioned reinforcer (Allport's "functional autonomy" (u); Berridge and Robinson's cue "wanting" (u)). (b) Cost paid on the target's behalf: error that rose while acting for it. This is effort justification (Aronson and Mills 1959 (u)). |
| Weight falls with | Disuse (slow fade). Repeated error rises after contact (disillusion). Habituation: repeated contact pays less, so there is no loop of standing next to a loved thing for ever. Today `WEAR` does this only for praise and insult. The general form belongs in the sense layer (9.8). |
| Limits | A ceiling and a gain, both set by the genome. These are capacities. The content is never inherited. |
| Controls | Placebo targets: features that predict nothing. Their weights must stay near 0, or the rule attaches to anything. |

The body is never switched off. A concern wins a choice when its weight times predicted gain beats the body terms in the same one-step lookahead the mind already uses. Hunger keeps rising as the mind starves, so the override has a limit. A very strong concern can carry a mind a long way. It cannot carry it past death.

**The two outcomes, neither written in**

| Outcome | How it forms | How we tell it is learned |
|---|---|---|
| The martyr | The target is a person or object. Contact paid off, and costs were paid on its behalf, so the weight grew. In a world where feeding means leaving the target in danger, the mind accepts a higher body error. | Swap the learned weight with another mind's at decision time. The behaviour must flip. Clones with different lives must part. |
| The hoarder | The target is the coin. Coin was paired with food and safety, so its weight grew. A mind that never spends never re-tests the pairing, so the weight never falls. | A spender's weight tracks purchases. A holder's does not. Both started from the same weight of 0. |

Grief comes from the same rule. If the target dies, its need stays high and the weight fades slowly. Grief should last longer for heavier weights. That is a free check (u).

**The tramplers.** Nothing in this mind weights other people's welfare unless it formed a concern for them. A mind whose only concerns are coin and itself will take from others whenever taking pays after reprisal. Whether it pays is decided by the world's consequence table. That is the point. It is where "good for the whole" enters the system: as an outcome we measure, never as a term in the reward.

**Staging.** First, concerns with persons and things as targets. Contest and deference outcomes (9.3) are added only if that works.

---

### 9.3 How dominance and status emerge

We do not model a need for dominance. A scalar drive, rank bonus or "social dominance orientation" score would name the behaviour (Pratto et al. 1994 (u) measures it as a trait, which is the opposite of emergent). We model the conditions under which control pays and let a mind learn to value it.

| Ingredient | How it is built | Exists today? |
|---|---|---|
| Contests with real stakes | Scarce things (the seat by the fire, the one ration, the chest) go to whoever wins a physical contest. The outcome comes from bodies, things and chance. There is no rank rule. | Partly. `strike`, `shove`, `threaten` exist. A blow's aim comes from anger, drink, fatigue and whether the target is down or asleep. Its damage comes from the weapon and chance. There is no body strength, so only weapons make lasting asymmetries. After a blow or a threat, the victim's fear rises scaled by its authored `bravery` trait. That is a trait doing a dial's job. Round 3 should replace it with the pain-weight dial and the victim's history with that partner. Scarcity of single items needs checking (u). |
| Yielding | The target's menu already has `give` and `refuse`. Extortion is `threaten`, then `give`. No new act is needed. | Yes |
| Winner and loser effects | Partner models start from a shared prior learned across all partners, so one win raises expected success against strangers too. | Needs a shared prior over the partner models P. Small change, not made. |
| Self-model of strength | The same shared prior is the mind's estimate of how it does in contests. | As above |
| Deference | Others' partner models record "gives in to X". Their acts toward X change. X's standing is how they treat it, as `SLIGHT` already works. | Yes |
| Prestige | Being sought out (`ask`, `offer`) and praised, spread by gossip. This is a second way to raise standing, without fear (Henrich and Gil-White 2001 (u)). | Yes, by log analysis |
| Taste for winning | No special slot. The concern rule's candidate targets (9.2) include relations the log records, such as who gave way to whom. If being given way to keeps coming before drive drops, it gains weight like any other target. No appetite curve is written for it. | New, staged late |

**How the taste forms.** Control repeatedly comes before drive drops: more food, a safer seat, fewer slights. By the 9.2 rule, winning and being deferred to become conditioned reinforcers, like money. The "need for dominance" you can understand but not model is then a learned appetite for a signal that used to predict satisfaction. Whether it appears is an open empirical question. If it does not, that is a finding about the world, not a failure to build it.

**Hierarchy from equals.** Chase et al. (2002 (u)) and Bonabeau et al. (1996 (u)) show that identical animals form a linear order from chance wins and winner effects alone. The inn can test this directly: five identical-nature minds, different orders each seed, and the order should vanish when the shared prior is switched off.

**Two routes to standing.** Report them separately: standing from fear-linked events (a threat followed by a `give`) and standing from being sought out. Which route a mind takes depends on what each costs in the world. This is the dominance versus prestige split (Cheng et al. 2013 (u)).

**When tyranny can emerge.** It needs all of these:

| Condition | Why |
|---|---|
| A scarce thing others need | Control has something to sell. |
| Challenge costs more than complying | Witnesses, reprisal and gossip are weak or slow. |
| An asymmetry that persists | A weapon, a monopoly of the chest, a corner of the room. |
| Targets cannot leave or coordinate | Otherwise counter-dominance coalitions form (Boehm 1999 (u)). |

When the first three hold and reprisal is switched off, we expect `threaten`, then `give`, to become a stable pattern, with rising inequality. With reprisal on, we expect it to fall. Acemoglu and Robinson (2012 (u)) make the same point about institutions: tyranny is what a world allows. We deliberately build the permissive world as a test bed and label it so.

---

### 9.4 What the outer loop selects

**Selection channels.** Each is a persistence measure, never a weighing of needs. The three-part lens comparison is in 9.5.

| Channel | Measure | Default? | What it predicts |
|---|---|---|---|
| P1. Body | Nights survived by the focal mind, risk-averse (mean minus SD) over world variants. | Yes | Energy and safety weights cannot fall far: low weights mean death. Attachment helps only where allies feed or protect. Martyrdom does not evolve. Hoarding evolves only if coin buys survival. Extraction evolves only where reprisal is weak. |
| P2. Lineage | The genome persists if any carrier does. With one focal mind, same as P1. In a cast where all minds share a genome (a clone cast), it differs. | Later | Higher attachment to clones. Bounded extraction. (u) |
| P3. Group | Nights the cast stays viable, such as at least k survivors. | Later arm | Higher attachment ceiling and gain. Martyr-like acts appear. Extraction falls. Inequality can still hide if the measure is an average. |
| P4. Idea | The attachment content spreads to other minds by imitation. | Out of scope | Needs a cultural channel. It is the only channel through which a martyr's cause can outlive the martyr. It is also the channel that spreads exploiters. |
| A0. Old `F_body` | The fixed bodily error of sections 1 to 3. | Control only | Pushes wealth weight up whether or not coin helps, because "solvent" is inside the score. This leak is the thing to measure. |

**What changes in the genome.** The 8 link and plasticity genes stay. We add capacity genes only: attachment gain, attachment ceiling, cost-credit coefficient, fade rate and strength of the shared contest prior. The bounds are [0, 2x] for acquired parts. Body drive weights, when evolved, are [0.1x, 2x] of today's `NATURE`, with a control run at [0, 2x]. We do not need the genome to zero out hunger. A concern can outweigh hunger without that.

**What each choice predicts.** Running P1 and A0 and comparing is the sensitivity check the earlier plan wanted. If the same nature wins under both, `F_body` was harmless. If not, it was a hidden value. Running P1 against P3 shows what group selection adds, namely pro-sociality, and nothing else.

**Cost.** One P1 run is unchanged (about 8 CPU hours per run, 25 for three). A clone-cast or group run at 1.0 to 1.5 s a night (u): population 8, 12 generations, 3 life seeds of 100 nights is 28,800 nights, about 10 CPU hours per run at 1.25 s, or about 30 for three (my arithmetic). Group runs wait until casts of several minds are cheap.

---

### 9.5 Observer lenses, and how packs encode moral consequences

**The lenses.** Each is computed from the logs after a run. None is in any reward, selection or tuning loop.

| Lens | What it measures | Why |
|---|---|---|
| Own measure | The mind's own weighted error at the end of life, including acquired concerns. | Your definition of a good life: needs plus what it cares for. A tyrant can score high. A martyr can score high on concerns and low on the body. |
| Body | The old `F_body` number, alive, fed, safe, rested, solvent. | What happened to the body. |
| Whole | Total of own measures, the worst-off mind (Rawls), spread of holdings and own measures (Gini), deaths and hp lost to others (traced through event causes), cast survivorship. | The group's good, in four numbers. Never one. |
| Rubric | A data file of weights over logged event kinds, such as "gave to the hungry" and "killed". | A creed or ethic, held by the observer. |

**Lens disagreement table.** Report one row per run, such as: tyrant (own measure high, whole low), martyr (body low, own measure mixed, whole variable), hoarder, gentle reciprocator. This table is the answer to "good to the individual, the whole, or a rubric?" We show where they split. We do not choose one.

**Packs encode consequences, not judgements.** A pack is data. It contains:
1. The world's facts: what exists (a shrine, ranks, realms).
2. A consequence table: what each act does, who perceives it, who reacts and how. Cultivation example: insulting a senior is physically dangerous, because the senior is stronger and may strike. That is physics plus social structure. It is not the sentence "insulting seniors is wrong".
3. Rubric files native to that world, for reports only.

The mind learns caution from the consequence table. The rubric tells us afterwards whether the result matches that world's creed. A character in the pack can hold the creed as an acquired concern (9.2), and then it is that character's value.

---

### 9.6 Changes to the round-3 plan

| Step | Change |
|---|---|
| 0a | Add archetypes: extractor (threaten, then take), hoarder, protector. Add a measure: does power pay? Run with reprisal and gossip off and on. If the extractor wins only when reprisal is off, the world sets tyranny. Minutes of CPU. |
| 0b | Registry adds ordinal dominance patterns (linearity of order, winner effect). Lens files and the rubric file are hashed too. Before drafting, you get a one-page example of what a registry entry looks like (your earlier answer 4). |
| 1 | Add an attachment-swap test beside belief-swap: swap the weight of Fenn with Oren's and measure the change in actions. |
| 2 | Unchanged, still conditional. |
| Core gate | Add module checks: a concern is acquired and extinguished (E-A below), no loop of standing by a loved cue, and a P-deaf partner earns weight 0. |
| 3 | Freeze the lens code and the new stability checks (9.6, risks). |
| 4 | Run under both P1 and A0. First run the acquired-value module with hand-set capacity genes, off against on. Evolve the capacity genes only if the module passes. |
| 5 | Reputation on or off is kept. Add a power pair: W_power (scarce items, weak reprisal) against W_check (reprisal, witnesses). |
| 5b (new) | P1 against P3 or a clone cast, 3 seeds each, once multi-mind casts are cheap. |
| 6 | The held-out report adds the lens table, the hierarchy pattern, and the martyr and hoarder probes. |
| 7 | You also read one tyrant night and one martyr night, and say whether each is recognisable. You are not asked whether either is good. |

**New cheap, decisive experiments**

| # | Test | Pass |
|---|---|---|
| E-A | A mind with a P-kind gift-giver: the concern weight rises, then falls when the gifts stop. The coin that is never spent keeps its weight. | Rise, fall and the hoarder asymmetry all appear. |
| E-B | Clones in a rich life and a poor life. | The acquired weights themselves part: D_life / D_rng at least 2. |
| E-C | Martyr probe: feeding needs the attached partner left in danger. | Behaviour flips with the weight swap, TV at least 0.1. Body error is higher for the attached clone. |
| E-D | Five identical-nature minds with contests. | A linear order forms (Landau h, de Vries 1995 (u)). It differs across seeds. It is weaker with the shared prior off. |
| E-E | Extractor tournament, power-off against power-on worlds. | Extractor wins only where reprisal is weak. |
| E-F | Tyranny: one-sided advantage, reprisal off, then on. | Extraction share and spread rise, then fall. The whole lens drops under the first. |

**New stability risks**

| Risk | Check |
|---|---|
| Runaway attachment or a loop on a loved cue | Ceiling, `WEAR`, fade. Test: a mind placed beside a loved cue must not stay. |
| Monoculture of extractors, or nobody ever contests | Report type spread per run. Collapse to either counts as failure. |
| Acquired weights just copy the authored table | Compare acquired wealth weights across minds with different lives. If equal, the module adds nothing. |
| Hierarchy caused by engine tie-breaks | Run E-D with shuffled turn order. |
| Goodhart on a lens | No lens is ever tuned on. Logged looks, as for the registry (Manheim and Garrabrant 2018 (u)). |
| P1 is a flat signal | Use 5 life seeds, shock variants and common random numbers. Fall back to the A0 score for coarse early search if needed. |

---

### 9.7 What stays out of scope, and why

| Item | Why |
|---|---|
| A built-in dominance drive | It names the behaviour. |
| Millions pushed into poverty | The inn has 5 people. It can show extraction, inequality and deaths at that scale, nothing more. |
| Churches, creeds and ideologies as living institutions | Need shared belief, imitation and several minds (channel P4). A pack can carry a creed as a world fact. |
| Selecting on a rubric | It would teach Goodhart gaming. At most one labelled demonstration, later. |
| Moral emotion (guilt, shame, indignation) | Not predicted by anything here. Fairness also stays out. |
| Hormones, kinship, reproduction | No such layer exists, and none is needed for the first tests. |
| Claiming "human-like" tyranny | The claim is narrower: control can be learned as a value when the world pays for it. |
| Genre packs | After step 5 passes, as before. |
| The user's broader aim (all lenses, all worlds) | The narrow scope of section 8 is the first stage, not the goal. |

---

### 9.8 Senses are modelled too (proposed, from your message)

The reward is not read from the true body state. It is read through senses, and a sense is a modelled part with dials, like everything else.

| Layer | What it is | Example |
|---|---|---|
| Body (world law) | The true state: body water, the energy store, tissue damage, core and skin temperature (the body-heat work in wave 13). Harm and death come from this, felt or not. | A body dehydrates whether or not it feels thirst. |
| Sense | A reading of one body state, with a gain, a threshold, noise, a delay and adaptation (a steady signal fades). | Thirst is the reading of body water. |
| Reward | The drop in weighted error over the sensed values, not the true ones. | |

Non-standard senses are dial settings, never flags:

| Real case | Dial setting | What follows, unwritten |
|---|---|---|
| No pain from birth (congenital insensitivity to pain (u)) | Pain gain 0 | Injuries are not avoided, because nothing teaches avoidance. |
| Blunted thirst in old age (u) | Thirst threshold high | Dehydration without wanting water. |
| Never feeling full (Prader-Willi (u)) | Satiety gain low | Eating past need. |
| Chronic pain | Pain gain high, adaptation low | Caution everywhere. |
| No pain in the heat of a fight (stress-induced analgesia (u)) | Current state lowers the pain gain for a while | Fighting on while hurt. |

The mind can still learn around a broken sense. It can learn that a day without water comes before collapse, and drink on a schedule. That is the same prediction machinery pointed at its own body.

The outer loop can evolve the sense dials too. Pain and hunger are themselves evolved signals. A lineage whose pain gain drifts to 0 dies of its injuries, so selection keeps the gains in a working range, with spread. Only the body's physics stays fixed.

Cost: one small sense function between the body state and `drive_error` (in `lab/world2/world.py` today, the voxel body later). Its habituation also serves 9.2: repeated contact with a concern pays less.

### 9.9 Priming in simpler worlds (proposed, from your message)

Minds start in simpler worlds and then move into the full one, like a childhood.
- The priming worlds use the same laws with fewer things: one room and plenty of food, then scarcity, then other people, then contests. They never use a script of correct acts. Copying authored behaviour would be the if-else ladder again.
- Test: primed minds against minds learning from scratch, on the same held-out full world. Priming must speed learning. It must not lock in habits that are wrong in the full world (negative transfer). Primed minds must still be able to learn, because long training can lose that ability (loss of plasticity, Dohare et al. 2024 (u)).
- The outer loop can also choose inherited starting values. What every life learns early can then drift into the starting point (the Baldwin effect (u)).

### 9.10 Questions for you

1. **Selection and the old score.** Is individual survival (P1) the default selection, with `F_body` demoted to a report and group arms later? *Recommendation: yes.* This also removes the "is solvent a value?" question from selection, because wealth now matters only when a mind learns it.
2. **Dominance as a learned taste.** Are you happy for dominance to exist only as a learned appetite, and for us to build deliberately permissive worlds (no reprisal, one monopoly) to see whether tyranny emerges? They would be labelled test beds. *Recommendation: yes.*
3. **Lenses.** Do you want own measure, body and whole in every report, with the first rubric arriving only with the cultivation pack, written by me from the books and open to your veto? *Recommendation: yes.*
4. **Death.** The brief says a dead character returns next night. For persistence and the martyr probe that blurs the cost. Keep the return for lifetimes, count nights survived, and add a permanent-death variant for selection and the martyr test only? *Recommendation: yes.*

---

**Sources (all u).** Keramati and Gutkin 2014; Allport 1937; Berridge and Robinson 1998; Aronson and Mills 1959; Chase et al. 2002; Bonabeau et al. 1996; de Vries 1995; Henrich and Gil-White 2001; Cheng et al. 2013; Boehm 1999; Acemoglu and Robinson 2012; Pratto et al. 1994; Manheim and Garrabrant 2018; Dohare et al. 2024; Baldwin 1896; Hinton and Nowlan 1987.
