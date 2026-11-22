# Family households: kin care from shared history

Scoped with the owner on 2026-11-21 (decisions Fam1-Fam6). This is the learned mind's next lever after the lethal-neglect step.

**Why.**
- The inn of five unrelated adults never paid for treating partners differently, not even with realistic hardship. The oracle gap stayed under 0.1, and care could gain at most about 0.02.
- The library's sources say need-based help flows mainly between KIN: 99 percent of the incapacitated were fed by others (Gurven 2012), while net food between unrelated Tsimane families is about 49 kcal a day and only weakly need-responsive (Hooper 2015).
- Five unrelated people is the setting where reciprocity pays least.

**The owner's two criteria:**
1. the behaviour MEANINGFULLY MATCHES HUMAN BEHAVIOUR;
2. it is INTERESTING.

"If it's realistic we can work towards interesting."

## Fam1. Where kin care comes from: shared history plus kin cues

- Kinship is an ESTIMATE FROM CUES, never a flag. The cues:
  - the years of co-residence as children;
  - seeing one's mother care for an infant (Lieberman, Tooby and Cosmides 2007).
- The estimate raises how fast ACQUIRED CONCERNS (docs/learned-mind-target.md 9.2) form, through a genetic CAPACITY dial that varies between people.
- Parents' care also grows by the concerns' own effort-justification route (the cost of raising a child).
- Edge cases follow with no rules: adopted and step siblings raised together act as kin; twins separated at birth do not; the same cue gives incest avoidance.

## Fam2. The past: a coarse pre-history on the same laws

- The years before play run in SEASON-SIZED STEPS with the SAME LAWS:
  - each step records the cue exposures (who lived with whom, who cared for whom, how much);
  - it grows the concerns by the same concern law, scaled to the step;
  - the last weeks before play run in full detail, so recent memories and moods are real.
- Precedents:
  - Bad News generated a 140-year town in about 5 minutes;
  - Dwarf Fortress keeps a two-layer memory in which each season's strongest events per category become permanent and shape personality;
  - Talk of the Town simulated the last week and implanted the rest.
- THE OWNER'S CONSISTENCY RULE: run both the FULL simulation and the COARSE pre-history, and they must be consistent to a degree. The coarse one must REACH A SIMILAR POINT (concern weights, closeness rank orders, kinship estimates).
- Use research, games, fiction and the understanding of human bonds as references for how it should look: siblings raised together closer, a favourite child, an absent parent, grief.
- Warning from Adams: summed happiness caused tantrum spirals, so concerns must not feed one summed mood.

## Fam3. Who lives at the inn

- A FAMILY-RUN INN WITH LODGERS: an innkeeper couple, a child, an elderly parent, and 2 to 3 unrelated regulars or lodgers.
- Kin and non-kin share one world under the same laws.
- Children are real bodies (child body sheets), with child needs and abilities in the inn (limited strength, needing to be fed). The mind's development comes later.

## Fam4. Who learns

- EVERYONE: every family member and lodger has a learned mind with acquired concerns and kin cues, and their bonds form in the coarse past.
- A rules-cast inn is the null.
- Read the lethal-neglect result first. If the fixed-partner tests pass there, that gives a clean single-learner comparison.
- Cost: an all-learner inn is several times slower than today's runs, on CPU, overnight.

## Fam5. The life course during play

- NOW, a FIXED household during play: births, growing up and ageing happen only in the coarse past. DEATH IS IN, since the lethal world needs it, and grief can follow.
- LATER, the full life course, with the genetics layer (inherited dials, sexes as body data, docs/magic-research/genetics.txt) and child development, so families persist across generations.

## Fam6. What counts as success

GATES (the build is correct):
1. The coarse pre-history and the full simulation reach similar bonds over a short span (one or two simulated years), within a set tolerance.
2. The kin cues work as in humans, in controlled pasts:
   - an adopted child raised together gets kin-like care;
   - a sibling raised apart does not;
   - SCRAMBLED cues make kin care disappear.
3. The usual controls: placebo targets near 0, a swap of bonds flips behaviour, and the rules null.

REPORTED against field data, never tuned:

4. Kin care emerges: more help flows to family than to lodgers.
5. It pays: family members survive hardship better than lone lodgers in the lethal hard inn.

A failure of 4 or 5 is a finding that points at the world (needs too easily met), the mind (concerns too weak) or the past (cues too faint), each to be fixed from real sources.

## Sources for the kin rule (library, 2026-11-21: responses/2026-11-21-mark1-concern-rates.md)

- Care follows YEARS CO-RESIDED: r = 0.41 without the maternal perinatal cue, and the lines cross at 14 to 18 years.
- Late entry costs about 2.5 percent of care time per year (Anderson).
- Current residence dominates time spent: resident stepfathers have 84 fewer interactions a year, nonresident genetic fathers 179 to 194 fewer.
- In Ache food sharing the reciprocity slope is about 6 times steeper for close kin, and relatedness alone drops out once reciprocity is in.
- Kin ties do not decay over 17 months.

The library's proposed single rule (no kin flag), to build in the family step:
1. an accumulated-history term H_T per pair, with a time constant of years;
2. H_T slows the fade of the concern;
3. an earlier start counts more, with no hard window;
4. the reciprocity coupling is scaled by H_T;
5. an OBSERVED-CARE cue: seeing someone you care about care for T sets your weight for T high (the general form of the maternal perinatal cue).

The concern constants change accordingly:
- fade = 0.0014 / (1 + age/tau) a night, with tau about a year, reduced by contact (the old 0.02 a night was 10 to 15 times too fast);
- learning per contact bout, not per turn;
- extinction slower than acquisition, and it does not erase;
- habituation per feature, with a nonzero set point and recovery by variety;
- effort justification about 2x value per completed act.

The acceptance tests are in the response's S5.

## Hours to friendship (library, 2026-11-22, full text of Hall 2018)

- Casual friend, friend and close friend took 94, 164 and 219 hours of contact for adults (43, 57 and 119 for students).
- At 3 to 5 minutes per turn, a friend-level concern needs about 1,000 or more contact turns, against about 11 turns for a_T at lr 0.2. So the concern acquisition per contact must be MUCH slower, and set from these hours.
- Talk about goals and shared concerns adds closeness; small talk and work time lower it.
- Effort justification (Aronson and Mills 1959, full, random assignment): severe 195.3 against mild 171.1 against control 166.7, a THRESHOLD, not a dose.
- Rescorla 2004 has no numbers, so the recovery rate stays unsourced.

