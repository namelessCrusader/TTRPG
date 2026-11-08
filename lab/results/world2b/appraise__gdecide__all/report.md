# Scene creativity: 10 runs

## Unexpected choices that fit the character (for rating: do they make sense?)

- 20.7 unexpected, +20.7 fit: t21 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t23 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t30 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t32 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t34 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t37 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t39 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t42 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t44 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)
- 20.7 unexpected, +20.7 fit: t48 Tom (peasant): accuse Oren of taking the copper coin  (seed0.events.jsonl)

## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 7.2 unexpected, +3.4 fit: t165 Fenn (thief): accuse Tom of taking the sword  (seed1.events.jsonl)
- 6.4 unexpected, +2.9 fit: t134 Sir Ada (knight): accuse Oren of taking the copper coin  (seed9.events.jsonl)
- 4.9 unexpected, -2.0 fit: t10 Fenn (thief): take the copper coin  (seed0.events.jsonl)
- 4.5 unexpected, +1.1 fit: t168 the bandit (bandit): take the axe  (seed4.events.jsonl)
- 2.3 unexpected, +0.4 fit: t91 Tom (peasant): take the mug of ale  (seed7.events.jsonl)
- 1.9 unexpected, -0.4 fit: t27 Oren (merchant): tell Sir Ada that Fenn took the copper coin  (seed4.events.jsonl)
- 1.9 unexpected, +0.2 fit: t262 Tom (peasant): drop the purse of silver  (seed3.events.jsonl)
- 1.8 unexpected, +0.4 fit: t95 Tom (peasant): drop the purse of silver  (seed7.events.jsonl)
- 1.7 unexpected, +0.0 fit: t159 Tom (peasant): drop the purse of silver  (seed3.events.jsonl)
- 1.7 unexpected, -0.8 fit: t9 Sir Ada (knight): tell Oren that Tom took the sword  (seed1.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 90% | 3.0 |
| betrayal | 0% | 0.0 |
| rescue | 90% | 4.2 |
| lie found out | 0% | 0.0 |
| coward acts bravely | 70% | 1.7 |
| gift returned | 0% | 0.0 |

Rarest and most surprising matches:

- rescue (seed1.events.jsonl): ?x=Fenn, ?v=Tom, ?h=Sir Ada
- rescue (seed0.events.jsonl): ?x=Tom, ?v=Oren, ?h=Fenn
- revenge (seed7.events.jsonl): ?x=Tom, ?a=Oren
- rescue (seed7.events.jsonl): ?x=Tom, ?v=Oren, ?h=Sir Ada
- rescue (seed7.events.jsonl): ?x=Tom, ?v=Oren, ?h=Fenn
- revenge (seed4.events.jsonl): ?x=Oren, ?a=Fenn
- rescue (seed4.events.jsonl): ?x=Oren, ?v=Fenn, ?h=Tom
- revenge (seed2.events.jsonl): ?x=Sir Ada, ?a=Tom
- rescue (seed2.events.jsonl): ?x=Sir Ada, ?v=Tom, ?h=Oren
- rescue (seed2.events.jsonl): ?x=Sir Ada, ?v=Tom, ?h=Fenn

## Chains

- choices that react to another character: 50%
- longest chain: 107 (Fenn > Tom > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Sir Ada > Fenn > Sir Ada > Fenn > Sir Ada > Fenn > Oren > Fenn > Sir Ada > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Oren > Fenn > Oren > Sir Ada > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Sir Ada > Oren > Sir Ada > Oren > Sir Ada > Fenn > Oren > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Sir Ada > Fenn > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Oren > Fenn > Oren > Sir Ada > Fenn > Sir Ada > Oren > Sir Ada > Fenn > Oren > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Sir Ada > Fenn > Sir Ada > Fenn > Oren > Sir Ada > Fenn > Oren > Sir Ada > Oren > Sir Ada > Fenn > Tom)

## Variety between runs

- map covered (harm x help, 5x5): 8%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.42

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 1.28 | drink 220, drop 46, take 43 |
| cook | 1.03 | take 762, drop 728, eat 3 |
| knight | 1.35 | accuse 841, take 230, tell 123 |
| merchant | 0.66 | accuse 817, tell 47, take 18 |
| peasant | 1.39 | accuse 864, take 532, tell 70 |
| thief | 0.88 | accuse 833, tell 97, strike 41 |

## The world

- choices: 6443; stopped before done: 29%; deaths per night: 0.00; lies told: 0, found out: 0
- most common actions: accuse 2675, drop 815, take 805, drink 221, strike 45, tell 29, steal 7, eat 3

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 0% | 0% | 100% | 4 |
| cook | 0% | 0% | 100% | 24 |
| knight | 50% | 0% | 50% | 4 |
| merchant | 0% | 0% | 100% | 1 |
| peasant | 12% | 0% | 88% | 17 |
| thief | 0% | 0% | 100% | 4 |
