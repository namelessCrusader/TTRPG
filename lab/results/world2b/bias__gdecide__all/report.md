# Scene creativity: 10 runs

## Unexpected choices that fit the character (for rating: do they make sense?)

- 8.6 unexpected, +7.1 fit: t149 Fenn (thief): steal the purse of silver from Oren  (seed1.events.jsonl)
- 8.4 unexpected, +6.7 fit: t11 Fenn (thief): steal the purse of silver from Oren  (seed3.events.jsonl)
- 8.3 unexpected, +6.0 fit: t285 Oren (merchant): flee to the kitchen  (seed9.events.jsonl)
- 8.1 unexpected, +6.7 fit: t17 Fenn (thief): steal the purse of silver from Oren  (seed3.events.jsonl)
- 8.1 unexpected, +6.2 fit: t62 Fenn (thief): steal the purse of silver from Oren  (seed3.events.jsonl)
- 8.0 unexpected, +7.1 fit: t6 Fenn (thief): steal the purse of silver from Oren  (seed3.events.jsonl)
- 7.9 unexpected, +5.3 fit: t70 Tom (peasant): flee to the yard  (seed8.events.jsonl)
- 7.9 unexpected, +5.8 fit: t178 Fenn (thief): tell Oren that Sir Ada took the lute  (seed5.events.jsonl)
- 7.8 unexpected, +4.9 fit: t184 Oren (merchant): flee to the cellar  (seed1.events.jsonl)
- 7.8 unexpected, +7.3 fit: t11 Fenn (thief): steal the purse of silver from Oren  (seed2.events.jsonl)

## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 9.4 unexpected, -0.7 fit: t98 Fenn (thief): tell Sir Ada that Oren took the copper coin  (seed8.events.jsonl)
- 9.0 unexpected, +4.4 fit: t194 Fenn (thief): tell Oren that Sir Ada took the copper coin  (seed5.events.jsonl)
- 8.9 unexpected, +0.4 fit: t16 Oren (merchant): give the lute to Fenn  (seed9.events.jsonl)
- 8.8 unexpected, +5.4 fit: t35 Fenn (thief): steal the sword from Sir Ada  (seed0.events.jsonl)
- 8.5 unexpected, +0.8 fit: t135 Tom (peasant): drop the kitchen knife  (seed0.events.jsonl)
- 8.2 unexpected, +1.5 fit: t78 Fenn (thief): hide the sword under your cloak  (seed7.events.jsonl)
- 8.2 unexpected, +1.1 fit: t139 Oren (merchant): tell Sir Ada that Marta took the dagger  (seed7.events.jsonl)
- 8.0 unexpected, +1.4 fit: t62 Tom (peasant): walk to the yard  (seed3.events.jsonl)
- 8.0 unexpected, +1.6 fit: t170 the bandit (bandit): strike Tom (hit Tom)  (seed8.events.jsonl)
- 7.9 unexpected, +4.3 fit: t8 Oren (merchant): steal the sword from Sir Ada  (seed7.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 100% | 14.3 |
| betrayal | 100% | 5.2 |
| rescue | 100% | 27.4 |
| lie found out | 30% | 0.4 |
| coward acts bravely | 100% | 2.5 |
| gift returned | 50% | 1.4 |

Rarest and most surprising matches:

- revenge (seed7.events.jsonl): ?x=Sir Ada, ?a=Fenn
- revenge (seed7.events.jsonl): ?x=Fenn, ?a=Sir Ada
- gift returned (seed9.events.jsonl): ?a=Tom, ?b=Sir Ada
- rescue (seed6.events.jsonl): ?x=Fenn, ?v=Sir Ada, ?h=Oren
- rescue (seed9.events.jsonl): ?x=the bandit, ?v=Tom, ?h=Sir Ada
- rescue (seed4.events.jsonl): ?x=Oren, ?v=Tom, ?h=Fenn
- gift returned (seed4.events.jsonl): ?a=Sir Ada, ?b=Fenn
- betrayal (seed5.events.jsonl): ?a=Oren, ?b=Sir Ada
- revenge (seed5.events.jsonl): ?x=Tom, ?a=Fenn
- rescue (seed3.events.jsonl): ?x=Marta, ?v=Sir Ada, ?h=Fenn

## Chains

- choices that react to another character: 33%
- longest chain: 25 (the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit)

## Variety between runs

- map covered (harm x help, 5x5): 8%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.63

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 2.95 | shove 159, take 94, strike 81 |
| cook | 3.33 | take 266, shove 181, strike 160 |
| knight | 3.20 | strike 405, shove 157, tell 96 |
| merchant | 3.51 | flee to 252, take 157, tell 125 |
| peasant | 3.47 | tell 176, flee to 172, take 80 |
| thief | 3.48 | strike 153, take 143, tell 95 |

## The world

- choices: 5309; stopped before done: 40%; deaths per night: 0.20; lies told: 36, found out: 5
- most common actions: shove 644, drop 440, strike 440, take 439, flee to 308, tell 180, accuse 140, walk to 105, drink 104, steal 65, give 46, hide 42

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 2% | 2% | 95% | 84 |
| cook | 2% | 10% | 88% | 231 |
| knight | 13% | 3% | 84% | 105 |
| merchant | 7% | 14% | 79% | 195 |
| peasant | 10% | 21% | 69% | 149 |
| thief | 6% | 9% | 85% | 102 |
