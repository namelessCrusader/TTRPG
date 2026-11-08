# Scene creativity: 20 runs

## Unexpected choices that fit the character (for rating: do they make sense?)

- 10.8 unexpected, +9.0 fit: t6 Fenn (thief): steal the purse of silver from Oren  (seed15.events.jsonl)
- 9.6 unexpected, +7.5 fit: t209 Tom (peasant): flee to the yard  (seed17.events.jsonl)
- 9.4 unexpected, +7.7 fit: t113 Tom (peasant): flee to the yard  (seed10.events.jsonl)
- 9.2 unexpected, +8.0 fit: t195 Fenn (thief): take the purse of silver from Oren  (seed7.events.jsonl)
- 8.9 unexpected, +8.1 fit: t8 Fenn (thief): steal the purse of silver from Oren  (seed1.events.jsonl)
- 8.4 unexpected, +5.8 fit: t22 Fenn (thief): steal the purse of silver from Oren  (seed1.events.jsonl)
- 8.3 unexpected, +7.8 fit: t247 Marta (cook): flee to the kitchen  (seed13.events.jsonl)
- 8.3 unexpected, +6.4 fit: t21 Fenn (thief): steal the purse of silver from Oren  (seed13.events.jsonl)
- 8.2 unexpected, +7.7 fit: t9 Fenn (thief): steal the purse of silver from Oren  (seed2.events.jsonl)
- 8.2 unexpected, +7.2 fit: t27 Oren (merchant): flee to the yard  (seed3.events.jsonl)

## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 9.9 unexpected, +2.9 fit: t81 Fenn (thief): tell Sir Ada that Marta took the purse of silver  (seed18.events.jsonl)
- 9.6 unexpected, -0.2 fit: t25 Fenn (thief): praise Sir Ada  (seed19.events.jsonl)
- 9.2 unexpected, +0.6 fit: t253 Sir Ada (knight): accuse Fenn of taking the purse of silver  (seed17.events.jsonl)
- 8.6 unexpected, +5.1 fit: t15 Fenn (thief): steal the purse of silver from Oren  (seed1.events.jsonl)
- 8.5 unexpected, +2.7 fit: t6 Fenn (thief): strike Oren (hit Oren)  (seed1.events.jsonl)
- 8.5 unexpected, +3.8 fit: t45 Fenn (thief): tell Oren that Sir Ada took the copper coin  (seed0.events.jsonl)
- 8.5 unexpected, +1.5 fit: t150 Oren (merchant): hide the sword under your cloak  (seed3.events.jsonl)
- 8.4 unexpected, -0.1 fit: t173 Oren (merchant): take the stool  (seed3.events.jsonl)
- 8.3 unexpected, +1.5 fit: t142 Marta (cook): walk to the main room  (seed2.events.jsonl)
- 8.3 unexpected, +1.5 fit: t170 Marta (cook): praise Fenn  (seed13.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 100% | 16.5 |
| betrayal | 100% | 5.2 |
| rescue | 100% | 30.5 |
| lie found out | 15% | 0.1 |
| coward acts bravely | 100% | 2.8 |
| gift returned | 75% | 1.2 |

Rarest and most surprising matches:

- revenge (seed11.events.jsonl): ?x=Oren, ?a=Fenn
- rescue (seed9.events.jsonl): ?x=Fenn, ?v=Sir Ada, ?h=the bandit
- rescue (seed1.events.jsonl): ?x=Fenn, ?v=Oren, ?h=Sir Ada
- rescue (seed18.events.jsonl): ?x=Oren, ?v=Tom, ?h=Fenn
- rescue (seed9.events.jsonl): ?x=the bandit, ?v=Fenn, ?h=Marta
- betrayal (seed19.events.jsonl): ?a=Fenn, ?b=Oren
- revenge (seed1.events.jsonl): ?x=Fenn, ?a=the bandit
- rescue (seed4.events.jsonl): ?x=Marta, ?v=Sir Ada, ?h=Fenn
- betrayal (seed4.events.jsonl): ?a=Marta, ?b=Fenn
- betrayal (seed10.events.jsonl): ?a=Oren, ?b=Tom

## Chains

- choices that react to another character: 33%
- longest chain: 86 (Marta > the bandit > Sir Ada > the bandit > Sir Ada > Marta > Sir Ada > Fenn > Marta > Oren > Sir Ada > the bandit > Sir Ada > Fenn > Oren > Sir Ada > Oren > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > Fenn > Sir Ada > Oren > Sir Ada > Fenn > Sir Ada > Oren > Sir Ada > the bandit > Oren > Sir Ada > Oren > Sir Ada > Fenn > Sir Ada > the bandit > Sir Ada > Oren > the bandit > Sir Ada > the bandit > Sir Ada > Oren > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Oren > Fenn > Oren > Sir Ada > the bandit > Sir Ada > Oren > the bandit > Fenn > the bandit > Sir Ada > Oren > Sir Ada > Oren > Sir Ada > the bandit > Sir Ada > the bandit > Oren > the bandit > Fenn > Sir Ada > the bandit > Sir Ada > Oren > Sir Ada > the bandit > Oren > Sir Ada > the bandit > Oren > Sir Ada > the bandit > Sir Ada > the bandit)

## Variety between runs

- map covered (harm x help, 5x5): 16%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.58

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 3.29 | strike 193, take 192, drop 161 |
| cook | 3.31 | take 584, strike 356, drop 347 |
| knight | 3.18 | strike 568, accuse 255, shove 189 |
| merchant | 3.64 | take 357, flee to 296, tell 276 |
| peasant | 3.50 | tell 415, flee to 342, take 267 |
| thief | 3.43 | strike 362, tell 325, take 239 |

## The world

- choices: 10999; stopped before done: 38%; deaths per night: 0.05; lies told: 92, found out: 9
- most common actions: take 1111, drop 1102, strike 899, shove 821, flee to 653, tell 545, accuse 433, drink 302, walk to 193, steal 159, give 101, insult 80

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 6% | 2% | 92% | 101 |
| cook | 7% | 8% | 85% | 443 |
| knight | 8% | 2% | 90% | 227 |
| merchant | 8% | 18% | 74% | 351 |
| peasant | 4% | 22% | 74% | 373 |
| thief | 10% | 7% | 82% | 283 |
