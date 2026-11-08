# Scene creativity: 10 runs

## Unexpected choices that fit the character (for rating: do they make sense?)


## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 4.7 unexpected, +0.0 fit: t284 Fenn (thief): praise Oren  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t285 Fenn (thief): strike Oren with the axe  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t286 Fenn (thief): insult Sir Ada  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t290 Fenn (thief): tell Oren that the bandit took the copper coin  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t292 Fenn (thief): strike Oren (missed)  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t294 Fenn (thief): give the iron pot to Marta  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t296 Fenn (thief): give the mug of ale to Marta  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t296 Fenn (thief): praise Tom  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t300 Fenn (thief): tell the bandit that Tom took the ladle  (seed6.events.jsonl)
- 4.7 unexpected, +0.0 fit: t283 Fenn (thief): tell Oren that the bandit took the copper coin  (seed6.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 100% | 22.2 |
| betrayal | 100% | 19.5 |
| rescue | 100% | 44.7 |
| lie found out | 0% | 0.0 |
| coward acts bravely | 100% | 2.8 |
| gift returned | 100% | 10.2 |

Rarest and most surprising matches:

- betrayal (seed8.events.jsonl): ?a=the bandit, ?b=Sir Ada
- gift returned (seed8.events.jsonl): ?a=Sir Ada, ?b=the bandit
- gift returned (seed6.events.jsonl): ?a=Fenn, ?b=the bandit
- rescue (seed8.events.jsonl): ?x=Oren, ?v=the bandit, ?h=Sir Ada
- rescue (seed8.events.jsonl): ?x=Marta, ?v=the bandit, ?h=Tom
- revenge (seed6.events.jsonl): ?x=Fenn, ?a=Marta
- revenge (seed8.events.jsonl): ?x=Oren, ?a=Marta
- revenge (seed6.events.jsonl): ?x=Marta, ?a=Fenn
- rescue (seed8.events.jsonl): ?x=Tom, ?v=the bandit, ?h=Sir Ada
- rescue (seed8.events.jsonl): ?x=Marta, ?v=the bandit, ?h=Sir Ada

## Chains

- choices that react to another character: 32%
- longest chain: 16 (Oren > the bandit > Tom > Oren > Fenn > the bandit > Oren > Fenn > the bandit > Marta > the bandit > Oren > Marta > Fenn > Sir Ada > Tom)

## Variety between runs

- map covered (harm x help, 5x5): 4%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.38

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 4.22 | strike 54, tell 42, search 38 |
| cook | 4.14 | take 160, drop 114, walk to 109 |
| knight | 4.17 | take 155, walk to 126, strike 109 |
| merchant | 4.17 | take 136, walk to 117, strike 108 |
| peasant | 4.21 | take 139, walk to 123, tell 101 |
| thief | 4.24 | take 135, walk to 110, drop 95 |

## The world

- choices: 6942; stopped before done: 34%; deaths per night: 0.00; lies told: 25, found out: 2
- most common actions: take 522, drop 501, praise 373, shove 355, walk to 343, insult 337, strike 307, wait 230, give 216, threaten 198, shout for help 194, tell 192

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 10% | 10% | 80% | 82 |
| cook | 11% | 12% | 78% | 219 |
| knight | 8% | 16% | 76% | 213 |
| merchant | 13% | 10% | 78% | 209 |
| peasant | 11% | 13% | 76% | 221 |
| thief | 13% | 10% | 78% | 207 |
