# Scene creativity: 20 runs

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
| revenge | 100% | 22.1 |
| betrayal | 100% | 19.4 |
| rescue | 100% | 46.5 |
| lie found out | 5% | 0.1 |
| coward acts bravely | 100% | 2.9 |
| gift returned | 100% | 10.9 |

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

- choices that react to another character: 31%
- longest chain: 16 (Oren > the bandit > Tom > Oren > Fenn > the bandit > Oren > Fenn > the bandit > Marta > the bandit > Oren > Marta > Fenn > Sir Ada > Tom)

## Variety between runs

- map covered (harm x help, 5x5): 4%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.38

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 4.22 | strike 111, take 106, walk to 94 |
| cook | 4.15 | take 343, drop 239, walk to 228 |
| knight | 4.17 | take 317, walk to 241, strike 196 |
| merchant | 4.17 | take 273, walk to 254, strike 210 |
| peasant | 4.21 | take 277, walk to 233, tell 176 |
| thief | 4.20 | take 303, walk to 252, drop 203 |

## The world

- choices: 13943; stopped before done: 33%; deaths per night: 0.00; lies told: 56, found out: 5
- most common actions: take 1113, drop 1057, walk to 732, praise 704, shove 704, insult 669, strike 607, wait 473, give 470, threaten 418, shout for help 392, flee to 358

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 8% | 15% | 77% | 189 |
| cook | 10% | 11% | 80% | 431 |
| knight | 9% | 16% | 76% | 414 |
| merchant | 12% | 10% | 78% | 417 |
| peasant | 10% | 10% | 80% | 414 |
| thief | 13% | 10% | 77% | 412 |
