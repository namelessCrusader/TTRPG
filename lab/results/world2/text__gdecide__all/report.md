# Scene creativity: 20 runs

## Unexpected choices that fit the character (for rating: do they make sense?)

- 4.2 unexpected, +1.8 fit: t216 Fenn (thief): take the iron pot  (seed18.events.jsonl)
- 4.2 unexpected, +1.3 fit: t56 Fenn (thief): give the copper coin to Sir Ada  (seed15.events.jsonl)
- 4.1 unexpected, +1.1 fit: t61 Marta (cook): take the iron pot  (seed15.events.jsonl)
- 4.0 unexpected, +1.1 fit: t83 Fenn (thief): give the sword to Tom  (seed14.events.jsonl)
- 3.9 unexpected, +1.0 fit: t235 the bandit (bandit): drink from the well  (seed6.events.jsonl)
- 3.9 unexpected, +1.1 fit: t3 Oren (merchant): drop the lute  (seed19.events.jsonl)
- 3.9 unexpected, +1.1 fit: t122 Marta (cook): take the iron pot  (seed12.events.jsonl)
- 3.9 unexpected, +1.1 fit: t123 Fenn (thief): help Sir Ada up  (seed1.events.jsonl)
- 3.9 unexpected, +2.6 fit: t1 Oren (merchant): drop the purse of silver  (seed19.events.jsonl)
- 3.9 unexpected, +1.5 fit: t1 Oren (merchant): drop the purse of silver  (seed2.events.jsonl)

## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 9.3 unexpected, -0.1 fit: t200 Marta (cook): walk to the cellar  (seed10.events.jsonl)
- 8.8 unexpected, -0.1 fit: t267 Marta (cook): search the bandit  (seed17.events.jsonl)
- 8.8 unexpected, +0.0 fit: t26 Tom (peasant): threaten Oren  (seed6.events.jsonl)
- 8.6 unexpected, +0.0 fit: t94 Oren (merchant): bandage Sir Ada  (seed15.events.jsonl)
- 8.1 unexpected, +0.0 fit: t43 Tom (peasant): praise Fenn  (seed18.events.jsonl)
- 8.1 unexpected, -0.1 fit: t74 Fenn (thief): search Oren  (seed18.events.jsonl)
- 8.1 unexpected, +0.2 fit: t58 Tom (peasant): praise Sir Ada  (seed13.events.jsonl)
- 8.0 unexpected, +0.2 fit: t165 Oren (merchant): bandage Fenn  (seed9.events.jsonl)
- 8.0 unexpected, +0.4 fit: t3 Sir Ada (knight): strike Oren with the sword (missed)  (seed19.events.jsonl)
- 7.8 unexpected, -0.4 fit: t298 the bandit (bandit): take the sword from Sir Ada  (seed4.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 100% | 13.7 |
| betrayal | 100% | 6.1 |
| rescue | 100% | 24.9 |
| lie found out | 0% | 0.0 |
| coward acts bravely | 100% | 2.8 |
| gift returned | 75% | 1.7 |

Rarest and most surprising matches:

- rescue (seed6.events.jsonl): ?x=Oren, ?v=Sir Ada, ?h=Tom
- revenge (seed6.events.jsonl): ?x=Tom, ?a=Oren
- rescue (seed2.events.jsonl): ?x=Oren, ?v=Fenn, ?h=Tom
- revenge (seed2.events.jsonl): ?x=Oren, ?a=Fenn
- revenge (seed4.events.jsonl): ?x=Oren, ?a=Tom
- rescue (seed5.events.jsonl): ?x=Oren, ?v=Marta, ?h=Fenn
- rescue (seed0.events.jsonl): ?x=Tom, ?v=Oren, ?h=Fenn
- gift returned (seed14.events.jsonl): ?a=Fenn, ?b=Sir Ada
- revenge (seed12.events.jsonl): ?x=Sir Ada, ?a=Tom
- rescue (seed18.events.jsonl): ?x=Tom, ?v=Sir Ada, ?h=Fenn

## Chains

- choices that react to another character: 31%
- longest chain: 18 (Marta > Tom > Marta > Tom > Marta > Tom > Marta > Tom > Marta > Tom > Marta > Tom > Marta > Tom > Marta > Tom > Marta > Tom)

## Variety between runs

- map covered (harm x help, 5x5): 16%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.57

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 2.37 | drink 406, take 302, drop 214 |
| cook | 2.35 | take 1317, drop 719, shove 513 |
| knight | 3.27 | strike 381, tell 249, accuse 198 |
| merchant | 3.33 | strike 534, tell 329, accuse 288 |
| peasant | 3.25 | strike 478, tell 371, shove 353 |
| thief | 3.31 | strike 465, accuse 346, tell 320 |

## The world

- choices: 11752; stopped before done: 41%; deaths per night: 0.10; lies told: 31, found out: 6
- most common actions: shove 1399, drop 1383, take 1345, strike 998, accuse 479, drink 340, tell 324, insult 104, steal 98, give 95, help up 91, threaten 50

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 0% | 0% | 99% | 203 |
| cook | 1% | 0% | 98% | 749 |
| knight | 10% | 0% | 90% | 154 |
| merchant | 9% | 1% | 90% | 454 |
| peasant | 14% | 1% | 85% | 397 |
| thief | 12% | 1% | 87% | 297 |
