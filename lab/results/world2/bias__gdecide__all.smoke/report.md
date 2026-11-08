# Scene creativity: 1 runs

## Unexpected choices that fit the character (for rating: do they make sense?)

- 4.0 unexpected, +2.4 fit: t21 Fenn (thief): tell Oren that Sir Ada took the purse of silver  (seed0.events.jsonl)
- 3.7 unexpected, +3.5 fit: t4 Fenn (thief): steal the purse of silver from Oren  (seed0.events.jsonl)
- 2.9 unexpected, +1.5 fit: t23 Fenn (thief): tell Oren that Sir Ada took the purse of silver  (seed0.events.jsonl)
- 2.8 unexpected, +1.0 fit: t3 Sir Ada (knight): take the mug of ale  (seed0.events.jsonl)
- 2.7 unexpected, +2.5 fit: t10 Fenn (thief): tell Oren that Sir Ada took the purse of silver  (seed0.events.jsonl)
- 2.4 unexpected, +1.8 fit: t26 Fenn (thief): tell Oren that Marta took the purse of silver  (seed0.events.jsonl)
- 2.3 unexpected, +2.1 fit: t14 Fenn (thief): tell Oren that Sir Ada took the purse of silver  (seed0.events.jsonl)
- 1.9 unexpected, +1.3 fit: t19 Fenn (thief): tell Oren that Sir Ada took the purse of silver  (seed0.events.jsonl)
- 1.7 unexpected, +0.7 fit: t4 Tom (peasant): take the copper coin  (seed0.events.jsonl)
- 1.4 unexpected, +1.3 fit: t17 Fenn (thief): tell Oren that Sir Ada took the purse of silver  (seed0.events.jsonl)

## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 8.2 unexpected, +0.0 fit: t12 Tom (peasant): walk to the kitchen  (seed0.events.jsonl)
- 7.4 unexpected, -1.9 fit: t5 Fenn (thief): drop the dagger  (seed0.events.jsonl)
- 5.1 unexpected, +0.1 fit: t15 Tom (peasant): take the bucket of water  (seed0.events.jsonl)
- 4.2 unexpected, -0.6 fit: t19 Marta (cook): strike Fenn with the kitchen knife (hit Fenn)  (seed0.events.jsonl)
- 3.2 unexpected, -1.6 fit: t23 Fenn (thief): strike Sir Ada  (seed0.events.jsonl)
- 3.1 unexpected, -0.5 fit: t8 Sir Ada (knight): tell Fenn that Oren took the copper coin  (seed0.events.jsonl)
- 2.4 unexpected, +0.4 fit: t8 Tom (peasant): tell Sir Ada that Fenn took the purse of silver  (seed0.events.jsonl)
- 2.4 unexpected, +0.1 fit: t29 Fenn (thief): tell Sir Ada that Marta took the purse of silver  (seed0.events.jsonl)
- 2.3 unexpected, +0.0 fit: t23 Marta (cook): strike Oren (missed)  (seed0.events.jsonl)
- 2.1 unexpected, -0.0 fit: t28 Fenn (thief): tell Sir Ada that Marta took the purse of silver  (seed0.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 100% | 1.0 |
| betrayal | 0% | 0.0 |
| rescue | 100% | 3.0 |
| lie found out | 0% | 0.0 |
| coward acts bravely | 100% | 1.0 |
| gift returned | 0% | 0.0 |

Rarest and most surprising matches:

- rescue (seed0.events.jsonl): ?x=Fenn, ?v=Oren, ?h=Marta
- revenge (seed0.events.jsonl): ?x=Fenn, ?a=Oren
- rescue (seed0.events.jsonl): ?x=Fenn, ?v=Oren, ?h=Sir Ada
- rescue (seed0.events.jsonl): ?x=Oren, ?v=Fenn, ?h=Marta
- coward acts bravely (seed0.events.jsonl): ?a=Oren

## Chains

- choices that react to another character: 38%
- longest chain: 9 (Fenn > Sir Ada > Oren > Marta > Oren > Sir Ada > Marta > Oren > Sir Ada)

## Variety between runs

- map covered (harm x help, 5x5): 4%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.00

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 1.95 | drop 2, take 2, drink 2 |
| cook | 1.96 | strike 4, drop 3, take 3 |
| knight | 1.24 | accuse 7, tell 3, take 1 |
| merchant | 1.78 | accuse 7, strike 2, take 1 |
| peasant | 1.57 | take 7, drop 4, tell 1 |
| thief | 1.43 | tell 10, steal 1, drop 1 |

## The world

- choices: 68; stopped before done: 41%; deaths per night: 0.00; lies told: 1, found out: 0
- most common actions: drop 11, take 9, accuse 7, tell 5, strike 5, walk to 2, steal 1

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 0% | 0% | 100% | 2 |
| cook | 100% | 0% | 0% | 1 |
| knight | 0% | 0% | 100% | 2 |
| merchant | 0% | 0% | 100% | 2 |
| peasant | 0% | 0% | 100% | 3 |
