# Scene creativity: 10 runs

## Unexpected choices that fit the character (for rating: do they make sense?)

- 9.9 unexpected, +7.2 fit: t190 the bandit (bandit): steal the lute from Oren  (seed8.events.jsonl)
- 9.6 unexpected, +7.3 fit: t237 the bandit (bandit): steal the sword from Sir Ada  (seed0.events.jsonl)
- 9.1 unexpected, +6.6 fit: t168 the bandit (bandit): steal the lute from Oren  (seed4.events.jsonl)
- 8.7 unexpected, +5.8 fit: t264 the bandit (bandit): steal the lute from Oren  (seed1.events.jsonl)
- 7.8 unexpected, +4.8 fit: t236 Tom (peasant): flee to the main room  (seed4.events.jsonl)
- 7.5 unexpected, +5.6 fit: t196 the bandit (bandit): steal the lute from Oren  (seed6.events.jsonl)
- 7.5 unexpected, +4.5 fit: t222 Fenn (thief): steal the sword from Sir Ada  (seed3.events.jsonl)
- 7.3 unexpected, +5.4 fit: t280 the bandit (bandit): steal the lute from Oren  (seed6.events.jsonl)
- 7.2 unexpected, +4.5 fit: t129 Oren (merchant): steal the sword from Sir Ada  (seed4.events.jsonl)
- 7.0 unexpected, +6.2 fit: t105 Fenn (thief): tell Oren that Marta took the purse of silver  (seed2.events.jsonl)

## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 10.8 unexpected, +3.8 fit: t60 Fenn (thief): tell Marta that Oren took the sword  (seed9.events.jsonl)
- 10.1 unexpected, +1.1 fit: t233 Tom (peasant): praise Fenn  (seed4.events.jsonl)
- 9.9 unexpected, +5.1 fit: t64 Oren (merchant): steal the sword from Sir Ada  (seed7.events.jsonl)
- 9.4 unexpected, +3.3 fit: t268 Fenn (thief): tell Marta that Sir Ada took the purse of silver  (seed2.events.jsonl)
- 9.3 unexpected, +3.0 fit: t141 Tom (peasant): walk to the kitchen  (seed7.events.jsonl)
- 9.3 unexpected, +3.1 fit: t292 Fenn (thief): tell Marta that the bandit took the lute  (seed7.events.jsonl)
- 9.0 unexpected, +1.1 fit: t120 Fenn (thief): tell Marta that Tom took the purse of silver  (seed3.events.jsonl)
- 9.0 unexpected, -0.4 fit: t161 Marta (cook): threaten Tom  (seed8.events.jsonl)
- 8.9 unexpected, +4.0 fit: t114 Fenn (thief): tell Tom that Sir Ada took the purse of silver  (seed5.events.jsonl)
- 8.8 unexpected, +0.9 fit: t114 Fenn (thief): tell Tom that Sir Ada took the purse of silver  (seed8.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 90% | 3.8 |
| betrayal | 70% | 0.9 |
| rescue | 90% | 8.0 |
| lie found out | 30% | 0.3 |
| coward acts bravely | 90% | 1.7 |
| gift returned | 0% | 0.0 |

Rarest and most surprising matches:

- revenge (seed3.events.jsonl): ?x=Fenn, ?a=Oren
- rescue (seed8.events.jsonl): ?x=Tom, ?v=Fenn, ?h=Marta
- lie found out (seed8.events.jsonl): ?a=Fenn, ?b=Tom
- rescue (seed3.events.jsonl): ?x=Oren, ?v=Fenn, ?h=Sir Ada
- rescue (seed8.events.jsonl): ?x=Sir Ada, ?v=Marta, ?h=Tom
- revenge (seed2.events.jsonl): ?x=Fenn, ?a=Marta
- rescue (seed8.events.jsonl): ?x=Marta, ?v=Tom, ?h=Sir Ada
- betrayal (seed7.events.jsonl): ?a=Tom, ?b=Sir Ada
- revenge (seed2.events.jsonl): ?x=Marta, ?a=Fenn
- revenge (seed3.events.jsonl): ?x=Oren, ?a=Fenn

## Chains

- choices that react to another character: 20%
- longest chain: 12 (Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit > Sir Ada > the bandit)

## Variety between runs

- map covered (harm x help, 5x5): 4%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.43

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 2.74 | walk to 105, wait 72, strike 43 |
| cook | 2.83 | wait 282, walk to 216, drink 87 |
| knight | 2.82 | wait 306, walk to 167, drink 79 |
| merchant | 2.94 | walk to 291, wait 177, flee to 106 |
| peasant | 2.68 | wait 275, walk to 273, flee to 83 |
| thief | 3.12 | wait 215, walk to 208, drink 93 |

## The world

- choices: 4509; stopped before done: 18%; deaths per night: 0.10; lies told: 34, found out: 4
- most common actions: wait 1195, walk to 1086, drink 387, flee to 240, accuse 155, strike 133, take 115, tell 62, search 57, hide 45, play 39, defend 25

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 0% | 50% | 50% | 12 |
| cook | 26% | 31% | 43% | 35 |
| knight | 13% | 16% | 71% | 31 |
| merchant | 2% | 70% | 27% | 44 |
| peasant | 15% | 61% | 24% | 46 |
| thief | 0% | 61% | 39% | 18 |
