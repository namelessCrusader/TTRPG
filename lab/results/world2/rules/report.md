# Scene creativity: 20 runs

## Unexpected choices that fit the character (for rating: do they make sense?)

- 9.9 unexpected, +8.5 fit: t278 the bandit (bandit): strike Oren with the club (knocked Oren down)  (seed13.events.jsonl)
- 9.1 unexpected, +6.6 fit: t168 the bandit (bandit): steal the lute from Oren  (seed4.events.jsonl)
- 8.9 unexpected, +6.4 fit: t211 the bandit (bandit): steal the sword from Sir Ada  (seed16.events.jsonl)
- 8.9 unexpected, +7.2 fit: t214 the bandit (bandit): steal the sword from Sir Ada  (seed16.events.jsonl)
- 8.6 unexpected, +7.9 fit: t262 the bandit (bandit): steal the sword from Tom  (seed17.events.jsonl)
- 7.8 unexpected, +5.7 fit: t148 the bandit (bandit): steal the sword from Sir Ada  (seed18.events.jsonl)
- 7.7 unexpected, +5.6 fit: t66 Fenn (thief): tell Sir Ada that Oren took the sword  (seed15.events.jsonl)
- 7.6 unexpected, +7.1 fit: t243 Oren (merchant): flee to the kitchen  (seed1.events.jsonl)
- 7.6 unexpected, +4.8 fit: t292 the bandit (bandit): tell Sir Ada that Tom took the sword  (seed18.events.jsonl)
- 7.5 unexpected, +5.6 fit: t196 the bandit (bandit): steal the lute from Oren  (seed6.events.jsonl)

## Unexpected choices the character does not explain (odd draws: sense, or noise?)

- 10.8 unexpected, +3.8 fit: t60 Fenn (thief): tell Marta that Oren took the sword  (seed9.events.jsonl)
- 10.4 unexpected, +6.3 fit: t156 the bandit (bandit): steal the dagger from Fenn  (seed18.events.jsonl)
- 10.1 unexpected, +0.0 fit: t208 Marta (cook): shove the bandit  (seed14.events.jsonl)
- 9.9 unexpected, +3.3 fit: t205 Sir Ada (knight): steal the copper coin from Tom  (seed19.events.jsonl)
- 9.4 unexpected, +0.1 fit: t98 Marta (cook): give the ladle to Fenn  (seed16.events.jsonl)
- 9.4 unexpected, +3.3 fit: t268 Fenn (thief): tell Marta that Sir Ada took the purse of silver  (seed2.events.jsonl)
- 9.0 unexpected, +3.9 fit: t132 the bandit (bandit): threaten Sir Ada  (seed18.events.jsonl)
- 9.0 unexpected, +1.1 fit: t120 Fenn (thief): tell Marta that Tom took the purse of silver  (seed3.events.jsonl)
- 8.9 unexpected, +4.0 fit: t114 Fenn (thief): tell Tom that Sir Ada took the purse of silver  (seed5.events.jsonl)
- 8.9 unexpected, +1.3 fit: t270 Sir Ada (knight): threaten Oren  (seed12.events.jsonl)

## Story patterns

| pattern | share of runs | matches per run |
| --- | --- | --- |
| revenge | 100% | 4.7 |
| betrayal | 75% | 1.3 |
| rescue | 100% | 8.6 |
| lie found out | 20% | 0.2 |
| coward acts bravely | 100% | 2.0 |
| gift returned | 0% | 0.0 |

Rarest and most surprising matches:

- revenge (seed3.events.jsonl): ?x=Fenn, ?a=Oren
- rescue (seed14.events.jsonl): ?x=Marta, ?v=the bandit, ?h=Sir Ada
- betrayal (seed18.events.jsonl): ?a=Fenn, ?b=Oren
- rescue (seed18.events.jsonl): ?x=the bandit, ?v=Fenn, ?h=Tom
- rescue (seed18.events.jsonl): ?x=the bandit, ?v=Fenn, ?h=Oren
- revenge (seed7.events.jsonl): ?x=Marta, ?a=Sir Ada
- revenge (seed14.events.jsonl): ?x=Marta, ?a=the bandit
- revenge (seed18.events.jsonl): ?x=the bandit, ?a=Fenn
- rescue (seed18.events.jsonl): ?x=the bandit, ?v=Fenn, ?h=Sir Ada
- lie found out (seed8.events.jsonl): ?a=Fenn, ?b=Tom

## Chains

- choices that react to another character: 22%
- longest chain: 14 (the bandit > Fenn > the bandit > Sir Ada > Fenn > the bandit > Fenn > the bandit > Fenn > the bandit > Fenn > the bandit > Fenn > Sir Ada)

## Variety between runs

- map covered (harm x help, 5x5): 8%
- how different two runs are, by what each type did (0 = the same, 1 = nothing shared): 0.41

| type | variety of actions (bits) | most common |
| --- | --- | --- |
| bandit | 3.02 | walk to 196, wait 181, strike 116 |
| cook | 2.93 | wait 504, walk to 482, drink 189 |
| knight | 2.96 | wait 539, walk to 350, strike 197 |
| merchant | 3.03 | walk to 508, wait 334, flee to 241 |
| peasant | 2.61 | walk to 613, wait 504, flee to 190 |
| thief | 3.17 | walk to 436, wait 385, strike 211 |

## The world

- choices: 9308; stopped before done: 21%; deaths per night: 0.10; lies told: 69, found out: 4
- most common actions: walk to 2199, wait 2164, drink 763, flee to 552, accuse 352, strike 336, take 227, tell 133, search 117, hide 96, play 72, defend 49

## Type behaviour: what a witness does next after seeing a blow or a threat

| type | brave | away | other | n |
| --- | --- | --- | --- | --- |
| bandit | 4% | 31% | 65% | 26 |
| cook | 18% | 44% | 39% | 96 |
| knight | 19% | 16% | 66% | 70 |
| merchant | 4% | 65% | 31% | 78 |
| peasant | 10% | 70% | 21% | 112 |
| thief | 8% | 48% | 44% | 63 |
