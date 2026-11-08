# Action catalog (design, for your notes)

Written in Simplified Technical English. This is Workstream D of
`docs/character-models-plan.md`. It lists what the player (and every other
character) can do, how the player chooses it, and how actions take time.
Nothing is built from it until you give notes.

## 1. Rules for the catalog

1. **The same actions for everyone.** The player's character and all other
   characters use the same catalog. A model that runs a character chooses
   from these actions. It does not invent new ones.
2. **The world decides the result.** An action is what a character tries.
   The world decides what happens: a thrown key lands where physics puts
   it, and a weak swing does little.
3. **Only possible actions are shown.** If a door has no lock, "lock" is
   not shown for it.
4. **Groups stay small.** A group shows at most about 7 actions (proposed).
   More than that goes into a second level.

## 2. How the player chooses

1. The player selects a **target**: a thing, a person, a place, or their own
   character.
2. The screen shows the **groups** that have at least one possible action
   for that target.
3. The player selects a group, then an **action**.
4. The player can add **parts** (all are optional):
   - with what ("with the torch")
   - how ("carefully", "quietly", "with force", "in anger")
   - then ("then close the door")
   - until ("until the guard looks away")
5. The player confirms. The character starts the action.

Example: select the guard → Fight → Strike → with: knee, how: with force.

Example: select the key → Handle → Throw → at: the well, how: in anger.
The world decides where the key lands.

## 3. The groups

Status column: **yes** = the voxel simulation can do it now; **part** = some
of it; **no** = not yet. "Step" refers to the build order in roadmap 95.

### Move

| action | target | parts | status |
| --- | --- | --- | --- |
| walk to | place, thing, person | how (quietly = sneak) | yes |
| run to | place, thing, person | | part (one speed now) |
| climb | thing, wall | | no |
| jump | place, gap | | yes |
| crouch, lie down, stand up | self | | part (crouch yes) |
| hide | behind or in a thing | | no (needs step 5, light) |
| follow | person | until | no |
| flee from | person, thing, place | | part (go to the exit) |
| swim | water | | no (step 6) |
| stop | self | | yes |

### Look and listen

| action | target | parts | status |
| --- | --- | --- | --- |
| look at, look toward | thing, person, direction | | yes |
| search | place, thing | carefully | no |
| listen | direction, door | | part (hearing is automatic) |
| watch | person | until | no |
| read | written thing | | no |

"Look" chooses where the eyes point. What the character sees next depends
on it. A character sees what is behind it only after it turns to look.

### Handle

| action | target | parts | status |
| --- | --- | --- | --- |
| take, pick up | thing | from where | yes |
| drop | held thing | | yes |
| put | held thing | in, on | no |
| give | held thing | to whom | no |
| open, close | door, lid, container | | no (step 8) |
| lock, unlock | door, container | with which key | no (step 8) |
| push, pull | thing, person | how hard, which way | part (shove) |
| lift, carry, drag | thing, person | | yes (drag a person) |
| throw | held thing | at, to, how hard | yes |
| pour | held vessel | into, onto | part (vessels spill) |
| light, put out | thing | with what | part (fire exists) |
| break | thing | with what | part (swing breaks glass) |
| cut, tie | rope, cloth | with what | no |
| dig | ground | with what | no |
| bar, wedge | door | with what | no (step 8) |
| pull a lever, press | mechanism | | no (step 8) |
| set, disarm | trap | | no (step 8) |

### Fight

| action | target | parts | status |
| --- | --- | --- | --- |
| strike | person, thing | with (fist, foot, knee, head, weapon), how hard | part (swing an arm or held thing) |
| shoot | person, thing | with (bow, sling) | part (projectiles exist, no bow) |
| throw at | person, thing | with (held thing) | yes |
| block, parry | incoming blow | with (shield, weapon) | no |
| dodge | incoming thing | | no |
| grapple, hold | person | | yes (hold) |
| shove | person | which way | yes |
| disarm | person | | no |

### Talk

Talk has a meaning and words (see the plan, Workstream G). The player
chooses the meaning. The words come from the chosen speech method.

| action | target | parts | status |
| --- | --- | --- | --- |
| tell | person | what (a fact) | no |
| ask | person | about what | no |
| order | person | to do what | no |
| threaten | person | or what | no |
| offer, promise | person | what, for what | no |
| plead | person | for what | no |
| insult, praise | person | | no |
| lie | person | what (a false fact) | no |
| greet | person | | no |
| shout | everyone near | what | part (fixed lines: "Fire!") |
| whisper | person | what | no |

### Help and care

| action | target | parts | status |
| --- | --- | --- | --- |
| press on a wound | person, self | | no (bleeding exists) |
| bandage | person, self | with what | no |
| carry | person | to where | yes (drag) |
| help up | person | | no |
| give food, give drink | person | what | no |
| wake | person | how (shake, shout) | part (noise wakes) |

### Use and make

| action | target | parts | status |
| --- | --- | --- | --- |
| use | held thing | on what | no |
| eat, drink | food, drink | | no |
| combine, brew | things | at (fire, bench) | no (step 11) |
| forge | metal | at (forge) | no (step 11) |

### Skill and magic (later)

Each spell is a source that acts through world laws that already exist (see
roadmap 95). It is not a special case. Magic and "rule of cool" effects that
have no law go to Workstream H (deciding effects), inside limits.

### Self

| action | target | parts | status |
| --- | --- | --- | --- |
| wait | self | until | yes (stay) |
| rest | self | how long | no |
| hold breath | self | | no (step 6) |
| brace | self | against what | part (lean) |

## 4. Actions take time

**The world does not wait.** While the player's character acts, all other
characters act, and the world laws run.

**Each action has three parts:**

1. Start: for example, raising the arm.
2. Act: for example, the arm swings.
3. Finish: for example, the arm comes back.

**Durations (proposed, in turns of 0.025 s):** a step is about 3 turns; a
swing is about 10; opening a door is about 20; picking a lock is about 200.
The world laws decide the real time. A heavy thing takes longer to lift.

**Changing an action.** At each turn, a character can stop its action and
choose again. Exceptions:

- It is in the air (jumping or falling).
- It cannot act (unconscious, asleep, held).
- Its act part has passed a point of no return (a thrown thing has left the
  hand).

**Momentum continues after a change.**

- Physical: a running body keeps its speed and must slow down.
- Emotional: anger and fear fade slowly. They push the next choices. A
  character that was angry a moment ago still throws harder.

**Stopping the player.** When something important happens (a loud crash, a
blow, fire in view), the game stops the player's action and asks the player
to choose again. This agrees with the "world keeps running" design (the job
system, with a stop when something important is seen).

## 5. What the simulation needs for this catalog

- A **duration** and a **stop point** for each action.
- An **"until"** condition that the world can test each turn.
- A **queue** for "then".
- **Manner** values that change the world result: "with force" adds force and
  takes away accuracy; "carefully" does the opposite; "quietly" makes less
  noise.
- **Emotion as momentum**: a mood that fades over time and changes force,
  accuracy and choice.

## 6. Questions for your notes

1. Are the groups right? Should "Help and care" be part of "Handle"?
2. At most 7 actions in a group: good?
3. The parts "with what", "how", "then", "until": is anything missing?
4. The durations in Section 4: good first values?
5. Which important events stop the player's action?
6. Talk: should the player see the speech acts (tell, ask, threaten), or
   choose from ready lines?
