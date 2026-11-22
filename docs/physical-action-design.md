# One interface for physical action

Scoped with the owner on 2026-11-21 (decisions PA1-PA6). The owner: "I don't want to make 100 specialised interfaces; a smaller set or just one which works across different physical interactions." For climbing: choose WHERE to try to grab, and then what the character can see and their specs decide the attempt.

## PA1. Six motor primitives

| Primitive | What it does | Examples |
|---|---|---|
| MOVE | put a body part at a point, or along a path | reach for a ledge, raise a shield |
| PRESS | apply force in a direction | push, pull, lift, brace, pry |
| GRIP / RELEASE | close or open a hand on something | take, hang, let go, the release of a throw |
| STRIKE | a fast, committed swing through a point | punch, kick, chop, headbutt |
| STEP / SHIFT | move the weight and the base of support | walk, jump, plant the feet, lean |
| HOLD | keep a configuration under load | hang, carry, hold a door shut |

- Each primitive takes three things:
  - a BODY PART: a hand, both hands, a foot, a knee, an elbow, the shoulder, the head, the whole body, or a held tool, which extends the part;
  - a TARGET the character PERCEIVES (K10 in docs/knowledge-design.md);
  - HOW HARD, HOW FAST and HOW CAREFULLY, where speed trades against precision.
- Everything else is a SEQUENCE:
  - climbing is grip, press and step, repeated;
  - a throw is grip, a fast move, then release at the right point;
  - a catch is a move and a grip, timed;
  - opening a latch is a move, then a press or a pull;
  - a tackle is a fast step plus a whole-body strike.
- Resolution goes through the existing and coming laws: reach and pose (pose.py), perception (the sight laws), exertion from the body (wave 14b G), balance and body contact (14b H and E), and grip and friction (wave 15 S2).
- The voxel backend already runs acts as limb jobs (limb, match, until). The primitives expose that layer.
- Today's 28 catalogue verbs (src/mind/data/verbs.yaml), the menus, cached plans and spell gestures become SHORTCUTS that fill in sequences of primitives.

## PA2. Choosing where (text first, ALWAYS a text-only option)

- NOW:
  - (a) a LIST of what the character perceives, described in the body's terms ("a ledge at chest height, a hand-span to your left", "his right wrist"); it is short and vague in the dark ("something rough under your fingers, about here");
  - (b) FREE TEXT translated into primitives within the fixed language (plan-cache Q18), refused if the target is not perceived, and a shortcut only for proven or implicit skill (magic Q15).
- LATER: (c) a spatial picker in a graphical front end.
- STANDING: MARK1 may move beyond text, but there must ALWAYS be a TEXT-ONLY OPTION. Graphics is another front end over the same primitives.
- Perception sets how accurately the chosen target is hit. A misjudged ledge in poor light makes the fingers slip, from the sight laws, with no dice.

## PA3. Time

- A short sequence is QUEUED and runs until it is done or until something SALIENT interrupts it: a slip, a hold giving way, a threat, a sound the character would notice. Salience comes from the character's perception and drives, so a nervous climber stops more often than a calm expert. It is the world-tick momentum design.
- STEP-BY-STEP is available whenever the player wants it.
- Real-time control is to be tried later, after this works well.

## PA4. One interface for everyone

- NPC verbs and cached plans expand into the same primitives. NPCs can learn new sequences through the plan cache (a thief works out how to climb the back wall).
- There is no difference in WHAT can be done, only in who chooses and how.
- NPCs stay cheap: a menu shortcut, with 0 model calls in reactive play.

## PA5. Skills

- A working sequence can be saved as a personal TECHNIQUE, a repertoire record in the plan cache (plan-cache Q17).
- PRACTICE CHUNKS it: fewer decisions, faster, less noise (magic Q10).
- It fades with disuse and can be mis-learned (use-dependent plasticity, magic MC3). It passes on by teaching, demonstration being best (knowledge K4).
- The PLAYER chooses the strategy, and the CHARACTER'S proficiency sets the execution (precision, speed, fatigue).
- There are no skill numbers; a readable summary is described from the records ("a practised climber, at ease on rough stone").

## PA6. When

- WAVE 16a: the primitive layer, on 14b's exertion and contact and wave 15's grip, with the 28 verbs re-expressed as shortcuts.
- WAVE 16b: the plan cache and the knowledge layer, built on the primitives.
- Magic's gesture interfaces come after, on the same primitives.
