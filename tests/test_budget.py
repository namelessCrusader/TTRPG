"""The anti-85k ratchet. The old engine died of unbounded accretion; this test
makes growth a DECISION instead of a drift.

Rules it enforces:
- CODE is budgeted. Exceeding a cap fails CI until you either compact something
  or consciously raise the cap in this file (a reviewed, one-line confession).
- DATA (yaml rules, eval cases) and TESTS are deliberately unbudgeted — content
  may accrete freely (NetHack-style); mechanisms may not.

When a module hits its cap, the FIRST question is "what does this collapse into?"
(see docs/OPEN_DESIGNS.md compression map), not "raise the cap".
"""

import os

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "core")

# module → line cap. Headroom ≈ +25% over today's size; raising a cap is a
# design decision that belongs in a diff, with a reason.
CAPS = {
    "composer.py": 900,     # fattest; synonyms/guards accrete — watch for code→data moves
    "mind.py": 540,         # the collapse to utility scoring is DONE: fight-or-flight + confront + social all
                            # folded into ONE drive-utility selector (_react + _aggress, seeded-softmax pick).
                            # Net -40 lines vs the three hand-authored handlers. Behaviour EMERGES from
                            # drives × beliefs × affordances — no if-then chain. (was 565 mid-fold-in; debt paid.)
    "engine.py": 690,       # raised 665→690: every Option now carries verb+args (the action's
                            # structured parts — decision-2 "composed tree from listed pieces").
                            # Previously 645→665: affordance_menu is now the D8 legality predicate —
                            # + the parts.affords hook (options a component advertises, D2) —
                            # any actor + PERCEPTION-FILTERED (entity options bind only to fresh
                            # belief nodes from perceive(); frozen-slice param = double-buffer).
                            # Previously 625→645: broadened speech-verb surface (taunt/reason/plead
                            # route to dialogue), out-of-earshot refusal, animals render.
                            # Previously 520→580: `describe` — the LOOK renderer. The same facts
                            # cell_info dumps, spoken as prose ("a puddle of water wets the
                            # floor"), plus the stats builtin. cell_info stays the debugger; the
                            # fiction never shows units or coordinates again. Previously 480→520:
                            # the pack-vocabulary seam in free_text (pack verbs first claim,
                            # inventory builtin, the concept gate).
    "social.py": 560,       # raised 545→560: _pack_line — a pack voices STANCES in its own
                            # register (the A/B ceiling fix: a farm NPC no longer says "...in this
                            # den!"). Previously 500→545: the AgentBrain — a big model plays DM by SELECTING a
                            # typed stance (choose() defers via DMChoice, the harness surfaces the
                            # choice-point, the DM picks; never free-writes a line). Grounded
                            # (ctx→pick) harvested. Previously 460→500: per-trait dialogue VARIANTS (a brave guard and a timid
                            # apprentice voice the same stance differently) — authored CONTENT, the kind the
                            # ratchet allows to accrete. When it grows more, move REACTIONS to a yaml (like reactions).
    "reactions.py": 475,    # raised 440→475: pack_affords — run 2 found pack verbs (open/descend)
                            # lived only on the free-text path; the piece table could not win
                            # the dungeon. Earlier 435→440: the parts.step hook — the D13 behavior layer proposes
                            # its effects at the top of tick, before rules read heat.
                            # Previously 400→420: narration VARIANTS (_voice: narrative: fields may be
                            # lists, seeded-hash pick — the seam big models pour book-voice into,
                            # offline, as data) + at:entity verb targeting + clock_tick op.
                            # Previously 340→400: PACKS — a LitRPG "system" (cultivation, a dungeon
                            # System) is DATA: concepts + verbs + reactions in data/packs/*.yaml.
                            # The interpreter grew one general seam (load_pack/pack_verb +
                            # transfer_prop, a conserved continuous transfer); each genre costs
                            # zero further code. Also cell_has_tag (location-coupled effects).
    "webserver.py": 280,
    "state.py": 220,
    "__main__.py": 360,     # raised 310→360: the DM choice-point protocol — session `choose`
                            # resumes a deferred social turn with the DM's stance (deterministic
                            # replay + _dm_prompt + harvest). Previously …→310: session --pack/--scene — playtests pick their
                            # world and its LitRPG systems. Previously: `session` driver — agent
                            # playtest loop harvesting traces.
                            # Now TAG-NAMESPACED so many Haiku agents play in PARALLEL (own world+shard), plus
                            # `session merge` to pool shards into the master corpus. Diverse intents → diverse traces.
    "seed.py": 290,         # raised 250→265: the dungeon grew a SECOND STOREY (z=1 gallery over
                            # z=0 dark; tiered Bronze/Silver/Gold boxes; the broodmother below).
                            # Scenes are CONTENT-adjacent but stay code until layer-map authoring
                            # lands — which should then shrink this whole module.
    "eval_composer.py": 180,
    "effects.py": 175,      # raised 160→175: spawn/despawn — entity lifecycle, the one primitive a
                            # closed-state VM still needs (summon/mint/birth). Through the bus, logged.
    "spatial.py": 150,
    "checks.py": 170,       # the adjudication organism: bands/consequences/oracle/clocks
    "director.py": 170,     # chaos dial + lull + GM-move menu; moves are content, keep the shape
    "acts.py": 130,         # NPC jobs (DwarfCorp port): job KINDS are content, the stepper is the shape
    "llm.py": 145,          # raised 100→145 for batched scoring + prior cache (harness-sweep port #1)
    "knn.py": 100,          # kNN-Prompting: frozen model as feature extractor, decision by vote
    "combat.py": 135,       # NEW seam: combat physics (to-hit + damage type×material) pulled OUT of composer
    "trace.py": 70,         # NEW seam: protoreasoning-trace harvest — capture scored picks, learn only the
                            # sim-BLESSED ones into the kNN datastore (the corpus any later distillation needs)
    "playtest.py": 135,     # NEW seam: the play-test loop on the real engine — prompt (view +
                            # piece table) → composed answer → parse → step; NPCs act on mind.py;
                            # a refusal names its piece and costs no time; corpus rows logged.
    "pieces.py": 200,       # 170→200: ticket 1 from AI run #1 — condition VALUES are listed
                            # pieces too (table lists ids/tags/carried; parser refuses a
                            # guessed tag by name — 'vial' vs has_prize burned us).
                            # NEW seam (decision 2): the piece TABLE (verb rows + slot values) +
                            # the reverse index + the PARSER: flat lines, bare-value slotting,
                            # (then …) sequences, (if (cond) A B) with run-time checks — every
                            # refusal NAMES the bad piece. contest waits for the dsl bridge.
    "reach.py": 90,         # NEW seam (item 8): the reachable-state counter — distinct salient
                            # states at depth k, no author/LM/harvest in the loop. The feedback
                            # loop that scores world edits (before/after) in milliseconds.
    "parts.py": 90,         # NEW seam (D13): the BEHAVIOR layer of the hybrid schema — stateful/
                            # resource-holding components (fuel_burn first). Private counters mutate
                            # internally; world-visible consequences ride the effect bus. Boundary:
                            # material→template, static-boolean→tag, changes-over-ticks→part.
    "perceive.py": 150,     # NEW seam: perceive() — an entity's belief-slice (observe() folds witnessed
                            # events; unseen beliefs persist STALE with an age). The SINGLE perception
                            # source behind the perception-filtered affordance menu (design doc D8).
                            # Ported from the data-engine prototype after the scratchpad was wiped.
    "dsl.py": 250,          # NEW seam: the TOTAL, loop-free, recursive program language the trained DM
                            # emits over the effect VM — the general (state,intent,target)→edit operation
                            # verbs/reactions/stances are all special cases of. The sim is the authoritative
                            # interpreter (executes the exact deterministic/seeded edit; refuses malformed
                            # atomically). + canonicalize (absorb free-form dialects) + validate (dry-run
                            # filter for the data engine). Design settled by the free-vs-scaffold spike.
}                           # — reads the sim, never fakes it; composer shrank 889→797 to make room for it
TOTAL_CAP = 6070            # 6030→6070: pack verbs join the menu (reactions.pack_affords). Earlier: 5990→6030: ticket 1 — condition values become listed pieces (pieces.py). Earlier: 5860→5990: playtest.py, the loop on the real engine. Earlier: 5730→5860: the decision-2 parser (pieces.py). Earlier: 5690→5730: Option verb/args + the piece-table. Earlier: 5610→5690: reach.py — the item-8 reachable-state counter (the world-
                            # design feedback loop; first measurement: +2 parts ≈ 3× states). Earlier:
                            # 5500→5610: parts.py — the D13 behavior layer (Entity.parts + stepper +
                            # liquid_volume, the first RESOURCE part: tip/draw, amounts sim-owned (D6);
                            # AFFORDS: parts advertise menu options; set_part = bus-clean part writes;
                            # fuel_burn first) + the any-actor/perception-filtered menu work. Earlier:
                            # 5370→5500: perceive.py — the belief-fold / single perception source (D8),
                            # ported into the repo as the input to the perception-filtered affordance menu
                            # (item 1 of the engine build). Earlier:
                            # 5110→5370: the DSL — a total, loop-free, recursive program language the
                            # trained DM emits over the effect VM (dsl.py, the general operation verbs/
                            # reactions/stances specialize) + spawn/despawn on the bus. This is the
                            # model-output design, settled by grill + free-vs-scaffold spike. Earlier:
                            # 5060→5110 A/B fixes. Old engine was 85,000 — ~15.9x under. (big-model-DM vs local torch) — the
                            # CEILING fix (pack-overridable stance lines: farm/beast registers, so the
                            # DM's right-lane pick finally speaks in-genre), the death-tick INCOHERENCE
                            # ("dead but hp=8" → death wins the tick), player fall grammar, animals
                            # render, talk-to-monster is dialogue, out-of-earshot refusal, no coordinate
                            # leaks. Earlier: 4990→5060 the DM brain. Old engine was 85,000 — ~16.6x under.


def _lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def test_no_module_exceeds_its_budget():
    over = []
    for name, cap in CAPS.items():
        n = _lines(os.path.join(SRC, name))
        if n > cap:
            over.append(f"{name}: {n} > {cap}")
    assert not over, ("COMPACT BEFORE YOU GROW — over budget: " + "; ".join(over)
                      + ". Collapse something (see OPEN_DESIGNS.md compression map) "
                        "or raise the cap here, on purpose, with a reason.")


def test_engine_total_stays_small():
    total = sum(_lines(os.path.join(SRC, f)) for f in os.listdir(SRC) if f.endswith(".py"))
    assert total <= TOTAL_CAP, \
        f"engine at {total} lines (cap {TOTAL_CAP}). The last engine died at 85,000. Compact."


def test_new_modules_are_a_decision():
    known = set(CAPS) | {"__init__.py"}
    strays = [f for f in os.listdir(SRC) if f.endswith(".py") and f not in known]
    assert not strays, (f"new module(s) {strays}: adding a module is a seam decision — "
                        "give it a cap here and say why it isn't part of an existing one.")
