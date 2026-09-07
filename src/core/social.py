"""Social layer (fork B) as CONSTRAINED SELECTION, not generation.

The engine authors a small set of in-character reactions per intent (a canned
line + pre-bounded deltas). The LM's only job is to CHOOSE one that fits the
npc's current state — a single forward pass, no token generation, no JSON. The
output is always inside the authored set, so it is coherent by construction and
a tiny (0.5B) model can do it fast. This is "define the feasible interactions,
let the LM select among them" — restrictive decoding over a fixed vocabulary.

Whatever the brain picks still flows through _sanitize and the one bus: content
matters (the brain reads composure/emotion/disposition), but nothing it decides
touches state except as validated, capped, logged deltas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import effects as fx
from .state import World

EMOTIONS = {"calm", "afraid", "hostile", "compliant"}
EDGE_KINDS = {"disposition", "fear", "trust"}

# kNN-Prompting: the frozen selector learns from play. DATASTORE is loaded once
# (None = disabled, e.g. fuzz); trace.py captures picks and flushes the blessed
# ones here via DATASTORE.add — see [[trace-harvest]].
DATASTORE = None
KNN_ALPHA = 0.4


def _trait_words(t: dict) -> str:
    """The 4 numeric traits → the salient adjectives (only the strong ones)."""
    say = []
    if t.get("bravery", .5) >= .7: say.append("brave")
    elif t.get("bravery", .5) <= .3: say.append("timid")
    if t.get("aggression", .5) >= .7: say.append("hot-tempered")
    if t.get("warmth", .5) >= .7: say.append("warm")
    elif t.get("warmth", .5) <= .3: say.append("cold")
    if t.get("self_interest", .5) >= .7: say.append("self-serving")
    elif t.get("self_interest", .5) <= .3: say.append("loyal")
    return ", ".join(say) or "even-keeled"


def persona_card(world: World, npc) -> str:
    """Scene-gated persona (harness #4 = World Info without keywords): each NPC's
    OWN sheet — traits + flaw + bond + live rapport — compact enough for a tiny
    model's context. This replaces the one hard-coded 'doorman' persona."""
    m = npc.mind
    disp = world.edges.get((npc.id, "player"), {}).get("disposition", 0.0)
    rapport = ("You are cold and wary toward the player — you distrust them."
               if disp <= -0.2 else
               "You have warmed to the player." if disp >= 0.2 else
               "You barely know the player.")
    lines = [f"You are {npc.name}. You are {_trait_words(m.get('traits', {}))}."]
    if m.get("flaw"):
        lines.append(f"Your flaw: {m['flaw']}.")
    if m.get("bond"):
        lines.append(f"What you care about: {m['bond']}.")
    lines.append(rapport)
    return "\n".join(lines)
MAX_COMPOSURE_STEP = 20.0
MAX_EDGE = 1.0


@dataclass
class Reaction:
    id: str
    line: str
    deltas: list  # ('emotion', tag) | ('composure', d) | ('edge', kind, w)
    variants: dict | None = None   # trait-word -> alternate surface line (same stance)


# trait words (from _trait_words) in the order we prefer to voice — the FIRST of
# the npc's strong traits that has a variant for this reaction wins, so a brave
# guard and a timid apprentice say the same STANCE in their own register.
_VOICE_ORDER = ("hot-tempered", "brave", "timid", "cold", "warm", "self-serving", "loyal")


def _voiced(reaction: "Reaction", npc) -> str:
    """The reaction's surface line, flavored by the npc's dominant trait if a
    variant exists — else the neutral default. Voice varies; stance/mechanics don't."""
    if reaction.variants:
        strong = set(_trait_words(npc.mind.get("traits", {})).split(", "))
        for w in _VOICE_ORDER:
            if w in strong and w in reaction.variants:
                return reaction.variants[w]
    return reaction.line


# The feasible interaction set. Each option is safe by construction; the LM only
# decides which fits. (Author more per intent to widen the NPC's repertoire.)
REACTIONS: dict[str, list[Reaction]] = {
    "threaten": [
        Reaction("cower", "P-please — I don't want any trouble!",
                 [("emotion", "afraid"), ("composure", -15), ("edge", "fear", 0.5)],
                 variants={"timid": "P-please — I don't want any trouble!",
                           "brave": "You'll regret laying a hand on me.",
                           "hot-tempered": "Touch me and I'll take your arm off!"}),
        Reaction("defy", "You don't frighten me. Try it.",
                 [("emotion", "hostile"), ("edge", "disposition", -0.3)],
                 variants={"hot-tempered": "Try it. I'll bury you.",
                           "cold": "You don't frighten me. Try it.",
                           "loyal": "I'll not betray the guild — do your worst."}),
        Reaction("comply", "Alright! Alright — what do you want?!",
                 [("emotion", "compliant"), ("composure", -8), ("edge", "fear", 0.4), ("edge", "disposition", 0.1)]),
    ],
    "plead": [
        Reaction("relent", "...Fine. I'm listening. For now.",
                 [("composure", 5), ("edge", "disposition", 0.2)],
                 variants={"warm": "Oh — alright, alright. Tell me what's wrong.",
                           "cold": "...Fine. I'm listening. For now.",
                           "self-serving": "And what's in it for me? ...Go on, then.",
                           "loyal": "For the guild's sake, then. Speak."}),
        Reaction("refuse", "Save your breath.",
                 [("edge", "disposition", -0.1)],
                 variants={"hot-tempered": "Save your breath before I lose my patience.",
                           "cold": "Save your breath."}),
    ],
    "talk": [
        Reaction("wary", "(warily) What do you want?",
                 [("edge", "disposition", 0.05)],
                 variants={"timid": "(shrinking back) Wh-what do you want?",
                           "brave": "State your business.",
                           "cold": "(warily) What do you want?"}),
        Reaction("warm", "Good to see a friendly face in this den.",
                 [("emotion", "calm"), ("edge", "disposition", 0.2)],
                 variants={"warm": "Good to see a friendly face in this den!",
                           "self-serving": "Well now — a friendly face. What can you do for me?",
                           "loyal": "Good to see someone honest about the place."}),
        Reaction("dismiss", "I'm busy. Go bother someone else.",
                 [("edge", "disposition", -0.1)],
                 variants={"hot-tempered": "I'm busy. Clear off.",
                           "cold": "I'm busy. Go bother someone else.",
                           "self-serving": "Unless you're paying, I'm busy."}),
    ],
}


def _build(npc: str, deltas: list) -> list:
    out = []
    for d in deltas:
        if d[0] == "emotion":
            out += [fx.clear_tag(npc, e) for e in EMOTIONS if e != d[1]]
            out.append(fx.set_tag(npc, d[1]))
        elif d[0] == "composure":
            out.append(fx.adjust_prop(npc, "composure", d[1]))
        elif d[0] == "edge":
            out.append(fx.set_edge(npc, "player", d[1], d[2]))
    return out


class SocialBrain:
    """choose(ctx, options) -> index into options."""

    def choose(self, ctx: dict, options: list) -> int:
        raise NotImplementedError


# flat repertoire for free-text ("say anything") — the LM routes to what fits
ALL_REACTIONS: list[Reaction] = []
for _rs in REACTIONS.values():
    for _r in _rs:
        if _r.id not in {x.id for x in ALL_REACTIONS}:
            ALL_REACTIONS.append(_r)


def _find(options: list, rid: str) -> int:
    for i, o in enumerate(options):
        if o.id == rid:
            return i
    return 0


# free-text intent → the reaction ids that fit it (stage-2 picks within by state)
INTENTS = {
    "threatening": ["cower", "defy", "comply"],
    "friendly":    ["warm", "relent", "wary"],
    "insulting":   ["defy", "dismiss"],
    "neutral":     ["wary", "warm", "dismiss"],
}


class MockBrain(SocialBrain):
    """Deterministic, state-dependent. Free; keeps fuzz reproducible."""

    def choose(self, ctx: dict, options: list) -> int:
        comp = ctx["composure"]
        utt = (ctx.get("utterance") or "").lower()
        soured = ctx.get("disposition", 0.0) <= -0.2   # they know what you did
        tone = ctx.get("tone")
        # patience wears: after being pestered several times in quick succession,
        # even a civil NPC waves you off — unless you're actively threatening.
        if ctx.get("pestered", 1) >= 4 and tone != "threatening" and "dismiss" in [o.id for o in options] \
                and not any(w in utt for w in ("kill", "hurt", "knife", "threat", "die")):
            return _find(options, "dismiss")
        if tone in INTENTS:                             # declared tone routes directly
            if tone == "threatening":
                return _find(options, "cower" if comp <= 45 else "defy")
            if tone == "friendly" and not soured:
                return _find(options, "relent" if comp <= 55 else "warm")
            if tone == "insulting":
                return _find(options, "defy" if comp > 45 else "dismiss")
        if utt:  # keyword routing for typed input
            if any(w in utt for w in ("kill", "hurt", "or else", "knife", "burn you", "threat", "die")):
                return _find(options, "cower" if comp <= 45 else "defy")
            if soured:                                  # no warmth for the disgraced
                return _find(options, "dismiss")
            if any(w in utt for w in ("please", "help", "sorry", "friend", "trust", "deal", "gold")):
                return _find(options, "relent" if comp <= 55 else "warm")
            return _find(options, "wary")
        if soured and ctx["intent"] == "talk":
            return _find(options, "dismiss")
        if ctx["intent"] == "threaten":
            return _find(options, "cower" if comp <= 45 else "defy")
        return 0


class TorchChoiceBrain(SocialBrain):
    """A real local LM that SCORES the options and picks one (single forward pass).

    Works with any small instruct model; default is a 0.5B. Falls back to a simple
    heuristic on any failure so a flaky model never breaks the turn.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B"):
        self.model_id = model_id

    @property
    def _lm(self):
        from .llm import get_lm           # ONE shared, cached model per process
        return get_lm(self.model_id)

    def _load(self):
        self._lm.load()

    def choose(self, ctx: dict, options: list) -> int:
        try:
            if ctx.get("utterance"):
                return self._choose_freeform(ctx, options)
            return self._score(ctx, options)
        except Exception:
            return MockBrain().choose(ctx, options)

    def _choose_freeform(self, ctx: dict, options: list) -> int:
        """Two-stage: classify the utterance's intent (easy for a small model),
        then pick the in-state reaction within that intent (already reliable).
        A player-declared TONE overrides classification — how you say it is yours."""
        intent = ctx.get("tone") if ctx.get("tone") in INTENTS else self._classify(ctx["utterance"])
        ids = INTENTS.get(intent, ["wary", "dismiss"])
        subset = [o for o in options if o.id in ids] or options
        s = dict(ctx); s["utterance"] = None; s["intent"] = intent   # stage 2 uses state, not text
        return _find(options, subset[self._score(s, subset)].id)

    def _classify(self, utterance: str) -> str:
        labels = list(INTENTS)
        sysmsg = "You classify the TONE of what a player says to a character. Judge only the words."
        cond = (f'The player says: "{utterance}"\n'
                "Is that threatening, friendly, insulting, or neutral? Answer with one word.")
        neutral = "Answer with one word: threatening, friendly, insulting, or neutral."
        return labels[self._lm.pick(sysmsg, cond, neutral, labels)]

    @staticmethod
    def _state_plausible(ctx: dict, options: list) -> list:
        """The sim gates, the LM chooses WITHIN the gate: composure is a hard
        band, not a suggestion a tiny model may overrule for flavor. Persona
        picks the HOW (cower vs comply; defy vs dismiss) — never the whether."""
        c = ctx.get("composure", 50.0)
        drop = {"defy"} if c < 30 else {"cower"} if c > 75 else set()
        kept = [o for o in options if o.id not in drop]
        return kept or options

    def _score(self, ctx: dict, options: list) -> int:
        """Contrastive (PMI) selection: pick argmax of logP(option | real state) minus
        logP(option | neutral). Subtracting the neutral score cancels the model's
        built-in prior toward certain option strings, so the choice reflects STATE,
        not which word the model likes to say. Robust with a tiny (0.6B) model."""
        self._load()
        sub = self._state_plausible(ctx, options)
        if len(sub) == 1:
            return _find(options, sub[0].id)
        return _find(options, sub[self._score_sub(ctx, sub)].id)

    def _score_sub(self, ctx: dict, options: list) -> int:
        feel = ", ".join(sorted(ctx["tags"] & EMOTIONS)) or "calm"
        # qualitative state — small models reason far better from words than numbers
        c = ctx["composure"]
        nerve = ("You are trembling, terrified, with no courage left" if c < 25 else
                 "You are rattled and easily cowed" if c < 50 else
                 "You are holding steady" if c < 75 else
                 "You feel bold and very hard to intimidate")
        d = ctx["disposition"]
        rapport = ("you loathe the player" if d < -0.2 else
                   "you have warmed to the player" if d > 0.2 else "you barely know the player")
        sysmsg = (ctx.get("persona", f"You role-play {ctx['npc_name']}.")
                  + "\nPick the reaction that best fits your character and current mood.")
        listing = "\n".join(f"- {o.id}: \"{o.line}\"" for o in options)
        tail = f"\nReply with exactly one word: {', '.join(o.id for o in options)}."
        act = (f'says to you: "{ctx["utterance"]}"' if ctx.get("utterance")
               else f"chose to **{ctx['intent']}** you")
        cond = (f"Right now: {nerve}; you feel {feel}; {rapport}. The player {act}. "
                f"Choose the most in-character reaction:\n" + listing + tail)
        neutral = "The player approaches you. Choose a reaction:\n" + listing + tail
        ids = [o.id for o in options]
        argmax, scores = self._lm.pick_scored(sysmsg, cond, neutral, ids)
        chosen = ids[argmax]
        if DATASTORE is not None and DATASTORE.anchors(ctx["intent"], tuple(ids)):
            from . import knn
            mscores = {o: s for o, s in zip(ids, scores)}
            chosen = knn.blend_pick(DATASTORE, ctx["intent"], ids, scores, mscores, alpha=KNN_ALPHA)
        from . import trace
        trace.capture(ctx["intent"], ids, scores, chosen)  # harvest the FINAL pick if the turn is blessed
        return _find(options, chosen)


class DMChoice(Exception):
    """AgentBrain needs the DM (a big model) to pick — carries the decision."""

    def __init__(self, ctx: dict, options: list):
        self.ctx, self.options = ctx, options


class AgentBrain(SocialBrain):
    """The DM is a big model driving the session. choose() SURFACES the stance
    decision instead of making it: unanswered it raises DMChoice (the harness
    prints the choice-point); resumed with forced indices it returns them in
    order. The DM only SELECTS a typed stance — never free-writes the line — so
    the grounded-vocabulary invariant holds and (ctx→pick) stays harvestable."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.log = []

    def choose(self, ctx: dict, options: list) -> int:
        if not self.answers:
            raise DMChoice(ctx, options)
        i = self.answers.pop(0)
        i = i if 0 <= i < len(options) else 0
        self.log.append({"intent": ctx["intent"], "utterance": ctx.get("utterance"),
                         "npc": ctx["npc_name"], "options": [o.id for o in options],
                         "chose": options[i].id})
        return i


BRAIN: SocialBrain = MockBrain()


def use_brain(name: str, **kw) -> SocialBrain:
    global BRAIN
    BRAIN = (TorchChoiceBrain(**kw) if name == "torch"
             else AgentBrain(**kw) if name == "agent" else MockBrain())
    return BRAIN


# ── VOICE MODEL ─────────────────────────────────────────────────────────────
# Optional. The decision model already chose the reaction (deltas); a voice model
# only GENERATES the spoken line, conditioned on the chosen emotional stance. It
# never touches state, so it cannot break coherence — a canned line is the
# fallback. This is the right home for a small *story/RP fine-tune* (Reyna-0.5B).
# harness #6: constraint lives in COMMITTED TOKENS — each stance opens with
# forced words the tiny voice model must continue from (instructions get
# ignored; a begun sentence gets finished in register).
def stance_prefix(rid: str) -> str:
    return STANCE.get(rid, ("", "Well — "))[1]


_META = ("as an ai", "language model", "i cannot roleplay", "as a dm", "assistant")
_STATSPEAK = re.compile(r"\b(hp|d\d+|damage|stats?)\b", re.I)


def sanitize_line(text: str) -> str | None:
    """Harness #7 (Kobold banned_strings + Clover sanitizer, as a filter): the
    voice lane is mechanically hygienic. Returns the cleaned line, or None =
    REJECT (caller retries once, then uses the canned line)."""
    t = re.sub(r"\*[^*]*\*", "", text).strip()          # RP-asterisk actions
    if not t or any(m in t.lower() for m in _META):
        return None                                     # meta-leakage: reject
    if _STATSPEAK.search(t):
        return None                                     # numbers are the sim's voice, not theirs
    t = re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1", t, flags=re.I)   # stutter loops
    if t.count('"') % 2:
        t = t.replace('"', "")                          # unbalanced quotes
    if t and t[-1] not in ".!?…":                       # trailing fragment → last full sentence
        ends = list(re.finditer(r"[.!?…]", t))
        if ends:
            t = t[:ends[-1].end()]
    return re.sub(r"\s+", " ", t).strip() or None


# stance id → (how it feels, the FORCED OPENER the voice must continue from)
STANCE = {
    "cower": ("terrified and cowering", "P-please, I— "),
    "defy": ("defiant and unafraid", "Hah! You think "),
    "comply": ("frightened but giving in", "Alright, alright — "),
    "relent": ("grudgingly relenting", "Fine. But "),
    "refuse": ("coldly refusing", "No. And "),
    "wary": ("wary and guarded", "Careful, now — "),
    "warm": ("warm and friendly", "Ah, friend, "),
    "dismiss": ("dismissive and busy", "Hmph. "),
}


class VoiceModel:
    def line(self, ctx: dict, chosen: Reaction) -> str:
        raise NotImplementedError


class TorchVoice(VoiceModel):
    def __init__(self, model_id: str = "aloobun/Reyna-RP-Qwen1.5-0.5B-Chat-v0.1"):
        self.model_id = model_id
        self._tok = self._model = self._dev = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._dev == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(self._dev).eval()

    def line(self, ctx: dict, chosen: Reaction) -> str:
        for bump in (0, 7919):                  # one reseeded retry, then canned
            try:
                c = dict(ctx, tick=ctx.get("tick", 0) + bump) if bump else ctx
                clean = sanitize_line(self._gen(c, chosen) or "")
            except Exception:
                return chosen.line
            if clean:
                return clean
        return chosen.line

    def _gen(self, ctx: dict, chosen: Reaction) -> str:
        import torch
        self._load()
        stance = STANCE.get(chosen.id, ("in character", ""))[0]
        what = (f'The player says to you: "{ctx["utterance"]}".' if ctx.get("utterance")
                else f'The player tries to {ctx["intent"]} you.')
        sysmsg = (ctx.get("persona", f"You are {ctx['npc_name']}.")
                  + " Stay fully in character. Reply with only ONE short spoken line.")
        usr = f"{what} You feel {stance}. What do you say?"
        prompt = self._tok.apply_chat_template(
            [{"role": "system", "content": sysmsg}, {"role": "user", "content": usr}],
            tokenize=False, add_generation_prompt=True)
        opener = stance_prefix(chosen.id)
        prompt += opener                     # committed tokens: the register is decided
        ids = self._tok(prompt, return_tensors="pt").to(self._dev)
        torch.manual_seed(int(ctx.get("tick", 0)))       # variety, but replayable per tick
        with torch.no_grad():
            out = self._model.generate(**ids, max_new_tokens=32, do_sample=True,
                                       temperature=0.8, top_p=0.9, pad_token_id=self._tok.eos_token_id)
        txt = self._tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        txt = txt.split("\n")[0].strip().strip('"').strip()
        return (opener + txt)[:140]          # the forced opener IS part of the line


VOICE: VoiceModel | None = None


def use_voice(name: str, **kw) -> VoiceModel | None:
    global VOICE
    VOICE = TorchVoice(**kw) if name == "reyna" else None
    return VOICE


def _sanitize(world: World, npc_id: str, effects: list) -> tuple[list, list]:
    kept, dropped = [], []
    for ef in effects:
        d, k = ef.data, ef.kind
        ok = False
        if k in ("set_tag", "clear_tag") and d.get("t") == npc_id and d.get("tag") in EMOTIONS:
            ok = True
        elif k == "adjust_prop" and d.get("t") == npc_id and d.get("prop") == "composure":
            d["delta"] = max(-MAX_COMPOSURE_STEP, min(MAX_COMPOSURE_STEP, d["delta"]))
            ok = True
        elif k == "set_edge" and d.get("a") == npc_id and d.get("b") == "player" and d.get("kind") in EDGE_KINDS:
            d["w"] = max(-MAX_EDGE, min(MAX_EDGE, d["w"]))
            ok = True
        (kept if ok else dropped).append(ef)
    return kept, dropped


def _pack_line(world, npc, chosen):
    """The A/B playtest's ceiling finding: the DM picks the right STANCE but every
    stance led to one static guildhall line ("...in this den!") — so a farm NPC
    sounded wrong-genre. An active pack may voice a stance in its OWN register via
    a `stances:` map (stance-id → line|variants, optionally nested under an
    archetype tag the NPC carries, so a beast and a person voice `defy` apart)."""
    from .reactions import _voice
    for p in getattr(world, "packs", []):
        st = p.get("stances")
        if not st:
            continue
        pool = next((st[t][chosen.id] for t in npc.tags
                     if isinstance(st.get(t), dict) and chosen.id in st[t]), None)
        pool = pool or st.get(chosen.id)
        if pool:
            return _voice(world, f"stance:{npc.id}:{chosen.id}", pool)
    return None


def address(world: World, actor_id: str, npc_id: str, intent: str | None = None,
            utterance: str | None = None, tone: str | None = None) -> None:
    """Player addresses an npc — via a menu verb (intent) or free-typed words
    (utterance). Free text uses the full repertoire so content decides the reaction."""
    npc = world.entities[npc_id]
    if utterance is not None:
        options = ALL_REACTIONS
    else:
        options = REACTIONS.get(intent)
        if not options:
            return
    # conversation fatigue: pestering an NPC turn after turn wears their patience.
    # Decays over time, so coming back later is fresh. This is why talking 5x in a
    # row no longer echoes the same warm line — repetition now COSTS something.
    conv = npc.mind.setdefault("conv", {"n": 0, "last": -99})
    conv["n"] = conv["n"] + 1 if world.tick - conv["last"] <= 4 else 1
    conv["last"] = world.tick
    ctx = {
        "npc_id": npc_id, "npc_name": npc.name, "intent": intent or "say",
        "utterance": utterance, "tick": world.tick, "tone": tone,
        "composure": npc.props.get("composure", 50.0), "tags": set(npc.tags),
        "disposition": world.edges.get((npc_id, "player"), {}).get("disposition", 0.0),
        "persona": persona_card(world, npc), "pestered": conv["n"],
    }
    idx = BRAIN.choose(ctx, options)
    chosen = options[idx if 0 <= idx < len(options) else 0]
    # the voice model only earns its call when there is player CONTENT to answer;
    # contentless menu stances use the canned line — now trait-voiced, so a brave
    # guard and a timid apprentice say the same STANCE in their own register.
    line = (_pack_line(world, npc, chosen)
            or (VOICE.line(ctx, chosen) if (VOICE is not None and utterance) else _voiced(chosen, npc)))
    kept, dropped = _sanitize(world, npc_id, _build(npc_id, chosen.deltas))
    if utterance:
        fx.apply(world, fx.speech(actor_id, utterance, cause=f'You: "{utterance}"'))
    fx.apply(world, fx.speech(npc_id, line, cause=f'{npc.name}: "{line}"'))
    fx.apply_all(world, kept)
    for ef in dropped:
        world.log.append(_note(world, f"[social] dropped out-of-bounds effect {ef.kind}"))


def _note(world, msg):
    from .state import LogEntry
    return LogEntry(world.tick, "note", {}, msg)
