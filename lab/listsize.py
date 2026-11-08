"""How does Laya's pick hold up as the list grows? One need, one right item,
N-1 distractors, the right one at a random place in the list."""
import random
import laya
from laya.common import build_sequence, render_options

NEEDS = [("cut the rope", "the knife"), ("light the dark cellar", "the lantern"),
         ("carry water from the well", "the bucket"), ("open the locked door", "the key"),
         ("dig up the buried box", "the shovel"), ("mend the torn sail", "the needle and thread"),
         ("start a fire in the hearth", "the flint and steel"), ("climb to the high window", "the ladder"),
         ("write a letter home", "the quill"), ("catch fish in the river", "the fishing rod"),
         ("fix the loose nail", "the hammer"), ("see who is far down the road", "the spyglass"),
         ("stay dry in the storm", "the oilskin cloak"), ("tie up the thief", "the rope"),
         ("cross the icy river", "the boat"), ("heal a deep cut", "the bandages"),
         ("pay the ferryman", "the silver coin"), ("feed the hungry horse", "the sack of oats"),
         ("chop firewood", "the axe"), ("read the ancient runes", "the dictionary of runes")]
JUNK = ["the pebble", "the feather", "the spoon", "the pillow", "the broom", "the bell", "the teacup",
        "the drum", "the mirror", "the comb", "the carpet", "the vase", "the painting", "the candle stub",
        "the dice", "the bowl", "the scarf", "the boot", "the button", "the pinecone", "the shell",
        "the jar of honey", "the wooden toy", "the saddle", "the harp", "the chessboard", "the blanket",
        "the flag", "the fan", "the clock", "the kettle", "the hat", "the glove", "the pipe", "the basket",
        "the wreath", "the lute", "the barrel", "the stool", "the brick", "the chain", "the cage",
        "the anvil", "the loom", "the plate", "the mask", "the whistle", "the horn", "the doll",
        "the scroll case", "the inkpot", "the tongs", "the sieve", "the churn", "the yoke", "the bellows",
        "the trowel", "the rake", "the sickle", "the spindle", "the ladle", "the pestle", "the mortar",
        "the tankard", "the goblet", "the crown", "the sceptre", "the banner", "the shield", "the helm",
        "the gauntlet", "the belt", "the pouch", "the satchel", "the lamp oil", "the soap", "the towel",
        "the sponge", "the rug", "the curtain", "the peg", "the hook", "the nail"]
a = laya.load("convaiinnovations/laya")
tok, cfg = a.tok, a.cfg
rng = random.Random(0)
print(f"{'N':>4} {'right':>6} {'p(right)':>9} {'tokens/opt':>11} {'opts cut':>9}")
for N in (4, 8, 16, 32, 64, 96):
    hits, pr, cut = 0, 0.0, 0
    for need, item in NEEDS:
        opts = rng.sample([j for j in JUNK if j != item], N - 1) + [item]
        rng.shuffle(opts)
        q = {"pick": {"type": "choice", "instructions": "What does Kael take?",
                      "criteria": {o: None for o in opts}}}
        st = f"Kael needs to {need}. He looks at what is in the storeroom."
        try:
            p = a.predict(st, q)["answers"]["pick"]["probabilities"]
        except ValueError as e:
            cut += 1
            continue
        best = max(p, key=p.get)
        hits += best == item
        pr += p[item]
        seq, marks = build_sequence(tok, st, {"t": "choice", "ins": "What does Kael take?",
                                              "crit": {o: None for o in opts}},
                                    cfg["max_len"], cfg["head_max_len"])
    n = len(NEEDS) - cut
    per = (cfg["head_max_len"] - 12) / N
    print(f"{N:>4} {hits}/{n:<4} {pr / max(n, 1):>9.2f} {per:>11.1f} {cut:>9}")
