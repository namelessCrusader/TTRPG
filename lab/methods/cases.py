"""Test cases for Workstream A: which method, and why does Laya get worse?

    python3 -m lab.methods.cases            # writes lab/methods/cases_things.jsonl
                                            # and lab/results/methods/label_check.md

KIND "things": "Kael needs to cut the rope. What can he use?"

THERE IS NO SINGLE CORRECT ANSWER. Each item has properties (it cuts, it
holds water, it gives light); each need says which properties make an item
GOOD and which make it only POSSIBLE. So a need has an acceptable SET, and
the labels come from one table instead of six thousand separate calls. The
table is the large model's labelling; label_check.md is for the human to
audit it.

Lists are built from GOOD items plus BAD distractors. POSSIBLE items are
left out of the lists, so that a pick is either in the acceptable set or
clearly not.

ONE FACTOR AT A TIME. Every cell holds all factors at the middle value except
one:

    size        4 8 16 32 64 128          (middle 16)
    optlen      short, long               (middle short)
    position    first, middle, last       (middle: random)
    similar     unrelated, same, near     (middle: any bad item)
    situation   short, long               (middle short)
    good        1 2 4                     (middle 1)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent

# name | category | properties | description
ITEMS = """
the kitchen knife | kitchen | cut sharp pierce weapon | a short blade with a wooden handle
the butter knife | kitchen | utensil | a dull round-ended table knife
the wooden spoon | kitchen | utensil | a long spoon for stirring
the iron pot | kitchen | cook liquid_hold blunt_heavy | a heavy pot for stews
the frying pan | kitchen | cook blunt_heavy | a black iron pan
the ladle | kitchen | utensil | a deep tin ladle
the rolling pin | kitchen | blunt_heavy | a wooden rolling pin
the cleaver | kitchen | cut chop sharp weapon | a heavy square blade for meat
the tin cup | kitchen | liquid_hold | an empty tin cup
the clay jug | kitchen | liquid_hold pour | a jug with a narrow neck
the funnel | kitchen | pour | a tin funnel
the salt jar | kitchen | dry_hold | a jar half full of salt
the cutting board | kitchen | board | a thick wooden board
the fork | kitchen | pierce utensil | a two-pronged kitchen fork
the kettle | kitchen | liquid_hold cook | a copper kettle
the bowl | kitchen | liquid_hold dry_hold | a wooden bowl
the plate | kitchen | dish | a glazed clay plate
the tea cloth | kitchen | cover clean | a clean cotton cloth
the mortar and pestle | kitchen | grind | a stone bowl and grinder
the cauldron | kitchen | cook liquid_hold | a large black cauldron
the hammer | tool | blunt_heavy weapon pry | a claw hammer
the saw | tool | cut chop | a wood saw with a long blade
the axe | tool | chop cut weapon | a woodcutter's axe
the hatchet | tool | chop cut weapon | a small one-handed axe
the shovel | tool | dig blunt_heavy reach | a long-handled shovel
the pickaxe | tool | dig blunt_heavy weapon | a miner's pickaxe
the crowbar | tool | pry blunt_heavy | an iron bar with a bent end
the chisel | tool | pry sharp | a steel chisel
the pliers | tool | grip | iron pliers
the whetstone | tool | sharpen | a grey stone for sharpening blades
the file | tool | sharpen | a rough metal file
the ladder | tool | climb reach | a wooden ladder
the rope | tool | bind climb | a coil of hemp rope
the twine | tool | bind | a ball of rough twine
the chain | tool | bind blunt_heavy | a length of iron chain
the bucket | tool | liquid_hold | a wooden bucket
the wheelbarrow | tool | carry_load | a one-wheeled barrow
the sack | tool | carry_load dry_hold | a burlap sack
the basket | tool | carry_load dry_hold | a wicker basket
the nails | tool | pierce sharp | a handful of iron nails
the measuring rod | tool | measure reach | a rod marked in feet
the knotted cord | tool | measure bind | a cord knotted at every foot
the scales | tool | weigh | a merchant's balance scales
the wire | tool | lockpick bind | a length of thin wire
the hairpin | tool | lockpick | a bent iron hairpin
the key ring | tool | keyring | a ring of old keys
the pliers pair | tool | grip | a small pair of pliers
the tongs | tool | grip | blacksmith's tongs
the grindstone | tool | sharpen | a round grindstone on a stand
the bellows | tool | fan_fire | leather bellows for a fire
the yoke | tool | carry_load | a shoulder yoke for two buckets
the sieve | tool | sieve | a wooden sieve
the padlock | tool | lock | a heavy iron padlock
the tent pegs | tool | pierce | a bundle of wooden pegs
the harness | tool | harness | a leather horse harness
the lasso | tool | bind | a rope with a running loop
the pitchfork | farm | weapon reach pierce | a three-pronged fork
the scythe | farm | cut weapon reach | a long curved blade on a pole
the sickle | farm | cut weapon | a small curved blade
the hoe | farm | dig | a garden hoe
the rake | farm | reach | a wooden rake
the churn | farm | churn | a butter churn
the snare | farm | trap | a wire loop for catching rabbits
the cage | farm | trap dry_hold | a wicker cage with a door
the hay bale | farm | fodder fuel | a bale of dry hay
the sack of oats | farm | fodder | a sack of horse oats
the horseshoe | farm | iron | an old horseshoe
the saddle | farm | saddle | a worn leather saddle
the mousetrap | farm | trap | a small spring trap
the net | fishing | catch_fish trap bind | a fishing net
the fishing rod | fishing | catch_fish reach | a rod with line and hook
the fish hook | fishing | catch_fish pierce | a single iron hook
the spear | weapon | weapon pierce reach catch_fish | a long hunting spear
the sword | weapon | cut weapon sharp pierce | a steel sword
the dagger | weapon | cut weapon sharp pierce | a double-edged dagger
the club | weapon | weapon blunt_heavy | a knotted wooden club
the bow | weapon | weapon | a hunting bow
the arrows | weapon | pierce | a quiver of arrows
the shield | weapon | shield | a round wooden shield
the helmet | weapon | armor | an iron helmet
the sling | weapon | weapon | a leather sling
the mace | weapon | weapon blunt_heavy | a flanged iron mace
the lantern | light | light | an oil lantern, lit
the candle | light | light ignite | a lit tallow candle
the torch | light | light ignite weapon | a burning torch
the oil lamp | light | light | a small clay oil lamp, lit
the brazier | light | heat light | an iron brazier full of coals
the flint and steel | light | ignite | a striker and a flint
the tinderbox | light | ignite | a box of tinder and a striker
the lamp oil | light | fuel | a flask of lamp oil
the firewood | light | fuel | a bundle of dry firewood
the charcoal | light | fuel write | a lump of charcoal
the empty lantern | light | lamp_empty | a lantern with no oil
the unlit candle stub | light | fuel | a short unlit candle
the mirror | light | signal_sight | a small hand mirror
the magnifying lens | light | lens | a round glass lens
the waterskin | container | liquid_hold drink | a full leather waterskin
the empty flask | container | liquid_hold | an empty tin flask
the barrel | container | liquid_hold dry_hold | a small wooden barrel
the wooden box | container | dry_hold | a small box with a lid
the pouch | container | dry_hold | a leather pouch
the glass bottle | container | liquid_hold | an empty glass bottle
the bottle of wine | container | drink | a corked bottle of wine
the wool cloak | clothing | warm cover | a thick wool cloak
the oilskin coat | clothing | waterproof warm cover | a coat of waxed cloth
the leather gloves | clothing | gloves | a pair of leather gloves
the fur hat | clothing | warm | a hat of rabbit fur
the boots | clothing | boots | a pair of riding boots
the scarf | clothing | warm | a long wool scarf
the apron | clothing | cover | a leather apron
the tarp | clothing | waterproof cover | a sheet of waxed canvas
the loaf of bread | food | food | a fresh loaf of bread
the apple | food | food fodder | a red apple
the carrots | food | food fodder | a bunch of carrots
the cheese | food | food | a wheel of hard cheese
the dried meat | food | food | strips of dried meat
the honey jar | food | food | a jar of honey
the salted fish | food | food | a salted fish
the eggs | food | food | a basket of eggs
the quill and ink | writing | write | a goose quill and a pot of ink
the chalk | writing | write | a stick of white chalk
the parchment | writing | write_surface | a sheet of parchment
the wax tablet | writing | write write_surface | a wooden tablet coated in wax
the book | writing | book | a thick leather book
the map | writing | map | a map of the valley
the seal | writing | seal | a brass seal
the ink pot | writing | ink | a pot of black ink
the sealing wax | writing | wax | a stick of red sealing wax
the horn | music | signal_sound | a hunting horn
the bell | music | signal_sound | a brass hand bell
the drum | music | signal_sound | a small drum
the whistle | music | signal_sound | a tin whistle
the flute | music | music | a wooden flute
the lute | music | music | a lute with six strings
the flag | music | signal_sight | a red cloth flag
the bandages | medical | bandage | a roll of linen bandages
the herbs | medical | clean_wound | a bundle of healing herbs
the bottle of spirits | medical | clean_wound fuel drink | a bottle of strong spirits
the salve | medical | clean_wound | a tin of healing salve
the splint | medical | splint | two flat sticks for a broken bone
the needle and thread | sewing | sew pierce | a needle with strong thread
the scissors | sewing | cut sharp | a pair of iron scissors
the thimble | sewing | thimble | a brass thimble
the leather patch | sewing | cover | a patch of soft leather
the ball of wool | sewing | yarn | a ball of spun wool
the spindle | sewing | spin | a wooden spindle
the boat | travel | float | a small rowing boat
the raft | travel | float | a raft of lashed logs
the log | travel | float fuel | a dry log
the pole | travel | reach | a long wooden pole
the plank | travel | float | a long wooden plank
the spyglass | travel | see_far | a brass spyglass
the compass | travel | navigate | a small compass
the bedroll | travel | sleep warm | a rolled blanket
the blanket | travel | sleep warm cover | a heavy wool blanket
the tent | travel | sleep waterproof cover | a small canvas tent
the pillow | travel | sleep | a feather pillow
the silver coin | money | pay | a silver coin
the copper coins | money | pay | a few copper coins
the gold ring | money | pay trade | a plain gold ring
the gem | money | pay trade | a small green gem
the bolt of cloth | money | trade cover | a bolt of blue cloth
the holy symbol | religious | holy | a wooden holy symbol
the prayer beads | religious | holy | a string of prayer beads
the dice | toy | game | a pair of bone dice
the chessboard | toy | game | a chessboard and pieces
the doll | toy | toy | a rag doll
the kite | toy | toy signal_sight | a paper kite
the ball | toy | toy | a leather ball
the vase | decor | decor | a painted vase
the painting | decor | decor | a small painting of a ship
the pinecone | nature | nature | a dry pinecone
the feather | nature | feather | a white goose feather
the pebble | nature | stone | a smooth pebble
the rock | nature | blunt_heavy | a heavy rock
the stick | nature | reach | a long stick
the flat stone | nature | sharpen | a flat rough stone
the shell | nature | shell | a large sea shell
the bone | nature | bone | an old animal bone
the brick | nature | blunt_heavy | a red brick
the soap | cleaning | clean | a bar of soap
the sponge | cleaning | clean | a sea sponge
the towel | cleaning | clean cover | a linen towel
the broom | cleaning | clean reach | a straw broom
the mop | cleaning | clean | a mop and a pail
the comb | cleaning | comb | a bone comb
"""

# need | GOOD properties (any) | POSSIBLE properties (any)
NEEDS = """
cut the rope | cut | sharp chop
light the dark cellar | light | ignite
carry water from the well | liquid_hold | -
dig up the buried box | dig | pry
tie up the thief | bind | yarn cover
climb to the high window | climb | reach
start a fire in the hearth | ignite | light fan_fire
write a letter home | write | ink
catch fish in the river | catch_fish | pierce bind
fix the loose nail in the door | blunt_heavy | grip
see who is far down the road | see_far | climb
stay dry in the storm | waterproof | cover
pay the ferryman | pay | trade
feed the hungry horse | fodder | food
chop firewood | chop | cut
treat a bleeding cut | bandage | clean_wound cover clean
keep warm on a freezing night | warm | sleep ignite heat
open a locked chest whose key is lost | lockpick pry | blunt_heavy keyring
signal a ship far out at sea | signal_sight | light signal_sound
call the village for help | signal_sound | signal_sight
cross the flooded stream | float | reach bind
carry the harvest to the barn | carry_load | dry_hold
cook the stew | cook | heat liquid_hold
drink on a long walk | drink | liquid_hold
fight off the wolf | weapon | ignite reach blunt_heavy
reach the apple high in the tree | reach | climb
mend the torn cloak | sew | bind cover
measure the length of the field | measure | bind
weigh the silver | weigh | -
clean the muddy floor | clean | liquid_hold cover
hide the stolen coins | dry_hold | cover
block the arrows | shield | armor cover board
break the ice on the pond | blunt_heavy | chop dig pry
trap a rabbit | trap | bind dry_hold
sharpen the blade | sharpen | blunt_heavy
sleep outside | sleep | warm cover
eat something before the journey | food | fodder drink
pour the oil into the lamp | pour | liquid_hold
"""

# HAND-PICKED NEAR MISSES: things that look, sound or seem as if they would
# do the job, and cannot. The first "near" pool was a rule ("shares any
# property with an acceptable item"), and it turned every weapon into a near
# miss for lighting a cellar, because the torch counts as a weapon.
NEARMISS = {
    "cut the rope": "the butter knife, the fork, the wooden spoon, the needle and thread, the tent pegs, the thimble, the comb, the file, the whetstone",
    "light the dark cellar": "the empty lantern, the unlit candle stub, the lamp oil, the mirror, the magnifying lens, the firewood, the charcoal, the bellows",
    "carry water from the well": "the sieve, the basket, the sack, the funnel, the wheelbarrow, the plate, the pouch, the wooden box, the ladle",
    "dig up the buried box": "the rake, the pitchfork, the wooden spoon, the ladle, the stick, the broom, the sickle, the scythe",
    "tie up the thief": "the harness, the scarf, the needle and thread, the leather gloves, the saddle, the padlock, the belt, the ball of wool",
    "climb to the high window": "the twine, the wire, the chain, the net, the lasso, the knotted cord, the bench",
    "start a fire in the hearth": "the unlit candle stub, the firewood, the lamp oil, the charcoal, the empty lantern, the magnifying lens, the hay bale, the bellows",
    "write a letter home": "the parchment, the feather, the book, the seal, the sealing wax, the map, the ink pot",
    "catch fish in the river": "the bucket, the sieve, the basket, the cage, the snare, the mousetrap, the pole, the twine",
    "fix the loose nail in the door": "the chisel, the file, the nails, the wooden spoon, the butter knife, the pliers, the tongs, the whetstone",
    "see who is far down the road": "the magnifying lens, the mirror, the map, the compass, the lantern, the ladder, the oil lamp",
    "stay dry in the storm": "the fur hat, the scarf, the leather gloves, the boots, the bedroll, the pillow, the wool cloak, the apron",
    "pay the ferryman": "the key ring, the horseshoe, the seal, the holy symbol, the prayer beads, the dice, the salt jar",
    "feed the hungry horse": "the saddle, the harness, the horseshoe, the bucket, the pinecone, the loaf of bread, the cheese",
    "chop firewood": "the butter knife, the hammer, the shovel, the chisel, the file, the rolling pin, the kitchen knife, the scissors",
    "treat a bleeding cut": "the splint, the thimble, the leather gloves, the scarf, the salt jar, the bottle of wine, the soap",
    "keep warm on a freezing night": "the tarp, the leather gloves, the boots, the apron, the lamp oil, the firewood, the unlit candle stub",
    "open a locked chest whose key is lost": "the padlock, the butter knife, the nails, the fork, the tent pegs, the needle and thread, the key ring, the seal",
    "signal a ship far out at sea": "the magnifying lens, the spyglass, the map, the compass, the empty lantern, the unlit candle stub, the flute",
    "call the village for help": "the flute, the lute, the ball, the doll, the pillow, the mirror, the flag",
    "cross the flooded stream": "the bucket, the wooden box, the tent, the tarp, the boots, the ladder, the rope",
    "carry the harvest to the barn": "the bucket, the tin cup, the plate, the tarp, the bolt of cloth, the blanket, the pouch",
    "cook the stew": "the plate, the ladle, the wooden spoon, the salt jar, the cutting board, the tin cup, the bowl",
    "drink on a long walk": "the honey jar, the ink pot, the lamp oil, the salted fish, the empty flask, the glass bottle, the soap",
    "fight off the wolf": "the fork, the wooden spoon, the butter knife, the ladle, the pillow, the flute, the feather",
    "reach the apple high in the tree": "the ball, the kite, the lasso, the twine, the chain, the rope, the wire",
    "mend the torn cloak": "the thimble, the spindle, the ball of wool, the scissors, the comb, the fish hook, the leather patch",
    "measure the length of the field": "the scales, the compass, the map, the ladder, the stick, the pole, the rope",
    "weigh the silver": "the measuring rod, the knotted cord, the bowl, the salt jar, the bucket, the compass, the copper coins",
    "clean the muddy floor": "the rake, the comb, the sieve, the feather, the pinecone, the pillow, the bucket",
    "hide the stolen coins": "the sieve, the funnel, the plate, the net, the flag, the tarp, the book",
    "block the arrows": "the plate, the vase, the painting, the pillow, the book, the cutting board, the helmet",
    "break the ice on the pond": "the wooden spoon, the broom, the stick, the rake, the ladle, the feather, the fork",
    "trap a rabbit": "the fishing rod, the fish hook, the bucket, the sieve, the carrots, the lasso, the rope",
    "sharpen the blade": "the pebble, the shell, the bone, the comb, the chisel, the scissors, the rock",
    "sleep outside": "the saddle, the boots, the leather gloves, the mop, the tarp, the scarf, the fur hat",
    "eat something before the journey": "the salt jar, the soap, the pinecone, the sealing wax, the lamp oil, the sack of oats, the hay bale",
    "pour the oil into the lamp": "the sieve, the wooden spoon, the plate, the basket, the sponge, the fork, the net",
}

FILLER = [
    "Rain drums on the roof.", "Somewhere a dog barks.", "The air smells of smoke and wet straw.",
    "An old woman hums by the fire.", "The floorboards creak under every step.",
    "Two farmers argue about the price of wool.", "A cat watches from the top of a shelf.",
    "The wind rattles the shutters.", "Somebody laughs in the next room.",
    "The storeroom is cluttered and dusty.", "A lamp gutters in the draught.",
    "It is late, and most of the village is asleep.",
]


def parse():
    items = []
    for line in ITEMS.strip().splitlines():
        name, cat, props, desc = [p.strip() for p in line.split("|")]
        items.append({"name": name, "cat": cat, "props": set(props.split()), "desc": desc})
    needs = []
    for line in NEEDS.strip().splitlines():
        need, good, poss = [p.strip() for p in line.split("|")]
        needs.append({"need": need, "good": set(good.split()),
                      "possible": set() if poss == "-" else set(poss.split())})
    return items, needs


def label(item, need):
    if item["props"] & need["good"]:
        return "good"
    if item["props"] & need["possible"]:
        return "possible"
    return "bad"


def pools(items, need):
    good = [i for i in items if label(i, need) == "good"]
    bad = [i for i in items if label(i, need) == "bad"]
    gcats = {i["cat"] for i in good}
    gprops = set().union(*(i["props"] for i in good)) - need["good"]
    same = [i for i in bad if i["cat"] in gcats]
    near = [i for i in bad if i["props"] & gprops or
            any(w in i["name"] for g in good for w in g["name"].split()[1:] if len(w) > 3)]
    unrelated = [i for i in bad if i["cat"] not in gcats and not (i["props"] & gprops)]
    return good, bad, {"same": same, "near": near, "unrelated": unrelated}


MIDDLE = {"size": 16, "optlen": "short", "position": "random", "similar": "any",
          "situation": "short", "good": 1}
FACTORS = {"size": [4, 8, 16, 32, 64, 128], "optlen": ["short", "long"],
           "position": ["first", "middle", "last"], "similar": ["unrelated", "same", "near"],
           "situation": ["short", "long"], "good": [1, 2, 4]}


def situation_text(need, kind, rng):
    base = f"Kael needs to {need}."
    if kind == "short":
        return base + " He looks at what is in the storeroom."
    f = rng.sample(FILLER, 6)
    return " ".join(f[:3] + [base] + f[3:] + ["He looks at what is in the storeroom."])


def make_case(items, need, cfg, rng):
    good, bad, pl = pools(items, need)
    k = cfg["good"]
    if len(good) < k:
        return None
    src = bad if cfg["similar"] == "any" else pl[cfg["similar"]]
    n_bad = cfg["size"] - k
    if len(src) < n_bad:
        return None
    chosen_good = rng.sample(good, k)
    chosen_bad = rng.sample(src, n_bad)
    opts = chosen_bad[:]
    rng.shuffle(opts)
    if cfg["position"] == "first":
        opts = chosen_good + opts
    elif cfg["position"] == "last":
        opts = opts + chosen_good
    elif cfg["position"] == "middle":
        m = len(opts) // 2
        opts = opts[:m] + chosen_good + opts[m:]
    else:
        opts = opts + chosen_good
        rng.shuffle(opts)
    text = (lambda i: i["name"]) if cfg["optlen"] == "short" else (lambda i: f"{i['name']}: {i['desc']}")
    return {"need": need["need"], "situation": situation_text(need["need"], cfg["situation"], rng),
            "question": "Which thing can Kael use for this?",
            "options": [text(i) for i in opts],
            "names": [i["name"] for i in opts],
            "good": [i for i, o in enumerate(opts) if label(o, need) == "good"]}


def build_nearmiss(items, needs, n_per_cell=60, seed=0, size=8):
    """A clean comparison at 8 options: the SAME needs with random bad items,
    and with hand-picked near misses. Only near misses that the labels call
    bad are used; a need with fewer than 7 is left out of both cells."""
    by = {i["name"]: i for i in items}
    usable, dropped = [], []
    for nd in needs:
        names = [n.strip() for n in NEARMISS.get(nd["need"], "").split(",") if n.strip()]
        near = [by[n] for n in names if n in by and label(by[n], nd) == "bad"]
        dropped += [(nd["need"], n) for n in names if n not in by or label(by[n], nd) != "bad"]
        good, bad, _ = pools(items, nd)
        if len(near) >= size - 1 and good:
            usable.append((nd, good, bad, near))
    cases = []
    for value in ("random", "hand"):
        rng = random.Random(f"{seed}-nearmiss-{value}")
        for k in range(n_per_cell):
            nd, good, bad, near = usable[k % len(usable)]
            g = rng.choice(good)
            src = near if value == "hand" else bad
            opts = rng.sample(src, size - 1) + [g]
            rng.shuffle(opts)
            cases.append({"need": nd["need"], "situation": situation_text(nd["need"], "short", rng),
                          "question": "Which thing can Kael use for this?",
                          "options": [o["name"] for o in opts], "names": [o["name"] for o in opts],
                          "good": [i for i, o in enumerate(opts) if label(o, nd) == "good"],
                          "id": f"things/nearmiss={value}/{k}", "kind": "things",
                          "factor": "nearmiss", "value": value})
    return cases, usable, dropped


def build(n_per_cell=60, seed=0):
    items, needs = parse()
    cases = []
    for factor, values in FACTORS.items():
        for v in values:
            cfg = dict(MIDDLE, **{factor: v})
            rng = random.Random(f"{seed}-{factor}-{v}")
            made, tries = 0, 0
            order = needs[:]
            while made < n_per_cell and tries < n_per_cell * 20:
                need = order[tries % len(order)]
                if tries % len(order) == 0:
                    rng.shuffle(order)
                tries += 1
                c = make_case(items, need, cfg, rng)
                if c is None:
                    continue
                c.update(id=f"things/{factor}={v}/{made}", kind="things", factor=factor, value=v)
                cases.append(c)
                made += 1
            if made < n_per_cell:
                print(f"  only {made} cases for {factor}={v}")
    return cases, items, needs


def label_check(items, needs, path, n=50, seed=1):
    """For the human: 50 random (need, item) labels to audit, plus each need's
    full good and possible sets."""
    rng = random.Random(seed)
    lines = ["# Label check (Workstream A, things)", "",
             "Mark any label you disagree with. If more than 20% are wrong, the labels are fixed before use.", ""]
    lines += ["## Each need: good and possible items", ""]
    for nd in needs:
        g = [i["name"] for i in items if label(i, nd) == "good"]
        p = [i["name"] for i in items if label(i, nd) == "possible"]
        lines.append(f"- **{nd['need']}**: good = {', '.join(g)}; possible = {', '.join(p) or '(none)'}")
    lines += ["", "## 50 random single labels", ""]
    for _ in range(n):
        nd = rng.choice(needs)
        it = rng.choice(items)
        lines.append(f"- {nd['need']} / {it['name']} → **{label(it, nd)}**")
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    cases, items, needs = build()
    nm, usable, dropped = build_nearmiss(items, needs)
    cases += nm
    print(f"near-miss cells use {len(usable)} needs; left out of the lists "
          f"(not labelled bad, or unknown): {dropped}")
    out = HERE / "cases_things.jsonl"
    with out.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    res = HERE.parent / "results" / "methods"
    res.mkdir(parents=True, exist_ok=True)
    label_check(items, needs, res / "label_check.md")
    print(f"{len(items)} items, {len(needs)} needs, {len(cases)} cases -> {out}")
    for nd in needs:
        good, bad, pl = pools(items, nd)
        if len(good) < 4 or len(pl["same"]) < 15 or len(pl["near"]) < 15:
            print(f"  {nd['need']}: good {len(good)}, bad {len(bad)}, same {len(pl['same'])}, "
                  f"near {len(pl['near'])}, unrelated {len(pl['unrelated'])}")
