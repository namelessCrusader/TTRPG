"""Two more kinds of test cases for Workstream A, built the way "things" is.

    lab/.venv/bin/python -m lab.methods.cases_kinds
        # writes lab/methods/cases_places.jsonl and cases_people.jsonl
        # and lab/results/methods/label_check_places.md and label_check_people.md

The factors, their middle values and the fillers are the ones in cases.py,
so a cell here is the same cell as in things, only the kind changes.

KIND "places": "Mara needs to find a drink of water. Where can she go?"

Each place has properties; each need says which properties make a place GOOD
and which make it only POSSIBLE. Possible places are left out of the lists.
Drinkable water (the well) is not open water (the river): the river is only
possible for a drink. The ~ tags (~water, ~holy) are used by no label; they
only say what SORT of place it is, and each need names the sorts that SEEM
to fit it. (A rule, "shares a sort with any good place", made every church a
near miss for a drink, because the holy well is holy.)
    same       a bad place in the same part of the village as a good one
    near       a bad place of a sort the need names (the horse trough for a
               drink of water)
    unrelated  the rest

KIND "people": "A bandit grabs a child in the market square. What can Tom do?"

The options are actions AIMED AT the people there, so the model has to bind
the action to the right person: "hit the bandit" is good, "hit the child" is
not. Every action has a PURPOSE. Each situation says which (purpose, role)
pairs are GOOD and which only POSSIBLE; "-" is the role of an action aimed at
nobody ("shout for help"). Everything else is bad. The options of a situation
are every action with every person, plus the actions aimed at nobody.
    same       a good purpose aimed at the wrong person ("hit the child"):
               the entity-binding test
    near       a person of a good option, with a bad purpose ("ask the
               bandit to dance")
    unrelated  the rest: bad actions aimed at nobody, and bad purposes aimed
               at people no good option is aimed at

A purpose holds only actions that are right or wrong together. That is why
"throw a stone at" (pelt) is not "hit" (attack): against a wolf the first is
sensible and the second is not.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from lab.methods.cases import FACTORS, FILLER, MIDDLE, label

HERE = Path(__file__).parent
RES = HERE.parent / "results" / "methods"

# ---------------------------------------------------------------- places

# name | part of the village | properties and ~sorts | description
PLACES = """
the inn common room | inn | meal ale drinks gossip crowd fire shelter rough_sleep buys_food in ~food ~drink ~fire | a long room with benches, a hearth and a busy bar
the inn guest room | inn | bed private lockable in ~sleep | a small room upstairs with a straw mattress and a shuttered window
the inn kitchen | inn | kitchen warm buys_food in ~food ~fire | a hot room with a spit over the fire and pots on hooks
the inn stable | inn | fodder horses warm rough_sleep shelter nook stall in ~animals | stalls for travellers' mounts, with hay racks along the wall
the inn cellar | inn | hide cool_store cold lockable casks in ~dark ~cold ~drink | a low vault of ale barrels under the inn
the inn yard | inn | ~travel | a cobbled yard where the carts unload
the inn attic | inn | hide private high dry_store lockable in ~dark | a low dusty space under the roof beams, full of old trunks
the pump in the inn yard | inn | drinkwater ~water | an iron pump with a stone basin beneath it
the inn washroom | inn | wash tubs in ~water ~wash | a stone room with tubs, soap and a mangle
the woodshed behind the inn | inn | firewood nook dry_store in ~wood | a lean-to stacked with split logs and kindling
the inn's kitchen garden | inn | ~plants ~food | rows of cabbages, leeks and parsley behind the inn
the inn porch | inn | shelter ~travel | a covered porch with a bench beside the inn door
the inn's chicken coop | inn | hens ~animals | a coop of nesting boxes behind the kitchen
the snug | inn | ale drinks meal private in ~drink | a small wood-panelled room off the bar, with one table
the innkeeper's back room | inn | lockable private in ~money ~trade | a cramped room with a desk, a ledger and a strongbox
the inn's dovecote | inn | ~animals ~high | a round tower where pigeons roost
the coach stop | inn | chatter ~travel | a post outside the inn where the mail coach stops twice a week
the well | square | drinkwater chatter cold ~water | a stone well in the square with a bucket on a rope
the fountain | square | drinkwater ~water ~stone | a carved stone fountain in the square, always running
the market square | square | market crowd gossip food_sell ~trade ~food | an open square crowded with stalls on market day
the market hall | square | market crowd shelter in ~trade | a timber hall on stone pillars where stallholders set out their goods
the bread stall | square | bread food_sell ~food ~trade | a trestle of round loaves under a striped awning
the bakery | square | bread food_sell warm in ~food ~fire | a low shop with a brick oven and a smell of baking
the butcher's shop | square | food_sell in ~food ~dead | a shop with hooks of hanging meat and a sawdust floor
the alehouse | square | ale drinks food_sell chatter shelter warm in ~drink | a cottage with a green bush hung over the door
the lodging house | square | bed in ~sleep | a tall timber house by the square, with a painted moon on its sign
the stocks | square | lockable ~law | a wooden frame with holes for the head and hands of wrongdoers
the notice board | square | announce chatter ~law | a board by the reeve's house where notices are nailed up
the village green | square | grazing ~grass ~animals | a wide common of grass where geese graze and children play
the barber's shop | square | barber gossip surgeon in ~health | a shop with a striped pole, a chair and a basin
the general store | square | trade food_sell goods in ~trade | a shop of candles, rope, salt and sacks of flour
the tailor's shop | square | clothes in ~cloth ~trade | a shop with bolts of cloth and a wooden dummy in the window
the apothecary | square | healer herbs in ~health | a narrow shop of jars, dried roots and bitter smells
the reeve's house | square | law literate announce guards in ~law | a tall house on the square where the reeve keeps the village rolls
the horse trough | square | ~water ~animals | a long stone trough where the carters stop
the alarm bell | square | alarm ~metal ~law | an iron bell hung on a post in the middle of the square
the cobbler's shop | square | in ~craft ~cloth | a shop of lasts, awls and hanging boots
the moneylender's house | square | in ~money ~trade | a narrow house with barred windows and a heavy door
the bathhouse | square | bathe tubs warm shaves in ~water ~wash | a steamy house of wooden tubs and hot stones
the washhouse | square | wash tubs in ~water ~wash | a stone house with coppers of boiling water and scrubbing boards
the healer's hut | square | healer herbs in ~health | a small hut by the green with a cot, clean linen and jars
the candlemaker's shop | square | in ~fire ~craft | a shop that smells of tallow, with candles hung in bunches
the midwife's cottage | square | herbs in ~health | a cottage with a birthing chair and a shelf of remedies
the church | church | holy shelter sanctuary literate announce quiet in ~holy ~stone | a stone church with a square tower and rows of wooden pews
the chapel | church | holy shelter sanctuary quiet in ~holy | a small chapel with a single altar and a candle burning
the bell tower | church | alarm lookout high nook in ~holy ~high | the church tower, with a great bronze bell at the top
the churchyard | church | burial quiet ~dead ~holy | the ground around the church, with rows of headstones
the old graveyard | church | burial quiet ~dead | a walled field of leaning stones at the edge of the village
the crypt | church | vault quiet hide cold in ~dead ~dark ~cold ~stone | a cold vault of stone coffins beneath the church
the priest's house | church | priest literate in ~holy ~books | a small house beside the church with a herb garden
the lychgate | church | shelter ~holy ~dead | a roofed wooden gate at the entrance to the churchyard
the wayside shrine | church | holy cover ~holy ~travel | a little roofed shrine with a carved saint, beside the road
the holy well | church | drinkwater holy ~water ~holy | a spring under a carved stone arch, hung with ribbons
the tithe barn | church | grain_store shelter nook hay rough_sleep stall in ~grain ~holy | a huge stone barn beside the church
the parish school | church | literate in ~books | a one-room school with benches and slates
the sexton's cottage | church | gravedigger in ~dead | a cottage by the churchyard where the gravedigger lives
the almshouse | church | rough_sleep alms in ~holy ~sleep | a row of small rooms where the poor are given a bed and soup
the church garden | church | quiet ~plants ~holy | a walled garden of roses with a stone bench, behind the church
the monastery chapel | monastery | holy shelter sanctuary quiet in ~holy | a dim chapel where the monks sing at dawn
the monastery library | monastery | books literate quiet in ~books | a long room of chained volumes and lecterns
the scriptorium | monastery | scribe literate in ~books | a bright room where monks copy pages at sloping desks
the infirmary | monastery | healer rough_sleep in ~health | a long room of cots, tended by an old monk
the pilgrims' hostel | monastery | bed meal in ~sleep ~holy | a plain house of bunks by the monastery gate
the monastery kitchen | monastery | kitchen warm in ~food ~fire | a smoky kitchen with a great hearth and copper pots
the monastery brewhouse | monastery | drinks nook in ~drink ~grain | a building of vats with a sweet smell of malt
the cloister | monastery | shelter quiet in ~holy ~stone | a covered walk around a square lawn
the monastery herb garden | monastery | herbs quiet ~plants ~health | beds of sage, feverfew and comfrey behind the infirmary
the monastery granary | monastery | grain_store dry_store grain nook in ~grain | a raised stone building on mushroom-shaped legs
the monks' cells | monastery | rough_sleep private in ~sleep ~holy | a row of tiny bare rooms with a cot in each
the chapter house | monastery | in ~holy | a round hall with stone benches where the monks meet
the monastery gatehouse | monastery | alms in ~holy ~travel | an arched gate where a monk hands out loaves to the poor
the monastery fishpond | monastery | water ~water | a square pond stocked with the monks' carp
the monastery orchard | monastery | quiet branches hay cover ~plants ~food | rows of old pear trees inside the monastery wall
the abbot's lodging | monastery | literate in ~holy ~books | a fine stone house with glass windows
the barn | farm | grain_store shelter nook fodder rough_sleep pen in ~grain ~animals | a big timber barn with a threshing floor
the hayloft | farm | hide rough_sleep fodder warm high in ~grain | a loft under the barn roof, piled high with hay
the cow shed | farm | cows warm stall hay nook in ~animals | a low shed of stalls, with a three-legged stool by the door
the henhouse | farm | hens ~animals | a wooden coop with nesting boxes and a little ramp
the pigsty | farm | ~animals | a muddy pen where three pigs root about
the sheepfold | farm | pen ~animals ~stone | a ring of high stone walls with a wattle gate
the hurdle pen | farm | pen ~animals ~wood | a square of woven hurdles beside the barn, with a gate
the farmyard | farm | farmyard ~animals | a muddy yard between the farmhouse and the sheds
the cow pasture | farm | grazing cows ~animals ~grass | a fenced field of grass where the cows graze
the wheat field | farm | nook ~grain ~plants | a field of ripe wheat as high as a man's chest
the haystack | farm | hide fodder warm rough_sleep ~grain | a tall stack of hay at the edge of the field
the farmhouse kitchen | farm | fire kitchen quern in ~food ~fire | a kitchen with a long table and a fire in the grate
the dairy | farm | cool_store cold in ~food ~cold | a chilly stone room with pans of cream on the shelves
the root cellar | farm | cool_store hide cold in ~dark ~cold ~food | a cellar dug into the hillside, with turnips packed in straw
the duck pond | farm | water ~water ~animals | a muddy pond where the ducks paddle
the orchard | farm | branches hay cover quiet ~plants ~food | rows of apple trees with windfalls in the grass
the beehives | farm | ~food ~animals | a row of straw skeps humming with bees
the farm stable | farm | fodder horses warm shelter nook stall rough_sleep in ~animals | a stable with two stalls and a hay rack
the granary | farm | grain_store dry_store grain nook in ~grain | a wooden shed raised on stone legs to keep out rats
the woodpile | farm | firewood nook ~wood | a stack of split logs against the farmhouse wall
the farm well | farm | drinkwater cold ~water | a well with a wooden cover in the corner of the yard
the cart shed | farm | shelter nook in ~wood ~travel | an open-fronted shed where the hay cart is kept
the vegetable garden | farm | ~plants ~food | rows of beans, onions and cabbages
the ploughed field | farm | ~plants | a wide brown field, freshly ploughed
the river bank | river | water fishing wash cold ~water | a grassy bank above a wide, slow river
the bridge | river | crossing angle ~water ~stone ~travel | a stone bridge of three arches
under the bridge | river | hide shelter water angle ~water ~dark | the damp shadow beneath the stone arches
the ford | river | crossing water ~water ~travel | a shallow stretch of water with a gravel bottom
the ferry landing | river | crossing boat angle ~water ~travel | a wooden jetty where a flat boat is hauled from bank to bank on a rope
the stepping stones | river | crossing ~water ~stone | a line of flat stones from bank to bank where the water narrows
the quay | river | boat trade crowd water ~water ~trade | a timber quay where the river barges unload
the fish market | river | fish_trade food_sell crowd ~water ~trade ~food | a row of wet stalls by the quay, piled with eels and trout
the boathouse | river | boat nook in ~water ~wood | a shed over the water where the rowing boats are kept
the watermill | river | mill nook in ~grain ~water | a mill whose great wheel turns in the river
the mill pond | river | water fishing ~water | the still pond above the mill weir
the reed beds | river | hide ~water ~plants | a thick bed of tall reeds along the river
the washing stones | river | wash water ~water ~wash | flat stones at the river's edge where the women kneel with their baskets
the fishing jetty | river | fishing water ~water ~wood | a narrow jetty with a bench at the far end
the river tavern | river | ale meal drinks gossip crowd fire shelter rough_sleep buys_food in ~drink ~water | a low tavern on the bank, busy with bargemen
the weir | river | angle risky_crossing ~water ~stone | a low dam of stones where the river spills over
the tannery | river | ~water ~craft | a stinking yard of vats by the river where hides are soaked
the dye works | river | buys_wool ~water ~cloth ~craft | vats of blue and red dye beside the river
the net shed | river | ~water ~craft | a shed hung with fishing nets waiting to be mended
the swimming hole | river | water risky_crossing ~water | a deep, clear pool in a bend of the river where boys swim
the edge of the woods | woods | secret_exit firewood cover nook ~wild ~wood | where the fields end and the first oaks begin
the deep woods | woods | hide firewood cover woods ~wild ~wood ~dark | dense old forest where the paths fade out
the woodcutter's hut | woods | firewood timber in ~wood | a log hut with an axe stuck in a chopping block
the charcoal burners' camp | woods | warm timber ~fire ~wood | a clearing of smoking mounds covered in turf
the hollow oak | woods | hide shelter ~wood | an ancient oak with a hollow trunk big enough for a person
the hunting lodge | woods | ~wood ~sleep | the lord's timber lodge, shut up outside the hunting season
the forest spring | woods | drinkwater cold ~water ~wild | a clear spring bubbling up between mossy rocks
the woodland stream | woods | water fishing cold ~water ~wild | a quick brown stream running over stones
the glade | woods | quiet grazing ~wild ~grass | an open glade of grass and clover
the bramble thicket | woods | nook ~wild ~plants | tangled bushes heavy with blackberries
the fallen beech | woods | firewood nook ~wood | a great beech blown down across the path
the wolf den | woods | ~wild ~dark ~animals | a hollow under tree roots, strewn with bones
the herbwife's cottage | woods | healer herbs in ~health ~wild | a crooked cottage hung with drying plants
the hunters' hide | woods | hide ~wild ~wood | a screen of woven branches where hunters wait for deer
the forest track | woods | secret_exit ~wild ~travel | a narrow track under the trees that leads out of the valley
the mushroom ring | woods | ~wild ~food | a circle of pale mushrooms in the leaf litter
the sawpit | woods | timber ~wood ~craft | a pit where two men saw logs into planks
the bog | woods | ~water ~wild | a soft green bog that sucks at your boots
the forest pool | woods | water cold ~water ~wild ~dark | a dark, still pool under the trees
the hilltop | hills | lookout quiet high ~high ~wild | a bare hilltop with the whole valley spread out below
the watchtower | hills | lookout alarm high guards in ~high ~law | a stone tower on the ridge, with a horn hung by the door
the cave | hills | hide shelter cold rough_sleep ~dark ~stone ~wild | a dry cave in the side of the hill
the old ruins | hills | hide cover ~stone | the tumbled walls of an old fort, overgrown with ivy
the standing stones | hills | quiet ~stone ~holy | a ring of tall grey stones on the moor
the shepherd's hut | hills | shelter rough_sleep warm in ~animals ~sleep | a stone hut with a turf roof, used at lambing time
the hill pasture | hills | grazing ~animals ~grass | a high field of short grass cropped by sheep
the quarry | hills | ~stone | a pit of cut grey stone with ladders down the side
the old mine | hills | hide cold cover ~dark ~stone | a timber-framed hole into the hill, abandoned for years
the windmill | hills | mill high nook in ~grain ~high | a windmill on the rise with four canvas sails
the gallows hill | hills | ~law ~dead | a bare rise with an old gallows on it
the beacon | hills | alarm lookout high ~fire ~high ~wood | a stack of tarred wood on the hill, ready to be lit
the moor | hills | ~wild | open heather under a wide sky
the hot spring | hills | bathe water warm ~water ~wild | a steaming pool among the rocks
the mountain lake | hills | water fishing cold ~water ~wild | a cold, deep lake high in the hills
the cliff | hills | high ~high ~stone | a sheer cliff where ravens nest
the shepherds' fire | hills | fire ~fire ~animals | a ring of stones where the shepherds keep a fire going at night
the guardhouse | keep | law guards jail in ~law ~weapons | a squat stone building where the watch keep their pikes, with a barred cell
the gaol | keep | jail in ~law ~dark | a stone cell with a heavy barred door
the town gate | keep | lookout guards ~law ~travel | the timber gate in the stockade where the carts come in
the postern gate | keep | secret_exit ~law ~travel | a small door in the back wall, seldom used
the palisade walk | keep | lookout high ~law ~wood ~high | a walkway along the top of the village stockade
the lord's hall | keep | law guards in ~law | the great hall of the manor, hung with shields
the armoury | keep | lockable in ~metal ~weapons ~law | a locked room of racked spears and shields
the smugglers' tunnel | keep | secret_exit hide cold ~dark | a low tunnel from an old cellar out under the wall
the drain | keep | secret_exit nook ~water ~dark | a stone culvert that runs under the wall into the fields
the courthouse | keep | law in ~law | a hall where the magistrate hears disputes
the constable's house | keep | law guards in ~law | a narrow house with a lantern over the door
the keep tower | keep | high guards in ~high ~law | the lord's stone tower, with guards on the roof
the moat | keep | ~water | a ditch of green, still water around the keep
the training yard | keep | ~weapons | a sandy yard with straw targets where men practise with swords
the dungeon | keep | jail in ~dark ~law | cells cut into the rock beneath the keep
the steward's office | keep | in ~law ~money | a room where the lord's steward counts the rents
the lord's stables | keep | hay horses in ~animals | long stables where the lord's hunting horses are kept
the kennels | keep | ~animals | pens where the lord's hounds are kept
the ice house | keep | cool_store cold in ~cold ~dark | a domed brick pit packed with winter ice and straw
the smithy | crafts | smith shoe metalwork warm in ~metal ~fire ~craft | a smoky forge with an anvil and a tub of water
the farrier's shed | crafts | shoe metalwork warm ~metal ~animals ~fire | a shed with a small forge and iron rings on the wall for tethering
the armourer's workshop | crafts | smith metalwork in ~metal ~weapons ~craft | a workshop hung with mail shirts and helmets
the tinker's cart | crafts | metalwork ~metal ~travel | a painted cart hung with pots and pans
the carpenter's workshop | crafts | timber in ~wood ~craft | a workshop of sawdust, planks and half-made chairs
the potter's kiln | crafts | warm ~fire ~craft ~stone | a brick kiln and a wheel under a lean-to
the weaver's workshop | crafts | buys_wool cloth in ~cloth ~craft | a room with two big looms clacking away
the wool merchant's house | crafts | wool_trade in ~cloth ~trade | a merchant's house with sacks of fleece stacked in the yard
the cloth hall | crafts | wool_trade cloth in ~cloth ~trade | a long hall where fleeces and cloth change hands
the cooper's yard | crafts | nook ~wood ~craft | a yard of barrels and iron hoops
the brewhouse | crafts | drinks nook in ~drink ~grain | the village brewhouse, with copper vats and a smell of malt
the scribe's booth | crafts | scribe literate ~books | a narrow booth with a desk, quills and pots of ink
the bookbinder's | crafts | books literate in ~books ~craft | a workshop of leather covers and stacks of paper
the rope walk | crafts | in ~craft | a long shed where hemp is twisted into rope
the saddler's shop | crafts | in ~animals ~craft | a shop of saddles, bridles and harness
the horse fair | crafts | horse_trade horses crowd ~animals ~trade | a field of pens and shouting dealers on fair day
the horse dealer's yard | crafts | horse_trade horses ~animals ~trade | a yard of paddocks behind a dealer's house
the livery stable | crafts | fodder horses shelter nook in ~animals | stables where mounts are kept for hire
the fodder merchant | crafts | fodder ~grain ~trade | a shed of hay bales and oat sacks for sale
the scholar's house | crafts | books literate in ~books | a tall narrow house where an old scholar lives among his papers
the notary's office | crafts | scribe literate in ~books ~law | a room with a desk, wax seals and a strongbox of deeds
the bookseller's stall | crafts | books ~books ~trade | a stall of second-hand volumes under a canvas awning
the clothier's stall | crafts | clothes ~cloth ~trade | a market stall of hoods, capes and woollen hose
the furrier's shop | crafts | cloth in ~cloth ~trade ~animals | a shop of fox and rabbit furs
her bedroom | home | bed private warm in ~sleep | a small room with a bed under the eaves
her kitchen | home | fire kitchen meal quern in ~food ~fire | her own kitchen, with a pot hanging over the fire
the cellar at home | home | cool_store hide cold lockable in ~dark ~cold | a chilly cellar under her house, reached by a trapdoor
the loft at home | home | hide private high dry_store in ~dark | a low space under the thatch, reached by a ladder
the back garden | home | quiet ~plants | a patch of beans and herbs behind her house
the privy | home | private ~dark | a wooden outhouse at the end of the garden
the rain barrel | home | water ~water ~wood | a barrel under the eaves that catches rain from the roof
the front step | home | ~travel | the step outside her front door, facing the lane
the front room | home | in ~cloth | the room at the front of her house, with a table and a spinning wheel
the washtub | home | wash tubs ~wash ~wood | a big wooden tub and a scrubbing board by her back door
the chicken run | home | hens ~animals | a fenced run behind the house where her hens scratch about
the woodshed at home | home | firewood nook dry_store in ~wood | a lean-to against her house, stacked with split logs
the goat shed | home | stall nook in ~animals | a small shed where her two goats sleep
"""

# need | GOOD properties (any) | POSSIBLE properties (any) | NEAR sorts: a bad
# place of one of these sorts SEEMS to fit (the horse trough for a drink)
PLACE_NEEDS = """
find a drink of water | drinkwater | water drinks kitchen | ~water
hide from the angry guard | hide | crowd sanctuary private nook | ~dark ~wild
see who is coming down the road | lookout | high | ~high ~travel ~law
warn the whole village | alarm | crowd guards announce | ~law ~high
find a healer for a wound | healer | priest herbs surgeon | ~health ~plants ~holy
report a theft | law | guards | ~law ~money
pray | holy | burial quiet priest | ~holy ~dead ~books
sleep for the night | bed | rough_sleep | ~sleep ~grain ~health
shelter from the rain | shelter | in cover | ~wood ~stone ~wild
get warm | fire | warm bed | ~fire ~wash ~sleep ~cold
fix a broken sword | smith | metalwork | ~metal ~weapons ~craft
feed the horse | fodder grazing | grain market hay | ~animals ~grain
cross the river | crossing | boat risky_crossing | ~water ~travel
catch some fish | fishing | water angle | ~water
keep the wine cool | cool_store | cold | ~cold ~dark ~drink
leave the village unseen | secret_exit | woods | ~travel ~dark ~wild
find firewood | firewood | timber branches | ~wood ~fire
buy bread | bread market | food_sell meal alms kitchen | ~food ~trade
store the grain | grain_store | dry_store mill | ~grain ~food ~wood
bury the dead | burial | holy priest gravedigger vault | ~dead ~holy ~stone
trade wool | wool_trade market | trade buys_wool | ~cloth ~trade ~craft
wash the muddy clothes | wash | water drinkwater bathe tubs | ~water ~wash
take a bath | bathe | water tubs | ~water ~wash
get a hot meal | meal | food_sell kitchen alms bread | ~food ~drink ~fire
have a letter written | scribe | literate | ~books ~law
buy a horse | horse_trade | horses market | ~animals ~trade
shoe the horse | shoe | smith metalwork | ~metal ~animals ~fire
find a book to read | books | literate | ~books ~money ~holy
hear the latest news | gossip | crowd chatter | ~drink ~trade ~travel
milk a cow | cows | farmyard | ~animals ~food
collect eggs | hens | farmyard | ~animals
grind the grain into flour | mill | quern | ~grain ~food
get a shave | barber | shaves | ~health ~wash
lock up the thief | jail | lockable guards | ~law ~dark
drink some ale | ale | drinks meal casks | ~drink ~food
keep the sheep safe for the night | pen | stall | ~animals
sell a basket of fish | fish_trade market | trade food_sell buys_food | ~food ~trade ~water
buy a new cloak | clothes | market goods cloth | ~cloth ~trade ~craft
"""


def parse_places():
    places = []
    for line in PLACES.strip().splitlines():
        name, cat, props, desc = [p.strip() for p in line.split("|")]
        toks = props.split()
        places.append({"name": name, "cat": cat, "desc": desc,
                       "props": {t for t in toks if not t.startswith("~")},
                       "sorts": {t for t in toks if t.startswith("~")}})
    needs = []
    for line in PLACE_NEEDS.strip().splitlines():
        need, good, poss, near = [p.strip() for p in line.split("|")]
        needs.append({"need": need, "good": set(good.split()), "near": set(near.split()),
                      "possible": set() if poss == "-" else set(poss.split())})
    return places, needs


def place_unit(places, need):
    """One need: its good and bad options (possible ones are dropped) and the
    three bad pools."""
    opt = lambda p: {"short": p["name"], "long": f"{p['name']}: {p['desc']}",
                     "label": label(p, need), "place": p}
    good = [opt(p) for p in places if label(p, need) == "good"]
    bad = [opt(p) for p in places if label(p, need) == "bad"]
    gcats = {o["place"]["cat"] for o in good}
    same = [o for o in bad if o["place"]["cat"] in gcats]
    near = [o for o in bad if o["place"]["sorts"] & need["near"]]
    unrelated = [o for o in bad if o["place"]["cat"] not in gcats
                 and not o["place"]["sorts"] & need["near"]]
    return {"need": need["need"], "base": f"Mara needs to {need['need']}.",
            "good": good, "bad": bad,
            "pools": {"same": same, "near": near, "unrelated": unrelated}}


# ---------------------------------------------------------------- people

# short | long (the same act with a concrete manner, no hint of wisdom) | purpose
TEMPLATES = """
hit {x} | hit {x} hard, with your fist | attack
kick {x} | kick {x} in the leg, as hard as you can | attack
throw a stone at {x} | pick up a stone and throw it at {x} | pelt
hit {x} with a stick | grab a stick and swing it at {x} | pelt
raise your fist at {x} | raise your fist at {x} and scowl | threaten
wave a stick at {x} | wave a long stick at {x} and stamp your feet | threaten
shout at {x} to stop | shout at {x} to stop, in your loudest voice | stop
tell {x} to stop at once | tell {x}, firmly, to stop at once | stop
order {x} to back off | point at {x} and order them to back off | back_off
tell {x} to keep away | tell {x} to keep away, and mean it | back_off
grab {x}'s arm | grab {x}'s arm with both hands | restrain
hold {x} back | wrap your arms around {x} from behind and hold on | restrain
chase {x} | run after {x} as fast as you can | chase
trip {x} | stick out your foot and trip {x} | chase
pull {x} to safety | take hold of {x} and pull them to safety | rescue
drag {x} out of harm's way | grab {x} by the collar and drag them out of harm's way | rescue
step between {x} and the danger | step in front of {x} with your arms spread wide | shield
shield {x} with your body | crouch over {x} to shield them with your body | shield
warn {x} of the danger | warn {x} of the danger, urgently | warn
warn {x} to watch out | lean close and warn {x} to watch out | warn
tell {x} to run | tell {x} to run, right now | urge_flee
tell {x} to get away | shout to {x} to get away from there | urge_flee
call {x} for help | call out to {x} for help | alert
run to fetch {x} | run as fast as you can to fetch {x} | alert
wave {x} over | wave both arms at {x} to come over quickly | alert
tell {x} what you saw | tell {x} exactly what you saw, step by step | report
offer {x} your coins | hold out your purse of coins to {x} | give_coins
give {x} a silver coin | press a silver coin into {x}'s hand | give_coins
beg {x} for mercy | fall to your knees and beg {x} for mercy | beg
plead with {x} | plead with {x}, hands clasped together | beg
ask {x} to calm down | ask {x}, in a low voice, to calm down | calm
talk quietly to {x} | talk quietly and slowly to {x}, with open hands | calm
bandage {x}'s wound | tear a strip of cloth and bandage {x}'s wound | heal
press a cloth on {x}'s wound | press a folded cloth firmly on {x}'s wound | heal
give {x} water | give {x} a cup of water from the jug | give_water
fetch {x} a drink of water | hurry off and fetch {x} a drink of water | give_water
wrap {x} in your cloak | take off your cloak and wrap it round {x} | cover
help {x} up | give {x} your hand and help them up | lift
speak kindly to {x} | speak kindly and softly to {x} | comfort
sit with {x} | sit down beside {x} and stay a while | comfort
put a hand on {x}'s shoulder | put a gentle hand on {x}'s shoulder | comfort
hug {x} | put your arms around {x} and hug them tight | hug
give {x} some bread | give {x} a piece of bread from your bag | feed
lend {x} a hand | roll up your sleeves and lend {x} a hand | assist
help {x} with the heavy load | put your back into it and help {x} with the heavy load | assist
help {x} pull | take hold beside {x} and pull hard | assist
slap {x} on the back | slap {x} hard between the shoulder blades | backslap
thump {x} on the back | thump {x} on the back with the heel of your hand | backslap
tell {x} the way | tell {x} the way, pointing down the road | inform
draw {x} a map | draw {x} a map in the dust with a stick | inform
point {x} down the right road | turn {x} round and point down the right road | inform
ask {x} what happened | ask {x}, slowly and clearly, what happened | ask
ask {x} what is wrong | lean closer and ask {x} what is wrong | ask
accuse {x} of lying | point at {x} and accuse them of lying | accuse_lie
say that {x} is lying | say, loud enough for all to hear, that {x} is lying | accuse_lie
call {x} a thief | call {x} a thief, loudly, in front of everyone | accuse_theft
speak up for {x} | step forward and speak up for {x} | defend
say that {x} is innocent | say firmly, to everyone, that {x} is innocent | defend
buy something from {x} | buy something from {x} for a few coppers | buy
buy some of {x}'s wares | point at the best of {x}'s wares and buy some | buy
buy a basketful from {x} | fill a basket from {x}'s stock and pay for it | buy
haggle with {x} | haggle with {x} over the price, coin by coin | haggle
sell {x} your goods | offer to sell {x} your goods for a good sum | sell
sell {x} what you brought | hand over what you brought and sell it to {x} | sell
pay {x} what is owed | count out coins and pay {x} what is owed | pay
insult {x} | call {x} a fool to their face | insult
laugh at {x} | point at {x} and laugh out loud | insult
praise {x} | praise {x} warmly, in front of everyone | praise
compliment {x}'s clothes | tell {x} how fine their clothes look today | praise
ask {x} to dance | bow and ask {x} to dance | social
tell {x} a joke | tell {x} your best joke about a goat | social
invite {x} for a drink | invite {x} to the inn for a mug of ale | social
greet {x} | nod and greet {x} with a smile | greet
ignore {x} | ignore {x} and say nothing at all | ignore
turn your back on {x} | turn your back on {x} and look away | ignore
pick {x}'s pocket | slip your hand quietly into {x}'s pocket | steal
thank {x} | thank {x}, with a little bow | thank
"""

# short | long | purpose (aimed at nobody: role "-")
UNTARGETED = """
shout for help | shout "Help!" as loud as you can | alert
run for help | run down the lane to find help | alert
cry out for help | cry out for help, again and again | alert
run away | turn and run away as fast as you can | flee
climb up out of reach | climb onto the nearest wall, out of reach | flee
go home | walk home without a word | leave
walk away | walk away slowly down the lane | leave
hide under the table | crawl under the nearest table and keep still | hide
hide behind a barrel | crouch down low behind a barrel | hide
duck behind a cart | duck down behind a cart and wait | hide
keep eating your stew | keep eating your stew, one spoonful at a time | idle
sit down and wait | sit down on a bench and wait | idle
yawn and stretch | yawn, and stretch your arms over your head | idle
tie your bootlaces | kneel and tie your bootlaces | idle
look up at the sky | look up at the clouds for a while | idle
scratch your head | scratch your head slowly | idle
whistle a tune | whistle an old tune to yourself | idle
take a nap | lie down and take a nap | idle
order another ale | order another mug of ale | idle
brush off your coat | brush the dust off your coat with your hand | idle
crack your knuckles | crack your knuckles one by one | idle
play a game of dice | take out your dice and play a game | idle
read a book | open a book and read a page | idle
hum to yourself | hum a quiet song to yourself | idle
eat an apple | take an apple from your pocket and eat it | idle
sing a song | sing a cheerful song about spring | entertain
dance a jig | dance a jig on the spot | entertain
juggle three apples | juggle three apples, high in the air | entertain
play the flute | take out your flute and play a tune | entertain
sweep the floor | sweep the floor with a broom | chore
fetch more firewood | go out and fetch an armful of firewood | chore
wash the dishes | wash the dishes in a tub of water | chore
mend your boot | sew up the hole in your boot | chore
count your coins | count the coins in your purse | chore
feed the chickens | scatter grain for the chickens | chore
carry the sacks inside | carry the flour sacks inside, one by one | chore
polish your boots | polish your boots with a rag | chore
pray quietly | kneel and pray quietly | pray
say a prayer | say a prayer under your breath | pray
throw water on the flames | fill a bucket and throw it on the flames | douse
beat out the flames | beat the flames with a wet sack | douse
pick up a stick | pick up a heavy stick from the ground | arm
grab a pitchfork | grab a pitchfork from against the wall | arm
watch what happens | stand still and watch what happens | watch
stand and stare | stand and stare with your mouth open | watch
"""

# key sentence
#   cast        role: name; ...
#   GOOD        purposes > roles; ...     ("-" = an action aimed at nobody)
#   POSSIBLE    purposes > roles; ...
# The chooser is always Tom. Cast names avoid the nouns in FILLER (an old
# woman, a dog, a cat, farmers), so a filler never seems to be one of them.
SITUATIONS = """
A bandit grabs a child in the market square.
  threat: the bandit; victim: the child; ally: the blacksmith; authority: the guard; bystander: the flower seller
  attack pelt threaten stop back_off restrain > threat; rescue shield > victim; alert report > authority; alert > ally -
  chase give_coins beg calm > threat; comfort urge_flee calm alert > victim; warn urge_flee alert > bystander; warn report > ally; warn > authority; flee hide arm > -

A roofer falls from his ladder in the inn yard; his leg is twisted under him and his head is bleeding.
  injured: the roofer; healer: the herbwife; ally: the stable boy; bystander: the pedlar; stranger: the traveller
  heal give_water cover comfort > injured; alert > healer ally -
  ask calm > injured; alert > bystander stranger; report > healer

Two drunk men start a fist fight in the tavern, knocking over a table.
  fighter: the drover; fighter2: the carter; authority: the innkeeper; bystander: the fiddler; bystander2: the barmaid
  stop back_off restrain calm > fighter fighter2; alert > authority
  threaten shield rescue > fighter fighter2; report warn > authority; warn urge_flee alert > bystander bystander2; alert flee hide leave watch > -

A thief snatches the cheesewife's purse at her stall and runs off through the crowd.
  threat: the thief; victim: the cheesewife; ally: the butcher; authority: the watchman; bystander: the juggler
  chase stop > threat; alert report > authority; alert > ally -
  attack pelt threaten restrain accuse_theft > threat; comfort ask report > victim; report > ally; alert > bystander

A baker accuses a ragged girl of stealing a loaf, but Tom saw her pay for it.
  accuser: the baker; victim: the ragged girl; authority: the constable; bystander: the weaver; bystander2: the cooper
  defend > victim; report > accuser authority
  calm accuse_lie stop back_off pay ask > accuser; comfort shield ask > victim; alert > authority

A small boy stands crying in the busy square; he has lost his mother.
  child: the small boy; authority: the watchman; stranger: the tall stranger; trader: the apple seller; bystander: the chimney sweep
  comfort ask > child; alert > authority
  hug feed give_water greet social calm > child; alert ask > trader bystander stranger; alert > -; report > authority

Tom's friend Jory sits alone on the church steps, weeping; his old horse died this morning.
  friend: Jory; authority: the priest; bystander: the sexton; bystander2: the washerwoman; stranger: the traveller
  comfort hug > friend
  ask give_water feed social greet cover calm > friend; alert > authority

At dusk a wolf gets into the sheep pen and goes for the lambs.
  threat: the wolf; victim: the lambs; ally: the shepherd; ally2: the farmhand; bystander: the miller's boy
  pelt threaten > threat; alert warn > ally ally2; alert arm > -
  attack stop back_off chase > threat; rescue shield > victim; report > ally ally2; warn urge_flee alert > bystander; flee hide > -

Smoke pours from the inn kitchen and the fire is spreading, while the cook sleeps upstairs.
  victim: the cook; ally: the stable boy; authority: the innkeeper; bystander: the pedlar; stranger: the traveller
  warn > victim ally authority bystander stranger; urge_flee rescue > victim; urge_flee > bystander stranger; alert > ally authority -; douse > -
  urge_flee > ally authority; alert > bystander stranger victim; flee > -

A tired stranger on a mule asks Tom the way to the monastery.
  stranger: the stranger; authority: the reeve; bystander: the potter; bystander2: the goose girl; child: the reeve's son
  inform > stranger
  give_water feed social greet > stranger

After the funeral, the miller's widow stands alone by the grave, crying.
  grieving: the widow; authority: the priest; bystander: the gravedigger; bystander2: the smith's wife; child: the widow's grandson
  comfort > grieving
  hug give_water cover calm > grieving; comfort > child; alert > authority; pray > -

A drunk lout shoves a younger boy into the mud and laughs at him.
  threat: the lout; victim: the younger boy; authority: the watchman; ally: the boy's big sister; bystander: the tanner
  stop back_off > threat; lift comfort shield > victim; alert > authority
  threaten restrain calm attack > threat; rescue ask urge_flee alert calm > victim; alert > ally bystander -

The neighbour's horse lies in its stall, sweating and groaning with colic.
  patient: the horse; owner: the neighbour; healer: the farrier; bystander: the thatcher; child: the neighbour's daughter
  alert > healer owner; report > owner
  comfort calm lift give_water > patient; warn > owner; report > healer; alert report > child; alert > bystander -

In the tavern a traveller cheats at dice with loaded bones and wins a farmhand's last coins.
  liar: the traveller; victim: the farmhand; authority: the innkeeper; ally: the drover; bystander: the fiddler
  accuse_theft > liar; report > victim authority
  accuse_lie stop restrain report > liar; alert > authority; comfort defend give_coins > victim; report > ally

Tom needs a dozen eggs, and the egg wife at her stall has a full basket.
  trader: the egg wife; authority: the market warden; bystander: the bargeman; bystander2: the tinker; child: the egg wife's son
  buy > trader
  haggle greet praise thank pay give_coins > trader; buy > child

Tom sees the miller press his thumb on the scale to give the carter's wife short weight of flour.
  cheat: the miller; victim: the carter's wife; authority: the reeve; ally: the carter; bystander: the cobbler
  accuse_theft > cheat; report > victim authority ally
  accuse_lie stop report > cheat; defend warn ask > victim; alert > authority; warn alert > ally

A little girl slips off the jetty into the river and struggles in the water.
  victim: the little girl; ally: the boatman; bystander: the net mender; bystander2: the washerwoman; stranger: the pilgrim
  rescue > victim; alert > ally -
  alert > bystander bystander2 stranger; lift alert > victim

An old shepherd collapses in the hot hayfield, pale and gasping.
  injured: the old shepherd; healer: the herbwife; ally: the reaper; bystander: the gleaner; child: the shepherd's grandson
  give_water > injured; alert > healer ally -
  comfort lift rescue ask calm > injured; alert > bystander child

A thin, ragged beggar at the inn door holds out his hand and asks for food.
  beggar: the beggar; authority: the innkeeper; bystander: the pedlar; bystander2: the drover; child: the potboy
  feed give_coins > beggar
  give_water social greet ignore ask comfort cover > beggar; buy > authority

A haystack catches fire beside the barn where two small children are playing.
  victim: the two children; ally: the farmhand; ally2: the ploughman; authority: the reeve; bystander: the pedlar
  rescue warn urge_flee > victim; alert warn > ally ally2; alert douse > -
  shield alert > victim; alert warn > authority bystander; urge_flee > ally ally2 authority bystander

A wild boar bursts out of the trees and charges at the woodcutter's daughter.
  threat: the boar; victim: the woodcutter's daughter; ally: the woodcutter; bystander: the charcoal burner; stranger: the pilgrim
  urge_flee warn rescue > victim; alert warn > ally; alert > -
  pelt threaten stop back_off > threat; shield alert > victim; alert warn > bystander stranger; flee arm hide > -

A carter's wagon is stuck deep in the mud, and his horse cannot pull it free.
  owner: the carter; animal: the horse; ally: the blacksmith; bystander: the swineherd; stranger: the pilgrim
  assist > owner; alert > ally
  alert > bystander stranger -; ask > owner; comfort calm assist > animal

A hungry urchin climbs the wall and steals apples from the priest's orchard.
  thief: the urchin; owner: the priest; authority: the watchman; bystander: the laundress; bystander2: the thatcher
  stop feed > thief; report > owner
  back_off chase restrain accuse_theft ask calm ignore give_coins > thief; alert > owner authority; report > authority

At supper, the reeve chokes on a fish bone and cannot breathe.
  injured: the reeve; healer: the herbwife; ally: the reeve's wife; bystander: the fiddler; bystander2: the potboy
  backslap > injured; alert > healer -
  calm > injured; alert > ally bystander bystander2

At the harvest dance, Tom's friend Wenna stands alone by the wall, hoping someone will ask her to dance.
  friend: Wenna; authority: the reeve; bystander: the fiddler; stranger: the young drover; child: the smith's boy
  social praise greet > friend
  comfort hug > friend; social greet > stranger; praise thank > bystander; entertain > -

At night Tom sees a stranger climbing out of the chandler's window with a sack of candles.
  threat: the stranger; owner: the chandler; authority: the night watchman; ally: Tom's brother; bystander: the drunk tinker
  stop > threat; alert > authority owner ally -
  back_off chase restrain accuse_theft threaten pelt attack > threat; report warn > owner authority; warn > ally; hide watch > -

A pedlar swears his cloudy bottle cures every sickness, and an old couple are about to buy it.
  liar: the pedlar; victim: the old couple; authority: the reeve; healer: the herbwife; bystander: the milkmaid
  warn > victim; accuse_lie > liar
  stop back_off ask accuse_theft > liar; report > victim authority; stop > victim; alert > healer authority; warn > authority

On the road Tom finds a merchant lying beaten and bleeding in the ditch; robbers have taken his purse.
  injured: the merchant; ally: the carter; authority: the reeve; bystander: the swineherd; stranger: the pilgrim
  heal give_water cover comfort ask > injured; alert > ally authority -
  lift rescue calm > injured; alert > bystander stranger; report > authority

A soldier on the lonely road draws his sword and demands Tom's purse.
  threat: the soldier; ally: the ploughman; stranger: the pilgrim; bystander: the milkmaid; bystander2: the goat boy
  give_coins > threat
  beg calm > threat; alert > ally; warn urge_flee > bystander bystander2 stranger; alert flee > -

Two boys fight over a ball in the lane, and the smaller one already has a bloody nose.
  fighter: the bigger boy; injured: the smaller boy; authority: the smaller boy's father; bystander: the washerwoman; bystander2: the tinker
  stop back_off restrain > fighter; comfort > injured
  calm ask > fighter injured; heal give_water shield lift rescue stop back_off restrain > injured; alert report > authority

A travelling tinker's wife starts to give birth early, in the inn stable.
  patient: the tinker's wife; healer: the midwife; ally: the innkeeper's wife; bystander: the ostler; stranger: the drover
  alert > healer ally; comfort give_water > patient
  cover ask calm > patient; alert > bystander stranger -

An old man at the well is trying to haul up a heavy bucket and cannot manage it.
  helpee: the old man; bystander: the goose girl; bystander2: the cobbler; child: the potboy; stranger: the knight
  assist > helpee
  greet give_water > helpee; alert > bystander bystander2 stranger

A shy new apprentice sits alone at the feast, knowing nobody.
  newcomer: the apprentice; authority: the master weaver; bystander: the fiddler; bystander2: the brewer; child: the weaver's daughter
  greet social > newcomer
  praise comfort ask > newcomer

A wool buyer offers Tom a fair price for his fleeces.
  trader: the wool buyer; authority: the market warden; bystander: the shepherd; bystander2: the carter; child: the buyer's clerk
  sell > trader
  haggle thank greet social > trader

A crowd is kicking a thief who lies curled up on the ground.
  threat: the angry crowd; victim: the thief; authority: the constable; ally: the priest; bystander: the tanner
  stop back_off calm > threat; alert > authority; shield > victim
  rescue heal lift alert > victim; report > authority; alert > ally -
"""


def _pairs(spec):
    """'a b > r s; c > -' -> {(a, r), (a, s), (b, r), (b, s), (c, '-')}"""
    out = set()
    for group in spec.split(";"):
        if group.strip():
            purposes, roles = group.split(">")
            out |= {(p, r) for p in purposes.split() for r in roles.split()}
    return out


def _fill(template, name):
    poss = name + ("'" if name.endswith("s") else "'s")
    return template.replace("{x}'s", poss).replace("{x}", name)


def parse_people():
    templates = [[p.strip() for p in l.split("|")] for l in TEMPLATES.strip().splitlines()]
    untargeted = [[p.strip() for p in l.split("|")] for l in UNTARGETED.strip().splitlines()]
    sits = []
    for block in SITUATIONS.strip().split("\n\n"):
        key, cast, good, poss = [l.strip() for l in block.strip().splitlines()]
        cast = dict(tuple(p.strip() for p in c.split(":")) for c in cast.split(";"))
        sits.append({"key": key, "cast": cast, "good": _pairs(good), "possible": _pairs(poss)})
    return templates, untargeted, sits


def people_unit(templates, untargeted, sit):
    """One situation: every action with every person, plus the actions aimed
    at nobody, each labelled from its (purpose, role)."""
    opts = []
    for short, long, purpose in templates:
        for role, name in sit["cast"].items():
            opts.append({"short": _fill(short, name), "long": _fill(long, name),
                         "purpose": purpose, "role": role})
    for short, long, purpose in untargeted:
        opts.append({"short": short, "long": long, "purpose": purpose, "role": "-"})
    for o in opts:
        pr = (o["purpose"], o["role"])
        o["label"] = "good" if pr in sit["good"] else "possible" if pr in sit["possible"] else "bad"
    good = [o for o in opts if o["label"] == "good"]
    bad = [o for o in opts if o["label"] == "bad"]
    gpurp = {p for p, _ in sit["good"]}
    groles = {r for _, r in sit["good"]} - {"-"}
    same = [o for o in bad if o["role"] != "-" and o["purpose"] in gpurp]
    near = [o for o in bad if o["role"] in groles and o["purpose"] not in gpurp]
    unrelated = [o for o in bad if o not in same and o not in near]
    return {"need": sit["key"], "base": sit["key"], "good": good, "bad": bad, "all": opts,
            "pools": {"same": same, "near": near, "unrelated": unrelated}}


# ---------------------------------------------------------------- cases

KINDS = {
    "places": {"tail": "She thinks about where she could go.",
               "question": "Where can Mara go for this?"},
    "people": {"tail": "Tom is there.", "question": "What can Tom do?"},
}


def situation_text(base, tail, length, rng):
    if length == "short":
        return f"{base} {tail}"
    f = rng.sample(FILLER, 6)
    return " ".join(f[:3] + [base] + f[3:] + [tail])


def make_case(unit, kind, cfg, rng):
    k = cfg["good"]
    if len(unit["good"]) < k:
        return None
    src = unit["bad"] if cfg["similar"] == "any" else unit["pools"][cfg["similar"]]
    n_bad = cfg["size"] - k
    if len(src) < n_bad:
        return None
    chosen_good = rng.sample(unit["good"], k)
    opts = rng.sample(src, n_bad)
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
    kd = KINDS[kind]
    return {"need": unit["need"],
            "situation": situation_text(unit["base"], kd["tail"], cfg["situation"], rng),
            "question": kd["question"],
            "options": [o[cfg["optlen"]] for o in opts],
            "names": [o["short"] for o in opts],
            "good": [i for i, o in enumerate(opts) if o["label"] == "good"]}


def build(kind, units, n_per_cell=60, seed=0):
    """Every cell holds all factors at the middle value except one. Units
    (needs, situations) are taken in shuffled rounds; a unit that cannot make
    the cell (too few good options, or too few in the pool) is skipped."""
    cases, short = [], []
    for factor, values in FACTORS.items():
        for v in values:
            cfg = dict(MIDDLE, **{factor: v})
            rng = random.Random(f"{seed}-{kind}-{factor}-{v}")
            made, tries = 0, 0
            order = units[:]
            while made < n_per_cell and tries < n_per_cell * 20:
                if tries % len(order) == 0:
                    rng.shuffle(order)
                u = order[tries % len(order)]
                tries += 1
                c = make_case(u, kind, cfg, rng)
                if c is None:
                    continue
                c.update(id=f"{kind}/{factor}={v}/{made}", kind=kind, factor=factor, value=v)
                cases.append(c)
                made += 1
            if made < n_per_cell:
                short.append((factor, v, made))
    return cases, short


def audit_sample(units, rng, n_good=20, n_bad=30):
    """Single labels for the human to audit. A plain random draw is almost all
    obvious bad ones ("pick the old man's pocket"), so it is 20 good options
    and 30 bad ones from the same and near pools, shuffled."""
    rows = []
    for _ in range(n_good):
        u = rng.choice([u for u in units if u["good"]])
        rows.append((u["need"], rng.choice(u["good"])))
    for _ in range(n_bad):
        u = rng.choice(units)
        hard = u["pools"]["same"] + [o for o in u["pools"]["near"] if o not in u["pools"]["same"]]
        rows.append((u["need"], rng.choice(hard)))
    rng.shuffle(rows)
    return [f"- {need} / {o['short']} → **{o['label']}**" for need, o in rows]


def label_check_places(places, needs, units, path, seed=1):
    """For the human: each need's good and possible places, and 50 single
    (need, place) labels to audit."""
    rng = random.Random(seed)
    lines = ["# Label check (Workstream A, places)", "",
             "Mark any label you disagree with. If more than 20% are wrong, the labels are fixed before use.", "",
             "## Each need: good and possible places", ""]
    for nd in needs:
        g = [p["name"] for p in places if label(p, nd) == "good"]
        p_ = [p["name"] for p in places if label(p, nd) == "possible"]
        lines.append(f"- **{nd['need']}**: good = {', '.join(g)}; possible = {', '.join(p_) or '(none)'}")
    lines += ["", "## 50 single labels (20 good, 30 bad that look alike), shuffled", ""]
    lines += audit_sample(units, rng)
    Path(path).write_text("\n".join(lines) + "\n")


def label_check_people(sits, units, path, seed=1):
    """For the human: each situation's cast and its good and possible
    (purpose → person) pairs, and 50 single (situation, action) labels."""
    rng = random.Random(seed)
    by = {}
    for short, _, purpose in parse_people()[0]:
        by.setdefault(purpose, []).append(short)
    for short, _, purpose in parse_people()[1]:
        by.setdefault(purpose, []).append(short)

    def show(pairs, cast):
        per = {}
        for p, r in sorted(pairs):
            per.setdefault(r, []).append(p)
        return "; ".join(f"{', '.join(ps)} → {cast.get(r, 'nobody')}" for r, ps in per.items()) or "(none)"

    lines = ["# Label check (Workstream A, people)", "",
             "Mark any label you disagree with. If more than 20% are wrong, the labels are fixed before use.", "",
             "Tom is the one choosing. A purpose is a group of actions (below); a pair",
             "\"attack → the bandit\" means every attack action aimed at the bandit.", "",
             "## Purposes", ""]
    lines += [f"- **{p}**: {' / '.join(ts)}" for p, ts in by.items()]
    lines += ["", "## Each situation: good and possible", ""]
    for s in sits:
        lines.append(f"- **{s['key']}** ({', '.join(f'{r}: {nm}' for r, nm in s['cast'].items())})")
        lines.append(f"  - good: {show(s['good'], s['cast'])}")
        lines.append(f"  - possible: {show(s['possible'], s['cast'])}")
    lines += ["", "## 50 single labels (20 good, 30 bad that look alike), shuffled", ""]
    lines += audit_sample(units, rng)
    Path(path).write_text("\n".join(lines) + "\n")


def pool_problems(units):
    out = []
    for u in units:
        pl = u["pools"]
        if len(u["good"]) < 4 or len(u["bad"]) < 127 or len(pl["same"]) < 15 or len(pl["near"]) < 15:
            out.append(f"  {u['need'][:60]}: good {len(u['good'])}, bad {len(u['bad'])}, "
                       f"same {len(pl['same'])}, near {len(pl['near'])}, unrelated {len(pl['unrelated'])}")
    return out


def checks(places, templates, untargeted, sits, punits):
    """Nothing twice in one list, and no cast name hiding in a filler."""
    names = [p["name"] for p in places]
    dup = {n for n in names if names.count(n) > 1}
    assert not dup, f"places named twice: {dup}"
    for u in punits:
        texts = [o["short"] for o in u["all"]]
        dup = {t for t in texts if texts.count(t) > 1}
        assert not dup, f"{u['need']}: options twice: {dup}"
    filler = set(" ".join(FILLER).lower().replace(".", " ").replace(",", " ").split())
    for s in sits:
        for nm in s["cast"].values():
            head = nm.split()[-1].lower()
            assert head not in filler and head + "s" not in filler, \
                f"{s['key']}: '{nm}' is also in a filler"
        for pr in s["good"] | s["possible"]:
            assert pr[1] == "-" or pr[1] in s["cast"], f"{s['key']}: no role {pr[1]}"
    purposes = {t[2] for t in templates} | {t[2] for t in untargeted}
    for s in sits:
        unknown = {p for p, _ in s["good"] | s["possible"]} - purposes
        assert not unknown, f"{s['key']}: unknown purposes {unknown}"
        both = s["good"] & s["possible"]
        assert not both, f"{s['key']}: both good and possible: {both}"


def main():
    places, needs = parse_places()
    templates, untargeted, sits = parse_people()
    punits = [people_unit(templates, untargeted, s) for s in sits]
    checks(places, templates, untargeted, sits, punits)
    lunits = [place_unit(places, nd) for nd in needs]
    RES.mkdir(parents=True, exist_ok=True)
    for kind, units in (("places", lunits), ("people", punits)):
        cases, short = build(kind, units)
        out = HERE / f"cases_{kind}.jsonl"
        with out.open("w") as f:
            for c in cases:
                f.write(json.dumps(c) + "\n")
        print(f"{kind}: {len(units)} units, {len(cases)} cases -> {out}")
        for factor, v, made in short:
            print(f"  only {made} cases for {factor}={v}")
        probs = pool_problems(units)
        if probs:
            print("  units that cannot fill every cell (good < 4, bad < 127, same or near < 15):")
            print("\n".join(probs))
    label_check_places(places, needs, lunits, RES / "label_check_places.md")
    label_check_people(sits, punits, RES / "label_check_people.md")
    print(f"{len(places)} places, {len(needs)} needs; {len(sits)} situations, "
          f"{len(templates)} aimed actions, {len(untargeted)} actions aimed at nobody")


if __name__ == "__main__":
    main()
