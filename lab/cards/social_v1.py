"""Social cards, v1: the questions a mind has to answer, with the answers.

Two tracks, and they are kept apart because they fail for different reasons.

READ — someone says something. What are they doing, to whom, how hard?
  This is parsing. It has a right answer that does not depend on who hears it,
  and it is the part a player's free text has to go through before anything in
  the world can respond to it.

MIND — a person, their state, and what just happened. What do they do, how
  afraid are they, do they believe it? This is judgement. The gold answers here
  are HAND-LABELLED, a reasonable GM's call, and they are provisional in exactly
  the way the reflex table was: when the sim can produce these situations and
  score the outcomes, its labels replace these. Every card is written so that
  the state settles the answer — a brave armed guard with friends does not fold
  to an unarmed man — rather than leaving it to taste.

The questions are the same on every card of a track (only a target list
varies), because that is how they will be asked in play: a fixed set of typed
questions, one state that changes. Run this file to write social_v1.jsonl.
"""
import json
import random
from pathlib import Path

INTENTS = {
    "threaten": "warns of harm unless the listener does something",
    "bribe": "offers money or goods for a favour or a blind eye",
    "flatter": "praises the listener to win them over",
    "insult": "demeans or mocks the listener",
    "plead": "begs for mercy, help or pity",
    "ask_info": "asks for information: directions, news, who or where",
    "deceive": "tells the listener something false to get an advantage",
    "command": "gives an order, expecting to be obeyed by right of rank",
    "comfort": "reassures or consoles the listener",
    "trade": "proposes a fair exchange or haggles over a price",
    "small_talk": "idle friendly chat with no request in it",
}
HOSTILITY = ["friendly", "neutral", "curt", "hostile"]

REPLIES = {
    "comply": "does what was asked or demanded",
    "refuse": "says no and stands their ground",
    "haggle": "counters with terms or a price",
    "flee": "tries to get away",
    "attack": "strikes first",
    "call_help": "shouts for guards, friends or anyone nearby",
    "lie": "answers with something untrue to protect themselves or others",
    "question": "asks for more before deciding",
}
FEAR = ["calm", "uneasy", "afraid", "terrified"]


def read(cid, scene, speaker, line, people, intent, target, hostility, asks):
    state = {"scene": scene, "present": ", ".join(people),
             "speaker": speaker, "says": line}
    return {
        "id": cid, "track": "read", "state": state,
        "questions": {
            "intent": {"type": "choice",
                       "instructions": "What is the speaker doing with these words?",
                       "criteria": INTENTS},
            "target": {"type": "choice",
                       "instructions": "Who are the words aimed at?",
                       # shuffled by the card's own id: listed in the order
                       # they were written, the target was nearly always
                       # first, and guessing "the first one" scored 57%
                       "criteria": {p: f"the person called {p}" for p in
                                    random.Random(cid).sample(
                                        [p for p in people if p != speaker],
                                        len(people) - 1)}},
            "hostility": {"type": "score",
                          "instructions": "How hostile is the speaker being?",
                          "criteria": HOSTILITY},
            "asks": {"type": "noul",
                     "instructions": "Is the speaker asking the listener to do or give something?"},
        },
        "gold": {"intent": intent, "target": target,
                 "hostility": HOSTILITY.index(hostility), "asks": asks},
    }


def mind(cid, who, state, event, reply, fear, believes):
    return {
        "id": cid, "track": "mind",
        "state": {"person": who, "situation": state, "what_happened": event},
        "questions": {
            "reply": {"type": "choice",
                      "instructions": "What does this person do next?",
                      "criteria": REPLIES},
            "fear": {"type": "score",
                     "instructions": "How afraid is this person right now?",
                     "criteria": FEAR},
            "believes": {"type": "noul",
                         "instructions": "Does this person believe what they were just told?"},
        },
        "gold": {"reply": reply, "fear": FEAR.index(fear), "believes": believes},
    }


CARDS = [
    # ---------------------------------------------------------------- READ
    read("r01", "a tavern at night", "Kael", "Hand over the strongbox key, Brenna, or I open your throat instead.",
         ["Kael", "Brenna", "Old Tam"], "threaten", "Brenna", "hostile", True),
    read("r02", "a city gate", "Kael", "Five silver says you never saw our wagon, sergeant.",
         ["Kael", "Sergeant Voss", "a carter"], "bribe", "Sergeant Voss", "neutral", True),
    read("r03", "a noble's garden party", "Isra", "Lady Marrow, your roses put the royal gardens to shame.",
         ["Isra", "Lady Marrow", "a steward"], "flatter", "Lady Marrow", "friendly", False),
    read("r04", "a training yard", "Dorn", "You swing that sword like a farmer swings a hoe, Pell.",
         ["Dorn", "Pell", "the drillmaster"], "insult", "Pell", "hostile", False),
    read("r05", "a bandit camp", "Ana", "Please, I have two children at home, let me go and you will never see me again.",
         ["Ana", "the bandit chief", "a lookout"], "plead", "the bandit chief", "friendly", True),
    read("r06", "a crossroads", "Kael", "Which of these roads takes us to Harrowgate before dark?",
         ["Kael", "a shepherd"], "ask_info", "a shepherd", "neutral", True),
    read("r07", "a manor door", "Isra", "The Duke sent me personally; he expects me in the study.",
         ["Isra", "the butler"], "deceive", "the butler", "neutral", True),
    read("r08", "a barracks", "Captain Hale", "Corporal, take four men and hold the east bridge until relieved.",
         ["Captain Hale", "the corporal", "a recruit"], "command", "the corporal", "neutral", True),
    read("r09", "a burned farmhouse", "Ana", "It is not your fault, Pim. Nobody could have stopped that fire.",
         ["Ana", "Pim", "a neighbour"], "comfort", "Pim", "friendly", False),
    read("r10", "a market stall", "Dorn", "Three wolf pelts for that hatchet, and I will throw in the fox.",
         ["Dorn", "the smith", "a child"], "trade", "the smith", "neutral", True),
    read("r11", "a riverside inn", "Old Tam", "Fine weather for the harvest this year, eh, Brenna?",
         ["Old Tam", "Brenna", "Kael"], "small_talk", "Brenna", "friendly", False),
    read("r12", "a dungeon cell", "the jailer", "Keep your mouth shut or you eat nothing for a week, rat.",
         ["the jailer", "the prisoner"], "threaten", "the prisoner", "hostile", True),
    read("r13", "a guildhall", "Isra", "I am told the guildmaster owes you a debt; I could see it paid tomorrow if my name reaches the council.",
         ["Isra", "the clerk", "a porter"], "bribe", "the clerk", "friendly", True),
    read("r14", "a tavern at night", "Pell", "Where did the tall stranger with the scarred hand go, Old Tam?",
         ["Pell", "Old Tam", "Brenna"], "ask_info", "Old Tam", "neutral", True),
    read("r15", "a toll bridge", "Kael", "I already paid your partner on the far side, he waved us through.",
         ["Kael", "the tollman"], "deceive", "the tollman", "neutral", True),
    read("r16", "a throne room", "the queen", "Kneel.",
         ["the queen", "the envoy", "a guard"], "command", "the envoy", "curt", True),
    read("r17", "a dockside", "Dorn", "Nice boat. Shame if the tar caught, with all this lamp oil about.",
         ["Dorn", "the fisherman", "a gull"], "threaten", "the fisherman", "hostile", True),
    read("r18", "a temple", "Ana", "Holy mother, my brother is dying of the grey fever, I beg you to come.",
         ["Ana", "the priestess", "an acolyte"], "plead", "the priestess", "friendly", True),
    read("r19", "a guard post", "Pell", "Your mother was a goat and your father smelt of elderberries.",
         ["Pell", "the guard", "a merchant"], "insult", "the guard", "hostile", False),
    read("r20", "a market stall", "the merchant", "Twelve silver is robbery. Eight, and you have a deal.",
         ["the merchant", "Isra"], "trade", "Isra", "curt", True),
    read("r21", "a war camp", "Captain Hale", "You did well today, lad. Your first battle and you held the line.",
         ["Captain Hale", "a recruit", "the quartermaster"], "comfort", "a recruit", "friendly", False),
    read("r22", "a scholar's library", "Isra", "Master Orrin, your treatise on the old runes is the finest I have ever read.",
         ["Isra", "Master Orrin", "a student"], "flatter", "Master Orrin", "friendly", False),
    read("r23", "a mine entrance", "Dorn", "How deep does the old shaft go, and is there bad air down there?",
         ["Dorn", "the foreman"], "ask_info", "the foreman", "neutral", True),
    read("r24", "a thieves' den", "Kael", "The watch are coming, I saw them at the corner, we have to move the goods now!",
         ["Kael", "the fence", "a pickpocket"], "deceive", "the fence", "neutral", True),
    read("r25", "a village well", "Brenna", "Morning, Pim. The swallows are back early this year.",
         ["Brenna", "Pim"], "small_talk", "Pim", "friendly", False),
    read("r26", "a ship's deck", "the captain", "Reef the mainsail and get that line secured, now!",
         ["the captain", "the bosun", "a passenger"], "command", "the bosun", "curt", True),
    read("r27", "an arena", "Pell", "Crawl back to your mother, you wet little worm.",
         ["Pell", "the challenger", "the crowd"], "insult", "the challenger", "hostile", False),
    read("r28", "a smugglers' cove", "Isra", "A crate of this brandy is yours if the lighthouse stays dark tonight.",
         ["Isra", "the lighthouse keeper"], "bribe", "the lighthouse keeper", "friendly", True),
    read("r29", "a ruined chapel", "Ana", "Spare the boy. Take me instead, I will not fight you.",
         ["Ana", "the knight", "a boy"], "plead", "the knight", "friendly", True),
    read("r30", "a caravan camp", "Dorn", "My salt for your dried meat, measure for measure. Fair?",
         ["Dorn", "the caravan master", "a mule driver"], "trade", "the caravan master", "friendly", True),

    # ---------------------------------------------------------------- MIND
    mind("m01", "Brenna, innkeeper, timid, unarmed",
         "alone behind her bar at night; a stranger has a knife at her throat",
         "He demands the strongbox key or he will kill her.",
         "comply", "terrified", True),
    mind("m02", "Sergeant Voss, veteran gate guard, brave, armed, loyal to the city",
         "at the gate with six of his men around him in daylight",
         "An unarmed traveller threatens to burn the gate down unless he is let through.",
         "refuse", "calm", False),
    mind("m03", "Sergeant Voss, gate guard, underpaid, known to take coin",
         "alone at the side gate at night, nobody watching",
         "A carter offers him five silver to forget a wagon passed.",
         "comply", "calm", True),
    mind("m04", "Lady Marrow, noblewoman, vain, proud of her roses",
         "hosting a garden party among friends",
         "A young woman praises her roses above the royal gardens.",
         "question", "calm", True),
    mind("m05", "the butler, suspicious, twenty years in service, knows the Duke is abroad",
         "at the manor door, guards within call",
         "A stranger says the Duke sent her personally and expects her in the study.",
         "call_help", "uneasy", False),
    mind("m06", "Pim, a farm boy of twelve",
         "standing in the ashes of his family's farmhouse",
         "A kind neighbour tells him the fire was not his fault.",
         "question", "uneasy", True),
    mind("m07", "the smith, shrewd trader",
         "at her stall; the hatchet is her best work and worth more than three pelts",
         "A hunter offers three wolf pelts and a fox for the hatchet.",
         "haggle", "calm", True),
    mind("m08", "a lone goblin scout, cowardly",
         "cornered in a dead-end tunnel by three armed adventurers, no way past them",
         "One of them raises a sword and shouts at it to surrender.",
         "comply", "terrified", True),
    mind("m09", "a lone goblin scout, cowardly",
         "in a tunnel with an open side passage behind it; three adventurers approach from ahead",
         "They shout that they are coming for it.",
         "flee", "afraid", True),
    mind("m10", "Grask, orc war chief, hot-tempered, proud",
         "in his own hall surrounded by his warband",
         "A human envoy calls him a pig who rules pigs.",
         "attack", "calm", True),
    mind("m11", "the tollman, careful, keeps a tally with his partner",
         "at his booth; his partner's signal flag shows nobody has paid today",
         "A traveller claims he already paid the partner on the far side.",
         "refuse", "calm", False),
    mind("m12", "the priestess, devout, sworn to heal the sick",
         "in the temple at evening, no urgent patients",
         "A weeping woman begs her to come heal a brother dying of fever.",
         "comply", "calm", True),
    mind("m13", "the clerk, ambitious, wants a council seat",
         "at his desk in the guildhall, alone",
         "A visitor hints that his name could reach the council if he opens the sealed ledgers for her.",
         "haggle", "uneasy", True),
    mind("m14", "the fence, paranoid, has been raided before",
         "in his back room with a stack of stolen silver",
         "A young thief bursts in saying the watch is at the corner.",
         "flee", "afraid", True),
    mind("m15", "the fence, paranoid, has a lookout posted at the corner",
         "in his back room; his lookout just whistled all clear a moment ago",
         "A young thief bursts in saying the watch is at the corner.",
         "question", "uneasy", False),
    mind("m16", "Old Tam, a gossip who likes the stranger's coin",
         "drinking in the tavern; the scarred stranger paid him to keep quiet",
         "A young man asks where the tall stranger with the scarred hand went.",
         "lie", "calm", True),
    mind("m17", "the shepherd, friendly, knows the roads well",
         "at a crossroads at noon with his flock",
         "A traveller politely asks which road leads to Harrowgate.",
         "comply", "calm", True),
    mind("m18", "a merchant, alone on a forest road",
         "three bandits step out with drawn bows; his guards are an hour behind",
         "The bandits demand his purse.",
         "comply", "terrified", True),
    mind("m19", "a merchant, travelling with twelve armed caravan guards",
         "on a forest road; a single ragged bandit steps out with a club",
         "The bandit demands his purse.",
         "refuse", "calm", True),
    mind("m20", "the knight, honourable, sworn never to harm a child",
         "has captured a boy who stole bread; the boy's mother is before him",
         "The mother begs him to spare the boy and take her instead.",
         "comply", "calm", True),
    mind("m21", "the lighthouse keeper, dutiful, lost his son to a wreck",
         "in the lighthouse at dusk",
         "A smuggler offers him a crate of brandy to leave the lamp dark tonight.",
         "refuse", "calm", True),
    mind("m22", "a village child, shy",
         "playing near the well; a large armoured stranger crouches beside her",
         "The stranger asks her kindly where the elder lives.",
         "comply", "uneasy", True),
    mind("m23", "the jailer, cruel, holds the keys",
         "outside a locked cell; the prisoner is chained",
         "The prisoner threatens to curse his whole family.",
         "attack", "calm", False),
    mind("m24", "a city watchman, new to the job, nervous",
         "alone in an alley at night; a masked figure with a crossbow blocks the way out",
         "The figure tells him to walk away and forget what he saw.",
         "comply", "terrified", True),
    mind("m25", "a city watchman, new to the job, nervous",
         "alone in an alley at night, a whistle on his belt, a patrol two streets away",
         "A drunk shoves past him and calls him a coward.",
         "call_help", "uneasy", True),
    mind("m26", "the caravan master, fair dealer",
         "at camp, short on salt, long on dried meat",
         "A hunter offers salt for dried meat, measure for measure.",
         "comply", "calm", True),
    mind("m27", "the queen's envoy, loyal to a rival crown",
         "in a foreign throne room surrounded by the queen's guards",
         "The queen orders him to kneel.",
         "comply", "uneasy", True),
    mind("m28", "the drillmaster, stern, respected",
         "in the training yard with his recruits watching",
         "A recruit mocks his swordsmanship to his face.",
         "attack", "calm", True),
    mind("m29", "Master Orrin, an old scholar, lonely",
         "in his library, rarely visited",
         "A visitor says his treatise on the old runes is the finest she has read.",
         "question", "calm", True),
    mind("m30", "a dungeon rat-man, starving",
         "in a flooded tunnel; an adventurer holds out a strip of dried meat and asks the way to the lower level",
         "The adventurer offers the meat for directions.",
         "comply", "uneasy", True),
]


def main():
    out = Path(__file__).with_name("social_v1.jsonl")
    with out.open("w") as f:
        for c in CARDS:
            f.write(json.dumps(c) + "\n")   # NOT sort_keys: option order is part of the card
    tracks = {}
    for c in CARDS:
        tracks[c["track"]] = tracks.get(c["track"], 0) + 1
    print(out, tracks)


if __name__ == "__main__":
    main()
