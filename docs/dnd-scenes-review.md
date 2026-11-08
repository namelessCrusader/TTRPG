# D&D scenes in real play: a review

Simplified Technical English. 2026-10-01.

**The question.** NPCs in our village scene (an inn at night: a peasant, a
knight, a merchant, a thief, a cook, a bandit) choose actions from a menu.
The world simulation decides facts. We found two problems. The model picks
whatever the scene text names. A personality sentence ("Tom is
hot-tempered") overrides the situation. The target is scenes that feel
seamless, consistent and real. Characters sometimes do the obvious thing.
Sometimes they act on their nature against the obvious. Sometimes they
misread the scene, make mistakes, or do something stupid. A game master
reads odd actions in the way that is best for the scene. So we looked at
real D&D play to see how people do this.

**Method.** We searched the web and opened each source. We read raw text
(the transcript files, the forum pages, the saved Reddit posts) and we
checked each quote word for word against it. We did not invent or repair
any quote. Where a quote keeps a typing error, the error is in the source.
"..." marks a cut. We counted some patterns in 17 Critical Role episodes
(Section 4.1). Numbers in brackets point to Section 6.

**Honesty notes.**

- Critical Role and FIREBALL lines: checked against the raw data files.
- Reddit and forum stories: the words are real, but the stories are told by
  one player or GM and can be exaggerated. People post stories *because*
  they are strange, so these over-represent mistakes.
- Items marked **(not verified)** we could not open or could not check.

| term | meaning |
| --- | --- |
| GM, DM | Game master, dungeon master. The person who describes the world, plays all NPCs and decides results. |
| PC, NPC | Player character (run by a player). Non-player character (run by the GM). |
| actual play | A recorded or streamed real game (for example *Critical Role*, *Dimension 20*). |
| play-by-post | A game played in writing, one post at a time, in a forum or chat. |
| check, roll, DC | The GM asks for a die roll ("make an animal handling check"). DC is the number to beat. A "natural 1" or "natural 20" is the lowest or highest result on the die. |
| ruling | A decision by the GM for a case that the rules do not cover. |
| "you can try" | A common GM answer: the act is allowed; the dice and the world decide the result. |
| rule of cool | A GM habit: allow a dramatic act if it is fun enough, even if the rules are not clear. |
| "yes, and" | An improvisation rule: accept what the other person said, then add to it. |
| in character, out of character | Speaking as the character, or speaking as the real person at the table. |
| retcon | Take back something that was said or done in the story. |
| misreading, misbelief | Misreading: a *player* has a wrong picture of the scene. Misbelief: a *character* believes something false inside the story. |

## 1. Where the transcripts are

| # | source | what it contains | size | how to get it | licence | use for us |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **CRD3**, Critical Role Dungeons and Dragons Dataset (Rameshkumar and Bailey, ACL 2020) [1][2] | Spoken play from the show *Critical Role*. One JSON file for each episode. Each turn has `NAMES` (the actor), `UTTERANCES` and `NUMBER`. `METADATA` has the wiki summary and a synopsis. Made from fan transcripts. | 159 episodes (campaign 1: 113, campaign 2: 46), 398,682 turns. About 105 MB in total; 0.45 to 1.06 MB for each file. | One file at a time: `https://raw.githubusercontent.com/RevanthRameshkumar/CRD3/master/data/cleaned%20data/C1E081.json`. Also on Hugging Face as `microsoft/crd3`. | CC BY-SA 4.0 | Real table talk with one GM. No game state. Speakers are actors, not characters. Best source for misreadings and GM rulings. |
| 2 | **Critical Role wiki transcripts** [3] | The same fan transcripts. Topic pages (for example "Deck of Many Things") give the episode and the time of each famous moment. | All episodes, also campaign 3. | MediaWiki API with a browser user agent: `criticalrole.fandom.com/api.php?action=parse&page=<Page>&prop=wikitext&format=json`. `?action=raw` returns a Cloudflare check. | Fandom (CC BY-SA) | Use the topic pages to find scenes. Not independent from CRD3. |
| 3 | **CritRoleStats** [4] | Counts: dice rolls, natural 20s and 1s, damage, spells, kills, deaths, level-ups. | Up to campaign 3, episode 81. | critrolestats.com. The site says: "As of January 1, 2024, CritRoleStats has sunset". The Omen Archive continues it. | Not stated | How often things happen (crits, deaths). No dialogue. We did not check if the data downloads as tables. |
| 4 | **FIREBALL** (Zhu et al., ACL 2023) [5][6] | D&D played in text on Discord with the Avrae bot (a bot that rolls dice and keeps hit points). Each row: the chat lines before a bot command, the command, the bot result, the chat lines after, and the full game state before and after (hit points, conditions such as "Blinded"). | Paper: 25k combats, 8M utterances, 3.6k authors, 1.3M states. Hugging Face release: 153,829 rows in 1,471 JSONL files (about 2 GB). | One file at a time: `huggingface.co/datasets/lara-martin/FIREBALL/resolve/main/filtered/<hash>.jsonl`. The Hugging Face rows API does not work for this dataset. | CC BY 4.0, research use | The only large source with real game state next to each choice. Weak points: every speaker is "Player N", so the GM is not marked. Much text is about bot commands. Combat only. |
| 5 | **D&D Beyond play-by-post corpus** (Callison-Burch et al., EMNLP 2022) [7] | Play-by-post games from the D&D Beyond forums. Game state was added by rules, so it is partial. Note: the paper used Giant in the Playground only to train a classifier for in-character and out-of-character posts. | 896 campaigns, 815,106 turns, 58.2M words, 532,270 dice rolls. | We found no download. Treat it as not public. | — | Long campaigns with dice. Not available. |
| 6 | **G4C, "I Cast Detect Thoughts"** (Zhou et al., ACL 2023) [8] | 47k GM–player dialogues rebuilt from source 5. 2.5k have human labels: "does this GM turn guide the players?" | 47k dialogues | "Supplementary materials". We found no public URL. | — | Labels for GM guidance. |
| 7 | **CALYPSO** (Zhu et al., AIIDE 2023) [9] | Not a dataset. Language-model tools for GMs on Discord, studied with 71 players and GMs for four months. | — | Paper only. | — | Finding: a rephrased stat block "doesn't really help in running an encounter". GMs asked for monsters' "motives, tactics, and who they might prioritize [in a fight]". |
| 8 | **Deep Dungeons and Dragons** (Louis and Sutton, NAACL 2018) [10] | Play-by-post transcripts from roleplayerguild.com. Built to predict a character's actions from that character's description. | 91 MB archive | `groups.inf.ed.ac.uk/cup/ddd/` **(not verified: reported by a helper; we did not open it)** | Not stated | Closest to our task. Their result: "Action sequences are better predicted when the character performing the action is also taken into account". |
| 9 | **STORIUM** (Akoury et al., EMNLP 2020) [11] | An online story game. Each entry has "cards" next to the text (a character's goal, strength, weakness, challenge). | 5,743 stories, 30,119 authors | storium.cs.umass.edu (it showed a maintenance page when we tried). | Research release | An explicit trait label next to each written choice. |
| 10 | **Dimension 20** [12] | No research dataset. Volunteer fan transcripts (a Google Drive linked from the Dimension 20 wiki) and `transcripts.foreverdreaming.org`. | Several seasons | **(not verified: reported by a helper; we did not open them)** | Unclear. Treat as copyrighted. | Professional comedy play. Many deliberate "stupid" acts. |
| 11 | **Forums and Reddit** [15]–[23] | First-hand stories: r/DnD, r/DMAcademy, r/rpghorrorstories, D&D Beyond forums, Giant in the Playground, RPG.net. | — | `reddit.com` blocks scripts. This works: `https://arctic-shift.photon-reddit.com/api/posts/ids?ids=<post id>`. Giant in the Playground and RPG.net returned Cloudflare blocks (403). | Site terms | Self-reported and selected for being strange. |
| 12 | **Classic table stories** [13][14] | "Eric and the Dread Gazebo" by Richard Aronson (© 1986; printed in *The Spell Book* #13, 1987). Old story collections. | — | rec.humor.funny copy at netfunny.com | Copyright of the authors | The best-known real misreading story. Second-hand: Aronson knew the GM and the player. |

**Which source fits which question.**

- How players misread scenes and how a GM answers: CRD3 (1).
- What a choice looks like next to the exact game state: FIREBALL (4).
- Character description → action: Deep Dungeons and Dragons (8), STORIUM (9).
- How often the dice decide: CritRoleStats (3).

## 2. Excerpts from real play

**How to read these.** Speakers in Critical Role are actors. Campaign 1:
TRAVIS = Grog (a strong, simple barbarian), LAURA = Vex (a ranger), LIAM =
Vax (a rogue), SAM = Scanlan (a bard), TALIESIN = Percy (a gunslinger),
MARISHA = Keyleth (a druid). Campaign 2: LAURA = Jester (a playful cleric),
MARISHA = Beau (a monk), LIAM = Caleb (a wizard), SAM = Nott (a goblin
rogue). MATT = Matthew Mercer, the GM. "C1E081" = campaign 1, episode 81.
Turn numbers are the CRD3 `NUMBER` field. FIREBALL rows are given as file
and line (counting from 0).

### (a) A player or character misreads the scene

**a1. The gazebo.** Aronson, "Eric and the Dread Gazebo" [13]. The GM
describes a garden with a gazebo. Eric does not know the word. He treats
the gazebo as a monster.

> ERIC: I use my sword to detect good on it.
> ED: It's not good, Eric. It's a gazebo.
> ...
> ED: (Thoroughly frustrated) It's too late. You've awakened the gazebo.

*Shows:* a player treats a *named* thing as an actor. The GM's naming made
it look important. In the end the GM plays along, and the "gazebo" eats
Eric. Second-hand story.

**a2. The back door.** CRD3 C2E005 "The Open Road", turns 1082–1086 [1].
Beau jumps through a window to hit a fleeing man.

> MARISHA: I thought it was much closer than that.
> MATT: Unfortunately, no. But you do manage to make it all the way to this door there.
> MARISHA: Oh! They're going out the back door. I thought they were going out that way!
> MATT: Nope!

*Shows:* a wrong picture of distance and direction. The GM does not cancel
the act. He resolves it as far as the world allows, then corrects the
picture in one word.

**a3. Talking to friends who cannot hear.** FIREBALL, file
`930d5994ee67e86436e60a097677fce2.jsonl`, lines 44 and 53 [6].

> "Hey guys!" *She yells out to rally the party, not quite realizing that they can neither see nor hear her.*
> ...
> *Right. Okay.* "Althea," *Aliseth shouts to the literal sky since she's not aware Althea is right there.*

*Shows:* in text play, players *write* their character's misbelief on
purpose. The player knows; the character does not.

**a4. The ring of feather falling.** u/Vexra (the GM), r/DnD, 2019-08-13
[16]. The party jumps from a tower. The cleric has a ring that protects
only the wearer. He thinks he casts a spell from it.

> "I don't want to cast the spell through the ring until we're halfway down the mountain"
> ...
> The BBEG was impressed by the clerics ruthlessness in apparently murdering his own party and offered him a job which the cleric happily accepted

*Shows:* one misunderstood rule kills the party. The GM gives the villain
("BBEG", big bad evil guy) a reaction that turns the disaster into a new
story. The same post has a second mistake: the GM had warned the wizard
that summoned demons "are completely uncontrollable"; the wizard summoned
them "anyway", and they attacked the party.

### (b) A "stupid", impulsive act against the obvious

**b1. Grog draws from the Deck of Many Things.** CRD3 C1E081 "What Lies
Beneath the Surface", turns 733–734, 773, 808 [1]. The deck can give great
gifts or destroy a character. The other players know it. Travis does not.

> TRAVIS: Yeah, I'll pull one card out, but I want to pull the one in the middle because I remember magicians be like, choose a card, any card.
> LIAM: Travis, you could destroy the world.
> TRAVIS: I genuinely am only doing this because I don't know what the hell it is.
> MATT: You said you were just pulling one card, so you pull one card.

*Shows:* the player's ignorance makes the character's choice honest. The
table keeps the secret on purpose (LIAM, turn 772: "We'll tell you later.
... 'Cause we're all out of character."). The GM holds Grog to his exact
words. The deck then follows Grog through 34 episodes (see g8).

**b2. "The time to be bold is now!"** CRD3 C2E045 "The Stowaway", turns
2399–2402 [1]. Beau touched a magic window and it pulled her and Jester
in. The guest character Twiggy then does the same.

> DEBORAH: Twiggy says  The time to be bold is now! I'll run up and touch the same spot she did.
> SAM: No! Twiggy! Don't!
> MATT: (glass shattering) Twiggy's gone.
> TRAVIS: What a fucking bunch of idiots we are! It's so good.

*Shows:* the table warns, the character acts on her nature, the GM lets it
land at once. The table enjoys it.

**b3. A sword in a bar fight.** u/Tyler9351 (a player), r/DnD, 2019-06-12
[17]. In a tavern, a half-orc PC bumps into an orc.

> "WaNnA fIgHt?" Mercy asks, and then proceeds to draw his sword."This is a BAR fight, right?" Some people ask.
> ...
> Before he answers, the orc smashes a chair on him, chokes him, and smashes a beer bottle on his head. People start to bet on who's going to win, even our own party.

*Shows:* an inn scene. The PC breaks the norm of a bar fight (no blades).
The crowd reacts as a crowd: it bets. The post says the PC "DIES".

### (c) The GM reads an odd act charitably

**c1. "Heel!"** CRD3 C1E045 "Those Who Walk Away", turns 1099–1105 [1].
Galdric is a wolf. The party tried a ritual and a treat to make him join
Vax. The wolf ignored them and walked out. Grog acts.

> TRAVIS: No, I do, I take out Craven Edge and I sink it into the earth right beside his head. And I say, (shouts) heel!
> MATT: Make an animal handling check.
> TRAVIS: Stay.
> MATT: He sits.

*Shows:* the GM does not call an attack or an intimidation roll. He maps
the violent display to the nearest fitting skill. It works.

**c2. Pickpocket a giant.** CRD3 C1E075 "Where the Cards Fall", turns
401–405 [1]. Vax the rogue looks at a fire giant's belt pouch.

> LIAM: How high up is that side?
> MATT: It's about eight feet up.
> TRAVIS: Don't fucking-- don't even.
> MATT: Yeah, you'd have to leap for that, man. You're not pickpocketing a giant without some high rolls. Mind you, you can certainly try. I would love for you to try.

*Shows:* "you can try" comes with a fact from the world (eight feet) and a
cost (high rolls). The GM wants the odd act; another player objects.

**c3. A fireball shaped like a bell.** u/FrizzleMira (a player), r/DnD,
2019-06-05 [18].

> DM: You create a bell of fire that tolls loudly sending out a wave of radiant fire. Roll 8d6 fire damage. Any undead who fail their save are now considered turned. However, mark off a use of your channel divinity.
> Not at all how that should've worked. But it was awesome and fit with the arcane cleric idea.

*Shows:* rule of cool, but with a price (a used resource). The reason the
GM gives is that it fits the character.

**c4. The cleric converts the goblins.** u/DietCokeAndTuna (the GM), r/DnD,
2019-07-30 [19]. The GM invited his pastor to play. The pastor plays a
cleric. The first ambush starts.

> and the cleric... asks me if he could convert the Goblins to whatever religion was in the game.
> Not wanting to ruin the game for him, I said, "Sure. Give me a persuasion check followed by a religion check."

*Shows:* a non-violent act in a violent scene. The GM makes it hard (two
rolls), not impossible. The post says both rolls were natural 20s.

### (d) An NPC reacts against the obvious, but it makes sense later

**d1. "Kill me, give me honor."** CRD3 C1E025 "Crimson Diplomacy", turns
1765–1769 [1]. The last enemy soldier is held by magic. Grog wants to kill
him. Vax shows mercy.

> TRAVIS: Lillith, can I go play t-ball with his head?
> LIAM: No. Let him live. Let this one live.
> MATT: At which point the head lifts he goes  "No. Kill me, give me honor. I have fallen."
> LIAM: Oh, I'm sorry I retract that.

*Shows:* the obvious NPC act is to accept mercy. This NPC has its own
value (honour). The table accepts it at once.

**d2. The acolyte prays with the vandal.** CRD3 C2E016 "A Favor in Kind",
turns 2085–2102 [1][3]. Jester wants to put a mustache on a holy statue.
She fails her roll and an acolyte sees her.

> LAURA: Can I get in my disguise kit and affix a mustache onto her porcelain mask, and put a wig on her, and then write across the bottom of it "The Traveler was here," and leave a pamphlet?
> MATT: You can certainly try.
> LAURA: There was a smudge on her face, and I know that the people get really pissed when there's stuff on their statues. I was trying to clean it off for you.
> MATT: "Please do. Please I can pray with you if you'd like."

*Shows:* the obvious NPC act is to throw her out. This NPC is young,
polite and pious. He believes the lie and prays with her. The GM did not
ask for a deception roll.

**d3. Kaylie.** CRD3 C1E038 "Echoes of the Past", turns 863, 893, 924–925
[1]. Kaylie, a young bard, comes to Scanlan's room and pours wine. The
players read it as flirting or theft. She is his daughter, and she came to kill him.

> LAURA: She's going to rob you blind.
> MATT: ... "This day, Scanlan Shorthalt, I beat you at your own song. Now I come to take your dignity. Now take up your sword."
> SAM: I slowly unbutton my shirt and show her my chest and say, "Stab me right here, if you'd like, and I will not resist. You've earned it. Take me."
> MATT: You see the tear begin to roll down her cheek as her jaw clenches. (grunts) And she thrusts forward-- -- and rolls a natural one.

*Shows:* the table misread the NPC (a), and the NPC's real motive was
hidden but consistent. The GM rolled her attack. The natural 1 became "her
will falters" (turn 928). The dice chose between two acts that both fit her.

### (e) Personality shows through a choice

**e1. Grog pays.** CRD3 C1E085 "A Bard's Lament", turns 1357–1361 [1].
Vex is bargaining with a merchant.

> TRAVIS: I know how to do this. I take a bag of gold--
> LAURA: No, Grog! Grog, Grog, Grog.
> TRAVIS: -- pour it onto the desk. Just a pile of gold.
> LAURA: Grog, do not do this. Please, please, please.

*Shows:* the obvious act is to let the skilled bargainer speak. Grog uses
his own method (force, or here a pile of money). The table predicts it and
objects before it lands.

**e2. The druid becomes a woodworker.** Tablee5 (the GM), D&D Beyond
forums, 2019-05-16 [15].

> they encounter the Redbrands at the tap room.
> Also, he decided to take a job as a woodworker there. Idk why, but he started running a very successful business and did not finish the rest of the campaign.

*Shows:* a character follows its own leaning, not the plot. The GM lets
the world absorb it.

### (f) A character acts on a wrong belief

**f1. The fake deck.** CRD3 C1E082 "Deadly Echoes", turn 757, and C1E086
"Daring Days", turns 2352–2387 [1]. Vex secretly gave Grog a pouch "with
rocks in it" in place of the deck. Later she confesses.

> LAURA: ... I gave you a false deck.
> TRAVIS: No, no, you gave it to me, it's in my bag.
> ...
> MATT: It's a lot of words. It's a lot of words. But you got your deck back.
> TRAVIS: But I always had it.

*Shows:* the belief survives the truth because Grog failed his insight
check (he rolled 6). The GM: "She's talking at you, but nothing changed."
A low-intelligence character keeps a false belief in a way the table finds
both funny and right.

**f2. Bad information.** CRD3 C2E026 "Found & Lost", turns 2148–2149 and
3044–3046 [1]. Keg (a PC played by a guest) told the party about their
enemy Lorenzo. She believed it.

> MARISHA: Lorenzo is a fighter type?
> ASHLEY: Yes.
> ...
> SAM: He can cast?! I thought you said he was a fighter! Your information is the worst!
> ASHLEY: I'm not very smart!

*Shows:* a plan built on a true-seeming but wrong belief. The ambush fails.
The source of the error is a character, so the error has a reason.

**f3. The safe raft.** FIREBALL, file
`fbf652feb458b3cc1a375c6c5f76815e.jsonl`, lines 57–58 [6].

> As your action, you slowly navigate your raft to the bank on the left side of the map. You think it's safe there.
> They ran off the map to hide.
> The party thinks the raft is safely lodged up against the rocks.
> dr4 attacks with a Light Crossbow! dr4 attacked Thyme (Tammy) and crit!

*Shows:* the GM states the party's belief in the narration ("You think
it's safe"), and the game state shows the hidden truth. Belief and fact
are separate, and the gap creates the scene.

**f4. The man who could not lie.** u/MrGumberculese (a player), r/DnD,
2019-06-04 [20]. The game was a White Wolf game, not D&D. The player gave
his character a visible "tell" when lying. For the first time, the
character lies to a politician.

> The DM looks up at me: Wait is that true?
> Me: No, Roy just lied, I did the shifty eye thing he always does when he lies.
> DM: Well I wasn't paying attention and didn't see it, so neither does he. He has no reason to believe you aren't telling the truth.

*Shows:* the GM decides what the NPC believes from what the NPC could
perceive. The lie works. The player says it "went against what everyone
was expecting".

### (g) More verified lines (short)

These support Sections 4 and 5. All are verified against the raw source.

| id | source | line | what it shows |
| --- | --- | --- | --- |
| g1 | CRD3 C1E025, turn 1297 | LIAM: "I thought that I could get a peeky-poo at what they actually were, who they were, and let's face it, I fucked up is what I did." | Vax scouted alone and was captured. He explains his own mistake. |
| g2 | CRD3 C2E015, turns 445–447 | LIAM: "... I'm going to fire at where I think this thing is." / LAURA: "Oh no." / SAM: "I'm dead." | Caleb is blind and fires anyway. All three rays hit the monster. |
| g3 | CRD3 C2E035, turns 2262–2269 | MATT: "No. How it works is, you give them a task. He went and hid under the table for safety, which he did." ... "But it wasn't specific enough." | An NPC under a magic order does exactly what it was told, for six seconds. |
| g4 | CRD3 C1E025, turns 1180–1181 | LAURA: "I thought she was disguised." / MATT: "That's right. Never mind, I retract that entire sequence." | The GM's own error. He retcons it openly. |
| g5 | CRD3 C1E082, turn 194 | MATT: "... Correcting my mistake from the card last session." | The GM misread a card. He fixes it inside the story (a curse). |
| g6 | CRD3 C1E085, turns 2495–2510 | TRAVIS: "Well, I thought you were being taken over." | Vax makes a hand-shadow "dragon". Grog crushes his hands. A misbelief that fits Grog. |
| g7 | CRD3 C1E005, turns 399–403 | MATT: "You can certainly try. There's plates that have been broken and cracked." | Grog cuts armour from a dead monster. It works: "three saucer-shaped pieces". |
| g8 | CRD3 C1E081, turns 887–891; C1E115, turn 1792 | MATT: "Make a slight of hand check, Grog." / TRAVIS: "Shit, that's a natural 1!" ... MATT: "Give me the card. Let this be a lesson to everybody out there." | Grog tries to steal the deck and fails. 34 episodes later he draws "Void" and loses his soul. |
| g9 | CRD3 C1E010, turns 1927–1928 | MARISHA: "No, I'll wait. I thought we were next to a tree." / MATT: "No, most of the forest starts further up this incline ..." | A player's wrong map, fixed in one sentence. |
| g10 | CRD3 C1E069, turns 1344, 1358 | MATT: "... I come to you as an ally." ... "There were a few holes in Asum's story when he first got here nobody picked up on." | An enemy dragon in disguise offers an alliance. The clues were there. |
| g11 | FIREBALL `e9ef444f8149386d323c0a04cc0c3d59.jsonl`, lines 15, 18 | "dasyes 60 ft in wrng direction is now 100 ft away" / "moves 30 ft but in wrong direction and is now 30 ft away" | A blinded NPC moves the wrong way. The state records "Blinded (Sunbeam)". |
| g12 | FIREBALL `196f767932c2384af7de08fc6b27bb2f.jsonl`, line 22 | "*The man picks you up and lays  you on a bench, performing a quick battlefield patch up.* "Dont worry kid. You aint fyin and yet."" | An NPC wins the fight, then spares and heals the PC. |
| g13 | FIREBALL `34cb54bbf4effd21220f993fa5ad94f8.jsonl`, lines 30, 34 | "(dont move back unless you take the "disengage action" ..." / "(i'll count it as a BA" | A new player breaks a rule. Someone, probably the GM, rules in the player's favour. |
| g14 | D&D Beyond forums, Cardinal_Fang, 2019-04-08 [15] | "He asks if he can eat one of the shards (thinking it to be candy [?]), so I, the DM, say "yes, but roll a d6"." ... "So, I had the NPC cleric administer the Heimlich." | A drunk dwarf eats a glowing rock. "Yes, but roll". When no PC helps, an NPC does. |
| g15 | dndadventure.com compilation [14] | "The troll looks on and shouts 'SEND FORTH YOUR BEST RIDDLER!'" ... "The troll splats him dead. 'I hate riddlers.'" | An NPC with a surprising but simple motive. Second-hand, undated. |
| g16 | Steuer, "The Head of Vecna", 1998 [21] | "I knew that there was no way he could CUT HIS OWN HEAD OFF. Alas I was mistaken" | A planted false rumour. PCs act on it. The GM did not expect it. |
| g17 | u/M4dmaddy, r/DnD, 2019-06-24 [22] | "the high priest sends a messenger to the city in which the seat of the church is, to convince them to urge the king to declare holy war on the cultists." | A PC's lie spreads through NPCs and grows. |
| g18 | u/kareth117 (the GM), r/DnD, 2019-07-28 [23] | "After a NAT 1 on my insight behind the screen and a NAT 20 on the wood elf's deception" | A bank clerk believes an old note. The dice set her belief. |

## 3. What makes a scene good: principles

Each principle has a source and, where we could, a short quote checked
against the raw page. "For us" says how it applies to NPC choice.

**Scene structure**

1. **Prepare situations, not plots.** Know who wants what and with which
   means; then react. "Don't prep plots, prep situations." [24] *For us:*
   each NPC is a set of wants and means. The menu is the means.
2. **Start with an event.** "The key driver for the strong start is
   "something happens."" [25] *For us:* each night starts with an event
   (a stranger enters, a purse is missing), so each NPC has something to
   react to.
3. **Pace in waves.** Left 4 Dead's AI director: "Constant, unchanging
   combat is fatiguing"; "Long periods of inactivity are boring"; "If
   intensity is too high, remove major threats for awhile". [26] *For us:*
   one tension number can re-weight the menu (eat and talk when calm,
   confront when high), with no model call.
4. **Keep a default activity, react briefly, then return.** Façade: "Beat
   mix-in jdbs are beat-specific reactions used to respond to player
   actions and connect the interaction back to the canonical sequence."
   [27] *For us:* the cook keeps serving. An odd event gets a short
   reaction, then the work goes on. This keeps the scene continuous.
5. **The world has momentum.** GMs interviewed by Acharya et al. describe a
   world where "the world and its agents are headed in a particular
   direction"; "both player action and inaction have consequences". GMs
   decide results "by thinking through the logic of the world". Novice GMs
   "struggled more with managing unexpected player actions". [58] *For
   us:* the simulation is the "logic of the world". NPC plans move forward
   when nobody acts.
6. **Let the NPCs decide, and play to find out.** Apocalypse World: "Play
   to find out what happens."; "Sometimes, disclaim decision-making."; "Put
   it in your NPCs' hands." [28] *For us:* the narrator interprets, but the
   NPC's pick and the simulation decide what happens.

**Characters**

7. **Every named character wants something.** "What Do They Want? In method
   acting you'll hear "what's my motivation?"" [29] "Name every person"
   [30]. Colville, as summarised by a reader: "Always know what the bad
   guys want" [56] **(notes by a third person, not Colville's own
   words)**. *For us:* 1 to 3 explicit wants for each NPC; score the menu
   against them.
8. **Model passions, not whole people, and make them visible.** "Traditional
   media narrative models passions, not people."; "PASSIONS are TRANSPARENT
   to the PLAYER" [31]. *For us:* a few visible passions (the bandit hates
   the knight; the merchant guards his box).
9. **The approach is where character lives.** The Angry GM: "the Approach is
   the part of the action where all of the role-playing lives"; "Outcomes
   come from Intentions and Consequences come from Approaches." [32] *For
   us:* a menu item is an (intent, approach) pair. The thief lies and the
   knight confesses: same intent, different character, different
   consequences.
10. **Characters sometimes act against their own interest, at a cost.**
    Blades in the Dark: "they don't always act in their own best
    interests" [33]. *For us:* allow picks from nature, not from the best
    score, but make each one cost something in the world. Then it reads as
    character, not as a bug.
11. **Norms are usually kept, can always be broken, and a breach is
    noticed.** Versu (Evans): "any character can violate the norms at any
    moment – it's just that NPCs should not typically want to do so"; "if a
    character does violate a norm, it should be noticed." [34] *For us:*
    the inn evening is a social practice (order, pay, wait your turn).
    Norm-breaking acts stay in the menu at low weight. A breach is an event
    that others see (compare b3, the bar fight).
12. **A social move has an intent; the other side accepts or rejects by
    rules.** Prom Week: "Volition is scored by counting the weight of the
    true influence rules."; "If the sum is zero or greater, the game
    responder accepts the intent of the game." [35] *For us:* NPCs that are
    not spoken to can accept or reject by rule alone, with no model call.
13. **A refusal has a named reason.** The Angry GM: "Objections are the
    reasons an NPC wants to not help the party. Incentives are the reasons
    the NPC wants to help the party." [36] Players should learn "why that
    conflict happened — from the NPC's perspective" [37]. *For us:* store
    each NPC's objections; refusals name them.

**Mistakes and misunderstandings**

14. **How a character reads another's act is itself a choice.** Versu:
    "replies to a particular action are not automatic"; "a trivial initial
    remark can snowball into something dramatic". [34] *For us:* a
    misunderstanding is a choice among readings (insult, joke, ignore),
    weighted by personality. Others then judge the reading.
15. **Wrong beliefs need interesting causes, and the system keeps the
    truth.** "Character knowledge may be inaccurate, but for reasons that
    may in and of themselves be interesting" [38]; in Talk of the Town,
    "Characters are not consciously aware of transferences, confabulations,
    or mutations" [39]. *For us:* each NPC has a belief store apart from
    world facts. Each wrong belief records its cause (did not see, was told,
    was lied to).
16. **Uncertainty should be a few states that the player can see.** Thief
    (Leonard): "The result of their work must be perceivable and
    understandable by the player." [40] *For us:* suspicion as unaware →
    suspicious → certain. Each level opens different acts and lines.
17. **A mistake needs an explanation that the audience can see.** Riedl and
    Stern: "inexplicable goal failure results in a loss of agent
    believability"; "failing believably" means to "enact a plausible and
    believable explanation for goal failure". [59] *For us:* every
    deviation from the obvious act must be traceable to a belief, a trait
    or a perception, and the narration can show it.
18. **Human errors come in kinds.** James Reason's model (summary slides,
    not the book): slips are "incorrect execution of a planned action";
    lapses are a forgotten step; rule-based mistakes come from "picking an
    inappropriate rule", caused by "misconstrued view of state, over-zealous
    pattern matching, frequency gambling"; knowledge-based mistakes come
    from "incomplete/inaccurate understanding". [60] **(secondary source)**
    *For us:* this is a menu of *understandable* mistakes. Slip = dice.
    Lapse = memory decay. Rule-based = a habit used in the wrong place.
    Knowledge-based = a wrong belief.
19. **Use the character's nature to cover what the system cannot handle.**
    Façade: "By design, Grace and Trip are self-absorbed, allowing them to
    occasionally believably ignore unrecognized or unhandleable player
    actions." [27] *For us:* when the parse fails, the drunk peasant
    mishears, the busy cook waves you off.

**How the GM reads odd actions**

20. **Treat an odd act as meant, and build on it.** Tina Fey: "respect what
    your partner has created"; "THERE ARE NO MISTAKES, only
    opportunities." [41] *For us:* when the model picks a strange act, the
    narrator gives it the motive that best serves the scene.
21. **Add to the simulation's story; never contradict it.** Emily Short on
    James Ryan: "allowing a human curator to build on (but not contradict)
    the stories produced by a simulation" [42]; curation "is greatly
    assisted by the simulation maintaining extensive records" [43]. The
    Angry GM: "Never contradict a Consequence once it exists." [32] *For
    us:* narration adds motive and colour; facts come only from logged
    world changes.
22. **Be permissive about the approach, strict about the result.** "lean on
    the permissive side" [44]; "You don't want to create a world in which
    any action can bring about any goal." [32] *For us:* the narrator picks
    the kindest reading; the legality check still decides the result.
23. **Expect people to miss things.** Three Clue Rule: "the PCs will
    probably miss the first; ignore the second; and misinterpret the
    third"; if all fail, "the next part of the bad guy's plan happens".
    [44] *For us:* key facts should reach the player through 3 or more
    NPCs or objects. When the scene stalls, the bandit acts.
24. **Failure must change the situation; certain acts just succeed.** "Failure
    is meaningful (and interesting) when it creates an obstacle or
    consequences, and therefore requires the creation of a new path." [45]
    "If the action can fail, but failure carries no risk or cost … THE
    ACTION SUCCEEDS." [32] *For us:* an NPC slip (spilled ale, a whisper
    overheard) creates new state that others can act on.
25. **Rulings, not rules; and rule of cool.** "you don't use a rule; you
    make a ruling." [53] Rule of cool, as quoted from TV Tropes by a blog:
    "the limit of the willing suspension of disbelief … directly
    proportional to its awesomeness" [54] **(secondary quote; TV Tropes
    returned 403)**. Colville, as summarised: "It never hurts to give the
    players the benefit of the doubt" [56] **(third-person notes)**.
26. **Never show the list.** Dungeon World: "You never show the players that
    you're picking a move from a list."; "Be a fan of the characters". [30]
    *For us:* narrate picks as story ("Bran shoves back his stool"), never
    as option names.

**Legibility: what the player can see**

27. **Only what the player perceives counts.** Tynan Sylvester: "In play,
    the Game Model is irrelevant. Players can't perceive it directly.";
    "Story-Richness : The percentage of interactions in a game that are
    interesting to the player."; "Choose the minimum representation that
    supports the kinds of stories you want to generate". Of Ultima
    Online's hidden ecology: "literally no-one ever noticed". [46] *For us:*
    every hidden belief or motive needs a visible sign.
28. **Hidden complexity adds nothing.** Wardrip-Fruin's "Tale-Spin effect"
    is "what happens when the system's complex internal processes are never
    apparent to the audience". [47] *For us:* a belief model is wasted
    unless NPCs visibly act or speak on their false beliefs.
29. **Say the intention out loud; explain inaction with a line.** F.E.A.R.:
    "We also use dialogue to explain a lack of action."; "Vocalizing
    intentions can sometimes even be enough" [48]. *For us:* template lines
    for "can't" and "won't" ("Not with the knight watching") cost no model
    call.
30. **Players fill the gaps.** Tarn Adams: "people just fill in the blanks"
    [49]. Will Wright: "The player's imagination and the simulator are
    together weaving this thing" [50]. RimWorld: "RimWorld is a story
    generator." [51] *For us:* a small, clear signal is enough. The player
    makes the story from it.
31. **The player experience has several parts.** Liapis and Denisova name
    six: cognitive challenge, immersion, agency, attachment, group dynamics
    and refereeing. "Another common reason for dysfunction is a lack of
    shared mental model of the game state and gameworld ... between
    players, which leads to confusion and arguments." [57] *For us:* misreadings (Section 4.1) are a group-dynamics cost when
    they are the *player's* error, and a story gain when they are the
    *character's*.
32. **NPCs must react, not only pursue goals.** Colville: "What we really
    mean is they're not reacting." [52] *For us:* react to salient events
    first, then return to the plan.

Robin's Laws of Good Game Mastering: we could not open the book. A
secondary page quotes Laws on player "traits" as "the particular emotional
impulses that give your players a sense of reward" [55]. **(not verified
against the book)**

## 4. Patterns in the excerpts

### 4.1 How often do mistakes and misreadings happen?

We counted word patterns in 17 Critical Role episodes from CRD3: C1E005,
010, 015, 025, 035, 045, 055, 065, 075, 085, 095, 105 and C2E005, 015, 025,
035, 045. This is a rough count. We did not read each episode. We count
MATT as the GM and every other speaker (also guests) as a player.

| measure (17 episodes, 44,046 turns) | count |
| --- | --- |
| player turns | 31,555 |
| player turns that end with "?" | 6,571 (21%) |
| player turns with "I thought", "we thought", "I meant", "misheard", "what do you mean" | 145 |
| of a hand-checked sample of 48 of these, real misreadings of the scene or of game state | 28 (58%) |
| estimated spoken misreadings for each episode (an episode is about 3 to 4 hours) | about 5 (a lower limit) |
| GM replies just after a player question that start with "Yes", "Yeah", "Sure", "You can", "You do" ... | 469 |
| GM replies just after a player question that start with "No", "Nope", "Not" | 197 |
| GM says "you can (certainly) try" | 19 |
| GM says "go for it" | 113 |
| a player declares an act and the next player objects ("No!", "Don't", "Oh no"): pattern matches | 27 |
| of these, real objections to an act (we read all 27) | 16 |

What the 28 real misreadings were about:

| about | count | example |
| --- | --- | --- |
| where a thing or a person is | 5 | "I thought we were next to a tree." |
| the state of an object or an NPC | 6 | "I thought it was cracked or something." |
| a plan, or words between players | 6 | "I meant convince him to come inside." |
| time, rest, turn order | 3 | "I thought we rested." |
| numbers and rules | 3 | "13? I thought that said 93." |
| world history | 2 | "Is it only 800? I thought it was 8,000." |
| the character's own wrong belief | 2 | "I thought you were being taken over." |
| the GM's own error | 1 | "I thought she was disguised." |

**Findings.**

1. **Misreadings are common.** One player turn in five ends with a
   question mark.
   Players build the picture of the scene by asking. About 5 spoken
   misreadings in each episode is a lower limit, because most misreadings
   have no marker phrase.
2. **Most misreadings are the player's, not the character's.** They are
   about where, when and what state. The GM fixes them fast and out of
   character, in one sentence (a2, g9). The story does not change because
   of them.
3. **Character misbeliefs are rarer, and the table keeps them.** Grog's
   fake deck (f1), Keg's wrong report (f2), Grog's "taken over" (g6). They
   are played out, not corrected.
4. **"Stupid" acts are rarer still, but they are what people remember.**
   Other players objected to a declared act 16 times in 17 episodes. Half
   of the objections (8) were to two characters: Grog, and the guest
   character Twiggy (4 in her one episode). The Deck of Many Things alone
   is a 34-episode thread.
5. **The GM says yes-type words more than twice as often as no-type
   words** (469 to 197) when answering a question. Many "no" answers
   correct a fact ("No, most of the forest starts further up"). We did
   not count how many.
6. **Forum stories are biased.** Reddit and forum stories (a4, b3, c3, c4,
   e2, f4) are chosen because something strange happened. In recorded
   play, most turns are ordinary: move, attack, heal, ask.

### 4.2 What makes a mistake feel understandable, not random?

1. **It follows from what the character perceived or was told.** Caleb is
   blind and fires "at where I think this thing is" (g2). Beau did not
   know the room was so large (a2). Keg believed what she said (f2).
2. **It follows from the character's nature.** Grog solves problems with
   force or gold (c1, e1). Twiggy is bold and naive (b2). The table
   predicts these mistakes before they happen ("Grog, do not do this").
   A mistake that others can predict is not random.
3. **It is a good habit in the wrong place.** Grog uses on a wolf the
   threat that works on enemies (c1). The rogue thinks of pickpocketing a
   giant (c2). The half-orc brings a sword to a bar fight (b3). Reason
   calls this a rule-based mistake: "picking an inappropriate rule" [60].
4. **It is visible before it lands.** Someone objects or warns: "Travis,
   you could destroy the world" (b1); "No! Twiggy! Don't!" (b2); the GM
   warned the wizard about the demons (a4). The warning tells the
   audience that this is a choice, not a bug.
5. **It has a reason after it lands.** Vax explains himself (g1). The
   cleric's ring rule explains the disaster (a4). Riedl and Stern: failure
   without an explanation loses believability [59].
6. **The player's ignorance matches the character's.** Travis does not know
   the deck, and the table keeps it that way (b1). Text players write their
   character's ignorance on purpose (a3).

Mistakes that do *not* feel understandable: a character forgets what it
said a minute ago; a character acts on facts it could not know; an act
serves no goal and no trait.

### 4.3 How does the GM turn an odd choice into a good scene?

| GM move | excerpts | what the GM does |
| --- | --- | --- |
| Allow the attempt, name the obstacle, attach a roll | c2, c4, g7, g14 | "You can certainly try" + a hard fact ("eight feet up") + a roll. |
| Map the act to the nearest rule | c1, c3 | A violent display becomes an animal handling check. A bell of fire is paid for with a resource. |
| Resolve as far as the world allows | a2 | Beau reaches the door, not the target. |
| Hold the character to its exact words | b1, g3 | "You said you were just pulling one card." The NPC hides for six seconds because the order said no more. |
| Decide NPC belief from NPC perception | f4, d2, g18 | The politician did not see the tell, so he believes. The acolyte believes the excuse. |
| Fix the player's picture fast | a2, g9 | One sentence of fact, then play on. |
| Take back the GM's own error | g4, g5 | "Never mind, I retract that entire sequence." Or fix it inside the story. |
| Let the impulsive act land with full weight | b2, g8 | "Twiggy's gone." Grog draws "Void". |
| Give an NPC a reaction that makes a disaster into story | a4, g12, g14 | The villain hires the cleric. An NPC spares the PC. An NPC does the Heimlich. |
| Give an NPC its own values | d1, d2, d3, g15 | "Kill me, give me honor." The acolyte prays. Kaylie cannot strike. |
| Let the dice choose between two acts that both fit | d3, c4, g18 | Kaylie's natural 1 turns an attack into an embrace. |

The gazebo (a1) is the counter-example. The GM got frustrated and punished
the player for a misreading of a word. Modern advice says to fix the
player's picture instead (principles 20, 23).

### 4.4 How do consistency and surprise balance?

- **The character stays fixed; the situation changes.** Grog acts like Grog
  in every episode. The surprise is *which* situation meets his nature. The
  table knows the trait and still does not know the act.
- **Facts are fixed; choices and dice give surprise.** The GM never changes
  a known fact to make a scene fun. He corrects his own errors openly
  (g4, g5). Surprise comes from choices, rolls and hidden facts that were
  there all along (Kaylie's motive, the clues to Raishan, g10).
- **Most mistakes are small.** Most misreadings cost seconds. Big mistakes
  (Vax captured, Twiggy pulled into the glass, the party falling) are rare,
  and each comes from a nature or a belief.
- **The obvious act is the default.** Most turns are obvious. A scene where
  every character acts against the obvious would feel random. A scene
  where nobody ever does would feel scripted.
- **The odd act gets a fair chance, not a sure one.** "You can certainly
  try" with a cost. Two natural 20s convert the goblins (c4); a natural 1
  ruins Grog's theft (g8). The dice make surprise fair.

## 5. What a game simulation needs to make such scenes

**The short answer.** Most of what made the excerpts work is a *fact* that
a simulation can store and check: who saw what, who believes what, who
lied, what each character's habits are, how far a body can move, what an
order said. A smaller part needs *judgement over open text*: what an odd
act means, if it fits the character, if it is funny or harmful, and the
words of the narration.

### 5.1 Element by element

| element seen in real play | excerpts | facts the simulation can track | needs judgement over open text |
| --- | --- | --- | --- |
| Perception limits cause wrong acts | g2, g11, a2, a3 | Line of sight, light, sound, distance, conditions ("blinded"). The character acts on what it perceives, not on the truth. | Only the words of the narration. |
| Each character has its own beliefs, true or false | f1, f2, f3, g17 | A belief record for each character: the claim, its source (saw it; told by X), and whether it is true. A lie is a fact: who said it, to whom, and if the listener believed it. | Is a *new* lie plausible to this listener? What does an excuse ("a smudge on her face") mean? |
| The character knows less than the player | b1, a3 | Knowledge is per character. The model that chooses for a character sees only that character's beliefs. | None. This is a rule about inputs. |
| A habit used in the wrong place | c1, c2, b3, e1 | A small habit table for each character: situation type → preferred family of acts (force, money, charm, flight). The situation type comes from perception, so a wrong reading gives the wrong habit. | Does a new, specific act fit the habit in an unusual situation? |
| Execution slips | g8 | Skill and dice. | None. |
| Resolve as far as the world allows | a2 | Movement, reach, time. Partial results are facts. | None. |
| An NPC follows an order to the letter | g3 | An order is a structured fact with a scope and a duration. The NPC does the minimum it was told. | Turning a spoken order into that structure. |
| The NPC has values of its own | d1, d2, d3 | Values and norms as data (honour, piety, fear, family), weighted for each character. | The line of dialogue. |
| Others react before a risky act | e1, b2, c2 | Other characters perceive the *start* of an act and react from their own drives. This shows the act is a choice, not a bug. | The words of the warning. |
| The NPC's belief comes from the NPC's perception | f4, d2, g18 | Who could see the tell. Insight against deception as dice plus traits. | Judging an excuse that has no ready rule. |
| The GM maps an odd act to the nearest rule | c1, c3 | If the act is in the menu, there is nothing to map. | For free text from the *player*: find the nearest legal act and its roll. This is the main judgement task. |
| The GM says yes, with a cost | c2, c4, g7 | The difficulty comes from world facts (height, armour, size). | Choosing *which* fact to tell the player. |
| Explain the failure after it lands | g1, a4 | A cause log: belief X, trait Y and perception Z led to act A. The narrator reads it. | Writing the explanation. |
| Player misreading versus character misbelief | g9 versus g6 | In a computer game, the *player's* misreading is an interface problem: make the screen clear. The *character's* misbelief lives in the simulation. Keep the two apart. | None. |
| Tone: is the stupid act a gift or a harm? | b2 ("It's so good.") versus a1 | Rate limits: how often each character deviates from its best act. Never against a hard rule of the scene. | Is this act funny, sad, or does it ruin the scene for others? |

### 5.2 Two lessons for our current problems

1. **"The model picks whatever the scene text names."** Real players do
   this too. Eric attacked the gazebo because the GM named it (a1).
   Players assume that a named thing matters. A GM solves this by naming
   what matters, and players ask questions (21% of player turns in our
   sample). For us: the scene text must not name inert things in the same
   way as actors and threats. The menu, not the prose, must carry what each
   character knows about each thing.
2. **"A personality sentence overrides the situation."** In real play a
   trait does not override facts. Grog does not attack a friend. He grabs
   Vax's hands only when he believes Vax is "being taken over" (g6), a
   misbelief from a hand-shadow trick. The trait chooses *among* the acts
   that the situation and the character's beliefs make possible. It
   changes *how* and *which*, not *whether the facts apply*. Louis and
   Sutton found the same in data: actions are predicted better when the
   character is taken into account [10]. Our reading: the character adds
   to the situation; it does not replace it.

### 5.3 A rough recipe from the excerpts

1. **Most turns:** the character takes the obvious act for its beliefs.
2. **Sometimes:** the character takes the act its habit prefers, even when
   another act scores higher. Others can predict it, and they can object.
3. **Sometimes:** the character's belief is wrong (poor perception, a lie,
   an old fact). The act is right for the belief and wrong for the world.
4. **Rarely:** a slip in execution (dice).
5. **Always:** other characters can see the act start and react. After the
   act, the cause log can explain it.
6. **The GM layer** (a narrator, or a rules step) answers every odd act
   with "you can try", takes the cost from world facts, and resolves as far
   as the world allows. It never changes a known fact to make the scene
   fun. It may change its *own* earlier error, openly.

## 6. References

We or a research helper opened each URL below during this review, unless
it is marked. Reddit posts were read through the arctic-shift archive API
(`https://arctic-shift.photon-reddit.com/api/posts/ids?ids=<id>`), because
reddit.com blocks scripts; the permalink is given.

**Datasets and transcripts**

1. CRD3 repository (README, LICENSE, cleaned data). https://github.com/RevanthRameshkumar/CRD3
2. Rameshkumar, R., Bailey, P. (2020). Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset. ACL. https://aclanthology.org/2020.acl-main.459/
3. Critical Role wiki, transcript of C2E016 via the API. https://criticalrole.fandom.com/api.php?action=parse&page=A_Favor_in_Kind/Transcript&prop=wikitext&format=json
4. CritRoleStats. https://www.critrolestats.com/
5. Zhu, A., Aggarwal, K., Feng, A., Martin, L. J., Callison-Burch, C. (2023). FIREBALL: A Dataset of Dungeons and Dragons Actual-Play with Structured Game State Information. ACL. https://arxiv.org/abs/2305.01528
6. FIREBALL dataset files. https://huggingface.co/datasets/lara-martin/FIREBALL ; code: https://github.com/zhudotexe/FIREBALL
7. Callison-Burch, C., Singh Tomar, G., Martin, L. J., Ippolito, D., Bailis, S., Reitter, D. (2022). Dungeons and Dragons as a Dialog Challenge for Artificial Intelligence. EMNLP. https://arxiv.org/abs/2210.07109
8. Zhou, P., Zhu, A., Hu, J., Pujara, J., Ren, X., Callison-Burch, C., Choi, Y., Ammanabrolu, P. (2023). I Cast Detect Thoughts: Learning to Converse and Guide with Intents and Theory-of-Mind in Dungeons and Dragons. ACL. https://arxiv.org/abs/2212.10060
9. Zhu, A., Martin, L., Head, A., Callison-Burch, C. (2023). CALYPSO: LLMs as Dungeon Masters' Assistants. AIIDE. https://arxiv.org/abs/2308.07540
10. Louis, A., Sutton, C. (2018). Deep Dungeons and Dragons: Learning Character-Action Interactions from Role-Playing Game Transcripts. NAACL. https://aclanthology.org/N18-2111/
11. Akoury, N., Wang, S., Whiting, J., Hood, S., Peng, N., Iyyer, M. (2020). STORIUM: A Dataset and Evaluation Platform for Machine-in-the-Loop Story Generation. EMNLP. https://aclanthology.org/2020.emnlp-main.525/
12. Dimension 20 wiki, Fan Content and Resources. https://dimension20.fandom.com/wiki/Fan_Content_and_Resources **(not verified: reported by a helper)**

**Stories of play**

13. Aronson, R. The Tale of Eric and the Dread Gazebo (© 1986), rec.humor.funny archive, July 1998. https://www.netfunny.com/rhf/jokes/98/Jul/gazebo.html
14. Gaming Stories (compilation, undated). https://www.dndadventure.com/html/articles/gaming_stories.html
15. D&D Beyond forums, "Funny D&D stories" thread (2019). https://www.dndbeyond.com/forums/dungeons-dragons-discussion/dungeon-masters-only/36498-funny-d-d-stories
16. u/Vexra (2019-08-13). A ring of feather falling just murdered my entire party. r/DnD. https://www.reddit.com/r/DnD/comments/cpxnzp/
17. u/Tyler9351 (2019-06-12). Never bring a sword to a bar fight. r/DnD. https://www.reddit.com/r/DnD/comments/bzroxx/
18. u/FrizzleMira (2019-06-05). r/DnD. https://www.reddit.com/r/DnD/comments/bwza11/
19. u/DietCokeAndTuna (2019-07-30). r/DnD. https://www.reddit.com/r/DnD/comments/cjqznt/
20. u/MrGumberculese (2019-06-04). Story Time - The Man Who Couldn't Lie & And Excellent DMing. r/DnD. https://www.reddit.com/r/DnD/comments/bwnnow/
21. Steuer, M. F. (1998). The Head of Vecna. https://www.rpglibrary.org/articles/storytelling/headofvecna.php
22. u/M4dmaddy (2019-06-24). r/DnD. https://www.reddit.com/r/DnD/comments/c4ky8v/
23. u/kareth117 (2019-07-28). r/DnD. https://www.reddit.com/r/DnD/comments/cixrgs/

**Game master advice**

24. Alexander, J. Don't Prep Plots. The Alexandrian. https://thealexandrian.net/wordpress/4147/roleplaying-games/dont-prep-plots
25. Shea, M. Eight Steps of the Lazy DM (2023). Sly Flourish. https://slyflourish.com/eight_steps_2023.html
26. Booth, M. (2009). The AI Systems of Left 4 Dead. https://steamcdn-a.akamaihd.net/apps/valve/2009/ai_systems_of_l4d_mike_booth.pdf
27. Mateas, M., Stern, A. (2005). Structuring Content in the Façade Interactive Drama Architecture. AIIDE. https://users.soe.ucsc.edu/~michaelm/publications/mateas-aiide2005.pdf
28. Baker, D. V., Baker, M. Apocalypse World 2nd ed., MC threats preview. http://apocalypse-world.com/AW2ndEdThreatsPreview.pdf
29. Shea, M. A New Method for NPCs. Sly Flourish. https://slyflourish.com/method_npcs.html
30. Dungeon World SRD, Gamemastering. https://www.dungeonworldsrd.com/gamemastering/
31. Levine, K. (2014). Narrative Legos. GDC. https://media.gdcvault.com/GDC2014/Presentations/Levine_Ken_Narrative_Legos.pdf
32. The Angry GM. Adjudicate Actions Like a Boss. https://theangrygm.com/adjudicate-actions-like-a-boss/
33. Blades in the Dark SRD, Action Roll. https://bladesinthedark.com/action-roll
34. Evans, R., on Emily Short's blog (2013). Versu: Conversation Implementation. https://emshort.blog/2013/02/26/versu-conversation-implementation/
35. McCoy, J., Treanor, M., Samuel, B., Mateas, M., Wardrip-Fruin, N. (2011). Prom Week: Social Physics as Gameplay. FDG. http://www.ben-samuel.com/wp-content/uploads/2015/09/FDG-2011-Prom-Week-Social-Physics-as-Gameplay.pdf
36. The Angry GM. Help! My Players Are Talking to Things! https://theangrygm.com/help-my-players-are-talking-to-things/
37. The Angry GM. Resolving Social Actions. https://theangrygm.com/resolving-social-actions/
38. Ryan, J. O., Summerville, A., Mateas, M., Wardrip-Fruin, N. Toward Characters Who Observe, Tell, Misremember, and Lie. https://eis.ucsc.edu/papers/ryanEtAl_TowardCharactersWhoObserveTellMisrememberLie.pdf
39. Ryan, J., Mateas, M. Simulating Character Knowledge Phenomena in Talk of the Town. Game AI Pro 3, ch. 37. https://www.gameaipro.com/GameAIPro3/GameAIPro3_Chapter37_Simulating_Character_Knowledge_Phenomena_in_Talk_of_the_Town.pdf
40. Leonard, T. Building an AI Sensory System: Examining the Design of Thief: The Dark Project. https://www.gamedeveloper.com/programming/building-an-ai-sensory-system-examining-the-design-of-i-thief-the-dark-project-i-
41. Fey, T. Tina Fey's Rules of Improvisation (from *Bossypants*). https://kicp-workshops.uchicago.edu/eo2014/pdf/Tina-Feys-rules-of-improv.pdf
42. Short, E. (2019). Curating Simulated Storyworlds (James Ryan), ch. 6. https://emshort.blog/2019/05/28/curating-simulated-storyworlds-james-ryan-ch-6f/
43. Ryan, J. Curating Simulated Storyworlds (excerpt). https://catn.decontextualize.com/public/ryan-curating-excerpt.pdf
44. Alexander, J. The Three Clue Rule. The Alexandrian. https://thealexandrian.net/wordpress/1118/roleplaying-games/three-clue-rule
45. Alexander, J. Failure for the Beginning GM. The Alexandrian. https://thealexandrian.net/wordpress/46186/roleplaying-games/failure-for-the-beginning-gm
46. Sylvester, T. The Simulation Dream. Game Developer. https://www.gamedeveloper.com/design/the-simulation-dream
47. Wardrip-Fruin, N. Expressive Processing: Interpretation and Creation. https://escholarship.org/content/qt5453x2sb/qt5453x2sb.pdf
48. Orkin, J. Three States and a Plan: The A.I. of F.E.A.R. https://www.gamedevs.org/uploads/three-states-plan-ai-of-fear.pdf
49. Interview: The Making of Dwarf Fortress (Tarn Adams). Game Developer. https://www.gamedeveloper.com/design/interview-the-making-of-dwarf-fortress
50. The Sims' Creator Shares His Secrets (Will Wright). App Store story. https://apps.apple.com/us/story/id1449205380
51. RimWorld official site. https://rimworldgame.com/
52. Colville, M. Essay in *Your Best Game Ever*. https://www.montecookgames.com/matt-colvilles-essay-from-your-best-game-ever/
53. Finch, M. A Quick Primer for Old School Gaming (mirror). https://friendorfoe.com/d/Old%20School%20Primer.pdf
54. 3 Wise DMs. Skills, Saves, the Impossible, and the Rule of Cool. https://3wisedms.com/skills-saves-the-impossible-and-the-rule-of-cool/ **(quotes TV Tropes, which we could not open)**
55. Critical Hits (2008). Robin's Laws Revisited, Part 2. https://critical-hits.com/blog/2008/01/23/robins-laws-revisited-part-2-player-types-and-traits/ **(secondary; we could not open Robin's Laws)**
56. Ansel, M. Notes on Matt Colville, "Running the Game". https://listed.to/@michaelansel/3838/matt-colville-running-the-game **(third-person notes; we could not check the videos)**

**Research on play and believable characters**

57. Liapis, A., Denisova, A. (2023). The Challenge of Evaluating Player Experience in Tabletop Role-Playing Games. FDG. https://antoniosliapis.com/articles/ttrpgpx.php
58. Acharya, D., Mateas, M., Wardrip-Fruin, N. (2021). Story Improvisation in Tabletop Roleplaying Games: Towards a Computational Assistant for Game Masters. IEEE CoG. https://ieee-cog.org/2021/assets/papers/paper_164.pdf
59. Riedl, M. O., Stern, A. (2006). Failing Believably: Toward Drama Management with Autonomous Actors in Interactive Narratives. TIDSE. https://faculty.cc.gatech.edu/~riedl/pubs/tidse06b.pdf
60. Brown, A. (2001). An Overview of Human Error, drawn from J. Reason, *Human Error* (1990). Course slides. http://roc.cs.berkeley.edu/294fall01/slides/human-error.pdf **(secondary source for Reason)**

**Could not open:** the TV Tropes "Rule of Cool" page; *Robin's Laws of Good
Game Mastering*; Matt Colville's videos; Tynan Sylvester's book *Designing
Games* and his GDC talk (we used his article [46]); the original
"Boatmurdered" thread; the Versu IEEE paper (we used [34]); Giant in the
Playground and RPG.net threads (Cloudflare blocks).
