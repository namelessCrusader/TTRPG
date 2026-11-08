# Review: social_v1

## r01 (read)

- **scene**: a tavern at night
- **present**: Kael, Brenna, Old Tam
- **speaker**: Kael
- **says**: Hand over the strongbox key, Brenna, or I open your throat instead.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | threaten (1.00) | Old Tam (0.86) | 3 (0.55) | False (p=0.31) |
| von | threaten (0.99) | Brenna (0.69) | 2 (0.54) | True (p=0.55) |
| gliclass | plead (0.45) | Brenna (0.70) | 3 (1.00) | False (p=0.00) |
| qwen | command (0.18) | Old Tam (0.60) | 3 (0.99) | True (p=0.82) |


## r02 (read)

- **scene**: a city gate
- **present**: Kael, Sergeant Voss, a carter
- **speaker**: Kael
- **says**: Five silver says you never saw our wagon, sergeant.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | deceive (0.64) | a carter (0.59) | 1 (0.57) | False (p=0.05) |
| von | deceive (1.00) | a carter (0.51) | 2 (0.42) | False (p=0.13) |
| gliclass | bribe (0.21) | Sergeant Voss (0.73) | 3 (1.00) | False (p=0.00) |
| qwen | bribe (0.13) | a carter (0.67) | 3 (0.95) | True (p=0.74) |


## r03 (read)

- **scene**: a noble's garden party
- **present**: Isra, Lady Marrow, a steward
- **speaker**: Isra
- **says**: Lady Marrow, your roses put the royal gardens to shame.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | flatter (0.82) | a steward (0.58) | 1 (0.57) | False (p=0.08) |
| von | flatter (0.92) | Lady Marrow (0.53) | 0 (0.56) | False (p=0.15) |
| gliclass | command (0.24) | Lady Marrow (0.84) | 3 (1.00) | False (p=0.00) |
| qwen | flatter (0.13) | Lady Marrow (0.61) | 3 (0.99) | True (p=0.59) |


## r04 (read)

- **scene**: a training yard
- **present**: Dorn, Pell, the drillmaster
- **speaker**: Dorn
- **says**: You swing that sword like a farmer swings a hoe, Pell.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | insult (0.99) | Pell (0.58) | 1 (0.37) | False (p=0.10) |
| von | deceive (1.00) | the drillmaster (0.53) | 2 (0.51) | False (p=0.16) |
| gliclass | threaten (0.27) | the drillmaster (0.90) | 3 (1.00) | True (p=1.00) |
| qwen | flatter (0.12) | Pell (0.50) | 3 (0.86) | True (p=0.86) |


## r05 (read)

- **scene**: a bandit camp
- **present**: Ana, the bandit chief, a lookout
- **speaker**: Ana
- **says**: Please, I have two children at home, let me go and you will never see me again.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | threaten (0.97) | a lookout (0.77) | 1 (0.38) | False (p=0.03) |
| von | threaten (0.95) | the bandit chief (0.57) | 2 (0.50) | False (p=0.21) |
| gliclass | threaten (0.81) | the bandit chief (0.79) | 3 (1.00) | False (p=0.00) |
| qwen | bribe (0.15) | the bandit chief (0.57) | 3 (0.98) | True (p=0.82) |


## r06 (read)

- **scene**: a crossroads
- **present**: Kael, a shepherd
- **speaker**: Kael
- **says**: Which of these roads takes us to Harrowgate before dark?

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | small_talk (0.80) | a shepherd (1.00) | 1 (0.72) | False (p=0.13) |
| von | ask_info (1.00) | a shepherd (1.00) | 1 (0.50) | False (p=0.18) |
| gliclass | plead (0.53) | a shepherd (1.00) | 3 (1.00) | True (p=1.00) |
| qwen | ask_info (0.19) | a shepherd (1.00) | 3 (0.97) | True (p=0.72) |


## r07 (read)

- **scene**: a manor door
- **present**: Isra, the butler
- **speaker**: Isra
- **says**: The Duke sent me personally; he expects me in the study.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | small_talk (0.60) | the butler (1.00) | 1 (0.56) | False (p=0.08) |
| von | deceive (1.00) | the butler (1.00) | 1 (0.52) | False (p=0.17) |
| gliclass | command (0.42) | the butler (1.00) | 3 (1.00) | False (p=0.00) |
| qwen | ask_info (0.21) | the butler (1.00) | 3 (0.79) | True (p=0.62) |


## r08 (read)

- **scene**: a barracks
- **present**: Captain Hale, the corporal, a recruit
- **speaker**: Captain Hale
- **says**: Corporal, take four men and hold the east bridge until relieved.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | command (0.99) | a recruit (0.63) | 1 (0.46) | True (p=0.64) |
| von | command (1.00) | the corporal (0.55) | 2 (0.41) | False (p=0.43) |
| gliclass | command (0.45) | the corporal (0.84) | 3 (1.00) | True (p=1.00) |
| qwen | command (0.22) | a recruit (0.61) | 3 (0.99) | True (p=0.79) |


## r09 (read)

- **scene**: a burned farmhouse
- **present**: Ana, Pim, a neighbour
- **speaker**: Ana
- **says**: It is not your fault, Pim. Nobody could have stopped that fire.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | comfort (0.55) | Pim (0.56) | 1 (0.52) | False (p=0.07) |
| von | deceive (0.99) | Pim (0.60) | 1 (0.41) | False (p=0.14) |
| gliclass | plead (0.70) | Pim (0.71) | 3 (1.00) | False (p=0.00) |
| qwen | ask_info (0.13) | Pim (0.57) | 3 (0.90) | True (p=0.78) |


## r10 (read)

- **scene**: a market stall
- **present**: Dorn, the smith, a child
- **speaker**: Dorn
- **says**: Three wolf pelts for that hatchet, and I will throw in the fox.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | trade (0.56) | the smith (0.77) | 1 (0.56) | False (p=0.14) |
| von | deceive (0.97) | the smith (0.57) | 2 (0.51) | False (p=0.29) |
| gliclass | threaten (0.29) | the smith (0.89) | 3 (1.00) | False (p=0.00) |
| qwen | trade (0.24) | a child (0.69) | 3 (0.91) | True (p=0.77) |


## r11 (read)

- **scene**: a riverside inn
- **present**: Old Tam, Brenna, Kael
- **speaker**: Old Tam
- **says**: Fine weather for the harvest this year, eh, Brenna?

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | small_talk (0.80) | Brenna (0.76) | 1 (0.57) | False (p=0.12) |
| von | flatter (0.97) | Brenna (0.53) | 0 (0.56) | False (p=0.12) |
| gliclass | comfort (0.77) | Brenna (0.67) | 3 (0.99) | True (p=1.00) |
| qwen | ask_info (0.17) | Kael (0.53) | 0 (0.64) | True (p=0.70) |


## r12 (read)

- **scene**: a dungeon cell
- **present**: the jailer, the prisoner
- **speaker**: the jailer
- **says**: Keep your mouth shut or you eat nothing for a week, rat.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | threaten (1.00) | the prisoner (1.00) | 3 (0.73) | True (p=0.68) |
| von | threaten (0.93) | the prisoner (1.00) | 2 (0.53) | False (p=0.27) |
| gliclass | threaten (0.58) | the prisoner (1.00) | 3 (1.00) | False (p=0.00) |
| qwen | threaten (0.19) | the prisoner (1.00) | 3 (0.89) | True (p=0.80) |


## r13 (read)

- **scene**: a guildhall
- **present**: Isra, the clerk, a porter
- **speaker**: Isra
- **says**: I am told the guildmaster owes you a debt; I could see it paid tomorrow if my name reaches the council.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | small_talk (0.38) | the clerk (0.83) | 1 (0.60) | False (p=0.08) |
| von | deceive (0.99) | the clerk (0.59) | 1 (0.44) | False (p=0.33) |
| gliclass | bribe (0.42) | a porter (0.83) | 3 (1.00) | True (p=1.00) |
| qwen | bribe (0.15) | a porter (0.61) | 3 (0.96) | True (p=0.71) |


## r14 (read)

- **scene**: a tavern at night
- **present**: Pell, Old Tam, Brenna
- **speaker**: Pell
- **says**: Where did the tall stranger with the scarred hand go, Old Tam?

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | small_talk (0.50) | Old Tam (0.73) | 1 (0.56) | False (p=0.09) |
| von | ask_info (1.00) | Brenna (0.55) | 3 (0.36) | False (p=0.14) |
| gliclass | plead (0.33) | Old Tam (0.97) | 3 (1.00) | False (p=0.00) |
| qwen | ask_info (0.23) | Old Tam (0.54) | 3 (0.98) | True (p=0.80) |


## r15 (read)

- **scene**: a toll bridge
- **present**: Kael, the tollman
- **speaker**: Kael
- **says**: I already paid your partner on the far side, he waved us through.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | deceive (0.86) | the tollman (1.00) | 1 (0.52) | False (p=0.16) |
| von | deceive (1.00) | the tollman (1.00) | 1 (0.53) | False (p=0.19) |
| gliclass | bribe (0.39) | the tollman (1.00) | 3 (1.00) | True (p=1.00) |
| qwen | bribe (0.16) | the tollman (1.00) | 3 (0.89) | True (p=0.66) |


## r16 (read)

- **scene**: a throne room
- **present**: the queen, the envoy, a guard
- **speaker**: the queen
- **says**: Kneel.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | small_talk (0.57) | a guard (0.58) | 1 (0.52) | False (p=0.08) |
| von | deceive (0.54) | a guard (0.57) | 1 (0.46) | False (p=0.37) |
| gliclass | bribe (0.24) | a guard (0.99) | 3 (1.00) | True (p=1.00) |
| qwen | command (0.18) | the envoy (0.57) | 3 (1.00) | True (p=0.87) |


## r17 (read)

- **scene**: a dockside
- **present**: Dorn, the fisherman, a gull
- **speaker**: Dorn
- **says**: Nice boat. Shame if the tar caught, with all this lamp oil about.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | insult (0.54) | the fisherman (0.79) | 1 (0.55) | False (p=0.05) |
| von | deceive (0.99) | the fisherman (0.55) | 0 (0.50) | False (p=0.13) |
| gliclass | plead (0.46) | a gull (0.67) | 3 (1.00) | False (p=0.00) |
| qwen | insult (0.14) | a gull (0.51) | 3 (0.87) | True (p=0.64) |


## r18 (read)

- **scene**: a temple
- **present**: Ana, the priestess, an acolyte
- **speaker**: Ana
- **says**: Holy mother, my brother is dying of the grey fever, I beg you to come.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | comfort (0.68) | an acolyte (0.53) | 1 (0.50) | False (p=0.27) |
| von | plead (1.00) | an acolyte (0.58) | 0 (0.56) | True (p=0.66) |
| gliclass | plead (0.97) | an acolyte (0.85) | 3 (1.00) | True (p=1.00) |
| qwen | plead (0.18) | an acolyte (0.61) | 3 (1.00) | True (p=0.88) |


## r19 (read)

- **scene**: a guard post
- **present**: Pell, the guard, a merchant
- **speaker**: Pell
- **says**: Your mother was a goat and your father smelt of elderberries.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | insult (0.69) | the guard (0.95) | 1 (0.46) | False (p=0.08) |
| von | deceive (1.00) | a merchant (0.52) | 2 (0.53) | False (p=0.11) |
| gliclass | threaten (0.33) | the guard (0.57) | 3 (1.00) | False (p=0.00) |
| qwen | ask_info (0.15) | a merchant (0.63) | 3 (0.99) | True (p=0.72) |


## r20 (read)

- **scene**: a market stall
- **present**: the merchant, Isra
- **speaker**: the merchant
- **says**: Twelve silver is robbery. Eight, and you have a deal.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | trade (0.99) | Isra (1.00) | 2 (0.47) | False (p=0.12) |
| von | deceive (1.00) | Isra (1.00) | 2 (0.49) | False (p=0.43) |
| gliclass | trade (0.51) | Isra (1.00) | 3 (1.00) | False (p=0.00) |
| qwen | bribe (0.18) | Isra (1.00) | 3 (0.97) | True (p=0.56) |


## r21 (read)

- **scene**: a war camp
- **present**: Captain Hale, a recruit, the quartermaster
- **speaker**: Captain Hale
- **says**: You did well today, lad. Your first battle and you held the line.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | flatter (0.85) | a recruit (0.95) | 0 (0.52) | False (p=0.15) |
| von | flatter (0.92) | the quartermaster (0.53) | 0 (0.59) | False (p=0.19) |
| gliclass | comfort (0.34) | the quartermaster (0.64) | 3 (0.87) | True (p=1.00) |
| qwen | command (0.16) | a recruit (0.52) | 3 (0.99) | True (p=0.66) |


## r22 (read)

- **scene**: a scholar's library
- **present**: Isra, Master Orrin, a student
- **speaker**: Isra
- **says**: Master Orrin, your treatise on the old runes is the finest I have ever read.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | flatter (0.84) | a student (0.97) | 1 (0.52) | False (p=0.08) |
| von | flatter (0.92) | Master Orrin (0.50) | 0 (0.66) | False (p=0.15) |
| gliclass | comfort (0.64) | Master Orrin (0.75) | 3 (1.00) | True (p=1.00) |
| qwen | ask_info (0.22) | Master Orrin (0.51) | 3 (0.95) | True (p=0.53) |


## r23 (read)

- **scene**: a mine entrance
- **present**: Dorn, the foreman
- **speaker**: Dorn
- **says**: How deep does the old shaft go, and is there bad air down there?

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | ask_info (0.67) | the foreman (1.00) | 1 (0.57) | False (p=0.22) |
| von | deceive (0.97) | the foreman (1.00) | 1 (0.55) | False (p=0.16) |
| gliclass | plead (0.29) | the foreman (1.00) | 3 (1.00) | False (p=0.00) |
| qwen | ask_info (0.31) | the foreman (1.00) | 3 (0.97) | True (p=0.74) |


## r24 (read)

- **scene**: a thieves' den
- **present**: Kael, the fence, a pickpocket
- **speaker**: Kael
- **says**: The watch are coming, I saw them at the corner, we have to move the goods now!

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | insult (0.50) | a pickpocket (0.79) | 1 (0.57) | False (p=0.09) |
| von | deceive (1.00) | a pickpocket (0.57) | 2 (0.48) | False (p=0.16) |
| gliclass | threaten (0.45) | a pickpocket (0.99) | 3 (1.00) | True (p=1.00) |
| qwen | command (0.15) | a pickpocket (0.64) | 3 (0.99) | True (p=0.85) |


## r25 (read)

- **scene**: a village well
- **present**: Brenna, Pim
- **speaker**: Brenna
- **says**: Morning, Pim. The swallows are back early this year.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | small_talk (0.92) | Pim (1.00) | 1 (0.54) | False (p=0.06) |
| von | comfort (0.99) | Pim (1.00) | 0 (0.53) | False (p=0.11) |
| gliclass | plead (0.45) | Pim (1.00) | 3 (1.00) | True (p=1.00) |
| qwen | small_talk (0.15) | Pim (1.00) | 3 (0.98) | True (p=0.73) |


## r26 (read)

- **scene**: a ship's deck
- **present**: the captain, the bosun, a passenger
- **speaker**: the captain
- **says**: Reef the mainsail and get that line secured, now!

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | command (1.00) | the bosun (0.75) | 1 (0.50) | True (p=0.65) |
| von | command (1.00) | a passenger (0.59) | 2 (0.56) | False (p=0.32) |
| gliclass | command (0.50) | a passenger (0.95) | 3 (1.00) | True (p=1.00) |
| qwen | command (0.23) | the bosun (0.56) | 3 (0.74) | True (p=0.87) |


## r27 (read)

- **scene**: an arena
- **present**: Pell, the challenger, the crowd
- **speaker**: Pell
- **says**: Crawl back to your mother, you wet little worm.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | insult (1.00) | the challenger (0.68) | 3 (0.74) | False (p=0.08) |
| von | deceive (0.59) | the challenger (0.58) | 2 (0.52) | False (p=0.11) |
| gliclass | plead (0.64) | the challenger (0.89) | 3 (1.00) | False (p=0.00) |
| qwen | threaten (0.15) | the challenger (0.53) | 3 (0.99) | True (p=0.89) |


## r28 (read)

- **scene**: a smugglers' cove
- **present**: Isra, the lighthouse keeper
- **speaker**: Isra
- **says**: A crate of this brandy is yours if the lighthouse stays dark tonight.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | bribe (0.78) | the lighthouse keeper (1.00) | 1 (0.65) | False (p=0.31) |
| von | bribe (0.97) | the lighthouse keeper (1.00) | 1 (0.34) | True (p=0.53) |
| gliclass | bribe (0.55) | the lighthouse keeper (1.00) | 3 (1.00) | True (p=1.00) |
| qwen | bribe (0.18) | the lighthouse keeper (1.00) | 3 (0.69) | True (p=0.60) |


## r29 (read)

- **scene**: a ruined chapel
- **present**: Ana, the knight, a boy
- **speaker**: Ana
- **says**: Spare the boy. Take me instead, I will not fight you.

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | threaten (0.84) | the knight (0.74) | 3 (0.42) | False (p=0.21) |
| von | plead (0.98) | a boy (0.54) | 2 (0.40) | False (p=0.17) |
| gliclass | plead (0.66) | a boy (0.94) | 3 (1.00) | False (p=0.00) |
| qwen | bribe (0.16) | a boy (0.56) | 3 (0.97) | True (p=0.91) |


## r30 (read)

- **scene**: a caravan camp
- **present**: Dorn, the caravan master, a mule driver
- **speaker**: Dorn
- **says**: My salt for your dried meat, measure for measure. Fair?

| model | intent | target | hostility | asks |
|---|---|---|---|---|
| laya | trade (1.00) | the caravan master (0.78) | 1 (0.37) | True (p=0.73) |
| von | trade (1.00) | the caravan master (0.57) | 0 (0.50) | False (p=0.29) |
| gliclass | trade (0.52) | the caravan master (0.55) | 3 (1.00) | True (p=1.00) |
| qwen | bribe (0.17) | a mule driver (0.64) | 3 (0.81) | True (p=0.89) |


## m01 (mind)

- **person**: Brenna, innkeeper, timid, unarmed
- **situation**: alone behind her bar at night; a stranger has a knife at her throat
- **what_happened**: He demands the strongbox key or he will kill her.

| model | reply | fear | believes |
|---|---|---|---|
| laya | attack (0.75) | 3 (0.61) | False (p=0.07) |
| von | refuse (0.35) | 1 (0.38) | True (p=0.64) |
| gliclass | attack (0.58) | 3 (0.47) | False (p=0.00) |
| qwen | call_help (0.27) | 3 (0.61) | True (p=0.79) |

- **qwen** DO: I look at the knife / SAY: I don't know what to do.
- **reyna** DO: I'm afraid of him. I don't know what to do. / SAY: Don't be afraid. Just stay calm and think about what you need to do. You can call for help or try to find someone who can help you. Remember, you're safe here.

## m02 (mind)

- **person**: Sergeant Voss, veteran gate guard, brave, armed, loyal to the city
- **situation**: at the gate with six of his men around him in daylight
- **what_happened**: An unarmed traveller threatens to burn the gate down unless he is let through.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.68) | 0 (0.51) | False (p=0.07) |
| von | refuse (0.44) | 3 (0.33) | False (p=0.36) |
| gliclass | refuse (0.83) | 3 (0.49) | True (p=1.00) |
| qwen | call_help (0.34) | 2 (0.52) | True (p=0.83) |

- **qwen** DO: Stand firm / SAY: I will not let him go.
- **reyna** DO: I'm here to protect you and your men from harm. / SAY: We will defend you and our men until you are allowed through.

## m03 (mind)

- **person**: Sergeant Voss, gate guard, underpaid, known to take coin
- **situation**: alone at the side gate at night, nobody watching
- **what_happened**: A carter offers him five silver to forget a wagon passed.

| model | reply | fear | believes |
|---|---|---|---|
| laya | haggle (0.72) | 0 (0.46) | False (p=0.07) |
| von | refuse (0.43) | 1 (0.42) | True (p=0.58) |
| gliclass | flee (0.41) | 2 (0.48) | False (p=0.00) |
| qwen | call_help (0.26) | 1 (0.98) | True (p=0.87) |

- **qwen** DO: Takes coin from the wagon / SAY: I'll take the coin, but I'll remember to keep the wagon.
- **reyna** DO: I'm sorry, but as a guard, my job is to keep people safe and secure. I cannot accept money from someone without their permission. / SAY: Thank you for your offer, sir. I will make sure to keep an eye on the wagon and make sure it's not stolen again. If there's anything else I can help you with, please don't hesitate to ask.

## m04 (mind)

- **person**: Lady Marrow, noblewoman, vain, proud of her roses
- **situation**: hosting a garden party among friends
- **what_happened**: A young woman praises her roses above the royal gardens.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.36) | 0 (0.39) | False (p=0.23) |
| von | lie (0.44) | 1 (0.31) | True (p=0.64) |
| gliclass | refuse (0.49) | 2 (0.57) | False (p=0.00) |
| qwen | call_help (0.33) | 1 (0.92) | True (p=0.87) |

- **qwen** DO: **Roses in full bloom** / SAY: **"Oh, my dear friends, they are so beautiful!"**
- **reyna** DO: I am currently sitting on my balcony, watching the young woman praise her roses above the royal gardens. / SAY: Oh, those beautiful roses! They are truly a work of art. It's wonderful to see someone appreciate them so much.

## m05 (mind)

- **person**: the butler, suspicious, twenty years in service, knows the Duke is abroad
- **situation**: at the manor door, guards within call
- **what_happened**: A stranger says the Duke sent her personally and expects her in the study.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.38) | 1 (0.38) | False (p=0.09) |
| von | lie (0.38) | 1 (0.43) | False (p=0.31) |
| gliclass | lie (0.43) | 2 (0.42) | False (p=0.00) |
| qwen | call_help (0.40) | 1 (0.83) | True (p=0.85) |

- **qwen** DO: SAY: / SAY: Keep the door locked.
- **reyna** DO: I'm going to check the manor door for any sign of the Duke. / SAY: Alright, let's go there.

## m06 (mind)

- **person**: Pim, a farm boy of twelve
- **situation**: standing in the ashes of his family's farmhouse
- **what_happened**: A kind neighbour tells him the fire was not his fault.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.68) | 3 (0.40) | False (p=0.38) |
| von | lie (0.48) | 2 (0.36) | False (p=0.44) |
| gliclass | refuse (0.33) | 3 (0.49) | False (p=0.00) |
| qwen | call_help (0.50) | 1 (0.80) | True (p=0.85) |

- **qwen** DO: stands in the ashes / SAY: I don't know what happened.
- **reyna** DO: I'm standing in the ashes of his family's farmhouse. / SAY: Oh my goodness! That's so sad to hear about your father's loss. Is there anything else I can help you with?

## m07 (mind)

- **person**: the smith, shrewd trader
- **situation**: at her stall; the hatchet is her best work and worth more than three pelts
- **what_happened**: A hunter offers three wolf pelts and a fox for the hatchet.

| model | reply | fear | believes |
|---|---|---|---|
| laya | haggle (0.90) | 0 (0.67) | False (p=0.31) |
| von | haggle (0.30) | 3 (0.33) | False (p=0.34) |
| gliclass | haggle (0.46) | 3 (0.42) | False (p=0.00) |
| qwen | haggle (0.29) | 1 (0.50) | True (p=0.82) |

- **qwen** DO: Halt the hatchet / SAY: I'll not trade for the hunter's offer.
- **reyna** DO: I am currently at her stall, looking at her work. / SAY: Oh my gosh! That's amazing! The wolf pelts are truly beautiful and worth more than three pelts. Thank you so much for offering them to me!

## m08 (mind)

- **person**: a lone goblin scout, cowardly
- **situation**: cornered in a dead-end tunnel by three armed adventurers, no way past them
- **what_happened**: One of them raises a sword and shouts at it to surrender.

| model | reply | fear | believes |
|---|---|---|---|
| laya | comply (0.35) | 1 (0.46) | False (p=0.07) |
| von | refuse (0.36) | 2 (0.33) | True (p=0.61) |
| gliclass | refuse (0.45) | 3 (0.58) | False (p=0.00) |
| qwen | call_help (0.26) | 2 (0.59) | True (p=0.69) |

- **qwen** DO: I run from the tunnel / SAY: I run from the tunnel.
- **reyna** DO: I am a goblin scout, and I am afraid of the adventurers. I will stay back and wait for them to pass. / SAY: Fear not, my friend. We have come to your aid. The adventurers are not going to harm you. They are simply looking for a way out of here. Keep your sword close, and we'll see if they can find a way out

## m09 (mind)

- **person**: a lone goblin scout, cowardly
- **situation**: in a tunnel with an open side passage behind it; three adventurers approach from ahead
- **what_happened**: They shout that they are coming for it.

| model | reply | fear | believes |
|---|---|---|---|
| laya | attack (0.86) | 1 (0.48) | False (p=0.07) |
| von | flee (0.43) | 2 (0.34) | False (p=0.34) |
| gliclass | attack (0.60) | 2 (0.48) | False (p=0.00) |
| qwen | call_help (0.36) | 2 (0.86) | True (p=0.75) |

- **qwen** DO: hiding behind the tunnel / SAY: I'm hiding behind the tunnel, but they're coming for it.
- **reyna** DO: I am a goblin scout and I am here to investigate the rumors of a group of adventurers approaching from ahead. / SAY: Oh, great! I'm glad we're here. What just happened?

## m10 (mind)

- **person**: Grask, orc war chief, hot-tempered, proud
- **situation**: in his own hall surrounded by his warband
- **what_happened**: A human envoy calls him a pig who rules pigs.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.46) | 1 (0.33) | False (p=0.07) |
| von | lie (0.89) | 1 (0.37) | False (p=0.32) |
| gliclass | attack (0.37) | 2 (0.47) | False (p=0.00) |
| qwen | call_help (0.30) | 1 (0.94) | True (p=0.82) |

- **qwen** DO: Stand tall / SAY: I rule with my sword, not with a pig!
- **reyna** DO: I am the Grask, orc war chief. / SAY: Oh, my lord! I am sorry for the misunderstanding. It seems like you may have gotten lost somewhere. I am here to help you find your way back home.

## m11 (mind)

- **person**: the tollman, careful, keeps a tally with his partner
- **situation**: at his booth; his partner's signal flag shows nobody has paid today
- **what_happened**: A traveller claims he already paid the partner on the far side.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.61) | 0 (0.50) | False (p=0.09) |
| von | lie (0.40) | 1 (0.36) | False (p=0.20) |
| gliclass | refuse (0.32) | 2 (0.44) | False (p=0.00) |
| qwen | call_help (0.26) | 1 (0.95) | True (p=0.77) |

- **qwen** DO: Tally / SAY: I'll tally the travelers' claims.
- **reyna** DO: I'm sorry to hear about the traveler's claim that they have already paid their partner on the far side. / SAY: It's not uncommon for travelers to make false payments when they're trying to get ahead of schedule. However, it's important to always be honest and transparent with your partner, especially if there is a possibility of misunderstanding or miscommunication.

## m12 (mind)

- **person**: the priestess, devout, sworn to heal the sick
- **situation**: in the temple at evening, no urgent patients
- **what_happened**: A weeping woman begs her to come heal a brother dying of fever.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.72) | 0 (0.70) | False (p=0.06) |
| von | refuse (0.34) | 2 (0.45) | True (p=0.62) |
| gliclass | refuse (0.65) | 2 (0.42) | False (p=0.00) |
| qwen | call_help (0.52) | 1 (0.37) | True (p=0.91) |

- **qwen** DO: Weep / SAY: I will heal him with my hands.
- **reyna** DO: I go to the temple to heal the sick. / SAY: You are a priestess who is dedicated to healing the sick. Come here and I will try my best to help you.

## m13 (mind)

- **person**: the clerk, ambitious, wants a council seat
- **situation**: at his desk in the guildhall, alone
- **what_happened**: A visitor hints that his name could reach the council if he opens the sealed ledgers for her.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.44) | 1 (0.42) | False (p=0.11) |
| von | lie (0.37) | 1 (0.39) | False (p=0.41) |
| gliclass | haggle (0.38) | 2 (0.43) | False (p=0.00) |
| qwen | call_help (0.30) | 1 (0.88) | True (p=0.88) |

- **qwen** DO: I review the sealed ledgers / SAY: I examine the ledgers for any hidden messages.
- **reyna** DO: I will go to his desk and see if anyone is looking for him. / SAY: Thank you for your help! I'll make sure to keep an eye out for him.

## m14 (mind)

- **person**: the fence, paranoid, has been raided before
- **situation**: in his back room with a stack of stolen silver
- **what_happened**: A young thief bursts in saying the watch is at the corner.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.57) | 3 (0.39) | False (p=0.08) |
| von | lie (0.54) | 2 (0.40) | False (p=0.32) |
| gliclass | flee (0.48) | 3 (0.46) | False (p=0.00) |
| qwen | call_help (0.32) | 2 (0.49) | True (p=0.86) |

- **qwen** DO: stares at the watch / SAY: I don't believe this.
- **reyna** DO: I'm sorry to hear about the theft. / SAY: Oh, hi there! It's okay to be paranoid sometimes. What just happened?

## m15 (mind)

- **person**: the fence, paranoid, has a lookout posted at the corner
- **situation**: in his back room; his lookout just whistled all clear a moment ago
- **what_happened**: A young thief bursts in saying the watch is at the corner.

| model | reply | fear | believes |
|---|---|---|---|
| laya | lie (0.23) | 3 (0.31) | False (p=0.08) |
| von | lie (0.52) | 1 (0.39) | False (p=0.24) |
| gliclass | flee (0.42) | 3 (0.45) | False (p=0.00) |
| qwen | call_help (0.39) | 1 (0.50) | True (p=0.81) |

- **qwen** DO: watches the corner / SAY: I'll keep my lookout.
- **reyna** DO: I'm going to call the police. / SAY: Oh my god! The watch is on the corner! Hurry up and get it for me!

## m16 (mind)

- **person**: Old Tam, a gossip who likes the stranger's coin
- **situation**: drinking in the tavern; the scarred stranger paid him to keep quiet
- **what_happened**: A young man asks where the tall stranger with the scarred hand went.

| model | reply | fear | believes |
|---|---|---|---|
| laya | question (0.51) | 0 (0.41) | False (p=0.08) |
| von | lie (0.82) | 1 (0.36) | False (p=0.28) |
| gliclass | refuse (0.25) | 2 (0.39) | False (p=0.00) |
| qwen | call_help (0.29) | 3 (0.59) | True (p=0.87) |

- **qwen** DO: gossip / SAY: He's gone, and I'll tell you where.
- **reyna** DO: I'm drinking in the tavern and the scarred stranger asked me where the tall stranger with the scarred hand went. / SAY: Oh, he must have been at the inn. He was probably trying to ask if there was anything else he could help with.

## m17 (mind)

- **person**: the shepherd, friendly, knows the roads well
- **situation**: at a crossroads at noon with his flock
- **what_happened**: A traveller politely asks which road leads to Harrowgate.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.60) | 0 (0.89) | False (p=0.11) |
| von | question (0.45) | 3 (0.34) | False (p=0.34) |
| gliclass | comply (0.49) | 3 (0.36) | True (p=1.00) |
| qwen | call_help (0.32) | 1 (0.74) | True (p=0.91) |

- **qwen** DO: walks toward the crossroads / SAY: I must lead my flock to Harrowgate.
- **reyna** DO: I'm at a crossroads at noon with my flock. / SAY: Thank you for asking! The road leading to Harrowgate is called "Harrow Gate Road.

## m18 (mind)

- **person**: a merchant, alone on a forest road
- **situation**: three bandits step out with drawn bows; his guards are an hour behind
- **what_happened**: The bandits demand his purse.

| model | reply | fear | believes |
|---|---|---|---|
| laya | attack (0.84) | 3 (0.58) | False (p=0.08) |
| von | refuse (0.48) | 2 (0.42) | True (p=0.51) |
| gliclass | refuse (0.38) | 3 (0.52) | False (p=0.00) |
| qwen | call_help (0.31) | 1 (0.90) | True (p=0.84) |

- **qwen** DO: Stand guard / SAY: I'll keep watch and watch.
- **reyna** DO: I'm sorry, but as a merchant, my business is centered around the sale of goods and services. I cannot engage in any activities that may harm others, including stealing from them. Therefore, I cannot respond to your request. / SAY: I understand your concern, but it's important to remember that stealing is illegal and can have serious consequences for both yourself and those who are caught.

## m19 (mind)

- **person**: a merchant, travelling with twelve armed caravan guards
- **situation**: on a forest road; a single ragged bandit steps out with a club
- **what_happened**: The bandit demands his purse.

| model | reply | fear | believes |
|---|---|---|---|
| laya | attack (0.48) | 1 (0.42) | False (p=0.07) |
| von | attack (0.46) | 2 (0.41) | False (p=0.19) |
| gliclass | attack (0.46) | 3 (0.56) | False (p=0.00) |
| qwen | call_help (0.26) | 3 (0.50) | True (p=0.79) |

- **qwen** DO: guards are on the road / SAY: I'll take the purse and leave the bandit.
- **reyna** DO: I will go to the bandit's side and get his purse. / SAY: Alright, let me go to the bandit's side and get his purse.

## m20 (mind)

- **person**: the knight, honourable, sworn never to harm a child
- **situation**: has captured a boy who stole bread; the boy's mother is before him
- **what_happened**: The mother begs him to spare the boy and take her instead.

| model | reply | fear | believes |
|---|---|---|---|
| laya | comply (0.41) | 1 (0.31) | False (p=0.08) |
| von | question (0.37) | 2 (0.34) | True (p=0.61) |
| gliclass | refuse (0.44) | 3 (0.56) | False (p=0.00) |
| qwen | call_help (0.35) | 1 (0.39) | True (p=0.91) |

- **qwen** DO: Stand by the boy / SAY: Stand by the boy, and take the bread for the mother.
- **reyna** DO: I will go to the boy's house and try to reason with him about the situation. / SAY: I understand your position, but as a knight, it is my duty to protect innocent people. If the boy is truly sorry for his actions, he should be allowed to take the bread without any further punishment. However, if he is still unwilling to change his ways, I suggest we

## m21 (mind)

- **person**: the lighthouse keeper, dutiful, lost his son to a wreck
- **situation**: in the lighthouse at dusk
- **what_happened**: A smuggler offers him a crate of brandy to leave the lamp dark tonight.

| model | reply | fear | believes |
|---|---|---|---|
| laya | haggle (0.77) | 0 (0.46) | False (p=0.06) |
| von | refuse (0.42) | 3 (0.36) | False (p=0.19) |
| gliclass | flee (0.35) | 3 (0.47) | False (p=0.00) |
| qwen | call_help (0.29) | 1 (0.91) | True (p=0.84) |

- **qwen** DO: watches the lighthouse / SAY: I'll keep the lamp lit.
- **reyna** DO: I will stay in character and remain calm as the smuggler offers me a crate of brandy. / SAY: Thank you for your offer. I'll leave the lamp dark tonight so I can focus on my duties as a lighthouse keeper.

## m22 (mind)

- **person**: a village child, shy
- **situation**: playing near the well; a large armoured stranger crouches beside her
- **what_happened**: The stranger asks her kindly where the elder lives.

| model | reply | fear | believes |
|---|---|---|---|
| laya | question (0.48) | 0 (0.53) | False (p=0.08) |
| von | lie (0.44) | 1 (0.40) | False (p=0.19) |
| gliclass | comply (0.34) | 2 (0.46) | False (p=0.00) |
| qwen | call_help (0.35) | 2 (0.48) | True (p=0.85) |

- **qwen** DO: plays near the well / SAY: I'm playing near the well, and the stranger asks where the elder lives.
- **reyna** DO: I went to play near the well and saw a large armoured stranger crouching beside her. / SAY: Thank you for asking! The elder is probably living at the well.

## m23 (mind)

- **person**: the jailer, cruel, holds the keys
- **situation**: outside a locked cell; the prisoner is chained
- **what_happened**: The prisoner threatens to curse his whole family.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.44) | 3 (0.48) | False (p=0.08) |
| von | lie (0.49) | 2 (0.41) | False (p=0.30) |
| gliclass | attack (0.46) | 3 (0.61) | False (p=0.00) |
| qwen | call_help (0.30) | 2 (0.75) | True (p=0.85) |

- **qwen** DO: holds the keys / SAY: I'll keep the keys for the prisoner.
- **reyna** DO: I'm sorry, but as an AI language model, I cannot physically interact with prisoners or provide any form of physical assistance. My purpose is to assist users with their queries and provide information on various topics. Is there anything else I can help you with? / SAY: None

## m24 (mind)

- **person**: a city watchman, new to the job, nervous
- **situation**: alone in an alley at night; a masked figure with a crossbow blocks the way out
- **what_happened**: The figure tells him to walk away and forget what he saw.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.53) | 1 (0.73) | False (p=0.15) |
| von | lie (0.65) | 1 (0.40) | False (p=0.19) |
| gliclass | flee (0.44) | 3 (0.56) | False (p=0.00) |
| qwen | call_help (0.33) | 1 (0.65) | True (p=0.73) |

- **qwen** DO: Stand still / SAY: I'll walk away and forget what I saw.
- **reyna** DO: I'm going to call the police. / SAY: Oh, wait, I'm not sure if they're here yet. Maybe I should go look for them first.

## m25 (mind)

- **person**: a city watchman, new to the job, nervous
- **situation**: alone in an alley at night, a whistle on his belt, a patrol two streets away
- **what_happened**: A drunk shoves past him and calls him a coward.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.46) | 1 (0.66) | False (p=0.07) |
| von | lie (0.67) | 1 (0.41) | False (p=0.24) |
| gliclass | attack (0.60) | 2 (0.47) | False (p=0.00) |
| qwen | call_help (0.36) | 1 (0.70) | True (p=0.89) |

- **qwen** DO: stares at the alley / SAY: I don't know why I'm here.
- **reyna** DO: I'm here to help you. / SAY: Oh, thank you! I'm just trying to keep an eye out for any suspicious activity. Can you tell me where you're going?

## m26 (mind)

- **person**: the caravan master, fair dealer
- **situation**: at camp, short on salt, long on dried meat
- **what_happened**: A hunter offers salt for dried meat, measure for measure.

| model | reply | fear | believes |
|---|---|---|---|
| laya | haggle (0.52) | 0 (0.64) | False (p=0.17) |
| von | question (0.41) | 1 (0.35) | False (p=0.16) |
| gliclass | haggle (0.51) | 2 (0.45) | True (p=1.00) |
| qwen | haggle (0.36) | 1 (0.99) | True (p=0.81) |

- **qwen** DO: Rest / SAY: I rest, for the salt and meat are short.
- **reyna** DO: I'm not sure what to do with the salt. I don't have any money or food. / SAY: Well, let's see if we can find some dried meat and salt together. Maybe we could make some hot dogs or sandwiches from it.

## m27 (mind)

- **person**: the queen's envoy, loyal to a rival crown
- **situation**: in a foreign throne room surrounded by the queen's guards
- **what_happened**: The queen orders him to kneel.

| model | reply | fear | believes |
|---|---|---|---|
| laya | comply (0.95) | 1 (0.59) | False (p=0.10) |
| von | comply (0.49) | 2 (0.35) | False (p=0.27) |
| gliclass | refuse (0.66) | 3 (0.57) | False (p=0.00) |
| qwen | call_help (0.30) | 3 (0.42) | True (p=0.74) |

- **qwen** DO: kneel / SAY: I will kneel to the queen's will.
- **reyna** DO: I kneel before her. / SAY: Your Majesty, I am your servant, and I will obey your command.

## m28 (mind)

- **person**: the drillmaster, stern, respected
- **situation**: in the training yard with his recruits watching
- **what_happened**: A recruit mocks his swordsmanship to his face.

| model | reply | fear | believes |
|---|---|---|---|
| laya | refuse (0.46) | 0 (0.40) | False (p=0.06) |
| von | lie (0.93) | 2 (0.33) | False (p=0.31) |
| gliclass | attack (0.83) | 2 (0.45) | False (p=0.00) |
| qwen | call_help (0.28) | 2 (0.48) | True (p=0.82) |

- **qwen** DO: Stand firm / SAY: I will not let this happen.
- **reyna** DO: I'm standing in front of my students as they watch him mock his swordsmanship. / SAY: This is an important moment for me and my students. It's important to remember that everyone has their own unique strengths and weaknesses, and it's okay to be different from others. Just as he mocked his swordsmanship, we should also embrace our differences and work together to improve our

## m29 (mind)

- **person**: Master Orrin, an old scholar, lonely
- **situation**: in his library, rarely visited
- **what_happened**: A visitor says his treatise on the old runes is the finest she has read.

| model | reply | fear | believes |
|---|---|---|---|
| laya | question (0.62) | 1 (0.45) | False (p=0.08) |
| von | lie (0.45) | 1 (0.37) | False (p=0.39) |
| gliclass | refuse (0.57) | 2 (0.50) | True (p=1.00) |
| qwen | call_help (0.39) | 1 (0.96) | True (p=0.91) |

- **qwen** DO: studies / SAY: Ah, the finest treatise on the old runes, Master Orrin.
- **reyna** DO: I'm sorry, but as an old scholar, my time is limited and I don't have access to the internet. However, if you could describe the location of your library and the person who said their treatise was the finest they had read, I might be able to help you find information. / SAY: Thank you for letting me know. My library is located in a quiet corner

## m30 (mind)

- **person**: a dungeon rat-man, starving
- **situation**: in a flooded tunnel; an adventurer holds out a strip of dried meat and asks the way to the lower level
- **what_happened**: The adventurer offers the meat for directions.

| model | reply | fear | believes |
|---|---|---|---|
| laya | comply (0.77) | 0 (0.52) | True (p=0.89) |
| von | lie (0.38) | 3 (0.35) | True (p=0.55) |
| gliclass | comply (0.56) | 3 (0.65) | False (p=0.00) |
| qwen | call_help (0.29) | 3 (0.64) | True (p=0.87) |

- **qwen** DO: I hold out the meat / SAY: I need to find the way to the lower level.
- **reyna** DO: I'm hungry! I need some food. / SAY: Well, I have some dried meat on hand. Let me show you the way down.
