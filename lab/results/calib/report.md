# Calibration of the decision models (KnowNo-style split conformal)

alpha 0.1: the target coverage is 90%. Fitted on one half of the labeled cases (by a hash of the id), judged on the other. A set is every option with p >= p_min; the model is SURE when the set has one option, else it abstains.

| source | n cal / test | T | p_min | coverage % | mean set | abstain % | acc when sure % | acc@1 % | ECE T=1 -> T | coverage over random halves: mean (sd, p5) % | T=1: coverage % / set / abstain % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gdecide mark1-mind-v2 | 3261 / 3125 | 0.8446 | 0.0289 | 90.2 | 6.298 | 77.0 | 81.2 | 61.7 | 0.219 -> 0.186 | 90.0 (0.6, 88.9) | 89.8 / 6.346 / 78.0 |
| gdecide v1 (wave 10 text) | 3261 / 3125 | 0.9667 | 0.0289 | 89.5 | 6.421 | 76.9 | 81.2 | 60.7 | 0.192 -> 0.183 | 90.0 (0.7, 88.8) | 89.5 / 6.542 / 77.1 |
| gdecide lab text | 3261 / 3125 | 1.2907 | 0.0202 | 89.9 | 11.272 | 92.1 | 22.7 | 53.0 | 0.282 -> 0.333 | 89.9 (0.7, 88.6) | 89.6 / 10.518 / 89.8 |
| gdecide1b (wave 10 text) | 3261 / 3125 | 0.7055 | 0.0230 | 89.5 | 8.547 | 91.8 | 58.4 | 50.9 | 0.246 -> 0.175 | 89.9 (0.8, 88.7) | 89.3 / 8.903 / 93.5 |
| laya (wave 10 text) | 3235 / 3088 | 3.3035 | 0.0231 | 90.3 | 10.53 | 94.1 | 71.4 | 41.0 | 0.186 -> 0.171 | 90.0 (0.8, 88.7) | 89.8 / 12.727 / 87.2 |
| gliclass (wave 10 text) | 3261 / 3125 | 10.7268 | 0.0268 | 89.7 | 12.14 | 88.3 | 11.8 | 34.7 | 0.653 -> 0.061 | 90.0 (0.7, 88.9) | 90.6 / 18.659 / 75.7 |
| von (wave 10 text) | 3261 / 3125 | 6.097 | 0.0174 | 89.2 | 12.705 | 99.4 | 31.6 | 33.6 | 0.308 -> 0.121 | 90.1 (0.7, 88.9) | 89.2 / 15.498 / 90.9 |
| vegaml (wave 10 text) | 3261 / 3125 | 1.2819 | 0.0238 | 90.1 | 12.13 | 99.7 | 0.0 | 28.8 | 0.080 -> 0.113 | 90.0 (0.7, 88.9) | 90.2 / 12.228 / 99.9 |

## Mondrian: one threshold for each menu size

Conformal coverage is promised on average over the cases; a big menu can fall short of it while a small one is over. A threshold for each menu-size class (fitted on that class's calibration cases) promises the coverage within each class.

| source | coverage % | mean set | abstain % | acc when sure % | coverage over random halves: mean (sd, p5) % |
|---|---|---|---|---|---|
| gdecide mark1-mind-v2 | 89.8 | 8.358 | 73.6 | 83.0 | 90.0 (0.8, 88.6) |
| gdecide v1 (wave 10 text) | 90.2 | 8.82 | 74.2 | 82.2 | 90.0 (0.8, 88.6) |
| gdecide lab text | 90.1 | 10.503 | 88.9 | 48.5 | 90.0 (0.8, 88.8) |
| gdecide1b (wave 10 text) | 89.6 | 11.025 | 89.0 | 66.8 | 90.0 (0.8, 88.7) |
| laya (wave 10 text) | 89.2 | 13.002 | 91.3 | 69.3 | 90.1 (0.7, 88.9) |
| gliclass (wave 10 text) | 90.2 | 17.29 | 77.1 | 57.1 | 90.0 (0.7, 88.8) |
| von (wave 10 text) | 89.0 | 16.183 | 92.0 | 67.3 | 90.1 (0.7, 88.9) |
| vegaml (wave 10 text) | 89.8 | 16.102 | 100.0 | 100.0 | 90.1 (0.8, 88.8) |

## By menu size: gdecide mark1-mind-v2

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 100.0 | 2.831 | 64.9 | 0.2891 | 87.0 / 1.065 / 9.1 |
| <= 8 | 830 | 88.3 | 4.336 | 64.8 | 0.0343 | 87.4 / 4.078 / 63.5 |
| <= 16 | 1792 | 96.3 | 7.671 | 81.5 | 0.0501 | 90.7 / 4.575 / 76.8 |
| <= 32 | 97 | 89.7 | 6.289 | 81.4 | 0.0302 | 89.7 / 6.01 / 81.4 |
| <= 64 | 240 | 66.2 | 4.692 | 85.8 | 0.0084 | 92.1 / 31.979 / 93.8 |
| > 64 | 89 | 42.7 | 4.281 | 80.9 | 0.0023 | 88.8 / 69.607 / 95.5 |

## Coverage by case set: gdecide mark1-mind-v2

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 74.6 |
| creative | 127 | 93.7 |
| lure | 113 | 97.4 |
| people | 593 | 90.7 |
| personality | 24 | 100.0 |
| personality2 | 77 | 100.0 |
| pipe | 427 | 93.4 |
| places | 541 | 94.6 |
| present | 234 | 78.2 |
| state | 348 | 80.5 |
| things | 582 | 91.6 |

## By menu size: gdecide v1 (wave 10 text)

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 98.7 | 2.766 | 64.9 | 0.2318 | 87.0 / 1.221 / 20.8 |
| <= 8 | 830 | 87.7 | 4.276 | 64.3 | 0.0359 | 87.1 / 3.947 / 62.2 |
| <= 16 | 1792 | 95.7 | 7.951 | 81.5 | 0.0466 | 91.5 / 5.188 / 78.0 |
| <= 32 | 97 | 89.7 | 6.887 | 81.4 | 0.0310 | 88.7 / 6.392 / 81.4 |
| <= 64 | 240 | 65.4 | 4.371 | 85.8 | 0.0085 | 93.3 / 34.417 / 94.6 |
| > 64 | 89 | 40.5 | 3.798 | 83.2 | 0.0031 | 87.6 / 67.584 / 94.4 |

## Coverage by case set: gdecide v1 (wave 10 text)

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 72.9 |
| creative | 127 | 93.7 |
| lure | 113 | 98.2 |
| people | 593 | 87.7 |
| personality | 24 | 100.0 |
| personality2 | 77 | 100.0 |
| pipe | 427 | 91.6 |
| places | 541 | 94.6 |
| present | 234 | 78.6 |
| state | 348 | 80.2 |
| things | 582 | 92.4 |

## By menu size: gdecide lab text

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 100.0 | 3.909 | 98.7 | 0.2437 | 90.9 / 1.416 / 37.7 |
| <= 8 | 830 | 85.8 | 6.678 | 84.1 | 0.0033 | 87.8 / 7.058 / 88.2 |
| <= 16 | 1792 | 96.1 | 13.993 | 94.8 | 0.0511 | 91.1 / 6.795 / 89.5 |
| <= 32 | 97 | 95.9 | 17.309 | 96.9 | 0.0311 | 90.7 / 8.124 / 91.8 |
| <= 64 | 240 | 70.0 | 9.221 | 98.8 | 0.0120 | 91.2 / 35.854 / 99.2 |
| > 64 | 89 | 41.6 | 4.652 | 84.3 | 0.0057 | 87.6 / 59.393 / 97.8 |

## Coverage by case set: gdecide lab text

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 83.0 |
| creative | 127 | 100.0 |
| lure | 113 | 0.0 |
| people | 593 | 94.8 |
| personality | 24 | 100.0 |
| personality2 | 77 | 100.0 |
| pipe | 427 | 96.2 |
| places | 541 | 96.3 |
| present | 234 | 80.8 |
| state | 348 | 85.1 |
| things | 582 | 95.0 |

## By menu size: gdecide1b (wave 10 text)

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 98.7 | 3.416 | 89.6 | 0.1499 | 93.5 / 1.974 / 59.7 |
| <= 8 | 830 | 89.2 | 5.717 | 87.4 | 0.0327 | 86.1 / 5.119 / 84.6 |
| <= 16 | 1792 | 95.7 | 10.555 | 94.3 | 0.0393 | 90.3 / 6.891 / 91.0 |
| <= 32 | 97 | 84.5 | 10.66 | 95.9 | 0.0163 | 90.7 / 15.485 / 96.9 |
| <= 64 | 240 | 62.1 | 6.021 | 89.6 | 0.0089 | 91.7 / 32.379 / 92.5 |
| > 64 | 89 | 40.5 | 3.461 | 85.4 | 0.0029 | 95.5 / 94.73 / 97.8 |

## Coverage by case set: gdecide1b (wave 10 text)

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 71.2 |
| creative | 127 | 96.1 |
| lure | 113 | 98.2 |
| people | 593 | 89.2 |
| personality | 24 | 100.0 |
| personality2 | 77 | 100.0 |
| pipe | 427 | 93.4 |
| places | 541 | 90.9 |
| present | 234 | 81.6 |
| state | 348 | 78.7 |
| things | 582 | 92.3 |

## By menu size: laya (wave 10 text)

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 100.0 | 4.0 | 100.0 | 0.2291 | 92.2 / 2.312 / 72.7 |
| <= 8 | 830 | 99.5 | 7.782 | 98.9 | 0.0947 | 87.7 / 4.548 / 88.4 |
| <= 16 | 1792 | 96.8 | 12.619 | 92.9 | 0.0416 | 90.1 / 9.11 / 92.9 |
| <= 32 | 97 | 76.3 | 15.247 | 89.7 | 0.0214 | 84.5 / 16.753 / 89.7 |
| <= 64 | 240 | 30.4 | 6.658 | 92.9 | 0.0080 | 90.0 / 53.583 / 96.2 |
| > 64 | 52 | 7.7 | 1.135 | 65.4 | 0.0045 | 84.6 / 103.596 / 94.2 |

## Coverage by case set: laya (wave 10 text)

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 55.9 |
| creative | 127 | 100.0 |
| lure | 113 | 100.0 |
| people | 558 | 93.4 |
| personality | 24 | 91.7 |
| personality2 | 77 | 97.4 |
| pipe | 427 | 96.7 |
| places | 539 | 88.3 |
| present | 234 | 59.8 |
| state | 348 | 98.9 |
| things | 582 | 90.2 |

## By menu size: gliclass (wave 10 text)

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 100.0 | 4.0 | 100.0 | 0.1016 | 89.6 / 2.597 / 53.2 |
| <= 8 | 830 | 100.0 | 8.0 | 100.0 | 0.0722 | 88.3 / 5.52 / 64.6 |
| <= 16 | 1792 | 100.0 | 16.0 | 100.0 | 0.0461 | 91.1 / 13.305 / 82.0 |
| <= 32 | 97 | 76.3 | 20.495 | 62.9 | 0.0266 | 95.9 / 27.845 / 86.6 |
| <= 64 | 240 | 10.8 | 1.0 | 0.0 | 0.0144 | 89.6 / 55.075 / 85.8 |
| > 64 | 89 | 4.5 | 1.0 | 0.0 | 0.0075 | 85.4 / 106.596 / 83.2 |

## Coverage by case set: gliclass (wave 10 text)

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 64.4 |
| creative | 127 | 100.0 |
| lure | 113 | 100.0 |
| people | 593 | 88.0 |
| personality | 24 | 100.0 |
| personality2 | 77 | 100.0 |
| pipe | 427 | 100.0 |
| places | 541 | 88.0 |
| present | 234 | 50.8 |
| state | 348 | 100.0 |
| things | 582 | 91.4 |

## By menu size: von (wave 10 text)

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 100.0 | 4.0 | 100.0 | 0.2359 | 88.3 / 2.649 / 96.1 |
| <= 8 | 830 | 99.2 | 7.722 | 100.0 | 0.1036 | 88.5 / 5.513 / 100.0 |
| <= 16 | 1792 | 95.1 | 13.985 | 100.0 | 0.0295 | 89.3 / 11.289 / 86.2 |
| <= 32 | 97 | 82.5 | 24.608 | 95.9 | 0.0112 | 91.8 / 27.753 / 100.0 |
| <= 64 | 240 | 41.7 | 22.113 | 98.3 | 0.0087 | 89.6 / 56.354 / 100.0 |
| > 64 | 89 | 5.6 | 2.607 | 87.6 | 0.0045 | 84.3 / 105.0 / 100.0 |

## Coverage by case set: von (wave 10 text)

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 61.0 |
| creative | 127 | 97.6 |
| lure | 113 | 100.0 |
| people | 593 | 83.1 |
| personality | 24 | 100.0 |
| personality2 | 77 | 98.7 |
| pipe | 427 | 97.9 |
| places | 541 | 86.0 |
| present | 234 | 67.5 |
| state | 348 | 100.0 |
| things | 582 | 91.8 |

## By menu size: vegaml (wave 10 text)

| options | n test | coverage % | mean set | abstain % | Mondrian p_min | Mondrian: coverage % / mean set / abstain % |
|---|---|---|---|---|---|---|
| <= 4 | 77 | 98.7 | 3.948 | 100.0 | 0.1414 | 92.2 / 3.26 / 98.7 |
| <= 8 | 830 | 100.0 | 7.89 | 100.0 | 0.0953 | 89.3 / 4.995 / 100.0 |
| <= 16 | 1792 | 96.9 | 14.665 | 100.0 | 0.0383 | 89.2 / 12.269 / 100.0 |
| <= 32 | 97 | 69.1 | 17.278 | 100.0 | 0.0103 | 94.8 / 26.773 / 100.0 |
| <= 64 | 240 | 40.0 | 10.708 | 100.0 | 0.0029 | 92.1 / 47.958 / 100.0 |
| > 64 | 89 | 12.4 | 5.91 | 91.0 | 0.0010 | 92.1 / 110.427 / 100.0 |

## Coverage by case set: vegaml (wave 10 text)

| set | n test | coverage % |
|---|---|---|
| ablate | 59 | 57.6 |
| creative | 127 | 100.0 |
| lure | 113 | 97.4 |
| people | 593 | 81.8 |
| personality | 24 | 100.0 |
| personality2 | 77 | 100.0 |
| pipe | 427 | 99.3 |
| places | 541 | 90.9 |
| present | 234 | 70.9 |
| state | 348 | 100.0 |
| things | 582 | 91.1 |
