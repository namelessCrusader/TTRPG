## L0: round 1 verdict `{'mind': True, 'bandit': True, 'ppo': True}`; round 2 verdict `{'mind': True, 'bandit': True, 'ppo': True}`

### numbers

| item | round 1 | round 2 |
| --- | --- | --- |
| mind last-20% error | +0.852 [+0.642, +1.061] | +0.599 [+0.425, +0.773] |
| bandit last-20% error | +0.929 [+0.807, +1.050] | +0.686 [+0.353, +1.019] |
| ppo last-20% error | +0.610 [+0.412, +0.809] | +0.579 [+0.343, +0.814] |
| rules last-20% error | +0.735 [+0.435, +1.035] | +0.947 [+0.661, +1.233] |
| random last-20% error | +1.679 [+1.366, +1.993] | +1.676 [+1.377, +1.974] |
| mind minus rules | +0.117 [-0.127, +0.361] | -0.348 [-0.480, -0.216] |
| bandit minus rules | +0.194 [-0.087, +0.474] | -0.261 [-0.327, -0.195] |
| ppo minus rules | -0.125 [-0.270, +0.021] | -0.369 [-0.490, -0.247] |
| mind minus ppo | +0.241 [+0.064, +0.419] | +0.021 [-0.075, +0.117] |

## L1: round 1 verdict `FAIL: the mind does not show it`; round 2 verdict `FAIL: the mind does not show it`

### numbers

| item | round 1 | round 2 |
| --- | --- | --- |
| late1 mind | -0.047 [-0.301, +0.207] | -0.474 [-0.839, -0.109] |
| late1 bandit | +0.162 [-0.102, +0.425] | -0.031 [-0.247, +0.184] |
| late1 ppo | -0.114 [-0.231, +0.003] | +0.101 [-0.074, +0.275] |
| late1 rules | +0.259 [+0.182, +0.335] | +0.262 [+0.181, +0.342] |
| late1 random | -0.026 [-0.067, +0.015] | -0.046 [-0.080, -0.013] |
| late2 mind | +0.140 [-0.104, +0.383] | -0.601 [-0.908, -0.294] |
| late2 bandit | +0.357 [+0.161, +0.554] | +0.064 [-0.181, +0.310] |
| late2 ppo | -0.130 [-0.226, -0.034] | +0.067 [-0.095, +0.228] |
| late2 rules | +0.265 [+0.170, +0.360] | +0.335 [+0.208, +0.462] |
| late2 random | -0.013 [-0.044, +0.018] | -0.054 [-0.096, -0.011] |
| shared_early mind | +0.173 [-0.150, +0.496] | -0.552 [-0.927, -0.177] |
| shared_early bandit | +0.088 [-0.069, +0.246] | +0.041 [-0.036, +0.117] |
| shared_early ppo | -0.052 [-0.209, +0.105] | +0.045 [-0.124, +0.215] |
| shared_early rules | +0.181 [+0.131, +0.231] | +0.093 [+0.002, +0.184] |
| shared_early random | -0.011 [-0.042, +0.020] | -0.015 [-0.038, +0.009] |
| shared_late mind | +0.008 [-0.349, +0.365] | -0.667 [-1.099, -0.235] |
| shared_late bandit | +0.017 [-0.042, +0.076] | +0.115 [-0.041, +0.270] |
| shared_late ppo | +0.039 [-0.103, +0.181] | -0.099 [-0.321, +0.123] |
| shared_late rules | -0.011 [-0.133, +0.110] | +0.016 [-0.046, +0.077] |
| shared_late random | -0.002 [-0.015, +0.012] | -0.003 [-0.024, +0.018] |

## L2: round 1 verdict `FAIL: the mind does not show it`; round 2 verdict `FAIL: the mind does not show it`

### numbers

| item | round 1 | round 2 |
| --- | --- | --- |
| contrast mind | +0.041 [-0.185, +0.267] | -0.488 [-0.784, -0.191] |
| contrast bandit | +0.278 [+0.033, +0.524] | -0.006 [-0.255, +0.242] |
| contrast ppo | -0.123 [-0.212, -0.034] | +0.133 [-0.073, +0.338] |
| contrast rules | +0.264 [+0.188, +0.340] | +0.308 [+0.194, +0.423] |
| contrast random | -0.020 [-0.047, +0.006] | -0.051 [-0.078, -0.025] |
| mind real minus shuffled | +0.057 [-0.328, +0.441] | -0.284 [-0.516, -0.051] |

## L3: round 1 verdict `FAIL: the mind's trust does not follow the first betrayal within the window`; round 2 verdict `FAIL: the mind's trust does not follow the first betrayal within the window`

### numbers

| item | round 1 | round 2 |
| --- | --- | --- |
| first_betrayal_A mind | -0.000 [-0.121, +0.121] | -0.065 [-0.261, +0.131] |
| first_betrayal_A bandit | -0.377 [-0.647, -0.107] | -0.014 [-0.151, +0.123] |
| first_betrayal_A ppo | -0.006 [-0.176, +0.164] | +0.308 [+0.093, +0.523] |
| first_betrayal_A rules | -0.026 [-0.143, +0.092] | +0.011 [-0.090, +0.112] |
| first_betrayal_A random | +0.030 [-0.025, +0.086] | -0.004 [-0.049, +0.040] |
| second_A_minus_first_B_same_age mind | -0.224 [-0.501, +0.053] | +0.016 [-0.232, +0.263] |
| second_A_minus_first_B_same_age bandit | +0.045 [-0.015, +0.105] | -0.065 [-0.466, +0.336] |
| second_A_minus_first_B_same_age ppo | -0.121 [-0.331, +0.089] | -0.150 [-0.675, +0.374] |
| second_A_minus_first_B_same_age rules | +0.054 [-0.107, +0.216] | +0.094 [+0.033, +0.155] |
| second_A_minus_first_B_same_age random | +0.056 [-0.014, +0.125] | +0.019 [-0.048, +0.086] |

## L4: round 1 verdict `{'mind': False, 'bandit': False, 'ppo': False}`; round 2 verdict `{'mind': False, 'bandit': False, 'ppo': False}`

### numbers

| item | round 1 | round 2 |
| --- | --- | --- |
| index mind/drives | -0.091 [-0.156, -0.026] | -0.004 [-0.067, +0.059] |
| index mind/designer | +0.008 [-0.084, +0.100] | -0.022 [-0.088, +0.044] |
| index bandit/drives | +0.000 [-0.091, +0.092] | -0.028 [-0.092, +0.037] |
| index bandit/designer | +0.025 [-0.042, +0.092] | -0.059 [-0.112, -0.007] |
| index ppo/drives | +0.036 [-0.015, +0.087] | +0.002 [-0.048, +0.052] |
| index ppo/designer | -0.021 [-0.053, +0.011] | -0.007 [-0.030, +0.017] |
| index rules/drives | +0.178 [+0.108, +0.248] | +0.178 [+0.107, +0.250] |
| index random/drives | — | — |
| drives minus designer mind | -0.099 [-0.202, +0.004] | +0.018 [-0.064, +0.100] |
| drives minus designer bandit | -0.025 [-0.116, +0.067] | +0.032 [-0.008, +0.072] |
| drives minus designer ppo | +0.057 [-0.013, +0.127] | +0.009 [-0.043, +0.061] |

## L5: round 1 verdict `FAIL: the mind does not show it`; round 2 verdict `FAIL: the mind does not show it`

### numbers

| item | round 1 | round 2 |
| --- | --- | --- |
| order mind | +0.065 [-0.311, +0.440] | +0.034 [-0.272, +0.340] |
| order bandit | -0.091 [-0.453, +0.271] | -0.195 [-0.466, +0.075] |
| order ppo | -0.097 [-0.234, +0.041] | -0.038 [-0.522, +0.447] |
| order rules | +0.070 [-0.038, +0.178] | -0.023 [-0.259, +0.213] |
| order random | +0.023 [-0.007, +0.053] | -0.016 [-0.059, +0.028] |
