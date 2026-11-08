| model | size | license | acc@1 % [95% CI] | moves rev labeled % | moves rev M9 % | ms M9 | ms labeled | RAM peak MiB (anon) | VRAM peak MiB (torch / nvidia-smi) | conformal covers % / mean set |
|---|---|---|---|---|---|---|---|---|---|---|
| gdecide | 0.34B | Apache-2.0 | 61.5 [60.3-62.7] | 17.2 | 6.0 | 22.95 | 22.84 | 1487.5 (1084.2) | 2105.7 / 1122 | - |
| vegaml | 0.8B frozen Qwen3.5 + 14M engine | Apache-2.0 (vegaml and Qwen3.5-0.8B) | 29.5 [28.4-30.7] | 60.4 | 62.5 | 98.13 | 94.33 | 2342.1 (1338.4) | 1650.1 / 1850 | 30.8 / 1.07 |
| gliclass | 0.15B | Apache-2.0 | 35.8 [34.6-37.0] | 35.7 | 15.3 | 13.04 | 12.62 | 1378.6 (841.4) | 639.9 / 762 | - |
| gdecide1b | 1B | Apache-2.0 | 51.2 [50.0-52.4] | 20.6 | 17.9 | 26.34 | 20.93 | 1446.7 (1047.7) | 4713.1 / 3220 | - |
| von | 0.4B | see the model card | 34.7 [33.5-35.9] | 70.9 | 32.3 | 20.99 | 20.59 | 2328.2 (2003.2) | 3016.0 / 3318 | - |
| laya | 0.42B | see the model card | 41.6 [40.4-42.8] | 47.8 | 29.2 | 22.44 | 22.4 | 3002.8 (2080.0) | 2337.1 / 2650 | - |
| bev | 0.4B | CC-BY-NC-4.0 (testing only) | not screened: the OOM killer took it while loading, under the 2500 MB cap and again under 3300 MB (lab/results/screen/run_bev_1.log, run_bev_2.log) | | | | | | | |
