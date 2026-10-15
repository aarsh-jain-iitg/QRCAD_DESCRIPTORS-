# Phase 2 augmentation sanity

Percentile u of (part, augmentation) distances within the random-pair distribution of each block (0 = closest possible). Target: at least 90 % of pairs below the median random pair (u < 0.5). Blocks below that are eps_b candidates.

| block | median u | p90 u | pairs with u<0.5 | eps_b |
|---|---|---|---|---|
| d1 | 0.0030 | 0.034 | 99.8% | 0.003 |
| d2 | 0.0063 | 0.050 | 99.8% | 0.004 |
| triangle | 0.0040 | 0.011 | 100.0% | 0.002 |
| d4 | 0.0050 | 0.015 | 99.3% | 0.002 |
| shps | 0.1150 | 0.615 | 83.9% | 0.074 |
| ncf | 0.0121 | 0.251 | 96.0% | 0.016 |
| lbe | 0.0021 | 0.011 | 97.7% | 0.022 |
| dad | 0.8680 | 0.961 | 18.1% | 0.968 |
| curvature | 0.4523 | 0.860 | 52.3% | 0.578 |
| crs_r | 0.0037 | 0.016 | 99.5% | 0.002 |
| t1 | 0.0165 | 0.722 | 84.9% | 0.267 |
| phd | 0.0085 | 0.076 | 97.8% | 0.033 |
| crs_a | 0.0217 | 0.248 | 95.0% | 0.062 |
| paf | 0.5711 | 0.912 | 44.6% | 0.674 |
| mts | 0.4092 | 0.846 | 54.5% | 0.045 |

Blocks under 90 %: shps, dad, curvature, t1, paf, mts
