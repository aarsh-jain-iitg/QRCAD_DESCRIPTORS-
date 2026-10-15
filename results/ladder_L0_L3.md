# BLR ladder L0-L3, design protocol

299 parts x 2 augmentations = 598 queries; gallery = the 299 originals; target = the source part.
Cluster bootstrap (by part) 95% CI on MRR. L2 exponent p* = 1.0 (chosen on these queries).

| level | MRR | 95% CI | R@1 | R@5 | R@10 | median rank |
|---|---|---|---|---|---|---|
| L0 | 0.7736 | [0.7419, 0.8052] | 0.6538 | 0.9181 | 0.9565 | 1 |
| L1 | 0.8431 | [0.8164, 0.8689] | 0.7224 | 0.9783 | 0.9967 | 1 |
| L1r | 0.8740 | [0.8499, 0.8964] | 0.7642 | 1.0000 | 1.0000 | 1 |
| L2 | 0.8431 | [0.8174, 0.8692] | 0.7224 | 0.9783 | 0.9967 | 1 |
| L3 | 0.8686 | [0.8450, 0.8917] | 0.7525 | 0.9967 | 1.0000 | 1 |

Paired Wilcoxon on per-query reciprocal rank (Holm across these comparisons):

| pair | p | p (Holm) |
|---|---|---|
| L0 vs L1 | 1.74e-08 | 8.68e-08 |
| L1 vs L1r | 0.000182 | 0.000729 |
| L1r vs L2 | 0.000182 | 0.000729 |
| L2 vs L3 | 0.0297 | 0.0594 |
| L1 vs L3 | 0.0297 | 0.0594 |

Blocks with eps_b > 0.2: dad, curvature, t1, paf

| block | eps_b |
|---|---|
| d1 | 0.003 |
| d2 | 0.004 |
| triangle | 0.002 |
| d4 | 0.002 |
| shps | 0.074 |
| ncf | 0.016 |
| lbe | 0.022 |
| dad | 0.968 |
| curvature | 0.578 |
| crs_r | 0.002 |
| t1 | 0.267 |
| phd | 0.033 |
| crs_a | 0.062 |
| paf | 0.674 |
| mts | 0.045 |

Latency (S2 + S3, ms per query): K=N p50 2.30, K=N p95 2.48, K=100 p50 1.31, K=100 p95 1.55
