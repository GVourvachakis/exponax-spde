# Stochastic Allen–Cahn validation

Notebook commit `c4e1abda8975e87c1552cf0ad2c58d3ce7121064` · 2026-11-28 · Python 3.12.3, jax 0.12.1, jaxlib 0.12.1, numpy 2.5.5, exponax 0.2.1, exponax-spde 0.1.0 · CPU float64.

PASS meets the gate · INCONCLUSIVE leaves a signal or resolution unresolved · DIAGNOSTIC is descriptive · FAIL misses the gate.

| Check | Verdict | Evidence |
| --- | --- | --- |
| §1 deterministic parity | PASS | bit-exact physical=True, Fourier=True, 1–50 steps |
| §2 h=0.001 | PASS | 90 groups. all 3 comparisons agree; largest: §2 h=0.001 mode (3,) = 2.1041e-06 +- 1.5e-08, exact 2.0923e-06 (Bonferroni over 3, false-alarm rate 0.001) |
| §2 h=0.002 | PASS | 77 groups. all 3 comparisons agree; largest: §2 h=0.002 mode (3,) = 2.1056e-06 +- 1.4e-08, exact 2.0923e-06 (Bonferroni over 3, false-alarm rate 0.001) |
| §3 axes and interior | PASS | 142 groups. all 4 comparisons agree; largest: §3 axes and interior mode (1, 0) = 0.00051031 +- 9.9e-06, exact 0.00049178 (Bonferroni over 4, false-alarm rate 0.001) |
| §4 alpha=0 | PASS | 178 groups. all 3 comparisons agree; largest: §4 alpha=0 mode (16,) = 0.00036372 +- 6.4e-06, exact 0.00035621 (Bonferroni over 3, false-alarm rate 0.001) |
| §4 alpha=1 | PASS | 124 groups. all 3 comparisons agree; largest: §4 alpha=1 mode (4,) = 9.1709e-06 +- 2.1e-07, exact 9.0086e-06 (Bonferroni over 3, false-alarm rate 0.001) |
| §4 alpha=2 | PASS | 143 groups. all 3 comparisons agree; largest: §4 alpha=2 mode (1,) = 5.5334e-05 +- 1.1e-06, exact 5.5505e-05 (Bonferroni over 3, false-alarm rate 0.001) |
| §4 alpha=3 | PASS | 147 groups. all 3 comparisons agree; largest: §4 alpha=3 mode (1,) = 1.3595e-06 +- 2.3e-08, exact 1.3712e-06 (Bonferroni over 3, false-alarm rate 0.001) |
| §5 conditional kick moments | PASS | worst 0.308 SE, five-SE two-sided band |
| §8 radial PSD normalisation | PASS | max Parseval residual 2.22e-16 |
| §10 exact OU coupling | PASS | max Fourier difference 9.93e-16 |
| §10 additive fixed-grid order | PASS | order 0.9797, paired 95% [0.9577, 0.9996], band [0.75, 1.25], physverify PASS |
| §11 N=64 Euler–Maruyama | DIAGNOSTIC | order 0.4914, paired 95% [0.4614, 0.5231] |
| §11 N=64 Milstein | DIAGNOSTIC | order 0.9835, paired 95% [0.9789, 0.9884] |
| §11 #20 N=32 Euler–Maruyama | PASS | order 0.5170, paired 95% [0.5071, 0.5277], band [0.35, 0.7], physverify PASS |
| §11 #20 N=32 Milstein | PASS | order 1.0082, paired 95% [0.9980, 1.0190], band [0.8, 1.25], physverify PASS |
| §11 geometric Brownian Euler–Maruyama | PASS | order 0.5185, paired 95% [0.4936, 0.5448], band [0.4, 0.65], physverify PASS |
| §11 geometric Brownian Milstein | PASS | order 0.9949, paired 95% [0.9670, 1.0225], band [0.85, 1.2], physverify PASS |
| Schemes untamed order 1 | PASS | observed order 1.03 (expected 1) |
| Schemes tamed order 1 | PASS | observed order 1.02 (expected 1) |
| Schemes upstream ETDRK1 parity | PASS | one and 20 steps |
| Schemes untamed order 2 | PASS | observed order 2.07 (expected 2) |
| Schemes tamed order 2 | PASS | observed order 0.91 (expected 1) |
| Schemes upstream ETDRK2 parity | PASS | one and 20 steps |
| Schemes untamed order 3 | PASS | observed order 3.04 (expected 3) |
| Schemes tamed order 3 | PASS | observed order 1.00 (expected 1) |
| Schemes upstream ETDRK3 parity | PASS | one and 20 steps |
| Schemes untamed order 4 | PASS | observed order 4.06 (expected 4) |
| Schemes tamed order 4 | PASS | observed order 0.99 (expected 1) |
| Schemes upstream ETDRK4 parity | PASS | one and 20 steps |
| Schemes small-noise crossover | PASS | order-4 h*=0.02872452, largest MSE deviation 1.27 SE, slope bands ±0.2 |
| Schemes additive k=1 | PASS | order 1.0248, paired 95% [1.0026, 1.0459], band [0.8, 1.2], physverify PASS |
| Schemes additive k=2 | PASS | order 1.0243, paired 95% [1.0040, 1.0452], band [0.8, 1.2], physverify PASS |
| Schemes additive k=3 | PASS | order 1.0226, paired 95% [1.0021, 1.0437], band [0.8, 1.2], physverify PASS |
| Schemes additive k=4 | PASS | order 1.0227, paired 95% [1.0022, 1.0437], band [0.8, 1.2], physverify PASS |
| Schemes Euler–Maruyama k=2 | PASS | order 0.6024, paired 95% [0.5778, 0.6261], band [0.35, 0.65], physverify PASS |
| Schemes Euler–Maruyama k=4 | PASS | order 0.6019, paired 95% [0.5771, 0.6256], band [0.35, 0.65], physverify PASS |
| Schemes Milstein k=2 | PASS | order 1.0220, paired 95% [1.0119, 1.0323], band [0.8, 1.2], physverify PASS |
| Schemes Milstein k=4 | PASS | order 1.0205, paired 95% [1.0103, 1.0309], band [0.8, 1.2], physverify PASS |
| §12 coupled weak signal, coarse 8 | PASS | coarse–fine 245.97 SE, fine–reference 8.22 SE |
| §12 independent weak signal, coarse 8 | PASS | coarse–fine 6.42 SE, fine–reference 8.96 SE |
| §12 coupled weak signal, coarse 16 | PASS | coarse–fine 490.89 SE, fine–reference 7.19 SE |
| §12 independent weak signal, coarse 16 | PASS | coarse–fine 5.21 SE, fine–reference 7.73 SE |
| §12 exact Gaussian null | PASS | coupled gap 0, max discrepancy 1.21 SE |
| §12 exact four-level Gaussian marginals | PASS | largest pathwise gap 5.22e-15, marginal discrepancy 0.73 SE |
| §12 coupled four-level residual c=8 | INCONCLUSIVE | mean -0.000194976538, SE 3.91e-05, predicted -0.00024968, agreement 1.40 SE, detection 4.98 SE, M=256 |
| §12 coupled four-level residual c=16 | PASS | mean -6.1840018e-05, SE 9.09e-06, predicted -6.0416e-05, agreement 0.16 SE, detection 6.80 SE, M=1088 |
| §13 split composition | PASS | max error 1.67e-16 |
| §13 exact OU half-step | PASS | mean=0.29924859 (target 0.29925047), var=5.6321e-06 (target 5.618e-06), max 0.16 SE |
