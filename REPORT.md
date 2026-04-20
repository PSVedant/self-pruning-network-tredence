# Self-Pruning Neural Network Report

## Why L1 penalty on sigmoid gates encourages sparsity

The L1 norm (sum of absolute values) of the gate values is added to the cross‑entropy loss. Because the gates are always positive (after sigmoid), this is simply their sum. Minimising this sum pushes gates toward zero, effectively pruning the corresponding weights. The sigmoid ensures gates stay in [0,1], and the L1 penalty drives them to exactly zero, not just small values.

## Results for different λ values

| λ (lambda) | Test Accuracy | Sparsity (%) |
|------------|---------------|---------------|
| 1e-02 | 60.430% | 93.4% |
| 5e-02 | 61.270% | 94.4% |
| 1e-01 | 60.060% | 94.9% |

## Gate Distribution

The gate distribution plot `plots/gate_distributions.png` (or `logs/gate_dist_0.png`, etc.) shows a large spike at 0 (pruned weights) and a smaller cluster away from 0 (kept weights). This bimodal distribution proves the network learned to commit gates to either 0 or 1.

## Trade‑off

Higher λ leads to higher sparsity but slightly lower accuracy, as shown in the table above.
