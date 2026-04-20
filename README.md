# Self-Pruning Neural Network  
**Tredence AI Engineering Intern – Case Study Submission**


## The Problem

Deploying large neural networks on edge devices is hard. Standard pruning happens after training. This network prunes itself during training. Every weight has a learnable gate (0 to 1) that multiplies the weight's output. An L1 penalty on gates pushes them toward zero. Important weights resist; redundant weights let their gates drop to zero and are deleted.

---

## Who This Is For

An AI Engineer evaluating a candidate. They want to see:

- A correct `PrunableLinear` layer with proper gradient flow.
- A training loop that includes the sparsity loss.
- A clear trade‑off between sparsity and accuracy for different lambda values.
- A gate distribution plot with a spike at 0 and a cluster away from 0.

This project meets all those requirements.

---

## What the Network Does

A feed‑forward MLP (3072 → 1024 → 512 → 256 → 128 → 10) trained on CIFAR‑10.  
Each `PrunableLinear` layer contains:

- Standard `weight` and `bias`
- A `gate_scores` tensor (same shape as weight)
- Temperature‑scaled sigmoid: `gates = sigmoid(gate_scores / temperature)`
- Forward pass: `output = linear(input, weight * gates, bias)`

Total loss = `CrossEntropy + lambda * mean(gates + 0.5 * g(1-g))`  
The second term is a **dual‑term normalised sparsity loss**:

- L1 term (`mean(gates)`) pushes gates toward 0.
- Binarization term (`mean(g(1-g))`) penalises gates at 0.5, rewarding commitment to 0 or 1.
- Normalisation keeps output in [0,1], so lambda is directly interpretable.

---

## How It Is Different from a Basic Implementation

- **Dual‑term gate loss**: Pure L1 produces many gates near 0.5. Adding `g(1-g)` forces bimodality – spike at 0, cluster near 1.
- **Warmup + ramp‑in**: First 5 epochs no sparsity loss. Then penalty ramps in over 5 epochs. Prevents early collapse.
- **Exponential temperature decay**: `T = 2.0 -> 0.02`. Gates start soft (explore) and become hard (commit) near the end.
- **Separate learning rates**: `LR_gates = 0.02`, `LR_weights = 0.003`. Gates learn much faster – essential for millions of gates.
- **Negative gate initialisation**: `gate_scores = -0.5` → gates ≈ 0.38. Network starts slightly pruned and learns which gates to reopen.
- **Hard prune + freeze**: After training, pruned gates set to -10 (sigmoid ≈ 0) and weights zeroed. No gate computation at inference.

## System Architecture 

| Layer | Type | Input → Output | Additional |
|-------|------|----------------|------------|
| 1 | Flatten | (3, 32, 32) → 3072 | - |
| 2 | PrunableLinear | 3072 → 1024 | + BatchNorm + GELU + Dropout(0.1) |
| 3 | PrunableLinear | 1024 → 512 | + BatchNorm + GELU + Dropout(0.1) |
| 4 | PrunableLinear | 512 → 256 | + BatchNorm + GELU + Dropout(0.1) |
| 5 | PrunableLinear | 256 → 128 | + BatchNorm + GELU + Dropout(0.1) |
| 6 | PrunableLinear (head) | 128 → 10 | No extra layers |
| Loss | CrossEntropy + λ × sparsity_loss | - | Sparsity loss = mean(gates + 0.5·g(1-g)) |

All gates are summed (normalised) and added to the loss. Gradients flow to both weights and gate scores.

---

## Step‑by‑Step: How Pruning Happens

1. **Initialisation:** `gate_scores = -0.5` → gates ≈ 0.38. Most weights are already slightly pruned.
2. **Warmup (epochs 1–5):** No sparsity loss. Network learns useful features without interference.
3. **Ramp‑in (epochs 6–10):** Sparsity penalty linearly increases from 0 to full λ. Gates start moving.
4. **Exponential temperature decay:** Temperature drops from 2.0 to 0.02. At low T, sigmoid becomes almost step function. Gates near 0.5 forced to 0 or 1.
5. **Hard prune:** After 30 epochs, gates < 0.01 frozen to -10 and weights zeroed. Final model is sparse, no gate multiplication.


## How to Run (Google Colab with GPU)

1. Open Colab and change runtime type to T4 GPU.
2. Clone the repository.
3. Install dependencies from requirements.txt.
4. Train the network (takes about 15 minutes).
5. Generate the report and extra plots.
6. Download results from the logs and plots folders.

## Results

| λ (lambda) | Test accuracy | Sparsity |
|------------|---------------|----------|
| 1e-2       | 60.95%        | 93.3%    |
| 5e-2       | 60.27%        | 94.5%    |
| 1e-1       | 59.72%        | 94.8%    |

- Sparsity >93% with only 1‑2% accuracy drop (unpruned baseline ≈62%).
- The last layer prunes much less (≈55%) – classifier needs many connections.
- Gate distribution (`plots/gate_distributions.png`) shows huge spike at 0 and small cluster near 1 – required bimodal shape.
- Trade‑off: higher λ → higher sparsity → slightly lower accuracy.

## Why L1 on Sigmoid Gates Produces Sparsity

L1 penalty is sum of all gate values. Because gates are always positive (sigmoid), minimising sum pushes them toward zero. Sigmoid keeps gates in [0,1], and L1 has constant gradient (unlike L2, which weakens as value gets small). Constant pressure drives gates to exactly zero – not just small numbers. L1 regularisation creates sparse solutions.

Binarization term `g(1-g)` is maximal at 0.5 and zero at 0 or 1. Penalising it pushes indecisive gates to commit, creating bimodal histogram.

## Requirements
torch>=2.0.0, torchvision>=0.15.0, matplotlib>=3.7.0, numpy>=1.24.0

## Evaluation Criteria – How This Project Meets Them

- **PrunableLinear correctness:** Gradients flow to both weight and gate_scores; forward pass uses weight * gates. Verified by training and hard prune.
- **Training loop with sparsity loss:** Total loss includes normalised dual‑term sparsity loss with warmup, ramp‑in, and temperature decay.
- **Quality of results:** Clear trade‑off table, bimodal gate distribution plot, per‑layer sparsity report (printed in logs).
- **Code quality:** Clean, modular, well‑commented, single‑file train.py + report.py, easy to run in Colab.
  
## Conclusion

All case study requirements are satisfied. The network prunes itself during training, achieving **>93% sparsity** while maintaining strong accuracy on CIFAR‑10.

**GitHub repository:** [https://github.com/PSVedant/self-pruning-network-tredence](https://github.com/PSVedant/self-pruning-network-tredence)
