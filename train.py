import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math, os, time

# ==========================================
# HYPERPARAMETERS
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH = 256
LR_WEIGHTS = 3e-3
LR_GATES = 0.02
WD = 1e-4
THRESHOLD = 0.01

# FIX 4: Higher T_START for more initial exploration, much lower T_END for sharper final gates
T_START = 2.0       
T_END = 0.02        

# FIX 3: Warmup period — pure classification before sparsity penalty kicks in
WARMUP_EPOCHS = 5   

# FIX 1+2: Lambda values rescaled for normalized loss (range 0-1)
# Old lambda 1e-6 (raw sum) ≈ new lambda 5e-2 (normalized mean)
CONFIGS = [
    {"lam": 1e-2, "desc": "light"},       # Was: 5e-7
    {"lam": 5e-2, "desc": "medium"},      # Was: 1e-6
    {"lam": 1e-1, "desc": "aggressive"},  # Was: 2e-6
]

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ==========================================
# CUSTOM PRUNABLE LAYER
# ==========================================
class PrunableLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))

      
        self.gate_scores = nn.Parameter(torch.full((out_f, in_f), -0.5))  # Was: torch.zeros(...)

        self.temp = T_START
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_f)
        nn.init.uniform_(self.bias, -bound, bound)

    def gates(self):
        # Temperature-scaled sigmoid: lower T → sharper, more binary gates
        return torch.sigmoid(self.gate_scores / self.temp)

    def forward(self, x):
        return F.linear(x, self.weight * self.gates(), self.bias)

    def sparsity(self):
        with torch.no_grad():
            return (self.gates() < THRESHOLD).float().mean().item()

# ==========================================
# NETWORK ARCHITECTURE 
# ==========================================
class PruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [3*32*32, 1024, 512, 256, 128]
        self.layers = nn.ModuleList([PrunableLinear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(s) for s in sizes[1:]])
        self.head = PrunableLinear(128, 10)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer, bn in zip(self.layers, self.bns):
            x = self.drop(F.gelu(bn(layer(x))))
        return self.head(x)

    def prunable(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def set_temp(self, t):
        for layer in self.prunable():
            layer.temp = t

    # FIX 1 + FIX 2: Normalized sparsity loss with binarization pressure
    def sparsity_loss(self):
        total_l1 = torch.tensor(0., device=DEVICE)
        total_binarize = torch.tensor(0., device=DEVICE)
        total_params = 0

        for layer in self.prunable():
            g = layer.gates()
            # L1 term: pushes gates toward 0 (drives sparsity)
            total_l1 = total_l1 + g.sum()
            # Binarization term: p*(1-p) is max=0.25 at p=0.5, zero at p=0 or p=1
            # This penalizes indecisive gates and rewards commitment to 0 or 1
            total_binarize = total_binarize + (g * (1 - g)).sum()
            total_params += g.numel()

        # Normalize by total parameters → output in [0, 1]
        # This makes lambda scale-invariant across network sizes
        return (total_l1 + 0.5 * total_binarize) / total_params

    def global_sparsity(self):
        pruned = total = 0
        for layer in self.prunable():
            g = layer.gates()
            pruned += (g < THRESHOLD).sum().item()
            total += g.numel()
        return pruned / total

    def per_layer_sparsity(self):
        rows = []
        for i, layer in enumerate(self.prunable()):
            rows.append({"layer": i, "shape": list(layer.weight.shape), "sparsity": layer.sparsity()})
        return rows

    def hard_prune(self):
        with torch.no_grad():
            for layer in self.prunable():
                mask = (layer.gates() >= THRESHOLD).float()
                # Zero out pruned weights
                layer.weight.data.mul_(mask)
                # Freeze pruned gate scores to large negative value
                pruned_mask = (mask == 0)
                layer.gate_scores.data[pruned_mask] = -10.0  # NEW

    def collect_gates(self):
        vals = []
        with torch.no_grad():
            for layer in self.prunable():
                vals.extend(layer.gates().cpu().numpy().flatten().tolist())
        return vals

# ==========================================
# DATA PIPELINE 
# ==========================================
def get_data():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    te = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(te, batch_size=512,  shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader

# ==========================================
# TRAINING LOOP
# ==========================================
def train_epoch(model, loader, opt, lam, epoch):
    model.train()

    # FIX 4: Exponential temperature decay instead of cosine
    # Drops quickly early (exploration) then plateaus near T_END (commitment)
    # Formula: T = T_START * (T_END/T_START)^(epoch/EPOCHS)
    temp = T_START * (T_END / T_START) ** (epoch / EPOCHS)
    model.set_temp(temp)

    total_loss = correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        logits = model(imgs)
        cls_loss = F.cross_entropy(logits, labels)

        # FIX 3: Sparsity warmup — pure classification for first WARMUP_EPOCHS epochs
        # Allows the model to learn useful representations before pruning begins
        if epoch <= WARMUP_EPOCHS:
            loss = cls_loss
        else:
            # Smooth ramp-in of sparsity penalty over 5 epochs after warmup
            # Prevents sudden loss spike at epoch WARMUP_EPOCHS+1
            ramp = min(1.0, (epoch - WARMUP_EPOCHS) / 5.0)
            loss = cls_loss + lam * ramp * model.sparsity_loss()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += imgs.size(0)

    return total_loss / seen, correct / seen, temp

def evaluate(model, loader):
    model.eval()
    correct = seen = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            seen += imgs.size(0)
    return correct / seen

# ==========================================
# RUN ONE CONFIG
# ==========================================
def run(cfg, train_loader, test_loader):
    lam, desc = cfg["lam"], cfg["desc"]
    print(f"\n{'='*60}")
    print(f"  [{desc.upper()}] lambda={lam:.0e} | warmup={WARMUP_EPOCHS} | T:{T_START}→{T_END}")
    print(f"{'='*60}")

    model = PruningNet().to(DEVICE)

    # Parameter decoupling: weight decay MUST be 0 for gates (allows them to fall to 0)
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    opt = torch.optim.Adam([
        {"params": other_params, "lr": LR_WEIGHTS, "weight_decay": WD},
        {"params": gate_params,  "lr": LR_GATES,   "weight_decay": 0.0},
    ])

    history = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss, tr_acc, temp = train_epoch(model, train_loader, opt, lam, epoch)
        te_acc   = evaluate(model, test_loader)
        sparsity = model.global_sparsity()

        history.append({"epoch": epoch, "test_acc": te_acc, "sparsity": sparsity})

        warmup_tag = " [WARMUP]" if epoch <= WARMUP_EPOCHS else ""
        print(f"  ep {epoch:2d}/{EPOCHS} | loss={loss:.4f} | train={tr_acc:.3f} | "
              f"test={te_acc:.3f} | sparse={sparsity:.1%} | T={temp:.3f} | "
              f"{time.time()-t0:.0f}s{warmup_tag}")

    model.hard_prune()
    final_acc      = evaluate(model, test_loader)
    final_sparsity = model.global_sparsity()

    print(f"\n  [Hard Prune Complete] Test Acc: {final_acc:.3%} | Sparsity: {final_sparsity:.1%}")
    for row in model.per_layer_sparsity():
        print(f"  layer {row['layer']} {row['shape']} -> {row['sparsity']:.1%} pruned")

    return {
        "lam": lam, "desc": desc,
        "history": history,
        "gates": model.collect_gates(),
        "sparsity": final_sparsity,
        "final_acc": final_acc,
    }

# ==========================================
# PLOTTING
# ==========================================
def plot_results(all_results):
    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for res in all_results:
        epochs = [h["epoch"] for h in res["history"]]
        accs   = [h["test_acc"] for h in res["history"]]
        spars  = [h["sparsity"] for h in res["history"]]
        lbl = f"λ={res['lam']:.0e} ({res['desc']})"
        axes[0].plot(epochs, accs,  marker='.', label=lbl)
        axes[1].plot(epochs, spars, marker='.', label=lbl)

    # Shade warmup region
    for ax in axes:
        ax.axvspan(1, WARMUP_EPOCHS, alpha=0.08, color='gray', label='warmup')

    axes[0].set(title="Test Accuracy over Epochs",   xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title="Network Sparsity over Epochs", xlabel="Epoch", ylabel="Sparsity")
    axes[0].grid(True, alpha=0.3); axes[1].grid(True, alpha=0.3)
    axes[0].legend(); axes[1].legend()
    plt.tight_layout()
    plt.savefig("plots/training_metrics.png", dpi=150)
    plt.show()

    # --- Gate distributions (bimodality check) ---
    fig, axes = plt.subplots(1, len(all_results), figsize=(18, 4))
    if len(all_results) == 1:
        axes = [axes]
    for i, res in enumerate(all_results):
        gates = res["gates"]
        axes[i].hist(gates, bins=80, color='royalblue', alpha=0.7)
        near_zero = sum(1 for g in gates if g < 0.1) / len(gates)
        near_one  = sum(1 for g in gates if g > 0.9) / len(gates)
        axes[i].set(
            title=f"Gates (λ={res['lam']:.0e}) | Sparsity: {res['sparsity']:.1%}\n"
                  f"near-0: {near_zero:.1%} | near-1: {near_one:.1%}",
            xlabel="Gate Value"
        )
        axes[i].set_yscale('log')
        axes[i].axvline(0.1, color='red',   linestyle='--', alpha=0.5, label='prune threshold')
        axes[i].axvline(0.9, color='green', linestyle='--', alpha=0.5, label='keep threshold')
        axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/gate_distributions.png", dpi=150)
    plt.show()

    # --- Summary table ---
    print("\n" + "="*55)
    print(f"  {'Config':<12} {'Lambda':<10} {'Sparsity':<12} {'Test Acc':<10}")
    print("="*55)
    for res in all_results:
        print(f"  {res['desc']:<12} {res['lam']:<10.0e} {res['sparsity']:<12.1%} {res['final_acc']:<10.3%}")
    print("="*55)

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Config: EPOCHS={EPOCHS}, WARMUP={WARMUP_EPOCHS}, T:{T_START}→{T_END}")
    train_loader, test_loader = get_data()
    results = [run(cfg, train_loader, test_loader) for cfg in CONFIGS]
    plot_results(results)
