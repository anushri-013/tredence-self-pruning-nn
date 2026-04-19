"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
Tredence AI Engineering Intern – Case Study Submission

Problem : "The Self-Pruning Neural Network"
Dataset : CIFAR-10  (torchvision.datasets.CIFAR10)
Framework: PyTorch

Overview
--------
A feed-forward network is augmented with learnable "gate" parameters
(one per weight).  During training, an L1 sparsity loss on sigmoid(gate_scores)
drives most gates toward 0, effectively pruning the corresponding weights in-place
— no post-training step needed.

Total Loss = CrossEntropy + λ * Σ |σ(gate_scores)|
           = ClassificationLoss + λ * SparsityLoss

Usage
-----
    python self_pruning_nn.py               # default: 3 λ values, 30 epochs
    python self_pruning_nn.py --epochs 50   # longer training
"""

import argparse
import math
import os
import time
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")          # works in headless / server environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear  –  Custom Gated Linear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that attaches one learnable gate scalar
    (gate_score) to *every* weight.

    Forward pass
    ────────────
    1. gates          = σ(gate_scores)           gates ∈ (0, 1)
    2. pruned_weights = weight ⊙ gates           element-wise multiply
    3. output         = x @ pruned_weights.T + bias

    Why gradients flow
    ──────────────────
    All operations (sigmoid, element-wise multiply, F.linear) are standard
    differentiable PyTorch ops.  The autograd engine automatically builds a
    computation graph that lets ∂Loss/∂weight and ∂Loss/∂gate_scores be
    computed and updated by any optimizer — no custom backward() needed.

    Gate initialisation
    ───────────────────
    gate_scores are initialised to 0, so σ(0) = 0.5 — all gates start
    half-open, a neutral state that neither prunes nor amplifies weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Standard weight & bias (identical to nn.Linear) ──
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None

        # ── Gate scores – same shape as weight ──
        # Registered as a Parameter so the optimizer updates them alongside
        # the weights.  Initialised to 0 → gates start at 0.5.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        self._reset_parameters()

    # ──────────────────────────────────────────────────────────────
    def _reset_parameters(self) -> None:
        """Kaiming uniform (same as nn.Linear default)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 – squash gate_scores into (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2 – mask weights with gates (pruned_weights ≈ 0 when gate ≈ 0)
        pruned_weights = self.weight * gates

        # Step 3 – standard linear operation  (handles bias=None cleanly)
        return F.linear(x, pruned_weights, self.bias)

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> Tuple[float, int, int]:
        """
        Return (sparsity_fraction, pruned_count, total_count).
        A weight is considered pruned when its gate < threshold.
        """
        gates  = torch.sigmoid(self.gate_scores)
        pruned = int((gates < threshold).sum().item())
        total  = gates.numel()
        return pruned / total, pruned, total

    # ──────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Self-Pruning Network
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier built entirely from PrunableLinear layers.

    Architecture  (default)
    ──────────────────────
    Input  3×32×32 → flatten → 3072
    FC1    3072  → 512   (PrunableLinear + BN + ReLU + Dropout)
    FC2    512   → 256   (PrunableLinear + BN + ReLU + Dropout)
    FC3    256   → 128   (PrunableLinear + BN + ReLU + Dropout)
    FC4    128   → 10    (PrunableLinear – logits)
    """

    def __init__(self, hidden_sizes: List[int] = [512, 256, 128]) -> None:
        super().__init__()

        dims = [3 * 32 * 32] + hidden_sizes + [10]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation after final layer
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.2))

        self.net = nn.Sequential(*layers)

    # ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten spatial dims
        return self.net(x)

    # ──────────────────────────────────────────────────────────────
    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    # ──────────────────────────────────────────────────────────────
    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values  =  Σ |σ(gate_score)|

        Why L1 encourages sparsity
        ──────────────────────────
        The L1 norm is NOT differentiable at exactly 0, but its sub-gradient
        is sign(x).  For values > 0 (all our gates, since sigmoid > 0) this
        sub-gradient is +1, creating a constant downward pull on every gate
        value regardless of its magnitude.  Unlike L2, which produces weaker
        gradients near 0, the L1 penalty produces the *same* pressure whether
        a gate is 0.9 or 0.01.  This drives many gate values all the way to
        (near) zero — i.e. sparsity.  Combined with the classification loss,
        only gates whose weights genuinely help prediction survive above zero.
        """
        device = next(self.parameters()).device
        total  = torch.zeros(1, device=device)
        for layer in self.prunable_layers():
            # sigmoid always ≥ 0, so abs() is redundant but kept for clarity
            total = total + torch.sigmoid(layer.gate_scores).abs().sum()
        return total.squeeze()

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of all weights whose gate is below `threshold`."""
        pruned = total = 0
        for layer in self.prunable_layers():
            _, p, t = layer.sparsity(threshold)
            pruned += p;  total += t
        return pruned / total if total > 0 else 0.0

    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        """All post-sigmoid gate values as a flat numpy array (for plotting)."""
        return np.concatenate([
            torch.sigmoid(l.gate_scores).cpu().numpy().ravel()
            for l in self.prunable_layers()
        ])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Data Loaders  (CIFAR-10)
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(
    batch_size:  int = 128,
    data_dir:    str = "./data",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Download (if needed) and return train / test DataLoaders for CIFAR-10."""

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Training & Evaluation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     SelfPruningNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    lam:       float,
    device:    torch.device,
) -> Tuple[float, float]:
    """One training epoch; returns (mean_total_loss, mean_cls_loss)."""
    model.train()
    total_sum = cls_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()
        total_loss  = cls_loss + lam * sparse_loss

        total_loss.backward()
        optimizer.step()

        total_sum += total_loss.item()
        cls_sum   += cls_loss.item()

    n = len(loader)
    return total_sum / n, cls_sum / n


@torch.no_grad()
def evaluate(
    model:  SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Top-1 accuracy on `loader`."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full Experiment for One λ Value
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    lam:          float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    device:       torch.device,
    epochs:       int   = 30,
    lr:           float = 1e-3,
    hidden_sizes: List[int] = [512, 256, 128],
    print_every:  int   = 5,
) -> Tuple[float, float, np.ndarray]:
    """
    Train a SelfPruningNet with sparsity coefficient `lam`.

    Returns
    -------
    test_acc  : final test accuracy (0–1)
    sparsity  : global sparsity fraction (0–1)
    gate_vals : all gate values as numpy array (for distribution plot)
    """
    print(f"\n{'='*65}")
    print(f"  λ = {lam:.0e}  |  device = {device}  |  epochs = {epochs}")
    print(f"{'='*65}")

    model     = SelfPruningNet(hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing decays LR smoothly to near-zero over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tot_loss, cls_loss = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()

        if epoch % print_every == 0 or epoch == epochs:
            sp  = model.global_sparsity()
            acc = evaluate(model, test_loader, device)
            print(f"  Ep {epoch:3d}/{epochs} | "
                  f"TotLoss={tot_loss:.4f} | ClsLoss={cls_loss:.4f} | "
                  f"TestAcc={acc*100:.2f}% | Sparsity={sp*100:.1f}% | "
                  f"Elapsed={time.time()-t0:.0f}s")

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.global_sparsity()
    gate_vals = model.all_gate_values()

    print(f"\n  ✓ FINAL  →  TestAcc={test_acc*100:.2f}%  |  Sparsity={sparsity*100:.2f}%")
    return test_acc, sparsity, gate_vals


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(
    gate_vals: np.ndarray,
    lam:       float,
    sparsity:  float,
    save_path: str = "gate_distribution.png",
) -> None:
    """
    Histogram of all sigmoid gate values.
    A well-pruned model shows a large spike at 0 (pruned) and a smaller
    cluster near 1 (active weights important for classification).
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(gate_vals, bins=120, color="#4C72B0", alpha=0.85, edgecolor="none")
    ax.axvline(0.01, color="#E74C3C", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.01)")

    pruned_pct = (gate_vals < 0.01).mean() * 100
    ax.set_xlabel("Gate value  σ(gate_score)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Distribution of Gate Values  (λ = {lam:.0e})\n"
        f"{pruned_pct:.1f}% of weights pruned (gate < 0.01)  —  "
        f"Global sparsity = {sparsity*100:.1f}%",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved → {save_path}")


def plot_lambda_tradeoff(
    lambdas:    List[float],
    accuracies: List[float],
    sparsities: List[float],
    save_path:  str = "lambda_tradeoff.png",
) -> None:
    """
    Dual-axis line chart: Accuracy and Sparsity vs λ.
    Shows the trade-off clearly.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    c_acc, c_sp = "#2196F3", "#FF5722"

    ax1.set_xlabel("λ  (sparsity coefficient, log scale)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", color=c_acc, fontsize=12)
    ax1.plot(lambdas, [a * 100 for a in accuracies],
             "o-", color=c_acc, linewidth=2, markersize=8, label="Test Accuracy")
    ax1.tick_params(axis="y", labelcolor=c_acc)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Sparsity (%)", color=c_sp, fontsize=12)
    ax2.plot(lambdas, [s * 100 for s in sparsities],
             "s--", color=c_sp, linewidth=2, markersize=8, label="Sparsity")
    ax2.tick_params(axis="y", labelcolor=c_sp)

    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    plt.title("Sparsity vs Accuracy Trade-off  (λ sweep on CIFAR-10)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network – Tredence Case Study")
    p.add_argument("--epochs",      type=int,   default=30,          help="Epochs per λ run")
    p.add_argument("--batch_size",  type=int,   default=128,         help="Training batch size")
    p.add_argument("--lr",          type=float, default=1e-3,        help="Initial learning rate")
    p.add_argument("--lambdas",     type=float, nargs="+",
                   default=[1e-5, 1e-4, 1e-3],                       help="Sparsity coefficients to sweep")
    p.add_argument("--data_dir",    type=str,   default="./data",    help="CIFAR-10 data directory")
    p.add_argument("--num_workers", type=int,   default=2,           help="DataLoader workers")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'─'*65}")
    print(f"  Self-Pruning Neural Network  —  Tredence AI Intern Case Study")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Lambdas : {args.lambdas}")
    print(f"{'─'*65}")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )

    results: dict = {}   # lam → (acc, sparsity, gate_vals)
    best_lam = args.lambdas[0]
    best_acc = -1.0

    for lam in args.lambdas:
        acc, sparsity, gate_vals = run_experiment(
            lam          = lam,
            train_loader = train_loader,
            test_loader  = test_loader,
            device       = device,
            epochs       = args.epochs,
            lr           = args.lr,
        )
        results[lam] = (acc, sparsity, gate_vals)
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    # ── Summary table ────────────────────────────────────────────────
    print("\n\n" + "=" * 55)
    print(f"  {'Lambda':>10} │ {'Test Accuracy':>14} │ {'Sparsity (%)':>12}")
    print("=" * 55)
    for lam in args.lambdas:
        acc, sp, _ = results[lam]
        marker = " ◀ best" if lam == best_lam else ""
        print(f"  {lam:>10.0e} │ {acc*100:>13.2f}% │ {sp*100:>11.2f}%{marker}")
    print("=" * 55)

    # ── Plots ────────────────────────────────────────────────────────
    best_gate_vals = results[best_lam][2]
    best_sparsity  = results[best_lam][1]

    plot_gate_distribution(
        gate_vals = best_gate_vals,
        lam       = best_lam,
        sparsity  = best_sparsity,
        save_path = "gate_distribution.png",
    )
    plot_lambda_tradeoff(
        lambdas    = args.lambdas,
        accuracies = [results[l][0] for l in args.lambdas],
        sparsities = [results[l][1] for l in args.lambdas],
        save_path  = "lambda_tradeoff.png",
    )

    print("\n✅  All done!  Outputs: gate_distribution.png | lambda_tradeoff.png")


if __name__ == "__main__":
    main()