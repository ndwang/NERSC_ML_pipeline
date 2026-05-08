#!/usr/bin/env python
"""Evaluate a checkpoint by visualizing best, median, and worst reconstructions.

For each group (best, median, worst — 3 samples each by default), saves a
per-sample side-by-side plot (input / recon / residual, all channels stacked).
Also writes a CSV with per-sample recon losses for the evaluated subset.

Usage:
    python scripts/eval_recon_cases.py runs/<run>
    python scripts/eval_recon_cases.py runs/<run> --n-per-group 3 --n-samples 5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from beam_vae.data.preprocessing import PLANE_NAMES
from scripts.analyze_model import load_run


@torch.no_grad()
def per_sample_recon_loss(model, dataset, n_samples, batch_size=256, device="cpu"):
    """Run inference and return per-sample recon loss matching training definition.

    Training loss: ((recon - target) ** 2).sum(dim=(2,3)).mean()  — sum over HW,
    mean over batch and channels. Per-sample: sum over HW, mean over channels.
    """
    n = min(len(dataset), n_samples)
    subset = Subset(dataset, range(n))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device).eval()

    losses, inputs, recons = [], [], []
    for maps, scales, centroids in loader:
        maps = maps.to(device)
        scales = scales.to(device)
        centroids = centroids.to(device)
        recon, _, _, _, _ = model(maps, scales, centroids)
        loss = ((recon - maps) ** 2).sum(dim=(2, 3)).mean(dim=1)
        losses.append(loss.cpu().numpy())
        inputs.append(maps.cpu().numpy())
        recons.append(recon.cpu().numpy())
    return (np.concatenate(losses),
            np.concatenate(inputs),
            np.concatenate(recons))


def select_groups(losses, n):
    sorted_idx = np.argsort(losses)
    best = sorted_idx[:n]
    worst = sorted_idx[-n:][::-1]
    mid = len(sorted_idx) // 2
    half = n // 2
    median = sorted_idx[mid - half: mid - half + n]
    return {"best": best, "median": median, "worst": worst}


def plot_sample(inp, rec, idx, group, loss_val, out_dir):
    n_channels = inp.shape[0]
    res = inp - rec
    fig, axes = plt.subplots(n_channels, 3, figsize=(12, 3.2 * n_channels),
                             constrained_layout=True)
    fig.suptitle(f"{group} — sample {idx}   recon_loss={loss_val:.4e}", fontsize=20)
    for ch in range(n_channels):
        vmin = min(inp[ch].min(), rec[ch].min())
        vmax = max(inp[ch].max(), rec[ch].max())
        axes[ch, 0].imshow(inp[ch], vmin=vmin, vmax=vmax, cmap="viridis")
        axes[ch, 0].set_ylabel(PLANE_NAMES[ch], fontsize=14)
        axes[ch, 1].imshow(rec[ch], vmin=vmin, vmax=vmax, cmap="viridis")
        rmax = float(np.abs(res[ch]).max())
        axes[ch, 2].imshow(res[ch], cmap="RdBu_r", vmin=-rmax, vmax=rmax)
        for ax in axes[ch]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("Input", fontsize=16)
    axes[0, 1].set_title("Reconstruction", fontsize=16)
    axes[0, 2].set_title("Residual", fontsize=16)
    out_path = out_dir / f"s{idx}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_cumulative_loss(losses, out_dir):
    sorted_l = np.sort(losses)
    cdf = np.arange(1, len(sorted_l) + 1) / len(sorted_l) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_l, cdf, color="steelblue", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Recon loss")
    ax.set_ylabel("% of samples with loss ≤ x")
    ax.set_title(f"Cumulative recon loss (n={len(losses)})")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", alpha=0.3)

    for pct in [50, 90, 95, 99]:
        thresh = np.percentile(losses, pct)
        ax.axhline(pct, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(f"{pct}% ≤ {thresh:.2e}", xy=(thresh, pct),
                    xytext=(5, -12), textcoords="offset points",
                    fontsize=9, color="black")

    lo, hi = sorted_l[0], sorted_l[-1]
    ax.axvline(lo, color="C3", linewidth=1.0)
    ax.axvline(hi, color="C3", linewidth=1.0)
    ax.annotate(f"min = {lo:.2e}", xy=(lo, 50),
                xytext=(5, 0), textcoords="offset points",
                fontsize=9, color="C3", rotation=90, va="center")
    ax.annotate(f"max = {hi:.2e}", xy=(hi, 50),
                xytext=(-5, 0), textcoords="offset points",
                ha="right", fontsize=9, color="C3", rotation=90, va="center")

    fig.tight_layout()
    out_path = out_dir / "loss_cdf.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_loss_distribution(losses, groups, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(losses, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    colors = {"best": "C2", "median": "C0", "worst": "C3"}
    for name, idx in groups.items():
        for i in idx:
            ax.axvline(losses[i], color=colors[name], linewidth=1.0, alpha=0.7)
        ax.plot([], [], color=colors[name], label=name)  # legend stub
    ax.set_xlabel("Per-sample recon loss")
    ax.set_ylabel("Count")
    ax.set_title(f"Recon loss distribution (n={len(losses)})")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "loss_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument("--n-per-group", type=int, default=3,
                        help="Samples per group (best/median/worst)")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Max validation samples to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <run_dir>/eval_cases)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, model, val_dataset = load_run(args.run_dir)

    out_dir = Path(args.output) if args.output else Path(args.run_dir) / "eval_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    print("Computing per-sample recon losses...")
    losses, inputs, recons = per_sample_recon_loss(
        model, val_dataset, args.n_samples, device=device,
    )
    print(f"  Evaluated {len(losses)} samples  "
          f"min={losses.min():.4e}  median={np.median(losses):.4e}  max={losses.max():.4e}")

    groups = select_groups(losses, args.n_per_group)
    for name, idx in groups.items():
        print(f"  {name:6s}: indices {list(idx)}  losses={losses[idx].round(6)}")

    np.savetxt(out_dir / "recon_losses.csv",
               np.column_stack([np.arange(len(losses)), losses]),
               delimiter=",", header="sample_idx,recon_loss", comments="")

    plot_loss_distribution(losses, groups, out_dir)
    plot_cumulative_loss(losses, out_dir)

    for name, indices in groups.items():
        grp_dir = out_dir / name
        grp_dir.mkdir(exist_ok=True)
        print(f"\n--- {name} ---")
        for idx in indices:
            plot_sample(inputs[idx], recons[idx], int(idx), name,
                        float(losses[idx]), grp_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
