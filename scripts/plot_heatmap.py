#!/usr/bin/env python
"""Generate the latent_dim × beta heatmap of val metrics.

Run from project root:
    python scripts/plot_heatmap.py
Output: runs/heatmap_dim_beta.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── data ──────────────────────────────────────────────────────────────────────
# Rows: latent_dim ∈ {128, 256, 512}
# Cols: beta ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3}
# Values: val metric at best epoch; None = not run.
# SPECIAL: cells with a non-numeric outcome (collapse, crash, timeout).

DIMS  = [64, 128, 256, 512]
BETAS = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 1e-3]

# (dim, beta) → (val_recon, val_scale, val_centroid)
DATA = {
    # Scan 6 — dim=64 beta sweep
    (64,  1e-6): (1.02e-5, 2.70e-5, 5.24e-5),  # partial collapse (KL=25.4)
    (64,  2e-6): (9.5e-6,  3.24e-5, 5.88e-5),  # partial collapse (KL=5.78)
    (64,  5e-6): (1.02e-5, 7.41e-5, 1.51e-4),
    (64,  1e-5): (1.01e-5, 4.60e-5, 6.37e-4),

    # Scan 1 — dim=128 beta sweep
    (128, 0):    (9.9e-6,  7.6e-5,  8.8e-5),
    (128, 1e-6): (1.1e-5,  5.6e-5,  1.3e-4),
    (128, 1e-5): (9.6e-6,  6.1e-5,  1.1e-4),
    (128, 1e-4): (9.4e-6,  3.3e-5,  1.4e-4),

    # Scan 5 — dim=128 fine beta sweep
    (128, 2e-6): (1.17e-5, 3.78e-5, 5.87e-5),
    (128, 5e-6): (1.08e-5, 1.82e-5, 4.72e-5),
    (128, 2e-5): (9.2e-6,  1.34e-4, 3.66e-4),
    (128, 5e-5): (1.18e-5, 2.99e-4, 2.36e-3),

    # Scan 2 / 3 / 4 / 5 — dim=256
    (256, 0):    (4.3e-4,  1.0e-2,  2.8e-2),  # collapsed epoch 9 (lr=1e-3)
    (256, 1e-6): (8.6e-6,  2.52e-5, 1.224e-4),  # Scan 7, lr=5e-4, epoch 468
    (256, 2e-6): (2.33e-4, 9.3e-3,  6.3e-2),  # collapsed epoch 13 (lr=1e-3)
    (256, 5e-6): (9.5e-6,  3.80e-5, 8.85e-5),  # Scan 7, lr=5e-4, epoch 500
    (256, 1e-5): (7.9e-6,  2.1e-5,  7.3e-5),
    (256, 2e-5): (9.51e-6, 3.87e-5, 2.53e-4),  # Scan 5 rerun, 500 epochs
    (256, 5e-5): (9.97e-6, 1.11e-4, 7.76e-4),  # Scan 5 rerun, epoch-500 converged value
    (256, 1e-4): (9.9e-6,  2.9e-4,  1.5e-3),
    (256, 1e-3): (1.4e-5,  3.6e-2,  5.7e-2),

    # Scan 3 / 4 / 5 — dim=512 (512+1e-6 timed out at epoch 459)
    (512, 1e-6): (9.9e-6,  1.9e-5,  4.8e-5),  # t/o epoch 459, not collapsed (KL=1.39)
    (512, 1e-5): (1.1e-5,  2.0e-5,  4.3e-5),
    (512, 2e-5): (1.30e-5, 3.40e-5, 5.70e-5),  # Scan 5 rerun, 500 epochs
    (512, 5e-5): (1.08e-5, 4.73e-5, 1.46e-4),  # Scan 5 rerun, 500 epochs
    (512, 1e-4): (1.5e-5,  3.5e-3,  5.5e-4),
}

# Cells with a special label overlaid on (or instead of) the numeric value.
# "unreg" = posterior collapse (beta too low); "crash" = NaN/exception
# "t/o" = timed out; "partial" = partial collapse, still shows numeric value
SPECIAL = {
    (256, 0):   "unreg",  # collapsed epoch 9, lr=1e-3
    (256, 2e-6): "unreg",  # collapsed epoch 13, lr=1e-3
    (512, 0):   "unreg",  # collapsed before epoch 50, lr=1e-3
}

# ── build arrays ──────────────────────────────────────────────────────────────
nrows, ncols = len(DIMS), len(BETAS)
recon    = np.full((nrows, ncols), np.nan)
scale    = np.full((nrows, ncols), np.nan)
centroid = np.full((nrows, ncols), np.nan)

for ri, d in enumerate(DIMS):
    for ci, b in enumerate(BETAS):
        if (d, b) in DATA:
            r, s, c = DATA[(d, b)]
            recon[ri, ci]    = r
            scale[ri, ci]    = s
            centroid[ri, ci] = c

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 5))
fig.suptitle("Val loss heatmaps: latent_dim × beta  (lower = better, log scale)",
             fontsize=13)

beta_labels = ["0", "1e-6", "2e-6", "5e-6", "1e-5", "2e-5", "5e-5", "1e-4", "1e-3"]
dim_labels  = [str(d) for d in DIMS]

panels = [
    ("Reconstruction loss", recon),
    ("Scale loss",          scale),
    ("Centroid loss",       centroid),
]

for ax, (title, mat) in zip(axes, panels):
    valid = mat[~np.isnan(mat)]
    vmin = valid.min()
    vmax = np.percentile(valid, 85)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r

    im = ax.imshow(mat, norm=norm, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, extend="max")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(beta_labels)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(dim_labels)
    ax.set_xlabel("beta")
    ax.set_ylabel("latent_dim")
    ax.set_title(title)

    for ri, d in enumerate(DIMS):
            for ci, b in enumerate(BETAS):
                v = mat[ri, ci]
                special = SPECIAL.get((d, b))
                if special == "unreg":
                    ax.text(ci, ri, "unregularized", ha="center", va="center",
                            fontsize=6, color="#ff4444", fontweight="bold")
                elif special == "crash":
                    ax.text(ci, ri, "NaN", ha="center", va="center",
                            fontsize=8, color="gray", fontweight="bold")
                elif special == "t/o":
                    ax.text(ci, ri, "t/o", ha="center", va="center",
                            fontsize=8, color="gray", fontweight="bold")
                elif np.isnan(v):
                    ax.text(ci, ri, "—", ha="center", va="center",
                            fontsize=8, color="gray", fontweight="bold")
                else:
                    txt = f"{v:.1e}"
                    normed = (np.log10(v) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
                    color = "white" if normed > 0.5 else "black"
                    ax.text(ci, ri, txt, ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")

plt.tight_layout()
out = "runs/heatmap_dim_beta.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
