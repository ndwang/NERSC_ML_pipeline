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

DIMS  = [128, 256, 512]
BETAS = [0, 1e-6, 1e-5, 1e-4, 1e-3]

# (dim, beta) → (val_recon, val_scale, val_centroid)
DATA = {
    # Scan 1 — dim=128 beta sweep
    (128, 0):    (9.9e-6,  7.6e-5,  8.8e-5),
    (128, 1e-6): (1.1e-5,  5.6e-5,  1.3e-4),
    (128, 1e-5): (9.6e-6,  6.1e-5,  1.1e-4),
    (128, 1e-4): (9.4e-6,  3.3e-5,  1.4e-4),

    # Scan 2 / 3 — dim=256
    (256, 1e-5): (7.9e-6,  2.1e-5,  7.3e-5),
    (256, 1e-4): (9.9e-6,  2.9e-4,  1.5e-3),
    (256, 1e-3): (1.4e-5,  3.6e-2,  5.7e-2),

    # Scan 4 — dim=512
    (512, 1e-5): (1.1e-5,  2.0e-5,  4.3e-5),
    (512, 1e-4): (1.5e-5,  3.5e-3,  5.5e-4),
}

# Cells with a special label instead of a numeric value.
# "unreg" = unregularized failure (beta too low, z becomes noise); "crash" = NaN/exception; "t/o" = timed out
SPECIAL = {
    (256, 0):    "unreg",
    (256, 1e-6): "unreg",
    (512, 0):    "unreg",
    (512, 1e-6): "t/o",
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
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Val loss heatmaps: latent_dim × beta  (lower = better, log scale)",
             fontsize=13)

beta_labels = ["0", "1e-6", "1e-5", "1e-4", "1e-3"]
dim_labels  = [str(d) for d in DIMS]

panels = [
    ("Reconstruction loss", recon),
    ("Scale loss",          scale),
    ("Centroid loss",       centroid),
]

for ax, (title, mat) in zip(axes, panels):
    valid = mat[~np.isnan(mat)]
    vmin, vmax = valid.min(), valid.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis_r

    im = ax.imshow(mat, norm=norm, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

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
