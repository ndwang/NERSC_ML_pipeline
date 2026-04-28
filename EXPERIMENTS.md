# Experiment Log

**Defaults unless noted:** `model=vae2d` · `lr=1e-3` · `latent_dim=128` · `beta=1e-5` · `scheduler=ReduceOnPlateau` · `500 epochs max`

**Dataset:** All v2 runs use `v2_sectioned_1sec_10k` (`/pscratch/sd/n/ndwang/latent_beam_dynamics/data/v2/vae_training/`). Legacy v1 runs (sectioned_10k, linear_10k, frequency_maps datasets) are archived in `runs/v1/` and not documented here.

---

## Scan 1 — Beta sweep (2026-04-27)

**Question:** What KL weight produces the best reconstruction and downstream scale/centroid metrics on v2 data?

Fixed: `latent_dim=128`, `lr=1e-3`. W&B group: `v2_beta_scan`.

| beta  | best epoch | val_recon | val_kl | val_scale | val_centroid | epochs to val_recon=1.6e-5 |
|-------|-----------|-----------|--------|-----------|--------------|---------------------------|
| 0     | 440       | 9.9e-6    | 0.589  | 7.6e-5    | 8.8e-5       | 109                       |
| 1e-6  | 472       | 1.1e-5    | 0.615  | 5.6e-5    | 1.3e-4       | 134                       |
| 1e-5  | 493       | 1.3e-5    | 0.642  | 6.4e-5    | 1.1e-4       | **174**                   |
| 1e-4  | 440       | **9.4e-6**| 0.624  | **3.3e-5**| 1.4e-4       | **107**                   |

**Conclusions:**
- `beta=1e-4` is the winner: best val_recon and best val_scale, fastest convergence. Counterintuitive — stronger KL regularization helps rather than hurts on this dataset.
- `beta=1e-5` (previous default from v1 scans) is the worst here — slowest to converge and worst final reconstruction.
- `beta=0` has the lowest val_centroid but worse val_scale than 1e-4. Without KL, the latent space organizes scale better but suffers on centroid.
- All betas produce similar KL (~0.6) except beta=0 which is unconstrained.

---

## Scan 2 — Latent dim sweep (2026-04-27)

**Question:** How many latent dimensions does the model need? Is 256 the ceiling?

Fixed: `beta=1e-5`, `lr=1e-3`. W&B group: `v2_latent_dim_scan`.

| latent_dim | best epoch | val_recon  | val_kl | val_scale | val_centroid |
|------------|-----------|-----------|--------|-----------|--------------|
| 32         | 445       | 1.3e-5     | 1.496  | 6.4e-3    | 1.2e-3       |
| 64         | 443       | 9.3e-6     | 0.985  | 6.8e-5    | 6.0e-4       |
| 128        | 453       | 9.6e-6     | 0.621  | 6.1e-5    | 2.9e-4       |
| 256        | 435       | **7.9e-6** | **0.431** | **2.1e-5** | **7.3e-5** |

**Conclusions:**
- `dim=256` wins on every metric by a clear margin. The KL drop from 0.62 → 0.43 going from 128→256 indicates the larger model uses the latent space more efficiently.
- `dim=32` is clearly bottlenecked: val_scale is 100× worse than dim=256, KL=1.5 (posterior far from prior), and train/val scale ratio is only 1.9× (train and val both fail — capacity-limited, not overfit).
- `dim=64` has notably high centroid error (6.0e-4 vs 7.3e-5 at dim=256); centroid information needs more latent capacity.
- Whether 256 is a capacity ceiling or still scaling is the key open question — the falling KL suggests 512 might help further.

---

## Scan 3 — 2D grid: latent_dim × beta (submitted 2026-04-27, job 52161516)

**Question:** What is the joint optimum of latent_dim and beta on v2 data, and does the optimal beta shift as latent capacity increases?

**Motivation:** Scans 1 and 2 were run independently at fixed latent_dim=128 and beta=1e-5 respectively, so they don't tell us how the two hyperparameters interact. The Scan 1 winner (beta=1e-4) and Scan 2 winner (dim=256) have never been combined; the beta scan also only ran at dim=128. There are two open questions: (1) does beta=1e-4 remain the best choice at dim=256, or does the optimal beta shift as the latent space grows? (2) is dim=256 the capacity ceiling, or does dim=512 still improve? The Scan 2 KL trend (1.50 → 0.98 → 0.62 → 0.43 from dim=32→256, all at beta=1e-5) suggests the model has not saturated — each doubling of capacity still produces a meaningful KL drop, which typically tracks with better latent utilization and lower reconstruction error.

**Design:** 2D grid at latent_dim ∈ {256, 512} × beta ∈ {1e-4, 1e-3}. Fixed: `lr=1e-3`, `data=v2_sectioned_1sec_10k`. beta=1e-3 added to bracket the optimum — if beta=1e-4 already over-regularizes at dim=512, a stronger prior may hurt; if the model has more capacity to absorb the KL penalty, a larger beta could improve scale/centroid metrics further.

**What we expect:** `dim=256 + beta=1e-4` should improve on both Scan 1 and 2 individually — this is the straightforward combination of winners. If the KL trend continues, `dim=512` should further reduce both val_recon and val_kl. Whether beta=1e-3 helps or hurts likely depends on whether the model is still capacity-limited at that dim: with more room in the latent space, stronger regularization could encourage better-organized representations, or it could simply increase reconstruction error without a compensating gain in KL.

**What the result implies either way:** If dim=512 does not improve over dim=256, the model has plateaued and the bottleneck has shifted away from latent capacity (likely to dataset size or the fixed architecture). If beta=1e-4 remains the winner at both dims, the optimal regularization is robust to latent size and we can fix it for future runs. If beta shifts to 1e-3 at dim=512, it implies the latent space uses the extra capacity primarily for distributional structure rather than reconstruction detail.

W&B group: `v2_grid_latent_beta`.

| latent_dim | beta=1e-4 | beta=1e-3 |
|------------|-----------|-----------|
| 256        | pending   | pending   |
| 512        | pending   | pending   |

---

## Open Questions

- **Is dim=512 still capacity-limited?** The KL trend (1.50 → 0.98 → 0.62 → 0.43 from dim=32→256) suggests the model hasn't plateaued. Pending grid result will answer this.
- **Why does stronger beta help on v2 but not v1?** On the v1 dataset, beta=1e-5 was the preferred value. The v2 dataset (single-section, v2 generation constraints) may have a different latent structure. Worth revisiting once the grid results are in.
- **Large train/val ratio on scale/centroid.** All runs show 10–40× higher training error than validation on scale/centroid metrics. This is the opposite of typical overfitting and is likely a logging artifact (train metrics computed on mini-batches mid-epoch, val on a clean full pass). Not a reliability concern given the small absolute val numbers.
