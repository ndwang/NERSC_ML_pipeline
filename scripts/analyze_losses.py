#!/usr/bin/env python
"""Compare training runs from a sweep.

Usage:
    # Compare all runs matching a prefix
    python scripts/compare_runs.py runs/beta_*_260401_*

    # Compare specific runs
    python scripts/compare_runs.py runs/lr_2e-4_260401_1523 runs/lr_5e-4_260401_1523

    # Sort by a specific metric
    python scripts/compare_runs.py runs/beta_*_260401_* --sort val_recon

    # Show convergence speed
    python scripts/compare_runs.py runs/lr_*_260401_* --convergence

    # Show loss trajectory at key epochs
    python scripts/compare_runs.py runs/beta_*_260401_* --trajectory
"""

import argparse
import csv
import sys
from pathlib import Path


def load_history(run_dir):
    """Load history CSV from a run directory. Returns list of row dicts."""
    run_dir = Path(run_dir)
    csvs = list(run_dir.glob("*_history.csv"))
    if not csvs:
        return None
    with open(csvs[0]) as f:
        return list(csv.DictReader(f))


def best_epoch(rows, metric="val_total"):
    """Return the row with the minimum value of the given metric."""
    return min(rows, key=lambda r: float(r[metric]))


def print_summary(run_dirs, sort_by="val_total"):
    """Print best-epoch summary table for all runs."""
    results = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            print(f"  {Path(d).name}: no history CSV found, skipping")
            continue
        best = best_epoch(rows)
        results.append((Path(d).name, best, rows))

    if not results:
        print("No valid runs found.")
        return

    # Sort
    results.sort(key=lambda r: float(r[1][sort_by]))

    # Determine available fields
    fields = ["val_total", "val_recon", "val_kl", "val_scale", "val_centroid"]
    train_fields = ["train_scale", "train_centroid"]

    # Header
    header = f"{'run':<35} {'epoch':>5}"
    for f in fields:
        header += f"  {f:>14}"
    for f in train_fields:
        header += f"  {f:>15}"
    print(header)
    print("-" * len(header))

    for name, best, _ in results:
        line = f"{name:<35} {best['epoch']:>5}"
        for f in fields:
            line += f"  {float(best[f]):>14.7f}"
        for f in train_fields:
            line += f"  {float(best[f]):>15.7f}"
        print(line)


def print_convergence(run_dirs, metric="val_recon"):
    """Print epochs to reach various thresholds."""
    # First pass: find thresholds from the data
    all_best = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            continue
        best_val = float(best_epoch(rows, metric)[metric])
        all_best.append(best_val)

    if not all_best:
        print("No valid runs found.")
        return

    # Auto-generate thresholds: 10x, 5x, 2x, 1.5x best
    global_best = min(all_best)
    thresholds = sorted(set([
        round(global_best * 10, -int(f"{global_best * 10:.1e}".split("e")[1]) + 1),
        round(global_best * 5, -int(f"{global_best * 5:.1e}".split("e")[1]) + 1),
        round(global_best * 2, -int(f"{global_best * 2:.1e}".split("e")[1]) + 1),
    ]), reverse=True)

    # Header
    header = f"{'run':<35}"
    for t in thresholds:
        header += f"  {t:<12.1e}"
    header += f"  {'final':>12}"
    print(f"Convergence: epochs to reach {metric} threshold")
    print(header)
    print("-" * len(header))

    results = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            continue
        name = Path(d).name
        final = float(rows[-1][metric])
        hits = []
        for t in thresholds:
            hit = None
            for r in rows:
                if float(r[metric]) <= t:
                    hit = int(r["epoch"])
                    break
            hits.append(hit)
        results.append((name, hits, final))

    # Sort by first threshold reached (earliest = best)
    results.sort(key=lambda r: r[1][-1] if r[1][-1] is not None else 99999)

    for name, hits, final in results:
        line = f"{name:<35}"
        for h in hits:
            line += f"  {str(h) if h else '>'+str(int(rows[-1]['epoch'])):<12}"
        line += f"  {final:>12.7f}"
        print(line)


def print_trajectory(run_dirs, epochs=None):
    """Print loss values at key epochs for all runs."""
    if epochs is None:
        epochs = [1, 10, 50, 100, 200, 300, 400, 500]

    fields = ["val_recon", "val_scale", "val_centroid", "val_kl"]

    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            continue
        name = Path(d).name
        max_ep = int(rows[-1]["epoch"])
        print(f"=== {name} ===")
        header = f"  {'epoch':>5}"
        for f in fields:
            header += f"  {f:>14}"
        print(header)
        for ep in epochs:
            if ep > max_ep:
                continue
            r = rows[ep - 1]
            line = f"  {int(r['epoch']):>5}"
            for f in fields:
                line += f"  {float(r[f]):>14.7f}"
            print(line)
        print()


def print_overfitting(run_dirs):
    """Print train/val ratios at best epoch to check for overfitting."""
    pairs = [("train_scale", "val_scale"), ("train_centroid", "val_centroid")]

    header = f"{'run':<35} {'epoch':>5}"
    for train_f, val_f in pairs:
        name = train_f.replace("train_", "")
        header += f"  {'train_'+name:>13}  {'val_'+name:>13}  {'ratio':>7}"
    print("Overfitting check (train/val ratio at best epoch)")
    print(header)
    print("-" * len(header))

    results = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            continue
        best = best_epoch(rows)
        results.append((Path(d).name, best))

    results.sort(key=lambda r: float(r[1]["val_total"]))

    for name, best in results:
        line = f"{name:<35} {best['epoch']:>5}"
        for train_f, val_f in pairs:
            tv = float(best[train_f])
            vv = float(best[val_f])
            ratio = tv / vv if vv > 0 else float("inf")
            line += f"  {tv:>13.7f}  {vv:>13.7f}  {ratio:>7.1f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Compare training runs from a sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")
    parser.add_argument("--sort", default="val_total",
                        help="Metric to sort summary by (default: val_total)")
    parser.add_argument("--convergence", action="store_true",
                        help="Show convergence speed analysis")
    parser.add_argument("--trajectory", action="store_true",
                        help="Show loss trajectory at key epochs")
    parser.add_argument("--overfitting", action="store_true",
                        help="Show train/val ratio analysis")
    parser.add_argument("--all", action="store_true",
                        help="Show all analyses")
    args = parser.parse_args()

    # Expand globs (shell should handle this, but just in case)
    run_dirs = [Path(d) for d in args.run_dirs if Path(d).is_dir()]
    if not run_dirs:
        print("No valid run directories found.")
        sys.exit(1)

    run_dirs.sort(key=lambda d: d.name)
    print(f"Comparing {len(run_dirs)} runs\n")

    show_all = args.all or not (args.convergence or args.trajectory or args.overfitting)

    if show_all or not (args.convergence or args.trajectory or args.overfitting):
        print_summary(run_dirs, sort_by=args.sort)
        print()

    if args.convergence or args.all:
        print_convergence(run_dirs)
        print()

    if args.overfitting or args.all:
        print_overfitting(run_dirs)
        print()

    if args.trajectory or args.all:
        print_trajectory(run_dirs)


if __name__ == "__main__":
    main()
