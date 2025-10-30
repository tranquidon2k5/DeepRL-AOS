#!/usr/bin/env python3
"""
Multi-seed aggregation & visualization for PPO-CEC logs (.npz)

Features
- Load one or many .npz runs (same experiment) and aggregate across seeds
- Compute per-update stats across seeds: median, IQR (q25–q75), mean, std
- Plot scalar metrics: best, loss, alpha_sum
- Plot vector metrics: probs, p_used, op_improve (per-operator curves)
- Export PNGs to outdir and a single summary CSV with both
  * across-seeds per-update stats
  * per-seed summary across updates (mean, std, last)

Usage examples (PowerShell)
    python scripts/plot_multi_seed.py --inputs "logs/cec_f1_seed*.npz" --outdir plots/f1
    foreach ($f in 1,5,10) {
      python scripts/plot_multi_seed.py --inputs "logs/cec_f${f}_seed*.npz" --outdir "plots/f${f}"
    }

Assumptions
- Each .npz has arrays saved by scripts/train_ppo_cec.py (updates, loss, best, ...)
- All runs in a group share the same operator count (num_ops) and similar length; we align to the min length across seeds.
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

# -------------- Helpers --------------

def find_group_and_seed(path: str) -> Tuple[str, str]:
    """Infer a group key (e.g., f1/f5/f10) and seed from filename."""
    base = os.path.basename(path).lower()
    g = re.search(r"(?:cec|ppo_cec)_(f\d+)", base)
    group = g.group(1) if g else "all"
    s = re.search(r"seed(\d+)", base)
    seed = s.group(1) if s else os.path.splitext(base)[0]
    return group, seed


def to2d_object_series(obj_series: np.ndarray) -> np.ndarray:
    """Convert dtype=object array of vectors (len U) -> (U, K)."""
    return np.stack([np.asarray(v, dtype=float) for v in obj_series], axis=0)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def nan_stats_across_seeds(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Given x shape (S, U[, K]), compute stats across axis=0 (seeds)."""
    q25 = np.nanquantile(x, 0.25, axis=0)
    med = np.nanmedian(x, axis=0)
    q75 = np.nanquantile(x, 0.75, axis=0)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    return {"q25": q25, "median": med, "q75": q75, "mean": mean, "std": std}


def plot_scalar(upd: np.ndarray, stats: Dict[str, np.ndarray], out_png: str, title: str, per_seed: List[np.ndarray] = None):
    plt.figure(figsize=(8, 5))
    if per_seed:
        for y in per_seed:
            plt.plot(upd, y, alpha=0.25, linewidth=1)
    plt.fill_between(upd, stats["q25"], stats["q75"], alpha=0.2, linewidth=0)
    plt.plot(upd, stats["median"], linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel("update")
    plt.ylabel(os.path.splitext(os.path.basename(out_png))[0])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_vector_per_op(upd: np.ndarray, stats: Dict[str, np.ndarray], out_prefix: str, title_prefix: str, per_seed: List[np.ndarray] = None):
    """Plot each operator as its own figure.
    stats arrays have shape (U, K). per_seed elements have shape (U, K).
    """
    U, K = stats["median"].shape
    for k in range(K):
        out_png = f"{out_prefix}_op{k}.png"
        plt.figure(figsize=(8, 5))
        if per_seed:
            for y in per_seed:
                plt.plot(upd, y[:, k], alpha=0.25, linewidth=1)
        plt.fill_between(upd, stats["q25"][:, k], stats["q75"][:, k], alpha=0.2, linewidth=0)
        plt.plot(upd, stats["median"][:, k], linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel("update")
        plt.ylabel(os.path.basename(out_prefix))
        plt.title(f"{title_prefix} · op{k}")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()


# -------------- Core --------------

def load_runs(paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load npz files and group by function key.
    Returns: dict[group] -> list of runs, each run is dict with arrays and metadata.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in paths:
        try:
            data = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Skip {p}: {e}")
            continue
        group, seed = find_group_and_seed(p)
        run = {
            "path": p,
            "seed": seed,
            "updates": np.asarray(data["updates"], dtype=int),
            "loss": np.asarray(data["loss"], dtype=float),
            "best": np.asarray(data["best"], dtype=float),
            "alpha_sum": np.asarray(data["alpha_sum"], dtype=float),
            "probs": to2d_object_series(data["probs"]),
            "p_used": to2d_object_series(data["p_used"]),
            "op_improve": to2d_object_series(data["op_improve"]),
            "num_ops": int(data["num_ops"]),
        }
        groups[group].append(run)
    return groups


def align_truncate(runs: List[Dict[str, Any]]):
    """Truncate all runs to the minimum number of updates found so arrays align."""
    min_len = min(len(r["updates"]) for r in runs)
    for r in runs:
        for key in ["updates", "loss", "best", "alpha_sum"]:
            r[key] = r[key][:min_len]
        for key in ["probs", "p_used", "op_improve"]:
            r[key] = r[key][:min_len, :]
    return min_len


def aggregate_and_plot(group: str, runs: List[Dict[str, Any]], outdir: str, title_hint: str = "", show_per_seed=False) -> List[Dict[str, Any]]:
    ensure_dir(outdir)
    align_truncate(runs)

    seeds = [r["seed"] for r in runs]
    upd = runs[0]["updates"]
    S = len(runs)

    # Stack across seeds
    scalars = {}
    vectors = {}

    def stack_scalar(key):
        return np.stack([r[key] for r in runs], axis=0)  # (S, U)

    def stack_vector(key):
        return np.stack([r[key] for r in runs], axis=0)  # (S, U, K)

    scalars["loss"] = stack_scalar("loss")
    scalars["best"] = stack_scalar("best")
    scalars["alpha_sum"] = stack_scalar("alpha_sum")

    vectors["probs"] = stack_vector("probs")
    vectors["p_used"] = stack_vector("p_used")
    vectors["op_improve"] = stack_vector("op_improve")

    # Compute stats across seeds
    scalar_stats = {k: nan_stats_across_seeds(v) for k, v in scalars.items()}
    vector_stats = {k: nan_stats_across_seeds(v) for k, v in vectors.items()}

    # Plots
    prefix = f"{group}" if group else "all"
    title_base = f"{title_hint} {group} (S={S})".strip()

    for m in ["best", "loss", "alpha_sum"]:
        out_png = os.path.join(outdir, f"{prefix}_{m}.png")
        per_seed = [scalars[m][i] for i in range(S)] if show_per_seed else None
        plot_scalar(upd, scalar_stats[m], out_png, f"{title_base} · {m}", per_seed)

    for m in ["probs", "p_used", "op_improve"]:
        out_prefix = os.path.join(outdir, f"{prefix}_{m}")
        per_seed = [vectors[m][i] for i in range(S)] if show_per_seed else None
        plot_vector_per_op(upd, vector_stats[m], out_prefix, f"{title_base} · {m}", per_seed)

    # Summaries to return for CSV
    summaries: List[Dict[str, Any]] = []

    # Across-seeds per-update stats
    for m, st in scalar_stats.items():
        for i, u in enumerate(upd):
            summaries.append({
                "scope": "across_seeds",
                "group": group,
                "metric": m,
                "op": "",
                "update": int(u),
                "n_seeds": S,
                "q25": float(st["q25"][i]),
                "median": float(st["median"][i]),
                "q75": float(st["q75"][i]),
                "mean": float(st["mean"][i]),
                "std": float(st["std"][i]),
            })
    for m, st in vector_stats.items():
        U, K = st["median"].shape
        for i, u in enumerate(upd):
            for k in range(K):
                summaries.append({
                    "scope": "across_seeds",
                    "group": group,
                    "metric": m,
                    "op": f"op{k}",
                    "update": int(u),
                    "n_seeds": S,
                    "q25": float(st["q25"][i, k]),
                    "median": float(st["median"][i, k]),
                    "q75": float(st["q75"][i, k]),
                    "mean": float(st["mean"][i, k]),
                    "std": float(st["std"][i, k]),
                })

    # Per-seed summaries across updates (mean, std, last)
    for r in runs:
        sid = r["seed"]
        for m in ["best", "loss", "alpha_sum"]:
            y = r[m]
            summaries.append({
                "scope": "per_seed",
                "group": group,
                "seed": sid,
                "metric": m,
                "op": "",
                "update": -1,
                "n_seeds": 1,
                "q25": np.nan,
                "median": np.nan,
                "q75": np.nan,
                "mean": float(np.nanmean(y)),
                "std": float(np.nanstd(y)),
                "last": float(y[-1]),
            })
        for m in ["probs", "p_used", "op_improve"]:
            y = r[m]  # (U, K)
            U, K = y.shape
            for k in range(K):
                col = y[:, k]
                summaries.append({
                    "scope": "per_seed",
                    "group": group,
                    "seed": sid,
                    "metric": m,
                    "op": f"op{k}",
                    "update": -1,
                    "n_seeds": 1,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "mean": float(np.nanmean(col)),
                    "std": float(np.nanstd(col)),
                    "last": float(col[-1]),
                })

    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Glob(s) or file paths for .npz logs")
    ap.add_argument("--outdir", default="plots", help="Output directory for PNGs & CSV")
    ap.add_argument("--title", default="PPO-CEC", help="Title prefix for plots")
    ap.add_argument("--show_per_seed", action="store_true", help="Overlay faint per-seed lines")
    args = ap.parse_args()

    # Expand globs
    paths: List[str] = []
    for pat in args.inputs:
        found = glob.glob(pat)
        if not found and os.path.isfile(pat):
            found = [pat]
        paths.extend(found)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No input files found. Check --inputs pattern.")

    runs_by_group = load_runs(paths)

    ensure_dir(args.outdir)

    all_summaries: List[Dict[str, Any]] = []
    n_figs = 0

    for group, runs in sorted(runs_by_group.items()):
        group_out = os.path.join(args.outdir, group)
        summaries = aggregate_and_plot(group, runs, group_out, args.title, show_per_seed=args.show_per_seed)
        all_summaries.extend(summaries)
        # Heuristic count of figures saved: 3 scalars + 3*K vectors per group
        K = runs[0]["num_ops"] if runs else 0
        n_figs += 3 + 3 * K

    # Write summary CSV (single file across all groups)
    out_csv = os.path.join(args.outdir, "summary.csv")
    fieldnames = [
        "scope", "group", "seed", "metric", "op", "update", "n_seeds",
        "q25", "median", "q75", "mean", "std", "last"
    ]
    # Fill missing keys with blanks/NaNs
    for row in all_summaries:
        row.setdefault("seed", "")
        row.setdefault("last", np.nan)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in all_summaries:
            wr.writerow(row)

    print(f"Saved figures and summary to: {args.outdir}")
    print(f"Total figures (approx): {n_figs}")


if __name__ == "__main__":
    main()
