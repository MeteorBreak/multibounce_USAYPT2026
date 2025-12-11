#!/usr/bin/env python3
"""
Plot 2D points from three CSV files onto a single figure with distinct colors per file,
after aligning each group's points to a common center for clearer presentation of central tendency.

- Expected input directory: 25_10_23 containing b_1.csv, b_2.csv, b_3.csv
- Each CSV: 32 pairs of x,y coordinates. Headers optional.
- Output image: 25_10_23/combined_points_aligned.png

Usage:
  python plot_points_aligned.py [DATA_DIR] [--method mean|median]

Where DATA_DIR defaults to the sibling folder "25_10_23" next to this script.

Notes:
- Uses a headless matplotlib backend (Agg), so it always saves the figure to disk.
- Points from each file are individually centered by their own mean/median and overlaid at (0, 0).
- This highlights spread/shape while ignoring absolute offsets between files.
"""
from __future__ import annotations

import csv
import sys
import os
from pathlib import Path
from typing import List, Tuple
import argparse
import statistics

# Use a headless backend so this works on servers or without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_points_csv(path: Path) -> List[Tuple[float, float]]:
    """Read pairs of floats (x, y) from a CSV file.

    - Accepts optional header row (will be skipped if it can't parse as floats).
    - Skips rows that don't have at least two columns.
    - Trims whitespace.
    """
    points: List[Tuple[float, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            cells = [c.strip() for c in row if c is not None]
            if len(cells) < 2:
                sys.stderr.write(f"Warning: {path.name}: line {i} has fewer than 2 columns; skipped.\n")
                continue
            try:
                x = float(cells[0])
                y = float(cells[1])
                points.append((x, y))
            except ValueError:
                sys.stderr.write(f"Warning: {path.name}: line {i} not numeric; skipped.\n")
                continue
    return points


def compute_center(points: List[Tuple[float, float]], method: str = "mean") -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    xs, ys = zip(*points)
    if method == "median":
        return (statistics.median(xs), statistics.median(ys))
    # default mean
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def align_points(points: List[Tuple[float, float]], center: Tuple[float, float]) -> List[Tuple[float, float]]:
    cx, cy = center
    return [(x - cx, y - cy) for (x, y) in points]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Plot aligned points from 3 CSV files")
    parser.add_argument("data_dir", nargs="?", default=None, help="Directory containing b_1.csv, b_2.csv, b_3.csv (default: 25_10_23 next to this script)")
    parser.add_argument("--method", choices=["mean", "median"], default="mean", help="Centering method per group (default: mean)")
    args = parser.parse_args(argv[1:])

    # Determine data directory
    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir / "25_10_23"
    data_dir = Path(args.data_dir).resolve() if args.data_dir else default_dir

    # Input files
    files = [data_dir / "b_1.csv", data_dir / "b_2.csv", data_dir / "b_3.csv"]
    missing = [p for p in files if not p.exists()]
    if missing:
        sys.stderr.write("Error: Missing expected CSV files:\n" + "\n".join(f"  - {m}" for m in missing) + "\n")
        return 1

    # Colors for three files
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Read, center, and plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    total_points = 0
    for idx, path in enumerate(files):
        pts = read_points_csv(path)
        total_points += len(pts)
        if not pts:
            sys.stderr.write(f"Warning: {path.name}: no valid points parsed.\n")
            continue
        center = compute_center(pts, method=args.method)
        aligned = align_points(pts, center)
        xs, ys = zip(*aligned)
        ax.scatter(xs, ys, s=36, color=colors[idx], label=f"{path.name} (centered)", alpha=0.9, edgecolors="white", linewidths=0.5)

    # Draw a faint crosshair at the common center (0,0)
    ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":", zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.8, linestyle=":", zorder=0)

    # Axes formatting
    title_method = "Mean" if args.method == "mean" else "Median"
    ax.set_title(f"Aligned points by {title_method} center (b_1.csv, b_2.csv, b_3.csv)")
    ax.set_xlabel("x (centered)")
    ax.set_ylabel("y (centered)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")

    # Tight layout and save
    out_path = data_dir / "combined_points_aligned.png"
    fig.tight_layout()
    fig.savefig(out_path)

    print(f"Saved aligned plot ({args.method}) with {total_points} points to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
