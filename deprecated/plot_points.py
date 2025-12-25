#!/usr/bin/env python3
"""
Plot 2D points from three CSV files onto a single figure with distinct colors per file.

- Expected input directory: 25_10_23 containing b_1.csv, b_2.csv, b_3.csv
- Each CSV: 32 pairs of x,y coordinates. Headers optional.
- Output image: 25_10_23/combined_points.png

Usage:
  python plot_points.py [DATA_DIR]

Where DATA_DIR defaults to the sibling folder "25_10_23" next to this script.

Notes:
- This script uses a non-interactive matplotlib backend (Agg) so it runs headless and
  always saves the figure to disk.
- It will tolerate a single header row or stray whitespace; rows that cannot be parsed
  as two floats are skipped with a warning to stderr.
"""
from __future__ import annotations

import csv
import sys
import os
from pathlib import Path
from typing import List, Tuple

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
            # Normalize and skip empty strings
            cells = [c.strip() for c in row if c is not None]
            if len(cells) < 2:
                sys.stderr.write(f"Warning: {path.name}: line {i} has fewer than 2 columns; skipped.\n")
                continue
            try:
                x = float(cells[0])
                y = float(cells[1])
                points.append((x, y))
            except ValueError:
                # Likely a header; be forgiving and only warn for the first unparsable line
                sys.stderr.write(f"Warning: {path.name}: line {i} not numeric; skipped.\n")
                continue
    return points


def main(argv: List[str]) -> int:
    # Determine data directory
    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir / "25_10_23"
    data_dir = Path(argv[1]).resolve() if len(argv) > 1 else default_dir

    # Input files
    files = [data_dir / "b_1.csv", data_dir / "b_2.csv", data_dir / "b_3.csv"]
    missing = [p for p in files if not p.exists()]
    if missing:
        sys.stderr.write("Error: Missing expected CSV files:\n" + "\n".join(f"  - {m}" for m in missing) + "\n")
        return 1

    # Colors for three files
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Read and plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    total_points = 0
    for idx, path in enumerate(files):
        pts = read_points_csv(path)
        total_points += len(pts)
        if not pts:
            sys.stderr.write(f"Warning: {path.name}: no valid points parsed.\n")
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=36, color=colors[idx], label=path.name, alpha=0.9, edgecolors="white", linewidths=0.5)

    # Axes formatting
    ax.set_title("Points from b_1.csv, b_2.csv, b_3.csv")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")

    # Tight layout and save
    out_path = data_dir / "combined_points.png"
    fig.tight_layout()
    fig.savefig(out_path)

    # Also print a short summary to stdout
    print(f"Saved plot with {total_points} points to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
