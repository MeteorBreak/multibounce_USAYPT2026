#!/usr/bin/env python3
"""
Plot 2D points from multiple CSV files onto a single figure with distinct colors per file,
after aligning each group's points to a common center.
Calculates and displays statistical metrics (Variance, RMS spread).

Usage:
  python plot_points_aligned.py [FILE1] [FILE2] ... [--method mean|median]
"""
from __future__ import annotations

import csv
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
import statistics
import math
import random

scaler = 1 # 1000 for m to mm, 1 for m to m

# Try importing numpy for advanced stats
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def read_points_csv(path: Path) -> List[Tuple[float, float]]:
    """Read pairs of floats (x, y) from a CSV file."""
    points: List[Tuple[float, float]] = []
    if not path.exists():
        sys.stderr.write(f"Warning: File not found: {path}\n")
        return points
        
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            cells = [c.strip() for c in row if c is not None]
            if len(cells) < 2:
                continue
            try:
                x = float(cells[0])
                y = float(cells[1])
                points.append((x, y))
            except ValueError:
                continue
    return points

def compute_stats(points: List[Tuple[float, float]], method: str = "mean") -> Dict[str, Any]:
    if not points:
        return {}
    
    if HAS_NUMPY:
        arr = np.array(points)
        if method == "median":
            center = np.median(arr, axis=0)
        else:
            center = np.mean(arr, axis=0)
            
        centered = arr - center
        
        # Variance and Std Dev
        var = np.var(arr, axis=0)
        std = np.std(arr, axis=0)
        
        # Radial stats
        dists = np.linalg.norm(centered, axis=1)
        rms_r = np.sqrt(np.mean(dists**2))
        
        # Covariance for ellipse
        cov = np.cov(arr.T)
        
        return {
            "center": tuple(center),
            "var": tuple(var),
            "std": tuple(std),
            "rms_r": rms_r,
            "cov": cov,
            "count": len(points)
        }
    else:
        # Fallback to standard library
        xs, ys = zip(*points)
        if method == "median":
            cx, cy = statistics.median(xs), statistics.median(ys)
        else:
            cx, cy = statistics.mean(xs), statistics.mean(ys)
            
        # Variance/Std
        if len(points) > 1:
            vx = statistics.variance(xs)
            vy = statistics.variance(ys)
            sx = statistics.stdev(xs)
            sy = statistics.stdev(ys)
        else:
            vx, vy, sx, sy = 0.0, 0.0, 0.0, 0.0
            
        # Radial
        sq_dists = [(x-cx)**2 + (y-cy)**2 for x, y in points]
        rms_r = math.sqrt(sum(sq_dists) / len(points))
        
        return {
            "center": (cx, cy),
            "var": (vx, vy),
            "std": (sx, sy),
            "rms_r": rms_r,
            "cov": None,
            "count": len(points)
        }

def draw_confidence_ellipse(ax, center, cov, n_std=2.0, facecolor='none', **kwargs):
    """
    Draw a confidence ellipse of a 2D dataset.
    """
    if cov is None:
        return
        
    # Calculate eigenvalues and eigenvectors
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    # Ellipse geometry
    width = lambda_[0] * n_std * 2
    height = lambda_[1] * n_std * 2
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    
    ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                  facecolor=facecolor, **kwargs)
    ax.add_patch(ell)

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Plot aligned points from multiple CSV files with stats.")
    parser.add_argument("files", nargs="*", help="Paths to CSV files to analyze")
    parser.add_argument("--method", choices=["mean", "median"], default="mean", help="Centering method (default: mean)")
    parser.add_argument("--random", nargs=9, type=float, metavar='ERR',
                        help="Generate 3 groups of 20 random points from 9 error inputs (m). "
                             "Supply 9 floats: (Par1, Norm, Par2) for Plane 1, then Plane 2, then Plane 3. "
                             "Applies err_p + 0.2*err_n logic. OVERRIDES file input.")
    args = parser.parse_args(argv[1:])

    # Prepare data sources: List of (label, points)
    data_sources: List[Tuple[str, List[Tuple[float, float]]]] = []
    out_dir = Path.cwd()

    if args.random:
        if args.files:
            print("Info: --random specified, ignoring input files.")
        
        inputs = args.random
        # Split into 3 groups of 3
        groups = [inputs[i:i+3] for i in range(0, 9, 3)]
        
        print("Simulation Mode: Processing 3 planes (Input m -> Plot mm)...")
        
        for i, (e_p1, e_norm, e_p2) in enumerate(groups, start=1):
            # Parse error inputs
            # "For the input's second error, i.e., normal error, multiply it by 0.2 
            # and add it directly to the other two directions"
            eff_p1 = e_p1 + 0.2 * e_norm
            eff_p2 = e_p2 + 0.2 * e_norm
            
            print(f"  Plane {i}: Inputs ({e_p1}, {e_norm}, {e_p2}) -> Effective Bounds: X={eff_p1:.5f}m, Y={eff_p2:.5f}m")
            
            # Generator: random 20 points
            # Using uniform distribution [-eff, +eff]
            sim_points = []
            for _ in range(20):
                rx = random.uniform(-eff_p1, eff_p1)
                ry = random.uniform(-eff_p2, eff_p2)
                sim_points.append((rx, ry))
                
            data_sources.append((f"Sim_Plane_{i}", sim_points))
        
    else:
        if not args.files:
            parser.print_help()
            return 1
            
        files = [Path(f).resolve() for f in args.files]
        if files:
            out_dir = files[0].parent
            
        for path in files:
            pts = read_points_csv(path)
            # Filter empty results here or inside loop
            data_sources.append((path.name, pts))
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    cmap = plt.get_cmap("tab10")
    
    print(f"{'File':<40} | {'N':<3} | {'Std X (mm)':<12} | {'Std Y (mm)':<12} | {'RMS R (mm)':<12}")
    print("-" * 90)

    for idx, (name, pts_m) in enumerate(data_sources):
        if not pts_m:
            print(f"{name:<40} | 0   | -            | -            | -")
            continue
            
        # Convert to mm for plotting and stats
        pts = [(x * scaler, y * scaler) for x, y in pts_m]
        
        stats = compute_stats(pts, method=args.method)
        center = stats["center"]
        
        # Align points for plotting
        aligned = [(x - center[0], y - center[1]) for x, y in pts]
        xs, ys = zip(*aligned)
        
        color = cmap(idx % 10)
        
        # Plot points
        label = f"{name}\n$\sigma_x={stats['std'][0]:.2f}, \sigma_y={stats['std'][1]:.2f}$ mm\n$RMS={stats['rms_r']:.2f}$ mm"
        ax.scatter(xs, ys, s=30, color=color, label=label, alpha=0.7, edgecolors="white", linewidths=0.5)
        
        # Draw ellipse (2-sigma)
        if HAS_NUMPY and stats["cov"] is not None:
            # We draw the ellipse centered at (0,0) because we plotted aligned points
            draw_confidence_ellipse(ax, (0,0), stats["cov"], n_std=2.0, edgecolor=color, linestyle='--', linewidth=1.5)

        # Print to terminal
        print(f"{name:<40} | {stats['count']:<3} | {stats['std'][0]:<12.4f} | {stats['std'][1]:<12.4f} | {stats['rms_r']:<12.4f}")

    ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":", zorder=0)
    ax.axvline(0, color="0.5", linewidth=0.8, linestyle=":", zorder=0)
    
    ax.set_title(f"Aligned Points & Error Analysis ({args.method} centered)")
    ax.set_xlabel("x (centered) [mm]")
    ax.set_ylabel("y (centered) [mm]")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    
    # Save output
    out_path = out_dir / "combined_analysis.png"
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    print(f"\nSaved analysis plot to: {out_path}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
