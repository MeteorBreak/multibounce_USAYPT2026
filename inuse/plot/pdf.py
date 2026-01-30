import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_pdf_heatmap(csv_path, invert=False):
    """
    Reads coordinates from a CSV and plots a 2D Probability Density Function heatmap.
    Input CSV is expected to have at least 2 columns (X, Y).
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)

    try:
        # Load data. Attempt to handle headers dynamically.
        # First read with header=None to inspect first row
        # using low_memory=False to act more like a standard file read on small files
        try:
            df_raw = pd.read_csv(csv_path, header=None)
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty.")
            sys.exit(1)

        if df_raw.empty:
            print("Error: CSV file is empty.")
            sys.exit(1)

        # Check if 1st row is likely a header (contains non-numeric)
        is_header = False
        try:
            # Check first two columns of first row
            pd.to_numeric(df_raw.iloc[0, 0])
            pd.to_numeric(df_raw.iloc[0, 1])
        except (ValueError, TypeError, IndexError):
            # If conversion fails, assume it's a header
            is_header = True
        
        if is_header:
            df = pd.read_csv(csv_path) # Reload with header
        else:
            df = df_raw
            
        # Ensure we have at least 2 columns
        if df.shape[1] < 2:
            print(f"Error: CSV file must contain at least 2 columns (X, Y). Found {df.shape[1]} columns.")
            sys.exit(1)
            
        # Extract X and Y, convert to numeric, drop NaNs
        # We assume the first two columns are X and Y
        # coerce errors to NaN, then drop rows with any NaNs in the first two columns
        df_clean = df.iloc[:, [0, 1]].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_clean) < 2:
             print("Error: Not enough valid numeric data points (>= 2 required) to plot density.")
             sys.exit(1)
             
        if invert:
            x = df_clean.iloc[:, 1].values
            y = df_clean.iloc[:, 0].values
        else:
            x = df_clean.iloc[:, 0].values
            y = df_clean.iloc[:, 1].values

        # Calculate KDE (Kernel Density Estimation)
        # gaussian_kde expects shape (dims, n_points)
        values = np.vstack([x, y])
        
        try:
            kernel = gaussian_kde(values)
        except np.linalg.LinAlgError:
            print("Error: Singular matrix in KDE calculation. Points might be collinear or identical (variance is zero).")
            sys.exit(1)
        except ValueError as ve:
             print(f"Error in KDE calculation: {ve}")
             sys.exit(1)

        # Define grid for plotting
        # Fixed range 160x160 (-80 to 80 centered) as requested
        limit = 100
        xmin_grid, xmax_grid = -limit, limit
        ymin_grid, ymax_grid = -limit, limit

        # Grid resolution

        resolution = 100j
        X, Y = np.mgrid[xmin_grid:xmax_grid:resolution, ymin_grid:ymax_grid:resolution]
        
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)

        # Normalize and apply modest scaling/gamma correction to improve visibility
        # 1. Normalize to 0-1 relative to peak density
        z_max = Z.max() if Z.max() > 0 else 1.0
        Z_norm = Z / z_max
        
        # 2. Apply Gamma correction (power < 1) to lift lower values and brighten the plot
        #    Gamma = 0.6 is a "moderate" scaling to make regions visible
        gamma = 0.6 
        Z_plot = np.power(Z_norm, gamma)

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Heatmap (filled contour)
        # 0 to 1 range after normalization
        cf = plt.contourf(X, Y, Z_plot, levels=60, cmap='viridis')
        cbar = plt.colorbar(cf)
        cbar.set_label(f'Relative Density (Normalized, $\gamma$={gamma})')

        # Overlay data points
        plt.scatter(x, y, s=15, color='red', alpha=0.6, label='Data Points', edgecolors='white', linewidths=0.6)

        plt.title(f'2D Probability Density Function\nSource: {os.path.basename(csv_path)}')
        plt.xlabel('X Coordinate (mm)')
        plt.ylabel('Y Coordinate (mm)')
        plt.legend(loc='upper right')
        
        # Enforce fixed 160x160 aligned view (-80 to 80)
        plt.xlim(xmin_grid, xmax_grid)
        plt.ylim(ymin_grid, ymax_grid)
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.grid(True, linestyle='--', alpha=0.3)

        print(f"Plotting PDF for {len(x)} points from '{csv_path}'...")
        
        # Save figure to the same directory as input CSV
        output_png = os.path.splitext(csv_path)[0] + '_pdf.png'
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Saved plot image to '{output_png}'")

        plt.show()

    except Exception as e:
        print(f"Unexpected error: {e}")
        # optional: print full traceback for debugging
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 2D Probability Density Function from CSV coordinates.')
    parser.add_argument('csv_path', type=str, help='Relative path to the input CSV file.')
    parser.add_argument("-i", "--invert", action="store_true", help="Invert (swap) X and Y axes")
    args = parser.parse_args()

    plot_pdf_heatmap(args.csv_path, invert=args.invert)
