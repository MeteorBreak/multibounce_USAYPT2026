import pandas as pd
import numpy as np
import argparse
import sys
import os

def generate_points(csv_rel_path, n):
    # Construct absolute path or use relative
    if not os.path.exists(csv_rel_path):
        print(f"Error: File '{csv_rel_path}' not found.")
        sys.exit(1)

    # Load data
    try:
        # Assuming no header first
        df = pd.read_csv(csv_rel_path, header=None)
        
        # Check if first row contains non-numeric chunks (indicating a header)
        # We try to convert the first row of the first 2 columns to float
        try:
            df.iloc[0, :2].astype(float)
            # If successful, likely no header, or header is numbers (unlikely)
            data = df.iloc[:, :2].astype(float).values
        except ValueError:
            # Conversion failed, assume first row is header
            df = pd.read_csv(csv_rel_path)
            data = df.iloc[:, :2].astype(float).values
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if len(data) < 2:
        print("Not enough data points to fit distribution.")
        sys.exit(1)

    # Fit Gaussian (Mean and Covariance)
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    # Generate points
    # 2-sigma ellipse corresponds to Mahalanobis distance <= 2
    generated = []
    
    # Pre-calculate inverse covariance matrix for distance check
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("Covariance matrix is singular. Cannot fit ellipse effectively.")
        sys.exit(1)

    batch_size = max(n, 100) # Generate at least 100 at a time
    
    while len(generated) < n:
        # Sample from multivariate normal distribution
        # This captures the pattern of the input usage (Gaussian)
        samples = np.random.multivariate_normal(mean, cov, batch_size)
        
        # Filter points within 2-sigma ellipse
        # Mahalanobis distance squared: (x-mu)^T * Sigma^-1 * (x-mu)
        diff = samples - mean
        
        # Efficient calculation of Mahalanobis distance squared for all samples
        # (diff @ inv_cov) * diff performs row-wise dot product logic
        dist_sq = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        
        # 2-sigma check: Distance <= 2 => Distance^2 <= 4
        valid_indices = dist_sq <= 4.0
        valid_samples = samples[valid_indices]
        
        generated.extend(valid_samples)
        
        # If we have enough, truncate
        if len(generated) >= n:
            generated = generated[:n]
            break

    # Output to stdout in CSV format
    for p in generated:
        print(f"{p[0]:.6f},{p[1]:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a 2-sigma ellipse to input points and generate n new points within that range.")
    parser.add_argument("csv_path", help="Relative path to the input CSV file containing 2D coordinates")
    parser.add_argument("n", type=int, help="Number of points to generate")
    
    args = parser.parse_args()
    
    generate_points(args.csv_path, args.n)
