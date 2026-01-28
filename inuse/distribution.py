import sys
import os
import numpy as np

# Try importing required libraries
try:
    from scipy import stats
except ImportError:
    print("Error: 'scipy' module is required. Please install it using 'pip install scipy'.")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: 'pandas' module is required. Please install it using 'pip install pandas'.")
    sys.exit(1)

def load_data(file_path):
    """
    Load data from a CSV file.
    Attempts to define X and Y coordinates.
    """
    try:
        # First attempt: Read without header
        df = pd.read_csv(file_path, header=None)
        
        # Check if the first row probably contains strings (header)
        # If the first row values are strings that can't be converted to float, assume it is a header
        is_header = False
        try:
             # Check first few columns
             for col in df.columns[:2]:
                 float(df.iloc[0][col])
        except ValueError:
             is_header = True
             
        if is_header:
            df = pd.read_csv(file_path, header=0)
            
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            # Fallback: maybe valid cols were read as object due to dirty data?
            # For now, just report error
            print("Error: CSV file must contain at least two numeric columns for X and Y coordinates.")
            return None, None

        # Heuristic to find X and Y columns
        cols = numeric_df.columns
        x_col, y_col = cols[0], cols[1]  # Default to first two
        
        # If columns have names (strings), try to find 'x' and 'y' case-insensitive
        col_names = [str(c).lower() for c in cols]
        
        # Exact match or contains 'x'/'y' priority?
        # Let's check independent "x" / "y" first
        if 'x' in col_names and 'y' in col_names:
            x_col = cols[col_names.index('x')]
            y_col = cols[col_names.index('y')]
        
        return numeric_df[x_col].values, numeric_df[y_col].values
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def analyze_1d_distribution(data, name="Data"):
    """
    Analyze the distribution of 1D data using KS test against common distributions.
    """
    if data is None or len(data) == 0:
        return
        
    print(f"\nAnalyzing distribution for: {name}")
    print(f"  Sample size: {len(data)}")
    print(f"  Basic Stats: Mean={np.mean(data):.4f}, Std={np.std(data):.4f}, Min={np.min(data):.4f}, Max={np.max(data):.4f}")
    
    # 1. Check if data is constant
    if np.min(data) == np.max(data):
        print("  Data is constant. (Point mass distribution)")
        return

    # Distributions to check
    # Mapping friendly name -> scipy.stats object
    dist_candidates = {
        'Normal (正态分布)': stats.norm,
        'Uniform (均匀分布)': stats.uniform,
        # 'Exponential': stats.expon  # Optional, usually data needs to be positive or shifted
    }
    
    results = []
    
    for dist_label, dist_func in dist_candidates.items():
        # Fit the distribution
        # params usually (loc, scale) for norm/uniform
        try:
            params = dist_func.fit(data)
            
            # KS Test
            # Null hypothesis: The sample comes from the specified distribution
            D, p_val = stats.kstest(data, dist_func.name, args=params)
            
            results.append({
                'label': dist_label,
                'p_value': p_val,
                'D': D,
                'params': params
            })
        except Exception as e:
            print(f"  Failed to fit {dist_label}: {e}")
        
    # Sort by p-value (descending - higher p-value is better 'fit')
    # Or by statistic D (ascending)
    results.sort(key=lambda x: x['p_value'], reverse=True)
    
    print(f"  {'Distribution':<20} | {'P-Value':<12} | {'KS Stat':<10}")
    print(f"  {'-'*46}")
    for res in results:
        print(f"  {res['label']:<20} | {res['p_value']:.4e}   | {res['D']:.4f}")
        
    best_fit = results[0]
    print(f"  {'-'*46}")
    
    # Interpretation threshold
    alpha = 0.05
    if best_fit['p_value'] > alpha:
        print(f"  Result: Data likely follows {best_fit['label']} (p > {alpha}).")
    else:
        # If all p-values are low, it might be none of them, or mixed, or just noisy/large sample
        print(f"  Result: Closest form is {best_fit['label']}, but statistical fit is poor (p < {alpha}).\n          This might be due to a constrained range, outliers, or a different distribution.")
    
    return results

def main():
    if len(sys.argv) < 2:
        try:
            file_input = input("Please enter the relative path to the CSV file: ").strip()
        except EOFError:
            file_input = ""
            
        if not file_input:
            print("No file provided. Usage: python distribution.py <path_to_csv>")
            sys.exit(1)
        file_path = file_input
    else:
        file_path = sys.argv[1]

    # Handle path
    full_path = os.path.abspath(file_path)

    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}")
        sys.exit(1)

    print(f"\nProcessing file: {file_path}")
    
    x_data, y_data = load_data(full_path)
    
    if x_data is not None and y_data is not None:
        analyze_1d_distribution(x_data, "X Coordinate")
        analyze_1d_distribution(y_data, "Y Coordinate")
    else:
        print("Could not extract X/Y data.")

if __name__ == "__main__":
    main()
