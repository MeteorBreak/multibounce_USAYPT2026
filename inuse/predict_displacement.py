import numpy as np
import argparse
import sys

def generate_random_points(n, errors_m):
    """
    Generate n random points within the specified error range.
    
    Args:
        n (int): Number of points.
        errors_m (list or tuple): 3D error [e_x, e_y, e_z] in meters. 
                                  e_y (index 1) is assumed to be the normal error.
    
    Returns:
        np.ndarray: nx3 array of coordinates in mm.
    """
    # Create a copy to avoid modifying the original list if it's reused
    errors = np.array(errors_m, dtype=float)
    
    # According to instructions: 
    # For the second input error (normal error, index 1), multiply by 0.2 
    # and add directly to the other two directions (indices 0 and 2).
    normal_error_contribution = errors[1] * 0.2
    
    # Modify the error bounds for the parallel directions
    # We assume the inputs represent the half-width of the range (or similar magnitude)
    # The logic provided: new_error_parallel = old_error_parallel + 0.2 * error_normal
    bound_x = errors[0] + normal_error_contribution
    bound_y = errors[1]  # Normal error itself is assumed to be unchanged
    bound_z = errors[2] + normal_error_contribution
    
    # Generate points uniformly distributed within [-bound, +bound]
    # We assume the coordinates are centered at (0,0,0)
    points_x = np.random.uniform(-bound_x, bound_x, n)
    points_y = np.random.uniform(-bound_y, bound_y, n)
    points_z = np.random.uniform(-bound_z, bound_z, n)
    
    points_m = np.column_stack((points_x, points_y, points_z))
    
    # Convert from meters to millimeters
    points_mm = points_m * 1000.0
    
    return points_mm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random landing coordinates based on theoretical errors.")
    parser.add_argument("n", type=int, help="Number of points to generate")
    parser.add_argument("ex", type=float, help="Parallel error 1 (m) (X-axis)")
    parser.add_argument("ey", type=float, help="Normal error (m) (Y-axis) - Used to augment parallel errors")
    parser.add_argument("ez", type=float, help="Parallel error 2 (m) (Z-axis)")
    
    args = parser.parse_args()
    
    errors_input = [args.ex, args.ey, args.ez]
    
    points = generate_random_points(args.n, errors_input)
    
    # Output definition: Simply print coordinates
    # print(f"Generated {args.n} points (Unit: mm):")
    # print("x, z")
    for p in points:
        print(f"{p[0]:.4f}, {p[2]:.4f}")
