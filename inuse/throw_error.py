import argparse
import numpy as np
import os
import sys

def calculate_velocity_error(csv_path, flight_time):
    """
    Calculate initial velocity error based on landing coordinates and flight time.
    
    Args:
        csv_path (str): Path to the CSV file containing landing coordinates (x, y).
        flight_time (float): Flight time in seconds.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    try:
        # Load data. Assuming no header and comma delimiter based on user context.
        # If the file has a header, we might need to skip it. 
        # Let's try to load it; if it fails, try skipping one row.
        try:
            data = np.loadtxt(csv_path, delimiter=',')
        except ValueError:
             data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
             
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if data.ndim == 1:
        # Handle case with single data point
        data = data.reshape(1, -1)

    if data.shape[1] < 2:
        print("Error: CSV must have at least two columns representing x and y coordinates.")
        return

    # Extract coordinates (assuming relative to target point as per description)
    # x and y are the offsets from the target.
    # Convert from mm to m
    x_offsets = data[:, 0] / 1000.0
    y_offsets = data[:, 1] / 1000.0

    # Calculate velocity errors
    # The error in velocity is simply the displacement error divided by time.
    # Delta_v = Delta_d / t
    vx_errors = x_offsets / flight_time
    vy_errors = y_offsets / flight_time

    # Calculate statistics
    vx_mean = np.mean(vx_errors)
    vx_std = np.std(vx_errors, ddof=1) if len(data) > 1 else 0.0 # Sample standard deviation
    vy_mean = np.mean(vy_errors)
    vy_std = np.std(vy_errors, ddof=1) if len(data) > 1 else 0.0

    # Calculate radial error (magnitude of velocity error vector)
    v_error_magnitude = np.sqrt(vx_errors**2 + vy_errors**2)
    v_mag_mean = np.mean(v_error_magnitude)
    v_mag_std = np.std(v_error_magnitude, ddof=1) if len(data) > 1 else 0.0

    print(f"Analysis Results")
    print(f"----------------")
    print(f"Data file: {csv_path}")
    print(f"Number of throws: {len(data)}")
    print(f"Flight time: {flight_time} s")
    print(f"----------------")
    print(f"X Direction (Direction 1) Velocity Error:")
    print(f"  Mean Bias:       {vx_mean:.4f} m/s")
    print(f"  Std Deviation:   {vx_std:.4f} m/s")
    print(f"----------------")
    print(f"Y Direction (Direction 2) Velocity Error:")
    print(f"  Mean Bias:       {vy_mean:.4f} m/s")
    print(f"  Std Deviation:   {vy_std:.4f} m/s")
    print(f"----------------")
    print(f"Total Velocity Error Magnitude:")
    print(f"  Mean:            {v_mag_mean:.4f} m/s")
    print(f"  Std Deviation:   {v_mag_std:.4f} m/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate initial velocity error from landing coordinates.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing landing coordinates (relative to target).")
    parser.add_argument("flight_time", type=float, help="Flight time of the ball in seconds.")
    
    args = parser.parse_args()
    
    calculate_velocity_error(args.csv_path, args.flight_time)
