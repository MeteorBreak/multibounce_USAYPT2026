import numpy as np
import os

def calculate_friction_coefficient(file_path, angle_degrees=0.0):
    """
    Calculates the friction coefficient and Coefficient of Restitution (COR)
    between a ping pong ball and a board using tracking data (t, x, y, vx, vy).
    
    Uses linear regression on velocity data to extrapolate velocities at the exact bounce time,
    minimizing errors from gravity and discrete sampling.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load data
    try:
        data = np.genfromtxt(file_path, delimiter=',', names=True, encoding='utf-8')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    try:
        t = data['t']
        y = data['y']
        vx = data['vx']
        vy = data['vy']
    except ValueError as e:
        print(f"Error extracting data columns: {e}")
        return
    
    # Step 1: Find the index of the bounce (minimum y)
    min_y_idx = np.argmin(y)
    t_bounce = t[min_y_idx]
    
    # Step 2: Select analysis window
    # We use a window of points before and after the bounce to fit the velocity trajectory.
    # This helps smooth out noise and accounts for gravity by extrapolating to the bounce time.
    window_size = 10  # Number of frames to use for fitting
    gap = 1           # Frames to skip immediately around the bounce to avoid impact dynamics
    
    start_pre = min_y_idx - gap - window_size
    end_pre = min_y_idx - gap
    
    start_post = min_y_idx + gap
    end_post = min_y_idx + gap + window_size
    
    # Ensure we don't go out of bounds
    if start_pre < 0 or end_post > len(t):
        print("Error: Bounce detected too close to the start or end of the dataset for the defined window.")
        return

    # Extract data for fitting
    t_pre = t[start_pre:end_pre]
    vx_pre_data = vx[start_pre:end_pre]
    vy_pre_data = vy[start_pre:end_pre]

    t_post = t[start_post:end_post]
    vx_post_data = vx[start_post:end_post]
    vy_post_data = vy[start_post:end_post]
    
    # Step 3: Linear Regression to find velocities at impact
    # Fit v = at + b and evaluate at t_bounce
    
    # Pre-bounce
    coeffs_vx_pre = np.polyfit(t_pre, vx_pre_data, 1)
    vx_in = np.polyval(coeffs_vx_pre, t_bounce)
    
    coeffs_vy_pre = np.polyfit(t_pre, vy_pre_data, 1)
    vy_in = np.polyval(coeffs_vy_pre, t_bounce)

    # Post-bounce
    coeffs_vx_post = np.polyfit(t_post, vx_post_data, 1)
    vx_out = np.polyval(coeffs_vx_post, t_bounce)
    
    coeffs_vy_post = np.polyfit(t_post, vy_post_data, 1)
    vy_out = np.polyval(coeffs_vy_post, t_bounce)

    # Transform velocities to surface coordinates (t: tangential, n: normal)
    theta = np.radians(angle_degrees)
    
    vt_in = vx_in * np.cos(theta) + vy_in * np.sin(theta)
    vn_in = -vx_in * np.sin(theta) + vy_in * np.cos(theta)
    
    vt_out = vx_out * np.cos(theta) + vy_out * np.sin(theta)
    vn_out = -vx_out * np.sin(theta) + vy_out * np.cos(theta)
    
    # Step 4: Calculate deltas and coefficients
    delta_vt = vt_out - vt_in
    delta_vn = vn_out - vn_in
    
    if delta_vn == 0:
        print("Error: No change in normal velocity detected.")
        return

    # Friction Coefficient (mu)
    mu = abs(delta_vt / delta_vn)
    
    # Normal COR (e_n)
    e_n = abs(vn_out / vn_in)
    
    # Output results
    print("=" * 50)
    print("   Bounce Analysis (Linear Regression Fit)")
    print("=" * 50)
    print(f"Data File: {file_path}")
    print(f"Incline Angle: {angle_degrees} degrees")
    print(f"Bounce detected at index {min_y_idx}, time t = {t_bounce:.4f} s")
    print(f"Fitting Window: {window_size} frames (Gap: {gap})")
    print("-" * 50)
    print(f"{'Parameter':<20} | {'Pre-bounce':<12} | {'Post-bounce':<12}")
    print("-" * 50)
    print(f"{'vx (m/s)':<20} | {vx_in:12.4f} | {vx_out:12.4f}")
    print(f"{'vy (m/s)':<20} | {vy_in:12.4f} | {vy_out:12.4f}")
    print(f"{'vt (m/s)':<20} | {vt_in:12.4f} | {vt_out:12.4f}")
    print(f"{'vn (m/s)':<20} | {vn_in:12.4f} | {vn_out:12.4f}")
    print("-" * 50)
    print(f"Delta vt: {delta_vt:.4f}")
    print(f"Delta vn: {delta_vn:.4f}")
    print("-" * 50)
    print(f"Friction Coefficient (mu) : {mu:.4f}")
    print(f"Normal COR (e_n)          : {e_n:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    # Ask user for file path at runtime
    csv_file = input("Please enter the relative path of the CSV file to analyze: ").strip()
    try:
        angle_input = input("Please enter the incline angle in degrees (default 0): ").strip()
        angle = float(angle_input) if angle_input else 0.0
    except ValueError:
        print("Invalid angle. Using 0 degrees.")
        angle = 0.0
        
    calculate_friction_coefficient(csv_file, angle)
