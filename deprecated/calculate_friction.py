import numpy as np
import os

# Constants
RADIUS = 0.02  # Radius of a standard ping pong ball in meters (40mm diameter)

def calculate_friction_coefficient(file_path):
    """
    Calculates the friction coefficient and Coefficient of Restitution (COR)
    between a ping pong ball and a board using tracking data (t, x, y, vx, vy, omega).
    
    Based on the Garwin Model for bouncing balls.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load data
    # We use numpy to load the CSV. 
    # Handling the 'ω' (omega) column requires correct encoding or column index access.
    try:
        # encoding='utf-8' is important for the Greek letter omega
        data = np.genfromtxt(file_path, delimiter=',', names=True, encoding='utf-8')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Extract arrays for easier access
    # We use the column names from the file. 
    # If 'ω' is not found by name, we try to access by index if possible, 
    # but structured arrays are accessed by name.
    
    try:
        t = data['t']
        y = data['y']
        vx = data['vx']
        vy = data['vy']
        
        # Try to find the omega column. It might be named 'ω', 'omega', or similar.
        # We check the available field names.
        keys = data.dtype.names
        omega_key = None
        for k in keys:
            if 'ω' in k or 'omega' in k.lower():
                omega_key = k
                break
        
        if omega_key:
            omega = data[omega_key]
        else:
            # Fallback: assume it's the 6th column if it exists and we couldn't find the name
            if len(keys) >= 6:
                omega = data[keys[5]]
                print(f"Warning: Could not find column 'ω' or 'omega'. Using 6th column '{keys[5]}' as angular velocity.")
            else:
                print("Error: Angular velocity column not found.")
                return

    except ValueError as e:
        print(f"Error extracting data columns: {e}")
        return
    
    # --- Algorithm Explanation ---
    # 1. Identify the bounce event:
    #    The bounce occurs when the ball reaches its lowest vertical position (minimum y).
    #
    # 2. Select analysis points:
    #    We select data points slightly before and after the exact bounce moment.
    #
    # 3. Calculate Friction Coefficient (mu):
    #    mu = | delta_vx / delta_vy |
    #
    # 4. Calculate Coefficient of Restitution (COR) - Garwin Model:
    #    The Garwin model characterizes the bounce using two coefficients:
    #    a) Vertical COR (e_y): Ratio of vertical speeds.
    #       e_y = - v_y_out / v_y_in
    #
    #    b) Tangential COR (e_x): Ratio of tangential velocities at the contact point.
    #       The velocity of the contact point (v_cp) is: v_cp = v_x + R * omega
    #       e_x = - v_cp_out / v_cp_in
    #       
    #       Note: R is the radius of the ball (0.02m).
    
    # Step 1: Find the index of the bounce (minimum y)
    min_y_idx = np.argmin(y)
    
    # Step 2: Select points before and after
    buffer = 2
    
    # Ensure we don't go out of bounds
    if min_y_idx - buffer < 0 or min_y_idx + buffer >= len(t):
        print("Error: Bounce detected too close to the start or end of the dataset.")
        return

    idx_before = min_y_idx - buffer
    idx_after = min_y_idx + buffer
    
    # Step 3: Get velocities
    vx_in = vx[idx_before]
    vy_in = vy[idx_before]
    omega_in = omega[idx_before]
    
    vx_out = vx[idx_after]
    vy_out = vy[idx_after]
    omega_out = omega[idx_after]
    
    # Step 4: Calculate deltas and coefficients
    delta_vx = vx_out - vx_in
    delta_vy = vy_out - vy_in
    
    if delta_vy == 0:
        print("Error: No change in vertical velocity detected.")
        return

    # Friction Coefficient (mu)
    mu = abs(delta_vx / delta_vy)
    
    # Vertical COR (e_y)
    # We use absolute values to handle direction signs robustly
    e_y = abs(vy_out / vy_in)
    
    # Tangential COR (e_x) - Garwin Model
    # v_cp = v_x + R * omega

    v_cp_in = vx_in - RADIUS * omega_in
    v_cp_out = vx_out - RADIUS * omega_out
    
    # Avoid division by zero
    if abs(v_cp_in) < 1e-6:
        e_x = 0.0 # Or undefined
    else:
        e_x = abs(v_cp_out / v_cp_in)
    
    # Output results
    print("=" * 50)
    print("   Bounce Analysis (Garwin Model)")
    print("=" * 50)
    print(f"Data File: {file_path}")
    print(f"Bounce detected at index {min_y_idx}, time t = {t[min_y_idx]:.4f} s")
    print("-" * 50)
    print(f"{'Parameter':<20} | {'Pre-bounce':<12} | {'Post-bounce':<12}")
    print("-" * 50)
    print(f"{'vx (m/s)':<20} | {vx_in:12.4f} | {vx_out:12.4f}")
    print(f"{'vy (m/s)':<20} | {vy_in:12.4f} | {vy_out:12.4f}")
    print(f"{'omega (rad/s)':<20} | {omega_in:12.4f} | {omega_out:12.4f}")
    print("-" * 50)
    print(f"Delta vx: {delta_vx:.4f}")
    print(f"Delta vy: {delta_vy:.4f}")
    print("-" * 50)
    print(f"Friction Coefficient (mu) : {mu:.4f}")
    print(f"Vertical COR (e_y)        : {e_y:.4f}")
    print(f"Tangential COR (e_x)      : {e_x:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    # Ask user for file path at runtime
    csv_file = input("Please enter the relative path of the CSV file to analyze: ").strip()
    calculate_friction_coefficient(csv_file)
