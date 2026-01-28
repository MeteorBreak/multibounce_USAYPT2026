import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import sys

# ==========================================
# Configuration / Parameters
# ==========================================
# Default parameters for a standard table tennis ball
# Users can modify these default values or input them at runtime
PARAMS = {
    'm': 0.0027,       # Mass (kg)
    'R': 0.020,        # Radius (m)
    'rho': 1.225,      # Air density (kg/m^3)
    'Cd': 0.4,         # Drag coefficient (approximate)
    'Cl': 0.2,         # Lift coefficient (example default)
    'omega': np.array([0.0, 0.0, 0.0]), # Angular velocity vector (rad/s)
    'wind_v': np.array([0.0, 0.0, 0.0]), # Wind velocity (m/s)
    'r0': np.array([0.0, 0.0, 0.0]),     # Launch position (x, y, z) in m
    'target_x': 2.35,  # Target distance from launcher (m)
    'v0_nominal': np.array([10.0, 0.0, 0.1]) # Estimated/Nominal launch velocity (m/s)
}

# Simulation constants
DT = 0.001 # Time step (s)
G = 9.81   # Gravity (m/s^2)
PI = np.pi

# ==========================================
# Physics Engine
# ==========================================

def compute_acceleration(v, params):
    """
    Computes acceleration based on Air Resistance and Magnus Effect.
    
    Formulas:
    F_D = -0.5 * Cd * rho * pi * R^2 * |v - vw| * (v - vw)
    F_M = Cl * rho * pi * R^3 * (omega x (v - vw))
    """
    v_w = params['wind_v']
    v_rel = v - v_w
    v_mag = np.linalg.norm(v_rel)
    
    # Avoid division by zero or computations on static ball
    if v_mag == 0:
        return np.array([0.0, 0.0, -G])
        
    m = params['m']
    R = params['R']
    rho = params['rho']
    Cd = params['Cd']
    Cl = params['Cl']
    omega = params['omega']
    
    # Drag Force (Air Resistance)
    # F_D = -0.5 * c_D * rho * pi * R^2 * |v-vw| * (v-vw)
    Area = PI * R**2
    F_D = -0.5 * Cd * rho * Area * v_mag * v_rel
    
    # Magnus Force
    # F_M = c_L * rho * pi * R^3 * (omega x (v-vw))
    F_M = Cl * rho * PI * R**3 * np.cross(omega, v_rel)
    
    # Gravity Force
    F_G = np.array([0.0, 0.0, -m * G])
    
    # Total Force
    F_total = F_G + F_D + F_M
    
    # Newton's Second Law: a = F / m
    return F_total / m

def simulate_trajectory(v0, params):
    """
    Simulates the ball trajectory using the Verlet integration algorithm provided.
    X(t + dt) = X(t) + V(t)dt + 0.5 * a(t)dt^2
    Returns the impact point (y, z) on the plane x = target_x.
    """
    r = params['r0'].copy()
    v = v0.copy()
    a = compute_acceleration(v, params)
    
    dt = DT
    target_x = params['target_x']
    
    # Limit steps to prevent infinite loops
    max_steps = 10000 
    
    for _ in range(max_steps):
        # Check if we passed the target plane
        if r[0] >= target_x:
            break
            
        # Verlet Position Update (using Taylor expansion provided)
        # X(t+dt) = X(t) + V(t)dt + 0.5 * a(t)dt^2
        r_next = r + v * dt + 0.5 * a * dt**2
        
        # Velocity Update
        # To calculate the next position, we need the next velocity for the drag force.
        # We use Velocity Verlet which is symplectic and stable:
        # V(t+dt) = V(t) + 0.5 * (a(t) + a(t+dt)) * dt
        # But a(t+dt) depends on V(t+dt), so we first predict V
        
        # Predict V (Euler) to estimate new forces
        v_pred = v + a * dt
        a_next = compute_acceleration(v_pred, params)
        
        # Correct V
        v_next = v + 0.5 * (a + a_next) * dt
        
        # Update state
        r = r_next
        v = v_next
        a = a_next
        
    # Interpolation for precise impact point
    # current r is just past target_x. We can backtrack linearly using velocity.
    dx = r[0] - target_x
    if v[0] != 0:
        dt_back = dx / v[0]
        y_impact = r[1] - v[1] * dt_back
        z_impact = r[2] - v[2] * dt_back
    else:
        y_impact = r[1]
        z_impact = r[2]

    return np.array([y_impact, z_impact])

# ==========================================
# Inverse Calculation (Solver)
# ==========================================

def solve_initial_velocity(target_yz, v_nominal, params):
    """
    Finds the initial velocity v0 that results in the ball hitting target_yz.
    Since there are multiple v0 that can hit the target (3 components vs 2 constraints),
    we look for the v0 closest to the 'v_nominal' estimate.
    """
    
    # Objective function to minimize
    def objective(v_flat):
        # 1. Error in impact position
        impact = simulate_trajectory(v_flat, params)
        pos_error = np.sum((impact - target_yz)**2)
        
        # 2. Regularization: Deviation from nominal launch velocity
        # This resolves the redundancy (chooses the "most likely" shot close to expected settings)
        vel_deviation = np.sum((v_flat - v_nominal)**2)
        
        # Weight position error much higher because hitting the target is the hard constraint
        return pos_error * 1e4 + vel_deviation

    # Use Nelder-Mead simplex method as it doesn't require analytical gradients
    result = minimize(objective, v_nominal, method='Nelder-Mead', tol=1e-6)
    
    final_v0 = result.x
    impact_check = simulate_trajectory(final_v0, params)
    
    return final_v0, impact_check

# ==========================================
# Main Execution
# ==========================================

def main():
    print("=======================================================")
    print("   Table Tennis Launcher Initial Velocity Calculator   ")
    print("=======================================================")
    
    # 1. Parameter Input
    print(f"\nDefault Parameters (Standard Ball):")
    print(f"Mass: {PARAMS['m']} kg, Radius: {PARAMS['R']} m, Air Density: {PARAMS['rho']} kg/m^3")
    print(f"Launch Pos: {PARAMS['r0']}, Target Dist: {PARAMS['target_x']} m")
    
    # Optional: Update nominal velocity
    print(f"\nCurrent Nominal Velocity Guess: {PARAMS['v0_nominal']} m/s")
    use_custom_v = input("Do you want to update parameters? (y/n) [n]: ").lower()
    if use_custom_v == 'y':
        try:
            vx = float(input(f"Enter Nominal Vx (m/s): ") or PARAMS['v0_nominal'][0])
            vy = float(input(f"Enter Nominal Vy (m/s): ") or PARAMS['v0_nominal'][1])
            vz = float(input(f"Enter Nominal Vz (m/s): ") or PARAMS['v0_nominal'][2])
            PARAMS['v0_nominal'] = np.array([vx, vy, vz])
            
            w_str = input("Enter Spin Vector 'wx,wy,wz' (rad/s) [0,0,0]: ")
            if w_str.strip():
                PARAMS['omega'] = np.array([float(x) for x in w_str.split(',')])
                
            cd_val = input(f"Enter Drag Coeff Cd [{PARAMS['Cd']}]: ")
            if cd_val.strip(): PARAMS['Cd'] = float(cd_val)
            
            cl_val = input(f"Enter Lift Coeff Cl [{PARAMS['Cl']}]: ")
            if cl_val.strip(): PARAMS['Cl'] = float(cl_val)
            
        except ValueError as e:
            print(f"Input Error: {e}")
            return

    # 2. CSV Input
    csv_path = input("\nEnter relative path to CSV file with impact coordinates: ").strip()
    full_path = os.path.abspath(csv_path)
    
    if not os.path.exists(full_path):
        print(f"Error: File '{full_path}' does not exist.")
        return
        
    try:
        df = pd.read_csv(full_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"\nColumns found: {list(df.columns)}")
    print("Please specify which columns describe the impact coordinates (Y, Z).")
    col_y = input(f"Column for Y (Horizontal on target) [default '{df.columns[0]}']: ") or df.columns[0]
    
    # Auto-guess second column if not specified
    default_z = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    col_z = input(f"Column for Z (Vertical on target)   [default '{default_z}']: ") or default_z
    
    if col_y not in df.columns or col_z not in df.columns:
        print("Error: Columns not found.")
        return
        
    y_data = df[col_y].values
    z_data = df[col_z].values
    
    # 3. Processing
    print(f"\nCalculating inverse trajectories for {len(y_data)} points...")
    print("This may take a few seconds per point depending on convergence...")
    
    stats_vx, stats_vy, stats_vz = [], [], []
    errors = []
    
    for i in range(len(y_data)):
        target = np.array([y_data[i], z_data[i]])
        
        # Solve
        v_sol, impact = solve_initial_velocity(target, PARAMS['v0_nominal'], PARAMS)
        
        # Calculate error relative to nominal
        v_err = v_sol - PARAMS['v0_nominal']
        
        stats_vx.append(v_sol[0])
        stats_vy.append(v_sol[1])
        stats_vz.append(v_sol[2])
        errors.append(v_err)
        
        print(f"Point {i+1}: Target({target[0]:.3f}, {target[1]:.3f}) -> "
              f"V0_calc([{v_sol[0]:.2f}, {v_sol[1]:.2f}, {v_sol[2]:.2f}]) "
              f"Err_mag: {np.linalg.norm(impact-target):.2e}")

    # 4. Results
    errors = np.array(errors)
    mean_v = np.array([np.mean(stats_vx), np.mean(stats_vy), np.mean(stats_vz)])
    std_v = np.array([np.std(stats_vx), np.std(stats_vy), np.std(stats_vz)])
    
    print("\n" + "="*40)
    print("              RESULTS               ")
    print("="*40)
    print(f"Nominal Velocity: {PARAMS['v0_nominal']}")
    print(f"Mean Calc Velocity: {mean_v}")
    print(f"Std Dev (Dispersion): {std_v}")
    print("\nVelocity Errors (Calculated - Nominal):")
    print(f"Mean Error: {np.mean(errors, axis=0)}")
    print(f"RMS Error:  {np.sqrt(np.mean(errors**2, axis=0))}")
    
    save = input("\nSave calculated velocities to CSV? (y/n): ").lower()
    if save == 'y':
        out_name = "velocity_analysis_result.csv"
        out_df = pd.DataFrame({
            'target_y': y_data,
            'target_z': z_data,
            'calc_vx': stats_vx,
            'calc_vy': stats_vy,
            'calc_vz': stats_vz,
            'error_vx': errors[:,0],
            'error_vy': errors[:,1],
            'error_vz': errors[:,2]
        })
        out_df.to_csv(out_name, index=False)
        print(f"Saved to {out_name}")

if __name__ == "__main__":
    main()
