#!/usr/bin/env python3
"""
Variance Propagation Script for Ping Pong Ball Trajectory
Calculates the propagation of the full 9x9 Covariance Matrix (Position, Velocity, Spin)
through flight (Kinematic Drift) and bounce (Rolling Model + Curvature Coupling).

Output:
    Prints the Covariance Matrix elements at each impact point, formatted for
    use with the visualization tool 'plot_points_aligned.py'.

Usage:
    python variance_propagation.py --sigma_r 0.001 0.001 0.001 --sigma_v 0.05 0.05 0.05 --sigma_w 1.0 1.0 1.0
"""

import numpy as np
import argparse
import sys

# ==========================================
# 1. Physics & Math Helpers
# ==========================================

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return v / norm

def skew(v):
    """Returns the skew-symmetric matrix of a vector v."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def get_rotation_matrix_to_local(n, v_in):
    """
    Constructs Rotation Matrix R (World -> Local)
    Local z = n
    Local x = tangent along velocity projection
    """
    v_n_val = np.dot(v_in, n)
    v_n_vec = v_n_val * n
    v_t_vec = v_in - v_n_vec
    
    if np.linalg.norm(v_t_vec) < 1e-6:
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, n)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        t1 = normalize(np.cross(arbitrary, n))
    else:
        t1 = normalize(v_t_vec)
        
    t2 = np.cross(n, t1)
    
    # R rows are the new basis vectors expressed in world frame
    # So v_local = R @ v_world
    Q = np.array([t1, t2, n]) 
    return Q

# ==========================================
# 2. Dynamics Models
# ==========================================

class PingPongSystem:
    def __init__(self):
        # Parameters
        self.R = 0.02           # Radius (m)
        self.m = 0.0027         # Mass (kg)
        self.rho = 1.225        # Air density
        self.g = np.array([0, -9.8, 0]) # Gravity (Y is vertical)
        
        # Aerodynamics (Used for nominal trajectory, ignored for error sensitivity)
        self.Cd = 0.5
        self.Cl = 0.2
        self.A = np.pi * self.R**2
        self.kD = 0.5 * self.Cd * self.rho * self.A
        self.kL = self.Cl * np.pi * self.R**3 * self.rho
        self.v_wind = np.zeros(3)

    def flight_step(self, r, v, w, dt):
        """
        Nominal state update using Velocity Verlet
        """
        v_rel = v - self.v_wind
        v_norm = np.linalg.norm(v_rel)
        
        F_gravity = self.m * self.g
        F_drag = -self.kD * v_norm * v_rel
        F_magnus = self.kL * np.cross(w, v_rel)
        
        a = (F_gravity + F_drag + F_magnus) / self.m
        
        r_new = r + v * dt + 0.5 * a * dt**2
        
        v_pred = v + a * dt
        v_rel_pred = v_pred - self.v_wind
        v_norm_pred = np.linalg.norm(v_rel_pred)
        
        F_drag_new = -self.kD * v_norm_pred * v_rel_pred
        F_magnus_new = self.kL * np.cross(w, v_rel_pred)
        a_new = (F_gravity + F_drag_new + F_magnus_new) / self.m
        
        v_new = v + 0.5 * (a + a_new) * dt
        return r_new, v_new, w

    def get_flight_transition_mask(self, dt):
        """
        Returns the simplified Flight Transition Matrix Phi (9x9).
        Model: Kinematic Drift Only.
        dr = v * dt
        dv = 0 (No aerodynamic error amplification)
        dw = 0
        """
        Phi = np.eye(9)
        # Position += Velocity * dt
        Phi[0:3, 3:6] = np.eye(3) * dt
        return Phi

    def get_bounce_transition(self, v_in, w_in, normal, curvature_matrix, e_n):
        """
        Calculates Phi_bounce (9x9) including Curvature Coupling.
        """
        # 1. Rotation to Local Frame
        Q = get_rotation_matrix_to_local(normal, v_in)
        R_6x6 = np.zeros((6, 6))
        R_6x6[0:3, 0:3] = Q
        R_6x6[3:6, 3:6] = Q
        
        # 2. Local Bounce Matrix M (rolling model)
        M_loc = np.eye(6)
        
        # Tangential V
        M_loc[0, 0] = 3/5.0
        M_loc[0, 4] = 2 * self.R / 5.0
        
        M_loc[1, 1] = 3/5.0
        M_loc[1, 3] = - 2 * self.R / 5.0
        
        # Normal V
        M_loc[2, 2] = -e_n
        
        # Spin
        M_loc[3, 1] = - 3 / (5.0 * self.R)
        M_loc[3, 3] = 2/5.0
        
        M_loc[4, 0] = 3 / (5.0 * self.R)
        M_loc[4, 4] = 2/5.0
        
        # Transform to World
        T_world = R_6x6.T @ M_loc @ R_6x6
        
        # 3. Sensitivity to Normal Vector (Jacobian J)
        v_dot_n = np.dot(v_in, normal)
        k_factor_v = (-e_n - 3.0/5.0)
        J_v_n = k_factor_v * (v_dot_n * np.eye(3) + np.outer(normal, v_in))
        
        k_factor_w = 3.0 / (5.0 * self.R)
        J_w_n = k_factor_w * (-skew(v_in))
        
        J_total = np.vstack([J_v_n, J_w_n]) # 6x3
        
        # 4. Curvature Coupling
        Coupling = J_total @ curvature_matrix # 6x3
        
        # 5. Assemble Phi
        Phi = np.eye(9)
        Phi[3:9, 0:3] = Coupling # Pos -> Vel/Spin coupling
        Phi[3:9, 3:9] = T_world  # Vel/Spin -> Vel/Spin
        
        return Phi

def format_covariance_for_plot(cov_matrix):
    """
    Extracts the X-Z Position block (Horizontal Plane) and formats as string.
    We assume Y is vertical, so we plot X (Lateral) and Z (Forward).
    Returns: var_x cov_xz cov_zx var_z
    """
    # X index 0, Z index 2
    vx = cov_matrix[0, 0]
    vz = cov_matrix[2, 2]
    cxz = cov_matrix[0, 2]
    czx = cov_matrix[2, 0]
    return f"{vx:.6e} {cxz:.6e} {czx:.6e} {vz:.6e}"

def main():
    parser = argparse.ArgumentParser(description="Propagate Covariance Matrix through bounces.")
    parser.add_argument("--sigma_r", nargs=3, type=float, default=[0.005, 0.005, 0.005], help="Initial Pos Std (m)")
    parser.add_argument("--sigma_v", nargs=3, type=float, default=[0.05, 0.05, 0.05], help="Initial Vel Std (m/s)")
    parser.add_argument("--sigma_w", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Initial Spin Std (rad/s)")
    parser.add_argument("--no_print", action='store_true', help="Suppress detailed output")
    args = parser.parse_args()

    # 1. Initialize Covariance P0 (9x9)
    P = np.zeros((9, 9))
    # Diagonal variances = sigma^2
    np.fill_diagonal(P, np.concatenate([
        np.array(args.sigma_r)**2,
        np.array(args.sigma_v)**2,
        np.array(args.sigma_w)**2
    ]))

    # 2. System Setup
    sys_model = PingPongSystem()
    
    # Nominal State (Y is Vertical Up, Z is Forward, X is Right)
    r = np.array([0., 0.705, 0.])     # 1m height
    v = np.array([0., 1.02, 5.76])   # 5 m/s forward (Z), 2 m/s down (-Y)
    w = np.array([0., 0., 0.])  # Topspin (Rotation around X axis)
    
    # 3. Define Scenarios (The Sequence)
    # Using the same sequence as final_propagation.py for consistency
    sequence = [
        # Bounce 1: Slanted Flat
        {'type': 'flight', 'dt': 0.335},
        {'type': 'bounce', 'normal': normalize(np.array([0.4297, 0.9030, 0])), 'e_n': 0.7023, 'is_curved': False},
        
        # Bounce 2: Curved Surface (Cylindrical-ish)
        {'type': 'flight', 'dt': 0.327},
        {'type': 'bounce', 'normal': normalize(np.array([-0.3681, 0.9298, 0])), 'e_n': 0.7616, 'is_curved': False}, # Flat in original? Check.
        
        # Bounce 3: Curved
        {'type': 'flight', 'dt': 0.299},
        {'type': 'bounce', 'normal': normalize(np.array([0, 1, 0])), 'e_n': 0.7616, 'is_curved': True, 'curvature_radius': 0.3},
        
        # Final Landing
        {'type': 'flight', 'dt': 0.243}
    ]

    print("=== Covariance Propagation Simulation ===")
    print(f"Initial Sigma R: {args.sigma_r}")
    
    for i, step in enumerate(sequence):
        # A. Flight Phase
        if step['type'] == 'flight':
            dt = step['dt']
            
            # 1. Nominal Trajectory Integration (Substeps for accuracy of nominal, though STM is linear)
            sub_steps = 50
            d_dt = dt / sub_steps
            for _ in range(sub_steps):
                r, v, w = sys_model.flight_step(r, v, w, d_dt)
                
            # 2. Covariance Propagation (Exact for kinematic model)
            # P_new = Phi * P * Phi.T
            Phi_flight = sys_model.get_flight_transition_mask(dt)
            P = Phi_flight @ P @ Phi_flight.T
            
            if not args.no_print:
                print(f"\n[Step {i+1}] Flight ({dt}s) -> Pos: {r}")
                print(f"  Pos Covariance Diag: {np.diag(P)[0:3]}")

        # B. Bounce Phase
        elif step['type'] == 'bounce':
            n = step['normal']
            e_n = step['e_n']
            
            # Print Pre-Bounce Covariance for Plotting
            # This is the covariance "at the landing plane collision" for this bounce
            print(f"\n>>> BOUNCE {i+1} IMPACT (Plot Args 2D):")
            plot_args = format_covariance_for_plot(P)
            print(f"    --random {plot_args}")
            print(f"    (Std dev X: {np.sqrt(P[0,0]):.4f} m, Z: {np.sqrt(P[2,2]):.4f} m)")
            
            # Calculate Curvature Matrix K
            if step.get('is_curved'):
                R_surf = step['curvature_radius']
                # Convex Surface Assumption (Normal points out of ball, out of surface usually)
                # K = (1/R) * (I - nnT)
                K = (1.0 / R_surf) * (np.eye(3) - np.outer(n, n))
            else:
                K = np.zeros((3, 3))
            
            # Calculate Transition Phi_bounce
            Phi_bounce = sys_model.get_bounce_transition(v, w, n, K, e_n)
            
            # Propagate Covariance
            # P_post = Phi_bounce * P_pre * Phi_bounce.T
            P = Phi_bounce @ P @ Phi_bounce.T
            
            # Update Nominal Velocity/Spin (Instantaneous)
            # We need valid v_out, w_out for the next flight
            # Use the matrix T_world logic from get_bounce_transition implicitly or recompute
            # Recomputing using the same logic for safety:
            d_state = np.zeros(9) # Dummy error
            # Actually we need the nominal update. 
            # Re-using the logic manually or trusting the helper isn't enough, we need actual `v_new`
            # Let's just create a quick helper or do it here.
            Q = get_rotation_matrix_to_local(n, v)
            v_loc = Q @ v
            
            # Impulse equations (Rolling)
            v_loc_new = np.zeros(3)
            # Tangent
            v_loc_new[0] = (3/5)*v_loc[0] # + spin coupling (omitted in simple update, wait, rolling model HAS spin coupling)
            # We must be consistent with the STM.
            # The STM used: v_t' = 3/5 v_t + 2R/5 (w x n) ...
            # Let's simple apply the impulse properly:
            
            # V, W in world
            v_t = v - np.dot(v, n) * n
            v_n = np.dot(v, n) * n
            
            # Rolling collision (No slip) Nominal Update
            # v_out = 3/5 v_t - e_n v_n + 2/5 R (w x n)
            cross_term = np.cross(w, n)
            v_new = (3/5.0)*v_t - e_n*v_n + (2.0/5.0)*sys_model.R * cross_term
            
            # w_out = 3/(5R) (n x v_in) + 2/5 w_in
            w_new = (3.0/(5.0*sys_model.R)) * np.cross(n, v) + (2.0/5.0)*w
            
            v = v_new
            w = w_new
            
            if not args.no_print:
                print(f"  -> Bounce complete. New Vel: {v}")

    # Final State
    print(f"\n>>> FINAL STATE (Plot Args 2D):")
    plot_args = format_covariance_for_plot(P)
    print(f"    --random {plot_args}")
    print(f"    (Std dev X: {np.sqrt(P[0,0]):.4f} m, Z: {np.sqrt(P[2,2]):.4f} m)")


if __name__ == "__main__":
    main()
