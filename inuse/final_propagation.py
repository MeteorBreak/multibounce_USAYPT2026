import numpy as np

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
    # 1. Normal velocity component
    v_n_val = np.dot(v_in, n)
    v_n_vec = v_n_val * n
    
    # 2. Tangential velocity vector
    v_t_vec = v_in - v_n_vec
    
    # 3. Tangent basis t1 (Main tangent)
    if np.linalg.norm(v_t_vec) < 1e-6:
        # Perpendicular impact: choose arbitrary tangent
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, n)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        t1 = normalize(np.cross(arbitrary, n))
    else:
        t1 = normalize(v_t_vec)
        
    # 4. Secondary tangent t2
    t2 = np.cross(n, t1)
    
    # R rows are the new basis vectors expressed in world frame
    Q = np.array([t1, t2, n]) # 3x3
    
    return Q

# ==========================================
# 2. Dynamics Models
# ==========================================

class PingPongSystem:
    def __init__(self):
        # --- System Parameters ---
        self.R = 0.02           # Radius (m)
        self.m = 0.0027         # Mass (kg)
        self.rho = 1.225        # Air density (kg/m^3)
        self.Cd = 0.5           # Drag coefficient
        self.Cl = 0.2           # Lift coefficient (Magnus)
        self.g = np.array([0, -9.8, 0]) # Gravity Y-down
        self.v_wind = np.array([0, 0, 0])
        
        # Precompute constants
        self.A = np.pi * self.R**2
        self.kD = 0.5 * self.Cd * self.rho * self.A
        self.kL = self.Cl * np.pi * self.R**3 * self.rho

    def flight_step(self, r, v, w, dt):
        """
        Nominal state update using Verlet integration (Velocity Verlet approximation for error prop context)
        """
        # Relative velocity
        v_rel = v - self.v_wind
        v_norm = np.linalg.norm(v_rel)
        
        # Forces
        F_gravity = self.m * self.g
        F_drag = -self.kD * v_norm * v_rel
        F_magnus = self.kL * np.cross(w, v_rel)
        
        a = (F_gravity + F_drag + F_magnus) / self.m
        
        # Velocity Verlet Part 1
        r_new = r + v * dt + 0.5 * a * dt**2
        
        # Predictor for velocity roughly to update accel
        v_pred = v + a * dt
        v_rel_pred = v_pred - self.v_wind
        v_norm_pred = np.linalg.norm(v_rel_pred)
        
        F_drag_new = -self.kD * v_norm_pred * v_rel_pred
        F_magnus_new = self.kL * np.cross(w, v_rel_pred)
        a_new = (F_gravity + F_drag_new + F_magnus_new) / self.m
        
        v_new = v + 0.5 * (a + a_new) * dt
        # Assuming angular velocity w is constant during short flight
        w_new = w 
        
        return r_new, v_new, w_new

    def get_flight_jacobian_9x9(self, v, w):
        """
        Calculates the simplified Jacobian F for dX/dt = F * X
        Only considers kinematic error drift: d(pos)/dt = vel_error
        Ignores aerodynamic error sensitivities (d(acc)/d(vel) = 0).
        """
        # --- Construct 9x9 F ---
        # dr_dot = dv
        # dv_dot = 0 (Simplified: flight forces don't amplify errors)
        # dw_dot = 0 (Simplified)
        
        F = np.zeros((9, 9))
        
        # Block (0,0): d(v)/dr = 0
        # Block (0,1): d(v)/dv = I (velocity error integrates to position error)
        F[0:3, 3:6] = np.eye(3)
        
        # All other blocks are zero
        
        return F

    def get_bounce_transition_9x9(self, v_in, w_in, normal, curvature_matrix, e_n):
        """
        Calculates the 9x9 Error Transition Matrix Phi_bounce
        Consider both Velocity/Spin transition (M) and Curvature Coupling (J*K).
        """
        mu_t = 3.0/5.0
        mu_rot = 2.0/5.0
        
        # 1. Construct standard 6x6 impact matrix T_world (Velocity & Spin)
        #    Based on the rolling model:
        #    v_out = mu_t*v_t - e_n*v_n + mu_rot*R*(w x n)
        #    w_out = ...
        
        #    Let's use the local basis approach for M, then rotate to world
        Q = get_rotation_matrix_to_local(normal, v_in) # 3x3 World->Local
        R_6x6 = np.zeros((6, 6))
        R_6x6[0:3, 0:3] = Q
        R_6x6[3:6, 3:6] = Q
        
        # Local Bounce Matrix M_local (6x6)
        # Indexes: vx(0), vy(1), vz(2), wx(3), wy(4), wz(5) 
        # Local definition: z is normal, x is main tangent.
        
        M_loc = np.eye(6)
        
        # v_x (tangential) update: v_x' = 3/5 v_x + 2R/5 w_y
        M_loc[0, 0] = 3/5.0
        M_loc[0, 4] = 2 * self.R / 5.0
        
        # v_y (tangential) update: v_y' = 3/5 v_y - 2R/5 w_x
        M_loc[1, 1] = 3/5.0
        M_loc[1, 3] = - 2 * self.R / 5.0
        
        # v_z (normal) update: v_z' = -e_n v_z
        M_loc[2, 2] = -e_n
        
        # w_x update: w_x' = -3/5R v_y + 2/5 w_x
        M_loc[3, 1] = - 3 / (5.0 * self.R)
        M_loc[3, 3] = 2/5.0
        
        # w_y update: w_y' = 3/5R v_x + 2/5 w_y
        M_loc[4, 0] = 3 / (5.0 * self.R)
        M_loc[4, 4] = 2/5.0
        
        # w_z usually conserved or damped, keep 1.0 for now
        
        # Transform M to world frame: T = R.T * M_loc * R
        T_world = R_6x6.T @ M_loc @ R_6x6
        
        # 2. Construct Sensitivity Jacobian J (World Frame)
        #    J_v_n = d(v_out)/d(n) approx (mu_n - mu_t) * [ (v.n)I + n v.T ]
        #    We ignore the spin coupling term for sensitivity as discussed (small magnitude)
        
        v_dot_n = np.dot(v_in, normal)
        # Coeff k = (-e_n - 3/5)
        k_factor_v = (-e_n - 3.0/5.0)
        
        J_v_n = k_factor_v * (v_dot_n * np.eye(3) + np.outer(normal, v_in))
        
        # J_w_n = d(w_out)/d(n)
        # Even though spin coupling to v is small, N changing changes friction direction significant for spin
        # w_out linear part approx: (3/5R) * (n x v_in)
        # d(n x v)/dn = - [v]x
        k_factor_w = 3.0 / (5.0 * self.R)
        J_w_n = k_factor_w * (-skew(v_in)) # Approximated main term
        
        # Stack Jacobian 6x3
        J_total = np.vstack([J_v_n, J_w_n])
        
        # 3. Calculate Coupling Matrix (Phi_vr)
        #    Coupling = J_total * Curvature (6x3 * 3x3 = 6x3)
        Coupling = J_total @ curvature_matrix
        
        # 4. Assemble full 9x9 Phi
        Phi = np.eye(9)
        # Position continuity (Top-left I is already there, others 0)
        
        # Coupling block (Bottom-Left 6x3) -> d(Motion)/d(r)
        Phi[3:9, 0:3] = Coupling
        
        # Motion block (Bottom-Right 6x6) -> d(Motion)/d(Motion)
        Phi[3:9, 3:9] = T_world
        
        return Phi 

# ==========================================
# 3. Simulation Control
# ==========================================

def run_simulation():
    sys = PingPongSystem()
    
    # --- A. Initial Conditions ---
    # Nominal State: r(3), v(3), w(3)
    # Coordinate System: Y-Up, Z-Forward, X-Right
    r_nom = np.array([0., 1.0, 0.])    # 1m height
    v_nom = np.array([0., -2.0, 5.0])  # Forward Z, Down Y
    w_nom = np.array([50.0, 0.0, 0.0]) # Topspin around X
    
    # Initial Error Vector (9x1)
    # Let's assume we have some uncertainty in initial velocity and spin
    delta_X = np.zeros(9)
    delta_X[3] = 0.1   # dv_x = 0.1 m/s
    delta_X[7] = 5.0   # dw_y = 5.0 rad/s
    
    # Accumulation Matrix (Phi_total)
    Phi_accum = np.eye(9)
    
    # --- B. Sequence Definition ---
    # 1. Flight (0.4s) -> Bounce 1 (Flat, slanted) -> Flight (0.3s) -> Bounce 2 (Curved) -> Flight (0.3s)
    
    sequence = [
        {'type': 'flight', 'dt': 0.335},
        {'type': 'bounce', 'normal': normalize(np.array([0.4297, 0.9030, 0])), 'e_n': 0.7023, 'is_curved': False},
        {'type': 'flight', 'dt': 0.327},
        {'type': 'bounce', 'normal': normalize(np.array([-0.3681, 0.9298, 0])), 'e_n': 0.7616, 'is_curved': False},
        {'type': 'flight', 'dt': 0.299},
        {'type': 'bounce', 'normal': normalize(np.array([0, 1, 0])), 'e_n': 0.7616, 'is_curved': True, 'curvature_radius': 0.5},
        {'type': 'flight', 'dt': 0.243}
    ]
    
    print("--- 9D Error Propagation Simulation ---")
    print(f"Initial Error Vector:\n{delta_X}")
    print("-" * 50)
    
    for i, step in enumerate(sequence):
        step_type = step['type']
        
        print(f"Step {i+1}: {step_type.upper()}")
        
        if step_type == 'flight':
            dt = step['dt']
            sub_steps = 50
            d_dt = dt / sub_steps
            
            # Numerically integrate Flight Error
            for _ in range(sub_steps):
                # 1. Calculate Jacobian at current nominal state
                F = sys.get_flight_jacobian_9x9(v_nom, w_nom)
                
                # 2. Integrate Phi: dPhi/dt = F*Phi => Phi(t+dt) = (I + F*dt)*Phi(t)
                # Note: We update the Accumulation Matrix directly.
                # E_new = (I + F*d_dt) * E_old
                Update_Mat = np.eye(9) + F * d_dt
                Phi_accum = Update_Mat @ Phi_accum
                
                # 3. Update nominal state for next step
                r_nom, v_nom, w_nom = sys.flight_step(r_nom, v_nom, w_nom, d_dt)
                
            print(f"  -> Pos after flight: {r_nom}")
            print(f"  -> Vel after flight: {v_nom}")
            
        elif step_type == 'bounce':
            n = step['normal']
            e_n = step['e_n']
            
            # Determine Curvature Matrix K (3x3)
            # K = dn/dr.
            # For Flat: K = 0
            # For Sphere (Radius R_surf): K = I/R_surf - (n n.T)/R_surf (approximate on surface)
            # Here we assume a simple isotropic curvature for demonstration.
            if step.get('is_curved'):
                R_surf = step['curvature_radius']
                # Concave or Convex? Let's assume Convex (dispersing error) -> K > 0 (simplification)
                # Actually for normal pointing OUT, convex surface K is usually defined positive.
                # Approximation: d_n / d_r = (1/R) * I_tangent
                # Identity minus normal projection to keep n unit length
                K = (1.0 / R_surf) * (np.eye(3) - np.outer(n, n))
                print(f"  -> Curved Surface Impact (R={R_surf}m)")
            else:
                K = np.zeros((3, 3))
                print(f"  -> Flat Surface Impact")

            # 1. Update nominal velocity (Instantaneous Bounce)
            # Using rolling model equations
            Q = get_rotation_matrix_to_local(n, v_nom)
            
            # Transform to local
            v_loc = Q @ v_nom
            w_loc = Q @ w_nom
            
            # Apply physics (manual implementation of rolling model for Nominal)
            # This must match what is used in the matrix derivation
            R_b = sys.R
            v_x, v_y, v_z = v_loc
            w_x, w_y, w_z = w_loc
            
            v_x_new = 0.6*v_x + 0.4*R_b*w_y
            v_y_new = 0.6*v_y - 0.4*R_b*w_x
            v_z_new = -e_n * v_z
            
            w_x_new = - (0.6/R_b)*v_y + 0.4*w_x
            w_y_new = (0.6/R_b)*v_x + 0.4*w_y
            w_z_new = w_z
            
            v_loc_new = np.array([v_x_new, v_y_new, v_z_new])
            w_loc_new = np.array([w_x_new, w_y_new, w_z_new])
            
            # Back to world
            v_nom = Q.T @ v_loc_new
            w_nom = Q.T @ w_loc_new
            
            # 2. Get 9x9 Bounce Matrix
            Phi_bounce = sys.get_bounce_transition_9x9(v_nom, w_nom, n, K, e_n)
            
            # 3. Update Accumulation
            Phi_accum = Phi_bounce @ Phi_accum
            
            print(f"  -> Vel after bounce: {v_nom}")

        # Calculate current total error
        current_error = Phi_accum @ delta_X
        # Print position error trace for monitoring
        pos_err_norm = np.linalg.norm(current_error[0:3])
        print(f"  -> Current Pos Error Magnitude: {pos_err_norm:.6f} m")
        print("-" * 30)

    # --- C. Final Output ---
    final_error = Phi_accum @ delta_X
    
    print("\n" + "="*50)
    print("FINAL ERROR TENSOR (Direct Propagation)")
    print("="*50)
    print(f"Nominal Final State:")
    print(f"  Pos: {r_nom}")
    print(f"  Vel: {v_nom}")
    print(f"  Spin: {w_nom}")
    print("\nPropagated Error vector (delta_X_final):")
    print(f"  d_Pos (3): {final_error[0:3]}")
    print(f"  d_Vel (3): {final_error[3:6]}")
    print(f"  d_Spin(3): {final_error[6:9]}")

    return final_error

if __name__ == "__main__":
    run_simulation()