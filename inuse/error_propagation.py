import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def get_local_basis(v_full, n):
    """
    根据入射速度(世界坐标)和板法向量(世界坐标)构建局部基底。
    v_full: 6维状态向量 [vx, vy, vz, wx, wy, wz]
    n: 3维单位法向量
    """
    # 修复点：只取前3维(线速度)来计算几何关系
    v_3d = v_full[:3]
    
    # 1. 计算法向速度分量
    v_n_val = np.dot(v_3d, n)
    v_n_vec = v_n_val * n
    
    # 2. 计算切向速度矢量 (v_planar)
    v_t_vec = v_3d - v_n_vec
    
    # 3. 构建单位切向向量 t1 (主切向)
    # 如果垂直入射(没有切向速度)，选取任意垂直于n的向量作为t1
    if np.linalg.norm(v_t_vec) < 1e-6:
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, n)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        t1 = normalize(np.cross(arbitrary, n)) # 此时t1方向不重要，只要在面内即可
    else:
        t1 = normalize(v_t_vec)
        
    # 4. 构建次切向向量 t2 (由右手定则确定)
    t2 = np.cross(n, t1)
    
    # 返回局部基底向量 t1(x), t2(y), n(z)
    return t1, t2, n

def get_rotation_matrix_6x6(t1, t2, n):
    """
    构建从世界坐标系到局部坐标系的旋转矩阵 R (6x6)
    局部系定义: x轴=t1, y轴=t2, z轴=n
    """
    # 3x3 方向余弦矩阵 Q
    # Q 的行向量是新基底在旧基底(世界系XYZ)上的投影
    Q = np.array([
        t1,  # Local x row
        t2,  # Local y row
        n    # Local z row
    ])
    
    # 扩充为 6x6 块对角矩阵
    # | Q  0 |
    # | 0  Q |
    R = np.zeros((6, 6))
    R[:3, :3] = Q
    R[3:, 3:] = Q
    return R

def get_bounce_matrix_6x6(R_ball, mu, e_n):
    """
    构建 6x6 的局部误差传播矩阵 M_local
    这里使用了你提供的矩阵结构，并进行了合理的扩充。
    """
    # 初始化为单位阵 (意味着默认没有变化)
    M = np.eye(6)
    
    # --- 填入你的物理模型参数 ---
    # 你的模型: M_k (3x3) 作用于 [vt, vn, omega]
    # 对应局部坐标系:
    # vt -> 局部vx (索引0)
    # vn -> 局部vz (索引2)
    # omega -> 局部wy (索引4, 绕t2轴旋转对应主切向平面的运动)
    
    # 1. 主切向 (x) 与 上旋 (wy) 耦合 (来自你的公式)
    # [3/5, 0, 2R/5]
    M[0, 0] = 3/5.0
    M[0, 2] = 0.0       # x 与 z 通常不直接耦合，除非斜面极其粗糙
    M[0, 4] = 2*R_ball/5.0
    
    # 2. 法向 (z) (来自你的公式)
    # [0, -en, 0]
    M[2, 0] = 0.0
    M[2, 2] = -e_n      # 碰撞反弹系数
    M[2, 4] = 0.0
    
    # 3. 角速度 wy (来自你的公式)
    # [3/5R, 0, 2/5]
    M[4, 0] = 3/(5.0*R_ball)
    M[4, 2] = 0.0
    M[4, 4] = 2/5.0
    
    # --- 侧向扩充 (假设侧向与主切向物理机制相同) ---
    # 侧向 (y) 与 侧旋 (wx) 耦合
    # 注意符号：y轴速度对应绕x轴旋转(wx)的耦合需要符合右手定则
    M[1, 1] = 3/5.0
    M[1, 3] = -2*R_ball/5.0 # 符号可能需根据具体旋转方向调整，暂取负
    
    M[3, 1] = -3/(5.0*R_ball)
    M[3, 3] = 2/5.0

    return M

def simulate_bounce_nominal(v_in, n, e_n, R_ball=0.02):
    """
    更新标称速度，使用与误差矩阵一致的“完全滚动模型”。
    考虑了旋转与切向速度的耦合。
    """
    # 1. 获取局部基底
    t1, t2, n_vec = get_local_basis(v_in, n)
    
    # 2. 构建旋转矩阵 R (World -> Local)
    R_mat = get_rotation_matrix_6x6(t1, t2, n_vec)
    
    # 3. 转换到局部坐标系
    v_local = R_mat @ v_in
    # v_local = [vx, vy, vz, wx, wy, wz]
    vx, vy, vz = v_local[0], v_local[1], v_local[2]
    wx, wy, wz = v_local[3], v_local[4], v_local[5]
    
    # 4. 应用物理模型 (Rolling Model)
    # 对应 get_bounce_matrix_6x6 中的逻辑
    
    # x方向 (主切向) 与 wy (绕次切向转动) 耦合
    vx_new = (3/5.0) * vx + (2*R_ball/5.0) * wy
    
    # y方向 (次切向) 与 wx (绕主切向转动) 耦合
    # 注意符号: 矩阵中 M[1,3] = -2R/5, M[3,1] = -3/5R
    vy_new = (3/5.0) * vy - (2*R_ball/5.0) * wx
    
    # z方向 (法向)
    vz_new = -e_n * vz
    
    # wx 更新
    wx_new = - (3/(5.0*R_ball)) * vy + (2/5.0) * wx
    
    # wy 更新
    wy_new = (3/(5.0*R_ball)) * vx + (2/5.0) * wy
    
    # wz 更新 (假设法向摩擦力矩忽略不计，保持不变)
    wz_new = wz
    
    v_local_new = np.array([vx_new, vy_new, vz_new, wx_new, wy_new, wz_new])
    
    # 5. 转换回世界坐标系
    v_out = R_mat.T @ v_local_new
    
    return v_out

# ==========================================
# 主程序
# ==========================================

print("--- Multibounce Error Propagation Simulation ---")

# 1. 系统参数
R_ball = 0.02   # 乒乓球半径 2cm
g = np.array([0, 0, -9.8])
dt_steps = [0.5, 0.4, 0.3] # 每次飞行的持续时间 (秒) - 假设值

# 2. 初始状态
# 标称状态 (用于计算基底): 水平发射
current_nominal_state = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # vx=5, vy=0
# 初始误差 (这是我们要传播的)
delta_v0 = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0]) # x方向有0.1m/s误差，y方向有0.05m/s误差

# 3. 场景定义: 3次弹跳的板法向量
normals = [
    normalize(np.array([0, 0, 1])),       # 第一次: 地面 (水平)
    normalize(np.array([-0.2, 0.5, 1])),  # 第二次: 侧向倾斜板 (产生侧向耦合)
    normalize(np.array([0.5, 0, 1]))      # 第三次: 向前倾斜板
]
restitutions = [0.9, 0.8, 0.8] # 恢复系数

# 4. 误差传播初始化
current_error_matrix = np.eye(6) # 累积的速度误差传播矩阵 (Product of T_j)
total_position_error = np.zeros(3) # 最终位移误差

print(f"Initial Error (World Frame): {delta_v0}")
print("-" * 50)

for i in range(3):
    t_flight = dt_steps[i]
    normal = normals[i]
    e_n = restitutions[i]
    
    print(f"\n[Phase {i}: Flight -> Bounce {i+1}]")
    
    # A. 飞行阶段产生的位移误差累积
    # 公式项: t_i * (当前累积的速度误差)
    # 当前的速度误差向量 = current_error_matrix * delta_v0
    current_velocity_error = current_error_matrix @ delta_v0
    
    position_drift = t_flight * current_velocity_error[:3] # 只取速度部分
    total_position_error += position_drift
    
    print(f"  Flight Time: {t_flight}s")
    print(f"  Pos Error Drift this flight: {position_drift}")
    
    # 更新标称速度 (重力作用)
    # v = v0 + g*t
    current_nominal_state[:3] += g * t_flight
    print(f"  Impact Velocity (Nominal): {current_nominal_state[:3]}")
    
    # B. 碰撞阶段 (矩阵变换)
    
    # 1. 获取局部基底 (关键修复点)
    t1, t2, n_vec = get_local_basis(current_nominal_state, normal)
    print(f"  Plate Normal: {n_vec}")
    print(f"  Local Tangent1 (Main): {t1}")
    
    # 2. 构建旋转矩阵 R (World -> Local)
    R = get_rotation_matrix_6x6(t1, t2, n_vec)
    
    # 3. 构建局部碰撞矩阵 M (Local)
    M_local = get_bounce_matrix_6x6(R_ball, 0.2, e_n) # mu取0.2
    
    # 4. 计算世界坐标下的传递矩阵 T = R.T * M * R
    T_world = R.T @ M_local @ R
    
    # 5. 更新总累积矩阵 (注意乘法顺序: 新矩阵在左)
    # Accumulated_New = T_current * Accumulated_Old
    current_error_matrix = T_world @ current_error_matrix
    
    print(f"  Error Matrix Trace (Check stability): {np.trace(current_error_matrix):.4f}")
    
    # C. 更新标称速度 (为了下一次弹跳的基底计算)
    current_nominal_state = simulate_bounce_nominal(current_nominal_state, n_vec, e_n, R_ball)

print("-" * 50)
print("\nFinal Result:")
print(f"Total Position Error (World Frame): {total_position_error}")
print(f"Final Velocity Error (World Frame): {(current_error_matrix @ delta_v0)[:3]}")