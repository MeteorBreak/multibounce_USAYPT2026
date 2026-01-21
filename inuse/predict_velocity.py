import sys

def main():
    print("=== 球体切向出射速度计算工具 ===")
    print("公式: V_x_out = (Term * v_x) / (1 + Term)")
    print("      其中 Term = R^2 / (K * R1^2)")
    
    # 1. 定义预设参数
    m = 0.02294  # kg (尽管公式中未直接体现质量，但此处按要求定义)
    R_mm = 32.0  # mm (初始半径)
    K = 2.0 / 5.0  # 0.4 (系数)
    
    print(f"\n参数信息:")
    print(f"  质量 m = {m} kg")
    print(f"  半径 R = {R_mm} mm")
    print(f"  系数 K = {K}")
    print("-" * 30)
    
    try:
        # 2. 获取用户输入
        vx_str = input("请输入切向入射速度 v_x (m/s): ")
        vx = float(vx_str)
        
        delta_d_str = input("请输入球体碰撞瞬间直径的缩小量 (mm): ")
        delta_d = float(delta_d_str)
        
        # 3. 计算形变后的半径 R1
        # 逻辑: 碰撞后半径 = 初始半径 - (直径缩小量 / 2)
        R1_mm = R_mm - (delta_d / 2.0)
        
        print(f"\n内部计算:")
        print(f"  形变后半径 R1: {R1_mm:.6f} mm")
        
        if R1_mm <= 0:
            print("错误: 形变过大，计算出的半径无效 (<=0)。")
            return

        # 4. 代入公式计算
        # Term = R^2 / (K * R1^2)
        # 注意: R和R1使用相同单位(mm)即可，比值是无量纲的
        term = (R_mm ** 2) / (K * (R1_mm ** 2))
        
        print(f"  中间系数 (R^2 / K*R1^2): {term:.6f}")
        
        # V_x_out = ( Term * v_x ) / ( 1 + Term )
        vx_out = (term * vx) / (1 + term)
        
        print("=" * 30)
        print(f"计算结果:")
        print(f"  切向出射速度 V_x_out = {vx_out:.6f} m/s")
        print("=" * 30)

    except ValueError:
        print("\n输入错误：请输入有效的数字。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")

if __name__ == "__main__":
    main()