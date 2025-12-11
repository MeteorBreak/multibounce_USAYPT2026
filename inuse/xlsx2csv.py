import pandas as pd
import os
import sys

def process_excel_to_csv(file_path, camera_fps, tracker_fps):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    try:
        # 1. 读取Excel文件
        # 用户反馈新的数据可能没有第二行作为表头的情况，改为默认 header=0 (第一行作为表头)
        print(f"正在读取文件: {file_path} ...")
        df = pd.read_excel(file_path, header=0)

        # 2. 定义需要保留的列
        # 注意：这里使用了希腊字母 ω (small omega)
        # 根据用户需求，主要关注 t, y, vx, vy，但也保留其他可能存在的列
        target_columns = ['t', 'y', 'vx', 'vy', 'x', 'ω']

        # 检查这些列是否都在文件中，防止报错
        # 有时候列名可能有空格，先去除空格
        df.columns = df.columns.str.strip()
        
        missing_cols = [col for col in target_columns if col not in df.columns]
        if missing_cols:
            # print(f"警告: 文件中缺少以下列: {missing_cols}") # 减少干扰信息
            pass
            
        # 只保留存在的列
        existing_cols = [col for col in target_columns if col in df.columns]
        if not existing_cols:
            print("错误: 未找到任何目标列 (t, x, y, vx, vy, ω)。请检查Excel文件的列名。")
            return
            
        df = df[existing_cols]

        # 3. 将数据转化为纯小数 (处理科学记数法)
        # pandas 的 to_numeric 会自动处理 '1.23E-04' 这种格式并转为浮点数
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 数据换算 (根据帧率调整时间和速度)
        # 时间缩放因子: tracker_fps / camera_fps
        # 速度缩放因子: camera_fps / tracker_fps (因为 v = d/t，时间变长，速度变慢)
        if camera_fps != tracker_fps:
            print(f"正在进行帧率换算 (Camera: {camera_fps} fps, Tracker: {tracker_fps} fps)...")
            time_scale = tracker_fps / camera_fps
            vel_scale = camera_fps / tracker_fps
            
            if 't' in df.columns:
                df['t'] = df['t'] * time_scale
            
            # 速度相关列
            # 注意：y 是位置坐标，不随帧率改变（它是每一帧的物理位置快照），因此不需要换算。
            # 只有涉及时间维度的量（如速度 vx, vy, 角速度 ω）才需要根据帧率缩放。
            for v_col in ['vx', 'vy', 'ω']:
                if v_col in df.columns:
                    df[v_col] = df[v_col] * vel_scale

        # 5. 输出为 .csv 文件
        output_path = os.path.splitext(file_path)[0] + ".csv"
        
        # float_format='%.10f' 可以强制输出非科学记数法的小数，
        # 但如果不指定，pandas通常也会默认输出标准小数格式（除非数值极小）。
        # 根据你的附件样例，默认格式通常效果最好，因为它不会截断精度或增加无用的0。
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"成功! 已输出文件至: {output_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    # 获取用户输入
    input_path = input("请输入.xlsx文件的相对路径: ").strip()
    # 去除可能存在的引号（如果用户是拖拽文件进终端的）
    input_path = input_path.replace("'", "").replace('"', "")
    
    try:
        c_fps = float(input("请输入摄像机原始帧率 (Camera FPS): ").strip())
        t_fps = float(input("请输入Tracker处理帧率 (Tracker FPS): ").strip())
    except ValueError:
        print("错误: 帧率必须是数字。")
        sys.exit(1)
    
    process_excel_to_csv(input_path, c_fps, t_fps)