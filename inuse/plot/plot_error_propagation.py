import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='分析球体多次弹跳的误差传播 (Analyze error propagation).')
    
    # 添加 mode 参数，选择 'x', 'y' 或 'r'
    parser.add_argument('--mode', '-m', choices=['x', 'y', 'r'], default='r',
                        help='选择要绘制的偏移量类型: x (X轴偏移), y (Y轴偏移), r (距离原点的总偏移, 默认)')

    parser.add_argument('files', metavar='CSV_FILE', type=str, nargs=3,
                        help='三个CSV文件的相对路径，分别对应三次弹跳的数据 (Relative paths to the three CSV files)')

    args = parser.parse_args()
    
    # 用于存储每次弹跳的所有球的距离数据
    bounces_data = []

    # 遍历输入的三个文件路径
    for i, file_path in enumerate(args.files):
        if not os.path.exists(file_path):
            print(f"错误: 找不到文件 - {file_path}")
            sys.exit(1)
            
        try:
            # 读取CSV，假设没有表头，前两列为x和y
            # header=None 表示不将第一行作为列名
            df = pd.read_csv(file_path, header=None)
            
            if df.shape[1] < 2:
                print(f"错误: 文件 {file_path} 至少需要两列数据 (x, y)。")
                sys.exit(1)
                
            # 提取 x 和 y (前两列)
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            
            # 根据选择的模式计算偏移量
            if args.mode == 'x':
                data = x
            elif args.mode == 'y':
                data = y
            else: # r
                # 计算距离原点的偏移量 sqrt(x^2 + y^2)
                data = np.sqrt(x**2 + y**2)
            
            bounces_data.append(data.values)
            
        except Exception as e:
            print(f"读取或处理文件 {file_path} 时发生错误: {e}")
            sys.exit(1)

    # 检查球的数量是否一致
    num_balls = len(bounces_data[0])
    for i, data in enumerate(bounces_data):
        if len(data) != num_balls:
            print(f"错误: 文件 {args.files[i]} 包含 {len(data)} 个数据点，但第一个文件包含 {num_balls} 个。所有文件行数必须一致。")
            sys.exit(1)

    # 转置数据矩阵，使其形状为 (球的数量, 弹跳次数)
    # 这样每一行代表一个球在三次弹跳中的数据
    data_matrix = np.array(bounces_data).T 

    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 横轴坐标：1, 2, 3
    x_axis = [1, 2, 3]
    
    # 为每个球绘制一条折线
    for ball_idx in range(num_balls):
        plt.plot(x_axis, data_matrix[ball_idx, :], marker='o', label=f'Ball {ball_idx + 1}')

    # 根据不同模式设置图表标题和纵轴标签
    if args.mode == 'x':
        title_str = 'X Offset per Bounce'
        ylabel_str = 'X Offset (mm)'
    elif args.mode == 'y':
        title_str = 'Y Offset per Bounce'
        ylabel_str = 'Y Offset (mm)'
    else:
        title_str = 'Combined Offset per Bounce'
        ylabel_str = r'Offset from Origin (mm)'

    plt.title(title_str)
    plt.xlabel('Bounce Number')
    plt.ylabel(ylabel_str)
    plt.xticks(x_axis, ['Bounce 1', 'Bounce 2', 'Bounce 3'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放在外侧以免遮挡
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # 调整布局防止标签被截断

    # 保存图表，文件名包含模式以便区分
    output_filename = f'error_propagation_plot_{args.mode}.png'
    plt.savefig(output_filename)
    print(f"图表已保存至 {output_filename}")
    
    # 如果环境支持显示GUI，则显示图表
    try:
        plt.show()
    except Exception:
        pass

if __name__ == '__main__':
    main()