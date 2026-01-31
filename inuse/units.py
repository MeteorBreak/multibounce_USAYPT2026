import pandas as pd
import argparse
import os

def process_csv(n, file_path, move=None):
    # 获取绝对路径
    abs_path = os.path.abspath(file_path)
    
    if not os.path.exists(abs_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    try:
        # 读取CSV，header=None 假设数据无表头
        df = pd.read_csv(abs_path, header=None)
        
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        name, ext = os.path.splitext(filename)

        if move is not None:
            # 偏移操作
            if len(df.columns) == len(move):
                df_result = df + move
            else:
                df_result = df.add(move, axis=1)

            # 格式化文件名中的偏移标识
            move_strs = [str(int(x)) if x.is_integer() else str(x) for x in move]
            suffix = f"_m_{'_'.join(move_strs)}"
        else:
            # 乘法操作
            df_result = df * n
            
            # 格式化文件名中的倍数标识
            n_str = str(int(n)) if n.is_integer() else str(n)
            suffix = f"_x{n_str}"
        
        new_filename = f"{name}{suffix}{ext}"
        new_path = os.path.join(directory, new_filename)
        
        # 保存新的CSV
        # float_format='%.4f' 用于指定保留小数点后4位
        df_result.to_csv(new_path, index=False, header=False, float_format='%.4f')
        print(f"处理完成。已保存至: {new_path}")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取CSV文件并将所有元素乘以n，保留4位小数。也可使用 -m 选项进行平移。")
    parser.add_argument("n", type=float, help="乘数 (如果使用了 -m 选项，此参数将被忽略，但仍需提供占位值)")
    parser.add_argument("file_path", type=str, help="CSV文件的相对路径")
    parser.add_argument("-m", "--move", nargs=2, type=float, metavar=('X', 'Y'), help="二维偏移量。启用此选项时，将忽略乘数 n，改为将数据加上偏移量。")
    
    args = parser.parse_args()
    
    process_csv(args.n, args.file_path, args.move)