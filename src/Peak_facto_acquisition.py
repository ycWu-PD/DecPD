import os
import pandas as pd
import numpy as np


def calculate_par(file_path):
    """读取CSV文件并计算par值（针对第四列）。"""
    data = pd.read_csv(file_path)

    # 获取第四列的数据（索引从0开始，第四列是 data.iloc[:, 3]）
    par_value = data.iloc[:, 3].values

    # # 计算 max(|x_PD|) 和 σ_PD
    # max_abs = np.max(np.abs(column_data))
    # std_dev = np.std(column_data)
    #
    # # 避免除以0的错误
    # if std_dev == 0:
    #     return None  # 或返回 float('inf')
    #
    # # 计算 par 值
    # par_value = max_abs / (np.sqrt(2) * std_dev)
    return par_value


def process_folder(folder_path):
    """遍历文件夹中的所有CSV文件并计算par值。"""
    par_values = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            par_value = calculate_par(file_path)
            if par_value is not None:
                par_values.append(par_value)

    return par_values


# 使用示例
folder_path = r'D:\Original_dataset\Data_PD\Seven_classification\normal_PEI_200pc\train_data'  # 替换为实际的文件夹路径
par_values = process_folder(folder_path)

# 计算并输出最大值、平均值、最小值
if par_values:
    max_par = np.max(par_values)
    mean_par = np.mean(par_values)
    min_par = np.min(par_values)

    print(f"最大值 (Max): {max_par}")
    print(f"平均值 (Mean): {mean_par}")
    print(f"最小值 (Min): {min_par}")
else:
    print("没有计算出有效的par值。")
