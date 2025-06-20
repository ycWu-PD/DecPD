# 提取时域特征，均值、标准差、最小值、最大值、峰值因子、偏度和峰度、百分位数
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


# 定义特征提取函数
def extract_time_domain_features(data):
    num_segments = len(data) // 400  # 计算可提取特征的段数
    features = []

    for i in tqdm(range(num_segments)):
        segment = data[i * 400: (i + 1) * 400]  # 提取每个段的数据
        segment_features = [
            np.mean(segment),
            np.std(segment),
            np.min(segment),
            np.max(segment),
            np.max(np.abs(segment)) / (np.sqrt(2) * np.std(segment)),
            pd.Series(segment).skew(),
            pd.Series(segment).kurtosis(),
            np.percentile(segment, [0, 1, 10, 25, 50, 75, 90, 99, 100]),

        ]
        features.append(segment_features)

    return np.array(features)


def main():
    # 定义输入文件夹路径和输出文件夹路径
    input_parent_folder = 'D:\Original_dataset\JIEYA\jieya_WN_100pc'
    # input_parent_folder = 'D:\Original_dataset\JIEYA\jieya_normal'
    output_parent_folder = 'D:\Original_dataset\JIEYA\WN_100PC_data-time'
    # output_parent_folder = 'D:\Original_dataset\JIEYA\jieya_normal_data-time'
    folders = ['111', '222', '333', '444', '555', '666']

    if not os.path.exists(output_parent_folder):
        os.makedirs(output_parent_folder)

    for folder in folders:
        input_folder_path = os.path.join(input_parent_folder, folder)
    #     input_folder_path = os.path.join(input_parent_folder)
        # 获取输入文件夹中所有csv文件
        csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
        # 构建输出文件夹的完整路径
        output_folder_path = os.path.join(output_parent_folder, folder)
        # output_folder_path = os.path.join(output_parent_folder)

        # 检查并创建输出文件夹
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # 循环处理每个csv文件
        for csv_file in csv_files:
            # 构建输入文件的完整路径
            input_file_path = os.path.join(input_folder_path, csv_file)

            # 读取CSV文件的数据列（假设数据列名称为'MyData'）
            df = pd.read_csv(input_file_path)
            data = df['C2 in V'].values

            # 提取时域特征
            features = extract_time_domain_features(data)

            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_folder_path, f"features_{csv_file}")

            # 将特征保存到文件
            pd.DataFrame(features).to_csv(output_file_path, index=False, header=None)

            print(f"已保存特征文件: {output_file_path}")


if __name__ == '__main__':
    main()
