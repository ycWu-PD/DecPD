# 提取频域特征，功率谱密度、频谱峰值、频谱平均值、频谱能量和频谱熵
import os
import pandas as pd
import numpy as np
from scipy import signal
from tqdm import tqdm



# 定义特征提取函数
def extract_frequency_domain_features(data):
    num_segments = len(data) // 400  # 计算可提取特征的段数
    features = []

    for i in tqdm(range(num_segments)):
        segment = data[i * 400: (i + 1) * 400]  # 提取每个段的数据

        # 使用傅里叶变换计算功率谱密度
        f, psd = signal.welch(segment)
        spectral_peak = f[np.argmax(psd)]
        spectral_mean = np.mean(f * psd) / np.mean(psd)
        spectral_energy = np.sum(psd)
        spectral_entropy = -np.sum(psd * np.log2(psd + 1e-10))

        segment_features = [
            spectral_peak,
            spectral_mean,
            spectral_energy,
            spectral_entropy
        ]
        features.append(segment_features)

    return np.array(features)


def main():
    count = 1
    # 定义输入文件夹路径和输出文件夹路径
    # input_parent_folder = 'D:\Original_dataset\JIEYA\jieya_WN_500pc'
    input_parent_folder = 'D:\Original_dataset\JIEYA\jieya_normal'
    # output_parent_folder = 'D:\Original_dataset\JIEYA\WN_500PC_data-frequnecy'
    output_parent_folder = 'D:\Original_dataset\JIEYA\jieya_normal_data-frequnecy'

    # folders = ['111', '222', '333', '444', '555', '666']

    if not os.path.exists(output_parent_folder):
        os.makedirs(output_parent_folder)

    # for folder in folders:
    #     input_folder_path = os.path.join(input_parent_folder, folder)
        input_folder_path = os.path.join(input_parent_folder)
        # 获取输入文件夹中所有csv文件
        csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
        # 构建输出文件夹的完整路径
        # output_folder_path = os.path.join(output_parent_folder, folder)
        output_folder_path = os.path.join(output_parent_folder)

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

            # 提取频域特征
            features = extract_frequency_domain_features(data)

            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_folder_path, f"features_{csv_file}")

            # 将特征保存到文件
            pd.DataFrame(features).to_csv(output_file_path, index=False, header=None)

            print(f"已保存特征文件: {output_file_path}")

    # # 删除解压文件
    # delete_files_in_folder('D:\pd_FL_learning\code_projection\data\pd-csv(2) -\pd-csv(2) - 副本\有干扰（电力电子）200pc\jieya')


if __name__ == '__main__':
    main()
