import os
import pandas as pd
import numpy as np
import pywt

# 设置输入和输出文件夹路径
input_folder = r'D:\Original_dataset\Data_PD\JIEYA\合并解压\normal'  # 输入CSV文件夹路径
output_folder = r'D:\Original_dataset\Data_PD\Seven_classification\wavelet_noraml'  # 输出CSV文件夹路径

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理从0到200的文件
for index in range(1000):
    # 构建输入文件名
    input_file = f'{index}.csv'
    input_file_path = os.path.join(input_folder, input_file)

    # 检查输入文件是否存在
    if not os.path.exists(input_file_path):
        print(f'文件 {input_file} 不存在，跳过处理')
        continue

    # 读取CSV文件的数据列
    df = pd.read_csv(input_file_path)
    data_time = df['in s'].values
    data_phase = df['C1 in V'].values
    data_voltage = df['C2 in V'].values

    data = np.column_stack((data_time, data_phase, data_voltage))

    # 选择一列数据进行去噪
    column = 2  # 替换为你想要处理的列
    signal = data[:, column]

    # 定义小波去噪函数
    def wavelet_denoise(signal, wavelet='db4', level=1):
        coeffs = pywt.wavedec(signal, wavelet, mode="per")
        sigma = np.median(np.abs(coeffs[-level]))
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = (pywt.threshold(c, threshold, 'soft') for c in coeffs[1:])
        denoised_signal = pywt.waverec(coeffs, wavelet, mode="per")
        return denoised_signal

    # 进行小波去噪
    denoised_signal = wavelet_denoise(signal)

    # 将去噪后的信号添加到DataFrame中
    df['C2 in V'] = denoised_signal

    # 构建输出文件名
    output_file = os.path.join(output_folder, f'{index}.csv')

    # 保存到新的CSV文件中
    df.to_csv(output_file, index=False)

    print(f'文件 {input_file} 处理完成，保存为 {output_file}')
