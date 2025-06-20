import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import entropy
import pywt


def band_stop_filter(data, fs, f_range, order):
    nyquist = 0.5 * fs
    low = f_range[0] / nyquist
    high = f_range[1] / nyquist
    b, a = signal.butter(order, [low, high], btype='bandstop')
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def main():
    # 定义输入文件夹路径和输出文件夹路径
    input_parent_folder = r'D:\Original_dataset\Data_PD\JIEYA\合并解压\PEI_200pc'
    output_folder = r'D:\Original_dataset\Data_PD\PEI_200PC\Bandstop_filtering\PEI_FILTER'  # 输出CSV文件夹路径

    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    global index

    # 循环处理每个csv文件
    for index in range(1363):

        # 构建输入文件名
        input_file = f'{index}.csv'
        input_file_path = os.path.join(input_parent_folder, input_file)

        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f'文件 {input_file} 不存在，跳过处理')
            continue

        # 读取CSV文件的数据列（假设数据列名称为'MyData'）
        df = pd.read_csv(input_file_path)
        data_time = df['in s'].values
        data_phase = df['C1 in V'].values
        data_voltage = df['C2 in V'].values

        # 生成示例信号
        fs = 120  # 采样率  # 50hz 采样
        t = data_time
        f1 = 15  # 需要滤除的频率范围起始频率
        f2 = 45  # 需要滤除的频率范围结束频率

        dft_result = np.fft.fft(data_voltage)
        freq_dft = np.fft.fftfreq(len(data_voltage), 1 / fs)

        # 带阻滤波
        filtered_data = band_stop_filter(data_voltage, fs, [f1, f2], order=4)

        # denoised_signal = wavelet_denoise(filtered_data)

        df['C2 in V'] = filtered_data

        # 构建输出文件名
        output_file = os.path.join(output_folder, f'{index}.csv')

        # 保存到新的CSV文件中
        df.to_csv(output_file, index=False)

        print(f'文件 {input_file} 处理完成，保存为 {output_file}')


if __name__ == '__main__':
    main()
