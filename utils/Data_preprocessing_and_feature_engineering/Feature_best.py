# 提取时域特征，均值、标准差、最小值、最大值、峰值因子、偏度和峰度、百分位数
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import entropy
import pywt


# 定义特征提取函数
def extract_time_domain_features(data):
    num_segments = len(data) // 400  # 计算可提取特征的段数
    features = []

    for i in tqdm(range(num_segments)):
        segment = data[i * 400: (i + 1) * 400]  # 提取每个段的数据

        peaks, properties = find_peaks(segment[:, 2], height=None, threshold=None, prominence=0.0008)  # 峰值 突出度 # 0.001

        peaks_voltages = segment[peaks, 2]

        f, psd = signal.welch(segment[:, 2])  # f：采样频率阵列 ；psd 功率谱密度，由于电网白噪声是在功率谱密度上功率谱密度为常数的随机信号

        mean = np.mean(segment[:, 2])  # 平均值
        max = np.max(np.abs(segment[:, 2]))

        std = np.std(segment[:, 2])  # 标准差
        par = np.max(np.abs(segment[:, 2])) / (np.sqrt(2) * np.std(segment[:, 2]))  # 峰值因数form factor

        """
        分布相对于正态分布的不对称程度。正偏度表示分布是不对称的，左侧较大，零偏度表示分布是对称的，负偏度表示分布不对称，右侧较大
        """
        skew = pd.Series(segment[:, 2]).skew()  # 偏度
        """
        峰度是分布相对于正态分布的尖锐程度。零峰度表示分布为正态分布，正峰度表示分布为尖峰形，负峰度表示分布为平坦形
        """
        kurt = pd.Series(segment[:, 2]).kurtosis()  # 峰度

        percentile0 = np.percentile(segment[:, 2], 0)  # 分位点
        percentile1 = np.percentile(segment[:, 2], 1)
        percentile10 = np.percentile(segment[:, 2], 10)
        percentile25 = np.percentile(segment[:, 2], 25)
        percentile50 = np.percentile(segment[:, 2], 50)
        percentile75 = np.percentile(segment[:, 2], 75)
        percentile90 = np.percentile(segment[:, 2], 90)
        percentile99 = np.percentile(segment[:, 2], 99)
        percentile100 = np.percentile(segment[:, 2], 100)

        relative_percentiles0 = percentile0 - mean  # 相对百分位数
        relative_percentiles1 = percentile1 - mean
        relative_percentiles10 = percentile10 - mean
        relative_percentiles25 = percentile25 - mean
        relative_percentiles50 = percentile50 - mean
        relative_percentiles75 = percentile75 - mean
        relative_percentiles90 = percentile90 - mean
        relative_percentiles99 = percentile99 - mean
        relative_percentiles100 = percentile100 - mean

        spectral_peaks = f[np.argmax(psd)]  # 频谱峰值
        spectral_mean = np.mean(f * psd) / np.mean(psd)  # 频谱平均值
        spectral_energy = np.sum(psd)  # 频谱能量
        spectral_entropy = -np.sum(psd * np.log2(psd + 1e-10))  # 频谱熵

        # peak_heights = properties['peak_height']
        # 瞬态脉冲提取
        if peaks.size == 0:  # 没有脉冲时

            peaks_voltages_max = 0
            peaks_voltages_min = 0
            widths_mean = 0
            width_max = 0
            width_min = 0
            prominence_mean = 0
            prominence_max = 0
            prominence_min = 0
        else:
            peaks_voltages_max = np.max(peaks_voltages)
            peaks_voltages_min = np.min(peaks_voltages)

            # rms = np.sqrt(np.mean(peaks_voltages ** 2))
            # # 峰值因数
            # peaks_fac = np.max(peaks_voltages) / rms  # 峰值因数
            # # 形状因数
            # peaks_cre = rms / np.mean(peaks_voltages)  # 形状因数

            widths = signal.peak_widths(segment[:, 2], peaks)[0]
            prominences = signal.peak_prominences(segment[:, 2], peaks)[0]
            if widths.size == 0:  # 单个峰值无法计算时
                widths_mean = 0
                width_max = 0
                width_min = 0
            else:
                widths_mean = widths.mean()
                width_max = widths.max()
                width_min = widths.min()

            prominence_mean = prominences.mean()
            prominence_max = prominences.max()
            prominence_min = prominences.min()

        segment_features = [

            mean,
            std,
            max,
            par,
            skew,
            kurt,

            spectral_peaks,
            spectral_mean,
            spectral_entropy,
            spectral_energy,

            percentile0,
            percentile1,
            percentile10,
            percentile25,
            percentile50,
            percentile75,
            percentile90,
            percentile99,
            percentile100,

            relative_percentiles0,
            relative_percentiles1,
            relative_percentiles10,
            relative_percentiles25,
            relative_percentiles50,
            relative_percentiles75,
            relative_percentiles90,
            relative_percentiles99,
            relative_percentiles100,

            # discharge_magnitude_pC,

            # # 瞬态脉冲
            peaks_voltages_max,
            peaks_voltages_min,

            widths_mean,
            width_max,
            width_min,
            #
            prominence_mean,
            prominence_max,
            prominence_min

        ]
        features.append(segment_features)

    return np.array(features)


def main():
    # 定义输入文件夹路径和输出文件夹路径
    input_parent_folder = r'D:\Original_dataset\Data_PD\Seven_classification\wavelet_noraml'

    output_parent_folder = r'D:\Original_dataset\Data_PD\Seven_classification\wavelet_denoise_Normal'

    column_names = ['mean', 'std',
                    'max',
                    # 'min',
                    'par', 'skew', 'kurt',

                    'spectral_peaks',
                    'spectral_mean',
                    'spectral_entropy',
                    'spectral_energy',

                    'percentile0',
                    'percentile1', 'percentile10',
                    'percentile25',
                    'percentile50',
                    'percentile75',
                    'percentile90',
                    'percentile99',
                    'percentile100',

                    'relative_percentiles0', 'relative_percentiles1',
                    'relative_percentiles10', 'relative_percentiles25', 'relative_percentiles50',
                    'relative_percentiles75',
                    'relative_percentiles90', 'relative_percentiles99', 'relative_percentiles100',

                    'peaks_voltages_max',
                    'peaks_voltages_min',

                    'widths_mean',
                    'width_max',
                    'width_min',

                    'prominence_mean',
                    'prominence_max',
                    'prominence_min'
                    ]

    if not os.path.exists(output_parent_folder):
        os.makedirs(output_parent_folder)

    global index
    # 循环处理每个csv文件
    for index in range(1000):

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

        # # 归一化数据
        # min_val = np.min(data_voltage)
        # max_val = np.max(data_voltage)
        # data_voltage = -1 + 2 * (data_voltage - min_val) / (max_val - min_val)

        data = np.column_stack((data_time, data_phase, data_voltage))
        # 提取时域特征
        features = extract_time_domain_features(data)

        # 构建输出文件名
        output_file = os.path.join(output_parent_folder, f'{index}.csv')

        # 将去噪后的信号保存到新的CSV文件中
        pd.DataFrame(features).to_csv(output_file, index=False, header=column_names)

        print(f'文件 {input_file} 特征提取完成，保存为 {output_file}')


if __name__ == '__main__':
    main()
