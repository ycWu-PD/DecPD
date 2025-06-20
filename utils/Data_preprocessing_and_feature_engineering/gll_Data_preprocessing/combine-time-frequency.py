# 将提取出来的时域、频域特征结合起来，频域特征列在时域特征列的右面
"""
   时域: '平均值', '标准差', '最小值', '最大值', '峰值因子', '偏度', '峰度', '25分位数', '50分位数', '75分位数',
   频域：'频谱峰值', '频谱平均值', '频谱能量','频谱熵'
    label: PEI:1-6 WN:7-12
"""
import os
import pandas as pd
import shutil

folders = ['111', '222', '333', '444', '555', '666']
data_parents_frequency_folder = 'D:\Original_dataset\JIEYA\WN_500PC_data-frequnecy'
data_parents_time_folder = 'D:\Original_dataset\JIEYA\WN_500PC_data-time'
out_parents_folder = 'D:\Original_dataset\JIEYA\WN_500PC_data-time-frequnecy'

label_time = 7


# 删除解压文件夹
def delete_files_in_folder(folder_clear_path):
    try:
        shutil.rmtree(folder_clear_path)
        print(f"成功删除{folder_clear_path}")
    except OSError as e:
        print(f"发生错误：{e.filename}{e.strerror}")


for folder in folders:
    # 定义数据文件夹路径
    data_frequency_folder = os.path.join(data_parents_frequency_folder, folder)
    data_time_folder = os.path.join(data_parents_time_folder, folder)

    # 定义输出文件夹路径
    output_folder = os.path.join(out_parents_folder, folder)

    # 检查并创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取两个文件夹中的 CSV 文件列表
    data_time_files = [f for f in os.listdir(data_time_folder) if f.endswith('.csv')]
    data_frequency_files = [f for f in os.listdir(data_frequency_folder) if f.endswith('.csv')]

    # 遍历同名文件进行数据拼接
    for file in data_time_files:
        if file in data_frequency_files:
            # 构建文件路径
            data_time_file_path = os.path.join(data_time_folder, file)
            data_frequency_file_path = os.path.join(data_frequency_folder, file)

            # 读取数据
            data_time = pd.read_csv(data_time_file_path)
            data_frequency = pd.read_csv(data_frequency_file_path)

            # 拼接数据
            merged_data = pd.concat([data_time, data_frequency], axis=1)

            # 添加列名
            merged_data.columns = ['平均值', '标准差', '最小值', '最大值', '峰值因子', '偏度', '峰度', '25分位数', '50分位数', '75分位数', '频谱峰值',
                                   '频谱平均值', '频谱能量',
                                   '频谱熵']

            # # 添加label值
            # merged_data['label'] = label_time

            # 构建输出文件路径
            output_file_path = os.path.join(output_folder, file)

            # 保存拼接后的数据
            merged_data.to_csv(output_file_path, index=False)

            print(f"已保存拼接文件: {output_file_path}")
    label_time = label_time + 1

# 删除解压文件
delete_files_in_folder(data_parents_frequency_folder)
delete_files_in_folder(data_parents_time_folder)

