import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import MaxAbsScaler

# 设置特征文件夹路径
feature_folder = r'D:\Original_dataset\Data_PD\normal+WN_100pC\train(7_Classification)'
out_folder = r'D:\pd_FL_learning\code_projection\data\train'
# 获取所有csv文件
csv_files = [file for file in os.listdir(feature_folder) if file.endswith('.csv')]

# 初始化MinMaxScaler
scaler = MinMaxScaler()

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
# 遍历每个csv文件，进行归一化
for file in csv_files:
    file_path = os.path.join(feature_folder, file)
    df = pd.read_csv(file_path)
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    output_path = os.path.join(out_folder, file)
    scaled_df.to_csv(output_path, index=False)
    print(f"文件{file}已经成功特征归一化至{output_path}")
