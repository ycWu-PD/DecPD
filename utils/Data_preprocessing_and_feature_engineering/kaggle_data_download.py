import os
import pandas as pd
from tqdm import tqdm

# 从 Parquet 文件中读取数据
df = pd.read_parquet(r'D:\Original_dataset\open_pd_data\vsb\train.parquet')

# 设置输出路径
output_dir = r'D:\Original_dataset\Data_PD\NEW_FEATURE\2分类测试\open_data\train'
# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将每一列保存到对应的 CSV 文件
for column in tqdm(df.columns):
    column_df = df[[column]]
    output_path = os.path.join(output_dir, f'{column}.csv')
    column_df.to_csv(output_path, index=False, header=False)

print(f"CSV 文件已保存到 {output_dir} 文件夹中。")


