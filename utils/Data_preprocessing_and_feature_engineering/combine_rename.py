import os
import shutil

# 删除解压文件夹
def delete_files_in_folder(folder_clear_path):
    try:
        shutil.rmtree(folder_clear_path)
        print(f"成功删除{folder_clear_path}")
    except OSError as e:
        print(f"发生错误：{e.filename}{e.strerror}")

def merge_csv_files(source_folders, target_folder, start_index):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 初始化计数器
    index = start_index

    # 遍历每个源文件夹
    for source_folder in source_folders:
        # 获取源文件夹中的所有文件
        files = os.listdir(source_folder)

        # 遍历文件夹中的所有文件
        for file_name in files:
            # 检查文件是否为 CSV 文件
            if file_name.endswith('.csv'):
                # 构造新的文件名
                new_file_name = f"{index}.csv"

                # 构造源文件路径和目标文件路径
                old_file_path = os.path.join(source_folder, file_name)
                new_file_path = os.path.join(target_folder, new_file_name)

                # 重命名并移动文件
                shutil.copy(old_file_path, new_file_path)
                print(new_file_name)
                # 计数器递增
                index += 1


# 指定源文件夹路径、目标文件夹路径和起始索引
source_folders = [
    # r"D:\Original_dataset\Data_PD\Two-phase\already_features\normal"
    # r"D:\Original_dataset\Data_PD\NEW_FEATURE\bandstop_filter_test\filter_data\WN_100pc(1253)"

]
target_folder_path = r""
start_index = 1305

# 调用函数进行文件合并和重命名
merge_csv_files(source_folders, target_folder_path, start_index)
# 删除文件
# delete_files_in_folder(r"D:\Original_dataset\Data_PD\Two-phase\already_filter\1")