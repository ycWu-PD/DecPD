"""

          在所有zip文件中，在文件下载解压时，有丢失导致zip文件中无csv文件，故解压后文件与解压包数不一致

"""
# 将文件夹里的zip压缩包里的csv文件解压并重新命名
import os
import zipfile
import csv
from tqdm import tqdm
import shutil


def extract_and_rename_zip_files(folder_path, output_folder_path):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_path, exist_ok=True)

    # 初始化计数器
    count = 1

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否为ZIP文件
        if os.path.isfile(file_path) and file_path.lower().endswith('.zip'):
            # 证明访问到了ZIP文件
            print('访问到ZIP文件:', file_path)


            # 打开ZIP文件
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # 遍历ZIP文件中的所有文件
                for file_info in zip_ref.infolist():
                    # 检查文件是否为CSV文件
                    if file_info.filename.lower().endswith('.csv'):
                        # 构建新的文件名
                        new_file_name = '{}.csv'.format(count)
                        new_file_path = os.path.join(output_folder_path, new_file_name)

                        # 解压并重命名CSV文件，并修改文件类型为CSV UTF-8 (逗号分隔)
                        with zip_ref.open(file_info, 'r') as source_file, open(new_file_path, 'w', newline='',
                                                                               encoding='UTF-8') as target_file:
                            csv_reader = csv.reader([line.decode('UTF-8') for line in source_file])
                            csv_writer = csv.writer(target_file, delimiter=',')
                            csv_writer.writerows(csv_reader)

                        print('解压和重命名文件:', new_file_path)

                        # 增加计数器
                        count += 1


def delete_files_in_folder(folder_clear_path):
    try:
        shutil.rmtree(folder_clear_path)
        print(f"成功删除{folder_clear_path}")
    except OSError as e:
        print(f"发生错误：{e.filename}{e.strerror}")


def main():
    folder_list = ['111', '222', '333', '444', '555', '666']
    for folder_name in tqdm(folder_list):
        folder_path = os.path.join('D:\Original_dataset\原始数据\pd-csv(2)\有干扰（白噪声）100pc',
                                   folder_name)
        output_folder_path = os.path.join(
            'D:\Original_dataset\Data_PD\JIEYA\jieya_WN_100pc',
            folder_name)
        extract_and_rename_zip_files(folder_path, output_folder_path)

    # # 指定文件夹路径和输出文件夹路径
    # folder_path = 'D:\Original_dataset\pd-csv(2)\正常'
    # output_folder_path = 'D:\Original_dataset\jieya_normal'

    #调用函数解压和重命名ZIP文件中的CSV文件，并保存为"CSV UTF-8 (逗号分隔)"类型
    extract_and_rename_zip_files(folder_path, output_folder_path)


if __name__ == '__main__':
    main()
