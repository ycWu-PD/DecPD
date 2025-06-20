import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def draw_picture_save_date(x1, y1, x2, y2):
    # 绘制训练集和验证集的损失曲线
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 4))

    axs[0].plot(x1, label='Train Loss')
    axs[0].plot(y1, label='Val Loss')
    axs[0].set_title('Train and Val Loss', fontsize=16)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # 绘制训练集和验证集的准确率曲线
    axs[1].plot(x2, label='Val Accuracy')
    axs[1].plot(y2, label='Train Accuracy')
    axs[1].set_yticks(np.arange(0, 1.1, 0.1))
    axs[1].set_title('Train and Val Accuracy', fontsize=16)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    history_train = {
        'loss_train': x1,
        'loss_val': y1,
        'accuracy_val': x2,
        'accuracy_train': y2,
    }

    # 指定输出路径
    output_directory = r'C:\Users\86178\Desktop\论文\消融实验\后期准确率测试'
    output_file = 'history_train_test.csv'
    output_path = os.path.join(output_directory, output_file)

    # 如果文件夹不存在，则创建
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 如果文件已经存在，则删除
    if os.path.exists(output_path):
        user_input = input("文件已存在! 输入 'Y' 删除并重建，输入 'N' 退出程序: \n")
        if user_input.lower() == 'Y':
            os.remove(output_path)
            # 创建 DataFrame
            df = pd.DataFrame(history_train)
            # 保存到 CSV 文件
            df.to_csv(output_path, index=False)
            print(f'已保存train_data至{output_path}')
        elif user_input == 'N':
            print("程序退出")
            exit()
    else:
        # 创建 DataFrame
        df = pd.DataFrame(history_train)
        # 保存到 CSV 文件
        df.to_csv(output_path, index=False)
        print(f'已保存train_data至{output_path}')


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'  # 设置格式为小数点后两位
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, format(cm_normalized[i, j], fmt) + '%',
                 ha="center", va="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
