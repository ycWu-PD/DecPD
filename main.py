"""
1、Provide half of the data for verifying the authenticity of the experiment

2、Data related to power grid privacy, leakage will be investigated

"""
import time
import csv
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.DecPD_Net import DecPD
from src.improved_Focal_Loss import Focal_Loss
from src.Visualization import draw_picture_save_date
from src.Visualization import plot_confusion_matrix
from src.option import args_parser


class myDataSet(Dataset):
    def __init__(self, data_dir, label_dir):
        """
        :param data_dir: 数据文件路径
        :param label_dir: 标签文件路径
        """

        # 读文件夹下每个数据文件名称
        # os.listdir读取文件夹内的文件名称
        self.file_name = os.listdir(data_dir)

        self.data_path = []
        self.label_path = label_dir
        self.labels = pd.read_csv(self.label_path, header=None, skiprows=1)

        # 让每一个文件的路径拼接起来
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name[index]))
            # self.label_path.append(os.path.join(label_dir, self.label_name[index]))

    def __len__(self):
        # 返回数据集长度
        return len(self.file_name)

    def __getitem__(self, index):
        # 获取每一个数据

        # 读取数据
        data = pd.read_csv(self.data_path[index], header=None, skiprows=1)
        index_data = int(self.data_path[index][11:-4])
        # 读取标签

        # label = label[index, 3]
        label = self.labels.iloc[index_data, 3]

        # 转成张量
        numpy_array = data.values
        data = torch.tensor(numpy_array, dtype=torch.float32)
        # data = torch.tensor(data.values)
        label = torch.tensor(label)

        return index_data, data, label


def main_PD():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = torch.FloatTensor([1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]).to(device)

    train_iter = DataLoader(train_dataset, args.batch_size, drop_last=True, shuffle=True)
    val_iter = DataLoader(val_dataset, args.batch_size, drop_last=True, shuffle=True)
    test_iter = DataLoader(test_dataset, args.batch_size, drop_last=True, shuffle=True)

    model = DecPD(args.input_size, args.hidden_size, args.num_layers, args.num_classes, device).to(device)
    # criterion = nn.CrossEntropyLoss(class_weights)
    criterion = Focal_Loss(alpha=alpha, par=args.par, a=args.a, reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 余弦退火算法
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=2, eta_min=1e-4)

    # 保存训练历史
    history_train = {'loss_train': [], 'loss_val': [], 'accuracy_train': [], 'accuracy_val': []}
    best_precision = 0
    best_precision_test = 0
    print("-------------------------started training--------------------------------")
    model.train()
    for epoch in range(args.num_epochs):
        model.train()
        time_start = time.time()
        loss_train = 0
        loss_val = 0
        optimizer.zero_grad()
        for index, data, label in train_iter:
            data = data.reshape(-1, args.sequence_length, args.input_size).to(device)
            label = label.to(device)
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, label)
            loss_train += loss
            loss.backward()

        # 记录训练损失函数值
        history_train['loss_train'].append(loss_train.item())
        time_epoch = time.time() - time_start
        print(
            f"\033[91m Epoch：{epoch + 1}/{args.num_epochs}, Loss：{loss_train.item():.4f}, Time：{time_epoch:.3f} \033[0m")

        model.eval()
        y_true_train = []
        y_pred_train = []
        with torch.no_grad():
            for index_train, data_train, label_train in train_iter:
                data_train = data_train.reshape(-1, args.sequence_length, args.input_size).to(device)
                label_train = label_train.to(device)
                # 前向传播
                outputs = model(data_train)
                _, predicted_train = torch.max(outputs.data, 1)
                y_true_train.extend(label_train.cpu().numpy())
                y_pred_train.extend(predicted_train.cpu().numpy())
        # 计算准确率和召回率
        accuracy_train = accuracy_score(y_true_train, y_pred_train)
        confusionmatrix_train = confusion_matrix(y_true_train, y_pred_train)
        history_train['accuracy_train'].append(accuracy_train)
        print(f"train_test Accuracy: {accuracy_train:.4f}\nconfusionmatrix_train:\n{confusionmatrix_train}")

        y_true_val = []
        y_pred_val = []
        with torch.no_grad():
            for index_val, data_val, label_val in val_iter:
                data_val = data_val.reshape(-1, args.sequence_length, args.input_size).to(device)
                label_val = label_val.to(device)
                # 前向传播
                outputs_val = model(data_val)
                loss = criterion(outputs_val, label_val)
                loss_val += loss
                _, predicted_val = torch.max(outputs_val.data, 1)
                # predicted = (outputs > threshold).int()
                y_true_val.extend(label_val.cpu().numpy())
                y_pred_val.extend(predicted_val.cpu().numpy())
        # 计算准确率和召回率
        accuracy_val = accuracy_score(y_true_val, y_pred_val)
        history_train['loss_val'].append(loss_val.item())
        history_train['accuracy_val'].append(accuracy_val)
        recall = recall_score(y_true_val, y_pred_val, average='macro', zero_division=1)
        # 预测为正类的样本中，实际上属于正类的样本所占的比例
        precision = precision_score(y_true_val, y_pred_val, average='macro', zero_division=1)
        F1_score = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        confusionmatrix_val = confusion_matrix(y_true_val, y_pred_val)
        print(f"Validation Accuracy: {accuracy_val:.4f} \n"
              f"Validation recall:{recall:.4f}\nValidation precision:{precision:.4f}\nF1_Score:{F1_score:.4f}")

        if best_precision <= (confusionmatrix_val[0][0] + confusionmatrix_val[1][1] + confusionmatrix_val[2][2] +
                              confusionmatrix_val[3][3] +
                              confusionmatrix_val[4][4] + confusionmatrix_val[5][5] + confusionmatrix_val[6][6]):
            torch.save(model.state_dict(), f"my_models_ckpt_val//val_best" + '_model.ckpt')
            best_precision = (confusionmatrix_val[0][0] + confusionmatrix_val[1][1] + confusionmatrix_val[2][2] +
                              confusionmatrix_val[3][3] +
                              confusionmatrix_val[4][4] + confusionmatrix_val[5][5] + confusionmatrix_val[6][6])
            print(f"\033[94m Improve!!! Successfully val_predicted:{best_precision}\033[0m")
            print(f"confusionmatrix: \n{confusionmatrix_val}")

        else:
            print(f"confusionmatrix:\n{confusionmatrix_val}")

        #  Only for comparative reference
        y_true_test = []
        y_pred_test = []
        with torch.no_grad():
            for index_test, data_test, label_test in test_iter:
                data_test = data_test.reshape(-1, args.sequence_length, args.input_size).to(device)
                label_test = label_test.to(device)
                # 前向传播
                outputs_test = model(data_test)
                _, predicted_test = torch.max(outputs_test.data, 1)
                # predicted = (outputs > threshold).int()
                y_true_test.extend(label_test.cpu().numpy())
                y_pred_test.extend(predicted_test.cpu().numpy())
        # 计算准确率和召回率
        accuracy_test = accuracy_score(y_true_test, y_pred_test)
        confusionmatrix_test = confusion_matrix(y_true_test, y_pred_test)
        print(f"Test Accuracy: {accuracy_test:.4f}")
        if best_precision_test <= (
                confusionmatrix_test[0][0] + confusionmatrix_test[1][1] + confusionmatrix_test[2][2] +
                confusionmatrix_test[3][3] +
                confusionmatrix_test[4][4] + confusionmatrix_test[5][5] + confusionmatrix_test[6][6]):
            # torch.save(model.state_dict(), f"my_models_ckpt_test//test_best" + '_model.ckpt')
            best_precision_test = (
                    confusionmatrix_test[0][0] + confusionmatrix_test[1][1] + confusionmatrix_test[2][2] +
                    confusionmatrix_test[3][3] +
                    confusionmatrix_test[4][4] + confusionmatrix_test[5][5] + confusionmatrix_test[6][6])
            print(f"\033[95m Improve!!! Successfully test_predicted:{best_precision}\033[0m")
            print(f"confusionmatrix: \n{confusionmatrix_test}")
        else:
            print(f"confusionmatrix:\n{confusionmatrix_test}")
        optimizer.step()
        # scheduler.step(epoch)

    # 绘制训练集和验证集的损失曲线
    draw_picture_save_date(history_train['loss_train'], history_train['loss_val'], history_train['accuracy_val'],
                           history_train['accuracy_train'])


def test_PD(weight_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_iter = DataLoader(test_dataset, args.batch_size, drop_last=True, shuffle=True)
    weights = torch.load(weight_model)

    model = DecPD(args.input_size, args.hidden_size, args.num_layers, args.num_classes, device).to(device)
    model.load_state_dict(weights)

    print(f"\033[92m start test \033[0m")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for index_test1, data, label in test_iter:
            data = data.reshape(-1, args.sequence_length, args.input_size).to(device)
            label_test1 = label.to(device)
            # 前向传播
            outputs_test1 = model(data)
            _, predicted_test1 = torch.max(outputs_test1.data, 1)
            y_true.extend(label_test1.cpu().numpy())
            y_pred.extend(predicted_test1.cpu().numpy())

            # 记录分类错误数据
            index_test1_np = index_test1.cpu().numpy()
            combined_test_array = np.empty((len(index_test1), 3))
            combined_test_array[:, 0] = index_test1_np
            combined_test_array[:, 1] = label_test1.cpu().numpy()
            combined_test_array[:, 2] = predicted_test1.cpu().numpy()

            output_file = "my_models_ckpt_test//test_compare_model.csv"
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = ['index', 'true_label', 'predicted_label']
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                for row in combined_test_array:
                    writer.writerow(row)
        print(f"Save comparative data to: {output_file}")

    # 计算准确率和召回率
    accuracy_test = accuracy_score(y_true, y_pred)
    confusionmatrix = confusion_matrix(y_true, y_pred)

    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    F1_score = f1_score(y_true, y_pred, average='macro')
    print(f"test Accuracy: {accuracy_test:.4f}\ntest recall:{recall:.4f}\n"
          f"test precision:{precision:.4f}\nconfusionmatrix:\n{confusionmatrix}\nF1_Score:{F1_score:.4f}")

    classes = ['Normal', 'Defect 1', 'Defect 2', 'Defect 3', 'Defect 4', 'Defect 5', 'Defect 6']
    plot_confusion_matrix(y_true, y_pred, classes)


def train_ckpt(weight_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter = DataLoader(train_dataset, args.batch_size, drop_last=True)
    weights = torch.load(weight_model)

    model = DecPD(args.input_size, args.hidden_size, args.num_layers, args.num_classes, device).to(device)
    model.load_state_dict(weights)

    print(f"\033[93m start train_test \033[0m")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for _, data, label in train_iter:
            data = data.reshape(-1, args.sequence_length, args.input_size).to(device)
            label = label.to(device)
            # 前向传播
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    # 计算准确率和召回率
    accuracy = accuracy_score(y_true, y_pred)
    confusionmatrix = confusion_matrix(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    F1_score = f1_score(y_true, y_pred, average='macro')
    print(f"train_test Accuracy: {accuracy:.4f} \ntrain_test recall:{recall:.4f}\n"
          f"train_test precision:{precision:.4f}\ntrain_confusionmatrix:\n{confusionmatrix}\nF1_Score:{F1_score:.4f}")


def Val_PD(weight_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_iter = DataLoader(val_dataset, args.batch_size, drop_last=True, shuffle=True)
    weights = torch.load(weight_model)

    model = DecPD(args.input_size, args.hidden_size, args.num_layers, args.num_classes, device).to(device)
    model.load_state_dict(weights)

    print(f"\033[94m start val \033[0m")
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for index_test1, data, label in val_iter:
            data = data.reshape(-1, args.sequence_length, args.input_size).to(device)
            label_test1 = label.to(device)
            # 前向传播
            outputs_test1 = model(data)
            _, predicted_test1 = torch.max(outputs_test1.data, 1)
            y_true.extend(label_test1.cpu().numpy())
            y_pred.extend(predicted_test1.cpu().numpy())

    # 计算准确率和召回率
    accuracy = accuracy_score(y_true, y_pred)
    confusionmatrix = confusion_matrix(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    F1_score = f1_score(y_true, y_pred, average='macro')
    print(f"test Accuracy: {accuracy:.4f}\ntest recall:{recall:.4f}\n"
          f"test precision:{precision:.4f}\nconfusionmatrix:\n{confusionmatrix}\nF1_Score:{F1_score:.4f}")


if __name__ == "__main__":
    args = args_parser()
    data_dir = 'data/train'
    test_dir = 'data/test1'
    label_dir = 'data/label/jf-data-label.csv'

    # 读取数据集
    train_dataset = myDataSet(
        data_dir=data_dir,
        label_dir=label_dir,
    )
    test_dataset = myDataSet(
        data_dir=test_dir,
        label_dir=label_dir,
    )
    # train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.1, random_state=random_Seed, shuffle=True)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.25, random_state=args.random_Seed,
                                                  shuffle=True)
    train_samples = len(train_dataset)
    test_samples = len(test_dataset)
    val_samples = len(val_dataset)
    print(
        f"\nNumber of training sets：{train_samples} "
        f"\nNumber of Verification set：{val_samples} "
        f"\nNumber of Testing set：{test_samples}")

    weight_val_ckpt = "my_models_ckpt_val//val_best" + '_model.ckpt'
    main_PD()
    test_PD(weight_val_ckpt)
    # train_ckpt(weight_val_ckpt)
    # Val_PD(weight_val_ckpt)
    # for i in range(20):
    #     test_PD(weight_val_ckpt)
    print("\n╭◜ oh ha It's over ԅ(≖‿≖ԅ)◝╮ ")
