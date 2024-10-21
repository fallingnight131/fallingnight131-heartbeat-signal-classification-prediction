import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from data_process import reduce_mem_usage
from early_stopping import EarlyStopping
from model import Net1  # 从 model.py 导入模型
import os


if __name__ == '__main__':
    # 数据加载
    train = pd.read_csv('./datas/train.csv')

    # 处理训练数据
    train_list = []
    for items in train.values:
        train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])
    train = pd.DataFrame(np.array(train_list))
    train.columns = ['id'] + ['s_' + str(i) for i in range(len(train_list[0])-2)] + ['label']
    train = reduce_mem_usage(train)

    y_train = train['label']
    x_train = train.drop(['id', 'label'], axis=1)

    # 处理SMOTE数据平衡
    knn = NearestNeighbors(n_jobs=-1)
    smote = SMOTE(random_state=2021, k_neighbors=knn)
    k_x_train, k_y_train = smote.fit_resample(x_train, y_train)
    k_x_train = np.array(k_x_train).reshape(k_x_train.shape[0], k_x_train.shape[1], 1)

    # 划分训练集和验证集
    k_x_train, k_x_val, k_y_train, k_y_val = train_test_split(k_x_train, k_y_train, test_size=0.2, random_state=2021)

    # 将数据转换为PyTorch的Dataset
    train_dataset = TensorDataset(torch.tensor(k_x_train, dtype=torch.float32), torch.tensor(k_y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(k_x_val, dtype=torch.float32), torch.tensor(k_y_val.values, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    input_length = k_x_train.shape[1]  # 计算输入的长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net1(input_length).to(device)  # 使用导入的模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 实例化 EarlyStopping
    early_stopping = EarlyStopping(patience=10, delta=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证集损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}')

        # 早停
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break

    # 保存模型

    # 检查并创建 model 文件夹
    if not os.path.exists('model'):
        os.makedirs('model')

    torch.save(model.state_dict(), 'model/my_model.pth')
    print("模型已保存到 model/my_model.pth")
