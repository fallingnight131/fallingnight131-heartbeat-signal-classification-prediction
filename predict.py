import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import Net1  # 从 model.py 导入模型


# 数据加载
test = pd.read_csv('./datas/testA.csv')

# 处理测试数据
test_list = []
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])
test = pd.DataFrame(np.array(test_list))
test.columns = ['id'] + ['s_'+str(i) for i in range(len(test_list[0])-1)]

X_test = test.drop(['id'], axis=1)
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

# 加载模型
input_length = X_test.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net1(input_length).to(device)  # 使用导入的模型
model.load_state_dict(torch.load('model/my_model.pth', map_location=device))
model.eval()

# 预测
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions_nn1 = model(X_test_tensor).cpu().numpy()

# 保存预测结果
submit = pd.DataFrame()
submit['id'] = range(100000, 100000 + len(predictions_nn1))
submit['label_0'] = predictions_nn1[:, 0]
submit['label_1'] = predictions_nn1[:, 1]
submit['label_2'] = predictions_nn1[:, 2]
submit['label_3'] = predictions_nn1[:, 3]

# 打印前几行
print(submit.head())

# 保存结果
submit.to_csv(("./submit/submit.csv"), index=False)
print("预测结果已保存到 ./submit/submit.csv")
