import torch
import torch.nn as nn

#构建模型
class Net1(nn.Module):
    def __init__(self, input_length):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * (input_length // 2), 4)  # 根据输入长度计算全连接层的大小

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度顺序
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.max_pool1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return torch.softmax(self.fc(x), dim=1)



if __name__ == '__main__':
    pass