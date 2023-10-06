"""
@Author: yanzx
@Date: 2023/10/6 11:55
@Description: 
"""

import numpy as np
import torch
from torch import nn, optim

# 创建一些示例数据（特征和标签）
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 0, 0, 1])  # 0表示负类，1表示正类

# 定义逻辑回归模型参数
learning_rate = 0.01
num_iterations = 1000

X = torch.Tensor(X)
y = torch.Tensor(y)


class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层
        self.relu = nn.Sigmoid()  # 非线性激活函数
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层，输出为一个值

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = XORModel()

criterion = nn.MSELoss()  # 均方差误差损失
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 训练模型
for epoch in range(10000):  # 10000个epoch示例
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y.view(-1, 1))  # 将目标形状调整为与输出相同
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/10000], Loss: {loss.item()}")

# 评估模型
with torch.no_grad():
    predicted = model(X).round()  # 四舍五入为0或1
    print("Predicted:", predicted)
    print("Ground Truth:", y.view(-1, 1))
