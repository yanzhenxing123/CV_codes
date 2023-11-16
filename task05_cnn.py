"""
@Time : 2023/11/15 10:34
@Author : yanzx
@Description :
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F  # 激活函数

from torch.utils.data import DataLoader, random_split


# 1. 定义模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.softmax(x)
        return x


# 2. 定义损失函数和优化器
model = SimpleCNN(num_classes=15)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 3. 加载数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_path = 'imgs/scene_categories'
img_dataset = ImageFolder(root=dataset_path, transform=transform)

data_loader = DataLoader(img_dataset, batch_size=32, shuffle=True)


# 假设你有一个名为 img_dataset 的数据集
dataset_size = len(img_dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(img_dataset, [train_size, test_size])


# 创建用于训练和测试的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    accuracy = total_correct / total_samples
    end = time.time()
    print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}, Cost time: {(end-start):.4f}')

# 5. 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy}')
