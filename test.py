import torch
import torch.nn.functional as F

# 模拟神经网络的输出，假设有三个类别
logits = torch.tensor([[1.0, 1, 1.0]])

# 使用 F.softmax 将输出转换为概率分布
probabilities = F.softmax(logits, dim=0)

print("Softmax probabilities:", probabilities)
