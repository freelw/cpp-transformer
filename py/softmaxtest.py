import torch
import torch.nn as nn

# 构造输入特征 x
x = [
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6],
]

labels = [2, 3]

# Convert labels to integer class indices
labels = torch.tensor(labels, dtype=torch.long)

# 将输入特征 x 转换为张量

x = torch.tensor(x, dtype=torch.float32)

x.requires_grad = True


softmax_size = 6

# 初始化一个 Softmax 层

softmax = nn.Softmax(dim=1)

# 将输入特征 x 传入 Softmax 层

y = softmax(x)
y.retain_grad()

loss = nn.CrossEntropyLoss()

# 计算交叉熵损失

loss_value = loss(y, labels)

print("Softmax 层的输出：", y)

# 打印交叉熵损失

print("交叉熵损失：", loss_value)

loss_value.backward()

# print x grad
print("x grad:", x.grad)

# print y grad
print("y grad:", y.grad)
