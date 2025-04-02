import torch
import torch.nn as nn

# 假设输入特征的维度为 10
normalized_shape = 6
layer_norm = nn.LayerNorm(normalized_shape)

# 设定 gamma（weight）的初始化值
# 例如，将 gamma 初始化为全 1
nn.init.ones_(layer_norm.weight)

# 设定 beta（bias）的初始化值
# 例如，将 beta 初始化为全 0
nn.init.zeros_(layer_norm.bias)

# 打印初始化后的 gamma 和 beta
print("Gamma (weight):", layer_norm.weight)
print("Beta (bias):", layer_norm.bias)

# 构造输入特征 x
x = [
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6],
]

# 将输入特征 x 转换为张量

x = torch.tensor(x, dtype=torch.float32)

x.requires_grad = True

# 将输入特征 x 传入 LayerNorm 层

y = layer_norm(x)

# 打印 LayerNorm 层的输出

print("LayerNorm 层的输出：", y)

# 打印 LayerNorm 层的输出的均值

print("LayerNorm 层的输出的均值：", y.mean())

# 打印 LayerNorm 层的输出的方差

print("LayerNorm 层的输出的方差：", y.var())

labels = [2, 3]

# Reshape y to have a batch size of 1 and 6 classes
#y = y.unsqueeze(0)

# print y again

print("Reshaped y:", y)

# Convert labels to integer class indices
labels = torch.tensor(labels, dtype=torch.long)

loss = nn.CrossEntropyLoss()

# 计算交叉熵损失

loss_value = loss(y, labels)

# 打印交叉熵损失

print("交叉熵损失：", loss_value)

loss_value.backward()

# 打印 LayerNorm 层的 gamma 和 beta 的梯度

print("Gamma (weight) 的梯度：", layer_norm.weight.grad)

print("Beta (bias) 的梯度：", layer_norm.bias.grad)

# 打印 LayerNorm 层的输出的梯度

print("LayerNorm 层的输出的梯度：", x.grad)