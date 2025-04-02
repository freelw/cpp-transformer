import torch
import torch.nn as nn
from torch.nn.init import constant_

# 定义初始化权重的钩子函数
def init_weights(module, input):
    # print("init_weights 0")
    # print(module)
    if isinstance(module, nn.Linear):
        constant_(module.weight, 1)
        constant_(module.bias, 0)
        module.weight.data[0, 0] = 0.1
        # eye_(module.weight)
        print("init_weights")
        # 移除钩子，保证只执行一次
        module._forward_pre_hooks.pop(list(module._forward_pre_hooks.keys())[0])

def get_qkv_labels1():
    queries = [
        [3.5, 3.1, 3.1, 3.1],
        [4.5, 4.1, 4.1, 4.1]
    ]
    queries = torch.tensor(queries, dtype=torch.float32)
    queries.requires_grad = True
    labels = [0, 0]
    labels = torch.tensor(labels, dtype=torch.long)
    return queries, labels

class PositionWiseFFN():  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens, bias=True)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs, bias=True)

        self.dense1.register_forward_pre_hook(init_weights)
        self.dense2.register_forward_pre_hook(init_weights)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

def test():
    queries, labels = get_qkv_labels1()
    ffn = PositionWiseFFN(8, 5)
    res = ffn.forward(queries)
    print("res:", res)
    loss = nn.CrossEntropyLoss()
    res = res.reshape(-1, res.shape[-1])
    res.retain_grad()
    print(ffn.dense1.weight)
    print(ffn.dense2.weight)
    print("res:", res)
    print("labels:", labels)
    loss_val = loss(res, labels)
    print("loss_val:", loss_val)
    loss_val.backward()
    print("queries.grad:", queries.grad)
    print("res.grad:", res.grad)

if '__main__' == __name__:
    test()