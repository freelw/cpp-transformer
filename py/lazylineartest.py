import torch
import torch.nn as nn
import math
from torch.nn.init import constant_


# 定义初始化权重的钩子函数
def init_weights(module, input):
    # print("init_weights 0")
    # print(module)
    if isinstance(module, nn.Linear):
        constant_(module.weight, 1.0)
        print("init_weights")
        # 移除钩子，保证只执行一次
        module._forward_pre_hooks.pop(list(module._forward_pre_hooks.keys())[0])


def test():

    W_q = nn.LazyLinear(3, bias=False)
    W_q.register_forward_pre_hook(init_weights)
    queries = [
        [0.1, 0.1],
    ]

    queries = torch.tensor(queries, dtype=torch.float32)
    queries.requires_grad = True
    res = W_q(queries)
    

    labels = [2]

    #convert labels to tensor

    labels = torch.tensor(labels, dtype=torch.long)

    loss = nn.CrossEntropyLoss()

    
    res.retain_grad()
    

    # print res again

    print("Reshaped res:", res)

    loss_value = loss(res, labels)

    print("loss_value:", loss_value)

    loss_value.backward()

    
    print("res.grad:", res.grad)

    print("query.grad:", queries.grad)
    



if '__main__' == __name__:
    test()
