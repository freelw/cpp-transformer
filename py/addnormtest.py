import torch
import torch.nn as nn

def get_qkv_labels1():
    queries = [
        [
            [3.5, 3.1, 3.1, 3.1],
        ],
        [
            [4.5, 4.1, 4.1, 4.1],
        ]
    ]
    queries = torch.tensor(queries, dtype=torch.float32)
    queries.requires_grad = True
    labels = [0, 0]
    labels = torch.tensor(labels, dtype=torch.long)
    return queries, labels

class AddNorm():  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
        # 设定 gamma（weight）的初始化值
        # 例如，将 gamma 初始化为全 1
        nn.init.ones_(self.ln.weight)
        # 设定 beta（bias）的初始化值
        # 例如，将 beta 初始化为全 0
        nn.init.zeros_(self.ln.bias)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

def test():
    queries, labels = get_qkv_labels1()
    addnorm = AddNorm(4, 0)
    res = addnorm.forward(queries, queries)
    print("res:", res)
    loss = nn.CrossEntropyLoss()
    res = res.reshape(-1, res.shape[-1])
    res.retain_grad()
    print("res:", res)
    print("labels:", labels)
    loss_val = loss(res, labels)
    print("loss_val:", loss_val)
    loss_val.backward()
    print("queries.grad:", queries.grad)
    print("res.grad:", res.grad)

    #print addnorm.ln gamma and beta grad
    print("addnorm.ln.weight.grad:", addnorm.ln.weight.grad)
    print("addnorm.ln.bias.grad:", addnorm.ln.bias.grad)


if '__main__' == __name__:
    test()