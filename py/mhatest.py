import torch
import torch.nn as nn
import math
from torch.nn.init import constant_

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 定义初始化权重的钩子函数
def init_weights(module, input):
    # print("init_weights 0")
    # print(module)
    if isinstance(module, nn.Linear):
        constant_(module.weight, 1)
        module.weight.data[0, 0] = 0.1
        # eye_(module.weight)
        print("init_weights")
        # 移除钩子，保证只执行一次
        module._forward_pre_hooks.pop(list(module._forward_pre_hooks.keys())[0])

class MultiHeadAttention:
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_q.register_forward_pre_hook(init_weights)
        self.W_k.register_forward_pre_hook(init_weights)
        self.W_v.register_forward_pre_hook(init_weights)
        self.W_o.register_forward_pre_hook(init_weights)
        # constant_(self.W_q.weight, 1.0)
        # constant_(self.W_k.weight, 1.0)
        # constant_(self.W_v.weight, 1.0)
        # constant_(self.W_o.weight, 1.0)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

def get_qkv_labels1():
    queries = [
        [
            [0.1, 0.1],
        ],
        [
            [0.2, 0.2],
        ]
    ]

    queries = torch.tensor(queries, dtype=torch.float32)
    #queries = torch.normal(0, 1, (2, 1, 2))

    keys = [
        [
            [1.1, 1.1],
            [1.2, 1.2],
            [1.3, 1.3],
            [1.4, 1.4],
            [1.5, 1.5],
        ],
        [
            [2.1, 2.1],
            [2.2, 2.2],
            [2.3, 2.3],
            [2.4, 2.4],
            [2.5, 2.5],
        ]
    ]

    keys = torch.tensor(keys, dtype=torch.float32)

    values = [
        [
            [3.1, 3.1, 3.1, 3.1],
            [3.2, 3.2, 3.2, 3.2],
            [3.3, 3.3, 3.3, 3.3],
            [3.4, 3.4, 3.4, 3.4],
            [3.5, 3.5, 3.5, 3.5],
        ],
        [
            [4.1, 4.1, 4.1, 4.1],
            [4.2, 4.2, 4.2, 4.2],
            [4.3, 4.3, 4.3, 4.3],
            [4.4, 4.4, 4.4, 4.4],
            [4.5, 4.5, 4.5, 4.5],
        ]
    ]

    values = torch.tensor(values, dtype=torch.float32)

    queries.requires_grad = True
    keys.requires_grad = True
    values.requires_grad = True

    labels = [0, 0]
    labels = torch.tensor(labels, dtype=torch.long)

    return queries, keys, values, labels

def get_qkv_labels0():
    queries = [
        [
            [0.1, 0.1],
        ],
        [
            [0.2, 0.2],
        ]
    ]

    queries = torch.tensor(queries, dtype=torch.float32)
    #queries = torch.normal(0, 1, (2, 1, 2))

    keys = [
        [
            [1.1, 1.1],
            [1.2, 1.2],
        ],
        [
            [2.1, 2.1],
            [2.1, 2.1],
        ]
    ]

    keys = torch.tensor(keys, dtype=torch.float32)

    values = [
        [
            [3.1, 3.1],
            [3.2, 3.5],
        ],
        [
            [4.1, 4.1],
            [4.1, 4.1],
        ]
    ]

    values = torch.tensor(values, dtype=torch.float32)

    queries.requires_grad = True
    keys.requires_grad = True
    values.requires_grad = True

    labels = [0, 0]
    labels = torch.tensor(labels, dtype=torch.long)

    return queries, keys, values, labels

def print_grads_res(queries, keys, values, res):
    print("res:", res)  
    print("queries.grad:", queries.grad)
    print("keys.grad:", keys.grad)
    print("values.grad:", values.grad)
    print("res.grad:", res.grad)

def test0(valid_lens, queries, keys, values, labels, num_hidden, num_heads=1):
    
    attention = MultiHeadAttention(num_hidden, num_heads, 0, bias=False)
    res = attention.forward(queries, keys, values, valid_lens)
    loss = nn.CrossEntropyLoss()
    res = res.reshape(-1, res.shape[-1])
    res.retain_grad()
    loss_value = loss(res, labels)
    print("loss_value:", loss_value)
    loss_value.backward()
    print_grads_res(queries, keys, values, res)

def test_mha0():
    valid_lens = torch.tensor([5, 5])
    queries, keys, values, labels = get_qkv_labels0()
    test0(valid_lens, queries, keys, values, labels, 3)

def test_mha1():
    valid_lens = torch.tensor([5, 5])
    queries, keys, values, labels = get_qkv_labels1()
    test0(valid_lens, queries, keys, values, labels, 10)

def test_mha_with_mask():
    valid_lens = torch.tensor([2, 4])
    queries, keys, values, labels = get_qkv_labels1()
    test0(valid_lens, queries, keys, values, labels, 10, 2)

def test1(valid_lens):
    
    queries, keys, values, labels = get_qkv_labels0()
    attention = DotProductAttention(0)
    res = attention.forward(queries, keys, values, valid_lens)
    loss = nn.CrossEntropyLoss()
    res = res.reshape(-1, res.shape[-1])
    res.retain_grad()
    loss_value = loss(res, labels)
    print("loss_value:", loss_value)
    loss_value.backward()
    print_grads_res(queries, keys, values, res)

def test_attention():
    valid_lens = torch.tensor([5, 5])
    test1(valid_lens)

if '__main__' == __name__:

    # print ("------test_mha0------")
    # test_mha0()
    # print ("------test_mha0 end------")

    # print ("------test_mha1------")
    # test_mha1()
    # print ("------test_mha1 end------")
    # print ("------test_attention------")
    # test_attention()
    # print ("------test_attention end------")

    test_mha_with_mask()
