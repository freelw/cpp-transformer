import torch
import torch.nn as nn
import math

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

def test_atttion():
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

    valid_lens = torch.tensor([2, 4])

    attention = DotProductAttention(dropout=0)
    attention.eval()
    res = attention(queries, keys, values, valid_lens)

    print("res:", res)

    labels = [2, 3]

    #convert labels to tensor

    labels = torch.tensor(labels, dtype=torch.long)

    loss = nn.CrossEntropyLoss()

    res = res.reshape(-1, res.shape[-1])

    # print res again

    print("Reshaped res:", res)

    loss_value = loss(res, labels)

    print("loss_value:", loss_value)

    loss_value.backward()

    print("queries.grad:", queries.grad)
    print("keys.grad:", keys.grad)
    print("values.grad:", values.grad)


if '__main__' == __name__:
    test_atttion()