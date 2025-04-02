import torch
import torch.nn as nn

class PositionalEncoding:  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

if '__main__' == __name__:
    num_hidden = 20
    # num_hidden = 256
    encoding = PositionalEncoding(num_hidden, 0)
    X = torch.ones((1, 2, num_hidden))
    print(encoding.forward(X))
