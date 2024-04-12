import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)  # Query
        self.w_k = nn.Linear(d_model, d_model)  # Key
        self.w_v = nn.Linear(d_model, d_model)  # Value
        self.w_combine = nn.Linear(d_model, d_model)  # Combination
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(-2, -1) / math.sqrt(n_d)
        # Apply mask here if necessary
        attention_weights = self.softmax(score)
        weighted_v = attention_weights @ v  # Multiply by V
        weighted_v = weighted_v.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        output = self.w_combine(weighted_v)
        return output


# Example usage:
d_model = 512
n_head = 8
X = torch.randn(128, 64, d_model)
attention = MultiHeadAttention(d_model, n_head)
output = attention(X, X, X)
print(output, output.shape)
