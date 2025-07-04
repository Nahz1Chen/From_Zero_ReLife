import math

import torch
from torch import nn
from torch import F


class TokenEmbedding(nn.Embedding):
    def __int__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__int__(vocab_size, d_model, padding_idx=1)


class PostionalEmbedding(nn.Module):
    def __int__(self, d_model, maxlen, device):
        super(PostionalEmbedding, self).__int__()
        self.encoding = torch.zeros(maxlen, d_model, device)
        self.encoding.requires_grad_(False)

        pos = torch.arange(0, maxlen, device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]


class TranformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TranformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PostionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask is not None:
            mask = torch.tril(torch.ones(time, time, dtype=bool))
            score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        return output


class EncoderLayer(nn.Module):
    def __int__(self, d_model, ffn_hidden, n_head, drop_prob) -> None:
        super(EncoderLayer, self).__int__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x
