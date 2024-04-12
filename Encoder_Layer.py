import torch
from torch import nn


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
        seq_len=x.shape[1]
        return self.encoding[:seq_len,:]
    
