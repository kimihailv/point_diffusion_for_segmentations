import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, in_channels, dim=3):
        super().__init__()
        conv = nn.Conv3d if dim == 3 else nn.Conv1d

        self.q = conv(in_channels, in_channels, 1, bias=False)
        self.v = conv(in_channels, in_channels, 1, bias=False)
        self.k = conv(in_channels, in_channels, 1, bias=False)
        self.out = conv(in_channels, in_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.GroupNorm(8, in_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        bs = x.size(0)
        channels = x.size(1)

        q = self.q(x).view(bs, channels, -1)
        k = self.k(x).view(bs, channels, -1)
        v = self.v(x).view(bs, channels, -1)
        qk = torch.matmul(q.permute(0, 2, 1), k)
        w = self.softmax(qk)
        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(bs, channels, *x.shape[2:])
        h = self.out(h)
        return self.activation(self.norm(x + h))
