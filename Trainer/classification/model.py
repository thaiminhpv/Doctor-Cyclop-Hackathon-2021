import timm
import torch
import torch.nn as nn

from config import *


class B1_ns(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(B1_ns, self).__init__()
        self.enet = timm.create_model(enet_type, True)
        self.dropout = nn.Dropout(0.5)
        self.enet.conv_stem.weight = nn.Parameter(self.enet.conv_stem.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
        self.myfc = nn.Linear(self.enet.classifier.in_features, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        h = self.myfc(self.dropout(x))
        return h
