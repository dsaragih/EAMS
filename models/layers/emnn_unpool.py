import torch.nn as nn
from torch_geometric.nn import LayerNorm, knn_interpolate
from models.utils import *

class Unpooling(torch.nn.Module):
    def __init__(self, in_dim, out_dim, k=3):
        super().__init__()
        self.k = k

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.ln = LayerNorm(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pos_x_origin, x, batch_x, pos_y_original, y, batch_y):
        x = knn_interpolate(x, pos_x_origin, pos_y_original, batch_x, batch_y, k=self.k)
        x = torch.cat([x, y], dim=-1)
        x = self.relu(self.ln(self.linear(x)))
        return x