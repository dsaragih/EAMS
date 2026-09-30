import torch.nn as nn
from torch_geometric.nn import global_mean_pool, LayerNorm, fps, radius
from models.utils import *

class Pooling(torch.nn.Module):
    def __init__(self, in_dim, out_dim, ratio=0.25, r=0.2, global_pool = False):
        super().__init__()
        self.ratio = ratio
        self.r = r

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.ln = LayerNorm(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = global_pool
        

    def forward(self, pos_origin, pos, x, batch):
        if self.global_pool:
            return global_mean_pool(x, batch)

        else:
            index = fps(pos_origin, batch, ratio=self.ratio, random_start=True)
            row, col = radius(pos_origin, pos_origin[index], self.r, batch, batch[index], max_num_neighbors=64)
            pos_origin_out = pos_origin[index]
            pos_out = pos[index]
            x_out = aggregate_max(self.relu(self.ln(self.linear(x)))[col], row, pos_out.shape[0])
            batch_out = batch[index]
            return pos_origin_out, pos_out, x_out, batch_out

