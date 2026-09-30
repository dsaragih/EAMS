import torch
import torch.nn as nn
from torch_geometric.nn import knn_interpolate


class FeatureUnpooling(nn.Module):
    def __init__(self, coarse_dim: int, skip_dim: int, out_dim: int, k: int = 3):
        super().__init__()
        self.k = k

        self.coarse_proj = nn.Linear(coarse_dim, out_dim, bias=False)
        self.skip_proj = nn.Linear(skip_dim, out_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(coarse_dim + skip_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        pos_coarse_origin: torch.Tensor,
        x_coarse: torch.Tensor,
        batch_coarse: torch.Tensor,
        pos_fine_origin: torch.Tensor,
        x_skip: torch.Tensor,
        batch_fine: torch.Tensor,
    ) -> torch.Tensor:
        lifted = knn_interpolate(
            x_coarse,
            pos_coarse_origin,
            pos_fine_origin,
            batch_coarse,
            batch_fine,
            k=self.k,
        )
        gate = self.gate(torch.cat([lifted, x_skip], dim=-1))
        fused = gate * self.skip_proj(x_skip) + (1.0 - gate) * self.coarse_proj(lifted)
        return self.act(self.norm(fused))
