import torch
import torch.nn as nn
from torch_geometric.nn import fps, radius

from models.utils import aggregate_max, aggregate_mean, aggregate_sum


class FeaturePooling(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ratio: float = 0.25,
        r: float = 0.2,
        max_num_neighbors: int = 64,
        aggr: str = "max",
        random_start: bool = True,
        use_feature_scores: bool = True,
    ):
        super().__init__()
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got: {ratio}")
        if aggr not in {"max", "mean", "sum"}:
            raise ValueError(
                f"aggr must be one of {{'max', 'mean', 'sum'}}, got: {aggr}"
            )

        self.ratio = ratio
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.aggr = aggr
        self.random_start = random_start
        self.use_feature_scores = use_feature_scores

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()
        self.score_proj = nn.Linear(out_dim, 1) if use_feature_scores else None

    def _aggregate(
        self, src: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        if self.aggr == "max":
            return aggregate_max(src, index, dim_size)
        if self.aggr == "mean":
            return aggregate_mean(src, index, dim_size)
        return aggregate_sum(src, index, dim_size)

    def forward(
        self,
        pos_origin: torch.Tensor,
        pos: torch.Tensor,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sample_idx = fps(
            pos_origin,
            batch,
            ratio=self.ratio,
            random_start=self.random_start,
        )
        if sample_idx.numel() == 0:
            raise ValueError("FeaturePooling produced no sampled nodes")

        pooled_batch = batch[sample_idx]
        row, col = radius(
            pos_origin,
            pos_origin[sample_idx],
            self.r,
            batch,
            pooled_batch,
            max_num_neighbors=self.max_num_neighbors,
        )

        projected = self.act(self.norm(self.linear(x)))
        src = projected[col]
        if self.score_proj is not None:
            src = src * torch.sigmoid(self.score_proj(src))

        pooled_x = self._aggregate(src, row, sample_idx.shape[0])
        return {
            "sample_idx": sample_idx,
            "pos_origin": pos_origin[sample_idx],
            "pos": pos[sample_idx],
            "x": pooled_x,
            "batch": pooled_batch,
        }
