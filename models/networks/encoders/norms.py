import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm


DEFAULT_NORM_TYPE = "stateless_batchnorm"


def canonicalize_norm_type(norm_type: str | None) -> str:
    if norm_type is None:
        return DEFAULT_NORM_TYPE

    normalized = norm_type.strip().lower().replace("-", "_")
    aliases = {
        "batchnorm": DEFAULT_NORM_TYPE,
        "bn": DEFAULT_NORM_TYPE,
        "stateless_batch_norm": DEFAULT_NORM_TYPE,
        "stateless_batchnorm": DEFAULT_NORM_TYPE,
        "graphnorm": "graphnorm",
        "graph_norm": "graphnorm",
        "gn": "graphnorm",
        "layernorm": "layernorm",
        "layer_norm": "layernorm",
        "ln": "layernorm",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported normalization type "
            f"'{norm_type}'. Expected one of: stateless_batchnorm, graphnorm, layernorm"
        )
    return aliases[normalized]


class StatelessBatchNorm(nn.Module):
    """BatchNorm over the last feature dimension without running statistics."""

    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features, track_running_stats=False)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        del batch
        if x.ndim == 2:
            return self.norm(x)
        if x.ndim == 3:
            original_shape = x.shape
            x = x.reshape(-1, original_shape[-1])
            x = self.norm(x)
            return x.reshape(original_shape)
        raise ValueError(
            f"StatelessBatchNorm expects rank-2 or rank-3 input, got shape {tuple(x.shape)}"
        )


class LastDimLayerNorm(nn.Module):
    """LayerNorm over the last feature dimension for rank-2 or rank-3 tensors."""

    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        del batch
        if x.ndim not in {2, 3}:
            raise ValueError(
                f"LastDimLayerNorm expects rank-2 or rank-3 input, got shape {tuple(x.shape)}"
            )
        return self.norm(x)


class BatchGraphNorm(nn.Module):
    """GraphNorm over rank-2 node/edge/face feature tensors."""

    def __init__(self, num_features: int):
        super().__init__()
        self.norm = GraphNorm(num_features)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"BatchGraphNorm expects rank-2 input, got shape {tuple(x.shape)}"
            )
        if batch is None:
            raise ValueError("BatchGraphNorm requires `batch`")
        return self.norm(x, batch)


def build_norm_layer(num_features: int, norm_type: str | None = None) -> nn.Module:
    normalized = canonicalize_norm_type(norm_type)
    if normalized == DEFAULT_NORM_TYPE:
        return StatelessBatchNorm(num_features)
    if normalized == "graphnorm":
        return BatchGraphNorm(num_features)
    if normalized == "layernorm":
        return LastDimLayerNorm(num_features)
    raise ValueError(f"Unhandled normalization type: {normalized}")
