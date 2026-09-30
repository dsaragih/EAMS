"""Graph-level residual matrix prediction for direct canonicalization."""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ..contracts import EncoderOutput
from ..mlp_decoder import MLP_Decoder


class GraphRotationDecoder(nn.Module):
    """Pool node embeddings and predict a residual 3-by-3 matrix per graph."""

    def __init__(
        self,
        node_dim: int | None = None,
        input_dim: int | None = None,
        out_dim: int = 9,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        norm_type: str | None = None,
        use_layernorm: bool = False,
        activation: str = "silu",
        use_bias: bool = True,
        output_level: str = "graph",
        pool_reduce: str = "max",
        **kwargs,
    ):
        super().__init__()
        del kwargs

        feature_dim = node_dim if node_dim is not None else input_dim
        if feature_dim is None:
            raise ValueError("GraphRotationDecoder requires node_dim or input_dim")
        if output_level != "graph":
            raise ValueError(
                "GraphRotationDecoder only supports output_level='graph', got: "
                f"{output_level}"
            )
        if out_dim != 9:
            raise ValueError(
                "GraphRotationDecoder requires out_dim=9 for a 3-by-3 matrix, "
                f"got {out_dim}."
            )
        if pool_reduce not in {"mean", "sum", "max"}:
            raise ValueError(
                "GraphRotationDecoder pool_reduce must be one of {'mean', 'sum', 'max'}, "
                f"got {pool_reduce!r}."
            )

        self.output_level = output_level
        self.pool_reduce = pool_reduce
        self.head = MLP_Decoder(
            input_dim=feature_dim,
            target_dim=out_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            norm_type=norm_type,
            use_layernorm=use_layernorm,
            activation=activation,
            use_bias=use_bias,
        )
        final_linear = self.head.linears[-1]
        nn.init.zeros_(final_linear.weight)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

    def forward(self, encoded: EncoderOutput) -> dict[str, torch.Tensor | str]:
        if encoded.node_embedding is None:
            raise ValueError("GraphRotationDecoder requires node_embedding")
        if encoded.batch is None:
            raise ValueError("GraphRotationDecoder requires batch indices")

        batch = encoded.batch.long()
        num_graphs = int(batch.max().item() + 1) if batch.numel() else 0
        pooled = scatter(
            encoded.node_embedding,
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.pool_reduce,
        )
        graph_batch = torch.arange(num_graphs, device=pooled.device)
        return {"pred": self.head(pooled, graph_batch), "level": self.output_level}
