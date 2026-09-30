import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ..contracts import EncoderOutput
from ..mlp_decoder import MLP_Decoder


class GraphEnergyDecoder(nn.Module):
    """Pool node embeddings to graph-level scalar energies."""

    def __init__(
        self,
        node_dim: int | None = None,
        input_dim: int | None = None,
        out_dim: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        norm_type: str | None = None,
        use_layernorm: bool = False,
        activation: str = "silu",
        use_bias: bool = True,
        output_level: str = "graph",
        pool_reduce: str = "mean",
        **kwargs,
    ):
        super().__init__()
        del kwargs

        feature_dim = node_dim if node_dim is not None else input_dim
        if feature_dim is None:
            raise ValueError("GraphEnergyDecoder requires node_dim or input_dim")
        if output_level != "graph":
            raise ValueError(
                "GraphEnergyDecoder only supports output_level='graph', got: "
                f"{output_level}"
            )
        if out_dim != 1:
            raise ValueError(
                f"GraphEnergyDecoder expects out_dim=1 for scalar energies, got {out_dim}."
            )
        if pool_reduce not in {"mean", "sum", "max"}:
            raise ValueError(
                "GraphEnergyDecoder pool_reduce must be one of {'mean', 'sum', 'max'}, "
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
        nn.init.normal_(final_linear.weight, mean=0.0, std=1.0e-3)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

    def forward(self, encoded: EncoderOutput) -> dict[str, torch.Tensor | str]:
        if encoded.node_embedding is None:
            raise ValueError("GraphEnergyDecoder requires node_embedding")
        if encoded.batch is None:
            raise ValueError("GraphEnergyDecoder requires batch indices")

        node_embedding = encoded.node_embedding
        batch = encoded.batch.long()
        num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        pooled = scatter(
            node_embedding,
            batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.pool_reduce,
        )
        graph_batch = torch.arange(num_graphs, device=pooled.device)
        pred = self.head(pooled, graph_batch)
        return {"pred": pred, "level": self.output_level}
