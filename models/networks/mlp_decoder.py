import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import LayerNorm

from models.utils import *


class MLP_Decoder(nn.Module):
    """Small MLP decoder with configurable hidden-layer normalization.

    This intentionally avoids LazyLinear: all dimensions are inferred from
    constructor parameters.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        norm_type: str | None = None,
        use_layernorm: bool = True,
        activation: str = "relu",
        use_bias: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim]

        if norm_type is None:
            norm_type = "layernorm" if use_layernorm else "none"
        norm_type = norm_type.lower()
        if norm_type not in {"none", "layernorm", "batchnorm1d"}:
            raise ValueError(
                "norm_type must be one of {'none', 'layernorm', 'batchnorm1d'}, "
                f"got: {norm_type}"
            )

        self.use_layernorm = norm_type == "layernorm"
        self.use_batchnorm1d = norm_type == "batchnorm1d"
        self.norm_type = norm_type
        self.activation = activation

        dims = [input_dim] + list(hidden_dims) + [target_dim]
        self.linears = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=use_bias)
                for i in range(len(dims) - 1)
            ]
        )

        self.norms = nn.ModuleList()
        for i in range(len(dims) - 2):
            hidden_dim = dims[i + 1]
            if self.use_layernorm:
                self.norms.append(LayerNorm(hidden_dim))
            elif self.use_batchnorm1d:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(dims) - 2)]
        )

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.2)
        raise ValueError(f"Unsupported activation={self.activation}")

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Hidden layers
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            if self.use_layernorm:
                if batch is None:
                    raise ValueError(
                        "MLP_Decoder(norm_type='layernorm') requires `batch` in forward()."
                    )
                x = self.norms[i](x, batch)
            elif self.use_batchnorm1d:
                x = self.norms[i](x)
            x = self._act(x)
            x = self.dropouts[i](x)

        # Final projection (no norm/activation)
        return self.linears[-1](x)


def _pos_to_xyz(pos: torch.Tensor) -> torch.Tensor:
    """Return an (N, 3) position tensor.

    EMNN may run in mc-mode where `pos` becomes (N, 3, num_vector).
    For scalar decoding, we use the mean across vectors.
    """

    if pos.ndim == 2:
        return pos
    if pos.ndim == 3 and pos.shape[1] == 3:
        return pos.mean(dim=-1)
    raise ValueError(f"Unsupported pos shape for decoding: {tuple(pos.shape)}")


class EdgePositionMessageDecoder(nn.Module):
    """Decode using edgewise messages that mix hidden + coordinate information.

    Inspired by PositionDecoder: compute edgewise features (including distance
    embedding), decode on edges, and optionally scatter to vertices.
    """

    def __init__(
        self,
        node_dim: int,
        edge_attr_dim: int,
        out_dim: int,
        *,
        msg_hidden_dims: list[int] | None = None,
        dist_hidden_dims: list[int] | None = None,
        dist_emb_dim: int = 16,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str | None = None,
        use_layernorm: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps

        # Distance embedding MLP: (E,1) -> (E,dist_emb_dim)
        if dist_hidden_dims is None:
            dist_hidden_dims = [max(dist_emb_dim, 8)]
        self.dist_mlp = MLP_Decoder(
            input_dim=1,
            target_dim=dist_emb_dim,
            hidden_dims=dist_hidden_dims,
            dropout=dropout,
            norm_type=norm_type,
            use_layernorm=use_layernorm,
            activation=activation,
        )

        msg_in_dim = 2 * node_dim + edge_attr_dim + dist_emb_dim
        if msg_hidden_dims is None:
            msg_hidden_dims = [node_dim, node_dim]

        self.msg_mlp = MLP_Decoder(
            input_dim=msg_in_dim,
            target_dim=out_dim,
            hidden_dims=msg_hidden_dims,
            dropout=dropout,
            norm_type=norm_type,
            use_layernorm=use_layernorm,
            activation=activation,
        )

    def forward_edge(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (edge_pred, edge_embedding, edge_batch)."""

        pos_xyz = _pos_to_xyz(pos)
        src, dst = edge_index[0].long(), edge_index[1].long()

        edge_batch = batch[src]
        dist = torch.norm(pos_xyz[src] - pos_xyz[dst], dim=-1, keepdim=True)
        dist_emb = self.dist_mlp(dist, edge_batch)

        edge_embedding = torch.cat([x[src], x[dst], edge_attr, dist_emb], dim=-1)
        edge_pred = self.msg_mlp(edge_embedding, edge_batch)
        return edge_pred, edge_embedding, edge_batch

    def forward_vertex(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        reduce: str = "sum",
    ) -> torch.Tensor:
        """Decode on edges then scatter predictions to vertices."""

        edge_pred, _, _ = self.forward_edge(x, pos, edge_index, edge_attr, batch)
        num_nodes = x.shape[0]
        return scatter(
            edge_pred, edge_index[1].long(), dim=0, dim_size=num_nodes, reduce=reduce
        )
