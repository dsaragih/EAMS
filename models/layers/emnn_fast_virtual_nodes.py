import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from models.networks.encoders.norms import build_norm_layer


class FastVirtualNodeLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_virtual_nodes: int,
        edge_attr_dim: int = 0,
        dropout: float = 0.0,
        residual_weight: float = 1.0,
        learnable_residual_weight: bool = False,
        eps: float = 1e-8,
        norm_type: str = "stateless_batchnorm",
    ):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got: {dim}")
        if num_virtual_nodes < 1:
            raise ValueError(
                f"num_virtual_nodes must be >= 1, got: {num_virtual_nodes}"
            )

        self.dim = dim
        self.num_virtual_nodes = num_virtual_nodes
        self.edge_attr_dim = edge_attr_dim
        self.eps = float(eps)

        self.virtual_feature_seed = nn.Parameter(
            torch.randn(1, num_virtual_nodes, dim) * 0.02
        )

        self.node_norm = build_norm_layer(dim, norm_type)
        self.virtual_norm = build_norm_layer(dim, norm_type)
        self.output_norm = build_norm_layer(dim, norm_type)

        real_edge_input_dim = dim * 2 + 1 + edge_attr_dim
        virtual_edge_input_dim = dim * 2 + 2
        self.real_edge_mlp = nn.Sequential(
            nn.Linear(real_edge_input_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.virtual_edge_mlp = nn.Sequential(
            nn.Linear(virtual_edge_input_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.node_update_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.virtual_update_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.node_coord_gate = self._build_coord_gate(dim)
        self.virtual_coord_gate = self._build_coord_gate(dim)

        if learnable_residual_weight:
            self.residual_weight = nn.Parameter(torch.tensor(float(residual_weight)))
        else:
            self.residual_weight = float(residual_weight)

    def _build_coord_gate(self, dim: int) -> nn.Sequential:
        gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1, bias=False),
        )
        nn.init.xavier_uniform_(gate[-1].weight, gain=0.001)
        return gate

    def initialize_virtual_state(
        self, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs == 0:
            empty_features = self.virtual_feature_seed.new_zeros(
                (0, self.num_virtual_nodes, self.dim)
            )
            empty_coords = pos.new_zeros((0, self.num_virtual_nodes, 3))
            return empty_features, empty_coords

        graph_centers = scatter(pos, batch, dim=0, dim_size=num_graphs, reduce="mean")
        virtual_node_features = self.virtual_feature_seed.expand(num_graphs, -1, -1)
        virtual_node_coords = graph_centers.unsqueeze(1).expand(
            -1, self.num_virtual_nodes, -1
        )
        return virtual_node_features, virtual_node_coords

    def _normalize_virtual_features(
        self, virtual_node_features: torch.Tensor
    ) -> torch.Tensor:
        if virtual_node_features.numel() == 0:
            return virtual_node_features
        batch_size, num_virtual_nodes, dim = virtual_node_features.shape
        flat = virtual_node_features.reshape(batch_size * num_virtual_nodes, dim)
        flat = self.virtual_norm(flat)
        return flat.reshape(batch_size, num_virtual_nodes, dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        virtual_node_features: torch.Tensor | None = None,
        virtual_node_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (N, C), got: {tuple(x.shape)}")
        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got: {tuple(pos.shape)}")

        num_nodes = x.shape[0]
        if num_nodes == 0:
            if virtual_node_features is None or virtual_node_coords is None:
                virtual_node_features, virtual_node_coords = (
                    self.initialize_virtual_state(pos, batch)
                )
            return x, pos, virtual_node_features, virtual_node_coords

        if edge_attr is not None and edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        if virtual_node_features is None or virtual_node_coords is None:
            virtual_node_features, virtual_node_coords = self.initialize_virtual_state(
                pos, batch
            )

        batch_size = int(batch.max().item()) + 1
        if virtual_node_features.shape != (
            batch_size,
            self.num_virtual_nodes,
            self.dim,
        ):
            raise ValueError(
                "virtual_node_features must have shape "
                f"({batch_size}, {self.num_virtual_nodes}, {self.dim}), got: {tuple(virtual_node_features.shape)}"
            )
        if virtual_node_coords.shape != (batch_size, self.num_virtual_nodes, 3):
            raise ValueError(
                "virtual_node_coords must have shape "
                f"({batch_size}, {self.num_virtual_nodes}, 3), got: {tuple(virtual_node_coords.shape)}"
            )

        x_norm = self.node_norm(x)
        virtual_norm = self._normalize_virtual_features(virtual_node_features)

        row, col = edge_index.long()
        real_radial = (pos[row] - pos[col]).square().sum(dim=-1, keepdim=True)
        real_edge_inputs = [x_norm[row], x_norm[col], real_radial]
        if edge_attr is not None:
            real_edge_inputs.append(edge_attr.to(dtype=x.dtype))
        real_edge_messages = self.real_edge_mlp(torch.cat(real_edge_inputs, dim=-1))
        real_agg = scatter(
            real_edge_messages,
            row,
            dim=0,
            dim_size=num_nodes,
            reduce="mean",
        )

        graph_centers = scatter(pos, batch, dim=0, dim_size=batch_size, reduce="mean")
        virtual_centered = virtual_node_coords - graph_centers.unsqueeze(1)
        virtual_spread = torch.linalg.norm(virtual_centered, dim=-1, keepdim=True)

        node_to_virtual = virtual_node_coords[batch] - pos.unsqueeze(1)
        virtual_radial = node_to_virtual.square().sum(dim=-1, keepdim=True)
        node_virtual_inputs = torch.cat(
            [
                x_norm.unsqueeze(1).expand(-1, self.num_virtual_nodes, -1),
                virtual_norm[batch],
                virtual_radial,
                virtual_spread[batch],
            ],
            dim=-1,
        )
        virtual_messages = self.virtual_edge_mlp(node_virtual_inputs)  # m_i^v

        node_virtual_agg = virtual_messages.mean(dim=1)
        node_update = self.node_update_mlp(
            torch.cat([x_norm, real_agg, node_virtual_agg], dim=-1)
        )
        x_out = self.output_norm(x + self.residual_weight * node_update)

        node_coord_delta = (-node_to_virtual) * self.node_coord_gate(virtual_messages)
        pos_out = pos + self.residual_weight * node_coord_delta.mean(dim=1)

        graph_virtual_messages = scatter(
            virtual_messages.reshape(num_nodes, self.num_virtual_nodes * self.dim),
            batch,
            dim=0,
            dim_size=batch_size,
            reduce="mean",
        ).reshape(batch_size, self.num_virtual_nodes, self.dim)
        virtual_update = self.virtual_update_mlp(
            torch.cat([virtual_norm, graph_virtual_messages], dim=-1)
        )
        virtual_features_out = (
            virtual_node_features + self.residual_weight * virtual_update
        )

        virtual_coord_delta = node_to_virtual * self.virtual_coord_gate(
            virtual_messages
        )
        virtual_coord_delta = scatter(
            virtual_coord_delta.reshape(num_nodes, self.num_virtual_nodes * 3),
            batch,
            dim=0,
            dim_size=batch_size,
            reduce="mean",
        ).reshape(batch_size, self.num_virtual_nodes, 3)
        virtual_coords_out = (
            virtual_node_coords + self.residual_weight * virtual_coord_delta
        )

        return x_out, pos_out, virtual_features_out, virtual_coords_out
