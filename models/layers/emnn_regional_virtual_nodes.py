import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps
from torch_geometric.utils import scatter

from models.networks.encoders.norms import build_norm_layer


class RegionalVirtualNodeLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_virtual_nodes: int,
        coord_dim: int,
        assignment_hidden_dim: int | None = None,
        num_attention_heads: int = 4,
        num_attention_layers: int = 1,
        attention_dropout: float = 0.0,
        transformer_hidden_mult: float = 2.0,
        use_coordinate_features: bool = True,
        geometry_feature_mode: str | None = None,
        geometry_anchor_count: int | None = None,
        geometry_rbf_bins: int = 8,
        geometry_rbf_sigma: float = 0.2,
        geometry_feature_normalization: str = "max",
        use_region_geometry_features: bool = True,
        residual_weight: float = 1.0,
        learnable_residual_weight: bool = False,
        eps: float = 1e-6,
        geometry_prior_mode: str = "none",
        geometry_prior_weight: float = 1.0,
        geometry_prior_sigma: float = 0.2,
        geometry_radius: float | None = None,
        geometry_random_start: bool = False,
        geometry_detach_coords: bool = True,
        norm_type: str = "stateless_batchnorm",
    ):
        super().__init__()
        if num_virtual_nodes < 1:
            raise ValueError(
                f"num_virtual_nodes must be >= 1, got: {num_virtual_nodes}"
            )
        if dim % num_attention_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        self.num_virtual_nodes = num_virtual_nodes
        self.coord_dim = coord_dim
        self.use_coordinate_features = use_coordinate_features
        self.eps = eps
        if geometry_feature_mode is None:
            geometry_feature_mode = (
                "raw_coordinates" if use_coordinate_features else "none"
            )
        valid_geometry_feature_modes = {
            "none",
            "raw_coordinates",
            "fps_anchor_distances",
            "fps_anchor_rbf",
            "centroid_distance",
        }
        if geometry_feature_mode not in valid_geometry_feature_modes:
            raise ValueError(
                "geometry_feature_mode must be one of "
                f"{sorted(valid_geometry_feature_modes)}, got: {geometry_feature_mode}"
            )
        if geometry_feature_normalization not in {"none", "max"}:
            raise ValueError(
                "geometry_feature_normalization must be one of {'none', 'max'}, "
                f"got: {geometry_feature_normalization}"
            )
        if geometry_rbf_bins < 1:
            raise ValueError(
                f"geometry_rbf_bins must be >= 1, got: {geometry_rbf_bins}"
            )
        if geometry_rbf_sigma <= 0.0:
            raise ValueError(
                f"geometry_rbf_sigma must be > 0, got: {geometry_rbf_sigma}"
            )
        valid_geometry_prior_modes = {"none", "fps_bias", "radius_bias", "fps_radius"}
        if geometry_prior_mode not in valid_geometry_prior_modes:
            raise ValueError(
                f"geometry_prior_mode must be one of {sorted(valid_geometry_prior_modes)}, got: {geometry_prior_mode}"
            )
        if geometry_prior_sigma <= 0.0:
            raise ValueError(
                f"geometry_prior_sigma must be > 0, got: {geometry_prior_sigma}"
            )

        self.geometry_prior_mode = geometry_prior_mode
        self.geometry_prior_weight = float(geometry_prior_weight)
        self.geometry_prior_sigma = float(geometry_prior_sigma)
        self.geometry_radius = geometry_radius
        self.geometry_random_start = geometry_random_start
        self.geometry_detach_coords = geometry_detach_coords
        self.geometry_feature_mode = geometry_feature_mode
        self.geometry_anchor_count = geometry_anchor_count or num_virtual_nodes
        self.geometry_rbf_bins = geometry_rbf_bins
        self.geometry_rbf_sigma = float(geometry_rbf_sigma)
        self.geometry_feature_normalization = geometry_feature_normalization
        self.use_region_geometry_features = use_region_geometry_features
        self.input_norm = build_norm_layer(dim, norm_type)
        self.residual_norm = build_norm_layer(dim, norm_type)

        assignment_hidden_dim = assignment_hidden_dim or dim
        geometry_feature_dim = self._geometry_feature_dim()
        assignment_input_dim = dim + geometry_feature_dim
        self.assignment_mlp = nn.Sequential(
            nn.Linear(assignment_input_dim, assignment_hidden_dim),
            nn.SiLU(),
            nn.Linear(assignment_hidden_dim, num_virtual_nodes),
        )

        self.geometry_feature_proj = (
            nn.Linear(geometry_feature_dim, dim)
            if use_region_geometry_features and geometry_feature_dim > 0
            else None
        )
        feedforward_dim = max(dim, int(dim * transformer_hidden_mult))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.region_attention = nn.TransformerEncoder(
            encoder_layer, num_layers=num_attention_layers
        )
        self.region_output_norm = build_norm_layer(dim, norm_type)
        self.output_proj = nn.Sequential(
            build_norm_layer(dim, norm_type),
            nn.Linear(dim, dim),
        )
        nn.init.xavier_uniform_(self.output_proj[1].weight, gain=0.1)
        if self.output_proj[1].bias is not None:
            nn.init.zeros_(self.output_proj[1].bias)
        if learnable_residual_weight:
            self.residual_weight = nn.Parameter(torch.tensor(float(residual_weight)))
        else:
            self.residual_weight = float(residual_weight)

    def _geometry_feature_dim(self) -> int:
        if self.geometry_feature_mode == "none":
            return 0
        if self.geometry_feature_mode == "raw_coordinates":
            return self.coord_dim
        if self.geometry_feature_mode == "fps_anchor_distances":
            return self.geometry_anchor_count
        if self.geometry_feature_mode == "fps_anchor_rbf":
            return self.geometry_anchor_count * self.geometry_rbf_bins
        if self.geometry_feature_mode == "centroid_distance":
            return 1
        raise ValueError(
            f"Unsupported geometry_feature_mode: {self.geometry_feature_mode}"
        )

    def _normalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        if self.geometry_feature_normalization == "none":
            return distances

        scale = distances.amax(dim=-1, keepdim=True).clamp_min(self.eps)
        return distances / scale

    def _rbf_expand(self, distances: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(
            0.0,
            1.0,
            steps=self.geometry_rbf_bins,
            device=distances.device,
            dtype=distances.dtype,
        )
        expanded = distances.unsqueeze(-1) - centers.view(1, 1, -1)
        return torch.exp(-(expanded.square()) / (2.0 * (self.geometry_rbf_sigma**2)))

    def _compute_node_geometry_features(
        self,
        coord: torch.Tensor,
        geometry_coord: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor | None:
        if self.geometry_feature_mode == "none":
            return None
        if self.geometry_feature_mode == "raw_coordinates":
            return coord.reshape(coord.shape[0], -1).clone()

        geometry_pos = self._geometry_positions(geometry_coord)
        if self.geometry_detach_coords:
            geometry_pos = geometry_pos.detach()

        if self.geometry_feature_mode == "centroid_distance":
            out = geometry_pos.new_zeros((geometry_pos.shape[0], 1))
            for graph_idx in range(num_graphs):
                node_mask = batch == graph_idx
                graph_pos = geometry_pos[node_mask]
                if graph_pos.shape[0] == 0:
                    continue
                centroid = graph_pos.mean(dim=0, keepdim=True)
                dists = torch.linalg.norm(graph_pos - centroid, dim=-1, keepdim=True)
                out[node_mask] = self._normalize_distances(dists)
            return out

        seed_coords = self._build_seed_coordinates(
            geometry_pos,
            batch,
            num_graphs,
            num_seeds=self.geometry_anchor_count,
        )
        out_dim = self._geometry_feature_dim()
        out = geometry_pos.new_zeros((geometry_pos.shape[0], out_dim))
        for graph_idx in range(num_graphs):
            node_mask = batch == graph_idx
            graph_pos = geometry_pos[node_mask]
            if graph_pos.shape[0] == 0:
                continue
            dists = torch.cdist(graph_pos, seed_coords[graph_idx], p=2)
            dists = self._normalize_distances(dists)
            if self.geometry_feature_mode == "fps_anchor_distances":
                out[node_mask] = dists
                continue
            out[node_mask] = self._rbf_expand(dists).reshape(graph_pos.shape[0], -1)
        return out

    def _geometry_positions(self, coord: torch.Tensor) -> torch.Tensor:
        if coord.ndim == 2:
            return coord
        if coord.ndim == 3 and coord.shape[1] == 3:
            return coord.mean(dim=-1)
        raise ValueError(
            f"Unsupported coordinate shape for geometry prior: {tuple(coord.shape)}"
        )

    def _build_seed_coordinates(
        self,
        geometry_pos: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        num_seeds: int,
    ) -> torch.Tensor:
        seed_coords = []
        for graph_idx in range(num_graphs):
            node_mask = batch == graph_idx
            graph_pos = geometry_pos[node_mask]
            if graph_pos.shape[0] == 0:
                seed_coords.append(
                    geometry_pos.new_zeros((num_seeds, geometry_pos.shape[-1]))
                )
                continue

            ratio = min(1.0, num_seeds / max(graph_pos.shape[0], 1))
            local_seed_idx = fps(
                graph_pos,
                ratio=ratio,
                random_start=self.geometry_random_start,
            )
            if local_seed_idx.numel() == 0:
                local_seed_idx = torch.zeros(
                    1, dtype=torch.long, device=geometry_pos.device
                )
            if local_seed_idx.numel() < num_seeds:
                repeats = (
                    num_seeds + local_seed_idx.numel() - 1
                ) // local_seed_idx.numel()
                local_seed_idx = local_seed_idx.repeat(repeats)
            local_seed_idx = local_seed_idx[:num_seeds]
            seed_coords.append(graph_pos[local_seed_idx])

        return torch.stack(seed_coords, dim=0)

    def _compute_geometry_prior_logits(
        self,
        geometry_coord: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.geometry_prior_mode == "none" or self.geometry_prior_weight == 0.0:
            return geometry_coord.new_zeros(
                (geometry_coord.shape[0], self.num_virtual_nodes)
            )

        geometry_pos = self._geometry_positions(geometry_coord)
        if self.geometry_detach_coords:
            geometry_pos = geometry_pos.detach()

        seed_coords = self._build_seed_coordinates(
            geometry_pos,
            batch,
            num_graphs,
            num_seeds=self.num_virtual_nodes,
        )
        prior_logits = geometry_pos.new_zeros(
            (geometry_pos.shape[0], self.num_virtual_nodes)
        )
        sigma_sq = max(self.geometry_prior_sigma, self.eps) ** 2

        for graph_idx in range(num_graphs):
            node_mask = batch == graph_idx
            graph_pos = geometry_pos[node_mask]
            if graph_pos.shape[0] == 0:
                continue

            graph_seed_coords = seed_coords[graph_idx]
            dists = torch.cdist(graph_pos, graph_seed_coords, p=2)

            if self.geometry_prior_mode == "fps_bias":
                graph_logits = -(dists.square()) / (2.0 * sigma_sq)
            elif self.geometry_prior_mode == "radius_bias":
                graph_logits = torch.zeros_like(dists)
            else:
                graph_logits = -(dists.square()) / (2.0 * sigma_sq)

            if self.geometry_prior_mode in {"radius_bias", "fps_radius"}:
                if self.geometry_radius is None or self.geometry_radius <= 0.0:
                    raise ValueError(
                        "geometry_radius must be set to a positive value when using radius-based geometry priors"
                    )
                valid_mask = dists <= self.geometry_radius
                rows_with_valid_seed = valid_mask.any(dim=-1, keepdim=True)
                outside_penalty = torch.full_like(graph_logits, -1e4)
                graph_logits = torch.where(valid_mask, graph_logits, outside_penalty)
                graph_logits = torch.where(
                    rows_with_valid_seed, graph_logits, torch.zeros_like(graph_logits)
                )

            prior_logits[node_mask] = graph_logits

        return self.geometry_prior_weight * prior_logits

    def _scatter_sum(
        self, src: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        return scatter(src, batch, dim=0, dim_size=num_graphs, reduce="sum")

    def forward(
        self,
        x: torch.Tensor,
        coord: torch.Tensor,
        batch: torch.Tensor,
        geometry_coord: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if coord.shape[0] != x.shape[0] or batch.shape[0] != x.shape[0]:
            raise ValueError(
                "RegionalVirtualNodeLayer expects x, coord, and batch to share the same node dimension"
            )
        if geometry_coord is not None and geometry_coord.shape[0] != x.shape[0]:
            raise ValueError(
                "RegionalVirtualNodeLayer expects geometry_coord and x to share the same node dimension"
            )

        num_nodes = x.shape[0]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        coord_shape = coord.shape[1:]
        # Clone the flattened coordinates so autograd does not retain a view into
        # a tensor that later EMNN layers may update in-place.
        coord_flat = coord.reshape(num_nodes, -1).clone()
        geometry_feature_source = (
            geometry_coord if geometry_coord is not None else coord
        )

        if num_graphs == 0:
            empty_regions = x.new_zeros((0, self.num_virtual_nodes, x.shape[-1]))
            empty_coords = coord.new_zeros((0, self.num_virtual_nodes, *coord_shape))
            empty_mass = x.new_zeros((0, self.num_virtual_nodes))
            empty_assignment = x.new_zeros((num_nodes, self.num_virtual_nodes))
            return {
                "x": x,
                "assignment_matrix": empty_assignment,
                "region_features": empty_regions,
                "region_coords": empty_coords,
                "region_mass": empty_mass,
            }

        assignment_input = self.input_norm(x)
        node_geometry_features = self._compute_node_geometry_features(
            coord=coord,
            geometry_coord=geometry_feature_source,
            batch=batch,
            num_graphs=num_graphs,
        )
        if node_geometry_features is not None:
            assignment_input = torch.cat(
                [assignment_input, node_geometry_features], dim=-1
            )

        assignment_logits = self.assignment_mlp(assignment_input)
        geometry_logits = self._compute_geometry_prior_logits(
            geometry_feature_source,
            batch,
            num_graphs,
        )
        assignment_logits = assignment_logits + geometry_logits
        assignment_logits = assignment_logits - assignment_logits.amax(
            dim=-1, keepdim=True
        )
        assignment_logits = assignment_logits.clamp(min=-20.0, max=20.0)
        assignment = torch.softmax(assignment_logits, dim=-1)

        region_mass = self._scatter_sum(assignment, batch, num_graphs)
        region_mass_denom = region_mass.clamp_min(self.eps).unsqueeze(-1)

        weighted_features = assignment.unsqueeze(-1) * x.unsqueeze(1)
        feature_sums = self._scatter_sum(
            weighted_features.reshape(num_nodes, -1), batch, num_graphs
        ).reshape(num_graphs, self.num_virtual_nodes, x.shape[-1])
        region_features = feature_sums / region_mass_denom

        weighted_coords = assignment.unsqueeze(-1) * coord_flat.unsqueeze(1)
        coord_sums = self._scatter_sum(
            weighted_coords.reshape(num_nodes, -1), batch, num_graphs
        ).reshape(num_graphs, self.num_virtual_nodes, coord_flat.shape[-1])
        region_coords_flat = coord_sums / region_mass_denom

        region_tokens = region_features
        if (
            self.geometry_feature_proj is not None
            and node_geometry_features is not None
        ):
            weighted_geometry = assignment.unsqueeze(
                -1
            ) * node_geometry_features.unsqueeze(1)
            geometry_sums = self._scatter_sum(
                weighted_geometry.reshape(num_nodes, -1), batch, num_graphs
            ).reshape(
                num_graphs,
                self.num_virtual_nodes,
                node_geometry_features.shape[-1],
            )
            region_geometry_features = geometry_sums / region_mass_denom
            region_tokens = region_tokens + self.geometry_feature_proj(
                region_geometry_features
            )
        region_tokens = self.region_attention(region_tokens)
        region_tokens = self.region_output_norm(region_tokens)

        node_region_tokens = region_tokens[batch]
        node_update = (assignment.unsqueeze(-1) * node_region_tokens).sum(dim=1)
        node_update = self.output_proj(node_update)
        node_update = 5.0 * torch.tanh(node_update / 5.0)

        residual_weight = self.residual_weight
        x = self.residual_norm(x + residual_weight * node_update)

        return {
            "x": x,
            "assignment_matrix": assignment,
            "region_features": region_tokens,
            "region_coords": region_coords_flat.reshape(
                num_graphs, self.num_virtual_nodes, *coord_shape
            ),
            "region_mass": region_mass,
        }
