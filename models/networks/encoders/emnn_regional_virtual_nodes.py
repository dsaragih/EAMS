import torch
import torch.nn as nn
from torch_geometric.nn import knn_interpolate, radius

from data.types import MeshData
from loguru import logger

from models.layers.emnn_feature_pool import FeaturePooling
from models.layers.emnn_feature_unpool import FeatureUnpooling
from models.layers.emnn_regional_virtual_nodes import RegionalVirtualNodeLayer
from models.networks.encoders.norms import build_norm_layer
from models.utils import aggregate_max, aggregate_mean, aggregate_sum

from ..contracts import EncoderOutput
from .emnn import EMNN


class PooledContextBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        radius_value: float,
        max_num_neighbors: int = 64,
        aggr: str = "mean",
        dropout: float = 0.0,
        geometry_mode: str = "raw_relative",
        geometry_rbf_bins: int = 8,
        geometry_rbf_sigma: float = 0.2,
        norm_type: str = "stateless_batchnorm",
    ):
        super().__init__()
        if aggr not in {"max", "mean", "sum"}:
            raise ValueError(
                f"aggr must be one of {{'max', 'mean', 'sum'}}, got: {aggr}"
            )
        if geometry_mode not in {
            "none",
            "raw_relative",
            "distance_only",
            "distance_rbf",
        }:
            raise ValueError(
                "geometry_mode must be one of {'none', 'raw_relative', 'distance_only', 'distance_rbf'}, "
                f"got: {geometry_mode}"
            )
        if geometry_rbf_bins < 1:
            raise ValueError(
                f"geometry_rbf_bins must be >= 1, got: {geometry_rbf_bins}"
            )
        if geometry_rbf_sigma <= 0.0:
            raise ValueError(
                f"geometry_rbf_sigma must be > 0, got: {geometry_rbf_sigma}"
            )

        self.radius_value = radius_value
        self.max_num_neighbors = max_num_neighbors
        self.aggr = aggr
        self.geometry_mode = geometry_mode
        self.geometry_rbf_bins = geometry_rbf_bins
        self.geometry_rbf_sigma = float(geometry_rbf_sigma)
        self.norm = build_norm_layer(dim, norm_type)
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + self._edge_geometry_dim(), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def _edge_geometry_dim(self) -> int:
        if self.geometry_mode == "none":
            return 0
        if self.geometry_mode == "raw_relative":
            return 3
        if self.geometry_mode == "distance_only":
            return 1
        if self.geometry_mode == "distance_rbf":
            return self.geometry_rbf_bins
        raise ValueError(f"Unsupported geometry_mode: {self.geometry_mode}")

    def _edge_geometry_features(
        self, pos_origin: torch.Tensor, row: torch.Tensor, col: torch.Tensor
    ) -> torch.Tensor | None:
        rel = pos_origin[col] - pos_origin[row]
        if self.geometry_mode == "none":
            return None
        if self.geometry_mode == "raw_relative":
            return rel

        distance = torch.linalg.norm(rel, dim=-1, keepdim=True)
        if self.geometry_mode == "distance_only":
            return distance

        centers = torch.linspace(
            0.0,
            self.radius_value,
            steps=self.geometry_rbf_bins,
            device=distance.device,
            dtype=distance.dtype,
        )
        expanded = distance - centers.view(1, -1)
        return torch.exp(-(expanded.square()) / (2.0 * (self.geometry_rbf_sigma**2)))

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
        x: torch.Tensor,
        pos_origin: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x

        row, col = radius(
            pos_origin,
            pos_origin,
            self.radius_value,
            batch,
            batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        valid = row != col
        row = row[valid]
        col = col[valid]
        if row.numel() == 0:
            return x

        x_norm = self.norm(x)
        edge_geometry = self._edge_geometry_features(pos_origin, row, col)
        edge_inputs = [x_norm[row], x_norm[col]]
        if edge_geometry is not None:
            edge_inputs.append(edge_geometry)
        msg = self.edge_mlp(torch.cat(edge_inputs, dim=-1))
        agg = self._aggregate(msg, row, x.shape[0])
        update = self.node_mlp(torch.cat([x_norm, agg], dim=-1))
        return x + update


class EMNNRegionalVirtualNodes(EMNN):
    def __init__(
        self,
        input_dim: int,
        edges_input_dim: int,
        width: int,
        width_mult: list[int],
        num_vector: int = 1,
        num_regional_virtual_nodes: int = 16,
        regional_assignment_hidden_dim: int | None = None,
        regional_num_attention_heads: int = 4,
        regional_num_attention_layers: int = 1,
        regional_attention_dropout: float = 0.0,
        regional_transformer_hidden_mult: float = 2.0,
        regional_use_coordinate_features: bool = True,
        regional_geometry_feature_mode: str | None = None,
        regional_geometry_anchor_count: int | None = None,
        regional_geometry_rbf_bins: int = 8,
        regional_geometry_rbf_sigma: float = 0.2,
        regional_geometry_feature_normalization: str = "max",
        regional_use_region_geometry_features: bool = True,
        regional_residual_weight: float = 1.0,
        regional_learnable_residual_weight: bool = False,
        regional_eps: float = 1e-6,
        regional_only_last_layer: bool = False,
        regional_geometry_prior_mode: str = "none",
        regional_geometry_coord_source: str = "origin",
        regional_geometry_prior_weight: float = 1.0,
        regional_geometry_prior_sigma: float = 0.2,
        regional_geometry_radius: float | None = None,
        regional_geometry_random_start: bool = False,
        regional_geometry_detach_coords: bool = True,
        regional_end_stage_mode: str = "flat",
        regional_context_pool_ratio: float = 0.25,
        regional_context_pool_radius: float = 0.2,
        regional_context_pool_aggr: str = "mean",
        regional_context_max_num_neighbors: int = 64,
        regional_context_random_start: bool = False,
        regional_context_use_feature_scores: bool = True,
        regional_context_dim: int | None = None,
        regional_context_num_blocks: int = 1,
        regional_context_fusion: str = "gated",
        regional_context_lift_k: int = 3,
        regional_context_fusion_dropout: float = 0.0,
        regional_context_geometry_mode: str = "raw_relative",
        regional_context_rbf_bins: int = 8,
        regional_context_rbf_sigma: float = 0.2,
        **kwargs,
    ):
        compat_hier = bool(kwargs.get("hier", False))
        if regional_end_stage_mode == "flat" and compat_hier:
            logger.warning(
                "EMNNRegionalVirtualNodes received hier=True with regional_end_stage_mode='flat'; "
                "using legacy_pool_unpool for backward compatibility."
            )
            regional_end_stage_mode = "legacy_pool_unpool"

        if regional_end_stage_mode != "legacy_pool_unpool":
            kwargs["hier"] = False
            kwargs.pop("num_pooling_layers", None)

        super().__init__(
            input_dim=input_dim,
            edges_input_dim=edges_input_dim,
            width=width,
            width_mult=width_mult,
            num_vector=num_vector,
            **kwargs,
        )
        if num_regional_virtual_nodes < 1:
            raise ValueError(
                "num_regional_virtual_nodes must be >= 1 for EMNNRegionalVirtualNodes"
            )

        self.num_regional_virtual_nodes = num_regional_virtual_nodes
        if regional_geometry_coord_source not in {"origin", "layer"}:
            raise ValueError(
                "regional_geometry_coord_source must be either 'origin' or 'layer'"
            )
        if regional_end_stage_mode not in {
            "flat",
            "legacy_pool_unpool",
            "pooled_context_fusion",
        }:
            raise ValueError(
                "regional_end_stage_mode must be one of {'flat', 'legacy_pool_unpool', 'pooled_context_fusion'}"
            )
        if regional_context_fusion not in {"gated", "residual"}:
            raise ValueError(
                "regional_context_fusion must be one of {'gated', 'residual'}"
            )

        self.regional_geometry_coord_source = regional_geometry_coord_source
        self.regional_end_stage_mode = regional_end_stage_mode
        coord_dim = 3 * self.num_vector
        self.regional_virtual_node_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            layer_dim = width * width_mult[layer_idx + 1]
            # The final EMNN layer returns a single coordinate channel.
            if layer_idx == self.num_layers - 1:
                coord_dim = 3
            self.regional_virtual_node_layers.append(
                RegionalVirtualNodeLayer(
                    dim=layer_dim,
                    num_virtual_nodes=num_regional_virtual_nodes,
                    coord_dim=coord_dim,
                    assignment_hidden_dim=regional_assignment_hidden_dim,
                    num_attention_heads=regional_num_attention_heads,
                    num_attention_layers=regional_num_attention_layers,
                    attention_dropout=regional_attention_dropout,
                    transformer_hidden_mult=regional_transformer_hidden_mult,
                    use_coordinate_features=regional_use_coordinate_features,
                    geometry_feature_mode=regional_geometry_feature_mode,
                    geometry_anchor_count=regional_geometry_anchor_count,
                    geometry_rbf_bins=regional_geometry_rbf_bins,
                    geometry_rbf_sigma=regional_geometry_rbf_sigma,
                    geometry_feature_normalization=regional_geometry_feature_normalization,
                    use_region_geometry_features=regional_use_region_geometry_features,
                    residual_weight=regional_residual_weight,
                    learnable_residual_weight=regional_learnable_residual_weight,
                    eps=regional_eps,
                    geometry_prior_mode=regional_geometry_prior_mode,
                    geometry_prior_weight=regional_geometry_prior_weight,
                    geometry_prior_sigma=regional_geometry_prior_sigma,
                    geometry_radius=regional_geometry_radius,
                    geometry_random_start=regional_geometry_random_start,
                    geometry_detach_coords=regional_geometry_detach_coords,
                    norm_type=self.norm_type,
                )
            )
        if regional_only_last_layer:
            for layer_idx in range(self.num_layers - 1):
                # Detect this in forward pass
                self.regional_virtual_node_layers[layer_idx] = None

        final_dim = width * width_mult[-1]
        context_dim = (
            final_dim if regional_context_dim is None else regional_context_dim
        )
        self.context_pool = None
        self.context_blocks = nn.ModuleList()
        self.context_fusion = None
        self.context_residual_proj = None
        self.context_lift_k = regional_context_lift_k
        if self.regional_end_stage_mode == "pooled_context_fusion":
            if regional_context_num_blocks < 1:
                raise ValueError("regional_context_num_blocks must be >= 1")
            self.context_pool = FeaturePooling(
                in_dim=final_dim,
                out_dim=context_dim,
                ratio=regional_context_pool_ratio,
                r=regional_context_pool_radius,
                max_num_neighbors=regional_context_max_num_neighbors,
                aggr=regional_context_pool_aggr,
                random_start=regional_context_random_start,
                use_feature_scores=regional_context_use_feature_scores,
            )
            for _ in range(regional_context_num_blocks):
                self.context_blocks.append(
                    PooledContextBlock(
                        dim=context_dim,
                        radius_value=regional_context_pool_radius,
                        max_num_neighbors=regional_context_max_num_neighbors,
                        aggr=regional_context_pool_aggr,
                        dropout=regional_context_fusion_dropout,
                        geometry_mode=regional_context_geometry_mode,
                        geometry_rbf_bins=regional_context_rbf_bins,
                        geometry_rbf_sigma=regional_context_rbf_sigma,
                        norm_type=self.norm_type,
                    )
                )
            if regional_context_fusion == "gated":
                self.context_fusion = FeatureUnpooling(
                    coarse_dim=context_dim,
                    skip_dim=final_dim,
                    out_dim=final_dim,
                    k=regional_context_lift_k,
                )
            else:
                self.context_residual_proj = nn.Sequential(
                    nn.Linear(context_dim, final_dim),
                    nn.SiLU(),
                    nn.Dropout(regional_context_fusion_dropout),
                    nn.Linear(final_dim, final_dim),
                )
            self.context_output_norm = build_norm_layer(final_dim, self.norm_type)
            self.regional_context_fusion = regional_context_fusion

        logger.info(
            "Using "
            f"{num_regional_virtual_nodes} regional virtual nodes after every EMNN conv layer "
            f"(geometry_feature_mode={regional_geometry_feature_mode or ('raw_coordinates' if regional_use_coordinate_features else 'none')}, "
            f"context_geometry_mode={regional_context_geometry_mode})"
        )

    def _run_pooled_context_fusion(
        self,
        *,
        x: torch.Tensor,
        pos_origin: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.context_pool is None:
            return x

        pooled = self.context_pool(pos_origin, pos, x, batch)
        pooled_x = pooled["x"]
        pooled_pos_origin = pooled["pos_origin"]
        pooled_batch = pooled["batch"]

        for context_block in self.context_blocks:
            pooled_x = context_block(pooled_x, pooled_pos_origin, pooled_batch)

        if self.context_fusion is not None:
            x_update = self.context_fusion(
                pooled_pos_origin,
                pooled_x,
                pooled_batch,
                pos_origin,
                x,
                batch,
            )
            return self.context_output_norm(x + x_update)

        lifted = knn_interpolate(
            pooled_x,
            pooled_pos_origin,
            pos_origin,
            pooled_batch,
            batch,
            k=self.context_lift_k,
        )
        x_update = self.context_residual_proj(lifted)
        return self.context_output_norm(x + x_update)

    def _run_end_stage(
        self,
        *,
        x: torch.Tensor,
        pos_origin: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.regional_end_stage_mode == "flat":
            return x
        if self.regional_end_stage_mode == "legacy_pool_unpool":
            return self._run_hierarchy(x=x, pos_origin=pos_origin, pos=pos, batch=batch)
        return self._run_pooled_context_fusion(
            x=x,
            pos_origin=pos_origin,
            pos=pos,
            batch=batch,
        )

    def forward(self, data: MeshData) -> EncoderOutput:
        inputs = self._prepare_forward_inputs(data)
        batch = inputs["batch"]
        pos = inputs["pos"]
        pos_origin = inputs["pos_origin"]
        face = inputs["face"]
        x = inputs["x"]
        edge_attr = inputs["edge_attr"]
        thickness = inputs["thickness"]
        thickness_paired_idx = inputs["thickness_paired_idx"]
        thickness_embed = inputs["thickness_embed"]

        assignment_matrix = None
        region_features = None
        region_coords = None
        region_mass = None

        for layer_idx, (conv_layer, regional_layer) in enumerate(
            zip(self.conv_layers, self.regional_virtual_node_layers)
        ):
            x, pos, thickness_embed = self._run_conv_layer(
                conv_layer=conv_layer,
                layer_idx=layer_idx,
                x=x,
                pos=pos,
                batch=batch,
                face=face,
                edge_attr=edge_attr,
                data=data,
                thickness=thickness,
                thickness_embed=thickness_embed,
                thickness_paired_idx=thickness_paired_idx,
            )
            if regional_layer is not None:
                geometry_coord = (
                    pos_origin
                    if self.regional_geometry_coord_source == "origin"
                    else pos
                )
                regional_output = regional_layer(
                    x=x,
                    coord=pos,
                    batch=batch,
                    geometry_coord=geometry_coord,
                )
                x = regional_output["x"]
                assignment_matrix = regional_output["assignment_matrix"]
                region_features = regional_output["region_features"]
                region_coords = regional_output["region_coords"]
                region_mass = regional_output["region_mass"]

            if self.detect_nan and not torch.isfinite(x).all():
                raise ValueError(
                    f"Non-finite values detected in EMNNRegionalVirtualNodes after regional layer {layer_idx}"
                )

        x = self._run_end_stage(x=x, pos_origin=pos_origin, pos=pos, batch=batch)

        return EncoderOutput(
            node_embedding=x,
            pos=pos,
            batch=batch,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            assignment_matrix=assignment_matrix,
            region_features=region_features,
            region_coords=region_coords,
            region_mass=region_mass,
        )
