import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.utils import scatter

from data.types import MeshData
from models.layers.emnn_conv import (
    MeshEGNN,
    MeshEGNNFastVirtual,
    MeshEGNNmc,
    MeshEGNNmcFastVirtual,
)

from ..contracts import EncoderOutput
from .emnn import EMNN


class EMNNFastVirtualNodes(EMNN):
    def __init__(
        self,
        input_dim: int,
        edges_input_dim: int,
        width: int,
        width_mult: list[int],
        num_vector: int = 1,
        num_fast_virtual_nodes: int = 8,
        fast_virtual_dropout: float = 0.0,
        fast_virtual_residual_weight: float = 1.0,
        fast_virtual_learnable_residual_weight: bool = False,
        fast_virtual_only_last_layer: bool = False,
        fast_virtual_eps: float = 1e-8,
        fast_virtual_end_stage_mode: str = "flat",
        **kwargs,
    ):
        compat_hier = bool(kwargs.get("hier", False))
        if fast_virtual_end_stage_mode == "flat" and compat_hier:
            logger.warning(
                "EMNNFastVirtualNodes received hier=True with fast_virtual_end_stage_mode='flat'; "
                "using legacy_pool_unpool for backward compatibility."
            )
            fast_virtual_end_stage_mode = "legacy_pool_unpool"

        if fast_virtual_end_stage_mode != "legacy_pool_unpool":
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

        if num_fast_virtual_nodes < 1:
            raise ValueError(
                f"num_fast_virtual_nodes must be >= 1, got: {num_fast_virtual_nodes}"
            )

        self.num_fast_virtual_nodes = num_fast_virtual_nodes
        self.fast_virtual_only_last_layer = fast_virtual_only_last_layer
        self.fast_virtual_dropout = fast_virtual_dropout
        if fast_virtual_end_stage_mode not in {"flat", "legacy_pool_unpool"}:
            raise ValueError(
                "fast_virtual_end_stage_mode must be one of {'flat', 'legacy_pool_unpool'}"
            )
        self.fast_virtual_end_stage_mode = fast_virtual_end_stage_mode

        integrated_start_idx = (
            self.num_layers - 1 if fast_virtual_only_last_layer else 0
        )
        first_dim = width * width_mult[integrated_start_idx]
        self.initial_virtual_feature_seed = nn.Parameter(
            torch.randn(1, num_fast_virtual_nodes, first_dim) * 0.02
        )

        self.conv_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            in_dim = width * width_mult[layer_idx]
            out_dim = width * width_mult[layer_idx + 1]
            use_integrated_virtual = (
                not fast_virtual_only_last_layer or layer_idx == self.num_layers - 1
            )
            if use_integrated_virtual:
                layer_cls = MeshEGNNmcFastVirtual if self.mc else MeshEGNNFastVirtual
            else:
                layer_cls = MeshEGNNmc if self.mc else MeshEGNN
            layer_kwargs = dict(
                input_nf=in_dim,
                output_nf=out_dim,
                hidden_nf=in_dim,
                edges_in_d=self.edges_input_dim,
                attention=self.conv_attention,
                normalize=self.normalize_coord_diff,
                coords_agg="sum",
                tanh=self.coord_update_tanh,
            )
            if self.mc:
                num_vectors_in = self.num_vector
                num_vectors_out = (
                    self.num_vector if layer_idx < self.num_layers - 1 else 1
                )
                layer_kwargs.update(
                    num_vectors_in=num_vectors_in,
                    num_vectors_out=num_vectors_out,
                    last_layer=(layer_idx == self.num_layers - 1),
                )
            if use_integrated_virtual:
                layer_kwargs.update(
                    num_virtual_nodes=num_fast_virtual_nodes,
                    virtual_dropout=fast_virtual_dropout,
                )
            self.conv_layers.append(layer_cls(**layer_kwargs))

        if (
            fast_virtual_residual_weight != 1.0
            or fast_virtual_learnable_residual_weight
        ):
            logger.warning(
                "fast_virtual_residual_weight and fast_virtual_learnable_residual_weight "
                "are not used by the integrated virtual-node conv implementation."
            )
        if fast_virtual_eps != 1e-8:
            logger.warning(
                "fast_virtual_eps is currently reserved and not used by the integrated virtual-node conv implementation."
            )

        logger.info(
            "Using FastEGNN-style virtual nodes "
            f"(num_fast_virtual_nodes={num_fast_virtual_nodes}, only_last_layer={fast_virtual_only_last_layer})"
        )

    def _initialize_virtual_state(
        self, pos: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        feature_dim = self.initial_virtual_feature_seed.shape[-1]
        if num_graphs == 0:
            return (
                self.initial_virtual_feature_seed.new_zeros(
                    (0, self.num_fast_virtual_nodes, feature_dim)
                ),
                pos.new_zeros((0, self.num_fast_virtual_nodes, *pos.shape[1:])),
            )

        graph_centers = scatter(pos, batch, dim=0, dim_size=num_graphs, reduce="mean")
        virtual_node_features = self.initial_virtual_feature_seed.expand(
            num_graphs, -1, -1
        )
        if graph_centers.ndim == 2:
            virtual_node_coords = graph_centers.unsqueeze(1).expand(
                -1, self.num_fast_virtual_nodes, -1
            )
        else:
            virtual_node_coords = graph_centers.unsqueeze(1).expand(
                -1, self.num_fast_virtual_nodes, -1, -1
            )
        return virtual_node_features, virtual_node_coords

    def _run_end_stage(
        self,
        *,
        x: torch.Tensor,
        pos_origin: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.fast_virtual_end_stage_mode == "flat":
            return x
        return self._run_hierarchy(x=x, pos_origin=pos_origin, pos=pos, batch=batch)

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

        virtual_node_features = None
        virtual_node_coords = None

        for layer_idx, conv_layer in enumerate(self.conv_layers):
            use_integrated_virtual = (
                not self.fast_virtual_only_last_layer
                or layer_idx == self.num_layers - 1
            )

            if not use_integrated_virtual:
                if thickness_embed is None:
                    x, pos = conv_layer(
                        x,
                        edge_index=data.edge_index.long(),
                        coord=pos,
                        edge_attr=edge_attr,
                        face=face.long(),
                    )
                else:
                    thickness_scale = self._compute_thickness_gate(thickness)
                    x, pos, thickness_embed = conv_layer(
                        x,
                        edge_index=data.edge_index.long(),
                        coord=pos,
                        edge_attr=edge_attr,
                        face=face.long(),
                        thickness_embed=thickness_embed,
                        thickness_paired_idx=thickness_paired_idx,
                        thickness_scale=thickness_scale.unsqueeze(-1),
                    )
            else:
                if virtual_node_features is None or virtual_node_coords is None:
                    virtual_node_features, virtual_node_coords = (
                        self._initialize_virtual_state(pos, batch)
                    )
                if thickness_embed is None:
                    x, pos, virtual_node_features, virtual_node_coords = conv_layer(
                        x,
                        edge_index=data.edge_index.long(),
                        coord=pos,
                        edge_attr=edge_attr,
                        face=face.long(),
                        batch=batch,
                        virtual_node_feat=virtual_node_features,
                        virtual_node_coord=virtual_node_coords,
                    )
                else:
                    thickness_scale = self._compute_thickness_gate(thickness)
                    (
                        x,
                        pos,
                        virtual_node_features,
                        virtual_node_coords,
                        thickness_embed,
                    ) = conv_layer(
                        x,
                        edge_index=data.edge_index.long(),
                        coord=pos,
                        edge_attr=edge_attr,
                        face=face.long(),
                        batch=batch,
                        virtual_node_feat=virtual_node_features,
                        virtual_node_coord=virtual_node_coords,
                        thickness_embed=thickness_embed,
                        thickness_paired_idx=thickness_paired_idx,
                        thickness_scale=thickness_scale.unsqueeze(-1),
                    )

            if self.virtual_global_node:
                x = self._apply_virtual_global_node(x, batch=batch, layer_idx=layer_idx)

            if self.detect_nan and not torch.isfinite(x).all():
                raise ValueError(
                    "Non-finite values detected in EMNNFastVirtualNodes after "
                    f"virtual-node layer {layer_idx}"
                )

        x = self._run_end_stage(x=x, pos_origin=pos_origin, pos=pos, batch=batch)

        face_batch = None
        if face is not None:
            face_batch = (
                batch[face[0].long()]
                if face.numel() > 0
                else batch.new_empty((0,), dtype=batch.dtype)
            )

        return EncoderOutput(
            node_embedding=x,
            pos=pos,
            batch=batch,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            face=face,
            face_batch=face_batch,
            virtual_node_features=virtual_node_features,
            virtual_node_coords=virtual_node_coords,
        )
