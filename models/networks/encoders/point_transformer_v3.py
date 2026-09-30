import importlib

import torch
import torch.nn as nn

from data.types import MeshData

from ..contracts import EncoderOutput
from .point_conv import (
    _compute_face_batch,
    _gather_sampled_node_features,
    _interpolate_to_original_nodes,
    _validate_pointcloud_layout,
)


class PointTransformerV3Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edges_input_dim: int | None = None,
        use_node_features: bool = False,
        use_normals: bool = True,
        order: tuple[str, ...] | list[str] = (
            "z",
            "z-trans",
            "hilbert",
            "hilbert-trans",
        ),
        stride: tuple[int, ...] | list[int] = (2, 2, 2, 2),
        enc_depths: tuple[int, ...] | list[int] = (2, 2, 2, 6, 2),
        enc_channels: tuple[int, ...] | list[int] = (32, 64, 128, 256, 512),
        enc_num_head: tuple[int, ...] | list[int] = (2, 4, 8, 16, 32),
        enc_patch_size: tuple[int, ...] | list[int] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: tuple[int, ...] | list[int] = (2, 2, 2, 2),
        dec_channels: tuple[int, ...] | list[int] = (64, 64, 128, 256),
        dec_num_head: tuple[int, ...] | list[int] = (4, 4, 8, 16),
        dec_patch_size: tuple[int, ...] | list[int] = (1024, 1024, 1024, 1024),
        grid_size: float = 0.01,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        shuffle_orders: bool = True,
        enable_rpe: bool = False,
        enable_flash: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cls_mode: bool = False,
        detect_nan: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if cls_mode:
            raise ValueError(
                "PointTransformerV3Encoder baseline requires cls_mode=False to preserve one feature vector per input vertex."
            )
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got: {grid_size}")

        self.input_dim = input_dim
        self.edges_input_dim = 0 if edges_input_dim is None else edges_input_dim
        self.use_node_features = use_node_features
        self.use_normals = use_normals
        self.detect_nan = detect_nan
        self.grid_size = float(grid_size)
        self.cls_mode = cls_mode
        self.point_feature_dim = (input_dim if use_node_features else 0) + (
            3 if use_normals else 0
        )
        if self.point_feature_dim < 1:
            raise ValueError(
                "PointTransformerV3Encoder requires at least one point feature channel; enable normals or node features."
            )
        self.output_dim = tuple(dec_channels)[0]

        ptv3_cls = self._load_ptv3_class()
        self.backbone = ptv3_cls(
            in_channels=self.point_feature_dim,
            order=tuple(order),
            stride=tuple(stride),
            enc_depths=tuple(enc_depths),
            enc_channels=tuple(enc_channels),
            enc_num_head=tuple(enc_num_head),
            enc_patch_size=tuple(enc_patch_size),
            dec_depths=tuple(dec_depths),
            dec_channels=tuple(dec_channels),
            dec_num_head=tuple(dec_num_head),
            dec_patch_size=tuple(dec_patch_size),
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=cls_mode,
        )

    @staticmethod
    def _load_ptv3_class():
        try:
            module = importlib.import_module("PointTransformerV3.model")
        except Exception as exc:
            raise ImportError(
                "PointTransformerV3 dependencies are unavailable. Install the PTv3 runtime stack, including spconv, torch-scatter, timm, and addict. flash-attn is only required when enable_flash=true."
            ) from exc
        return module.PointTransformerV3

    @property
    def required_batch_attributes(self) -> list[str]:
        required = [
            "pos",
            "batch",
            "pointcloud_pos",
            "pointcloud_original_idx",
            "pointcloud_valid_mask",
            "pointcloud_num_points",
        ]
        if self.use_node_features:
            required.append("x")
        if self.use_normals:
            required.append("pointcloud_normal")
        return required

    def _check_finite(
        self, tensor: torch.Tensor | None, name: str, error_message: str
    ) -> None:
        if tensor is None or not self.detect_nan:
            return
        if torch.isfinite(tensor).all():
            return
        raise ValueError(error_message)

    def _extract_point_value(self, point, key: str) -> torch.Tensor | None:
        if isinstance(point, dict):
            return point.get(key)
        return getattr(point, key, None)

    def _compute_grid_coord(
        self, coord: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        if coord.shape[0] == 0:
            return coord.new_empty((0, 3), dtype=torch.int32)

        num_graphs = int(batch.max().item()) + 1
        min_coord = torch.empty((num_graphs, 3), device=coord.device, dtype=coord.dtype)
        for graph_idx in range(num_graphs):
            graph_coord = coord[batch == graph_idx]
            min_coord[graph_idx] = graph_coord.min(dim=0).values

        return torch.div(
            coord - min_coord[batch],
            self.grid_size,
            rounding_mode="trunc",
        ).int()

    def _build_point_features(
        self,
        *,
        sampled_node_features: torch.Tensor | None,
        pointcloud_normal: torch.Tensor | None,
        num_graphs: int,
        n_points: int,
    ) -> torch.Tensor:
        feature_parts = []
        if sampled_node_features is not None:
            feature_parts.append(sampled_node_features.view(num_graphs, n_points, -1))
        if pointcloud_normal is not None:
            feature_parts.append(pointcloud_normal.view(num_graphs, n_points, 3))
        if not feature_parts:
            raise ValueError(
                "PointTransformerV3Encoder requires at least one point feature tensor"
            )
        return torch.cat(feature_parts, dim=-1).view(num_graphs * n_points, -1)

    def _prepare_inputs(self, data: MeshData) -> dict[str, torch.Tensor | float]:
        (
            pos,
            batch,
            pointcloud_pos,
            pointcloud_valid_mask,
            pointcloud_original_idx,
            pointcloud_normal,
            num_graphs,
            n_points,
        ) = _validate_pointcloud_layout(
            data,
            require_normals=self.use_normals,
            require_original_idx=True,
            detect_nan=self.detect_nan,
        )

        sampled_node_features = None
        if self.use_node_features:
            sampled_node_features = _gather_sampled_node_features(
                data,
                input_dim=self.input_dim,
                pointcloud_original_idx=pointcloud_original_idx,
                pointcloud_pos=pointcloud_pos,
                detect_nan=self.detect_nan,
            )

        feat = self._build_point_features(
            sampled_node_features=sampled_node_features,
            pointcloud_normal=pointcloud_normal,
            num_graphs=num_graphs,
            n_points=n_points,
        )
        self._check_finite(
            feat, "point features", "Non-finite values in PTv3 point features"
        )

        sampled_batch = torch.arange(
            num_graphs,
            device=batch.device,
            dtype=batch.dtype,
        ).repeat_interleave(n_points)
        sampled_pos = pointcloud_pos.view(num_graphs, n_points, 3)
        sampled_source_pos = pos[pointcloud_original_idx].view(num_graphs, n_points, 3)
        sampled_valid_mask = pointcloud_valid_mask.view(num_graphs, n_points)

        inputs: dict[str, torch.Tensor | float] = {
            "coord": pointcloud_pos,
            "feat": feat,
            "batch": sampled_batch,
        }

        grid_coord = getattr(data, "pointcloud_grid_coord", None)
        if grid_coord is None:
            grid_coord = self._compute_grid_coord(pointcloud_pos, sampled_batch)
        else:
            grid_coord = grid_coord.int()
        if grid_coord.ndim != 2 or grid_coord.shape[-1] != 3:
            raise ValueError(
                f"PointTransformerV3Encoder expects grid_coord with shape (N, 3), got: {tuple(grid_coord.shape)}"
            )
        if grid_coord.shape[0] != pointcloud_pos.shape[0]:
            raise ValueError(
                "PointTransformerV3Encoder expects pointcloud grid_coord to align with pointcloud_pos"
            )
        inputs["grid_coord"] = grid_coord

        return {
            "inputs": inputs,
            "pos": pos,
            "batch": batch,
            "sampled_source_pos": sampled_source_pos,
            "sampled_valid_mask": sampled_valid_mask,
        }

    def forward(self, data: MeshData) -> EncoderOutput:
        prepared = self._prepare_inputs(data)
        point = self.backbone(prepared["inputs"])
        sampled_embedding = self._extract_point_value(point, "feat")

        if sampled_embedding is None:
            raise ValueError(
                "PointTransformerV3 backbone did not return per-point features"
            )
        self._check_finite(
            sampled_embedding,
            "PTv3 node embedding",
            "Non-finite values in PTv3 node embedding",
        )

        if sampled_embedding.shape[0] != prepared["inputs"]["coord"].shape[0]:
            raise ValueError(
                "PointTransformerV3Encoder baseline expects one embedding per sampled point. Ensure cls_mode=False."
            )

        node_embedding = _interpolate_to_original_nodes(
            sampled_embedding=sampled_embedding.view(
                prepared["sampled_source_pos"].shape[0],
                prepared["sampled_source_pos"].shape[1],
                -1,
            ),
            sampled_pos=prepared["sampled_source_pos"],
            sampled_valid_mask=prepared["sampled_valid_mask"],
            original_pos=prepared["pos"],
            original_batch=prepared["batch"],
        )
        face, face_batch = _compute_face_batch(
            getattr(data, "face", None), prepared["batch"]
        )

        return EncoderOutput(
            node_embedding=node_embedding,
            pos=data.pos,
            batch=data.batch,
            edge_index=getattr(data, "edge_index", None),
            edge_attr=getattr(data, "edge_attr", None),
            face=face,
            face_batch=face_batch,
        )
