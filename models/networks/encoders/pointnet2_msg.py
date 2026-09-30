import torch
import torch.nn as nn

from data.types import MeshData
from models.point2_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)

from ..contracts import EncoderOutput
from .point_conv import (
    _compute_face_batch,
    _gather_sampled_node_features,
    _interpolate_to_original_nodes,
    _validate_pointcloud_layout,
)


def _run_msg_set_abstraction(
    layer: PointNetSetAbstractionMsg,
    xyz: torch.Tensor,
    features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_points = int(xyz.shape[-1])
    if num_points < 1:
        raise ValueError("PointNet++ MSG set abstraction requires at least one point")

    original_npoint = layer.npoint
    original_nsample_list = list(layer.nsample_list)
    layer.npoint = min(layer.npoint, num_points)
    layer.nsample_list = [min(nsample, num_points) for nsample in layer.nsample_list]

    try:
        return layer(xyz, features)
    finally:
        layer.npoint = original_npoint
        layer.nsample_list = original_nsample_list


def _run_set_abstraction(
    layer: PointNetSetAbstraction,
    xyz: torch.Tensor,
    features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_points = int(xyz.shape[-1])
    if num_points < 1:
        raise ValueError("PointNet++ set abstraction requires at least one point")

    original_npoint = layer.npoint
    original_nsample = layer.nsample
    if not layer.group_all:
        layer.npoint = min(layer.npoint, num_points)
        layer.nsample = min(layer.nsample, num_points)

    try:
        return layer(xyz, features)
    finally:
        layer.npoint = original_npoint
        layer.nsample = original_nsample


class PointNet2MSGVertexEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 0,
        edges_input_dim: int | None = None,
        use_node_features: bool = False,
        use_normals: bool = True,
        num_object_classes: int = 1,
        object_class_index: int = 0,
        sa1_npoint: int = 512,
        sa1_radius_list: tuple[float, ...] | list[float] = (0.1, 0.2, 0.4),
        sa1_nsample_list: tuple[int, ...] | list[int] = (32, 64, 128),
        sa1_mlp_list: tuple[tuple[int, ...], ...] | list[list[int]] = (
            (32, 32, 64),
            (64, 64, 128),
            (64, 96, 128),
        ),
        sa2_npoint: int = 128,
        sa2_radius_list: tuple[float, ...] | list[float] = (0.4, 0.8),
        sa2_nsample_list: tuple[int, ...] | list[int] = (64, 128),
        sa2_mlp_list: tuple[tuple[int, ...], ...] | list[list[int]] = (
            (128, 128, 256),
            (128, 196, 256),
        ),
        sa3_mlp: tuple[int, ...] | list[int] = (256, 512, 1024),
        fp3_mlp: tuple[int, ...] | list[int] = (256, 256),
        fp2_mlp: tuple[int, ...] | list[int] = (256, 128),
        fp1_mlp: tuple[int, ...] | list[int] = (128, 128),
        detect_nan: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if input_dim < 0:
            raise ValueError(f"input_dim must be non-negative, got: {input_dim}")
        if num_object_classes < 0:
            raise ValueError(
                f"num_object_classes must be non-negative, got: {num_object_classes}"
            )
        if num_object_classes > 0 and not (
            0 <= object_class_index < num_object_classes
        ):
            raise ValueError(
                "object_class_index must be in [0, num_object_classes) when class conditioning is enabled"
            )

        self.input_dim = input_dim
        self.edges_input_dim = 0 if edges_input_dim is None else edges_input_dim
        self.use_node_features = use_node_features
        self.use_normals = use_normals
        self.num_object_classes = num_object_classes
        self.object_class_index = object_class_index
        self.detect_nan = detect_nan

        sa1_radius_list = list(sa1_radius_list)
        sa1_nsample_list = list(sa1_nsample_list)
        sa1_mlp_list = [list(mlp) for mlp in sa1_mlp_list]
        sa2_radius_list = list(sa2_radius_list)
        sa2_nsample_list = list(sa2_nsample_list)
        sa2_mlp_list = [list(mlp) for mlp in sa2_mlp_list]
        sa3_mlp = list(sa3_mlp)
        fp3_mlp = list(fp3_mlp)
        fp2_mlp = list(fp2_mlp)
        fp1_mlp = list(fp1_mlp)

        if len(sa1_radius_list) != len(sa1_nsample_list) or len(sa1_radius_list) != len(
            sa1_mlp_list
        ):
            raise ValueError(
                "sa1 radius, nsample, and mlp lists must have matching lengths"
            )
        if len(sa2_radius_list) != len(sa2_nsample_list) or len(sa2_radius_list) != len(
            sa2_mlp_list
        ):
            raise ValueError(
                "sa2 radius, nsample, and mlp lists must have matching lengths"
            )

        self.sampled_feature_dim = (
            3 + (input_dim if use_node_features else 0) + (3 if use_normals else 0)
        )
        sa1_out_dim = sum(mlp[-1] for mlp in sa1_mlp_list)
        sa2_out_dim = sum(mlp[-1] for mlp in sa2_mlp_list)
        fp3_out_dim = fp3_mlp[-1]
        fp2_out_dim = fp2_mlp[-1]
        fp1_skip_dim = 3 + self.sampled_feature_dim + num_object_classes

        self.sa1 = PointNetSetAbstractionMsg(
            sa1_npoint,
            sa1_radius_list,
            sa1_nsample_list,
            self.sampled_feature_dim,
            sa1_mlp_list,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            sa2_npoint,
            sa2_radius_list,
            sa2_nsample_list,
            sa1_out_dim,
            sa2_mlp_list,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=sa2_out_dim + 3,
            mlp=sa3_mlp,
            group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=sa2_out_dim + sa3_mlp[-1],
            mlp=fp3_mlp,
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=sa1_out_dim + fp3_out_dim,
            mlp=fp2_mlp,
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=fp1_skip_dim + fp2_out_dim,
            mlp=fp1_mlp,
        )
        self.output_dim = fp1_mlp[-1]

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

    def _build_point_features(
        self,
        *,
        xyz: torch.Tensor,
        sampled_node_features: torch.Tensor | None,
        pointcloud_normal: torch.Tensor | None,
        num_graphs: int,
        n_points: int,
    ) -> torch.Tensor:
        feature_parts = [xyz.transpose(1, 2).contiguous()]
        if sampled_node_features is not None:
            feature_parts.append(sampled_node_features.view(num_graphs, n_points, -1))
        if pointcloud_normal is not None:
            feature_parts.append(pointcloud_normal.view(num_graphs, n_points, 3))
        point_features = torch.cat(feature_parts, dim=-1)
        return point_features.transpose(1, 2).contiguous()

    def _build_object_conditioning(
        self, num_graphs: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        if self.num_object_classes == 0:
            return None
        conditioning = torch.zeros(
            (num_graphs, self.num_object_classes), device=device, dtype=dtype
        )
        conditioning[:, self.object_class_index] = 1.0
        return conditioning

    def forward(self, data: MeshData) -> EncoderOutput:
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

        if num_graphs == 0:
            node_embedding = pos.new_empty((0, self.output_dim))
        else:
            sampled_pos = pointcloud_pos.view(num_graphs, n_points, 3)
            sampled_source_pos = pos[pointcloud_original_idx].view(
                num_graphs, n_points, 3
            )
            sampled_valid_mask = pointcloud_valid_mask.view(num_graphs, n_points)
            xyz = sampled_pos.transpose(1, 2).contiguous()

            sampled_node_features = None
            if self.use_node_features:
                sampled_node_features = _gather_sampled_node_features(
                    data,
                    input_dim=self.input_dim,
                    pointcloud_original_idx=pointcloud_original_idx,
                    pointcloud_pos=pointcloud_pos,
                    detect_nan=self.detect_nan,
                )

            l0_points = self._build_point_features(
                xyz=xyz,
                sampled_node_features=sampled_node_features,
                pointcloud_normal=pointcloud_normal,
                num_graphs=num_graphs,
                n_points=n_points,
            )

            l1_xyz, l1_points = _run_msg_set_abstraction(self.sa1, xyz, l0_points)
            l2_xyz, l2_points = _run_msg_set_abstraction(self.sa2, l1_xyz, l1_points)
            l3_xyz, l3_points = _run_set_abstraction(self.sa3, l2_xyz, l2_points)

            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

            object_conditioning = self._build_object_conditioning(
                num_graphs=num_graphs,
                device=xyz.device,
                dtype=xyz.dtype,
            )
            fp1_inputs = [xyz, l0_points]
            if object_conditioning is not None:
                fp1_inputs.insert(
                    0, object_conditioning.unsqueeze(-1).repeat(1, 1, n_points)
                )

            l0_fused = torch.cat(fp1_inputs, dim=1)
            l0_points = self.fp1(xyz, l1_xyz, l0_fused, l1_points)
            sampled_embedding = l0_points.transpose(1, 2).contiguous()

            if self.detect_nan and not torch.isfinite(sampled_embedding).all():
                raise ValueError(
                    "Non-finite values in PointNet++ MSG sampled embeddings"
                )

            node_embedding = _interpolate_to_original_nodes(
                sampled_embedding=sampled_embedding,
                sampled_pos=sampled_source_pos,
                sampled_valid_mask=sampled_valid_mask,
                original_pos=pos,
                original_batch=batch,
            )

        face, face_batch = _compute_face_batch(getattr(data, "face", None), batch)

        return EncoderOutput(
            node_embedding=node_embedding,
            pos=data.pos,
            batch=data.batch,
            edge_index=getattr(data, "edge_index", None),
            edge_attr=getattr(data, "edge_attr", None),
            face=face,
            face_batch=face_batch,
        )
