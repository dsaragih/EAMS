import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from data.types import MeshData
from models.layers.point_conv import PointConvDensitySetAbstraction
from models.point2_utils import PointNetFeaturePropagation

from ..contracts import EncoderOutput


def _run_pointconv_set_abstraction(
    layer: PointConvDensitySetAbstraction,
    xyz: torch.Tensor,
    features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_points = int(xyz.shape[-1])
    if num_points < 1:
        raise ValueError("PointConv set abstraction requires at least one point")

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


def _validate_pointcloud_layout(
    data: MeshData,
    *,
    require_normals: bool,
    require_original_idx: bool,
    detect_nan: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    int,
]:
    required = [
        "pos",
        "batch",
        "pointcloud_pos",
        "pointcloud_valid_mask",
        "pointcloud_num_points",
    ]
    if require_normals:
        required.append("pointcloud_normal")
    if require_original_idx:
        required.append("pointcloud_original_idx")
    for attr in required:
        if not hasattr(data, attr):
            raise ValueError(f"Input data is missing required attribute '{attr}'")

    pos = data.pos.float()
    batch = data.batch.long()
    pointcloud_pos = data.pointcloud_pos.float()
    pointcloud_valid_mask = data.pointcloud_valid_mask.bool()
    pointcloud_num_points = data.pointcloud_num_points.long().view(-1)
    pointcloud_original_idx = getattr(data, "pointcloud_original_idx", None)
    if pointcloud_original_idx is not None:
        pointcloud_original_idx = pointcloud_original_idx.long().view(-1)
    pointcloud_normal = getattr(data, "pointcloud_normal", None)
    if pointcloud_normal is not None:
        pointcloud_normal = pointcloud_normal.float()

    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(
            f"Expected data.pos with shape (N, 3), got: {tuple(pos.shape)}"
        )
    if batch.ndim != 1 or batch.shape[0] != pos.shape[0]:
        raise ValueError("Expected data.batch to align with data.pos")
    if pointcloud_pos.ndim != 2 or pointcloud_pos.shape[-1] != 3:
        raise ValueError(
            "Expected data.pointcloud_pos with shape (B * n_points, 3) after batching"
        )
    if (
        pointcloud_valid_mask.ndim != 1
        or pointcloud_valid_mask.shape[0] != pointcloud_pos.shape[0]
    ):
        raise ValueError(
            "Expected data.pointcloud_valid_mask with shape (B * n_points,) aligned with pointcloud_pos"
        )
    if (
        pointcloud_normal is not None
        and pointcloud_normal.shape != pointcloud_pos.shape
    ):
        raise ValueError(
            "Expected data.pointcloud_normal to align with data.pointcloud_pos"
        )
    if (
        pointcloud_original_idx is not None
        and pointcloud_original_idx.shape[0] != pointcloud_pos.shape[0]
    ):
        raise ValueError(
            "Expected data.pointcloud_original_idx to align with data.pointcloud_pos"
        )

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if pointcloud_num_points.numel() != num_graphs:
        raise ValueError(
            "Expected one pointcloud_num_points entry per graph in the batch"
        )
    if num_graphs == 0:
        return (
            pos,
            batch,
            pointcloud_pos,
            pointcloud_valid_mask,
            pointcloud_original_idx,
            pointcloud_normal,
            0,
            0,
        )

    n_points = int(pointcloud_num_points[0].item())
    if n_points < 1:
        raise ValueError("pointcloud_num_points must be positive")
    if not torch.equal(
        pointcloud_num_points, pointcloud_num_points.new_full((num_graphs,), n_points)
    ):
        raise ValueError(
            "All graphs in the batch must share the same pointcloud_num_points"
        )
    if pointcloud_pos.shape[0] != num_graphs * n_points:
        raise ValueError(
            "Batched pointcloud_pos does not match batch size times point count"
        )

    if detect_nan:
        tensors = [(pos, "positions"), (pointcloud_pos, "sampled positions")]
        if pointcloud_normal is not None:
            tensors.append((pointcloud_normal, "sampled normals"))
        for tensor, name in tensors:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite values in PointConv {name}")

    return (
        pos,
        batch,
        pointcloud_pos,
        pointcloud_valid_mask,
        pointcloud_original_idx,
        pointcloud_normal,
        num_graphs,
        n_points,
    )


def _gather_sampled_node_features(
    data: MeshData,
    *,
    input_dim: int,
    pointcloud_original_idx: torch.Tensor | None,
    pointcloud_pos: torch.Tensor,
    detect_nan: bool,
) -> torch.Tensor:
    if input_dim < 1:
        raise ValueError(
            "PointConv node features requested with non-positive input_dim"
        )
    if pointcloud_original_idx is None:
        raise ValueError(
            "PointConv configured with use_node_features=True but pointcloud_original_idx is missing"
        )

    node_features = getattr(data, "x", None)
    if node_features is None:
        raise ValueError(
            "PointConv configured with use_node_features=True but data.x is missing"
        )

    node_features = node_features.float()
    if node_features.ndim != 2:
        raise ValueError(
            f"Expected data.x with shape (N, C), got: {tuple(node_features.shape)}"
        )
    if node_features.shape[-1] != input_dim:
        raise ValueError(
            f"PointConv expected input feature dim {input_dim}, got: {node_features.shape[-1]}"
        )
    if pointcloud_original_idx.numel() > 0:
        if (
            pointcloud_original_idx.min().item() < 0
            or pointcloud_original_idx.max().item() >= node_features.shape[0]
        ):
            raise ValueError("pointcloud_original_idx contains invalid node indices")

    sampled_features = node_features[pointcloud_original_idx]
    if sampled_features.shape[0] != pointcloud_pos.shape[0]:
        raise ValueError("Sampled node features do not align with pointcloud_pos")
    if detect_nan and not torch.isfinite(sampled_features).all():
        raise ValueError("Non-finite values in PointConv sampled node features")
    return sampled_features


def _compute_face_batch(
    face: torch.Tensor | None,
    batch: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if face is None:
        return None, None
    face = face.long()
    face_batch = (
        batch[face[0].long()]
        if face.numel() > 0
        else batch.new_empty((0,), dtype=batch.dtype)
    )
    return face, face_batch


def _interpolate_to_original_nodes(
    sampled_embedding: torch.Tensor,
    sampled_pos: torch.Tensor,
    sampled_valid_mask: torch.Tensor,
    original_pos: torch.Tensor,
    original_batch: torch.Tensor,
) -> torch.Tensor:
    if original_pos.shape[0] == 0:
        return sampled_embedding.new_empty((0, sampled_embedding.shape[-1]))

    if sampled_pos.ndim != 3:
        raise ValueError(
            f"Expected sampled_pos with shape (B, n_points, 3), got: {tuple(sampled_pos.shape)}"
        )

    dense_original_pos, original_mask = to_dense_batch(original_pos, original_batch)
    pairwise_distance = torch.cdist(dense_original_pos, sampled_pos)
    invalid_sampled = ~sampled_valid_mask.unsqueeze(1)
    pairwise_distance = pairwise_distance.masked_fill(invalid_sampled, float("inf"))

    if (~sampled_valid_mask).all(dim=1).any():
        raise ValueError("Each graph must contain at least one valid sampled point")

    nearest_idx = pairwise_distance.argmin(dim=-1)
    gather_idx = nearest_idx.unsqueeze(-1).expand(-1, -1, sampled_embedding.shape[-1])
    dense_embedding = sampled_embedding.gather(1, gather_idx)
    return dense_embedding[original_mask]


class PointConvVertexEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 0,
        edges_input_dim: int | None = None,
        use_node_features: bool = False,
        use_normals: bool = True,
        sa1_npoint: int = 512,
        sa1_nsample: int = 32,
        sa1_mlp: tuple[int, ...] | list[int] = (64, 64, 128),
        sa1_bandwidth: float = 0.1,
        sa2_npoint: int = 128,
        sa2_nsample: int = 64,
        sa2_mlp: tuple[int, ...] | list[int] = (128, 128, 256),
        sa2_bandwidth: float = 0.2,
        sa3_mlp: tuple[int, ...] | list[int] = (256, 512, 1024),
        sa3_bandwidth: float = 0.4,
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

        self.input_dim = input_dim
        self.edges_input_dim = 0 if edges_input_dim is None else edges_input_dim
        self.use_node_features = use_node_features
        self.use_normals = use_normals
        self.detect_nan = detect_nan
        self.point_feature_dim = (input_dim if use_node_features else 0) + (
            3 if use_normals else 0
        )

        sa1_mlp = list(sa1_mlp)
        sa2_mlp = list(sa2_mlp)
        sa3_mlp = list(sa3_mlp)
        fp3_mlp = list(fp3_mlp)
        fp2_mlp = list(fp2_mlp)
        fp1_mlp = list(fp1_mlp)

        self.sa1 = PointConvDensitySetAbstraction(
            npoint=sa1_npoint,
            nsample=sa1_nsample,
            in_channel=self.point_feature_dim + 3,
            mlp=sa1_mlp,
            bandwidth=sa1_bandwidth,
            group_all=False,
        )
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=sa2_npoint,
            nsample=sa2_nsample,
            in_channel=sa1_mlp[-1] + 3,
            mlp=sa2_mlp,
            bandwidth=sa2_bandwidth,
            group_all=False,
        )
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=1,
            nsample=None,
            in_channel=sa2_mlp[-1] + 3,
            mlp=sa3_mlp,
            bandwidth=sa3_bandwidth,
            group_all=True,
        )
        self.sa3.bn_linear = nn.GroupNorm(1, sa3_mlp[-1])

        self.fp3 = PointNetFeaturePropagation(
            in_channel=sa2_mlp[-1] + sa3_mlp[-1],
            mlp=fp3_mlp,
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=sa1_mlp[-1] + fp3_mlp[-1],
            mlp=fp2_mlp,
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=self.point_feature_dim + fp2_mlp[-1],
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
            point_feature_parts = []
            if self.use_node_features:
                sampled_node_features = _gather_sampled_node_features(
                    data,
                    input_dim=self.input_dim,
                    pointcloud_original_idx=pointcloud_original_idx,
                    pointcloud_pos=pointcloud_pos,
                    detect_nan=self.detect_nan,
                )
                sampled_node_features = sampled_node_features.view(
                    num_graphs, n_points, self.input_dim
                )
                point_feature_parts.append(sampled_node_features)
            if self.use_normals:
                point_feature_parts.append(
                    pointcloud_normal.view(num_graphs, n_points, 3)
                )

            point_features = None
            if point_feature_parts:
                point_features = torch.cat(point_feature_parts, dim=-1)
                point_features = point_features.transpose(1, 2).contiguous()

            l1_xyz, l1_points = _run_pointconv_set_abstraction(
                self.sa1, xyz, point_features
            )
            l2_xyz, l2_points = _run_pointconv_set_abstraction(
                self.sa2, l1_xyz, l1_points
            )
            l3_xyz, l3_points = _run_pointconv_set_abstraction(
                self.sa3, l2_xyz, l2_points
            )

            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(xyz, l1_xyz, point_features, l1_points)
            sampled_embedding = l0_points.transpose(1, 2).contiguous()

            if self.detect_nan and not torch.isfinite(sampled_embedding).all():
                raise ValueError("Non-finite values in PointConv sampled embeddings")

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


class PointConvEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 0,
        edges_input_dim: int | None = None,
        use_node_features: bool = False,
        use_normals: bool = True,
        sa1_npoint: int = 512,
        sa1_nsample: int = 32,
        sa1_mlp: tuple[int, ...] | list[int] = (64, 64, 128),
        sa1_bandwidth: float = 0.1,
        sa2_npoint: int = 128,
        sa2_nsample: int = 64,
        sa2_mlp: tuple[int, ...] | list[int] = (128, 128, 256),
        sa2_bandwidth: float = 0.2,
        sa3_mlp: tuple[int, ...] | list[int] = (256, 512, 1024),
        sa3_bandwidth: float = 0.4,
        detect_nan: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if input_dim < 0:
            raise ValueError(f"input_dim must be non-negative, got: {input_dim}")

        self.input_dim = input_dim
        self.edges_input_dim = 0 if edges_input_dim is None else edges_input_dim
        self.use_node_features = use_node_features
        self.use_normals = use_normals
        self.detect_nan = detect_nan
        self.point_feature_dim = (input_dim if use_node_features else 0) + (
            3 if use_normals else 0
        )

        sa1_mlp = list(sa1_mlp)
        sa2_mlp = list(sa2_mlp)
        sa3_mlp = list(sa3_mlp)

        self.sa1 = PointConvDensitySetAbstraction(
            npoint=sa1_npoint,
            nsample=sa1_nsample,
            in_channel=self.point_feature_dim + 3,
            mlp=sa1_mlp,
            bandwidth=sa1_bandwidth,
            group_all=False,
        )
        self.sa2 = PointConvDensitySetAbstraction(
            npoint=sa2_npoint,
            nsample=sa2_nsample,
            in_channel=sa1_mlp[-1] + 3,
            mlp=sa2_mlp,
            bandwidth=sa2_bandwidth,
            group_all=False,
        )
        self.sa3 = PointConvDensitySetAbstraction(
            npoint=1,
            nsample=None,
            in_channel=sa2_mlp[-1] + 3,
            mlp=sa3_mlp,
            bandwidth=sa3_bandwidth,
            group_all=True,
        )
        self.sa3.bn_linear = nn.GroupNorm(1, sa3_mlp[-1])
        self.output_dim = sa3_mlp[-1]

    @property
    def required_batch_attributes(self) -> list[str]:
        required = [
            "pointcloud_pos",
            "pointcloud_valid_mask",
            "pointcloud_num_points",
            "batch",
        ]
        if self.use_node_features:
            required.extend(["x", "pointcloud_original_idx"])
        if self.use_normals:
            required.append("pointcloud_normal")
        return required

    def forward(self, data: MeshData) -> EncoderOutput:
        (
            _,
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
            require_original_idx=self.use_node_features,
            detect_nan=self.detect_nan,
        )

        if num_graphs == 0:
            node_embedding = pointcloud_pos.new_empty((0, self.output_dim))
            graph_pos = pointcloud_pos.new_empty((0, 3))
            graph_batch = batch.new_empty((0,), dtype=batch.dtype)
        else:
            sampled_pos = pointcloud_pos.view(num_graphs, n_points, 3)
            xyz = sampled_pos.transpose(1, 2).contiguous()
            point_feature_parts = []
            if self.use_node_features:
                sampled_node_features = _gather_sampled_node_features(
                    data,
                    input_dim=self.input_dim,
                    pointcloud_original_idx=pointcloud_original_idx,
                    pointcloud_pos=pointcloud_pos,
                    detect_nan=self.detect_nan,
                )
                sampled_node_features = sampled_node_features.view(
                    num_graphs, n_points, self.input_dim
                )
                point_feature_parts.append(sampled_node_features)
            if self.use_normals:
                point_feature_parts.append(
                    pointcloud_normal.view(num_graphs, n_points, 3)
                )

            point_features = None
            if point_feature_parts:
                point_features = torch.cat(point_feature_parts, dim=-1)
                point_features = point_features.transpose(1, 2).contiguous()

            l1_xyz, l1_points = _run_pointconv_set_abstraction(
                self.sa1, xyz, point_features
            )
            l2_xyz, l2_points = _run_pointconv_set_abstraction(
                self.sa2, l1_xyz, l1_points
            )
            _, l3_points = _run_pointconv_set_abstraction(self.sa3, l2_xyz, l2_points)
            node_embedding = l3_points.squeeze(-1).contiguous()

            if self.detect_nan and not torch.isfinite(node_embedding).all():
                raise ValueError("Non-finite values in PointConv graph embeddings")

            graph_pos = sampled_pos.mean(dim=1)
            graph_batch = torch.arange(
                num_graphs, device=batch.device, dtype=batch.dtype
            )

        return EncoderOutput(
            node_embedding=node_embedding,
            pos=graph_pos,
            batch=graph_batch,
            edge_index=None,
            edge_attr=None,
            face=None,
            face_batch=None,
        )
