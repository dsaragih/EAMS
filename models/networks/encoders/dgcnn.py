import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from data.types import MeshData

from ..contracts import EncoderOutput


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    idx: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


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


def _validate_pointcloud_layout(
    data: MeshData,
    *,
    detect_nan: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    required = [
        "pos",
        "batch",
        "pointcloud_pos",
        "pointcloud_original_idx",
        "pointcloud_valid_mask",
        "pointcloud_num_points",
    ]
    for attr in required:
        if not hasattr(data, attr):
            raise ValueError(f"Input data is missing required attribute '{attr}'")

    pos = data.pos.float()
    batch = data.batch.long()
    pointcloud_pos = data.pointcloud_pos.float()
    pointcloud_original_idx = data.pointcloud_original_idx.long().view(-1)
    pointcloud_valid_mask = data.pointcloud_valid_mask.bool()
    pointcloud_num_points = data.pointcloud_num_points.long().view(-1)

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
    if pointcloud_original_idx.shape[0] != pointcloud_pos.shape[0]:
        raise ValueError(
            "Expected data.pointcloud_original_idx aligned with data.pointcloud_pos"
        )
    if pointcloud_original_idx.numel() > 0:
        if (
            pointcloud_original_idx.min().item() < 0
            or pointcloud_original_idx.max().item() >= pos.shape[0]
        ):
            raise ValueError("pointcloud_original_idx contains invalid node indices")

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
            pointcloud_original_idx,
            pointcloud_valid_mask,
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
        for tensor, name in (
            (pos, "positions"),
            (pointcloud_pos, "sampled point positions"),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite values in DGCNN {name}")

    return (
        pos,
        batch,
        pointcloud_pos,
        pointcloud_original_idx,
        pointcloud_valid_mask,
        num_graphs,
        n_points,
    )


def _interpolate_to_original_nodes(
    sampled_embedding: torch.Tensor,
    sampled_pos: torch.Tensor,
    sampled_valid_mask: torch.Tensor,
    original_pos: torch.Tensor,
    original_batch: torch.Tensor,
) -> torch.Tensor:
    if original_pos.shape[0] == 0:
        return sampled_embedding.new_empty((0, sampled_embedding.shape[-1]))

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


class DGCNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 0,
        edges_input_dim: int | None = None,
        k: int = 20,
        emb_dims: int = 256,
        detect_nan: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if input_dim < 0:
            raise ValueError(f"input_dim must be non-negative, got: {input_dim}")
        if k < 1:
            raise ValueError(f"k must be positive, got: {k}")
        if emb_dims < 1:
            raise ValueError(f"emb_dims must be positive, got: {emb_dims}")

        self.input_dim = input_dim
        self.edges_input_dim = 0 if edges_input_dim is None else edges_input_dim
        self.k = k
        self.emb_dims = emb_dims
        self.detect_nan = detect_nan
        self.output_dim = emb_dims
        self.feature_dim = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.feature_dim * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

    @property
    def required_batch_attributes(self) -> list[str]:
        return [
            "pos",
            "batch",
            "pointcloud_pos",
            "pointcloud_original_idx",
            "pointcloud_valid_mask",
            "pointcloud_num_points",
        ]

    def forward(self, data: MeshData) -> EncoderOutput:
        (
            pos,
            batch,
            pointcloud_pos,
            pointcloud_original_idx,
            pointcloud_valid_mask,
            num_graphs,
            n_points,
        ) = _validate_pointcloud_layout(data, detect_nan=self.detect_nan)

        if num_graphs == 0:
            node_embedding = pos.new_empty((0, self.output_dim))
        else:
            sampled_pos = pointcloud_pos.view(num_graphs, n_points, 3)
            sampled_source_pos = pos[pointcloud_original_idx].view(
                num_graphs, n_points, 3
            )
            sampled_valid_mask = pointcloud_valid_mask.view(num_graphs, n_points)

            x = sampled_pos.transpose(1, 2).contiguous()
            k = min(self.k, n_points)
            idx = knn(x, k=k)

            x = get_graph_feature(x, k=k, idx=idx)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x1, k=k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x2, k=k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]

            x = get_graph_feature(x3, k=k)
            x = self.conv4(x)
            x4 = x.max(dim=-1, keepdim=False)[0]

            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = self.conv5(x)
            sampled_embedding = x.transpose(1, 2).contiguous()

            if self.detect_nan and not torch.isfinite(sampled_embedding).all():
                raise ValueError("Non-finite values in DGCNN sampled embeddings")

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
