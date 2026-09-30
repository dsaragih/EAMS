import torch
import torch.nn as nn

from data.types import MeshData
from loguru import logger
from torch_geometric.utils import scatter

from ..contracts import EncoderOutput


def _legacy_face_adjacency(face: torch.Tensor) -> torch.Tensor:
    face = face.long()
    num_faces = face.shape[1]
    if num_faces == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=face.device)

    faces = face.transpose(0, 1).contiguous()
    adj_1_1 = faces[:, 0].unsqueeze(1) == faces[:, 0].unsqueeze(0)
    adj_1_2 = faces[:, 0].unsqueeze(1) == faces[:, 1].unsqueeze(0)
    adj_1_3 = faces[:, 0].unsqueeze(1) == faces[:, 2].unsqueeze(0)
    adj_2_1 = faces[:, 1].unsqueeze(1) == faces[:, 0].unsqueeze(0)
    adj_2_2 = faces[:, 1].unsqueeze(1) == faces[:, 1].unsqueeze(0)
    adj_2_3 = faces[:, 1].unsqueeze(1) == faces[:, 2].unsqueeze(0)
    adj_3_1 = faces[:, 2].unsqueeze(1) == faces[:, 0].unsqueeze(0)
    adj_3_2 = faces[:, 2].unsqueeze(1) == faces[:, 1].unsqueeze(0)
    adj_3_3 = faces[:, 2].unsqueeze(1) == faces[:, 2].unsqueeze(0)
    adj = (
        adj_1_1
        + adj_1_2
        + adj_1_3
        + adj_2_1
        + adj_2_2
        + adj_2_3
        + adj_3_1
        + adj_3_2
        + adj_3_3
    )
    adj = torch.where(adj >= 1, 1.0, 0.0).to(dtype=torch.float32)
    return adj @ adj


def _face_incidence(face: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    face = face.long()
    face_index = torch.arange(face.shape[1], device=face.device).repeat_interleave(
        face.shape[0]
    )
    vertex_index = face.transpose(0, 1).reshape(-1)
    return vertex_index, face_index


class PointwiseBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm = nn.BatchNorm1d(out_dim, track_running_stats=False)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class PointwiseHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm = nn.BatchNorm1d(out_dim, track_running_stats=False)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x)))


class AFF(nn.Module):
    def __init__(self, channels: int, r: float = 4.0):
        super().__init__()
        inter_channels = max(1, int(channels // r))
        self.local_att = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels, track_running_stats=False),
        )
        self.global_att = nn.Sequential(
            nn.Linear(channels, inter_channels),
            nn.BatchNorm1d(inter_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inter_channels, channels),
            nn.BatchNorm1d(channels, track_running_stats=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = self.sigmoid(self.local_att(x) + self.global_att(y))
        return x * weight + y * (1.0 - weight)


class FaceGraph(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = PointwiseBlock(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return adjacency @ x


class FaceGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int):
        super().__init__()
        self.head = PointwiseHead(in_dim, out_dim)
        self.gcn = nn.ModuleList(FaceGraph(out_dim, out_dim) for _ in range(depth))
        self.tail = PointwiseHead(out_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        shortcut = x
        for block in self.gcn:
            x = block(x, adjacency)
        return self.tail(x) + shortcut


class BaselineVertexEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int | None = None,
        edges_input_dim: int | None = None,
        face_aggregate: str = "mean",
        output_node_embedding: bool = True,
        output_face_embedding: bool = True,
        detect_nan: bool = True,
        **kwargs,
    ):
        super().__init__()
        del input_dim, edges_input_dim, kwargs

        if face_aggregate not in {"mean", "sum", "max"}:
            raise ValueError(
                f"face_aggregate must be one of {{'mean', 'sum', 'max'}}, got: {face_aggregate}"
            )

        self.face_aggregate = face_aggregate
        self.output_node_embedding = output_node_embedding
        self.output_face_embedding = output_face_embedding
        self.detect_nan = detect_nan
        self.output_dim = 1024
        self.face_output_dim = 1024

        self.conv1_c = PointwiseBlock(12, 64)
        self.conv1_n = PointwiseBlock(12, 64)
        self.conv1_n.norm = self.conv1_c.norm

        self.fa_1 = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.aff_coor_2 = AFF(256, 0.5)
        self.aff_coor_3 = AFF(512, 0.5)
        self.aff_coor_4 = AFF(512, 0.5)

        self.aff_1 = AFF(128, 1.0)
        self.aff_2 = AFF(256, 1.0)
        self.aff_3 = AFF(512, 1.0)
        self.aff_4 = AFF(512, 1.0)

        self.gcn_coor_1_1 = FaceGCN(64, 128, 2)
        self.gcn_coor_1_2 = FaceGCN(128, 128, 2)
        self.gcn_coor_1_3 = FaceGCN(128, 128, 2)
        self.gcn_nor_1_1 = FaceGCN(64, 128, 2)

        self.gcn_coor_2_1 = FaceGCN(128, 256, 1)
        self.gcn_coor_2_2 = FaceGCN(128, 256, 1)
        self.gcn_nor_2_1 = FaceGCN(128, 256, 1)

        self.gcn_coor_3_1 = FaceGCN(256, 512, 2)
        self.gcn_coor_3_2 = FaceGCN(256, 512, 2)
        self.gcn_nor_3_1 = FaceGCN(256, 512, 2)

        self.gcn_coor_4_1 = FaceGCN(512, 512, 2)
        self.gcn_coor_4_2 = FaceGCN(512, 512, 2)
        self.gcn_nor_4_1 = FaceGCN(512, 512, 2)

        self.fu_1 = PointwiseBlock(128 + 256, 512)
        self.fu_2 = PointwiseBlock(512 + 512, 512)
        self.fu_2.norm = self.fu_1.norm

    @property
    def required_batch_attributes(self) -> list[str]:
        return ["pos", "normal", "face", "batch"]

    def _precomputed_face_features(
        self, data: MeshData
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        coor = getattr(data, "tcgn_face_coor", None)
        nor = getattr(data, "tcgn_face_normal", None)
        if coor is None or nor is None:
            return None

        if coor.ndim != 2 or nor.ndim != 2 or coor.shape[1] != 12 or nor.shape[1] != 12:
            raise ValueError(
                "Expected tcgn_face_coor/tcgn_face_normal with shape (F, 12)"
            )
        if coor.shape[0] != data.face.shape[1] or nor.shape[0] != data.face.shape[1]:
            raise ValueError(
                "Precomputed tcgn face features must match the current face count"
            )
        return coor.float(), nor.float()

    def _build_face_features(
        self, pos: torch.Tensor, normal: torch.Tensor, face: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        face_vertices = pos[face.transpose(0, 1).long()]
        face_vertex_normals = normal[face.transpose(0, 1).long()]

        face_centers = face_vertices.mean(dim=1)
        xy = face_vertices[:, 0] - face_vertices[:, 1]
        xz = face_vertices[:, 0] - face_vertices[:, 2]
        face_normals = torch.cross(xy, xz, dim=1)

        coor = torch.cat(
            [face_vertices.reshape(face_vertices.shape[0], 9), face_centers], dim=1
        )
        nor = torch.cat(
            [
                face_vertex_normals.reshape(face_vertex_normals.shape[0], 9),
                face_normals,
            ],
            dim=1,
        )
        return coor, nor

    def _vertex_embedding(
        self, face_embedding: torch.Tensor, face: torch.Tensor, num_vertices: int
    ) -> torch.Tensor:
        vertex_index, face_index = _face_incidence(face)
        return scatter(
            face_embedding[face_index],
            vertex_index,
            dim=0,
            dim_size=num_vertices,
            reduce=self.face_aggregate,
        )

    def _check_nan(self, x: torch.Tensor, name: str):
        if self.detect_nan and torch.isnan(x).any():
            logger.error(f"NaN detected in {name}: shape={tuple(x.shape)}")
            raise ValueError(f"NaN detected in {name}")

    def forward(self, data: MeshData) -> EncoderOutput:
        for attr in self.required_batch_attributes:
            if not hasattr(data, attr):
                raise ValueError(f"Input data is missing required attribute '{attr}'")

        face = data.face.long()
        if face.ndim != 2 or face.shape[0] != 3:
            raise ValueError(
                f"Expected data.face with shape (3, F), got: {tuple(face.shape)}"
            )

        adjacency = _legacy_face_adjacency(face)
        # breakpoint()
        precomputed = self._precomputed_face_features(data)
        if precomputed is None:
            coor, nor = self._build_face_features(
                data.pos.float(), data.normal.float(), face
            )
        else:
            coor, nor = precomputed

        coor = self.conv1_c(coor)
        nor = self.conv1_n(nor)

        coor1 = self.gcn_coor_1_1(coor, adjacency)
        coor1 = self.gcn_coor_1_2(coor1, adjacency)
        coor1 = self.gcn_coor_1_3(coor1, adjacency)
        nor1 = self.gcn_nor_1_1(nor, adjacency)
        coor_nor1 = self.aff_1(coor1, nor1)

        coor2_1 = self.gcn_coor_2_1(coor1, adjacency)
        coor2_2 = self.gcn_coor_2_2(coor1, adjacency)
        coor2 = self.aff_coor_2(coor2_1, coor2_2)
        nor2 = self.gcn_nor_2_1(nor1, adjacency)
        coor_nor2 = self.aff_2(coor2, nor2)

        coor3_1 = self.gcn_coor_3_1(coor2, adjacency)
        coor3_2 = self.gcn_coor_3_2(coor2, adjacency)
        coor3 = self.aff_coor_3(coor3_1, coor3_2)
        nor3 = self.gcn_nor_3_1(nor2, adjacency)
        coor_nor3 = self.aff_3(coor3, nor3)

        coor4_1 = self.gcn_coor_4_1(coor3, adjacency)
        coor4_2 = self.gcn_coor_4_2(coor3, adjacency)
        coor4 = self.aff_coor_4(coor4_1, coor4_2)
        nor4 = self.gcn_nor_4_1(nor3, adjacency)
        coor_nor4 = self.aff_4(coor4, nor4)

        x1 = self.fu_1(torch.cat((coor_nor1, coor_nor2), dim=1))
        x2 = self.fu_2(torch.cat((coor_nor3, coor_nor4), dim=1))
        face_embedding = self.fa_1(torch.cat((x1, x2), dim=1))
        self._check_nan(face_embedding, "face_embedding")

        node_embedding = None
        if self.output_node_embedding:
            node_embedding = self._vertex_embedding(
                face_embedding=face_embedding,
                face=face,
                num_vertices=data.pos.shape[0],
            )
            self._check_nan(node_embedding, "node_embedding")

        face_batch = (
            data.batch[face[0].long()]
            if face.numel() > 0
            else data.batch.new_empty((0,), dtype=data.batch.dtype)
        )

        return EncoderOutput(
            node_embedding=node_embedding,
            pos=data.pos,
            batch=data.batch,
            edge_index=getattr(data, "edge_index", None),
            edge_attr=getattr(data, "edge_attr", None),
            face=face,
            face_batch=face_batch,
            face_embedding=face_embedding if self.output_face_embedding else None,
        )
