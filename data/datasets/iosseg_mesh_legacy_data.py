"""Raw PLY IOSSeg dataset for legacy TCGN parity.

This dataset intentionally avoids cached pre-transforms so samples stay close to
the original fast-tgcn data path:
- load raw ASCII .ply on every access
- preserve original face list and PLY normals
- apply only runtime transforms configured for the active stage
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger as log
from plyfile import PlyData
from torch.utils.data import Dataset

from data.transforms.transforms import compose_graph_transforms
from data.types import MeshData


FAST_TGCN_GINGIVA_CLASS_ID = 16
FAST_TGCN_NUM_CLASSES = 17
FAST_TGCN_GINGIVA_RGB = (255, 255, 255)
FAST_TGCN_LOWER_LABELS = [
    (255, 48, 48),
    (0, 191, 255),
    (255, 165, 0),
    (202, 255, 112),
    (200, 255, 255),
    (255, 228, 255),
    (255, 155, 255),
    (255, 69, 0),
    (139, 0, 0),
    (144, 238, 144),
    (0, 139, 139),
    (0, 0, 139),
    (139, 0, 139),
    (255, 105, 180),
    (230, 230, 250),
    (255, 228, 181),
]
FAST_TGCN_UPPER_LABELS = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (30, 144, 255),
    (0, 255, 127),
    (127, 255, 0),
    (255, 246, 143),
    (60, 179, 113),
    (255, 106, 106),
    (131, 111, 255),
    (255, 211, 155),
    (255, 99, 71),
    (155, 48, 255),
]


FAST_TGCN_RGB_TO_CLASS: dict[Tuple[int, int, int], int] = {
    FAST_TGCN_GINGIVA_RGB: FAST_TGCN_GINGIVA_CLASS_ID,
}
for class_id, rgb in enumerate(FAST_TGCN_LOWER_LABELS):
    FAST_TGCN_RGB_TO_CLASS[rgb] = class_id
for class_id, rgb in enumerate(FAST_TGCN_UPPER_LABELS):
    FAST_TGCN_RGB_TO_CLASS[rgb] = class_id


def _build_edge_index_from_faces(face: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if face.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    a, b, c = face[0], face[1], face[2]
    src = torch.cat([a, b, c, b, c, a], dim=0)
    dst = torch.cat([b, c, a, a, b, c], dim=0)

    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    key = src.to(torch.int64) * int(num_nodes) + dst.to(torch.int64)
    key = torch.unique(key)
    src_u = (key // int(num_nodes)).to(torch.long)
    dst_u = (key % int(num_nodes)).to(torch.long)
    return torch.stack([src_u, dst_u], dim=0)


def load_legacy_ply_file(ply_path: str) -> MeshData:
    """Load a PLY mesh using fast-tgcn-compatible face color semantics.

    Key parity choices vs the generic IOSSeg loader:
    - use `plyfile.PlyData.read` like fast-tgcn
    - preserve raw vertex coordinates (no centering here)
    - map tooth colors to class ids 0..15 and gingiva to class id 16
    - accept both lower and upper palette variants for the same tooth id
    """

    if not osp.isfile(ply_path):
        raise FileNotFoundError(f"Missing PLY file: {ply_path}")

    row_data = PlyData.read(ply_path)
    vertex_data = row_data.elements[0].data
    face_data = row_data.elements[1].data

    points = np.column_stack([vertex_data[name] for name in vertex_data.dtype.names])
    xyz = points[:, :3].astype(np.float32)

    if points.shape[1] >= 6:
        normal = points[:, 3:6].astype(np.float32)
    else:
        normal = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    index_face = np.vstack(face_data["vertex_indices"]).astype(np.int64)
    rgb_face = np.column_stack(
        [face_data["red"], face_data["green"], face_data["blue"]]
    ).astype(np.int64)

    face_labels = np.zeros((index_face.shape[0],), dtype=np.int64)
    unknown_rgb = 0
    for face_idx, rgb in enumerate(rgb_face):
        rgb_key = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if rgb_key not in FAST_TGCN_RGB_TO_CLASS:
            unknown_rgb += 1
        face_labels[face_idx] = FAST_TGCN_RGB_TO_CLASS.get(rgb_key, 0)

    # if unknown_rgb > 0:
    #     log.warning(
    #         "[IOSSegLegacy] {} face(s) had unknown RGB and were mapped to class 0 in {}",
    #         unknown_rgb,
    #         osp.basename(ply_path),
    #     )

    vertex_counts = np.zeros((xyz.shape[0], FAST_TGCN_NUM_CLASSES), dtype=np.int64)
    for face_idx, class_id in enumerate(face_labels):
        v0, v1, v2 = index_face[face_idx]
        vertex_counts[v0, class_id] += 1
        vertex_counts[v1, class_id] += 1
        vertex_counts[v2, class_id] += 1
    vertex_labels = np.argmax(vertex_counts, axis=1).astype(np.int64)

    pos = torch.tensor(xyz, dtype=torch.float32)
    norm = torch.tensor(normal, dtype=torch.float32)
    face = torch.tensor(index_face, dtype=torch.long).t().contiguous()
    edge_index = _build_edge_index_from_faces(face, num_nodes=pos.shape[0])

    data = MeshData(
        pos=pos,
        normal=norm.clone(),
        face=face,
        edge_index=edge_index,
        face_labels=torch.tensor(face_labels, dtype=torch.long),
    )
    data.norm = norm
    data.vertex_labels = torch.tensor(vertex_labels, dtype=torch.long)
    data.face_rgb = torch.tensor(rgb_face, dtype=torch.long)
    return data


class IOSSegLegacyMeshDataset(Dataset):
    """IOSSeg dataset that reads raw PLY meshes without preprocessing cache."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        file_list: Optional[Iterable[str]] = None,
        transforms: Optional[List[Any]] = None,
        profile_transforms: bool = False,
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.split = split
        self.profile_transforms = profile_transforms

        mesh_ids = (
            list(file_list) if file_list is not None else self._discover_all_mesh_ids()
        )
        self.samples = [
            self._normalize_mesh_id(mesh_id_raw) for mesh_id_raw in mesh_ids
        ]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Loaded 0 meshes for split='{self.split}' from root='{self.root}'"
            )

        self.transform = compose_graph_transforms(
            transforms or [], profile=profile_transforms, stage=split
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MeshData:
        mesh_id, filename = self.samples[idx]
        ply_path = osp.join(self.root, filename)

        data = load_legacy_ply_file(ply_path)
        data.mesh_id = mesh_id
        data.filename = filename

        data = self.transform(data)
        return MeshData.from_data(data)

    def get_transform_timing_report(self) -> str:
        if self.profile_transforms and hasattr(self.transform, "get_timing_report"):
            return self.transform.get_timing_report()
        return "Transform profiling is not enabled. Set profile_transforms=True when creating the dataset."

    def reset_transform_timing(self) -> None:
        if self.profile_transforms and hasattr(self.transform, "reset_timing_stats"):
            self.transform.reset_timing_stats()

    @staticmethod
    def _normalize_mesh_id(mesh_id_raw: str) -> Tuple[str, str]:
        s = str(mesh_id_raw).strip().replace("\\", "/")
        base = osp.basename(s)
        if base.endswith(".ply"):
            base = base[: -len(".ply")]
        if not base:
            raise ValueError("Empty mesh id")
        return base, f"{base}.ply"

    def _discover_all_mesh_ids(self) -> List[str]:
        if not osp.isdir(self.root):
            return []
        return sorted(
            fname[: -len(".ply")]
            for fname in os.listdir(self.root)
            if fname.lower().endswith(".ply")
        )
