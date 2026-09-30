"""IntrA (Intracranial Aneurysm) mesh dataset.

This dataset is vertex-supervised (Data.vertex_labels) and optionally supports
geodesic-distance edge weights (Data.edge_weight) loaded from .gd files.

Expected directory layout (configurable via DataModule args):
    <obj_root>/ (e.g., AN1_full.obj)
    <label_root>/ (e.g., AN1-_norm.ad)
    <gd_root>/ (e.g., AN1_full.gd)

Split ids are typically simple mesh ids like 'AN1'.
"""

from __future__ import annotations

import os
import os.path as osp
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger as log
from torch_geometric.data import Data, InMemoryDataset

from data.transforms.pre_transforms import compose_graph_pretransforms
from data.transforms.transforms import compose_graph_transforms
from data.types import MeshData


@dataclass(frozen=True)
class IntrAMeshId:
    """Identifier for a single IntrA mesh."""

    stem: str


def _parse_mesh_id(mesh_id: str) -> IntrAMeshId:
    mesh_id = str(mesh_id).strip()
    if not mesh_id:
        raise ValueError("Empty mesh id")

    # annotated/ad/AN44-3-_norm.ad -> AN44-3
    mesh_id = os.path.basename(mesh_id)
    # Split from the last "-"
    mesh_id = mesh_id.rsplit("-", 1)[0]

    # Allow passing filenames in split files.
    for suffix in (".obj", ".ad", ".gd", ".txt"):
        if mesh_id.endswith(suffix):
            mesh_id = mesh_id[: -len(suffix)]
            break

    # Commonly split ids are like 'AN1' (no suffix).
    return IntrAMeshId(stem=mesh_id)


def _triangulate_faces(face_indices: List[List[int]]) -> np.ndarray:
    """Triangulate OBJ faces with fan triangulation.

    Returns:
        (F, 3) array with 0-based vertex indices.
    """

    tris: List[Tuple[int, int, int]] = []
    for f in face_indices:
        if len(f) < 3:
            continue
        if len(f) == 3:
            tris.append((f[0], f[1], f[2]))
        else:
            v0 = f[0]
            for i in range(1, len(f) - 1):
                tris.append((v0, f[i], f[i + 1]))

    if not tris:
        return np.zeros((0, 3), dtype=np.int64)

    return np.asarray(tris, dtype=np.int64)


def _build_edge_index_from_faces(face_f3: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build a bidirectional, coalesced edge_index from triangular faces."""

    if face_f3.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    a, b, c = face_f3[0], face_f3[1], face_f3[2]

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


def _load_ad_vertex_labels(
    label_path: str,
    vertices_xyz: np.ndarray,
    merge_label_2_to_1: bool = True,
) -> torch.Tensor:
    """Load vertex labels from IntrA .ad file.

    The .ad file contains per-vertex rows: (x, y, z, nx, ny, nz, label).
    If the number of rows does not match the OBJ vertex count, falls back to
    nearest-neighbour mapping by xyz.

    Returns:
        labels: LongTensor shape (N,) with values in [0..] or -1 for unknown.
    """

    if not osp.isfile(label_path):
        return torch.full((vertices_xyz.shape[0],), -1, dtype=torch.long)

    arr = np.loadtxt(label_path, dtype=np.float64)
    arr = np.atleast_2d(arr)
    if arr.size == 0:
        return torch.full((vertices_xyz.shape[0],), -1, dtype=torch.long)

    if arr.shape[1] < 7:
        raise ValueError(
            f"Expected at least 7 columns in label file (x y z nx ny nz label): {label_path}"
        )

    label_xyz = arr[:, :3]
    labels = arr[:, -1].astype(np.int64)

    if merge_label_2_to_1:
        labels[labels == 2] = 1

    n = int(vertices_xyz.shape[0])
    if labels.shape[0] == n:
        return torch.tensor(labels, dtype=torch.long)

    try:
        from scipy.spatial import cKDTree
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SciPy is required for label mapping when .ad rows != vertex count. "
            "Install 'scipy' or ensure labels match vertex order."
        ) from e

    tree = cKDTree(vertices_xyz)
    _, nn_idx = tree.query(label_xyz, k=1, workers=1)
    nn_idx = nn_idx.astype(np.int64)

    mapped = np.full((n,), fill_value=-1, dtype=np.int64)
    # If duplicates occur, later assignments win; acceptable for fallback.
    mapped[nn_idx] = labels
    return torch.tensor(mapped, dtype=torch.long)


def _load_gd_edge_weights(
    gd_path: str,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Load geodesic distance weights for directed edges from a .gd matrix file."""

    if not gd_path or not osp.isfile(gd_path) or edge_index.numel() == 0:
        return torch.empty((edge_index.shape[1],), dtype=torch.float32)

    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    m = int(src.shape[0])

    # adjacency[i] = list of (j, edge_k) for directed edge i->j
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(int(num_nodes))]
    for k in range(m):
        adjacency[int(src[k])].append((int(dst[k]), k))

    weights = np.zeros((m,), dtype=np.float32)

    with open(gd_path, "r") as f:
        for i, raw in enumerate(f):
            if i >= int(num_nodes):
                break
            line = raw.strip()
            if not line:
                continue

            dist_row = np.fromstring(line, sep=" ", dtype=np.float32)
            if dist_row.size < int(num_nodes):
                raise ValueError(
                    f"Row {i} in gd file has {dist_row.size} entries, expected >= {num_nodes}: {gd_path}"
                )

            for j, k in adjacency[i]:
                weights[k] = dist_row[j]

    weights = torch.tensor(weights, dtype=torch.float32)
    # Higher weight for smaller geodesic distance; add small epsilon to avoid division by zero.
    weights = 1.0 / (weights + 1e-8)
    return weights


class IntrAMeshDataset(InMemoryDataset):
    """IntrA mesh dataset.

    Output Data fields:
      - pos: (N, 3)
      - face: (3, F)
      - edge_index: (2, E)
      - edge_weight: (E,) optional (if .gd is present and enabled)
      - vertex_labels: (N,) with ignore_index -1
      - mesh_id: '<ID>'
      - filename: '<ID>_full.obj' (for logging/compat)
    """

    def __init__(
        self,
        root: str,
        obj_root: str,
        label_root: str,
        gd_root: str,
        split: str = "train",
        file_list: Optional[Iterable[str]] = None,
        transforms: Optional[List[Any]] = None,
        pre_transforms: Optional[List[str]] = None,
        pre_filter: Optional[Callable] = None,
        profile_transforms: bool = False,
        fold_tag: str = "",
        obj_suffix: str = "_full.obj",
        label_suffix: str = "-_norm.ad",
        gd_suffix: str = "_full.gd",
        use_geodesic_edge_weight: bool = True,
        merge_label_2_to_1: bool = True,
    ):
        self.split = split
        self.file_list = list(file_list) if file_list is not None else None
        self.profile_transforms = profile_transforms
        self.fold_tag = str(fold_tag or "")

        self.obj_root = str(obj_root)
        self.label_root = str(label_root)
        self.gd_root = str(gd_root)

        self.obj_suffix = obj_suffix
        self.label_suffix = label_suffix
        self.gd_suffix = gd_suffix

        self.use_geodesic_edge_weight = bool(use_geodesic_edge_weight)
        self.merge_label_2_to_1 = bool(merge_label_2_to_1)

        if self.file_list is not None and pre_filter is None:
            allowed = {
                _parse_mesh_id(str(s)).stem for s in self.file_list if str(s).strip()
            }
            pre_filter = lambda data: getattr(data, "mesh_id", None) in allowed

        transform = (
            compose_graph_transforms(
                transforms or [], profile=profile_transforms, stage=split
            )
            if transforms is not None
            else None
        )
        pre_transform = (
            compose_graph_pretransforms(pre_transforms or [])
            if pre_transforms is not None
            else None
        )

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_dir(self) -> str:
        suffix = f"_{self.fold_tag}" if self.fold_tag else ""
        return osp.join(self.root, f"processed_{self.split}{suffix}")

    @property
    def raw_file_names(self) -> List[str]:
        # Avoid expensive enumeration; we validate existence during processing.
        return []

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def get_transform_timing_report(self) -> str:
        if self.profile_transforms and hasattr(self.transform, "get_timing_report"):
            return self.transform.get_timing_report()
        return "Transform profiling is not enabled. Set profile_transforms=True when creating the dataset."

    def reset_transform_timing(self):
        if self.profile_transforms and hasattr(self.transform, "reset_timing_stats"):
            self.transform.reset_timing_stats()

    def process(self):
        mesh_ids = self.file_list
        if mesh_ids is None:
            mesh_ids = self._discover_all_mesh_ids()

        data_list: List[MeshData] = []
        signature_cache_dir = osp.join(self.processed_dir, "signature_spectra")
        for mesh_id_raw in mesh_ids:
            start = time.time()
            mesh_id_raw = str(mesh_id_raw).strip()
            if not mesh_id_raw:
                continue
            mesh_id = _parse_mesh_id(mesh_id_raw)

            obj_path = osp.join(self.obj_root, f"{mesh_id.stem}{self.obj_suffix}")
            label_path = osp.join(self.label_root, f"{mesh_id.stem}{self.label_suffix}")
            gd_path = osp.join(self.gd_root, f"{mesh_id.stem}{self.gd_suffix}")

            data = self._load_pyg_data(
                obj_path=obj_path,
                label_path=label_path,
                gd_path=gd_path,
                mesh_id=mesh_id.stem,
            )
            data.signature_cache_dir = signature_cache_dir

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(MeshData.from_data(data))
            end = time.time()
            log.info(f"Processed mesh {mesh_id.stem} in {end - start:.2f} seconds")

        if len(data_list) == 0:
            raise RuntimeError(
                f"Processed 0 meshes for split='{self.split}'. "
                "Check split ids and that expected files exist under the annotated root."
            )

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _discover_all_mesh_ids(self) -> List[str]:
        ids: List[str] = []
        if not osp.isdir(self.obj_root):
            return ids

        for fname in sorted(os.listdir(self.obj_root)):
            if not fname.endswith(".obj"):
                continue
            if not fname.endswith(self.obj_suffix):
                continue
            stem = fname[: -len(self.obj_suffix)]
            if stem:
                ids.append(stem)
        return ids

    def _load_pyg_data(
        self,
        obj_path: str,
        label_path: str,
        gd_path: str,
        mesh_id: str,
    ) -> Data:
        if not osp.isfile(obj_path):
            raise FileNotFoundError(f"Missing OBJ file: {obj_path}")

        vertices: List[List[float]] = []
        faces_raw: List[List[int]] = []

        with open(obj_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue

                if parts[0] == "v":
                    if len(parts) < 4:
                        continue
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

                elif parts[0] == "f":
                    idxs: List[int] = []
                    for token in parts[1:]:
                        if not token:
                            continue
                        vstr = token.split("/")[0]
                        if not vstr:
                            continue
                        vid = int(vstr)
                        if vid < 0:
                            vid = len(vertices) + vid
                        else:
                            vid = vid - 1
                        idxs.append(vid)
                    if idxs:
                        faces_raw.append(idxs)

        vertices_np = np.asarray(vertices, dtype=np.float64)
        faces_f3 = _triangulate_faces(faces_raw)

        if vertices_np.shape[0] == 0:
            raise RuntimeError(f"No vertices found in OBJ: {obj_path}")

        pos = torch.tensor(vertices_np, dtype=torch.float32)
        pos = pos - pos.mean(dim=0, keepdim=True)

        if faces_f3.shape[0] > 0:
            face = torch.tensor(faces_f3, dtype=torch.long).t().contiguous()  # (3, F)
        else:
            face = torch.empty((3, 0), dtype=torch.long)

        edge_index = _build_edge_index_from_faces(face, num_nodes=pos.shape[0])

        vertex_labels = _load_ad_vertex_labels(
            label_path=label_path,
            vertices_xyz=vertices_np,
            merge_label_2_to_1=self.merge_label_2_to_1,
        )
        if vertex_labels.shape[0] != pos.shape[0]:
            raise ValueError(
                f"Vertex label length mismatch for {label_path}: got {vertex_labels.shape[0]}, expected {pos.shape[0]}"
            )

        data = Data(pos=pos, face=face, edge_index=edge_index)
        data.vertex_labels = vertex_labels
        data.mesh_id = mesh_id
        data.filename = f"{mesh_id}{self.obj_suffix}"

        if self.use_geodesic_edge_weight and gd_path and osp.isfile(gd_path):
            data.edge_weight = _load_gd_edge_weights(
                gd_path=gd_path,
                edge_index=edge_index,
                num_nodes=pos.shape[0],
            )

        return data


if __name__ == "__main__":
    # Prefer running from repo root:
    #   python -m data.datasets.intra_mesh_data
    pass
