"""IOSSeg 3D (PLY) mesh dataset.

Implements PLY loading and face-color -> vertex label mapping based on the logic
in the original IOSSeg face-color preprocessing pipeline.

Expected behavior:
- Reads ASCII .ply with vertices and faces.
- Face RGBA colors are mapped to class ids (gingiva=0, teeth=1..16).
- Vertex labels are computed by majority vote over incident faces.
- Produces PyTorch Geometric `Data` with `pos`, `face`, `edge_index`, `vertex_labels`, `norm`.

Dataset is cached via `InMemoryDataset` under a split-specific processed folder.

Note: If you run this file directly (e.g. `python data/datasets/iosseg_mesh_data.py`)
from outside the project root, Python may not be able to resolve the top-level
`data` package. The small block below makes direct execution more robust by
adding the repository root to `sys.path`.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import os
import os.path as osp
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger as log
from torch_geometric.data import Data, InMemoryDataset

from data.transforms.pre_transforms import compose_graph_pretransforms
from data.transforms.transforms import compose_graph_transforms
from data.types import MeshData


GINGIVA_CLASS_ID = 0
N_TEETH = 16
N_CLASSES = 1 + N_TEETH  # 17

# Build a deterministic RGB -> class_id mapping:
#   0 = gingiva
#   1..16 = tooth index (shared between lower/upper palettes)
_GINGIVA_RGB = (255, 255, 255)
_LOWER_LABELS = [
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
_UPPER_LABELS = [
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

LABEL_RGB_TO_CLASS: dict[Tuple[int, int, int], int] = {_GINGIVA_RGB: GINGIVA_CLASS_ID}
for i in range(N_TEETH):
    class_id = i + 1
    LABEL_RGB_TO_CLASS[_LOWER_LABELS[i]] = class_id
    LABEL_RGB_TO_CLASS[_UPPER_LABELS[i]] = class_id


def _triangulate_polygon(idx: List[int]) -> List[Tuple[int, int, int]]:
    if len(idx) < 3:
        return []
    if len(idx) == 3:
        return [(idx[0], idx[1], idx[2])]
    v0 = idx[0]
    return [(v0, idx[t], idx[t + 1]) for t in range(1, len(idx) - 1)]


def _build_edge_index_from_faces(face_f3: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build bidirectional, unique edge_index from triangular faces."""

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


def load_ply_file(ply_path: str) -> Data:
    """Load an ASCII .ply mesh where face colors define classes.

    This matches the notebook logic:
    - Face RGB is mapped via LABEL_RGB_TO_CLASS (unknown -> gingiva=0)
    - Vertex labels are a majority vote over incident faces.

    Returns `Data` with pos, face, edge_index, norm, vertex_labels.
    """

    if not osp.isfile(ply_path):
        raise FileNotFoundError(f"Missing PLY file: {ply_path}")

    with open(ply_path, "r") as f:
        fmt = None
        n_verts: Optional[int] = None
        n_faces: Optional[int] = None

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {ply_path}")
            line = line.strip()

            if line.startswith("format "):
                fmt = line.split()[1].lower()
            elif line.startswith("element vertex "):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face "):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break

        if fmt != "ascii":
            raise NotImplementedError(
                f"Only ASCII PLY is supported (got format={fmt}) for file {ply_path}"
            )
        if n_verts is None or n_faces is None:
            raise ValueError(f"PLY header missing element counts: {ply_path}")

        verts = np.zeros((n_verts, 3), dtype=np.float64)
        norms = np.zeros((n_verts, 3), dtype=np.float64)

        for i in range(n_verts):
            parts = f.readline().strip().split()
            if len(parts) < 3:
                raise ValueError(
                    f"Vertex line {i} has < 3 floats in {ply_path}: {parts}"
                )
            verts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            if len(parts) >= 6:
                norms[i] = [float(parts[3]), float(parts[4]), float(parts[5])]

        faces: List[Tuple[int, int, int]] = []
        face_classes: List[int] = []
        unknown_rgb = 0

        for i in range(n_faces):
            parts = f.readline().strip().split()
            if len(parts) < 4:
                continue

            k = int(parts[0])
            if len(parts) < 1 + k + 3:
                raise ValueError(f"Face line {i} too short in {ply_path}: {parts}")

            idx = [int(x) for x in parts[1 : 1 + k]]

            # Expect RGB or RGBA after indices.
            # If RGBA exists, take first 3.
            tail = parts[1 + k :]
            if len(tail) >= 4:
                rgb = (int(tail[0]), int(tail[1]), int(tail[2]))
            elif len(tail) >= 3:
                rgb = (int(tail[0]), int(tail[1]), int(tail[2]))
            else:
                rgb = _GINGIVA_RGB

            c = LABEL_RGB_TO_CLASS.get(rgb, GINGIVA_CLASS_ID)
            if rgb not in LABEL_RGB_TO_CLASS:
                unknown_rgb += 1

            for tri in _triangulate_polygon(idx):
                faces.append(tri)
                face_classes.append(c)

    faces_np = (
        np.asarray(faces, dtype=np.int64)
        if len(faces)
        else np.zeros((0, 3), dtype=np.int64)
    )
    face_classes_np = (
        np.asarray(face_classes, dtype=np.int64)
        if len(face_classes)
        else np.zeros((0,), dtype=np.int64)
    )

    if unknown_rgb > 0:
        log.warning(
            f"[IOSSeg] {unknown_rgb} face(s) had unknown RGB and were mapped to gingiva (0) in {osp.basename(ply_path)}"
        )

    # Face labels -> vertex labels via majority vote over incident faces.
    if faces_np.shape[0] > 0:
        counts = np.zeros((n_verts, N_CLASSES), dtype=np.int64)
        for fi in range(faces_np.shape[0]):
            c = int(face_classes_np[fi])
            if c < 0 or c >= N_CLASSES:
                continue
            v0, v1, v2 = faces_np[fi]
            counts[v0, c] += 1
            counts[v1, c] += 1
            counts[v2, c] += 1

        vertex_labels_np = np.argmax(counts, axis=1).astype(np.int64)
    else:
        vertex_labels_np = np.zeros((n_verts,), dtype=np.int64)

    pos = torch.tensor(verts, dtype=torch.float32)
    pos = pos - pos.mean(dim=0, keepdim=True)

    face = (
        torch.tensor(faces_np, dtype=torch.long).t().contiguous()
        if faces_np.size > 0
        else torch.empty((3, 0), dtype=torch.long)
    )

    edge_index = _build_edge_index_from_faces(face, num_nodes=pos.shape[0])

    data = Data(pos=pos, face=face, edge_index=edge_index)
    data.norm = torch.tensor(norms, dtype=torch.float32)
    data.face_labels = torch.tensor(face_classes_np, dtype=torch.long)
    data.vertex_labels = torch.tensor(vertex_labels_np, dtype=torch.long)

    return data


def _append_face_labels_to_data(data: Data) -> MeshData:
    mesh_data = MeshData.from_data(data)
    if hasattr(mesh_data, "face_labels") and mesh_data.face_labels is not None:
        if int(mesh_data.face_labels.shape[0]) == int(mesh_data.face.shape[1]):
            return mesh_data

    if not hasattr(mesh_data, "face") or mesh_data.face is None:
        raise ValueError("Cannot append face_labels to mesh without face connectivity")
    if not hasattr(mesh_data, "vertex_labels") or mesh_data.vertex_labels is None:
        raise ValueError("Cannot append face_labels to mesh without vertex_labels")

    face_vertex_labels = mesh_data.vertex_labels[mesh_data.face]
    mesh_data.face_labels = face_vertex_labels[0].contiguous()
    return mesh_data


def _resolved_processed_dir_arg(
    processed_dir: Optional[str], split: str
) -> Optional[str]:
    if processed_dir is None:
        return None
    return f"{processed_dir}_{split}"


def _upgrade_processed_cache(
    *,
    data_dir: str,
    split: str,
    processed_dir: str,
    output_processed_dir: Optional[str],
    in_place: bool,
) -> str:
    source_dir = _resolved_processed_dir_arg(processed_dir, split)
    if source_dir is None:
        raise ValueError("processed_dir is required when appending face labels")

    dataset = IOSSegMeshDataset(
        root=osp.abspath(osp.expanduser(data_dir)),
        split=split,
        processed_dir=source_dir,
    )

    upgraded = [
        _append_face_labels_to_data(dataset.get(i)) for i in range(len(dataset))
    ]
    target_dir = (
        source_dir
        if in_place
        else _resolved_processed_dir_arg(output_processed_dir, split)
    )
    if target_dir is None:
        raise ValueError(
            "Provide --output-processed-dir or use --in-place when appending face labels"
        )

    os.makedirs(target_dir, exist_ok=True)
    torch.save(dataset.collate(upgraded), osp.join(target_dir, "data.pt"))
    return target_dir


class IOSSegMeshDataset(InMemoryDataset):
    """IOSSeg 3D dataset with meshes stored as ASCII .ply files.

    The dataset root points to a directory containing PLY meshes.
    Split files (managed by the DataModule) should contain one mesh id per line,
    either with or without the `.ply` extension.

    Output Data fields:
      - pos: (N, 3)
      - face: (3, F)
      - edge_index: (2, E)
      - vertex_labels: (N,)
        - face_labels: (F,)
      - norm: (N, 3) (zeros if not present in PLY)
      - filename: '<mesh_id>.ply'
      - mesh_id: '<mesh_id>'
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        file_list: Optional[Iterable[str]] = None,
        transforms: Optional[List[Any]] = None,
        pre_transforms: Optional[List[str]] = None,
        pre_filter: Optional[Callable] = None,
        profile_transforms: bool = False,
        processed_dir: Optional[str] = None,
    ):
        self.split = split
        self.file_list = list(file_list) if file_list is not None else None
        self.profile_transforms = profile_transforms
        self._processed_dir_override = (
            osp.abspath(osp.expanduser(processed_dir)) if processed_dir else None
        )

        if self.file_list is not None and pre_filter is None:
            allowed = {
                self._normalize_mesh_id(s)[0] for s in self.file_list if str(s).strip()
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
        if self._processed_dir_override is not None:
            return self._processed_dir_override
        return osp.join(os.path.dirname(self.root), f"new_processed_{self.split}")

    @property
    def raw_file_names(self) -> List[str]:
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

    @staticmethod
    def _normalize_mesh_id(mesh_id_raw: str) -> Tuple[str, str]:
        """Return (mesh_id, filename).

        Accepts entries like:
        - '007_L'
        - '007_L.ply'
        - 'subdir/007_L.ply'

        mesh_id is always the base filename without extension.
        filename is always '<mesh_id>.ply'.
        """

        s = str(mesh_id_raw).strip()
        s = s.replace("\\", "/")
        base = osp.basename(s)
        if base.endswith(".ply"):
            base = base[: -len(".ply")]
        if not base:
            raise ValueError("Empty mesh id")
        return base, f"{base}.ply"

    def process(self):
        mesh_ids = self.file_list
        if mesh_ids is None:
            mesh_ids = self._discover_all_mesh_ids()

        data_list: List[MeshData] = []
        signature_cache_dir = osp.join(self.processed_dir, "signature_spectra")
        for mesh_id_raw in mesh_ids:
            t0 = time.time()
            mesh_id, filename = self._normalize_mesh_id(mesh_id_raw)

            ply_path = osp.join(self.root, filename)
            data = load_ply_file(ply_path)

            data.mesh_id = mesh_id
            data.filename = filename
            data.signature_cache_dir = signature_cache_dir

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(MeshData.from_data(data))
            log.info(f"Processed IOSSeg mesh {mesh_id} in {time.time() - t0:.2f}s")

        if len(data_list) == 0:
            raise RuntimeError(
                f"Processed 0 meshes for split='{self.split}'. "
                f"Check that PLY files exist under root='{self.root}' and split ids are correct."
            )

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _discover_all_mesh_ids(self) -> List[str]:
        if not osp.isdir(self.root):
            return []
        out: List[str] = []
        for fname in sorted(os.listdir(self.root)):
            if fname.lower().endswith(".ply"):
                out.append(fname[: -len(".ply")])
        return out


if __name__ == "__main__":
    import argparse

    def open_ids_file(ids_file: Optional[str]) -> List[str]:
        if ids_file and osp.isfile(ids_file):
            with open(ids_file, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def _split_csv(value: Optional[str]) -> Optional[List[str]]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]

    # Prefer running as a module from the repo root:
    #   python -m data.datasets.iosseg_mesh_data --data-dir /path/to/ply_dir --split train
    # Direct execution also works thanks to the sys.path bootstrap above:
    #   python data/datasets/iosseg_mesh_data.py --data-dir /path/to/ply_dir --split train

    parser = argparse.ArgumentParser(description="IOSSegMeshDataset quick smoke test")
    parser.add_argument(
        "--append-face-labels",
        action="store_true",
        help="Upgrade an existing processed cache by appending face_labels.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing IOSSeg ASCII .ply meshes.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Split name (affects processed directory naming).",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Optional path to a text file containing mesh ids (one per line).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Optional override: directory to store/load processed 'data.pt'.",
    )
    parser.add_argument(
        "--output-processed-dir",
        type=str,
        default=None,
        help="Optional output directory prefix when writing an upgraded processed cache.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the existing processed cache when appending face labels.",
    )
    parser.add_argument(
        "--pre",
        type=str,
        default=None,
        help="Optional comma-separated list of pre_transforms (applied during processing).",
    )
    args = parser.parse_args()

    if args.append_face_labels:
        upgraded_dir = _upgrade_processed_cache(
            data_dir=args.data_dir,
            split=args.split,
            processed_dir=args.processed_dir,
            output_processed_dir=args.output_processed_dir,
            in_place=args.in_place,
        )
        print(f"Wrote upgraded processed cache with face_labels to {upgraded_dir}")
        raise SystemExit(0)

    splits_dir = osp.join(osp.dirname(args.data_dir), "splits")
    file_ids = open_ids_file(f"{splits_dir}/{args.split}_ids.txt") if args.split != "all" else None

    dataset = IOSSegMeshDataset(
        root=osp.abspath(osp.expanduser(args.data_dir)),
        split=args.split,
        file_list=file_ids if file_ids else None,
        pre_transforms=_split_csv(args.pre),
        processed_dir=_resolved_processed_dir_arg(args.processed_dir, args.split),
        # Keep the smoke test minimal: no transforms unless you pass them from code.
    )

    print(f"Loaded {len(dataset)} meshes (split={args.split})")
    mesh = dataset[0]
    print(mesh)
